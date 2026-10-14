from copy import deepcopy
import unittest
import test_chunker
from app.chunker.parent_child import build_chunks
from app.chunker.source_mapping import map_chunk_sources
from app.review.chunk_operations import apply_chunk_operations


class ChunkOperationTests(unittest.TestCase):
    def setUp(self):
        self.result = test_chunker.ParentChildChunkerTests().make_result()
        template = next(c for c in build_chunks(self.result, "drug", "product") if not c.is_parent and "summary" not in c.content_type)
        self.chunks = []
        for index, ids in enumerate((["p1", "p2"], ["list"])):
            chunk = deepcopy(template)
            chunk.id = f"c{index}"
            chunk.reading_order = index * 100
            chunk.source_element_ids = ids
            chunk.content = "\n".join(next(e.raw_text for e in self.result.elements if e.id == key) for key in ids)
            self.chunks.append(chunk)

    def directive(self, operation, **kwargs):
        signatures = {"source_chunk_id": self.signature(self.chunks[0]), "target_chunk_id": self.signature(self.chunks[1]), "anchor_chunk_id": self.signature(self.chunks[1])}
        self.result.chunk_policy["chunk_operations"] = [{"operation": operation, "source_chunk_id": "c0", "target_chunk_id": "c1", "anchor_chunk_id": "c1", "signatures": signatures, **kwargs}]

    def signature(self, chunk):
        return {"kind": chunk.content_type, "content": chunk.content, "ids": chunk.source_element_ids.copy()}

    def test_split_preserves_all_source_evidence(self):
        self.directive("SPLIT_CHUNK", split_element_id="p1", split_position="after")
        chunks = apply_chunk_operations(self.result, self.chunks)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(sorted(i for c in chunks for i in c.source_element_ids), ["list", "p1", "p2"])
        for chunk in chunks:
            if chunk.id != "c1":
                self.assertTrue(all(s.location_status == "LOCATABLE" for s in map_chunk_sources(self.result, [chunk])[chunk.id]))

    def test_merge_and_move_do_not_exclude_elements(self):
        self.directive("MOVE_ELEMENT", element_ids=["p2"])
        chunks = apply_chunk_operations(self.result, self.chunks)
        self.assertEqual(chunks[0].source_element_ids, ["p1"])
        self.assertEqual(chunks[1].source_element_ids, ["p2", "list"])
        self.assertIn("必要时调整剂量", chunks[1].content)

    def test_merge(self):
        self.directive("MERGE_CHUNK")
        chunks = apply_chunk_operations(self.result, self.chunks)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_element_ids, ["p1", "p2", "list"])

    def test_order_and_metadata(self):
        self.directive("MOVE_CHUNK_AFTER")
        apply_chunk_operations(self.result, self.chunks, phase="metadata")
        self.assertGreater(self.chunks[0].reading_order, self.chunks[1].reading_order)
        self.directive("UPDATE_METADATA", fields={"summary": "correct", "keywords": ["剂量"]})
        apply_chunk_operations(self.result, self.chunks, phase="metadata")
        self.assertEqual(self.chunks[0].metadata["summary"], "correct")

    def test_parent_order_moves_cross_page_descendants_without_changing_ownership(self):
        a, b = deepcopy(self.chunks[0]), deepcopy(self.chunks[1])
        a.id, b.id = "parent-a", "parent-b"
        a.is_parent = b.is_parent = True
        a.parent_id = b.parent_id = None
        a.content_type = b.content_type = "topic_parent"
        child_a, nested, grandchild, child_b = [deepcopy(self.chunks[0]) for _ in range(4)]
        child_a.id, nested.id, grandchild.id, child_b.id = "a-child", "nested", "grandchild", "b-child"
        child_a.parent_id = nested.parent_id = a.id
        grandchild.parent_id, child_b.parent_id = nested.id, b.id
        nested.is_parent = True
        child_a.page_end = grandchild.page_start = grandchild.page_end = 2
        self.chunks = [a, b, child_a, nested, grandchild, child_b]
        for index, chunk in enumerate(self.chunks):
            chunk.reading_order = index * 100
        self.result.chunk_policy["chunk_operations"] = [{"operation":"MOVE_CHUNK_AFTER", "source_chunk_id":a.id, "anchor_chunk_id":b.id,
            "signatures":{"source_chunk_id":self.signature(a), "anchor_chunk_id":self.signature(b)}}]
        original = {c.id:(c.parent_id,c.content,c.source_element_ids.copy(),c.page_start,c.page_end) for c in self.chunks}
        apply_chunk_operations(self.result, self.chunks, phase="metadata")
        self.assertEqual([c.id for c in sorted(self.chunks,key=lambda c:c.reading_order)], [b.id,child_b.id,a.id,child_a.id,nested.id,grandchild.id])
        self.assertEqual(original, {c.id:(c.parent_id,c.content,c.source_element_ids,c.page_start,c.page_end) for c in self.chunks})

    def test_parent_order_is_executable_in_actual_build_pipeline(self):
        original = build_chunks(self.result, "drug", "product")
        parents = [c for c in original if c.is_parent]
        self.assertGreaterEqual(len(parents), 2)
        source, target = parents[:2]
        self.result.chunk_policy["chunk_operations"] = [{"operation":"MOVE_CHUNK_AFTER", "source_chunk_id":"source", "anchor_chunk_id":"anchor",
            "signatures":{"source_chunk_id":self.signature(source), "anchor_chunk_id":self.signature(target)}}]
        actual = build_chunks(self.result, "drug", "product")
        self.assertEqual(len(actual), len(original))
        by_id = {c.id:c for c in actual}
        self.assertGreater(by_id[source.id].reading_order, by_id[target.id].reading_order)
        target_children = [c for c in actual if c.parent_id == target.id]
        source_children = [c for c in actual if c.parent_id == source.id]
        self.assertTrue(all(c.reading_order < by_id[source.id].reading_order for c in target_children))
        self.assertTrue(all(c.reading_order > by_id[source.id].reading_order for c in source_children))

    def test_empty_split_and_ambiguous_fragments_fail(self):
        self.directive("SPLIT_CHUNK", split_element_id="p1", split_position="before")
        with self.assertRaises(ValueError):
            apply_chunk_operations(self.result, self.chunks)
        self.chunks[0].content = "片段"
        self.directive("MOVE_ELEMENT", element_ids=["p2"])
        with self.assertRaises(ValueError):
            apply_chunk_operations(self.result, self.chunks)

    def test_policy_is_remapped_for_next_clone(self):
        from app.storage.parsed_result import _remap_policy
        policy={"chunk_operations":[{"signatures":{"source_chunk_id":{"ids":["p1","p2"],"content":"unchanged"}},"element_ids":["p1"]}],"heading_parents":{"p2":"p1"}}
        copied=_remap_policy(policy,{"p1":"new1","p2":"new2"})
        self.assertEqual(copied["chunk_operations"][0]["signatures"]["source_chunk_id"]["ids"],["new1","new2"])
        self.assertEqual(copied["heading_parents"],{"new2":"new1"})
        self.assertEqual(policy["heading_parents"],{"p2":"p1"})

    def test_metadata_reaches_actual_built_version(self):
        chunk=next(c for c in build_chunks(self.result,"drug","product") if not c.is_parent and "summary" not in c.content_type)
        self.result.chunk_policy["chunk_operations"]=[{"operation":"UPDATE_METADATA","source_chunk_id":"original","signatures":{"source_chunk_id":self.signature(chunk)},"fields":{"keywords":["核对"]}}]
        built=build_chunks(self.result,"drug","product")
        updated=next(c for c in built if c.metadata.get("keywords")==["核对"])
        self.assertEqual(updated.content,chunk.content)

    def test_heading_parent_changes_path_without_replacing_text(self):
        from app.review.corrections import apply_proposals
        from app.chunker.parent_child import _blocks
        from app.models import TextElement
        ref=deepcopy(self.result.elements[1].ref)
        self.result.elements.extend([TextElement("parent",ref,"subsection_heading","父标题","父标题",3,1),TextElement("child",deepcopy(ref),"subsection_heading","子标题","子标题",4,1)])
        mapping={e.id:e.id for e in self.result.elements}
        apply_proposals(self.result,mapping,[{"id":"proposal","issue_type":"heading_assignment","element_ids":["child"],"suggested":{"parent_id":"parent"}}],"unused",self.result.document_id)
        child=next(e for e in self.result.elements if e.id=="child")
        self.assertEqual(child.raw_text,"子标题")
        blocks=_blocks(self.result.sections[0],self.result.elements,heading_parents=self.result.chunk_policy["heading_parents"])
        self.assertEqual(next(b.path for b in blocks if "child" in b.source_ids),["用法用量","父标题","子标题"])
