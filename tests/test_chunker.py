import unittest

from app.chunker.parent_child import build_chunks, estimated_token_count
from app.models import (
    ImageRecord,
    ParseResult,
    Section,
    SourceRef,
    TableRecord,
    TableRow,
    TextElement,
    ValuePair,
)


class ParentChildChunkerTests(unittest.TestCase):
    def make_result(self):
        document_id = "doc"
        section_id = "dose-section"
        ref = lambda page, bbox=(1, 1, 10, 10): SourceRef(document_id, section_id, page, bbox)
        section = Section(
            section_id,
            document_id,
            "【用法用量】",
            "用法用量",
            1,
            2,
            ref(1),
            1.0,
        )
        elements = [
            TextElement("heading", ref(1), "section_heading", "【用法用量】", "用法用量", 0, 1.0),
            TextElement("p1", ref(1, (1, 20, 100, 30)), "paragraph", "肾功能不全患者需要监测。", "肾功能不全患者需要监测。", 1, 0.9),
            TextElement("p2", ref(1, (1, 31, 100, 40)), "paragraph", "必要时调整剂量。", "必要时调整剂量。", 2, 0.9),
            TextElement("header", ref(2), "header", "说明书页眉", "说明书页眉", 0, 0.9),
            TextElement("list", ref(2), "list_item", "1. 漏服后不要服用双倍剂量。", "1. 漏服后不要服用双倍剂量。", 1, 0.9),
        ]
        table_ref = ref(2, (1, 100, 200, 180))
        row = TableRow(
            "row-1",
            "table-1",
            1,
            ["腹泻", "3级", "暂停用药"],
            ["腹泻", "3级", "暂停用药"],
            [table_ref],
            "测试药；不良反应：腹泻；等级：3级；处理：暂停用药",
        )
        table = TableRecord(
            "table-1",
            document_id,
            section_id,
            "不良反应剂量调整",
            None,
            "| 不良反应 | 等级 | 处理 |\n| --- | --- | --- |\n| 腹泻 | 3级 | 暂停用药 |",
            {"rows": []},
            {"rows": []},
            ["table.png"],
            [table_ref],
            [],
            [row],
        )
        image = ImageRecord(
            "image-1",
            ref(2, (10, 200, 100, 300)),
            "dosage_diagram",
            "dose.png",
            "剂量调整流程图",
            0.8,
        )
        return ParseResult(
            document_id,
            "leaflet.pdf",
            "a" * 64,
            "0.3.0",
            ValuePair("测试药", "测试药"),
            2,
            [section],
            elements,
            [table],
            [image],
            [],
            [],
            brand_names=[ValuePair("测试牌", "测试牌")],
            revision_date="2026-01-01",
        )

    def test_parent_child_relationships_and_metadata(self):
        result = self.make_result()
        chunks = build_chunks(result, "drug-1", "product-1", max_tokens=40, overlap_tokens=5)

        parents = [chunk for chunk in chunks if chunk.is_parent]
        children = [
            chunk for chunk in chunks
            if not chunk.is_parent and chunk.content_type not in {"section_summary", "topic_summary"}
        ]
        summaries = [chunk for chunk in chunks if chunk.content_type.endswith("_summary")]
        self.assertTrue(summaries)
        self.assertTrue(all(chunk.metadata["evidence_role"] == "navigation" for chunk in summaries))
        self.assertEqual(
            {chunk.content_type for chunk in parents},
            {"topic_parent", "table", "row_group_parent", "image_parent"},
        )
        self.assertEqual(
            {chunk.content_type for chunk in children},
            {"paragraph", "list_item", "table_row", "image"},
        )
        parent_ids = {chunk.id for chunk in parents}
        self.assertTrue(all(chunk.parent_id in parent_ids for chunk in children))
        self.assertFalse(any("说明书页眉" in chunk.content for chunk in chunks))

        paragraph = next(chunk for chunk in children if chunk.content_type == "paragraph")
        self.assertEqual(paragraph.metadata["drug_id"], "drug-1")
        self.assertEqual(paragraph.metadata["section"], "用法用量")
        self.assertIn("renal_impairment", paragraph.metadata["populations"])
        self.assertEqual(paragraph.split_reason, "paragraph_boundary")
        self.assertEqual(paragraph.raw_text, paragraph.content)
        self.assertEqual(paragraph.metadata["source_element_ids"], ["p1"])
        self.assertEqual(paragraph.section_path[0], result.sections[0].normalized_name)
        self.assertEqual(paragraph.page_start, 1)
        self.assertIsNotNone(paragraph.bbox)

        table_row = next(chunk for chunk in children if chunk.content_type == "table_row")
        table_parent = next(chunk for chunk in parents if chunk.content_type == "table")
        row_group = next(chunk for chunk in parents if chunk.content_type == "row_group_parent")
        self.assertEqual(table_row.parent_id, row_group.id)
        self.assertEqual(row_group.parent_id, table_parent.id)
        self.assertIn("不良反应剂量调整", table_row.retrieval_text)

    def test_chunk_ids_are_stable_and_long_text_is_split(self):
        result = self.make_result()
        result.elements[1].raw_text = "肾功能不全患者需要调整剂量。" * 8
        result.elements[1].normalized_text = result.elements[1].raw_text
        first = build_chunks(result, "drug-1", "product-1", max_tokens=20, overlap_tokens=4)
        second = build_chunks(result, "drug-1", "product-1", max_tokens=20, overlap_tokens=4)
        self.assertEqual([chunk.id for chunk in first], [chunk.id for chunk in second])
        paragraph_chunks = [
            chunk for chunk in first if not chunk.is_parent and chunk.content_type == "paragraph"
        ]
        self.assertGreater(len(paragraph_chunks), 1)
        self.assertTrue(all(estimated_token_count(chunk.content) <= 20 for chunk in paragraph_chunks))

    def test_subheadings_group_related_facts_into_local_parents(self):
        result = self.make_result()
        result.tables = []
        result.images = []
        ref = lambda page, top: SourceRef("doc", "dose-section", page, (1, top, 100, top + 9))
        result.elements = [
            TextElement("h1", ref(1, 20), "subsection_heading", "腹泻", "腹泻", 1, 1.0),
            TextElement("a", ref(1, 30), "paragraph", "首次发生腹泻3级时暂停给药。", "首次发生腹泻3级时暂停给药。", 2, 0.9),
            TextElement("b", ref(1, 40), "paragraph", "恢复至1级后降低剂量。", "恢复至1级后降低剂量。", 3, 0.9),
            TextElement("h2", ref(1, 50), "subsection_heading", "皮疹", "皮疹", 4, 1.0),
            TextElement("c", ref(2, 20), "paragraph", "皮疹严重时暂停给药。", "皮疹严重时暂停给药。", 1, 0.9),
        ]
        chunks = build_chunks(result, "drug", "product")
        children = {c.source_element_ids[0]: c for c in chunks if not c.is_parent}
        self.assertEqual(children["a"].parent_id, children["b"].parent_id)
        self.assertNotEqual(children["a"].parent_id, children["c"].parent_id)
        self.assertEqual(children["b"].topic_name, "腹泻")
        self.assertEqual(children["b"].section_path, ["用法用量", "腹泻"])
        self.assertEqual(children["b"].previous_chunk_id, children["a"].id)
        self.assertEqual(children["a"].next_chunk_id, children["b"].id)
        self.assertEqual(children["c"].page_start, 2)
        self.assertEqual(children["c"].refs[0].bbox, (1, 20, 100, 29))

    def test_list_lead_in_and_sentence_only_fallback(self):
        result = self.make_result()
        result.tables = []
        result.images = []
        ref = lambda order: SourceRef("doc", "dose-section", 1, (1, order * 10, 100, order * 10 + 9))
        result.elements = [
            TextElement("lead", ref(1), "paragraph", "处理原则：", "处理原则：", 1, 0.9),
            TextElement("item", ref(2), "list_item", "（1）不得服用45 mg。", "（1）不得服用45 mg。", 2, 0.9),
            TextElement("long", ref(3), "paragraph",
                        "腹泻达到3级时暂停给药。" * 5,
                        "腹泻达到3级时暂停给药。" * 5, 3, 0.9),
        ]
        children = [c for c in build_chunks(result, "drug", "product", max_tokens=28, overlap_tokens=15)
                    if not c.is_parent]
        item = next(c for c in children if c.content_type == "list_item")
        self.assertEqual(item.lead_in_id, "lead")
        self.assertIn("45 mg", item.raw_text)
        self.assertIn("不得", item.raw_text)
        long_parts = [c for c in children if "long" in c.source_element_ids]
        self.assertGreater(len(long_parts), 1)
        self.assertTrue(all(c.raw_text.endswith("。") for c in long_parts))
        self.assertTrue(all(c.split_reason == "token_fallback" for c in long_parts))
        self.assertTrue(any(c.overlap_source_ids for c in long_parts[1:]))

    def test_table_row_keeps_raw_cells_and_inherited_group_key(self):
        result = self.make_result()
        result.elements = []
        result.images = []
        table = result.tables[0]
        table.normalized_json["rows"] = [["不良反应", "等级", "处理"]]
        table.rows[0].raw_cells = ["腹泻", "2级", "暂停给药"]
        table.rows[0].normalized_cells = ["腹泻", "2级", "暂停给药"]
        table.rows.append(TableRow(
            "row-2", table.id, 2, [None, "再次2级", "降低剂量"],
            ["腹泻", "再次2级", "降低剂量"],
            [SourceRef("doc", "dose-section", 2, (1, 190, 200, 205))],
            "测试药；不良反应：腹泻；等级：再次2级；处理：降低剂量",
        ))
        rows = [c for c in build_chunks(result, "drug", "product")
                if c.content_type == "table_row"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].parent_id, rows[1].parent_id)
        self.assertIn("不良反应：腹泻", rows[1].retrieval_text)
        self.assertNotIn("腹泻", rows[1].raw_text)
        self.assertEqual(rows[1].page_start, 2)

    def test_long_topic_splits_at_sentences_and_keeps_heading(self):
        result = self.make_result()
        result.tables = []
        result.images = []
        ref = lambda top: SourceRef("doc", "dose-section", 1, (1, top, 100, top + 9))
        result.elements = [
            TextElement("head", ref(20), "subsection_heading", "腹泻", "腹泻", 1, 1.0),
            TextElement("body", ref(30), "paragraph", "首次发生腹泻时暂停给药。" * 8,
                        "首次发生腹泻时暂停给药。" * 8, 2, 0.9),
        ]
        chunks = build_chunks(result, "drug", "product", max_tokens=25,
                              overlap_tokens=0, parent_soft_max_tokens=40)
        parents = [c for c in chunks if c.content_type == "topic_parent"]
        children = [c for c in chunks if not c.is_parent]
        self.assertGreater(len(parents), 1)
        self.assertTrue(all(parent.raw_text.startswith("腹泻\n") for parent in parents))
        self.assertTrue(all(child.raw_text.endswith("。") for child in children))
        self.assertTrue(all(child.parent_id in {parent.id for parent in parents} for child in children))

    def test_indivisible_sentence_is_flagged_without_mid_sentence_cut(self):
        result = self.make_result()
        result.tables = []
        result.images = []
        raw = "肾功能不全患者不得服用45 mg"
        result.elements = [TextElement(
            "one", SourceRef("doc", "dose-section", 1, (1, 20, 100, 29)),
            "paragraph", raw, raw, 1, 0.9,
        )]
        children = [c for c in build_chunks(result, "drug", "product", max_tokens=10,
                                            overlap_tokens=2) if not c.is_parent]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].raw_text, raw)
        self.assertEqual(children[0].split_reason, "oversized_sentence")

    def test_raw_and_normalized_nested_heading_are_traceable(self):
        result = self.make_result()
        result.tables = []
        result.images = []
        ref = SourceRef("doc", "dose-section", 1, (1, 20, 100, 29))
        result.elements = [
            TextElement("nested", ref, "subsection_heading", "严重的不良反应",
                        "严重的不良反应", 1, 1.0),
            TextElement("fact", SourceRef("doc", "dose-section", 1, (1, 30, 100, 39)),
                        "paragraph", "间质性肺病应立即停药。", "间质性肺病应立即停药。", 2, 0.9),
        ]
        child = next(c for c in build_chunks(result, "drug", "product") if not c.is_parent)
        self.assertEqual(child.section_path, ["用法用量", "严重不良反应", "间质性肺病"])
        self.assertEqual(child.metadata["raw_subsection_name"], "严重的不良反应")
        self.assertEqual(child.metadata["normalized_subsection_name"], "严重不良反应")


if __name__ == "__main__":
    unittest.main()
