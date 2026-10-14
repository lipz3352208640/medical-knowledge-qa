import unittest
from app.models import ParseResult, SourceRef, TextElement, ValuePair, Section
from app.review.corrections import apply_proposals


class ElementStructureCorrectionTests(unittest.TestCase):
    def base(self):
        items = [TextElement(f"new-{i}", SourceRef('new', 'section', 1, (0, i*10, 100, i*10+8)),
                             'paragraph', str(i), str(i), i, .9) for i in range(1, 5)]
        result = ParseResult('new', 'none', 'hash', 'v', ValuePair('test','test'), 2,
                             [], items, [], [], [], [])
        return result, {f"old-{i}": f"new-{i}" for i in range(1, 5)}

    def apply(self, result, ids, kind, targets, suggested):
        apply_proposals(result, ids, [{"id": "proposal", "issue_type": kind,
                        "element_ids": targets, "suggested": suggested}], '', 'old')

    def test_move_only_current_preserves_other_elements(self):
        for source, anchor, operation, expected in [('old-2','old-4','MOVE_AFTER',[1,3,4,2]),
                                                   ('old-4','old-1','MOVE_BEFORE',[4,1,2,3])]:
            result, ids = self.base()
            self.apply(result, ids, 'reading_order', [source, anchor],
                       {"operation": operation, "move_id": source, "reference_id": anchor})
            self.assertEqual([e.raw_text for e in result.elements], list(map(str, expected)))
            self.assertEqual([e.reading_order for e in result.elements], [1,2,3,4])

    def test_move_rejects_cross_page_and_self(self):
        for anchor in ['old-2','old-3']:
            result, ids = self.base()
            result.elements[2].ref.page = 2
            with self.assertRaises(ValueError):
                self.apply(result, ids, 'reading_order', ['old-2',anchor],
                           {"operation": 'MOVE_AFTER', "move_id": 'old-2', "reference_id": anchor})

    def test_merge_keeps_first_and_union_bbox(self):
        result, ids = self.base()
        self.apply(result, ids, 'element_merge_split', ['old-2','old-3'], {"merge": True})
        self.assertEqual([e.raw_text for e in result.elements], ['1','2\n3','4'])
        self.assertEqual(result.elements[1].id, 'new-2')
        self.assertEqual(result.elements[1].ref.bbox, (0,20,100,38))

    def test_split_keeps_parts_contiguous_before_next_paragraph(self):
        result, ids = self.base()
        self.apply(result, ids, 'element_merge_split', ['old-2'], {"split_texts": ['成人。','儿童。','老人。']})
        self.assertEqual([e.raw_text for e in result.elements], ['1','成人。','儿童。','老人。','3','4'])
        self.assertEqual([e.reading_order for e in result.elements], [1,2,3,4,5,6])
        self.assertEqual(result.chunk_policy.get('force_split_after_ids',[]), [])

    def test_split_preserves_explicit_chunk_end_after_final_part(self):
        result,ids=self.base()
        result.chunk_policy={'force_split_after_ids':['new-2','new-4']}
        self.apply(result,ids,'element_merge_split',['old-2'],{'split_texts':['甲','乙']})
        extra=result.elements[2].id
        self.assertEqual(set(result.chunk_policy['force_split_after_ids']),{'new-4',extra})

    def test_real_chunk_builder_keeps_split_paragraph_in_same_chunk(self):
        from app.chunker.parent_child import build_chunks
        result,ids=self.base()
        result.sections=[Section('section','new','说明','说明',1,2,SourceRef('new','section',1,(0,0,100,10)),1.0)]
        for element in result.elements:
            element.raw_text=element.normalized_text='普通说明内容。'
        result.elements[1].raw_text=result.elements[1].normalized_text='成人患者。儿童患者。'
        children=lambda:[c for c in build_chunks(result,'drug','product') if not c.is_parent and 'summary' not in c.content_type]
        before=children()
        self.apply(result,ids,'element_merge_split',['old-2'],{'split_texts':['成人患者。','儿童患者。']})
        after=children()
        self.assertEqual(len(after),len(before))
        added=result.elements[2].id
        self.assertTrue(any({'new-2',added}<=set(c.source_element_ids) for c in after))

    def test_legacy_full_order_remains_supported(self):
        result, ids = self.base()
        self.apply(result, ids, 'reading_order', ['old-1','old-2'], {"ordered_element_ids": ['old-2','old-1']})
        self.assertEqual([e.raw_text for e in result.elements], ['2','1','3','4'])

    def test_split_persists_edited_text_in_new_version(self):
        result, ids = self.base()
        result.elements[1].raw_text = '成人。儿童。'
        self.apply(result, ids, 'element_merge_split', ['old-2'],
                   {"split_texts": ['成人患者。', '儿童及青少年。']})
        self.assertEqual([e.raw_text for e in result.elements],
                         ['1', '成人患者。', '儿童及青少年。', '3', '4'])
        self.assertEqual(result.elements[1].normalized_text, '成人患者。')
        self.assertEqual(result.elements[2].normalized_text, '儿童及青少年。')
