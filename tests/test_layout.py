import unittest

from app.parser.layout import Line, build_sections_and_elements, section_at


class LayoutTests(unittest.TestCase):
    def test_heading_and_traceability(self):
        lines = [
            Line(1, (10, 40, 90, 54), "【适应证】", 12, 0),
            Line(1, (10, 60, 180, 75), "治疗示例 45 mg", 10, 1),
            Line(2, (10, 40, 90, 54), "【用法用量】", 12, 0),
        ]
        sections, elements = build_sections_and_elements("doc", lines, {1: 800, 2: 800})
        self.assertEqual([s.normalized_name for s in sections], ["未分类", "适应症", "用法用量"])
        self.assertEqual(elements[1].ref.section_id, sections[1].id)
        self.assertEqual(elements[1].ref.bbox, (10, 60, 180, 75))
        self.assertEqual(section_at(sections, 2, 60), sections[2].id)
        self.assertEqual(sections[1].end_page, 1)

    def test_heading_and_body_on_same_line(self):
        sections, elements = build_sections_and_elements(
            "doc", [Line(1, (10, 100, 300, 115), "【禁忌】对本品过敏者禁用", 10, 0)], {1: 800})
        self.assertEqual(sections[1].normalized_name, "禁忌")
        self.assertEqual(elements[0].raw_text, "【禁忌】对本品过敏者禁用")

    def test_table_reference_is_not_a_section_heading(self):
        lines = [Line(1, (10, 100, 100, 115), "【用法用量】", 10, 0),
                 Line(1, (20, 200, 130, 215), "【注意事项】", 10, 1)]
        sections, elements = build_sections_and_elements("doc", lines, {1: 800}, [(1, (10, 180, 300, 300))])
        self.assertEqual([s.normalized_name for s in sections], ["未分类", "用法用量"])
        self.assertEqual(elements[1].element_type, "table_text")

    def test_inline_section_reference_is_not_heading(self):
        lines = [Line(1, (10, 100, 100, 115), "【用法用量】", 10, 0),
                 Line(1, (10, 140, 300, 155), "【注意事项】中的数据反映了患者情况", 10, 1),
                 Line(1, (10, 180, 100, 195), "【药代动力学】）。", 10, 2)]
        sections, _ = build_sections_and_elements("doc", lines, {1: 800})
        self.assertEqual([s.normalized_name for s in sections], ["未分类", "用法用量"])

    def test_numbered_chinese_and_english_section_names(self):
        lines = [
            Line(1, (10, 100, 200, 115), "6. 不良反应", 12, 0),
            Line(1, (10, 130, 200, 145), "腹泻发生率较高。", 10, 1),
            Line(2, (10, 100, 240, 115), "6 ADVERSE REACTIONS", 12, 0),
        ]
        sections, _ = build_sections_and_elements("doc", lines, {1: 800, 2: 800})
        self.assertEqual([s.normalized_name for s in sections[1:]], ["不良反应", "不良反应"])
        self.assertEqual(sections[1].raw_name, "6. 不良反应")


if __name__ == "__main__":
    unittest.main()
