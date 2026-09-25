import unittest

from app.parser.tables import Fragment, build_table, group_fragments, normalize_rows


class TableTests(unittest.TestCase):
    def test_controlled_fill(self):
        raw = [["不良反应", "等级", "处理"], ["腹泻", "2级", "暂停"], [None, "3级", None]]
        normalized = normalize_rows(raw, {(2, 0), (2, 2)})
        self.assertEqual(normalized[2], ["腹泻", "3级", None])
        self.assertIsNone(raw[2][0])

    def test_cross_page_header_dedup_and_refs(self):
        a = Fragment(1, 800, (20, 600, 500, 790), [["事件", "等级"], ["腹泻", "2级"]],
                     [[(20, 600, 250, 630), (250, 600, 500, 630)], [(20, 630, 250, 660), (250, 630, 500, 660)]],
                     "a.png", "section", "lines")
        b = Fragment(2, 800, (20, 20, 500, 180), [["事件", "等级"], ["腹泻", "3级"]],
                     [[(20, 20, 250, 50), (250, 20, 500, 50)], [(20, 50, 250, 80), (250, 50, 500, 80)]],
                     "b.png", "section", "lines")
        groups = group_fragments([a, b])
        self.assertEqual(len(groups), 1)
        table = build_table(groups[0], "doc", "达可替尼")
        self.assertEqual(len(table.raw_json["rows"]), 3)
        self.assertEqual([r.refs[0].page for r in table.rows], [1, 2])
        self.assertEqual(len(table.image_paths), 2)

    def test_cross_page_without_repeated_header(self):
        a = Fragment(5, 800, (78, 625, 517, 750), [["药品名", "临床表现"], ["保钾利尿药", "升高"]],
                     [[(78, 625, 200, 650), (200, 625, 517, 650)],
                      [(78, 650, 200, 750), (200, 650, 517, 750)]], "a.png", "interaction", "lines")
        b = Fragment(6, 800, (78, 70, 517, 530), [["利尿降压药", "低血压"], ["锂剂", "锂毒性"]],
                     [[(78, 70, 200, 100), (200, 70, 517, 100)],
                      [(78, 100, 200, 140), (200, 100, 517, 140)]], "b.png", "interaction", "lines")
        table = build_table(group_fragments([a, b])[0], "doc", "坎地沙坦酯")
        self.assertEqual(len(table.rows), 3)
        self.assertEqual(table.rows[-1].refs[0].page, 6)
        self.assertTrue(any("无重复表头" in warning for warning in table.warnings))

    def test_cross_page_inherits_key_and_deduplicates_repeated_data_row(self):
        a = Fragment(
            1, 800, (20, 600, 500, 790),
            [["事件", "等级"], ["腹泻", "2级"]],
            [[(20, 600, 250, 630), (250, 600, 500, 630)],
             [(20, 630, 250, 660), (250, 630, 500, 660)]],
            "a.png", "section", "lines",
        )
        b = Fragment(
            2, 800, (20, 20, 500, 180),
            [["事件", "等级"], [None, "2级"], [None, "3级"]],
            [[(20, 20, 250, 50), (250, 20, 500, 50)],
             [None, (250, 50, 500, 80)],
             [None, (250, 80, 500, 110)]],
            "b.png", "section", "lines",
        )
        table = build_table([a, b], "doc", "达可替尼")
        self.assertEqual([row.normalized_cells for row in table.rows],
                         [["腹泻", "2级"], ["腹泻", "3级"]])
        self.assertEqual(table.rows[-1].raw_cells, [None, "3级"])
        self.assertTrue(any("重复数据行" in warning for warning in table.warnings))


if __name__ == "__main__":
    unittest.main()
