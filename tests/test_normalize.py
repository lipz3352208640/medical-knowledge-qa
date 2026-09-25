import unittest

from app.parser.normalize import extract_values, normalize_drug_alias, normalize_drug_name, normalize_section


class NormalizeTests(unittest.TestCase):
    def test_sections_preserve_variant_in_caller(self):
        self.assertEqual(normalize_section("【适应证】"), "适应症")
        self.assertEqual(normalize_section("【成份】"), "成分")

    def test_dose_pairs(self):
        values = extract_values("每日 45 mg；3级；≤ 2.5 mL")
        self.assertIn(("45 mg", "45mg"), [(x.raw, x.normalized) for x in values])
        self.assertIn(("3级", "3级"), [(x.raw, x.normalized) for x in values])

    def test_drug_name(self):
        self.assertEqual(normalize_drug_name(" 奥 美 拉 唑 "), "奥美拉唑")
        self.assertEqual(normalize_drug_alias("VIZIMPRO®"), "vizimpro")


if __name__ == "__main__":
    unittest.main()
