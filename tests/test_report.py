import unittest

from app.models import ParseResult, ValuePair
from app.report.quality import make_report


class ReportTests(unittest.TestCase):
    def test_empty_page_flag(self):
        result = ParseResult("doc", "example.pdf", "hash", "1", ValuePair("药", "药"), 2,
                             [], [], [], [], ["review"], [2])
        report = make_report(result)
        self.assertTrue(report["pages"][1]["review_required"])
        self.assertEqual(report["file_hash"], "hash")


if __name__ == "__main__":
    unittest.main()
