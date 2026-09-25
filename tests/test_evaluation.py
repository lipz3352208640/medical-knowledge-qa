import json
from pathlib import Path
import unittest

from app.evaluation.metrics import average_metrics, coverage_counts, ranking_metrics


class EvaluationTests(unittest.TestCase):
    def test_ranking_metrics_credit_all_relevant_chunks(self):
        scores = ranking_metrics(["noise", "gold-b", "noise-2", "gold-a"], ["gold-a", "gold-b"])
        self.assertEqual(scores["recall_at_1"], 0)
        self.assertEqual(scores["recall_at_3"], 0.5)
        self.assertEqual(scores["recall_at_5"], 1)
        self.assertEqual(scores["mrr"], 0.5)
        self.assertGreater(scores["ndcg_at_5"], 0)
        self.assertLess(scores["ndcg_at_5"], 1)

    def test_empty_retrieval_is_zero_and_missing_values_are_excluded(self):
        self.assertEqual(ranking_metrics([], ["gold"])["mrr"], 0)
        self.assertEqual(
            average_metrics([{"score": 1.0}, {"score": None}, {"score": 0.0}], ["score"]),
            {"score": 0.5},
        )

    def test_golden_set_has_unique_ids_and_multiple_drugs(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "golden.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [case["id"] for case in data["retrieval_cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(coverage_counts(data["retrieval_cases"])), 5)
        self.assertTrue(data["comparison_cases"])
        self.assertTrue(any(not case["answerable"] for case in data["answer_cases"]))


if __name__ == "__main__":
    unittest.main()
