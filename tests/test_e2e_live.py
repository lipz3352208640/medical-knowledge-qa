"""Opt-in test against real MySQL, Qdrant, reranker and local Ollama."""

import os
import unittest

from app.runtime import build_runtime
from app.storage.retrieval import load_chunks_by_ids


@unittest.skipUnless(
    os.getenv("RUN_LIVE_E2E") == "1" and os.getenv("MYSQL_URL") and os.getenv("RERANKER_MODEL"),
    "Set RUN_LIVE_E2E=1, MYSQL_URL and RERANKER_MODEL for the live chain",
)
class LiveEndToEndTests(unittest.TestCase):
    def test_adverse_overview_keeps_traceable_evidence(self):
        mysql_url = os.environ["MYSQL_URL"]
        service, _ = build_runtime(mysql_url)
        result = service.answer("坎地沙坦酯片的不良反应有哪些")
        self.assertNotEqual(result.confidence, "low", result.warnings)
        self.assertTrue(result.evidence, result.warnings)
        self.assertTrue(any(term in result.answer for term in ("恶心", "头晕", "乏力")))
        sources = {chunk.id: chunk for chunk in load_chunks_by_ids(
            mysql_url, [item["chunk_id"] for item in result.evidence],
        )}
        for citation in result.evidence:
            source = sources[citation["chunk_id"]]
            self.assertEqual(citation["section"], "不良反应")
            self.assertEqual(source.metadata["generic_name"], "坎地沙坦酯片")
            self.assertIn(citation["quote"], source.content)

    def test_table_row_question_reaches_traceable_answer(self):
        mysql_url = os.environ["MYSQL_URL"]
        service, _ = build_runtime(mysql_url)
        result = service.answer("达可替尼发生3级腹泻时如何调整剂量？")
        self.assertTrue(result.evidence, result.warnings)
        self.assertNotEqual(result.confidence, "low", result.warnings)
        self.assertIn("腹泻", result.answer)
        sources = {chunk.id: chunk for chunk in load_chunks_by_ids(
            mysql_url, [item["chunk_id"] for item in result.evidence],
        )}
        self.assertEqual(len(sources), len(result.evidence))
        self.assertTrue(any(chunk.content_type == "table_row" for chunk in sources.values()))
        for citation in result.evidence:
            source = sources[citation["chunk_id"]]
            self.assertIn(citation["quote"], source.content)
            self.assertEqual(source.metadata["section"], citation["section"])
            self.assertEqual(source.metadata["generic_name"], "达可替尼片")
            self.assertEqual(citation["page"], source.page_start)


if __name__ == "__main__":
    unittest.main()
