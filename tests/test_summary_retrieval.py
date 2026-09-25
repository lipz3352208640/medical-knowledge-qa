import json
import unittest
from unittest.mock import MagicMock, patch

from app.storage.retrieval import lexical_search, load_summary_sources, summary_search


class SummaryRetrievalTests(unittest.TestCase):
    def test_summary_hit_resolves_only_original_document_children_in_source_order(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = (
            "document-1", json.dumps({"source_chunk_ids": ["row-b", "row-a"]}),
        )

        def row(chunk_id):
            return (
                chunk_id, "document-1", "section-1", "parent-1",
                "原文", "检索文本", "table_row", 1, 1, None, "[]", "{}",
            )

        cursor.fetchall.return_value = [row("row-a"), row("row-b")]
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with (
            patch("app.storage.retrieval.parse_mysql_url"),
            patch("app.storage.retrieval._connect", return_value=connection),
        ):
            sources = load_summary_sources("mysql://test", "summary-1")

        self.assertEqual([source.id for source in sources], ["row-b", "row-a"])
        sql, params = cursor.execute.call_args_list[1].args
        self.assertIn("document_id=%s", sql)
        self.assertEqual(params, ("document-1", "row-b", "row-a"))

    def test_summary_and_evidence_search_use_separate_channels(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with (
            patch("app.storage.retrieval.parse_mysql_url"),
            patch("app.storage.retrieval._connect", return_value=connection),
        ):
            lexical_search("mysql://test", "腹泻")
            summary_search("mysql://test", "腹泻")

        evidence_sql = cursor.execute.call_args_list[0].args[0]
        summary_sql = cursor.execute.call_args_list[1].args[0]
        self.assertIn("content_type NOT IN ('section_summary','topic_summary')", evidence_sql)
        self.assertIn("content_type IN ('section_summary','topic_summary')", summary_sql)


if __name__ == "__main__":
    unittest.main()
