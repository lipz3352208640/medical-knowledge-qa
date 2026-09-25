import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError as exc:
    raise unittest.SkipTest("Install the web extra and httpx to test the FastAPI UI") from exc

from app.generator.answer import AnswerResult
from app.storage.retrieval import StoredChunk
from app.web.api import create_app


class _Answer:
    def __init__(self):
        self.calls = []

    def answer(self, query, *, drug_names=()):
        self.calls.append((query, drug_names))
        return AnswerResult("说明书原文答案", [], "high", compression=[{"strategies": ["direct_top3"]}])


class _Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, drug_names=()):
        self.calls.append((query, drug_names))
        return type("Result", (), {"to_dict": lambda self: {"query": query, "groups": []}})()


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.answer = _Answer()
        self.retriever = _Retriever()
        self.client = TestClient(create_app(
            answer_service=self.answer, retriever=self.retriever,
            mysql_url="mysql://unused", enable_debug=True,
        ))

    def test_frontend_and_health_are_served(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("从说明书原文", page.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/health").json(), {"status": "ok"})

    def test_chat_passes_drug_filter_and_debug(self):
        response = self.client.post("/api/v1/chat/query", json={
            "query": "  腹泻怎么办？  ",
            "filters": {"drug_names": ["达可替尼"], "document_versions": ["current"]},
            "debug": True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.answer.calls, [("腹泻怎么办？", ("达可替尼",))])
        self.assertEqual(response.json()["answer"], "说明书原文答案")
        self.assertEqual(response.json()["compression"][0]["strategies"], ["direct_top3"])

    def test_rejects_unsupported_versions(self):
        response = self.client.post("/api/v1/chat/query", json={
            "query": "腹泻怎么办？", "filters": {"document_versions": ["old"]},
        })
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.answer.calls)

    def test_debug_endpoint_is_restricted_by_config(self):
        request = {"query": "腹泻怎么办？"}
        self.assertEqual(
            self.client.post("/api/v1/retrieval/debug", json=request).json()["query"],
            "腹泻怎么办？",
        )
        restricted = TestClient(create_app(
            answer_service=self.answer, retriever=self.retriever, enable_debug=False,
        ))
        self.assertEqual(restricted.post("/api/v1/retrieval/debug", json=request).status_code, 404)

    def test_evidence_returns_active_original_and_404_for_missing(self):
        chunk = StoredChunk(
            "chunk-a", "doc-a", "section-a", "parent-a", "原文内容", "检索内容",
            "table_row", 2, 2, [1, 2, 3, 4],
            [{"page": 2, "bbox": [1, 2, 3, 4]}],
            {"generic_name": "甲药", "section": "用法用量", "source_file": "C:/private/甲药.pdf",
             "source_element_ids": ["element-a"], "table_id": "table-a"},
        )
        with patch("app.web.api.load_chunks_by_ids", return_value=[chunk]):
            response = self.client.get("/api/v1/evidence/chunk-a")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["parent_id"], "parent-a")
        self.assertEqual(response.json()["source_file"], "甲药.pdf")
        self.assertEqual(response.json()["content"], "原文内容")
        with patch("app.web.api.load_chunks_by_ids", return_value=[]):
            self.assertEqual(self.client.get("/api/v1/evidence/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
