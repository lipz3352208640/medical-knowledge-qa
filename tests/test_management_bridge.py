"""Contracts between the existing RAG API and independent management app."""

import unittest
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.generator.answer import AnswerResult
from app.storage.audit import AuditPersistenceError
from app.web.api import create_app


class _Answer:
    def answer(self, _query, *, drug_names=()):
        return AnswerResult("说明书原文回答", [], "medium")


class _Audit:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def record_answer(self, **value):
        self.calls.append(value)
        if self.fail:
            raise AuditPersistenceError("database unavailable")
        return "a" * 32


class BridgeTests(unittest.TestCase):
    def _client(self, audit, *, mysql_url=None):
        return TestClient(create_app(answer_service=_Answer(), retriever=object(),
                                     internal_token="service-secret", audit_store=audit,
                                     audit_required=True, mysql_url=mysql_url))

    def test_answer_is_returned_only_after_audit_commit(self):
        audit = _Audit()
        response = self._client(audit).post("/api/v1/chat/query", json={"query": "药品有什么注意事项"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer_id"], "a" * 32)
        self.assertEqual(audit.calls[0]["status"], "COMPLETED")
        self.assertEqual(len(audit.calls[0]["trace_id"]), 32)

    def test_audit_failure_never_returns_success(self):
        response = self._client(_Audit(fail=True)).post(
            "/api/v1/chat/query", json={"query": "药品有什么注意事项"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("answer_id", response.json())

    def test_internal_endpoints_require_separate_service_token(self):
        client = self._client(_Audit())
        self.assertEqual(client.get("/internal/v1/config-status").status_code, 401)
        response = client.get("/internal/v1/config-status",
                              headers={"X-Service-Token": "service-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audit_status"], "configured")

    def test_reprocess_is_queued_in_mysql(self):
        client = self._client(_Audit())
        with patch("app.web.api.enqueue_reprocess", return_value={"job_id": "job-1", "status": "PENDING"}) as enqueue:
            response = client.post("/internal/v1/parse-runs/run-1/reprocess",
                                   headers={"X-Service-Token": "service-secret"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "PENDING")
        self.assertEqual(enqueue.call_args.args[1], "run-1")

    def test_knowledge_import_requires_service_token_and_queues_pdf(self):
        client = self._client(_Audit(), mysql_url="mysql://unused")
        with tempfile.TemporaryDirectory() as directory, \
             patch("app.web.api.setting", side_effect=lambda key, default=None: directory if key == "KNOWLEDGE_IMPORT_DIR" else default), \
             patch("app.web.api.enqueue_import", return_value={"job_id": "job-1", "status": "PENDING"}) as enqueue:
            path = "/internal/v1/knowledge/imports"
            headers = {"X-Filename": "sample.pdf"}
            self.assertEqual(client.post(path, headers=headers, content=b"%PDF-example").status_code, 401)
            self.assertEqual(client.post(path, headers={**headers, "X-Service-Token": "service-secret"},
                                         content=b"not a pdf").status_code, 422)
            response = client.post(path, headers={**headers, "X-Service-Token": "service-secret"},
                                   content=b"%PDF-example")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["job_id"], "job-1")
            self.assertTrue(enqueue.call_args.args[1].endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()
