"""Opt-in live Ollama/Qdrant correction workflow against isolated MySQL data."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import pymupdf

from app.config import setting
from app.models import ParseResult, Section, SourceRef, TextElement, ValuePair
from app.parser.layout import stable_id
from app.storage.jobs import enqueue_correction, get_job, publish_import
from app.storage.mysql import _connect, ingest, initialize, parse_mysql_url
from app.worker import process_one


@unittest.skipUnless(os.getenv("LIVE_REVIEW_TEST") == "1" and setting("TEST_MYSQL_URL", test=True),
                     "set LIVE_REVIEW_TEST=1 with test MySQL, Ollama and Qdrant running")
class LiveReviewWorkerTests(unittest.TestCase):
    def test_real_worker_builds_new_vector_version(self):
        url = setting("TEST_MYSQL_URL", test=True)
        initialize(url)
        token = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "review.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.save(pdf)
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            document_id, section_id = token[:24], token[8:32]
            ref = SourceRef(document_id, section_id, 1, (20, 20, 200, 40))
            item = TextElement(token[4:28], ref, "paragraph",
                               "Take one tablet daily as directed.",
                               "Take one tablet daily as directed.", 1, 1.0)
            result = ParseResult(document_id, str(pdf), digest, "live-review-" + token[:8],
                                 ValuePair("ReviewDrug" + token[:8], "ReviewDrug" + token[:8]), 1,
                                 [Section(section_id, document_id, "Warnings", "Warnings", 1, 1, ref, 1.0)],
                                 [item], [], [], [], [])
            ingest(result, url)
            run = stable_id("parse-run", document_id, result.parser_version, digest)
            conn = _connect(parse_mysql_url(url))
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT chunk_manifest_hash FROM document_index_state WHERE document_id=%s", (document_id,))
                    version = cursor.fetchone()[0]
            finally:
                conn.close()
            payload = {"batch_key": hashlib.sha256(token.encode()).hexdigest(),
                       "document_id": document_id, "parse_run_id": run,
                       "index_version": version, "file_hash": digest,
                       "proposals": [{"id": uuid.uuid4().hex, "issue_type": "text_correction",
                                      "element_ids": [item.id], "chunk_ids": [], "pages": [1],
                                      "suggested": {"text": "Take two tablets daily as directed."}}]}
            job = enqueue_correction(url, payload)
            original_setting = setting
            def test_setting(key, default=None):
                if key == "QDRANT_COLLECTION":
                    return "medical_review_worker_test"
                return original_setting(key, default)
            with patch("app.worker.setting", side_effect=test_setting), patch("app.worker.parse_pdf", side_effect=AssertionError("PDF parser must not run")):
                self.assertTrue(process_one(url))
            completed = get_job(url, job["job_id"])
            self.assertEqual(completed["status"], "READY", completed.get("error"))
            self.assertNotEqual(completed["document_id"], document_id)
            self.assertEqual(completed["result"]["index"]["failed_count"], 0)
            self.assertEqual(publish_import(url, job["job_id"])["version_status"], "ACTIVE")
