import hashlib
from pathlib import Path
import tempfile
import unittest
import uuid

import pymupdf

from app.chunker.parent_child import build_chunks
from app.config import setting
from app.indexer.pipeline import build_vector_index
from app.models import ParseResult, Section, SourceRef, TextElement, ValuePair
from app.parser.layout import stable_id
from app.review.corrections import apply_proposals
from app.storage.jobs import enqueue_correction, finish_job, get_job, publish_import
from app.storage.mysql import _connect, ingest, initialize, parse_mysql_url
from app.storage.parsed_result import load_review_base
from app.storage.report import chunk_sources


@unittest.skipUnless(setting("TEST_MYSQL_URL", test=True), "QA test MySQL required")
class ReviewJobIntegrationTests(unittest.TestCase):
    def test_idempotent_batch_new_version_and_atomic_publication(self):
        url = setting("TEST_MYSQL_URL", test=True)
        initialize(url)
        token = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "review.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.save(pdf_path)
            file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            doc_id, section_id = token[:24], token[8:32]
            def ref(y):
                return SourceRef(doc_id, section_id, 1, (20, y, 180, y + 20))
            result = ParseResult(
                doc_id, str(pdf_path), file_hash, "review-base-" + token[:8],
                ValuePair("测试审核药" + token[:8], "测试审核药" + token[:8]), 1,
                [Section(section_id, doc_id, "注意事项", "注意事项", 1, 1, ref(5), 1.0)],
                [TextElement("first" + token[:19], ref(30), "paragraph", "保留的原文。", "保留的原文。", 1, 1.0),
                 TextElement("second" + token[:18], ref(60), "paragraph", "需排除的噪声。", "需排除的噪声。", 2, 1.0)],
                [], [], [], [],
            )
            ingest(result, url)
            run_id = stable_id("parse-run", doc_id, result.parser_version, file_hash)
            conn = _connect(parse_mysql_url(url))
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT chunk_manifest_hash FROM document_index_state WHERE document_id=%s", (doc_id,))
                    index_version = cursor.fetchone()[0]
            finally:
                conn.close()
            proposal = {"id": uuid.uuid4().hex, "issue_type": "chunk_boundary",
                        "element_ids": [result.elements[1].id], "chunk_ids": [], "pages": [1],
                        "suggested": {"exclude_element_ids": [result.elements[1].id]}}
            payload = {"batch_key": hashlib.sha256(token.encode()).hexdigest(),
                       "document_id": doc_id, "parse_run_id": run_id,
                       "index_version": index_version, "file_hash": file_hash,
                       "proposals": [proposal]}
            job = enqueue_correction(url, payload)
            self.assertEqual(job["job_id"], enqueue_correction(url, payload)["job_id"])
            version = "review-correction-" + job["job_id"][:8]
            cloned, ids = load_review_base(url, run_id, version)
            apply_proposals(cloned, ids, [proposal], url, doc_id)
            self.assertEqual(len(cloned.elements), 2)
            self.assertEqual(cloned.chunk_policy["exclude_element_ids"], [ids[result.elements[1].id]])
            self.assertFalse(any("噪声" in chunk.content for chunk in build_chunks(cloned, "drug", "product")))
            ingest(cloned, url, publish=False)
            # A failed rebuild leaves the old document searchable.
            finish_job(url, job["job_id"], "FAILED", error="simulated index failure")
            conn = _connect(parse_mysql_url(url))
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version_status FROM document WHERE id=%s", (doc_id,))
                    self.assertEqual(cursor.fetchone()[0], "ACTIVE")
                    cursor.execute("SELECT version_status FROM document WHERE id=%s", (cloned.document_id,))
                    self.assertEqual(cursor.fetchone()[0], "PENDING")
            finally:
                conn.close()
            class Embedder:
                model = "test-embedder"
                dimensions = 4
                def embed(self, values):
                    return [[1.0, 0.0, 0.0, 0.0] for _ in values]
            class Vectors:
                collection_name = "test-collection"
                def ensure_collection(self, dimensions):
                    self.assert_dimensions = dimensions
                def replace_document(self, document_id, chunks, vectors, *, vector_size):
                    assert document_id == cloned.document_id and len(chunks) == len(vectors)
                    return len(chunks)
            report = build_vector_index(url, Embedder(), Vectors(),
                                        document_ids=[cloned.document_id], include_pending=True)
            self.assertEqual(report.failed_count, 0)
            finish_job(url, job["job_id"], "READY", document_id=cloned.document_id,
                       result={"document_id": cloned.document_id})
            conn = _connect(parse_mysql_url(url))
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM chunk WHERE document_id=%s AND content_type='paragraph' LIMIT 1", (cloned.document_id,))
                    chunk_id = cursor.fetchone()[0]
                    cursor.execute("SELECT chunk_manifest_hash FROM document_index_state WHERE document_id=%s", (cloned.document_id,))
                    new_index = cursor.fetchone()[0]
            finally:
                conn.close()
            new_run = stable_id("parse-run", cloned.document_id, cloned.parser_version, file_hash)
            from app.storage.report import version_snapshot
            snapshot=version_snapshot(url,new_run)
            self.assertEqual(snapshot['document_id'],cloned.document_id)
            self.assertEqual(len(snapshot['pages']),cloned.page_count)
            self.assertTrue(snapshot['pages'][0]['elements'])
            from fastapi.testclient import TestClient
            from app.web.api import create_app
            client=TestClient(create_app(mysql_url=url,internal_token='snapshot-test'))
            path=f'/internal/v1/parse-runs/{new_run}/snapshot'
            self.assertIn(client.get(path).status_code,(401,403))
            response=client.get(path,headers={'X-Service-Token':'snapshot-test'})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['pages'][0]['index_version'],new_index)
            self.assertTrue(chunk_sources(url, chunk_id, new_run, new_index)["locatable"])
            self.assertEqual(publish_import(url, job["job_id"])["version_status"], "ACTIVE")
            conn = _connect(parse_mysql_url(url))
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version_status FROM document WHERE id=%s", (doc_id,))
                    self.assertEqual(cursor.fetchone()[0], "SUPERSEDED")
            finally:
                conn.close()
