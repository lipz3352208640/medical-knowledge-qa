import unittest

from app.storage.retrieval import ChunkFilters, StoredChunk
from app.storage.vector_store import QdrantConfig, QdrantError, QdrantVectorStore, qdrant_point_id


class _FakeStore(QdrantVectorStore):
    def __init__(self):
        super().__init__(QdrantConfig(upload_batch_size=2))
        self.requests = []

    def _collection_info(self):
        return {
            "config": {"params": {"vectors": {"size": 3}}},
            "payload_schema": {
                name: {"data_type": "keyword"}
                for name in (
                    "document_id",
                    "drug_id",
                    "product_id",
                    "section",
                    "content_type",
                    "parent_id",
                )
            },
        }

    def _request(self, method, path, payload=None, **kwargs):
        self.requests.append((method, path, payload))
        return {"status": "ok", "result": {}}


class VectorStoreTests(unittest.TestCase):
    def test_search_applies_qdrant_metadata_filter_and_restores_chunk(self):
        store = _FakeStore()
        original_request = store._request

        def request(method, path, payload=None, **kwargs):
            original_request(method, path, payload, **kwargs)
            return {"status": "ok", "result": {"points": [{
                "score": 0.82,
                "payload": {
                    "chunk_id": "c1", "document_id": "d1", "section_id": "s1",
                    "parent_id": "p1", "content": "正文", "retrieval_text": "检索",
                    "content_type": "paragraph", "page_start": 1, "page_end": 1,
                    "bbox": [1, 2, 3, 4], "refs": [], "section": "不良反应",
                    "drug_id": "drug-1", "index_channel": "evidence",
                    "populations": ["elderly"],
                },
            }]}}

        store._request = request
        hits = store.search([0.1, 0.2, 0.3], ChunkFilters(
            document_ids=("d1",), drug_ids=("drug-1",), sections=("不良反应",),
            populations=("elderly",),
        ))
        self.assertEqual(hits[0].chunk.id, "c1")
        self.assertEqual(hits[0].score, 0.82)
        payload = store.requests[-1][2]
        self.assertEqual(payload["limit"], 20)
        self.assertEqual(payload["filter"]["must"][0]["match"]["value"], "evidence")
        self.assertIn({"key": "drug_id", "match": {"any": ["drug-1"]}}, payload["filter"]["must"])

    def test_search_rejects_cross_drug_payload(self):
        store = _FakeStore()
        store._request = lambda *_args, **_kwargs: {
            "result": {"points": [{"score": 0.9, "payload": {
                "drug_id": "other-drug", "index_channel": "evidence",
            }}]}
        }
        with self.assertRaisesRegex(QdrantError, "outside drug_id filter"):
            store.search([0.1], ChunkFilters(drug_ids=("drug-1",)))

    def test_qdrant_point_id_is_stable_uuid(self):
        first = qdrant_point_id("abc123")
        self.assertEqual(first, qdrant_point_id("abc123"))
        self.assertNotEqual(first, qdrant_point_id("abc124"))
        self.assertEqual(len(first), 36)

    def test_replace_document_deletes_old_points_then_upserts_payload(self):
        chunk = StoredChunk(
            "chunk-1",
            "doc-1",
            "section-1",
            "parent-1",
            "正文",
            "测试药；正文",
            "paragraph",
            1,
            1,
            [1, 1, 10, 10],
            [{"page": 1}],
            {"drug_id": "drug-1", "product_id": "product-1", "section": "用法用量"},
        )
        store = _FakeStore()
        count = store.replace_document("doc-1", [chunk], [[1, 0, 0]], vector_size=3)

        self.assertEqual(count, 1)
        delete_request = next(item for item in store.requests if "/points/delete" in item[1])
        self.assertEqual(
            delete_request[2]["filter"]["must"][0]["match"]["value"],
            "doc-1",
        )
        upsert_request = next(item for item in store.requests if "/points?" in item[1])
        point = upsert_request[2]["points"][0]
        self.assertEqual(point["payload"]["chunk_id"], "chunk-1")
        self.assertEqual(point["payload"]["parent_id"], "parent-1")


if __name__ == "__main__":
    unittest.main()
