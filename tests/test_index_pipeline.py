import unittest
from unittest.mock import patch

from app.indexer.pipeline import build_vector_index
from app.storage.retrieval import StoredChunk


class _Embedder:
    model = "bge-m3:latest"
    dimensions = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _VectorStore:
    collection_name = "chunks"

    def __init__(self):
        self.recreated = False
        self.replaced = []
        self.pruned = None

    def ensure_collection(self, size):
        self.size = size

    def recreate_collection(self, size):
        self.recreated = True
        self.size = size

    def replace_document(self, document_id, chunks, vectors, *, vector_size):
        self.replaced.append((document_id, len(chunks), vector_size))
        return len(chunks)

    def prune_documents(self, active_ids):
        self.pruned = active_ids
        return 2


class IndexPipelineTests(unittest.TestCase):
    def test_indexes_active_documents_and_prunes_stale_points(self):
        chunk = StoredChunk(
            "chunk-1",
            "doc-1",
            "section-1",
            "parent-1",
            "正文",
            "测试药；用法用量；正文",
            "paragraph",
            1,
            1,
            [1, 1, 10, 10],
            [],
            {"drug_id": "drug-1"},
        )
        store = _VectorStore()
        states = []
        with (
            patch("app.indexer.pipeline.list_active_document_ids", return_value=["doc-1"]),
            patch("app.indexer.pipeline.load_vector_chunks", return_value=[chunk]),
            patch("app.indexer.pipeline.set_vector_index_state", side_effect=lambda *a, **k: states.append((a, k))),
            patch("app.indexer.pipeline.mark_inactive_vector_indexes_stale", return_value=1),
        ):
            report = build_vector_index("mysql://unused", _Embedder(), store)

        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.documents[0].point_count, 1)
        self.assertEqual(report.pruned_point_count, 2)
        self.assertEqual(store.replaced, [("doc-1", 1, 3)])
        self.assertEqual(store.pruned, {"doc-1"})
        self.assertEqual([call[0][2] for call in states], ["INDEXING", "READY"])


if __name__ == "__main__":
    unittest.main()
