import json
import unittest
from unittest.mock import patch

from app.indexer.embeddings import EmbeddingError, OllamaEmbedder, OllamaEmbeddingConfig


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OllamaEmbedderTests(unittest.TestCase):
    def test_batches_and_validates_embeddings(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            inputs = json.loads(request.data.decode("utf-8"))["input"]
            return _Response({"embeddings": [[1, 0, 0] for _ in inputs]})

        embedder = OllamaEmbedder(
            OllamaEmbeddingConfig(dimensions=3, batch_size=2, timeout_seconds=5)
        )
        with patch("app.indexer.embeddings.urlopen", side_effect=fake_urlopen):
            vectors = embedder.embed(["一", "二", "三"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0][1], 5)
        self.assertTrue(requests[0][0].full_url.endswith("/api/embed"))

    def test_rejects_wrong_vector_size(self):
        embedder = OllamaEmbedder(OllamaEmbeddingConfig(dimensions=3))
        with patch(
            "app.indexer.embeddings.urlopen",
            return_value=_Response({"embeddings": [[1, 0]]}),
        ):
            with self.assertRaises(EmbeddingError):
                embedder.embed(["文本"])


if __name__ == "__main__":
    unittest.main()
