"""Minimal Ollama embedding client with strict response validation."""

from dataclasses import dataclass
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "bge-m3:latest"
    dimensions: int = 1024
    batch_size: int = 32
    timeout_seconds: float = 120.0


class OllamaEmbedder:
    def __init__(self, config: OllamaEmbeddingConfig | None = None):
        self.config = config or OllamaEmbeddingConfig()
        if self.config.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.config.batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not contain blank text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.config.batch_size]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {
                "model": self.config.model,
                "input": texts,
                "truncate": True,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.config.base_url.rstrip('/')}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"Ollama embedding request failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 0} "
                f"embeddings for {len(texts)} inputs"
            )
        vectors: list[list[float]] = []
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or len(embedding) != self.config.dimensions:
                actual = len(embedding) if isinstance(embedding, list) else 0
                raise EmbeddingError(
                    f"Embedding {index} has {actual} dimensions; "
                    f"expected {self.config.dimensions}"
                )
            vector = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError(f"Embedding {index} contains a non-finite value")
            vectors.append(vector)
        return vectors
