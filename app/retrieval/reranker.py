"""Pairwise Cross-Encoder scoring for fused evidence candidates."""

from typing import Protocol

from app.storage.retrieval import StoredChunk


class Reranker(Protocol):
    def score(self, query: str, chunks: list[StoredChunk]) -> list[float]: ...


def candidate_text(chunk: StoredChunk) -> str:
    metadata = chunk.metadata
    kind = {
        "paragraph": "正文", "list_item": "列表项", "table_row": "表格行", "image": "图片说明",
    }.get(chunk.content_type, chunk.content_type)
    content = chunk.retrieval_text if chunk.content_type == "table_row" else chunk.content
    return (
        f"[药品] {metadata.get('generic_name') or metadata.get('drug_name') or ''}\n"
        f"[章节] {metadata.get('section') or ''}\n"
        f"[类型] {kind}\n"
        f"[内容] {content}"
    )


class CrossEncoderReranker:
    """Load a local sentence-transformers CrossEncoder only when first used."""

    def __init__(self, model_name: str):
        if not model_name.strip():
            raise ValueError("Cross-Encoder model name is required")
        self.model_name = model_name
        self._model = None

    def score(self, query: str, chunks: list[StoredChunk]) -> list[float]:
        if not chunks:
            return []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install the rerank extra: pip install -e '.[rerank]'") from exc
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, candidate_text(chunk)) for chunk in chunks])
        return [float(score) for score in scores]
