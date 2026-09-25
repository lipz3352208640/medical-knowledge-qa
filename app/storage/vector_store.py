"""Qdrant REST storage for dense child-chunk vectors."""

from dataclasses import dataclass
import json
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.storage.retrieval import ChunkFilters, StoredChunk


_POINT_NAMESPACE = uuid.UUID("38e47755-6830-51b8-86b2-99b767091233")


def qdrant_point_id(chunk_id: str) -> str:
    """Map the project's 24-character chunk id to a stable Qdrant UUID."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"medical-knowledge-qa:{chunk_id}"))


class QdrantError(RuntimeError):
    pass


@dataclass(slots=True)
class VectorHit:
    chunk: StoredChunk
    score: float


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    url: str = "http://127.0.0.1:6333"
    api_key: str | None = None
    collection_name: str = "medical_knowledge_chunks"
    timeout_seconds: float = 30.0
    upload_batch_size: int = 64


class QdrantVectorStore:
    def __init__(self, config: QdrantConfig | None = None):
        self.config = config or QdrantConfig()
        if self.config.upload_batch_size <= 0:
            raise ValueError("upload_batch_size must be positive")

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    @property
    def _collection_path(self) -> str:
        return f"/collections/{quote(self.collection_name, safe='')}"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        allow_not_found: bool = False,
    ) -> dict | None:
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["api-key"] = self.config.api_key
        request = Request(
            f"{self.config.url.rstrip('/')}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise QdrantError(f"Qdrant request failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise QdrantError(f"Qdrant request failed: {exc}") from exc
        if not body:
            return {}
        try:
            result = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise QdrantError("Qdrant returned invalid JSON") from exc
        if result.get("status") not in (None, "ok"):
            raise QdrantError(f"Qdrant operation failed: {result.get('status')}")
        return result

    def _collection_info(self) -> dict | None:
        response = self._request("GET", self._collection_path, allow_not_found=True)
        return response.get("result") if response else None

    def ensure_collection(self, vector_size: int) -> None:
        info = self._collection_info()
        if info is None:
            self._request(
                "PUT",
                self._collection_path,
                {"vectors": {"size": vector_size, "distance": "Cosine"}},
            )
            info = self._collection_info() or {}
        else:
            vectors = (
                info.get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
            actual_size = vectors.get("size")
            if actual_size != vector_size:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} uses vector size "
                    f"{actual_size}; expected {vector_size}. Rebuild the collection."
                )

        payload_schema = info.get("payload_schema", {})
        for field_name in (
            "document_id",
            "drug_id",
            "product_id",
            "section",
            "content_type",
            "index_channel",
            "parent_id",
            "populations",
        ):
            if field_name not in payload_schema:
                self._request(
                    "PUT",
                    f"{self._collection_path}/index?wait=true",
                    {"field_name": field_name, "field_schema": "keyword"},
                )

    def recreate_collection(self, vector_size: int) -> None:
        if self._collection_info() is not None:
            self._request("DELETE", self._collection_path)
        self.ensure_collection(vector_size)

    def replace_document(
        self,
        document_id: str,
        chunks: list[StoredChunk],
        vectors: list[list[float]],
        *,
        vector_size: int,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for index, vector in enumerate(vectors):
            if len(vector) != vector_size:
                raise ValueError(
                    f"vector {index} has {len(vector)} dimensions; expected {vector_size}"
                )

        self.ensure_collection(vector_size)
        self.delete_document(document_id)
        if not chunks:
            return 0

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "parent_id": chunk.parent_id,
                "content_type": chunk.content_type,
                "content": chunk.content,
                "retrieval_text": chunk.retrieval_text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "bbox": chunk.bbox,
                "refs": chunk.refs,
            }
            points.append(
                {
                    "id": qdrant_point_id(chunk.id),
                    "vector": vector,
                    "payload": payload,
                }
            )
        for start in range(0, len(points), self.config.upload_batch_size):
            self._request(
                "PUT",
                f"{self._collection_path}/points?wait=true",
                {"points": points[start : start + self.config.upload_batch_size]},
            )
        return len(points)

    def delete_document(self, document_id: str) -> None:
        if self._collection_info() is None:
            return
        self._request(
            "POST",
            f"{self._collection_path}/points/delete?wait=true",
            {
                "filter": {
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            },
        )

    def search(
        self,
        vector: list[float],
        filters: ChunkFilters | None = None,
        *,
        limit: int = 20,
        channel: str = "evidence",
    ) -> list[VectorHit]:
        """Query Qdrant's dense index with the same metadata scope as MySQL."""
        if not vector or limit <= 0:
            raise ValueError("vector and positive limit are required")
        if channel not in {"evidence", "summary"}:
            raise ValueError(f"Unsupported index channel: {channel}")
        selected = filters or ChunkFilters()
        must: list[dict] = [
            {"key": "index_channel", "match": {"value": channel}}
        ]
        for field, values in (
            ("document_id", selected.document_ids),
            ("drug_id", selected.drug_ids),
            ("section", selected.sections),
            ("content_type", selected.content_types),
        ):
            if values:
                must.append({"key": field, "match": {"any": list(values)}})
        if selected.populations:
            must.append({"key": "populations", "match": {"any": list(selected.populations)}})
        response = self._request(
            "POST",
            f"{self._collection_path}/points/query",
            {
                "query": vector,
                "filter": {"must": must},
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
            },
        ) or {}
        points = response.get("result", {}).get("points", [])
        hits: list[VectorHit] = []
        for point in points:
            payload = point.get("payload") or {}
            try:
                for field, values in (
                    ("document_id", selected.document_ids),
                    ("drug_id", selected.drug_ids),
                    ("section", selected.sections),
                    ("content_type", selected.content_types),
                ):
                    if values and payload.get(field) not in values:
                        raise QdrantError(f"Qdrant returned a hit outside {field} filter")
                if payload.get("index_channel") != channel:
                    raise QdrantError("Qdrant returned a hit outside index channel")
                if selected.populations and not set(payload.get("populations") or []).intersection(selected.populations):
                    raise QdrantError("Qdrant returned a hit outside populations filter")
                chunk = StoredChunk(
                    id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    section_id=payload["section_id"],
                    parent_id=payload.get("parent_id"),
                    content=payload["content"],
                    retrieval_text=payload["retrieval_text"],
                    content_type=payload["content_type"],
                    page_start=payload["page_start"],
                    page_end=payload["page_end"],
                    bbox=payload.get("bbox"),
                    refs=payload["refs"],
                    metadata=payload,
                )
                hits.append(VectorHit(chunk=chunk, score=float(point["score"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise QdrantError("Qdrant returned an incomplete search hit") from exc
        return hits

    def prune_documents(self, active_document_ids: set[str]) -> int:
        """Delete points belonging to document versions that are no longer active."""
        if self._collection_info() is None:
            return 0
        offset = None
        stale_ids: list[object] = []
        while True:
            payload = {
                "limit": 256,
                "with_payload": ["document_id"],
                "with_vector": False,
            }
            if offset is not None:
                payload["offset"] = offset
            response = self._request(
                "POST",
                f"{self._collection_path}/points/scroll",
                payload,
            ) or {}
            result = response.get("result", {})
            stale_ids.extend(
                record["id"]
                for record in result.get("points", [])
                if (record.get("payload") or {}).get("document_id")
                not in active_document_ids
            )
            offset = result.get("next_page_offset")
            if offset is None:
                break
        for start in range(0, len(stale_ids), self.config.upload_batch_size):
            self._request(
                "POST",
                f"{self._collection_path}/points/delete?wait=true",
                {"points": stale_ids[start : start + self.config.upload_batch_size]},
            )
        return len(stale_ids)
