"""Orchestrate recoverable MySQL-to-Qdrant indexing."""

from dataclasses import asdict, dataclass, field

from app.indexer.embeddings import OllamaEmbedder
from app.storage.retrieval import (
    list_active_document_ids,
    load_vector_chunks,
    mark_inactive_vector_indexes_stale,
    set_vector_index_state,
)
from app.storage.vector_store import QdrantVectorStore


@dataclass(slots=True)
class DocumentIndexResult:
    document_id: str
    status: str
    point_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class IndexBuildReport:
    collection_name: str
    embedding_model: str
    vector_size: int
    recreated_collection: bool
    pruned_point_count: int = 0
    documents: list[DocumentIndexResult] = field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "FAILED" for item in self.documents)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["failed_count"] = self.failed_count
        return value


def build_vector_index(
    mysql_url: str,
    embedder: OllamaEmbedder,
    vector_store: QdrantVectorStore,
    *,
    document_ids: list[str] | None = None,
    recreate_collection: bool = False,
    include_pending: bool = False,
) -> IndexBuildReport:
    """Index active document child chunks and keep failures explicit and recoverable."""
    if include_pending and not document_ids:
        raise ValueError("include_pending requires explicit document_ids")
    active_ids = list_active_document_ids(mysql_url, document_ids, include_pending=include_pending)
    if document_ids:
        missing = sorted(set(document_ids) - set(active_ids))
        if missing:
            raise ValueError(
                "Documents do not exist or are not ACTIVE: " + ", ".join(missing)
            )

    if recreate_collection:
        vector_store.recreate_collection(embedder.dimensions)
    else:
        vector_store.ensure_collection(embedder.dimensions)

    report = IndexBuildReport(
        collection_name=vector_store.collection_name,
        embedding_model=embedder.model,
        vector_size=embedder.dimensions,
        recreated_collection=recreate_collection,
    )
    for document_id in active_ids:
        try:
            set_vector_index_state(
                mysql_url,
                document_id,
                "INDEXING",
                embedding_model=embedder.model,
                collection_name=vector_store.collection_name,
                point_count=0,
            )
            chunks = load_vector_chunks(mysql_url, document_id, include_pending=include_pending)
            if not chunks:
                raise ValueError(
                    "No child chunks found; rerun document ingestion after migration 004"
                )
            # Embed everything before replacing points so an Ollama failure does
            # not remove the document's last usable vector index.
            vectors = embedder.embed([chunk.retrieval_text for chunk in chunks])
            point_count = vector_store.replace_document(
                document_id,
                chunks,
                vectors,
                vector_size=embedder.dimensions,
            )
            set_vector_index_state(
                mysql_url,
                document_id,
                "READY",
                embedding_model=embedder.model,
                collection_name=vector_store.collection_name,
                point_count=point_count,
            )
            report.documents.append(
                DocumentIndexResult(document_id, "READY", point_count)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                set_vector_index_state(
                    mysql_url,
                    document_id,
                    "FAILED",
                    embedding_model=embedder.model,
                    collection_name=vector_store.collection_name,
                    point_count=0,
                    error=error,
                )
            except Exception:
                pass
            report.documents.append(
                DocumentIndexResult(document_id, "FAILED", error=error)
            )

    # A full active-document run is also the cleanup boundary for superseded
    # versions. A targeted run must not delete unrelated points.
    if document_ids is None:
        report.pruned_point_count = vector_store.prune_documents(set(active_ids))
        mark_inactive_vector_indexes_stale(mysql_url)
    return report
