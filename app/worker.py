"""Dedicated slow document worker backed by the RAG MySQL job table."""

from __future__ import annotations

import logging
from app.config import setting
from pathlib import Path
import time

from app.indexer.embeddings import OllamaEmbedder, OllamaEmbeddingConfig
from app.indexer.pipeline import build_vector_index
from app.parser.pdf import PARSER_VERSION, parse_pdf
from app.storage.jobs import claim_job, finish_job, correction_batch, finish_correction_batch
from app.storage.mysql import ingest
from app.storage.parsed_result import load_review_base
from app.review.corrections import apply_proposals
from app.storage.vector_store import QdrantConfig, QdrantVectorStore


def process_one(mysql_url: str) -> bool:
    claimed = claim_job(mysql_url)
    if claimed is None:
        return False
    job_id, kind, _old_document_id, source_file = claimed
    try:
        # A review run has its own parser version and document ID. The prior
        # PDF, parse run and index remain available for audit and rollback.
        parser_version = PARSER_VERSION if kind == "IMPORT" else f"{PARSER_VERSION}.review.{job_id[:8]}"
        if kind == "CORRECTION":
            batch = correction_batch(mysql_url, job_id)
            if batch is None:
                raise ValueError("修正批次不存在")
            result, id_map = load_review_base(mysql_url, batch["parse_run_id"], parser_version)
            if result.file_hash != batch["file_hash"] or result.document_id == batch["document_id"]:
                raise ValueError("修正来源版本不一致")
            apply_proposals(result, id_map, batch["proposals"], mysql_url,
                            batch["document_id"])
            if any(proposal["issue_type"] in {"element_missing", "reading_order",
                      "element_merge_split", "table_structure"} for proposal in batch["proposals"]):
                result.warnings.append("人工重组的解析结构需审核后发布")
        else:
            result = parse_pdf(source_file, Path(setting("PARSER_OUTPUT_DIR", "data/output")),
                               parser_version)
        ingest(result, mysql_url, publish=False)
        finish_job(mysql_url, job_id, "VALIDATING", result={"document_id": result.document_id},
                   document_id=result.document_id)
        embedder = OllamaEmbedder(OllamaEmbeddingConfig(
            base_url=setting("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=setting("EMBEDDING_MODEL", "bge-m3:latest"),
            dimensions=int(setting("EMBEDDING_DIMENSIONS", "1024")),
        ))
        vector_store = QdrantVectorStore(QdrantConfig(
            url=setting("QDRANT_URL", "http://127.0.0.1:6333"),
            api_key=setting("QDRANT_API_KEY"),
            collection_name=setting("QDRANT_COLLECTION", "medical_knowledge_chunks"),
        ))
        report = build_vector_index(mysql_url, embedder, vector_store,
                                    document_ids=[result.document_id], include_pending=True)
        if report.failed_count:
            raise RuntimeError(report.documents[0].error or "index failed")
        status = "NEEDS_REVIEW" if result.failed_pages or result.warnings else "READY"
        finish_job(mysql_url, job_id, status,
                   result={"document_id": result.document_id,
                           "page_count": result.page_count, "index": report.to_dict()})
        if kind == "CORRECTION":
            finish_correction_batch(mysql_url, job_id, result.document_id)
    except Exception as exc:
        logging.exception("Reprocess job %s failed", job_id)
        finish_job(mysql_url, job_id, "FAILED", error=f"{type(exc).__name__}: {exc}")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mysql_url = setting("MYSQL_URL")
    if not mysql_url:
        raise RuntimeError("Set mysql_url in config.yaml before starting the worker")
    while True:
        if not process_one(mysql_url):
            time.sleep(2)


if __name__ == "__main__":
    main()
