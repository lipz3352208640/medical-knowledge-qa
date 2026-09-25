"""Local parsing, indexing, retrieval and answer commands."""

import argparse
import json
import os
from pathlib import Path

from app.indexer.embeddings import OllamaEmbedder, OllamaEmbeddingConfig
from app.indexer.pipeline import build_vector_index
from app.generator.answer import AnswerService
from app.generator.ollama import OllamaChatClient, OllamaChatConfig
from app.parser.pdf import PARSER_VERSION, parse_pdf
from app.retrieval.compression import ContextCompressor
from app.retrieval.pipeline import RetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rewrite import LLMQueryRewriter
from app.report.quality import write_report
from app.storage.mysql import ingest, initialize
from app.storage.vector_store import QdrantConfig, QdrantVectorStore


def _retrieval_dependencies(args):
    embedder = OllamaEmbedder(
        OllamaEmbeddingConfig(
            base_url=args.ollama_url,
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
            batch_size=args.embedding_batch_size,
        )
    )
    vector_store = QdrantVectorStore(
        QdrantConfig(
            url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            collection_name=args.qdrant_collection,
        )
    )
    chat = OllamaChatClient(OllamaChatConfig(
        base_url=args.ollama_url, model=args.llm_model,
        timeout_seconds=float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "120")),
    ))
    rewrite_enabled = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() not in {
        "0", "false", "no", "off",
    }
    engine = RetrievalEngine(
        args.mysql_url, embedder, vector_store,
        CrossEncoderReranker(args.reranker_model),
        query_rewriter=LLMQueryRewriter(chat) if rewrite_enabled else None,
    )
    return embedder, engine, chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, index and answer from Chinese drug leaflets")
    parser.add_argument(
        "command",
        choices=["parse", "parse-samples", "init-db", "index", "rebuild-index", "retrieve", "answer"],
    )
    parser.add_argument("files", nargs="*", help="PDF files for the parse command")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--mysql-url", default=os.getenv("MYSQL_URL"))
    parser.add_argument("--parser-version", default=os.getenv("PARSER_VERSION", PARSER_VERSION))
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--query", help="Chinese medical question for retrieve or answer")
    parser.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "qwen3.5:latest"))
    parser.add_argument("--debug", action="store_true", help="Show compression decisions for answer")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("QDRANT_COLLECTION", "medical_knowledge_chunks"),
    )
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"))
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    )
    args = parser.parse_args()
    if args.command in {"retrieve", "answer"}:
        if not args.mysql_url:
            parser.error("MYSQL_URL or --mysql-url is required")
        if not args.query:
            parser.error("--query is required for retrieve or answer")
        if not args.reranker_model:
            parser.error("RERANKER_MODEL or --reranker-model is required for Cross-Encoder reranking")
        embedder, engine, chat = _retrieval_dependencies(args)
        if args.command == "retrieve":
            print(json.dumps(engine.retrieve(args.query).to_dict(), ensure_ascii=False))
        else:
            service = AnswerService(engine, ContextCompressor(embedder, chat), chat)
            print(json.dumps(service.answer(args.query).to_dict(debug=args.debug), ensure_ascii=False))
        return
    if args.command == "init-db":
        if not args.mysql_url:
            parser.error("MYSQL_URL or --mysql-url is required")
        initialize(args.mysql_url)
        print("Database schema initialized")
        return
    if args.command in {"index", "rebuild-index"}:
        if not args.mysql_url:
            parser.error("MYSQL_URL or --mysql-url is required")
        embedder = OllamaEmbedder(
            OllamaEmbeddingConfig(
                base_url=args.ollama_url,
                model=args.embedding_model,
                dimensions=args.embedding_dimensions,
                batch_size=args.embedding_batch_size,
            )
        )
        vector_store = QdrantVectorStore(
            QdrantConfig(
                url=args.qdrant_url,
                api_key=args.qdrant_api_key,
                collection_name=args.qdrant_collection,
            )
        )
        report = build_vector_index(
            args.mysql_url,
            embedder,
            vector_store,
            document_ids=args.document_ids,
            recreate_collection=args.command == "rebuild-index",
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False))
        if report.failed_count:
            raise SystemExit(1)
        return
    files = args.files
    if args.command == "parse-samples":
        manifest = Path(__file__).resolve().parents[1] / "samples.json"
        files = json.loads(manifest.read_text(encoding="utf-8"))
    if not files:
        parser.error("At least one PDF is required")
    for path in files:
        result = parse_pdf(path, args.output, args.parser_version)
        parsed, report = write_report(result, args.output)
        status = "parsed"
        if args.mysql_url:
            status = "inserted" if ingest(result, args.mysql_url) else "already present"
        print(json.dumps({"file": path, "status": status, "document_id": result.document_id,
                          "pages": result.page_count, "sections": len(result.sections),
                          "tables": len(result.tables), "images": len(result.images),
                          "failed_pages": result.failed_pages, "blank_pages": result.blank_pages,
                          "parsed_json": str(parsed),
                          "report_json": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
