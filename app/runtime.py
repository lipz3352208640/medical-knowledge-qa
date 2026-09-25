"""Construct reusable retrieval and answer services from local configuration."""

import os

from app.generator.answer import AnswerService
from app.generator.ollama import OllamaChatClient, OllamaChatConfig
from app.indexer.embeddings import OllamaEmbedder, OllamaEmbeddingConfig
from app.retrieval.compression import ContextCompressor
from app.retrieval.pipeline import RetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rewrite import LLMQueryRewriter
from app.storage.vector_store import QdrantConfig, QdrantVectorStore


def build_runtime(mysql_url: str) -> tuple[AnswerService, RetrievalEngine]:
    reranker_model = os.getenv("RERANKER_MODEL")
    if not reranker_model:
        raise RuntimeError("RERANKER_MODEL is required")
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embedder = OllamaEmbedder(OllamaEmbeddingConfig(
        base_url=ollama_url,
        model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    ))
    vector_store = QdrantVectorStore(QdrantConfig(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION", "medical_knowledge_chunks"),
    ))
    chat = OllamaChatClient(OllamaChatConfig(
        base_url=ollama_url, model=os.getenv("LLM_MODEL", "qwen3.5:latest"),
        timeout_seconds=float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "120")),
    ))
    rewrite_enabled = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() not in {
        "0", "false", "no", "off",
    }
    retriever = RetrievalEngine(
        mysql_url, embedder, vector_store, CrossEncoderReranker(reranker_model),
        query_rewriter=LLMQueryRewriter(chat) if rewrite_enabled else None,
    )
    return AnswerService(retriever, ContextCompressor(embedder, chat), chat), retriever
