"""Construct reusable retrieval and answer services from local configuration."""

from app.config import setting

from app.generator.answer import AnswerService
from app.generator.ollama import OllamaChatClient, OllamaChatConfig
from app.indexer.embeddings import OllamaEmbedder, OllamaEmbeddingConfig
from app.retrieval.compression import ContextCompressor
from app.retrieval.pipeline import RetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rewrite import LLMQueryRewriter
from app.storage.vector_store import QdrantConfig, QdrantVectorStore


def build_runtime(mysql_url: str) -> tuple[AnswerService, RetrievalEngine]:
    reranker_model = setting("RERANKER_MODEL")
    if not reranker_model:
        raise RuntimeError("RERANKER_MODEL is required")
    ollama_url = setting("OLLAMA_URL", "http://127.0.0.1:11434")
    embedder = OllamaEmbedder(OllamaEmbeddingConfig(
        base_url=ollama_url,
        model=setting("EMBEDDING_MODEL", "bge-m3:latest"),
        dimensions=int(setting("EMBEDDING_DIMENSIONS", "1024")),
        batch_size=int(setting("EMBEDDING_BATCH_SIZE", "32")),
    ))
    vector_store = QdrantVectorStore(QdrantConfig(
        url=setting("QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=setting("QDRANT_API_KEY"),
        collection_name=setting("QDRANT_COLLECTION", "medical_knowledge_chunks"),
    ))
    chat = OllamaChatClient(OllamaChatConfig(
        base_url=ollama_url, model=setting("LLM_MODEL", "qwen3.5:latest"),
        timeout_seconds=float(setting("OLLAMA_CHAT_TIMEOUT_SECONDS", "120")),
    ))
    rewrite_enabled = setting("QUERY_REWRITE_ENABLED", "true").lower() not in {
        "0", "false", "no", "off",
    }
    retriever = RetrievalEngine(
        mysql_url, embedder, vector_store, CrossEncoderReranker(reranker_model),
        query_rewriter=LLMQueryRewriter(chat) if rewrite_enabled else None,
    )
    return AnswerService(retriever, ContextCompressor(embedder, chat), chat), retriever
