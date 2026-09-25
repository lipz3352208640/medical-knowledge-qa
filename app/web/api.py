"""FastAPI interface for grounded leaflet questions and source inspection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pymysql
from pydantic import BaseModel, Field

from app.generator.answer import AnswerService
from app.generator.ollama import ChatError
from app.indexer.embeddings import EmbeddingError
from app.retrieval.pipeline import RetrievalEngine
from app.runtime import build_runtime
from app.storage.retrieval import load_chunks_by_ids
from app.storage.vector_store import QdrantError


_LOG = logging.getLogger(__name__)
_STATIC = Path(__file__).parent / "static"


class QueryFilters(BaseModel):
    drug_names: list[str] = Field(default_factory=list, max_length=5)
    document_versions: list[str] = Field(default_factory=lambda: ["current"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    debug: bool = False


def create_app(
    *,
    answer_service: AnswerService | None = None,
    retriever: RetrievalEngine | None = None,
    mysql_url: str | None = None,
    enable_debug: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="医药知识问答", version="0.1.0")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    lock = Lock()
    runtime: tuple[AnswerService, RetrievalEngine] | None = None
    debug_allowed = (
        os.getenv("ENABLE_RETRIEVAL_DEBUG") == "1" if enable_debug is None else enable_debug
    )

    def services() -> tuple[AnswerService, RetrievalEngine]:
        nonlocal runtime
        if runtime is None:
            with lock:
                if runtime is None:
                    if answer_service is not None and retriever is not None:
                        runtime = (answer_service, retriever)
                    else:
                        url = mysql_url or os.getenv("MYSQL_URL")
                        if not url:
                            raise HTTPException(503, "请先配置 MYSQL_URL")
                        runtime = build_runtime(url)
        return runtime

    def validate_filters(request: QueryRequest) -> tuple[str, ...]:
        if request.filters.document_versions != ["current"]:
            raise HTTPException(422, "目前只支持 current 版本的说明书")
        names = tuple(name.strip() for name in request.filters.drug_names)
        if any(not name for name in names):
            raise HTTPException(422, "药品筛选名称不能为空")
        return names

    @app.get("/")
    def homepage() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/v1/chat/query")
    def query(request: QueryRequest) -> dict:
        names = validate_filters(request)
        try:
            answer, _ = services()
            return answer.answer(request.query.strip(), drug_names=names).to_dict(debug=request.debug)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ChatError, EmbeddingError, QdrantError, pymysql.MySQLError, ImportError, OSError, RuntimeError):
            _LOG.exception("Question answering failed")
            raise HTTPException(503, "问答服务暂不可用，请检查 MySQL、Qdrant 和 Ollama") from None

    @app.post("/api/v1/retrieval/debug")
    def retrieval_debug(request: QueryRequest) -> dict:
        if not debug_allowed:
            raise HTTPException(404, "检索调试接口未启用")
        names = validate_filters(request)
        try:
            _, engine = services()
            return engine.retrieve(request.query.strip(), drug_names=names).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (EmbeddingError, QdrantError, pymysql.MySQLError, ImportError, OSError, RuntimeError):
            _LOG.exception("Retrieval debug failed")
            raise HTTPException(503, "检索服务暂不可用") from None

    @app.get("/api/v1/evidence/{chunk_id}")
    def evidence(chunk_id: str) -> dict:
        if not 1 <= len(chunk_id) <= 128:
            raise HTTPException(422, "无效的 chunk ID")
        url = mysql_url or os.getenv("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            chunks = load_chunks_by_ids(url, [chunk_id])
        except (pymysql.MySQLError, OSError):
            _LOG.exception("Evidence lookup failed")
            raise HTTPException(503, "来源查询暂不可用") from None
        if not chunks:
            raise HTTPException(404, "未找到当前有效的原文证据")
        chunk = chunks[0]
        meta = chunk.metadata
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "drug_name": meta.get("generic_name") or meta.get("drug_name"),
            "source_file": Path(meta.get("source_file") or "").name,
            "revision_date": meta.get("revision_date"),
            "section": meta.get("section"),
            "section_path": meta.get("section_path"),
            "content_type": chunk.content_type,
            "content": chunk.content,
            "retrieval_text": chunk.retrieval_text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "bbox": chunk.bbox,
            "refs": chunk.refs,
            "parent_id": chunk.parent_id,
            "table_id": meta.get("table_id"),
            "source_element_ids": meta.get("source_element_ids", []),
        }

    return app


app = create_app()
