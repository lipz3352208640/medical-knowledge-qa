"""FastAPI interface for grounded leaflet questions and source inspection."""

from __future__ import annotations

import logging
from app.config import setting
from pathlib import Path
from urllib.parse import unquote
from threading import Event, Lock, Thread
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
import pymysql
from pydantic import BaseModel, Field

from app.generator.answer import AnswerService
from app.generator.ollama import ChatError
from app.indexer.embeddings import EmbeddingError
from app.observability.metrics import METRICS
from app.observability.tracing import new_trace
from app.report.page_asset import page_geometry, render_page
from app.rules.cache import RuleSnapshotCache
from app.rules.runtime import REQUEST_RULE_SNAPSHOT
from app.storage.index_release import release_status
from app.storage.jobs import (enqueue_import, enqueue_reprocess, enqueue_correction,
                              get_job, list_jobs, publish_import, retry_correction, discard_pending_document)
from app.retrieval.pipeline import RetrievalEngine
from app.runtime import build_runtime
from app.storage.audit import AuditPersistenceError, MySQLAuditStore
from app.storage.retrieval import load_chunks_by_ids
from app.storage.report import (
    get_qa_audit, list_documents, list_parse_runs, list_qa_audits, parse_page,
    run_document_id, chunk_sources, chunk_hierarchy, document_versions, document_identity,
)
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


class DiscardDocumentRequest(BaseModel):
    actor: str = Field(min_length=1,max_length=64)
    reason: str = Field(min_length=1,max_length=2000)


class CorrectionBatchRequest(BaseModel):
    batch_key: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=1, max_length=24)
    parse_run_id: str = Field(min_length=1, max_length=24)
    index_version: str = Field(min_length=64, max_length=64)
    file_hash: str = Field(min_length=64, max_length=64)
    proposals: list[dict] = Field(min_length=1, max_length=100)


def create_app(
    *,
    answer_service: AnswerService | None = None,
    retriever: RetrievalEngine | None = None,
    mysql_url: str | None = None,
    enable_debug: bool | None = None,
    internal_token: str | None = None,
    audit_store: MySQLAuditStore | None = None,
    audit_required: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="医药知识问答", version="0.1.0")
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    lock = Lock()
    runtime: tuple[AnswerService, RetrievalEngine] | None = None
    debug_allowed = (
        (setting("ENABLE_RETRIEVAL_DEBUG") or "").lower() in {"1", "true", "yes"}
        if enable_debug is None else enable_debug
    )
    audit_url = setting("QA_AUDIT_URL") or mysql_url or setting("MYSQL_URL")
    audit = audit_store
    audit_flag = setting("QA_AUDIT_ENABLED")
    # A production RAG process with MYSQL_URL should audit by default. Tests
    # that inject an AnswerService/RetrievalEngine remain opt-in and isolated.
    production = answer_service is None and retriever is None
    audit_enabled = (
        audit_flag.lower() in {"1", "true", "yes"}
        if audit_flag is not None
        else (production and bool(audit_url))
    )
    if audit is None and production and audit_enabled and audit_url:
        audit = MySQLAuditStore(audit_url)
    require_audit = (
        audit_required if audit_required is not None else
        (setting("QA_AUDIT_REQUIRED", "1" if audit is not None else "0").lower()
         in {"1", "true", "yes"} if production or audit is not None else False)
    )
    service_token = internal_token or setting("RAG_INTERNAL_TOKEN")
    rule_cache = RuleSnapshotCache(
        setting("RAG_RULE_SNAPSHOT_PATH", "data/rule_snapshot.json"),
        admin_url=setting("ADMIN_INTERNAL_URL") if production else None,
        token=service_token,
    )
    rule_lock = Lock()
    stop_rules = Event()
    @app.on_event("startup")
    def start_rule_refresh() -> None:
        if not rule_cache.admin_url:
            return
        with rule_lock:
            rule_cache.refresh()
        def poll() -> None:
            while not stop_rules.wait(30):
                with rule_lock:
                    rule_cache.refresh()
        Thread(target=poll, name="rule-snapshot-poller", daemon=True).start()
    @app.on_event("shutdown")
    def stop_rule_refresh() -> None:
        stop_rules.set()

    def services() -> tuple[AnswerService, RetrievalEngine]:
        nonlocal runtime
        if runtime is None:
            with lock:
                if runtime is None:
                    if answer_service is not None and retriever is not None:
                        runtime = (answer_service, retriever)
                    else:
                        url = mysql_url or setting("MYSQL_URL")
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
        # Keep the public health contract stable for existing clients.
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return METRICS.render()

    @app.post("/api/v1/chat/query")
    def query(request: QueryRequest, x_request_id: str | None = Header(default=None)) -> dict:
        names = validate_filters(request)
        request_id = x_request_id or uuid.uuid4().hex
        started = time.perf_counter()
        trace = new_trace(request_id=request_id, route="chat_query")
        with rule_lock:
            request_snapshot, rule_version = rule_cache.snapshot, rule_cache.version
        rule_token = REQUEST_RULE_SNAPSHOT.set(request_snapshot)
        METRICS.inc("rag_requests_total", status="started", intent="unknown")
        try:
            with trace.span("rag_request", request_id=request_id):
                with trace.span("llm_generate"):
                    answer_service, _ = services()
                    result = answer_service.answer(request.query.strip(), drug_names=names)
                payload = result.to_dict(debug=request.debug)
                if audit is None and require_audit:
                    raise AuditPersistenceError("QA_AUDIT_REQUIRED=1 but no audit store is configured")
                if audit is not None:
                    with trace.span("audit_persist"):
                        try:
                            answer_id = audit.record_answer(
                                request_id=request_id, trace_id=trace.trace_id,
                                question=request.query.strip(), answer=result.answer,
                                status="COMPLETED", evidence=[
                                    {key: citation[key] for key in ("chunk_id", "document_id", "page", "page_end", "section") if key in citation}
                                    for citation in payload.get("evidence", [])
                                ],
                                rule_version=rule_version,
                                index_version=setting("RAG_INDEX_VERSION"),
                                prompt_version=setting("PROMPT_VERSION", "answer-v1"),
                                model_id=setting("LLM_MODEL", "qwen3.5:latest"),
                                no_evidence_reason=(result.warnings[0] if not payload.get("evidence") and result.warnings else None),
                                citation_check_status="PASSED" if payload.get("evidence") else "FAILED",
                            )
                        except AuditPersistenceError:
                            METRICS.inc("qa_audit_failures_total", reason="persistence")
                            raise
                    payload["answer_id"] = answer_id
                payload["request_id"] = request_id
                payload["trace_id"] = trace.trace_id
                METRICS.inc("rag_requests_total", status="completed", intent="unknown")
                return payload
        except AuditPersistenceError as exc:
            _LOG.exception("Question audit persistence failed")
            raise HTTPException(503, "答案审计未成功写入，未返回可审计成功结果") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ChatError, EmbeddingError, QdrantError, pymysql.MySQLError, ImportError, OSError, RuntimeError):
            _LOG.exception("Question answering failed")
            raise HTTPException(503, "问答服务暂不可用，请检查 MySQL、Qdrant 和 Ollama") from None
        finally:
            METRICS.observe("rag_request_duration_seconds", time.perf_counter() - started, route="chat_query")
            trace.finish()
            REQUEST_RULE_SNAPSHOT.reset(rule_token)

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
        url = mysql_url or setting("MYSQL_URL")
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

    def guard_internal(token: str | None) -> None:
        if not service_token or token != service_token:
            raise HTTPException(401, "内部服务认证失败")

    @app.post("/internal/v1/knowledge/imports", status_code=202)
    async def internal_import(request: Request, x_service_token: str | None = Header(default=None),
                              x_filename: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 mysql_url")
        filename = Path(unquote(x_filename or "")).name
        if not filename.lower().endswith(".pdf") or filename in {".", ".."}:
            raise HTTPException(422, "仅接受 PDF 文件")
        content = await request.body()
        if not content.startswith(b"%PDF-") or len(content) > 50 * 1024 * 1024:
            raise HTTPException(422, "PDF 格式无效或超过 50 MB")
        target_dir = Path(setting("KNOWLEDGE_IMPORT_DIR", "data/imports"))
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{uuid.uuid4().hex}.pdf"
        target.write_bytes(content)
        try:
            job = enqueue_import(url, str(target.resolve()))
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return {"job_id": job["job_id"], "status": job["status"], "filename": filename}

    @app.post("/internal/v1/jobs/{job_id}/publish")
    def internal_publish(job_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 mysql_url")
        try:
            result = publish_import(url, job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if result is None:
            raise HTTPException(404, "待发布任务不存在或尚未完成")
        return result

    @app.get("/internal/v1/documents")
    def internal_documents(include_withdrawn: bool = False, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            return {"documents": list_documents(url,include_withdrawn=include_withdrawn)}
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "文档列表暂不可用") from exc

    @app.get('/internal/v1/documents/{document_id}/versions')
    def internal_document_versions(document_id: str,page: int=Query(default=1,ge=1),page_size: int=Query(default=20,ge=1,le=100),x_service_token: str | None=Header(default=None)):
        guard_internal(x_service_token)
        url=mysql_url or setting('MYSQL_URL')
        if not url: raise HTTPException(503,'请先配置 MYSQL_URL')
        try: result=document_versions(url,document_id,page,page_size)
        except (pymysql.MySQLError,OSError) as exc: raise HTTPException(503,'版本历史暂不可用') from exc
        if result is None: raise HTTPException(404,'文档不存在')
        return result

    @app.get('/internal/v1/documents/{document_id}/identity')
    def internal_document_identity(document_id: str,x_service_token: str | None=Header(default=None)):
        guard_internal(x_service_token)
        url=mysql_url or setting('MYSQL_URL')
        if not url:raise HTTPException(503,'请先配置 MYSQL_URL')
        try:result=document_identity(url,document_id)
        except (pymysql.MySQLError,OSError) as exc:raise HTTPException(503,'文档身份暂不可用') from exc
        if result is None:raise HTTPException(404,'文档不存在')
        return result

    @app.post('/internal/v1/documents/{document_id}/discard')
    def internal_discard(document_id: str, payload: DiscardDocumentRequest, x_service_token: str | None = Header(default=None)):
        guard_internal(x_service_token)
        url = mysql_url or setting('MYSQL_URL')
        if not url:
            raise HTTPException(503,'请先配置 MYSQL_URL')
        try:
            result = discard_pending_document(url,document_id,payload.actor,payload.reason)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        if result is None:
            raise HTTPException(404,'文档版本不存在')
        return result

    @app.get("/internal/v1/documents/{document_id}/parse-runs")
    def internal_parse_runs(document_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            return {"parse_runs": list_parse_runs(url, document_id)}
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "解析运行列表暂不可用") from exc

    @app.get("/internal/v1/parse-runs/{run_id}/snapshot")
    def internal_version_snapshot(run_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        from app.storage.report import version_snapshot
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        value = version_snapshot(url, run_id)
        if value is None:
            raise HTTPException(404, "解析版本不存在")
        for index, page in enumerate(value["pages"], 1):
            if page is None:
                raise HTTPException(409, "版本缺少解析页面，不能生成完整变化清单")
            try:
                geometry = page_geometry(url, run_id, index)
                if geometry:
                    page.update(geometry)
            except OSError:
                pass
        return value

    @app.get("/internal/v1/parse-runs/{run_id}/pages/{page}")
    def internal_parse_page(run_id: str, page: int, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            value = parse_page(url, run_id, page)
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "解析页面暂不可用") from exc
        if value is None:
            raise HTTPException(404, "解析页面不存在")
        try:
            geometry = page_geometry(url, run_id, page)
        except (OSError, pymysql.MySQLError):
            geometry = None
        if geometry:
            value.update(geometry)
        return value

    @app.get("/internal/v1/parse-runs/{run_id}/pages/{page}/image")
    def internal_parse_page_image(run_id: str, page: int,
                                  x_service_token: str | None = Header(default=None)) -> Response:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            content = render_page(url, run_id, page)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "原 PDF 页面暂不可用") from exc
        if content is None:
            raise HTTPException(404, "原 PDF 页面不存在")
        return Response(content, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    @app.get('/internal/v1/parse-runs/{run_id}/tables/{table_id}')
    def internal_table_structure(run_id: str, table_id: str, x_service_token: str | None = Header(default=None)):
        guard_internal(x_service_token)
        from app.review.table_audit import read_table
        value=read_table(mysql_url or setting('MYSQL_URL'),run_id,table_id)
        if value is None:raise HTTPException(404,'表格不属于当前解析版本')
        return value

    @app.post('/internal/v1/parse-runs/{run_id}/tables/{table_id}/validate')
    def internal_validate_table(run_id: str, table_id: str, value: dict, x_service_token: str | None = Header(default=None)):
        guard_internal(x_service_token)
        from app.review.table_audit import read_table,validate_correction
        url=mysql_url or setting('MYSQL_URL');table=read_table(url,run_id,table_id)
        if not table:raise HTTPException(404,'表格不属于当前版本')
        try:cells=validate_correction(table,chunk_hierarchy(url,run_id)['chunks'],value)
        except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
        return {'table':table,'after_cells':cells}

    @app.get("/internal/v1/parse-runs/{run_id}/chunk-hierarchy")
    def internal_chunk_hierarchy(run_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:raise HTTPException(503, "请先配置 MYSQL_URL")
        try:value=chunk_hierarchy(url,run_id)
        except pymysql.MySQLError as exc:raise HTTPException(503, "分块层级暂不可用") from exc
        if value is None:raise HTTPException(404, "解析运行不存在")
        return value

    @app.get("/internal/v1/chunks/{chunk_id}/sources")
    def internal_chunk_sources(chunk_id: str, parse_run_id: str, index_version: str,
                               x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            value = chunk_sources(url, chunk_id, parse_run_id, index_version)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "分块来源暂不可用") from exc
        if value is None:
            raise HTTPException(404, "分块或解析批次不存在")
        return value

    @app.post("/internal/v1/parse-runs/{run_id}/reprocess", status_code=202)
    def internal_reprocess(run_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            job = enqueue_reprocess(url, run_id)
        except pymysql.MySQLError as exc:
            raise HTTPException(503, "任务数据库暂不可用") from exc
        if job is None:
            raise HTTPException(404, "解析运行不存在")
        return {"job_id": job["job_id"], "status": job["status"], "parse_run_id": run_id}

    @app.post("/internal/v1/review-batches", status_code=202)
    def internal_review_batch(payload: CorrectionBatchRequest,
                              x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        try:
            job = enqueue_correction(url, payload.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"job_id": job["job_id"], "status": job["status"]}

    @app.post("/internal/v1/jobs/{job_id}/retry", status_code=202)
    def internal_retry_job(job_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        value = retry_correction(url, job_id)
        if value is None:
            raise HTTPException(409, "只有失败的修正任务可以重试")
        return value

    @app.get("/internal/v1/jobs/{job_id}")
    def internal_job(job_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        value = get_job(url, job_id)
        if value is None:
            raise HTTPException(404, "任务不存在")
        return value

    @app.get("/internal/v1/jobs")
    def internal_jobs(x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 mysql_url")
        return {"jobs": list_jobs(url)}

    @app.get("/internal/v1/qa-audits")
    def internal_audits(limit: int = 50, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url or audit is None:
            return {"audits": [], "status": "disabled"}
        try:
            return {"audits": list_qa_audits(url, max(1, min(limit, 200)))}
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "问答审计暂不可用") from exc

    @app.get("/internal/v1/qa-audits/{answer_id}")
    def internal_audit(answer_id: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url or audit is None:
            raise HTTPException(404, "问答审计未配置")
        try:
            value = get_qa_audit(url, answer_id)
        except (pymysql.MySQLError, OSError) as exc:
            raise HTTPException(503, "问答审计暂不可用") from exc
        if value is None:
            raise HTTPException(404, "问答审计不存在")
        return value

    @app.get("/internal/v1/index-versions/{version}")
    def internal_index_version(version: str, x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        url = mysql_url or setting("MYSQL_URL")
        if not url:
            raise HTTPException(503, "请先配置 MYSQL_URL")
        value = release_status(url, version)
        if value is None:
            raise HTTPException(404, "索引版本不存在")
        return value

    @app.get("/internal/v1/config-status")
    def internal_config_status(x_service_token: str | None = Header(default=None)) -> dict:
        guard_internal(x_service_token)
        with rule_lock:
            loaded_version, refresh_error = rule_cache.version, rule_cache.error
        return {
            "status": "ok", "service_instance": setting("RAG_INSTANCE_ID", "rag-local"),
            "rule_version": loaded_version, "rule_refresh_error": refresh_error,
            "index_version": setting("RAG_INDEX_VERSION"),
            "audit_status": "configured" if audit is not None else "not_configured",
            "trace_status": "configured" if setting("LANGFUSE_HOST") else "disabled",
        }

    return app


app = create_app()
