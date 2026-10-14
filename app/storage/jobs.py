"""Durable MySQL work queue for slow reparse and reindex operations."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
import hashlib

from app.storage.mysql import _connect, _json, parse_mysql_url


def enqueue_import(mysql_url: str, source_file: str) -> dict:
    job_id = uuid.uuid4().hex
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO rag_job (id,kind,document_id,source_file,status)
                   VALUES (%s,'IMPORT',NULL,%s,'PENDING')""",
                (job_id, source_file),
            )
        conn.commit()
        return get_job(mysql_url, job_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def discard_pending_document(mysql_url: str, document_id: str, actor: str, reason: str) -> dict | None:
    reason = reason.strip()
    if not actor or not reason or len(reason) > 2000:
        raise ValueError('请填写废弃原因（最多 2000 字）')
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT version_status FROM document WHERE id=%s FOR UPDATE', (document_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            if row[0] != 'PENDING':
                raise ValueError('只能废弃待发布 PENDING 版本，不能操作正式或历史版本')
            cursor.execute("SELECT id FROM rag_job WHERE document_id=%s AND status IN ('PENDING','RUNNING','VALIDATING') LIMIT 1 FOR UPDATE", (document_id,))
            if cursor.fetchone():
                raise ValueError('该版本仍有运行中的任务，请等待任务结束后再废弃')
            cursor.execute("UPDATE document SET version_status='WITHDRAWN',effective_to=NOW() WHERE id=%s", (document_id,))
            cursor.execute('INSERT INTO document_discard_log (document_id,actor,reason) VALUES (%s,%s,%s)', (document_id,actor,reason))
        conn.commit()
        return {'document_id':document_id,'version_status':'WITHDRAWN'}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def publish_import(mysql_url: str, job_id: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT j.document_id,d.product_id,d.parse_status,d.version_status,
                          s.vector_status
                   FROM rag_job j JOIN document d ON d.id=j.document_id
                   JOIN document_index_state s ON s.document_id=d.id
                   WHERE j.id=%s AND j.kind IN ('IMPORT','REPROCESS','CORRECTION')
                     AND j.status IN ('READY','NEEDS_REVIEW') FOR UPDATE""",
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            document_id, product_id, parse_status, version_status, vector_status = row
            if parse_status not in {"READY", "REVIEW"} or vector_status != "READY" or version_status not in {"PENDING", "ACTIVE"}:
                raise ValueError("Document is not ready for publication")
            cursor.execute("SELECT source_document_id,source_index_version FROM review_correction_batch WHERE job_id=%s", (job_id,))
            correction = cursor.fetchone()
            if correction:
                cursor.execute("""SELECT d.version_status,s.chunk_manifest_hash FROM document d
                                  JOIN document_index_state s ON s.document_id=d.id WHERE d.id=%s FOR UPDATE""",
                               (correction[0],))
                source_state = cursor.fetchone()
                if source_state != ("ACTIVE", correction[1]):
                    raise ValueError("修正来源版本已变化，不能发布旧修正")
            if version_status == "PENDING":
                cursor.execute("SELECT id FROM drug_product WHERE id=%s FOR UPDATE", (product_id,))
                cursor.execute(
                    """SELECT id FROM document WHERE product_id=%s AND version_status='ACTIVE'
                       FOR UPDATE""", (product_id,),
                )
                previous = cursor.fetchone()
                if previous:
                    cursor.execute(
                        """UPDATE document SET version_status='SUPERSEDED',effective_to=NOW()
                           WHERE id=%s""", (previous[0],),
                    )
                cursor.execute(
                    """UPDATE document SET version_status='ACTIVE',effective_from=NOW(),
                       effective_to=NULL,supersedes_document_id=%s WHERE id=%s""",
                    (previous[0] if previous else None, document_id),
                )
        conn.commit()
        return {"job_id": job_id, "document_id": document_id, "version_status": "ACTIVE"}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enqueue_reprocess(mysql_url: str, run_id: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT r.document_id,d.source_file FROM parse_run r
                   JOIN document d ON d.id=r.document_id WHERE r.id=%s""", (run_id,),
            )
            source = cursor.fetchone()
            if not source:
                return None
            cursor.execute(
                """SELECT id FROM rag_job WHERE document_id=%s AND kind='REPROCESS'
                   AND status IN ('PENDING','RUNNING','VALIDATING')
                   ORDER BY created_at DESC LIMIT 1 FOR UPDATE""", (source[0],),
            )
            existing = cursor.fetchone()
            job_id = existing[0] if existing else uuid.uuid4().hex
            if not existing:
                cursor.execute(
                    """INSERT INTO rag_job (id,kind,document_id,source_file,status)
                       VALUES (%s,'REPROCESS',%s,%s,'PENDING')""",
                    (job_id, source[0], source[1]),
                )
        conn.commit()
        return get_job(mysql_url, job_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enqueue_correction(mysql_url: str, payload: dict) -> dict:
    key = payload.get("batch_key", "")
    if len(key) != 64 or not payload.get("proposals"):
        raise ValueError("修正批次必须有确认的结构化建议和幂等键")
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT job_id FROM review_correction_batch WHERE batch_key=%s FOR UPDATE", (key,))
            existing = cursor.fetchone()
            if existing:
                conn.commit()
                return get_job(mysql_url, existing[0])
            cursor.execute(
                """SELECT d.source_file,d.file_hash,d.version_status,s.chunk_manifest_hash
                   FROM document d JOIN parse_run r ON r.document_id=d.id
                   JOIN document_index_state s ON s.document_id=d.id
                   WHERE d.id=%s AND r.id=%s FOR UPDATE""",
                (payload["document_id"], payload["parse_run_id"]),
            )
            source = cursor.fetchone()
            if (not source or source[1] != payload["file_hash"] or
                    source[2] != "ACTIVE" or source[3] != payload["index_version"]):
                raise ValueError("来源 PDF、解析批次或索引版本已变化")
            file = Path(source[0])
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != source[1]:
                raise ValueError("原始 PDF 不存在或哈希已变化")
            for proposal in payload["proposals"]:
                if proposal.get("issue_type") == "manual_review":
                    raise ValueError("纯描述问题单不能进入执行批次")
                if proposal.get('issue_type')=='table_review':
                    from app.review.table_audit import read_table,validate_correction
                    from app.storage.report import chunk_hierarchy
                    v=proposal['suggested']
                    if v.get('source_run_id')!=payload['parse_run_id']:raise ValueError('表格来源版本不匹配')
                    table=read_table(mysql_url,payload['parse_run_id'],v['table_id'])
                    if not table or table['document_id']!=payload['document_id']:raise ValueError('表格不属于来源文档')
                    validate_correction(table,chunk_hierarchy(mysql_url,payload['parse_run_id'])['chunks'],v)
                for element_id in proposal.get("element_ids", []):
                    if proposal.get('issue_type')=='table_review':
                        if element_id not in {c['id'] for c in table['cells']}:raise ValueError('Cell 不属于来源表格')
                        continue
                    cursor.execute("SELECT 1 FROM element_report WHERE parse_run_id=%s AND element_id=%s LIMIT 1",
                                   (payload["parse_run_id"], element_id))
                    if not cursor.fetchone():
                        raise ValueError("修正元素不属于来源解析批次")
                for chunk_id in proposal.get("chunk_ids", []):
                    cursor.execute("SELECT 1 FROM chunk WHERE document_id=%s AND id=%s",
                                   (payload["document_id"], chunk_id))
                    if not cursor.fetchone():
                        raise ValueError("修正分块不属于来源索引版本")
            # Validate the complete typed correction before persisting a job.
            # Descriptive or malformed payloads never enter the worker queue.
            from app.storage.parsed_result import load_review_base
            from app.review.corrections import apply_proposals
            from app.chunker.parent_child import build_chunks
            from app.parser.pdf import PARSER_VERSION
            preview, id_map = load_review_base(
                mysql_url, payload["parse_run_id"],
                f"{PARSER_VERSION}.review.preview.{key[:8]}",
            )
            apply_proposals(preview, id_map, payload["proposals"], mysql_url,
                            payload["document_id"])
            build_chunks(preview, "preview-drug", "preview-product")
            job_id = uuid.uuid4().hex
            cursor.execute("""INSERT INTO rag_job (id,kind,document_id,source_file,status)
                              VALUES (%s,'CORRECTION',%s,%s,'PENDING')""",
                           (job_id, payload["document_id"], str(file)))
            cursor.execute(
                """INSERT INTO review_correction_batch
                   (batch_key,job_id,source_document_id,source_parse_run_id,
                    source_index_version,source_file_hash,proposal_manifest_json)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (key, job_id, payload["document_id"], payload["parse_run_id"],
                 payload["index_version"], payload["file_hash"], _json(payload["proposals"])),
            )
        conn.commit()
        return get_job(mysql_url, job_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def correction_batch(mysql_url: str, job_id: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("""SELECT source_document_id,source_parse_run_id,source_index_version,
                              source_file_hash,proposal_manifest_json FROM review_correction_batch
                              WHERE job_id=%s""", (job_id,))
            row = cursor.fetchone()
        if not row:
            return None
        return {"document_id": row[0], "parse_run_id": row[1], "index_version": row[2],
                "file_hash": row[3], "proposals": json.loads(row[4]) if isinstance(row[4], str) else row[4]}
    finally:
        conn.close()


def finish_correction_batch(mysql_url: str, job_id: str, document_id: str) -> None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("""UPDATE review_correction_batch SET target_document_id=%s,
                              finished_at=NOW(6) WHERE job_id=%s""", (document_id, job_id))
        conn.commit()
    finally:
        conn.close()


def retry_correction(mysql_url: str, job_id: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("""SELECT id FROM rag_job WHERE id=%s AND kind='CORRECTION'
                              AND status='FAILED' FOR UPDATE""", (job_id,))
            if not cursor.fetchone():
                return None
            cursor.execute("""UPDATE rag_job SET status='PENDING',error_summary=NULL,
                              finished_at=NULL WHERE id=%s""", (job_id,))
        conn.commit()
        return get_job(mysql_url, job_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_job(mysql_url: str, job_id: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT id,kind,document_id,source_file,status,attempts,created_at,started_at,
                          finished_at,result_json,error_summary FROM rag_job WHERE id=%s""", (job_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        names = ("job_id","kind","document_id","source_file","status","attempts","created_at",
                 "started_at","finished_at","result","error")
        value = dict(zip(names, row, strict=True))
        if isinstance(value["result"], str):
            value["result"] = json.loads(value["result"])
        return value
    finally:
        conn.close()


def list_jobs(mysql_url: str, limit: int = 50) -> list[dict]:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM rag_job ORDER BY created_at DESC,id DESC LIMIT %s", (limit,))
            ids = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    return [job for job_id in ids if (job := get_job(mysql_url, job_id)) is not None]


def claim_job(mysql_url: str) -> tuple[str, str, str, str] | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT id,kind,document_id,source_file FROM rag_job WHERE status='PENDING'
                   ORDER BY created_at,id LIMIT 1 FOR UPDATE SKIP LOCKED"""
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    """UPDATE rag_job SET status='RUNNING',attempts=attempts+1,
                       started_at=NOW(6) WHERE id=%s""", (row[0],),
                )
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_job(mysql_url: str, job_id: str, status: str, *, result: dict | None = None,
               error: str | None = None, document_id: str | None = None) -> None:
    if status not in {"VALIDATING", "READY", "NEEDS_REVIEW", "FAILED"}:
        raise ValueError(status)
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """UPDATE rag_job SET status=%s,result_json=%s,error_summary=%s,
                   document_id=COALESCE(%s,document_id),
                   finished_at=IF(%s='VALIDATING',NULL,NOW(6)) WHERE id=%s""",
                (status, _json(result) if result is not None else None, error, document_id, status, job_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
