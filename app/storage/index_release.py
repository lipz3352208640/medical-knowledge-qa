"""Read and record verified full index releases."""

from __future__ import annotations

from app.storage.mysql import _connect, parse_mysql_url


def release_status(mysql_url: str, version: str) -> dict | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT version,status,collection_name,document_count,failed_count,
                          started_at,completed_at,error_summary
                   FROM index_release WHERE version=%s""", (version,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return dict(zip(("version","status","collection_name","document_count",
                         "failed_count","started_at","completed_at","error_summary"), row, strict=True))
    finally:
        conn.close()


def record_release(mysql_url: str, version: str, status: str, collection_name: str,
                   document_count: int = 0, failed_count: int = 0,
                   error_summary: str | None = None) -> None:
    if status not in {"RUNNING", "READY", "FAILED"}:
        raise ValueError(status)
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO index_release
                   (version,status,collection_name,document_count,failed_count,completed_at,error_summary)
                   VALUES (%s,%s,%s,%s,%s,IF(%s='RUNNING',NULL,NOW(6)),%s)
                   ON DUPLICATE KEY UPDATE status=VALUES(status),collection_name=VALUES(collection_name),
                     document_count=VALUES(document_count),failed_count=VALUES(failed_count),
                     completed_at=VALUES(completed_at),error_summary=VALUES(error_summary)""",
                (version, status, collection_name, document_count, failed_count, status, error_summary),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
