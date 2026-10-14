"""Backfill page audit records for documents ingested before migration 005.

This command only inserts missing audit rows. It reads the persisted parse
report and source elements; it never reparses, reindexes, or edits source data.
"""

from __future__ import annotations

import argparse
import json
from app.config import setting
from pathlib import Path

import pymupdf

from app.parser.layout import stable_id
from app.storage.mysql import _connect, _json, parse_mysql_url


def _value(raw):
    return json.loads(raw) if isinstance(raw, (str, bytes)) else raw


def _dimensions(path: str, page_count: int) -> list[tuple[float | None, float | None]]:
    sizes = [(None, None)] * page_count
    source = Path(path)
    if source.is_file() and source.suffix.lower() == ".pdf":
        try:
            with pymupdf.open(source) as pdf:
                for index in range(min(page_count, len(pdf))):
                    sizes[index] = (pdf[index].rect.width, pdf[index].rect.height)
        except (OSError, ValueError, RuntimeError):
            pass
    return sizes


def backfill(mysql_url: str, *, dry_run: bool = False) -> dict[str, int]:
    connection = _connect(parse_mysql_url(mysql_url))
    result = {"scanned": 0, "created": 0, "skipped": 0, "missing_report": 0}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM document ORDER BY created_at,id")
            document_ids = [row[0] for row in cursor.fetchall()]
        for document_id in document_ids:
            result["scanned"] += 1
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """SELECT id,source_file,file_hash,parser_version,page_count,
                                  parse_status,revision_date,created_at
                           FROM document WHERE id=%s FOR UPDATE""", (document_id,),
                    )
                    document = cursor.fetchone()
                    if document is None:
                        result["skipped"] += 1
                        connection.rollback()
                        continue
                    doc_id, source, file_hash, parser_version, count, parse_status, revision, created = document
                    cursor.execute("SELECT id FROM parse_run WHERE document_id=%s", (doc_id,))
                    if cursor.fetchone():
                        result["skipped"] += 1
                        connection.rollback()
                        continue
                    cursor.execute("SELECT report_json FROM parse_report WHERE document_id=%s", (doc_id,))
                    row = cursor.fetchone()
                    if not row:
                        result["missing_report"] += 1
                        connection.rollback()
                        continue
                    report = _value(row[0])
                    if dry_run:
                        result["created"] += 1
                        connection.rollback()
                        continue
                    run_id = stable_id("parse-run", doc_id, parser_version, file_hash)
                    warnings = report.get("warnings", [])
                    failed = report.get("failed_pages", [])
                    status = "NEEDS_REVIEW" if parse_status == "REVIEW" or warnings or failed else "READY"
                    cursor.execute(
                        """INSERT INTO parse_run
                           (id,document_id,document_version,file_hash,parser_version,
                            status,started_at,finished_at,error_summary)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (run_id, doc_id, str(revision or file_hash), file_hash,
                         parser_version, status, created, created,
                         _json({"failed_pages": failed, "warnings": warnings,
                                "backfilled_from": "parse_report"})),
                    )
                    cursor.execute(
                        """SELECT e.id,e.page,e.bbox,e.reading_order,e.element_type,
                                  e.raw_text,e.normalized_text,e.confidence,s.normalized_name
                           FROM element e JOIN section s ON s.id=e.section_id
                           WHERE e.document_id=%s ORDER BY e.page,e.reading_order,e.id""",
                        (doc_id,),
                    )
                    elements = cursor.fetchall()
                    element_count: dict[int, int] = {}
                    for element in elements:
                        eid, page, bbox, order, kind, raw, normalized, confidence, section = element
                        element_count[page] = element_count.get(page, 0) + 1
                        cursor.execute(
                            """INSERT INTO element_report
                               (id,parse_run_id,element_id,page,bbox,reading_order,
                                element_type,raw_text,normalized_text,section_path,
                                confidence,warning_codes)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (stable_id("element-report", run_id, eid), run_id, eid, page,
                             bbox, order, kind, raw, normalized, _json([section]),
                             confidence, _json([])),
                        )
                    page_summaries = {item["page"]: item for item in report.get("pages", [])}
                    sizes = _dimensions(source, count)
                    for page in range(1, count + 1):
                        summary = page_summaries.get(page, {})
                        page_warnings = []
                        if page in failed:
                            page_warnings.append("page_parse_failed")
                        if page in report.get("blank_pages", []) or summary.get("blank"):
                            page_warnings.append("blank_page")
                        cursor.execute(
                            """INSERT INTO page_report
                               (parse_run_id,page,width,height,text_count,table_count,
                                image_count,warnings_json)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (run_id, page, *sizes[page - 1], element_count.get(page, 0),
                             summary.get("tables", 0), summary.get("images", 0),
                             _json(page_warnings)),
                        )
                connection.commit()
                result["created"] += 1
            except Exception:
                connection.rollback()
                raise
    finally:
        connection.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    url = setting("MYSQL_URL")
    if not url:
        parser.error("mysql_url in config.yaml is required")
    print(json.dumps(backfill(url, dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
