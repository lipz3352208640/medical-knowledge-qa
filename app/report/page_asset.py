"""Render an authenticated, bounded original PDF page for parse review."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.storage.mysql import _connect, parse_mysql_url


def _source(mysql_url: str, run_id: str) -> tuple[Path, int] | None:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT d.source_file,d.page_count FROM parse_run r
                   JOIN document d ON d.id=r.document_id WHERE r.id=%s""", (run_id,),
            )
            row = cursor.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    path = Path(row[0])
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise FileNotFoundError("original PDF is unavailable")
    return path, row[1]


def page_geometry(mysql_url: str, run_id: str, page: int) -> dict | None:
    source = _source(mysql_url, run_id)
    if source is None or not 1 <= page <= source[1]:
        return None
    with pymupdf.open(source[0]) as document:
        original = document[page - 1]
        return {"width": original.rect.width, "height": original.rect.height,
                "rotation": original.rotation, "coordinate_system": "pdf_points_top_left",
                "rotation_matrix": tuple(original.rotation_matrix)}


def render_page(mysql_url: str, run_id: str, page: int) -> bytes | None:
    source = _source(mysql_url, run_id)
    if source is None or not 1 <= page <= source[1]:
        return None
    with pymupdf.open(source[0]) as document:
        original = document[page - 1]
        if original.rect.width * original.rect.height > 4_000_000:
            raise ValueError("PDF page is too large to render")
        return original.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png")
