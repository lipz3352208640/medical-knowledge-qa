"""Rehydrate an immutable parse run for review corrections without reparsing PDF."""

from __future__ import annotations

import json

from app.models import (ImageRecord, ParseResult, Section, SourceRef, TableCell,
                        TableRecord, TableRow, TextElement, ValuePair)
from app.parser.layout import stable_id
from app.storage.mysql import _connect, parse_mysql_url


def _j(value):
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def _remap_policy(value, id_map):
    """Carry replay policies into another immutable clone with its new IDs."""
    if isinstance(value, dict):
        return {id_map.get(key, key): _remap_policy(item, id_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_policy(item, id_map) for item in value]
    if isinstance(value, str):
        return id_map.get(value, value)
    return value


def load_review_base(mysql_url: str, run_id: str, parser_version: str) -> tuple[ParseResult, dict[str, str]]:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT d.id,d.source_file,d.file_hash,d.page_count,d.revision_date,p.report_json
                   FROM parse_run r JOIN document d ON d.id=r.document_id
                   JOIN parse_report p ON p.document_id=d.id WHERE r.id=%s""",
                (run_id,),
            )
            document = cursor.fetchone()
            if document is None:
                raise ValueError("来源解析运行不存在")
            old_doc, source_file, file_hash, page_count, revision_date, report_json = document
            report = _j(report_json)
            new_doc = stable_id(file_hash, parser_version)
            cursor.execute("SELECT * FROM section WHERE document_id=%s ORDER BY start_page,id", (old_doc,))
            section_rows = cursor.fetchall()
            cursor.execute("SELECT * FROM element WHERE document_id=%s ORDER BY page,reading_order,id", (old_doc,))
            element_rows = cursor.fetchall()
            cursor.execute("SELECT * FROM parsed_table WHERE document_id=%s ORDER BY id", (old_doc,))
            table_rows = cursor.fetchall()
            cursor.execute("SELECT * FROM table_cell WHERE document_id=%s ORDER BY table_id,row_index,column_index", (old_doc,))
            cell_rows = cursor.fetchall()
            cursor.execute("SELECT * FROM table_row WHERE document_id=%s ORDER BY table_id,row_index", (old_doc,))
            row_rows = cursor.fetchall()
            cursor.execute("SELECT * FROM image WHERE document_id=%s ORDER BY page,id", (old_doc,))
            image_rows = cursor.fetchall()
        all_ids = [row[0] for row in [*section_rows, *element_rows, *table_rows, *row_rows, *image_rows]]
        id_map = {old: stable_id("review-clone", new_doc, old) for old in all_ids}

        def ref(section_id: str, page: int, bbox) -> SourceRef:
            return SourceRef(new_doc, id_map[section_id], page, tuple(_j(bbox)))

        def from_ref(value) -> SourceRef:
            item = _j(value)
            return ref(item["section_id"], item["page"], item["bbox"])

        sections = [Section(id_map[row[0]], new_doc, row[2], row[3], row[4], row[5],
                            ref(row[0], row[6], row[7]), row[8]) for row in section_rows]
        # Column order follows 001_initial and the additive is_bold migration.
        elements = [TextElement(id_map[row[0]], ref(row[2], row[3], row[4]), row[5],
                                row[6], row[7], row[8], row[9],
                                [ValuePair(**value) for value in _j(row[10])], bool(row[11]))
                    for row in element_rows]
        tables = []
        for row in table_rows:
            old_id = row[0]
            from app.review.table_audit import cell_id
            for cell in cell_rows:
                if cell[0]==old_id:id_map[cell_id(old_id,cell[1],cell[2])]=cell_id(id_map[old_id],cell[1],cell[2])
            cells = [TableCell(cell[7], cell[8], cell[1], cell[2],
                               ref(cell[4], cell[5], cell[6]), cell[9])
                     for cell in cell_rows if cell[0] == old_id]
            rows = [TableRow(id_map[item[0]], id_map[old_id], item[4],
                             _j(item[8]), _j(item[9]),
                             [from_ref(value) for value in _j(item[7])], item[10],
                             [ValuePair(**value) for value in _j(item[11])])
                    for item in row_rows if item[1] == old_id]
            tables.append(TableRecord(
                id_map[old_id], new_doc, id_map[row[2]], row[3],
                from_ref(row[4]) if row[4] else None, row[5], _j(row[6]),
                _j(row[7]), _j(row[8]), [from_ref(value) for value in _j(row[9])],
                cells, rows, _j(row[10]),
            ))
            if tables[-1].raw_json.get('audit_cells'):
                for c in tables[-1].raw_json['audit_cells']:
                    c['table_id']=id_map[old_id];c['id']=cell_id(id_map[old_id],c['row'],c['column'])
        images = [ImageRecord(id_map[row[0]], ref(row[2], row[3], row[4]),
                              row[5], row[6], row[7], row[8]) for row in image_rows]
        result = ParseResult(
            new_doc, source_file, file_hash, parser_version,
            ValuePair(**report["drug_name"]), page_count, sections, elements,
            tables, images, report.get("warnings", []), report.get("failed_pages", []),
            report.get("blank_pages", []),
            [ValuePair(**value) for value in report.get("brand_names", [])],
            ValuePair(**report["english_name"]) if report.get("english_name") else None,
            report.get("approval_numbers", []), str(revision_date) if revision_date else None,
            _remap_policy(report.get("chunk_policy", {}), id_map), report.get("summary_overrides", {}),
        )
        return result, id_map
    finally:
        conn.close()
