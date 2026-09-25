"""Coordinate-preserving table extraction and cautious cross-page merging."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from app.models import Section, SourceRef, TableCell, TableRecord, TableRow
from app.parser.layout import section_at, stable_id
from app.parser.normalize import clean_space, extract_values


INHERITABLE_HEADERS = ("不良反应", "事件", "项目", "系统器官", "器官", "分组", "类别", "症状", "药品名")


def normalize_rows(raw_rows: list[list[str | None]], vertical_spans: set[tuple[int, int]],
                   headers: list[str | None] | None = None) -> list[list[str | None]]:
    """Fill only blanks proven to be part of a vertical span in a semantic key column."""
    if not raw_rows:
        return []
    width = max(map(len, raw_rows))
    result = [[clean_space(v) if v is not None else None for v in row] + [None] * (width - len(row))
              for row in raw_rows]
    header_values = headers if headers is not None else result[0]
    eligible = {c for c, header in enumerate(header_values) if header and any(k in header for k in INHERITABLE_HEADERS)}
    for row in range(1, len(result)):
        for col in eligible:
            if (row, col) in vertical_spans and result[row][col] is None:
                result[row][col] = result[row - 1][col]
    return result


def markdown_table(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    width = max(map(len, rows))
    def render(row: list[str | None]) -> str:
        return "| " + " | ".join((x or "").replace("|", "\\|").replace("\n", "<br>") for x in row + [None] * (width - len(row))) + " |"
    return "\n".join([render(rows[0]), "| " + " | ".join(["---"] * width) + " |", *map(render, rows[1:])])


@dataclass(slots=True)
class Fragment:
    page: int
    page_height: float
    bbox: tuple[float, float, float, float]
    raw_rows: list[list[str | None]]
    cell_bboxes: list[list[tuple[float, float, float, float] | None]]
    image_path: str
    section_id: str
    strategy: str


def _row_bbox(cells: list[tuple[float, float, float, float] | None], fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    valid = [x for x in cells if x is not None]
    if not valid:
        return fallback
    return (min(x[0] for x in valid), min(x[1] for x in valid), max(x[2] for x in valid), max(x[3] for x in valid))


def _vertical_spans(fragment: Fragment) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for row_index in range(1, len(fragment.cell_bboxes)):
        row_boxes = fragment.cell_bboxes[row_index]
        row_box = _row_bbox(row_boxes, fragment.bbox)
        for col, box in enumerate(row_boxes):
            if box is not None:
                continue
            for previous in range(row_index - 1, -1, -1):
                if col < len(fragment.cell_bboxes[previous]):
                    prior = fragment.cell_bboxes[previous][col]
                    if prior is not None:
                        if prior[3] > row_box[1] + 1 and prior[0] <= row_box[0] + 5:
                            spans.add((row_index, col))
                        break
    return spans


def extract_fragments(plumber_page: object, fitz_page: object, page_number: int,
                      document_id: str, sections: list[Section], output_dir: Path) -> list[Fragment]:
    import pymupdf  # Imported only when PDF parsing is requested.

    found = plumber_page.find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
    strategy = "lines"
    if not found:
        candidates = plumber_page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text",
                                               "min_words_vertical": 3, "min_words_horizontal": 2})
        found = [t for t in candidates if len(t.rows) >= 3 and len(t.columns) >= 2 and
                 any(re.search(r"剂量|等级|频率|不良反应|项目|发生率", str(x)) for x in t.extract()[0])]
        strategy = "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[Fragment] = []
    for index, table in enumerate(found):
        if len(table.rows) < 2 or len(table.columns) < 2:
            continue
        bbox = tuple(round(float(x), 2) for x in table.bbox)
        raw_rows = table.extract()
        cell_bboxes = [[tuple(round(float(v), 2) for v in c) if c is not None else None for c in row.cells]
                       for row in table.rows]
        image_id = stable_id(document_id, "table-crop", page_number, index, bbox)
        image_path = output_dir / f"{image_id}.png"
        if not image_path.exists():
            pixmap = fitz_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(bbox), alpha=False)
            pixmap.save(str(image_path))
        result.append(Fragment(page_number, float(plumber_page.height), bbox, raw_rows, cell_bboxes,
                               str(image_path), section_at(sections, page_number, bbox[1]), strategy))
    return result


def _same_header(a: Fragment, b: Fragment) -> bool:
    return bool(a.raw_rows and b.raw_rows and
                [clean_space(x or "") for x in a.raw_rows[0]] ==
                [clean_space(x or "") for x in b.raw_rows[0]])


def can_merge(a: Fragment, b: Fragment) -> bool:
    if b.page != a.page + 1 or a.section_id != b.section_id:
        return False
    if len(a.raw_rows[0]) != len(b.raw_rows[0]):
        return False
    if a.bbox[3] < a.page_height * 0.7 or b.bbox[1] > b.page_height * 0.3:
        return False
    if abs(a.bbox[0] - b.bbox[0]) > 18 or abs(a.bbox[2] - b.bbox[2]) > 18:
        return False
    if _same_header(a, b):
        return True
    header_words = ("药品名", "剂量", "等级", "严重程度", "临床表现", "不良反应", "项目")
    first_header = any(len(clean_space(cell or "")) <= 12 and any(word in (cell or "") for word in header_words)
                       for cell in a.raw_rows[0])
    next_header = any(len(clean_space(cell or "")) <= 12 and any(word in (cell or "") for word in header_words)
                      for cell in b.raw_rows[0])
    return first_header and not next_header


def group_fragments(fragments: list[Fragment]) -> list[list[Fragment]]:
    groups: list[list[Fragment]] = []
    for fragment in sorted(fragments, key=lambda f: (f.page, f.bbox[1])):
        if groups and can_merge(groups[-1][-1], fragment):
            groups[-1].append(fragment)
        else:
            groups.append([fragment])
    return groups


def build_table(group: list[Fragment], document_id: str, drug_name: str) -> TableRecord:
    first = group[0]
    table_id = stable_id(document_id, "table", first.page, first.bbox)
    refs = [SourceRef(document_id, f.section_id, f.page, f.bbox) for f in group]
    cells: list[TableCell] = []
    rows: list[TableRow] = []
    all_raw: list[list[str | None]] = []
    all_normalized: list[list[str | None]] = []
    warnings: list[str] = []
    for fragment_index, fragment in enumerate(group):
        spans = _vertical_spans(fragment)
        normalized = normalize_rows(fragment.raw_rows, spans, all_normalized[0] if fragment_index else None)
        skip = 1 if fragment_index and _same_header(group[fragment_index - 1], fragment) else 0
        if skip:
            warnings.append(f"第 {fragment.page} 页重复表头已去重")
        elif fragment_index:
            warnings.append(f"第 {fragment.page} 页无重复表头，按前页表头解释；请人工核验")
        continuation_cols: set[int] = set()
        for local_row in range(skip, len(fragment.raw_rows)):
            raw = fragment.raw_rows[local_row]
            norm = normalized[local_row]
            cross_page_inherited: set[int] = set()
            if fragment_index and all_normalized:
                for col, header in enumerate(all_normalized[0]):
                    if (header and any(key in header for key in INHERITABLE_HEADERS)
                            and col < len(norm) and not norm[col]
                            and (local_row == skip or col in continuation_cols)):
                        source_row = normalized[local_row - 1] if local_row > skip else all_normalized[-1]
                        if col >= len(source_row) or not source_row[col]:
                            continue
                        norm[col] = source_row[col]
                        cross_page_inherited.add(col)
                        continuation_cols.add(col)
                # PDF extractors sometimes repeat the last data row on both pages.
                if local_row == skip and norm == all_normalized[-1]:
                    warnings.append(f"第 {fragment.page} 页重复数据行已去重")
                    continue
            row_index = len(all_raw)
            all_raw.append(raw)
            all_normalized.append(norm)
            boxes = fragment.cell_bboxes[local_row]
            row_bbox = _row_bbox(boxes, fragment.bbox)
            row_ref = SourceRef(document_id, fragment.section_id, fragment.page, row_bbox)
            for col, value in enumerate(raw):
                cell_bbox = boxes[col] if col < len(boxes) and boxes[col] else row_bbox
                inherited = (row_index - 1 if col in cross_page_inherited else
                             local_row - 1 if (local_row, col) in spans else None)
                cells.append(TableCell(value, norm[col] if col < len(norm) else None, row_index, col,
                                       SourceRef(document_id, fragment.section_id, fragment.page, cell_bbox), inherited))
            if row_index:
                headers = all_normalized[0]
                detail = "；".join(f"{headers[col] or f'列{col+1}'}：{value}" for col, value in enumerate(norm) if value)
                retrieval = f"{drug_name}；{detail}"
                rows.append(TableRow(stable_id(table_id, "row", row_index), table_id, row_index,
                                     raw, norm, [row_ref], retrieval, extract_values(retrieval)))
    if len(group) > 1:
        warnings.append("跨页表格按列数、页面位置和续页内容拼接；请人工核验")
    raw_json = {"rows": all_raw, "fragments": [{"page": f.page, "bbox": f.bbox, "strategy": f.strategy} for f in group]}
    normalized_json = {"rows": all_normalized, "cells": [asdict(c) for c in cells]}
    return TableRecord(id=table_id, document_id=document_id, section_id=first.section_id,
                       title="", title_ref=None, markdown=markdown_table(all_normalized),
                       raw_json=raw_json, normalized_json=normalized_json,
                       image_paths=[f.image_path for f in group], refs=refs, cells=cells,
                       rows=rows, warnings=warnings)
