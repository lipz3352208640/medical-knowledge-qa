"""Exact, ordered PDF source mapping for newly generated chunks.

No fuzzy matching is used: an unmatched chunk remains explicitly unlocatable.
PDF boxes describe elements, never individual characters.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import ChunkRecord, ParseResult, SourceRef


@dataclass(frozen=True)
class SourceSegment:
    element_id: str | None
    source_id: str | None
    source_kind: str
    page_number: int | None
    bbox: tuple[float, float, float, float] | None
    element_start: int | None
    element_end: int | None
    chunk_start: int | None
    chunk_end: int | None
    location_status: str = "LOCATABLE"


def _sources(result: ParseResult) -> dict[str, tuple[str, str, list[SourceRef]]]:
    sources = {
        item.id: (item.raw_text, "text", [item.ref]) for item in result.elements
    }
    for table in result.tables:
        from app.review.table_audit import cell_id
        grid=table.raw_json.get('audit_cells')
        if grid:
            from app.models import SourceRef
            for c in grid:
                refs=[SourceRef(result.document_id,table.section_id,c['page'],tuple(c['bbox']))] if c.get('precise') and c.get('bbox') else []
                sources[cell_id(table.id,c['row'],c['column'])]=(c['text'],'table_cell',refs)
        else:
            from collections import Counter
            boxes=Counter((c.ref.page,c.ref.bbox) for c in table.cells)
            row_boxes={(row.row_index,ref.page,ref.bbox) for row in table.rows for ref in row.refs}
            for c in table.cells:
                precise=boxes[(c.ref.page,c.ref.bbox)]==1 and (c.row,c.ref.page,c.ref.bbox) not in row_boxes
                sources[cell_id(table.id,c.row,c.column)]=(c.raw or '','table_cell',[c.ref] if precise else [])
        for row in table.rows:
            sources[row.id] = (
                "；".join(str(value or "") for value in row.raw_cells),
                "table_row",
                row.refs,
            )
    for image in result.images:
        sources[image.id] = (image.caption, "image_caption", [image.ref])
    return sources


def _exact_segments(
    chunk: ChunkRecord, sources: dict[str, tuple[str, str, list[SourceRef]]]
) -> list[SourceSegment]:
    ids = chunk.source_element_ids
    if not ids or any(source_id not in sources for source_id in ids):
        return []
    # The chunk builder joins PDF lines and table rows with these separators.
    separator = "\n"
    joined = ""
    spans: list[tuple[str, int, int, int, str, list[SourceRef]]] = []
    for source_id in ids:
        raw, kind, refs = sources[source_id]
        stripped = raw.strip()
        if not stripped:
            return []
        if joined:
            joined += separator
        start = len(joined)
        joined += stripped
        spans.append((source_id, start, len(joined), raw.find(stripped), kind, refs))
    # A child may contain only a sentence from a longer PDF line. The strict
    # substring position preserves overlap without guessing per-character boxes.
    expected = chunk.metadata.get("source_text_start")
    if isinstance(expected, int) and joined[expected:expected + len(chunk.content)] == chunk.content:
        offset = expected
    else:
        offset = joined.find(chunk.content)
        # Repeated identical text is ambiguous without generator offsets.
        if offset >= 0 and joined.find(chunk.content, offset + 1) >= 0:
            return []
    if offset < 0:
        return []
    end = offset + len(chunk.content)
    segments: list[SourceSegment] = []
    for source_id, start, stop, leading, kind, refs in spans:
        left, right = max(start, offset), min(stop, end)
        if left >= right:
            continue
        for ref in refs:
            segments.append(SourceSegment(
                element_id=source_id,
                source_id=source_id,
                source_kind=kind,
                page_number=ref.page,
                bbox=ref.bbox,
                element_start=leading + left - start,
                element_end=leading + right - start,
                chunk_start=left - offset,
                chunk_end=right - offset,
            ))
    return segments


def map_chunk_sources(result: ParseResult, chunks: list[ChunkRecord]) -> dict[str, list[SourceSegment]]:
    sources = _sources(result)
    mapped: dict[str, list[SourceSegment]] = {}
    for chunk in chunks:
        if chunk.content_type in {"section_summary", "topic_summary"}:
            continue
        exact = _exact_segments(chunk, sources)
        mapped[chunk.id] = exact or [SourceSegment(
            element_id=None, source_id=None, source_kind="unresolved",
            page_number=None, bbox=None, element_start=None, element_end=None,
            chunk_start=None, chunk_end=None, location_status="UNLOCATABLE",
        )]
    for chunk in chunks:
        if chunk.content_type not in {"section_summary", "topic_summary"}:
            continue
        evidence: list[SourceSegment] = []
        seen: set[tuple[str, str | None, int | None, int | None]] = set()
        for source_chunk_id in chunk.metadata.get("source_chunk_ids", []):
            for item in mapped.get(source_chunk_id, []):
                key = (source_chunk_id, item.element_id, item.element_start, item.page_number)
                if item.location_status != "LOCATABLE" or key in seen:
                    continue
                seen.add(key)
                evidence.append(SourceSegment(
                    element_id=item.element_id, source_id=source_chunk_id,
                    source_kind="summary_evidence", page_number=item.page_number,
                    bbox=item.bbox, element_start=item.element_start,
                    element_end=item.element_end, chunk_start=None, chunk_end=None,
                ))
        mapped[chunk.id] = evidence or [SourceSegment(
            element_id=None, source_id=None, source_kind="summary_evidence",
            page_number=None, bbox=None, element_start=None, element_end=None,
            chunk_start=None, chunk_end=None, location_status="UNLOCATABLE",
        )]
    return mapped
