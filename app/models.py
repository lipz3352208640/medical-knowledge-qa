"""Loss-aware, JSON serializable parse records.

Coordinates use PDF points with the origin at the top-left of the rendered page.
Page numbers are one-based. A multi-page object has one SourceRef per page.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceRef:
    document_id: str
    section_id: str
    page: int
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class ValuePair:
    raw: str
    normalized: str


@dataclass(slots=True)
class Section:
    id: str
    document_id: str
    raw_name: str
    normalized_name: str
    start_page: int
    end_page: int
    heading_ref: SourceRef
    confidence: float


@dataclass(slots=True)
class TextElement:
    id: str
    ref: SourceRef
    element_type: str
    raw_text: str
    normalized_text: str
    reading_order: int
    confidence: float
    values: list[ValuePair] = field(default_factory=list)
    is_bold: bool = False


@dataclass(slots=True)
class TableCell:
    raw: str | None
    normalized: str | None
    row: int
    column: int
    ref: SourceRef
    inherited_from_row: int | None = None


@dataclass(slots=True)
class TableRow:
    id: str
    table_id: str
    row_index: int
    raw_cells: list[str | None]
    normalized_cells: list[str | None]
    refs: list[SourceRef]
    retrieval_text: str
    values: list[ValuePair] = field(default_factory=list)


@dataclass(slots=True)
class TableRecord:
    id: str
    document_id: str
    section_id: str
    title: str
    title_ref: SourceRef | None
    markdown: str
    raw_json: dict[str, Any]
    normalized_json: dict[str, Any]
    image_paths: list[str]
    refs: list[SourceRef]
    cells: list[TableCell]
    rows: list[TableRow]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImageRecord:
    id: str
    ref: SourceRef
    image_type: str
    path: str
    caption: str | None
    confidence: float


@dataclass(slots=True)
class ChunkRecord:
    """A traceable parent or child retrieval unit."""

    id: str
    document_id: str
    section_id: str
    parent_id: str | None
    content: str
    retrieval_text: str
    content_type: str
    is_parent: bool
    token_count: int
    page_start: int
    page_end: int
    bbox: tuple[float, float, float, float] | None
    reading_order: int
    refs: list[SourceRef]
    metadata: dict[str, Any]
    content_hash: str
    chunker_version: str
    section_path: list[str] = field(default_factory=list)
    raw_section_name: str = ""
    normalized_section_name: str = ""
    topic_id: str | None = None
    topic_name: str | None = None
    parent_type: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    lead_in_id: str | None = None
    split_reason: str = ""
    overlap_source_ids: list[str] = field(default_factory=list)
    source_element_ids: list[str] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return self.content


@dataclass(slots=True)
class ParseResult:
    document_id: str
    source_file: str
    file_hash: str
    parser_version: str
    drug_name: ValuePair
    page_count: int
    sections: list[Section]
    elements: list[TextElement]
    tables: list[TableRecord]
    images: list[ImageRecord]
    warnings: list[str]
    failed_pages: list[int]
    blank_pages: list[int] = field(default_factory=list)
    brand_names: list[ValuePair] = field(default_factory=list)
    english_name: ValuePair | None = None
    approval_numbers: list[str] = field(default_factory=list)
    revision_date: str | None = None
    chunk_policy: dict[str, Any] = field(default_factory=dict)
    summary_overrides: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
