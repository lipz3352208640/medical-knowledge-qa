"""Turn PDF text lines into ordered, section-linked elements."""

from dataclasses import dataclass
import hashlib
import re

from app.models import Section, SourceRef, TextElement
from app.parser.normalize import SECTION_ALIASES, clean_space, extract_values, normalize_section


def stable_id(*parts: object) -> str:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


@dataclass(slots=True)
class Line:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    order: int
    is_bold: bool = False


HEADING = re.compile(r"^([【\[（(]([^】\]）)]{2,30})[】\]）)])")
NUMBERED_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2})?\s*[.、]?\s*(.+?)\s*$")
ENGLISH_SECTION_NAMES = {"ADVERSE REACTIONS": "不良反应", "DOSAGE AND ADMINISTRATION": "用法用量"}


def heading_name(text: str) -> tuple[str, float] | None:
    raw = clean_space(text)
    match = HEADING.match(raw)
    if match and normalize_section(match.group(1)) in SECTION_ALIASES.values():
        tail = raw[match.end():].lstrip()
        # A bracketed section name followed by a sentence reference is not a heading.
        if not tail.startswith(("中", "）", ")", "。", "，", ",", "、")):
            return match.group(1), 0.98 if not tail else 0.88
    # Unbracketed headings require an exact known label and no trailing prose.
    if raw.rstrip(":：") in SECTION_ALIASES:
        return raw, 0.85
    numbered = NUMBERED_HEADING.fullmatch(raw)
    if numbered:
        label = numbered.group(1).rstrip(":：")
        if normalize_section(label) in SECTION_ALIASES.values() or label.upper() in ENGLISH_SECTION_NAMES:
            return raw, 0.85
    return None


def classify_line(text: str, page: int, bbox: tuple[float, float, float, float], height: float) -> str:
    if heading_name(text):
        return "section_heading"
    if bbox[1] < height * 0.045:
        return "header"
    if bbox[3] > height * 0.955:
        return "footer" if not re.fullmatch(r"\s*[-—]?\s*\d+\s*[-—]?\s*", text) else "page_number"
    if re.match(r"^\s*(?:[（(]?\d+[）).、]|[①-⑳]|[-•])", text):
        return "list_item"
    return "paragraph"


def lines_from_page(page: object, page_number: int) -> list[Line]:
    """
    Use PDF text blocks, preserving visual coordinates and font size.
    使用 PDF 文本块，保留视觉坐标和字体大小。
    """
    lines: list[Line] = []
    data = page.get_text("dict", sort=True)
    for block in data["blocks"]:
        if "lines" not in block:
            continue
        for item in block["lines"]:
            spans = item.get("spans", [])
            value = "".join(span.get("text", "") for span in spans).strip()
            if not value:
                continue
            bbox = tuple(round(float(v), 2) for v in item["bbox"])
            visible_spans = [x for x in spans if x.get("text", "").strip()]
            lines.append(Line(page_number, bbox, value, max((float(x.get("size", 0)) for x in spans), default=0), len(lines),
                              bool(visible_spans) and all("bold" in str(x.get("font", "")).lower() for x in visible_spans)))
    return lines


def build_sections_and_elements(
    document_id: str,
    lines: list[Line],
    page_heights: dict[int, float],
    table_regions: list[tuple[int, tuple[float, float, float, float]]] | None = None,
) -> tuple[list[Section], list[TextElement]]:
    if not lines:
        raise ValueError("No selectable PDF text found")
    lines = sorted(lines, key=lambda x: (x.page, x.order))
    first = lines[0]
    unclassified_id = stable_id(document_id, "unclassified")
    sections = [Section(unclassified_id, document_id, "未分类", "未分类", first.page, first.page,
                        SourceRef(document_id, unclassified_id, first.page, first.bbox), 1.0)]
    current = sections[0]
    elements: list[TextElement] = []
    for line in lines:
        center = ((line.bbox[0] + line.bbox[2]) / 2, (line.bbox[1] + line.bbox[3]) / 2)
        in_table = any(page == line.page and box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]
                       for page, box in (table_regions or []))
        kind = "table_text" if in_table else classify_line(line.text, line.page, line.bbox, page_heights[line.page])
        heading = heading_name(line.text) if kind == "section_heading" else None
        if heading:
            raw, confidence = heading
            section_id = stable_id(document_id, "section", line.page, line.order, raw)
            label = NUMBERED_HEADING.fullmatch(raw)
            normalized = ENGLISH_SECTION_NAMES.get(label.group(1).upper(), normalize_section(label.group(1))) if label else normalize_section(raw)
            current = Section(section_id, document_id, raw, normalized, line.page, line.page,
                              SourceRef(document_id, section_id, line.page, line.bbox), confidence)
            sections.append(current)
        current.end_page = max(current.end_page, line.page)
        ref = SourceRef(document_id, current.id, line.page, line.bbox)
        elements.append(TextElement(stable_id(document_id, "line", line.page, line.order, line.text), ref,
                                    kind, line.text, clean_space(line.text), line.order,
                                    heading[1] if heading else 0.9, extract_values(line.text), line.is_bold))
    return sections, elements


def section_at(sections: list[Section], page: int, top: float) -> str:
    selected = sections[0].id
    for section in sections[1:]:
        if (section.heading_ref.page, section.heading_ref.bbox[1]) <= (page, top):
            selected = section.id
        else:
            break
    return selected
