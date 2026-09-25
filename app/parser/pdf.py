"""End-to-end PDF extraction. No medical content is inferred."""

from pathlib import Path
import hashlib
import re

from app.models import ParseResult, ValuePair
from app.parser.images import extract_images, extract_chemical_diagrams, extract_chart_regions
from app.parser.layout import build_sections_and_elements, lines_from_page, section_at, stable_id
from app.parser.normalize import normalize_drug_alias, normalize_drug_name
from app.parser.tables import build_table, extract_fragments, group_fragments


PARSER_VERSION = "0.3.0"

TABLE_TITLE_PATTERN = re.compile(r"^表\s*[0-9一二三四五六七八九十]+\s*[.．、:：]")


def _find_table_title(table: object, elements: list[object], page_heights: dict[int, float]):
    """Find a same-page title or a numbered title at the bottom of the previous page."""
    first_ref = table.refs[0]
    same_page = [e for e in elements if e.ref.page == first_ref.page and
                 e.ref.bbox[3] <= first_ref.bbox[1] and first_ref.bbox[1] - e.ref.bbox[3] < 45 and
                 e.element_type not in {"header", "page_number", "footer", "table_text"}]
    numbered = [e for e in same_page if TABLE_TITLE_PATTERN.match(e.raw_text.strip())]
    candidates = numbered or same_page
    if candidates:
        return max(candidates, key=lambda e: e.ref.bbox[3])
    if first_ref.page > 1 and first_ref.bbox[1] <= page_heights[first_ref.page] * 0.18:
        previous_page = [e for e in elements if e.ref.page == first_ref.page - 1 and
                         e.ref.bbox[1] >= page_heights[first_ref.page - 1] * 0.60 and
                         TABLE_TITLE_PATTERN.match(e.raw_text.strip())]
        if previous_page:
            return max(previous_page, key=lambda e: e.ref.bbox[3])
    return None


def _drug_name(source: Path, all_lines: list[object]) -> ValuePair:
    first_page = [x.text for x in all_lines if x.page == 1]
    for index, line in enumerate(first_page[:35]):
        match = re.search(r"(?:通用名称|药品名称)\s*[：:]\s*([^\s【】]{2,35})", line)
        if match:
            raw = match.group(1)
            return ValuePair(raw, normalize_drug_name(raw))
        if re.fullmatch(r"【(?:通用名称|药品名称)】", line.strip()) and index + 1 < len(first_page):
            raw = re.sub(r"^\s*(?:通用名称|药品名称)\s*[：:]\s*", "", first_page[index + 1]).strip()
            return ValuePair(raw, normalize_drug_name(raw))
    raw = re.sub(r"__?说明书.*$|肠溶胶囊$|片$", "", source.stem)
    raw = raw.split("_")[-1] if raw.startswith("必洛斯_") else raw
    return ValuePair(raw, normalize_drug_name(raw))


def _labeled_value(all_lines: list[object], label: str) -> str | None:
    first_page = [line.text.strip() for line in all_lines if line.page == 1]
    pattern = re.compile(rf"^{re.escape(label)}\s*[：:]\s*(.+)$")
    for index, text in enumerate(first_page[:40]):
        match = pattern.match(text)
        if match and match.group(1).strip():
            return match.group(1).strip()
        if re.fullmatch(rf"【?{re.escape(label)}】?\s*[：:]?", text) and index + 1 < len(first_page):
            return first_page[index + 1].strip()
    return None


def _drug_metadata(all_lines: list[object]) -> tuple[list[ValuePair], ValuePair | None, list[str], str | None]:
    brand_raw = _labeled_value(all_lines, "商品名称")
    brand_names = []
    if brand_raw:
        for value in re.split(r"[/／、]", brand_raw):
            value = value.strip()
            if value:
                brand_names.append(ValuePair(value, normalize_drug_alias(value)))
    english_raw = _labeled_value(all_lines, "英文名称")
    english_name = ValuePair(english_raw, normalize_drug_alias(english_raw)) if english_raw else None

    text = "\n".join(line.text for line in all_lines)
    approvals = set()
    for match in re.finditer(r"国药准字\s*[A-Z]{1,2}\s*\d{8}|(?<![A-Za-z0-9])[HJ]\d{8}", text, re.I):
        approvals.add(re.sub(r"\s+", "", match.group()).upper())

    revision_dates: list[tuple[int, int, int]] = []
    for line in all_lines:
        if "修改日期" not in line.text and "修订日期" not in line.text:
            continue
        for year, month, day in re.findall(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", line.text):
            revision_dates.append((int(year), int(month), int(day)))
    revision_date = "%04d-%02d-%02d" % max(revision_dates) if revision_dates else None
    return brand_names, english_name, sorted(approvals), revision_date


def parse_pdf(source: str | Path, output_dir: str | Path, parser_version: str = PARSER_VERSION) -> ParseResult:
    try:
        import pymupdf
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF dependencies missing; run: .\\.venv\\Scripts\\python.exe -m pip install -e .") from exc


    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()
    document_id = stable_id(file_hash, parser_version)
    target = Path(output_dir).resolve() / document_id
    image_dir = target / "images"
    table_dir = target / "tables"
    warnings: list[str] = []
    failed_pages: list[int] = []
    blank_pages: list[int] = []
    all_lines = []
    heights: dict[int, float] = {}
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        page_count = len(pdf)
        for index, page in enumerate(pdf):
            number = index + 1
            heights[number] = float(page.rect.height)
            try:
                page_lines = lines_from_page(page, number)
                if not page_lines:
                    image_count = sum(block.get("type") == 1 for block in page.get_text("dict")["blocks"])
                    if image_count == 0 and len(page.get_drawings()) <= 2:
                        blank_pages.append(number)
                    else:
                        warnings.append(f"第 {number} 页无可选择文本；需要人工复核或 OCR")
                        failed_pages.append(number)
                all_lines.extend(page_lines)
            except Exception as exc:
                failed_pages.append(number)
                warnings.append(f"第 {number} 页文字提取失败：{type(exc).__name__}: {exc}")
        if all_lines:
            sections, elements = build_sections_and_elements(document_id, all_lines, heights)
        else:
            raise ValueError("PDF 无可选择文本；无法建立可追溯章节。请提供 OCR 方案。")
        drug_name = _drug_name(path, all_lines)
        brand_names, english_name, approval_numbers, revision_date = _drug_metadata(all_lines)
        fragments = []
        with pdfplumber.open(path) as plumber:
            for index, page in enumerate(pdf):
                number = index + 1
                try:
                    fragments.extend(extract_fragments(plumber.pages[index], page, number, document_id, sections, table_dir))
                except Exception as exc:
                    warnings.append(f"第 {number} 页表格提取失败：{type(exc).__name__}: {exc}")
                    if number not in failed_pages:
                        failed_pages.append(number)
        regions = [(fragment.page, fragment.bbox) for fragment in fragments]
        sections, elements = build_sections_and_elements(document_id, all_lines, heights, regions)
        for fragment in fragments:
            fragment.section_id = section_at(sections, fragment.page, fragment.bbox[1])
        images = []
        for index, page in enumerate(pdf):
            number = index + 1
            try:
                images.extend(extract_images(page, number, document_id, sections, image_dir))
                images.extend(extract_chemical_diagrams(page, number, document_id, sections,
                                                        [line for line in all_lines if line.page == number], image_dir))
                images.extend(extract_chart_regions(page, number, document_id, sections,
                                                    [line for line in all_lines if line.page == number], image_dir))
            except Exception as exc:
                warnings.append(f"第 {number} 页图片提取失败：{type(exc).__name__}: {exc}")
                if number not in failed_pages:
                    failed_pages.append(number)
    tables = [build_table(group, document_id, drug_name.normalized) for group in group_fragments(fragments)]
    for table in tables:
        title_element = _find_table_title(table, elements, heights)
        if title_element:
            table.title = title_element.raw_text
            table.title_ref = title_element.ref
        warnings.extend(f"表格 {table.id}: {item}" for item in table.warnings)
    return ParseResult(document_id, str(path), file_hash, parser_version, drug_name, page_count,
                       sections, elements, tables, images, warnings, sorted(failed_pages), blank_pages,
                       brand_names, english_name, approval_numbers, revision_date)
