"""Machine-readable summary for human review."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json

from app.models import ParseResult


def make_report(result: ParseResult) -> dict:
    page_elements = Counter(e.ref.page for e in result.elements)
    page_tables = Counter(ref.page for table in result.tables for ref in table.refs)
    page_images = Counter(image.ref.page for image in result.images)
    return {
        "document_id": result.document_id,
        "source_file": result.source_file,
        "file_hash": result.file_hash,
        "parser_version": result.parser_version,
        "drug_name": asdict(result.drug_name),
        "brand_names": [asdict(name) for name in result.brand_names],
        "english_name": asdict(result.english_name) if result.english_name else None,
        "approval_numbers": result.approval_numbers,
        "revision_date": result.revision_date,
        "page_count": result.page_count,
        "section_count": len(result.sections),
        "sections": [{"id": section.id, "raw_name": section.raw_name,
                      "normalized_name": section.normalized_name, "start_page": section.start_page,
                      "end_page": section.end_page, "heading_bbox": section.heading_ref.bbox,
                      "confidence": section.confidence} for section in result.sections],
        "element_count": len(result.elements),
        "table_count": len(result.tables),
        "table_row_count": sum(len(t.rows) for t in result.tables),
        "image_count": len(result.images),
        "failed_pages": result.failed_pages,
        "blank_pages": result.blank_pages,
        "review_required": bool(result.failed_pages or result.warnings),
        "warnings": result.warnings,
        "pages": [{"page": page, "elements": page_elements[page], "tables": page_tables[page],
                   "images": page_images[page], "blank": page in result.blank_pages,
                   "review_required": page in result.failed_pages or (page_elements[page] == 0 and page not in result.blank_pages)}
                  for page in range(1, result.page_count + 1)],
        "tables": [{"id": table.id, "title": table.title,
                    "title_ref": asdict(table.title_ref) if table.title_ref else None,
                    "pages": [ref.page for ref in table.refs],
                    "rows": len(table.rows), "warnings": table.warnings, "image_paths": table.image_paths}
                   for table in result.tables],
    }


def write_report(result: ParseResult, output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir).resolve() / result.document_id
    target.mkdir(parents=True, exist_ok=True)
    parsed = target / "parsed.json"
    report = target / "report.json"
    parsed.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(json.dumps(make_report(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed, report
