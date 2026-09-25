"""Extract embedded raster images with page coordinates."""

from pathlib import Path
import hashlib
import re

from app.models import ImageRecord, Section, SourceRef
from app.parser.layout import section_at, stable_id


def extract_images(page: object, page_number: int, document_id: str,
                   sections: list[Section], output_dir: Path) -> list[ImageRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    found: list[ImageRecord] = []
    for index, block in enumerate(page.get_text("dict")["blocks"]):
        if block.get("type") != 1 or not block.get("image"):
            continue
        bbox = tuple(round(float(x), 2) for x in block["bbox"])
        raw = block["image"]
        extension = block.get("ext", "png")
        if extension not in {"png", "jpg", "jpeg", "jpx", "bmp", "tiff"}:
            extension = "bin"
        asset_hash = hashlib.sha256(raw).hexdigest()[:16]
        image_id = stable_id(document_id, "image", page_number, index, bbox, asset_hash)
        target = output_dir / f"{image_id}.{extension}"
        if not target.exists():
            target.write_bytes(raw)
        section_id = section_at(sections, page_number, bbox[1])
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        image_type = "glyph_fragment" if width < 16 or height < 16 else "chart" if width > 200 and height > 90 else "unclassified"
        confidence = 0.8 if image_type == "glyph_fragment" else 0.65 if image_type == "chart" else 0.0
        found.append(ImageRecord(image_id, SourceRef(document_id, section_id, page_number, bbox),
                                 image_type, str(target), None, confidence))
    return found


def extract_chemical_diagrams(page: object, page_number: int, document_id: str,
                              sections: list[Section], lines: list[object],
                              output_dir: Path) -> list[ImageRecord]:
    """Render vector chemical diagrams between printed diagram and formula labels."""
    import pymupdf

    result: list[ImageRecord] = []
    ordered = sorted(lines, key=lambda line: line.order)
    for index, line in enumerate(ordered):
        if "化学结构式" not in line.text and "结构式：" not in line.text:
            continue
        lower = next((other for other in ordered[index + 1:]
                      if any(term in other.text for term in ("分子式", "分子量", "【性状】"))), None)
        if lower is None or lower.bbox[1] - line.bbox[3] < 20 or lower.bbox[1] - line.bbox[3] > 250:
            continue
        bbox = (max(0.0, line.bbox[0] - 5), line.bbox[3] + 2,
                min(float(page.rect.width), max(line.bbox[2] + 200, float(page.rect.width) - 50)),
                lower.bbox[1] - 2)
        image_id = stable_id(document_id, "chemical-diagram", page_number, bbox)
        target = output_dir / f"{image_id}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(bbox), alpha=False).save(str(target))
        section_id = section_at(sections, page_number, bbox[1])
        result.append(ImageRecord(image_id, SourceRef(document_id, section_id, page_number, bbox),
                                  "chemical_structure", str(target), line.text, 0.8))
    return result


def extract_chart_regions(page: object, page_number: int, document_id: str,
                          sections: list[Section], lines: list[object],
                          output_dir: Path) -> list[ImageRecord]:
    """Save full-page chart regions when a numbered figure caption is present."""
    import pymupdf

    captions = sorted((line for line in lines if re.match(r"^图\s*\d+\s*[.．、:]", line.text.strip())),
                      key=lambda line: line.bbox[1])
    result: list[ImageRecord] = []
    for index, caption in enumerate(captions):
        bottom = captions[index + 1].bbox[1] - 5 if index + 1 < len(captions) else float(page.rect.height) - 60
        if bottom - caption.bbox[3] < 50:
            continue
        bbox = (max(0.0, caption.bbox[0] - 10), caption.bbox[1] - 4,
                min(float(page.rect.width), float(page.rect.width) - 35), bottom)
        image_id = stable_id(document_id, "chart-region", page_number, bbox)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{image_id}.png"
        if not target.exists():
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(bbox), alpha=False).save(str(target))
        section_id = section_at(sections, page_number, bbox[1])
        result.append(ImageRecord(image_id, SourceRef(document_id, section_id, page_number, bbox),
                                  "chart", str(target), caption.text, 0.85))
    return result
