"""Structure-first, traceable parent/child chunks for drug leaflets."""

from dataclasses import dataclass
import hashlib
import re

from app.models import (
    ChunkRecord,
    ImageRecord,
    ParseResult,
    Section,
    SourceRef,
    TableRecord,
    TableRow,
    TextElement,
)
from app.parser.layout import stable_id
from app.chunker.tokens import estimated_token_count
from app.chunker.summaries import build_summary_chunks

CHUNKER_VERSION = "0.3.0"
DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_PARENT_SOFT_MAX_TOKENS = 1200
_SENTENCE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?|[。！？!?；;]+")
_NOISE = {"header", "footer", "page_number", "table_text", "excluded_from_retrieval"}
_TOPICS = (
    "间质性肺病",
    "肾功能不全",
    "肾功能损害",
    "肝功能不全",
    "肝功能损害",
    "皮肤不良反应",
    "皮疹",
    "腹泻",
    "低血压",
    "高血钾",
    "肾功能异常",
    "过敏反应",
    "孕妇",
    "哺乳期",
    "儿童",
    "老年人",
    "老年患者",
    "PPI",
    "质子泵抑制剂",
    "NSAIDs",
    "非甾体抗炎药",
    "CYP抑制剂",
    "CYP诱导剂",
    "推荐剂量",
    "漏服",
    "剂量调整",
    "降低剂量",
    "暂停用药",
    "恢复用药",
    "永久停药",
    "监测",
)
_TREATMENT = {"降低剂量", "暂停用药", "恢复用药", "永久停药", "监测"}
_SUBHEADING_ALIASES = {"严重的不良反应": "严重不良反应", "严重不良事件": "严重不良反应"}
_TOPIC_ALIASES = {
    "质子泵抑制剂": "PPI",
    "非甾体抗炎药": "NSAIDs",
    "肾功能不全": "肾功能损害",
    "肝功能不全": "肝功能损害",
    "老年患者": "老年人",
}
_POPULATIONS = {
    "elderly": ("老年", "高龄"),
    "pediatric": ("儿童", "小儿", "婴儿", "新生儿"),
    "pregnancy": ("孕妇", "妊娠", "怀孕"),
    "lactation": ("哺乳", "乳汁"),
    "renal_impairment": ("肾功能不全", "肾损害", "肾功能损害", "肾功能异常"),
    "hepatic_impairment": ("肝功能不全", "肝损害", "肝功能损害", "肝功能异常"),
}
_SECTION_INTENTS = {
    "适应症": "indication",
    "用法用量": "dose",
    "不良反应": "adverse_reaction",
    "禁忌": "contraindication",
    "注意事项": "warning",
    "药物相互作用": "interaction",
    "儿童用药": "special_population",
    "老年用药": "special_population",
    "孕妇及哺乳期妇女用药": "special_population",
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_refs(refs: list[SourceRef]) -> list[SourceRef]:
    seen = set()
    result = []
    for ref in refs:
        key = (ref.document_id, ref.section_id, ref.page, ref.bbox)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _bbox(refs: list[SourceRef]) -> tuple[float, float, float, float] | None:
    if len({ref.page for ref in refs}) != 1:
        return None
    return (
        min(r.bbox[0] for r in refs),
        min(r.bbox[1] for r in refs),
        max(r.bbox[2] for r in refs),
        max(r.bbox[3] for r in refs),
    )


@dataclass
class _Block:
    raw: str
    refs: list[SourceRef]
    source_ids: list[str]
    kind: str
    order: int
    path: list[str]
    topic: str = ""
    topic_reason: str = ""
    split_reason: str = "paragraph_boundary"
    lead_source_id: str | None = None
    source_offset: int = 0


def _subheading(element: TextElement) -> str | None:
    text = element.raw_text.strip()
    if element.element_type == "subsection_heading":
        return text.rstrip("：:")
    if not text or len(text) > 42:
        return None
    if element.is_bold and not re.search(r"[。！？!?；;]", text):
        return text.rstrip("：:")
    if re.fullmatch(
        r"(?:\d+(?:\.\d+)+[.、]?\s*|[（(]\d+[）)]\s*)[^。！？!?；;：:]{2,30}[：:]?",
        text,
    ):
        return text.rstrip("：:")
    if (
        text.endswith(("：", ":"))
        and len(text) <= 22
        and not re.search(r"\d+\s*(?:mg|g|ml|mL|%)", text)
    ):
        return text[:-1]
    return None


"""
将文本行整理成段落、列表项和子标题。
  1.忽略页眉、页脚、页码、重复表格文本；
  2.章节标题不作为普通正文块；
  3.识别显式小标题、粗体短文本、编号标题和以冒号结尾的短标题；
  4.普通 PDF 行会持续合并，直到遇到句号、分号等结束符；
  5.列表项单独形成块；
  6.“处理原则：”这类前导段落会记录为后续列表的 lead_source_id；
  7.文本即使跨页，只要语义没有结束，也允许合并。
"""


def _blocks(section: Section, elements: list[TextElement],
            excluded: set[str] | None = None,
            force_split_after: set[str] | None = None,
            heading_parents: dict[str, str] | None = None,
            element_split_groups: list[list[str]] | None = None) -> list[_Block]:
    blocks: list[_Block] = []
    pending: list[TextElement] = []
    path = [section.normalized_name]
    lead: str | None = None
    excluded = excluded or set()
    force_split_after = force_split_after or set()
    heading_parents = heading_parents or {}
    by_id = {item.id: item for item in elements}
    # Multiple editable Elements can still represent one original semantic unit.
    # Group only known split lineage; unrelated paragraphs retain normal rules.
    from copy import copy
    grouped = {}
    continuations = set()
    positions={e.id:i for i,e in enumerate(sorted(elements,key=lambda e:(e.ref.page,e.reading_order)))}
    for ids in element_split_groups or []:
        members=[by_id[key] for key in ids if key in by_id and key not in excluded]
        if len(members)<2 or len({e.ref.page for e in members})!=1:
            continue
        indices=[positions[e.id] for e in members]
        if any(b!=a+1 for a,b in zip(indices,indices[1:])) or any(e.id in heading_parents or e.id in force_split_after for e in members[:-1]):
            continue
        grouped[members[0].id]=members
        continuations.update(e.id for e in members[1:])
    units=[]
    for element in elements:
        if element.id in continuations:continue
        if element.id in grouped:
            unit=copy(element)
            unit.raw_text=''.join(e.raw_text for e in grouped[element.id])
            unit.normalized_text=unit.raw_text
            units.append(unit)
        else:units.append(element)
    def sources(item):return grouped.get(item.id,[item])
    def boundary(item):return any(e.id in force_split_after for e in sources(item))

    def parent_path(element_id: str, seen: set[str] | None = None) -> list[str]:
        seen = set() if seen is None else seen
        if element_id in seen or element_id not in by_id:
            raise ValueError("标题归属存在循环或父级不在当前章节")
        seen.add(element_id)
        item = by_id[element_id]
        prefix = parent_path(heading_parents[element_id], seen) if element_id in heading_parents else [section.normalized_name]
        if item.element_type != "section_heading":
            prefix.append(_subheading(item) or item.raw_text.strip())
        return prefix

    # 将正文合并成一个块
    def flush() -> None:
        nonlocal lead
        if not pending:
            return
        first = pending[0]
        raw = "\n".join(e.raw_text.strip() for e in pending)
        blocks.append(
            _Block(
                raw,
                [member.ref for e in pending for member in sources(e)],
                [member.id for e in pending for member in sources(e)],
                first.element_type,
                first.ref.page * 100_000 + first.reading_order,
                path.copy(),
                split_reason="review_boundary" if boundary(pending[-1]) else (
                    "list_item_boundary" if first.element_type == "list_item" else "paragraph_boundary"),
                lead_source_id=lead if first.element_type == "list_item" else None,
            )
        )
        lead = (
            first.id
            if raw.endswith(("：", ":"))
            else (lead if first.element_type == "list_item" else None)
        )
        pending.clear()

    for element in sorted(units, key=lambda e: (e.ref.page, e.reading_order)):
        if element.id in excluded or element.element_type in _NOISE or not element.raw_text.strip():
            continue
        assigned_path = parent_path(heading_parents[element.id]) if element.id in heading_parents else None
        if assigned_path is not None:
            flush()
            path = assigned_path
        if element.element_type == "section_heading":
            flush()
            inline = re.match(r"^【[^】]+】\s*(.+)$", element.raw_text.strip())
            if inline:
                blocks.append(
                    _Block(
                        inline.group(1),
                        [member.ref for member in sources(element)],
                        [member.id for member in sources(element)],
                        "paragraph",
                        element.ref.page * 100_000 + element.reading_order,
                        path.copy(),
                        split_reason="section_boundary",
                    )
                )
            continue
        heading = _subheading(element)
        if heading:
            flush()
            path = [*(assigned_path or [section.normalized_name]), _SUBHEADING_ALIASES.get(heading, heading)]
            blocks.append(
                _Block(
                    element.raw_text.strip(),
                    [member.ref for member in sources(element)],
                    [member.id for member in sources(element)],
                    "subsection_heading",
                    element.ref.page * 100_000 + element.reading_order,
                    path.copy(),
                    split_reason="subheading_boundary",
                )
            )
            lead = (
                element.id if element.raw_text.strip().endswith(("：", ":")) else None
            )
            continue
        if element.element_type == "list_item":
            flush()
            pending.append(element)
            flush()
            continue
        if pending and pending[0].element_type != element.element_type:
            flush()
        pending.append(element)
        if boundary(element):
            flush()
            continue
        # PDF lines can continue across a page; only punctuation or structure ends a unit.
        if re.search(r"[。！？!?；;]\s*$", element.raw_text):
            flush()
    flush()
    return blocks


def _topic_candidate(text: str) -> str | None:
    prefix = re.match(r"^([^：:。；;]{2,22})[：:]", text.strip())
    if prefix and not re.search(r"\d+\s*(?:mg|g|ml|mL|%)", prefix.group(1)):
        value = re.sub(r"^[（(]?\d+(?:\.\d+)*[）).、]?\s*", "", prefix.group(1)).strip()
        if value:
            return _TOPIC_ALIASES.get(value, value)
    term = next(
        (term for term in _TOPICS if re.search(re.escape(term), text, re.I)), None
    )
    return _TOPIC_ALIASES.get(term, term)


"""
给每个块标记主题
"""


def _assign_topics(blocks: list[_Block], section: Section) -> None:
    current = section.normalized_name
    for block in blocks:
        if block.kind == "subsection_heading":
            current = block.path[-1]
            block.topic_reason = "subheading"
        else:
            candidate = _topic_candidate(block.raw)
            # Keep explicit field labels as topic levels. A correction replay
            # must not collapse existing 药品名称 / 通用名称 paths.
            if candidate and (
                candidate not in _TREATMENT or current == section.normalized_name
            ):
                current = candidate
                block.topic_reason = "term_or_label"
            else:
                block.topic_reason = (
                    "adjacent_context"
                    if current != section.normalized_name
                    else "section_default"
                )
        block.topic = current
        if len(block.path) == 1 and current != section.normalized_name:
            block.path = [section.normalized_name, current]
        elif (
            len(block.path) > 1
            and current != block.path[-1]
            and block.kind != "subsection_heading"
        ):
            block.path = [*block.path[:2], current]


"""
超长语义块的句子边界兜底切分
  1. 块没有超过 max_tokens：原样返回。
  2. 超过上限：按完整句子组成几个子块；如果空间允许，会在相邻子块间重复前一个完整句子，并标记重叠。超过上限：按完整句子组成几个子块；如果空间允许，会在相邻子块间重复前一个完整句子，并标记重叠。
  3. 单个句子本身就超长：保留整句，标记为 oversized_sentence，避免把条件、否定词或剂量单位截断。
"""


def _fallback(
    block: _Block, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, str, bool, int, int]]:
    """Sentence-boundary fallback. An indivisible long sentence stays whole and flagged."""
    if estimated_token_count(block.raw) <= max_tokens:
        return [(block.raw, block.split_reason, False, 0, len(block.raw))]
    sentences = [(m.group(), m.start(), m.end()) for m in _SENTENCE.finditer(block.raw) if m.group().strip()]
    if len(sentences) <= 1:
        return [(block.raw, "oversized_sentence", False, 0, len(block.raw))]
    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    for sentence in sentences:
        if (
            current
            and estimated_token_count("".join(item[0] for item in [*current, sentence])) > max_tokens
        ):
            groups.append(current)
            current = []
            if (
                overlap_tokens
                and estimated_token_count(groups[-1][-1][0] + sentence[0]) <= max_tokens
            ):
                current.append(groups[-1][-1])
        current.append(sentence)
    if current:
        groups.append(current)
    result = []
    for index, group in enumerate(groups):
        start, end = group[0][1], group[-1][2]
        original = block.raw[start:end]
        raw = original.strip()
        start += len(original) - len(original.lstrip())
        end -= len(original) - len(original.rstrip())
        reason = (
            "oversized_sentence"
            if estimated_token_count(raw) > max_tokens
            else "token_fallback"
        )
        overlapped = index > 0 and bool(
            overlap_tokens and group[0] == groups[index - 1][-1]
        )
        result.append((raw, reason, overlapped, start, end))
    return result


def _chunk(
    result: ParseResult,
    drug_id: str,
    product_id: str,
    section: Section,
    kind: str,
    raw: str,
    retrieval: str,
    refs: list[SourceRef],
    order: int,
    identity: tuple[object, ...],
    *,
    parent_id: str | None,
    is_parent: bool,
    path: list[str],
    topic: str | None,
    parent_type: str | None,
    reason: str,
    source_ids: list[str],
    lead_in_id: str | None = None,
    overlap_ids: list[str] | None = None,
    table_id: str | None = None,
    topic_reason: str | None = None,
) -> ChunkRecord:
    refs = _unique_refs(refs)
    if not refs:
        raise ValueError("Chunk has no page and bbox source")
    raw = raw.strip()
    digest = _hash(raw)
    chunk_id = stable_id(
        result.document_id, "chunk", CHUNKER_VERSION, kind, *identity, digest
    )
    topic_id = (
        stable_id(result.document_id, "topic", section.id, topic) if topic else None
    )
    populations = sorted(
        k
        for k, terms in _POPULATIONS.items()
        if any(term in raw or term in section.normalized_name for term in terms)
    )
    intents = (
        [_SECTION_INTENTS[section.normalized_name]]
        if section.normalized_name in _SECTION_INTENTS
        else []
    )
    for key, terms in {
        "dose_adjustment": ("调整剂量", "剂量调整", "降低剂量", "减量"),
        "missed_dose": ("漏服", "漏用"),
        "monitoring": ("监测", "定期检查"),
        "discontinuation": ("停药", "暂停用药", "暂停给药", "永久停用"),
    }.items():
        if any(term in raw for term in terms):
            intents.append(key)
    metadata = {
        "document_id": result.document_id,
        "drug_id": drug_id,
        "product_id": product_id,
        "drug_name": result.drug_name.raw,
        "generic_name": result.drug_name.normalized,
        "brand_name": [x.normalized for x in result.brand_names],
        "english_name": result.english_name.normalized if result.english_name else None,
        "section": section.normalized_name,
        "raw_section_name": section.raw_name,
        "normalized_section_name": section.normalized_name,
        "section_path": path.copy(),
        "subsection": path[1] if len(path) > 1 else None,
        "topic_id": topic_id,
        "topic_name": topic,
        "topic_reason": topic_reason,
        "parent_id": parent_id,
        "parent_type": parent_type,
        "content_type": kind,
        "index_channel": "evidence",
        "populations": populations,
        "diseases": [],
        "drug_classes": [],
        "intent_tags": sorted(set(intents)),
        "version": result.parser_version,
        "revision_date": result.revision_date,
        "source_file": result.source_file,
        "content_hash": digest,
        "table_id": table_id,
        "raw_text": raw,
        "split_reason": reason,
        "source_element_ids": source_ids.copy(),
        "lead_in_id": lead_in_id,
        "overlap_source_ids": overlap_ids or [],
        "previous_chunk_id": None,
        "next_chunk_id": None,
    }
    pages = [ref.page for ref in refs]
    first_ref = min(refs, key=lambda ref: (ref.page, ref.bbox[1], ref.bbox[0]))
    # One sortable integer across text, tables and images. The small tail
    # distinguishes sentence pieces that share exactly the same source box.
    visual_order = (
        first_ref.page * 5_000_000
        + int(first_ref.bbox[1] * 50) * 100
        + min(int(first_ref.bbox[0] / 10), 99)
        + order % 100
    )
    return ChunkRecord(
        chunk_id,
        result.document_id,
        section.id,
        parent_id,
        raw,
        retrieval.strip(),
        kind,
        is_parent,
        estimated_token_count(raw),
        min(pages),
        max(pages),
        _bbox(refs),
        visual_order,
        refs,
        metadata,
        digest,
        CHUNKER_VERSION,
        path.copy(),
        section.raw_name,
        section.normalized_name,
        topic_id,
        topic,
        parent_type,
        None,
        None,
        lead_in_id,
        reason,
        overlap_ids or [],
        source_ids.copy(),
    )


def _link(children: list[ChunkRecord]) -> None:
    ordered = sorted(children, key=lambda c: (c.reading_order, c.id))
    for index, child in enumerate(ordered):
        child.previous_chunk_id = ordered[index - 1].id if index else None
        child.next_chunk_id = (
            ordered[index + 1].id if index + 1 < len(ordered) else None
        )
        child.metadata["previous_chunk_id"] = child.previous_chunk_id
        child.metadata["next_chunk_id"] = child.next_chunk_id


@dataclass(frozen=True)
class _BuildContext:
    result: ParseResult
    drug_id: str
    product_id: str
    sections: dict[str, Section]
    max_tokens: int
    overlap_tokens: int
    parent_soft_max_tokens: int


def _elements_by_section(
    result: ParseResult, sections: dict[str, Section]
) -> dict[str, list[TextElement]]:
    grouped: dict[str, list[TextElement]] = {section_id: [] for section_id in sections}
    for element in result.elements:
        if element.ref.section_id in grouped:
            grouped[element.ref.section_id].append(element)
    return grouped


def _split_oversized_blocks(blocks: list[_Block], soft_max_tokens: int) -> list[_Block]:
    expanded: list[_Block] = []
    for block in blocks:
        if estimated_token_count(block.raw) <= soft_max_tokens:
            expanded.append(block)
            continue
        for part, _, _, start, _ in _fallback(block, soft_max_tokens, 0):
            expanded.append(
                _Block(
                    part,
                    block.refs,
                    block.source_ids,
                    block.kind,
                    block.order,
                    block.path.copy(),
                    block.topic,
                    block.topic_reason,
                    "sentence_boundary",
                    block.lead_source_id,
                    block.source_offset + start,
                )
            )
    return expanded


"""
连续、同主题且大小合适的块放进一组。
"""


def _group_topic_blocks(
    blocks: list[_Block], soft_max_tokens: int
) -> list[list[_Block]]:
    groups: list[list[_Block]] = []
    for block in blocks:
        if not groups:
            groups.append([block])
            continue

        current = groups[-1]
        heading_waits_for_body = (
            len(current) == 1 and current[0].kind == "subsection_heading"
        )
        over_limit = (
            not heading_waits_for_body
            and estimated_token_count(
                "\n".join(item.raw for item in current) + block.raw
            )
            > soft_max_tokens
        )
        if (
            block.kind == "subsection_heading"
            or block.split_reason == "review_boundary"
            or current[-1].topic != block.topic
            or current[-1].path != block.path
            or over_limit
        ):
            groups.append([])
        groups[-1].append(block)
    return groups


def _block_children(
    context: _BuildContext,
    section: Section,
    block: _Block,
    parent: ChunkRecord,
    heading_by_path: dict[tuple[str, ...], _Block],
    source_to_chunk: dict[str, str],
) -> list[ChunkRecord]:
    children: list[ChunkRecord] = []
    previous_piece: ChunkRecord | None = None
    heading = heading_by_path.get(tuple(block.path[:2]))
    for piece_index, (piece, reason, overlapped, start, end) in enumerate(
        _fallback(block, context.max_tokens, context.overlap_tokens)
    ):
        child = _chunk(
            context.result,
            context.drug_id,
            context.product_id,
            section,
            block.kind,
            piece,
            f"{context.result.drug_name.normalized}；{section.normalized_name}；{block.topic}；{piece}",
            block.refs,
            block.order + piece_index,
            (*block.source_ids, piece_index),
            parent_id=parent.id,
            is_parent=False,
            path=block.path,
            topic=block.topic,
            parent_type="topic_group",
            reason=reason,
            source_ids=block.source_ids,
            lead_in_id=source_to_chunk.get(block.lead_source_id or "")
            or block.lead_source_id,
            overlap_ids=[previous_piece.id] if overlapped and previous_piece else [],
            topic_reason=block.topic_reason,
        )
        child.metadata["raw_subsection_name"] = (
            heading.raw.rstrip("：:") if heading else None
        )
        child.metadata["normalized_subsection_name"] = (
            block.path[1] if heading else None
        )
        child.metadata["source_text_start"] = block.source_offset + start
        child.metadata["source_text_end"] = block.source_offset + end
        children.append(child)
        previous_piece = child

    if previous_piece:
        for source_id in block.source_ids:
            source_to_chunk[source_id] = previous_piece.id
    return children


"""
每组生成一个局部 Parent，并为正文或列表生成 Child
"""
def _topic_group_chunks(
    context: _BuildContext,
    section: Section,
    group: list[_Block],
    group_index: int,
    heading_by_path: dict[tuple[str, ...], _Block],
    source_to_chunk: dict[str, str],
) -> tuple[ChunkRecord, list[ChunkRecord]]:
    heading = heading_by_path.get(tuple(group[0].path[:2]))
    parent_blocks = ([heading] if heading and group[0] is not heading else []) + group
    raw = "\n".join(block.raw for block in parent_blocks)
    topic = group[0].topic
    parent = _chunk(
        context.result,
        context.drug_id,
        context.product_id,
        section,
        "topic_parent",
        raw,
        f"{context.result.drug_name.normalized}；{section.normalized_name}；{topic}；{raw}",
        [ref for block in parent_blocks for ref in block.refs],
        group[0].order,
        (section.id, topic, group_index),
        parent_id=None,
        is_parent=True,
        path=group[0].path,
        topic=topic,
        parent_type="topic_group",
        reason="topic_boundary",
        source_ids=[
            source_id for block in parent_blocks for source_id in block.source_ids
        ],
        topic_reason=group[0].topic_reason,
    )
    parent.metadata["raw_subsection_name"] = (
        heading.raw.rstrip("：:") if heading else None
    )
    parent.metadata["normalized_subsection_name"] = (
        group[0].path[1] if heading else None
    )

    children: list[ChunkRecord] = []
    for block in group:
        if block.kind != "subsection_heading":
            children.extend(
                _block_children(
                    context,
                    section,
                    block,
                    parent,
                    heading_by_path,
                    source_to_chunk,
                )
            )
    return parent, children


def _section_chunks(
    context: _BuildContext,
    section: Section,
    elements: list[TextElement],
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    policy = context.result.chunk_policy
    blocks = _blocks(section, elements, set(policy.get("exclude_element_ids", [])),
                     set(policy.get("force_split_after_ids", [])), policy.get("heading_parents", {}),
                     policy.get("element_split_groups", []))
    _assign_topics(blocks, section)
    blocks = _split_oversized_blocks(blocks, context.parent_soft_max_tokens)
    groups = _group_topic_blocks(blocks, context.parent_soft_max_tokens)
    heading_by_path = {
        tuple(block.path): block
        for block in blocks
        if block.kind == "subsection_heading"
    }
    source_to_chunk: dict[str, str] = {}
    parents: list[ChunkRecord] = []
    children: list[ChunkRecord] = []

    for index, group in enumerate(groups):
        if all(block.kind == "subsection_heading" for block in group):
            continue
        parent, group_children = _topic_group_chunks(
            context,
            section,
            group,
            index,
            heading_by_path,
            source_to_chunk,
        )
        parents.append(parent)
        children.extend(group_children)
    _link(children)
    return parents, children


def _table_row_groups(
    table: TableRecord, title: str
) -> list[tuple[str, list[TableRow]]]:
    groups: list[tuple[str, list[TableRow]]] = []
    for row in table.rows:
        key = (
            str(row.normalized_cells[0])
            if row.normalized_cells and row.normalized_cells[0]
            else title
        )
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(row)
    return groups


def _table_row_group_chunks(
    context: _BuildContext,
    section: Section,
    table: TableRecord,
    table_parent: ChunkRecord,
    title: str,
    headers: list[str],
    group_index: int,
    key: str,
    rows: list[TableRow],
) -> tuple[ChunkRecord, list[ChunkRecord]]:
    group_raw = "\n".join(
        "；".join(str(value or "") for value in row.raw_cells) for row in rows
    )
    parent = _chunk(
        context.result,
        context.drug_id,
        context.product_id,
        section,
        "row_group_parent",
        group_raw,
        f"{title}；{key}；{group_raw}",
        [ref for row in rows for ref in row.refs],
        rows[0].refs[0].page * 100_000 + rows[0].row_index,
        (table.id, "group", group_index),
        parent_id=table_parent.id,
        is_parent=True,
        path=[section.normalized_name, title, key],
        topic=key,
        parent_type="row_group",
        reason="table_group_boundary",
        source_ids=[row.id for row in rows],
        table_id=table.id,
    )
    children: list[ChunkRecord] = []
    for row in rows:
        raw = "；".join(str(value or "") for value in row.raw_cells)
        normalized = "；".join(
            f"{headers[index] if index < len(headers) else f'列{index + 1}'}：{value}"
            for index, value in enumerate(row.normalized_cells)
            if value
        )
        children.append(
            _chunk(
                context.result,
                context.drug_id,
                context.product_id,
                section,
                "table_row",
                raw,
                f"{context.result.drug_name.normalized}；{section.normalized_name}；{title}；{key}；{normalized}",
                row.refs,
                row.refs[0].page * 100_000 + row.row_index,
                (table.id, row.id),
                parent_id=parent.id,
                is_parent=False,
                path=[section.normalized_name, title, key],
                topic=key,
                parent_type="row_group",
                reason="table_row_boundary",
                source_ids=[row.id],
                table_id=table.id,
            )
        )
    _link(children)
    return parent, children


def _table_chunks(
    context: _BuildContext,
    table: TableRecord,
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    section = context.sections[table.section_id]
    title = table.title.strip() or "表格"
    table_refs = ([table.title_ref] if table.title_ref else []) + table.refs
    table_parent = _chunk(
        context.result,
        context.drug_id,
        context.product_id,
        section,
        "table",
        f"{title}\n{table.markdown}",
        f"{context.result.drug_name.normalized}；{section.normalized_name}；{title}；{table.markdown}",
        table_refs,
        table.refs[0].page * 100_000,
        (table.id,),
        parent_id=None,
        is_parent=True,
        path=[section.normalized_name, title],
        topic=title,
        parent_type="table",
        reason="table_boundary",
        source_ids=[table.id],
        table_id=table.id,
    )
    normalized_rows = table.normalized_json.get("rows", [])
    headers = (
        [
            str(value or f"列{index + 1}")
            for index, value in enumerate(normalized_rows[0])
        ]
        if normalized_rows
        else []
    )

    parents = [table_parent]
    children: list[ChunkRecord] = []
    for index, (key, rows) in enumerate(_table_row_groups(table, title)):
        row_parent, row_children = _table_row_group_chunks(
            context,
            section,
            table,
            table_parent,
            title,
            headers,
            index,
            key,
            rows,
        )
        parents.append(row_parent)
        children.extend(row_children)
    return parents, children


def _image_chunks(
    context: _BuildContext,
    image: ImageRecord,
) -> tuple[ChunkRecord, ChunkRecord]:
    section = context.sections[image.ref.section_id]
    parent = _chunk(
        context.result,
        context.drug_id,
        context.product_id,
        section,
        "image_parent",
        image.caption,
        image.caption,
        [image.ref],
        image.ref.page * 100_000,
        (image.id, "parent"),
        parent_id=None,
        is_parent=True,
        path=[section.normalized_name],
        topic=image.caption,
        parent_type="image",
        reason="image_boundary",
        source_ids=[image.id],
    )
    child = _chunk(
        context.result,
        context.drug_id,
        context.product_id,
        section,
        "image",
        image.caption,
        f"{context.result.drug_name.normalized}；{section.normalized_name}；{image.caption}",
        [image.ref],
        image.ref.page * 100_000,
        (image.id,),
        parent_id=parent.id,
        is_parent=False,
        path=[section.normalized_name],
        topic=image.caption,
        parent_type="image",
        reason="image_boundary",
        source_ids=[image.id],
    )
    return parent, child


def build_chunks(
    result: ParseResult,
    drug_id: str,
    product_id: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    parent_soft_max_tokens: int = DEFAULT_PARENT_SOFT_MAX_TOKENS,
) -> list[ChunkRecord]:
    """Build text, table and image chunks; return parents before their children."""
    if max_tokens <= 0 or parent_soft_max_tokens <= 0:
        raise ValueError("token limits must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens")

    sections = {section.id: section for section in result.sections}
    context = _BuildContext(
        result,
        drug_id,
        product_id,
        sections,
        max_tokens,
        overlap_tokens,
        parent_soft_max_tokens,
    )
    by_section = _elements_by_section(result, sections)
    parents: list[ChunkRecord] = []
    children: list[ChunkRecord] = []

    for section in result.sections:
        section_parents, section_children = _section_chunks(
            context,
            section,
            by_section[section.id],
        )
        parents.extend(section_parents)
        children.extend(section_children)

    for table in result.tables:
        table_parents, table_children = _table_chunks(context, table)
        from app.review.table_audit import cell_id
        cells=table.raw_json.get('audit_cells') or [dict(row=c.row,column=c.column,text=c.raw or '',rowspan=1,colspan=1) for c in table.cells]
        rows={row.id:row.row_index for row in table.rows}
        for chunk in [*table_parents,*table_children]:
            chunk.metadata['cell_ids']=[cell_id(table.id,c['row'],c['column']) for c in cells if table.id in chunk.source_element_ids or any(rows.get(key)==c['row'] for key in chunk.source_element_ids)]
        parents.extend(table_parents)
        children.extend(table_children)

    for image in result.images:
        if image.caption and image.caption.strip():
            parent, child = _image_chunks(context, image)
            parents.append(parent)
            children.append(child)

    if result.chunk_policy.get("chunk_operations"):
        from app.review.chunk_operations import apply_chunk_operations
        children = apply_chunk_operations(result, children)
        # Rebuild affected retrieval parents so moved/merged evidence is not
        # served through a stale parent body. Children retain exact PDF refs.
        affected = {key for c in children for key in c.metadata.get("audit_previous_parent_ids", [])}
        retained = []
        for parent in parents:
            if parent.id not in affected:
                retained.append(parent)
                continue
            members = sorted((c for c in children if c.parent_id == parent.id), key=lambda c: (c.reading_order, c.id))
            if not members:
                continue
            parent.source_element_ids = list(dict.fromkeys(key for c in members for key in c.source_element_ids))
            parent.content = "\n".join(c.content for c in members)
            parent.retrieval_text = parent.content
            parent.refs = _unique_refs([ref for c in members for ref in c.refs])
            parent.page_start = min(ref.page for ref in parent.refs)
            parent.page_end = max(ref.page for ref in parent.refs)
            parent.bbox = _bbox(parent.refs)
            parent.token_count = estimated_token_count(parent.content)
            parent.content_hash = _hash(parent.content)
            parent.id = stable_id(result.document_id, "parent-review", parent.parent_type, *parent.source_element_ids, parent.content_hash)
            parent.metadata.update(raw_text=parent.content, source_element_ids=parent.source_element_ids.copy(), content_hash=parent.content_hash)
            for child in members:
                child.parent_id = parent.id
                child.metadata["parent_id"] = parent.id
            retained.append(parent)
        parents = retained
    parent_ids = {parent.id for parent in parents}
    if any(child.parent_id not in parent_ids for child in children):
        raise AssertionError("Every child must have a stored parent")
    summaries = build_summary_chunks(
        result, children, drug_id, product_id, CHUNKER_VERSION
    )
    applied_overrides: set[str] = set()
    for summary in summaries:
        key = "|".join((summary.content_type, summary.normalized_section_name,
                        summary.topic_name or ""))
        replacement = result.summary_overrides.get(key)
        if replacement:
            applied_overrides.add(key)
            summary.content = replacement
            summary.retrieval_text = replacement
            summary.content_hash = _hash(replacement)
            summary.id = stable_id(result.document_id, "summary-review", CHUNKER_VERSION,
                                   key, summary.content_hash)
            summary.token_count = estimated_token_count(replacement)
            summary.metadata["content_hash"] = summary.content_hash
            summary.metadata["raw_text"] = replacement
            summary.metadata["summary_edit_version"] = result.parser_version
    if set(result.summary_overrides) - applied_overrides:
        raise ValueError("摘要目标在新版本中不存在，需人工重新核对")
    if result.chunk_policy.get("chunk_operations"):
        from app.review.chunk_operations import apply_chunk_operations
        reviewed = apply_chunk_operations(result, [*parents, *children, *summaries], phase="metadata")
        parents = sorted((chunk for chunk in reviewed if chunk.is_parent), key=lambda chunk: (chunk.reading_order, chunk.id))
        combined = [chunk for chunk in reviewed if not chunk.is_parent]
    else:
        combined = [*children, *summaries]
    return [
        *parents,
        *sorted(combined, key=lambda chunk: (chunk.reading_order, chunk.id)),
    ]
