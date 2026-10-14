"""Source-grounded navigation summaries for broad leaflet queries.

These records describe where to look. They never replace the referenced
paragraph or table row as medical evidence.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re

from app.chunker.tokens import estimated_token_count
from app.models import ChunkRecord, ParseResult, Section, SourceRef
from app.parser.layout import stable_id


SUMMARY_TYPES = ("section_summary", "topic_summary")
_DIMENSIONS = {
    "发生率": re.compile(r"发生率|发生频率|常见|百分比|%"),
    "严重程度": re.compile(r"严重|[1-4１-４]级|等级|致命"),
    "发生时间": re.compile(r"发生时间|首次|再次|发生后|第\s*\d+\s*天"),
    "处理措施": re.compile(r"暂停|停药|恢复|支持治疗|处理|监测"),
    "剂量调整": re.compile(r"剂量|减量|mg|毫克", re.I),
    "特殊人群": re.compile(r"儿童|老年|妊娠|孕妇|肾功能|肝功能"),
    "相互作用": re.compile(r"相互作用|PPI|NSAID|合用", re.I),
}


@dataclass(frozen=True)
class SummaryPolicy:
    section_min_tokens: int = 2000
    section_min_children: int = 8
    section_min_topics: int = 3
    topic_min_tokens: int = 800
    topic_min_dimensions: int = 2


def _ordered_sources(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    return sorted(chunks, key=lambda chunk: (chunk.reading_order, chunk.id))


def _dimensions(chunks: list[ChunkRecord]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for label, pattern in _DIMENSIONS.items():
        ids = [chunk.id for chunk in chunks if pattern.search(chunk.raw_text)]
        if ids:
            matches[label] = ids
    return matches


def _make_summary(
    result: ParseResult,
    section: Section,
    sources: list[ChunkRecord],
    *,
    drug_id: str,
    product_id: str,
    chunker_version: str,
    summary_type: str,
    topic_name: str | None,
    key_point_sources: dict[str, list[str]],
) -> ChunkRecord:
    sources = _ordered_sources(sources)
    source_ids = [chunk.id for chunk in sources]
    points = list(key_point_sources)
    heading = f"{result.drug_name.normalized}；{section.normalized_name}"
    if topic_name:
        heading += f"；{topic_name}"
    content = heading + "。检索导航：" + "；".join(points) + "。"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    summary_id = stable_id(
        result.document_id, "summary", chunker_version, summary_type,
        section.id, topic_name or "", digest,
    )
    refs: list[SourceRef] = []
    seen_refs: set[tuple] = set()
    for source in sources:
        for ref in source.refs:
            key = (ref.page, ref.section_id, ref.bbox)
            if key not in seen_refs:
                seen_refs.add(key)
                refs.append(ref)
    pages = [ref.page for ref in refs]
    # A generated summary has evidence areas, never its own PDF bbox.
    bbox = None
    source_element_ids = list(dict.fromkeys(
        element_id for source in sources for element_id in source.source_element_ids
    ))
    path = sources[0].section_path if topic_name else [section.normalized_name]
    metadata = {
        **sources[0].metadata,
        "document_id": result.document_id,
        "drug_id": drug_id,
        "product_id": product_id,
        "section": section.normalized_name,
        "section_path": path.copy(),
        "topic_id": sources[0].topic_id if topic_name else None,
        "topic_name": topic_name,
        "parent_id": None,
        "parent_type": None,
        "content_type": summary_type,
        "content_hash": digest,
        "raw_text": content,
        "source_element_ids": source_element_ids,
        "summary_type": summary_type,
        "source_chunk_ids": source_ids,
        "key_points": points,
        "key_point_sources": key_point_sources,
        "evidence_role": "navigation",
        "index_channel": "summary",
        "summary_weight": 0.25,
        "split_reason": "summary_routing",
        "lead_in_id": None,
        "overlap_source_ids": [],
        "previous_chunk_id": None,
        "next_chunk_id": None,
    }
    return ChunkRecord(
        id=summary_id,
        document_id=result.document_id,
        section_id=section.id,
        parent_id=None,
        content=content,
        retrieval_text=content,
        content_type=summary_type,
        is_parent=False,
        token_count=estimated_token_count(content),
        page_start=min(pages),
        page_end=max(pages),
        bbox=bbox,
        reading_order=sources[0].reading_order,
        refs=refs,
        metadata=metadata,
        content_hash=digest,
        chunker_version=chunker_version,
        section_path=path.copy(),
        raw_section_name=section.raw_name,
        normalized_section_name=section.normalized_name,
        topic_id=sources[0].topic_id if topic_name else None,
        topic_name=topic_name,
        parent_type=None,
        split_reason="summary_routing",
        source_element_ids=source_element_ids,
    )

"""
生成摘要索引
"""
def build_summary_chunks(
    result: ParseResult,
    evidence_children: list[ChunkRecord],
    drug_id: str,
    product_id: str,
    chunker_version: str,
    policy: SummaryPolicy | None = None,
) -> list[ChunkRecord]:
    """Create summaries only when section size or topic complexity warrants one."""
    selected = policy or SummaryPolicy()
    by_section: dict[str, list[ChunkRecord]] = defaultdict(list)
    for child in evidence_children:
        if child.content_type in {"paragraph", "list_item", "table_row"}:
            by_section[child.section_id].append(child)

    summaries: list[ChunkRecord] = []
    for section in result.sections:
        children = _ordered_sources(by_section[section.id])
        if not children:
            continue
        by_topic: dict[str, list[ChunkRecord]] = defaultdict(list)
        for child in children:
            by_topic[child.topic_name or section.normalized_name].append(child)
        distinct_topics = {name for name in by_topic if name != section.normalized_name}
        section_tokens = sum(child.token_count for child in children)
        needs_section = (
            section_tokens >= selected.section_min_tokens
            or len(children) >= selected.section_min_children
            or len(distinct_topics) >= selected.section_min_topics
        )
        if needs_section:
            points = {
                f"主题：{name}": [child.id for child in topic_children]
                for name, topic_children in by_topic.items()
                if name != section.normalized_name
            }
            if not points:
                points = {
                    f"内容维度：{name}": ids
                    for name, ids in _dimensions(children).items()
                }
            if not points:
                points = {"本章节原始段落与表格行": [child.id for child in children]}
            summaries.append(_make_summary(
                result, section, children, drug_id=drug_id, product_id=product_id,
                chunker_version=chunker_version, summary_type="section_summary",
                topic_name=None, key_point_sources=points,
            ))

        for topic_name, topic_children in by_topic.items():
            if topic_name == section.normalized_name:
                continue
            dimensions = _dimensions(topic_children)
            if (sum(child.token_count for child in topic_children) < selected.topic_min_tokens
                    or len(dimensions) < selected.topic_min_dimensions):
                continue
            points = {f"内容维度：{name}": ids for name, ids in dimensions.items()}
            summaries.append(_make_summary(
                result, section, topic_children, drug_id=drug_id,
                product_id=product_id, chunker_version=chunker_version,
                summary_type="topic_summary", topic_name=topic_name,
                key_point_sources=points,
            ))
    return summaries
