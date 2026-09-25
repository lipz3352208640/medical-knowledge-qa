"""Conditional context pruning and exact-source extraction (chapter 12)."""

from dataclasses import dataclass, field
import math

from app.chunker.tokens import estimated_token_count
from app.generator.ollama import ChatError, ChatModel
from app.indexer.embeddings import EmbeddingError
from app.retrieval.pipeline import DrugEvidence
from app.retrieval.query import ParsedQuery
from app.storage.retrieval import StoredChunk


class ContextBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    prompt_token_budget: int = 5000
    direct_query_tokens: int = 32
    direct_evidence_tokens: int = 1000
    embedding_context_count: int = 8
    embedding_context_tokens: int = 1600
    embedding_keep: int = 5
    llm_filter_total_candidates: int = 9
    extract_chunk_tokens: int = 700


@dataclass(slots=True)
class EvidenceUnit:
    chunk: StoredChunk
    text: str
    drug_name: str
    primary: bool
    owner_id: str
    required: bool = False
    relevance: float = 0.0

    @property
    def token_count(self) -> int:
        context = (
            estimated_token_count(self.chunk.retrieval_text)
            if self.chunk.content_type == "table_row" else 0
        )
        return estimated_token_count(self.text) + context + 48


@dataclass(slots=True)
class PreparedEvidence:
    units: list[EvidenceUnit]
    strategies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return sum(unit.token_count for unit in self.units)


_FILTER_SCHEMA = {
    "type": "object",
    "properties": {"keep_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["keep_ids"],
}
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"extracts": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["chunk_id", "quote"],
        },
    }},
    "required": ["extracts"],
}


class ContextCompressor:
    def __init__(self, embedder, chat: ChatModel, config: CompressionConfig | None = None):
        self.embedder = embedder
        self.chat = chat
        self.config = config or CompressionConfig()
        if any(value <= 0 for value in (
            self.config.prompt_token_budget, self.config.direct_query_tokens,
            self.config.direct_evidence_tokens, self.config.embedding_context_count,
            self.config.embedding_context_tokens, self.config.embedding_keep,
            self.config.llm_filter_total_candidates, self.config.extract_chunk_tokens,
        )):
            raise ValueError("Compression thresholds must be positive")

    def prepare(
        self,
        query: str,
        parsed: ParsedQuery,
        group: DrugEvidence,
        *,
        total_candidates: int,
    ) -> PreparedEvidence:
        if not group.evidence:
            return PreparedEvidence([])
        direct = self._can_generate_directly(query, parsed, group)
        selected = group.evidence[:3] if direct else group.evidence
        prepared = PreparedEvidence(self._collect_units(selected, group.drug or ""))
        if direct:
            prepared.strategies.append("direct_top3")
            self._fit_budget(prepared)
            return prepared
        contexts = [unit for unit in prepared.units if not unit.primary]
        #未走 direct_top3，并且可选上下文超过 8 条，或超过 1600 token
        if (any(not unit.required for unit in contexts)
            and (len(contexts) > self.config.embedding_context_count or
                 sum(unit.token_count for unit in contexts) > self.config.embedding_context_tokens)):
            try:
                self._embedding_filter(query, prepared)
            except EmbeddingError:
                prepared.warnings.append("向量裁剪不可用，保留展开的原文上下文")
            prepared.strategies.append("embedding_filter")
        #候选数大于 1，同时是多药比较、多条件或多实体问题，且总候选数 ≥ 9
        if (len(group.evidence) > 1
            and (parsed.comparison or len(parsed.conditions) > 1 or len(parsed.entities) > 1)
            and total_candidates >= self.config.llm_filter_total_candidates):
            try:
                self._llm_filter(query, prepared)
            except ChatError:
                prepared.warnings.append("LLM候选过滤不可用，保留原始候选")
            prepared.strategies.append("llm_filter")
        #非概览问题，并且至少有一个证据 chunk 超过 700 token
        if not parsed.broad and any(
            estimated_token_count(unit.text) > self.config.extract_chunk_tokens
            for unit in prepared.units
        ):
            try:
                self._extract(query, prepared)
            except ChatError:
                prepared.warnings.append("LLM原文摘取不可用，保留原文")
            prepared.strategies.append("llm_extractor")
        self._fit_budget(prepared)
        return prepared

    """
    direct_top3策略
      1.除了 多药品、多章节、多人群、多条件、多实体时、比较、概览等情况
      2.校验前三条证据是否覆盖问题要求的章节，覆盖走direct_top3
      2.在前三条证据覆盖问题要求的章节时， 问题 token ≤ 32，且前 3 条证据 ≤ 1000 token

    """
    def _can_generate_directly(
        self, query: str, parsed: ParsedQuery, group: DrugEvidence
    ) -> bool:
        if (
            parsed.comparison
            or parsed.broad
            or len(parsed.drug_ids) > 1
            or len(parsed.sections) > 1
            or len(parsed.populations) > 1
            or len(parsed.conditions) > 1
            or len(parsed.entities) > 1
            or estimated_token_count(query) > self.config.direct_query_tokens
        ):
            return False
        top = self._collect_units(group.evidence[:3], group.drug or "")
        if not self._covers_requested_sections(parsed, top):
            return False
        return sum(unit.token_count for unit in top) <= min(
            self.config.direct_evidence_tokens, self.config.prompt_token_budget
        )

    @staticmethod
    def _covers_requested_sections(
        parsed: ParsedQuery, units: list[EvidenceUnit]
    ) -> bool:
        if not parsed.sections:
            return True
        available: set[str] = set()
        for unit in units:
            section = unit.chunk.metadata.get("section")
            if isinstance(section, str) and section:
                available.add(section)
            path = unit.chunk.metadata.get("section_path")
            if isinstance(path, (list, tuple)):
                available.update(
                    item for item in path if isinstance(item, str) and item
                )
        return set(parsed.sections).issubset(available)

    @staticmethod
    def _collect_units(candidates: list, drug_name: str) -> list[EvidenceUnit]:
        primary_ids = {item.chunk.id for item in candidates}
        seen: set[str] = set()
        units: list[EvidenceUnit] = []
        for item in candidates:
            chunk = item.chunk
            if chunk.id not in seen:
                units.append(EvidenceUnit(
                    chunk, chunk.content, drug_name or chunk.metadata.get("generic_name") or "",
                    True, chunk.id,
                ))
                seen.add(chunk.id)
            for context in item.context_chunks:
                if context.id in seen or context.id in primary_ids:
                    continue
                required = (
                    context.id == chunk.metadata.get("lead_in_id")
                    or (chunk.content_type == "table_row" and context.id == chunk.parent_id)
                )
                units.append(EvidenceUnit(
                    context, context.content,
                    drug_name or context.metadata.get("generic_name") or "",
                    False, chunk.id, required,
                ))
                seen.add(context.id)
        return units

    def _embedding_filter(self, query: str, prepared: PreparedEvidence) -> None:
        contexts = [unit for unit in prepared.units if not unit.primary and not unit.required]
        if not contexts:
            return
        vectors = self.embedder.embed([query] + [unit.text for unit in contexts])
        if len(vectors) != len(contexts) + 1:
            raise ValueError("Embedder returned an unexpected vector count")
        for unit, vector in zip(contexts, vectors[1:], strict=True):
            unit.relevance = _cosine(vectors[0], vector)
        keep = {
            unit.chunk.id
            for unit in sorted(contexts, key=lambda unit: (-unit.relevance, unit.chunk.id))
            [:self.config.embedding_keep]
        }
        prepared.units = [
            unit for unit in prepared.units
            if unit.primary or unit.required or unit.chunk.id in keep
        ]

    def _llm_filter(self, query: str, prepared: PreparedEvidence) -> None:
        primaries = [unit for unit in prepared.units if unit.primary]
        if len(primaries) <= 1:
            return
        candidates = "\n".join(
            f"[{unit.chunk.id}] {unit.drug_name}｜{unit.chunk.metadata.get('section', '')}｜{unit.text[:360]}"
            for unit in primaries
        )
        response = self.chat.complete(
            "你只筛选与问题直接相关的说明书原文候选。只返回给定的 chunk_id；不推断医学事实。",
            f"问题：{query}\n候选：\n{candidates}\n返回 keep_ids 数组。",
            _FILTER_SCHEMA,
            max_output_tokens=256,
        )
        raw_ids = response.get("keep_ids")
        allowed = {unit.chunk.id for unit in primaries}
        if not isinstance(raw_ids, list) or any(not isinstance(value, str) for value in raw_ids):
            prepared.warnings.append("LLM候选过滤结果无效，保留原始候选")
            return
        keep = set(raw_ids) & allowed
        if not keep:
            prepared.warnings.append("LLM候选过滤未保留证据，保留原始候选")
            return
        keep.add(primaries[0].chunk.id)
        prepared.units = [
            unit for unit in prepared.units if (
                unit.chunk.id in keep if unit.primary else unit.owner_id in keep
            )
        ]

    def _extract(self, query: str, prepared: PreparedEvidence) -> None:
        long_units = [
            unit for unit in prepared.units
            if estimated_token_count(unit.text) > self.config.extract_chunk_tokens
        ]
        for unit in long_units:
            response = self.chat.complete(
                "从给定说明书原文中复制与问题相关的连续句段。可返回多条，但每条 quote 必须分别逐字复制，不能改写、拼接或补充。",
                f"问题：{query}\nchunk_id：{unit.chunk.id}\n原文：\n{unit.chunk.content}\n"
                "返回 extracts，包含 chunk_id 和 quote。",
                _EXTRACT_SCHEMA,
                max_output_tokens=512,
            )
            extracts = response.get("extracts")
            quotes: list[str] = []
            if isinstance(extracts, list):
                for extract in extracts:
                    if isinstance(extract, dict) and extract.get("chunk_id") == unit.chunk.id:
                        quote = extract.get("quote")
                        if isinstance(quote, str) and quote.strip() and quote in unit.chunk.content:
                            quotes.append(quote)
            quotes = list(dict.fromkeys(quotes))[:3]
            excerpt = "\n[省略]\n".join(quotes)
            if quotes and estimated_token_count(excerpt) < estimated_token_count(unit.text):
                unit.text = excerpt
            else:
                prepared.warnings.append(f"原文摘取未通过校验：{unit.chunk.id}")
    """
    budget_pruning策略：
      1. 所有处理后上下文总量仍不超过 5000 token，直接返回
      2. 在压缩经过重排序的原始证据之前，先丢弃可选的上下
      3. 将主证据及其必要的上下文共同构成一个证据包
      4. 最高排名证据包超过预算时直接抛出异常
    """
    def _fit_budget(self, prepared: PreparedEvidence) -> None:
        budget = self.config.prompt_token_budget
        if prepared.token_count <= budget:
            return
        # Drop optional context before reducing the reranked original evidence.
        #在压缩经过重排序的原始证据之前，先丢弃可选的上下
        for unit in sorted(
            [unit for unit in prepared.units if not unit.primary and not unit.required],
            key=lambda unit: (unit.relevance, -unit.token_count),
        ):
            if prepared.token_count <= budget:
                break
            prepared.units.remove(unit)
        primaries = [unit for unit in prepared.units if unit.primary]
        #一个主要条目及其必要的上下文共同构成一个证据包。
        #如果只移除表格的表头或引导语，可能会改变剩余医疗证据的含义。
        for unit in reversed(primaries[1:]):
            if prepared.token_count <= budget:
                break
            # A primary and its required context form one evidence bundle.
            # Removing only a table header or lead-in can change the meaning
            # of the remaining medical evidence.
            prepared.units = [
                part for part in prepared.units
                if part.owner_id != unit.chunk.id
            ]
        if prepared.token_count > budget:
            raise ContextBudgetError(
                "The highest-ranked evidence bundle exceeds the evidence budget"
            )
        prepared.strategies.append("budget_pruning")
        prepared.warnings.append("部分低排序证据或上下文因预算限制未进入回答")


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors have incompatible dimensions")
    product = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return product / (norm_left * norm_right) if norm_left and norm_right else 0.0
