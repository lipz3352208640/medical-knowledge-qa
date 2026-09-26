"""Chapter 11: scoped lexical/vector recall, RRF, reranking and evidence."""

from dataclasses import dataclass, field, replace
import math

from app.chunker.tokens import estimated_token_count
from app.generator.ollama import ChatError
from app.retrieval.query import ParsedQuery, drug_alias_map, parse_query
from app.retrieval.reranker import Reranker
from app.retrieval.rewrite import QueryRewrite, QueryRewriteError
from app.storage.retrieval import (
    ChunkFilters, StoredChunk, bm25_search, list_active_document_ids,
    list_active_drug_aliases, list_ready_vector_document_ids,
    load_chunks_by_ids, load_parent_chunks,
    load_summary_sources, summary_search,
)
from app.storage.vector_store import QdrantVectorStore


@dataclass(slots=True)
class RankedCandidate:
    chunk: StoredChunk
    bm25_rank: int | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    rrf_score: float = 0.0
    summary_route_score: float = 0.0
    rerank_score: float | None = None
    context_chunks: list[StoredChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "section_id": self.chunk.section_id,
            "section": self.chunk.metadata.get("section"),
            "section_path": self.chunk.metadata.get("section_path", []),
            "topic_name": self.chunk.metadata.get("topic_name"),
            "content_type": self.chunk.content_type,
            "content": self.chunk.content,
            "retrieval_text": self.chunk.retrieval_text,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "refs": self.chunk.refs,
            "source_element_ids": self.chunk.metadata.get("source_element_ids", []),
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "vector_rank": self.vector_rank,
            "vector_score": self.vector_score,
            "rrf_score": self.rrf_score,
            "summary_route_score": self.summary_route_score,
            "rerank_score": self.rerank_score,
            "context": [
                {"chunk_id": chunk.id, "content": chunk.content,
                 "retrieval_text": chunk.retrieval_text,
                 "content_type": chunk.content_type, "refs": chunk.refs}
                for chunk in self.context_chunks
            ],
        }


@dataclass(slots=True)
class DrugEvidence:
    drug: str | None
    drug_id: str | None
    evidence: list[RankedCandidate]

    def to_dict(self) -> dict:
        return {
            "drug": self.drug,
            "drug_id": self.drug_id,
            "missing_evidence": not self.evidence,
            "evidence_status": "当前说明书未检索到相关表述" if not self.evidence else "已检索到原文证据",
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class ComparisonDimension:
    dimension: str
    groups: list[DrugEvidence]

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "groups": [group.to_dict() for group in self.groups]}


@dataclass(slots=True)
class RetrievalResult:
    query: str
    parsed: ParsedQuery
    groups: list[DrugEvidence]
    comparison_dimensions: list[ComparisonDimension] = field(default_factory=list)
    query_rewrite: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "query": self.query,
            "parsed": self.parsed.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
        }
        if self.parsed.comparison:
            result["comparison_dimensions"] = [dimension.to_dict() for dimension in self.comparison_dimensions]
        if self.query_rewrite is not None:
            result["query_rewrite"] = self.query_rewrite
        return result

#计算语义检索+bm25检索的混合检索后的RRF
def reciprocal_rank_fusion(
    bm25_hits: list,
    vector_hits: list,
    *,
    k: int = 60,
) -> list[RankedCandidate]:
    """Fuse only recall ranks; keep each channel's rank and raw score."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    by_id: dict[str, RankedCandidate] = {}
    for rank, hit in enumerate(bm25_hits, start=1):
        candidate = by_id.setdefault(hit.chunk.id, RankedCandidate(hit.chunk))
        if candidate.bm25_rank is None:
            candidate.bm25_rank = rank
            candidate.bm25_score = hit.score
            candidate.rrf_score += 1 / (k + rank)
    for rank, hit in enumerate(vector_hits, start=1):
        candidate = by_id.setdefault(hit.chunk.id, RankedCandidate(hit.chunk))
        if candidate.vector_rank is None:
            candidate.vector_rank = rank
            candidate.vector_score = hit.score
            candidate.rrf_score += 1 / (k + rank)
    return sorted(by_id.values(), key=lambda item: (-item.rrf_score, item.chunk.id))


def _reciprocal_rank_fusion_queries(
    query_hits: list[tuple[list, list]], *, k: int
) -> list[RankedCandidate]:
    """Fuse original and rewritten recalls while keeping the original dominant."""
    by_id: dict[str, RankedCandidate] = {}
    for query_index, (bm25_hits, vector_hits) in enumerate(query_hits):
        weight = 1.0 if query_index == 0 else 0.75
        for channel, hits in (("bm25", bm25_hits), ("vector", vector_hits)):
            emitted: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                if hit.chunk.id in emitted:
                    continue
                emitted.add(hit.chunk.id)
                candidate = by_id.setdefault(hit.chunk.id, RankedCandidate(hit.chunk))
                candidate.rrf_score += weight / (k + rank)
                if channel == "bm25" and (
                    candidate.bm25_rank is None or rank < candidate.bm25_rank
                ):
                    candidate.bm25_rank = rank
                    candidate.bm25_score = hit.score
                if channel == "vector" and (
                    candidate.vector_rank is None or rank < candidate.vector_rank
                ):
                    candidate.vector_rank = rank
                    candidate.vector_score = hit.score
    return sorted(by_id.values(), key=lambda item: (-item.rrf_score, item.chunk.id))


class RetrievalEngine:
    def __init__(
        self,
        mysql_url: str,
        embedder,
        vector_store: QdrantVectorStore,
        reranker: Reranker,
        *,
        recall_limit: int = 20,
        evidence_limit: int = 6,
        rrf_k: int = 60,
        context_token_budget: int = 1200,
        query_rewriter=None,
        rewrite_min_candidates: int = 3,
    ):
        if recall_limit <= 0 or not 5 <= evidence_limit <= 8 or rrf_k <= 0:
            raise ValueError("recall_limit > 0, evidence_limit 5..8 and rrf_k > 0 required")
        if context_token_budget <= 0 or rewrite_min_candidates <= 0:
            raise ValueError("context and rewrite thresholds must be positive")
        self.mysql_url = mysql_url
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.recall_limit = recall_limit
        self.evidence_limit = evidence_limit
        self.rrf_k = rrf_k
        self.context_token_budget = context_token_budget
        self.query_rewriter = query_rewriter
        self.rewrite_min_candidates = rewrite_min_candidates

    def retrieve(self, query: str, *, drug_names: tuple[str, ...] = ()) -> RetrievalResult:
        aliases = list_active_drug_aliases(self.mysql_url)
        parsed = _apply_drug_filter(parse_query(query, aliases), query, drug_names, aliases)
        active_ids = tuple(list_active_document_ids(self.mysql_url))
        scopes = list(zip(parsed.drugs, parsed.drug_ids, strict=True)) or [(None, None)]
        if not active_ids:
            groups = [DrugEvidence(name, drug_id, []) for name, drug_id in scopes]
            dimensions = [ComparisonDimension(section, groups) for section in (parsed.sections or ("综合",))] if parsed.comparison else []
            return RetrievalResult(query, parsed, groups, dimensions)
        ready_ids = tuple(list_ready_vector_document_ids(
            self.mysql_url,
            collection_name=self.vector_store.collection_name,
            embedding_model=self.embedder.model,
        ))
        rewrite: QueryRewrite | None = None
        rewrite_trace: dict | None = None
        if (
            self.query_rewriter is not None
            and self.query_rewriter.should_rewrite_before_recall(query, parsed)
        ):
            rewrite, rewrite_trace = self._try_rewrite(
                query, parsed, aliases, trigger="unresolved_colloquial_intent"
            )
        result = self._retrieve_parsed(
            query, parsed, scopes, active_ids, ready_ids, rewrite
        )
        if (
            rewrite is None
            and self.query_rewriter is not None
            and self._needs_rewrite_after_recall(result)
        ):
            rewrite, rewrite_trace = self._try_rewrite(
                query, parsed, aliases, trigger="insufficient_initial_recall"
            )
            if rewrite is not None:
                result = self._retrieve_parsed(
                    query, parsed, scopes, active_ids, ready_ids, rewrite
                )
        result.query_rewrite = rewrite_trace
        return result

    def _retrieve_parsed(
        self,
        query: str,
        parsed: ParsedQuery,
        scopes: list[tuple[str | None, str | None]],
        active_ids: tuple[str, ...],
        ready_ids: tuple[str, ...],
        rewrite: QueryRewrite | None,
    ) -> RetrievalResult:
        if parsed.comparison:
            dimensions: list[ComparisonDimension] = []
            combined = {drug_id: DrugEvidence(name, drug_id, []) for name, drug_id in scopes}
            for section in parsed.sections or ("综合",):
                section_filter = (section,) if parsed.sections else ()
                groups = [
                    self._retrieve_scope(
                        query, parsed, name, drug_id, active_ids, ready_ids,
                        section_filter, rewrite,
                    )
                    for name, drug_id in scopes
                ]
                dimensions.append(ComparisonDimension(section, groups))
                for group in groups:
                    combined[group.drug_id].evidence.extend(group.evidence)
            return RetrievalResult(query, parsed, list(combined.values()), dimensions)
        groups = [
            self._retrieve_scope(
                query, parsed, name, drug_id, active_ids, ready_ids,
                parsed.sections, rewrite,
            )
            for name, drug_id in scopes
        ]
        return RetrievalResult(query, parsed, groups)

    def _try_rewrite(
        self,
        query: str,
        parsed: ParsedQuery,
        aliases: list[tuple[str, str]],
        *,
        trigger: str,
    ) -> tuple[QueryRewrite | None, dict]:
        try:
            rewrite = self.query_rewriter.rewrite(query, parsed, aliases)
        except (ChatError, QueryRewriteError, ValueError) as exc:
            return None, {
                "trigger": trigger,
                "applied": False,
                "warning": str(exc),
            }
        return rewrite, {
            "trigger": trigger,
            "applied": True,
            **rewrite.to_dict(),
        }

    def _needs_rewrite_after_recall(self, result: RetrievalResult) -> bool:
        groups = (
            [group for dimension in result.comparison_dimensions for group in dimension.groups]
            if result.parsed.comparison else result.groups
        )
        if not groups:
            return True
        counts = [len(group.evidence) for group in groups]
        if not any(counts):
            return True
        return (
            not result.parsed.sections
            and not result.parsed.requires_exact_value
            and min(counts) < self.rewrite_min_candidates
        )

    def _retrieve_scope(
        self,
        query: str,
        parsed: ParsedQuery,
        drug: str | None,
        drug_id: str | None,
        active_ids: tuple[str, ...],
        ready_ids: tuple[str, ...],
        sections: tuple[str, ...],
        rewrite: QueryRewrite | None,
    ) -> DrugEvidence:
        scope_query = _single_drug_query(query, parsed, drug)
        summary_candidates: list[RankedCandidate] = []
        summary_first = self._should_use_summary_first(parsed, drug_id, sections)
        vector = None
        filters = self._scope_filters(drug_id, active_ids, sections)
        if summary_first:
            vector = self._query_vector(scope_query, ready_ids)
            complete_section_summary = self._route_summaries(
                scope_query, vector, filters, ready_ids, summary_candidates
            )
            if complete_section_summary and summary_candidates:
                return self._rerank_scope(
                    drug, drug_id, scope_query, summary_candidates, parsed
                )

        lexical, dense, vector, filters = self._recall_scope(
            scope_query, drug_id, active_ids, ready_ids, sections,
            prepared_vector=vector,
        )
        query_hits = [(lexical, dense)]
        rerank_query = scope_query
        if rewrite is not None:
            rewrite_sections = sections or rewrite.sections
            for rewritten_query in rewrite.retrieval_queries:
                scoped_rewrite = _single_drug_query(rewritten_query, parsed, drug)
                rewritten_lexical, rewritten_dense, _, _ = self._recall_scope(
                    scoped_rewrite,
                    drug_id,
                    active_ids,
                    ready_ids,
                    rewrite_sections,
                )
                query_hits.append((rewritten_lexical, rewritten_dense))
            rerank_query = f"{scope_query}；检索规范化：{rewrite.normalized_query}"
        candidates = _reciprocal_rank_fusion_queries(query_hits, k=self.rrf_k)
        if summary_candidates:
            self._merge_summary_candidates(candidates, summary_candidates)
        elif parsed.broad:
            self._route_summaries(scope_query, vector, filters, ready_ids, candidates)
        return self._rerank_scope(drug, drug_id, rerank_query, candidates, parsed)

    @staticmethod
    def _should_use_summary_first(
        parsed: ParsedQuery,
        drug_id: str | None,
        sections: tuple[str, ...],
    ) -> bool:
        """Use a complete section summary as the primary route for scoped overviews."""
        return (
            parsed.intent == "overview"
            and parsed.broad
            and not parsed.comparison
            and not parsed.requires_exact_value
            and drug_id is not None
            and len(parsed.drug_ids) == 1
            and len(sections) == 1
        )

    @staticmethod
    def _scope_filters(
        drug_id: str | None,
        active_ids: tuple[str, ...],
        sections: tuple[str, ...],
    ) -> ChunkFilters:
        return ChunkFilters(
            document_ids=active_ids,
            drug_ids=(drug_id,) if drug_id else (),
            sections=sections,
            # Population labels are sparse on short dose/table rows. Keep them
            # in the query for ranking, but never hard-exclude unlabelled facts.
            populations=(),
        )

    def _query_vector(
        self, query: str, ready_ids: tuple[str, ...]
    ) -> list[float] | None:
        if not ready_ids:
            return None
        vectors = self.embedder.embed([query])
        if len(vectors) != 1:
            raise ValueError("Embedder must return one vector for one query")
        return vectors[0]

    def _rerank_scope(
        self,
        drug: str | None,
        drug_id: str | None,
        rerank_query: str,
        candidates: list[RankedCandidate],
        parsed: ParsedQuery,
    ) -> DrugEvidence:
        if not candidates:
            return DrugEvidence(drug, drug_id, [])
        candidates.sort(
            key=lambda item: (-(item.rrf_score + item.summary_route_score), item.chunk.id)
        )
        scores = self.reranker.score(rerank_query, [item.chunk for item in candidates])
        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker returned invalid scores")
        for item, score in zip(candidates, scores, strict=True):
            item.rerank_score = score
        candidates.sort(key=lambda item: (
            -item.rerank_score,
            -(item.rrf_score + item.summary_route_score),
            item.chunk.id,
        ))
        selected = candidates[:self.evidence_limit]
        self._expand_context(selected, parsed)
        return DrugEvidence(drug, drug_id, selected)

    def _recall_scope(
        self,
        query: str,
        drug_id: str | None,
        active_ids: tuple[str, ...],
        ready_ids: tuple[str, ...],
        sections: tuple[str, ...],
        *,
        prepared_vector: list[float] | None = None,
    ) -> tuple[list, list, list[float] | None, ChunkFilters]:
        filters = self._scope_filters(drug_id, active_ids, sections)
        lexical = bm25_search(self.mysql_url, query, filters, limit=self.recall_limit)
        vector = prepared_vector
        dense = []
        if ready_ids:
            if vector is None:
                vector = self._query_vector(query, ready_ids)
            vector_filters = ChunkFilters(
                document_ids=ready_ids,
                drug_ids=filters.drug_ids,
                sections=filters.sections,
                populations=filters.populations,
            )
            dense = self.vector_store.search(vector, vector_filters, limit=self.recall_limit)
        return lexical, dense, vector, filters

    def _route_summaries(
        self,
        query: str,
        vector: list[float] | None,
        filters: ChunkFilters,
        ready_ids: tuple[str, ...],
        candidates: list[RankedCandidate],
    ) -> bool:
        lexical = summary_search(self.mysql_url, query, filters, limit=5)
        dense = []
        if vector is not None:
            vector_filters = ChunkFilters(
                document_ids=ready_ids,
                drug_ids=filters.drug_ids,
                sections=filters.sections,
                populations=filters.populations,
            )
            dense = self.vector_store.search(vector, vector_filters, limit=5, channel="summary")
        routed = reciprocal_rank_fusion(lexical, dense, k=self.rrf_k)
        by_id = {item.chunk.id: item for item in candidates}
        complete_section_summary = False
        for rank, summary in enumerate(routed[:5], start=1):
            try:
                sources = load_summary_sources(self.mysql_url, summary.chunk.id)
            except ValueError:
                # A stale or incomplete summary must not suppress primary recall.
                continue
            if summary.chunk.content_type == "section_summary":
                complete_section_summary = True
            key_points = summary.chunk.metadata.get("key_point_sources") or {}
            representative_ids = {
                source_id for source_ids in key_points.values() for source_id in source_ids
            }
            representatives = [source for source in sources if source.id in representative_ids]
            for source in (representatives or sources)[:8]:
                candidate = by_id.get(source.id)
                if candidate is None:
                    candidate = RankedCandidate(source)
                    by_id[source.id] = candidate
                    candidates.append(candidate)
                candidate.summary_route_score += 0.25 / (self.rrf_k + rank)
        return complete_section_summary

    @staticmethod
    def _merge_summary_candidates(
        candidates: list[RankedCandidate],
        summary_candidates: list[RankedCandidate],
    ) -> None:
        by_id = {item.chunk.id: item for item in candidates}
        for summary_candidate in summary_candidates:
            existing = by_id.get(summary_candidate.chunk.id)
            if existing is None:
                candidates.append(summary_candidate)
                by_id[summary_candidate.chunk.id] = summary_candidate
            else:
                existing.summary_route_score += summary_candidate.summary_route_score

    def _expand_context(self, selected: list[RankedCandidate], parsed: ParsedQuery) -> None:
        needed: list[str] = []
        parents: list[str] = []
        for item in selected:
            metadata = item.chunk.metadata
            lead = metadata.get("lead_in_id")
            if lead:
                needed.append(lead)
            if not parsed.requires_exact_value and not parsed.broad:
                needed.extend(
                    value for value in (
                        metadata.get("previous_chunk_id"), metadata.get("next_chunk_id")
                    ) if value
                )
            if item.chunk.parent_id and (parsed.broad or item.chunk.content_type == "table_row"):
                parents.append(item.chunk.parent_id)
        found = {chunk.id: chunk for chunk in load_chunks_by_ids(self.mysql_url, needed)}
        parent_map = {chunk.id: chunk for chunk in load_parent_chunks(self.mysql_url, parents)}
        selected_ids = {item.chunk.id for item in selected}
        emitted_context_ids: set[str] = set()
        for item in selected:
            chunk = item.chunk
            metadata = chunk.metadata
            local_ids = [
                metadata.get("previous_chunk_id"), metadata.get("lead_in_id"),
                metadata.get("next_chunk_id"),
            ]
            context: list[StoredChunk] = []
            budget = self.context_token_budget
            for chunk_id in local_ids:
                if not chunk_id or chunk_id not in found or chunk_id in selected_ids:
                    continue
                neighbor = found[chunk_id]
                if neighbor.document_id != chunk.document_id or neighbor.section_id != chunk.section_id:
                    continue
                if chunk_id != metadata.get("lead_in_id") and (
                    parsed.requires_exact_value or parsed.broad or neighbor.parent_id != chunk.parent_id
                ):
                    continue
                cost = estimated_token_count(neighbor.content)
                if cost <= budget and neighbor.id not in emitted_context_ids:
                    context.append(neighbor)
                    emitted_context_ids.add(neighbor.id)
                    budget -= cost
            parent = parent_map.get(chunk.parent_id or "")
            if parent and parent.document_id == chunk.document_id and parent.id not in emitted_context_ids:
                cost = estimated_token_count(parent.content)
                if cost <= budget:
                    context.append(parent)
                    emitted_context_ids.add(parent.id)
            item.context_chunks = context


def _single_drug_query(query: str, parsed: ParsedQuery, current: str | None) -> str:
    if not parsed.drugs or current is None:
        return query
    reduced = query
    for name in parsed.drugs:
        if name != current:
            reduced = reduced.replace(name, "")
    return reduced.strip() or current


def _apply_drug_filter(
    parsed: ParsedQuery,
    query: str,
    drug_names: tuple[str, ...],
    aliases: list[tuple[str, str]],
) -> ParsedQuery:
    """Resolve explicit API filters to canonical drug IDs without changing the question."""
    if not drug_names:
        return parsed
    alias_ids = drug_alias_map(aliases)
    selected: dict[str, str] = {}
    for name in drug_names:
        matches = alias_ids.get(name.strip().casefold(), set())
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous drug filter: {name}")
        selected[next(iter(matches))] = name.strip()
    if not set(parsed.drug_ids).issubset(selected):
        raise ValueError("Drug filter conflicts with a drug named in the question")
    names_by_id = dict(zip(parsed.drug_ids, parsed.drugs, strict=True))
    names = tuple(names_by_id.get(drug_id, name) for drug_id, name in selected.items())
    ids = tuple(selected)
    comparison = len(ids) > 1 and any(
        term in query for term in ("比较", "对比", "区别", "差异", "相比", "分别", "哪个", "与", "和", "及")
    )
    return replace(
        parsed, drugs=names, drug_ids=ids, comparison=comparison,
        intent="comparison" if comparison and not parsed.requires_exact_value else parsed.intent,
    )
