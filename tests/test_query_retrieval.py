import unittest
from unittest.mock import MagicMock, patch

from app.retrieval.pipeline import RetrievalEngine, _apply_drug_filter, reciprocal_rank_fusion
from app.retrieval.query import parse_query
from app.retrieval.reranker import candidate_text
from app.retrieval.rewrite import QueryRewrite
from app.storage.retrieval import ChunkFilters, LexicalHit, StoredChunk, bm25_search
from app.storage.vector_store import VectorHit


def _chunk(chunk_id, drug_id, *, kind="paragraph", parent_id="parent-1", metadata=None):
    return StoredChunk(
        chunk_id, f"doc-{drug_id}", "section-1", parent_id,
        f"原文 {chunk_id}", f"检索 {chunk_id}", kind,
        2, 2, [1, 2, 3, 4], [{"document_id": f"doc-{drug_id}", "page": 2}],
        {"drug_id": drug_id, "generic_name": drug_id, "section": "不良反应", **(metadata or {})},
    )


class _Embedder:
    model = "test-embedding"

    def embed(self, texts):
        return [[0.5, 0.5] for _ in texts]


class _Store:
    collection_name = "test-chunks"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, vector, filters, *, limit, channel="evidence"):
        self.calls.append((filters, channel))
        return self.hits.get((filters.drug_ids, channel), [])


class _Reranker:
    def score(self, query, chunks):
        return [1.0 if "best" in chunk.id else 0.0 for chunk in chunks]


class _Rewriter:
    def __init__(self, rewrite):
        self.rewrite_plan = rewrite
        self.calls = []

    def should_rewrite_before_recall(self, query, parsed):
        return not parsed.sections and "不能吃" in query

    def rewrite(self, query, parsed, aliases):
        self.calls.append((query, parsed, aliases))
        return self.rewrite_plan


class QueryRetrievalTests(unittest.TestCase):
    def test_bm25_ranks_short_matching_evidence_above_long_noise(self):
        def row(chunk_id, text):
            return (
                chunk_id, "doc-a", "section-1", None, text, text,
                "paragraph", 1, 1, None, "[]", "{}",
            )

        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchall.return_value = [
            row("long", "腹泻" + "其他无关内容" * 100),
            row("short", "腹泻"),
            row("unmatched", "头痛"),
        ]
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with (
            patch("app.storage.retrieval.parse_mysql_url"),
            patch("app.storage.retrieval._connect", return_value=connection),
        ):
            hits = bm25_search(
                "mysql://unused", "腹泻", ChunkFilters(drug_ids=("drug-a",)), limit=2,
            )
        self.assertEqual([hit.chunk.id for hit in hits], ["short", "long"])
        sql, params = cursor.execute.call_args.args
        self.assertIn("p.drug_id IN (%s)", sql)
        self.assertEqual(params, ("drug-a",))

    def test_query_understanding_resolves_aliases_and_exact_entities(self):
        parsed = parse_query(
            "甲药与乙药的3级腹泻如何调整剂量？",
            [("甲药", "drug-a"), ("乙药", "drug-b")],
        )
        self.assertEqual(parsed.drug_ids, ("drug-a", "drug-b"))
        self.assertTrue(parsed.comparison)
        self.assertTrue(parsed.requires_exact_value)
        self.assertEqual(parsed.entities, ("3级",))
        self.assertEqual(parsed.sections, ("用法用量",))
        self.assertEqual(parsed.conditions, ("腹泻",))
        self.assertNotIn("drug_ids", parsed.to_dict())

    def test_ambiguous_drug_alias_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous drug alias"):
            parse_query("甲药的剂量", [("甲药", "drug-a"), ("甲药", "drug-b")])

    def test_common_drug_name_without_dosage_form_resolves(self):
        aliases = [("达可替尼片", "drug-a"), ("奥美拉唑肠溶胶囊", "drug-b")]
        self.assertEqual(parse_query("达可替尼的剂量？", aliases).drug_ids, ("drug-a",))
        self.assertEqual(parse_query("奥美拉唑的禁忌？", aliases).drug_ids, ("drug-b",))
        self.assertEqual(parse_query("达可替尼片的剂量？", aliases).drug_ids, ("drug-a",))

    def test_drug_filter_resolves_alias_and_rejects_conflict(self):
        aliases = [("达可替尼片", "drug-a"), ("厄贝沙坦片", "drug-b")]
        parsed = parse_query("发生腹泻怎么办？", aliases)
        filtered = _apply_drug_filter(parsed, "发生腹泻怎么办？", ("达可替尼",), aliases)
        self.assertEqual(filtered.drug_ids, ("drug-a",))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _apply_drug_filter(
                parse_query("厄贝沙坦的剂量？", aliases), "厄贝沙坦的剂量？",
                ("达可替尼",), aliases,
            )

    def test_explicit_attention_section_keeps_interaction_inference(self):
        parsed = parse_query("甲药与补钾药合用有什么注意事项？", [("甲药", "drug-a")])
        self.assertIn("注意事项", parsed.sections)
        self.assertIn("药物相互作用", parsed.sections)

    def test_population_is_soft_context_for_sparse_dose_chunks(self):
        row = _chunk("child-dose", "drug-a", metadata={"section": "用法用量", "populations": []})
        filters_seen = []

        def bm25(_mysql, _query, filters, *, limit):
            filters_seen.append(filters)
            return [LexicalHit(row, 1.0)]

        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[("甲药片", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[]),
            patch("app.retrieval.pipeline.bm25_search", side_effect=bm25),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), _Store({}), _Reranker()).retrieve(
                "甲药儿童体重不超过30公斤一次用多少？"
            )
        self.assertEqual([item.chunk.id for item in result.groups[0].evidence], ["child-dose"])
        self.assertEqual(filters_seen[0].populations, ())

    def test_colloquial_query_rewrite_adds_scoped_recall_and_trace(self):
        source = _chunk(
            "best-warning", "drug-a",
            metadata={"section": "注意事项"},
        )
        rewrite = QueryRewrite(
            "高血压患者禁用、慎用或应避免联合使用哪些药物",
            "overview",
            ("禁忌", "注意事项", "药物相互作用"),
            ("高血压",),
            (),
            (),
            ("高血压患者慎用药物",),
            "口语风险表达标准化",
        )
        rewriter = _Rewriter(rewrite)
        calls = []

        def bm25(_mysql, query, filters, *, limit):
            calls.append((query, filters.sections))
            if query == rewrite.normalized_query:
                return [LexicalHit(source, 2.0)]
            return []

        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[]),
            patch("app.retrieval.pipeline.bm25_search", side_effect=bm25),
            patch("app.retrieval.pipeline.summary_search", return_value=[]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine(
                "mysql://unused", _Embedder(), _Store({}), _Reranker(),
                query_rewriter=rewriter,
            ).retrieve("高血压患者不能吃哪些药")

        self.assertEqual([item.chunk.id for item in result.groups[0].evidence], ["best-warning"])
        self.assertEqual(len(rewriter.calls), 1)
        self.assertTrue(result.query_rewrite["applied"])
        self.assertIn(
            (rewrite.normalized_query, rewrite.sections), calls
        )

    def test_rrf_retains_channel_ranks_and_scores(self):
        first = _chunk("a", "drug-a")
        second = _chunk("b", "drug-a")
        fused = reciprocal_rank_fusion(
            [LexicalHit(first, 8.0), LexicalHit(second, 4.0)],
            [VectorHit(second, 0.9), VectorHit(first, 0.7)], k=60,
        )
        self.assertEqual([item.chunk.id for item in fused], ["a", "b"])
        self.assertEqual((fused[0].bm25_rank, fused[0].vector_rank), (1, 2))
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 61 + 1 / 62)
        self.assertIn("[药品] drug-a", candidate_text(first))
        self.assertIn("[章节] 不良反应", candidate_text(first))
        table_row = _chunk("row", "drug-a", kind="table_row")
        table_row.retrieval_text = "表头：严重程度；数值：3级"
        self.assertIn("[类型] 表格行", candidate_text(table_row))
        self.assertIn("表头：严重程度", candidate_text(table_row))

    def test_multi_drug_retrieval_keeps_evidence_scoped_and_reports_missing(self):
        hit = _chunk("best-a", "drug-a")
        store = _Store({(("drug-a",), "evidence"): [VectorHit(hit, 0.9)]})

        def lexical(_mysql, _query, filters, *, limit):
            return [LexicalHit(hit, 2.0)] if filters.drug_ids == ("drug-a",) else []

        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[
                ("甲药", "drug-a"), ("乙药", "drug-b")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=[
                "doc-drug-a", "doc-drug-b"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[
                "doc-drug-a", "doc-drug-b"]),
            patch("app.retrieval.pipeline.bm25_search", side_effect=lexical),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), store, _Reranker()).retrieve(
                "甲药与乙药的不良反应有什么区别？"
            )

        self.assertEqual([group.drug_id for group in result.groups], ["drug-a", "drug-b"])
        self.assertEqual([item.chunk.id for item in result.groups[0].evidence], ["best-a"])
        self.assertFalse(result.groups[1].evidence)
        self.assertTrue(result.to_dict()["groups"][1]["missing_evidence"])
        self.assertEqual(
            result.to_dict()["comparison_dimensions"][0]["groups"][1]["evidence_status"],
            "当前说明书未检索到相关表述",
        )
        self.assertEqual({call[0].drug_ids for call in store.calls}, {("drug-a",), ("drug-b",)})

    def test_summary_routes_to_original_child_and_recovers_context(self):
        source = _chunk("best-source", "drug-a", metadata={"lead_in_id": "lead"})
        lead = _chunk("lead", "drug-a")
        summary = _chunk(
            "summary", "drug-a", kind="section_summary", parent_id=None,
            metadata={"key_point_sources": {"腹泻": ["best-source"]}},
        )
        store = _Store({(("drug-a",), "summary"): [VectorHit(summary, 0.6)]})
        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[("甲药", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.bm25_search", return_value=[]),
            patch("app.retrieval.pipeline.summary_search", return_value=[]),
            patch("app.retrieval.pipeline.load_summary_sources", return_value=[source]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[lead]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), store, _Reranker()).retrieve(
                "甲药有哪些不良反应？"
            )
        evidence = result.groups[0].evidence
        self.assertEqual([item.chunk.id for item in evidence], ["best-source"])
        self.assertGreater(evidence[0].summary_route_score, 0)
        self.assertEqual([part.id for part in evidence[0].context_chunks], ["lead"])

    def test_scoped_overview_uses_section_summary_without_primary_recall(self):
        source = _chunk(
            "best-attention", "drug-a", parent_id=None,
            metadata={"section": "注意事项"},
        )
        summary = _chunk(
            "attention-summary", "drug-a", kind="section_summary", parent_id=None,
            metadata={
                "section": "注意事项",
                "key_point_sources": {"联合用药": ["best-attention"]},
            },
        )
        store = _Store({
            (("drug-a",), "summary"): [VectorHit(summary, 0.9)],
        })
        primary_search = MagicMock(return_value=[])
        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[
                ("甲药", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=[
                "doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[
                "doc-drug-a"]),
            patch("app.retrieval.pipeline.bm25_search", primary_search),
            patch("app.retrieval.pipeline.summary_search", return_value=[
                LexicalHit(summary, 2.0)]),
            patch("app.retrieval.pipeline.load_summary_sources", return_value=[source]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine(
                "mysql://unused", _Embedder(), store, _Reranker()
            ).retrieve("甲药有哪些注意事项？")

        self.assertEqual(
            [item.chunk.id for item in result.groups[0].evidence],
            ["best-attention"],
        )
        primary_search.assert_not_called()
        self.assertEqual([channel for _, channel in store.calls], ["summary"])

    def test_scoped_overview_falls_back_when_section_summary_is_unavailable(self):
        source = _chunk(
            "best-attention", "drug-a", parent_id=None,
            metadata={"section": "注意事项"},
        )
        primary_search = MagicMock(return_value=[LexicalHit(source, 2.0)])
        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[
                ("甲药", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=[
                "doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[]),
            patch("app.retrieval.pipeline.bm25_search", primary_search),
            patch("app.retrieval.pipeline.summary_search", return_value=[]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[]),
        ):
            result = RetrievalEngine(
                "mysql://unused", _Embedder(), _Store({}), _Reranker()
            ).retrieve("甲药有哪些注意事项？")

        self.assertEqual(result.groups[0].evidence[0].chunk.id, "best-attention")
        primary_search.assert_called_once()

    def test_comparison_aligns_each_section_for_each_drug(self):
        adverse = _chunk("best-adverse", "drug-a")
        store = _Store({})

        def bm25(_mysql, _query, filters, *, limit):
            if filters.drug_ids == ("drug-a",) and filters.sections == ("不良反应",):
                return [LexicalHit(adverse, 1.0)]
            return []

        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[
                ("甲药", "drug-a"), ("乙药", "drug-b")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=[
                "doc-drug-a", "doc-drug-b"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[
                "doc-drug-a", "doc-drug-b"]),
            patch("app.retrieval.pipeline.bm25_search", side_effect=bm25),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), store, _Reranker()).retrieve(
                "甲药与乙药的不良反应和禁忌有什么区别？"
            )
        dimensions = result.to_dict()["comparison_dimensions"]
        self.assertEqual([item["dimension"] for item in dimensions], ["不良反应", "禁忌"])
        self.assertEqual([group["missing_evidence"] for group in dimensions[0]["groups"]], [False, True])
        self.assertTrue(all(group["missing_evidence"] for group in dimensions[1]["groups"]))
        self.assertEqual(
            [call[0].sections for call in store.calls],
            [("不良反应",), ("不良反应",), ("禁忌",), ("禁忌",)],
        )

    def test_active_document_without_ready_vectors_uses_bm25_only(self):
        source = _chunk("best-source", "drug-a", parent_id=None)
        store = _Store({})
        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[("甲药", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[]),
            patch("app.retrieval.pipeline.bm25_search", return_value=[LexicalHit(source, 1.2)]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), store, _Reranker()).retrieve(
                "甲药的不良反应是什么？"
            )
        self.assertFalse(store.calls)
        self.assertEqual(result.groups[0].evidence[0].chunk.id, "best-source")
        self.assertIsNone(result.groups[0].evidence[0].vector_rank)

    def test_shared_parent_is_emitted_once(self):
        first = _chunk("best-row", "drug-a", kind="table_row")
        second = _chunk("other-row", "drug-a", kind="table_row")
        parent = _chunk("parent-1", "drug-a", kind="row_group_parent", parent_id=None)
        with (
            patch("app.retrieval.pipeline.list_active_drug_aliases", return_value=[("甲药", "drug-a")]),
            patch("app.retrieval.pipeline.list_active_document_ids", return_value=["doc-drug-a"]),
            patch("app.retrieval.pipeline.list_ready_vector_document_ids", return_value=[]),
            patch("app.retrieval.pipeline.bm25_search", return_value=[
                LexicalHit(first, 2.0), LexicalHit(second, 1.0)]),
            patch("app.retrieval.pipeline.summary_search", return_value=[]),
            patch("app.retrieval.pipeline.load_chunks_by_ids", return_value=[]),
            patch("app.retrieval.pipeline.load_parent_chunks", return_value=[parent]),
        ):
            result = RetrievalEngine("mysql://unused", _Embedder(), _Store({}), _Reranker()).retrieve(
                "甲药有哪些不良反应？"
            )
        context_ids = [
            part.id for item in result.groups[0].evidence for part in item.context_chunks
        ]
        self.assertEqual(context_ids, ["parent-1"])


if __name__ == "__main__":
    unittest.main()
