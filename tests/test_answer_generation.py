import io
import json
import unittest
from unittest.mock import patch

from app.generator.answer import AnswerService, _source_quote, _unsupported_values
from app.generator.ollama import ChatError, OllamaChatClient, OllamaChatConfig
from app.retrieval.compression import CompressionConfig, ContextCompressor
from app.retrieval.pipeline import (
    ComparisonDimension, DrugEvidence, RankedCandidate, RetrievalResult,
)
from app.retrieval.query import ParsedQuery
from app.storage.retrieval import StoredChunk


def _parsed(*, comparison=False, broad=False, sections=("用法用量",)):
    names = ("甲药", "乙药") if comparison else ("甲药",)
    ids = ("drug-a", "drug-b") if comparison else ("drug-a",)
    return ParsedQuery(
        names, "comparison" if comparison else "dose", sections,
        (), (), (), comparison, False, ids, broad,
    )


def _chunk(chunk_id, content, *, drug_id="drug-a", doc_id="doc-a", metadata=None):
    return StoredChunk(
        chunk_id, doc_id, "section-1", None, content, content,
        "paragraph", 2, 2, [1, 2, 3, 4],
        [{"document_id": doc_id, "section_id": "section-1", "page": 2, "bbox": [1, 2, 3, 4]}],
        {"drug_id": drug_id, "generic_name": "甲药", "section": "用法用量",
         "source_file": "甲药说明书.pdf", **(metadata or {})},
    )


class _Embedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0]] + [
            [0.9, 0.1] if "相关" in text else [0.0, 1.0]
            for text in texts[1:]
        ]


class _Chat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, schema, *, max_output_tokens=1024):
        self.calls.append((system, user, schema, max_output_tokens))
        if not self.responses:
            raise ChatError("No mocked chat response remains")
        return self.responses.pop(0)


class _Retriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query):
        return self.result


class CompressionTests(unittest.TestCase):
    def test_short_question_uses_top_three_without_compression_calls(self):
        group = DrugEvidence("甲药", "drug-a", [
            RankedCandidate(_chunk(str(index), f"短证据{index}")) for index in range(4)
        ])
        chat = _Chat([])
        embedder = _Embedder()
        prepared = ContextCompressor(embedder, chat).prepare(
            "甲药的剂量？", _parsed(), group, total_candidates=4,
        )
        self.assertEqual([unit.chunk.id for unit in prepared.units], ["0", "1", "2"])
        self.assertEqual(prepared.strategies, ["direct_top3"])
        self.assertFalse(chat.calls)
        self.assertFalse(embedder.calls)

    def test_embedding_filter_prunes_context_but_keeps_lead(self):
        primary = _chunk("main", "正文", metadata={"lead_in_id": "lead"})
        candidate = RankedCandidate(primary, context_chunks=[
            _chunk("lead", "引导句"),
            _chunk("c1", "无关段落一"),
            _chunk("c2", "相关段落"),
            _chunk("c3", "无关段落二"),
        ])
        group = DrugEvidence("甲药", "drug-a", [candidate])
        embedder = _Embedder()
        prepared = ContextCompressor(
            embedder, _Chat([]), CompressionConfig(
                direct_evidence_tokens=1, embedding_context_count=2, embedding_keep=1,
            ),
        ).prepare("甲药怎么处理？", _parsed(), group, total_candidates=1)
        self.assertEqual([unit.chunk.id for unit in prepared.units], ["main", "lead", "c2"])
        self.assertIn("embedding_filter", prepared.strategies)

    def test_llm_filter_keeps_best_ranked_child_even_if_model_drops_it(self):
        group = DrugEvidence("甲药", "drug-a", [
            RankedCandidate(_chunk(chunk_id, "原文"))
            for chunk_id in ("first", "middle", "last")
        ])
        chat = _Chat([{"keep_ids": ["last", "unknown"]}])
        prepared = ContextCompressor(
            _Embedder(), chat, CompressionConfig(direct_evidence_tokens=1),
        ).prepare("甲药与乙药比较", _parsed(comparison=True), group, total_candidates=10)
        self.assertEqual([unit.chunk.id for unit in prepared.units], ["first", "last"])
        self.assertIn("llm_filter", prepared.strategies)

    def test_extractor_keeps_exact_substring_and_source_mapping(self):
        source = _chunk("long", "前文无关内容。发生3级腹泻时暂停给药。后文无关内容。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        chat = _Chat([{"extracts": [{
            "chunk_id": "long", "quote": "发生3级腹泻时暂停给药。",
        }]}])
        prepared = ContextCompressor(
            _Embedder(), chat, CompressionConfig(
                direct_evidence_tokens=1, extract_chunk_tokens=10,
            ),
        ).prepare("甲药腹泻如何处理？", _parsed(), group, total_candidates=1)
        self.assertEqual(prepared.units[0].text, "发生3级腹泻时暂停给药。")
        self.assertEqual(prepared.units[0].chunk.id, "long")
        self.assertIn("llm_extractor", prepared.strategies)

    def test_extractor_keeps_distinct_quotes_when_model_repeats_one(self):
        source = _chunk(
            "long", "前文无关内容。发生3级腹泻时暂停给药。中间无关内容。症状恢复后恢复给药。后文无关内容。",
        )
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        first = "发生3级腹泻时暂停给药。"
        second = "症状恢复后恢复给药。"
        chat = _Chat([{"extracts": [
            {"chunk_id": "long", "quote": first},
            {"chunk_id": "long", "quote": first},
            {"chunk_id": "long", "quote": first},
            {"chunk_id": "long", "quote": second},
        ]}])
        prepared = ContextCompressor(
            _Embedder(), chat, CompressionConfig(
                direct_evidence_tokens=1, extract_chunk_tokens=10,
            ),
        ).prepare("甲药腹泻如何处理？", _parsed(), group, total_candidates=1)
        self.assertEqual(prepared.units[0].text, first + "\n[省略]\n" + second)
        self.assertEqual(prepared.units[0].chunk.id, "long")

    def test_invalid_extractor_quote_preserves_original(self):
        source = _chunk("long", "前文无关内容。发生3级腹泻时暂停给药。后文无关内容。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        chat = _Chat([{"extracts": [{
            "chunk_id": "long", "quote": "发生4级腹泻时停药。",
        }]}])
        prepared = ContextCompressor(
            _Embedder(), chat, CompressionConfig(
                direct_evidence_tokens=1, extract_chunk_tokens=10,
            ),
        ).prepare("甲药腹泻如何处理？", _parsed(), group, total_candidates=1)
        self.assertEqual(prepared.units[0].text, source.content)
        self.assertTrue(any("未通过校验" in warning for warning in prepared.warnings))


class AnswerGenerationTests(unittest.TestCase):
    def test_adverse_overview_falls_back_to_exact_excerpts(self):
        first = _chunk("adverse-1", "恶心、呕吐、腹泻。", metadata={"section": "不良反应"})
        second = _chunk("adverse-2", "头晕、低血压。", metadata={"section": "不良反应"})
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(first), RankedCandidate(second)])
        retrieval = RetrievalResult(
            "问题", _parsed(broad=True, sections=("不良反应",)), [group],
        )
        chat = _Chat([{
            "answer": "1. 恶心、呕吐、腹泻；2. 头晕、低血压，发生率0.1%。",
            "citations": [
                {"chunk_id": "adverse-1", "quote": first.content},
                {"chunk_id": "adverse-2", "quote": second.content},
            ],
            "confidence": "high", "warnings": [],
        }])
        result = AnswerService(
            _Retriever(retrieval), ContextCompressor(_Embedder(), chat), chat,
        ).answer("甲药有哪些不良反应？")
        self.assertEqual(result.confidence, "medium")
        self.assertIn("原文摘录", result.answer)
        self.assertNotIn("0.1%", result.answer)
        self.assertEqual([item["chunk_id"] for item in result.evidence], ["adverse-1", "adverse-2"])
        self.assertIn("extractive_fallback", result.compression[0]["strategies"])
        self.assertEqual(len(chat.calls), 1)
        self.assertIn("不要添加发生率", chat.calls[0][1])

    def test_listed_grade_alternative_is_supported_by_exact_quote(self):
        quote = "腹泻；3或4级；暂时停用本品。"
        self.assertEqual(_unsupported_values("3级腹泻时暂时停用本品。", quote), [])
        self.assertEqual(_unsupported_values("4级腹泻时暂时停用本品。", quote), [])
        self.assertIn("5级", _unsupported_values("5级腹泻时暂时停用本品。", quote))
        self.assertIn("3毫克", _unsupported_values("服用3毫克。", quote))

    def test_list_markers_are_not_treated_as_medical_values(self):
        quote = "有高钾风险，应避免联合使用。"
        self.assertEqual(
            _unsupported_values("1. 有高钾风险，应避免联合使用。", quote), []
        )
        self.assertEqual(
            _unsupported_values("（1）有高钾风险，应避免联合使用。", quote), []
        )
        self.assertIn("1.5毫克", _unsupported_values("使用1.5毫克。", quote))

    @staticmethod
    def _service(response, *, comparison=False, second=None):
        source = _chunk("row-1", "发生3级腹泻时暂停给药。")
        first = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        groups = [first]
        dimensions = []
        if comparison:
            groups.append(second or DrugEvidence("乙药", "drug-b", []))
            dimensions = [ComparisonDimension("用法用量", groups)]
        result = RetrievalResult("问题", _parsed(comparison=comparison), groups, dimensions)
        chat = _Chat(response if isinstance(response, list) else [response])
        service = AnswerService(_Retriever(result), ContextCompressor(_Embedder(), chat), chat)
        return service, chat

    def test_valid_answer_citations_are_filled_from_original_chunk(self):
        service, chat = self._service({
            "answer": "发生3级腹泻时暂停给药。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "high", "warnings": [],
        })
        result = service.answer("甲药腹泻怎么办？")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.evidence[0]["source_file"], "甲药说明书.pdf")
        self.assertEqual(result.evidence[0]["page"], 2)
        self.assertEqual(result.evidence[0]["chunk_id"], "row-1")
        self.assertIn("只依据", chat.calls[0][0])

    def test_duplicate_parent_quote_prefers_original_child(self):
        quote = "发生3级腹泻时暂停给药。"
        child = _chunk("child", quote)
        parent = _chunk("parent", "前文。" + quote + "后文。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(child, context_chunks=[parent])])
        result = RetrievalResult("问题", _parsed(), [group])
        chat = _Chat([{
            "answer": quote,
            "citations": [
                {"chunk_id": "parent", "quote": quote},
                {"chunk_id": "child", "quote": quote},
            ],
            "confidence": "high", "warnings": [],
        }])
        answer = AnswerService(
            _Retriever(result), ContextCompressor(_Embedder(), chat), chat,
        ).answer("甲药腹泻怎么办？")
        self.assertEqual([item["chunk_id"] for item in answer.evidence], ["child"])

    def test_citation_restores_pdf_line_break_without_changing_words(self):
        source = _chunk("dose", "通常建议的初始剂量为每日\n150mg。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        retrieval = RetrievalResult("问题", _parsed(), [group])
        chat = _Chat([{
            "answer": "通常建议的初始剂量为每日150mg。",
            "citations": [{"chunk_id": "dose", "quote": "通常建议的初始剂量为每日150mg。"}],
            "confidence": "high", "warnings": [],
        }])
        result = AnswerService(
            _Retriever(retrieval), ContextCompressor(_Embedder(), chat), chat,
        ).answer("甲药通常的初始剂量是多少？")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.evidence[0]["quote"], source.content)

        self.assertIsNone(_source_quote("通常建议的初始剂量为每日75mg。", source.content, source.content))

    def test_failed_numeric_citation_is_repaired_once(self):
        quote = "发生3级腹泻时暂停给药。"
        service, chat = self._service([
            {"answer": "发生4级腹泻时暂停给药。",
             "citations": [{"chunk_id": "row-1", "quote": quote}],
             "confidence": "high", "warnings": []},
            {"answer": quote,
             "citations": [{"chunk_id": "row-1", "quote": quote}],
             "confidence": "high", "warnings": []},
        ])
        result = service.answer("甲药腹泻怎么办？")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(chat.calls), 2)
        self.assertIn("validation_retry", result.compression[0]["strategies"])

    def test_model_refusal_drops_unrelated_citations(self):
        service, chat = self._service({
            "answer": "说明书未提及这一特定情境，仅有常规剂量。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "low", "warnings": [],
        })
        result = service.answer("甲药在未知情境的剂量？")
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.evidence)
        self.assertIn("无法依据证据", result.answer)
        self.assertEqual(len(chat.calls), 1)

    def test_model_refusal_with_not_provided_drops_unrelated_citations(self):
        service, chat = self._service({
            "answer": "说明书未提供这一特定情境的剂量，仅给出常规剂量。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "low", "warnings": [],
        })
        result = service.answer("甲药在未知情境的剂量？")
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.evidence)
        self.assertIn("无法依据证据", result.answer)
        self.assertEqual(len(chat.calls), 1)

    def test_hallucinated_quote_and_number_are_blocked(self):
        invalid_quote, _ = self._service({
            "answer": "发生4级腹泻时暂停给药。",
            "citations": [{"chunk_id": "row-1", "quote": "发生4级腹泻时暂停给药。"}],
            "confidence": "high", "warnings": [],
        })
        self.assertEqual(invalid_quote.answer("甲药腹泻怎么办？").confidence, "low")
        invalid_number, _ = self._service({
            "answer": "发生4级腹泻时暂停给药。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "high", "warnings": [],
        })
        result = invalid_number.answer("甲药腹泻怎么办？")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("数值" in warning for warning in result.warnings))

    def test_same_number_with_wrong_unit_and_personal_instruction_are_blocked(self):
        source = _chunk("dose", "每次3片。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        retrieval = RetrievalResult("问题", _parsed(), [group])
        chat = _Chat([
            {"answer": "每次3毫克。", "citations": [{"chunk_id": "dose", "quote": source.content}],
             "confidence": "high", "warnings": []},
            {"answer": "每次3毫克。", "citations": [{"chunk_id": "dose", "quote": source.content}],
             "confidence": "high", "warnings": []},
            {"answer": "您应该服用每次3片。", "citations": [{"chunk_id": "dose", "quote": source.content}],
             "confidence": "high", "warnings": []},
        ])
        service = AnswerService(_Retriever(retrieval), ContextCompressor(_Embedder(), chat), chat)
        self.assertEqual(service.answer("甲药剂量？").confidence, "low")
        self.assertEqual(service.answer("甲药剂量？").confidence, "low")

    def test_frequency_basis_must_match_cited_text(self):
        source = _chunk("frequency", "每日1次。")
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        retrieval = RetrievalResult("问题", _parsed(), [group])
        chat = _Chat([{
            "answer": "每周1次。",
            "citations": [{"chunk_id": "frequency", "quote": source.content}],
            "confidence": "high", "warnings": [],
        }])
        service = AnswerService(_Retriever(retrieval), ContextCompressor(_Embedder(), chat), chat)
        result = service.answer("甲药的使用频次？")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("数值" in warning for warning in result.warnings))

    def test_unsupported_contraindication_claim_is_blocked(self):
        service, _ = self._service({
            "answer": "甲药禁用。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "high", "warnings": [],
        })
        result = service.answer("甲药怎么办？")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("禁用" in warning for warning in result.warnings))

    def test_unspecified_drug_with_mixed_sources_requests_clarification(self):
        first = RankedCandidate(_chunk("a", "甲药说明。"))
        second = RankedCandidate(_chunk("b", "乙药说明。", drug_id="drug-b"))
        group = DrugEvidence(None, None, [first, second])
        parsed = ParsedQuery((), "fact", (), (), (), (), False, False, (), False)
        chat = _Chat([])
        service = AnswerService(
            _Retriever(RetrievalResult("有哪些注意事项？", parsed, [group])),
            ContextCompressor(_Embedder(), chat), chat,
        )
        result = service.answer("有哪些注意事项？")
        self.assertIn("请明确要查询的药品", result.answer)
        self.assertFalse(chat.calls)

    def test_cross_drug_overview_keeps_supported_partial_answer(self):
        first = _chunk(
            "drug-a-warning", "未说明高血压患者的相关限制。",
            metadata={"generic_name": "甲药"},
        )
        second = _chunk(
            "drug-b-warning", "有高钾风险，应避免联合使用。",
            drug_id="drug-b", doc_id="doc-b",
            metadata={"generic_name": "乙药"},
        )
        group = DrugEvidence(None, None, [
            RankedCandidate(first), RankedCandidate(second),
        ])
        parsed = ParsedQuery(
            (), "overview", (), (), ("高血压",), (),
            False, False, (), True,
        )
        chat = _Chat([
            {
                "answer": "说明书未提供这一特定情境的信息。",
                "citations": [], "confidence": "low", "warnings": [],
            },
            {
                "answer": "1. 有高钾风险，应避免联合使用。",
                "citations": [{
                    "chunk_id": "drug-b-warning", "quote": second.content,
                }],
                "confidence": "high", "warnings": [],
            },
        ])
        result = AnswerService(
            _Retriever(RetrievalResult("问题", parsed, [group])),
            ContextCompressor(_Embedder(), chat), chat,
        ).answer("高血压患者不能吃哪些药")
        self.assertEqual(result.confidence, "medium")
        self.assertEqual(len(result.evidence), 1)
        self.assertIn("有高钾风险", result.answer)
        self.assertTrue(any("部分药品" in warning for warning in result.warnings))

    def test_unfittable_source_returns_low_confidence_without_llm(self):
        source = _chunk("large", "较长的原文证据。" * 10)
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(source)])
        chat = _Chat([])
        service = AnswerService(
            _Retriever(RetrievalResult("问题", _parsed(), [group])),
            ContextCompressor(_Embedder(), chat, CompressionConfig(
                prompt_token_budget=20, direct_evidence_tokens=1,
            )), chat,
        )
        result = service.answer("甲药说明？")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("预算" in warning for warning in result.warnings))
        self.assertFalse(chat.calls)

    def test_comparison_missing_drug_is_explicit_and_not_sent_to_model(self):
        service, chat = self._service({
            "answer": "发生3级腹泻时暂停给药。",
            "citations": [{"chunk_id": "row-1", "quote": "发生3级腹泻时暂停给药。"}],
            "confidence": "high", "warnings": [],
        }, comparison=True)
        result = service.answer("甲药和乙药腹泻处理比较")
        self.assertIn("乙药：当前说明书未检索到相关表述", result.answer)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(result.confidence, "low")

    def test_comparison_with_no_evidence_reports_each_drug(self):
        first = DrugEvidence("甲药", "drug-a", [])
        second = DrugEvidence("乙药", "drug-b", [])
        groups = [first, second]
        retrieval = RetrievalResult(
            "比较", _parsed(comparison=True), groups,
            [ComparisonDimension("用法用量", groups)],
        )
        chat = _Chat([])
        service = AnswerService(_Retriever(retrieval), ContextCompressor(_Embedder(), chat), chat)
        result = service.answer("甲药与乙药用法用量比较")
        self.assertIn("甲药：当前说明书未检索到相关表述", result.answer)
        self.assertIn("乙药：当前说明书未检索到相关表述", result.answer)
        self.assertEqual(result.confidence, "low")
        self.assertFalse(chat.calls)

    def test_multiple_source_documents_are_answered_separately(self):
        first = _chunk("a", "发生3级腹泻时暂停给药。", metadata={
            "source_file": "旧版.pdf", "revision_date": "2020-01-01",
        })
        second = _chunk("b", "发生3级腹泻时降低剂量。", doc_id="doc-b", metadata={
            "source_file": "新版.pdf", "revision_date": "2024-01-01",
        })
        group = DrugEvidence("甲药", "drug-a", [RankedCandidate(first), RankedCandidate(second)])
        result = RetrievalResult("问题", _parsed(), [group])
        chat = _Chat([
            {"answer": "发生3级腹泻时暂停给药。", "citations": [{"chunk_id": "a", "quote": first.content}],
             "confidence": "high", "warnings": []},
            {"answer": "发生3级腹泻时降低剂量。", "citations": [{"chunk_id": "b", "quote": second.content}],
             "confidence": "high", "warnings": []},
        ])
        service = AnswerService(_Retriever(result), ContextCompressor(_Embedder(), chat), chat)
        answer = service.answer("甲药腹泻怎么处理？")
        self.assertIn("旧版.pdf，修订日期2020-01-01", answer.answer)
        self.assertIn("新版.pdf，修订日期2024-01-01", answer.answer)
        self.assertEqual(len(answer.evidence), 2)
        self.assertEqual(len(chat.calls), 2)


class OllamaChatTests(unittest.TestCase):
    def test_sends_qwen_chat_with_schema_and_parses_json(self):
        reply = {"done": True, "message": {"content": json.dumps({"answer": "ok"})}}
        captured = []

        def urlopen(request, timeout):
            captured.append((request, timeout))
            return io.BytesIO(json.dumps(reply).encode("utf-8"))

        with patch("app.generator.ollama.urlopen", side_effect=urlopen):
            result = OllamaChatClient(OllamaChatConfig()).complete(
                "system", "user", {"type": "object"}, max_output_tokens=123,
            )
        body = json.loads(captured[0][0].data)
        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(body["model"], "qwen3.5:latest")
        self.assertEqual(body["format"], {"type": "object"})
        self.assertFalse(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["options"]["num_predict"], 123)


if __name__ == "__main__":
    unittest.main()
