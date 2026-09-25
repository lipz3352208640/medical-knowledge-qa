import unittest

from app.chunker.parent_child import CHUNKER_VERSION, build_chunks
from app.chunker.summaries import SummaryPolicy, build_summary_chunks
from app.models import ParseResult, Section, SourceRef, TextElement, ValuePair


class SummaryChunkTests(unittest.TestCase):
    def make_result(self, count: int) -> ParseResult:
        document_id = "summary-doc"
        section_id = "adverse-section"
        heading_ref = SourceRef(document_id, section_id, 1, (1, 10, 100, 20))
        section = Section(
            section_id, document_id, "【不良反应】", "不良反应",
            1, 1, heading_ref, 1.0,
        )
        texts = (
            "腹泻发生率见原文。", "腹泻3级时暂停给药。",
            "腹泻再次发生时需要调整剂量。",
        )
        elements = [
            TextElement(
                f"line-{index}",
                SourceRef(document_id, section_id, 1, (1, 30 + index * 10, 100, 39 + index * 10)),
                "paragraph", texts[index % len(texts)], texts[index % len(texts)],
                index, 0.9,
            )
            for index in range(count)
        ]
        return ParseResult(
            document_id, "leaflet.pdf", "a" * 64, "test-parser",
            ValuePair("测试药", "测试药"), 1, [section], elements,
            [], [], [], [],
        )

    def test_section_summary_routes_to_real_children(self):
        result = self.make_result(9)
        chunks = build_chunks(result, "drug-id", "product-id")
        evidence = [chunk for chunk in chunks if chunk.content_type == "paragraph"]
        summaries = [chunk for chunk in chunks if chunk.content_type.endswith("_summary")]
        self.assertEqual(len(evidence), 9)
        self.assertEqual([chunk.content_type for chunk in summaries], ["section_summary"])
        summary = summaries[0]
        self.assertEqual(set(summary.metadata["source_chunk_ids"]), {chunk.id for chunk in evidence})
        self.assertEqual(summary.metadata["index_channel"], "summary")
        self.assertEqual(summary.metadata["evidence_role"], "navigation")
        self.assertTrue(summary.metadata["key_points"])
        self.assertTrue(all(
            set(ids) <= set(summary.metadata["source_chunk_ids"]) and ids
            for ids in summary.metadata["key_point_sources"].values()
        ))
        self.assertEqual(
            [chunk.id for chunk in chunks],
            [chunk.id for chunk in build_chunks(result, "drug-id", "product-id")],
        )

    def test_topic_summary_requires_size_and_multiple_dimensions(self):
        result = self.make_result(9)
        evidence = [
            chunk for chunk in build_chunks(result, "drug-id", "product-id")
            if chunk.content_type == "paragraph"
        ]
        summaries = build_summary_chunks(
            result, evidence, "drug-id", "product-id", CHUNKER_VERSION,
            SummaryPolicy(topic_min_tokens=20),
        )
        topic = next(chunk for chunk in summaries if chunk.content_type == "topic_summary")
        self.assertEqual(topic.topic_name, "腹泻")
        self.assertGreaterEqual(len(topic.metadata["key_points"]), 2)
        self.assertEqual(set(topic.metadata["source_chunk_ids"]), {chunk.id for chunk in evidence})

        short_result = self.make_result(2)
        self.assertFalse(any(
            chunk.content_type.endswith("_summary")
            for chunk in build_chunks(short_result, "drug-id", "product-id")
        ))


if __name__ == "__main__":
    unittest.main()
