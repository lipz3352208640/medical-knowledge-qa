import unittest

from app.retrieval.query import parse_query
from app.retrieval.rewrite import LLMQueryRewriter, QueryRewriteError


class _Chat:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user, schema, *, max_output_tokens=1024):
        self.calls.append((system, user, schema, max_output_tokens))
        return self.response


class QueryRewriteTests(unittest.TestCase):
    def test_colloquial_overview_is_aligned_to_canonical_sections(self):
        query = "高血压患者不能吃哪些药"
        parsed = parse_query(query, [])
        chat = _Chat({
            "normalized_query": "高血压患者禁用、慎用或应避免联合使用哪些药物",
            "intent": "overview",
            "sections": ["禁忌", "注意事项", "药物相互作用"],
            "conditions": ["高血压"],
            "populations": [],
            "entities": [],
            "query_expansions": [
                "高血压患者禁用药物",
                "高血压患者慎用及避免合用药物",
            ],
            "rewrite_reason": "将口语化的不能吃扩展到说明书标准风险表达",
        })
        rewriter = LLMQueryRewriter(chat)
        self.assertTrue(rewriter.should_rewrite_before_recall(query, parsed))
        rewrite = rewriter.rewrite(query, parsed, [])
        self.assertEqual(rewrite.intent, "overview")
        self.assertEqual(
            rewrite.sections, ("禁忌", "注意事项", "药物相互作用")
        )
        self.assertIn("不能吃", chat.calls[0][1])

    def test_rewrite_cannot_introduce_number_or_drug(self):
        number_query = "腹泻怎么办"
        number_parsed = parse_query(number_query, [])
        number_chat = _Chat({
            "normalized_query": "3级腹泻怎么办",
            "intent": "fact",
            "sections": [], "conditions": ["腹泻"], "populations": [],
            "entities": ["3级"], "query_expansions": [],
            "rewrite_reason": "错误地增加了等级",
        })
        with self.assertRaisesRegex(QueryRewriteError, "new number"):
            LLMQueryRewriter(number_chat).rewrite(number_query, number_parsed, [])

        drug_query = "腹泻怎么办"
        aliases = [("甲药", "drug-a")]
        drug_chat = _Chat({
            "normalized_query": "甲药腹泻怎么办",
            "intent": "fact",
            "sections": [], "conditions": ["腹泻"], "populations": [],
            "entities": [], "query_expansions": [],
            "rewrite_reason": "错误地增加了药品",
        })
        with self.assertRaisesRegex(QueryRewriteError, "new drug"):
            LLMQueryRewriter(drug_chat).rewrite(
                drug_query, parse_query(drug_query, aliases), aliases
            )


if __name__ == "__main__":
    unittest.main()
