"""Validated LLM query rewriting for low-confidence retrieval plans."""

from dataclasses import dataclass
import json
import re

from app.generator.ollama import ChatModel
from app.parser.normalize import normalize_section
from app.retrieval.query import ParsedQuery, parse_query


_ALLOWED_INTENTS = {
    "fact", "overview", "exact_value", "comparison", "indication", "dose",
    "adverse_reaction", "contraindication", "warning", "interaction",
    "special_population",
}
_ALLOWED_SECTIONS = {
    "适应症", "用法用量", "不良反应", "禁忌", "注意事项", "药物相互作用",
    "儿童用药", "老年用药", "孕妇及哺乳期妇女用药",
}
_INTENT_SECTIONS = {
    "indication": ("适应症",),
    "dose": ("用法用量",),
    "adverse_reaction": ("不良反应",),
    "contraindication": ("禁忌",),
    "warning": ("注意事项",),
    "interaction": ("药物相互作用",),
}
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?")
_NEGATIONS = ("不能", "不得", "不应", "不可", "禁止", "禁用", "避免", "慎用", "未", "无")
_COLLOQUIAL = (
    "不能吃", "能不能吃", "怎么吃", "咋吃", "一起吃", "一块吃", "有啥",
    "咋办", "怎么办", "能不能", "要不要", "会不会",
)

_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "normalized_query": {"type": "string"},
        "intent": {"type": "string", "enum": sorted(_ALLOWED_INTENTS)},
        "sections": {
            "type": "array", "items": {"type": "string"},
        },
        "conditions": {
            "type": "array", "items": {"type": "string"},
        },
        "populations": {
            "type": "array", "items": {"type": "string"},
        },
        "entities": {
            "type": "array", "items": {"type": "string"},
        },
        "query_expansions": {
            "type": "array", "items": {"type": "string"},
        },
        "rewrite_reason": {"type": "string"},
    },
    "required": [
        "normalized_query", "intent", "sections", "conditions",
        "populations", "entities", "query_expansions", "rewrite_reason",
    ],
}


class QueryRewriteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QueryRewrite:
    normalized_query: str
    intent: str
    sections: tuple[str, ...]
    conditions: tuple[str, ...]
    populations: tuple[str, ...]
    entities: tuple[str, ...]
    query_expansions: tuple[str, ...]
    rewrite_reason: str

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.normalized_query, *self.query_expansions)))

    def to_dict(self) -> dict:
        return {
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "sections": list(self.sections),
            "conditions": list(self.conditions),
            "populations": list(self.populations),
            "entities": list(self.entities),
            "query_expansions": list(self.query_expansions),
            "rewrite_reason": self.rewrite_reason,
        }


class LLMQueryRewriter:
    def __init__(self, chat: ChatModel, *, max_expansions: int = 3):
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        self.chat = chat
        self.max_expansions = max_expansions

    @staticmethod
    def should_rewrite_before_recall(query: str, parsed: ParsedQuery) -> bool:
        return (
            not parsed.sections
            and not parsed.requires_exact_value
            and any(term in query for term in _COLLOQUIAL)
        )

    def rewrite(
        self,
        query: str,
        parsed: ParsedQuery,
        aliases: list[tuple[str, str]],
    ) -> QueryRewrite:
        response = self.chat.complete(
            "你是药品说明书检索查询规划器。只改写检索表达，不回答医学问题。"
            "必须保留原问题中的药品、疾病、人群、否定、数值、单位和等级；"
            "不得新增药品、剂量或临床事实。口语化问题可以扩展为多个标准说明书章节。",
            "原问题：" + query + "\n规则解析："
            + json.dumps(parsed.to_dict(), ensure_ascii=False)
            + "\n输出一个规范问题和最多三个检索扩展。"
            "intent 使用给定枚举；‘哪些/有哪些’等列表概览使用 overview，"
            "具体数值使用 exact_value，多药比较使用 comparison；"
            "sections 只能使用标准说明书章节名称。",
            _REWRITE_SCHEMA,
            max_output_tokens=512,
        )
        return self._validate(query, parsed, aliases, response)

    def _validate(
        self,
        query: str,
        parsed: ParsedQuery,
        aliases: list[tuple[str, str]],
        response: dict,
    ) -> QueryRewrite:
        normalized = response.get("normalized_query")
        intent = response.get("intent")
        reason = response.get("rewrite_reason")
        if not isinstance(normalized, str) or not normalized.strip() or len(normalized) > 300:
            raise QueryRewriteError("Invalid normalized query")
        if intent not in _ALLOWED_INTENTS:
            raise QueryRewriteError("Invalid rewrite intent")
        if not isinstance(reason, str):
            raise QueryRewriteError("Invalid rewrite reason")
        normalized = normalized.strip()

        sections = tuple(dict.fromkeys(
            normalize_section(value)
            for value in _string_list(response.get("sections"))
            if normalize_section(value) in _ALLOWED_SECTIONS
        ))
        if parsed.sections and not set(parsed.sections).issubset(sections):
            raise QueryRewriteError("Rewrite dropped an explicit section")
        if not sections:
            sections = parsed.sections or _INTENT_SECTIONS.get(intent, ())
        if parsed.broad and not sections:
            raise QueryRewriteError("Overview rewrite did not select a standard section")

        if parsed.comparison and intent != "comparison":
            raise QueryRewriteError("Rewrite changed comparison intent")
        if parsed.requires_exact_value and intent != "exact_value":
            raise QueryRewriteError("Rewrite changed exact-value intent")
        if parsed.broad and intent != "overview":
            raise QueryRewriteError("Rewrite changed overview intent")

        expansions = _string_list(response.get("query_expansions"))
        queries = list(dict.fromkeys(
            value.strip() for value in (normalized, *expansions)
            if value.strip() and value.strip() != query.strip()
        ))[:self.max_expansions + 1]
        if not queries:
            raise QueryRewriteError("Rewrite produced no new query")

        original_numbers = set(_NUMBER.findall(query))
        original_drugs = set(parsed.drug_ids)
        for value in queries:
            if not set(_NUMBER.findall(value)).issubset(original_numbers):
                raise QueryRewriteError("Rewrite introduced a new number")
            rewritten = parse_query(value, aliases)
            rewritten_drugs = set(rewritten.drug_ids)
            if rewritten_drugs - original_drugs:
                raise QueryRewriteError("Rewrite introduced a new drug")
        normalized_drugs = set(parse_query(normalized, aliases).drug_ids)
        if original_drugs and normalized_drugs != original_drugs:
            raise QueryRewriteError("Normalized query changed explicit drugs")
        if any(term in query for term in _NEGATIONS) and not any(
            term in normalized for term in _NEGATIONS
        ):
            raise QueryRewriteError("Rewrite dropped negation")
        if any(entity not in normalized for entity in parsed.entities):
            raise QueryRewriteError("Rewrite dropped an exact entity")

        return QueryRewrite(
            normalized,
            intent,
            sections,
            tuple(dict.fromkeys((*parsed.conditions, *_string_list(response.get("conditions"))))),
            tuple(dict.fromkeys((*parsed.populations, *_string_list(response.get("populations"))))),
            parsed.entities,
            tuple(query_value for query_value in queries if query_value != normalized),
            reason.strip(),
        )


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise QueryRewriteError("Rewrite list field is invalid")
    return tuple(item.strip() for item in value if item.strip())
