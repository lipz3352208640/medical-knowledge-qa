"""Deterministic medical query parsing; drug aliases come from MySQL."""

from dataclasses import asdict, dataclass
import re

from app.parser.normalize import normalize_section
from app.rules.runtime import REQUEST_RULE_SNAPSHOT


_SECTION_RULES = (
    ("用法用量", ("剂量", "用量", "给药", "服用", "漏服", "减量")),
    ("不良反应", ("不良反应", "副作用", "发生率")),
    ("禁忌", ("禁忌", "禁用")),
    ("注意事项", ("注意事项", "警告", "注意什么")),
    ("适应症", ("适应症", "适应证", "治疗什么")),
    ("药物相互作用", ("相互作用", "合用", "联用")),
)
_POPULATIONS = {
    "elderly": ("老年", "高龄"),
    "pediatric": ("儿童", "小儿", "婴儿", "新生儿"),
    "pregnancy": ("孕妇", "妊娠", "怀孕"),
    "lactation": ("哺乳", "乳汁"),
    "renal_impairment": ("肾功能不全", "肾损害", "肾功能损害", "肾功能异常"),
    "hepatic_impairment": ("肝功能不全", "肝损害", "肝功能损害", "肝功能异常"),
}
_CONDITIONS = ("腹泻", "皮疹", "高血压", "低血压", "肾损伤", "肝损伤", "恶心", "呕吐")
_EXACT_ENTITY = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*(?:mg|μg|ug|g|ml|mL|毫克|微克|克|毫升|%|％|级|期|天|周|月|年)(?![A-Za-z])",
    re.IGNORECASE,
)
_DOSAGE_SUFFIXES = (
    "肠溶胶囊",
    "缓释胶囊",
    "软胶囊",
    "肠溶片",
    "缓释片",
    "分散片",
    "胶囊",
    "注射液",
    "口服液",
    "颗粒",
    "片剂",
    "片",
)


def drug_alias_map(aliases: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Accept common form-free drug names only when they resolve uniquely."""
    names: dict[str, set[str]] = {}
    for name, drug_id in aliases:
        key = name.casefold().strip()
        if not key:
            continue
        names.setdefault(key, set()).add(drug_id)
        for suffix in _DOSAGE_SUFFIXES:
            if key.endswith(suffix) and len(key) - len(suffix) >= 2:
                names.setdefault(key[: -len(suffix)], set()).add(drug_id)
                break
        for suffix in (" tablets", " capsules"):
            if key.endswith(suffix) and len(key) - len(suffix) >= 3:
                names.setdefault(key[: -len(suffix)], set()).add(drug_id)
                break
    return names


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    drugs: tuple[str, ...]
    intent: str
    sections: tuple[str, ...]
    populations: tuple[str, ...]
    conditions: tuple[str, ...]
    entities: tuple[str, ...]
    comparison: bool
    requires_exact_value: bool
    drug_ids: tuple[str, ...]
    broad: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("drug_ids")
        result.pop("broad")
        return result


"""
查询意图：
  parsed.sections：限制 BM25、向量检索的章节范围；
  parsed.broad：触发摘要索引路由；
  parsed.comparison：进入多药品、多维度检索；
  parsed.requires_exact_value：控制上下文扩展；
  parsed.conditions、parsed.entities：触发压缩阶段的 LLM 筛选；
  parsed.populations：目前主要被识别，但在线检索中被主动设置为空，避免漏掉未标注人群的证据。
"""


def parse_query(query: str, aliases: list[tuple[str, str]]) -> ParsedQuery:
    if not query.strip():
        raise ValueError("query must not be blank")
    lower = query.lower()
    names = drug_alias_map(aliases)
    mentions: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for name in sorted(names, key=lambda value: (-len(value), value)):
        for match in re.finditer(re.escape(name), lower):
            start, end = match.span()
            if name.isascii() and (
                (start > 0 and lower[start - 1].isalnum())
                or (end < len(lower) and lower[end].isalnum())
            ):
                continue
            if any(start < right and end > left for left, right in occupied):
                continue
            if len(names[name]) != 1:
                raise ValueError(f"Ambiguous drug alias in query: {query[start:end]}")
            occupied.append((start, end))
            mentions.append((start, end, query[start:end], next(iter(names[name]))))
    mentions.sort()
    drugs: list[str] = []
    drug_ids: list[str] = []
    for _, _, name, drug_id in mentions:
        if drug_id not in drug_ids:
            drugs.append(name)
            drug_ids.append(drug_id)
    # 明确章节
    explicit_sections = [
        normalize_section(alias)
        for alias in (
            "用法用量",
            "不良反应",
            "禁忌",
            "注意事项",
            "适应症",
            "适应证",
            "药物相互作用",
            "儿童用药",
            "老年用药",
            "孕妇及哺乳期妇女用药",
        )
        if alias in query
    ]
    # 推断章节
    inferred_sections = [
        section
        for section, terms in _SECTION_RULES
        if any(term in query for term in terms)
    ]
    # 人群
    populations = tuple(
        key
        for key, terms in _POPULATIONS.items()
        if any(term in query for term in terms)
    )
    # 人群章节
    population_sections = {
        "elderly": "老年用药",
        "pediatric": "儿童用药",
        "pregnancy": "孕妇及哺乳期妇女用药",
        "lactation": "孕妇及哺乳期妇女用药",
    }
    sections = tuple(
        dict.fromkeys(
            explicit_sections
            + inferred_sections
            + [
                population_sections[key]
                for key in populations
                if key in population_sections
            ]
        )
    )
    conditions = tuple(condition for condition in _CONDITIONS if condition in query)
    entities = tuple(
        dict.fromkeys(match.group().strip() for match in _EXACT_ENTITY.finditer(query))
    )
    exact = bool(entities) or any(
        term in query
        for term in ("多少", "几", "具体数值", "发生率", "百分比", "最大剂量")
    )
    broad = (
        any(
            term in query
            for term in ("概括", "总结", "有哪些", "哪些", "总体", "全面", "主要")
        )
        and not exact
    )
    comparison = len(drug_ids) > 1 and any(
        term in query
        for term in (
            "比较",
            "对比",
            "区别",
            "差异",
            "相比",
            "分别",
            "哪个",
            "与",
            "和",
            "及",
        )
    )
    if exact:
        intent = "exact_value"
    elif comparison:
        intent = "comparison"
    elif broad:
        intent = "overview"
    elif sections:
        intent = sections[0]
    else:
        intent = "fact"
    published = REQUEST_RULE_SNAPSHOT.get()
    if published:
        matches = []
        for rule in published.get("rules", []):
            if not rule.get("enabled") or any(term in lower for term in rule.get("exclusions", [])):
                continue
            phrases = [term.casefold() for term in rule.get("phrases", []) if term]
            kind = rule.get("match_type")
            hit = any(lower == term for term in phrases) if kind == "exact" else (
                any(term in lower.split() for term in phrases) if kind == "token" else
                any(term in lower for term in phrases)
            )
            if hit:
                matches.append(rule)
        matches.sort(key=lambda rule: (-rule.get("priority", 0), rule.get("id", "")))
        if matches:
            hints = tuple(normalize_section(value) for value in matches[0].get("section_hints", []))
            sections = tuple(dict.fromkeys(sections + hints))
            if not exact and not comparison:
                intent = matches[0]["intent_code"]
                if intent in {"overview", "broad"}:
                    broad = True
    return ParsedQuery(
        tuple(drugs),
        intent,
        sections,
        populations,
        conditions,
        entities,
        comparison,
        exact,
        tuple(drug_ids),
        broad,
    )
