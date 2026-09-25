"""Conservative normalization: never replace the source value."""

import re
import unicodedata

from app.models import ValuePair


SECTION_ALIASES = {
    "药品名称": "药品名称", "通用名称": "药品名称", "商品名称": "药品名称",
    "成份": "成分", "成分": "成分", "性状": "性状", "适应症": "适应症",
    "适应证": "适应症", "规格": "规格", "用法用量": "用法用量",
    "不良反应": "不良反应", "禁忌": "禁忌", "注意事项": "注意事项",
    "孕妇及哺乳期妇女用药": "孕妇及哺乳期妇女用药",
    "儿童用药": "儿童用药", "老年用药": "老年用药",
    "药物相互作用": "药物相互作用", "药理毒理": "药理毒理",
    "药代动力学": "药代动力学", "临床试验": "临床试验",
    "贮藏": "贮藏", "有效期": "有效期", "批准文号": "批准文号",
    "生产企业": "生产企业", "包装": "包装", "执行标准": "执行标准",
}


def clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def normalize_section(raw: str) -> str:
    key = re.sub(r"^[【\[（(\s]+|[】\]）)\s:：]+$", "", clean_space(raw))
    return SECTION_ALIASES.get(key, key)


def normalize_drug_name(raw: str) -> str:
    return clean_space(raw).replace(" ", "")


def normalize_drug_alias(raw: str) -> str:
    value = clean_space(raw).replace("®", "").replace("™", "").strip(" /／、")
    return value.casefold() if value.isascii() else value.replace(" ", "")


VALUE_PATTERN = re.compile(
    r"(?<!\w)(?:[≥≤<>±]?[\d]+(?:\.\d+)?(?:\s*[-~～至]\s*[\d]+(?:\.\d+)?)?)"
    r"\s*(?:mg|g|μg|ug|mL|ml|L|mmol|片|粒|次|天|周|月|年|级|%|％)?",
    re.IGNORECASE,
)


def extract_values(text: str) -> list[ValuePair]:
    out: list[ValuePair] = []
    for match in VALUE_PATTERN.finditer(text):
        raw = match.group().strip()
        if raw:
            normalized = clean_space(raw).replace("μg", "ug").replace("ml", "mL")
            normalized = re.sub(r"(?<=\d)\s+(?=(?:mg|g|ug|mL|L|mmol|%|级)\b)", "", normalized, flags=re.I)
            out.append(ValuePair(raw, normalized))
    return out
