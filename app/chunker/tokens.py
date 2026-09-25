"""Shared deterministic token estimate for chunk sizing and summaries."""

import re


_TOKEN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[./:%+_-][A-Za-z0-9]+)*|[^\s]")


def estimated_token_count(text: str) -> int:
    """Count approximate tokens without depending on an embedding tokenizer."""
    return sum(1 for _ in _TOKEN.finditer(text))
