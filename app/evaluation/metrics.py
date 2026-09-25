"""Deterministic retrieval and answer checks for human-annotated gold cases."""

from collections import Counter
import math


def ranking_metrics(
    ranked_ids: list[str], relevant_ids: list[str], *, k: int = 5,
) -> dict[str, float]:
    if k <= 0 or not relevant_ids:
        raise ValueError("k and relevant_ids must be nonempty")
    relevant = set(relevant_ids)
    first = next((rank for rank, chunk_id in enumerate(ranked_ids, 1) if chunk_id in relevant), None)
    gains = [1 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked_ids[:k], 1) if chunk_id in relevant]
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        "recall_at_1": len(relevant.intersection(ranked_ids[:1])) / len(relevant),
        "recall_at_3": len(relevant.intersection(ranked_ids[:3])) / len(relevant),
        "recall_at_5": len(relevant.intersection(ranked_ids[:5])) / len(relevant),
        "recall_at_10": len(relevant.intersection(ranked_ids[:10])) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_5": sum(gains) / ideal if ideal else 0.0,
    }


def average_metrics(rows: list[dict], keys: list[str]) -> dict[str, float | None]:
    return {
        key: round(sum(row[key] for row in rows if row.get(key) is not None)
                   / sum(row.get(key) is not None for row in rows), 4)
        if any(row.get(key) is not None for row in rows) else None
        for key in keys
    }


def coverage_counts(cases: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {}
    for case in cases:
        counts.setdefault(case["drug"], Counter())[case["category"]] += 1
    return {name: dict(values) for name, values in counts.items()}
