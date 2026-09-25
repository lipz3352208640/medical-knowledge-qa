"""Show which filter or ranking stage loses a golden retrieval chunk."""

import argparse
import json
import os
from pathlib import Path

from app.evaluation.runner import _drug_id
from app.retrieval.query import parse_query
from app.storage.retrieval import (
    ChunkFilters, bm25_search, list_active_document_ids, list_active_drug_aliases,
    load_chunks_by_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a golden retrieval case")
    parser.add_argument("case_id")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/golden.json"))
    args = parser.parse_args()
    mysql_url = os.getenv("MYSQL_URL")
    if not mysql_url:
        parser.error("MYSQL_URL is required")
    gold = json.loads(args.dataset.read_text(encoding="utf-8"))
    case = next((item for item in gold["retrieval_cases"] if item["id"] == args.case_id), None)
    if case is None:
        parser.error(f"Unknown case: {args.case_id}")
    aliases = list_active_drug_aliases(mysql_url)
    parsed = parse_query(case["query"], aliases)
    drug_id = _drug_id(case["drug"], aliases)
    documents = tuple(list_active_document_ids(mysql_url))
    strict = ChunkFilters(
        document_ids=documents, drug_ids=(drug_id,), sections=parsed.sections,
        populations=parsed.populations,
    )
    broad = ChunkFilters(document_ids=documents, drug_ids=(drug_id,))
    results = {}
    for label, filters in (("strict", strict), ("no_population", ChunkFilters(
        document_ids=documents, drug_ids=(drug_id,), sections=parsed.sections,
    )), ("drug_only", broad)):
        hits = bm25_search(mysql_url, case["query"], filters, limit=200)
        ids = [hit.chunk.id for hit in hits]
        results[label] = {
            "total_positive": len(ids),
            "gold_ranks": {chunk_id: ids.index(chunk_id) + 1 if chunk_id in ids else None
                           for chunk_id in case["relevant_chunk_ids"]},
            "top_ids": ids[:5],
        }
    sources = load_chunks_by_ids(mysql_url, case["relevant_chunk_ids"])
    print(json.dumps({
        "case": case["id"], "parsed": parsed.to_dict(), "bm25": results,
        "gold_metadata": {source.id: {
            "section": source.metadata.get("section"),
            "populations": source.metadata.get("populations"),
            "content_type": source.content_type,
        } for source in sources},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
