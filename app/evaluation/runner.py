"""Run parsing, staged retrieval and grounded-answer evaluations."""

import argparse
from datetime import datetime, timezone
import json
from app.config import setting
from pathlib import Path
import time

from app.evaluation.metrics import average_metrics, coverage_counts, ranking_metrics
from app.retrieval.pipeline import RetrievalEngine, reciprocal_rank_fusion
from app.retrieval.query import drug_alias_map, parse_query
from app.runtime import build_runtime
from app.storage.mysql import _connect, parse_mysql_url
from app.storage.retrieval import (
    ChunkFilters, bm25_search, list_active_document_ids, list_active_drug_aliases,
    list_ready_vector_document_ids, load_chunks_by_ids,
)


_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5")


def _drug_id(name: str, aliases: list[tuple[str, str]]) -> str:
    matches = drug_alias_map(aliases).get(name.casefold(), set())
    if len(matches) != 1:
        raise ValueError(f"Gold drug is unknown or ambiguous: {name}")
    return next(iter(matches))


def evaluate_parsing(mysql_url: str, cases: list[dict]) -> dict:
    connection = _connect(parse_mysql_url(mysql_url))
    rows = []
    try:
        with connection.cursor() as cursor:
            for case in cases:
                document_id = case["document_id"]
                cursor.execute("SELECT normalized_name FROM section WHERE document_id=%s", (document_id,))
                actual_sections = {row[0] for row in cursor.fetchall()}
                required = set(case["required_sections"])
                cursor.execute(
                    """SELECT c.id, c.content_type, c.content, c.retrieval_text,
                              c.page_start, c.page_end, c.refs_json, s.normalized_name
                       FROM chunk AS c JOIN section AS s ON s.id=c.section_id
                       WHERE c.document_id=%s AND c.is_parent=FALSE
                         AND c.content_type NOT IN ('section_summary','topic_summary')""",
                    (document_id,),
                )
                chunks = {}
                for chunk_id, kind, content, retrieval, start, end, refs, section in cursor.fetchall():
                    if isinstance(refs, str):
                        refs = json.loads(refs)
                    chunks[chunk_id] = (kind, content, retrieval, start, end, refs, section)
                table_count = sum(record[0] == "table_row" for record in chunks.values())
                traced = sum(
                    bool(record[5]) and record[3] >= 1 and record[3] <= record[4]
                    and all(ref.get("page", 0) >= 1 and ref.get("bbox") for ref in record[5])
                    for record in chunks.values()
                )
                samples = case.get("text_samples", [])
                retained = sum(
                    sample["chunk_id"] in chunks
                    and sample["text"] in chunks[sample["chunk_id"]][1]
                    for sample in samples
                )
                table_samples = case.get("table_samples", [])
                table_retained = sum(
                    sample["chunk_id"] in chunks
                    and chunks[sample["chunk_id"]][0] == "table_row"
                    and sample["text"] in chunks[sample["chunk_id"]][2]
                    for sample in table_samples
                )
                rows.append({
                    "document_id": document_id, "drug": case["drug"],
                    "required_section_coverage": len(required & actual_sections) / len(required) if required else 1.0,
                    "missing_sections": sorted(required - actual_sections),
                    "table_rows": table_count,
                    "table_rows_ok": table_count >= case.get("min_table_rows", 0),
                    "traceability_rate": traced / len(chunks) if chunks else 0.0,
                    "text_sample_retention": retained / len(samples) if samples else None,
                    "table_sample_retention": table_retained / len(table_samples) if table_samples else None,
                    "unclassified_chunk_rate": sum(record[6] == "未分类" for record in chunks.values()) / len(chunks) if chunks else None,
                    "chunk_count": len(chunks),
                })
    finally:
        connection.close()
    return {"cases": rows, "summary": average_metrics(
        rows, ["required_section_coverage", "traceability_rate", "text_sample_retention",
               "table_sample_retention", "table_rows_ok", "unclassified_chunk_rate"],
    )}


def _stage_metrics(chunks: list, case: dict, expected_drug_id: str) -> dict:
    ids = [chunk.id for chunk in chunks]
    result = ranking_metrics(ids, case["relevant_chunk_ids"])
    top = chunks[:5]
    result.update({
        "correct_drug_rate": sum(chunk.metadata.get("drug_id") == expected_drug_id for chunk in top) / len(top) if top else None,
        "correct_section_at_5": float(any(chunk.metadata.get("section") == case["section"] for chunk in top)),
        "cross_drug_contamination_rate": sum(chunk.metadata.get("drug_id") != expected_drug_id for chunk in top) / len(top) if top else None,
        "table_row_hit_at_5": float(any(
            chunk.id in case["relevant_chunk_ids"] and chunk.content_type == "table_row" for chunk in top
        )) if case["category"] == "table_row" else None,
        "top_ids": ids[:10],
    })
    return result


def evaluate_retrieval(engine: RetrievalEngine, cases: list[dict], comparison_cases: list[dict]) -> dict:
    aliases = list_active_drug_aliases(engine.mysql_url)
    active = tuple(list_active_document_ids(engine.mysql_url))
    ready = tuple(list_ready_vector_document_ids(
        engine.mysql_url, collection_name=engine.vector_store.collection_name,
        embedding_model=engine.embedder.model,
    ))
    stage_rows: dict[str, list[dict]] = {name: [] for name in ("bm25", "vector", "hybrid_rrf", "reranker")}
    case_rows = []
    for case in cases:
        started = time.perf_counter()
        expected_drug_id = _drug_id(case["drug"], aliases)
        parsed = parse_query(case["query"], aliases)
        if parsed.drug_ids != (expected_drug_id,):
            raise ValueError(f"Drug resolution failed for gold case {case['id']}: {parsed.drug_ids}")
        known = {chunk.id for chunk in load_chunks_by_ids(engine.mysql_url, case["relevant_chunk_ids"])}
        if known != set(case["relevant_chunk_ids"]):
            raise ValueError(f"Stale gold chunk IDs in {case['id']}")
        lexical = bm25_search(engine.mysql_url, case["query"], ChunkFilters(
            document_ids=active, drug_ids=(expected_drug_id,), sections=parsed.sections,
            populations=(),
        ), limit=engine.recall_limit)
        dense = []
        if ready:
            vectors = engine.embedder.embed([case["query"]])
            dense = engine.vector_store.search(vectors[0], ChunkFilters(
                document_ids=ready, drug_ids=(expected_drug_id,), sections=parsed.sections,
                populations=(),
            ), limit=engine.recall_limit)
        fused = reciprocal_rank_fusion(lexical, dense, k=engine.rrf_k)
        scores = engine.reranker.score(case["query"], [item.chunk for item in fused]) if fused else []
        for item, score in zip(fused, scores, strict=True):
            item.rerank_score = score
        reranked = sorted(fused, key=lambda item: (-item.rerank_score, -item.rrf_score, item.chunk.id))
        candidates = {
            "bm25": [hit.chunk for hit in lexical],
            "vector": [hit.chunk for hit in dense],
            "hybrid_rrf": [item.chunk for item in fused],
            "reranker": [item.chunk for item in reranked],
        }
        row = {"id": case["id"], "category": case["category"], "query": case["query"],
               "elapsed_seconds": round(time.perf_counter() - started, 2), "stages": {}}
        for stage, chunks in candidates.items():
            metrics = _stage_metrics(chunks, case, expected_drug_id)
            row["stages"][stage] = metrics
            stage_rows[stage].append(metrics)
        case_rows.append(row)
    comparison_rows = []
    for case in comparison_cases:
        result = engine.retrieve(case["query"])
        by_drug_id = {group.drug_id: group for group in result.groups}
        for drug, relevant in case["expected_by_drug"].items():
            drug_id = _drug_id(drug, aliases)
            group = by_drug_id.get(drug_id)
            chunks = [item.chunk for item in group.evidence] if group else []
            comparison_rows.append({
                "id": case["id"], "drug": drug,
                "recall_at_5": ranking_metrics([chunk.id for chunk in chunks], relevant)["recall_at_5"],
                "cross_drug_contamination_rate": sum(
                    chunk.metadata.get("drug_id") != drug_id for chunk in chunks
                ) / len(chunks) if chunks else None,
                "top_ids": [chunk.id for chunk in chunks[:5]],
            })
    keys = list(_METRICS) + [
        "correct_drug_rate", "correct_section_at_5", "cross_drug_contamination_rate", "table_row_hit_at_5",
    ]
    return {
        "cases": case_rows,
        "comparison_cases": comparison_rows,
        "summary": {stage: average_metrics(rows, keys) for stage, rows in stage_rows.items()},
        "comparison_summary": average_metrics(comparison_rows, ["recall_at_5", "cross_drug_contamination_rate"]),
    }


def evaluate_answers(service, mysql_url: str, cases: list[dict], *, limit: int = 0) -> dict:
    aliases = list_active_drug_aliases(mysql_url)
    selected = cases[:limit] if limit else cases
    rows = []
    for case in selected:
        started = time.perf_counter()
        answer = service.answer(case["query"])
        ids = [item["chunk_id"] for item in answer.evidence]
        chunks = {chunk.id: chunk for chunk in load_chunks_by_ids(mysql_url, ids)}
        traced = all(
            citation["chunk_id"] in chunks and citation["quote"] in chunks[citation["chunk_id"]].content
            for citation in answer.evidence
        )
        expected_drug = _drug_id(case["drug"], aliases) if case.get("drug") else None
        ownership = all(
            chunks[citation["chunk_id"]].metadata.get("drug_id") == expected_drug
            for citation in answer.evidence if citation["chunk_id"] in chunks
        ) if expected_drug else None
        expected_terms = case.get("expected_terms", [])
        normalized_answer = "".join(answer.answer.split()).casefold()
        terms_present = all("".join(term.split()).casefold() in normalized_answer for term in expected_terms) if case["answerable"] else None
        expected_cited = bool(set(ids) & set(case.get("expected_chunk_ids", []))) if case["answerable"] else None
        abstained = (
            answer.confidence == "low"
            and any(term in answer.answer for term in ("无法", "未检索到", "未提及", "证据不足", "不能确定", "资料不足"))
        ) if not case["answerable"] else None
        rows.append({
            "id": case["id"], "query": case["query"], "answerable": case["answerable"],
            "answer": answer.answer, "confidence": answer.confidence,
            "warnings": answer.warnings, "evidence": answer.evidence,
            "citation_traceability": float(traced and bool(answer.evidence)) if case["answerable"] else float(traced),
            "drug_ownership": float(ownership) if ownership is not None else None,
            "expected_terms_present": float(terms_present) if terms_present is not None else None,
            "expected_chunk_cited": float(expected_cited) if expected_cited is not None else None,
            "abstention_correct": float(abstained) if abstained is not None else None,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "manual_review": {"faithfulness": None, "answer_relevance": None, "citation_supports_claim": None},
        })
    return {
        "cases": rows,
        "summary": average_metrics(rows, [
            "citation_traceability", "drug_ownership", "expected_terms_present",
            "expected_chunk_cited", "abstention_correct", "elapsed_seconds",
        ]),
        "manual_review_required": "逐条判断回答是否由引用支持、是否切题；自动检查只验证字符串与来源归属。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the five-leaflet QA pipeline")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/golden.json"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/latest.json"))
    parser.add_argument("--mode", choices=("parse", "retrieval", "answers", "all"), default="all")
    parser.add_argument("--answer-limit", type=int, default=0, help="0 runs every answer case")
    args = parser.parse_args()
    if args.answer_limit < 0:
        parser.error("--answer-limit cannot be negative")
    mysql_url = setting("MYSQL_URL")
    if not mysql_url:
        parser.error("mysql_url in config.yaml is required")
    gold = json.loads(args.dataset.read_text(encoding="utf-8"))
    if gold.get("version") != 1:
        parser.error("Unsupported golden dataset version")
    report = {
        "dataset": str(args.dataset), "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "parse_documents": len(gold["parse_cases"]),
            "retrieval_cases": len(gold["retrieval_cases"]),
            "comparison_cases": len(gold["comparison_cases"]),
            "answer_cases": len(gold["answer_cases"]),
            "retrieval_by_drug": coverage_counts(gold["retrieval_cases"]),
            "spec_minimum_per_drug": {"exact": 10, "population": 5, "dose": 5, "adverse": 5, "interaction": 3},
        },
    }
    if args.mode in {"parse", "all"}:
        report["parse"] = evaluate_parsing(mysql_url, gold["parse_cases"])
    if args.mode in {"retrieval", "answers", "all"}:
        service, engine = build_runtime(mysql_url)
        if args.mode in {"retrieval", "all"}:
            report["retrieval"] = evaluate_retrieval(
                engine, gold["retrieval_cases"], gold["comparison_cases"],
            )
        if args.mode in {"answers", "all"}:
            report["answers"] = evaluate_answers(
                service, mysql_url, gold["answer_cases"], limit=args.answer_limit,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output), "coverage": report["coverage"],
        "parse": report.get("parse", {}).get("summary"),
        "retrieval": report.get("retrieval", {}).get("summary"),
        "comparison": report.get("retrieval", {}).get("comparison_summary"),
        "answers": report.get("answers", {}).get("summary"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
