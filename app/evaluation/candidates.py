"""List active source chunks to support manual golden-case annotation."""

import argparse
import json
from app.config import setting

from app.storage.mysql import _connect, parse_mysql_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect active leaflet chunks for gold labels")
    parser.add_argument("--section", default=None)
    parser.add_argument("--contains", default=None)
    parser.add_argument("--limit", type=int, default=3, help="Rows per drug and section")
    parser.add_argument("--summary", action="store_true", help="Show active documents and section coverage")
    args = parser.parse_args()
    mysql_url = setting("MYSQL_URL")
    if not mysql_url:
        parser.error("mysql_url in config.yaml is required")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    connection = _connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            if args.summary:
                cursor.execute(
                    """SELECT d.id,dr.generic_name_raw,s.normalized_name,
                              COUNT(DISTINCT CASE WHEN c.content_type='table_row' THEN c.id END)
                       FROM document AS d
                       JOIN drug_product AS p ON p.id=d.product_id
                       JOIN drug AS dr ON dr.id=p.drug_id
                       JOIN section AS s ON s.document_id=d.id
                       LEFT JOIN chunk AS c ON c.section_id=s.id AND c.is_parent=FALSE
                       WHERE d.version_status='ACTIVE'
                       GROUP BY d.id,dr.generic_name_raw,s.normalized_name
                       ORDER BY dr.generic_name_raw,s.normalized_name"""
                )
                grouped: dict[str, dict] = {}
                for doc_id, drug, section, table_rows in cursor.fetchall():
                    entry = grouped.setdefault(doc_id, {"document_id": doc_id, "drug": drug, "sections": [], "table_rows": 0})
                    entry["sections"].append(section)
                    entry["table_rows"] += table_rows
                print(json.dumps(list(grouped.values()), ensure_ascii=False, indent=2))
                return
            where = [
                "d.version_status='ACTIVE'", "c.is_parent=FALSE",
                "c.content_type NOT IN ('section_summary','topic_summary')",
            ]
            params: list[str] = []
            if args.section:
                where.append("s.normalized_name=%s")
                params.append(args.section)
            if args.contains:
                where.append("(c.content LIKE %s OR c.retrieval_text LIKE %s)")
                params.extend((f"%{args.contains}%", f"%{args.contains}%"))
            cursor.execute(
                """SELECT dr.generic_name_raw,s.normalized_name,c.id,c.content_type,
                          c.page_start,c.content,c.retrieval_text
                   FROM chunk AS c JOIN document AS d ON d.id=c.document_id
                   JOIN drug_product AS p ON p.id=d.product_id
                   JOIN drug AS dr ON dr.id=p.drug_id
                   JOIN section AS s ON s.id=c.section_id
                   WHERE """ + " AND ".join(where)
                + " ORDER BY dr.generic_name_raw,s.normalized_name,c.page_start,c.reading_order",
                tuple(params),
            )
            seen: dict[tuple[str, str], int] = {}
            rows = []
            for drug, section, chunk_id, content_type, page, content, retrieval in cursor.fetchall():
                key = (drug, section)
                if seen.get(key, 0) >= args.limit:
                    continue
                seen[key] = seen.get(key, 0) + 1
                rows.append({
                    "drug": drug, "section": section, "chunk_id": chunk_id,
                    "content_type": content_type, "page": page,
                    "content": content[:240].replace("\n", " "),
                    "retrieval_text": retrieval[:240].replace("\n", " "),
                })
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
