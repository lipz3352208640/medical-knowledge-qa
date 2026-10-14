"""Read chunk records and maintain external-index state in MySQL."""

from dataclasses import dataclass
from collections import Counter
import json
import math
import re
from typing import Any

from app.storage.mysql import _connect, parse_mysql_url


@dataclass(frozen=True, slots=True)
class ChunkFilters:
    document_ids: tuple[str, ...] = ()
    drug_ids: tuple[str, ...] = ()
    drug_names: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    populations: tuple[str, ...] = ()
    current_only: bool = True


@dataclass(slots=True)
class StoredChunk:
    id: str
    document_id: str
    section_id: str
    parent_id: str | None
    content: str
    retrieval_text: str
    content_type: str
    page_start: int
    page_end: int
    bbox: list[float] | None
    refs: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(slots=True)
class LexicalHit:
    chunk: StoredChunk
    score: float


_BM25_WORD = re.compile(r"[\u3400-\u9fff]+|[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?(?:mg|ml|g|%|级)?", re.IGNORECASE)


def _bm25_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _BM25_WORD.finditer(text.lower()):
        word = match.group()
        if "\u3400" <= word[0] <= "\u9fff":
            terms.extend(word)
            terms.extend(word[index:index + 2] for index in range(len(word) - 1))
        else:
            terms.append(word)
    return terms


def _decode_json(value: object) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def list_active_document_ids(
    mysql_url: str,
    requested_ids: list[str] | tuple[str, ...] | None = None,
    *, include_pending: bool = False,
) -> list[str]:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            statuses = "('ACTIVE','PENDING')" if include_pending else "('ACTIVE')"
            sql = f"SELECT id FROM document WHERE version_status IN {statuses}"
            params: list[object] = []
            if requested_ids:
                sql += " AND id IN (" + ",".join(["%s"] * len(requested_ids)) + ")"
                params.extend(requested_ids)
            sql += " ORDER BY id"
            cursor.execute(sql, tuple(params))
            return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def list_ready_vector_document_ids(
    mysql_url: str,
    *,
    collection_name: str,
    embedding_model: str,
) -> list[str]:
    """Only trust dense points backed by a READY index for this model/collection."""
    connection = _connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT d.id FROM document AS d
                   JOIN document_index_state AS state ON state.document_id=d.id
                   WHERE d.version_status='ACTIVE'
                     AND state.vector_status='READY'
                     AND state.collection_name=%s
                     AND state.embedding_model=%s
                   ORDER BY d.id""",
                (collection_name, embedding_model),
            )
            return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def load_vector_chunks(mysql_url: str, document_id: str, *, include_pending: bool = False) -> list[StoredChunk]:
    """Load evidence children and navigation summaries for separate vector channels."""
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            statuses = "('ACTIVE','PENDING')" if include_pending else "('ACTIVE')"
            cursor.execute(
                f"""SELECT c.id, c.document_id, c.section_id, c.parent_id,
                          c.content, c.retrieval_text, c.content_type,
                          c.page_start, c.page_end, c.bbox, c.refs_json,
                          c.metadata_json
                   FROM chunk AS c
                   JOIN document AS d ON d.id=c.document_id
                   WHERE c.document_id=%s AND c.is_parent=FALSE
                     AND d.version_status IN {statuses}
                   ORDER BY c.page_start, c.reading_order, c.id""",
                (document_id,),
            )
            return [_stored_chunk(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def load_parent_chunks(mysql_url: str, parent_ids: list[str] | tuple[str, ...]) -> list[StoredChunk]:
    """Load parent context in caller order, deduplicating repeated child parents."""
    unique_ids = list(dict.fromkeys(parent_ids))
    if not unique_ids:
        return []
    placeholders = ",".join(["%s"] * len(unique_ids))
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT id, document_id, section_id, parent_id,
                           content, retrieval_text, content_type,
                           page_start, page_end, bbox, refs_json, metadata_json
                    FROM chunk
                    WHERE is_parent=TRUE AND id IN ({placeholders})""",
                tuple(unique_ids),
            )
            by_id = {row[0]: _stored_chunk(row) for row in cursor.fetchall()}
            return [by_id[parent_id] for parent_id in unique_ids if parent_id in by_id]
    finally:
        connection.close()


def load_chunks_by_ids(mysql_url: str, chunk_ids: list[str] | tuple[str, ...]) -> list[StoredChunk]:
    """Load active original child or parent chunks in caller order."""
    unique_ids = list(dict.fromkeys(chunk_ids))
    if not unique_ids:
        return []
    placeholders = ",".join(["%s"] * len(unique_ids))
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT c.id, c.document_id, c.section_id, c.parent_id,
                           c.content, c.retrieval_text, c.content_type,
                           c.page_start, c.page_end, c.bbox, c.refs_json, c.metadata_json
                    FROM chunk AS c
                    JOIN document AS d ON d.id=c.document_id
                    WHERE d.version_status='ACTIVE'
                      AND c.content_type NOT IN ('section_summary','topic_summary')
                      AND c.id IN ({placeholders})""",
                tuple(unique_ids),
            )
            found = {chunk.id: chunk for chunk in map(_stored_chunk, cursor.fetchall())}
            return [found[chunk_id] for chunk_id in unique_ids if chunk_id in found]
    finally:
        connection.close()


def list_active_drug_aliases(mysql_url: str) -> list[tuple[str, str]]:
    """Return all known names of drugs with at least one active leaflet."""
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT dr.generic_name_normalized, dr.id
                   FROM drug AS dr
                   WHERE EXISTS (SELECT 1 FROM drug_product AS p
                     JOIN document AS d ON d.product_id=p.id
                     WHERE p.drug_id=dr.id AND d.version_status='ACTIVE')
                   UNION
                   SELECT da.normalized_name, da.drug_id
                   FROM drug_alias AS da
                   WHERE EXISTS (SELECT 1 FROM drug_product AS p
                     JOIN document AS d ON d.product_id=p.id
                     WHERE p.drug_id=da.drug_id AND d.version_status='ACTIVE')"""
            )
            return [(str(name), str(drug_id)) for name, drug_id in cursor.fetchall()]
    finally:
        connection.close()


def load_summary_sources(mysql_url: str, summary_id: str) -> list[StoredChunk]:
    """Resolve a summary hit to its original evidence children in source order."""
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT document_id, metadata_json FROM chunk
                   WHERE id=%s AND content_type IN ('section_summary','topic_summary')""",
                (summary_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Summary not found: {summary_id}")
            document_id, summary_metadata = row
            source_ids = list(dict.fromkeys(_decode_json(summary_metadata)["source_chunk_ids"]))
            if not source_ids:
                raise ValueError(f"Summary has no source chunks: {summary_id}")
            placeholders = ",".join(["%s"] * len(source_ids))
            cursor.execute(
                f"""SELECT id, document_id, section_id, parent_id,
                           content, retrieval_text, content_type,
                           page_start, page_end, bbox, refs_json, metadata_json
                    FROM chunk
                    WHERE is_parent=FALSE
                      AND document_id=%s
                      AND content_type NOT IN ('section_summary','topic_summary')
                      AND id IN ({placeholders})""",
                (document_id, *source_ids),
            )
            found = {record.id: record for record in map(_stored_chunk, cursor.fetchall())}
            if len(found) != len(source_ids):
                raise ValueError(f"Summary has missing source chunks: {summary_id}")
            return [found[source_id] for source_id in source_ids]
    finally:
        connection.close()


def _stored_chunk(row: tuple) -> StoredChunk:
    return StoredChunk(
        id=row[0],
        document_id=row[1],
        section_id=row[2],
        parent_id=row[3],
        content=row[4],
        retrieval_text=row[5],
        content_type=row[6],
        page_start=row[7],
        page_end=row[8],
        bbox=_decode_json(row[9]) if row[9] is not None else None,
        refs=_decode_json(row[10]),
        metadata=_decode_json(row[11]),
    )


def set_vector_index_state(
    mysql_url: str,
    document_id: str,
    status: str,
    *,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    point_count: int | None = None,
    error: str | None = None,
) -> None:
    if status not in {"PENDING", "INDEXING", "READY", "FAILED", "STALE"}:
        raise ValueError(f"Unsupported vector index status: {status}")
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            indexed_at = "NOW()" if status == "READY" else "indexed_at"
            cursor.execute(
                f"""UPDATE document_index_state SET
                       vector_status=%s,
                       embedding_model=COALESCE(%s, embedding_model),
                       collection_name=COALESCE(%s, collection_name),
                       point_count=COALESCE(%s, point_count),
                       last_error=%s,
                       indexed_at={indexed_at}
                   WHERE document_id=%s""",
                (
                    status,
                    embedding_model,
                    collection_name,
                    point_count,
                    error,
                    document_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Document has no chunk index state: {document_id}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def mark_inactive_vector_indexes_stale(mysql_url: str) -> int:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE document_index_state AS state
                   JOIN document AS d ON d.id=state.document_id
                   SET state.vector_status='STALE', state.point_count=0
                   WHERE d.version_status<>'ACTIVE' AND state.vector_status='READY'"""
            )
            changed = cursor.rowcount
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def lexical_search(
    mysql_url: str,
    query: str,
    filters: ChunkFilters | None = None,
    *,
    limit: int = 20,
) -> list[LexicalHit]:
    """Search original evidence; summary routing uses summary_search."""
    return _lexical_search(mysql_url, query, filters, limit=limit, summaries_only=False)

"""
bm25: 关键词相关性排序。
 按照分词规则对查询问题进行切分
 根据章节、药品产品名/ID 等信息过滤chunk
 根据bm25算法计算分词在文档中的分数
 返回超过阈值的文档
"""
def bm25_search(
    mysql_url: str,
    query: str,
    filters: ChunkFilters | None = None,
    *,
    limit: int = 20,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[LexicalHit]:
    """Rank the complete filtered evidence corpus with standard BM25.

    This reads all scoped child texts, favoring exact ranking over index speed.
    Summary routing continues to use the separate MySQL FULLTEXT index.
    """
    if limit <= 0 or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Invalid BM25 parameters")
    query_terms = set(_bm25_terms(query))
    if not query_terms:
        return []
    selected = filters or ChunkFilters()
    where = [
        "c.is_parent=FALSE",
        "c.content_type NOT IN ('section_summary','topic_summary')",
    ]
    params: list[object] = []
    if selected.current_only:
        where.append("d.version_status='ACTIVE'")

    def add_in(column: str, values: tuple[str, ...]) -> None:
        if values:
            where.append(f"{column} IN (" + ",".join(["%s"] * len(values)) + ")")
            params.extend(values)

    add_in("c.document_id", selected.document_ids)
    add_in("p.drug_id", selected.drug_ids)
    add_in("s.normalized_name", selected.sections)
    add_in("c.content_type", selected.content_types)
    if selected.populations:
        where.append("(" + " OR ".join(
            ["JSON_CONTAINS(c.metadata_json, JSON_QUOTE(%s), '$.populations')"]
            * len(selected.populations)
        ) + ")")
        params.extend(selected.populations)
    if selected.drug_names:
        placeholders = ",".join(["%s"] * len(selected.drug_names))
        where.append(
            "(dr.generic_name_normalized IN (" + placeholders
            + ") OR EXISTS (SELECT 1 FROM drug_alias AS da "
            "WHERE da.drug_id=dr.id AND da.normalized_name IN ("
            + placeholders + ")))"
        )
        params.extend(selected.drug_names)
        params.extend(selected.drug_names)
    sql = f"""SELECT c.id, c.document_id, c.section_id, c.parent_id,
                     c.content, c.retrieval_text, c.content_type,
                     c.page_start, c.page_end, c.bbox, c.refs_json, c.metadata_json
              FROM chunk AS c
              JOIN document AS d ON d.id=c.document_id
              JOIN drug_product AS p ON p.id=d.product_id
              JOIN drug AS dr ON dr.id=p.drug_id
              JOIN section AS s ON s.id=c.section_id
              WHERE {' AND '.join(where)}"""
    connection = _connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            chunks = [_stored_chunk(row) for row in cursor.fetchall()]
    finally:
        connection.close()
    if not chunks:
        return []
    documents = [Counter(_bm25_terms(chunk.retrieval_text)) for chunk in chunks]
    lengths = [sum(counts.values()) for counts in documents]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = Counter(
        term for counts in documents for term in query_terms if counts[term]
    )
    hits: list[LexicalHit] = []
    for chunk, counts, length in zip(chunks, documents, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log1p((len(chunks) - df + 0.5) / (df + 0.5))
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * (1 - b + b * length / average_length)
            )
        if score > 0:
            hits.append(LexicalHit(chunk, score))
    hits.sort(key=lambda hit: (-hit.score, hit.chunk.id))
    return hits[:limit]


def summary_search(
    mysql_url: str,
    query: str,
    filters: ChunkFilters | None = None,
    *,
    limit: int = 10,
) -> list[LexicalHit]:
    """Search the lower-weight summary routing channel."""
    return _lexical_search(mysql_url, query, filters, limit=limit, summaries_only=True)


def _lexical_search(
    mysql_url: str,
    query: str,
    filters: ChunkFilters | None,
    *,
    limit: int,
    summaries_only: bool,
) -> list[LexicalHit]:
    if not query.strip():
        return []
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = filters or ChunkFilters()
    channel_filter = (
        "c.content_type IN ('section_summary','topic_summary')"
        if summaries_only
        else "c.content_type NOT IN ('section_summary','topic_summary')"
    )
    where = ["c.is_parent=FALSE", channel_filter]
    params: list[object] = [query]
    if selected.current_only:
        where.append("d.version_status='ACTIVE'")

    def add_in(column: str, values: tuple[str, ...]) -> None:
        if values:
            where.append(f"{column} IN (" + ",".join(["%s"] * len(values)) + ")")
            params.extend(values)

    add_in("c.document_id", selected.document_ids)
    add_in("p.drug_id", selected.drug_ids)
    add_in("s.normalized_name", selected.sections)
    add_in("c.content_type", selected.content_types)
    if selected.populations:
        where.append("(" + " OR ".join(
            ["JSON_CONTAINS(c.metadata_json, JSON_QUOTE(%s), '$.populations')"]
            * len(selected.populations)
        ) + ")")
        params.extend(selected.populations)
    if selected.drug_names:
        placeholders = ",".join(["%s"] * len(selected.drug_names))
        where.append(
            "(dr.generic_name_normalized IN ("
            + placeholders
            + ") OR EXISTS (SELECT 1 FROM drug_alias AS da "
              "WHERE da.drug_id=dr.id AND da.normalized_name IN ("
            + placeholders
            + "))"
        )
        params.extend(selected.drug_names)
        params.extend(selected.drug_names)

    params.append(query)
    params.append(limit)
    sql = f"""SELECT c.id, c.document_id, c.section_id, c.parent_id,
                     c.content, c.retrieval_text, c.content_type,
                     c.page_start, c.page_end, c.bbox, c.refs_json,
                     c.metadata_json,
                     MATCH(c.retrieval_text) AGAINST (%s IN NATURAL LANGUAGE MODE) AS lexical_score
              FROM chunk AS c
              JOIN document AS d ON d.id=c.document_id
              JOIN drug_product AS p ON p.id=d.product_id
              JOIN drug AS dr ON dr.id=p.drug_id
              JOIN section AS s ON s.id=c.section_id
              WHERE {' AND '.join(where)}
                AND MATCH(c.retrieval_text) AGAINST (%s IN NATURAL LANGUAGE MODE)
              ORDER BY lexical_score DESC, c.id
              LIMIT %s"""

    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return [LexicalHit(_stored_chunk(row[:12]), float(row[12])) for row in cursor.fetchall()]
    finally:
        connection.close()
