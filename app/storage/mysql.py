"""Transactional, idempotent MySQL persistence."""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import parse_qs, unquote, urlsplit

from app.chunker import CHUNKER_VERSION, build_chunks
from app.models import ParseResult
from app.parser.layout import stable_id
from app.report.quality import make_report


@dataclass(frozen=True, slots=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"


def parse_mysql_url(mysql_url: str) -> MySQLConfig:
    parsed = urlsplit(mysql_url)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError("MySQL URL must start with mysql:// or mysql+pymysql://")
    database = unquote(parsed.path.lstrip("/"))
    if not database or not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError(
            "MySQL database name must contain only letters, digits, and underscores"
        )
    query = parse_qs(parsed.query)
    return MySQLConfig(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=unquote(parsed.username or "root"),
        password=unquote(parsed.password or ""),
        database=database,
        charset=query.get("charset", ["utf8mb4"])[0],
    )


def _connect(config: MySQLConfig, with_database: bool = True):
    import pymysql

    options = {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
        "charset": config.charset,
        "autocommit": False,
    }
    if with_database:
        options["database"] = config.database
    return pymysql.connect(**options)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _migration_statements(path: Path) -> list[str]:
    return [
        statement.strip()
        for statement in path.read_text(encoding="utf-8").split(";")
        if statement.strip()
    ]


def _product_identity(result: ParseResult) -> tuple[str, str, str]:
    if result.approval_numbers:
        source = "APPROVAL_NUMBER"
        payload = "|".join(sorted(result.approval_numbers))
    else:
        source = "GENERIC_BRAND"
        brands = "|".join(sorted(name.normalized for name in result.brand_names))
        payload = f"{result.drug_name.normalized}|{brands}"
    identity_hash = hashlib.sha256(f"{source}:{payload}".encode("utf-8")).hexdigest()
    return stable_id("product", identity_hash), identity_hash, source


def _store_aliases(
    cursor: object, result: ParseResult, drug_id: str, product_id: str, document_id: str
) -> None:
    names = [("GENERIC", result.drug_name)]
    names.extend(("BRAND", name) for name in result.brand_names)
    if result.english_name:
        names.append(("ENGLISH", result.english_name))
    for alias_type, name in names:
        alias_id = stable_id("alias", drug_id, product_id, alias_type, name.normalized)
        cursor.execute(
            """INSERT IGNORE INTO drug_alias
               (id, drug_id, product_id, source_document_id, alias_type, raw_name, normalized_name)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (
                alias_id,
                drug_id,
                product_id,
                document_id,
                alias_type,
                name.raw,
                name.normalized,
            ),
        )


def _product_metadata(result: ParseResult) -> tuple[str | None, str | None, str | None]:
    """Return denormalized product fields used by the product table."""
    brand_raw = " / ".join(name.raw for name in result.brand_names) or None
    brand_normalized = " / ".join(name.normalized for name in result.brand_names) or None
    english_raw = result.english_name.raw if result.english_name else None
    return brand_raw, brand_normalized, english_raw


def _find_product_by_identity(
    cursor: object,
    result: ParseResult,
    identity_hash: str,
) -> tuple[str, str] | None:
    """Find one product, preferring approval numbers over the identity hash."""
    if result.approval_numbers:
        placeholders = ",".join(["%s"] * len(result.approval_numbers))
        cursor.execute(
            f"""SELECT DISTINCT p.id, p.drug_id
                FROM drug_product AS p
                JOIN product_approval_number AS a ON a.product_id=p.id
                WHERE a.approval_number IN ({placeholders})
                FOR UPDATE""",
            tuple(result.approval_numbers),
        )
        matches = list(cursor.fetchall())
        if len(matches) > 1:
            raise ValueError("Approval number maps to multiple products")
        if matches:
            return matches[0]

    cursor.execute(
        """SELECT id, drug_id FROM drug_product
           WHERE identity_hash=%s FOR UPDATE""",
        (identity_hash,),
    )
    return cursor.fetchone()

#首先，通过标准通用名查找，查不到，在用别名在别名表查询，如果还查不到，那么就是新药
def _find_or_create_drug(cursor: object, result: ParseResult) -> str:
    """Resolve the canonical drug by name, alias, or create it once."""
    normalized_name = result.drug_name.normalized
    cursor.execute(
        """SELECT id FROM drug
           WHERE generic_name_normalized=%s FOR UPDATE""",
        (normalized_name,),
    )
    exact = cursor.fetchone()
    if exact:
        return exact[0]

    cursor.execute(
        """SELECT DISTINCT drug_id FROM drug_alias
           WHERE alias_type='GENERIC' AND normalized_name=%s
           FOR UPDATE""",
        (normalized_name,),
    )
    alias_drugs = [row[0] for row in cursor.fetchall()]
    if len(alias_drugs) > 1:
        raise ValueError("Generic alias maps to multiple drugs")
    if alias_drugs:
        return alias_drugs[0]

    # The deterministic id and the unique normalized-name key make this safe
    # when two workers discover the same drug at the same time.
    drug_id = stable_id("drug", normalized_name)
    # INSERT IGNORE 防止并发插入风险
    cursor.execute(
        """INSERT IGNORE INTO drug
           (id, generic_name_raw, generic_name_normalized)
           VALUES (%s,%s,%s)""",
        (drug_id, result.drug_name.raw, normalized_name),
    )
    cursor.execute(
        """SELECT id FROM drug
           WHERE generic_name_normalized=%s FOR UPDATE""",
        (normalized_name,),
    )
    drug = cursor.fetchone()
    if not drug:
        raise RuntimeError(f"Cannot create or read drug: {normalized_name}")
    return drug[0]


def _resolve_product(
    cursor: object,
    result: ParseResult,
) -> tuple[str, str]:
    """Resolve or create the product and return (product_id, drug_id)."""
    product_id, identity_hash, identity_source = _product_identity(result)
    metadata = _product_metadata(result)
    match = _find_product_by_identity(cursor, result, identity_hash)

    if match:
        stored_product_id, stored_drug_id = match
    else:
        #获取药品id
        stored_drug_id = _find_or_create_drug(cursor, result)
        #和上市药品关联
        cursor.execute(
            """INSERT IGNORE INTO drug_product
               (id, drug_id, identity_hash, identity_source, brand_name_raw,
                brand_name_normalized, english_name_raw)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (product_id, stored_drug_id, identity_hash, identity_source, *metadata),
        )
        # Re-read after INSERT IGNORE because another worker may have inserted
        # the same identity while this transaction was waiting on a unique key.
        cursor.execute(
            """SELECT id, drug_id FROM drug_product
               WHERE identity_hash=%s FOR UPDATE""",
            (identity_hash,),
        )
        match = cursor.fetchone()
        if not match:
            raise RuntimeError(f"Cannot create or read product: {identity_hash}")
        stored_product_id, stored_drug_id = match

    cursor.execute(
        """UPDATE drug_product SET
               brand_name_raw=COALESCE(%s, brand_name_raw),
               brand_name_normalized=COALESCE(%s, brand_name_normalized),
               english_name_raw=COALESCE(%s, english_name_raw)
           WHERE id=%s""",
        (*metadata, stored_product_id),
    )
    for approval_number in result.approval_numbers:
        cursor.execute(
            """INSERT IGNORE INTO product_approval_number
               (product_id, approval_number) VALUES (%s,%s)""",
            (stored_product_id, approval_number),
        )
    return stored_product_id, stored_drug_id


def _revision_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _find_existing_document(cursor: object, result: ParseResult) -> tuple | None:
    cursor.execute(
        """SELECT id, product_id, version_status, revision_date, file_hash
           FROM document
           WHERE file_hash=%s AND parser_version=%s
           FOR UPDATE""",
        (result.file_hash, result.parser_version),
    )
    return cursor.fetchone()

"""
更新存在的文档：
  1.旧文档产品名映射错误：
    1.1: 如果新产品名下挂载有新ACTIVE文档，旧文档置为SUPERSEDED，同时，给就文档设置正确的产品名
  2.无论旧文档产品名是否映射错误，新产品名挂在另一个新ACTIVE文档，
    2.1: 当前文档修订时间 >= 另一个 ACTIVE 文档时间。另一个 ACTIVE 文档置为SUPERSEDED，当前文档置为ACTIVE
    2.2: 当前文档修订时间 < 另一个 ACTIVE 文档时间，不需要处理
"""
def _update_existing_document(
    cursor: object,
    result: ParseResult,
    stored_product_id: str,
    stored_drug_id: str,
    existing: tuple,
) -> None:
    """Repair/reuse an existing parse artifact and refresh its aliases."""

    existing_id, old_product_id, existing_status, _, _ = existing
    # 当前文档仍属于旧产品。
    # 如果目标产品已经有 ACTIVE 文档，必须先将当前文档降级，
    # 然后才能把它迁移到目标产品，否则会触发“一产品一个 ACTIVE”唯一约束。
    if old_product_id != stored_product_id:
        cursor.execute(
            """SELECT id FROM document
               WHERE product_id=%s AND version_status='ACTIVE' AND id<>%s
               FOR UPDATE""",
            (stored_product_id, existing_id),
        )
        conflicting_active = cursor.fetchone()
        if existing_status == "ACTIVE" and conflicting_active:
            cursor.execute(
                """UPDATE document SET version_status='SUPERSEDED', effective_to=NOW()
                   WHERE id=%s""",
                (existing_id,),
            )
        cursor.execute(
            """UPDATE document SET product_id=%s,
                   revision_date=COALESCE(%s, revision_date)
               WHERE id=%s""",
            (stored_product_id, result.revision_date, existing_id),
        )

    #目标商品是否已经存在另一个 ACTIVE 文档。
    # 修订时间 > 另一个 ACTIVE 文档时间。旧 ACTIVE 变 SUPERSEDED，当前文档变 ACTIVE
    # 修订时间 < 另一个 ACTIVE 文档时间 什么都不做
    cursor.execute(
        """SELECT id, revision_date, file_hash FROM document
           WHERE product_id=%s AND version_status='ACTIVE' AND id<>%s
           FOR UPDATE""",
        (stored_product_id, existing_id),
    )
    active = cursor.fetchone()
    if active:
        active_id, active_revision, active_hash = active
        new_revision = _revision_date(result.revision_date)
        if active_hash == result.file_hash or (
            new_revision and (active_revision is None or new_revision >= active_revision)
        ):
            cursor.execute(
                """UPDATE document SET version_status='SUPERSEDED', effective_to=NOW()
                   WHERE id=%s""",
                (active_id,),
            )
            cursor.execute(
                """UPDATE document SET version_status='ACTIVE', supersedes_document_id=%s,
                       effective_from=NOW(), effective_to=NULL
                   WHERE id=%s""",
                (active_id, existing_id),
            )
    elif existing_status != "WITHDRAWN":
        cursor.execute(
            """UPDATE document SET version_status='ACTIVE', effective_from=NOW(),
                   effective_to=NULL WHERE id=%s""",
            (existing_id,),
        )

    _store_aliases(cursor, result, stored_drug_id, stored_product_id, existing_id)

"""
新增文档：
  1. 解析器升级，文档还是一样，将旧版本置为SUPERSEDED
  2. 同一解析器，文档升级，当前文档修订时间晚于旧文档，将旧文档版本置为SUPERSEDED
"""
def _insert_document(
    cursor: object,
    result: ParseResult,
    stored_product_id: str,
    stored_drug_id: str,
    publish: bool = True,
) -> None:
    """Insert a new document, selecting its version status under a row lock."""
    cursor.execute(
        """SELECT id, revision_date, file_hash FROM document
           WHERE product_id=%s AND version_status='ACTIVE'
           FOR UPDATE""",
        (stored_product_id,),
    )
    current = cursor.fetchone()
    new_revision = _revision_date(result.revision_date)
    supersedes_id = None

    if not publish:
        version_status = "PENDING"
    elif current is None:
        version_status = "ACTIVE"
    else:
        current_id, current_revision, current_hash = current
        if current_hash == result.file_hash or (
            new_revision and (current_revision is None or new_revision >= current_revision)
        ):
            version_status = "ACTIVE"
            supersedes_id = current_id
            cursor.execute(
                """UPDATE document SET version_status='SUPERSEDED', effective_to=NOW()
                   WHERE id=%s""",
                (current_id,),
            )
        elif new_revision:
            version_status = "SUPERSEDED"
        else:
            version_status = "PENDING"

    cursor.execute(
        """INSERT INTO document
           (id, product_id, source_file, file_hash, parser_version, revision_date,
            page_count, parse_status, version_status, supersedes_document_id,
            effective_from)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                   CASE WHEN %s='ACTIVE' THEN NOW() ELSE NULL END)""",
        (
            result.document_id,
            stored_product_id,
            result.source_file,
            result.file_hash,
            result.parser_version,
            result.revision_date,
            result.page_count,
            "REVIEW" if result.failed_pages or result.warnings else "READY",
            version_status,
            supersedes_id,
            version_status,
        ),
    )
    _store_aliases(cursor, result, stored_drug_id, stored_product_id, result.document_id)


def _insert_parsed_content(cursor: object, result: ParseResult) -> None:
    """Insert sections, elements, tables, images, and the quality report."""
    for section in result.sections:
        cursor.execute(
            """INSERT INTO section
               (id, document_id, raw_name, normalized_name, start_page, end_page,
                heading_page, heading_bbox, confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                section.id,
                result.document_id,
                section.raw_name,
                section.normalized_name,
                section.start_page,
                section.end_page,
                section.heading_ref.page,
                _json(section.heading_ref.bbox),
                section.confidence,
            ),
        )
    for element in result.elements:
        cursor.execute(
            """INSERT INTO element
               (id, document_id, section_id, page, bbox, element_type, raw_text,
                normalized_text, reading_order, confidence, values_json, is_bold)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                element.id,
                result.document_id,
                element.ref.section_id,
                element.ref.page,
                _json(element.ref.bbox),
                element.element_type,
                element.raw_text,
                element.normalized_text,
                element.reading_order,
                element.confidence,
                _json([asdict(value) for value in element.values]),
                element.is_bold,
            ),
        )
    for table in result.tables:
        cursor.execute(
            """INSERT INTO parsed_table
               (id, document_id, section_id, title, title_ref, markdown, raw_json,
                normalized_json, image_paths, refs_json, warnings_json)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                table.id,
                result.document_id,
                table.section_id,
                table.title,
                _json(asdict(table.title_ref)) if table.title_ref else None,
                table.markdown,
                _json(table.raw_json),
                _json(table.normalized_json),
                _json(table.image_paths),
                _json([asdict(ref) for ref in table.refs]),
                _json(table.warnings),
            ),
        )
        for cell in table.cells:
            cursor.execute(
                """INSERT INTO table_cell
                   (table_id, row_index, column_index, document_id, section_id, page,
                    bbox, raw_value, normalized_value, inherited_from_row)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    table.id,
                    cell.row,
                    cell.column,
                    result.document_id,
                    cell.ref.section_id,
                    cell.ref.page,
                    _json(cell.ref.bbox),
                    cell.raw,
                    cell.normalized,
                    cell.inherited_from_row,
                ),
            )
        for row in table.rows:
            ref = row.refs[0]
            cursor.execute(
                """INSERT INTO table_row
                   (id, table_id, document_id, section_id, row_index, page, bbox,
                    refs_json, raw_cells, normalized_cells, retrieval_text, values_json)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    row.id,
                    table.id,
                    result.document_id,
                    ref.section_id,
                    row.row_index,
                    ref.page,
                    _json(ref.bbox),
                    _json([asdict(item) for item in row.refs]),
                    _json(row.raw_cells),
                    _json(row.normalized_cells),
                    row.retrieval_text,
                    _json([asdict(value) for value in row.values]),
                ),
            )
    for image in result.images:
        cursor.execute(
            """INSERT INTO image
               (id, document_id, section_id, page, bbox, image_type, path, caption, confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                image.id,
                result.document_id,
                image.ref.section_id,
                image.ref.page,
                _json(image.ref.bbox),
                image.image_type,
                image.path,
                image.caption,
                image.confidence,
            ),
        )
    _insert_parse_audit(cursor, result)
    cursor.execute(
        "INSERT INTO parse_report (document_id, report_json) VALUES (%s,%s)",
        (result.document_id, _json(make_report(result))),
    )


def _insert_parse_audit(cursor: object, result: ParseResult) -> None:
    """Persist immutable run/page/element report rows alongside legacy report JSON."""
    run_id = stable_id("parse-run", result.document_id, result.parser_version, result.file_hash)
    status = "NEEDS_REVIEW" if result.failed_pages or result.warnings else "READY"
    # Re-ingesting the same immutable document is idempotent. Remove the old
    # report rows first so the deterministic run id can be written again.
    cursor.execute("DELETE FROM element_report WHERE parse_run_id=%s", (run_id,))
    cursor.execute("DELETE FROM page_report WHERE parse_run_id=%s", (run_id,))
    cursor.execute("DELETE FROM parse_run WHERE id=%s", (run_id,))
    cursor.execute(
        """INSERT INTO parse_run
           (id,document_id,document_version,file_hash,parser_version,config_version,
            status,started_at,finished_at,error_summary)
           VALUES (%s,%s,%s,%s,%s,%s,%s,NOW(6),NOW(6),%s)""",
        (run_id, result.document_id, result.revision_date or result.file_hash,
         result.file_hash, result.parser_version, None, status,
         _json({"failed_pages": result.failed_pages, "warnings": result.warnings})),
    )
    section_paths = {section.id: [section.normalized_name] for section in result.sections}
    page_text = {}
    for element in result.elements:
        page_text[element.ref.page] = page_text.get(element.ref.page, 0) + 1
        cursor.execute(
            """INSERT INTO element_report
               (id,parse_run_id,element_id,page,bbox,bbox_coordinate_system,
                reading_order,element_type,raw_text,normalized_text,section_path,
                confidence,source_asset_id,table_id,parent_id,warning_codes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (stable_id("element-report", run_id, element.id), run_id, element.id,
             element.ref.page, _json(element.ref.bbox), "pdf_points_top_left",
             element.reading_order, element.element_type, element.raw_text,
             element.normalized_text, _json(section_paths.get(element.ref.section_id, [])),
             element.confidence, None, None, None, _json([])),
        )
    # Table rows and image captions are first-class sources for retrieval
    # chunks. Their IDs must resolve through the same run-scoped audit API.
    for table in result.tables:
        for row in table.rows:
            raw = "；".join(str(value or "") for value in row.raw_cells)
            for ref in row.refs:
                cursor.execute(
                    """INSERT INTO element_report
                       (id,parse_run_id,element_id,page,bbox,bbox_coordinate_system,
                        reading_order,element_type,raw_text,normalized_text,section_path,
                        confidence,source_asset_id,table_id,parent_id,warning_codes)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (stable_id("element-report", run_id, row.id, ref.page), run_id,
                     row.id, ref.page, _json(ref.bbox), "pdf_points_top_left",
                     row.row_index, "table_row", raw, raw,
                     _json(section_paths.get(table.section_id, [])), 1.0,
                     None, table.id, None, _json([])),
                )
        from app.review.table_audit import cell_id
        grid=table.raw_json.get('audit_cells') or [dict(row=c.row,column=c.column,page=c.ref.page,bbox=list(c.ref.bbox),text=c.raw or '') for c in table.cells]
        for cell in grid:
            key=cell_id(table.id,cell['row'],cell['column'])
            cursor.execute('''INSERT INTO element_report
                (id,parse_run_id,element_id,page,bbox,bbox_coordinate_system,reading_order,element_type,
                 raw_text,normalized_text,section_path,confidence,source_asset_id,table_id,parent_id,warning_codes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                (stable_id('element-report',run_id,key),run_id,key,cell['page'],_json(cell.get('bbox')),
                 'pdf_points_top_left',cell['row']*1000+cell['column'],'table_cell',cell['text'],cell['text'],
                 _json([]),1.0,None,table.id,None,_json([])))
    for image in result.images:
        if not image.caption or not image.caption.strip():
            continue
        cursor.execute(
            """INSERT INTO element_report
               (id,parse_run_id,element_id,page,bbox,bbox_coordinate_system,
                reading_order,element_type,raw_text,normalized_text,section_path,
                confidence,source_asset_id,table_id,parent_id,warning_codes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (stable_id("element-report", run_id, image.id), run_id,
             image.id, image.ref.page, _json(image.ref.bbox),
             "pdf_points_top_left", 0, "image_caption", image.caption,
             image.caption, _json(section_paths.get(image.ref.section_id, [])),
             image.confidence, image.id, None, None, _json([])),
        )
    page_tables: dict[int, int] = {}
    for table in result.tables:
        for ref in table.refs:
            page_tables[ref.page] = page_tables.get(ref.page, 0) + 1
    page_images: dict[int, int] = {}
    for image in result.images:
        page_images[image.ref.page] = page_images.get(image.ref.page, 0) + 1
    for page in range(1, result.page_count + 1):
        warnings = []
        if page in result.failed_pages:
            warnings.append("page_parse_failed")
        if page in result.blank_pages:
            warnings.append("blank_page")
        cursor.execute(
            """INSERT INTO page_report
               (parse_run_id,page,width,height,text_count,table_count,image_count,warnings_json)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, page, None, None, page_text.get(page, 0), page_tables.get(page, 0),
             page_images.get(page, 0), _json(warnings)),
        )


def _replace_chunks(
    cursor: object,
    result: ParseResult,
    stored_drug_id: str,
    stored_product_id: str,
) -> int:
    """Rebuild one document's deterministic chunks inside the ingest transaction."""
    from app.chunker.source_mapping import map_chunk_sources

    chunks = build_chunks(result, stored_drug_id, stored_product_id)
    source_segments = map_chunk_sources(result, chunks)
    parse_run_id = stable_id("parse-run", result.document_id, result.parser_version, result.file_hash)
    manifest_payload = "|".join(
        f"{chunk.id}:{chunk.content_hash}:"
        f"{hashlib.sha256(_json(chunk.metadata).encode('utf-8')).hexdigest()}"
        for chunk in sorted(chunks, key=lambda item: item.id)
    )
    manifest_hash = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()

    cursor.execute(
        "SELECT chunk_manifest_hash FROM document_index_state WHERE document_id=%s FOR UPDATE",
        (result.document_id,),
    )
    existing_state = cursor.fetchone()

    cursor.execute("DELETE FROM chunk WHERE document_id=%s", (result.document_id,))
    for chunk in chunks:
        cursor.execute(
            """INSERT INTO chunk
               (id, document_id, section_id, parent_id, content, retrieval_text,
                content_type, is_parent, token_count, page_start, page_end, bbox,
                reading_order, refs_json, metadata_json, content_hash, chunker_version)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                chunk.id,
                chunk.document_id,
                chunk.section_id,
                chunk.parent_id,
                chunk.content,
                chunk.retrieval_text,
                chunk.content_type,
                chunk.is_parent,
                chunk.token_count,
                chunk.page_start,
                chunk.page_end,
                _json(chunk.bbox) if chunk.bbox else None,
                chunk.reading_order,
                _json([asdict(ref) for ref in chunk.refs]),
                _json(chunk.metadata),
                chunk.content_hash,
                chunk.chunker_version,
            ),
        )
        for ordinal, segment in enumerate(source_segments[chunk.id]):
            cursor.execute(
                """INSERT INTO chunk_source_segment
                   (chunk_id,parse_run_id,index_version,ordinal,element_id,source_id,
                    source_kind,page_number,bbox,element_start,element_end,
                    chunk_start,chunk_end,location_status)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (chunk.id, parse_run_id, manifest_hash, ordinal,
                 segment.element_id, segment.source_id, segment.source_kind,
                 segment.page_number, _json(segment.bbox) if segment.bbox else None,
                 segment.element_start, segment.element_end, segment.chunk_start,
                 segment.chunk_end, segment.location_status),
            )

    if existing_state is None:
        cursor.execute(
            """INSERT INTO document_index_state
               (document_id, chunker_version, chunk_manifest_hash, vector_status)
               VALUES (%s,%s,%s,'PENDING')""",
            (result.document_id, CHUNKER_VERSION, manifest_hash),
        )
    elif existing_state[0] != manifest_hash:
        cursor.execute(
            """UPDATE document_index_state SET
                   chunker_version=%s, chunk_manifest_hash=%s,
                   vector_status='PENDING', embedding_model=NULL,
                   collection_name=NULL, point_count=0,
                   last_error=NULL, indexed_at=NULL
               WHERE document_id=%s""",
            (CHUNKER_VERSION, manifest_hash, result.document_id),
        )
    else:
        cursor.execute(
            """UPDATE document_index_state SET chunker_version=%s
               WHERE document_id=%s""",
            (CHUNKER_VERSION, result.document_id),
        )
    return len(chunks)


def initialize(mysql_url: str) -> None:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config, with_database=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
        connection.commit()
    finally:
        connection.close()

    migration_dir = Path(__file__).resolve().parents[2] / "migrations"
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS schema_migration (
                       version VARCHAR(255) PRIMARY KEY,
                       checksum CHAR(64) NOT NULL,
                       applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"""
            )
            cursor.execute("SELECT version, checksum FROM schema_migration")
            applied = dict(cursor.fetchall())
            for migration in sorted(migration_dir.glob("*.sql")):
                content = migration.read_bytes()
                checksum = hashlib.sha256(content).hexdigest()
                if migration.name in applied:
                    if applied[migration.name] != checksum:
                        raise RuntimeError(
                            f"Applied migration was modified: {migration.name}"
                        )
                    continue
                for statement in _migration_statements(migration):
                    cursor.execute(statement)
                cursor.execute(
                    "INSERT INTO schema_migration (version, checksum) VALUES (%s,%s)",
                    (migration.name, checksum),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


"""Database model: drug -> drug_product -> document, with aliases and approvals."""


def ingest(result: ParseResult, mysql_url: str, *, publish: bool = True) -> bool:
    """Persist one parse result atomically.

    True means a new document was inserted. False means the same
    (file_hash, parser_version) had already been ingested; its aliases and
    version relationship may still be refreshed.
    """
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        connection.begin()
        with connection.cursor() as cursor:
            stored_product_id, stored_drug_id = _resolve_product(cursor, result)
            existing = _find_existing_document(cursor, result)

            if existing:
                if publish:
                    _update_existing_document(
                        cursor,
                        result,
                        stored_product_id,
                        stored_drug_id,
                        existing,
                    )
                is_new_document = False
            else:
                _insert_document(
                    cursor,
                    result,
                    stored_product_id,
                    stored_drug_id,
                    publish=publish,
                )
                _insert_parsed_content(cursor, result)
                is_new_document = True

            _replace_chunks(
                cursor,
                result,
                stored_drug_id,
                stored_product_id,
            )

        # There is exactly one commit point. Any failure before it rolls back
        # product, document, aliases, parsed content, and the quality report.
        connection.commit()
        return is_new_document
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


