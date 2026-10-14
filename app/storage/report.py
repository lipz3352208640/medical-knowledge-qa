"""Read-only RAG report and audit queries used by the internal API."""

from __future__ import annotations

import json

from app.storage.audit import MySQLAuditStore
from app.storage.mysql import _connect, parse_mysql_url


def _json_value(value):
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def list_documents(mysql_url: str, limit: int = 100, include_withdrawn: bool = False) -> list[dict]:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT d.id,d.source_file,d.file_hash,d.parser_version,d.revision_date,
                          d.page_count,d.parse_status,d.version_status,d.created_at,
                          p.id,dr.generic_name_normalized
                   FROM document d
                   JOIN drug_product p ON p.id=d.product_id
                   JOIN drug dr ON dr.id=p.drug_id
                   WHERE (%s OR d.version_status<>'WITHDRAWN')
                   ORDER BY d.created_at DESC LIMIT %s""",
                (include_withdrawn,limit),
            )
            rows = cursor.fetchall()
        return [dict(zip(
            ("document_id","source_file","file_hash","parser_version","revision_date",
             "page_count","parse_status","version_status","created_at","product_id","drug_name"),
            row, strict=True,
        )) for row in rows]
    finally:
        connection.close()


def document_identity(mysql_url: str, document_id: str) -> dict | None:
    connection=_connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT product_id FROM document WHERE id=%s',(document_id,))
            row=cursor.fetchone()
            if not row:return None
            cursor.execute('SELECT id FROM document WHERE product_id=%s ORDER BY id',(row[0],))
            return {'product_id':row[0],'document_ids':[d[0] for d in cursor.fetchall()]}
    finally:connection.close()


def document_versions(mysql_url: str, document_id: str, page: int = 1, page_size: int = 20) -> dict | None:
    """Paginate actual product versions, never infer lineage from names or hashes."""
    if page<1 or not 1<=page_size<=100: raise ValueError('invalid pagination')
    connection=_connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT product_id FROM document WHERE id=%s',(document_id,))
            source=cursor.fetchone()
            if not source: return None
            product_id=source[0]
            cursor.execute('SELECT COUNT(*) FROM document WHERE product_id=%s',(product_id,))
            total=cursor.fetchone()[0]
            cursor.execute('SELECT id,source_file,page_count,version_status,parse_status,revision_date,created_at FROM document WHERE product_id=%s ORDER BY created_at DESC,id DESC LIMIT %s OFFSET %s',
                           (product_id,page_size,(page-1)*page_size))
            names=('document_id','source_file','page_count','version_status','parse_status','revision_date','created_at')
            versions=[dict(zip(names,row,strict=True)) for row in cursor.fetchall()]
            # The worker persists this binding before an admin review session exists.
            # Preserve every explicit key; never choose an arbitrary match.
            bindings={row['document_id']:[] for row in versions}
            if bindings:
                slots=','.join(['%s']*len(bindings))
                cursor.execute(f'SELECT target_document_id,batch_key FROM review_correction_batch WHERE target_document_id IN ({slots}) ORDER BY batch_key',tuple(bindings))
                for target,key in cursor.fetchall(): bindings[target].append(key)
            for row in versions: row['review_batch_keys']=bindings[row['document_id']]
        return {'document_id':document_id,'product_id':product_id,'total':total,'versions':versions}
    finally: connection.close()


def list_parse_runs(mysql_url: str, document_id: str) -> list[dict]:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT id,document_id,document_version,file_hash,parser_version,
                          config_version,status,started_at,finished_at,error_summary
                   FROM parse_run WHERE document_id=%s ORDER BY started_at DESC""",
                (document_id,),
            )
            rows = cursor.fetchall()
        names = ("run_id","document_id","document_version","file_hash","parser_version",
                 "config_version","status","started_at","finished_at","error_summary")
        return [dict(zip(names, row, strict=True)) for row in rows]
    finally:
        connection.close()


def legacy_source_elements(page, existing_ids, table_rows, image_rows):
    """Read persisted sources omitted by old report backfills; never infer text."""
    from app.parser.layout import stable_id
    result = []
    def item(key, bbox, order, kind, text, section, table=None, image=None):
        return {'report_id':stable_id('legacy-source',key,page),'element_id':key,
                'page':page,'bbox':bbox,'bbox_coordinate_system':'pdf_points_top_left',
                'reading_order':order,'element_type':kind,'raw_text':text,
                'normalized_text':text,'section_path':[section] if section else [],
                'confidence':1.0,'source_asset_id':image,'table_id':table,
                'parent_id':None,'warning_codes':[]}
    for key, table, order, row_page, bbox, refs, raw, normalized, section in table_rows:
        if key in existing_ids:
            continue
        positions=[ref['bbox'] for ref in (_json_value(refs) or []) if ref['page']==page]
        if not positions and row_page==page:
            positions=[_json_value(bbox)]
        if not positions:
            continue
        raw, normalized = _json_value(raw), _json_value(normalized)
        value=item(key,positions[0],order,'table_row','；'.join(str(v or '') for v in raw),section,table=table)
        value['table_cells']={'raw':raw,'normalized':normalized}
        result.append(value)
    for key, image_page, bbox, caption, confidence, section in image_rows:
        if key not in existing_ids and image_page==page and caption and caption.strip():
            value=item(key,_json_value(bbox),0,'image_caption',caption,section,image=key)
            value['confidence']=confidence
            result.append(value)
    return result


def chunk_hierarchy(mysql_url: str, run_id: str) -> dict | None:
    """All chunks for one immutable run, including cross-page relationships."""
    connection = _connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT document_id FROM parse_run WHERE id=%s', (run_id,))
            run=cursor.fetchone()
            if not run:return None
            cursor.execute('''SELECT id,content_type,content,page_start,page_end,parent_id,is_parent,metadata_json
                              FROM chunk WHERE document_id=%s ORDER BY reading_order,id''', (run[0],))
            names=('chunk_id','content_type','content','page_start','page_end','parent_id','is_parent','metadata')
            chunks=[]
            for row in cursor.fetchall():
                item=dict(zip(names,row,strict=True))
                item['is_parent']=bool(item['is_parent'])
                item['metadata']=_json_value(item['metadata'])
                chunks.append(item)
            return {'run_id':run_id,'chunks':chunks}
    finally:
        connection.close()


def parse_page(mysql_url: str, run_id: str, page: int, *, include_legacy_sources=False, table_cache=None) -> dict | None:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM page_report WHERE parse_run_id=%s AND page=%s",
                (run_id, page),
            )
            page_row = cursor.fetchone()
            cursor.execute(
                """SELECT id,element_id,page,bbox,bbox_coordinate_system,reading_order,
                          element_type,raw_text,normalized_text,section_path,confidence,
                          source_asset_id,table_id,parent_id,warning_codes
                   FROM element_report WHERE parse_run_id=%s AND page=%s
                   ORDER BY reading_order,id""",
                (run_id, page),
            )
            elements = cursor.fetchall()
            cursor.execute("SELECT id,raw_cells,normalized_cells FROM table_row WHERE document_id=(SELECT document_id FROM parse_run WHERE id=%s) AND page=%s ORDER BY row_index,id", (run_id,page))
            table_rows = {row[0]:{'raw':_json_value(row[1]),'normalized':_json_value(row[2])} for row in cursor.fetchall()}
            legacy_tables, legacy_images = [], []
            if include_legacy_sources:
                cursor.execute("""SELECT t.id,t.table_id,t.row_index,t.page,t.bbox,t.refs_json,
                                  t.raw_cells,t.normalized_cells,s.normalized_name
                           FROM table_row t LEFT JOIN section s ON s.id=t.section_id
                           WHERE t.document_id=(SELECT document_id FROM parse_run WHERE id=%s)
                           ORDER BY t.row_index,t.id""", (run_id,))
                legacy_tables=cursor.fetchall()
                cursor.execute("""SELECT i.id,i.page,i.bbox,i.caption,i.confidence,s.normalized_name
                           FROM image i LEFT JOIN section s ON s.id=i.section_id
                           WHERE i.document_id=(SELECT document_id FROM parse_run WHERE id=%s)
                             AND i.page=%s ORDER BY i.id""", (run_id,page))
                legacy_images=cursor.fetchall()
            cursor.execute(
                """SELECT id,document_id,section_id,content_type,content,retrieval_text,
                          page_start,page_end,bbox,parent_id,metadata_json
                   FROM chunk WHERE document_id=(SELECT document_id FROM parse_run WHERE id=%s)
                     AND page_start<=%s AND page_end>=%s
                   ORDER BY reading_order,id""",
                (run_id, page, page),
            )
            chunks = cursor.fetchall()
            cursor.execute(
                """SELECT s.chunk_manifest_hash FROM document_index_state s
                   JOIN parse_run r ON r.document_id=s.document_id WHERE r.id=%s""",
                (run_id,),
            )
            version_row = cursor.fetchone()
        if page_row is None:
            return None
        page_names = ("run_id","page","width","height","text_count","table_count",
                      "image_count","warnings")
        page_value = dict(zip(page_names, page_row, strict=True))
        page_value["warnings"] = _json_value(page_value["warnings"])
        element_names = ("report_id","element_id","page","bbox","bbox_coordinate_system",
                         "reading_order","element_type","raw_text","normalized_text",
                         "section_path","confidence","source_asset_id","table_id","parent_id",
                         "warning_codes")
        element_values = []
        for row in elements:
            value = dict(zip(element_names, row, strict=True))
            for key in ("bbox", "section_path", "warning_codes"):
                value[key] = _json_value(value[key])
            if value['element_id'] in table_rows:
                value['table_cells'] = table_rows[value['element_id']]
            element_values.append(value)
        chunk_names = ("chunk_id","document_id","section_id","content_type","content",
                       "retrieval_text","page_start","page_end","bbox","parent_id","metadata")
        chunk_values = []
        for row in chunks:
            value = dict(zip(chunk_names, row, strict=True))
            value["bbox"] = _json_value(value["bbox"])
            value["metadata"] = _json_value(value["metadata"])
            chunk_values.append(value)
        page_value["elements"] = element_values
        page_value["chunks"] = chunk_values
        page_value["index_version"] = version_row[0] if version_row else None
        from app.review.table_audit import read_table
        tables=table_cache if table_cache is not None else {}
        if not include_legacy_sources:
            # Legacy report rows omit table_id. Discover actual persisted tables
            # through their rows on this PDF page, never via text heuristics.
            with connection.cursor() as cursor:
                cursor.execute('SELECT DISTINCT table_id FROM table_row WHERE document_id=(SELECT document_id FROM parse_run WHERE id=%s) AND page=%s', (run_id,page))
                page_tables=[row[0] for row in cursor.fetchall() if row[0]]
            for table_id in page_tables:
                if table_id not in tables:
                    tables[table_id]=read_table(mysql_url,run_id,table_id)
        for table_id in {e.get('table_id') for e in element_values if e.get('table_id')}:
            if table_id not in tables:
                tables[table_id]=read_table(mysql_url,run_id,table_id)
        for element in element_values:
            if element['element_type']!='table_cell':continue
            table=tables.get(element.get('table_id'))
            cell=next((c for c in table['cells'] if c['id']==element['element_id']),None) if table else None
            if cell:
                element['table_cells']={k:cell[k] for k in ('row','column','rowspan','colspan')}
                element['bbox']=cell['bbox'] if cell['precise'] else None
        if include_legacy_sources:
            page_value['_source_supplements']=legacy_source_elements(page,{e['element_id'] for e in element_values},legacy_tables,legacy_images)
            from app.review.table_audit import read_table
            table_ids={t[1] for t in legacy_tables if t[1]}
            for table_id in table_ids:
                if table_id not in tables:
                    tables[table_id]=read_table(mysql_url,run_id,table_id)
                table=tables[table_id]
                if not table:continue
                for cell in table['cells']:
                    if cell['page']!=page:continue
                    page_value['_source_supplements'].append({'element_id':cell['id'],'page':page,'element_type':'table_cell','table_id':table_id,'raw_text':cell['text'],'bbox':cell['bbox'] if cell['precise'] else None,'reading_order':cell['row']*1000+cell['column'],'section_path':[], 'table_cells':{k:cell[k] for k in ('row','column','rowspan','colspan')}})
        if not include_legacy_sources:
            page_value['tables'] = [table for table in tables.values() if table and any(cell['page']==page for cell in table['cells'])]
        return page_value
    finally:
        connection.close()


def version_snapshot(mysql_url: str, run_id: str) -> dict | None:
    connection = _connect(parse_mysql_url(mysql_url))
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT d.id,d.page_count,d.file_hash,d.version_status FROM document d JOIN parse_run r ON r.document_id=d.id WHERE r.id=%s", (run_id,))
            row = cursor.fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    table_cache = {}  # Request-local: immutable run grids are read once across all pages.
    return {"run_id":run_id, "document_id":row[0], "page_count":row[1], "file_hash":row[2],
            "version_status":row[3], "pages":[parse_page(mysql_url,run_id,page,include_legacy_sources=True,table_cache=table_cache) for page in range(1,row[1]+1)]}


def run_document_id(mysql_url: str, run_id: str) -> str | None:
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT document_id FROM parse_run WHERE id=%s", (run_id,))
            row = cursor.fetchone()
        return row[0] if row else None
    finally:
        connection.close()


def chunk_sources(mysql_url: str, chunk_id: str, run_id: str, index_version: str) -> dict | None:
    """Return a chunk only when PDF, run, current index and elements agree."""
    config = parse_mysql_url(mysql_url)
    connection = _connect(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT c.id,c.document_id,c.content,c.content_type,c.page_start,c.page_end,
                          c.metadata_json,r.id,r.file_hash,d.file_hash,d.source_file,
                          s.chunk_manifest_hash
                   FROM chunk c
                   JOIN parse_run r ON r.document_id=c.document_id
                   JOIN document d ON d.id=c.document_id
                   JOIN document_index_state s ON s.document_id=d.id
                   WHERE c.id=%s AND r.id=%s""",
                (chunk_id, run_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if row[8] != row[9] or row[11] != index_version:
                raise ValueError("PDF、解析批次或索引版本不一致")
            cursor.execute(
                """SELECT ordinal,element_id,source_id,source_kind,page_number,bbox,
                          element_start,element_end,chunk_start,chunk_end,location_status,
                          parse_run_id,index_version
                   FROM chunk_source_segment WHERE chunk_id=%s ORDER BY ordinal""",
                (chunk_id,),
            )
            source_rows = cursor.fetchall()
            if any(source[11] not in (None, run_id) or source[12] != index_version
                   for source in source_rows):
                raise ValueError("分块来源属于其他解析或索引版本，请重新生成")
            element_ids = list(dict.fromkeys(source[1] for source in source_rows if source[1]))
            elements = {}
            for element_id in element_ids:
                cursor.execute(
                    """SELECT element_id,page,bbox,element_type,raw_text,reading_order
                       FROM element_report WHERE parse_run_id=%s AND element_id=%s
                       ORDER BY page""",
                    (run_id, element_id),
                )
                found = cursor.fetchall()
                if not found:
                    raise ValueError("来源元素不属于该解析批次")
                elements[element_id] = [
                    {"element_id": value[0], "page": value[1],
                     "bbox": _json_value(value[2]), "element_type": value[3],
                     "raw_text": value[4], "reading_order": value[5]}
                    for value in found
                ]
        names = ("ordinal", "element_id", "source_id", "source_kind", "page_number",
                 "bbox", "element_start", "element_end", "chunk_start", "chunk_end",
                 "location_status", "parse_run_id", "index_version")
        segments = []
        for source in source_rows:
            value = dict(zip(names, source, strict=True))
            value["bbox"] = _json_value(value["bbox"])
            if value["element_id"] and not any(
                element["page"] == value["page_number"]
                for element in elements[value["element_id"]]
            ):
                raise ValueError("来源元素页码与解析批次不一致")
            segments.append(value)
        return {
            "chunk_id": row[0], "document_id": row[1], "content": row[2],
            "content_type": row[3], "page_start": row[4], "page_end": row[5],
            "metadata": _json_value(row[6]), "parse_run_id": run_id,
            "file_hash": row[8], "source_file": row[10],
            "index_version": index_version, "segments": segments,
            "elements": elements,
            "locatable": bool(segments) and all(
                segment["location_status"] == "LOCATABLE" for segment in segments
            ),
            "highlight_granularity": "element_bbox",
        }
    finally:
        connection.close()


def list_qa_audits(mysql_url: str, limit: int = 50) -> list[dict]:
    return MySQLAuditStore(mysql_url).list_answers(limit)


def get_qa_audit(mysql_url: str, answer_id: str) -> dict | None:
    return MySQLAuditStore(mysql_url).get_answer(answer_id)
