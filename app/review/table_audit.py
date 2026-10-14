"""Table evidence and corrections. Grid IDs are derived from persisted coordinates.

Legacy row fallback boxes are never advertised as precise Cell locations.
Corrected spans live in parsed_table.raw_json.audit_cells, avoiding a second store.
"""
from copy import deepcopy
import hashlib
import json
from app.parser.layout import stable_id


def cell_id(table_id, row, column):
    return stable_id('table-cell', table_id, row, column)


def grid_hash(cells):
    return hashlib.sha256(json.dumps(cells, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def grid_revision_matches(cells, revision):
    """Accept only exact hashes of known historical evidence representations.

    Read-only helper flags were added without changing the persisted grid.
    Keep text, IDs, geometry, spans and precise-source flags in every variant;
    do not accept client-supplied replacement revisions or arbitrary fields.
    """
    if not isinstance(revision, str):
        return False
    for omitted in ((), ('merge_anchor_row',), ('merge_anchor_row', 'merge_placeholder')):
        represented=[{k:v for k,v in c.items() if k not in omitted} for c in cells]
        if grid_hash(represented)==revision:
            return True
    return False


def validate_grid(cells):
    occupied = set()
    for c in cells:
        for key in ('row', 'column', 'rowspan', 'colspan'):
            if type(c.get(key)) is not int or c[key] < (1 if 'span' in key else 0):
                raise ValueError('单元格坐标和跨度必须为有效整数')
        if c['row']+c['rowspan'] > 10000 or c['column']+c['colspan'] > 1000 or c['rowspan']*c['colspan'] > 10000 or len(occupied) > 100000:
            raise ValueError('单元格跨度超出允许范围')
        for r in range(c['row'], c['row'] + c['rowspan']):
            for col in range(c['column'], c['column'] + c['colspan']):
                if (r, col) in occupied:
                    raise ValueError('单元格跨度重叠')
                occupied.add((r, col))
    return occupied


def preview_cells(cells, operation, value):
    """Validate against the complete authoritative grid, then return a new grid."""
    validate_grid(cells)
    result = deepcopy(cells)
    ids = value.get('cell_ids', [])
    chosen = [c for c in result if c['id'] in ids]
    if not ids or len(set(ids)) != len(ids) or len(chosen) != len(ids):
        raise ValueError('单元格不属于当前表格或引用已失效')
    if operation in {'UPDATE_CELL_TEXT', 'UPDATE_CELL_BBOX'}:
        if len(chosen) != 1:
            raise ValueError('请选择一个单元格')
        if operation == 'UPDATE_CELL_TEXT':
            if not isinstance(value.get('text'), str) or value['text'] == chosen[0]['text']:
                raise ValueError('请修改单元格文字')
            chosen[0]['text'] = value['text']
        else:
            box=value.get('bbox')
            import math
            if not isinstance(box,list) or len(box)!=4 or any(type(n) not in (int,float) or not math.isfinite(n) for n in box) or min(box)<0 or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('BBox 无效')
            if chosen[0].get('precise') and box==chosen[0].get('bbox'):raise ValueError('请调整单元格 BBox')
            chosen[0]['bbox']=box;chosen[0]['precise']=True
        return result
    if operation == 'MERGE_CELLS':
        if len(chosen) < 2 or len({c['page'] for c in chosen}) != 1:
            raise ValueError('只能合并同页相邻单元格')
        # A legacy NULL with no vertical inheritance is an explicit grid slot,
        # not evidence of a span. Reviewers may absorb it into a known cell in
        # the same row or column. An inherited NULL requires its exact source
        # anchor in the selection; inherited nonempty content stays protected.
        unknown=[c for c in chosen if not c.get('span_known')]
        placeholder_merge=((len({c['row'] for c in chosen})==1 or len({c['column'] for c in chosen})==1) and any(c.get('span_known') for c in chosen) and all(c['rowspan']==c['colspan']==1 for c in chosen) and all(not c['text'] and (c.get('merge_placeholder') or (type(c.get('merge_anchor_row')) is int and any(a.get('span_known') and a['row']==c['merge_anchor_row'] and a['column']==c['column'] for a in chosen))) for c in unknown))
        if unknown and not placeholder_merge:raise ValueError('所选单元格缺少可靠跨度，不能合并；空占位格仅支持同一行或同一列连续合并')
        top=min(c['row'] for c in chosen);left=min(c['column'] for c in chosen)
        bottom=max(c['row']+c['rowspan'] for c in chosen);right=max(c['column']+c['colspan'] for c in chosen)
        expected={(r,col) for r in range(top,bottom) for col in range(left,right)}
        if validate_grid(chosen) != expected:
            raise ValueError('合并选区必须为完整相邻矩形')
        first=min(chosen,key=lambda c:(c['row'],c['column']))
        first.update(rowspan=bottom-top,colspan=right-left,text=str(value.get('text', '\n'.join(c['text'] for c in chosen if c['text'].strip()))),span_known=True,merge_placeholder=False)
        if all(c.get('precise') and c.get('bbox') for c in chosen):
            first['bbox']=[min(c['bbox'][0] for c in chosen),min(c['bbox'][1] for c in chosen),max(c['bbox'][2] for c in chosen),max(c['bbox'][3] for c in chosen)]
        else:
            first['bbox']=None;first['precise']=False
        result=[c for c in result if c['id'] not in ids or c is first]
    elif operation == 'SPLIT_CELL':
        if len(chosen)!=1 or not isinstance(value.get('parts'),list) or len(value['parts'])<2:
            raise ValueError('拆分必须明确至少两个新单元格')
        source=chosen[0];parts=[]
        if not source.get('span_known'):raise ValueError('当前解析数据缺少可靠跨度，不能拆分')
        for p in value['parts']:
            part={**source,**{k:p[k] for k in ('row','column','rowspan','colspan','text') if k in p},'bbox':p.get('bbox'),'precise':bool(p.get('bbox'))}
            if not isinstance(part.get('text'),str) or not part['text'].strip():raise ValueError('拆分文本必须明确填写')
            part['id']=cell_id(source['table_id'],part['row'],part['column']);parts.append(part)
        if validate_grid(parts)!=validate_grid([source]):raise ValueError('拆分必须恰好覆盖原单元格，不能增删行列')
        for p in parts:
            if p['bbox']:
                box=p['bbox'];old=source.get('bbox')
                import math
                if not old or not isinstance(box,list) or len(box)!=4 or any(type(n) not in (int,float) or not math.isfinite(n) for n in box) or not(old[0]<=box[0]<box[2]<=old[2] and old[1]<=box[1]<box[3]<=old[3]):raise ValueError('新 BBox 必须位于原单元格内')
        result=[c for c in result if c['id']!=source['id']]+parts
    else:
        raise ValueError('不支持的单元格操作')
    validate_grid(result)
    return sorted(result,key=lambda c:(c['row'],c['column']))


def read_table(mysql_url, run_id, table_id):
    from app.storage.mysql import _connect,parse_mysql_url
    from app.storage.report import _json_value,chunk_hierarchy
    conn=_connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT document_id FROM parse_run WHERE id=%s',(run_id,));run=cur.fetchone()
            if not run:return None
            cur.execute('SELECT title,raw_json,normalized_json,refs_json,title_ref,section_id FROM parsed_table WHERE id=%s AND document_id=%s',(table_id,run[0]));record=cur.fetchone()
            if not record:return None
            title,raw,normalized,refs,title_ref,section=record;raw=_json_value(raw);normalized=_json_value(normalized)
            cur.execute('SELECT chunk_manifest_hash FROM document_index_state WHERE document_id=%s',(run[0],));version=cur.fetchone()
            cur.execute('SELECT id,row_index,bbox FROM table_row WHERE table_id=%s ORDER BY row_index',(table_id,));rows=cur.fetchall()
            row_ids={r[1]:r[0] for r in rows};row_boxes={r[1]:_json_value(r[2]) for r in rows}
            cur.execute('SELECT row_index,column_index,page,bbox,raw_value,inherited_from_row FROM table_cell WHERE table_id=%s ORDER BY row_index,column_index',(table_id,));values=cur.fetchall()
            cur.execute('SELECT normalized_name FROM section WHERE id=%s',(section,));section_row=cur.fetchone()
            cur.execute('SELECT page,width,height FROM page_report WHERE parse_run_id=%s',(run_id,));page_sizes={str(p):{'width':w,'height':h} for p,w,h in cur.fetchall()}
        cells=raw.get('audit_cells')
        if cells is None:
            cells=[dict(id=cell_id(table_id,r,c),table_id=table_id,row=r,column=c,rowspan=1,colspan=1,page=p,bbox=_json_value(b),text=text or '',precise=bool(b and _json_value(b)!=row_boxes.get(r)),span_known=inherited is None and text is not None,merge_placeholder=inherited is None and text is None,merge_anchor_row=inherited if text is None else None) for r,c,p,b,text,inherited in values]
        cells=deepcopy(cells)
        for c in cells:c['id']=cell_id(table_id,c['row'],c['column']);c['table_id']=table_id
        validate_grid(cells)
        if 'audit_cells' not in raw:
            from collections import Counter
            boxes=Counter((c['page'],tuple(c['bbox'] or [])) for c in cells)
            for c in cells:
                if boxes[(c['page'],tuple(c['bbox'] or []))]>1:c['precise']=False
        chunks=chunk_hierarchy(mysql_url,run_id)['chunks']
        mappings={}
        for ch in chunks:
            metadata=ch.get('metadata') or {}
            if metadata.get('table_id')!=table_id:continue
            explicit=metadata.get('cell_ids')
            ids=explicit if isinstance(explicit,list) else [c['id'] for c in cells if row_ids.get(c['row']) in metadata.get('source_element_ids',[]) or table_id in metadata.get('source_element_ids',[])]
            mappings[ch['chunk_id']]=ids
        return dict(table_id=table_id,run_id=run_id,document_id=run[0],index_version=version[0] if version else None,title=title,section_title=section_row[0] if section_row else '',section_id=section,refs=_json_value(refs),title_ref=_json_value(title_ref),cells=cells,chunk_cells=mappings,page_sizes=page_sizes,grid_revision=grid_hash(cells),warnings=[] if all(c['span_known'] for c in cells) else ['部分旧单元格缺少可靠跨度，仅支持查看和有明确来源的操作'])
    finally:conn.close()


def validate_correction(table, chunks, value):
    if not grid_revision_matches(table['cells'], value.get('grid_revision')):
        raise ValueError('表格证据已过期，请刷新后重新编辑')
    operation=value.get('operation');ids=value.get('cell_ids',[])
    pool={c['id']:c for c in table['cells']}
    if set(ids)-pool.keys():raise ValueError('Cell 引用不属于当前表格')
    if operation in {'MERGE_CELLS','SPLIT_CELL','UPDATE_CELL_TEXT','UPDATE_CELL_BBOX'}:
        after=preview_cells(table['cells'],operation,value)
        if operation=='UPDATE_CELL_BBOX':
            size=table.get('page_sizes',{}).get(str(pool[ids[0]]['page']),{})
            if not size.get('width') or not size.get('height'):raise ValueError('缺少页面尺寸，无法校验 BBox')
            if value['bbox'][2]>size['width'] or value['bbox'][3]>size['height']:raise ValueError('BBox 超出 PDF 页面范围')
        return after
    by_id={c['chunk_id']:c for c in chunks};source=by_id.get(value.get('source_chunk_id'))
    if not source or source['chunk_id'] not in table['chunk_cells']:raise ValueError('来源 Chunk 不属于当前表格')
    owned=table['chunk_cells'][source['chunk_id']]
    if source.get('is_parent'):raise ValueError('此操作只适用于表格子块')
    if operation=='UPDATE_CHUNK_CONTEXT':
        sources=value.get('context_sources');allowed={c['id'] for c in table['cells'] if c['row']==0 or c['column']==0 and any(pool[i]['row']==c['row'] for i in owned)}
        if not isinstance(sources,list) or not sources:raise ValueError('请选择上下文来源')
        seen=set()
        for item in sources:
            key=item.get('ref_id');kind=item.get('kind')
            valid=(kind in {'row_header','column_header'} and key in allowed and ((kind=='column_header' and pool[key]['row']==0) or (kind=='row_header' and pool[key]['column']==0))) or (kind=='table_title' and key==table['table_id'] and bool(table['title'])) or (kind=='section_heading' and key==table['section_id'])
            if not valid or key in seen:raise ValueError('上下文来源无效或重复')
            seen.add(key)
        if sources==source.get('metadata',{}).get('context_sources'):raise ValueError('上下文来源没有变化')
        return table['cells']
    target=by_id.get(value.get('target_chunk_id'))
    if operation=='CHANGE_CHUNK_PARENT':
        if not target or not target.get('is_parent') or target['chunk_id']==source.get('parent_id') or target['chunk_id'] not in table['chunk_cells']:raise ValueError('父块必须为同表格内不同的合法父块')
        seen={source['chunk_id']};current=target
        while current:
            if current['chunk_id'] in seen:raise ValueError('父子关系不能成环')
            seen.add(current['chunk_id']);current=by_id.get(current.get('parent_id'))
        return table['cells']
    mode=value.get('mode') if operation=='ADJUST_CHUNK_RANGE' else 'move'
    if operation not in {'MOVE_ELEMENT_TO_CHUNK','ADJUST_CHUNK_RANGE'}:raise ValueError('表格 Chunk 操作不支持')
    if mode not in {'merge','split','move'}:raise ValueError('边界操作无效')
    if mode!='split':
        if not target or target.get('is_parent') or target['chunk_id']==source['chunk_id'] or target['chunk_id'] not in table['chunk_cells']:raise ValueError('目标必须为同表格的其他子块')
    if mode=='merge':
        if target.get('parent_id')!=source.get('parent_id'):raise ValueError('合并目标必须为同父级的相邻子块')
        siblings=[c['chunk_id'] for c in chunks if c.get('parent_id')==source.get('parent_id') and not c.get('is_parent')]
        if abs(siblings.index(source['chunk_id'])-siblings.index(target['chunk_id']))!=1:raise ValueError('只能合并相邻子块')
    elif not ids or set(ids)-set(owned) or len(set(ids))!=len(ids) or not set(owned)-set(ids):raise ValueError('请选择来源 Cell，且原块必须保留至少一个 Cell')
    if target and mode=='move' and set(ids)&set(table['chunk_cells'][target['chunk_id']]):raise ValueError('目标已包含所选 Cell')
    return table['cells']


def apply_cell_correction(result, table, value):
    from app.models import TableCell,SourceRef
    cells=value['validated_cells']
    # The clone has fresh table/section IDs, while the saved grid is source evidence.
    for c in cells:c['id']=cell_id(table.id,c['row'],c['column']);c['table_id']=table.id
    table.raw_json['audit_cells']=deepcopy(cells)
    from app.parser.tables import markdown_table
    from app.parser.normalize import clean_space
    original={(c.row,c.column):c for c in table.cells}
    original_normalized=deepcopy(table.normalized_json.get('rows',[]))
    normalized=[]
    for c in cells:
        prior=original.get((c['row'],c['column']))
        text=prior.normalized if prior and (prior.raw or '')==c['text'] else clean_space(c['text'])
        normalized.append(TableCell(c['text'],text,c['row'],c['column'],SourceRef(result.document_id,table.section_id,c['page'],tuple(c['bbox'] or next(ref.bbox for ref in table.refs if ref.page==c['page'])))))
    table.cells=normalized
    count=max((c['column']+c['colspan'] for c in cells),default=0)
    matrix=[['']*count for _ in range(max((c['row']+c['rowspan'] for c in cells),default=0))]
    normalized_matrix=deepcopy(matrix)
    for c,n in zip(cells,normalized):
        matrix[c['row']][c['column']]=c['text']
        normalized_matrix[c['row']][c['column']]=n.normalized
        # Preserve unchanged inherited labels from the persisted normalization.
        prior=original.get((c['row'],c['column']))
        if prior and (prior.raw or '')==c['text'] and c['row']<len(original_normalized) and c['column']<len(original_normalized[c['row']]):
            normalized_matrix[c['row']][c['column']]=original_normalized[c['row']][c['column']]
            n.normalized=normalized_matrix[c['row']][c['column']]
            n.inherited_from_row=prior.inherited_from_row
    table.raw_json['rows']=deepcopy(matrix);table.normalized_json['rows']=deepcopy(normalized_matrix)
    for row in table.rows:
        row.raw_cells=matrix[row.row_index].copy();row.normalized_cells=normalized_matrix[row.row_index].copy();row.retrieval_text='；'.join(v or '' for v in row.normalized_cells)
    table.markdown=markdown_table(normalized_matrix)
