"""Replay explicit audit operations against exact source signatures, never fuzzy text."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from app.parser.layout import stable_id
from app.storage.mysql import _connect, parse_mysql_url


def prepare_operation(suggested, mysql_url, document_id, id_map):
    value = deepcopy(suggested)
    signatures = {}
    keys = ("source_chunk_id", "target_chunk_id", "anchor_chunk_id")
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            for key in keys:
                chunk_id = value.get(key)
                if not chunk_id:
                    continue
                cursor.execute("SELECT content_type,content,metadata_json FROM chunk WHERE id=%s AND document_id=%s", (chunk_id, document_id))
                row = cursor.fetchone()
                if not row:
                    raise ValueError("Chunk 目标不属于来源版本")
                metadata = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                ids = metadata.get("source_element_ids", [])
                if not ids or any(item not in id_map for item in ids):
                    raise ValueError("Chunk 缺少可重放的精确来源映射")
                signatures[key] = {"kind": row[0], "content": row[1], "ids": [id_map[item] for item in ids]}
    finally:
        conn.close()
    for key in ("element_ids",):
        ids = value.get(key, [])
        if any(item not in id_map for item in ids):
            raise ValueError("移动元素不属于来源版本")
        value[key] = [id_map[item] for item in ids]
    if value.get("split_element_id"):
        value["split_element_id"] = id_map.get(value["split_element_id"])
    value["signatures"] = signatures
    return value


def apply_chunk_operations(result, chunks, *, phase="boundary"):
    """Return modified chunks while preserving source refs and parent ownership.

    Ambiguous sentence fragments and non-text boundary operations fail explicitly.
    They must not drop evidence or silently execute a different operation.
    """
    elements = {item.id: item for item in result.elements}
    aliases = {}
    directives=[d for d in result.chunk_policy.get("chunk_operations", []) if
                (d["operation"] in {"UPDATE_METADATA", "MOVE_CHUNK_BEFORE", "MOVE_CHUNK_AFTER",'UPDATE_CHUNK_CONTEXT','CHANGE_CHUNK_PARENT','MOVE_ELEMENT_TO_CHUNK','ADJUST_CHUNK_RANGE'}) == (phase == "metadata")]
    # Bind immutable source identities before any operation rebuilds cell
    # membership or parent bodies. Later operations must keep pointing to the
    # same object, not search again against already-mutated source signatures.
    for planned in directives:
        for key,signature in planned.get('signatures',{}).items():
            old_id=planned.get(key)
            if not old_id or old_id in aliases:continue
            tracked=[c for c in chunks if old_id in c.metadata.get('audit_source_chunk_ids',[])]
            candidates=tracked if tracked else [c for c in chunks if c.content_type==signature['kind'] and c.source_element_ids==signature['ids']]
            exact=[c for c in candidates if c.content==signature['content']]
            candidates=exact or candidates
            if len(candidates)==1:
                aliases[old_id]=candidates[0]
                history=candidates[0].metadata.setdefault('audit_source_chunk_ids',[])
                if old_id not in history:history.append(old_id)
    for directive in directives:
        metadata_operation = directive["operation"] in {"UPDATE_METADATA", "MOVE_CHUNK_BEFORE", "MOVE_CHUNK_AFTER",'UPDATE_CHUNK_CONTEXT','CHANGE_CHUNK_PARENT','MOVE_ELEMENT_TO_CHUNK','ADJUST_CHUNK_RANGE'}
        if metadata_operation != (phase == "metadata"):
            continue
        def locate(key):
            old_id = directive.get(key)
            if old_id in aliases:
                if not any(c is aliases[old_id] for c in chunks):
                    raise ValueError(f"目标 Chunk 已被本批次较早的合并移除，请核对操作依赖（{directive['operation']}，{key}={old_id}）")
                return aliases[old_id]
            tracked = [c for c in chunks if old_id in c.metadata.get("audit_source_chunk_ids", [])]
            if len(tracked) == 1:
                aliases[old_id] = tracked[0]
                return tracked[0]
            signature = directive["signatures"].get(key)
            if not signature:
                raise ValueError("修正缺少目标 Chunk")
            candidates = [c for c in chunks if c.content_type == signature["kind"] and c.source_element_ids == signature["ids"]]
            exact = [c for c in candidates if c.content == signature["content"]]
            candidates = exact or candidates
            if len(candidates) != 1:
                raise ValueError(f"目标 Chunk 在新版本中不存在或有歧义，请重新核对（{directive['operation']}，{key}={old_id}，匹配 {len(candidates)} 个）")
            aliases[old_id] = candidates[0]
            candidates[0].metadata.setdefault("audit_source_chunk_ids", []).append(old_id)
            return candidates[0]

        def rewrite(chunk, ids):
            if not ids or any(item not in elements for item in ids):
                raise ValueError("边界操作仅支持有完整文本来源且非空的 Chunk")
            old_joined = "\n".join(elements[item].raw_text.strip() for item in chunk.source_element_ids)
            if chunk.content != old_joined:
                raise ValueError("句子片段或重叠 Chunk 需要人工调整，不能按完整元素重写")
            chunk.source_element_ids = list(dict.fromkeys(ids))
            chunk.content = "\n".join(elements[item].raw_text.strip() for item in chunk.source_element_ids)
            chunk.retrieval_text = chunk.content
            chunk.refs = [deepcopy(elements[item].ref) for item in chunk.source_element_ids]
            chunk.page_start = min(ref.page for ref in chunk.refs)
            chunk.page_end = max(ref.page for ref in chunk.refs)
            chunk.bbox = (min(r.bbox[0] for r in chunk.refs), min(r.bbox[1] for r in chunk.refs), max(r.bbox[2] for r in chunk.refs), max(r.bbox[3] for r in chunk.refs))
            from app.chunker.parent_child import estimated_token_count
            chunk.token_count = estimated_token_count(chunk.content)
            chunk.content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            chunk.id = stable_id(result.document_id, "chunk-review", chunk.content_type, *chunk.source_element_ids, chunk.content_hash)
            chunk.overlap_source_ids = []
            chunk.metadata.update(source_element_ids=chunk.source_element_ids.copy(), raw_text=chunk.content, content_hash=chunk.content_hash, overlap_source_ids=[], split_reason="audit_correction")
            chunk.metadata.pop("source_text_start", None)
            chunk.split_reason = "audit_correction"

        source = locate("source_chunk_id")
        op = directive["operation"]
        if op in {"MERGE_CHUNK", "SPLIT_CHUNK", "MOVE_ELEMENT"}:
            source.metadata.setdefault("audit_previous_parent_ids", []).append(source.parent_id)
        if op in {'UPDATE_CHUNK_CONTEXT','CHANGE_CHUNK_PARENT','MOVE_ELEMENT_TO_CHUNK','ADJUST_CHUNK_RANGE'}:
            affected_parent_ids={source.parent_id}
            from app.review.table_audit import cell_id
            table=next(t for t in result.tables if t.id==directive['table_id'])
            from app.chunker.source_mapping import _sources
            exact_sources=_sources(result)
            cells=table.raw_json.get('audit_cells') or [dict(row=c.row,column=c.column,text=c.raw or '',page=c.ref.page,bbox=list(c.ref.bbox),precise=bool(exact_sources[cell_id(table.id,c.row,c.column)][2])) for c in table.cells]
            pool={cell_id(table.id,c['row'],c['column']):c for c in cells}
            def rebuild(chunk, ids):
                if not ids or set(ids)-pool.keys():raise ValueError('表格 Chunk 不能丢失或引用无效 Cell')
                ids=list(dict.fromkeys(ids));chunk.metadata['cell_ids']=ids
                chunk.content='\n'.join(pool[i]['text'] for i in ids);chunk.retrieval_text=chunk.content
                chunk.source_element_ids=ids.copy()
                chunk.metadata.update(source_element_ids=chunk.source_element_ids.copy(),raw_text=chunk.content)
                from app.models import SourceRef
                refs=[SourceRef(result.document_id,table.section_id,pool[i]['page'],tuple(pool[i]['bbox'])) for i in ids if pool[i].get('bbox') and pool[i].get('precise',False)]
                chunk.refs=deepcopy(refs);chunk.page_start=min(pool[i]['page'] for i in ids);chunk.page_end=max(pool[i]['page'] for i in ids)
                chunk.bbox=(min(r.bbox[0] for r in refs),min(r.bbox[1] for r in refs),max(r.bbox[2] for r in refs),max(r.bbox[3] for r in refs)) if refs else None
                from app.chunker.parent_child import estimated_token_count
                chunk.token_count=estimated_token_count(chunk.content);chunk.content_hash=hashlib.sha256(chunk.content.encode()).hexdigest();chunk.metadata['content_hash']=chunk.content_hash
            if op=='UPDATE_CHUNK_CONTEXT':
                selected=directive['context_sources'];texts=[]
                for s in selected:
                    if s['kind'] in {'row_header','column_header'}:texts.append(pool[s['ref_id']]['text'])
                    elif s['kind']=='table_title':texts.append(table.title)
                    else:texts.append(next(s.normalized_name for s in result.sections if s.id==table.section_id))
                source.metadata['context_sources']=deepcopy(selected)
                source.metadata['raw_content']=source.content
                source.retrieval_text='\n'.join([*texts,source.content]);source.metadata['retrieval_content']=source.retrieval_text
            elif op=='CHANGE_CHUNK_PARENT':
                target=locate('target_chunk_id')
                affected_parent_ids.add(target.id)
                source.parent_id=target.id;source.metadata['parent_id']=target.id
            else:
                mode=directive.get('mode') if op=='ADJUST_CHUNK_RANGE' else 'move';owned=source.metadata['cell_ids'];moving=directive.get('cell_ids',[])
                if mode=='split':
                    for chunk in chunks:
                        if chunk.reading_order>source.reading_order:chunk.reading_order+=1
                    target=deepcopy(source);target.id=stable_id(source.id,'table-split',*moving);target.reading_order=source.reading_order+1;target.metadata['audit_source_chunk_ids']=[];chunks.append(target);rebuild(target,moving);rebuild(source,[i for i in owned if i not in moving])
                else:
                    target=locate('target_chunk_id')
                    affected_parent_ids.add(target.parent_id)
                    if mode=='merge':rebuild(source,[*owned,*target.metadata['cell_ids']]);chunks.remove(target)
                    else:rebuild(target,[*target.metadata['cell_ids'],*moving]);rebuild(source,[i for i in owned if i not in moving])
            # Retrieval parent bodies follow explicit child ownership, never PDF row order guesses.
            for parent in [c for c in chunks if c.id in affected_parent_ids and c.is_parent and c.metadata.get('table_id')==table.id and c.content_type!='table']:
                members=sorted([c for c in chunks if c.parent_id==parent.id],key=lambda c:c.reading_order)
                member_ids=list(dict.fromkeys(i for c in members for i in c.metadata.get('cell_ids',[])))
                # Preserve PDF grid order when a Cell changes its child owner.
                member_ids.sort(key=lambda key:(pool[key]['row'],pool[key]['column']))
                if member_ids:
                    rebuild(parent,member_ids)
                    parent.retrieval_text='\n'.join(c.retrieval_text for c in members)
                else:
                    parent.content=parent.retrieval_text=''
                    parent.source_element_ids=[];parent.refs=[];parent.bbox=None;parent.token_count=0
                    parent.content_hash=hashlib.sha256(b'').hexdigest()
                    parent.metadata.update(cell_ids=[],source_element_ids=[],raw_text='',content_hash=parent.content_hash)
        elif op == "UPDATE_METADATA":
            fields = directive["fields"]
            source.metadata.update(deepcopy(fields))
            if "section_path" in fields:
                source.section_path = fields["section_path"].copy()
            source.metadata["metadata_edit_version"] = result.parser_version
            if "summary" in fields and source.content_type in {"section_summary", "topic_summary"}:
                if not fields["summary"].strip():
                    raise ValueError("摘要 Chunk 正文不能为空")
                source.content = fields["summary"].strip()
                source.retrieval_text = source.content
                source.content_hash = hashlib.sha256(source.content.encode()).hexdigest()
                source.id = stable_id(result.document_id, "summary-review", source.content_type, source.content_hash)
                from app.chunker.parent_child import estimated_token_count
                source.token_count = estimated_token_count(source.content)
                source.metadata.update(raw_text=source.content, content_hash=source.content_hash)
        elif op in {"MOVE_CHUNK_BEFORE", "MOVE_CHUNK_AFTER"}:
            target = locate("anchor_chunk_id")
            ordered = sorted(chunks, key=lambda c: (c.reading_order, c.id))
            slots = sorted(c.reading_order for c in ordered)
            if source is target:
                raise ValueError("不能将分块移动到自身之前或之后")
            if source.is_parent:
                # Builders may store all parents before all children. Normalize
                # that storage layout into families before moving a root.
                by_id = {c.id: c for c in ordered}
                visited, grouped = set(), []
                def append_family(root):
                    if root.id in visited:
                        return
                    visited.add(root.id)
                    grouped.append(root)
                    for child in ordered:
                        if child.parent_id == root.id:
                            append_family(child)
                for root in ordered:
                    if not root.parent_id or root.parent_id not in by_id:
                        append_family(root)
                for root in ordered:
                    append_family(root)
                ordered = grouped
            def family(root):
                ids = {root.id}
                if root.is_parent:
                    while True:
                        children = {c.id for c in ordered if c.parent_id in ids}
                        if children <= ids:
                            break
                        ids.update(children)
                return [c for c in ordered if c.id in ids]
            moving, anchor = family(source), family(target)
            if set(c.id for c in moving) & set(c.id for c in anchor):
                raise ValueError("不能将父块移动到自身下属分块内")
            # Reorder the complete hierarchy, preserving each child's ownership
            # and relative order, including cross-page descendants.
            moving_ids = {c.id for c in moving}
            ordered = [c for c in ordered if c.id not in moving_ids]
            at = max(ordered.index(c) for c in anchor) + 1 if op == "MOVE_CHUNK_AFTER" else min(ordered.index(c) for c in anchor)
            ordered[at:at] = moving
            previous = slots[0] - 1
            for slot, chunk in zip(slots, ordered):
                chunk.reading_order = max(slot, previous + 1)
                previous = chunk.reading_order
        elif op == "MERGE_CHUNK":
            target = locate("target_chunk_id")
            source.metadata["audit_previous_parent_ids"].append(target.parent_id)
            ordered = sorted(chunks, key=lambda c: (c.reading_order, c.id))
            if abs(ordered.index(source) - ordered.index(target)) != 1 or source.is_parent or target.is_parent:
                raise ValueError("只能合并相邻的证据 Chunk")
            pair = sorted([source, target], key=lambda c: c.reading_order)
            if any(any(item not in elements for item in c.source_element_ids) or c.content != "\n".join(elements[item].raw_text.strip() for item in c.source_element_ids) for c in pair):
                raise ValueError("合并仅支持完整文本来源，句子片段不可扩展为全文")
            rewrite(source, [item for c in pair for item in c.source_element_ids])
            source.metadata["audit_source_chunk_ids"] = list(dict.fromkeys(source.metadata.get("audit_source_chunk_ids", []) + target.metadata.get("audit_source_chunk_ids", [])))
            chunks.remove(target)
            for key, chunk in list(aliases.items()):
                if chunk is target:
                    aliases[key] = source
        elif op == "SPLIT_CHUNK":
            ids = source.source_element_ids.copy()
            if directive.get("split_element_id") not in ids:
                raise ValueError("拆分点不属于来源 Chunk")
            index = ids.index(directive["split_element_id"]) + (directive["split_position"] == "after")
            if not 0 < index < len(ids) or source.is_parent:
                raise ValueError("拆分后两侧都必须保留来源元素")
            right = deepcopy(source)
            right.metadata["audit_source_chunk_ids"] = []
            rewrite(right, ids[index:])
            rewrite(source, ids[:index])
            right.reading_order = source.reading_order + 1
            chunks.append(right)
        elif op == "MOVE_ELEMENT":
            target = locate("target_chunk_id")
            target.metadata.setdefault("audit_previous_parent_ids", []).append(target.parent_id)
            moving = directive["element_ids"]
            if not moving or set(moving) - set(source.source_element_ids) or source.is_parent or target.is_parent:
                raise ValueError("只能移动来源证据 Chunk 内的元素")
            if set(moving) & set(target.source_element_ids):
                raise ValueError("目标 Chunk 已包含移动元素")
            target_ids = target.source_element_ids + moving if target.reading_order < source.reading_order else moving + target.source_element_ids
            rewrite(target, target_ids)
            rewrite(source, [item for item in source.source_element_ids if item not in moving])
        else:
            raise ValueError("不支持的 Chunk 操作")
    if phase=='metadata':
        # Keep empty groups until all directives finish (later directives may use
        # them as targets); then remove only childless, evidence-free table groups.
        occupied={c.parent_id for c in chunks if c.parent_id}
        chunks[:]=[c for c in chunks if not (c.content_type=='row_group_parent' and c.metadata.get('table_id') and c.id not in occupied and not c.content.strip() and not c.source_element_ids)]
    ordered = sorted(chunks, key=lambda c: (c.reading_order, c.id))
    for index, chunk in enumerate(ordered):
        chunk.previous_chunk_id = ordered[index - 1].id if index else None
        chunk.next_chunk_id = ordered[index + 1].id if index + 1 < len(ordered) else None
        chunk.metadata.update(previous_chunk_id=chunk.previous_chunk_id, next_chunk_id=chunk.next_chunk_id)
    return chunks
