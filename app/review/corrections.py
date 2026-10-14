"""Apply confirmed, typed corrections to a cloned parse result.

The original run is never modified. Unsupported payloads fail the job rather
than silently treating a description as an executable correction.
"""

from __future__ import annotations

import json

from app.models import ParseResult, TableRow, TextElement
from app.parser.layout import stable_id
from app.storage.mysql import _connect, parse_mysql_url


def _box(value) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox 必须是四个坐标")
    box = tuple(float(number) for number in value)
    if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox 坐标无效")
    return box


def _text(value: dict, key: str = "text") -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"修正缺少 {key}")
    return text.strip()


def _old_summary_key(mysql_url: str, chunk_id: str, document_id: str) -> str:
    conn = _connect(parse_mysql_url(mysql_url))
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT content_type,metadata_json FROM chunk WHERE id=%s AND document_id=%s",
                           (chunk_id, document_id))
            row = cursor.fetchone()
        if not row or row[0] not in {"section_summary", "topic_summary"}:
            raise ValueError("目标不是该版本的摘要分块")
        metadata = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        return "|".join((row[0], metadata["normalized_section_name"],
                         metadata.get("topic_name") or ""))
    finally:
        conn.close()


def apply_proposals(result: ParseResult, id_map: dict[str, str],
                    proposals: list[dict], mysql_url: str, source_document_id: str) -> None:
    elements = {item.id: item for item in result.elements}
    rows: dict[str, TableRow] = {row.id: row for table in result.tables for row in table.rows}
    for proposal in proposals:
        kind = proposal["issue_type"]
        suggested = proposal.get("suggested") or {}
        old_ids = proposal.get("element_ids") or []
        targets = [id_map.get(value) for value in old_ids]
        if kind!='table_review' and any(value is None for value in targets):
            raise ValueError("修正目标不属于来源解析版本")
        if kind == "manual_review":
            raise ValueError("纯描述问题单不能自动执行")
        if kind == 'table_review':
            from app.review.table_audit import read_table,validate_correction,apply_cell_correction,cell_id
            from app.storage.report import chunk_hierarchy
            source_run=suggested['source_run_id']
            table=read_table(mysql_url,source_run,suggested['table_id'])
            if not table:raise ValueError('来源表格不存在')
            after=validate_correction(table,chunk_hierarchy(mysql_url,source_run)['chunks'],suggested)
            target_table=next(t for t in result.tables if t.id==id_map.get(suggested['table_id']))
            if suggested['operation'] in {'MERGE_CELLS','SPLIT_CELL','UPDATE_CELL_TEXT','UPDATE_CELL_BBOX'}:
                if target_table.raw_json.get('audit_cells'):
                    from copy import deepcopy
                    from app.review.table_audit import preview_cells
                    current=deepcopy(target_table.raw_json['audit_cells'])
                    for c in current:c['table_id']=table['table_id'];c['id']=cell_id(table['table_id'],c['row'],c['column'])
                    after=preview_cells(current,suggested['operation'],suggested)
                apply_cell_correction(result,target_table,{**suggested,'validated_cells':after})
            else:
                from app.review.chunk_operations import prepare_operation
                for c in table['cells']:id_map[c['id']]=cell_id(target_table.id,c['row'],c['column'])
                id_map[table['section_id']]=target_table.section_id
                directive=prepare_operation(suggested,mysql_url,source_document_id,id_map)
                directive['table_id']=target_table.id
                directive['cell_ids']=[id_map[i] for i in suggested.get('cell_ids',[])]
                directive['context_sources']=[{**s,'ref_id':id_map.get(s['ref_id'],s['ref_id'])} for s in suggested.get('context_sources',[])]
                result.chunk_policy.setdefault('chunk_operations',[]).append(directive)
        elif kind == "text_correction":
            if len(targets) != 1 or targets[0] not in elements:
                raise ValueError("文字修正需要一个文本元素")
            elements[targets[0]].raw_text = _text(suggested)
            elements[targets[0]].normalized_text = elements[targets[0]].raw_text
        elif kind == "bbox_correction":
            if len(targets) != 1 or targets[0] not in elements:
                raise ValueError("bbox 修正需要一个文本元素")
            elements[targets[0]].ref.bbox = _box(suggested.get("bbox"))
        elif kind == "element_missing":
            if len(targets) != 1 or targets[0] not in elements:
                raise ValueError("补充元素需要选择一个相邻元素")
            anchor = elements[targets[0]]
            value = _text(suggested)
            bbox = _box(suggested.get("bbox"))
            item_id = stable_id(result.document_id, "added-element", proposal["id"])
            if item_id in elements:
                raise ValueError("补充元素 ID 冲突")
            added = TextElement(item_id, type(anchor.ref)(result.document_id,
                                 anchor.ref.section_id, anchor.ref.page, bbox),
                                 suggested.get("element_type", "paragraph"), value,
                                 value, anchor.reading_order + 1, 1.0)
            result.elements.append(added)
            elements[item_id] = added
        elif kind == "reading_order":
            if suggested.get("operation") in {"MOVE_BEFORE", "MOVE_AFTER"}:
                source_id = id_map.get(suggested.get("move_id"))
                anchor_id = id_map.get(suggested.get("reference_id"))
                if source_id not in elements or anchor_id not in elements or source_id == anchor_id:
                    raise ValueError("排序来源或参照元素无效")
                source, anchor = elements[source_id], elements[anchor_id]
                if source.ref.page != anchor.ref.page:
                    raise ValueError("阅读顺序修正必须在同一页")
                page_items = sorted([e for e in result.elements if e.ref.page == source.ref.page], key=lambda e: e.reading_order)
                ordered = [e for e in page_items if e.id != source_id]
                at = next(i for i, e in enumerate(ordered) if e.id == anchor_id)
                ordered.insert(at + (suggested["operation"] == "MOVE_AFTER"), source)
                for position, item in enumerate(ordered, 1):
                    item.reading_order = position
                continue
            order = suggested.get("ordered_element_ids")
            if not isinstance(order, list) or set(order) != set(old_ids) or len(order) != len(old_ids):
                raise ValueError("阅读顺序必须完整列出选中的元素")
            selected = [elements[id_map[value]] for value in order]
            if len({item.ref.page for item in selected}) != 1:
                raise ValueError("阅读顺序修正必须在同一页")
            slots = sorted(item.reading_order for item in selected)
            for slot, item in zip(slots, selected, strict=True):
                item.reading_order = slot
        elif kind == "element_merge_split":
            if len(targets) > 1 and suggested.get("merge") is True:
                selected = [elements[value] for value in targets]
                if len({(item.ref.page, item.ref.section_id) for item in selected}) != 1:
                    raise ValueError("合并元素必须同页同章节")
                first = selected[0]
                first.raw_text = "\n".join(item.raw_text for item in selected)
                first.normalized_text = first.raw_text
                first.ref.bbox = (min(e.ref.bbox[0] for e in selected),
                                  min(e.ref.bbox[1] for e in selected),
                                  max(e.ref.bbox[2] for e in selected),
                                  max(e.ref.bbox[3] for e in selected))
                result.elements = [item for item in result.elements if item.id not in set(targets[1:])]
                for value in targets[1:]:
                    elements.pop(value)
            elif len(targets) == 1 and isinstance(suggested.get("split_texts"), list):
                parts = suggested["split_texts"]
                if len(parts) < 2 or any(not isinstance(part, str) or not part.strip() for part in parts):
                    raise ValueError("拆分须提供至少两段文字")
                original = elements[targets[0]]
                # Make room before inserting parts: integer order collisions would
                # otherwise interleave the following paragraph with split results.
                for item in result.elements:
                    if item.ref.page == original.ref.page and item.reading_order > original.reading_order:
                        item.reading_order += len(parts) - 1
                original.raw_text = parts[0]
                original.normalized_text = parts[0]
                boundaries=result.chunk_policy.get("force_split_after_ids", [])
                preserve_end=original.id in boundaries
                last_id=original.id
                group=[original.id]
                for index, part in enumerate(parts[1:], 1):
                    new_id = stable_id(result.document_id, "split-element", proposal["id"], index)
                    extra = TextElement(new_id, type(original.ref)(result.document_id,
                                        original.ref.section_id, original.ref.page, original.ref.bbox),
                                        original.element_type, part, part,
                                        original.reading_order + index, original.confidence)
                    result.elements.append(extra)
                    elements[new_id] = extra
                    last_id=new_id
                    group.append(new_id)
                # Element splitting does not imply chunk splitting. An existing
                # explicit end boundary stays after the final replacement part.
                if preserve_end:
                    result.chunk_policy['force_split_after_ids']=[i for i in boundaries if i!=original.id]+[last_id]
                groups=result.chunk_policy.setdefault('element_split_groups',[])
                previous=next((ids for ids in groups if original.id in ids),None)
                if previous is not None:
                    at=previous.index(original.id);previous[at:at+1]=group
                else:
                    groups.append(group)
            else:
                raise ValueError("合并/拆分建议格式无效")
        elif kind == "table_structure":
            if len(targets) != 1 or targets[0] not in rows:
                raise ValueError("表格结构修正需要一个表格行")
            raw_cells = suggested.get("raw_cells")
            if not isinstance(raw_cells, list) or not raw_cells:
                raise ValueError("表格行修正缺少 raw_cells")
            row = rows[targets[0]]
            row.raw_cells = [str(value or "") for value in raw_cells]
            normalized = suggested.get("normalized_cells", raw_cells)
            if not isinstance(normalized, list) or len(normalized) != len(raw_cells):
                raise ValueError("normalized_cells 列数不一致")
            row.normalized_cells = [str(value or "") for value in normalized]
            table = next(table for table in result.tables if table.id == row.table_id)
            for cell in table.cells:
                if cell.row == row.row_index and cell.column < len(raw_cells):
                    cell.raw = row.raw_cells[cell.column]
                    cell.normalized = row.normalized_cells[cell.column]
            table.markdown = "\n".join("| " + " | ".join(str(value or "") for value in item.raw_cells) + " |"
                                       for item in table.rows)
            for key, values in (("raw_json", row.raw_cells),
                                ("normalized_json", row.normalized_cells)):
                matrix = getattr(table, key).get("rows")
                if isinstance(matrix, list) and row.row_index < len(matrix):
                    matrix[row.row_index] = values
        elif kind == "chunk_operation":
            from app.review.chunk_operations import prepare_operation
            directive = prepare_operation(suggested, mysql_url, source_document_id, id_map)
            result.chunk_policy.setdefault("chunk_operations", []).append(directive)
        elif kind == "chunk_boundary":
            excluded = suggested.get("exclude_element_ids", [])
            split_after = suggested.get("force_split_after_ids", [])
            if not isinstance(excluded, list) or not isinstance(split_after, list) or not (excluded or split_after):
                raise ValueError("分块边界建议需要排除或拆分的元素 ID")
            if any(value not in old_ids for value in [*excluded, *split_after]):
                raise ValueError("分块边界建议只能操作选中的来源元素")
            for key, values in (("exclude_element_ids", excluded), ("force_split_after_ids", split_after)):
                current = result.chunk_policy.setdefault(key, [])
                current.extend(id_map[value] for value in values if id_map[value] not in current)
        elif kind == "heading_assignment":
            if len(targets) != 1 or targets[0] not in elements:
                raise ValueError("标题归属需要一个文本元素")
            item = elements[targets[0]]
            if suggested.get("parent_id"):
                parent_id = id_map.get(suggested["parent_id"])
                if parent_id not in elements or parent_id == item.id:
                    raise ValueError("父级标题不属于来源版本或引用自身")
                parent = elements[parent_id]
                if "heading" not in parent.element_type and "title" not in parent.element_type:
                    raise ValueError("父级必须是标题元素")
                item.ref.section_id = parent.ref.section_id
                result.chunk_policy.setdefault("heading_parents", {})[item.id] = parent_id
            elif suggested.get("heading_text"):
                item.raw_text = _text(suggested, "heading_text")
                item.normalized_text = item.raw_text
                item.element_type = "subsection_heading"
            elif suggested.get("section_id") in id_map:
                item.ref.section_id = id_map[suggested["section_id"]]
            else:
                raise ValueError("标题建议需要 heading_text 或 section_id")
        elif kind == "summary_error":
            if len(proposal.get("chunk_ids") or []) != 1:
                raise ValueError("摘要修正需要一个摘要分块")
            key = _old_summary_key(mysql_url, proposal["chunk_ids"][0], source_document_id)
            result.summary_overrides[key] = _text(suggested)
        else:
            raise ValueError(f"无法执行的问题类型: {kind}")
    result.elements.sort(key=lambda item: (item.ref.page, item.reading_order, item.id))
