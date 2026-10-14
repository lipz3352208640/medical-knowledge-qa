from copy import deepcopy
import unittest
import test_chunker
from app.models import TableCell,TableRow,SourceRef
from app.review.table_audit import cell_id,grid_hash,validate_grid,grid_revision_matches,preview_cells,validate_correction,apply_cell_correction
from app.chunker.parent_child import build_chunks
from app.review.chunk_operations import apply_chunk_operations
from app.chunker.source_mapping import map_chunk_sources
from unittest.mock import patch
from app.review.corrections import apply_proposals


def grid():
    return [dict(id=cell_id('table-1',r,c),table_id='table-1',row=r,column=c,rowspan=1,colspan=1,page=2 if r<2 else 3,bbox=[c*50,r*20,(c+1)*50,(r+1)*20],text=f'{r}:{c}',precise=True,span_known=True) for r in range(3) for c in range(2)]


class TableAuditTests(unittest.TestCase):
    def test_two_disjoint_merges_replay_together_without_overwriting_in_either_order(self):
        original=grid()
        evidence={'table_id':'table-1','cells':original,'grid_revision':grid_hash(original),'chunk_cells':{}}
        proposals=[{'issue_type':'table_review','element_ids':[c['id'] for c in original if c['row']==r],'suggested':{'source_run_id':'run','table_id':'table-1','grid_revision':evidence['grid_revision'],'operation':'MERGE_CELLS','cell_ids':[c['id'] for c in original if c['row']==r],'text':f'merged row {r}'}} for r in (1,2)]
        for ordered in (proposals,list(reversed(proposals))):
            result=self.make_result();table=result.tables[0];table.id='new-table'
            with patch('app.review.table_audit.read_table',return_value=evidence),patch('app.storage.report.chunk_hierarchy',return_value={'chunks':[]}):
                apply_proposals(result,{'table-1':'new-table'},ordered,'unused-test-url','old-doc')
            cells=table.raw_json['audit_cells']
            self.assertEqual(len(cells),4)
            for row in (1,2):
                merged=next(c for c in cells if c['row']==row)
                self.assertEqual((merged['colspan'],merged['text']),(2,f'merged row {row}'))
                self.assertEqual(merged['id'],cell_id('new-table',row,0))
            self.assertEqual(original,grid())
            chunks=build_chunks(result,'drug','product')
            for row in (1,2):self.assertTrue(any(c.content_type=='table_row' and f'merged row {row}' in c.content for c in chunks))
    def test_adjacent_rectangle_merge_and_source_unchanged(self):
        cells=grid();original=deepcopy(cells)
        result=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[2]['id'],cells[3]['id']],'text':'merged'})
        self.assertEqual(cells,original);self.assertEqual(len(result),5)
        merged=next(c for c in result if c['id']==cells[2]['id'])
        self.assertEqual((merged['rowspan'],merged['colspan'],merged['bbox']),(1,2,[0,20,100,40]))
        self.assertEqual(len(validate_grid(result)),6)

    def test_nonrectangle_cross_page_unknown_span_and_missing_mapping_rejected(self):
        cells=grid()
        for ids in ([cells[0]['id'],cells[3]['id']],[cells[2]['id'],cells[4]['id']],['missing']):
            with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',{'cell_ids':ids})
        cells[0]['span_known']=False
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[1]['id']]})

    def test_split_exact_cover_without_inventing_bbox(self):
        cells=grid();merged=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[1]['id']]})
        parts=[dict(row=0,column=c,rowspan=1,colspan=1,text=f'part {c}') for c in range(2)]
        result=preview_cells(merged,'SPLIT_CELL',{'cell_ids':[cells[0]['id']],'parts':parts})
        self.assertTrue(all(c['bbox'] is None and not c['precise'] for c in result if c['row']==0))
        parts[1]['column']=3
        with self.assertRaises(ValueError):preview_cells(merged,'SPLIT_CELL',{'cell_ids':[cells[0]['id']],'parts':parts})

    def test_manual_horizontal_merge_absorbs_legacy_null_slot_not_vertical_inheritance(self):
        cells=grid();cells[2]['text']='< 0.0001'
        cells[3].update(text='',span_known=False,merge_placeholder=True,bbox=None,precise=False)
        value={'cell_ids':[cells[2]['id'],cells[3]['id']],'text':'< 0.0001'}
        result=preview_cells(cells,'MERGE_CELLS',value)
        merged=next(c for c in result if c['id']==cells[2]['id'])
        self.assertEqual(merged['colspan'],2);self.assertTrue(merged['span_known']);self.assertIsNone(merged['bbox'])
        self.assertFalse(merged['precise']);self.assertEqual(len(result),5)
        cells[3]['merge_placeholder']=False
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',value)
        cells[3]['merge_placeholder']=True;cells[3]['text']='inherited'
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',value)
        cells=grid();cells[2].update(text='',span_known=False,merge_placeholder=True)
        result=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[2]['id']]})
        self.assertEqual(next(c for c in result if c['id']==cells[0]['id'])['rowspan'],2)
        cells[2]['merge_placeholder']=False
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[2]['id']]})

    def test_inherited_null_vertical_merge_requires_exact_selected_anchor(self):
        cells=grid();cells[2].update(text='',span_known=False,merge_placeholder=False,merge_anchor_row=0)
        result=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[2]['id']]})
        self.assertEqual(next(c for c in result if c['id']==cells[0]['id'])['rowspan'],2)
        cells[2]['merge_anchor_row']=9
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[2]['id']]})
        cells[2].update(merge_anchor_row=0,text='inherited content')
        with self.assertRaises(ValueError):preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[0]['id'],cells[2]['id']]})

    def test_old_helper_field_revisions_accept_unchanged_evidence_only(self):
        original=grid()
        early=grid_hash(original)
        with_placeholder=[{**c,'merge_placeholder':False} for c in original]
        middle=grid_hash(with_placeholder)
        current=[{**c,'merge_anchor_row':None} for c in with_placeholder]
        newest=grid_hash(current)
        revisions=(early,middle,newest)
        for revision in revisions:self.assertTrue(grid_revision_matches(current,revision))
        for key,value in [('text','changed'),('bbox',[1,1,40,18]),('rowspan',2),('id','changed-cell'),('precise',False),('table_id','changed-table'),('span_known',False)]:
            changed=deepcopy(current);changed[0][key]=value
            for revision in revisions:self.assertFalse(grid_revision_matches(changed,revision),(key,revision))
        self.assertFalse(grid_revision_matches(current[:-1],early))
        self.assertFalse(grid_revision_matches(current,None))
        self.assertFalse(grid_revision_matches(current,'old'))
        table,chunks=self.evidence();old_revision=table['grid_revision']
        table['cells']=[{**c,'merge_placeholder':False,'merge_anchor_row':None} for c in table['cells']]
        table['grid_revision']=grid_hash(table['cells'])
        validate_correction(table,chunks,dict(operation='UPDATE_CELL_TEXT',cell_ids=[table['cells'][2]['id']],text='corrected',grid_revision=old_revision))
        table['cells'][2]['text']='new original text'
        with self.assertRaises(ValueError):validate_correction(table,chunks,dict(operation='UPDATE_CELL_TEXT',cell_ids=[table['cells'][2]['id']],text='corrected',grid_revision=old_revision))

    def evidence(self):
        cells=grid();a={'chunk_id':'a','parent_id':'p','is_parent':False};b={'chunk_id':'b','parent_id':'p','is_parent':False};p={'chunk_id':'p','is_parent':True}
        table=dict(table_id='table-1',cells=cells,grid_revision=grid_hash(cells),section_id='section',title='title',chunk_cells={'a':[c['id'] for c in cells[2:4]],'b':[c['id'] for c in cells[4:]],'p':[c['id'] for c in cells]})
        return table,[p,a,b]

    def test_stale_grid_cross_table_parent_cycle_empty_and_duplicate_cells(self):
        table,chunks=self.evidence();base=dict(operation='MOVE_ELEMENT_TO_CHUNK',source_chunk_id='a',target_chunk_id='b',cell_ids=[table['cells'][2]['id']],grid_revision=table['grid_revision'])
        validate_correction(table,chunks,base)
        for invalid in ({'grid_revision':'old'},{'target_chunk_id':'other-table'},{'cell_ids':[]},{'cell_ids':table['chunk_cells']['a']},{'cell_ids':[base['cell_ids'][0]]*2}):
            with self.assertRaises(ValueError):validate_correction(table,chunks,{**base,**invalid})
        with self.assertRaises(ValueError):validate_correction(table,chunks,{**base,'operation':'CHANGE_CHUNK_PARENT','target_chunk_id':'p'})

    def test_assignment_across_parents_stays_inside_table_and_preserves_source(self):
        table,chunks=self.evidence();chunks[2]['parent_id']='another-parent'
        value=dict(operation='MOVE_ELEMENT_TO_CHUNK',source_chunk_id='a',target_chunk_id='b',cell_ids=[table['cells'][2]['id']],grid_revision=table['grid_revision'])
        validate_correction(table,chunks,value)
        validate_correction(table,chunks,{**value,'operation':'ADJUST_CHUNK_RANGE','mode':'move'})
        with self.assertRaises(ValueError):validate_correction(table,chunks,{**value,'cell_ids':table['chunk_cells']['a']})
        with self.assertRaises(ValueError):validate_correction(table,chunks,{**value,'operation':'ADJUST_CHUNK_RANGE','mode':'merge'})
        del table['chunk_cells']['b']
        with self.assertRaises(ValueError):validate_correction(table,chunks,value)

    def test_bbox_requires_actual_page_bounds_and_rejects_noop(self):
        table,chunks=self.evidence();cell=table['cells'][2]
        value=dict(operation='UPDATE_CELL_BBOX',cell_ids=[cell['id']],bbox=[1,21,49,39],grid_revision=table['grid_revision'])
        with self.assertRaises(ValueError):validate_correction(table,chunks,value)
        table['page_sizes']={'2':{'width':100,'height':100}}
        validate_correction(table,chunks,value)
        for bbox in ([0,20,50,40],[1,21,101,39],[-1,21,49,39],[0,0,float('nan'),40]):
            with self.assertRaises(ValueError):validate_correction(table,chunks,{**value,'bbox':bbox})

    def test_merged_cell_correction_rebuilds_actual_row_content_and_precise_sources(self):
        result=self.make_result();table=result.tables[0];cells=grid()
        merged=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[2]['id'],cells[3]['id']],'text':'combined'})
        apply_cell_correction(result,table,{'validated_cells':merged})
        chunks=build_chunks(result,'drug','product')
        child=next(c for c in chunks if c.content_type=='table_row' and 'combined' in c.content)
        self.assertIn(cell_id(table.id,1,0),child.metadata['cell_ids'])
        self.assertNotIn(cell_id(table.id,1,1),child.metadata['cell_ids'])
        self.assertIn('combined',child.content)

    def make_result(self):
        result=test_chunker.ParentChildChunkerTests().make_result();table=result.tables[0]
        cells=grid();table.cells=[TableCell(c['text'],c['text'],c['row'],c['column'],SourceRef(result.document_id,table.section_id,c['page'],tuple(c['bbox']))) for c in cells]
        table.rows=[TableRow(f'row-{r}',table.id,r,[f'{r}:0',f'{r}:1'],[f'{r}:0',f'{r}:1'],[SourceRef(result.document_id,table.section_id,2 if r==1 else 3,(0,r*20,100,(r+1)*20))],'') for r in (1,2)]
        table.normalized_json={'rows':[['H1','H2'],['1:0','1:1'],['2:0','2:1']]}
        return result

    def test_context_retains_raw_and_assignment_rebuilds_exact_cell_sources(self):
        result=self.make_result();chunks=build_chunks(result,'drug','product')
        children=[c for c in chunks if c.content_type=='table_row'];source,target=children
        signatures=lambda c:dict(kind=c.content_type,content=c.content,ids=c.source_element_ids.copy())
        original=source.content
        context=dict(operation='UPDATE_CHUNK_CONTEXT',source_chunk_id='old-a',table_id='table-1',signatures={'source_chunk_id':signatures(source)},context_sources=[dict(kind='column_header',ref_id=cell_id('table-1',0,1))])
        result.chunk_policy['chunk_operations']=[context]
        apply_chunk_operations(result,chunks,phase='metadata')
        self.assertEqual(source.content,original);self.assertEqual(source.retrieval_text,'0:1\n'+original)
        move=dict(operation='MOVE_ELEMENT_TO_CHUNK',source_chunk_id='old-a',target_chunk_id='old-b',table_id='table-1',signatures={'source_chunk_id':signatures(source),'target_chunk_id':signatures(target)},cell_ids=[cell_id('table-1',1,1)])
        result.chunk_policy['chunk_operations']=[move]
        apply_chunk_operations(result,chunks,phase='metadata')
        self.assertEqual(source.source_element_ids,[cell_id('table-1',1,0)])
        self.assertIn(cell_id('table-1',1,1),target.source_element_ids)
        segments=map_chunk_sources(result,[source])[source.id]
        self.assertEqual(segments[0].element_id,cell_id('table-1',1,0));self.assertEqual(segments[0].bbox,(0,20,50,40))

    def test_split_preserves_parent_grid_order_and_leaves_other_parent_untouched(self):
        result=self.make_result();chunks=build_chunks(result,'drug','product')
        source,target=[c for c in chunks if c.content_type=='table_row']
        parent=next(c for c in chunks if c.id==source.parent_id)
        unrelated=next(c for c in chunks if c.id==target.parent_id)
        original=deepcopy(unrelated)
        signature=dict(kind=source.content_type,content=source.content,ids=source.source_element_ids.copy())
        result.chunk_policy['chunk_operations']=[dict(operation='ADJUST_CHUNK_RANGE',mode='split',source_chunk_id='old-source',table_id='table-1',cell_ids=[cell_id('table-1',1,0)],signatures={'source_chunk_id':signature})]
        apply_chunk_operations(result,chunks,phase='metadata')
        self.assertEqual(parent.source_element_ids,[cell_id('table-1',1,0),cell_id('table-1',1,1)])
        self.assertEqual(parent.content,'1:0\n1:1')
        self.assertEqual(unrelated.content,original.content)
        self.assertEqual(unrelated.source_element_ids,original.source_element_ids)

    def test_later_parent_target_survives_earlier_cell_assignment_rebuild(self):
        result=self.make_result();chunks=build_chunks(result,'drug','product')
        source,target=[c for c in chunks if c.content_type=='table_row']
        parent=next(c for c in chunks if c.id==target.parent_id)
        signature=lambda c:dict(kind=c.content_type,content=c.content,ids=c.source_element_ids.copy())
        source_sig,target_sig,parent_sig=signature(source),signature(target),signature(parent)
        move=dict(operation='MOVE_ELEMENT_TO_CHUNK',table_id='table-1',source_chunk_id='old-source',target_chunk_id='old-child-target',cell_ids=[cell_id('table-1',1,1)],signatures={'source_chunk_id':source_sig,'target_chunk_id':target_sig})
        reparent=dict(operation='CHANGE_CHUNK_PARENT',table_id='table-1',source_chunk_id='old-source',target_chunk_id='old-parent-target',signatures={'source_chunk_id':source_sig,'target_chunk_id':parent_sig})
        result.chunk_policy['chunk_operations']=[move,reparent]
        apply_chunk_operations(result,chunks,phase='metadata')
        self.assertEqual(source.parent_id,parent.id)
        self.assertEqual(source.source_element_ids,[cell_id('table-1',1,0)])
        self.assertIn(cell_id('table-1',1,1),target.source_element_ids)
        self.assertEqual(set(parent.source_element_ids),set(source.source_element_ids+target.source_element_ids))
        self.assertIn('old-parent-target',parent.metadata['audit_source_chunk_ids'])

    def test_cell_edit_rebuilds_rows_grid_and_keeps_original_pdf_refs(self):
        result=self.make_result();table=result.tables[0];refs=deepcopy(table.refs)
        cells=grid();cells[2]['text']='corrected'
        apply_cell_correction(result,table,{'validated_cells':cells})
        self.assertEqual(table.rows[0].raw_cells[0],'corrected');self.assertEqual(table.refs,refs)
        self.assertEqual(table.raw_json['audit_cells'][2]['text'],'corrected')
        rebuilt=build_chunks(result,'drug','product')
        self.assertTrue(any(c.content_type=='table_row' and 'corrected' in c.content for c in rebuilt))

class TableNormalizationRegressionTests(unittest.TestCase):
    def test_text_edit_preserves_inherited_category_and_chunk_group(self):
        from app.chunker.parent_child import _table_row_groups
        for category in ('腹泻', '皮肤不良反应'):
            with self.subTest(category=category):
                result=TableAuditTests().make_result();table=result.tables[0]
                table.cells[2].raw=category;table.cells[2].normalized=category
                table.cells[4].raw=None;table.cells[4].normalized=category
                table.cells[4].inherited_from_row=1
                table.normalized_json['rows'][1][0]=category
                table.normalized_json['rows'][2][0]=category
                for row in table.rows:
                    row.raw_cells[0]=category if row.row_index==1 else None
                    row.normalized_cells[0]=category
                cells=grid();cells[2]['text']=category;cells[4]['text']=''
                cells[3]['text']='corrected dose text'
                expected=[(key,[row.row_index for row in rows]) for key,rows in _table_row_groups(table,table.title)]
                apply_cell_correction(result,table,{'validated_cells':cells})
                self.assertEqual(table.rows[1].raw_cells[0],'')
                self.assertEqual(table.rows[1].normalized_cells[0],category)
                self.assertEqual(table.cells[4].inherited_from_row,1)
                self.assertEqual([(key,[row.row_index for row in rows]) for key,rows in _table_row_groups(table,table.title)],expected)
                chunks=build_chunks(result,'drug','product')
                children=[c for c in chunks if c.content_type=='table_row' and c.metadata.get('table_id')==table.id]
                self.assertEqual(len(children),2)
                self.assertEqual(len({c.parent_id for c in children}),1)
                self.assertTrue(all(c.section_path[-1]==category for c in children))
                self.assertTrue(any('corrected dose text' in c.content for c in children))

    def test_edit_preserves_unrelated_normalized_rows_and_markdown(self):
        result=TableAuditTests().make_result();table=result.tables[0]
        table.cells[-1].raw=None;table.cells[-1].normalized='inherited'
        table.normalized_json['rows'][2][1]='inherited'
        cells=grid();cells[-1]['text']='';cells[2]['text']='changed'
        apply_cell_correction(result,table,{'validated_cells':cells})
        self.assertEqual(table.normalized_json['rows'][2][1],'inherited')
        self.assertEqual(table.rows[1].normalized_cells[1],'inherited')
        self.assertIn('| --- | --- |',table.markdown)
        self.assertIn('changed',table.markdown)

    def test_reparent_removes_only_the_empty_source_group_after_all_operations(self):
        result=TableAuditTests().make_result();chunks=build_chunks(result,'drug','product')
        source,target=[c for c in chunks if c.content_type=='table_row']
        original_parent=source.parent_id;parent=next(c for c in chunks if c.id==target.parent_id)
        signature=lambda c:dict(kind=c.content_type,content=c.content,ids=c.source_element_ids.copy())
        result.chunk_policy['chunk_operations']=[dict(operation='CHANGE_CHUNK_PARENT',table_id='table-1',source_chunk_id='old-source',target_chunk_id='old-parent-target',signatures={'source_chunk_id':signature(source),'target_chunk_id':signature(parent)})]
        apply_chunk_operations(result,chunks,phase='metadata')
        self.assertEqual(source.parent_id,parent.id)
        self.assertFalse(any(c.id==original_parent for c in chunks))
        self.assertTrue(any(c.id==parent.id for c in chunks))

    def test_empty_merge_slots_do_not_add_line_breaks(self):
        cells=grid();cells[3]['text']=''
        merged=preview_cells(cells,'MERGE_CELLS',{'cell_ids':[cells[2]['id'],cells[3]['id']]})
        self.assertEqual(next(c for c in merged if c['id']==cells[2]['id'])['text'],cells[2]['text'])

if __name__=='__main__':unittest.main()
