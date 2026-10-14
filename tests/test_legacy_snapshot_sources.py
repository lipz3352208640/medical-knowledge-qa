import unittest
from app.storage.report import legacy_source_elements


class LegacySourceTests(unittest.TestCase):
    def test_tables_and_captions_are_actual_sources_not_new_content(self):
        row=('row','table',2,1,'[0,10,100,30]',[],['甲',''],['甲',''],'说明')
        image=('image',1,'[1,2,3,4]','图注',.9,'说明')
        result=legacy_source_elements(1,set(),[row],[image])
        self.assertEqual([e['element_type'] for e in result],['table_row','image_caption'])
        self.assertEqual(result[0]['raw_text'],'甲；')
        self.assertEqual(result[0]['table_cells']['raw'],['甲',''])
        self.assertEqual(result[0]['bbox'],[0,10,100,30])
        self.assertEqual(legacy_source_elements(1,{'row','image'},[row],[image]),[])
        self.assertEqual(legacy_source_elements(2,set(),[row],[image]),[])

    def test_multi_page_refs_use_actual_page_position(self):
        row=('row','table',1,1,[0,1,2,3],[{'page':1,'bbox':[0,1,2,3]},{'page':2,'bbox':[4,5,6,7]}],['乙'],['乙'],'说明')
        self.assertEqual(legacy_source_elements(2,set(),[row],[])[0]['bbox'],[4,5,6,7])
