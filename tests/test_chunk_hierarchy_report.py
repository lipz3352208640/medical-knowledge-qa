import unittest
from unittest.mock import MagicMock,patch
from app.storage.report import chunk_hierarchy

class ChunkHierarchyReportTests(unittest.TestCase):
    def test_hierarchy_is_bound_to_run_document_and_includes_cross_page_links(self):
        connection=MagicMock();cursor=connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value=('doc',)
        cursor.fetchall.return_value=[('p','topic_parent','父块',1,3,None,1,'{}'),('child','paragraph','跨页子块',3,3,'p',0,'{}')]
        with patch('app.storage.report._connect',return_value=connection):
            result=chunk_hierarchy('mysql+pymysql://u:p@localhost/db','run')
        self.assertEqual(result['chunks'][1]['parent_id'],'p')
        self.assertFalse(result['chunks'][1]['is_parent'])
        self.assertEqual(cursor.execute.call_args_list[0].args[1],('run',))
        self.assertEqual(cursor.execute.call_args_list[1].args[1],('doc',))
        self.assertNotIn('page_start<=',cursor.execute.call_args_list[1].args[0])
        connection.close.assert_called_once()
    def test_missing_run_returns_none_and_closes_connection(self):
        connection=MagicMock();connection.cursor.return_value.__enter__.return_value.fetchone.return_value=None
        with patch('app.storage.report._connect',return_value=connection):self.assertIsNone(chunk_hierarchy('mysql+pymysql://u:p@localhost/db','missing'))
        connection.close.assert_called_once()
