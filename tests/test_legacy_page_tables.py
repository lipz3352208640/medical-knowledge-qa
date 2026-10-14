import unittest
from unittest.mock import patch, MagicMock
from app.storage.report import parse_page


class LegacyPageTableTests(unittest.TestCase):
    def test_real_tables_are_returned_without_element_report_table_links(self):
        cursor=MagicMock()
        def execute(sql, args):
            cursor.fetchone.return_value = None
            cursor.fetchall.return_value = []
            if 'SELECT * FROM page_report' in sql:
                cursor.fetchone.return_value=('run',2,600,800,1,1,0,'[]')
            elif 'chunk_manifest_hash' in sql:
                cursor.fetchone.return_value=('index',)
            elif 'SELECT DISTINCT table_id' in sql:
                cursor.fetchall.return_value=[('persisted-table',)]
        cursor.execute.side_effect=execute
        conn=MagicMock();conn.cursor.return_value.__enter__.return_value=cursor
        table={'table_id':'persisted-table','cells':[{'id':'cell','page':2,'row':0,'column':0,'rowspan':1,'colspan':2,'text':'真实合并内容'}]}
        with patch('app.storage.report._connect',return_value=conn), patch('app.storage.report.parse_mysql_url'), patch('app.review.table_audit.read_table',return_value=table) as read:
            page=parse_page('test-url','run',2)
        self.assertEqual(page['elements'],[])
        self.assertEqual(page['tables'],[table])
        self.assertEqual(page['tables'][0]['cells'][0]['colspan'],2)
        read.assert_called_once_with('test-url','run','persisted-table')
