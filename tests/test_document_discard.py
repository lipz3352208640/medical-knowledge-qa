import unittest
import uuid
from app.config import setting
from app.models import ParseResult,Section,SourceRef,TextElement,ValuePair
from app.storage.mysql import _connect,ingest,initialize,parse_mysql_url
from app.storage.jobs import discard_pending_document,publish_import
from app.storage.report import list_documents
from app.web.api import create_app
from fastapi.testclient import TestClient


@unittest.skipUnless(setting('TEST_MYSQL_URL',test=True),'Dedicated QA test database required')
class DocumentDiscardTests(unittest.TestCase):
    def test_discard_preserves_data_hides_default_and_blocks_publication(self):
        url=setting('TEST_MYSQL_URL',test=True)
        self.assertTrue(parse_mysql_url(url).database.endswith('_test'))
        initialize(url);token=uuid.uuid4().hex;doc=token[:24];section=token[8:32]
        ref=SourceRef(doc,section,1,(20,20,180,40))
        result=ParseResult(doc,'discard-test.pdf','a'*64,'discard-'+token,
                           ValuePair('DiscardDrug'+token,'DiscardDrug'+token),1,
                           [Section(section,doc,'Warnings','Warnings',1,1,ref,1.0)],
                           [TextElement(token[4:28],ref,'paragraph','Preserved','Preserved',1,1.0)],[],[],[],[])
        ingest(result,url,publish=False)
        job=uuid.uuid4().hex
        conn=_connect(parse_mysql_url(url))
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO rag_job (id,kind,document_id,source_file,status) VALUES (%s,'IMPORT',%s,'discard-test.pdf','READY')",(job,doc))
                cur.execute("UPDATE document_index_state SET vector_status='READY' WHERE document_id=%s",(doc,))
                cur.execute('SELECT COUNT(*) FROM parse_run WHERE document_id=%s',(doc,));runs=cur.fetchone()[0]
                cur.execute('SELECT COUNT(*) FROM chunk WHERE document_id=%s',(doc,));chunks=cur.fetchone()[0]
            conn.commit()
            with self.assertRaises(ValueError):discard_pending_document(url,doc,'admin',' ')
            client=TestClient(create_app(mysql_url=url,internal_token='discard-test'))
            route=f'/internal/v1/documents/{doc}/discard'
            self.assertEqual(client.post(route,json={'actor':'admin','reason':'old test'}).status_code,401)
            response=client.post(route,json={'actor':'admin','reason':'old test'},headers={'x-service-token':'discard-test'})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['version_status'],'WITHDRAWN')
            self.assertNotIn(doc,[x['document_id'] for x in list_documents(url,limit=10000)])
            self.assertIn(doc,[x['document_id'] for x in list_documents(url,limit=10000,include_withdrawn=True)])
            with self.assertRaises(ValueError):publish_import(url,job)
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM parse_run WHERE document_id=%s',(doc,));self.assertEqual(cur.fetchone()[0],runs)
                cur.execute('SELECT COUNT(*) FROM chunk WHERE document_id=%s',(doc,));self.assertEqual(cur.fetchone()[0],chunks)
                cur.execute('SELECT actor,reason FROM document_discard_log WHERE document_id=%s',(doc,));self.assertEqual(cur.fetchone(),('admin','old test'))
                cur.execute("UPDATE document SET version_status='ACTIVE' WHERE id=%s",(doc,))
            conn.commit()
            with self.assertRaises(ValueError):discard_pending_document(url,doc,'admin','must not discard active')
            with conn.cursor() as cur:
                cur.execute("UPDATE document SET version_status='PENDING' WHERE id=%s",(doc,))
                cur.execute("UPDATE rag_job SET status='RUNNING' WHERE id=%s",(job,))
            conn.commit()
            with self.assertRaises(ValueError):discard_pending_document(url,doc,'admin','running')
        finally:
            # Only the isolated fixture is reset; no production records are touched.
            with conn.cursor() as cur:
                cur.execute("UPDATE rag_job SET status='FAILED' WHERE id=%s",(job,))
                cur.execute("UPDATE document SET version_status='WITHDRAWN' WHERE id=%s",(doc,))
            conn.commit();conn.close()
