import os
import unittest
import uuid

from app.models import ParseResult, Section, SourceRef, ValuePair
from app.storage.mysql import ingest, initialize, parse_mysql_url


class MySQLConfigTests(unittest.TestCase):
    def test_parse_mysql_url(self):
        config = parse_mysql_url("mysql+pymysql://root:p%40ss@127.0.0.1:3306/medical_qa?charset=utf8mb4")
        self.assertEqual(config.user, "root")
        self.assertEqual(config.password, "p@ss")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.database, "medical_qa")
        self.assertEqual(config.charset, "utf8mb4")

    def test_reject_unsafe_database_name(self):
        with self.assertRaises(ValueError):
            parse_mysql_url("mysql://root:pass@localhost/bad-name")

            


@unittest.skipUnless(os.getenv("TEST_MYSQL_URL"), "Set TEST_MYSQL_URL for MySQL integration")
class MySQLIntegrationTests(unittest.TestCase):
    def test_migration_and_idempotent_insert(self):
        import json
        import pymysql

        mysql_url = os.environ["TEST_MYSQL_URL"]
        initialize(mysql_url)
        token = uuid.uuid4().hex
        document_id = token[:24]
        section_id = token[8:32]
        result = ParseResult(document_id, "synthetic.pdf", token * 2, "test-1",
                             ValuePair("测试药品" + token, "测试药品" + token), 1,
                             [Section(section_id, document_id, "【药品名称】", "药品名称", 1, 1,
                                      SourceRef(document_id, section_id, 1, (1, 1, 10, 10)), 1.0)],
                             [], [], [], [], [])
        result.brand_names = [ValuePair("测试商品名", "测试商品名")]
        result.approval_numbers = ["TEST" + token]
        result.revision_date = "2024-01-01"
        self.assertTrue(ingest(result, mysql_url))
        self.assertFalse(ingest(result, mysql_url))
        newer_token = uuid.uuid4().hex
        newer_document_id = newer_token[:24]
        newer_section_id = newer_token[8:32]
        newer = ParseResult(newer_document_id, "synthetic-v2.pdf", newer_token * 2, "test-1",
                            result.drug_name, 1,
                            [Section(newer_section_id, newer_document_id, "【药品名称】", "药品名称", 1, 1,
                                     SourceRef(newer_document_id, newer_section_id, 1, (1, 1, 10, 10)), 1.0)],
                            [], [], [], [], [])
        newer.brand_names = result.brand_names
        newer.approval_numbers = result.approval_numbers
        newer.revision_date = "2025-01-01"
        self.assertTrue(ingest(newer, mysql_url))

        # The same source file parsed by a newer parser is a new parse artifact.
        # It becomes active even when the leaflet itself has no newer revision date.
        reparsed_document_id = (token[4:] + "reparse")[:24]
        reparsed_section_id = (token[12:] + "reparse-section")[:24]
        renamed_generic = ValuePair("测试药品别名" + token, "测试药品别名" + token)
        reparsed = ParseResult(
            reparsed_document_id,
            newer.source_file,
            newer.file_hash,
            "test-2",
            renamed_generic,
            1,
            [Section(
                reparsed_section_id,
                reparsed_document_id,
                "【药品名称】",
                "药品名称",
                1,
                1,
                SourceRef(reparsed_document_id, reparsed_section_id, 1, (1, 1, 10, 10)),
                1.0,
            )],
            [], [], [], [], [],
        )
        reparsed.brand_names = result.brand_names
        reparsed.approval_numbers = result.approval_numbers
        self.assertTrue(ingest(reparsed, mysql_url))
        self.assertFalse(ingest(reparsed, mysql_url))

        # A later product with another approval number can still resolve to the
        # canonical drug through a previously learned generic-name alias.
        alias_product_token = uuid.uuid4().hex
        alias_product_document_id = alias_product_token[:24]
        alias_product_section_id = alias_product_token[8:32]
        alias_product = ParseResult(
            alias_product_document_id,
            "synthetic-alias-product.pdf",
            alias_product_token * 2,
            "test-1",
            renamed_generic,
            1,
            [Section(
                alias_product_section_id,
                alias_product_document_id,
                "【药品名称】",
                "药品名称",
                1,
                1,
                SourceRef(
                    alias_product_document_id,
                    alias_product_section_id,
                    1,
                    (1, 1, 10, 10),
                ),
                1.0,
            )],
            [], [], [], [], [],
        )
        alias_product.approval_numbers = ["TEST-ALIAS-" + alias_product_token]
        alias_product.revision_date = "2026-01-01"
        self.assertTrue(ingest(alias_product, mysql_url))

        config = parse_mysql_url(mysql_url)
        connection = pymysql.connect(host=config.host, port=config.port, user=config.user,
                                     password=config.password, database=config.database,
                                     charset=config.charset)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT generic_name_raw FROM drug WHERE generic_name_normalized=%s",
                               (result.drug_name.normalized,))
                self.assertEqual(cursor.fetchone()[0], result.drug_name.raw)
                cursor.execute("SELECT report_json FROM parse_report WHERE document_id=%s",
                               (result.document_id,))
                report = cursor.fetchone()[0]
                if isinstance(report, str):
                    report = json.loads(report)
                self.assertEqual(report["drug_name"]["raw"], result.drug_name.raw)
                cursor.execute(
                    "SELECT COUNT(*), SUM(is_parent) FROM chunk WHERE document_id=%s",
                    (result.document_id,),
                )
                chunk_count, parent_count = cursor.fetchone()
                self.assertGreater(chunk_count, 0)
                self.assertGreater(parent_count, 0)
                cursor.execute(
                    """SELECT vector_status FROM document_index_state
                       WHERE document_id=%s""",
                    (result.document_id,),
                )
                self.assertEqual(cursor.fetchone()[0], "PENDING")
                cursor.execute(
                    """SELECT d.id, d.version_status FROM document d
                       JOIN drug_product p ON p.id=d.product_id
                       WHERE d.id IN (%s,%s,%s) ORDER BY d.id""",
                    (result.document_id, newer.document_id, reparsed.document_id),
                )
                statuses = dict(cursor.fetchall())
                self.assertEqual(statuses[result.document_id], "SUPERSEDED")
                self.assertEqual(statuses[newer.document_id], "SUPERSEDED")
                self.assertEqual(statuses[reparsed.document_id], "ACTIVE")
                cursor.execute(
                    "SELECT supersedes_document_id FROM document WHERE id=%s",
                    (reparsed.document_id,),
                )
                self.assertEqual(cursor.fetchone()[0], newer.document_id)
                cursor.execute(
                    """SELECT alias_type, normalized_name FROM drug_alias
                       WHERE drug_id=(SELECT id FROM drug WHERE generic_name_normalized=%s)""",
                    (result.drug_name.normalized,),
                )
                aliases = set(cursor.fetchall())
                self.assertIn(("GENERIC", result.drug_name.normalized), aliases)
                self.assertIn(("GENERIC", renamed_generic.normalized), aliases)
                self.assertIn(("BRAND", "测试商品名"), aliases)
                cursor.execute(
                    "SELECT COUNT(*) FROM drug WHERE generic_name_normalized=%s",
                    (renamed_generic.normalized,),
                )
                self.assertEqual(cursor.fetchone()[0], 0)
                cursor.execute(
                    """SELECT dr.generic_name_normalized
                       FROM document AS doc
                       JOIN drug_product AS product ON product.id=doc.product_id
                       JOIN drug AS dr ON dr.id=product.drug_id
                       WHERE doc.id=%s""",
                    (alias_product.document_id,),
                )
                self.assertEqual(cursor.fetchone()[0], result.drug_name.normalized)
                cursor.execute(
                    """SELECT TABLE_COMMENT FROM information_schema.TABLES
                       WHERE TABLE_SCHEMA=%s AND TABLE_NAME='document'""",
                    (config.database,),
                )
                self.assertIn("说明书文档版本", cursor.fetchone()[0])
                cursor.execute(
                    """SELECT COLUMN_COMMENT FROM information_schema.COLUMNS
                       WHERE TABLE_SCHEMA=%s AND TABLE_NAME='table_cell'
                         AND COLUMN_NAME='normalized_value'""",
                    (config.database,),
                )
                self.assertIn("规范化单元格值", cursor.fetchone()[0])
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
