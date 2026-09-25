import hashlib
import json
import os
from pathlib import Path
import unittest

from app.parser.layout import stable_id
from app.chunker.parent_child import build_chunks
from app.storage.mysql import ingest, initialize, parse_mysql_url
from app.storage.retrieval import load_summary_sources


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    Path(x) for x in json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
]


def _stored_drug_and_product_ids(mysql_url: str, document_id: str) -> tuple[str, str]:
    import pymysql

    config = parse_mysql_url(mysql_url)
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT product.drug_id, document.product_id
                   FROM document
                   JOIN drug_product AS product ON product.id=document.product_id
                   WHERE document.id=%s""",
                (document_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise AssertionError(f"Document was not stored: {document_id}")
            return row
    finally:
        connection.close()


class SampleManifestTests(unittest.TestCase):
    def test_five_readable_pdfs_and_distinct_hashes(self):
        self.assertEqual(len(SAMPLES), 5)
        hashes = []
        for path in SAMPLES:
            self.assertTrue(path.is_file(), str(path))
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF-"), str(path))
            hashes.append(hashlib.sha256(data).hexdigest())
        self.assertEqual(len(set(hashes)), 5)

    def test_document_identity_tracks_parser_version(self):
        payload = SAMPLES[0].read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(stable_id(digest, "0.1.0"), stable_id(digest, "0.1.0"))
        self.assertNotEqual(stable_id(digest, "0.1.0"), stable_id(digest, "0.1.1"))


@unittest.skipUnless(os.getenv("TEST_MYSQL_URL"), "Set TEST_MYSQL_URL for PDF integration")
class PDFIntegrationTests(unittest.TestCase):
    def test_all_samples_process_chunk(self):
        import tempfile
        from app.parser.pdf import parse_pdf

        mysql_url = os.environ["TEST_MYSQL_URL"]
        initialize(mysql_url)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            for path in SAMPLES:
                with self.subTest(path=path.name):
                    result = parse_pdf(path, temporary)
                    ingest(result, mysql_url)
                    drug_id, product_id = _stored_drug_and_product_ids(
                        mysql_url, result.document_id
                    )
                    chunks = build_chunks(result, drug_id, product_id)

                    self.assertTrue(chunks)
                    self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
                    self.assertEqual(
                        [chunk.id for chunk in chunks],
                        [chunk.id for chunk in build_chunks(result, drug_id, product_id)],
                    )

                    parents = {chunk.id: chunk for chunk in chunks if chunk.is_parent}
                    children = [chunk for chunk in chunks if not chunk.is_parent]
                    summaries = [
                        chunk for chunk in children
                        if chunk.content_type in {"section_summary", "topic_summary"}
                    ]
                    evidence_children = [
                        chunk for chunk in children
                        if chunk.content_type not in {"section_summary", "topic_summary"}
                    ]
                    by_id = {chunk.id: chunk for chunk in chunks}
                    section_ids = {section.id for section in result.sections}
                    source_ids = (
                        {element.id for element in result.elements}
                        | {row.id for table in result.tables for row in table.rows}
                        | {image.id for image in result.images}
                    )
                    self.assertTrue(parents)
                    self.assertTrue(children)
                    self.assertEqual(
                        children,
                        sorted(children, key=lambda chunk: (chunk.reading_order, chunk.id)),
                    )
                    self.assertEqual(
                        sum(chunk.content_type == "table_row" for chunk in evidence_children),
                        sum(len(table.rows) for table in result.tables),
                    )

                    expected_parent_types = {
                        "paragraph": "topic_parent",
                        "list_item": "topic_parent",
                        "table_row": "row_group_parent",
                        "image": "image_parent",
                    }
                    for chunk in chunks:
                        self.assertEqual(chunk.document_id, result.document_id)
                        self.assertIn(chunk.section_id, section_ids)
                        self.assertTrue(chunk.raw_text)
                        self.assertTrue(chunk.retrieval_text)
                        self.assertTrue(chunk.split_reason)
                        self.assertEqual(
                            chunk.content_hash,
                            hashlib.sha256(chunk.raw_text.encode("utf-8")).hexdigest(),
                        )
                        self.assertEqual(chunk.metadata["drug_id"], drug_id)
                        self.assertEqual(chunk.metadata["product_id"], product_id)
                        self.assertEqual(chunk.metadata["section_path"], chunk.section_path)
                        self.assertEqual(chunk.metadata["raw_text"], chunk.raw_text)
                        self.assertTrue(chunk.source_element_ids)
                        self.assertEqual(
                            chunk.metadata["source_element_ids"], chunk.source_element_ids
                        )
                        self.assertGreaterEqual(chunk.page_start, 1)
                        self.assertLessEqual(chunk.page_start, chunk.page_end)
                        self.assertLessEqual(chunk.page_end, result.page_count)
                        self.assertTrue(chunk.refs)
                        self.assertEqual(
                            (chunk.page_start, chunk.page_end),
                            (min(ref.page for ref in chunk.refs),
                             max(ref.page for ref in chunk.refs)),
                        )
                        for ref in chunk.refs:
                            self.assertEqual(ref.document_id, result.document_id)
                            self.assertEqual(ref.section_id, chunk.section_id)
                            self.assertLess(ref.bbox[0], ref.bbox[2])
                            self.assertLess(ref.bbox[1], ref.bbox[3])
                        if chunk.page_start == chunk.page_end:
                            self.assertIsNotNone(chunk.bbox)
                        else:
                            self.assertIsNone(chunk.bbox)

                    for summary in summaries:
                        self.assertIsNone(summary.parent_id)
                        self.assertEqual(summary.metadata["index_channel"], "summary")
                        self.assertEqual(summary.metadata["evidence_role"], "navigation")
                        self.assertTrue(summary.metadata["source_chunk_ids"])
                        self.assertTrue(
                            set(summary.metadata["source_chunk_ids"])
                            <= {chunk.id for chunk in evidence_children}
                        )
                        for ids in summary.metadata["key_point_sources"].values():
                            self.assertTrue(ids)
                            self.assertTrue(set(ids) <= set(summary.metadata["source_chunk_ids"]))
                        self.assertEqual(
                            [source.id for source in load_summary_sources(mysql_url, summary.id)],
                            summary.metadata["source_chunk_ids"],
                        )

                    for child in evidence_children:
                        self.assertIn(child.parent_id, parents)
                        self.assertEqual(
                            parents[child.parent_id].content_type,
                            expected_parent_types[child.content_type],
                        )
                        self.assertTrue(set(child.source_element_ids) <= source_ids)
                        self.assertTrue(set(child.overlap_source_ids) <= by_id.keys())
                        if child.lead_in_id:
                            self.assertIn(child.lead_in_id, source_ids | set(by_id))
                        if child.previous_chunk_id:
                            self.assertEqual(by_id[child.previous_chunk_id].next_chunk_id, child.id)
                        if child.next_chunk_id:
                            self.assertEqual(by_id[child.next_chunk_id].previous_chunk_id, child.id)

                    for parent in parents.values():
                        if parent.parent_id:
                            self.assertIn(parent.parent_id, parents)

    def test_all_samples_provenance_and_repeatable_identity(self):
        import tempfile
        from app.parser.pdf import parse_pdf

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            for path in SAMPLES:
                with self.subTest(path=path.name):
                    result = parse_pdf(path, temporary)
                    self.assertGreater(result.page_count, 0)
                    self.assertTrue(result.sections)
                    self.assertTrue(result.elements)
                    self.assertFalse(result.drug_name.raw.startswith("通用名称"))
                    self.assertEqual(
                        result.document_id, parse_pdf(path, temporary).document_id
                    )
                    section_ids = {section.id for section in result.sections}

                    def assert_ref(ref):
                        self.assertEqual(ref.document_id, result.document_id)
                        self.assertIn(ref.section_id, section_ids)
                        self.assertGreaterEqual(ref.page, 1)
                        self.assertLessEqual(ref.page, result.page_count)
                        self.assertEqual(len(ref.bbox), 4)
                        self.assertLess(ref.bbox[0], ref.bbox[2])
                        self.assertLess(ref.bbox[1], ref.bbox[3])

                    for element in result.elements:
                        assert_ref(element.ref)
                    for table in result.tables:
                        self.assertTrue(
                            table.image_paths
                            and table.markdown
                            and table.raw_json
                            and table.normalized_json
                        )
                        self.assertTrue(
                            all(
                                Path(image_path).is_file()
                                for image_path in table.image_paths
                            )
                        )
                        for ref in table.refs:
                            assert_ref(ref)
                        for cell in table.cells:
                            assert_ref(cell.ref)
                        for row in table.rows:
                            self.assertTrue(row.refs)
                            self.assertEqual(
                                len(row.raw_cells), len(row.normalized_cells)
                            )
                            for ref in row.refs:
                                assert_ref(ref)
                    for image in result.images:
                        assert_ref(image.ref)
                        self.assertTrue(Path(image.path).is_file())
                    if "坎地沙坦" in path.name:
                        self.assertIn(
                            "必洛斯", [name.normalized for name in result.brand_names]
                        )
                        self.assertTrue(result.approval_numbers)
                        spanning = [
                            t
                            for t in result.tables
                            if [ref.page for ref in t.refs] == [5, 6]
                        ]
                        self.assertEqual(len(spanning), 1)
                        self.assertIn(
                            "非甾体抗炎药", spanning[0].rows[-2].normalized_cells[0]
                        )
                    if "达可替尼" in path.name:
                        self.assertEqual(
                            [name.normalized for name in result.brand_names],
                            ["多泽润", "vizimpro"],
                        )
                        self.assertEqual(result.revision_date, "2025-06-28")
                        self.assertTrue(
                            any(
                                image.image_type == "chemical_structure"
                                for image in result.images
                            )
                        )
                        self.assertTrue(
                            any(
                                image.image_type == "chart" and image.caption
                                for image in result.images
                            )
                        )
                        third_table = result.tables[2]
                        self.assertTrue(third_table.title.startswith("表3."))
                        self.assertIsNotNone(third_table.title_ref)
                        self.assertEqual(third_table.title_ref.page, 3)
                        self.assertEqual(third_table.refs[0].page, 4)
                    if "厄贝沙坦" in path.name:
                        self.assertEqual(result.blank_pages, [16])
                        self.assertFalse(result.failed_pages)


if __name__ == "__main__":
    unittest.main()
