"""YAML configuration behavior shared by CLI, API and tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config


class ConfigTests(unittest.TestCase):
    def test_yaml_takes_precedence_and_converts_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("mysql_url: yaml-db\nquery_rewrite_enabled: false\nembedding_dimensions: 1024\nqdrant_api_key: null\n", encoding="utf-8")
            with patch.object(config, "CONFIG_PATH", path), patch.dict(os.environ, {"MYSQL_URL": "env-db", "QDRANT_API_KEY": "env-key"}):
                self.assertEqual(config.setting("MYSQL_URL"), "yaml-db")
                self.assertEqual(config.setting("QUERY_REWRITE_ENABLED"), "false")
                self.assertEqual(config.setting("EMBEDDING_DIMENSIONS"), "1024")
                self.assertIsNone(config.setting("QDRANT_API_KEY"))

    def test_test_config_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.test.yaml"
            path.write_text("test_mysql_url: test-db\nrun_live_e2e: true\n", encoding="utf-8")
            with patch.object(config, "TEST_CONFIG_PATH", path):
                self.assertEqual(config.setting("TEST_MYSQL_URL", test=True), "test-db")
                self.assertEqual(config.setting("RUN_LIVE_E2E", test=True), "true")

    def test_missing_yaml_uses_legacy_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config, "CONFIG_PATH", Path(directory) / "missing.yaml"), patch.dict(os.environ, {"MYSQL_URL": "env-db"}):
                self.assertEqual(config.setting("MYSQL_URL"), "env-db")

    def test_example_password_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("mysql_url: mysql+pymysql://root:CHANGE_PASSWORD@localhost/db\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "example mysql_url"):
                config.load_config(path)


if __name__ == "__main__":
    unittest.main()
