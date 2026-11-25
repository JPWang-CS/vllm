"""CPU-only checks for model-scoped repository layout and runtime isolation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from fullnet.config import resolve  # noqa: E402
from fullnet.execution import plan  # noqa: E402
from fullnet.store import experiment, inside, read_json  # noqa: E402
import yaml


class RepositoryLayoutTest(unittest.TestCase):
    def setUp(self):
        fixture_parent = FRAMEWORK_ROOT / "tmp" / "runs"
        fixture_parent.mkdir(parents=True, exist_ok=True)
        self.fixture_root = Path(
            tempfile.mkdtemp(prefix="repository-layout-", dir=str(fixture_parent))
        )
        self.addCleanup(shutil.rmtree, self.fixture_root)

    def write_fixture(self):
        model = self.fixture_root / "models" / "fixture-model"
        (model / "changes" / "replacement").mkdir(parents=True)
        (model / "model.yaml").write_text(
            "name: fixture-model\nfamily: fixture\nplatforms: [cpu-test]\n",
            encoding="utf-8",
        )
        (model / "README.md").write_text("fixture model\n", encoding="utf-8")
        (model / "changes" / "replacement" / "change.yaml").write_text(
            "platforms: [cpu-test]\nrequires: []\nconflicts: []\n"
            "verify_argv: [python, -c, pass]\n",
            encoding="utf-8",
        )
        dataset = self.fixture_root / "dataset.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "id": "sample-1",
                    "prompt": "reply",
                    "outputs": ["reply"],
                    "actual_chat_tokens": 1,
                    "output_tokens": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        config = {
            "schema_version": 1,
            "model": "fixture-model",
            "platform": "cpu-test",
            "changes": ["replacement"],
            "service": {
                "argv": [sys.executable, "-B", "-c", "pass"],
                "base_url": "http://127.0.0.1:18399/v1",
                "served_model": "fixture-model",
            },
            "evaluation": {
                "dataset": str(dataset),
                "scope": "layout-fixture",
                "metric": "exact_match",
                "context_limit": 8,
            },
        }
        config_path = self.fixture_root / "fixture-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_repository_root_and_models_are_present(self):
        for relative in ("src/fullnet", "models", "scripts", "tests"):
            self.assertTrue((FRAMEWORK_ROOT / relative).is_dir(), relative)
        model_dirs = [path for path in (FRAMEWORK_ROOT / "models").iterdir() if path.is_dir()]
        self.assertGreater(len(model_dirs), 0)
        for model_dir in model_dirs:
            manifest = model_dir / "model.yaml"
            self.assertTrue(manifest.is_file(), model_dir)
            self.assertEqual(yaml.safe_load(manifest.read_text(encoding="utf-8"))["name"], model_dir.name)

    def test_plan_keeps_runtime_record_under_declared_model(self):
        config_path = self.write_fixture()
        config = resolve(config_path, self.fixture_root)
        run = plan(self.fixture_root, config, "precision-001")

        expected = self.fixture_root / "models" / "fixture-model" / "experiments" / "precision-001"
        self.assertEqual(run, expected)
        self.assertEqual(experiment(self.fixture_root, "fixture-model", "precision-001"), expected)
        self.assertEqual(read_json(run / "config.json")["model"], "fixture-model")
        self.assertEqual(read_json(run / "changes.json")[0]["name"], "replacement")
        self.assertTrue((run / "dataset.jsonl").is_file())

    def test_model_and_path_escape_are_rejected(self):
        config_path = self.write_fixture()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"] = "other-model"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model.yaml is required"):
            resolve(config_path, self.fixture_root)

        with self.assertRaisesRegex(ValueError, "escapes project"):
            inside(self.fixture_root, self.fixture_root.parent / "outside")
        with self.assertRaises(ValueError):
            experiment(self.fixture_root, "../fixture-model", "run")


if __name__ == "__main__":
    unittest.main()
