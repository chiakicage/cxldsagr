import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.gr_cache_serving.src.report_dense_window import load_pairs, main
from experiments.gr_cache_serving.tests import test_fixed_budget as fixtures


class DenseWindowReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for schedule in ("layer_end", "attention_window"):
            parent = self.root / schedule
            parent.mkdir()
            for population in (64, 128):
                path = fixtures.FixedBudgetTests().make_run(parent, population, "dense_prefetch")
                file = path / "summary.json"
                s = json.loads(file.read_text())
                s["environment"] = {"torch": "same"}
                s["runner_provenance"].update(
                    echo_revision="same",
                    echo_patch_sha256="same",
                    dense_prefetch={"schedule": schedule},
                )
                file.write_text(json.dumps(s))
                self.paths.append(path)

    def test_complete_pairs_and_incomplete_rejected(self):
        self.assertEqual(len(load_pairs(self.paths)), 4)
        with self.assertRaisesRegex(ValueError, "both schedules"):
            load_pairs(self.paths[:-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            load_pairs(self.paths + self.paths[:1])

    def test_mismatched_input_or_environment_rejected(self):
        file = self.paths[-1] / "summary.json"
        original = json.loads(file.read_text())
        for field, value in (
            ("environment", {"torch": "different"}),
            ("trace_metadata", original["trace_metadata"] | {"requests_sha256": "different"}),
        ):
            with self.subTest(field=field):
                file.write_text(json.dumps(original | {field: value}))
                with self.assertRaises(ValueError):
                    load_pairs(self.paths)
        file.write_text(json.dumps(original))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "requires report environment")
    def test_report_renders_and_refuses_overwrite(self):
        output = self.root / "report"
        argv = [*(str(p) for p in self.paths), "--output-dir", str(output)]
        with patch("builtins.print"):
            main(argv)
        self.assertTrue((output / "overview.png").is_file())
        self.assertEqual(len(json.loads((output / "manifest.json").read_text())["run_ids"]), 4)
        with self.assertRaises(SystemExit):
            main(argv)
