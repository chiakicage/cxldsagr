import csv
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from GR.heat import DEFAULT_HEAT_CURVES, HeatPopulation
from GR.input_generator import create_input_generator, main


class CurveHeatTests(unittest.TestCase):
    def write_curve(self, root, points):
        path = root / "curves.csv"
        path.write_text(
            "dataset,field,user_fraction,traffic_fraction\n"
            + "".join(f"test,heat,{x},{y}\n" for x, y in points)
        )
        return path

    def test_interpolation_conservation_and_seeded_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_curve(Path(directory), [(0.25, 0.5), (0.5, 0.75), (1, 1)])
            state = random.getstate()
            heat = HeatPopulation.from_curve(path, dataset="test", field="heat", num_users=8)
            self.assertEqual(sorted(heat.weights), list(range(8)))
            self.assertEqual(
                sorted(heat.weights.values(), reverse=True),
                [0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625],
            )
            self.assertEqual(heat.metadata["selected_weight_sum"], 1)
            self.assertEqual(heat.metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            again = HeatPopulation.from_curve(path, dataset="test", field="heat", num_users=8)
            self.assertEqual(heat, again)
            other = HeatPopulation.from_curve(
                path, dataset="test", field="heat", num_users=8, seed=7
            )
            self.assertNotEqual(heat.weights, other.weights)
            self.assertEqual(sorted(heat.weights.values()), sorted(other.weights.values()))
            single = HeatPopulation.from_curve(path, dataset="test", field="heat", num_users=1)
            self.assertEqual(single.weights, {0: 1})
            self.assertEqual(random.getstate(), state)

    def test_invalid_curves_and_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for points in (
                [(0.5, 0.7)],  # Missing endpoint.
                [(0.5, 0.7), (0.5, 0.8), (1, 1)],  # Duplicate rank.
                [(0.5, 0.7), (0.75, 0.6), (1, 1)],  # Decreasing traffic.
                [(0.5, 0.7), (0.75, 0.7), (1, 1)],  # Zero weight interval.
                [(0.5, 0.2), (1, 1)],  # Increasing ranked heat.
                [(0.5, float("nan")), (1, 1)],
                [(float("inf"), 0.7), (1, 1)],
                [(0.5, 1.1), (1, 1)],
            ):
                with self.subTest(points=points), self.assertRaises(ValueError):
                    HeatPopulation.from_curve(
                        self.write_curve(root, points), dataset="test", field="heat"
                    )
            path = self.write_curve(root, [(0, 0), (1, 1)])
            self.assertEqual(
                HeatPopulation.from_curve(path, dataset="test", field="heat", num_users=4).weights,
                dict.fromkeys(range(4), 0.25),
            )
            with self.assertRaisesRegex(ValueError, "no curve"):
                HeatPopulation.from_curve(path)
            for size in (0, -1, True, 1.5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    HeatPopulation.from_curve(path, num_users=size)
            path.write_text("dataset,field\ntest,heat\n")
            with self.assertRaisesRegex(ValueError, "missing curve columns"):
                HeatPopulation.from_curve(path)

    def test_bundled_profiles_match_summary_concentration(self):
        summary = DEFAULT_HEAT_CURVES.with_name("heat_summary.csv")
        with summary.open() as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            with self.subTest(dataset=row["dataset"], field=row["field"]):
                heat = HeatPopulation.from_curve(
                    dataset=row["dataset"], field=row["field"], num_users=10000
                )
                ranked = sorted(heat.weights.values(), reverse=True)
                self.assertTrue(all(math.isfinite(w) and w > 0 for w in ranked))
                self.assertAlmostEqual(math.fsum(ranked), 1)
                for pct in (1, 5, 10, 20, 50):
                    self.assertAlmostEqual(
                        100 * math.fsum(ranked[: 100 * pct]),
                        float(row[f"top_{pct}pct_traffic_pct"]),
                        delta=0.05,  # Percentage points; source ranks use ceil.
                    )
                n = len(ranked)
                gini = (n + 1 - 2 * math.fsum(i * w for i, w in enumerate(ranked, 1))) / n
                self.assertAlmostEqual(gini, float(row["gini"]), delta=0.002)

    def test_factory_and_cli_defaults_need_no_raw_data(self):
        # Only exercise resource loading and metadata here. Exact request encoding
        # is covered separately with the real NOSA tokenizer when available.
        specials = ["[UNK]", "<|im_start|>", "<|im_end|>"]
        tokenizer = Tokenizer(WordLevel(dict(zip(specials, range(len(specials)))), "[UNK]"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gen = create_input_generator(
                tokenizer=tokenizer, data_root=root / "absent", num_users=7
            )
            self.assertIsNone(gen.titles)
            self.assertEqual(len(gen.population.weights), 7)
            self.assertEqual(gen.population.metadata["dataset"], "beauty")
            industrial = create_input_generator(
                tokenizer=tokenizer, curve_dataset="industrial_100k", num_users=7
            )
            self.assertEqual(industrial.population.metadata["field"], "pv_share")
            path = root / "tokenizer.json"
            tokenizer.save(str(path))
            output = root / "requests.jsonl"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--tokenizer",
                        str(path),
                        "--data-root",
                        str(root / "absent"),
                        "--num-users",
                        "7",
                        "--count",
                        "0",
                        "--output",
                        str(output),
                    ]
                )
            meta = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
            self.assertEqual(meta["heat_source"], "curve")
            self.assertEqual(meta["text_material"], "synthetic")
            self.assertIsNone(meta["text_catalog_path"])
            self.assertEqual(meta["heat"], gen.population.metadata)
            profiles = [
                json.loads(line)
                for line in output.with_suffix(".jsonl.users.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(profiles), 7)
            self.assertAlmostEqual(sum(p["normalized_probability"] for p in profiles), 1)


if __name__ == "__main__":
    unittest.main()
