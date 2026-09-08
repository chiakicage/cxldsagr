import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer

from GR.heat import HeatPopulation
from GR.input_generator import (
    DEFAULT_TOKENIZER,
    InputGenerator,
    TextConfig,
    create_input_generator,
    main,
)
from GR.scheduling import ScheduleConfig


class HeatTests(unittest.TestCase):
    def test_dataset_counts_subset_and_industrial_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "heat.json"
            path.write_text(json.dumps({"0": [[4, 2], [9, 1]], "1": [[4, 3], [9, 7]]}))
            heat = HeatPopulation.load(path, num_users=0)
            self.assertEqual(heat.weights, {4: 5, 9: 8})
            a = HeatPopulation.load(path, num_users=1, seed=42)
            self.assertEqual(a.weights, HeatPopulation.load(path, num_users=1, seed=42).weights)
            path = root / "users.csv"
            path.write_text("user_id,pv_share\n42,0.2\n104,0.8\n")
            heat = HeatPopulation.load(path, industrial_field="pv_share", num_users=0)
            self.assertEqual(heat.weights, {42: 0.2, 104: 0.8})
            with self.assertRaisesRegex(ValueError, "only 2"):
                HeatPopulation.load(path, industrial_field="pv_share", num_users=3)

    def test_config_validation(self):
        for kwargs in (
            {"user_lengths": (0,)},
            {"user_probabilities": (float("nan"),) * 5},
            {"user_probabilities": (0.1,) * 5},
            {"history_cache_users": -1},
            {"max_input_tokens": 100},
        ):
            with self.assertRaises(ValueError):
                TextConfig(**kwargs)


@unittest.skipUnless(DEFAULT_TOKENIZER.is_file(), "local DeepSeek tokenizer unavailable")
class RuleTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))

    def generator(self, cfg=None, titles=None):
        return InputGenerator(
            HeatPopulation({42: 1, 99: 3}, {}),
            self.tokenizer,
            text_config=cfg,
            titles=titles,
            schedule_config=ScheduleConfig(sampling="sequential"),
        )

    def test_api_and_file_output_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heat = root / "users.csv"
            heat.write_text("user_id,pv_share\n42,0.2\n99,0.8\n")
            gen = create_input_generator(
                heat_source="industrial",
                heat_path=str(heat),
                num_users=0,
                text_material="synthetic",
                tokenizer=self.tokenizer,
                text_config=TextConfig(
                    user_lengths=(4096,),
                    user_probabilities=(1,),
                    item_lengths=(1024,),
                    item_probabilities=(1,),
                ),
                schedule_config=ScheduleConfig(sampling="sequential"),
            )
            rows = list(gen.iter_generate(4))
            self.assertEqual(set(root.iterdir()), {heat})
            self.assertEqual(rows[2]["visit_index"], 1)
            self.assertEqual(gen.for_user(42)["input_ids"], rows[0]["input_ids"])
            output = root / "requests.jsonl"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--heat-source",
                        "industrial",
                        "--heat-path",
                        str(heat),
                        "--num-users",
                        "0",
                        "--text-material",
                        "synthetic",
                        "--tokenizer",
                        str(DEFAULT_TOKENIZER.parent),
                        "--user-lengths",
                        "4096",
                        "--item-lengths",
                        "1024",
                        "--sampling",
                        "sequential",
                        "--count",
                        "4",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual([json.loads(line) for line in output.read_text().splitlines()], rows)
            meta = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
            self.assertEqual(meta["stats"]["repeat_requests"], 2)
            self.assertTrue(output.with_suffix(".jsonl.users.jsonl").is_file())

    def test_all_length_buckets_are_exact_and_readable(self):
        # One user per fixed bucket makes coverage independent of random draws.
        for history in (4096, 8192, 16384, 32768, 65536):
            for other in (1024, 2048, 4096):
                with self.subTest(user=history, item=other):
                    cfg = TextConfig(
                        user_lengths=(history,),
                        user_probabilities=(1,),
                        item_lengths=(other,),
                        item_probabilities=(1,),
                    )
                    gen = self.generator(cfg)
                    row = gen.for_user(42)
                    ids = row["input_ids"]
                    self.assertEqual(len(ids), history + other)
                    self.assertEqual(row["user_tokens"], history)
                    self.assertEqual(row["item_tokens"], other)
                    self.assertEqual(
                        gen.tokenizer.decode(ids, skip_special_tokens=False), row["prompt"]
                    )
                    self.assertIn("User profile U42.", row["prompt"])
                    self.assertIn("Record 1:", row["prompt"])
                    self.assertIn("Candidate pool for visit 0:", row["prompt"])

    def test_revisit_prefix_candidates_and_reproducibility(self):
        cfg = TextConfig(
            user_lengths=(4096,),
            user_probabilities=(1,),
            item_lengths=(1024,),
            item_probabilities=(1,),
            history_cache_users=0,
        )
        gen = self.generator(cfg, {i: f"Real product title {i}" for i in range(30)})
        rows = list(gen.iter_generate(4))
        self.assertEqual(rows, list(gen.iter_generate(4)))
        for old, new in ((rows[0], rows[2]), (rows[1], rows[3])):
            length = old["stable_prefix_tokens"]
            self.assertEqual(old["input_ids"][:length], new["input_ids"][:length])
            self.assertEqual(old["history_sha256"], new["history_sha256"])
            self.assertNotEqual(old["candidate_item_ids"], new["candidate_item_ids"])
            actual = 0
            for a, b in zip(old["input_ids"], new["input_ids"]):
                if a != b:
                    break
                actual += 1
            self.assertEqual(new["common_prefix_tokens"], actual)
            self.assertEqual(new["previous_request_id"], old["task_id"])
            self.assertIn("Real product title", new["prompt"])
        other = self.generator(replace(cfg, history_cache_users=8))
        self.assertEqual(gen.lengths_for_user(42), other.lengths_for_user(42))

    def test_too_small_budget_fails_explicitly(self):
        cfg = TextConfig(
            user_lengths=(4096,),
            user_probabilities=(1,),
            item_lengths=(32,),
            item_probabilities=(1,),
        )
        with self.assertRaisesRegex(ValueError, "mandatory text"):
            self.generator(cfg).for_user(42)


if __name__ == "__main__":
    unittest.main()
