import copy
import errno
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.gr_cache_serving.src import workload
from experiments.gr_cache_serving.src.workload import text_config_kwargs, write_workload


def request(uid=7, visit=0, task=0, timestamp=0.0, previous=None, interval=None):
    return {
        "model": "deepseek_v32",
        "user_id": uid,
        "visit_index": visit,
        "task_id": task,
        "timestamp": timestamp,
        "previous_request_id": previous,
        "revisit_interval_s": interval,
        "input_ids": [10, 11, uid, 12, 20 + visit, 21],
        "instruction_tokens": 2,
        "stable_prefix_tokens": 4,
        "candidate_suffix_tokens": 2,
        "total_input_tokens": 6,
        "user_tokens": 2,
        "item_tokens": 4,
        "history_token_span": [2, 4],
        "candidate_token_span": [4, 6],
    }


class FakeGenerator:
    def __init__(self, rows):
        self.prefix_ids = [10, 11]
        self.population = SimpleNamespace(weights={7: 3.0, 8: 1.0})
        self.rows = rows

    def iter_generate(self, count):
        yield from self.rows


class WorkloadTests(unittest.TestCase):
    def write(self, output, rows, count=None):
        return write_workload(
            FakeGenerator(rows),
            output_dir=output,
            count=len(rows) if count is None else count,
            prefix_tokens=4,
            candidate_tokens=2,
            generator_metadata={"tokenizer_sha256": "test", "heat": {"sha256": "curve"}},
        )

    def test_budget_mapping_uses_encoded_instruction_length(self):
        for instruction in (2, 17, 31):
            result = text_config_kwargs(65536, 1024, instruction)
            self.assertEqual(result["user_lengths"], (65536 - instruction,))
            self.assertEqual(result["item_lengths"], (1024 + instruction,))
            self.assertEqual(result["max_input_tokens"], 66560)
        for values in ((2, 8, 2), (0, 8, 2), (8, 0, 2), (8, 2, 0), (True, 2, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                text_config_kwargs(*values)

    def test_stream_digests_counts_and_unvisited_users(self):
        rows = [request(), request(visit=1, task=1, timestamp=2.0, previous=0, interval=2.0)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output" / "data" / "run"
            metadata = self.write(output, rows)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"requests.jsonl", "users.jsonl", "metadata.json", "requests.sha256"},
            )
            raw = (output / "requests.jsonl").read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(metadata["requests_sha256"], digest)
            self.assertEqual(
                (output / "requests.sha256").read_text(), f"{digest}  requests.jsonl\n"
            )
            written = [json.loads(line) for line in raw.splitlines()]
            prefix_digest = hashlib.sha256(b"[10,11,7,12]").hexdigest()
            self.assertEqual(written[0]["stable_prefix_sha256"], prefix_digest)
            self.assertEqual(written[1]["stable_prefix_sha256"], prefix_digest)
            self.assertNotEqual(written[0]["input_ids"], written[1]["input_ids"])
            users = [json.loads(line) for line in (output / "users.jsonl").read_text().splitlines()]
            self.assertEqual(users[0]["request_count"], 2)
            self.assertEqual(users[0]["stable_prefix_sha256"], prefix_digest)
            self.assertEqual(users[0]["normalized_heat_weight"], 0.75)
            self.assertEqual(users[1]["request_count"], 0)
            self.assertIsNone(users[1]["stable_prefix_sha256"])
            self.assertEqual(
                users[0]["generator_metadata_sha256"], metadata["generator_metadata_sha256"]
            )
            self.assertFalse(metadata["serving_executed"])
            self.assertEqual(metadata["stats"]["observed_users"], 1)
            self.assertEqual(metadata["stats"]["revisit_requests"], 1)
            self.assertEqual(metadata["stats"]["population_users"], 2)
            self.assertEqual(metadata["stats"]["total_input_tokens"], 12)

    def test_invalid_boundaries_do_not_publish_or_leave_staging(self):
        changes = (
            {"stable_prefix_tokens": 3},
            {"candidate_suffix_tokens": 3},
            {"history_token_span": [1, 4]},
            {"candidate_token_span": [3, 6]},
            {"user_tokens": 3},
            {"item_tokens": 3},
            {"input_ids": [10, 11, 7, 12]},
            {"input_ids": [10, 11, 7, 12, -1, 21]},
            {"input_ids": [10, 99, 7, 12, 20, 21]},
            {"visit_index": 1},
            {"task_id": 4},
            {"timestamp": float("nan")},
            {"previous_request_id": 0},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(ValueError):
                    self.write(root / "output" / "data" / "bad", [{**request(), **change}])
                self.assertEqual(list(root.iterdir()), [])

    def test_revisit_requires_identical_prefix_and_progression(self):
        for change in (
            {"input_ids": [10, 11, 99, 12, 21, 21]},
            {"visit_index": 2},
            {"previous_request_id": 8},
            {"revisit_interval_s": 9.0},
            {"timestamp": -1.0},
        ):
            rows = [request(), request(visit=1, task=1, timestamp=2.0, previous=0, interval=2.0)]
            rows[1].update(change)
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(ValueError):
                    self.write(root / "run", rows)
                self.assertEqual(list(root.iterdir()), [])

    def test_count_mismatch_does_not_publish(self):
        for count in (1, 3):
            rows = [request(), request(uid=8, task=1, timestamp=1.0)]
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(ValueError):
                    self.write(root / "run", rows, count=count)
                self.assertEqual(list(root.iterdir()), [])

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                self.write(output, [request()])
            self.assertEqual(sentinel.read_text(), "keep")

    def test_target_created_at_publication_is_not_replaced(self):
        for kind in ("empty_directory", "nonempty_directory", "file", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "run"
                original_publish = workload._publish_directory
                target_inode = []

                def racing_publish(
                    source,
                    destination,
                    kind=kind,
                    root=root,
                    target_inode=target_inode,
                    original_publish=original_publish,
                ):
                    if kind.endswith("directory"):
                        destination.mkdir()
                        if kind == "nonempty_directory":
                            (destination / "keep.txt").write_text("keep")
                    elif kind == "file":
                        destination.write_text("keep")
                    else:
                        destination.symlink_to(root / "missing")
                    target_inode.append(destination.lstat().st_ino)
                    original_publish(source, destination)

                with (
                    patch.object(workload, "_publish_directory", side_effect=racing_publish),
                    self.assertRaises(FileExistsError),
                ):
                    self.write(output, [request()])
                self.assertEqual(output.lstat().st_ino, target_inode[0])
                self.assertEqual(list(root.iterdir()), [output])
                if kind == "empty_directory":
                    self.assertEqual(list(output.iterdir()), [])
                elif kind == "nonempty_directory":
                    self.assertEqual((output / "keep.txt").read_text(), "keep")
                elif kind == "file":
                    self.assertEqual(output.read_text(), "keep")
                else:
                    self.assertEqual(output.readlink(), root / "missing")

    def test_shared_parent_created_concurrently_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared"
            output = shared / "data" / "run"
            original_mkdir = Path.mkdir
            created_by_other = []

            def racing_mkdir(path, *args, **kwargs):
                if path == shared and not created_by_other:
                    original_mkdir(path)
                    created_by_other.append(path.stat().st_ino)
                return original_mkdir(path, *args, **kwargs)

            with patch.object(Path, "mkdir", new=racing_mkdir):
                self.write(output, [request()])
            self.assertEqual(shared.stat().st_ino, created_by_other[0])
            self.assertTrue((output / "metadata.json").is_file())

    def test_publication_failure_preserves_shared_parents_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "shared" / "data" / "run"
            with (
                patch.object(workload, "_publish_directory", side_effect=OSError(errno.EIO, "I/O")),
                self.assertRaises(OSError),
            ):
                self.write(output, [request()])
            self.assertTrue(output.parent.is_dir())
            self.assertEqual(list(output.parent.iterdir()), [])
            self.assertEqual(list(root.iterdir()), [root / "shared"])

    def test_unavailable_renameat2_fails_without_unsafe_fallback(self):
        with (
            patch.object(workload.sys, "platform", "linux"),
            patch.object(workload.ctypes, "CDLL", return_value=SimpleNamespace()),
            patch.object(Path, "rename", side_effect=AssertionError("unsafe fallback")),
            self.assertRaisesRegex(OSError, "libc does not expose renameat2"),
        ):
            workload._publish_directory(Path("unused_source"), Path("unused_destination"))
        with (
            patch.object(workload.sys, "platform", "darwin"),
            self.assertRaisesRegex(OSError, "requires Linux"),
        ):
            workload._publish_directory(Path("unused_source"), Path("unused_destination"))

    def test_unsupported_renameat2_kernel_or_filesystem_fails(self):
        for code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            with self.subTest(code=code):
                native = Mock(return_value=-1)
                with (
                    patch.object(workload.sys, "platform", "linux"),
                    patch.object(
                        workload.ctypes, "CDLL", return_value=SimpleNamespace(renameat2=native)
                    ),
                    patch.object(workload.ctypes, "get_errno", return_value=code),
                    patch.object(Path, "rename", side_effect=AssertionError("unsafe fallback")),
                    self.assertRaisesRegex(OSError, "RENAME_NOREPLACE") as caught,
                ):
                    workload._publish_directory(Path("unused_source"), Path("unused_destination"))
                self.assertEqual(caught.exception.errno, code)

    def test_failure_while_streaming_does_not_publish(self):
        def failing_rows():
            yield request()
            raise RuntimeError("generation failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.write(root / "data" / "run", failing_rows(), count=2)
            self.assertEqual(list(root.iterdir()), [])

    def test_reproducible_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [request(), request(uid=8, task=1, timestamp=1.0)]
            self.write(root / "a", rows)
            self.write(root / "b", copy.deepcopy(rows))
            for path in (root / "a").iterdir():
                self.assertEqual(path.read_bytes(), (root / "b" / path.name).read_bytes())

    def test_help_requires_only_standard_library(self):
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, "-S", "-m", "experiments.gr_cache_serving.src.workload", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prefix-tokens", result.stdout)


if __name__ == "__main__":
    unittest.main()
