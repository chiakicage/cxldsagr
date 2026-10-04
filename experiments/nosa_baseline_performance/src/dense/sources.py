"""Fingerprint the runtime and experiment code used by NOSA measurements."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NATIVE_SOURCE_SUFFIXES = {".cu", ".cuh", ".c", ".cpp", ".h", ".hpp"}


def source_hashes(*extra_sources):
    root = ROOT
    paths = {root / "GR/analysis/heat_curves.csv"}
    for directory in (
        "models/nosa",
        "layers",
        "cache",
        "executor",
        "serving",
        "GR",
        "operators",
        "experiments/nosa_baseline_performance/src/dense",
    ):
        paths.update((root / directory).glob("*.py"))
    for directory in ("operators/nosa", "operators/common"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "tests" not in path.relative_to(root / directory).parts
            and path.suffix in {".py", *NATIVE_SOURCE_SUFFIXES}
        )
    paths.update(Path(path).resolve() for path in extra_sources)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
