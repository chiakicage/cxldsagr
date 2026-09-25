"""Fingerprint the runtime and experiment code used by NOSA measurements."""

import hashlib
from pathlib import Path


def source_hashes(*extra_sources):
    root = Path(__file__).resolve().parents[3]
    paths = {root / "GR/analysis/heat_curves.csv"}
    for directory in (
        "models/nosa",
        "layers",
        "cache",
        "executor",
        "serving",
        "GR",
        "operators",
        "operators/sm90",
        "experiments/nosa_gr_65536_1024/src",
    ):
        paths.update((root / directory).glob("*.py"))
    paths.update(Path(path).resolve() for path in extra_sources)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
