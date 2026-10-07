"""Compare matrix source snapshots with one exact hardware-query deadline change."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.run_contract import digest

HARDWARE_SOURCE = "experiments/deepseek_v32_mfu/src/profile_hardware.py"
_PROBE_LINES = {
    timeout: (
        "    completed = subprocess.run(command, check=True, capture_output=True, "
        f"text=True, timeout={timeout})\n"
    ).encode()
    for timeout in (20, 120)
}


def hardware_probe_variant(path, digest_file=digest):
    path = Path(path).resolve(strict=True)
    expected = digest_file(path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"Hardware probe source changed while reading: {path}")
    lines = data.splitlines(keepends=True)
    matches = [timeout for timeout, line in _PROBE_LINES.items() if lines.count(line) == 1]
    if len(matches) != 1 or any(lines.count(line) > 1 for line in _PROBE_LINES.values()):
        raise ValueError(f"Hardware probe lacks the unique exact supported timeout line: {path}")
    timeout = matches[0]
    lines[lines.index(_PROBE_LINES[timeout])] = _PROBE_LINES[20]
    canonical = b"".join(lines)
    return {
        "path": str(path),
        "sha256": actual,
        "nvidia_smi_timeout_seconds": timeout,
        "canonical_timeout20_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def source_snapshot(manifest, archive, *, current_hardware=None, digest_file=digest):
    """Keep raw identities exact; canonicalize only the one complete deadline line."""
    archive = Path(archive)
    for name, expected in manifest.items():
        if digest_file(archive / name) != expected:
            raise ValueError(f"Archived source changed: {name}")
    comparison = dict(manifest)
    probe = None
    current = None
    if HARDWARE_SOURCE in manifest:
        probe = hardware_probe_variant(archive / HARDWARE_SOURCE, digest_file)
        if probe["sha256"] != manifest[HARDWARE_SOURCE]:
            raise ValueError(f"Archived source changed: {HARDWARE_SOURCE}")
        comparison[HARDWARE_SOURCE] = probe["canonical_timeout20_sha256"]
        if current_hardware is not None:
            current = hardware_probe_variant(current_hardware, digest_file)
            if current["canonical_timeout20_sha256"] != probe["canonical_timeout20_sha256"]:
                raise ValueError("Hardware probe differs beyond the exact timeout20-to-120 change")
    return {
        "source_identity_sha256": identity_digest(manifest),
        "comparison_source_sha256": comparison,
        "hardware_probe": probe,
        "current_hardware_probe": current,
    }


def compare_source_snapshots(snapshots):
    if not snapshots or any(
        item["comparison_source_sha256"] != snapshots[0]["comparison_source_sha256"]
        for item in snapshots[1:]
    ):
        raise ValueError("Execution sources differ beyond the exact hardware-probe timeout change")
    variants = {}
    archived_hashes = set()
    for snapshot in snapshots:
        if snapshot["hardware_probe"]:
            archived_hashes.add(snapshot["hardware_probe"]["sha256"])
        for key in ("hardware_probe", "current_hardware_probe"):
            item = snapshot[key]
            if item is None:
                continue
            variant = variants.setdefault(
                item["sha256"],
                {name: value for name, value in item.items() if name != "path"} | {"paths": []},
            )
            if item["path"] not in variant["paths"]:
                variant["paths"].append(item["path"])
    return {
        "rule": "Exact source manifests, except one complete nvidia-smi subprocess timeout line of 20 or 120 seconds; no other byte changes",
        "comparison_source_identity_sha256": identity_digest(
            snapshots[0]["comparison_source_sha256"]
        ),
        "original_source_identity_sha256": sorted(
            {item["source_identity_sha256"] for item in snapshots}
        ),
        "hardware_probe_source": HARDWARE_SOURCE if variants else None,
        "matrix_hardware_probe_deadline_differs": len(archived_hashes) > 1,
        "archived_or_current_hardware_probe_deadline_differs": len(variants) > 1,
        "hardware_probe_variants": [variants[key] for key in sorted(variants)],
    }
