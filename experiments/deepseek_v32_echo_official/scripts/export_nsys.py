"""Export an experiment NSYS capture with an explicit artifact binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    if not source.is_relative_to(OUTPUT_ROOT.resolve()) or source.suffix != ".nsys-rep":
        parser.error("input must be an .nsys-rep inside this experiment's output")
    target = source.with_suffix(".sqlite")
    receipt = source.with_suffix(".export.json")
    stdout = source.with_suffix(".export.stdout.log")
    stderr = source.with_suffix(".export.stderr.log")
    if any(path.exists() for path in (target, receipt, stdout, stderr)):
        raise FileExistsError("export artifacts already exist")
    nsys = shutil.which("nsys")
    if nsys is None:
        raise FileNotFoundError("nsys")
    before = sha256(source)
    command = [nsys, "export", "--type=sqlite", "--output", str(target), str(source)]
    environment = dict(os.environ)
    temporary = OUTPUT_ROOT / "runtime/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run(command, env=environment, stdout=out, stderr=err, check=True)
    if not target.is_file() or not target.stat().st_size or sha256(source) != before:
        raise ValueError("missing export or changed capture")
    record = {
        "schema": "echo-engine-nsys-export-v1",
        "command": command,
        "exitcode": result.returncode,
        "nsys_version": subprocess.check_output([nsys, "--version"], text=True).strip(),
        "input": {"path": str(source), "sha256": before},
        "output": {"path": str(target), "sha256": sha256(target)},
        "exporter_sha256": sha256(Path(__file__)),
        "stdout_sha256": sha256(stdout),
        "stderr_sha256": sha256(stderr),
    }
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
