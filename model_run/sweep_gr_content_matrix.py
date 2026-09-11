"""Run the 315-input / 945-layer independent-content matrix, resuming completed groups."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from GR.input_generator import DEFAULT_ITEM_LENGTHS, DEFAULT_USER_LENGTHS
from model_run.measure_gr_content_matrix import ROOT, case_name


def complete(root, h, u):
    path = root / f"h{h}_u{u}" / "complete.json"
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("new_lengths") != list(DEFAULT_ITEM_LENGTHS):
        return False
    return all(
        all(
            (root / f"layer{layer}" / case_name(h, u, n, i) / filename).is_file()
            for filename in (
                "result.json",
                "selected_indices.pt",
                "unique_token_ids.json",
                "input.jsonl",
                "prompt.txt",
            )
        )
        and (root / f"{case_name(h, u, n, i)}_validation.json").is_file()
        for n in DEFAULT_ITEM_LENGTHS
        for i in range(3)
        for layer in range(3)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--histories",
        type=int,
        nargs="+",
        choices=DEFAULT_USER_LENGTHS,
        default=DEFAULT_USER_LENGTHS,
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for h in args.histories:
        for u in range(3):
            if complete(args.root, h, u):
                print(f"Already completed h{h} u{u}", flush=True)
                continue
            print(f"Running h{h} u{u}: 21 inputs, 63 layer results", flush=True)
            with (args.root / f"h{h}_u{u}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "model_run.measure_gr_content_matrix",
                        "--history",
                        str(h),
                        "--history-variant",
                        str(u),
                        "--root",
                        str(args.root),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    print("Requested content matrix complete.", flush=True)


if __name__ == "__main__":
    main()
