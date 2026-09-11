"""Measure the selected GR shapes through dense layers 0/1 and the layer-2 indexer."""

import argparse
import subprocess
import sys
from pathlib import Path

from model_run.sweep_gr_mla_cache import HISTORIES, NEW_TOKENS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, nargs="+", choices=HISTORIES, default=HISTORIES)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    root = Path("GR/generated/multilayer_hits_validated")
    root.mkdir(parents=True, exist_ok=True)
    for h in args.histories:
        for n in NEW_TOKENS:
            name = f"h{h}_n{n}"
            log_path = root / f"{name}.log"
            if args.skip_completed and (
                (root / f"{name}_validation.json").is_file()
                and all(
                    (root / f"layer{layer}" / name / "result.json").is_file() for layer in range(3)
                )
                and log_path.is_file()
                and "Completed; peak allocated MiB:" in log_path.read_text()
            ):
                print(f"Already completed {h // 1024}K + {n // 1024}K", flush=True)
                continue
            print(f"Running {h // 1024}K + {n // 1024}K", flush=True)
            with (root / f"h{h}_n{n}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "model_run.measure_gr_multilayer_hits",
                        "--validate",
                        "--history",
                        str(h),
                        "--new",
                        str(n),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    print(f"All {len(args.histories) * len(NEW_TOKENS)} cases completed.", flush=True)


if __name__ == "__main__":
    main()
