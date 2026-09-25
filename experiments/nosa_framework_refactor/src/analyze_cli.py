"""Validate saved outputs from the serving and text-generation CLI smoke runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serving = (args.log_dir / "serving.stdout.log").read_text()
    requests = [json.loads(line) for line in serving.splitlines() if line.strip()]
    assert len(requests) == 2
    assert all(row["status"] == "completed" and row["feature_shape"] == [4096] for row in requests)
    assert all(row["metadata"]["total_input_tokens"] == 384 for row in requests)
    assert requests[0]["metadata"]["user_id"] == requests[1]["metadata"]["user_id"]
    generation = (args.log_dir / "generation.stderr.log").read_text()
    stats = next(
        json.loads(line) for line in reversed(generation.splitlines()) if line.startswith("{")
    )
    assert 1 <= stats["generated_tokens"] <= 4
    assert stats["decode_steps"] == stats["generated_tokens"] - 1
    summary = {
        "serving_requests": len(requests),
        "serving_input_tokens": [row["metadata"]["total_input_tokens"] for row in requests],
        "feature_shapes": [row["feature_shape"] for row in requests],
        "generation": stats,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
