"""Run standalone DeepSeek checkpoint prefill and extend."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.deepseek_v32.model import DeepSeekEchoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,6,7")
    parser.add_argument("--input-ids", type=Path, required=True, help="JSON token ID list")
    parser.add_argument("--history", type=int, default=65536)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--sparse-pool-tokens", "--slots", dest="slots", type=int, default=16384)
    parser.add_argument("--host-arena-tokens", type=int)
    parser.add_argument("--workspace-query-tokens", type=int)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--extend-chunk-size", type=int, default=None)
    parser.add_argument(
        "--num-layers", type=int, help="execute only the first N checkpoint layers for diagnostics"
    )
    args = parser.parse_args()
    ids = json.loads(args.input_ids.read_text())
    model = DeepSeekEchoModel(
        args.model,
        devices=[int(x) for x in args.devices.split(",")],
        capacity=len(ids),
        offload=args.offload,
        slots=args.slots,
        chunk_size=args.chunk_size,
        extend_chunk_size=args.extend_chunk_size,
        host_arena_tokens=args.host_arena_tokens,
        workspace_query_tokens=args.workspace_query_tokens
        or max(args.chunk_size, args.extend_chunk_size or len(ids) - args.history),
        num_layers=args.num_layers,
    )
    model.forward(ids[: args.history])
    result = model.forward(ids[args.history :])
    print(json.dumps({"length": model.length, "next_token_id": int(result.argmax(-1))}))


if __name__ == "__main__":
    main()
