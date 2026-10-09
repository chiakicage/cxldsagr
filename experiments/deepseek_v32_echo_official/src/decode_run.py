"""Run an ordinary H64K request that samples two tokens and executes one decode."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.engine_run import (
    EXPERIMENT_ROOT,
    HISTORY,
    VOCAB,
    deadline,
    engine_arguments,
    file_sha256,
    load_workload,
    token_hash,
    write_json,
)

PROTOCOL = {
    "id": "normal_h64k_one_decode_v1",
    "history_tokens": HISTORY,
    "max_new_tokens": 2,
    "decode_forwards": 1,
    "warmup_first_token_offsets": [1, 2],
    "cache_reset": False,
    "cuda_graph": "official default enabled, batch size 1",
}
SAMPLING = {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True}


def validate(response, input_ids):
    if not isinstance(response, dict) or "error" in response:
        raise ValueError("Engine request failed")
    meta = response["meta_info"]
    if any(
        meta[name] != value
        for name, value in {
            "prompt_tokens": HISTORY,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "finish_reason": {"type": "length", "length": 2},
        }.items()
    ):
        raise ValueError(f"Normal decode geometry/finish mismatch: {meta}")
    ids = response["output_ids"]
    if (
        len(ids) != 7
        or ids[:5] != input_ids[-5:]
        or any(type(x) is not int or not 0 <= x < VOCAB for x in ids)
    ):
        raise ValueError("Unexpected two-token detokenization output")
    return {
        "prompt_tokens": HISTORY,
        "cached_tokens": 0,
        "completion_tokens": 2,
        "output_ids": ids[-2:],
        "input_sha256": token_hash(input_ids),
    }


def check_output(response, ids, path):
    import numpy as np

    meta = response["meta_info"]
    hidden = np.concatenate(
        [np.asarray(x, dtype=np.float32).reshape(-1, 7168) for x in meta["hidden_states"]]
    )
    entries = meta["output_token_ids_logprobs"]
    if hidden.shape != (1025, 7168) or len(entries) != 2:
        raise ValueError(f"Expected final prefill chunk plus decode hidden; got {hidden.shape}")
    logprobs = np.asarray([[x[0] for x in row] for row in entries], dtype=np.float32)
    if (
        logprobs.shape != (2, VOCAB)
        or any([x[1] for x in row] != list(range(VOCAB)) for row in entries)
        or not np.isfinite(hidden).all()
        or not np.isfinite(logprobs).all()
        or not np.array_equal(logprobs.argmax(axis=1), ids)
        or not np.allclose(np.exp(logprobs.astype(np.float64)).sum(axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Non-finite or inconsistent returned output distribution")
    np.savez_compressed(path, hidden=hidden, logprobs=logprobs, output_ids=ids)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "hidden_shape": list(hidden.shape),
        "logprobs_shape": list(logprobs.shape),
        "finite": True,
        "numerical_acceptance": False,
    }


def run(args):
    if not args.output.resolve().is_relative_to((EXPERIMENT_ROOT / "output").resolve()):
        raise ValueError("Output must remain in the experiment")
    _, requests = load_workload(args.workload)
    source = requests[0]
    prompt = source["input_ids"][:HISTORY]
    args.output.mkdir(parents=True, exist_ok=False)
    from sglang.srt.entrypoints import engine as engine_module

    if args.mode == "profile":
        from experiments.deepseek_v32_echo_official.src.decode_profile_hooks import scheduler

        os.environ["ECHO_ENGINE_PROFILE_DIR"] = str(args.output / "hooks")
        os.environ["ECHO_ENGINE_PROFILE_CASE"] = "echo" if args.case == "echo" else "hbm"
        engine_module.run_scheduler_process = scheduler
    kwargs = engine_arguments(args.model, args.case)
    metadata = {
        "schema": "echo-normal-decode-worker-v1",
        "mode": args.mode,
        "case": args.case,
        "protocol": PROTOCOL,
        "sampling_params": SAMPLING,
        "engine_arguments": kwargs,
        "prefix_sha256": token_hash(prompt),
        "numerical_acceptance": False,
    }
    write_json(args.output / "metadata.json", metadata)
    engine, primary, profiling = None, None, False
    try:
        with deadline(args.startup_timeout):
            engine = engine_module.Engine(**kwargs)

        async def initialize():
            engine.tokenizer_manager.auto_create_handle_loop()
            await asyncio.sleep(0)

        with deadline(60):
            engine.loop.run_until_complete(initialize())
            write_json(args.output / "server_info.json", engine.get_server_info())
        warmup = []
        for offset in PROTOCOL["warmup_first_token_offsets"]:
            warm_prompt = list(prompt)
            warm_prompt[0] = (warm_prompt[0] + offset) % VOCAB
            with deadline(args.request_timeout):
                response = engine.generate(input_ids=warm_prompt, sampling_params=SAMPLING)
            warmup.append({"first_token_offset": offset, **validate(response, warm_prompt)})
        write_json(args.output / "warmup.json", warmup)
        if args.mode == "profile":
            with deadline(60):
                engine.collective_rpc("echo_engine_profile_begin")
                engine.start_profile(
                    activities=["CUDA_PROFILER"],
                    with_stack=False,
                    record_shapes=False,
                    output_dir=str(args.output / "hooks"),
                )
            profiling = True
        call = {
            "input_ids": prompt,
            "sampling_params": SAMPLING,
            "stream": False,
            "data_parallel_rank": 0,
        }
        if args.mode == "check":
            call.update(
                return_hidden_states=True,
                return_logprob=True,
                logprob_start_len=-1,
                token_ids_logprob=list(range(VOCAB)),
            )
        with deadline(args.request_timeout):
            scope = nullcontext()
            if args.mode == "profile":
                from torch.cuda import nvtx

                from experiments.deepseek_v32_echo_official.src.engine_profile_hooks import (
                    nvtx_range,
                )

                scope = nvtx_range(
                    nvtx, "echo_engine|kind=request|phase=normal_decode|q=65536|start=0|end=65537"
                )
            with scope:
                started_ns = time.time_ns()
                start = time.perf_counter_ns()
                response = engine.generate(**call)
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                finished_ns = time.time_ns()
        row = {**validate(response, prompt), "started_ns": started_ns, "finished_ns": finished_ns}
        if args.mode == "check":
            row["output_artifact"] = check_output(
                response, row["output_ids"], args.output / "output.npz"
            )
        else:
            row["wall_ms"] = wall_ms
            write_json(args.output / "response.json", response)
            row["response_sha256"] = file_sha256(args.output / "response.json")
        if args.mode == "profile":
            with deadline(60):
                engine.collective_rpc("echo_decode_finish")
                snapshot = json.loads((args.output / "hooks/residency.json").read_text())
                if snapshot["decode_input_ids"] != row["output_ids"][:1]:
                    raise ValueError("Decode did not consume the first sampled output token")
                engine.stop_profile()
            profiling = False
        write_json(
            args.output / "complete.json",
            {
                **metadata,
                "status": "completed",
                "row": row,
                "warmup_sha256": file_sha256(args.output / "warmup.json"),
            },
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup = []
        if engine is not None:
            if profiling:
                try:
                    with deadline(60):
                        engine.stop_profile()
                except BaseException as error:  # noqa: BLE001
                    cleanup.append(error)
            try:
                engine.shutdown()
                atexit.unregister(engine.shutdown)
            except BaseException as error:  # noqa: BLE001
                cleanup.append(error)
        if cleanup:
            raise BaseExceptionGroup(
                "Normal decode cleanup failed", ([primary] if primary else []) + cleanup
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--case", choices=("echo", "resident_reference"), required=True)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
