"""Warm an official SGLang Engine, then execute a history/extend pair without HTTP.

This worker is launched in an owned process group by scripts/run_engine.py.
Two synthetic disjoint-prefix pairs warm computation and normal eviction paths.
The formal H request has no matching prefix; H+A then reuses exactly H. Neither
the allocator nor official radix/ECHO state is flushed or otherwise modified.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
REPRO_ROOT = REPO_ROOT / "3rdparty/ECHO/reproduction/cxldsagr"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPRO_ROOT))

from src.preflight import file_sha256
from src.workload_client import canonical, load_workload, token_hash

HISTORY = 65536
CANDIDATE = 128
VOCAB = 129280
WARMUP_PROTOCOL = {
    "id": "two_disjoint_prefix_pairs_v1",
    "pair_count": 2,
    "first_token_offsets": [1, 2],
    "requests_per_pair": ["prefill", "extend"],
    "timed": False,
}


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def engine_arguments(model: Path, case: str, *, piecewise_cuda_graph: bool = False) -> dict:
    arguments = {
        "model_path": str(model),
        "tp_size": 1,
        "dp_size": 1,
        "page_size": 64,
        "chunked_prefill_size": 1024,
        "skip_server_warmup": True,
        "random_seed": 42,
        "disable_overlap_schedule": True,
        "cuda_graph_max_bs": 1,
        "max_running_requests": 1,
        "context_length": 131072,
        "enable_return_hidden_states": True,
        "mem_fraction_static": 0.8,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "max_total_tokens": 16777216 if case == "echo" else 131072,
        "log_level": "info",
    }
    if piecewise_cuda_graph:
        arguments["enable_piecewise_cuda_graph"] = True
    return arguments


@contextlib.contextmanager
def deadline(seconds: float):
    """Terminate a blocked control or request call; never retry it."""

    def expired(signum, frame):
        raise TimeoutError(f"Engine call exceeded its {seconds}-second deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def validate_response(response, input_ids: list[int], *, phase: str) -> dict:
    if not isinstance(response, dict) or "error" in response:
        raise ValueError(f"unsuccessful Engine response: {str(response)[:500]}")
    meta = response["meta_info"]
    expected_prompt = HISTORY if phase == "prefill" else HISTORY + CANDIDATE
    expected_cached = 0 if phase == "prefill" else HISTORY
    if meta["prompt_tokens"] != expected_prompt or meta["completion_tokens"] != 1:
        raise ValueError("Engine did not consume the exact prompt and sample one token")
    if meta["finish_reason"] != {"type": "length", "length": 1}:
        raise ValueError(f"unsuccessful finish reason: {meta['finish_reason']}")
    if type(meta["cached_tokens"]) is not int or meta["cached_tokens"] != expected_cached:
        raise ValueError(
            f"{phase} requires exactly {expected_cached} cached tokens; "
            f"observed {meta['cached_tokens']}"
        )
    # The pinned text API exposes the last five prompt IDs plus the generated ID.
    raw_ids = response["output_ids"]
    if (
        not isinstance(response.get("text"), str)
        or not isinstance(raw_ids, list)
        or len(raw_ids) != 6
        or raw_ids[:5] != input_ids[-5:]
        or any(type(value) is not int or not 0 <= value < VOCAB for value in raw_ids)
    ):
        raise ValueError("unexpected official Engine detokenization buffer")
    return {
        "phase": phase,
        "prompt_tokens": expected_prompt,
        "cached_tokens": expected_cached,
        "computed_prompt_tokens": expected_prompt - expected_cached,
        "completion_tokens": 1,
        "finish_reason": meta["finish_reason"],
        "output_ids": raw_ids[-1:],
        "raw_output_ids": raw_ids,
        "input_sha256": token_hash(input_ids),
    }


def save_check_output(response, output_ids: list[int], path: Path, phase: str) -> dict:
    """Save independent observations, without asserting cross-run equivalence."""
    import numpy as np

    meta = response["meta_info"]
    hidden = np.concatenate(
        [np.asarray(chunk, dtype=np.float32) for chunk in meta["hidden_states"]], axis=0
    )
    expected_shape = (1024 if phase == "prefill" else CANDIDATE, 7168)
    if hidden.shape != expected_shape:
        raise ValueError(f"unexpected final executed chunk shape: {hidden.shape}")
    entries = meta["output_token_ids_logprobs"]
    if len(entries) != 1 or len(entries[0]) != VOCAB:
        raise ValueError("complete final-token vocabulary log probabilities are required")
    ids = np.asarray([entry[1] for entry in entries[0]], dtype=np.int64)
    logprobs = np.asarray([entry[0] for entry in entries[0]], dtype=np.float32)
    if not np.array_equal(ids, np.arange(VOCAB)):
        raise ValueError("log-probability vocabulary IDs differ")
    if not np.isfinite(hidden).all() or not np.isfinite(logprobs).all():
        raise ValueError("non-finite official Engine output")
    if abs(float(np.exp(logprobs.astype(np.float64)).sum()) - 1.0) > 1e-4:
        raise ValueError("output probabilities do not normalize")
    if int(np.argmax(logprobs)) != output_ids[0]:
        raise ValueError("greedy output differs from the returned distribution")
    sampled = meta["output_token_logprobs"]
    if (
        len(sampled) != 1
        or len(sampled[0]) != 3
        or not math.isfinite(sampled[0][0])
        or sampled[0][1] != output_ids[0]
        or sampled[0][2] is not None
    ):
        raise ValueError("sampled-token log probability disagrees with the generated ID")
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            hidden=hidden,
            logprobs=logprobs,
            output_ids=np.asarray(output_ids, dtype=np.int64),
        )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "hidden_shape": list(hidden.shape),
        "logprobs_shape": list(logprobs.shape),
        "finite": True,
        "scope": "Output shape, finiteness and distribution consistency; no equivalence claim.",
    }


def warm_engine(engine, source, args, sampling_params):
    """Use ordinary requests to warm shapes and eviction with distinct prefixes."""
    records = []
    started_ns = time.time_ns()
    for offset in WARMUP_PROTOCOL["first_token_offsets"]:
        warm_ids = list(source["input_ids"])
        warm_ids[0] = (warm_ids[0] + offset) % VOCAB
        for phase, input_ids in (("prefill", warm_ids[:HISTORY]), ("extend", warm_ids)):
            with deadline(args.request_timeout):
                clock = time.perf_counter_ns()
                response = engine.generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    stream=False,
                    data_parallel_rank=0,
                )
                elapsed = (time.perf_counter_ns() - clock) / 1e6
            observed = validate_response(response, input_ids, phase=phase)
            response_path = args.output / f"warmup_{offset}_{phase}.response.json"
            write_json(response_path, response)
            observed.update(
                first_token_offset=offset,
                synthetic_first_token=warm_ids[0],
                wall_ms=elapsed,
                response_sha256=file_sha256(response_path),
            )
            records.append(observed)
            print(
                canonical(
                    {
                        "warmup_offset": offset,
                        "phase": phase,
                        "cached_tokens": observed["cached_tokens"],
                    }
                ),
                flush=True,
            )
    receipt = {
        "schema": "echo-engine-in-process-warmup-v1",
        "status": "completed",
        "source_input_sha256": source["input_sha256"],
        "source_first_token": source["input_ids"][0],
        "warmup_protocol": WARMUP_PROTOCOL,
        "rows": records,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "scope": "Four ordinary generate calls outside formal timing and CUDA profiling. Only token 0 is changed; distinct synthetic histories induce normal official cache eviction. Diagnostic wall times are not benchmark samples.",
    }
    path = args.output / "in_engine_warmup.json"
    write_json(path, receipt)
    return file_sha256(path)


def run(args) -> None:
    output_root = (EXPERIMENT_ROOT / "output").resolve()
    for path in (args.output, args.profile_dir):
        if path is not None and not path.resolve().is_relative_to(output_root):
            raise ValueError("worker outputs must remain inside this experiment's output directory")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("the Engine worker requires PYTHONDONTWRITEBYTECODE=1")
    for variable in (
        "XDG_CACHE_HOME",
        "DG_JIT_CACHE_DIR",
        "SGL_DG_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "TMPDIR",
    ):
        value = os.environ.get(variable)
        if value is None or not Path(value).resolve().is_relative_to(output_root):
            raise ValueError(
                f"launch through scripts/run_engine.sh to scope {variable} to the experiment"
            )
    _, requests = load_workload(args.workload)
    source = requests[0]
    args.output.mkdir(parents=True, exist_ok=False)
    from sglang.srt.entrypoints import engine as engine_module

    if args.mode == "profile":
        from experiments.deepseek_v32_echo_official.src.engine_profile_hooks import (
            scheduler_with_profile,
        )

        if args.profile_dir is None:
            raise ValueError("profile mode requires --profile-dir")
        args.profile_dir.mkdir(parents=True, exist_ok=False)
        os.environ["ECHO_ENGINE_PROFILE_DIR"] = str(args.profile_dir)
        os.environ["ECHO_ENGINE_PROFILE_CASE"] = "echo" if args.case == "echo" else "hbm"
        engine_module.run_scheduler_process = scheduler_with_profile

    kwargs = engine_arguments(
        args.model, args.case, piecewise_cuda_graph=args.enable_piecewise_cuda_graph
    )
    metadata = {
        "schema": "echo-sglang-engine-pair-v1",
        "mode": args.mode,
        "case": args.case,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        "engine_arguments": kwargs,
        "source_request_id": source["request_id"],
        "source_user_id": source["user_id"],
        "prefix_sha256": source["prefix_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "warmup_protocol": WARMUP_PROTOCOL,
        "measurement": (
            "Synchronous Engine.generate entry through its complete Python result. Includes "
            "TokenizerManager, scheduler/IPC, model, sampling and detokenization. Excludes "
            "HTTP, model load, control initialization, in-process synthetic warmup, output "
            "checks and artifact writes."
        ),
        "cache_semantics": (
            "Warmed Engine with synthetic prefixes already present. Formal H has no matching "
            "logical prefix; H+A then reuses exactly H. Normal official eviction is included; "
            "the allocator and ECHO mappings are not empty or reset. "
            "Official radix retains the prompt. The sampled token has no decode forward. "
            "History HBM residency after prefill is retained: no cold reset or flush."
        ),
        "warmup_semantics": (
            "Every worker executes two ordinary synthetic H->H+A pairs in this Engine first. "
            "Only the first token changes, by +1 and +2 modulo vocabulary size. This covers "
            "both query shapes, both completion paths and normal capacity-pressure eviction. "
            "Independent outer warmup also remains; it shares disk JIT caches only."
        ),
        "numerical_acceptance": False,
        "started_ns": time.time_ns(),
    }
    write_json(args.output / "metadata.json", metadata)
    engine = None
    profile_started = False
    primary = None
    try:
        with deadline(args.startup_timeout):
            engine = engine_module.Engine(**kwargs)
        control_started = time.time_ns()

        async def initialize_receive_loop():
            engine.tokenizer_manager.auto_create_handle_loop()
            await asyncio.sleep(0)

        # This is the normal official receive-loop initializer, with no request,
        # cache action, GPU inference, HTTP call or change to scheduler policy.
        with deadline(args.control_timeout):
            engine.loop.run_until_complete(initialize_receive_loop())
            before = engine.get_server_info()
        write_json(args.output / "server_info.json", before)
        write_json(
            args.output / "control_setup.json",
            {
                "started_ns": control_started,
                "finished_ns": time.time_ns(),
                "initializer": "TokenizerManager.auto_create_handle_loop on Engine.loop",
                "state_query": "Engine.get_server_info",
                "http_calls": 0,
                "inference_calls": 0,
            },
        )
        warmup_sha256 = warm_engine(engine, source, args, metadata["sampling_params"])
        if args.mode == "profile":
            with deadline(args.control_timeout):
                engine.collective_rpc("echo_engine_profile_begin")
                engine.start_profile(
                    output_dir=str(args.profile_dir),
                    activities=["CUDA_PROFILER"],
                    with_stack=False,
                    record_shapes=False,
                )
            profile_started = True
        rows = []
        with (args.output / "requests.jsonl").open("x") as stream:
            for phase, input_ids in (
                ("prefill", source["input_ids"][:HISTORY]),
                ("extend", source["input_ids"]),
            ):
                call = {
                    "input_ids": input_ids,
                    "sampling_params": metadata["sampling_params"],
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
                    if args.mode == "profile":
                        from torch.cuda import nvtx

                        start = 0 if phase == "prefill" else HISTORY
                        case = "echo" if args.case == "echo" else "hbm"
                        nvtx.range_push(
                            f"echo_engine|kind=request|case={case}|phase={phase}"
                            f"|q={len(input_ids) - start}|start={start}|end={len(input_ids)}"
                        )
                    try:
                        started_ns = time.time_ns()
                        clock = time.perf_counter_ns()
                        response = engine.generate(**call)
                        elapsed_ms = (time.perf_counter_ns() - clock) / 1e6
                        finished_ns = time.time_ns()
                    finally:
                        if args.mode == "profile":
                            nvtx.range_pop()
                observed = validate_response(response, input_ids, phase=phase)
                observed.update(
                    started_ns=started_ns,
                    finished_ns=finished_ns,
                    source_request_id=source["request_id"],
                )
                if args.mode == "check":
                    observed["output_artifact"] = save_check_output(
                        response, observed["output_ids"], args.output / f"{phase}.npz", phase
                    )
                else:
                    observed["wall_ms"] = elapsed_ms
                    response_path = args.output / f"{phase}.response.json"
                    write_json(response_path, response)
                    observed["response_sha256"] = file_sha256(response_path)
                if args.mode == "profile":
                    # Synchronized residency diagnostics are outside generate
                    # timing and all forward scopes, and never run in bench.
                    with deadline(args.control_timeout):
                        engine.collective_rpc("echo_engine_profile_snapshot", phase=phase)
                stream.write(canonical(observed) + "\n")
                stream.flush()
                rows.append(observed)
                print(
                    canonical({"phase": phase, "cached_tokens": observed["cached_tokens"]}),
                    flush=True,
                )
        if profile_started:
            profile_started = False
            with deadline(args.control_timeout):
                engine.stop_profile()
        with deadline(args.control_timeout):
            write_json(args.output / "server_info_after.json", engine.get_server_info())
        write_json(
            args.output / "complete.json",
            {
                **metadata,
                "completed": 2,
                "finished_ns": time.time_ns(),
                "request_phases": [row["phase"] for row in rows],
                "cached_tokens": [row["cached_tokens"] for row in rows],
                "requests_sha256": file_sha256(args.output / "requests.jsonl"),
                "in_engine_warmup_sha256": warmup_sha256,
                "status": "completed",
            },
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup = []
        if engine is not None:
            if profile_started:
                try:
                    with deadline(args.control_timeout):
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
                "Engine execution and cleanup failed" if primary else "Engine cleanup failed",
                ([primary] if primary else []) + cleanup,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--case", choices=("echo", "resident_reference"), required=True)
    parser.add_argument("--mode", choices=("check", "bench", "profile", "warmup"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--control-timeout", type=float, default=60)
    args = parser.parse_args()
    if min(args.startup_timeout, args.request_timeout, args.control_timeout) <= 0:
        parser.error("all deadlines must be positive")
    run(args)


if __name__ == "__main__":
    main()
