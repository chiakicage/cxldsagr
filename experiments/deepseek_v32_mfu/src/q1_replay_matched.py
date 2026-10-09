"""Run the unchanged reduced HBM timer with the accepted formal environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
FORMAL_ID = "deepseek_h64k_a1_free_20261008_01_h65536_a1_profile"
KIND = "deepseek-q1-replay-matched-v1"
PREFIXES = (
    "CXLDSAGR_",
    "DG_",
    "DJ_",
    "FLASHINFER_",
    "CUTE_DSL_",
    "TRITON_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
)
ENVIRONMENT_KEYS = {
    "PATH",
    "CXX",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_PATH",
    "TVM_FFI_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "PYTHONDONTWRITEBYTECODE",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected(name):
    return name in ENVIRONMENT_KEYS or name.startswith(PREFIXES)


def read_environment(sqlite_path):
    """Read only the execution whitelist; never copy unrelated profiler environment."""
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT value FROM META_DATA_CAPTURE WHERE name = 'PROCESS_0:ENVIRONMENT_VARIABLE'"
        )
        result = {}
        for (entry,) in rows:
            name, separator, quoted = entry.partition("=")
            if not selected(name) and name != "HOME":
                continue
            require(
                separator and quoted.startswith('"') and quoted.endswith('"'),
                "Malformed selected profiler environment entry",
            )
            require(name not in result, "Duplicate selected environment entry")
            result[name] = quoted[1:-1]
    require(result.get("HOME") == os.environ.get("HOME"), "Formal default-cache home differs")
    del result["HOME"]
    return result


def apply_environment(expected):
    for name in set(expected) | {name for name in os.environ if selected(name)}:
        if name in expected:
            os.environ[name] = expected[name]
        else:
            os.environ.pop(name, None)


def require_subset(actual, expected, label):
    """List entries are complete identities, not recursively relaxed dictionaries."""
    require(all(row in expected for row in actual), f"Formal runtime mismatch: {label}")
    require(
        len({json.dumps(row, sort_keys=True) for row in actual}) == len(actual),
        f"Duplicate observed runtime entry: {label}",
    )
    return [row for row in expected if row not in actual]


def match_runtime(actual, formal):
    require(actual["backend"] == formal["backend_provenance"], "Formal backend identity differs")
    observed = actual["loaded_jit"]
    expected = formal["execution_runtime_artifacts"]
    missing = {}
    for name in ("native_jit", "local_native_jit", "cute_jit"):
        require(observed[name], f"No participating runtime entries: {name}")
        missing[name] = require_subset(observed[name], expected[name], name)
    for name in ("linear_quantization_triton", "attention_decode_triton"):
        left, right = observed[name], expected[name]
        require(left and left["specializations"], f"No participating specializations: {name}")
        require(
            {k: v for k, v in left.items() if k != "specializations"}
            == {k: v for k, v in right.items() if k != "specializations"},
            f"Formal specialization policy differs: {name}",
        )
        missing[name] = require_subset(left["specializations"], right["specializations"], name)
    require(
        observed["q1_topk_cub_native"] == expected["q1_topk_cub_native"],
        "Formal Q1 top-k binary identity differs",
    )
    require(observed["indexer_adaptation_triton"]["page64"], "No observed page64 specialization")
    for name, entries in observed["indexer_adaptation_triton"].items():
        missing["indexer_adaptation_triton/" + name] = require_subset(
            entries, expected["indexer_adaptation_triton"][name], name
        )
    require(observed["schema_version"] == expected["schema_version"], "Runtime schema differs")
    return missing


def verify_concrete_reference(formal):
    """Fail before provider import if an explicitly retained HBM binary is absent."""
    runtime = formal["execution_runtime_artifacts"]
    files = [
        formal["backend_provenance"]["installed"][name]["files"]["native"]
        for name in ("deep_gemm", "flash_mla")
    ]
    files.extend(row["library"] for row in runtime["native_jit"] if row["loaded_in_this_process"])
    topk = runtime["q1_topk_cub_native"]
    files.append({"path": topk["artifact_path"], "sha256": topk["artifact_sha256"]})
    for row in runtime["linear_quantization_triton"]["specializations"]:
        for name, path in row["metadata_group"].items():
            kind = Path(name).suffix[1:]
            if kind in row["artifact_sha256"]:
                files.append({"path": path, "sha256": row["artifact_sha256"][kind]})
    for item in files:
        require(
            digest(item["path"]) == item["sha256"],
            f"Retained formal HBM artifact changed: {item['path']}",
        )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--formal-profile", type=Path, default=EXPERIMENT / "output/data" / FORMAL_ID
    )
    options, remaining = parser.parse_known_args()
    # Preserve the original CLI/help; no Torch/provider import before environment setup.
    formal_dir = options.formal_profile.resolve()
    reference_files = {
        name: formal_dir / name
        for name in (
            "result.json",
            "backend_provenance_after.json",
            "sources.json",
            "hardware.json",
        )
    }
    formal = json.loads(reference_files["result.json"].read_text())
    require(formal["accepted"] and formal["run_id"] == FORMAL_ID, "Unexpected formal reference")
    require(
        formal["prefix_tokens"] == 65536 and formal["extend_tokens"] == 1,
        "Unexpected formal input shape",
    )
    require(
        json.loads(reference_files["backend_provenance_after.json"].read_text())
        == formal["backend_provenance"],
        "Formal provider records disagree",
    )
    expected_env = read_environment(formal_dir / "capture_4.sqlite")
    compiler_env = formal["backend_provenance"]["recall_dispatch"]["identity"][
        "compiler_environment"
    ]
    require(
        all(expected_env.get(name) == value for name, value in compiler_env.items()),
        "Formal capture and compiler environments disagree",
    )
    require(
        expected_env["PATH"].split(":")[:3] == [str(ROOT / ".venv/bin")] * 3,
        "Expected the literal formal three-prefix PATH",
    )
    require(
        all(
            expected_env.get(name) == value
            for name, value in formal["execution_environment"]["variables"].items()
        ),
        "Formal capture and execution environments disagree",
    )
    require(
        not any(name in sys.modules for name in ("torch", "triton", "tvm_ffi")),
        "Matched environment must precede provider imports",
    )
    verify_concrete_reference(formal)
    apply_environment(expected_env)
    if "--help" not in remaining and "-h" not in remaining:
        require(
            sorted(os.sched_getaffinity(0)) == formal["execution_environment"]["cpu_affinity"],
            "Run under the formal CPU affinity with taskset -c 0-7",
        )

    import torch

    from experiments.deepseek_v32_mfu.src import q1_replay_timer as timer

    old_source, old_runtime, old_receipt = timer.source_identity, timer.runtime, timer.write_receipt
    reference = {
        name: {"path": str(path), "sha256": digest(path)} for name, path in reference_files.items()
    }
    reference["capture_environment"] = {
        "path": str(formal_dir / "capture_4.sqlite"),
        "sha256": digest(formal_dir / "capture_4.sqlite"),
        "selected_environment": expected_env,
    }
    artifacts = {}
    destination = None

    def environment():
        require(
            {name: value for name, value in os.environ.items() if selected(name)} == expected_env,
            "Matched environment changed during provider imports",
        )
        torch.set_num_threads(formal["execution_environment"]["torch_num_threads"])
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        precision = {
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        require(precision == formal["torch_precision"], "Formal precision differs")

    def source_identity(args):
        nonlocal destination
        require(args.physical_device == 0, "Matched control requires physical GPU0")
        source = old_source(args)
        require(
            {name: value for name, value in os.environ.items() if selected(name)} == expected_env,
            "Matched execution environment changed",
        )
        require(
            torch.get_num_threads() == formal["execution_environment"]["torch_num_threads"],
            "Formal execution thread count changed",
        )
        require(source["torch"] == formal["dependencies"]["torch"], "Formal Torch differs")
        require(
            source["uuid"] == formal["hardware"]["torch_device_uuid"], "Formal GPU UUID differs"
        )
        require(source["checkpoint"] == formal["checkpoint_identity"], "Formal checkpoint differs")
        require(source["request"]["sha256"] == formal["request_sha256"], "Formal request differs")
        frozen_sources = formal["source_sha256"]
        require(
            all(source["sources"].get(name) == value for name, value in frozen_sources.items()),
            "A formal source changed or is absent from the source snapshot",
        )
        source["sources"][str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
        source["formal_reference"] = reference
        source["contract"]["matched_environment"] = KIND
        source["contract"]["comparison_boundary"] = (
            "Same reduced timer lifecycle; matched formal execution/compiler environment and "
            "participating recorded native/CuTe/Triton identities. Unused offload entries are "
            "not loaded. No graph-edge instrumentation. DeepGEMM per-launch JIT binary ledger "
            "is not exposed by the formal collector; CuTe MLIR identities have no retained file."
        )
        destination = (
            Path("/tmp/cxldsagr-checks/q1-replay-timer")
            if args.mode == "check"
            else EXPERIMENT / "output/data"
        ) / args.run_id
        return source

    def archive(path, expected, category="runtime"):
        require(digest(path) == expected, f"Concrete artifact changed: {path}")
        target = destination / category / (expected + "_" + Path(path).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        require(digest(target) == expected, "Archived artifact differs")
        artifacts[str(target.relative_to(destination))] = target

    def runtime():
        actual = old_runtime()
        missing = match_runtime(actual, formal)
        for name, path in reference_files.items():
            archive(path, reference[name]["sha256"], "formal_reference")
        require(
            digest(formal_dir / "capture_4.sqlite") == reference["capture_environment"]["sha256"],
            "Formal capture changed",
        )
        loaded = actual["loaded_jit"]
        native_files = [row["library"] for row in loaded["local_native_jit"]]
        native_files.extend(
            row["library"] for row in loaded["native_jit"] if row["loaded_in_this_process"]
        )
        native_files.extend(
            actual["backend"]["installed"][name]["files"]["native"]
            for name in ("deep_gemm", "flash_mla")
        )
        for item in native_files:
            archive(item["path"], item["sha256"])
        for row in loaded["linear_quantization_triton"]["specializations"]:
            for filename, path in row["metadata_group"].items():
                kind = Path(filename).suffix[1:]
                if kind in row["artifact_sha256"]:
                    archive(path, row["artifact_sha256"][kind])
        # The remaining Triton collectors hash retained in-memory artifacts.
        from operators.deepseek_v32.attention.device_only import decode

        kernels = list(decode._observed.values())
        for name in ("page64", "decode_hint"):
            module = sys.modules.get("operators.deepseek_v32.indexer." + name)
            if module is not None and module._kernel.cache_info().currsize:
                for cache in module._kernel().device_caches.values():
                    kernels.extend(cache[0].values())
        for compiled in kernels:
            for kind in ("ptx", "cubin"):
                value = compiled.asm[kind]
                value = value.encode() if isinstance(value, str) else value
                identity = hashlib.sha256(value).hexdigest()
                target = destination / "runtime" / f"{identity}.{kind}"
                if not target.exists():
                    target.write_bytes(value)
                require(digest(target) == identity, "Retained Triton archive differs")
                artifacts[str(target.relative_to(destination))] = target
        actual["formal_match"] = {
            "passed": True,
            "unobserved_formal_entries": missing,
            "archived_artifacts": {name: digest(path) for name, path in sorted(artifacts.items())},
            "boundary": "Exact participating recorded identities; no per-launch DeepGEMM binary "
            "ledger or retained CuTe MLIR file is claimed.",
        }
        return actual

    def write_receipt(path, **kwargs):
        kwargs["artifacts"].update(artifacts)
        return old_receipt(path, **kwargs)

    timer.KIND = KIND
    timer.environment = environment
    timer.source_identity = source_identity
    timer.runtime = runtime
    timer.write_receipt = write_receipt
    sys.argv = [sys.argv[0], *remaining]
    timer.main()


if __name__ == "__main__":
    main()
