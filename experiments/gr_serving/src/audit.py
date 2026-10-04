"""Read-only, independent acceptance audit for the frozen GR serving run.

No repository modules, GPU APIs, process inspection, or model execution are used.
The default checks every saved HBM tensor on CPU, one tensor at a time.  A lighter
``--reference-check existence`` run is explicitly structural, never accepted.
The JSON destination is created exclusively, only after all selected gates pass.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SCHEMES = {
    "deepseek_v32": ("hbm", "echo", "serial_sparse", "dense_prefetch"),
    "nosa": ("hbm", "serial_sparse", "dense_prefetch", "overlap"),
}
TIERS = ("hbm", "dram")
WIDTHS = {"deepseek_v32": 7168, "nosa": 4096}
IDENTITY_FIELDS = (
    "request_id",
    "user_id",
    "visit_index",
    "visit_number",
    "is_revisit",
    "timestamp",
    "stable_prefix_tokens",
    "candidate_suffix_tokens",
    "prefix_sha256",
    "candidate_sha256",
    "input_sha256",
)
PHASES = ("admission_ms", "prefix_ms", "extend_ms", "cleanup_ms")
STATS = ("count", "mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms")
HEAT_DEFAULT_FIELDS = {"beauty": "interaction_count", "industrial_10M": "pv_share"}
ACCESS_TRACE_FIELDS = ("request_id", "user_id", "visit_index", "previous_request_id", "timestamp")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def equal(actual, expected, message):
    require(actual == expected, f"{message}: got {actual!r}, expected {expected!r}")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer {value!r}")
    return value


def number(value, label, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"{label}: invalid finite number {value!r}",
    )
    return value


def close(actual, expected, label):
    number(actual, label)
    require(
        math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-8),
        f"{label}: got {actual}, expected {expected}",
    )


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(
        text,
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            AuditError(f"nonfinite JSON constant {value}")
        ),
    )


def read_json(path):
    return parse_json(path.read_text())


def read_jsonl(path):
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            require(bool(line.strip()), f"{path}:{lineno}: empty JSONL row")
            require(line.endswith("\n"), f"{path}:{lineno}: incomplete JSONL row")
            row = parse_json(line)
            require(isinstance(row, dict), f"{path}:{lineno}: expected object")
            yield row


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(2**20), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def token_hash(tokens):
    return hashlib.sha256(canonical(tokens)).hexdigest()


def key(row, request=False):
    fields = (
        ("model", "scheme", "num_users", "request_id")
        if request
        else ("model", "scheme", "num_users")
    )
    return tuple(row[name] for name in fields)


def indexed(rows, request=False):
    result = {}
    for row in rows:
        item = key(row, request)
        require(item not in result, f"duplicate record {item}")
        result[item] = row
    return result


@dataclass(frozen=True)
class RunContract:
    models: tuple[str, ...]
    populations: tuple[int, ...]
    request_cap: int
    max_revisits: int | None
    history_tokens: int
    candidate_tokens: int
    seed: int
    hbm_bytes: int
    dram_bytes: int
    allow_empty_revisits: bool = False
    heat_dataset: str | None = "beauty"
    heat_field: str | None = "interaction_count"
    sampling: str = "weighted"

    @classmethod
    def from_parameters(cls, params):
        models, populations = params["models"], params["users"]
        require(isinstance(models, list) and bool(models), "models must be a nonempty list")
        require(all(model in SCHEMES for model in models), "unsupported model")
        require(len(set(models)) == len(models), "duplicate models")
        require(
            isinstance(populations, list) and bool(populations), "users must be a nonempty list"
        )
        require(
            all(type(users) is int and users > 0 for users in populations),
            "unsupported user population",
        )
        require(len(set(populations)) == len(populations), "duplicate populations")
        allow_empty = params.get("allow_empty_revisits", False)
        require(type(allow_empty) is bool, "allow_empty_revisits must be boolean")
        max_revisits = params.get("max_revisits")
        if max_revisits is not None:
            integer(max_revisits, "maximum revisits excluding first", 0)
        sampling = params.get("sampling", "weighted")
        require(sampling in ("weighted", "sequential"), "unsupported sampling mode")
        if sampling == "sequential":
            require(max_revisits is None, "sequential sampling requires no revisit cap")
            require(
                params.get("access_trace") is None, "sequential sampling does not use a heat trace"
            )
            heat_dataset, heat_field = None, None
        else:
            heat_dataset = params.get("heat_dataset", "beauty")
            require(
                isinstance(heat_dataset, str) and heat_dataset in HEAT_DEFAULT_FIELDS,
                "unsupported heat dataset",
            )
            heat_field = params.get("heat_field")
            if heat_field is None:
                heat_field = HEAT_DEFAULT_FIELDS[heat_dataset]
            require(isinstance(heat_field, str) and bool(heat_field), "invalid heat field")
        # The measurement runner executes supported models in this canonical order.
        canonical_models = tuple(model for model in SCHEMES if model in models)
        return cls(
            canonical_models,
            tuple(populations),
            integer(params["requests"], "explicit total-request upper cap", 1),
            max_revisits,
            integer(params["history_tokens"], "history tokens", 1),
            integer(params["candidate_tokens"], "candidate tokens", 1),
            integer(params["seed"], "seed"),
            integer(params["hbm_budget_bytes"], "HBM byte cap", 1),
            integer(params["dram_budget_bytes"], "DRAM byte cap", 1),
            allow_empty,
            heat_dataset,
            heat_field,
            sampling,
        )

    @property
    def caps(self):
        return {"hbm": self.hbm_bytes, "dram": self.dram_bytes}

    @property
    def request_counts(self):
        return {
            users: (
                self.request_cap
                if self.max_revisits is None
                else min(self.request_cap, users * (self.max_revisits + 1))
            )
            for users in self.populations
        }

    @property
    def cases(self):
        return [
            (model, scheme, users)
            for model in self.models
            for scheme in SCHEMES[model]
            for users in self.populations
        ]

    @property
    def measured_requests(self):
        return sum(self.request_counts[users] for _, _, users in self.cases)


def audit_metadata(meta, expected_run, expected_source):
    equal(meta["status"], "accepted", "run is not complete and accepted")
    require(
        type(meta["schema_version"]) is int and meta["schema_version"] in (1, 2), "metadata schema"
    )
    if expected_run is not None:
        equal(meta["run_id"], expected_run, "run ID")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", meta["run_id"])), "invalid run ID")
    require(
        isinstance(meta["source_sha256"], str)
        and re.fullmatch(r"[a-f0-9]{64}", meta["source_sha256"]),
        "invalid source identity",
    )
    if expected_source is not None:
        equal(meta["source_sha256"], expected_source, "frozen source identity")
    number(meta["started_unix"], "start time", positive=True)
    number(meta["completed_unix"], "completion time", positive=True)
    require(meta["completed_unix"] > meta["started_unix"], "completion must follow start")
    params = meta["parameters"]
    contract = RunContract.from_parameters(params)
    loops = meta.get("sequential_loops")
    if meta["schema_version"] == 2 and contract.sampling == "sequential":
        require(len(contract.populations) == 1, "complete loops require one user population")
        users = contract.populations[0]
        require(
            contract.request_cap >= 2 * users and contract.request_cap % users == 0,
            "sequential requests must contain >= 2 complete rounds",
        )
        equal(
            loops,
            {
                "schema_version": 1,
                "rounds": contract.request_cap // users,
                "users_per_round": users,
            },
            "complete sequential loops",
        )
    else:
        equal(loops, None, "unexpected complete-loop declaration")
    equal(
        integer(meta["measured_requests"], "metadata measured requests"),
        contract.measured_requests,
        "metadata request count",
    )
    equal(params["run_id"], meta["run_id"], "parameter run ID")
    integer(params["warmup"], "separate discarded warmups", 2)
    integer(params["chunk_size"], "prefix chunk size", 1)
    for name in ("atol", "rtol"):
        equal(number(params[name], name), 0, f"exact numerical {name}")
    for tier, cap in contract.caps.items():
        gib = number(params[f"{tier}_budget_gib"], f"{tier} GiB cap", positive=True)
        equal(int(gib * 2**30), cap, f"{tier} byte/GiB cap consistency")
    equal(meta["hardware"]["compute_capability"], [9, 0], "SM90 platform")
    require(re.fullmatch(r"cuda(?::[0-9]+)?", params["device"]), "single CUDA device required")
    equal(meta["hardware"]["device"], params["device"], "recorded device")
    integer(meta["hardware"]["total_memory_bytes"], "device memory", 1)
    equal(set(meta["models"]), set(contract.models), "model descriptions")
    if "deepseek_v32" in contract.models:
        deepseek = meta["models"]["deepseek_v32"]
        equal(params["deepseek_layers"], 10, "DeepSeek surrogate scope")
        equal(deepseek["physical_layers"], 10, "DeepSeek replay physical blocks")
        equal(deepseek["chunk_size"], params["chunk_size"], "DeepSeek chunk size")
        if "cache_policy_revision" in deepseek:
            require(
                "deepseek_slots" not in params,
                "shared-pool metadata cannot reinterpret legacy per-session slots",
            )
            equal(
                deepseek["sparse_pool_tokens"],
                integer(params["sparse_pool_tokens"], "shared DeepSeek pool tokens", 1),
                "DeepSeek shared pool tokens",
            )
            equal(
                deepseek["extend_chunk_size"], params["extend_chunk_size"], "DeepSeek extend chunk"
            )
            workspace = integer(params["workspace_query_tokens"], "workspace query tokens", 1)
            require(
                workspace
                >= max(
                    params["chunk_size"], params["extend_chunk_size"] or contract.candidate_tokens
                ),
                "workspace does not cover actual query batches",
            )
        else:
            equal(
                deepseek["sparse_slots"],
                integer(params["deepseek_slots"], "legacy per-session DeepSeek slots", 1),
                "legacy DeepSeek sparse slots",
            )
        equal(deepseek["source_layers"], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], "source blocks")
        equal(deepseek["total_parameters"], 7827793408, "DeepSeek replay parameters")
        equal(
            deepseek["input_semantics"],
            "copy_source_layer_hidden_and_residual_for_each_physical_copy",
            "DeepSeek activation replay",
        )
        equal(
            deepseek["output"],
            "all_candidate_normalized_hidden_and_last_token_lm_head",
            "DeepSeek output boundary",
        )
    if "nosa" in contract.models:
        nosa = meta["models"]["nosa"]
        equal(nosa["layers"], 32, "full NOSA layers")
        equal(nosa["prefix_chunk_size"], params["chunk_size"], "NOSA prefix chunk size")
        equal(
            nosa["max_seq_len"],
            contract.history_tokens + contract.candidate_tokens,
            "NOSA sequence capacity",
        )
        equal(nosa["dtype"], "torch.bfloat16", "NOSA dtype")
        equal(
            nosa["output"],
            "all candidate normalized hidden states; no LM head or decode",
            "NOSA output boundary",
        )
    for description in meta["models"].values():
        require(
            description["device"] == params["device"]
            or (params["device"] == "cuda" and re.fullmatch(r"cuda:[0-9]+", description["device"])),
            "backend device disagrees with request device",
        )
    cases = indexed(meta["cases"])
    equal(list(cases), contract.cases, "ordered complete model/scheme/population matrix")
    reservations = {}
    for case_key, case in cases.items():
        model, scheme, users = case_key
        equal(
            integer(case["requests"], f"{case_key} requests", 1),
            contract.request_counts[users],
            f"{case_key} bounded requests",
        )
        require(case["all_hidden_exact"] is True, f"{case_key} is not all-hidden exact")
        equal(number(case["max_abs"], str(case_key)), 0, f"{case_key} max error")
        number(case["duration_seconds"], f"{case_key} duration", positive=True)
        allocated = integer(case["gpu_peak_allocated_bytes"], str(case_key), 1)
        reserved = integer(case["gpu_peak_reserved_bytes"], str(case_key), 1)
        require(
            allocated <= reserved <= meta["hardware"]["total_memory_bytes"],
            f"{case_key} invalid CUDA allocator peaks",
        )
        reservation = case["session_reservation"]
        equal(set(reservation), set(TIERS), f"{case_key} reservation tiers")
        for tier, cap in contract.caps.items():
            size = integer(reservation[tier], f"{case_key} reserved {tier}")
            require(size <= cap, f"{case_key} session exceeds {tier} cap")
        require(reservation["hbm"] > 0, f"{case_key} zero HBM session reservation")
        shared = case.get("shared_reservation", {"hbm": 0, "dram": 0})
        equal(set(shared), set(TIERS), f"{case_key} shared reservation tiers")
        for tier, cap in contract.caps.items():
            size = integer(shared[tier], f"{case_key} shared {tier}")
            require(
                size + reservation[tier] <= cap,
                f"{case_key} shared plus one session exceeds {tier} cap",
            )
        cpu_workspace = _cpu_execution_workspace(case_key, case, shared)
        require(
            (reservation["dram"] + shared["dram"] - cpu_workspace == 0) == (scheme == "hbm"),
            f"{case_key} inconsistent host backing reservation",
        )
        equal(case.get("sequential_loops"), loops, f"{case_key} complete-loop declaration")
        if model == "deepseek_v32" and "cache_policy_revision" in meta["models"][model]:
            plan = case["cache_resource_plan"]
            equal(
                plan["cache_policy_revision"],
                meta["models"][model]["cache_policy_revision"],
                f"{case_key} cache implementation identity",
            )
            if scheme in ("echo", "serial_sparse"):
                equal(plan["pool_scope"], "backend_per_layer", f"{case_key} pool ownership")
                equal(
                    plan["sparse_pool_tokens"],
                    params["sparse_pool_tokens"],
                    f"{case_key} shared tokens",
                )
                equal(
                    plan["workspace_query_tokens"],
                    params["workspace_query_tokens"],
                    f"{case_key} fixed scratch bound",
                )
                equal(
                    integer(case["host_page_capacity"], "host pages", 1) * 64,
                    plan["host_arena_tokens"],
                    f"{case_key} arena page quota",
                )
                retained_tokens = contract.history_tokens + contract.candidate_tokens
                if plan.get("candidate_persistence") == "gpu_transient":
                    retained_tokens = contract.history_tokens
                equal(
                    case["session_host_pages"],
                    (retained_tokens + 63) // 64,
                    f"{case_key} session page charge",
                )
                equal(
                    reservation["dram"],
                    case["session_host_pages"] * 4,
                    f"{case_key} only session page-table storage is charged per session",
                )
        prior = reservations.setdefault((model, scheme), reservation)
        equal(reservation, prior, f"{case_key} population changed per-session reservation")
    return contract, cases


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def audit_sources(data, repo, meta):
    manifest = read_json(data / "source_manifest.json")
    aggregate = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    equal(aggregate, meta["source_sha256"], "source manifest aggregate")
    require(len(manifest) > 0, "source manifest is empty")
    for name, digest in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        require(re.fullmatch(r"[a-f0-9]{64}", digest), f"bad SHA-256 for {name}")
        for base, label in ((repo, "current"), (data / "source", "snapshot")):
            path = base / relative
            require(path.is_file(), f"{label} source missing: {name}")
            equal(sha256(path), digest, f"{label} source {name}")
    # Check coverage too: a new eligible current source must not evade a byte-only check.
    extensions = {".py", ".sh", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
    current = set()
    for directory in ("models", "layers", "operators", "cache", "executor", "serving", "GR"):
        for path in (repo / directory).rglob("*"):
            if (
                path.is_file()
                and path.suffix in extensions
                and not ({"__pycache__", "output", "generated", "build"} & set(path.parts))
            ):
                current.add(str(path.relative_to(repo)))
    for experiment in ("gr_serving", "deepseek_v32_echo_cache"):
        for path in (repo / "experiments" / experiment).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"} and "output" not in path.parts:
                current.add(str(path.relative_to(repo)))
    provenance = meta.get("backend_provenance")
    if provenance is not None:
        current.update(
            (
                ".gitmodules",
                "pyproject.toml",
                "uv.lock",
                "scripts/prepare_3rdparty.py",
                "experiments/deepseek_v32_echo_prefill/src/backend_provenance.py",
            )
        )
        for name, prefixes in {
            "DeepGEMM": ("csrc", "deep_gemm", "setup.py", "scripts", ".gitmodules"),
            "DeepJIT": ("include",),
            "FlashMLA": ("csrc", "flash_mla", "setup.py", ".gitmodules"),
            "cutlass": ("include", "tools/util/include"),
        }.items():
            listed = subprocess.check_output(
                ["git", "-C", str(repo / "3rdparty" / name), "ls-files", "-z", "--", *prefixes]
            )
            current.update(
                str(Path("3rdparty") / name / entry.decode())
                for entry in listed.split(b"\0")
                if entry and (repo / "3rdparty" / name / entry.decode()).is_file()
            )
    equal(current, set(manifest), "current source manifest coverage")
    snapshot = {
        str(path.relative_to(data / "source"))
        for path in (data / "source").rglob("*")
        if path.is_file()
    }
    equal(snapshot, set(manifest), "saved source manifest coverage")
    for required in (
        "layers/attention.py",
        "layers/feed_forward.py",
        "layers/normalization.py",
        "cache/prefix_pool.py",
        "serving/persistent.py",
        "models/nosa/serving.py",
        "models/deepseek_v32/serving_backend.py",
    ):
        require(required in manifest, f"source coverage omitted {required}")
    if provenance is not None:
        pins = {
            "DeepGEMM": "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7",
            "DeepJIT": "2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43",
            "FlashMLA": "ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
            "cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
        }
        equal(provenance["git_revisions"], pins, "official backend source pins")
        modules = {f"3rdparty/{name}": digest for name, digest in pins.items()}
        audit_installed_provenance(provenance)
        if "flashinfer_runtime_artifacts" in meta:
            audit_runtime_artifacts(meta["flashinfer_runtime_artifacts"])
    else:
        modules = {}
        for line in meta["submodules"].splitlines():
            match = re.fullmatch(r"\s*([0-9a-f]{40})\s+(\S+)(?:\s+.*)?", line)
            require(match is not None, f"unclean or malformed recorded submodule: {line!r}")
            digest, name = match.groups()
            require(name not in modules, f"duplicate submodule {name}")
            modules[name] = digest
        legacy_modules = {"3rdparty/DeepGEMM", "3rdparty/DeepJIT", "3rdparty/cutlass"}
        current_modules = legacy_modules | {"3rdparty/FlashMLA"}
        require(set(modules) in (legacy_modules, current_modules), "dependency set")
        if set(modules) == current_modules:
            # New NOSA-only runs do not load the DeepSeek backend provenance,
            # but still record the current four pinned dependency checkouts.
            # Their working-tree gitlinks may intentionally differ from HEAD;
            # do not reinterpret the captured sources as that historical tree.
            equal(
                modules,
                {
                    "3rdparty/DeepGEMM": "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7",
                    "3rdparty/DeepJIT": "2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43",
                    "3rdparty/FlashMLA": "ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
                    "3rdparty/cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
                },
                "current four-dependency source pins",
            )
    for name, digest in modules.items():
        if provenance is None and "3rdparty/FlashMLA" not in modules:
            tree = git(repo, "ls-tree", meta["git_revision"], "--", name).split()
            equal(tree, ["160000", "commit", digest, name], f"recorded gitlink {name}")
        equal(git(repo / name, "rev-parse", "HEAD"), digest, f"current dependency {name}")
        equal(
            git(repo / name, "status", "--porcelain=v1", "--untracked-files=no"),
            "",
            f"tracked dependency changes {name}",
        )
    return {
        "source_files": len(manifest),
        "source_sha256": aggregate,
        "recorded_git_revision": meta["git_revision"],
        "submodules": modules,
    }


def audit_installed_provenance(provenance):
    """Check the recorded loaded artifacts without importing a CUDA package."""
    require({"deep_gemm", "flash_mla"} <= set(provenance["installed"]), "missing loaded backends")
    for name, package in provenance["installed"].items():
        if name == "flashinfer":
            equal(package["distribution_version"], "0.6.18", "FlashInfer version")
            base = Path(package["package_path"])
            actual = {
                str(path.relative_to(base)): sha256(path)
                for path in base.rglob("*.py")
                if "data" not in path.relative_to(base).parts
            }
            equal(actual, package["python_sha256"], "installed FlashInfer Python sources")
            for field in ("source_sha256", "include_sha256"):
                require(package[field], f"missing FlashInfer {field}")
                for relative, digest in package[field].items():
                    equal(sha256(base / relative), digest, f"FlashInfer {relative}")
            for artifact in package["installed_native_files"]:
                equal(sha256(Path(artifact["path"])), artifact["sha256"], "FlashInfer native")
            continue
        require(name in ("deep_gemm", "flash_mla"), f"unknown backend provenance {name}")
        for artifact in package["files"].values():
            equal(sha256(Path(artifact["path"])), artifact["sha256"], "loaded backend artifact")
        base = Path(package["files"]["python"]["path"]).parent
        actual = {str(path.relative_to(base)): sha256(path) for path in base.rglob("*.py")}
        equal(actual, package["python_sha256"], "loaded backend Python sources")
        if "jit_headers" in package:
            headers = package["jit_headers"]
            require(headers["matches_pinned_sources"] is True, "unmatched installed JIT headers")
            for name, digest in headers["sha256"].items():
                equal(sha256(Path(headers["path"]) / name), digest, "installed JIT header")
    if "flashinfer" not in provenance["installed"]:
        require(provenance["flashinfer"]["files_sha256"], "missing FlashInfer artifacts")
        for name, digest in provenance["flashinfer"]["files_sha256"].items():
            equal(sha256(Path(name)), digest, "installed FlashInfer artifact")


def audit_runtime_artifacts(value):
    """Recheck exposed artifact files; an in-memory IR digest is identity only."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            equal(sha256(Path(value["path"])), value["sha256"], "FlashInfer runtime artifact")
        for child in value.values():
            audit_runtime_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            audit_runtime_artifacts(child)


def scheduler_replay(weights, request_cap, max_revisits, seed):
    """Fresh cumulative weights, independent of GR's mutable Fenwick tree.

    One random draw is used per request. Original weights are retained while a
    user is eligible; eligibility ends after its first visit plus its revisit
    allowance. Strict agreement is required, including near floating boundaries.
    """
    rng = random.Random(seed)
    counts = Counter()
    users = sorted(weights)
    count = (
        request_cap if max_revisits is None else min(request_cap, len(users) * (max_revisits + 1))
    )
    for _ in range(count):
        eligible = [uid for uid in users if max_revisits is None or counts[uid] <= max_revisits]
        cumulative, running = [], 0.0
        for uid in eligible:
            running += weights[uid]
            cumulative.append(running)
        target = rng.random() * math.fsum(weights[uid] for uid in eligible)
        index = min(bisect.bisect_right(cumulative, target), len(eligible) - 1)
        uid = eligible[index]
        counts[uid] += 1
        yield uid


def audit_workload(directory, model, users, contract):
    count = contract.request_counts[users]
    history, candidate = contract.history_tokens, contract.candidate_tokens
    manifest = read_json(directory / "workload.json")
    equal(manifest["schema_version"], 1, f"{model}/{users} workload schema")
    context = manifest["context"]
    default_limit = integer(context["format_default_tokens"], "format context limit", 1)
    override = history + candidate > default_limit
    require(context["override_explicit"] is override, "context override disagrees with geometry")
    equal(
        context["effective_limit_tokens"],
        max(default_limit, history + candidate),
        "effective generation context limit",
    )
    equal(
        context["interpretation"],
        "generation boundary only; does not validate model quality",
        "context interpretation",
    )
    config = {
        "model": model,
        "num_users": users,
        "requests": contract.request_cap,
        "history_tokens": history,
        "candidate_tokens": candidate,
        "seed": contract.seed,
        "sampling": contract.sampling,
        "heat_dataset": contract.heat_dataset,
        "heat_field": contract.heat_field,
        "max_revisits": contract.max_revisits,
        "context_limit": history + candidate if override else None,
    }
    # Legacy Beauty manifests predate the explicit field parameter.
    if "heat_field" not in manifest["config"] and contract.heat_dataset == "beauty":
        config.pop("heat_field")
    if "sampling" not in manifest["config"] and contract.sampling == "weighted":
        config.pop("sampling")
    equal(canonical(manifest["config"]), canonical(config), f"{model}/{users} workload config")
    identity = {
        name: manifest[name] for name in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    equal(
        hashlib.sha256(canonical(identity)).hexdigest(),
        manifest["workload_sha256"],
        f"{model}/{users} signed workload",
    )
    equal(len(manifest["requests"]), count, "manifest identities count")
    if contract.sampling == "sequential":
        equal(manifest["heat_sha256"], None, "sequential has no heat data identity")
        equal(
            manifest["heat"],
            {
                "source": "explicit_synthetic_ids",
                "synthetic": True,
                "selected_users": users,
                "selected_weight_sum": float(users),
                "user_identity": "synthetic IDs 0..N-1",
                "weight_semantics": (
                    "equal placeholders required by GR; no heat curve or probabilistic sampling"
                ),
            },
            "sequential synthetic population metadata",
        )
    else:
        equal(manifest["heat"]["sha256"], manifest["heat_sha256"], "heat identity")
        equal(manifest["heat"]["dataset"], contract.heat_dataset, "heat dataset")
        equal(manifest["heat"]["field"], contract.heat_field, "heat field")
    equal(
        canonical(manifest["schedule"]),
        canonical(
            {
                "seed": contract.seed,
                "qps": 1,
                "arrival": "constant",
                "sampling": contract.sampling,
                "start_timestamp": 0.0,
                "max_revisits": contract.max_revisits,
            }
        ),
        "trace schedule",
    )
    user_rows = manifest["users"]
    for row in user_rows:
        integer(row["user_id"], "population user ID")
    equal(sorted(row["user_id"] for row in user_rows), list(range(users)), "population IDs")
    weights = {
        row["user_id"]: number(row["weight"], "user weight", positive=True) for row in user_rows
    }
    weight_sum = math.fsum(weights.values())
    close(weight_sum, manifest["heat"]["selected_weight_sum"], "selected heat weight")
    if contract.sampling == "sequential":
        equal(weights, dict.fromkeys(range(users), 1.0), "sequential equal placeholder weights")
        replay = [rid % users for rid in range(count)]
    else:
        replay = list(
            scheduler_replay(weights, contract.request_cap, contract.max_revisits, contract.seed)
        )
    requests, counts, prefixes, candidates, previous = [], Counter(), {}, {}, {}
    for rid, request in enumerate(read_jsonl(directory / "requests.jsonl")):
        label = f"{model}/{users}/{rid}"
        equal(integer(request["request_id"], label), rid, f"{label} ordered request ID")
        require(rid < count, f"{label} excess request")
        uid = integer(request["user_id"], f"{label} user")
        require(uid < users, f"{label} out-of-population user")
        if contract.max_revisits is not None:
            require(counts[uid] <= contract.max_revisits, f"{label} user exceeded revisit cap")
        if contract.sampling == "sequential":
            equal(uid, replay[rid], f"{label} independent sequential schedule")
            equal(request["visit_index"], rid // users, f"{label} sequential visit index")
        else:
            equal(uid, replay[rid], f"{label} independent seeded heat schedule")
        equal(
            number(request["user_heat_weight"], label, positive=True),
            weights[uid],
            f"{label} request heat weight",
        )
        equal(request["model"], model, f"{label} model")
        equal(request["stable_prefix_tokens"], history, f"{label} prefix length")
        equal(request["candidate_suffix_tokens"], candidate, f"{label} suffix length")
        equal(request["total_input_tokens"], history + candidate, f"{label} token count")
        ids = request["input_ids"]
        equal(len(ids), history + candidate, f"{label} actual token count")
        require(
            all(type(token) is int and 0 <= token < 2**63 for token in ids),
            f"{label} invalid token IDs",
        )
        equal(request["attention_mask"], [1] * len(ids), f"{label} attention mask")
        prefix, suffix = ids[:history], ids[history:]
        for name, tokens in (("prefix", prefix), ("candidate", suffix), ("input", ids)):
            equal(token_hash(tokens), request[f"{name}_sha256"], f"{label} {name} hash")
        equal(prefixes.setdefault(uid, prefix), prefix, f"{label} stable prefix changed")
        require(uid not in candidates or suffix != candidates[uid], f"{label} unchanged candidate")
        equal(
            integer(request["visit_index"], label), counts[uid], f"{label} contiguous visit index"
        )
        equal(integer(request["visit_number"], label, 1), counts[uid] + 1, f"{label} visit number")
        require(request["is_revisit"] is (counts[uid] > 0), f"{label} revisit label")
        equal(request["previous_request_id"], previous.get(uid), f"{label} previous visit")
        equal(request["timestamp"], float(rid), f"{label} serial trace timestamp")
        equal(
            request["revisit_interval_s"],
            None if uid not in previous else rid - previous[uid],
            f"{label} revisit interval",
        )
        identity_row = {name: request[name] for name in IDENTITY_FIELDS}
        equal(manifest["requests"][rid], identity_row, f"{label} manifest pairing")
        counts[uid] += 1
        candidates[uid], previous[uid] = suffix, rid
        # All retained request fields must appear unchanged in each measured scheme.
        requests.append(
            {
                name: value
                for name, value in request.items()
                if name not in ("input_ids", "attention_mask", "prompt")
            }
        )
    equal(len(requests), count, f"{model}/{users} actual request count")
    if (
        "access_trace_sha256" in manifest
        or contract.heat_dataset == "industrial_10M"
        or contract.sampling == "sequential"
    ):
        trace_identity = [{name: row[name] for name in ACCESS_TRACE_FIELDS} for row in requests]
        equal(
            manifest.get("access_trace_sha256"),
            hashlib.sha256(canonical(trace_identity)).hexdigest(),
            f"{model}/{users} access trace identity",
        )
    observed = {
        "requests": count,
        "unique_users": len(counts),
        "first_visits": len(counts),
        "revisits": count - len(counts),
        "max_visits": max(counts.values(), default=0),
        "max_revisits": max((visits - 1 for visits in counts.values()), default=0),
    }
    if "returning_users" in manifest["observed"]:
        observed["returning_users"] = sum(visits > 1 for visits in counts.values())
    equal(manifest["observed"], observed, f"{model}/{users} visit counts")
    require(
        observed["revisits"] > 0 or contract.allow_empty_revisits,
        f"{model}/{users} no revisits without explicit allowance",
    )
    for row in user_rows:
        uid = row["user_id"]
        equal(row["visits"], counts[uid], "per-user visit count")
        if "revisits" in row:
            equal(row["revisits"], max(0, counts[uid] - 1), "per-user revisit count")
        equal(
            row["prefix_sha256"],
            token_hash(prefixes[uid]) if uid in prefixes else None,
            "per-user fixed prefix identity",
        )
        if contract.sampling == "sequential":
            equal(row["probability"], None, "sequential user has no sampling probability")
        else:
            close(row["probability"], row["weight"] / weight_sum, "user probability")
    return requests, manifest


def audit_expected_trace(directory, users, contract, requests, workload):
    """Match every access and population row to the independently archived CSVs.

    This supplements seeded replay: changing both recorded weights and requests
    coherently cannot substitute another trace for the user-selected dataset.
    No model tokens are expected in these access-only source artifacts.
    """
    archive = read_json(directory / "manifest.json")
    equal(archive["dataset"], contract.heat_dataset, "archived heat dataset")
    equal(archive["field"], contract.heat_field, "archived heat field")
    equal(archive["population_seed"], contract.seed, "archived population seed")
    equal(archive["sampling_seed"], contract.seed, "archived sampling seed")
    equal(archive["requests_per_population"], contract.request_cap, "archived request count")
    require(contract.max_revisits is None, "archived access trace requires no revisit cap")
    equal(archive["schedule"], workload["schedule"], "archived schedule")
    require(users in archive["users"], f"archived trace has no population {users}")
    populations = [row for row in archive["populations"] if row["selected_users"] == users]
    equal(len(populations), 1, "archived population metadata count")
    equal(
        {name: value for name, value in populations[0].items() if name != "path"},
        {name: value for name, value in workload["heat"].items() if name != "path"},
        "archived population heat identity",
    )
    files = {
        name: sha256(directory / name)
        for name in ("manifest.json", f"requests_{users}.csv", f"users_{users}.csv")
    }
    for name, digest in files.items():
        if name != "manifest.json":
            equal(digest, archive["artifact_sha256"][name], f"archived artifact hash {name}")
    weights = {row["user_id"]: row["weight"] for row in workload["users"]}
    ranked = sorted(weights, key=lambda uid: (-weights[uid], uid))
    rank = {uid: index + 1 for index, uid in enumerate(ranked)}
    first, last, counts = {}, {}, Counter()

    def csv_row(values):
        return {name: "" if value is None else str(value) for name, value in values.items()}

    with (directory / f"requests_{users}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = (
            "request_id",
            "user_id",
            "heat_rank",
            "visit_index",
            "is_revisit",
            "previous_request_id",
            "reuse_distance_users",
            "synthetic_timestamp",
        )
        equal(reader.fieldnames, list(fields), "archived request CSV columns")
        for rid, request in enumerate(requests):
            uid = request["user_id"]
            previous = last.get(uid)
            expected = csv_row(
                {
                    "request_id": request["request_id"],
                    "user_id": uid,
                    "heat_rank": rank[uid],
                    "visit_index": request["visit_index"],
                    "is_revisit": int(request["is_revisit"]),
                    "previous_request_id": request["previous_request_id"],
                    "reuse_distance_users": None
                    if previous is None
                    else sum(position > previous for position in last.values()),
                    "synthetic_timestamp": float(request["timestamp"]),
                }
            )
            equal(next(reader, None), expected, f"archived request CSV row {users}/{rid}")
            counts[uid] += 1
            first.setdefault(uid, rid)
            last[uid] = rid
        equal(next(reader, None), None, "archived request CSV excess rows")
    equal(len(requests), archive["requests_per_population"], "archived matched request count")
    weight_sum = math.fsum(weights.values())
    with (directory / f"users_{users}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = (
            "pool_users",
            "user_id",
            "heat_rank",
            "probability",
            "expected_visits",
            "visits",
            "revisits",
            "first_request_id",
            "last_request_id",
        )
        equal(reader.fieldnames, list(fields), "archived user CSV columns")
        for uid in ranked:
            expected = csv_row(
                {
                    "pool_users": users,
                    "user_id": uid,
                    "heat_rank": rank[uid],
                    "probability": weights[uid] / weight_sum,
                    "expected_visits": len(requests) * weights[uid] / weight_sum,
                    "visits": counts[uid],
                    "revisits": max(0, counts[uid] - 1),
                    "first_request_id": first.get(uid),
                    "last_request_id": last.get(uid),
                }
            )
            equal(next(reader, None), expected, f"archived user CSV row {users}/{uid}")
        equal(next(reader, None), None, "archived user CSV excess rows")
    return {"directory": str(directory), "requests": len(requests), "files_sha256": files}


def audit_access_trace_snapshot(data, meta, expected_traces):
    """The copied CSV digest is distinct from each workload's canonical JSON digest."""
    digest = sha256(data / "access_trace.csv")
    equal(digest, meta["access_trace_sha256"], "saved access CSV metadata identity")
    for trace in expected_traces.values():
        for name, expected in trace["files_sha256"].items():
            if name.startswith("requests_"):
                equal(digest, expected, "saved access CSV archived identity")
    return digest


def lru_oracle(requests, reservation, caps, *, shared=None, host_pages=0, session_pages=0):
    """Derive retention from reuse distance, without calling/copying the pool.

    All requests have the same capacity and immutable per-user prefix.  Therefore
    K is the minimum tier capacity, and a reused user hits iff fewer than K other
    users have been accessed since its prior visit.  Last-visit ranks independently
    determine the retained set and exact evicted identity.
    """
    shared = shared or {"hbm": 0, "dram": 0}
    capacities = [(caps[tier] - shared[tier]) // size for tier, size in reservation.items() if size]
    if session_pages:
        capacities.append(host_pages // session_pages)
    capacity = min(capacities)
    require(capacity >= 1, "no whole session fits")
    previous = {}
    for index, request in enumerate(requests):
        uid = request["user_id"]
        more_recent = sum(visit > previous.get(uid, -1) for visit in previous.values())
        hit = uid in previous and more_recent < capacity
        retained_before = set(sorted(previous, key=previous.get, reverse=True)[:capacity])
        previous[uid] = index
        retained_after = set(sorted(previous, key=previous.get, reverse=True)[:capacity])
        yield {
            "hit": hit,
            "evicted": sorted(retained_before - retained_after),
            "cached_users": len(retained_after),
            "capacity": capacity,
        }


def _cpu_execution_workspace(case_key, case, shared):
    """Validate the shared transient allowance; it is not retained host storage."""
    plan = case.get("cache_resource_plan", {})
    cpu_terms = {
        "workspace_cpu_indexer_bytes": 8,
        "workspace_cpu_scalar_bytes": 8,
        "workspace_cpu_metrics_bytes": 24,
    }
    if case_key[0] != "deepseek_v32" or not any(
        name in plan for name in (*cpu_terms, "workspace_cpu_bytes")
    ):
        return 0
    for name, expected in cpu_terms.items():
        equal(integer(plan.get(name), f"{case_key} {name}"), expected, f"{case_key} {name}")
    workspace = integer(plan.get("workspace_cpu_bytes"), f"{case_key} CPU workspace")
    equal(workspace, sum(cpu_terms.values()), f"{case_key} CPU workspace sum")
    require(shared["dram"] >= workspace, f"{case_key} unreserved CPU workspace")
    return workspace


def audit_case_rows(case_key, rows, case, requests, manifest, run_id, caps):
    model, scheme, users = case_key
    equal(
        [row["request_id"] for row in rows],
        list(range(len(requests))),
        f"{case_key} measured request order",
    )
    reservation = case["session_reservation"]
    shared = case.get("shared_reservation", {"hbm": 0, "dram": 0})
    cpu_workspace = _cpu_execution_workspace(case_key, case, shared)
    resource_args = {
        "shared": shared,
        "host_pages": case.get("host_page_capacity", 0),
        "session_pages": case.get("session_host_pages", 0),
    }
    for case_field, manifest_field in (
        ("observed_users", "unique_users"),
        ("revisits", "revisits"),
        ("max_revisits_observed", "max_revisits"),
    ):
        if case_field in case:
            equal(
                integer(case[case_field], case_field),
                manifest["observed"][manifest_field],
                f"{case_key} case {case_field}",
            )
    counts = Counter()
    peaks = Counter()
    for row, request, expected in zip(
        rows, requests, lru_oracle(requests, reservation, caps, **resource_args), strict=True
    ):
        label = f"{case_key}/{row['request_id']}"
        equal(key(row), case_key, f"{label} coordinates")
        equal(row["run_id"], run_id, f"{label} run ID")
        loops = case.get("sequential_loops")
        if loops is not None:
            equal(manifest["config"].get("sampling"), "sequential", f"{label} loop sampling")
            equal(row.get("sequential_loops_schema"), 1, f"{label} loop schema")
            equal(row.get("rounds"), loops["rounds"], f"{label} round count")
            equal(row.get("round_index"), row["request_id"] // users, f"{label} round index")
            equal(
                row.get("round_user_index"), row["request_id"] % users, f"{label} round user index"
            )
            equal(row["user_id"], row["round_user_index"], f"{label} complete user order")
            equal(row["visit_index"], row["round_index"], f"{label} complete round visits")
        else:
            require("sequential_loops_schema" not in row, f"{label} unexpected loop schema")
        equal(row["workload_sha256"], manifest["workload_sha256"], f"{label} workload")
        for name, value in request.items():
            equal(row[name], value, f"{label} request field {name}")
        for name, value in (
            ("history_tokens", manifest["config"]["history_tokens"]),
            ("candidate_tokens", manifest["config"]["candidate_tokens"]),
            ("seed", manifest["config"]["seed"]),
        ):
            equal(row[name], value, f"{label} {name}")
        require(row["prefix_cache_hit"] is expected["hit"], f"{label} LRU hit mismatch")
        equal(row["evicted_users"], expected["evicted"], f"{label} LRU victim mismatch")
        equal(
            integer(row["cached_users"], label, 1),
            expected["cached_users"],
            f"{label} retained user count",
        )
        tier = ("hbm" if scheme == "hbm" else "dram") if expected["hit"] else "miss"
        equal(row["prefix_hit_tier"], tier, f"{label} hit tier")
        number(row["latency_ms"], f"{label} latency", positive=True)
        for phase in PHASES:
            number(row[phase], f"{label} {phase}")
        close(row["latency_ms"], math.fsum(row[name] for name in PHASES), f"{label} phase sum")
        require(row["correctness_exact"] is True, f"{label} correctness flag")
        equal(number(row["correctness_max_abs"], label), 0, f"{label} correctness error")
        require(row.get("phase", "measured") == "measured", f"{label} non-measured row")
        for memory, cap in caps.items():
            equal(row[f"{memory}_budget_bytes"], cap, f"{label} budget equality")
            reserve = integer(row[f"reserved_{memory}_bytes"], label)
            actual = integer(row[f"cache_{memory}_bytes"], label)
            boundary = integer(row[f"request_cache_{memory}_bytes"], label)
            equal(
                reserve,
                shared[memory] + reservation[memory] * expected["cached_users"],
                f"{label} reserved capacity",
            )
            if "shared_reservation" in case:
                equal(
                    row[f"shared_reserved_{memory}_bytes"],
                    shared[memory],
                    f"{label} shared reservation",
                )
                equal(
                    row[f"session_reserved_{memory}_bytes"],
                    reservation[memory] * expected["cached_users"],
                    f"{label} session reservation",
                )
                require(
                    row[f"shared_cache_{memory}_bytes"] <= shared[memory],
                    f"{label} actual shared storage exceeds reservation",
                )
                if memory == "dram":
                    equal(
                        row["shared_cache_dram_bytes"],
                        shared["dram"] - cpu_workspace,
                        f"{label} fixed shared host storage",
                    )
            require(0 <= actual <= boundary <= reserve <= cap, f"{label} {memory} allocation bound")
            if memory == "hbm":
                require(actual > 0, f"{label} missing resident storage")
            else:
                # Both synchronized boundaries retain full host backing/metadata.
                # Discount only an independently validated execution allowance.
                fixed_host = reserve - cpu_workspace
                equal(actual, fixed_host, f"{label} fixed host backing capacity")
                equal(boundary, fixed_host, f"{label} pre-cleanup fixed host backing capacity")
            for name in (
                f"cache_{memory}_bytes",
                f"request_cache_{memory}_bytes",
                f"reserved_{memory}_bytes",
            ):
                peaks[name] = max(peaks[name], row[name])
        if "shared_reservation" in case:
            equal(
                row["cache_host_pages"],
                case["session_host_pages"] * expected["cached_users"],
                f"{label} retained host pages",
            )
            equal(
                row["host_page_capacity"], case["host_page_capacity"], f"{label} host page capacity"
            )
            require(
                row["cache_host_pages"] <= row["host_page_capacity"],
                f"{label} host pages overcommitted",
            )
        counts["hits"] += expected["hit"]
        counts["evictions"] += len(expected["evicted"])
        counts["revisits"] += row["is_revisit"]
        counts["revisit_misses"] += row["is_revisit"] and not expected["hit"]
    require(
        case["duration_seconds"] * 1000 + 1e-5 >= math.fsum(row["latency_ms"] for row in rows),
        f"{case_key} case duration excludes measured request time",
    )
    capacity = next(lru_oracle(requests[:1], reservation, caps, **resource_args))["capacity"]
    return {
        "model": model,
        "scheme": scheme,
        "num_users": users,
        "requests": len(rows),
        "admitted_session_capacity": capacity,
        "max_observed_cached_users": max(row["cached_users"] for row in rows),
        "session_reservation": reservation,
        **counts,
        "first_visits": len(rows) - counts["revisits"],
        "misses": len(rows) - counts["hits"],
        "peaks": dict(peaks),
    }


def audit_correctness(records, measurements, candidate_tokens):
    indexed_records = indexed(records, request=True)
    equal(set(indexed_records), set(measurements), "correctness/request bijection")
    for record_key, record in indexed_records.items():
        row = measurements[record_key]
        require(
            isinstance(record["shape"], list)
            and all(type(size) is int and size > 0 for size in record["shape"]),
            f"{record_key} invalid hidden shape",
        )
        equal(
            record["shape"],
            [candidate_tokens, WIDTHS[row["model"]]],
            f"{record_key} complete hidden shape",
        )
        equal(record["dtype"], "torch.bfloat16", f"{record_key} hidden dtype")
        require(
            record["exact"] is True and row["correctness_exact"] is True,
            f"{record_key} nonexact hidden",
        )
        for name in ("max_abs", "relative_l2", "atol", "rtol"):
            equal(number(record[name], f"{record_key} {name}"), 0, f"{record_key} {name}")
        equal(record["max_abs"], row["correctness_max_abs"], f"{record_key} error pairing")
    return len(indexed_records)


def audit_numerical_tensors(data, contract):
    """Recompute all saved scheme comparisons; reject missing/extra/tampered evidence."""
    import torch

    records = indexed(read_jsonl(data / "correctness.jsonl"), request=True)
    expected_files = {
        f"numerical/{model}/{scheme}/{users}/{rid:06d}.pt"
        for model, scheme, users in contract.cases
        for rid in range(contract.request_counts[users])
    }
    found = {
        str(path.relative_to(data)) for path in (data / "numerical").rglob("*") if path.is_file()
    }
    equal(found, expected_files, "complete numerical tensor file set")

    hashes = {}

    def load(model, scheme, users, rid):
        name = f"numerical/{model}/{scheme}/{users}/{rid:06d}.pt"
        record = records[(model, scheme, users, rid)]
        equal(record["tensor_evidence"]["path"], name, "numerical tensor identity")
        hashes[name] = sha256(data / name)
        equal(hashes[name], record["tensor_evidence"]["sha256"], "numerical tensor digest")
        tensors = torch.load(data / name, map_location="cpu", weights_only=True)
        equal(set(tensors), {"hidden", "logits"}, "numerical tensor fields")
        hidden = tensors["hidden"]
        require(isinstance(hidden, torch.Tensor), f"{name}: hidden must be tensor")
        equal(list(hidden.shape), [contract.candidate_tokens, WIDTHS[model]], "complete hidden")
        equal(hidden.dtype, torch.bfloat16, "hidden dtype")
        require(bool(torch.isfinite(hidden).all()), f"{name}: nonfinite hidden")
        logits = tensors["logits"]
        if model == "deepseek_v32":
            require(isinstance(logits, torch.Tensor), f"{name}: missing last-token logits")
            equal(list(logits.shape), [1, 129280], "complete DeepSeek last-token LM head")
            equal(logits.dtype, torch.float32, "logits dtype")
            require(bool(torch.isfinite(logits).all()), f"{name}: nonfinite logits")
            comparison = record["logits"]
            equal(comparison["shape"], list(logits.shape), "logit record shape")
            equal(comparison["dtype"], str(logits.dtype), "logit record dtype")
            require(comparison["exact"] is True, "nonexact logit record")
            for field in ("max_abs", "relative_l2", "atol", "rtol"):
                equal(number(comparison[field], field), 0, "nonexact logit record")
        else:
            require(logits is None and record["logits"] is None, "NOSA hidden-only output boundary")
        return tensors

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    compared = 0
    try:
        for model in contract.models:
            for users in contract.populations:
                for rid in range(contract.request_counts[users]):
                    reference = load(model, "hbm", users, rid)
                    saved = torch.load(
                        data / "reference" / model / str(users) / f"{rid:06d}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    reference_name = f"reference/{model}/{users}/{rid:06d}.pt"
                    hashes[reference_name] = sha256(data / reference_name)
                    require(torch.equal(saved, reference["hidden"]), "HBM reference files disagree")
                    for scheme in SCHEMES[model]:
                        actual = load(model, scheme, users, rid)
                        for field in ("hidden", "logits"):
                            if reference[field] is not None:
                                require(
                                    torch.equal(actual[field], reference[field]),
                                    f"{model}/{scheme}/{users}/{rid}: recomputed {field} differs",
                                )
                        compared += 1
    finally:
        torch.set_num_threads(threads)
    for name, expected in hashes.items():
        equal(sha256(data / name), expected, "numerical tensor changed during audit")
    return {
        "status": "accepted",
        "complete_hidden_comparisons": compared,
        "files": len(found),
        "files_sha256": hashes,
    }


def latency_stats(values):
    ordered = sorted(values)
    result = {name: None for name in STATS}
    result["count"] = len(ordered)
    if not ordered:
        return result
    # statistics.quantiles uses inclusive sample endpoints; this independently
    # checks the documented (n-1)*p convention used by the report.
    quantiles = (
        statistics.quantiles(ordered, n=100, method="inclusive")
        if len(ordered) > 1
        else ordered * 99
    )
    result.update(
        mean_ms=statistics.fmean(ordered),
        median_ms=statistics.median(ordered),
        p95_ms=quantiles[94],
        p99_ms=quantiles[98],
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )
    return result


def audit_analysis(data, groups, measurement_index):
    summary = read_json(data / "analysis/summary.json")
    new_schema = any("sequential_loops_schema" in row for row in measurement_index.values())
    if new_schema:
        equal(summary.get("schema_version"), 2, "new cache/round report schema")
    expected_rounds = {}
    for case_key, rows in groups.items():
        if "sequential_loops_schema" in rows[0]:
            for index in range(rows[0]["rounds"]):
                expected_rounds[(*case_key, index)] = [
                    row for row in rows if row["round_index"] == index
                ]
    actual_rounds = {}
    for item in summary.get("rounds", []):
        item_key = (*key(item), item["round_index"])
        require(item_key not in actual_rounds, f"duplicate round summary {item_key}")
        actual_rounds[item_key] = item
    equal(set(actual_rounds), set(expected_rounds), "complete per-round report")
    for item_key, rows in expected_rounds.items():
        item = actual_rounds[item_key]
        for name in ("run_id", "workload_sha256"):
            equal(item[name], rows[0][name], f"{item_key} round {name}")
        equal(
            item["prefix_hits"],
            sum(row["prefix_cache_hit"] for row in rows),
            f"{item_key} round hits",
        )
        for name, expected in latency_stats([row["latency_ms"] for row in rows]).items():
            if name == "count":
                equal(item[name], expected, f"{item_key} round {name}")
            else:
                close(item[name], expected, f"{item_key} round {name}")
    if expected_rounds:
        with (data / "analysis/rounds.csv").open(newline="") as handle:
            csv_rounds = list(csv.DictReader(handle))
        expected_csv = [
            {name: str(value) for name, value in item.items()} for item in summary["rounds"]
        ]
        equal(csv_rounds, expected_csv, "per-round CSV values")
    actual = {}
    for group in summary["groups"]:
        group_key = (*key(group), group["scope"])
        require(group_key not in actual, f"duplicate summary {group_key}")
        actual[group_key] = group
    expected_keys = {
        (*case_key, scope) for case_key in groups for scope in ("all", "first_visit", "revisit")
    }
    equal(set(actual), expected_keys, "complete all/first/revisit report scopes")
    for case_key, rows in groups.items():
        for scope in ("all", "first_visit", "revisit"):
            chosen = [
                row for row in rows if scope == "all" or row["is_revisit"] == (scope == "revisit")
            ]
            item = actual[(*case_key, scope)]
            for name in (
                "run_id",
                "workload_sha256",
                "history_tokens",
                "candidate_tokens",
                "seed",
                "hbm_budget_bytes",
                "dram_budget_bytes",
            ):
                equal(item[name], rows[0][name], f"{case_key}/{scope} summary {name}")

            def check_stats(stats, values, label):
                for name, expected in latency_stats(values).items():
                    if expected is None or name == "count":
                        equal(stats[name], expected, f"{label} {name}")
                    else:
                        close(stats[name], expected, f"{label} {name}")

            check_stats(item, [row["latency_ms"] for row in chosen], f"{case_key}/{scope}")
            for phase in ("prefix_ms", "extend_ms", "cleanup_ms"):
                if chosen:
                    check_stats(
                        item[phase], [row[phase] for row in chosen], f"{case_key}/{scope}/{phase}"
                    )
            hits = sum(row["prefix_cache_hit"] for row in chosen)
            equal(item["prefix_hits"], hits, "summary hits")
            equal(item["prefix_misses"], len(chosen) - hits, "summary misses")
            equal(item["unique_users"], len({row["user_id"] for row in chosen}), "summary users")
            equal(
                item["prefix_hit_tiers"],
                dict(Counter(row["prefix_hit_tier"] for row in chosen)),
                "summary tiers",
            )
            equal(
                item["evictions"],
                sum(len(row["evicted_users"]) for row in chosen),
                "summary evictions",
            )
            if chosen:
                close(item["prefix_hit_rate"], hits / len(chosen), "summary hit rate")
                for memory in TIERS:
                    equal(
                        item[f"peak_cache_{memory}_bytes"],
                        max(row[f"cache_{memory}_bytes"] for row in chosen),
                        "summary cache peak",
                    )
            else:
                equal(item["prefix_hit_rate"], None, "empty summary hit rate")
    with (data / "analysis/per_request.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    equal(len(csv_rows), len(measurement_index), "per-request CSV length")
    seen = set()
    for item in csv_rows:
        item_key = (item["model"], item["scheme"], int(item["num_users"]), int(item["request_id"]))
        require(
            item_key in measurement_index and item_key not in seen,
            "per-request CSV request pairing",
        )
        seen.add(item_key)
        source = measurement_index[item_key]
        for name, value in source.items():
            encoded = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else ("" if value is None else str(value))
            )
            equal(item[name], encoded, f"CSV {item_key}/{name}")
        equal(item["prefix_hit"], str(source["prefix_cache_hit"]), "CSV normalized hit")
    with (data / "analysis/summary.csv").open(newline="") as handle:
        csv_summary = list(csv.DictReader(handle))
    equal(len(csv_summary), len(actual), "summary CSV length")
    seen = set()
    for item in csv_summary:
        item_key = (item["model"], item["scheme"], int(item["num_users"]), item["scope"])
        require(item_key in actual and item_key not in seen, "summary CSV pairing")
        seen.add(item_key)
        for name, value in actual[item_key].items():
            encoded = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else ("" if value is None else str(value))
            )
            equal(item[name], encoded, f"summary CSV {item_key}/{name}")
    for filename in ("per_request.svg", "summary.svg"):
        path = data / "analysis" / filename
        require(path.is_file() and path.stat().st_size > 0, f"missing {filename}")
    return {
        "summary_groups": len(actual),
        "per_request_csv_rows": len(csv_rows),
        "figures": "existence only; rendered visual inspection is a separate gate",
    }


def audit_references(data, mode, contract):
    if mode == "existence":
        return _audit_reference_files(data, mode, contract, None)
    require(mode == "cpu", "unsupported reference check mode")
    # map_location forces CPU storage; this path never calls a CUDA API.
    import torch

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _audit_reference_files(data, mode, contract, torch)
    finally:
        torch.set_num_threads(previous_threads)


def _audit_reference_files(data, mode, contract, torch):
    expected = {
        f"reference/{model}/{users}/{rid:06d}.pt"
        for model in contract.models
        for users, count in contract.request_counts.items()
        for rid in range(count)
    }
    found = {
        str(path.relative_to(data)) for path in (data / "reference").rglob("*") if path.is_file()
    }
    equal(found, expected, "complete HBM reference file set")
    digests, total_bytes, elements, previous_group = {}, 0, 0, None
    for name in sorted(expected):
        path = data / name
        group = str(Path(name).parent)
        if group != previous_group:
            print(f"Checking {group} ({mode}).", file=sys.stderr, flush=True)
            previous_group = group
        require(path.stat().st_size > 0, f"empty reference {name}")
        total_bytes += path.stat().st_size
        if torch is not None:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            require(isinstance(tensor, torch.Tensor), f"{name}: reference is not a tensor")
            equal(tensor.device.type, "cpu", f"{name} reference device")
            equal(
                list(tensor.shape),
                [contract.candidate_tokens, WIDTHS[Path(name).parts[1]]],
                f"{name} reference shape",
            )
            equal(str(tensor.dtype), "torch.bfloat16", f"{name} reference dtype")
            require(bool(torch.isfinite(tensor).all().item()), f"{name}: nonfinite reference")
            elements += tensor.numel()
            del tensor
            digests[name] = sha256(path)
    return {
        "mode": mode,
        "files": len(expected),
        "file_bytes": total_bytes,
        "finite_elements_checked": elements,
        "file_hash_manifest_sha256": hashlib.sha256(canonical(digests)).hexdigest()
        if digests
        else None,
    }


def audit(
    data,
    repo,
    *,
    expected_run=None,
    expected_source=None,
    reference_check="cpu",
    expected_trace_directory=None,
):
    # Fail before touching tensor files or writing output when the run is still active.
    meta = read_json(data / "metadata.json")
    contract, cases = audit_metadata(meta, expected_run, expected_source)
    require(
        contract.heat_dataset != "industrial_10M" or expected_trace_directory is not None,
        "industrial runs require --expected-trace-directory to verify the selected access dataset",
    )
    require(
        contract.sampling != "sequential" or expected_trace_directory is None,
        "sequential runs do not use an archived heat trace",
    )
    provenance = audit_sources(data, repo, meta)
    rows = list(read_jsonl(data / "measurements.jsonl"))
    equal(len(rows), contract.measured_requests, "complete bounded measured rows")
    measurement_index = indexed(rows, request=True)
    equal(
        [key(row, True) for row in rows],
        [
            (*case_key, rid)
            for case_key in contract.cases
            for rid in range(contract.request_counts[case_key[2]])
        ],
        "ordered measured matrix",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    equal(set(groups), set(cases), "measured cases")
    correctness_count = audit_correctness(
        read_jsonl(data / "correctness.jsonl"), measurement_index, contract.candidate_tokens
    )
    case_results, workload_hashes, expected_traces = [], {}, {}
    for model in contract.models:
        for users in contract.populations:
            directory = data / "workloads" / model / str(users)
            requests, manifest = audit_workload(directory, model, users, contract)
            workload_hashes[f"{model}/{users}"] = manifest["workload_sha256"]
            if expected_trace_directory is not None:
                expected_traces[f"{model}/{users}"] = audit_expected_trace(
                    expected_trace_directory, users, contract, requests, manifest
                )
            if contract.sampling == "weighted":
                equal(
                    sha256(Path(manifest["heat"]["path"])),
                    manifest["heat_sha256"],
                    "current heat data hash",
                )
            for scheme in SCHEMES[model]:
                case_key = (model, scheme, users)
                case_results.append(
                    audit_case_rows(
                        case_key,
                        groups[case_key],
                        cases[case_key],
                        requests,
                        manifest,
                        meta["run_id"],
                        contract.caps,
                    )
                )
    access_csv_digest = None
    if "access_trace_sha256" in meta or contract.heat_dataset == "industrial_10M":
        access_csv_digest = audit_access_trace_snapshot(data, meta, expected_traces)
    analysis = audit_analysis(data, groups, measurement_index)
    print(
        "All structural, numerical-record, pairing, budget, LRU and report gates passed; checking HBM reference files.",
        file=sys.stderr,
        flush=True,
    )
    references = audit_references(data, reference_check, contract)
    tensors = None
    if meta.get("numerical_evidence_schema") == 1 and reference_check == "cpu":
        tensors = audit_numerical_tensors(data, contract)
    # Freeze the audited evidence against changes during this audit as well.
    equal(read_json(data / "metadata.json"), meta, "metadata changed during final audit")
    audit_sources(data, repo, meta)
    for trace in expected_traces.values():
        for name, digest in trace["files_sha256"].items():
            equal(
                sha256(Path(trace["directory"]) / name),
                digest,
                f"archived trace changed during final audit: {name}",
            )
    evidence = {
        name: sha256(data / name)
        for name in (
            "metadata.json",
            "source_manifest.json",
            "measurements.jsonl",
            "correctness.jsonl",
            "analysis/summary.json",
            "analysis/summary.csv",
            "analysis/per_request.csv",
            "analysis/summary.svg",
            "analysis/per_request.svg",
        )
    }
    if tensors is not None:
        evidence.update(tensors["files_sha256"])
    if meta.get("sequential_loops") is not None:
        evidence["analysis/rounds.csv"] = sha256(data / "analysis/rounds.csv")
    if access_csv_digest is not None:
        equal(sha256(data / "access_trace.csv"), access_csv_digest, "saved access CSV changed")
        evidence["access_trace.csv"] = access_csv_digest
    return {
        "audit_schema": 1,
        "status": "accepted" if reference_check == "cpu" else "structural_only",
        "run_id": meta["run_id"],
        "data_path": str(data),
        "audited_unix": time.time(),
        "auditor_sha256": sha256(Path(__file__)),
        "cases": len(cases),
        "measured_requests": len(rows),
        "correctness_records": correctness_count,
        "exact_non_hbm_comparisons": sum(row["scheme"] != "hbm" for row in rows),
        "requested_models": contract.models,
        "requested_populations": contract.populations,
        "history_tokens": contract.history_tokens,
        "candidate_tokens": contract.candidate_tokens,
        "request_upper_cap": contract.request_cap,
        "max_revisits": contract.max_revisits,
        "allow_empty_revisits": contract.allow_empty_revisits,
        "heat_dataset": contract.heat_dataset,
        "heat_field": contract.heat_field,
        "sampling": contract.sampling,
        "per_population_requests": contract.request_counts,
        "budgets": contract.caps,
        "provenance": provenance,
        "workloads": workload_hashes,
        "expected_access_traces": expected_traces,
        "references": references,
        "numerical_tensors": tensors,
        "analysis": analysis,
        "case_audit": case_results,
        "evidence_sha256": evidence,
        "limits": [
            "Saved complete hidden/logits comparisons were independently recomputed on CPU; no GPU execution was repeated."
            if tensors is not None
            else "Numerical records are audited; non-HBM outputs are not persisted, so this is not a fresh numerical rerun.",
            "Cache allocations are sampled request boundaries and reservations, not a continuous process peak.",
            "GPU allocator peaks are checked separately; whole-user LRU is not a token/page cache hit metric.",
            "Rendered figures, publication files, profile overlap and Supervisor updates require separate completion.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="completed output/data/<run_id> directory")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--json", dest="output", type=Path, required=True, help="new JSON path; never overwritten"
    )
    parser.add_argument("--expected-run-id", help="optional explicit expected run identity")
    parser.add_argument(
        "--expected-source-sha256", help="optional independent expected source digest"
    )
    parser.add_argument("--reference-check", choices=("cpu", "existence"), default="cpu")
    parser.add_argument(
        "--expected-trace-directory",
        type=Path,
        help="access-only dataset directory with manifest and requests/users CSVs; required for industrial runs",
    )
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), f"audit destination already exists: {args.output}")
        result = audit(
            args.data.resolve(),
            args.repo.resolve(),
            expected_run=args.expected_run_id,
            expected_source=args.expected_source_sha256,
            reference_check=args.reference_check,
            expected_trace_directory=args.expected_trace_directory.resolve()
            if args.expected_trace_directory is not None
            else None,
        )
        with args.output.open("x") as handle:
            handle.write(
                json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "audit_json": str(args.output),
                    "cases": result["cases"],
                    "measured_requests": result["measured_requests"],
                }
            )
        )
        return 0 if result["status"] == "accepted" else 2
    except (
        AuditError,
        KeyError,
        TypeError,
        OSError,
        ImportError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
