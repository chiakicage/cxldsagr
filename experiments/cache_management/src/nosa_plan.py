"""Offline NOSA fixed-quota allocation declarations, without model/GPU allocation.

This reuses the production layout and allocator formulas under explicitly
assumed SM90/native-allocator conditions. It does not validate those conditions
on a device, measure allocator occupancy, or establish that a population fits.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import torch

from cache.allocator.budget import allocator_policy
from cache.capacity import allocation_footprint
from models.nosa.config import NosaConfig
from models.nosa.execution.compute_graphs import NosaComputeGraphs
from models.nosa.execution.fixed_resources import NosaFixedResources
from models.nosa.execution.session_budget import SCHEMES, session_allocation_layout

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "cache/capacity.py",
    "cache/lifecycle.py",
    "cache/allocator/budget.py",
    "cache/allocator/snapshot.py",
    "cache/allocator/pool_referrers.py",
    "models/nosa/config.py",
    "models/nosa/execution/resources.py",
    "models/nosa/execution/fixed_resources.py",
    "models/nosa/execution/session_budget.py",
    "models/nosa/execution/compute_graphs.py",
    "operators/nosa/attention/workspace.py",
    "operators/nosa/attention/offload/api.py",
    "pyproject.toml",
    "uv.lock",
    "experiments/cache_management/src/nosa_plan.py",
)


class OfflineNosaFixedResources(NosaFixedResources):
    """Keep native formulas while explicitly replacing only live validation.

    The inherited planner never allocates storage. A concrete cuda:0 device
    describes the accounting role and avoids current-device discovery.
    """

    def validate_native(self):
        if (
            self.device != torch.device("cuda:0")
            or self.dtype != torch.bfloat16
            or self.head_dim != 128
            or self.query_heads != 16 * self.kv_heads
        ):
            raise ValueError("Offline native NOSA planning requires CUDA/BF16/D128/GQA16")

    def _validate_allocator(self):
        return {
            "policy": allocator_policy(self.device),
            "configuration_key": "offline_assumption_not_observed",
        }


def _footprint(allocations, *, payload=False):
    # Payload also obeys phase maxima and zero-charge alias rules. It is a
    # declared live-storage bound, not a sum of mutually exclusive temporaries.
    if payload:
        allocations = tuple(
            replace(item, charged_bytes=0 if item.alias_of else item.storage_bytes)
            for item in allocations
        )
    return asdict(allocation_footprint(tuple(allocations)))


def _category(item):
    name = item.name
    if item.owner == "shared":
        if name in ("pool.keys", "pool.values"):
            return "pool_history_and_candidate_tail"
        if name.startswith(("pool.tags", "prefetch.")):
            return "mapping_and_counters"
        return "workspace"
    if name.startswith(("host.keys", "host.values")):
        return "host_history"
    if name.startswith("indexer.layer"):
        return "derived_records"
    if name.startswith("indexer.workspace"):
        return "indexer_current_and_retiring_workspace"
    if name.startswith("pending."):
        return "pending_append"
    if name.startswith("helper."):
        return "helper_phase_peak"
    if name.startswith("metrics."):
        return "device_metrics"
    if name.startswith("host.") or name == "indexer.host_flag":
        return "host_validation_and_observation"
    return "resident_kv_and_cis"


def _summarize(allocations):
    groups = {}
    for item in allocations:
        if item.alias_of is None:
            groups.setdefault(_category(item), []).append(item)
    return {
        "storage_payload_bound": _footprint(allocations, payload=True),
        "allocator_reservation_bound": _footprint(allocations),
        "categories": {
            name: {
                "storage_payload_bound": _footprint(items, payload=True),
                "allocator_reservation_bound": _footprint(items),
            }
            for name, items in sorted(groups.items())
        },
        "allocations": [asdict(item) for item in allocations],
    }


def _population(count, ceiling, shared, session):
    def total(field):
        return {
            tier: shared[field][tier] + count * session[field][tier] for tier in ("hbm", "dram")
        }

    return {
        "sessions": count,
        "admissible_under_token_quota": count <= ceiling,
        "physical_fit_validated": False,
        "storage_payload_bound_excluding_graphs": total("storage_payload_bound"),
        "allocator_reservation_bound_excluding_graphs": total("allocator_reservation_bound"),
        "boundary": (
            "Shared storage is counted once. Every session retains its established full "
            "reservation, including temporary bounds. These sums are not measured peaks "
            "or a claim that this population can coexist in physical HBM/DRAM."
        ),
    }


def build_plan(
    config, *, history=65536, candidate=128, chunk=1024, pool=65536, host=16777216, users=16
):
    if type(users) is not int or users <= 0:
        raise ValueError("users must be a positive integer")
    limits = {
        "max_session_capacity": history + candidate,
        "max_history_tokens": history,
        "max_candidate_tokens": candidate,
    }
    cases = {}
    for scheme in SCHEMES:
        resources = OfflineNosaFixedResources(
            scheme=scheme,
            layers=config.num_hidden_layers,
            kv_heads=config.num_key_value_heads,
            query_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            chunk_size=chunk,
            sparse_pool_tokens=pool,
            host_arena_tokens=host,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        resource_plan = resources.plan_resources(None, limits)
        session_allocations, _ = session_allocation_layout(
            capacity=history + candidate,
            queries=max(min(chunk, history), candidate),
            layers=config.num_hidden_layers,
            kv_heads=config.num_key_value_heads,
            query_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
            scheme=scheme,
            host_capacity=history,
            dense_counter=True,
        )
        shared, session = _summarize(resource_plan.allocations), _summarize(session_allocations)
        history_pages = history // resource_plan.page_size
        ceiling = (
            resource_plan.hbm_tokens // history
            if scheme == "hbm"
            else resource_plan.host_pages // history_pages
        )
        cases[scheme] = {
            "shared": shared,
            "one_session": session,
            "quota": {
                "per_layer_pool_history_tokens": 0 if scheme == "hbm" else pool,
                "retained_hbm_history_tokens": resource_plan.hbm_tokens,
                "retained_host_history_pages": resource_plan.host_pages,
                "page_tokens": resource_plan.page_size,
                "full_history_sessions": ceiling,
                "host_allocation": "none" if scheme == "hbm" else "lazy_per_session",
            },
            "populations": {
                "requested_users_if_all_retained": _population(users, ceiling, shared, session),
                "admissible_requested_users": _population(
                    min(users, ceiling), ceiling, shared, session
                ),
                "full_token_quota": _population(ceiling, ceiling, shared, session),
            },
            "resource_metadata": dict(resource_plan.metadata),
        }
    private_limit = (
        inspect.signature(NosaComputeGraphs.__init__).parameters["private_limit_bytes"].default
    )
    return {
        "schema": "nosa-offline-allocation-plan-v1",
        "status": "static_plan_not_execution",
        "parameters": {
            "history_tokens": history,
            "candidate_tokens": candidate,
            "chunk_size": chunk,
            "sparse_pool_tokens": pool,
            "host_arena_tokens": host,
            "requested_users": users,
        },
        "assumptions": {
            "architecture": "SM90",
            "dtype": "bfloat16",
            "allocator_policy": allocator_policy("cuda:0"),
            "live_hardware_validated": False,
            "live_allocator_validated": False,
            "model_weights_loaded": False,
            "cuda_initialized_by_planner": False,
        },
        "graph": {
            "chosen_private_limit_bytes": private_limit,
            "static_allocation_limit_bytes": None,
            "static_allocated_bytes": None,
            "static_storage_bytes": None,
            "private_reserved_bytes": None,
            "boundary": (
                "The chosen private limit is read from the production constructor default. "
                "Static limits and actual graph accounting require model/graph preparation "
                "and are supplied separately by matched motivation metadata. The private "
                "limit is not a measured allocation or a complete graph reservation."
            ),
        },
        "excluded": [
            "model_weights",
            "ordinary_activations",
            "inactive_allocator_segments",
            "graph_static_and_private_actuals",
            "driver_and_library_memory",
        ],
        "boundaries": [
            "Token quotas do not establish physical fit; full-quota figures are offline sums.",
            "Fixed host payload uses H, while the existing reservation bounds its pinned bin at H+A.",
            "Derived records/workspaces remain lazy; declared payload bounds are not current allocations.",
            "Final shared storage excludes the preserved temporary fetch-K/V initializer peak.",
        ],
        "model_config": asdict(config),
        "cases": cases,
    }


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id", required=True)
    command.add_argument("--model-path", type=Path, default=Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B"))
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--history-tokens", type=int, default=65536)
    command.add_argument("--candidate-tokens", type=int, default=128)
    command.add_argument(
        "--context-limit",
        type=int,
        help="explicit execution limit; defaults to H+A as in motivation",
    )
    command.add_argument("--chunk-size", type=int, default=1024)
    command.add_argument("--sparse-pool-tokens", type=int, default=65536)
    command.add_argument("--host-arena-tokens", type=int, default=16777216)
    command.add_argument("--num-users", type=int, default=16)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    output = args.output_dir or ROOT / "experiments/cache_management/output/data" / args.run_id
    if output.exists():
        raise FileExistsError(output)
    config_path = args.model_path / "config.json"
    config = NosaConfig.from_pretrained(args.model_path)
    checkpoint_context = config.max_position_embeddings
    context_limit = (
        args.history_tokens + args.candidate_tokens
        if args.context_limit is None
        else args.context_limit
    )
    if not args.history_tokens + args.candidate_tokens <= context_limit <= 262144:
        raise ValueError("context limit must cover H+A and not exceed 262144")
    config = replace(config, max_position_embeddings=context_limit)
    result = build_plan(
        config,
        history=args.history_tokens,
        candidate=args.candidate_tokens,
        chunk=args.chunk_size,
        pool=args.sparse_pool_tokens,
        host=args.host_arena_tokens,
        users=args.num_users,
    )
    result.update(
        run_id=args.run_id,
        config_source={
            "path": str(config_path.resolve()),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        sources={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS
        },
    )
    result["parameters"]["context_limit"] = context_limit
    result["assumptions"].update(
        checkpoint_context_limit=checkpoint_context,
        execution_context_limit=context_limit,
        long_context_quality_validated=False,
    )
    output.mkdir(parents=True)
    (output / "checkpoint_config.json").write_bytes(config_path.read_bytes())
    for name, expected in result["sources"].items():
        contents = (ROOT / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise RuntimeError(f"Planning source changed during the run: {name}")
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    (output / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Offline NOSA allocation plan: {output / 'plan.json'}")
    return result


if __name__ == "__main__":
    main()
