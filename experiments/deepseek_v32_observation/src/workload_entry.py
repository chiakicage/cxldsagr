"""Shared fixture and identity preparation for independent check/bench/profile."""

from __future__ import annotations

import torch

from experiments.deepseek_v32_observation.src.contract import Shape, task_keys
from experiments.deepseek_v32_observation.src.provenance import execution_identity


def prepare(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Observation explicitly requires an SM90/Hopper GPU")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Observation supports SM90/Hopper")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import deep_gemm

    deep_gemm.get_num_sms()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    shape = Shape(args.B, args.Q, args.H)
    key = shape.key if args.group == "attention" else f"s{args.S}"
    if key not in task_keys(args.group):
        raise ValueError("Workload is outside the requested observation matrix")
    if args.group == "common":
        from experiments.deepseek_v32_observation.src.common_workload import build_common

        owner = build_common(
            args.S, device=f"cuda:{args.device}", seed=args.seed, checkpoint=args.checkpoint
        )
    elif args.group == "moe":
        from experiments.deepseek_v32_observation.src.moe_workload import build_moe

        owner = build_moe(
            args.S, device=f"cuda:{args.device}", seed=args.seed, checkpoint=args.checkpoint
        )
    else:
        from experiments.deepseek_v32_observation.src.attention_workload import (
            build_attention_workload,
        )

        owner = build_attention_workload(
            shape.as_dict(), device=f"cuda:{args.device}", seed=args.seed
        )
    if args.group == "attention":
        cases = [
            {
                "name": name,
                "call": call,
                "work": owner.operator_metadata[name],
                "metadata": owner.operator_metadata[name],
                "before_call": owner.before_call(name) if hasattr(owner, "before_call") else None,
            }
            for name, call in owner.operations.items()
        ]
    else:
        cases = [
            {
                "name": case.name,
                "call": case.call,
                "work": case.work.as_dict() if hasattr(case.work, "as_dict") else case.work,
                "metadata": case.metadata,
                "verify": case.verify,
                "before_call": owner.before_call(case.name)
                if hasattr(owner, "before_call")
                else None,
            }
            for case in owner.cases
        ]
    input_identity = owner.input_identity()
    identity = execution_identity(
        group=args.group,
        key=key,
        seed=args.seed,
        checkpoint=args.checkpoint,
        operators=[{k: case[k] for k in ("name", "work", "metadata")} for case in cases],
        workload_identity=input_identity,
    )
    return owner, cases, identity, shape, key


def refresh_attention(owner, group):
    if group == "attention":
        owner.run()
        torch.cuda.synchronize()


def planned_inner(case, requested):
    # Cache packing owns a full resident duplicate per graph call. This bound
    # is chosen before graph allocation, not after an allocation failure.
    if case["name"] in {"indexer_prepare", "page64_pack"} or case.get("before_call") is not None:
        return 1
    return requested
