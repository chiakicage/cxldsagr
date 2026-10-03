"""Fixed physical caches; only the workload population changes."""

from collections import OrderedDict

PROFILE = "fixed_budget_512"
POPULATIONS = (64, 128, 256, 384, 512)
REQUESTS = 512
HOST_USERS = 128
RESIDENT_USERS = 93
DEVICE_TOKENS = 66624
GIB = 1 << 30


def validate_trace(metadata):
    generator = metadata.get("generator", {})
    schedule = generator.get("schedule_config", {})
    if (
        metadata["artifact_type"] != "prepared_gr_workload"
        or metadata["stable_prefix_tokens"] != 65536
        or metadata["candidate_suffix_tokens"] != 1024
        or metadata["stats"]["population_users"] not in POPULATIONS
        or metadata["stats"]["requests"] != REQUESTS
        or generator.get("curve_dataset") != "beauty"
        or generator.get("model") != "deepseek_v32"
        or schedule.get("seed") != 42
        or schedule.get("sampling") != "weighted"
        or schedule.get("arrival") != "poisson"
    ):
        raise ValueError(
            "fixed-budget sweep requires 512 requests, 64K/1K and a supported population"
        )


def validate_configuration(args, metadata, plan):
    validate_trace(metadata)
    if (
        args.layers != 5
        or args.host_users != HOST_USERS
        or args.device_cache_tokens != DEVICE_TOKENS
        or args.hbm_budget_gib != 72
        or args.host_cache_budget_gib != 66
        or args.workspace_reserve_gib != 8
        or args.non_torch_reserve_gib != 2
        or args.collect_transfers
        or args.repetition != 1
    ):
        raise ValueError("fixed-budget profile parameters differ from the frozen protocol")
    expected = RESIDENT_USERS if args.mode == "resident" else HOST_USERS
    if plan.retained_users_capacity != expected:
        raise ValueError("fixed cache capacity changed; do not adapt it to the workload")


def lru_outcomes(user_ids, capacity):
    if capacity < 1:
        raise ValueError("positive LRU capacity required")
    cache, outcomes = OrderedDict(), []
    for uid in user_ids:
        hit = uid in cache
        evicted = []
        if not hit and len(cache) == capacity:
            victim, _ = cache.popitem(last=False)
            evicted.append(victim)
        cache[uid] = None
        cache.move_to_end(uid)
        outcomes.append((hit, evicted, len(cache)))
    return outcomes


def validate_outcomes(rows, capacity):
    expected = lru_outcomes((row["user_id"] for row in rows), capacity)
    seen = set()
    for ordinal, (row, (hit, evicted, retained)) in enumerate(zip(rows, expected, strict=True)):
        uid = row["user_id"]
        if (
            row["ordinal"] != ordinal
            or row["first_visit"] != (uid not in seen)
            or row["prefix_reused"] != hit
            or row["evicted_user_ids"] != evicted
            or row["retained_prefix_tokens"] != retained * 65536
            or (hit and row["prefill_ms"] != 0)
            or (not hit and row["prefill_ms"] <= 0)
        ):
            raise ValueError(f"request {ordinal} disagrees with the fixed-capacity user LRU")
        seen.add(uid)
