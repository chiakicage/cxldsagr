"""Reject clearly impossible pinned pools before loading the GPU model.

The pinned Torch 2.8 allocator rounds a single allocation to a power of two.
Managed non-Movable memory is only an upper bound, not available pinned memory.
This check neither reserves pages nor changes NUMA or memory-hotplug policy.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

GIB = 1 << 30


def rounded_pinned_bytes(payload):
    if type(payload) is not int or payload <= 0:
        raise ValueError("positive pinned payload required")
    return 1 << (payload - 1).bit_length()


def plan_host_cache(*, users, layers, prefix, suffix, budget_bytes):
    if any(type(x) is not int or x <= 0 for x in (users, layers, prefix, suffix, budget_bytes)):
        raise ValueError("positive Host cache dimensions and budget required")
    guard_tokens = ((suffix + 63) // 64 + 1) * 64
    history = users * prefix * layers * 1152
    payload = history + guard_tokens * layers * 1152
    rounded = rounded_pinned_bytes(payload)
    staging_each = (prefix + guard_tokens + 64) * 1152
    staging = 2 * rounded_pinned_bytes(staging_each)
    metadata, safety = GIB // 2, 5 * GIB // 4
    if rounded + staging + metadata + safety > budget_bytes:
        raise MemoryError(
            "rounded KV, worst-case dense staging, metadata and safety exceed Host budget"
        )
    return {
        "total_cache_budget_bytes": budget_bytes,
        "history_kv_bytes": history,
        "candidate_guard_bytes": payload - history,
        "main_kv_tensor_bytes": payload,
        "main_kv_rounded_pinned_bytes": rounded,
        "main_allocator_padding_bytes": rounded - payload,
        "dense_staging_tensor_bytes": 2 * staging_each,
        "dense_staging_rounded_pinned_bytes": staging,
        "metadata_reserve_bytes": metadata,
        "safety_reserve_bytes": safety,
        "max_pinned_reserved_bytes": rounded + staging,
        "scope": "cache only; model loading, input trace and system headroom checked separately",
    }


def cgroup_memory_limits(cgroup_text=None, mountinfo_text=None):
    """Inspect every visible ancestor limit, including memory.high on cgroup v2."""
    groups = (
        Path("/proc/self/cgroup").read_text() if cgroup_text is None else cgroup_text
    ).splitlines()
    mounts = (
        Path("/proc/self/mountinfo").read_text() if mountinfo_text is None else mountinfo_text
    ).splitlines()
    memberships = [line.split(":", 2) for line in groups]
    records, found = [], False
    for mount in mounts:
        before, after = mount.split(" - ", 1)
        fields, fs = before.split(), after.split()
        version = (
            2
            if fs[0] == "cgroup2"
            else 1
            if fs[0] == "cgroup" and "memory" in fs[2].split(",")
            else None
        )
        if version is None:
            continue
        member = next(
            (
                path
                for _, controllers, path in memberships
                if (version == 2 and not controllers)
                or (version == 1 and "memory" in controllers.split(","))
            ),
            None,
        )
        if member is None:
            continue
        unescape = lambda value: re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
        root, mountpoint = Path(unescape(fields[3])), Path(unescape(fields[4]))
        relative = Path(member).relative_to(root)
        current = mountpoint / relative
        found = True
        while True:
            names = ("memory.max", "memory.high") if version == 2 else ("memory.limit_in_bytes",)
            usage_name = "memory.current" if version == 2 else "memory.usage_in_bytes"
            for name in names:
                limit_path = current / name
                if limit_path.exists():
                    raw = limit_path.read_text().strip()
                    if raw != "max" and int(raw) < 1 << 60:
                        limit, used = int(raw), int((current / usage_name).read_text())
                        records.append(
                            {
                                "path": str(limit_path),
                                "limit_bytes": limit,
                                "usage_bytes": used,
                                "headroom_bytes": max(0, limit - used),
                            }
                        )
            if current == mountpoint:
                break
            current = current.parent
    if not found:
        raise RuntimeError(
            "cannot resolve the applicable memory cgroup; refusing unchecked allocation"
        )
    return records


def available_non_movable_bytes(zoneinfo, allowed_nodes, page_bytes):
    zone, values, totals = None, {}, []
    for line in zoneinfo.splitlines() + ["Node 999999, zone End"]:
        header = re.fullmatch(r"Node\s+(\d+), zone\s+(\w+)", line.strip())
        if header:
            if zone and zone[0] in allowed_nodes and zone[1] in {"DMA", "DMA32", "Normal"}:
                totals.append(
                    max(
                        0,
                        values.get("pages free", 0)
                        + values.get("nr_zone_inactive_file", 0)
                        - values.get("nr_zone_write_pending", 0)
                        - values.get("high", 0),
                    )
                )
            zone, values = (int(header[1]), header[2]), {}
        elif zone:
            match = re.fullmatch(
                r"(pages free|nr_zone_inactive_file|nr_zone_write_pending|high)\s+(\d+)",
                line.strip(),
            )
            if match:
                values[match[1]] = int(match[2])
    return sum(totals) * page_bytes


def inspect_host_availability(plan, *, startup_reserve_bytes=32 * GIB):
    status = Path("/proc/self/status").read_text()
    nodes = parse_nodes(
        next(
            line.split(":", 1)[1]
            for line in status.splitlines()
            if line.startswith("Mems_allowed_list:")
        )
    )
    zones = Path("/proc/zoneinfo").read_text()
    managed = non_movable_bytes(zones, nodes, os.sysconf("SC_PAGE_SIZE"))
    available = available_non_movable_bytes(zones, nodes, os.sysconf("SC_PAGE_SIZE"))
    meminfo = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    limits = cgroup_memory_limits()
    system_reserve = 10 * GIB
    cache = plan["total_cache_budget_bytes"]
    required = cache + startup_reserve_bytes + system_reserve
    if cache + system_reserve > min(managed, available):
        raise MemoryError(
            "current allowed non-Movable free/clean-inactive capacity cannot cover the fixed Host cache plus system reserve"
        )
    if meminfo["MemAvailable"] < required or any(
        item["headroom_bytes"] < required for item in limits
    ):
        raise MemoryError(
            "available RAM or an ancestor cgroup limit cannot cover cache, startup and system reserves"
        )
    return {
        "allowed_numa_nodes": sorted(nodes),
        "non_movable_managed_bytes": managed,
        "non_movable_free_clean_inactive_bytes": available,
        "mem_available_bytes": meminfo["MemAvailable"],
        "startup_reserve_bytes": startup_reserve_bytes,
        "system_reserve_bytes": system_reserve,
        "cgroup_limits": limits,
        "scope": "preflight only; free/clean-inactive estimate is not a reservation or NUMA placement guarantee",
    }


class HostCacheGuard:
    """Read allocator peaks and owned CPU storage outside request timing."""

    def __init__(self, torch, plan):
        self.torch, self.plan = torch, plan
        self.peak_reserved = self.peak_allocated = self.peak_metadata = self.peak_rss = 0
        self.samples = 0
        self.startup = None

    def sample(self, runner, manager=None):
        stats = self.torch.cuda.memory.host_memory_stats()
        if not {"reserved_bytes.peak", "allocated_bytes.peak"} <= stats.keys():
            raise RuntimeError("pinned allocator statistics unavailable")
        self.peak_reserved = max(self.peak_reserved, stats["reserved_bytes.peak"])
        self.peak_allocated = max(self.peak_allocated, stats["allocated_bytes.peak"])
        pool = runner.runner.token_to_kv_pool
        metadata, storage = 0, set()
        for name in ("mem_state", "free_slots"):
            value = getattr(pool, name, None)
            if value is not None and value.device.type == "cpu":
                backing = value.untyped_storage()
                if backing.data_ptr() not in storage:
                    metadata += backing.nbytes()
                    storage.add(backing.data_ptr())
        if manager is not None:
            metadata += sum(
                sys.getsizeof(value)
                for value in (manager._entries, manager._frequency, manager._last_arrival)
            )
            for uid, entry in manager._entries.items():
                metadata += sum(
                    sys.getsizeof(value)
                    for value in (
                        uid,
                        entry,
                        entry.key,
                        entry.key.stable_prefix_sha256,
                        entry.prefix,
                        entry.prefix.token_ids,
                    )
                )
        self.peak_metadata = max(self.peak_metadata, metadata)
        status = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/self/status").read_text().splitlines()
        }
        self.peak_rss = max(self.peak_rss, int(status["VmHWM"].split()[0]) * 1024)
        self.samples += 1
        # Small pinned control allocations share the metadata allowance, never the safety reserve.
        pinned_auxiliary = max(0, self.peak_reserved - self.plan["max_pinned_reserved_bytes"])
        if pinned_auxiliary + 2 * self.peak_metadata > self.plan["metadata_reserve_bytes"]:
            raise MemoryError(
                "actual pinned reservation and CPU metadata exceed the fixed cache plan"
            )
        if self.peak_rss > self.plan["total_cache_budget_bytes"] + 32 * GIB:
            raise MemoryError("process RSS exceeds the cache plus model/input startup envelope")
        if self.startup is None:
            self.startup = dict(stats)

    def metadata(self):
        return {
            "peak_pinned_reserved_bytes": self.peak_reserved,
            "peak_pinned_allocated_bytes": self.peak_allocated,
            "peak_owned_cpu_cache_metadata_bytes": self.peak_metadata,
            "charged_cpu_metadata_bytes": 2 * self.peak_metadata,
            "pinned_above_main_and_staging_bytes": max(
                0, self.peak_reserved - self.plan["max_pinned_reserved_bytes"]
            ),
            "peak_process_rss_bytes": self.peak_rss,
            "process_rss_envelope_bytes": self.plan["total_cache_budget_bytes"] + 32 * GIB,
            "samples": self.samples,
            "startup_allocator_stats": self.startup,
            "scope": "allocator peaks include startup and warmup; CPU metadata is owned buffers/containers plus 2x transient allowance, not full heap attribution; RSS includes weights/input and is not cache-only",
        }


def parse_nodes(value):
    nodes = set()
    for group in value.strip().split(","):
        bounds = [int(part) for part in group.split("-")]
        if len(bounds) not in (1, 2) or min(bounds) < 0 or bounds[0] > bounds[-1]:
            raise ValueError("invalid NUMA node list")
        nodes.update(range(bounds[0], bounds[-1] + 1))
    return nodes


def non_movable_bytes(zoneinfo, allowed_nodes, page_bytes):
    zone = None
    total = 0
    seen = set()
    for line in zoneinfo.splitlines():
        header = re.fullmatch(r"Node\s+(\d+), zone\s+(\w+)", line.strip())
        if header:
            zone = (int(header[1]), header[2])
        elif zone and (match := re.fullmatch(r"managed\s+(\d+)", line.strip())):
            if zone in seen:
                raise ValueError("duplicate managed count in zoneinfo")
            seen.add(zone)
            if zone[0] in allowed_nodes and zone[1] in {"DMA", "DMA32", "Normal"}:
                total += int(match[1]) * page_bytes
    if total <= 0:
        raise ValueError("could not determine managed non-Movable memory")
    return total


def check_pinned_capacity(payload_bytes, managed_bytes, reserve_bytes=10 * GIB):
    if payload_bytes <= 0 or managed_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("invalid pinned-memory capacity parameters")
    rounded = 1 << (payload_bytes - 1).bit_length()
    if rounded + reserve_bytes > managed_bytes:
        raise MemoryError(
            f"Pinned Host KV payload {payload_bytes / GIB:.2f} GiB rounds to "
            f"{rounded / GIB:.0f} GiB in the pinned Torch allocator; "
            f"allowed non-Movable managed memory is only {managed_bytes / GIB:.2f} GiB "
            f"with {reserve_bytes / GIB:g} GiB reserved. ZONE_MOVABLE is not counted. "
            "Refusing GPU initialization; use a verified larger pinned-memory tier."
        )
    return {
        "host_kv_payload_bytes": payload_bytes,
        "torch_pinned_allocation_rounded_bytes": rounded,
        "allowed_non_movable_managed_bytes": managed_bytes,
        "host_reserve_bytes": reserve_bytes,
        "scope": "necessary capacity upper-bound check only; not an availability, cgroup, placement or pinning guarantee",
    }


def inspect_pinned_capacity(payload_bytes):
    status = Path("/proc/self/status").read_text()
    allowed = next(
        line.split(":", 1)[1]
        for line in status.splitlines()
        if line.startswith("Mems_allowed_list:")
    )
    nodes = parse_nodes(allowed)
    managed = non_movable_bytes(
        Path("/proc/zoneinfo").read_text(), nodes, os.sysconf("SC_PAGE_SIZE")
    )
    return check_pinned_capacity(payload_bytes, managed) | {"allowed_numa_nodes": sorted(nodes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--host-users", type=int, required=True)
    args = parser.parse_args(argv)
    if args.host_users < 1:
        parser.error("host users must be positive")
    tokens = args.host_users * 65536 + 1088
    print(json.dumps(inspect_pinned_capacity(tokens * args.layers * 1152), indent=2))


if __name__ == "__main__":
    main()
