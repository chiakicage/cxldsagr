"""NOSA allocation bounds for the pinned PyTorch native CUDA allocator.

The bounds cover active device blocks, including rounding and an unsplit
large-pool tail, and owned pinned-host blocks rounded to a power of two. They
exclude inactive cached blocks, fragmentation, direct library allocations,
driver page overhead and CPU malloc overhead. Count independent storages once;
old/new scratch and asynchronous sources remain live through completed free.

Validated source: PyTorch 7269437d655783a26cba32aa88195b741ff496aa,
c10/cuda/CUDACachingAllocator.cpp, round_size / get_pool / should_split, and
ATen/core/CachingHostAllocator.h, allocate / get_free_block.
The numerical helpers never inspect CUDA state. Call validate_allocator before
planning/allocation and around execution, and freeze its configuration_key for
the lifetime of an allocated plan.

Pool detection rejects GC-visible Python MemPool wrappers and foreign private
segments, including inactive ones. Explicitly registered graph pools are
separate from the cache allocation bound. PyTorch exposes no current-pool/no_split
getter, and Python GC inspection excludes its frozen generation. Empty frozen
Python pools and empty pools routed directly through C++ or private bindings
cannot be observed; neither can a previously latched uncached-environment flag
subsequently removed. A pinned reserve segment configured earlier and hidden by
a later settings update is likewise not observable. Host allocator replacement,
external pool routing and allocator/environment mutation are unsupported. These
checks admit the model's default-pool path; they do not prove absence of every
external pool. Checks are not atomic against another thread changing allocator
state.
"""

import gc
import hashlib
import json
import os
import sys

import torch

from models.nosa import _pool_referrers
from models.nosa._allocator_snapshot import snapshot as _allocator_snapshot

CPU_POLICY = "nosa_cpu_payload_v1"
CUDA_POLICY = "nosa_torch_native_512b_1mib_tail_pinned_power2_v2"
POLICY_REVISION = CUDA_POLICY
SUPPORTED_TORCH_REVISION = "7269437d655783a26cba32aa88195b741ff496aa"
_QUANTUM = 512
_SMALL_MAX = 1 << 20
_NO_CACHE_VARIABLES = ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NO_HIP_MEMORY_CACHING")
_pool_scan_state = None
_ORIGINAL_GC_GET_OBJECTS = gc.get_objects
_ORIGINAL_GC_GET_REFERRERS = gc.get_referrers
_resolve_pool_referrers = _pool_referrers.identity_resolver


def allocator_policy(device):
    """Return the policy identifier without CUDA initialization or allocation."""
    kind = torch.device(device).type
    if kind == "cpu":
        return CPU_POLICY
    if kind == "cuda":
        return CUDA_POLICY
    raise NotImplementedError("NOSA allocation budgeting supports only CPU and CUDA")


def allocation_bytes(size, device):
    """Bound one storage allocation; CPU role tests retain their payload ledger.

    CUDA requires the separately validated allocator contract. Small-pool blocks
    split at 512 B; large-pool blocks split only when the remainder exceeds
    1 MiB. The 1 MiB tail is therefore an inclusive per-allocation bound.
    """
    if type(size) is not int:
        raise TypeError("Allocation size must be an integer byte count")
    if size < 0:
        raise ValueError("Allocation size must be nonnegative")
    policy = allocator_policy(device)
    if not size or policy == CPU_POLICY:
        return size
    rounded = (size + _QUANTUM - 1) // _QUANTUM * _QUANTUM
    return rounded + (_SMALL_MAX if rounded > _SMALL_MAX else 0)


def pinned_allocation_bytes(size, device):
    """Size one host allocation for the model's CPU or CUDA execution role.

    CUDA execution uses PyTorch's CachingHostAllocator: allocate rounds to the
    next power of two, and get_free_block reuses only the same bin. This is the
    complete owned block, independent of the tensor's logical storage size.
    CPU reference execution uses ordinary host tensors and retains payload
    accounting. A separately validated CUDA contract excludes pinned reserve
    segments; those reserve extra process memory beyond individual block bins.
    """
    if type(size) is not int:
        raise TypeError("Allocation size must be an integer byte count")
    if size < 0:
        raise ValueError("Allocation size must be nonnegative")
    policy = allocator_policy(device)
    if not size or policy == CPU_POLICY:
        return size
    return 1 << (size - 1).bit_length()


def _python_pool_types(pool_type):
    pending = [pool_type]
    kinds = {}
    while pending:
        kind = pending.pop()
        if id(kind) not in kinds:
            kinds[id(kind)] = kind
            # Bypass arbitrary metaclass/instance __subclasses__ overrides.
            pending.extend(type.__subclasses__(kind))
    return kinds


def _has_python_pool_referrers(pool_type):
    # CPython 3.12 subtype_traverse visits the exact heap instance type, or
    # delegates that visit to its heap base. get_referrers scans every ordinary
    # generation in C, with the same frozen-object boundary as get_objects.
    # Extension subclasses must obey that heap-type traversal contract too.
    while True:
        kinds = _python_pool_types(pool_type)
        referrers = _resolve_pool_referrers(gc.get_referrers)(*kinds.values())
        found = any(issubclass(type(value), pool_type) for value in referrers)
        del referrers
        if found:
            return True
        # GC/audit callbacks can create a subclass after the first inventory.
        if kinds.keys() == _python_pool_types(pool_type).keys():
            return False


def _reject_python_pools():
    # Public MemPool is a GC-tracked Python subclass of the untracked pybind
    # base. Checking object types avoids invoking arbitrary objects' __class__.
    pool_type = torch.cuda.MemPool
    if not pool_type.__flags__ & (1 << 14):  # CPython Py_TPFLAGS_HAVE_GC
        raise RuntimeError("Cannot inspect Python MemPool objects in this PyTorch build")
    global _pool_scan_state
    identity = (pool_type, gc.get_objects, gc.get_referrers, gc.get_stats, gc.get_freeze_count)

    def stamp():
        return tuple(item["collections"] for item in gc.get_stats()), gc.get_freeze_count()

    # A successfully checked older generation contains no MemPool. New tracked
    # wrappers start in generation zero. A generation-zero collection can move
    # them into generation one; a later-generation collection or freeze/unfreeze
    # requires a complete scan. This preserves fresh empty-pool detection while
    # avoiding an all-object Python walk at every execution boundary.
    while True:
        before = stamp()
        previous = _pool_scan_state
        if (
            previous is None
            or previous[:-2] != identity
            or previous[-1] != before[1]
            or previous[-2][1:] != before[0][1:]
        ):
            generations = None
        elif previous[-2][0] != before[0][0]:
            generations = (0, 1)
        else:
            generations = (0,)
        if (
            generations is None
            and sys.implementation.name == "cpython"
            and sys.version_info[:2] == (3, 12)
            and gc.get_objects is _ORIGINAL_GC_GET_OBJECTS
            and gc.get_referrers is _ORIGINAL_GC_GET_REFERRERS
        ):
            found = _has_python_pool_referrers(pool_type)
        else:
            if generations is None:
                objects = gc.get_objects()
            else:
                try:
                    objects = gc.get_objects(generations[0])
                    for generation in generations[1:]:
                        objects.extend(gc.get_objects(generation))
                except TypeError:
                    # Keep the strict full-object fallback for inspection test
                    # doubles and implementations without generation selection.
                    objects = gc.get_objects()
            # Most tracked objects share a few hundred concrete types. Integer
            # identities avoid arbitrary metaclass hashing/equality calls.
            kinds = tuple(map(type, objects))
            unique_kinds = dict(zip(map(id, kinds), kinds, strict=True)).values()
            found = any(issubclass(kind, pool_type) for kind in unique_kinds)
            del kinds, unique_kinds, objects
        if found:
            _pool_scan_state = None
            raise NotImplementedError(
                "NOSA allocator budgeting found live Python MemPool objects, "
                "including empty, inactive, no_split and custom-allocator pools"
            )
        after = stamp()
        if after == before:
            _pool_scan_state = (*identity, *after)
            return
        # Collection during inspection can promote an unseen object. Repeat
        # conservatively instead of caching an incomplete observation.
        _pool_scan_state = None


def _validate_settings(settings):
    if not isinstance(settings, dict):
        raise TypeError("CUDA allocator snapshot lacks effective settings")
    divisions = settings.get("roundup_power2_divisions")
    if (
        type(settings.get("max_split_size")) is not int
        or settings["max_split_size"] != -1
        or not isinstance(divisions, dict)
        or set(divisions) != {str(1 << index) for index in range(16)}
        or any(type(value) is not int or value != 0 for value in divisions.values())
        or settings.get("expandable_segments") is not False
        or settings.get("graph_capture_record_stream_reuse") is not False
        or not isinstance(settings.get("PYTORCH_CUDA_ALLOC_CONF"), str)
    ):
        raise NotImplementedError(
            "NOSA requires native unlimited splitting, 512 B rounding, "
            "non-expandable segments and no graph-capture reuse"
        )
    # This setting is not exported as a separate snapshot field. Its parser
    # accepts only positive sizes; any occurrence in the effective settings
    # selects an unsupported globally reserved host segment. Runtime settings
    # mutation can hide a previously latched segment and is outside the contract.
    if "pinned_reserve_segment_size_mb" in settings["PYTORCH_CUDA_ALLOC_CONF"]:
        raise NotImplementedError("NOSA pinned allocation bounds exclude pinned reserve segments")


def validate_allocator(device, *, allowed_graph_pools=()):
    """Inspect effective settings and observable pool state without tensors.

    CPU returns immediately without querying CUDA. CUDA inspection does not
    initialize an unused CUDA runtime. The first check walks all GC objects;
    later checks inspect young generations unless collection/freeze activity
    requires a full scan. Allocator blocks are checked every time and the
    snapshot omits trace history. A private adapter can omit unused block Python
    dictionaries; unavailable builds retain the official snapshot with explicit
    adapter evidence. Every call still reads fresh allocator state. Guard cost
    remains inside its caller's window.
    GC-visible Python MemPools are conservatively rejected regardless of thread
    or device. Empty frozen Python pools and direct C++/private-binding pool
    routing are outside this contract. Normal GC freezing remains permitted.
    """
    global _resolve_pool_referrers
    device = torch.device(device)
    policy = allocator_policy(device)
    if policy == CPU_POLICY:
        return {"policy": policy, "configuration_key": policy}
    allowed = set()
    for pool in allowed_graph_pools:
        if (
            type(pool) is not tuple
            or len(pool) != 2
            or any(type(value) is not int or value < 0 for value in pool)
            or pool == (0, 0)
        ):
            raise ValueError("Allowed graph pools must be nondefault integer pool ID pairs")
        allowed.add(pool)
    if torch.version.git_version != SUPPORTED_TORCH_REVISION:
        raise NotImplementedError("Revalidate NOSA allocator bounds for this PyTorch revision")
    try:
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise NotImplementedError("NOSA allocation bounds require the native CUDA allocator")
        if torch._C._cuda_cudaCachingAllocator_is_enabled() is not True:
            raise NotImplementedError("NOSA allocation bounds require enabled CUDA caching")
        no_cache = {name: os.environ.get(name) for name in _NO_CACHE_VARIABLES}
        if any(value not in (None, "0") for value in no_cache.values()):
            raise NotImplementedError("Uncached CUDA allocation is unsupported by NOSA budgeting")
        resolver = _pool_referrers.prepare(_has_python_pool_referrers, _python_pool_types)
        if resolver is not _pool_referrers.identity_resolver:
            _resolve_pool_referrers = resolver
        _reject_python_pools()
        if torch.cuda.is_initialized():
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError(
                        "NOSA allocation budgeting does not support CUDA Graph"
                    )
        previous = torch._C._accelerator_getAllocatorSettings()
        snapshot, snapshot_adapter = _allocator_snapshot()
        current = torch._C._accelerator_getAllocatorSettings()
    except AttributeError as exc:
        raise RuntimeError("Required PyTorch allocator inspection API is unavailable") from exc
    if not isinstance(snapshot, dict):
        raise TypeError("CUDA allocator snapshot must provide a settings and segments mapping")
    settings = snapshot.get("allocator_settings")
    _validate_settings(settings)
    if previous != current or settings["PYTORCH_CUDA_ALLOC_CONF"] != current:
        raise RuntimeError("CUDA allocator settings changed during validation")
    segments = snapshot.get("segments")
    if not isinstance(segments, list):
        raise TypeError("CUDA allocator snapshot lacks segment ownership evidence")
    permitted_pools = {(0, 0), *allowed}
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("segment_pool_id"), (tuple, list))
            or tuple(segment["segment_pool_id"]) not in permitted_pools
            or segment.get("is_expandable") is not False
        ):
            raise NotImplementedError("Private or expandable CUDA memory segments are unsupported")
    evidence = {
        "policy": policy,
        "torch_version": str(torch.__version__),
        "torch_git_revision": torch.version.git_version,
        "effective_settings": settings,
        "no_cache_environment": no_cache,
    }
    evidence["configuration_key"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # These separately budgeted graph allocations do not alter native cache
    # rounding. Registering an owned pool must not invalidate the cache plan's
    # allocator geometry key; foreign pools still fail above on every call.
    evidence["allowed_graph_pool_ids"] = sorted(allowed)
    evidence["snapshot_adapter"] = snapshot_adapter
    return evidence
