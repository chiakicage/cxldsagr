"""Serial cross-request prefix ownership for the local ECHO correctness runner.

This layer is not a serving benchmark. Host LRU uses whole, page-aligned prefixes;
optional HBM retention only drops user replicas, leaving block admission and
replacement to ECHO. Neither retained users nor prefix reuse measures HBM hits.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serving.echo_runner import EchoPrefix, EchoPrefixRunner


@dataclass(frozen=True)
class PrefixKey:
    model_instance_id: str
    stable_prefix_sha256: str


@dataclass(frozen=True)
class EchoCacheResult:
    hidden_states: Any = field(repr=False)
    prefix_key: PrefixKey
    prefix_reused: bool
    evicted_user_ids: tuple[int, ...]
    retained_tokens: int
    observed_frequency: float


@dataclass
class _Entry:
    key: PrefixKey
    prefix: EchoPrefix
    hbm_drop_applied: bool = False


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class EchoCacheManager:
    """Own all prefixes of an otherwise idle ``EchoPrefixRunner`` until close.

    Do not interleave direct runner allocations or another manager. Capacity is
    in allocator token slots, not total host/HBM bytes: index and other memory
    remain the backend's responsibility. ``model_instance_id`` must identify
    weights, precision, positional encoding and attention/cache semantics.

    ``hbm_retained_users=None`` leaves ECHO's native policy untouched. An integer
    limits which users may keep replicas after requests, ranked by observed
    decayed frequency and recency. It is not a byte budget, pin, or promotion:
    ECHO may still evict any replica internally. Resident mode rejects it.

    Healthy close releases owned prefixes without closing the runner. After a
    backend failure no slots are recycled here; the caller must close/discard
    the failed runner. This serial class is not thread-safe.
    """

    def __init__(
        self,
        runner: EchoPrefixRunner,
        *,
        host_capacity_tokens: int,
        heat_decay_interval: int = 1024,
        hbm_retained_users: int | None = None,
    ):
        runner._ensure_alive()
        if runner._prefixes:
            raise ValueError("the manager requires a runner with no existing prefixes")
        backend = runner.runner
        self.page_size = _integer("page_size", backend.page_size, 1)
        capacity = _integer("host_capacity_tokens", host_capacity_tokens, 1)
        actual = _integer("allocator.size", backend.token_to_kv_pool_allocator.size, 1)
        if capacity % self.page_size or capacity > actual:
            raise ValueError("host_capacity_tokens must be page aligned and <= allocator.size")
        self.heat_decay_interval = _integer("heat_decay_interval", heat_decay_interval, 1)
        if hbm_retained_users is not None:
            _integer("hbm_retained_users", hbm_retained_users)
            if not hasattr(backend.token_to_kv_pool, "free_req_device_pool"):
                raise ValueError("HBM retention requires an authoritative host copy")
        self.runner = runner
        self.host_capacity_tokens = capacity
        self.hbm_retained_users = hbm_retained_users
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._frequency: dict[int, float] = {}
        self._last_arrival: dict[int, int] = {}
        self._arrivals = 0
        self._pending: EchoPrefix | None = None
        self._failed = False
        self._closed = False

    @property
    def retained_tokens(self) -> int:
        """Logical token count of published prefixes, not measured pool usage."""
        return sum(len(entry.prefix.token_ids) for entry in self._entries.values())

    @property
    def cached_user_ids(self) -> tuple[int, ...]:
        """Published users in host LRU order, oldest first."""
        return tuple(self._entries)

    @property
    def observed_frequency(self) -> dict[int, float]:
        return dict(self._frequency)

    def _validate(self, request):
        if not isinstance(request, Mapping):
            raise TypeError("request must be a mapping")
        if request.get("model") != "deepseek_v32":
            raise ValueError("the ECHO manager requires model=deepseek_v32")
        model_id = self.runner.model_instance_id
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("runner.model_instance_id must be a nonempty string")
        uid = _integer("user_id", request.get("user_id"))
        ids = request.get("input_ids")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise TypeError("input_ids must be a sequence of token IDs")
        ids = tuple(ids)
        backend = self.runner.runner
        vocab_size = backend.model_config.hf_config.vocab_size
        if not ids or any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError("input_ids must contain integers in vocabulary")
        prefix_size = _integer("stable_prefix_tokens", request.get("stable_prefix_tokens"), 1)
        if prefix_size >= len(ids) or prefix_size % self.page_size:
            raise ValueError("stable prefix must be page aligned and leave a nonempty candidate")
        if len(ids) > backend.model_config.context_len:
            raise ValueError("request exceeds the configured model context")
        device_pool = getattr(backend.token_to_kv_pool, "device_pool", None)
        if device_pool is not None and len(ids) > device_pool.size:
            raise MemoryError(
                "the initial ECHO runner requires the active context to fit device KV"
            )
        suffix_size = len(ids) - prefix_size
        for name, expected in (
            ("total_input_tokens", len(ids)),
            ("candidate_suffix_tokens", suffix_size),
        ):
            if name in request and _integer(name, request[name]) != expected:
                raise ValueError(f"{name} disagrees with input_ids and the stable boundary")
        if "instruction_tokens" not in request and any(
            name in request for name in ("user_tokens", "item_tokens", "history_token_span")
        ):
            raise ValueError("GR section metadata requires instruction_tokens")
        if "instruction_tokens" in request:
            instruction = _integer("instruction_tokens", request["instruction_tokens"])
            if instruction > prefix_size:
                raise ValueError("instruction_tokens exceeds the stable prefix")
            for name, expected in (
                ("user_tokens", prefix_size - instruction),
                ("item_tokens", suffix_size + instruction),
            ):
                if name in request and _integer(name, request[name]) != expected:
                    raise ValueError(f"{name} disagrees with the GR token boundaries")
            if "history_token_span" in request:
                self._check_span(request["history_token_span"], (instruction, prefix_size))
        if "candidate_token_span" in request:
            self._check_span(request["candidate_token_span"], (prefix_size, len(ids)))
        if "attention_mask" in request:
            mask = request["attention_mask"]
            if (
                not isinstance(mask, Sequence)
                or isinstance(mask, (str, bytes))
                or len(mask) != len(ids)
                or any(type(value) is not int or value != 1 for value in mask)
            ):
                raise ValueError("attention_mask must contain one integer 1 per input token")
        reserve = ((suffix_size + self.page_size - 1) // self.page_size + 1) * self.page_size
        if prefix_size + reserve > self.host_capacity_tokens:
            raise MemoryError("prefix, rounded suffix and guard page exceed host capacity")
        prefix, candidate = ids[:prefix_size], ids[prefix_size:]
        digest = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
        return uid, PrefixKey(model_id, digest), prefix, candidate, reserve

    @staticmethod
    def _check_span(value, expected):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(type(x) is not int for x in value)
            or tuple(value) != expected
        ):
            raise ValueError("token span disagrees with the GR token boundaries")

    def _backend_call(self, method, *args):
        try:
            return method(*args)
        except BaseException:
            # A backend failure may leave async work or partly mutated mappings.
            self.runner._failed = True
            raise

    def _remove(self, uid: int):
        self._backend_call(self.runner.release, self._entries[uid].prefix)
        del self._entries[uid]

    def _observe(self, uid: int):
        self._arrivals += 1
        if self._arrivals % self.heat_decay_interval == 0:
            self._frequency = {user: value * 0.5 for user, value in self._frequency.items()}
        self._frequency[uid] = self._frequency.get(uid, 0.0) + 1.0
        self._last_arrival[uid] = self._arrivals

    def _apply_hbm_retention(self, uid: int, entry: _Entry):
        if self.hbm_retained_users is None:
            return
        entries = dict(self._entries)
        entries[uid] = entry
        ranked = sorted(
            entries,
            key=lambda user: (-self._frequency[user], -self._last_arrival[user], user),
        )
        keep = set(ranked[: self.hbm_retained_users])
        for user, cached in entries.items():
            if user not in keep and not cached.hbm_drop_applied:
                self._backend_call(self.runner.evict_hbm, cached.prefix)
                cached.hbm_drop_applied = True

    def execute(self, request: Mapping) -> EchoCacheResult:
        """Execute one arrived request; never inspect generator/oracle heat fields."""
        if self._closed or self._failed:
            raise RuntimeError("ECHO manager is closed or failed")
        try:
            self.runner._ensure_alive()
            uid, key, prefix_ids, candidate_ids, reserve = self._validate(request)
            self._observe(uid)
            evicted = []
            entry = self._entries.get(uid)
            reused = entry is not None and entry.key == key
            if entry is not None and not reused:
                self._remove(uid)
                evicted.append(uid)
                entry = None
            additional = 0 if reused else len(prefix_ids)
            while self.retained_tokens + additional + reserve > self.host_capacity_tokens:
                victim = next(user for user in self._entries if user != uid)
                self._remove(victim)
                evicted.append(victim)
            if entry is None:
                self._pending = self._backend_call(self.runner.prefill, prefix_ids)
                entry = _Entry(key, self._pending)
            hidden = self._backend_call(self.runner.extend, entry.prefix, candidate_ids)
            entry.hbm_drop_applied = False
            self._apply_hbm_retention(uid, entry)
            # A new prefix becomes visible only after its whole first request succeeds.
            self._entries[uid] = entry
            self._entries.move_to_end(uid)
            self._pending = None
            return EchoCacheResult(
                hidden, key, reused, tuple(evicted), self.retained_tokens, self._frequency[uid]
            )
        except BaseException:
            self._failed = True
            raise

    def close(self):
        """Release owned healthy prefixes; leave the runner itself open."""
        if self._closed:
            return
        try:
            if not self.runner._failed and not self.runner._closed:
                if self._pending is not None:
                    self._backend_call(self.runner.release, self._pending)
                for uid in tuple(self._entries):
                    self._remove(uid)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._entries.clear()
            self._pending = None
            self._closed = True
