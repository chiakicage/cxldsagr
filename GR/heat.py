"""Load user identities and heat independently of request text."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ijson


@dataclass
class HeatPopulation:
    weights: dict[int, float]
    metadata: dict

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        industrial_field: str | None = None,
        num_users: int = 1000,
        seed: int = 42,
    ) -> HeatPopulation:
        if num_users < 0:
            raise ValueError("num_users must be nonnegative (0 means all users)")
        rng = random.Random(f"{seed}:heat-population")
        if industrial_field is None:
            counts: Counter = Counter()
            with path.open("rb") as source:
                for _, pairs in ijson.kvitems(source, ""):
                    for uid, count in pairs:
                        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                            raise ValueError(f"{path}: invalid interaction count {count}")
                        counts[int(uid)] += count
            source_users = len(counts)
            users = sorted(counts)
            if num_users and num_users < source_users:
                users = rng.sample(users, num_users)
            rows = [(uid, float(counts[uid])) for uid in users]
        else:
            if industrial_field not in ("pv_share", "pv_int", "pv_scaled_1_100"):
                raise ValueError("unsupported industrial heat field")
            rows = []
            source_users = 0
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if not {"user_id", industrial_field}.issubset(reader.fieldnames or []):
                    raise ValueError(f"{path}: missing user_id or {industrial_field}")
                for line, row in enumerate(reader, 2):
                    try:
                        uid, weight = int(row["user_id"]), float(row[industrial_field])
                        if uid < 0 or not math.isfinite(weight) or weight <= 0:
                            raise ValueError("expected nonnegative ID and finite positive weight")
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}:{line}: {exc}") from exc
                    source_users += 1
                    if not num_users or len(rows) < num_users:
                        rows.append((uid, weight))
                    else:
                        index = rng.randrange(source_users)
                        if index < num_users:
                            rows[index] = (uid, weight)
        if not rows:
            raise ValueError("heat source contains no users")
        if num_users > source_users:
            raise ValueError(f"requested {num_users} users, source has only {source_users}")
        weights = dict(sorted(rows))
        if len(weights) != len(rows):
            raise ValueError("selected heat profiles contain duplicate user IDs")
        return cls(
            weights,
            {
                "path": str(path.resolve()),
                "field": industrial_field or "interaction_count",
                "source_users": source_users,
                "selected_users": len(weights),
                "seed": seed,
                "selection": "all"
                if len(weights) == source_users
                else "uniform_without_replacement",
                "selected_weight_sum": math.fsum(weights.values()),
                "user_identity": "original heat-source user ID; no text-user mapping",
            },
        )
