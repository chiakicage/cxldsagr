import random
import unittest
from collections import Counter
from itertools import pairwise

from GR.scheduling import ScheduleConfig, _schedule, index_letter


class SchedulingTests(unittest.TestCase):
    def test_weighted_frequency_poisson_rate_and_rng_isolation(self):
        state = random.getstate()
        cfg = ScheduleConfig(qps=100)
        rows = list(_schedule([1, 2], {1: 1, 2: 9}, 10000, cfg))
        count = Counter(uid for uid, _ in rows)
        self.assertTrue(0.88 < count[2] / len(rows) < 0.92)
        self.assertTrue(95 < rows[-1][1] < 105)
        self.assertTrue(all(a[1] <= b[1] for a, b in pairwise(rows)))
        self.assertEqual(state, random.getstate())

    def test_timeslot_capacity_and_cooldown(self):
        for users in ([1], list(range(30))):
            cfg = ScheduleConfig(arrival="timeslot", qps=3)
            rows = list(_schedule(users, dict.fromkeys(users, 1), 200, cfg))
            self.assertEqual(len(rows), 200)
            self.assertLessEqual(max(Counter(t for _, t in rows).values()), 3)
            last = {}
            for uid, timestamp in rows:
                if uid in last:
                    self.assertGreaterEqual(timestamp - last[uid], 5)
                last[uid] = timestamp
            self.assertEqual(sorted(t for _, t in rows), [t for _, t in rows])

    def test_config_and_constant_sequential_schedule(self):
        for kwargs in (
            {"qps": 0},
            {"start_timestamp": float("nan")},
            {"arrival": "timeslot", "sampling": "sequential"},
            {"max_revisits": -1},
            {"max_revisits": True},
            {"max_revisits": 1.5},
            {"max_revisits": "8"},
        ):
            with self.assertRaises(ValueError):
                ScheduleConfig(**kwargs)
        self.assertEqual(
            list(
                _schedule(
                    [1, 2], {}, 4, ScheduleConfig(arrival="constant", sampling="sequential", qps=2)
                )
            ),
            [(1, 0), (2, 0.5), (1, 1), (2, 1.5)],
        )
        self.assertEqual(
            [index_letter(i) for i in (0, 25, 26, 51, 52)], ["A", "Z", "AA", "AZ", "BA"]
        )

    def test_revisit_cap_all_modes_exhaustion_and_request_limit(self):
        for arrival in ("constant", "poisson", "timeslot"):
            for sampling in ("weighted", "uniform", "sequential"):
                if arrival == "timeslot" and sampling == "sequential":
                    continue
                for cap in (0, 2, 8):
                    with self.subTest(arrival=arrival, sampling=sampling, cap=cap):
                        cfg = ScheduleConfig(
                            arrival=arrival, sampling=sampling, max_revisits=cap, qps=2
                        )
                        rows = list(_schedule([1, 2, 3], {1: 1, 2: 3, 3: 7}, 100, cfg))
                        self.assertEqual(
                            Counter(uid for uid, _ in rows), dict.fromkeys([1, 2, 3], cap + 1)
                        )
                        self.assertEqual(
                            list(_schedule([1, 2, 3], {1: 1, 2: 3, 3: 7}, 2, cfg)), rows[:2]
                        )
                        self.assertEqual(sorted(t for _, t in rows), [t for _, t in rows])
                        if arrival == "timeslot":
                            self.assertLessEqual(max(Counter(t for _, t in rows).values()), 2)
                            last = {}
                            for uid, timestamp in rows:
                                if uid in last:
                                    self.assertGreaterEqual(timestamp - last[uid], 5)
                                last[uid] = timestamp

    def test_capped_weighted_draw_matches_eligible_population_oracle(self):
        weights = {1: 1, 2: 3, 3: 7}
        for seed in range(20):
            rng = random.Random(seed)
            expected, visits = [], Counter()
            for _ in range(7):
                eligible = [uid for uid in weights if visits[uid] < 3]
                uid = rng.choices(eligible, weights=[weights[uid] for uid in eligible])[0]
                visits[uid] += 1
                expected.append(uid)
            rows = list(
                _schedule(list(weights), weights, 7, ScheduleConfig(seed=seed, max_revisits=2))
            )
            self.assertEqual([uid for uid, _ in rows], expected)
        rows = list(_schedule([1, 2], {1: 1, 2: 10000}, 2, ScheduleConfig(max_revisits=8)))
        self.assertEqual([uid for uid, _ in rows], [2, 2])  # No mandatory first-visit pass.

    def test_unlimited_schedule_preserves_existing_trace(self):
        expected_users = {
            "weighted": [3, 1, 2, 2, 3, 3, 3, 1, 3, 1, 2, 3],
            "uniform": [2, 1, 1, 1, 3, 3, 3, 1, 2, 1, 1, 2],
            "sequential": [1, 2, 3] * 4,
        }
        expected_timeslot = {
            "weighted": [
                (3, 0),
                (1, 0),
                (2, 1),
                (2, 16),
                (2, 31),
                (3, 45),
                (2, 46),
                (2, 61),
                (1, 65),
                (2, 76),
                (3, 90),
                (2, 91),
            ],
            "uniform": [
                (2, 0),
                (1, 0),
                (3, 1),
                (2, 15),
                (2, 30),
                (2, 45),
                (3, 46),
                (2, 60),
                (1, 65),
                (2, 75),
                (2, 90),
                (3, 91),
            ],
        }
        for arrival in ("constant", "poisson", "timeslot"):
            for sampling in ("weighted", "uniform", "sequential"):
                if arrival == "timeslot" and sampling == "sequential":
                    continue
                args = {"arrival": arrival, "sampling": sampling, "qps": 2}
                rows = list(_schedule([1, 2, 3], {1: 1, 2: 3, 3: 7}, 12, ScheduleConfig(**args)))
                self.assertEqual(
                    rows,
                    list(
                        _schedule(
                            [1, 2, 3],
                            {1: 1, 2: 3, 3: 7},
                            12,
                            ScheduleConfig(**args, max_revisits=20),
                        )
                    ),
                )
                if arrival == "timeslot":
                    self.assertEqual(rows, expected_timeslot[sampling])
                else:
                    self.assertEqual([uid for uid, _ in rows], expected_users[sampling])


if __name__ == "__main__":
    unittest.main()
