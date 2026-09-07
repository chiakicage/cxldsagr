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


if __name__ == "__main__":
    unittest.main()
