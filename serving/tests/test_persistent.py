import pytest
import torch

from serving.persistent import PersistentGRRunner


class Backend:
    scheme = "hbm"
    device = torch.device("cpu")
    max_seq_len = 100

    def __init__(self):
        self.built = self.released = 0
        self.fail = False

    def estimate_session_bytes(self, capacity, prefix_tokens):
        return {"hbm": capacity * 8, "dram": 0}

    def create_session(self, capacity):
        return {"tokens": [], "capacity": capacity}

    def prefill(self, session, ids):
        self.built += 1
        session["tokens"].extend(ids.tolist())

    def extend(self, session, ids):
        if self.fail:
            raise RuntimeError("model failure")
        outputs = []
        for token in ids.tolist():
            session["tokens"].append(token)
            outputs.append([sum(session["tokens"])])
        return torch.tensor(outputs)

    def truncate(self, session, prefix):
        del session["tokens"][prefix:]

    def session_bytes(self, session):
        return {"hbm": session["capacity"] * 8, "dram": 0}

    def release_session(self, session):
        session.clear()
        self.released += 1

    def synchronize(self):
        pass


def request(user=0, prefix=(1, 2), candidate=(3, 4)):
    return {"user_id": user, "input_ids": list(prefix + candidate), "stable_prefix_tokens": 2}


def test_revisits_and_suffix_rollback():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        first = runner.execute(request())
        second = runner.execute(request(candidate=(5, 6)))
        assert first.hidden.tolist() == [[6], [10]]
        assert second.hidden.tolist() == [[8], [14]]
        assert not first.metrics["is_revisit"]
        assert second.metrics["is_revisit"]
        assert second.metrics["prefix_cache_hit"]
        assert second.metrics["prefix_hit_tier"] == "hbm"
        assert backend.built == 1
        assert second.metrics["latency_ms"] >= second.metrics["extend_ms"] >= 0
    assert backend.released == 1


def test_evicted_revisit_counts_as_revisit_but_miss():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=32, dram_budget_bytes=64) as runner:
        runner.execute(request(0))
        runner.execute(request(1))
        result = runner.execute(request(0))
        assert result.metrics["visit_index"] == 1
        assert result.metrics["is_revisit"]
        assert not result.metrics["prefix_cache_hit"]
        assert result.metrics["evicted_users"] == [1]
        assert backend.built == 3


def test_changed_history_is_not_reused():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        runner.execute(request())
        result = runner.execute(request(prefix=(7, 8)))
        assert result.hidden.tolist() == [[18], [22]]
        assert not result.metrics["prefix_cache_hit"]


def test_failed_candidate_discards_user_and_allows_recovery():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        runner.execute(request())
        backend.fail = True
        with pytest.raises(RuntimeError, match="model failure"):
            runner.execute(request())
        assert len(runner.pool) == 0
        backend.fail = False
        result = runner.execute(request())
        assert not result.metrics["prefix_cache_hit"]
        assert result.metrics["visit_index"] == 1


@pytest.mark.parametrize("ids", [[], [1.0, 2, 3], [1, -1, 3], [True, 2, 3]])
def test_bad_request_never_allocates(ids):
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=0) as runner:
        with pytest.raises(ValueError):
            runner.execute({"user_id": 0, "input_ids": ids, "stable_prefix_tokens": 2})
        assert len(runner.pool) == 0
