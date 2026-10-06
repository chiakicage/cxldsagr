from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_mfu.src import gap_profile


def test_graph_setup_restores_cache_created_during_inference(tmp_path, monkeypatch):
    model = SimpleNamespace(blocks=[], snapshot_prefix=lambda: object())

    @torch.inference_mode()
    def annotate(*args, **kwargs):
        model.cache = torch.ones(1)
        return torch.ones(1), None, [], 0

    def restore(*args):
        # This mirrors the existing prefix-restore in-place operation: tensors
        # created by the annotated prefill may only be updated in inference mode.
        model.cache.zero_()

    class SetupReached(Exception):
        pass

    def setup(*args):
        assert model.cache.item() == 0
        raise SetupReached

    monkeypatch.setattr(gap_profile, "METHODS", ("hbm",))
    monkeypatch.setattr(gap_profile, "measurement_sources", dict)
    monkeypatch.setattr(gap_profile, "annotate", annotate)
    monkeypatch.setattr(gap_profile.torch, "load", lambda *args, **kwargs: torch.ones(1))
    monkeypatch.setattr(gap_profile.common, "restore_extend_prefix", restore)
    monkeypatch.setattr(gap_profile.common, "profile_extend_graph", setup)
    args = SimpleNamespace(nsys=True, output=tmp_path, compute_graphs=False, prefix=1)
    result = {"correctness": {}}
    receipt = {
        "checks": {"comparisons": {}},
        "artifact_paths": {"hbm_control.pt": "unused", "hbm_prefix_logits.pt": "unused"},
    }
    with pytest.raises(SetupReached):
        gap_profile.run_profile(model, torch.zeros(2), args, result, receipt, trace_warmups=0)
