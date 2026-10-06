"""Publication must re-read every measured consumer against accepted HBM outputs."""

from copy import deepcopy

import pytest
import torch

from experiments.cache_manager_performance.src import report


@pytest.fixture
def evidence_run(tmp_path):
    profile, checked = tmp_path / "profile", tmp_path / "check"
    profile.mkdir()
    checked.mkdir()
    check_path = checked / "result.json"
    check_path.write_text("{}\n")
    tensors = {
        "indices": [torch.tensor([[1, 0, -1], [2, 0, 1]], dtype=torch.int32)],
        "attention": [torch.zeros(2, 64, 512, dtype=torch.bfloat16)],
    }

    def save(directory, scheme, phase):
        path = directory / f"{scheme}_{phase}_0.pt"
        torch.save(tensors, path)
        return {
            "scheme": scheme,
            "phase": phase,
            "sample": 0,
            "evidence": {"path": path.name, "sha256": report.file_sha256(path)},
        }

    check = {"samples": [save(checked, "hbm", phase) for phase in report.MEASURED_PHASES]}
    result = {
        "repeats": 1,
        "samples": [
            save(profile, scheme, phase)
            for scheme in report.MEASURED_SCHEMES
            for phase in report.MEASURED_PHASES
        ],
    }
    receipt = {
        "artifact_paths": {
            row["evidence"]["path"]: str(checked / row["evidence"]["path"])
            for row in check["samples"]
        }
    }
    config = {"layers": 1, "append": 2, "history": 1}
    return profile, result, check_path, check, receipt, config


def _replace_profile_tensors(fixture, update):
    profile, result, *_ = fixture
    record = result["samples"][0]["evidence"]
    path = profile / record["path"]
    tensors = torch.load(path, weights_only=True, map_location="cpu")
    update(tensors)
    torch.save(tensors, path)
    record["sha256"] = report.file_sha256(path)


def test_saved_consumers_are_compared_exactly_and_counted(evidence_run):
    assert report._verify_saved_consumers(*evidence_run) == {
        "resident_reference_files": 2,
        "profile_evidence_files": 4,
        "profile_layer_comparisons": 4,
        "exact_selection_match": True,
        "exact_attention_match": True,
        "all_attention_finite": True,
    }


def test_corrupt_profile_evidence_is_rejected_before_loading(evidence_run):
    profile, result, *_ = evidence_run
    (profile / result["samples"][0]["evidence"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        report._verify_saved_consumers(*evidence_run)


def test_missing_profile_evidence_is_rejected(evidence_run):
    profile, result, *_ = evidence_run
    (profile / result["samples"][0]["evidence"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        report._verify_saved_consumers(*evidence_run)


@pytest.mark.parametrize("field", ["indices", "attention"])
def test_validly_hashed_but_different_consumer_is_rejected(evidence_run, field):
    def update(tensors):
        tensors[field][0].flatten()[0] += 1

    _replace_profile_tensors(evidence_run, update)
    with pytest.raises(ValueError, match=f"profile {field} differs"):
        report._verify_saved_consumers(*evidence_run)


def test_nonfinite_saved_attention_is_rejected(evidence_run):
    def update(tensors):
        tensors["attention"][0][0, 0, 0] = float("nan")

    _replace_profile_tensors(evidence_run, update)
    with pytest.raises(ValueError, match="non-finite"):
        report._verify_saved_consumers(*evidence_run)


@pytest.mark.parametrize("field,dtype", [("indices", torch.int64), ("attention", torch.float32)])
def test_equal_values_with_wrong_dtype_are_rejected(evidence_run, field, dtype):
    def update(tensors):
        tensors[field][0] = tensors[field][0].to(dtype)

    _replace_profile_tensors(evidence_run, update)
    with pytest.raises(ValueError, match="shape or dtype"):
        report._verify_saved_consumers(*evidence_run)


def test_missing_layer_output_is_rejected(evidence_run):
    _replace_profile_tensors(evidence_run, lambda tensors: tensors["attention"].clear())
    with pytest.raises(ValueError, match="selection/attention layers"):
        report._verify_saved_consumers(*evidence_run)


def test_missing_resident_phase_is_rejected(evidence_run):
    evidence_run[3]["samples"].pop()
    with pytest.raises(ValueError, match="resident reference"):
        report._verify_saved_consumers(*evidence_run)


def test_resident_artifact_must_be_bound_to_receipt(evidence_run):
    evidence_run[4]["artifact_paths"].popitem()
    with pytest.raises(ValueError, match="not bound by the acceptance receipt"):
        report._verify_saved_consumers(*evidence_run)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_profile_sample_coverage_must_be_complete_and_unique(evidence_run, change):
    samples = evidence_run[1]["samples"]
    if change == "missing":
        samples.pop()
    else:
        samples.append(deepcopy(samples[0]))
    with pytest.raises(ValueError, match="missing measured|duplicate sample"):
        report._verify_saved_consumers(*evidence_run)


def test_different_samples_cannot_share_one_evidence_file(evidence_run):
    samples = evidence_run[1]["samples"]
    samples[1]["evidence"] = deepcopy(samples[0]["evidence"])
    with pytest.raises(ValueError, match="separate consumer evidence files"):
        report._verify_saved_consumers(*evidence_run)


def test_evidence_path_cannot_escape_run_directory(evidence_run):
    evidence_run[1]["samples"][0]["evidence"]["path"] = "../check/reference.pt"
    with pytest.raises(ValueError, match="inside its run directory"):
        report._verify_saved_consumers(*evidence_run)
