import hashlib
import json

import pytest

from experiments.gr_serving.src import audit

PINS = {
    "3rdparty/DeepGEMM": "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7",
    "3rdparty/DeepJIT": "2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43",
    "3rdparty/FlashMLA": "ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
    "3rdparty/cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
}


def source_fixture(tmp_path, modules):
    repo, data = tmp_path / "repo", tmp_path / "data"
    manifest = {}
    for name in (
        "layers/attention.py",
        "layers/feed_forward.py",
        "layers/normalization.py",
        "cache/prefix_pool.py",
        "serving/persistent.py",
        "models/nosa/serving.py",
        "models/deepseek_v32/serving_backend.py",
    ):
        contents = f"# {name}\n".encode()
        for base in (repo, data / "source"):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        manifest[name] = hashlib.sha256(contents).hexdigest()
    (data / "source_manifest.json").write_text(json.dumps(manifest))
    meta = {
        "source_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "git_revision": "historical_parent_revision",
        "submodules": "\n".join(f" {value} {name}" for name, value in modules.items()),
        "backend_provenance": None,
    }
    return repo, data, meta


@pytest.mark.parametrize("legacy", [True, False])
def test_old_three_and_current_four_dependency_records_keep_distinct_gitlink_rules(
    tmp_path, monkeypatch, legacy
):
    modules = {
        name: value for name, value in PINS.items() if not legacy or not name.endswith("FlashMLA")
    }
    if legacy:
        modules["3rdparty/DeepGEMM"] = "b64107f2b9599ca76445b7f62eedf66bae1d095b"
        modules["3rdparty/DeepJIT"] = "e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16"
    repo, data, meta = source_fixture(tmp_path, modules)
    gitlinks = []

    def git(path, *args):
        if args[0] == "ls-tree":
            assert legacy, "current captured checkouts need not equal historical HEAD gitlinks"
            name = args[-1]
            gitlinks.append(name)
            return f"160000 commit {modules[name]} {name}"
        name = str(path.relative_to(repo))
        return modules[name] if args[0] == "rev-parse" else ""

    monkeypatch.setattr(audit, "git", git)
    result = audit.audit_sources(data, repo, meta)
    assert result["submodules"] == modules
    assert len(gitlinks) == (3 if legacy else 0)


def test_current_four_dependency_record_rejects_unpinned_flashmla(tmp_path):
    modules = {**PINS, "3rdparty/FlashMLA": "a" * 40}
    repo, data, meta = source_fixture(tmp_path, modules)
    with pytest.raises(ValueError, match="four-dependency source pins"):
        audit.audit_sources(data, repo, meta)


def test_dependency_compatibility_does_not_accept_extra_untracked_backends(tmp_path):
    repo, data, meta = source_fixture(tmp_path, {**PINS, "3rdparty/unrelated": "a" * 40})
    with pytest.raises(ValueError, match="dependency set"):
        audit.audit_sources(data, repo, meta)
