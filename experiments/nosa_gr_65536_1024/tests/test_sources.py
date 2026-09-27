"""Captured source fingerprints cover native implementations and local headers."""

import hashlib

from experiments.nosa_gr_65536_1024.src import sources


def test_native_sources_are_recursively_fingerprinted_by_content(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    contents = {
        "GR/analysis/heat_curves.csv": "heat fixture\n",
        "operators/sm90/_native.py": "# loader fixture\n",
        "operators/sm90/csrc/nosa_scores.cu": '#include "detail/pipeline.cuh"\n',
        "operators/sm90/csrc/detail/pipeline.cuh": "// pipeline fixture\n",
        "operators/sm90/csrc/detail/binding.cpp": "// binding fixture\n",
        "operators/sm90/csrc/detail/layout.hpp": "// layout fixture\n",
    }
    for name, content in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    ignored = tmp_path / "operators/sm90/csrc/kernel.o"
    ignored.write_bytes(b"compiled object is not a source snapshot")
    first = sources.source_hashes()
    assert first == {
        name: hashlib.sha256(content.encode()).hexdigest() for name, content in contents.items()
    }
    header = "operators/sm90/csrc/detail/pipeline.cuh"
    (tmp_path / header).write_text("// changed pipeline\n")
    second = sources.source_hashes()
    assert second[header] != first[header]
    assert {name for name in first if first[name] != second[name]} == {header}
