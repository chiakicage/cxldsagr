import json

import pytest

from evaluation.validation import require_receipt, write_receipt


def test_acceptance_reuse_and_evidence_integrity(tmp_path):
    output = tmp_path / "hidden.bin"
    output.write_bytes(b"checked output")
    receipt = tmp_path / "acceptance.json"
    identity = {"source": "actual source", "shape": [128, 4096], "cache": "fixed"}
    result = write_receipt(
        receipt,
        kind="model",
        identity=identity,
        checks={"passed": True, "all_hidden": True},
        artifacts={"hidden": output},
    )
    assert result["artifact_paths"]["hidden"] == str(output)
    assert require_receipt(receipt, kind="model", identity=identity) == result
    output.write_bytes(b"different output")
    with pytest.raises(ValueError, match="evidence changed"):
        require_receipt(receipt, kind="model", identity=identity)


def test_rejects_different_execution_and_failed_checks(tmp_path):
    receipt = tmp_path / "acceptance.json"
    identity = {"source": "a", "graph": False}
    with pytest.raises(ValueError, match="successful"):
        write_receipt(receipt, kind="model", identity=identity, checks={"passed": False})
    assert not receipt.exists()
    write_receipt(receipt, kind="model", identity=identity, checks={"passed": True})
    with pytest.raises(ValueError, match="identity"):
        require_receipt(receipt, kind="model", identity={**identity, "graph": True})
    with pytest.raises(ValueError, match="kind"):
        require_receipt(receipt, kind="operator", identity=identity)
    with pytest.raises(FileExistsError):
        write_receipt(receipt, kind="model", identity=identity, checks={"passed": True})
    record = json.loads(receipt.read_text())
    record["identity"]["source"] = "b"
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity"):
        require_receipt(receipt, kind="model", identity=record["identity"])
