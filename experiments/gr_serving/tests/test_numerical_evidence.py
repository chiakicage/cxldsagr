import json
from types import SimpleNamespace

import pytest
import torch

from experiments.gr_serving.src import audit, measure


def evidence_fixture(path):
    model, users, queries = "deepseek_v32", 16, 2
    hidden = torch.ones((queries, 7168), dtype=torch.bfloat16)
    logits = torch.ones((1, 129280), dtype=torch.float32)
    reference = path / "reference" / model / str(users)
    reference.mkdir(parents=True)
    torch.save(hidden, reference / "000000.pt")
    rows = []
    for scheme in audit.SCHEMES[model]:
        evidence = measure.save_numerical_evidence(
            path,
            model=model,
            scheme=scheme,
            users=users,
            request_id=0,
            hidden=hidden,
            logits=logits,
        )
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "num_users": users,
                "request_id": 0,
                "tensor_evidence": evidence,
                "logits": measure.numerical_comparison(logits, logits, atol=0, rtol=0),
            }
        )
    (path / "correctness.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = SimpleNamespace(
        models=[model],
        populations=[users],
        candidate_tokens=queries,
        cases=[(model, scheme, users) for scheme in audit.SCHEMES[model]],
        request_counts={users: 1},
    )
    return contract, rows


def test_reopens_all_hidden_and_logits(tmp_path):
    contract, _ = evidence_fixture(tmp_path)
    result = audit.audit_numerical_tensors(tmp_path, contract)
    assert result["status"] == "accepted"
    assert result["complete_hidden_comparisons"] == result["files"] == 4
    assert len(result["files_sha256"]) == 5


@pytest.mark.parametrize("field", ["hidden", "logits"])
def test_recomputed_comparison_rejects_changed_tensor_even_with_updated_hash(tmp_path, field):
    contract, rows = evidence_fixture(tmp_path)
    evidence = rows[1]["tensor_evidence"]
    path = tmp_path / evidence["path"]
    tensors = torch.load(path, weights_only=True)
    tensors[field].flatten()[17] = 3
    torch.save(tensors, path)
    evidence["sha256"] = audit.sha256(path)
    (tmp_path / "correctness.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.AuditError, match=f"recomputed {field} differs"):
        audit.audit_numerical_tensors(tmp_path, contract)


@pytest.mark.parametrize("kind", ["missing", "extra", "digest", "identity"])
def test_rejects_incomplete_or_rebound_evidence(tmp_path, kind):
    contract, rows = evidence_fixture(tmp_path)
    evidence = rows[1]["tensor_evidence"]
    path = tmp_path / evidence["path"]
    if kind == "missing":
        path.unlink()
    elif kind == "extra":
        (path.parent / "extra.pt").write_bytes(path.read_bytes())
    elif kind == "digest":
        evidence["sha256"] = "0" * 64
    else:
        evidence["path"] = rows[0]["tensor_evidence"]["path"]
    (tmp_path / "correctness.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.AuditError):
        audit.audit_numerical_tensors(tmp_path, contract)
