"""Frozen full-model harness with immutable FlashInfer and a pre-gate envelope."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_flashinfer as flashinfer
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_model as model

MODULES = {"rope", "silu_and_mul", "topk"}
POLICY = "private-fused-prepare-model-immutable-flashinfer-v1"
_source_identity = model.source_identity
_runtime = model.runtime
_require_receipt = model.require_receipt
_write = model.write


def source_identity(args, accepted):
    identity = _source_identity(args, accepted)
    for path in (Path(__file__).resolve(), Path(flashinfer.__file__).resolve()):
        identity["sources"][str(path.relative_to(model.ROOT))] = model.digest(path)
    identity["private_flashinfer_policy"] = POLICY
    return identity


def runtime():
    observed = _runtime()
    records = flashinfer.native_info()
    model.require(set(records) == MODULES, "Incomplete private FlashInfer native inventory")
    rows = [row for row in observed["model"]["loaded_jit"]["native_jit"] if row["name"] in MODULES]
    indexed = {row["name"]: row for row in rows}
    model.require(len(rows) == len(indexed) == 3, "Ambiguous FlashInfer collector inventory")
    for name, record in records.items():
        row = indexed[name]
        request = record["build_identity"]["request"]
        model.require(
            row["loaded_in_this_process"] is True
            and row["library"] == record["library"]
            and row["build_metadata"] == record["build_metadata"]
            and row["sources"] == record["build_identity"]["sources"]
            and row["cuda_flags"] == request["extra_cuda_cflags"]
            and row["cxx_flags"] == request["extra_cflags"],
            "FlashInfer collector does not identify the pinned native: " + name,
        )
    observed["private_flashinfer_native"] = records
    return observed


def main():
    args = model.parser().parse_args()
    destination = args.output_dir.resolve()
    model.component.raw.environment()

    def require_receipt(path, *, kind, identity):
        if kind == model.KIND:
            # Retain the actual attempted identity before a rejecting gate; do
            # not catch, normalize or retry the original receipt check.
            _write(
                destination / "pre_gate_identity.json",
                {
                    "schema": POLICY,
                    "mode": args.mode,
                    "pairs": args.pairs,
                    "receipt_path": str(Path(path).resolve()) if path is not None else None,
                    "identity": identity,
                },
            )
        return _require_receipt(path, kind=kind, identity=identity)

    def write(path, value):
        if Path(path) == destination / "result.json":
            # Finish provenance archival before declaring completion. During
            # check, the frozen producer's following receipt includes these files.
            flashinfer.archive(destination)
        return _write(path, value)

    with (
        flashinfer.pinned(),
        patch.object(model, "source_identity", source_identity),
        patch.object(model, "runtime", runtime),
        patch.object(model, "require_receipt", require_receipt),
        patch.object(model, "write", write),
    ):
        model.main()


if __name__ == "__main__":
    main()
