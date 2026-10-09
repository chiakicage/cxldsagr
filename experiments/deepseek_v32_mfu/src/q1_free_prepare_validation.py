"""Private H64K/A1 cap64 adapter for the unchanged independent stage proofs.

Only each validator's declared preparation-cap expression changes. Every score,
selection, ownership, FIFO publication, record, counter and transition check is
compiled from the identified original source. Evidence and returned proofs keep
their actual cap64; this module never relabels them as a full FIFO preparation.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

from experiments.deepseek_v32_mfu.src import extend_graph_validation as graph_validation
from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as transition_audit

CONTRACT = "private-cold-h65536-a1-p65600-free-prefix64-v1"
_GEOMETRY = {"H": 65536, "A": 1, "slots": 65600}
_POLICY = {
    "prefetch_policy": transition_audit.OFFICIAL_POLICY,
    "max_prefetch": 64,
    "prepared_max_prefetch": 64,
    "hint_index": 1,
}
_EXECUTION = {
    "method": "echo",
    "residency": "cold",
    "single_session": True,
    "num_layers": 3,
}
_PATHS = tuple(
    Path(path).resolve()
    for path in (__file__, transition_audit.__file__, graph_validation.__file__)
)


def _hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in _PATHS}


_SOURCE_HASHES = _hashes()


def _require_fields(value, expected, label):
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not type(wanted) or value[key] != wanted
        for key, wanted in expected.items()
    ):
        raise ValueError(f"Unsupported private bounded-free {label}")


def _layer_cap(layer):
    _require_fields(layer, {**_GEOMETRY, **_POLICY}, "layer scope")
    _require_fields(
        layer,
        {
            "record_bytes": 1152,
            "topk": 2048,
            "host_arena_tokens": 65600,
            "session_host_tokens": 65600,
        },
        "record and host geometry",
    )
    if type(layer.get("layer")) is not int or layer["layer"] not in (0, 1, 2):
        raise ValueError("Private bounded-free proof requires layers 0-2")
    if layer["slots"] - layer["H"] < 64:
        raise ValueError("Private bounded-free preparation lacks 64-slot headroom")
    return 64


def _scope_cap(scope):
    _require_fields(scope, {**_EXECUTION, **_GEOMETRY, **_POLICY}, "execution scope")
    return 64


def _private_module(module, function_name, old_expression, helper_name, helper, argument):
    """Replace exactly one named assignment, leaving the rest of the AST intact."""
    path = Path(module.__file__).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != _SOURCE_HASHES[str(path)]:
        raise RuntimeError("Independent validation source changed during private compilation")
    tree = ast.parse(source, filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(functions) != 1:
        raise RuntimeError("Independent validator function layout changed")
    assignments = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "prepared_cap"
    ]
    expected = ast.parse(old_expression, mode="eval").body
    if len(assignments) != 1 or ast.dump(assignments[0].value) != ast.dump(expected):
        raise RuntimeError("Independent validator preparation-cap expression changed")
    assignments[0].value = ast.copy_location(
        ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=[ast.Name(id=argument, ctx=ast.Load())],
            keywords=[],
        ),
        assignments[0].value,
    )
    namespace = {
        "__name__": "_private_free_prepare_validation_" + module.__name__.rsplit(".", 1)[-1],
        "__file__": str(path),
        helper_name: helper,
    }
    exec(  # noqa: S102 -- compile only identified local validator source with one checked AST edit
        compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace
    )
    return namespace


_PRIVATE_TRANSITION = _private_module(
    transition_audit,
    "_metadata",
    "min(8192, slots - queries)",
    "_private_bounded_free_layer_cap",
    _layer_cap,
    "layer",
)
_PRIVATE_GRAPH = _private_module(
    graph_validation,
    "compare_cache_state",
    'min(8192, scope["slots"] - scope["A"])',
    "_private_bounded_free_scope_cap",
    _scope_cap,
    "scope",
)
_ORIGINAL_COMPARE = graph_validation.compare_cache_state
_ACTIVE = False


def validation_identity():
    """Return the exact private contract and all source inputs to its AST copies."""
    if _hashes() != _SOURCE_HASHES:
        raise RuntimeError("Private validation source changed after import")
    return {
        "contract": CONTRACT,
        "geometry": dict(_GEOMETRY),
        "policy": dict(_POLICY),
        "execution": dict(_EXECUTION),
        "source_sha256": dict(_SOURCE_HASHES),
        "ast_changes": {
            "prefetch_transition_audit._metadata.prepared_cap": "validated private layer cap64",
            "extend_graph_validation.compare_cache_state.prepared_cap": (
                "validated private execution cap64"
            ),
        },
        "boundary": (
            "Ordinary persistent append with exclusive sole-session preparation is declared by "
            "the separately bound integration sources. This CPU adapter checks observed cold "
            "stages and final state; it does not establish native dispatch or GPU execution."
        ),
    }


@contextmanager
def candidate_validation():
    """Let ColdPrefetchObserver import the private cap64 proof functions temporarily.

    Use only around candidate observation. Baseline observation must run outside
    this context. Both functions use the same private globals, so compact reread
    retains every raw-proof check and the actual cap64 throughout.
    """
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("Private candidate validation cannot nest")
    validation_identity()
    _ACTIVE = True
    saved = []
    try:
        for name in ("validate_execution", "compact_evidence"):
            replacement = _PRIVATE_TRANSITION[name]
            saved.append((name, getattr(transition_audit, name), replacement))
            setattr(transition_audit, name, replacement)
        yield
    finally:
        primary, errors = sys.exception(), []
        for name, original, replacement in reversed(saved):
            try:
                if getattr(transition_audit, name) is not replacement:
                    raise RuntimeError("Concurrent independent validator mutation")
                setattr(transition_audit, name, original)
            except BaseException as error:  # noqa: BLE001 -- preserve all restoration failures
                errors.append(error)
        _ACTIVE = False
        try:
            validation_identity()
        except BaseException as error:  # noqa: BLE001 -- retain body and identity errors together
            errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "Private validation restoration failed", ([primary] if primary else []) + errors
            )


def compare_cache_state(
    actual, expected, *, actual_prefetch=None, expected_prefetch=None, prepared_cap
):
    """Compare within one arm under an explicit cap; never normalize proof scopes."""
    if type(prepared_cap) is not int or prepared_cap not in (64, 8192):
        raise ValueError("Private model comparison requires explicit cap64 or baseline cap8192")
    if prepared_cap == 8192:
        return _ORIGINAL_COMPARE(
            actual,
            expected,
            actual_prefetch=actual_prefetch,
            expected_prefetch=expected_prefetch,
        )
    validation_identity()
    for proof in (actual_prefetch, expected_prefetch):
        if not isinstance(proof, dict):
            raise ValueError(  # noqa: TRY004 -- a missing or malformed acceptance record
                "Private cap64 comparison needs both bound transition proofs"
            )
        _scope_cap(proof.get("scope"))
        _require_fields(
            proof.get("proof", {}).get("scope"),
            {"record_bytes": 1152, "topk": 2048},
            "audited record geometry",
        )
    return _PRIVATE_GRAPH["compare_cache_state"](
        actual,
        expected,
        actual_prefetch=actual_prefetch,
        expected_prefetch=expected_prefetch,
    )
