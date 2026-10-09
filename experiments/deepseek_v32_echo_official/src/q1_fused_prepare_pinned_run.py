"""Run the frozen preparation component with an exactly pinned generic ECHO ELF."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_native as native
from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_run as component


def source_identity(args):
    identity = _source_identity(args)
    for path in (Path(__file__).resolve(), Path(native.__file__).resolve()):
        identity["sources"][str(path)] = component.digest(path)
    return identity


def execution_identity(args, sources):
    return {
        **_execution_identity(args, sources),
        "private_generic_native": native.native_info(),
    }


_source_identity = component.source_identity
_execution_identity = component.execution_identity


def main():
    # Parse first so --help remains CPU-only and creates no runtime directories.
    component.parser().parse_args()
    component.raw.environment()
    with (
        native.pinned(),
        patch.object(component, "source_identity", source_identity),
        patch.object(component, "execution_identity", execution_identity),
        patch.object(component, "__spec__", SimpleNamespace(name=__spec__.name)),
    ):
        component.main()


if __name__ == "__main__":
    main()
