"""CPU lifecycle tests: exact bytes, registered paths, failures and archival."""

import dataclasses
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_flashinfer as loader


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "private"
    csrc = tmp_path / "installed-csrc"
    csrc.mkdir()
    source_names = {
        "rope": ["rope.cu", "flashinfer_rope_binding.cu"],
        "topk": [
            "topk.cu",
            "flashinfer_topk_binding.cu",
            "flashinfer_fast_topk_clusters_binding.cu",
        ],
    }
    for name in itertools.chain.from_iterable(source_names.values()):
        (csrc / name).write_text(f"unchanged {name}\n")
    header = tmp_path / "declared.cuh"
    header.write_text("header-v1\n")
    env = SimpleNamespace(
        FLASHINFER_GEN_SRC_DIR=tmp_path / "mutable-generated",
        FLASHINFER_CSRC_DIR=csrc,
        FLASHINFER_JIT_DIR=tmp_path / "mutable-jit",
    )
    state = SimpleNamespace(builds=[], loaded=[], delegated=[], specs={}, failure=None)

    @dataclasses.dataclass
    class Spec:
        name: str
        sources: list
        extra_cflags: list | None = None
        extra_cuda_cflags: list | None = None
        extra_ldflags: list | None = None
        extra_include_dirs: list | None = None
        is_class: bool = False
        needs_device_linking: bool = False

        @property
        def is_aot(self):
            return False

        @property
        def ninja_path(self):
            return env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

        def get_library_path(self):
            return env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

        def build_and_load(self):
            state.delegated.append(self.name)
            return "original-loader"

    def generate(name, sources, extra_cuda_cflags=None):
        spec = Spec(name, sources, extra_cuda_cflags=extra_cuda_cflags)
        state.specs.setdefault(name, spec)
        return spec

    def ninja(**request):
        return json.dumps(
            {
                "name": request["name"],
                "output": str(env.FLASHINFER_JIT_DIR),
                "sources": [str(path) for path in request["sources"]],
            },
            sort_keys=True,
        )

    def build(entry):
        state.builds.append(entry)
        if state.failure is not None:
            raise state.failure
        recipe = json.loads((entry / "build.ninja").read_text())
        assert Path(recipe["output"]) / recipe["name"] == entry
        (entry / f"{entry.name}.so").write_bytes(
            b"\x7fELF exact compiler metadata " + entry.name.encode()
        )
        for source in state.specs[entry.name].sources:
            target = entry / f"{source.parent.name}_{source.stem}.cuda.o"
            target.with_suffix(".o.d").write_text(f"{target}: {source} \\\n {header}\n")

    def load(path):
        state.loaded.append(Path(path))
        return path

    def mapped():
        return {
            str(path): (
                f"{os.major(path.stat().st_dev):02x}:{os.minor(path.stat().st_dev):02x}",
                path.stat().st_ino,
            )
            for path in state.loaded
        }

    installed = SimpleNamespace(
        env=env,
        core=SimpleNamespace(
            JitSpecNvcc=Spec,
            gen_jit_spec=generate,
            generate_ninja_build_for_op=ninja,
            jit_spec_registry=SimpleNamespace(get_all_specs=lambda: state.specs.copy()),
            MissingJITCacheError=lambda message, **kwargs: RuntimeError(message),
        ),
        activation=SimpleNamespace(
            get_act_and_mul_cu_str=lambda *args: "unchanged silu source\n",
            act_func_def_str={"silu": "unchanged silu function"},
        ),
        ffi=SimpleNamespace(load_module=load),
    )

    def identity(spec, installed):
        return {
            "schema": loader.SCHEMA,
            "request": loader._request(spec),
            "sources": [loader._file(path) for path in spec.sources],
            "include_files": [loader._file(header)],
            "environment": loader._environment(),
            "ninja_template": loader._ninja(spec, installed, root / "ninja-template"),
        }

    def make(name):
        if name == "silu_and_mul":
            env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)
            path = env.FLASHINFER_GEN_SRC_DIR / "silu_and_mul.cu"
            path.write_text("unchanged silu source\n")
            paths = [path]
        else:
            paths = [csrc / source for source in source_names.get(name, [])]
        return generate(name, paths, ["-lineinfo"] if name == "topk" else None)

    def fresh():
        state.specs.clear()
        state.loaded.clear()
        loader._LOADED.clear()

    monkeypatch.setattr(loader, "RUNTIME", root)
    monkeypatch.setattr(loader, "_LOADED", {})
    monkeypatch.setattr(loader, "_ACTIVE", False)
    monkeypatch.setattr(loader, "_installed", lambda: installed)
    monkeypatch.setattr(loader, "_input_identity", identity)
    monkeypatch.setattr(loader, "_mapped_libraries", mapped)
    monkeypatch.setattr(loader, "_run_ninja", build)
    return SimpleNamespace(
        root=root,
        make=make,
        fresh=fresh,
        state=state,
        header=header,
        installed=installed,
        original=Spec.build_and_load,
    )


def test_exact_cache_hit_preserves_elf_and_registered_metadata(runtime):
    with loader.pinned():
        spec = runtime.make("rope")
        module = spec.build_and_load()
        first = loader.native_info()["rope"]
        assert module == first["library"]["path"] == str(spec.get_library_path())
        assert str(spec.ninja_path) == first["build_metadata"]["path"]
        assert runtime.state.specs["rope"] is spec
        with pytest.raises(RuntimeError, match="cannot be rebuilt"):
            spec.build()
    assert runtime.installed.core.JitSpecNvcc.build_and_load is runtime.original
    runtime.fresh()
    with loader.pinned():
        spec = runtime.make("rope")
        spec.build_and_load()
        assert loader.native_info()["rope"] == first
    assert len(runtime.state.builds) == 1
    assert not runtime.installed.env.FLASHINFER_JIT_DIR.exists()


def test_complete_archive_contains_actual_libraries_recipes_and_records(runtime, tmp_path):
    with loader.pinned():
        for name in sorted(loader.MODULES):
            runtime.make(name).build_and_load()
        records = loader.native_info()
        archived = loader.archive(tmp_path / "result")
    assert set(records) == set(archived) == loader.MODULES
    manifest = json.loads((tmp_path / "result/flashinfer_native/manifest.json").read_text())
    assert manifest["modules"] == archived
    for name, fields in archived.items():
        for field, copies in fields.items():
            if field == "dependency_metadata":
                assert [copy["original"] for copy in copies] == records[name][field]
                for copy in copies:
                    assert (
                        Path(copy["archived"]["path"]).read_bytes()
                        == Path(copy["original"]["path"]).read_bytes()
                    )
                continue
            assert copies["original"] == records[name][field]
            assert (
                Path(copies["archived"]["path"]).read_bytes()
                == Path(copies["original"]["path"]).read_bytes()
            )
    assert not runtime.installed.env.FLASHINFER_GEN_SRC_DIR.exists()


@pytest.mark.parametrize("field", ["library", "build_metadata", "manifest"])
def test_modified_cached_bytes_are_rejected_without_rebuild(runtime, field):
    with loader.pinned():
        runtime.make("rope").build_and_load()
        record = loader.native_info()["rope"]
    path = Path(record[field]["path"])
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    runtime.fresh()
    with loader.pinned(), pytest.raises(RuntimeError, match="changed"):
        runtime.make("rope").build_and_load()
    assert len(runtime.state.builds) == 1


def test_corrupt_record_never_loads_or_rebuilds(runtime):
    with loader.pinned():
        runtime.make("rope").build_and_load()
        record = loader.native_info()["rope"]
    path = Path(record["manifest"]["path"])
    changed = json.loads(path.read_text())
    changed["cache_key"] = "wrong"
    path.chmod(0o644)
    path.write_text(json.dumps(changed, sort_keys=True, indent=2) + "\n")
    runtime.fresh()
    with loader.pinned(), pytest.raises(RuntimeError, match="Invalid immutable"):
        runtime.make("rope").build_and_load()
    assert len(runtime.state.builds) == 1 and runtime.state.loaded == []


@pytest.mark.parametrize("failure_kind", ["build", "load"])
def test_original_failure_propagates_without_fallback(runtime, monkeypatch, failure_kind):
    failure = OSError("the original failure")
    if failure_kind == "build":
        runtime.state.failure = failure
    else:

        def fail(_path):
            raise failure

        monkeypatch.setattr(runtime.installed.ffi, "load_module", fail)
    with pytest.raises(OSError) as error, loader.pinned():
        runtime.make("rope").build_and_load()
    assert error.value is failure
    assert not runtime.state.delegated
    assert not loader._ACTIVE


def test_incomplete_build_is_never_retried(runtime):
    runtime.state.failure = RuntimeError("compiler failed")
    with loader.pinned(), pytest.raises(RuntimeError, match="compiler failed"):
        runtime.make("rope").build_and_load()
    runtime.state.failure = None
    runtime.fresh()
    with loader.pinned(), pytest.raises(FileNotFoundError):
        runtime.make("rope").build_and_load()
    assert len(runtime.state.builds) == 1


def test_include_change_uses_new_identity_and_preserves_existing_elf(runtime):
    with loader.pinned():
        runtime.make("rope").build_and_load()
        old = loader.native_info()["rope"]
    runtime.header.write_text("header-v2\n")
    runtime.fresh()
    with loader.pinned():
        runtime.make("rope").build_and_load()
        new = loader.native_info()["rope"]
    assert old["cache_key"] != new["cache_key"]
    assert loader._file(old["library"]["path"]) == old["library"]
    assert len(runtime.state.builds) == 2


def test_actual_dependency_change_rejects_existing_artifact(runtime, monkeypatch, tmp_path):
    system_header = tmp_path / "implicit-system-header.h"
    system_header.write_text("v1")
    original = loader._dependencies

    def dependencies(entry):
        result = original(entry)
        result["files"].append(loader._file(system_header))
        return result

    monkeypatch.setattr(loader, "_dependencies", dependencies)
    with loader.pinned():
        runtime.make("rope").build_and_load()
    system_header.write_text("v2")
    runtime.fresh()
    with loader.pinned(), pytest.raises(RuntimeError, match="changed"):
        runtime.make("rope").build_and_load()
    assert len(runtime.state.builds) == 1


def test_unrelated_spec_uses_original_loader(runtime):
    with loader.pinned():
        assert runtime.make("unrelated").build_and_load() == "original-loader"
    assert runtime.state.delegated == ["unrelated"] and runtime.state.builds == []


@pytest.mark.parametrize("change", ["flags", "source", "generated"])
def test_unexpected_supported_request_is_rejected(runtime, change):
    with loader.pinned():
        spec = runtime.make("silu_and_mul" if change == "generated" else "rope")
        if change == "flags":
            spec.extra_cuda_cflags = ["--different"]
        elif change == "source":
            spec.sources.reverse()
        else:
            spec.sources[0].write_text("changed generated function")
        with pytest.raises(RuntimeError, match="Unexpected FlashInfer"):
            spec.build_and_load()
    assert runtime.state.builds == []


def test_already_registered_spec_prevents_pinning(runtime):
    runtime.make("rope")
    with pytest.raises(RuntimeError, match="before any supported spec"), loader.pinned():
        pytest.fail("must not enter")


def test_already_mapped_elf_prevents_pinning(runtime, monkeypatch):
    monkeypatch.setattr(
        loader, "_mapped_libraries", lambda: {"/old/rope.so (deleted)": ("00:00", 0)}
    )
    with pytest.raises(RuntimeError, match="already loaded"), loader.pinned():
        pytest.fail("must not enter")


def test_missing_actual_mapping_cannot_claim_runtime_participation(runtime, monkeypatch):
    monkeypatch.setattr(loader, "_mapped_libraries", dict)
    with loader.pinned(), pytest.raises(RuntimeError, match="actually mapped"):
        runtime.make("rope").build_and_load()
    assert loader._LOADED == {}


def test_loaded_inode_replacement_cannot_claim_runtime_participation(runtime, monkeypatch):
    with loader.pinned():
        runtime.make("rope").build_and_load()
        old_maps = loader._mapped_libraries()
        record = loader.native_info()["rope"]
        path = Path(record["library"]["path"])
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        monkeypatch.setattr(loader, "_mapped_libraries", lambda: old_maps)
        with pytest.raises(RuntimeError, match="actually mapped"):
            loader.native_info()


def test_incomplete_inventory_cannot_be_archived(runtime, tmp_path):
    with loader.pinned():
        runtime.make("rope").build_and_load()
        with pytest.raises(RuntimeError, match="incomplete"):
            loader.archive(tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_disabled_jit_never_builds_missing_artifact(runtime, monkeypatch):
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    with loader.pinned(), pytest.raises(RuntimeError, match="absent"):
        runtime.make("rope").build_and_load()
    assert runtime.state.builds == []


def test_header_tree_follows_links_and_binds_exact_bytes(tmp_path):
    include = tmp_path / "include"
    outside = tmp_path / "shared"
    include.mkdir()
    outside.mkdir()
    header = outside / "definition.cuh"
    header.write_text("v1")
    (include / "shared").symlink_to(outside, target_is_directory=True)
    (outside / "cycle").symlink_to(include, target_is_directory=True)
    records = loader._tree_files([include])
    assert records == [loader._file(header)]
    header.write_text("v2")
    with pytest.raises(RuntimeError, match="changed"):
        loader._verify_files(records)


@pytest.mark.parametrize("text", ["missing deps", "other.o: /a.h\n", ""])
def test_malformed_compiler_dependencies_fail(tmp_path, text):
    (tmp_path / "source.cuda.o.d").write_text(text)
    with pytest.raises(RuntimeError, match="dependenc"):
        loader._dependencies(tmp_path)


def test_missing_compiler_dependencies_fail(tmp_path):
    with pytest.raises(RuntimeError, match="dependencies"):
        loader._dependencies(tmp_path)


def test_build_command_preserves_actual_depfiles(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        loader.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setenv("MAX_JOBS", "8")
    loader._run_ninja(tmp_path)
    command = calls[0][0][0]
    assert command[command.index("-d") + 1] == "keepdepfile"
    assert command[-2:] == ["-j", "8"]
    assert calls[0][1] == {"cwd": tmp_path, "check": True}
