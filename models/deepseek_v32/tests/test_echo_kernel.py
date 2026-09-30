"""CPU-only filesystem and lifecycle tests for the process-once kernel overlay."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32 import echo_kernel as kernel


class EchoKernelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "ECHO"
        self.library = self.root / "installed/deep_gemm"
        self.cache = self.root / "overlays"
        self.cuda = self.root / "cuda"
        self.cuda.mkdir()
        self.original = b"header\nbefore\nfooter\n"
        self.modified = b"header\nafter\nfooter\n"
        self.spec = kernel.HeaderPatch(
            identifier="test_patch",
            relative_path="include/deep_gemm/impls/target.cuh",
            source_sha256=hashlib.sha256(self.original).hexdigest(),
            patched_sha256=hashlib.sha256(self.modified).hexdigest(),
            before=b"before",
            after=b"after",
        )
        for base in (self.library, self.checkout / "DeepGEMM/deep_gemm"):
            target = base / self.spec.relative_path
            target.parent.mkdir(parents=True)
            target.write_bytes(self.original)
        for name, content in (
            ("include/deep_gemm/impls/other.cuh", b"other"),
            ("include/deep_gemm/common/types.cuh", b"types"),
            ("include/cute/cute.hpp", b"cute"),
            ("include/cutlass/cutlass.h", b"cutlass"),
            ("__init__.py", b"# fake package\n"),
            ("deep_gemm_cpp.so", b"fake binary"),
        ):
            path = self.library / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def build(self):
        return kernel.build_header_overlay(self.checkout, self.library, self.cache, patch=self.spec)

    def test_only_target_header_materialized_and_cache_identity_stable(self):
        overlay = self.build()
        target = overlay.root / self.spec.relative_path
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), self.modified)
        self.assertEqual((self.library / self.spec.relative_path).read_bytes(), self.original)
        self.assertEqual(
            (self.checkout / "DeepGEMM/deep_gemm" / self.spec.relative_path).read_bytes(),
            self.original,
        )
        for name in (
            "include/cute",
            "include/cutlass",
            "include/deep_gemm/common",
            "include/deep_gemm/impls/other.cuh",
        ):
            self.assertTrue((overlay.root / name).is_symlink())
            self.assertEqual((overlay.root / name).resolve(), self.library / name)
        self.assertEqual(self.build().root, overlay.root)
        self.assertEqual(overlay.metadata["patched_header_sha256"], self.spec.patched_sha256)
        # The C++ collect_files helper follows directory symlinks recursively.
        self.assertEqual(
            (overlay.root / "include/deep_gemm/common/types.cuh").read_bytes(), b"types"
        )

        def collect_cuh(directory):
            found = []
            for path in directory.iterdir():
                if path.is_dir():
                    found.extend(collect_cuh(path))
                elif path.is_file() and path.suffix == ".cuh":
                    found.append(path)
            return sorted(found)

        # Match Compiler::collect_files rather than rglob's directory-symlink policy.
        original_headers = collect_cuh(self.library / "include/deep_gemm")
        overlay_headers = collect_cuh(overlay.root / "include/deep_gemm")
        self.assertEqual(
            [path.relative_to(self.library).as_posix() for path in original_headers],
            [path.relative_to(overlay.root).as_posix() for path in overlay_headers],
        )
        differences = [
            old.relative_to(self.library).as_posix()
            for old, new in zip(original_headers, overlay_headers, strict=True)
            if old.read_bytes() != new.read_bytes()
        ]
        self.assertEqual(differences, [self.spec.relative_path])

    def test_other_header_change_changes_identity_not_existing_overlay(self):
        first = self.build()
        (self.library / "include/cute/cute.hpp").write_bytes(b"new cute")
        second = self.build()
        self.assertNotEqual(first.root, second.root)
        self.assertNotEqual(
            first.metadata["installed_include_manifest_sha256"],
            second.metadata["installed_include_manifest_sha256"],
        )

    def test_mismatched_source_and_tampered_overlay_are_rejected(self):
        path = self.library / self.spec.relative_path
        path.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "differs from"):
            self.build()
        path.write_bytes(self.original)
        overlay = self.build()
        (overlay.root / self.spec.relative_path).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.build()

    def test_patch_hash_anchor_and_path_are_all_guarded(self):
        for spec in (
            replace(self.spec, source_sha256="bad"),
            replace(self.spec, patched_sha256="bad"),
            replace(self.spec, before=b"missing"),
            replace(self.spec, before=b"e"),
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                kernel.apply_header_patch(self.original, spec)
        for name in (
            "../target.cuh",
            "/include/target.cuh",
            "include/../target.cuh",
            "include/target.h",
            "include\\target.cuh",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                kernel.build_header_overlay(
                    self.checkout,
                    self.library,
                    self.cache,
                    patch=replace(self.spec, relative_path=name),
                )
        with self.assertRaisesRegex(ValueError, "outside"):
            kernel.build_header_overlay(
                self.checkout, self.library, self.library / "cache", patch=self.spec
            )

    def test_scope_selects_overlay_once_and_never_restores_inconsistent_native_root(self):
        calls = []
        binary = self.library.parent / "deep_gemm_cpp.cpython-test.so"
        binary.write_bytes(b"top-level C++ extension")
        cpp = SimpleNamespace(
            __file__=str(binary),
            init=lambda root, cuda: calls.append((root, cuda)),
            fp8_mqa_logits_fuse_prefetch=lambda: None,
        )
        original_init = cpp.init
        package = SimpleNamespace(
            __file__=str(self.library / "__init__.py"),
            _find_cuda_home=lambda: str(self.cuda),
        )
        modules = {"torch": object(), "deep_gemm": package, "deep_gemm_cpp": cpp}
        imports = []

        def import_module(name):
            imports.append(name)
            return modules[name]

        with ExitStack() as stack:
            stack.enter_context(patch.object(kernel, "_PROCESS_STATE", "unused"))
            stack.enter_context(patch.dict(kernel._PATCHES, {"test_patch": self.spec}))
            stack.enter_context(
                patch(
                    "models.deepseek_v32.echo_adapter.verify_echo_checkout",
                    return_value=(self.checkout, "revision", "diff_sha"),
                )
            )
            stack.enter_context(patch.object(kernel.importlib, "import_module", import_module))
            stack.enter_context(
                patch.object(
                    kernel.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=package.__file__),
                )
            )
            stack.enter_context(
                patch.object(kernel.importlib.metadata, "version", return_value="2.1.1+bc1b75c")
            )
            with (
                self.assertRaisesRegex(RuntimeError, "caller"),
                kernel.scoped_echo_kernel(
                    self.checkout, patch_id="test_patch", cache_root=self.cache
                ) as info,
            ):
                self.assertEqual(imports, ["torch", "deep_gemm", "deep_gemm_cpp"])
                self.assertEqual(calls, [(info["overlay_root"], str(self.cuda))])
                cpp.init(str(self.library), str(self.cuda))
                self.assertTrue(all(root == info["overlay_root"] for root, _ in calls))
                with self.assertRaisesRegex(RuntimeError, "reconfigure"):
                    cpp.init(str(self.root / "foreign"), str(self.cuda))
                self.assertEqual(info["original_header_sha256"], self.spec.source_sha256)
                self.assertFalse(info["reversible"])
                raise RuntimeError("caller failed")
            self.assertTrue(info["scope_closed"])
            self.assertIsNot(cpp.init, original_init)
            self.assertTrue(all(root == info["overlay_root"] for root, _ in calls))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                cpp.init(str(self.library), str(self.cuda))
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                kernel.assert_echo_kernel_process_usable()
            with (
                self.assertRaisesRegex(RuntimeError, "process-once"),
                kernel.scoped_echo_kernel(self.checkout),
            ):
                pass

    def test_disabled_default_has_no_import_or_overlay_side_effect(self):
        with (
            patch.object(kernel, "_PROCESS_STATE", "unused"),
            patch.object(
                kernel.importlib, "import_module", side_effect=AssertionError("unexpected import")
            ),
        ):
            with kernel.scoped_echo_kernel(self.checkout) as info:
                self.assertFalse(info["enabled"])
                kernel.assert_echo_kernel_process_usable()
            self.assertTrue(info["scope_closed"])
            self.assertFalse(self.cache.exists())
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                kernel.assert_echo_kernel_process_usable()

    def test_preimported_deepgemm_cpp_or_sglang_rejected_before_preparation(self):
        for name in ("deep_gemm", "deep_gemm_cpp", "sglang.srt.model_executor"):
            with (
                self.subTest(name=name),
                patch.object(kernel, "_PROCESS_STATE", "unused"),
                patch.dict(sys.modules, {name: SimpleNamespace()}),
            ):
                with (
                    self.assertRaisesRegex(RuntimeError, "before importing"),
                    kernel.scoped_echo_kernel(self.checkout, patch_id="test_patch"),
                ):
                    pass
                self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
