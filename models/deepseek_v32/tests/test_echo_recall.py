import hashlib
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32 import echo_recall as recall


class EchoRecallAddressTests(unittest.TestCase):
    def test_exact_single_promotion_with_guarded_hashes(self):
        source = "def kernel():\n    " + recall.BEFORE + "\n"
        expected = source.replace(recall.BEFORE, recall.AFTER)
        with (
            patch.object(recall, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest()),
            patch.object(recall, "PATCHED_SHA256", hashlib.sha256(expected.encode()).hexdigest()),
        ):
            self.assertEqual(recall.patched_recall_source(source), expected)
            with self.assertRaisesRegex(ValueError, "source SHA"):
                recall.patched_recall_source(source + "# changed")
            with (
                patch.object(recall, "PATCHED_SHA256", "wrong"),
                self.assertRaisesRegex(ValueError, "patched SHA"),
            ):
                recall.patched_recall_source(source)

    def test_fresh_jit_scope_restores_original_even_on_failure(self):
        source = "def kernel():\n    " + recall.BEFORE + "\n"
        expected = source.replace(recall.BEFORE, recall.AFTER)
        original = SimpleNamespace(src=source)
        module = SimpleNamespace(_recall_update_extend_kernel=original)
        replacement = SimpleNamespace(
            _unsafe_update_src=lambda src: setattr(replacement, "src", src)
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(recall, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
            )
            stack.enter_context(
                patch.object(
                    recall, "PATCHED_SHA256", hashlib.sha256(expected.encode()).hexdigest()
                )
            )
            factory = stack.enter_context(
                patch.object(recall, "_new_jit", return_value=replacement)
            )
            with (
                self.assertRaisesRegex(RuntimeError, "caller"),
                recall.scoped_extend_recall_address_fix(module) as info,
            ):
                self.assertIs(module._recall_update_extend_kernel, replacement)
                self.assertEqual(replacement.src, expected)
                self.assertFalse(info["source_files_modified"])
                factory.assert_called_once_with(original)
                raise RuntimeError("caller failed")
        self.assertIs(module._recall_update_extend_kernel, original)
        self.assertEqual(original.src, source)

    def test_large_pool_really_crosses_signed_element_offset_boundary(self):
        high = (2**31 - 1) // 576 + 1
        self.assertGreater(high * 576, 2**31 - 1)
        self.assertLess(high, 128 * 65536)


if __name__ == "__main__":
    unittest.main()
