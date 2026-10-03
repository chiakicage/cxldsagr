import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32 import echo_index


class IndexAddressTests(unittest.TestCase):
    def test_promotes_only_large_buffers_before_call_and_restores(self):
        class GetK:
            @classmethod
            def execute(cls, pool, buf, seq_len, page_indices):
                return page_indices

        class GetS(GetK):
            execute = GetK.__dict__["execute"]

        old = GetK.__dict__["execute"]
        pages = SimpleNamespace(to=lambda dtype: "int64 pages")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accessor.py"
            path.write_bytes(b"fixture")
            accessor = SimpleNamespace(__file__=str(path), GetK=GetK, GetS=GetS)
            with (
                patch.object(
                    echo_index, "INDEX_SOURCE_SHA256", hashlib.sha256(b"fixture").hexdigest()
                ),
                self.assertRaisesRegex(RuntimeError, "caller"),
                echo_index.scoped_index_address_fix(accessor, SimpleNamespace(int64="int64")),
            ):
                for cls in (GetK, GetS):
                    self.assertIs(
                        cls.execute(None, SimpleNamespace(numel=lambda: 2**31 - 1), 1, pages),
                        pages,
                    )
                    self.assertEqual(
                        cls.execute(None, SimpleNamespace(numel=lambda: 2**31), 1, pages),
                        "int64 pages",
                    )
                raise RuntimeError("caller")
            self.assertIs(GetK.__dict__["execute"], old)
            with (
                self.assertRaisesRegex(ValueError, "source differs"),
                echo_index.scoped_index_address_fix(accessor, None),
            ):
                pass


if __name__ == "__main__":
    unittest.main()
