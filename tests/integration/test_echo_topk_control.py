"""Dependency-free CPU contracts for the test-only top-k order control."""

import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

try:
    from tests.integration.echo_topk_control import (
        canonical_selected_order,
        scoped_logical_topk_order,
    )
except ModuleNotFoundError:
    from echo_topk_control import canonical_selected_order, scoped_logical_topk_order


def elementwise(values, other, operation):
    if isinstance(values, list):
        if isinstance(other, list):
            return [elementwise(a, b, operation) for a, b in zip(values, other, strict=True)]
        return [elementwise(a, other, operation) for a in values]
    return operation(values, other)


class Array:
    def __init__(self, values, dtype="int32", device="fake"):
        self.values, self.dtype, self.device = values, dtype, device
        self.ndim = 2 if values and isinstance(values[0], list) else 1

    def numel(self):
        return sum(map(len, self.values)) if self.ndim == 2 else len(self.values)

    def unique(self):
        return Array(sorted(set(self.values)))

    def _op(self, other, operation):
        other = other.values if isinstance(other, Array) else other
        return Array(elementwise(self.values, other, operation))

    def __ge__(self, other):
        return self._op(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._op(other, lambda a, b: a < b)

    def __and__(self, other):
        return self._op(other, lambda a, b: a and b)

    def __or__(self, other):
        return self._op(other, lambda a, b: a or b)

    def __invert__(self):
        return self._op(None, lambda a, b: not a)

    def all(self):
        return all(all(row) for row in self.values) if self.ndim == 2 else all(self.values)

    def __getitem__(self, indices):
        return Array(elementwise(indices.values, None, lambda i, _: self.values[i]))

    def __setitem__(self, indices, values):
        for index, value in zip(indices.values, values.values, strict=True):
            self.values[index] = value

    def argsort(self, dim, stable=True):
        assert dim == -1 and stable
        return Array([sorted(range(len(row)), key=row.__getitem__) for row in self.values])

    def gather(self, dim, indices):
        assert dim == -1
        return Array(
            [
                [row[i] for i in selected]
                for row, selected in zip(self.values, indices.values, strict=True)
            ]
        )

    def sort(self, dim):
        return SimpleNamespace(values=Array([sorted(row) for row in self.values]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


FAKE_TORCH = SimpleNamespace(
    int32="int32",
    int64="int64",
    _assert_async=require,
    full=lambda shape, value, **kwargs: Array([value] * shape[0], **kwargs),
    arange=lambda length, **kwargs: Array(list(range(length)), **kwargs),
    where=lambda mask, values, otherwise: Array(
        elementwise(
            mask.values, values.values, lambda condition, value: value if condition else otherwise
        )
    ),
    equal=lambda a, b: a.values == b.values,
)


class EchoTopkControlTests(unittest.TestCase):
    def reorder(self, selected, locations=(9, 3, 7), fused=True):
        return canonical_selected_order(
            Array(selected),
            Array(list(locations)),
            capacity=16,
            fused=fused,
            torch_module=FAKE_TORCH,
        ).values

    def test_only_permutation_preserves_duplicates_and_padding(self):
        source = [[3, -1, 7, 9, 3], [-1, 7, -1, 9, 7]]
        result = self.reorder(source)
        self.assertEqual(result, [[9, 3, 3, 7, -1], [9, 7, 7, -1, -1]])
        for original, reordered in zip(source, result, strict=True):
            self.assertEqual(Counter(original), Counter(reordered))
        self.assertEqual(source, [[3, -1, 7, 9, 3], [-1, 7, -1, 9, 7]])

    def test_logical_order_is_independent_of_physical_assignment(self):
        self.assertEqual(self.reorder([[3, 7, 9]]), [[9, 3, 7]])
        self.assertEqual(self.reorder([[12, 2, 5]], (5, 12, 2)), [[5, 12, 2]])
        self.assertEqual(self.reorder([[2, 0, -1, 1]], fused=False), [[0, 1, 2, -1]])

    def test_rejects_foreign_and_invalid_locations(self):
        for source, locations in (
            ([[8]], (9, 3, 7)),
            ([[0]], (9, 3, 7)),
            ([[16]], (9, 3, 7)),
            ([[-2]], (9, 3, 7)),
            ([[9]], (9, 9, 7)),
            ([[9]], (9, 0, 7)),
        ):
            with self.subTest(source=source, locations=locations), self.assertRaises(ValueError):
                self.reorder(source, locations)

    def test_scope_restores_hooks_and_marker_on_exception(self):
        hooks = []

        def register(hook, with_kwargs):
            self.assertTrue(with_kwargs)
            hooks.append(hook)
            return SimpleNamespace(remove=lambda: hooks.remove(hook))

        runner = SimpleNamespace(
            token_to_kv_pool=object(),
            model=SimpleNamespace(
                model=SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            self_attn=SimpleNamespace(
                                indexer=SimpleNamespace(register_forward_hook=register)
                            )
                        )
                        for _ in range(3)
                    ]
                )
            ),
        )
        module_name = scoped_logical_topk_order.__module__
        with (
            patch(f"{module_name}.importlib.import_module", return_value=SimpleNamespace()),
            self.assertRaisesRegex(RuntimeError, "caller"),
            scoped_logical_topk_order(runner),
        ):
            self.assertEqual(len(hooks), 3)
            with self.assertRaisesRegex(RuntimeError, "nested"), scoped_logical_topk_order(runner):
                pass
            raise RuntimeError("caller failed")
        self.assertFalse(hooks)
        self.assertFalse(hasattr(runner, "_gr_correctness_topk_control"))


if __name__ == "__main__":
    unittest.main()
