import unittest

import torch

from plume.adapters import global_update, weighted_sum


def _effective(adapter, name="m"):
    value = adapter[name]
    return value["B"][0, 0].T @ value["A"][0, 0]


class AdapterTests(unittest.TestCase):
    def test_weighted_sum_is_exact(self):
        first = {"m": {"A": torch.randn(1, 2, 3, 5), "B": torch.randn(1, 2, 3, 4)}}
        second = {"m": {"A": torch.randn(1, 2, 3, 5), "B": torch.randn(1, 2, 3, 4)}}
        result = weighted_sum(((first, 1.25), (second, -0.5)))
        for layer in range(2):
            got = result["m"]["B"][0, layer].T @ result["m"]["A"][0, layer]
            expected = (1.25 * first["m"]["B"][0, layer].T @ first["m"]["A"][0, layer]
                        - 0.5 * second["m"]["B"][0, layer].T @ second["m"]["A"][0, layer])
            self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_global_update_coefficients(self):
        full = {"m": {"A": torch.ones(1, 1, 1, 1), "B": torch.ones(1, 1, 1, 1)}}
        old = {"m": {"A": torch.ones(1, 1, 1, 1), "B": 2 * torch.ones(1, 1, 1, 1)}}
        self.assertTrue(torch.allclose(_effective(global_update(full, old, 1.0, 0.75)),
                                       torch.tensor([[0.25]])))


if __name__ == "__main__":
    unittest.main()
