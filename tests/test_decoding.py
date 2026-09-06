import math
import unittest

import torch

from plume.decoding import js_divergence


class DecodingTests(unittest.TestCase):
    def test_js_zero_for_equal_distributions(self):
        log_p = torch.log_softmax(torch.tensor([[1.0, 2.0]]), dim=-1)
        self.assertTrue(torch.allclose(js_divergence(log_p, log_p), torch.zeros(1), atol=1e-7))

    def test_js_is_symmetric_and_bounded(self):
        p = torch.log(torch.tensor([[0.99, 0.01]]))
        q = torch.log(torch.tensor([[0.01, 0.99]]))
        self.assertTrue(torch.allclose(js_divergence(p, q), js_divergence(q, p)))
        self.assertLessEqual(js_divergence(p, q).item(), math.log(2))
        self.assertGreater(js_divergence(p, q).item(), 0)


if __name__ == "__main__":
    unittest.main()
