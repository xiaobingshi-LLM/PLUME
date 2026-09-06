import unittest

import torch

from plume.memory import MemoryUnit, activate_memory, lexical_scores


class MemoryTests(unittest.TestCase):
    def test_latest_unit_breaks_lexical_tie(self):
        units = [MemoryUnit(0, "Alice lives in Rome."), MemoryUnit(1, "Alice lives in Paris.")]
        self.assertEqual(activate_memory("Where does Alice live?", units).units[0].index, 1)

    def test_recency_margin_does_not_override_clear_match(self):
        units = [MemoryUnit(0, "red red red apple"), MemoryUnit(1, "blue pear")]
        self.assertEqual(activate_memory("red apple", units, recency_margin=0.01).units[0].index, 0)

    def test_hybrid_restores_temporal_order(self):
        units = [MemoryUnit(i, str(i)) for i in range(4)]
        activation = activate_memory("", units, mode="hybrid", embeddings=torch.eye(4),
                                     query_embedding=torch.tensor([0.0, 1.0, 0.9, 0.0]), top_k=2)
        self.assertEqual([unit.index for unit in activation.units], [1, 2])

    def test_empty_query_scores_zero(self):
        self.assertEqual(lexical_scores("???", [MemoryUnit(0, "text")]), [0.0])


if __name__ == "__main__":
    unittest.main()
