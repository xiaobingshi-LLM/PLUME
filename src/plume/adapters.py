from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

Adapter = Mapping[str, Mapping[str, Tensor]]


def _weight_tensor(weight: float | Tensor, reference: Tensor) -> Tensor:
    value = torch.as_tensor(weight, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value
    if value.ndim == 1 and value.numel() == reference.shape[1]:
        return value.reshape(1, -1, 1, 1)
    raise ValueError("A vector adapter weight must have one value per layer")


def weighted_sum(terms: Sequence[tuple[Adapter, float | Tensor]]) -> dict:
    """Represent a weighted sum of LoRA updates exactly by rank concatenation."""
    if not terms:
        raise ValueError("at least one adapter is required")
    module_names = tuple(terms[0][0])
    if any(tuple(adapter) != module_names for adapter, _ in terms[1:]):
        raise ValueError("all adapters must contain identical modules")
    result = {}
    for name in module_names:
        ref_a, ref_b = terms[0][0][name]["A"], terms[0][0][name]["B"]
        a_parts, b_parts = [], []
        for adapter, weight in terms:
            a, b = adapter[name]["A"], adapter[name]["B"]
            if a.shape[:2] != ref_a.shape[:2] or a.shape[-1] != ref_a.shape[-1]:
                raise ValueError(f"incompatible A tensor for {name}: {tuple(a.shape)}")
            if b.shape[:2] != ref_b.shape[:2] or b.shape[-1] != ref_b.shape[-1]:
                raise ValueError(f"incompatible B tensor for {name}: {tuple(b.shape)}")
            a_parts.append(a)
            b_parts.append(_weight_tensor(weight, b) * b)
        result[name] = {
            "A": torch.cat(a_parts, dim=2),
            "B": torch.cat(b_parts, dim=2),
        }
    return result


def global_update(full: Adapter, old: Adapter, alpha: float, beta: float) -> dict:
    """Compute alpha*full + beta*(full-old), exactly as Eq. (1)."""
    return weighted_sum(((full, alpha + beta), (old, -beta)))
