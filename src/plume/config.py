from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlumeConfig:
    """PaperV5 inference defaults."""

    alpha: float = 1.0
    beta: float = 0.75
    lambda_max: float = 1.0
    tau: float = 0.3
    lexical_k1: float = 1.5
    recency_margin: float = 0.0
    max_context_chunk_tokens: int = 8192
    max_memory_unit_tokens: int = 512
    max_new_tokens: int = 256
    router: str = "lexical"
    hybrid_embedding_weight: float = 0.25
    hybrid_top_k: int = 3
    iterative_mode: bool = False

    def validate(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.lambda_max < 0 or self.tau <= 0:
            raise ValueError("lambda_max must be non-negative and tau positive")
        if self.max_context_chunk_tokens <= 0 or self.max_memory_unit_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.router not in {"lexical", "hybrid"}:
            raise ValueError("router must be lexical or hybrid")
        if not 0 <= self.hybrid_embedding_weight <= 1:
            raise ValueError("hybrid_embedding_weight must be in [0, 1]")
        if self.hybrid_top_k <= 0:
            raise ValueError("hybrid_top_k must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
