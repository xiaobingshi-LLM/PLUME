from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

TERM_RE = re.compile(r"[a-z0-9]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s*|\n+")


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs): ...
    def decode(self, ids, **kwargs) -> str: ...


@dataclass(frozen=True)
class MemoryUnit:
    index: int
    text: str


@dataclass(frozen=True)
class Activation:
    units: tuple[MemoryUnit, ...]
    lexical_scores: tuple[float, ...]
    combined_scores: tuple[float, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(unit.text for unit in self.units)


def segment_memory(context: str, tokenizer: Tokenizer, max_tokens: int) -> list[MemoryUnit]:
    """Segment by paragraph/sentence, then bound oversized units by tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    source = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(context) if part.strip()]
    texts: list[str] = []
    for part in source or [context.strip()]:
        ids = tokenizer(part, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for start in range(0, len(ids), max_tokens):
            text = tokenizer.decode(ids[start : start + max_tokens], skip_special_tokens=True).strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError("context produced no memory units")
    return [MemoryUnit(index=index, text=text) for index, text in enumerate(texts)]


def lexical_scores(query: str, units: Sequence[MemoryUnit], k1: float = 1.5) -> list[float]:
    query_terms = set(TERM_RE.findall(query.lower()))
    term_counts = [Counter(TERM_RE.findall(unit.text.lower())) for unit in units]
    n_units = len(units)
    weights = {
        term: math.log(1 + (n_units - sum(term in counts for counts in term_counts) + 0.5)
                       / (sum(term in counts for counts in term_counts) + 0.5))
        for term in query_terms
    }
    total_weight = sum(weights.values())
    scores = []
    for counts in term_counts:
        if not query_terms or total_weight == 0:
            scores.append(0.0)
            continue
        covered = sum(weights[t] for t in query_terms if counts[t] > 0)
        frequency = sum(
            weights[t] * counts[t] * (k1 + 1) / (counts[t] + k1)
            for t in query_terms if counts[t] > 0
        )
        scores.append(0.5 * covered / total_weight + 0.5 * frequency / (frequency + total_weight))
    return scores


def activate_memory(
    query: str,
    units: Sequence[MemoryUnit],
    *,
    k1: float = 1.5,
    recency_margin: float = 0.0,
    mode: str = "lexical",
    embeddings: Tensor | None = None,
    query_embedding: Tensor | None = None,
    embedding_weight: float = 0.25,
    top_k: int = 3,
) -> Activation:
    if not units:
        raise ValueError("units cannot be empty")
    lexical = lexical_scores(query, units, k1)
    if mode == "lexical":
        best = max(lexical)
        eligible = [i for i, score in enumerate(lexical) if best - score <= recency_margin]
        chosen = [max(eligible)]
        combined = lexical
    elif mode == "hybrid":
        if embeddings is None or query_embedding is None:
            raise ValueError("hybrid routing requires unit and query embeddings")
        if embeddings.shape[0] != len(units):
            raise ValueError("one embedding is required per memory unit")
        cosine = F.normalize(embeddings.float(), dim=-1) @ F.normalize(query_embedding.float(), dim=-1)
        semantic = ((cosine + 1) / 2).tolist()
        combined = [(1 - embedding_weight) * lex + embedding_weight * sem
                    for lex, sem in zip(lexical, semantic)]
        ranked = sorted(range(len(units)), key=lambda i: (combined[i], i), reverse=True)
        chosen = sorted(ranked[: min(top_k, len(ranked))])
    else:
        raise ValueError(f"unsupported router: {mode}")
    return Activation(tuple(units[i] for i in chosen), tuple(lexical), tuple(combined))
