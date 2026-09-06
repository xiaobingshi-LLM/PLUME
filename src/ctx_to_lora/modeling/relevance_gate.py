import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _strip_pad(ids: Tensor, pad_token_id: int | None) -> Tensor:
    if pad_token_id is None:
        return ids
    return ids[ids != pad_token_id]


@dataclass
class RelevanceGateResult:
    scores: list[float]
    use_lora: list[bool]


class BaseRelevanceGate:
    def score(self, contexts: Sequence[str], questions: Sequence[str]) -> list[float]:
        raise NotImplementedError


class LexicalRelevanceGate(BaseRelevanceGate):
    """Small BM25-style lexical gate over each context/question pair."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, contexts: Sequence[str], questions: Sequence[str]) -> list[float]:
        scores = []
        for context, question in zip(contexts, questions):
            ctx_tokens = _tokenize(context)
            q_tokens = _tokenize(question)
            if not ctx_tokens or not q_tokens:
                scores.append(0.0)
                continue

            tf = Counter(ctx_tokens)
            ctx_len = len(ctx_tokens)
            avgdl = max(ctx_len, 1)
            bm25 = 0.0
            matched = 0
            for token in set(q_tokens):
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                matched += 1
                denom = freq + self.k1 * (1 - self.b + self.b * ctx_len / avgdl)
                bm25 += freq * (self.k1 + 1) / denom

            overlap = matched / max(len(set(q_tokens)), 1)
            bm25_norm = bm25 / (bm25 + len(set(q_tokens)))
            scores.append(float(0.5 * overlap + 0.5 * bm25_norm))
        return scores


class EmbeddingRelevanceGate(BaseRelevanceGate):
    """Cosine gate using either sentence-transformers or a HF encoder path."""

    def __init__(
        self,
        model_path: str | None,
        device: torch.device | str | None = None,
        fallback_encoder: nn.Module | None = None,
        fallback_tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sentence_model = None
        self.tokenizer = None
        self.model = None
        self.fallback_encoder = fallback_encoder
        self.fallback_tokenizer = fallback_tokenizer

        if model_path is None:
            if fallback_encoder is None or fallback_tokenizer is None:
                raise ValueError(
                    "--gate_model_path is required for embedding backend when the "
                    "current model has no ctx encoder fallback."
                )
            self.fallback_encoder.eval()
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self.sentence_model = SentenceTransformer(model_path, device=str(self.device))
            except ImportError:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    def _encode(self, texts: Sequence[str]) -> Tensor:
        if self.sentence_model is not None:
            embeddings = self.sentence_model.encode(
                list(texts),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.to(self.device)

        if self.fallback_encoder is not None:
            tokens = self.fallback_tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                features = self.fallback_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                if hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim == 4:
                    features = features.mean(dim=1)
                mask = tokens["attention_mask"].unsqueeze(-1).to(features.dtype)
                pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return F.normalize(pooled, dim=-1)

        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    def score(self, contexts: Sequence[str], questions: Sequence[str]) -> list[float]:
        if not contexts:
            return []
        ctx_emb = self._encode(contexts)
        q_emb = self._encode(questions)
        cosine = (ctx_emb * q_emb).sum(dim=-1)
        return ((cosine + 1.0) / 2.0).detach().float().cpu().tolist()


class ScopeRelevanceGate(BaseRelevanceGate):
    """GRACE-style learned scope classifier gate.

    Supported model shapes:
    - torch-loaded object with score(contexts, questions)
    - torch-loaded object with predict_proba(pairs) or predict(pairs)
    - HF sequence-classification directory in gate_model_path
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device | str | None = None,
    ):
        if not model_path:
            raise ValueError("--gate_model_path is required for scope backend.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tokenizer = None
        self.loaded_object = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(model_path)
                .to(self.device)
                .eval()
            )
        except Exception:
            self.loaded_object = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            if isinstance(self.loaded_object, nn.Module):
                self.loaded_object.to(self.device).eval()

    def score(self, contexts: Sequence[str], questions: Sequence[str]) -> list[float]:
        pairs = list(zip(contexts, questions))
        if self.loaded_object is not None:
            obj = self.loaded_object
            if hasattr(obj, "score"):
                scores = obj.score(contexts, questions)
            elif hasattr(obj, "predict_proba"):
                probs = obj.predict_proba(pairs)
                scores = [p[-1] if isinstance(p, (list, tuple)) else p for p in probs]
            elif hasattr(obj, "predict"):
                scores = obj.predict(pairs)
            else:
                raise ValueError(
                    "Scope gate object must expose score, predict_proba, or predict."
                )
            return [float(x) for x in scores]

        tokens = self.tokenizer(
            list(questions),
            list(contexts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**tokens).logits
            if logits.shape[-1] == 1:
                scores = torch.sigmoid(logits[:, 0])
            else:
                scores = torch.softmax(logits, dim=-1)[:, -1]
        return scores.detach().float().cpu().tolist()


def build_relevance_gate(
    backend: str,
    gate_model_path: str | None = None,
    device: torch.device | str | None = None,
    fallback_encoder: nn.Module | None = None,
    fallback_tokenizer: PreTrainedTokenizerBase | None = None,
) -> BaseRelevanceGate:
    if backend == "lexical":
        return LexicalRelevanceGate()
    if backend == "embedding":
        return EmbeddingRelevanceGate(
            gate_model_path,
            device=device,
            fallback_encoder=fallback_encoder,
            fallback_tokenizer=fallback_tokenizer,
        )
    if backend == "scope":
        return ScopeRelevanceGate(gate_model_path, device=device)
    raise ValueError(f"Unknown gate backend: {backend}")


class RelevanceGatedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        gate: BaseRelevanceGate,
        threshold: float,
        tokenizer: PreTrainedTokenizerBase,
        ctx_tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()
        self.model = model
        self.gate = gate
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.gate_scores: list[float] = []
        self.gate_decisions: list[bool] = []

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    @property
    def base_model(self):
        return self.model.base_model

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def _decode_inputs(
        self,
        input_ids: Tensor,
        ctx_ids: Tensor | None,
        n_ctx_chunks: Tensor | None,
    ) -> tuple[list[str], list[str]]:
        questions = []
        for row in input_ids.detach().cpu():
            row = _strip_pad(row, self.tokenizer.pad_token_id)
            questions.append(self.tokenizer.decode(row, skip_special_tokens=True))

        contexts = []
        if ctx_ids is None:
            return ["" for _ in questions], questions

        chunks_per_sample = (
            n_ctx_chunks.detach().cpu().tolist()
            if n_ctx_chunks is not None
            else [1 for _ in questions]
        )
        offset = 0
        for n_chunks in chunks_per_sample:
            chunk_texts = []
            for row in ctx_ids[offset : offset + int(n_chunks)].detach().cpu():
                row = _strip_pad(row, self.ctx_tokenizer.pad_token_id)
                chunk_texts.append(
                    self.ctx_tokenizer.decode(row, skip_special_tokens=True)
                )
            contexts.append("\n".join(chunk_texts))
            offset += int(n_chunks)
        return contexts, questions

    def _score_batch(
        self,
        input_ids: Tensor,
        ctx_ids: Tensor | None,
        n_ctx_chunks: Tensor | None,
    ) -> RelevanceGateResult:
        contexts, questions = self._decode_inputs(input_ids, ctx_ids, n_ctx_chunks)
        scores = self.gate.score(contexts, questions)
        decisions = [score >= self.threshold for score in scores]
        self.gate_scores.extend(scores)
        self.gate_decisions.extend(decisions)
        return RelevanceGateResult(scores=scores, use_lora=decisions)

    def _slice_model_kwargs(
        self,
        kwargs: dict[str, Any],
        batch_indices: list[int],
        batch_size: int,
    ) -> dict[str, Any]:
        sliced = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                index = torch.tensor(batch_indices, device=value.device)
                sliced[key] = value.index_select(0, index)
            else:
                sliced[key] = value
        return sliced

    def _slice_model_args(
        self,
        args: tuple[Any, ...],
        batch_indices: list[int],
        batch_size: int,
    ) -> tuple[Any, ...]:
        sliced = []
        for value in args:
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                index = torch.tensor(batch_indices, device=value.device)
                value = value.index_select(0, index)
            sliced.append(value)
        return tuple(sliced)

    def _slice_context(
        self,
        values: Tensor | None,
        chunk_ranges: list[tuple[int, int]],
        batch_indices: list[int],
    ) -> Tensor | None:
        if values is None:
            return None
        selected = [
            values[start:end]
            for idx in batch_indices
            for start, end in [chunk_ranges[idx]]
        ]
        return torch.cat(selected, dim=0)

    @torch.inference_mode()
    def generate(self, *args: Any, **kwargs: Any):
        ctx_ids = kwargs.pop("ctx_ids", None)
        ctx_attn_mask = kwargs.pop("ctx_attn_mask", None)
        ctx_position_ids = kwargs.pop("ctx_position_ids", None)
        n_ctx_chunks = kwargs.pop("n_ctx_chunks", None)
        n_queries = kwargs.pop("n_queries", None)
        scalers = kwargs.pop("scalers", None)
        bias_scaler = kwargs.pop("bias_scaler", None)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return self.model.generate(
                *args,
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
                n_ctx_chunks=n_ctx_chunks,
                n_queries=n_queries,
                scalers=scalers,
                bias_scaler=bias_scaler,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        gate_result = self._score_batch(input_ids, ctx_ids, n_ctx_chunks)
        if ctx_ids is None:
            if hasattr(self.model, "reset"):
                self.model.reset()
            return self.model.base_model.generate(*args, **kwargs)

        chunks_per_sample = (
            n_ctx_chunks.detach().cpu().tolist()
            if n_ctx_chunks is not None
            else [1 for _ in range(batch_size)]
        )

        chunk_ranges = []
        offset = 0
        for n_chunks in chunks_per_sample:
            n_chunks = int(n_chunks)
            chunk_ranges.append((offset, offset + n_chunks))
            offset += n_chunks

        lora_indices = [
            idx for idx, use_lora in enumerate(gate_result.use_lora) if use_lora
        ]
        base_indices = [
            idx for idx, use_lora in enumerate(gate_result.use_lora) if not use_lora
        ]
        outputs_by_index: dict[int, Tensor] = {}

        if base_indices:
            if hasattr(self.model, "reset"):
                self.model.reset()
            base_args = self._slice_model_args(args, base_indices, batch_size)
            base_kwargs = self._slice_model_kwargs(kwargs, base_indices, batch_size)
            base_outputs = self.model.base_model.generate(*base_args, **base_kwargs)
            for idx, output in zip(base_indices, base_outputs):
                outputs_by_index[idx] = output

        if lora_indices:
            if hasattr(self.model, "patch_lora_forward"):
                self.model.patch_lora_forward()
            lora_args = self._slice_model_args(args, lora_indices, batch_size)
            lora_kwargs = self._slice_model_kwargs(kwargs, lora_indices, batch_size)
            lora_kwargs["ctx_ids"] = self._slice_context(
                ctx_ids, chunk_ranges, lora_indices
            )
            lora_kwargs["ctx_attn_mask"] = self._slice_context(
                ctx_attn_mask, chunk_ranges, lora_indices
            )
            if ctx_position_ids is not None:
                lora_kwargs["ctx_position_ids"] = self._slice_context(
                    ctx_position_ids, chunk_ranges, lora_indices
                )
            lora_kwargs["n_ctx_chunks"] = torch.tensor(
                [chunks_per_sample[idx] for idx in lora_indices],
                dtype=torch.int32,
                device=input_ids.device,
            )
            if n_queries is not None and n_queries.shape[0] == batch_size:
                index = torch.tensor(lora_indices, device=n_queries.device)
                lora_kwargs["n_queries"] = n_queries.index_select(0, index)
            if scalers is not None:
                lora_kwargs["scalers"] = self._slice_context(
                    scalers, chunk_ranges, lora_indices
                )
            if bias_scaler is not None:
                lora_kwargs["bias_scaler"] = bias_scaler

            lora_outputs = self.model.generate(*lora_args, **lora_kwargs)
            for idx, output in zip(lora_indices, lora_outputs):
                outputs_by_index[idx] = output

        outputs = [outputs_by_index[idx] for idx in range(batch_size)]
        max_len = max(out.shape[-1] for out in outputs)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        padded = []
        for out in outputs:
            out = out.unsqueeze(0)
            pad_len = max_len - out.shape[-1]
            if pad_len > 0:
                pad = torch.full(
                    (out.shape[0], pad_len),
                    pad_token_id,
                    dtype=out.dtype,
                    device=out.device,
                )
                out = torch.cat([out, pad], dim=-1)
            padded.append(out)
        return torch.cat(padded, dim=0)

    def print_gate_stats(self) -> None:
        if not self.gate_scores:
            print("Relevance gate: no scored samples")
            return
        total = len(self.gate_scores)
        used = sum(self.gate_decisions)
        mean_score = math.fsum(self.gate_scores) / total
        print(
            "Relevance gate: "
            f"backend={self.gate.__class__.__name__}, "
            f"threshold={self.threshold}, "
            f"use_lora={used}/{total} ({used / total:.2%}), "
            f"mean_score={mean_score:.4f}"
        )

    def reset_gate_stats(self) -> None:
        self.gate_scores.clear()
        self.gate_decisions.clear()
