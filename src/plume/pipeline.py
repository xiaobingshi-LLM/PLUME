from __future__ import annotations

from .adapters import global_update
from .config import PlumeConfig
from .decoding import adaptive_generate
from .memory import activate_memory, segment_memory


class Plume:
    def __init__(self, backend, config: PlumeConfig) -> None:
        config.validate()
        self.backend, self.config = backend, config

    def prepare_document(self, old_context: str, full_context: str) -> dict:
        old = self.backend.parameterize(old_context, self.config.max_context_chunk_tokens)
        full = self.backend.parameterize(full_context, self.config.max_context_chunk_tokens)
        units = segment_memory(full_context, self.backend.ctx_tokenizer,
                               self.config.max_memory_unit_tokens)
        embeddings = None
        if self.config.router == "hybrid":
            embeddings = self.backend.embed([unit.text for unit in units],
                                            self.config.max_memory_unit_tokens)
        return {"global": global_update(full, old, self.config.alpha, self.config.beta),
                "units": units, "embeddings": embeddings}

    def answer(self, prompt: str, prepared: dict) -> tuple[str, dict]:
        query_embedding = None
        if self.config.router == "hybrid":
            query_embedding = self.backend.embed([prompt], self.config.max_memory_unit_tokens)[0]
        activation = activate_memory(
            prompt, prepared["units"], k1=self.config.lexical_k1,
            recency_margin=self.config.recency_margin, mode=self.config.router,
            embeddings=prepared["embeddings"], query_embedding=query_embedding,
            embedding_weight=self.config.hybrid_embedding_weight,
            top_k=self.config.hybrid_top_k,
        )
        evidence = self.backend.parameterize(activation.text, self.config.max_context_chunk_tokens)
        prompt_ids = self.backend.encode_prompt(prompt)
        output_ids, diagnostics = adaptive_generate(
            self.backend, prompt_ids, prepared["global"], evidence,
            max_new_tokens=self.config.max_new_tokens,
            lambda_max=self.config.lambda_max, tau=self.config.tau,
        )
        answer = self.backend.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        diagnostics.update({"memory_indices": [unit.index for unit in activation.units],
                            "memory_text": activation.text,
                            "lexical_scores": list(activation.lexical_scores),
                            "router_scores": list(activation.combined_scores)})
        return answer, diagnostics
