from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def js_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """Jensen-Shannon divergence for two log-probability vectors."""
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return 0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                  + (log_q.exp() * (log_q - log_m)).sum(-1))


@torch.inference_mode()
def adaptive_generate(backend, prompt_ids: Tensor, global_adapter: dict, evidence_adapter: dict,
                      *, max_new_tokens: int = 256, lambda_max: float = 1.0,
                      tau: float = 0.3) -> tuple[Tensor, dict]:
    """Algorithm 1: two parameter views, independent KV caches, adaptive fusion."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    base = backend.model.base_model
    attention = torch.ones_like(prompt_ids)
    backend.install(global_adapter)
    global_out = base(input_ids=prompt_ids, attention_mask=attention, use_cache=True)
    backend.install(evidence_adapter)
    evidence_out = base(input_ids=prompt_ids, attention_mask=attention, use_cache=True)
    global_past, evidence_past = global_out.past_key_values, evidence_out.past_key_values
    global_logits, evidence_logits = global_out.logits[:, -1], evidence_out.logits[:, -1]
    generated: list[Tensor] = []
    divergences, gates = [], []
    eos = backend.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    for _ in range(max_new_tokens):
        log_global = F.log_softmax(global_logits.float(), dim=-1)
        log_evidence = F.log_softmax(evidence_logits.float(), dim=-1)
        divergence = js_divergence(log_evidence, log_global)
        gate = lambda_max * divergence / (divergence + tau)
        token = (log_global + gate.unsqueeze(-1) * log_evidence).argmax(-1, keepdim=True)
        generated.append(token)
        divergences.append(float(divergence.item()))
        gates.append(float(gate.item()))
        if int(token.item()) in eos_ids:
            break
        attention = torch.cat((attention, torch.ones_like(token)), dim=1)
        backend.install(global_adapter)
        global_out = base(input_ids=token, attention_mask=attention,
                          past_key_values=global_past, use_cache=True)
        backend.install(evidence_adapter)
        evidence_out = base(input_ids=token, attention_mask=attention,
                            past_key_values=evidence_past, use_cache=True)
        global_past, evidence_past = global_out.past_key_values, evidence_out.past_key_values
        global_logits, evidence_logits = global_out.logits[:, -1], evidence_out.logits[:, -1]
    output = torch.cat(generated, dim=1) if generated else prompt_ids.new_empty((1, 0))
    diagnostics = {
        "tokens": output.shape[1],
        "mean_js": sum(divergences) / len(divergences) if divergences else 0.0,
        "mean_lambda": sum(gates) / len(gates) if gates else 0.0,
        "js": divergences,
        "lambda": gates,
    }
    return output, diagnostics
