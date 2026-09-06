from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora


def tokenize_context(text: str, tokenizer) -> list[int]:
    """Match D2L's chat-context preprocessing without dataset dependencies."""
    if not tokenizer.chat_template:
        raise ValueError("the D2L context tokenizer must define a chat template")
    tokenized = tokenizer.apply_chat_template(
        [[{"role": "system", "content": ""},
          {"role": "user", "content": text.strip()}]],
        tokenize=True, add_generation_prompt=True, return_attention_mask=False,
        padding=False, truncation=False, add_special_tokens=False, return_dict=True,
    )
    return tokenized["input_ids"][0]


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    direct = path / "pytorch_model.bin"
    if direct.is_file():
        return direct
    candidates = list(path.glob("checkpoint-*/pytorch_model.bin"))
    if not candidates:
        raise FileNotFoundError(f"no D2L checkpoint found under {path}")
    def step(candidate: Path) -> int:
        try:
            return int(candidate.parent.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1
    return max(candidates, key=step)


class D2LBackend:
    def __init__(self, model_path: str, checkpoint_path: str, device: str, dtype: torch.dtype,
                 iterative_mode: bool = False) -> None:
        state = torch.load(resolve_checkpoint(checkpoint_path), map_location="cpu", weights_only=False)
        state["base_model_name_or_path"] = model_path
        lora_config = state["hypernet_config"].lora_config
        lora_config.base_model_name_or_path = model_path
        if lora_config.r != 8:
            raise ValueError(f"PaperV5 requires rank-8 D2L adapters, checkpoint has rank {lora_config.r}")
        targets = set(lora_config.target_modules)
        if targets != {"down_proj"}:
            raise ValueError(f"PaperV5 requires target_modules={{'down_proj'}}, got {targets}")
        kwargs = {"device_map": device, "torch_dtype": dtype}
        self.model = ModulatedPretrainedModel.from_state_dict(
            state, train=False, use_sequence_packing=False, use_flash_attn=True,
            base_model_kwargs=kwargs, ctx_model_kwargs=kwargs,
        ).eval()
        self.model.enable_iterative_mode(iterative_mode)
        self.tokenizer = get_tokenizer(model_path)
        self.ctx_tokenizer = get_tokenizer(self.model.ctx_encoder.base_model.name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = torch.device(device)

    @torch.inference_mode()
    def parameterize(self, context: str, max_chunk_tokens: int = 8192) -> dict:
        tokenized = tokenize_context(context, self.ctx_tokenizer)
        count = max(1, math.ceil(len(tokenized) / max_chunk_tokens))
        width = max(1, math.ceil(len(tokenized) / count))
        chunks = [tokenized[i : i + width] for i in range(0, len(tokenized), width)] or [[]]
        ctx_device = next(self.model.ctx_encoder.parameters()).device
        ids = torch.full((len(chunks), max(map(len, chunks), default=1)),
                         self.ctx_tokenizer.pad_token_id or 0, dtype=torch.long, device=ctx_device)
        mask = torch.zeros_like(ids)
        for row, chunk in enumerate(chunks):
            if chunk:
                values = torch.tensor(chunk, dtype=torch.long, device=ctx_device)
                ids[row, : values.numel()] = values
                mask[row, : values.numel()] = 1
        generated, _ = self.model.generate_weights(ids, mask)
        return combine_lora(
            generated,
            torch.tensor([len(chunks)], dtype=torch.long, device=self.model.device),
            lora_bias=self.model.hypernet.get_head_bias() if self.model.hypernet.config.use_bias else None,
        )

    def install(self, adapter: dict) -> None:
        self.model.patch_lora_forward()
        apply_lora_to_layers(
            self.model.base_model, self.model.hypernet.layer_indices, adapter,
            torch.ones(1, dtype=torch.int32, device=self.model.device),
        )

    @torch.inference_mode()
    def embed(self, texts: Sequence[str], max_tokens: int = 512) -> Tensor:
        tokens = self.ctx_tokenizer(list(texts), padding=True, truncation=True,
                                    max_length=max_tokens, return_tensors="pt")
        encoder_device = next(self.model.ctx_encoder.parameters()).device
        tokens = {key: value.to(encoder_device) for key, value in tokens.items()}
        features = self.model.ctx_encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        if getattr(features, "hidden_states", None) is not None:
            features = torch.stack(features.hidden_states, dim=1).mean(dim=1)
        elif hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        elif isinstance(features, (tuple, list)):
            features = features[0]
        mask = tokens["attention_mask"].unsqueeze(-1).to(features.dtype)
        pooled = (features * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1)

    def encode_prompt(self, prompt: str) -> Tensor:
        if self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
            )
        else:
            ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return ids.to(self.device)
