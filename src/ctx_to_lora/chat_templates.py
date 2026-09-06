"""Repository chat-template path resolution."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
CHAT_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "chat_templates"


def find_chat_template(model_name_or_path: str) -> Path | None:
    """Resolve a repository chat template for a hub name or local model path."""
    model_path = Path(model_name_or_path)
    normalized_name = model_name_or_path.strip("/")

    direct_path = CHAT_TEMPLATES_DIR / f"{normalized_name}.jinja"
    if direct_path.is_file():
        return direct_path

    model_basename = model_path.name
    candidates = sorted(CHAT_TEMPLATES_DIR.glob(f"*/{model_basename}.jinja"))
    if not candidates and "-" in model_basename:
        provider, model_name = model_basename.split("-", 1)
        provider_path = CHAT_TEMPLATES_DIR / provider / f"{model_name}.jinja"
        if provider_path.is_file():
            return provider_path
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Multiple chat templates match local model %s: %s",
            model_name_or_path,
            candidates,
        )
    return None
