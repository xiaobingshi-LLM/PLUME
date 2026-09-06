#!/usr/bin/env python
from __future__ import annotations

import importlib
import sys


REQUIRED = ("torch", "transformers", "peft", "einops", "jaxtyping", "rouge_score")


def main() -> None:
    failures = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"{name:16s} {version}")
        except Exception as error:
            failures.append((name, str(error)))
            print(f"{name:16s} ERROR: {error}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
