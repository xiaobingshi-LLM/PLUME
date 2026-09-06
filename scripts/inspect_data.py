#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plume.data import load_paired_documents


def main() -> None:
    total_documents = total_questions = 0
    for group in ("benchmarks", "generalization"):
        print(f"[{group}]")
        for directory in sorted((PROJECT_ROOT / "data" / group).iterdir()):
            if not directory.is_dir():
                continue
            documents = load_paired_documents(directory / "old.jsonl", directory / "full.jsonl")
            questions = sum(len(document.prompts) for document in documents)
            total_documents += len(documents)
            total_questions += questions
            print(f"{directory.name:8s} documents={len(documents):5d} questions={questions:6d}")
    print(f"total    documents={total_documents:5d} questions={total_questions:6d}")


if __name__ == "__main__":
    main()
