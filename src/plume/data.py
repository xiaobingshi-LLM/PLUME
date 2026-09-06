from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

BENCHMARK_DATASETS = ("squad", "ropes", "2wiki", "mfqa", "qasper")
GENERALIZATION_DATASETS = ("gsm8k", "crux")
ALL_DATASETS = BENCHMARK_DATASETS + GENERALIZATION_DATASETS


def dataset_directory(data_root: str | Path, dataset: str) -> Path:
    if dataset in BENCHMARK_DATASETS:
        group = "benchmarks"
    elif dataset in GENERALIZATION_DATASETS:
        group = "generalization"
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return Path(data_root) / group / dataset


@dataclass(frozen=True)
class Document:
    document_id: int
    old_context: str
    full_context: str
    prompts: tuple[str, ...]
    responses: tuple[str, ...]
    old_responses: tuple[str, ...]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_paired_documents(old_path: str | Path, full_path: str | Path) -> list[Document]:
    old_records, full_records = _read_jsonl(Path(old_path)), _read_jsonl(Path(full_path))
    if len(old_records) != len(full_records):
        raise ValueError("old and full files must contain the same number of documents")
    documents = []
    for index, (old, full) in enumerate(zip(old_records, full_records)):
        prompts, responses = full.get("prompts"), full.get("responses")
        if not isinstance(prompts, list) or not isinstance(responses, list) or len(prompts) != len(responses):
            raise ValueError(f"document {index} has invalid prompts/responses")
        if old.get("prompts") != prompts:
            raise ValueError(f"document {index} has mismatched old/full prompts")
        old_context, full_context = old.get("context", ""), full.get("context", "")
        if not old_context or not full_context:
            raise ValueError(f"document {index} has an empty context")
        old_responses = old.get("responses", responses)
        if not isinstance(old_responses, list) or len(old_responses) != len(responses):
            raise ValueError(f"document {index} has invalid old responses")
        documents.append(Document(index, old_context, full_context, tuple(prompts),
                                  tuple(responses), tuple(old_responses)))
    return documents


def iter_questions(documents: list[Document]) -> Iterator[tuple[Document, int, str, str]]:
    for document in documents:
        for question_id, (prompt, response) in enumerate(zip(document.prompts, document.responses)):
            yield document, question_id, prompt, response


def append_jsonl(path: str | Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
