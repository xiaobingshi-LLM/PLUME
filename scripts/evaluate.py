#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plume.config import PlumeConfig
from plume.data import ALL_DATASETS, append_jsonl, dataset_directory, load_paired_documents
from plume.metrics import exact_match, rouge_l, token_f1
from plume.pipeline import Plume


def parse_dtype(value: str) -> torch.dtype:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if value not in values:
        raise argparse.ArgumentTypeError(f"dtype must be one of {tuple(values)}")
    return values[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PaperV5 PLUME method")
    parser.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    parser.add_argument("--model", required=True, help="Local or Hugging Face base-model identifier")
    parser.add_argument("--d2l-checkpoint", required=True, help="D2L .bin file or checkpoint directory")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--router", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--max-documents", type=int, default=-1)
    parser.add_argument("--max-questions-per-document", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterative-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--recency-margin", type=float, default=0.0)
    parser.add_argument("--max-context-chunk-tokens", type=int, default=8192)
    parser.add_argument("--max-memory-unit-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from plume.d2l import D2LBackend
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = PlumeConfig(
        alpha=args.alpha, beta=args.beta, lambda_max=args.lambda_max, tau=args.tau,
        recency_margin=args.recency_margin,
        max_context_chunk_tokens=args.max_context_chunk_tokens,
        max_memory_unit_tokens=args.max_memory_unit_tokens,
        max_new_tokens=args.max_new_tokens, router=args.router,
        iterative_mode=args.iterative_mode,
    )
    data_dir = dataset_directory(PROJECT_ROOT / "data", args.dataset)
    documents = load_paired_documents(data_dir / "old.jsonl", data_dir / "full.jsonl")
    if args.max_documents >= 0:
        documents = documents[: args.max_documents]
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    backend = D2LBackend(args.model, args.d2l_checkpoint, args.device, args.dtype,
                         iterative_mode=args.iterative_mode)
    method = Plume(backend, config)
    metric_rows = []
    for document in documents:
        prepared = method.prepare_document(document.old_context, document.full_context)
        count = len(document.prompts) if args.max_questions_per_document < 0 else min(
            len(document.prompts), args.max_questions_per_document)
        for question_id in range(count):
            prompt, reference = document.prompts[question_id], document.responses[question_id]
            prediction, diagnostics = method.answer(prompt, prepared)
            metrics = {"exact_match": exact_match(prediction, reference),
                       "token_f1": token_f1(prediction, reference),
                       **rouge_l(prediction, reference),
                       "is_locality_query": document.old_responses[question_id] == reference}
            metric_rows.append(metrics)
            append_jsonl(output, {
                "dataset": args.dataset, "document_id": document.document_id,
                "question_id": question_id, "prompt": prompt, "reference": reference,
                "prediction": prediction, "metrics": metrics, "diagnostics": diagnostics,
            })
            print(json.dumps({"document": document.document_id, "question": question_id,
                              "prediction": prediction, **metrics}, ensure_ascii=False), flush=True)
    summary = {
        "examples": len(metric_rows),
        "exact_match": sum(row["exact_match"] for row in metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "token_f1": sum(row["token_f1"] for row in metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "rouge_l_precision": sum(row["rouge_l_precision"] for row in metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "rouge_l_recall": sum(row["rouge_l_recall"] for row in metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "rouge_l_f1": sum(row["rouge_l_f1"] for row in metric_rows) / len(metric_rows) if metric_rows else 0.0,
        "locality": (sum(row["rouge_l_recall"] for row in metric_rows if row["is_locality_query"])
                     / sum(row["is_locality_query"] for row in metric_rows))
                    if any(row["is_locality_query"] for row in metric_rows) else None,
        "config": config.to_dict(),
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
