#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="${1:-squad}"
ROUTER="${2:-lexical}"
python scripts/evaluate.py \
  --dataset "${DATASET}" \
  --router "${ROUTER}" \
  --model models/Qwen3-4B-Instruct-2507 \
  --d2l-checkpoint checkpoints/qwen_4b_d2l/checkpoint-20000 \
  --output "outputs/qwen/${DATASET}_${ROUTER}.jsonl"
