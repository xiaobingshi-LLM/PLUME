#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="${1:-squad}"
ROUTER="${2:-lexical}"
python scripts/evaluate.py \
  --dataset "${DATASET}" \
  --router "${ROUTER}" \
  --model models/gemma-2-2b-it \
  --d2l-checkpoint checkpoints/gemma_2b_d2l/checkpoint-20000 \
  --output "outputs/gemma/${DATASET}_${ROUTER}.jsonl"
