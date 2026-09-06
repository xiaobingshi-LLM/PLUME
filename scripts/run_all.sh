#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL="${MODEL:-models/Qwen3-4B-Instruct-2507}"
D2L_CHECKPOINT="${D2L_CHECKPOINT:-checkpoints/qwen_4b_d2l/checkpoint-20000}"
ROUTER="${ROUTER:-lexical}"
for dataset in squad ropes 2wiki mfqa qasper gsm8k crux; do
  python scripts/evaluate.py \
    --dataset "${dataset}" \
    --router "${ROUTER}" \
    --model "${MODEL}" \
    --d2l-checkpoint "${D2L_CHECKPOINT}" \
    --output "outputs/all/${dataset}_${ROUTER}.jsonl"
done
