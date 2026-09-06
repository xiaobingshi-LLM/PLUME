#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
python -m unittest discover -s tests -v
python scripts/inspect_data.py
