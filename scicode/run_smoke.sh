#!/usr/bin/env bash
# Quick smoke: 1 main problem, AA-style background, against local vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIMIT="${LIMIT:-1}"
export MAX_CONNECTIONS="${MAX_CONNECTIONS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/eval/inspect_ai/outputs/smoke}"
export SPLIT="${SPLIT:-validation}"
exec "$ROOT/run_aa_eval.sh"
