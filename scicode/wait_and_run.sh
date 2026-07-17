#!/usr/bin/env bash
# Wait until vLLM answers, then start AA-aligned SciCode eval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "$ROOT/.env" && set +a
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
LOG="${LOG:-$ROOT/eval/inspect_ai/outputs/auto_run.log}"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] waiting for model at ${OPENAI_BASE_URL%/}/models ..."
until curl -sf --max-time 5 "${OPENAI_BASE_URL%/}/models" >/dev/null; do
  sleep 15
  echo "[$(date -Is)] still waiting..."
done
echo "[$(date -Is)] model up; starting eval"
exec "$ROOT/run_aa_eval.sh"
