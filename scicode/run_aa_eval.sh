#!/usr/bin/env bash
# Run SciCode with Artificial Analysis-aligned settings against a local
# OpenAI-compatible vLLM endpoint (default: http://127.0.0.1:8000/v1).
#
# AA methodology (https://artificialanalysis.ai/methodology/intelligence-benchmarking):
#   - split=test (288/291 subproblems; HF currently lists 291)
#   - scientist-annotated background ON
#   - subproblem-level pass@1 scoring
#   - temp 0.6 for reasoning models (lab may override; this vLLM was started with temp=1.0)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Prefer project venv if present
if [[ -x "$ROOT/.venv/bin/inspect" ]]; then
  INSPECT="$ROOT/.venv/bin/inspect"
  PYTHON="$ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  INSPECT="uv run inspect"
  PYTHON="uv run python"
else
  INSPECT="inspect"
  PYTHON="python"
fi

H5="${H5PY_FILE:-$ROOT/eval/data/test_data.h5}"
if [[ ! -f "$H5" ]]; then
  echo "Missing $H5 — download from Google Drive folder 1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" >&2
  exit 1
fi

# Discover raw model id from the OpenAI-compatible server unless overridden
discover_model() {
  if [[ -n "${SCICODE_MODEL_ID:-}" ]]; then
    echo "$SCICODE_MODEL_ID"
    return
  fi
  "$PYTHON" - <<'PY'
import json, os, urllib.request
base = os.environ["OPENAI_BASE_URL"].rstrip("/")
req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','EMPTY')}"})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)
ids = [m["id"] for m in data.get("data", [])]
if not ids:
    raise SystemExit("No models advertised by /v1/models")
print(ids[0])
PY
}

# Build inspect model spec. Prefer openai-api/vllm/<id> so path-like
# served model ids (e.g. /cpfs01/.../Ornith-1.0-35B) keep all slashes intact.
# Note: env exports must NOT live only inside $(...); command substitution
# runs in a subshell and would drop VLLM_* from the parent.
to_inspect_model() {
  local raw="$1"
  # strip accidental leading openai/ prefix from env overrides
  raw="${raw#openai/}"
  raw="${raw#openai-api/vllm/}"
  echo "openai-api/vllm/${raw}"
}

export VLLM_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export VLLM_BASE_URL="${OPENAI_BASE_URL}"

wait_for_model() {
  local tries="${1:-60}"
  local i
  for i in $(seq 1 "$tries"); do
    if curl -sf --max-time 5 "${OPENAI_BASE_URL%/}/models" >/dev/null 2>&1; then
      return 0
    fi
    echo "[wait] model not ready ($i/$tries) at $OPENAI_BASE_URL"
    sleep 5
  done
  return 1
}

LIMIT="${LIMIT:-}"          # e.g. LIMIT=1 for smoke
MAX_CONN="${MAX_CONNECTIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/eval/inspect_ai/outputs/ornith_aa}"
WITH_BG="${WITH_BACKGROUND:-True}"
SPLIT="${SPLIT:-test}"

echo "Waiting for model at $OPENAI_BASE_URL ..."
wait_for_model 120
RAW_MODEL_ID="$(discover_model)"
INSPECT_MODEL="$(to_inspect_model "$RAW_MODEL_ID")"
echo "Using served model id: $RAW_MODEL_ID"
echo "Using inspect model:   $INSPECT_MODEL"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT/eval/inspect_ai"

EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ -n "$TOP_P" ]]; then
  EXTRA+=(--top-p "$TOP_P")
fi

set -x
$INSPECT eval scicode.py \
  --model "$INSPECT_MODEL" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --max-connections "$MAX_CONN" \
  -T "split=${SPLIT}" \
  -T "with_background=${WITH_BG}" \
  -T "h5py_file=${H5}" \
  -T "output_dir=${OUTPUT_DIR}" \
  -T mode=normal \
  "${EXTRA[@]}"
