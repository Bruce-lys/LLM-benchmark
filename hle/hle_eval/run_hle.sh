#!/usr/bin/env bash
# HLE evaluation entrypoint.
#
# Usage:
#   export OPENAI_API_KEY=EMPTY
#   export DEEPSEEK_API_KEY=sk-xxx
#   ./run_hle.sh configs/models/current.yaml
#   ./run_hle.sh configs/models/example.yaml --dry-run
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
cd "${ROOT_DIR}"

CONFIG="${1:-}"
if [[ -z "${CONFIG}" || "${CONFIG}" == -* ]]; then
  cat <<'EOF'
Usage: ./run_hle.sh <config.yaml|config.json> [extra args...]

Examples:
  ./run_hle.sh configs/models/current.yaml
  ./run_hle.sh configs/models/example.yaml --dry-run
  ./run_hle.sh configs/models/current.yaml --parallel 1 --only Ornith-0.9_Qwen3.6-0.1
  ./run_hle.sh configs/models/current.yaml --max_samples 3

Environment:
  OPENAI_API_KEY     model endpoint key (use EMPTY for SGLang)
  DEEPSEEK_API_KEY   realtime judge key
EOF
  exit 1
fi
shift || true

if [[ ! -f "${CONFIG}" ]]; then
  if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then
    CONFIG="${ROOT_DIR}/${CONFIG}"
  else
    echo "Config not found: ${CONFIG}" >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
  echo "[run_hle] OPENAI_API_KEY unset; using EMPTY for SGLang."
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[run_hle] WARNING: DEEPSEEK_API_KEY is unset; realtime judge will fail if enabled." >&2
fi

export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_hle] python=${PYTHON_BIN}"
echo "[run_hle] config=${CONFIG}"
echo "[run_hle] root=${ROOT_DIR}"

exec "${PYTHON_BIN}" "${SRC_DIR}/run_sglang_batch.py" \
  --config "${CONFIG}" \
  "$@"
