#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  exit 2
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
  echo "Missing ${PROJECT_DIR}/.env; copy .env.example and fill in API keys." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.env"
set +a

export TAU2_DIR="${TAU2_DIR:-${PROJECT_DIR}/.tau2-bench}"
if [[ ! -x "${TAU2_DIR}/.venv/bin/python" ]]; then
  echo "Environment is missing. Run ./scripts/setup.sh first." >&2
  exit 2
fi

exec uv run --project "${TAU2_DIR}" --with pyyaml \
  python "${SCRIPT_DIR}/run_from_yaml.py" "${CONFIG_PATH}"

