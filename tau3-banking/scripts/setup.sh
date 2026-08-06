#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TAU2_DIR="${TAU2_DIR:-${PROJECT_DIR}/.tau2-bench}"
TAU2_REPO="https://github.com/sierra-research/tau2-bench.git"
TAU2_REF="v1.0.1"
TAU2_COMMIT="fc0055dc4e0a316c3f83133267fbd6faaa770992"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ ! -d "${TAU2_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${TAU2_REF}" "${TAU2_REPO}" "${TAU2_DIR}"
fi

actual_commit="$(git -C "${TAU2_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${TAU2_COMMIT}" ]]; then
  echo "Unexpected tau2-bench commit: ${actual_commit}" >&2
  echo "Expected: ${TAU2_COMMIT}" >&2
  exit 1
fi

uv sync --project "${TAU2_DIR}" --extra knowledge
uv run --project "${TAU2_DIR}" tau2 check-data

