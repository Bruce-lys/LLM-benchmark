#!/usr/bin/env bash
# Download numeric gold targets (~1GB) required for scoring.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/eval/data/test_data.h5"
if [[ -f "$OUT" ]]; then
  echo "Already present: $OUT ($(du -h "$OUT" | cut -f1))"
  exit 0
fi
if [[ -x "$ROOT/.venv/bin/gdown" ]]; then
  GDOWN="$ROOT/.venv/bin/gdown"
elif command -v gdown >/dev/null 2>&1; then
  GDOWN=gdown
else
  echo "gdown not found; install with: uv pip install gdown" >&2
  exit 1
fi
mkdir -p /tmp/scicode_testdata "$ROOT/eval/data"
"$GDOWN" --folder 'https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR' -O /tmp/scicode_testdata
cp -f /tmp/scicode_testdata/test_data.h5 "$OUT"
echo "Saved $OUT"
