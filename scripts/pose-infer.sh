#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POSE_MODEL="${POSE_MODEL:?Set POSE_MODEL to a trained Pose checkpoint}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 IMAGE OUTPUT_DIR [additional infer arguments]" >&2
  exit 2
fi

image="$1"
output="$2"
shift 2

cd "$ROOT_DIR"

exec "$PYTHON_BIN" -m idcard_detector infer \
  --model "$POSE_MODEL" \
  --image "$image" \
  --output "$output" \
  "$@"
