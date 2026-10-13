#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
POSE_CONFIG="${POSE_CONFIG:-$ROOT_DIR/config/config_v1.yaml}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" -m idcard_detector train --config "$POSE_CONFIG" "$@"
