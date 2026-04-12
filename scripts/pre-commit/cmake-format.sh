#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  exit 0
fi

if ! command -v cmake-format >/dev/null 2>&1; then
  echo "Error: cmake-format is not installed or not in PATH." >&2
  echo "Install it with: pip install cmakelang" >&2
  exit 1
fi

cmake-format -i "$@"
