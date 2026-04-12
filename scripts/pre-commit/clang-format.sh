#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
  exit 0
fi

if ! command -v clang-format >/dev/null 2>&1; then
  echo "Error: clang-format is not installed or not in PATH." >&2
  exit 1
fi

clang-format -i --style=file "$@"
