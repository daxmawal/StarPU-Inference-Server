#!/bin/bash
set -euo pipefail

pattern="^(feat|fix|docs|style|refactor|test|chore|perf|build|ci|revert): .+"
message=$(head -n1 "$1")

if ! echo "$message" | grep -Eq "$pattern"; then
  echo "X Commit message must follow 'type: message' format (e.g. 'fix: correct tensor lifetime')"
  exit 1
fi
