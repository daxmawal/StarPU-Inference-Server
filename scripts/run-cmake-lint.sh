#!/bin/bash

# Run cmake-lint on all CMakeLists.txt and *.cmake files.
# Usage: ./scripts/run-cmake-lint.sh [directory]
# The directory defaults to the current directory.

TARGET_DIR="${1:-.}"

if ! command -v cmake-lint >/dev/null 2>&1; then
  echo "Error: cmake-lint is not installed or not in PATH." >&2
  echo "Install it with: pip install cmakelang" >&2
  exit 1
fi

FILES=$(find "$TARGET_DIR" -name 'CMakeLists.txt' -o -name '*.cmake')

fail=0
for file in $FILES; do
  echo "====> Linting $file"
  if ! cmake-lint "$file"; then
    fail=1
  fi
  echo
done

exit $fail