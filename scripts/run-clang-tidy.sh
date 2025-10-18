#!/bin/bash

# Run clang-tidy using the project's compile_commands database.
#
# Requires jq for parsing compile_commands.json.
#
# Required environment variables (or positional arguments):
#   LIBTORCH_DIR - path to the LibTorch installation
#   GRPC_DIR     - path to the gRPC installation
#
# Optionally, LIBTORCH_DIR and GRPC_DIR can be provided as the first and
# second positional arguments of this script. BUILD_DIR may be overridden via
# the BUILD_DIR environment variable (defaults to '../build').
#
# Additional options:
#   --file <path>  - analyze only the specified file
#   --dir  <path>  - analyze files found under the specified directory

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is not installed or not in PATH." >&2
  echo "Install it with: apt-get install jq" >&2
  exit 1
fi

BUILD_DIR=${BUILD_DIR:-../build}

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found." >&2
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?" >&2
  exit 1
fi

LIBTORCH_DIR=${LIBTORCH_DIR:-}
GRPC_DIR=${GRPC_DIR:-}
TARGET_FILE=""
TARGET_DIR=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      TARGET_FILE="$2"
      shift 2
      ;;
    --dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ -z "$LIBTORCH_DIR" ]]; then
  LIBTORCH_DIR="$1"
  shift
fi
if [[ -z "$GRPC_DIR" ]]; then
  GRPC_DIR="$1"
  shift
fi

if [[ -z "$LIBTORCH_DIR" ] || [ -z "$GRPC_DIR" ]]; then
  echo "Usage: LIBTORCH_DIR=<path> GRPC_DIR=<path> $0 [--file <path/to/file.cpp>] [--dir <path>]"
  echo "   or: $0 <libtorch_dir> <grpc_dir> [--file <path/to/file.cpp>] [--dir <path>]"
  exit 1
fi

CLANG_TIDY_ARGS=(
  -p "$BUILD_DIR"

  -checks=performance-*,modernize-*,bugprone-*,readability-*,clang-analyzer-*,cppcoreguidelines-*,portability-*,clang-diagnostic-unused-includes

  #-header-filter=.*

  # Ensure clang uses the correct host target and SIMD features
  -extra-arg-before=--target=x86_64-pc-linux-gnu
  -extra-arg-before=--gcc-toolchain=/usr
  -extra-arg=-stdlib=libstdc++
  -extra-arg-before=-msse

  -extra-arg=-std=c++23
  -extra-arg=-isystem/usr/include/c++/13
  -extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13

  -extra-arg=-I"$LIBTORCH_DIR/include"
  -extra-arg=-I"$LIBTORCH_DIR/include/torch/csrc/api/include"
  -extra-arg=-I"$GRPC_DIR/include"
)

if [[ -n "$TARGET_FILE" ]]; then
  FILES="$TARGET_FILE"
else
  if [[ -n "$TARGET_DIR" ]]; then
    FILTER_DIR=$(realpath "$TARGET_DIR")
    FILES=$(jq -r --arg dir "$FILTER_DIR" '.[] | .file | select(startswith($dir))' \
      "$BUILD_DIR/compile_commands.json" | sort -u | \
      grep -vE '\.pb\.cc$|\.grpc\.pb\.cc$' | \
      grep -v '/_deps/')
  else
    FILES=$(jq -r '.[].file' "$BUILD_DIR/compile_commands.json" | sort -u | \
      grep -vE '\.pb\.cc$|\.grpc\.pb\.cc$' | \
      grep -v '/_deps/')
  fi
fi

for file in $FILES; do
  echo "====> Analyzing $file"
  clang-tidy "$file" "${CLANG_TIDY_ARGS[@]}"
  echo
done
