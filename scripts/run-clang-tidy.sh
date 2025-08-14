#!/bin/bash

# Run clang-tidy using the project's compile_commands database.
#
# Required environment variables (or positional arguments):
#   LIBTORCH_DIR - path to the LibTorch installation
#   GRPC_DIR     - path to the gRPC installation
#
# Optionally, LIBTORCH_DIR and GRPC_DIR can be provided as the first and
# second positional arguments of this script. BUILD_DIR may be overridden via
# the BUILD_DIR environment variable (defaults to '../build').

BUILD_DIR=${BUILD_DIR:-../build}

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found."
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?"
  exit 1
fi

LIBTORCH_DIR=${LIBTORCH_DIR:-$1}
GRPC_DIR=${GRPC_DIR:-$2}

if [ -z "$LIBTORCH_DIR" ] || [ -z "$GRPC_DIR" ]; then
  echo "Usage: LIBTORCH_DIR=<path> GRPC_DIR=<path> $0 [--file <path/to/file.cpp>]"
  echo "   or: $0 <libtorch_dir> <grpc_dir> [--file <path/to/file.cpp>]"
  exit 1
fi

TARGET_FILE=""
if [[ "$3" == "--file" ]]; then
  TARGET_FILE="$4"
elif [[ "$1" == "--file" ]]; then
  TARGET_FILE="$2"
fi

CLANG_TIDY_ARGS=(
  -p "$BUILD_DIR"

  -checks=performance-*,modernize-*,bugprone-*,readability-*,clang-analyzer-*,cppcoreguidelines-*,portability-*,clang-diagnostic-unused-includes

  -extra-arg=-std=c++23
  -extra-arg=-isystem/usr/include/c++/13
  -extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13
  -extra-arg=-isystem/usr/lib/gcc/x86_64-linux-gnu/13/include

  -extra-arg=-I"$LIBTORCH_DIR/include"
  -extra-arg=-I"$LIBTORCH_DIR/include/torch/csrc/api/include"
  -extra-arg=-I"$GRPC_DIR/include"
)

if [ -n "$TARGET_FILE" ]; then
  FILES="$TARGET_FILE"
else
  FILES=$(jq -r '.[].file' "$BUILD_DIR/compile_commands.json" | sort -u | \
    grep -vE '\.pb\.cc$|\.grpc\.pb\.cc$' | \
    grep -v '/_deps/')
fi

for file in $FILES; do
  echo "====> Analyzing $file"
  clang-tidy "$file" "${CLANG_TIDY_ARGS[@]}"
  echo
done
