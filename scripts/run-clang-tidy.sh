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
  echo "Usage: LIBTORCH_DIR=<path> GRPC_DIR=<path> $0"
  echo "   or: $0 <libtorch_dir> <grpc_dir>"
  exit 1
fi

CLANG_TIDY_ARGS=(
  -p "$BUILD_DIR"

  -checks=performance-*,modernize-*,bugprone-*,readability-*,clang-analyzer-*,cppcoreguidelines-*,portability-*

  -extra-arg=-std=c++23
  -extra-arg=-isystem/usr/include/c++/13
  -extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13
  -extra-arg=-isystem/usr/lib/gcc/x86_64-linux-gnu/13/include

  -extra-arg=-I"$LIBTORCH_DIR/include"
  -extra-arg=-I"$LIBTORCH_DIR/include/torch/csrc/api/include"
  -extra-arg=-I"$GRPC_DIR/include"
)

jq -r '.[].file' "$BUILD_DIR/compile_commands.json" | sort -u | \
  grep -vE '\.pb\.cc$|\.grpc\.pb\.cc$' | \
  while read -r file; do
    echo "====> Analyzing $file"
    clang-tidy "$file" "${CLANG_TIDY_ARGS[@]}"
    echo
done
