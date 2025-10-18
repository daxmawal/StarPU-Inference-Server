#!/bin/bash

# Run cppcheck on the whole project using compile_commands.json
# No per-file loop — only a single call with --project

BUILD_DIR=${BUILD_DIR:-../build}
LIBTORCH_DIR=${LIBTORCH_DIR:-$1}
GRPC_DIR=${GRPC_DIR:-$2}

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found." >&2
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?" >&2
  exit 1
fi

if [[ -z "$LIBTORCH_DIR" ]] || [[ -z "$GRPC_DIR" ]]; then
  echo "Usage: LIBTORCH_DIR=<path> GRPC_DIR=<path> $0"
  echo "   or: $0 <libtorch_dir> <grpc_dir>"
  exit 1
fi

INCLUDE_DIRS=(
  -I"$LIBTORCH_DIR/include"
  -I"$LIBTORCH_DIR/include/torch/csrc/api/include"
  -I"$GRPC_DIR/include"
)

CPPFLAGS=()
for dir in "${INCLUDE_DIRS[@]}"; do
  CPPFLAGS+=("-I$dir")
done

echo "====> Running cppcheck on the whole project..."

cppcheck \
  --enable=all \
  --inconclusive \
  --std=c++23 \
  --project="$BUILD_DIR/compile_commands.json" \
  "${CPPFLAGS[@]}"

echo "====> cppcheck finished."
