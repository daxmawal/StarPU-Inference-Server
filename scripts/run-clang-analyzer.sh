#!/bin/bash

# Run the Clang Static Analyzer using scan-build.
#
# Requires a pre-configured build directory with compile_commands.json.
# BUILD_DIR may be overridden via the BUILD_DIR environment variable
# (defaults to '../build'). The output directory can be set with
# ANALYZER_OUTPUT (defaults to 'clang-analyzer').

BUILD_DIR=${BUILD_DIR:-../build}
ANALYZER_OUTPUT=${ANALYZER_OUTPUT:-clang-analyzer}

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found." >&2
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?" >&2
  exit 1
fi

mkdir -p "$ANALYZER_OUTPUT"

echo "====> Running Clang Static Analyzer..."
scan-build \
  --status-bugs \
  -o "$ANALYZER_OUTPUT" \
  cmake --build "$BUILD_DIR"

echo "====> Reports saved to $ANALYZER_OUTPUT"
