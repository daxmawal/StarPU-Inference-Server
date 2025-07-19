#!/bin/bash

BUILD_DIR=../build

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found."
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?"
  exit 1
fi

CLANG_TIDY_ARGS=(
  -p "$BUILD_DIR"

  -checks=performance-*,modernize-*,bugprone-*,readability-*,clang-analyzer-*,cppcoreguidelines-*,portability-*

  -extra-arg=-std=c++23
  -extra-arg=-isystem/usr/include/c++/13
  -extra-arg=-isystem/usr/include/x86_64-linux-gnu/c++/13
  -extra-arg=-isystem/usr/lib/gcc/x86_64-linux-gnu/13/include

  -extra-arg=-I/local/home/jd258565/Install/libtorch/include
  -extra-arg=-I/local/home/jd258565/Install/libtorch/include/torch/csrc/api/include
  -extra-arg=-I/local/home/jd258565/Install/grpc/include
)

jq -r '.[].file' "$BUILD_DIR/compile_commands.json" | sort -u | \
  grep -vE '\.pb\.cc$|\.grpc\.pb\.cc$' | \
  while read -r file; do
    echo "====> Analyzing $file"
    clang-tidy "$file" "${CLANG_TIDY_ARGS[@]}"
    echo
done