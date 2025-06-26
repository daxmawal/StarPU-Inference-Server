#!/bin/bash
set -e

echo "Running build..."
# Simple build verification script for pre-commit
if [ -d build ]; then
    cmake --build build
else
    make -j$(nproc)
fi
echo "Build completed."
