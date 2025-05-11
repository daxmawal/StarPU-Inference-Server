#!/bin/bash

HOOK_SRC_DIR="hooks"
HOOK_DST_DIR=".git/hooks"

if [ ! -d "$HOOK_SRC_DIR" ]; then
    echo "Hook source directory not found: $HOOK_SRC_DIR"
    exit 1
fi

for hook in "$HOOK_SRC_DIR"/*; do
    hook_name=$(basename "$hook")
    cp "$hook" "$HOOK_DST_DIR/$hook_name"
    chmod +x "$HOOK_DST_DIR/$hook_name"
    echo "Installed $hook_name hook."
done

echo "All git hooks installed successfully."
