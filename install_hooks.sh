#!/bin/bash

HOOK_SRC="hooks/pre-commit"
HOOK_DST=".git/hooks/pre-commit"

if [ ! -f "$HOOK_SRC" ]; then
    echo "Hook not found at $HOOK_SRC"
    exit 1
fi

cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
echo "Git pre-commit hook installed successfully."
