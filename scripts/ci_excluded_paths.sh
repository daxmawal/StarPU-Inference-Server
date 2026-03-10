#!/usr/bin/env bash
set -euo pipefail

# Print newline-separated directory exclusions for CI checks.
# Includes the default vendor directory and all git submodule paths.

repo_root="${1:-.}"
cd "$repo_root"

declare -a excludes=("external")

if [[ -f .gitmodules ]]; then
  while read -r _ path; do
    if [[ -n "${path:-}" ]]; then
      excludes+=("$path")
    fi
  done < <(git config --file .gitmodules --get-regexp path || true)
fi

printf '%s\n' "${excludes[@]}" | awk 'NF' | sort -u
