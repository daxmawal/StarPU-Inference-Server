#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
cd "$repo_root"

fail=0

declare -a test_only_name_patterns=(
  "*test_api*"
  "*test_accessor*"
  "*test_hooks*"
  "*test_forwarders*"
  "*test_overrides*"
)

find_args=()
for pattern in "${test_only_name_patterns[@]}"; do
  if ((${#find_args[@]} > 0)); then
    find_args+=(-o)
  fi
  find_args+=(-name "$pattern")
done

mapfile -t leaked_test_helpers < <(find src -type f \( "${find_args[@]}" \) | sort)
if ((${#leaked_test_helpers[@]} > 0)); then
  echo "::error::Test-only helper files must live under tests/support, not src/." >&2
  printf '  - %s\n' "${leaked_test_helpers[@]}"
  fail=1
fi

collect_support_include_files() {
  if command -v rg >/dev/null 2>&1; then
    rg -l '^\s*#\s*include\s+"support/' src --glob '*.[ch]pp' --glob '*.h' || true
    return
  fi

  while IFS= read -r -d '' file; do
    if grep -Eq '^[[:space:]]*#[[:space:]]*include[[:space:]]+"support/' "$file"; then
      printf '%s\n' "$file"
    fi
  done < <(
    find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -print0
  )
}

mapfile -t support_include_files < <(collect_support_include_files)

expected_support_include_files=(
  "src/grpc/server/server_main.cpp"
  "src/monitoring/metrics.cpp"
  "src/utils/batching_trace_logger.cpp"
)

tmp_expected="$(mktemp)"
tmp_actual="$(mktemp)"
trap 'rm -f "$tmp_expected" "$tmp_actual"' EXIT

if ((${#expected_support_include_files[@]} > 0)); then
  printf '%s\n' "${expected_support_include_files[@]}" | sort > "$tmp_expected"
else
  : > "$tmp_expected"
fi

if ((${#support_include_files[@]} > 0)); then
  printf '%s\n' "${support_include_files[@]}" | sort > "$tmp_actual"
else
  : > "$tmp_actual"
fi

if ! diff -u "$tmp_expected" "$tmp_actual" > /dev/null; then
  echo "::error::Unexpected src/ files include test support headers. Keep includes constrained and update this allowlist intentionally." >&2
  diff -u "$tmp_expected" "$tmp_actual" || true
  fail=1
fi

if ((${#support_include_files[@]} > 0)) && ! python3 - "${support_include_files[@]}" <<'PY'
import pathlib
import re
import sys

DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
SUPPORT_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"support/')
DEFINED_TRUE_RE = re.compile(r"^defined\s*(?:\(\s*STARPU_TESTING\s*\)|\s+STARPU_TESTING)$")
DEFINED_FALSE_RE = re.compile(r"^!\s*defined\s*(?:\(\s*STARPU_TESTING\s*\)|\s+STARPU_TESTING)$")


def normalize_directive_expr(expr: str) -> str:
    return expr.split("//", 1)[0].strip()


def classify_if_expr(expr: str) -> str:
    if DEFINED_TRUE_RE.fullmatch(expr):
        return "starpu_true"
    if DEFINED_FALSE_RE.fullmatch(expr):
        return "starpu_false"
    return "unknown"


had_error = False

for file_path in sys.argv[1:]:
    path = pathlib.Path(file_path)
    stack: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.rstrip()
        directive_match = DIRECTIVE_RE.match(line)
        if directive_match:
            directive, expr = directive_match.group(1), normalize_directive_expr(
                directive_match.group(2)
            )
            if directive == "if":
                stack.append(classify_if_expr(expr))
            elif directive == "ifdef":
                stack.append("starpu_true" if expr == "STARPU_TESTING" else "unknown")
            elif directive == "ifndef":
                stack.append("starpu_false" if expr == "STARPU_TESTING" else "unknown")
            elif directive == "else":
                if stack:
                    top = stack.pop()
                    if top == "starpu_true":
                        stack.append("starpu_false")
                    elif top == "starpu_false":
                        stack.append("starpu_true")
                    else:
                        stack.append("unknown")
            elif directive == "elif":
                if stack:
                    stack[-1] = "unknown"
            elif directive == "endif":
                if stack:
                    stack.pop()

        if SUPPORT_INCLUDE_RE.match(line):
            if "starpu_true" not in stack:
                print(
                    f"::error file={file_path},line={line_number}::"
                    "support/* include must be inside #if defined(STARPU_TESTING)."
                )
                had_error = True

if had_error:
    sys.exit(1)
PY
then
  fail=1
fi

if ((fail != 0)); then
  exit "$fail"
fi

echo "Structural guard passed."
