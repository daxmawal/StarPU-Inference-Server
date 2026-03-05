#!/bin/bash

# Run clang-tidy using the project's compile_commands database.
#
# Requires jq for parsing compile_commands.json.
#
# Dependency paths:
#   LIBTORCH_DIR - path to the LibTorch installation
#   GRPC_DIR     - path to the gRPC installation
#
# Dependency paths can be supplied via environment variables, positional
# arguments, or auto-detected from BUILD_DIR/CMakeCache.txt.
# BUILD_DIR may be overridden via the BUILD_DIR environment variable
# (defaults to '<repo>/build').
#
# Additional options:
#   --file <path>  - analyze only the specified file
#   --dir  <path>  - analyze files found under the specified directory
#   --header-filter <regex> - limit which headers receive diagnostics (default: project headers outside of build/)

set -uo pipefail

usage() {
  cat <<EOF
Usage:
  LIBTORCH_DIR=<path> GRPC_DIR=<path> $0 [--file <path/to/file.cpp>] [--dir <path>] [--header-filter <regex>]
  $0 <libtorch_dir> <grpc_dir> [--file <path/to/file.cpp>] [--dir <path>] [--header-filter <regex>]
  $0 [--file <path/to/file.cpp>] [--dir <path>] [--header-filter <regex>]   # auto-detect from build/CMakeCache.txt

Options:
  --file <path>           Analyze only the specified file
  --dir <path>            Analyze only files under this directory
  --header-filter <regex> Header filter regex for clang-tidy
  -h, --help              Show this help
EOF
  return 0
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is not installed or not in PATH." >&2
    exit 1
  fi
  return 0
}

require_command jq
require_command clang-tidy
require_command realpath

read_cmake_cache_value() {
  local key="$1"
  local cache_file="$2"
  awk -F= -v key="$key" '
    {
      split($1, parts, ":");
      if (parts[1] == key) {
        print substr($0, index($0, "=") + 1);
        exit;
      }
    }
  ' "$cache_file"
  return $?
}

resolve_libtorch_dir_from_cache() {
  local cache_file="$1"
  local torch_dir=""
  local caffe2_dir=""
  local c10_cuda_library=""
  local candidate=""

  torch_dir=$(read_cmake_cache_value "Torch_DIR" "$cache_file" || true)
  if [[ -n "$torch_dir" ]]; then
    candidate=$(realpath "$torch_dir/../../.." 2>/dev/null || true)
    if [[ -n "$candidate" ]] && [[ -d "$candidate/include" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  caffe2_dir=$(read_cmake_cache_value "Caffe2_DIR" "$cache_file" || true)
  if [[ -n "$caffe2_dir" ]]; then
    candidate=$(realpath "$caffe2_dir/../../.." 2>/dev/null || true)
    if [[ -n "$candidate" ]] && [[ -d "$candidate/include" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  c10_cuda_library=$(read_cmake_cache_value "C10_CUDA_LIBRARY" "$cache_file" || true)
  if [[ -n "$c10_cuda_library" ]]; then
    candidate=$(realpath "$(dirname "$c10_cuda_library")/.." 2>/dev/null || true)
    if [[ -n "$candidate" ]] && [[ -d "$candidate/include" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

resolve_grpc_dir_from_cache() {
  local cache_file="$1"
  local grpc_source_dir=""
  local grpc_dir=""
  local candidate=""

  grpc_source_dir=$(read_cmake_cache_value "grpc_SOURCE_DIR" "$cache_file" || true)
  if [[ -n "$grpc_source_dir" ]] && [[ -d "$grpc_source_dir/include" ]]; then
    realpath "$grpc_source_dir"
    return 0
  fi

  grpc_dir=$(read_cmake_cache_value "gRPC_DIR" "$cache_file" || true)
  if [[ -n "$grpc_dir" ]]; then
    candidate=$(realpath "$grpc_dir/../../.." 2>/dev/null || true)
    if [[ -n "$candidate" ]] && [[ -d "$candidate/include" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
BUILD_DIR=${BUILD_DIR:-"$PROJECT_ROOT/build"}

LIBTORCH_DIR=${LIBTORCH_DIR:-}
GRPC_DIR=${GRPC_DIR:-}
HEADER_FILTER=${HEADER_FILTER:-}
TARGET_FILE=""
TARGET_DIR=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
        echo "Error: --file requires a path argument." >&2
        usage
        exit 1
      fi
      TARGET_FILE="$2"
      shift 2
      ;;
    --dir)
      if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
        echo "Error: --dir requires a path argument." >&2
        usage
        exit 1
      fi
      TARGET_DIR="$2"
      shift 2
      ;;
    --header-filter)
      if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
        echo "Error: --header-filter requires a regex argument." >&2
        usage
        exit 1
      fi
      HEADER_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL+=("$1")
        shift
      done
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ -z "$LIBTORCH_DIR" ]] && [[ $# -gt 0 ]]; then
  LIBTORCH_DIR="$1"
  shift
fi
if [[ -z "$GRPC_DIR" ]] && [[ $# -gt 0 ]]; then
  GRPC_DIR="$1"
  shift
fi

if [[ -z "$LIBTORCH_DIR" ]] || [[ -z "$GRPC_DIR" ]]; then
  CMAKE_CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
  if [[ -f "$CMAKE_CACHE_FILE" ]]; then
    if [[ -z "$LIBTORCH_DIR" ]]; then
      LIBTORCH_DIR=$(resolve_libtorch_dir_from_cache "$CMAKE_CACHE_FILE" || true)
    fi
    if [[ -z "$GRPC_DIR" ]]; then
      GRPC_DIR=$(resolve_grpc_dir_from_cache "$CMAKE_CACHE_FILE" || true)
    fi
  fi
fi

if [[ -z "$LIBTORCH_DIR" ]] || [[ -z "$GRPC_DIR" ]]; then
  echo "Error: LIBTORCH_DIR and GRPC_DIR are required (env vars, positional args, or detectable in $BUILD_DIR/CMakeCache.txt)." >&2
  usage
  exit 1
fi

if [[ $# -gt 0 ]]; then
  echo "Error: unexpected positional arguments: $*" >&2
  usage
  exit 1
fi

if [[ -z "$HEADER_FILTER" ]]; then
  # Only include project headers in src/ and tests/, exclude build/ and external/
  HEADER_FILTER="^${PROJECT_ROOT}/(src|tests)/"
fi

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found." >&2
  echo "Did you run: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ?" >&2
  exit 1
fi

CLANG_TIDY_CONFIG="$BUILD_DIR/.clang-tidy"
CLANG_TIDY_CONFIG_BAK=""

cleanup_clang_tidy_config() {
  if [[ -n "$CLANG_TIDY_CONFIG_BAK" ]] && [[ -f "$CLANG_TIDY_CONFIG_BAK" ]]; then
    mv "$CLANG_TIDY_CONFIG_BAK" "$CLANG_TIDY_CONFIG"
  else
    rm -f "$CLANG_TIDY_CONFIG"
  fi
  return 0
}

write_clang_tidy_config() {
  local gcc_major=""
  local stdinc1=""
  local stdinc2=""

  if command -v g++ >/dev/null 2>&1; then
    gcc_major=$(g++ -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)
  fi
  if [[ -n "$gcc_major" ]]; then
    stdinc1="/usr/include/c++/$gcc_major"
    stdinc2="/usr/include/x86_64-linux-gnu/c++/$gcc_major"
  fi

  cat > "$CLANG_TIDY_CONFIG" <<EOF
---
Checks: "performance-*,modernize-*,bugprone-*,readability-*,clang-analyzer-*,cppcoreguidelines-*,portability-*,clang-diagnostic-unused-includes"
HeaderFilterRegex: "$HEADER_FILTER"
WarningsAsErrors: ""
ExtraArgs:
  - "--target=x86_64-pc-linux-gnu"
  - "--gcc-toolchain=/usr"
  - "-stdlib=libstdc++"
  - "-msse"
  - "-std=c++23"
  - "-I$LIBTORCH_DIR/include"
  - "-I$LIBTORCH_DIR/include/torch/csrc/api/include"
  - "-I$GRPC_DIR/include"
EOF
  if [[ -n "$stdinc1" ]] && [[ -d "$stdinc1" ]]; then
    echo "  - \"-isystem$stdinc1\"" >> "$CLANG_TIDY_CONFIG"
  fi
  if [[ -n "$stdinc2" ]] && [[ -d "$stdinc2" ]]; then
    echo "  - \"-isystem$stdinc2\"" >> "$CLANG_TIDY_CONFIG"
  fi
  return 0
}

trap cleanup_clang_tidy_config EXIT INT TERM

if [[ -f "$CLANG_TIDY_CONFIG" ]]; then
  CLANG_TIDY_CONFIG_BAK="$CLANG_TIDY_CONFIG.bak.$$"
  cp "$CLANG_TIDY_CONFIG" "$CLANG_TIDY_CONFIG_BAK"
fi

write_clang_tidy_config

CLANG_TIDY_ARGS=(
  -p "$BUILD_DIR"
  --config-file "$CLANG_TIDY_CONFIG"
)
if [[ -n "$HEADER_FILTER" ]]; then
  CLANG_TIDY_ARGS+=(
    --header-filter "$HEADER_FILTER"
  )
fi

if [[ -n "$TARGET_FILE" ]]; then
  FILES=("$TARGET_FILE")
else
  if [[ -n "$TARGET_DIR" ]]; then
    if [[ ! -d "$TARGET_DIR" ]]; then
      echo "Error: --dir path does not exist or is not a directory: $TARGET_DIR" >&2
      exit 1
    fi
    if ! FILTER_DIR=$(realpath "$TARGET_DIR" 2>/dev/null); then
      echo "Error: --dir path does not exist: $TARGET_DIR" >&2
      exit 1
    fi
    mapfile -t FILES < <(
      jq -r --arg dir "$FILTER_DIR" \
        '.[] | .file | select(startswith($dir)) | select(test("\\.pb\\.cc$|\\.grpc\\.pb\\.cc$") | not) | select(contains("/_deps/") | not)' \
        "$BUILD_DIR/compile_commands.json" | sort -u
    )
  else
    mapfile -t FILES < <(
      jq -r \
        '.[] | .file | select(test("\\.pb\\.cc$|\\.grpc\\.pb\\.cc$") | not) | select(contains("/_deps/") | not)' \
        "$BUILD_DIR/compile_commands.json" | sort -u
    )
  fi
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files matched the provided filters." >&2
  exit 1
fi

status=0
for file in "${FILES[@]}"; do
  if [[ ! -f "$CLANG_TIDY_CONFIG" ]]; then
    write_clang_tidy_config
  fi
  echo "====> Analyzing $file"
  if ! clang-tidy "$file" "${CLANG_TIDY_ARGS[@]}"; then
    status=1
  fi
  echo
done
exit "$status"
