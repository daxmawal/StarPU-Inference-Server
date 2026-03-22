#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  run_perf_smoke.sh \
    --build-dir <dir> \
    --server <host:port> \
    --config <config-path> \
    --schedule-csv <path> \
    --output-dir <dir> \
    [--model <name>] \
    [--input <spec>]
EOF
  exit 2
}

BUILD_DIR=""
SERVER_ADDR=""
CONFIG_PATH=""
SCHEDULE_CSV=""
OUTPUT_DIR=""
MODEL_NAME="resnet152"
INPUT_SPEC="input0:1x3x224x224:float32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --server)
      SERVER_ADDR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --schedule-csv)
      SCHEDULE_CSV="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --input)
      INPUT_SPEC="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$BUILD_DIR" || -z "$SERVER_ADDR" || -z "$CONFIG_PATH" || -z "$SCHEDULE_CSV" || -z "$OUTPUT_DIR" ]]; then
  usage
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "Build directory not found: $BUILD_DIR" >&2
  exit 1
fi

pushd "$BUILD_DIR" >/dev/null

chmod +x starpu_server client_example || true
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

./starpu_server --config "$CONFIG_PATH" >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}

trap cleanup EXIT

READY=0
for attempt in $(seq 1 60); do
  if ./client_example \
    --server "$SERVER_ADDR" \
    --model "$MODEL_NAME" \
    --input "$INPUT_SPEC" \
    --request-number 1 \
    --summary-json "$OUTPUT_DIR/readiness-summary.json" \
    --verbose 0 \
    >"$OUTPUT_DIR/readiness.log" 2>&1; then
    READY=1
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before becoming ready" >&2
    cat "$OUTPUT_DIR/server.log" >&2
    exit 1
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "server did not become ready in time" >&2
  cat "$OUTPUT_DIR/server.log" >&2
  exit 1
fi

rm -f "$OUTPUT_DIR/readiness-summary.json"

./client_example \
  --server "$SERVER_ADDR" \
  --model "$MODEL_NAME" \
  --input "$INPUT_SPEC" \
  --schedule-csv "$SCHEDULE_CSV" \
  --summary-json "$OUTPUT_DIR/resnet152-summary.json" \
  --verbose 1 \
  >"$OUTPUT_DIR/client.log" 2>&1

popd >/dev/null
