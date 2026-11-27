#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
export YOLO26_NPU_CORE="${YOLO26_NPU_CORE:-0}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python venv not found at $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Error: cloudflared not found in PATH" >&2
  exit 1
fi

cleanup() {
  echo ""
  echo "Shutting down server and Cloudflare tunnel..."
  kill "${UVICORN_PID:-}" "${CF_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting YOLO26n-seg FastAPI server on http://${HOST}:${PORT} (NPU core furiosa:${YOLO26_NPU_CORE})..."
"$PYTHON_BIN" -m uvicorn web_server:app --host "$HOST" --port "$PORT" &
UVICORN_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "Starting named Cloudflare Tunnel..."
  cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
  CF_PID=$!
else
  echo "Starting Cloudflare Quick Tunnel (HTTPS)..."
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" &
  CF_PID=$!
fi

wait "$UVICORN_PID"
