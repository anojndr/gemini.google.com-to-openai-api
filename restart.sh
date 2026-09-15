#!/usr/bin/env bash
# Restart gemini.google.com-to-openai-api: kill existing server processes,
# then start a fresh one in the background (nohup + server.log).
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/server.log"
PIDFILE="$DIR/server.pid"
PORT="${PORT:-28407}"

echo "==> Killing existing gemini.google.com-to-openai-api server processes..."
pkill -f "gemini\.google\.com-to-openai-api/\.venv/bin/python.*server\.py" 2>/dev/null || true
pkill -f "gemini\.google\.com-to-openai-api/server\.py" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi
# Wait for the port to free up (max ~10s).
for _ in $(seq 1 20); do
  if ! ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    break
  fi
  sleep 0.5
done

echo "==> Starting server in background (log: $LOG)..."
cd "$DIR"
: >> "$LOG"
# shellcheck disable=SC2086
nohup "$DIR/.venv/bin/python" "$DIR/server.py" >>"$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PIDFILE"
disown 2>/dev/null || true
echo "==> Sent startup (shell pid $NEW_PID) on port $PORT, waiting for /health..."

# Wait until the server answers /health (max ~120s for model init).
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "==> Healthy: $(curl -s "http://127.0.0.1:${PORT}/health")"
    echo ""
    echo "Copy-paste (base URL):"
    echo "  http://127.0.0.1:${PORT}/v1"
    echo ""
    echo "Copy-paste to follow logs:"
    echo "  tail -f \"$LOG\""
    exit 0
  fi
  # Bail early if the process died.
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "!! Server process $NEW_PID died during startup. Last log lines:"
    tail -n 30 "$LOG"
    exit 1
  fi
  sleep 2
done

echo "!! Timed out waiting for /health. Last log lines:"
tail -n 30 "$LOG"
exit 1
