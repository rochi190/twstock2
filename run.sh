#!/usr/bin/env bash
# 台股量化實驗室 — 啟動腳本
#   ./run.sh                 # http://127.0.0.1:8000
#   PORT=8080 ./run.sh       # 換連接埠
#   HOST=0.0.0.0 ./run.sh    # 對區域網路開放（預設僅限本機）
#   ./run.sh --precompute    # 先預先計算快取（行情 + 預測），再啟動
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "找不到 Python 3（可用環境變數 PYTHON 指定）。" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "需要 Python 3.10 以上（建議 3.11）。目前：$("$PY" --version 2>&1)" >&2
  exit 1
fi
if ! "$PY" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "缺少相依套件，請先執行：$PY -m pip install -r requirements.txt" >&2
  exit 1
fi

if [ "${1:-}" = "--precompute" ]; then
  shift
  "$PY" -m backend.precompute "$@"   # 其餘參數交給預先計算程式（例如 --universe-bt），完成後照常啟動伺服器
  exec "$PY" -m backend.server
fi
exec "$PY" -m backend.server "$@"   # 例如 ./run.sh --port 8080
