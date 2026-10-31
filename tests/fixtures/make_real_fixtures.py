"""Capture REAL API output of a running backend into tests/fixtures/real/*.json (used by the frontend tests).

    HOST=127.0.0.1 python3 -m backend.server --port 8765 &        # in the project root
    python3 tests/fixtures/make_real_fixtures.py http://127.0.0.1:8765

Jobs (recommendations, backtests) are polled until ready (the first universe run takes minutes).  Network access is needed by the
server (Yahoo / TWSE), not by this script.

Audit notes (frontend fixer, 2026-10-30)
* recommendations_h{1,5,20}.json and backtest_universe_h5.json, backtest_2330_h20.json, backtest_0050_h20.json were captured from the
  FIXED backend (rank_signal / leaders / laggards / evidence / basket / benchmark_top_sigma ...).  The h = 5 payload carries a real
  ``evidence`` block: the server fills it from the cached universe validation of the same horizon, so run the universe validation
  FIRST (``/api/backtest/universe?h=5``) and fetch /api/recommendations afterwards; h = 1 and h = 20 were fetched without a cached
  validation and therefore have ``evidence = {"source": "none"}`` (a real fixture of both states).
* forecast_*, history_*, backtest_2330_h1/h5 and status are the earlier captures (2026-10-22): they stay valid under the additive
  contract and double as "old payload without the new keys" tests.  ``legacy_recommendations_h5.json`` / ``legacy_backtest_universe_h5.json``
  are the pre-fix payloads of the recommender / universe validation, kept on purpose to prove that the page still renders an old server.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "real"


def get(base, path, params=None, timeout=600):
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode("utf-8"))


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("saved", name, flush=True)


def poll(base, name, path, params, max_s=3600):
    t0 = time.time()
    while time.time() - t0 < max_s:
        js = get(base, path, params)
        if js.get("status") != "computing":
            save(name + ".json", js)
            return js
        print(name, js.get("progress"), flush=True)
        time.sleep(3)
    raise TimeoutError(name)


def main(base="http://127.0.0.1:8765"):
    save("status.json", get(base, "/api/status"))
    save("search.json", get(base, "/api/search", {"q": "台積"}))
    save("search_empty.json", get(base, "/api/search", {"limit": 20}))
    save("search_code.json", get(base, "/api/search", {"q": "0050"}))
    for sym in ("2330", "0050", "6488"):
        save(f"forecast_{sym}.json", get(base, "/api/forecast", {"symbol": sym}))
        save(f"history_{sym}.json", get(base, "/api/history", {"symbol": sym, "limit": 250}))
    save("error_unknown.json", get(base, "/api/forecast", {"symbol": "1234x"}))
    save("error_9999.json", get(base, "/api/forecast", {"symbol": "9999"}))
    for h in (5, 1, 20):
        poll(base, f"recommendations_h{h}", "/api/recommendations", {"h": h, "top": 10})
    for h in (1, 5, 20):
        poll(base, f"backtest_2330_h{h}", "/api/backtest", {"symbol": "2330", "h": h})
    poll(base, "backtest_0050_h20", "/api/backtest", {"symbol": "0050", "h": 20})
    poll(base, "backtest_universe_h5", "/api/backtest/universe", {"h": 5})
    poll(base, "recommendations_h5", "/api/recommendations", {"h": 5, "top": 10})      # again: now with the cached validation as evidence


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
