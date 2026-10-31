"""Shared helpers for the quant-core tests: synthetic price data and an (optional, skippable) real-data loader."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.types import PriceData  # noqa: E402


def business_dates(n: int, start: str = "2005-01-03") -> np.ndarray:
    """``n`` consecutive Monday-Friday dates (datetime64[D])."""
    d = np.arange(np.datetime64(start), np.datetime64(start) + int(n * 1.5) + 10)
    # 1970-01-01 (day 0) was a Thursday, so Monday = 0 needs ``+ 3``.  (The helper used ``+ 4`` for a long time, which
    # produced Sunday-Thursday series: no Fridays and a Sunday every week -- harmless until the engine got weekday logic.)
    wd = (d.astype("datetime64[D]").astype(np.int64) + 3) % 7
    return d[wd < 5][:n]


def make_pricedata(ret: np.ndarray, symbol: str = "SYN", seed: int = 0, p0: float = 100.0, kind: str = "stock",
                   start: str = "2005-01-03", volume_level: float = 5e6) -> PriceData:
    """Price data whose adjusted close follows the given daily log returns (``len(ret)+1`` rows).

    Open / high / low / volume are random but consistent (low <= open, close <= high, volume log-normal).
    """
    rng = np.random.default_rng(seed + 12345)
    ret = np.asarray(ret, float)
    n = len(ret) + 1
    adj = p0 * np.exp(np.concatenate([[0.0], np.cumsum(ret)]))
    close = adj.copy()
    gap = 0.15 * np.concatenate([[0.0], ret]) * rng.standard_normal(n)
    open_ = np.concatenate([[p0], close[:-1]]) * np.exp(gap)
    sd = np.maximum(np.abs(np.concatenate([[0.0], ret])), 0.004)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.4 * sd)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.4 * sd)
    volume = np.round(volume_level * np.exp(0.4 * rng.standard_normal(n))).astype(float)
    return PriceData(symbol, symbol, "TWSE", kind, business_dates(n, start), open_, high, low, close, adj, volume)


def iid_returns(n: int, sd: float = 0.015, nu: float = 5.0, drift: float = 0.0, seed: int = 1) -> np.ndarray:
    """I.i.d. Student-t(nu) daily log returns with standard deviation ``sd`` and mean ``drift``."""
    rng = np.random.default_rng(seed)
    return drift + sd * rng.standard_t(nu, n) * np.sqrt((nu - 2.0) / nu)


def garch_returns(n: int, seed: int = 2, drift: float = 0.0003, v: float = 0.016 ** 2, alpha: float = 0.03,
                  gamma: float = 0.07, beta: float = 0.90, nu: float = 6.0) -> np.ndarray:
    """GJR-GARCH(1,1)-t returns with unconditional variance ``v``."""
    from backend.quant import garch as G
    phi = alpha + beta + gamma / 2
    p = G.GarchParams(omega=(1 - phi) * v, alpha=alpha, gamma=gamma, beta=beta, nu=nu)
    return drift + G.simulate(p, n, np.random.default_rng(seed))


def ar1_returns(n: int, rho: float = 0.25, sd: float = 0.015, seed: int = 3, drift: float = 0.0) -> np.ndarray:
    """Returns with a planted predictable component: r_t = drift + rho (r_{t-1} - drift) + e_t (rho > 0: momentum)."""
    rng = np.random.default_rng(seed)
    e = sd * np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    r = np.empty(n)
    r[0] = drift + e[0]
    for t in range(1, n):
        r[t] = drift + rho * (r[t - 1] - drift) + e[t]
    return r


# ----------------------------------------------------------------------------- real data (smoke tests only)
def _yahoo(yahoo_symbol: str):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
           f"?period1=0&period2={int(time.time())}&interval=1d&events=div%2Csplits")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def load_real_pricedata(symbol: str, yahoo_symbol: str | None = None, kind: str = "stock"):
    """Real daily bars (adjclose) or ``None`` when unavailable (offline).

    Cached for 24 h in ``$TWQ_TEST_CACHE`` (default: ``<system temp dir>/twq_quant_test_cache``).
    """
    yahoo_symbol = yahoo_symbol or f"{symbol}.TW"
    cache = os.environ.get("TWQ_TEST_CACHE") or os.path.join(tempfile.gettempdir(), "twq_quant_test_cache")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, symbol.replace("^", "") + ".json")
    try:
        if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400):
            js = _yahoo(yahoo_symbol)
            with open(path, "w") as f:
                json.dump(js, f)
        with open(path) as f:
            js = json.load(f)
        res = js["chart"]["result"][0]
        ts = np.array(res["timestamp"], dtype=np.int64) + res["meta"].get("gmtoffset", 28800)
        q = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        cols = {k: np.array([np.nan if v is None else v for v in q[k]], float)
                for k in ("open", "high", "low", "close", "volume")}
        cols["adjclose"] = np.array([np.nan if v is None else v for v in adj], float)
    except Exception:
        return None
    dates = (ts // 86400).astype("datetime64[D]")
    ok = np.ones(len(dates), bool)
    for k in ("open", "high", "low", "close", "adjclose"):
        ok &= np.isfinite(cols[k]) & (cols[k] > 0)
    keep = np.ones(len(dates), bool)
    keep[:-1] = dates[1:] != dates[:-1]
    ok &= keep
    vol = np.nan_to_num(cols["volume"], nan=0.0)
    return PriceData(symbol, symbol, "INDEX" if kind == "index" else "TWSE", kind, dates[ok], cols["open"][ok],
                     cols["high"][ok], cols["low"][ok], cols["close"][ok], cols["adjclose"][ok], vol[ok])
