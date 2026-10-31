"""Strict checkers for the DESIGN.md section 6 payloads (reusable by every module and by the manual end-to-end run).

Every ``check_*`` function takes the decoded JSON object of a **ready** response and returns a list of problem
strings (``[]`` means the payload honours the contract).  Nothing here raises on malformed input -- a missing key or a
wrong type is simply reported.  ``strict_loads`` decodes raw response bytes and rejects ``NaN`` / ``Infinity`` tokens.

    check_status(d)            section 6.0   /api/status
    check_search(d)            section 6.0   /api/search
    check_popular(d, codes)    section 6.0   /api/search with an empty q: the popular list is complete (12 + 3 ETFs)
    check_history(d)           section 6.0   /api/history
    check_forecast(d)          section 6.1   /api/forecast
    check_recommendations(d)   section 6.3 + 9   /api/recommendations (rank_signal ordering, leaders / laggards, lean,
                                                  evidence block, portfolio suppression)
    check_evidence(ev, h)      section 9.4   the ``evidence`` object of the picks payload
    check_backtest(d)          section 6.4   /api/backtest      (envelope + ``evaluation.report.validate_backtest_report``)
    check_universe_report(d)   section 6.5   /api/backtest/universe  (``d`` is the envelope *or* the ``report`` object)
    check_envelope(d)          section 6.2   job envelope (computing / ready / error)
    check_error(d)             error object  {"status":"error","error":{"code","message"}}
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import date, datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:  # so `from backend... import validate_backtest_report` works from any working directory
    sys.path.insert(0, _ROOT)

HORIZON_KEYS = ("1", "5", "20")
QKEYS = ("0.05", "0.25", "0.5", "0.75", "0.95")
FAN_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
COMPONENTS = ("garch", "hmm", "dlm", "naive")
ERROR_CODES = ("bad_request", "unknown_symbol", "insufficient_history", "data_unavailable", "internal")
SIGNALS = ("buy", "hold", "avoid")
CONFIDENCES = ("高", "中", "低")
VERDICT_LEVELS = ("no_edge", "weak_edge", "moderate_edge", "strong_edge")
STATS_KEYS = ("total_return", "cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "exposure",
              "turnover_per_year", "n_trades", "trade_hit_rate", "psr", "dsr", "sharpe_ci95")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ----------------------------------------------------------------------------------------------- primitives
def strict_loads(raw) -> object:
    """``json.loads`` that refuses NaN / Infinity / -Infinity (they are not JSON) and decodes UTF-8."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")

    def bad(tok):
        raise ValueError(f"non-finite JSON token {tok!r}")

    return json.loads(raw, parse_constant=bad)


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_date(v) -> bool:
    if not isinstance(v, str) or not _DATE_RE.match(v):
        return False
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def is_iso_datetime(v) -> bool:
    if not isinstance(v, str):
        return False
    try:
        datetime.fromisoformat(v)
        return True
    except ValueError:
        return False


class _P:
    """Problem collector with path-aware helpers."""

    def __init__(self):
        self.items: list[str] = []

    def add(self, path: str, msg: str) -> None:
        self.items.append(f"{path}: {msg}")

    def keys(self, d, path: str, required, optional=()) -> bool:
        if not isinstance(d, dict):
            self.add(path, f"expected object, got {type(d).__name__}")
            return False
        for k in required:
            if k not in d:
                self.add(path, f"missing key {k!r}")
        return True

    def num(self, d, key: str, path: str, lo=None, hi=None, nullable: bool = False) -> float | None:
        v = d.get(key) if isinstance(d, dict) else None
        here = f"{path}.{key}"
        if v is None:
            if not nullable:
                self.add(here, "missing or null")
            return None
        if not is_num(v):
            self.add(here, f"not a finite number: {v!r}")
            return None
        if lo is not None and v < lo:
            self.add(here, f"{v} < {lo}")
        if hi is not None and v > hi:
            self.add(here, f"{v} > {hi}")
        return float(v)

    def text(self, d, key: str, path: str, nonempty: bool = True) -> str | None:
        v = d.get(key) if isinstance(d, dict) else None
        if not isinstance(v, str) or (nonempty and not v.strip()):
            self.add(f"{path}.{key}", f"expected non-empty string, got {v!r}")
            return None
        return v

    def date(self, d, key: str, path: str, nullable: bool = False):
        v = d.get(key) if isinstance(d, dict) else None
        if v is None and nullable:
            return None
        if not is_date(v):
            self.add(f"{path}.{key}", f"not a YYYY-MM-DD date: {v!r}")
            return None
        return v

    def numlist(self, v, path: str, n: int | None = None, nullable_items: bool = False) -> list | None:
        if not isinstance(v, list):
            self.add(path, f"expected list, got {type(v).__name__}")
            return None
        if n is not None and len(v) != n:
            self.add(path, f"length {len(v)} != {n}")
        for i, x in enumerate(v):
            if x is None and nullable_items:
                continue
            if not is_num(x):
                self.add(f"{path}[{i}]", f"not a finite number: {x!r}")
                break
        return v

    def monotone(self, vals, path: str, increasing: bool = True, tol: float = 1e-12) -> None:
        for a, b in zip(vals, vals[1:]):
            if a is None or b is None:
                continue
            if (b < a - tol) if increasing else (b > a + tol):
                self.add(path, f"not monotone ({a} then {b})")
                return


def _finish(p: _P) -> list[str]:
    return p.items


# ----------------------------------------------------------------------------------------------- 6.2 / errors
def check_error(d) -> list[str]:
    p = _P()
    if not isinstance(d, dict) or d.get("status") != "error":
        return ["envelope: status must be 'error'"]
    err = d.get("error")
    if not p.keys(err, "error", ("code", "message")):
        return p.items
    if err.get("code") not in ERROR_CODES:
        p.add("error.code", f"unknown code {err.get('code')!r}")
    p.text(err, "message", "error")
    return p.items


def check_envelope(d) -> list[str]:
    """Job envelope: computing / ready / error."""
    p = _P()
    if not isinstance(d, dict):
        return ["envelope: not an object"]
    st = d.get("status")
    if st == "computing":
        pr = d.get("progress")
        if p.keys(pr, "progress", ("done", "total", "message")):
            done, total = pr.get("done"), pr.get("total")
            if not (isinstance(done, int) and not isinstance(done, bool) and done >= 0):
                p.add("progress.done", f"not a non-negative int: {done!r}")
            if not (isinstance(total, int) and not isinstance(total, bool) and total >= 0):
                p.add("progress.total", f"not a non-negative int: {total!r}")
            if isinstance(done, int) and isinstance(total, int) and total and done > total:
                p.add("progress", f"done {done} > total {total}")
            if not isinstance(pr.get("message"), str):
                p.add("progress.message", "not a string")
    elif st == "ready":
        pass
    elif st == "error":
        return check_error(d)
    else:
        p.add("status", f"unexpected {st!r}")
    return p.items


# ----------------------------------------------------------------------------------------------- 6.0
def check_status(d) -> list[str]:
    p = _P()
    if not p.keys(d, "status", ("status", "server_time", "taipei_time", "market_open", "last_trading_date", "version")):
        return p.items
    if d.get("status") != "ready":
        p.add("status", f"expected ready, got {d.get('status')!r}")
    for k in ("server_time", "taipei_time"):
        if not is_iso_datetime(d.get(k)):
            p.add(k, f"not ISO datetime: {d.get(k)!r}")
    if not isinstance(d.get("market_open"), bool):
        p.add("market_open", "not a bool")
    p.date(d, "last_trading_date", "status")
    p.text(d, "version", "status")
    if is_date(d.get("last_trading_date")) and date.fromisoformat(d["last_trading_date"]).weekday() >= 5:
        p.add("last_trading_date", "falls on a weekend")
    return p.items


def check_search(d, expect_code: str | None = None) -> list[str]:
    p = _P()
    if not p.keys(d, "search", ("status", "results")):
        return p.items
    if d.get("status") != "ready":
        p.add("status", f"expected ready, got {d.get('status')!r}")
    res = d.get("results")
    if not isinstance(res, list):
        p.add("results", "not a list")
        return p.items
    seen = set()
    for i, r in enumerate(res):
        path = f"results[{i}]"
        if not p.keys(r, path, ("code", "name", "market", "kind")):
            continue
        p.text(r, "code", path)
        p.text(r, "name", path)
        if r.get("market") not in ("TWSE", "TPEx", "INDEX"):
            p.add(f"{path}.market", f"unexpected {r.get('market')!r}")
        if r.get("kind") not in ("stock", "etf", "index"):
            p.add(f"{path}.kind", f"unexpected {r.get('kind')!r}")
        code = r.get("code")
        if code in seen:
            p.add(path, f"duplicate code {code}")
        seen.add(code)
        if isinstance(code, str) and isinstance(r.get("kind"), str) and r["kind"] != "index":
            if (r["kind"] == "etf") != code.startswith("00"):
                p.add(path, f"kind {r['kind']} inconsistent with code {code}")
        if isinstance(r.get("name"), str) and r["name"] == code:
            p.add(f"{path}.name", "name equals the code (no Chinese name resolved)")
    if expect_code is not None and expect_code not in seen:
        p.add("results", f"expected code {expect_code} missing")
    return p.items


POPULAR_ETFS = ("0050", "0056", "00878")
POPULAR_UNIVERSE = 12


def check_popular(d, universe_codes) -> list[str]:
    """The empty-query list of ``/api/search`` (DESIGN.md 6.0): the first 12 names of the universe followed by
    0050 / 0056 / 00878 -- complete whatever ``limit`` the client sent (audit: ``limit=5`` used to cut it)."""
    p = _P()
    p.items.extend(check_search(d))
    res = d.get("results") if isinstance(d, dict) else None
    if isinstance(res, list):
        codes = [r.get("code") for r in res if isinstance(r, dict)]
        want = list(universe_codes)[:POPULAR_UNIVERSE] + list(POPULAR_ETFS)
        if codes != want:
            p.add("results", f"popular list is {codes}, expected {want}")
    return p.items


def check_history(d, limit: int | None = None) -> list[str]:
    p = _P()
    if not p.keys(d, "history", ("status", "symbol", "name", "market", "kind", "asof", "last_bar_complete", "stale",
                                 "bars", "dividends")):
        return p.items
    if d.get("status") != "ready":
        p.add("status", f"expected ready, got {d.get('status')!r}")
    for k in ("symbol", "name"):
        p.text(d, k, "history")
    if d.get("market") not in ("TWSE", "TPEx", "INDEX"):
        p.add("market", f"unexpected {d.get('market')!r}")
    if d.get("kind") not in ("stock", "etf", "index"):
        p.add("kind", f"unexpected {d.get('kind')!r}")
    for k in ("last_bar_complete", "stale"):
        if not isinstance(d.get(k), bool):
            p.add(k, "not a bool")
    bars = d.get("bars")
    if not isinstance(bars, list) or not bars:
        p.add("bars", "empty or not a list")
        return p.items
    if limit is not None and len(bars) > limit:
        p.add("bars", f"{len(bars)} bars > limit {limit}")
    prev = None
    for i, b in enumerate(bars):
        path = f"bars[{i}]"
        if not p.keys(b, path, ("d", "o", "h", "l", "c", "v")):
            break
        if not is_date(b["d"]):
            p.add(f"{path}.d", f"bad date {b['d']!r}")
            break
        if prev is not None and b["d"] <= prev:
            p.add(path, f"dates not strictly ascending ({prev} -> {b['d']})")
            break
        prev = b["d"]
        if not all(is_num(b[k]) for k in ("o", "h", "l", "c", "v")):
            p.add(path, f"non-numeric field in {b}")
            break
        if min(b["o"], b["h"], b["l"], b["c"]) <= 0 or b["v"] < 0:
            p.add(path, f"non-positive price or negative volume in {b}")
            break
        if b["h"] < max(b["o"], b["c"]) - 1e-9 or b["l"] > min(b["o"], b["c"]) + 1e-9:
            p.add(path, f"high/low do not bracket open/close in {b}")
            break
    if d.get("asof") != bars[-1].get("d"):
        p.add("asof", f"{d.get('asof')!r} != last bar date {bars[-1].get('d')!r}")
    divs = d.get("dividends")
    if not isinstance(divs, list):
        p.add("dividends", "not a list")
    else:
        pd = None
        for i, x in enumerate(divs):
            if not (isinstance(x, dict) and is_date(x.get("d")) and is_num(x.get("amt")) and x["amt"] > 0):
                p.add(f"dividends[{i}]", f"bad dividend {x!r}")
                break
            if pd is not None and x["d"] < pd:
                p.add(f"dividends[{i}]", "not ascending")
                break
            pd = x["d"]
            if bars and x["d"] < bars[0]["d"]:
                p.add(f"dividends[{i}]", "dividend before the first returned bar")
                break
    return p.items


# ----------------------------------------------------------------------------------------------- 6.1
def _check_horizon(p: _P, hz, path: str, h: int, last_close: float | None) -> None:
    req = ("h", "days_label", "p_up", "p_up_raw", "exp_return", "median_return", "sigma", "quantiles",
           "price_quantiles", "signal", "score", "net_exp_return", "confidence", "components")
    if not p.keys(hz, path, req):
        return
    if hz.get("h") != h:
        p.add(f"{path}.h", f"{hz.get('h')!r} != {h}")
    p.text(hz, "days_label", path)
    pu = p.num(hz, "p_up", path, 0.0, 1.0)
    pr = p.num(hz, "p_up_raw", path, 0.0, 1.0)
    for name, v in (("p_up", pu), ("p_up_raw", pr)):
        if v is not None and not (0.0 < v < 1.0):
            p.add(f"{path}.{name}", f"{v} not strictly inside (0, 1)")
    er = p.num(hz, "exp_return", path, -1.0, 5.0)
    med = p.num(hz, "median_return", path, -1.0, 5.0)
    sg = p.num(hz, "sigma", path, 0.0, 5.0)
    if sg is not None and sg <= 0:
        p.add(f"{path}.sigma", "must be > 0")
    p.num(hz, "score", path, nullable=True)
    p.num(hz, "net_exp_return", path, nullable=True)
    if hz.get("signal") not in SIGNALS:
        p.add(f"{path}.signal", f"unexpected {hz.get('signal')!r}")
    if hz.get("confidence") not in CONFIDENCES:
        p.add(f"{path}.confidence", f"unexpected {hz.get('confidence')!r}")
    q = hz.get("quantiles")
    pq = hz.get("price_quantiles")
    qs, pqs = [], []
    if p.keys(q, f"{path}.quantiles", QKEYS):
        qs = [p.num(q, k, f"{path}.quantiles", -1.0, 5.0) for k in QKEYS]
        p.monotone(qs, f"{path}.quantiles", True)
        if all(x is not None for x in qs) and not (qs[0] < qs[-1]):
            p.add(f"{path}.quantiles", "degenerate (q05 >= q95)")
        if h == 1 and qs and qs[0] is not None and qs[-1] is not None and (qs[0] < -0.1 - 1e-6 or qs[-1] > 0.1 + 1e-6):
            p.add(f"{path}.quantiles", f"1-day quantiles outside the +-10% limit: {qs[0]}, {qs[-1]}")
    if p.keys(pq, f"{path}.price_quantiles", QKEYS):
        pqs = [p.num(pq, k, f"{path}.price_quantiles", 0.0) for k in QKEYS]
        p.monotone(pqs, f"{path}.price_quantiles", True)
        if last_close is not None and qs and len(qs) == len(pqs):
            for k, qv, pv in zip(QKEYS, qs, pqs):
                if qv is None or pv is None:
                    continue
                expect = last_close * (1.0 + qv)
                if abs(pv - expect) > max(0.01, 2e-3 * abs(expect)):
                    p.add(f"{path}.price_quantiles.{k}", f"{pv} inconsistent with last_close*(1+q)={expect:.4f}")
    if med is not None and qs and len(qs) == 5 and qs[2] is not None and abs(med - qs[2]) > 1e-3 + 0.05 * abs(qs[2]):
        p.add(f"{path}.median_return", f"{med} != quantiles['0.5'] {qs[2]}")
    if pr is not None and qs and len(qs) == 5 and qs[2] is not None and abs(qs[2]) > 0.5 * (sg or 1.0) \
            and (qs[2] > 0) != (pr > 0.5):
        p.add(f"{path}.p_up_raw", f"p_up_raw {pr} contradicts the median {qs[2]}")
    comps = hz.get("components")
    if isinstance(comps, dict):
        for name in COMPONENTS:
            c = comps.get(name)
            if not isinstance(c, dict):
                p.add(f"{path}.components.{name}", "missing")
                continue
            p.num(c, "p_up", f"{path}.components.{name}", 0.0, 1.0)
            p.num(c, "mu", f"{path}.components.{name}", -1.0, 5.0)
    else:
        p.add(f"{path}.components", "not an object")
    if er is not None and sg is not None and abs(er) > 6 * sg + 0.01:
        p.add(f"{path}.exp_return", f"|{er}| implausibly large relative to sigma {sg}")


def check_forecast(d) -> list[str]:
    """Strict check of a ready ``/api/forecast`` payload (DESIGN.md 6.1)."""
    p = _P()
    req = ("status", "symbol", "name", "market", "kind", "asof", "asof_complete", "next_trading_day", "last_close",
           "chg_1d", "n_bars", "short_history", "horizons", "fan", "regime", "vol", "risk", "models", "drivers",
           "notes", "disclaimer")
    if not p.keys(d, "forecast", req):
        return p.items
    if d.get("status") != "ready":
        p.add("status", f"expected ready, got {d.get('status')!r}")
    p.text(d, "symbol", "forecast")
    name = p.text(d, "name", "forecast")
    if name is not None and name == d.get("symbol"):
        p.add("name", "equals the code (no Chinese name resolved)")
    if d.get("market") not in ("TWSE", "TPEx", "INDEX"):
        p.add("market", f"unexpected {d.get('market')!r}")
    if d.get("kind") not in ("stock", "etf", "index"):
        p.add("kind", f"unexpected {d.get('kind')!r}")
    asof = p.date(d, "asof", "forecast")
    nxt = p.date(d, "next_trading_day", "forecast")
    if asof and nxt:
        if nxt <= asof:
            p.add("next_trading_day", f"{nxt} not after asof {asof}")
        if date.fromisoformat(nxt).weekday() >= 5:
            p.add("next_trading_day", f"{nxt} is a weekend")
    for k in ("asof_complete", "short_history"):
        if not isinstance(d.get(k), bool):
            p.add(k, "not a bool")
    last_close = p.num(d, "last_close", "forecast", 0.0)
    p.num(d, "chg_1d", "forecast", -0.2, 0.2, nullable=True)
    nb = d.get("n_bars")
    if not (isinstance(nb, int) and nb >= 250):
        p.add("n_bars", f"expected int >= 250, got {nb!r}")
    # horizons
    hs = d.get("horizons")
    if isinstance(hs, dict):
        for k in HORIZON_KEYS:
            if k not in hs:
                p.add("horizons", f"missing horizon {k}")
            else:
                _check_horizon(p, hs[k], f"horizons.{k}", int(k), last_close)
        s1 = [hs.get(k, {}).get("sigma") for k in HORIZON_KEYS if isinstance(hs.get(k), dict)]
        if len(s1) == 3 and all(is_num(x) for x in s1) and not (s1[0] < s1[1] < s1[2]):
            p.add("horizons", f"sigma must grow with the horizon, got {s1}")
    else:
        p.add("horizons", "not an object")
    # fan chart
    fan = d.get("fan")
    if p.keys(fan, "fan", ("days", "levels", "prices", "returns")):
        days, levels = fan.get("days"), fan.get("levels")
        if days != list(range(1, 21)):
            p.add("fan.days", f"expected 1..20, got {days!r}")
        if not (isinstance(levels, list) and len(levels) == 5 and all(is_num(x) for x in levels)
                and all(abs(a - b) < 1e-9 for a, b in zip(levels, FAN_LEVELS))):
            p.add("fan.levels", f"expected {FAN_LEVELS}, got {levels!r}")
        for key in ("prices", "returns"):
            m = fan.get(key)
            if not (isinstance(m, list) and len(m) == 5 and all(isinstance(r, list) and len(r) == 20 for r in m)):
                p.add(f"fan.{key}", "expected 5 rows x 20 columns")
                continue
            if not all(is_num(x) for r in m for x in r):
                p.add(f"fan.{key}", "non-finite entries")
                continue
            for j in range(20):  # monotone in the level, every day
                col = [m[i][j] for i in range(5)]
                if any(b < a - 1e-9 for a, b in zip(col, col[1:])):
                    p.add(f"fan.{key}", f"day {j + 1}: bands cross {col}")
                    break
            if key == "prices" and any(x <= 0 for r in m for x in r):
                p.add("fan.prices", "non-positive price")
        pr, rt = fan.get("prices"), fan.get("returns")
        if isinstance(pr, list) and isinstance(rt, list) and last_close is not None and len(pr) == len(rt) == 5:
            try:
                for i in range(5):
                    for j in range(20):
                        if abs(pr[i][j] - last_close * (1 + rt[i][j])) > max(0.01, 2e-3 * pr[i][j]):
                            p.add("fan", f"prices[{i}][{j}]={pr[i][j]} inconsistent with last_close*(1+returns)")
                            raise StopIteration
                widths = [pr[4][j] - pr[0][j] for j in range(20)]
                if any(b < a - 1e-6 for a, b in zip(widths, widths[1:])):
                    p.add("fan", "90% band width must not shrink with the horizon")
            except StopIteration:
                pass
            except (TypeError, IndexError):
                pass
    # regime
    rg = d.get("regime")
    if p.keys(rg, "regime", ("labels", "probs", "current", "transition")):
        labels, probs, trans = rg.get("labels"), rg.get("probs"), rg.get("transition")
        if not (isinstance(labels, list) and labels and all(isinstance(x, str) and x for x in labels)):
            p.add("regime.labels", "expected non-empty list of strings")
            labels = []
        k = len(labels)
        if p.numlist(probs, "regime.probs", k or None) is not None and probs and all(is_num(x) for x in probs):
            if abs(sum(probs) - 1.0) > 1e-3 or min(probs) < -1e-12:
                p.add("regime.probs", f"not a probability vector: {probs}")
        if rg.get("current") not in labels:
            p.add("regime.current", f"{rg.get('current')!r} not in labels")
        if not (isinstance(trans, list) and len(trans) == k and all(isinstance(r, list) and len(r) == k for r in trans)):
            p.add("regime.transition", f"expected {k}x{k}")
        elif all(is_num(x) for r in trans for x in r):
            if any(abs(sum(r) - 1.0) > 1e-3 or min(r) < -1e-12 for r in trans):
                p.add("regime.transition", "rows must be probability vectors")
        else:
            p.add("regime.transition", "non-finite entries")
    # volatility
    vol = d.get("vol")
    if p.keys(vol, "vol", ("sigma_daily", "sigma_annual", "percentile", "regime_note")):
        sd = p.num(vol, "sigma_daily", "vol", 0.0, 0.2)
        sa = p.num(vol, "sigma_annual", "vol", 0.0, 3.0)
        p.num(vol, "percentile", "vol", 0.0, 1.0)
        p.text(vol, "regime_note", "vol")
        if sd is not None and sa is not None and not (0.8 * sd * 15.0 < sa < 1.2 * sd * 16.5):
            p.add("vol", f"sigma_annual {sa} inconsistent with sigma_daily {sd} (x sqrt(245)={sd * 245 ** 0.5:.4f})")
    # risk
    rk = d.get("risk")
    if p.keys(rk, "risk", ("var95_1d", "cvar95_1d", "var95_20d", "cvar95_20d", "median_max_drawdown_20d",
                           "prob_loss_gt_10pct_20d")):
        v1 = p.num(rk, "var95_1d", "risk", -1.0, 0.0)
        c1 = p.num(rk, "cvar95_1d", "risk", -1.0, 0.0)
        v20 = p.num(rk, "var95_20d", "risk", -1.0, 0.5)
        c20 = p.num(rk, "cvar95_20d", "risk", -1.0, 0.5)
        p.num(rk, "median_max_drawdown_20d", "risk", -1.0, 0.0)
        p.num(rk, "prob_loss_gt_10pct_20d", "risk", 0.0, 1.0)
        if v1 is not None and c1 is not None and c1 > v1 + 1e-9:
            p.add("risk", f"cvar95_1d {c1} must be <= var95_1d {v1}")
        if v20 is not None and c20 is not None and c20 > v20 + 1e-9:
            p.add("risk", f"cvar95_20d {c20} must be <= var95_20d {v20}")
    # models
    md = d.get("models")
    if p.keys(md, "models", ("weights", "names")):
        w, names = md.get("weights"), md.get("names")
        if isinstance(w, dict):
            for hk in HORIZON_KEYS:
                row = w.get(hk)
                if not isinstance(row, dict):
                    p.add(f"models.weights.{hk}", "missing")
                    continue
                vals = [row.get(c) for c in COMPONENTS]
                if not all(is_num(x) and -1e-12 <= x <= 1 + 1e-12 for x in vals):
                    p.add(f"models.weights.{hk}", f"bad weights {row}")
                elif abs(sum(vals) - 1.0) > 1e-3:
                    p.add(f"models.weights.{hk}", f"weights sum to {sum(vals)}")
        else:
            p.add("models.weights", "not an object")
        if not (isinstance(names, dict) and all(isinstance(names.get(c), str) and names[c] for c in COMPONENTS)):
            p.add("models.names", "expected a name for each of garch/hmm/dlm/naive")
    # drivers
    dr = d.get("drivers")
    if not isinstance(dr, list):
        p.add("drivers", "not a list")
    else:
        for i, x in enumerate(dr):
            path = f"drivers[{i}]"
            if p.keys(x, path, ("name", "key", "value", "z", "contribution")):
                p.text(x, "name", path)
                p.text(x, "key", path)
                for k in ("value", "z", "contribution"):
                    p.num(x, k, path)
    notes = d.get("notes")
    if not (isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes)):
        p.add("notes", "expected non-empty list of strings")
    disc = d.get("disclaimer")
    if not (isinstance(disc, str) and "非投資建議" in disc):
        p.add("disclaimer", "must be a string containing 非投資建議")
    return p.items


# ----------------------------------------------------------------------------------------------- 6.3
REC_KEYS = ("rank", "code", "name", "kind", "last_close", "chg_1d", "p_up", "exp_return", "net_exp_return", "sigma",
            "score", "score_shrunk", "signal", "confidence", "weight", "reason", "p_up_by_h", "regime",
            "avg_value_20d", "liquid",
            # DESIGN.md 9 (additive): the relative-strength ordering and its honesty columns
            "rank_signal", "score_rank_pct", "lean", "edge_covers_cost", "low_vol")
LEANS = ("bullish", "bearish", "neutral")
SUPPRESSED = (None, "B0", "no_positive_edge")
EVIDENCE_SOURCES = ("none", "universe_validation")


def _check_rec(p: _P, r, path: str) -> None:
    if not p.keys(r, path, REC_KEYS):
        return
    if not (isinstance(r.get("rank"), int) and r["rank"] >= 1):
        p.add(f"{path}.rank", f"bad rank {r.get('rank')!r}")
    p.text(r, "code", path)
    nm = p.text(r, "name", path)
    if nm is not None and nm == r.get("code"):
        p.add(f"{path}.name", "name equals the code")
    if r.get("kind") not in ("stock", "etf"):
        p.add(f"{path}.kind", f"unexpected {r.get('kind')!r}")
    p.num(r, "last_close", path, 0.0)
    p.num(r, "chg_1d", path, -0.2, 0.2, nullable=True)
    pu = p.num(r, "p_up", path, 0.0, 1.0)
    p.num(r, "exp_return", path, -1.0, 5.0)
    net = p.num(r, "net_exp_return", path, -1.0, 5.0)
    p.num(r, "sigma", path, 0.0, 5.0)
    p.num(r, "score", path)
    p.num(r, "score_shrunk", path)
    w = p.num(r, "weight", path, 0.0, 1.0)
    p.num(r, "avg_value_20d", path, 0.0, nullable=True)
    if r.get("signal") not in SIGNALS:
        p.add(f"{path}.signal", f"unexpected {r.get('signal')!r}")
    if r.get("confidence") not in CONFIDENCES:
        p.add(f"{path}.confidence", f"unexpected {r.get('confidence')!r}")
    p.text(r, "reason", path)
    if not isinstance(r.get("liquid"), bool):
        p.add(f"{path}.liquid", "not a bool")
    pb = r.get("p_up_by_h")
    if p.keys(pb, f"{path}.p_up_by_h", HORIZON_KEYS):
        for k in HORIZON_KEYS:
            p.num(pb, k, f"{path}.p_up_by_h", 0.0, 1.0)
    p.num(r, "rank_signal", path)
    p.num(r, "score_rank_pct", path, 0.0, 1.0)
    p.num(r, "edge_vs_baseline", path, -1.0, 1.0, nullable=True)  # optional key: null when the naive component is missing
    if r.get("lean") not in LEANS:
        p.add(f"{path}.lean", f"unexpected {r.get('lean')!r}")
    for k in ("edge_covers_cost", "low_vol"):
        if not isinstance(r.get(k), bool):
            p.add(f"{path}.{k}", "not a bool")
    if net is not None and isinstance(r.get("edge_covers_cost"), bool) and r["edge_covers_cost"] != (net > 0):
        p.add(f"{path}.edge_covers_cost", f"{r['edge_covers_cost']} but net_exp_return is {net}")
    if r.get("signal") != "buy" and w not in (None, 0, 0.0):
        p.add(f"{path}.weight", f"non-buy name {r.get('code')} carries weight {w}")
    if pu is not None and r.get("signal") == "buy" and pu < 0.5:
        p.add(f"{path}.signal", f"buy with p_up {pu} < 0.5")
    if pu is not None and r.get("signal") == "avoid" and pu > 0.5:
        p.add(f"{path}.signal", f"avoid with p_up {pu} > 0.5")


def check_evidence(ev, horizon: int | None = None) -> list[str]:
    """The ``evidence`` object of the picks payload (DESIGN.md 9.4 + the optional after-cost keys of 6.3)."""
    p = _P()
    if not p.keys(ev, "evidence", ("source",)):
        return p.items
    if ev.get("source") not in EVIDENCE_SOURCES:
        p.add("evidence.source", f"unexpected {ev.get('source')!r}")
        return p.items
    if horizon is not None and "horizon" in ev and ev["horizon"] != horizon:
        p.add("evidence.horizon", f"{ev['horizon']!r} != {horizon}")
    if ev["source"] == "none":
        for k in ("verdict_level", "ic", "net_cagr", "beats_ew_after_costs"):
            if ev.get(k) is not None:
                p.add(f"evidence.{k}", f"source is 'none' but {k} = {ev[k]!r}")
        return p.items
    if not p.keys(ev, "evidence", ("horizon", "verdict_level", "ic", "ic_pvalue", "hit_rate", "base_rate", "computed_at")):
        return p.items
    if ev.get("verdict_level") not in VERDICT_LEVELS:
        p.add("evidence.verdict_level", f"unexpected {ev.get('verdict_level')!r}")
    p.num(ev, "ic", "evidence", -1.0, 1.0, nullable=True)
    p.num(ev, "ic_pvalue", "evidence", 0.0, 1.0, nullable=True)
    p.num(ev, "hit_rate", "evidence", 0.0, 1.0, nullable=True)
    p.num(ev, "base_rate", "evidence", 0.0, 1.0, nullable=True)
    for k in ("net_cagr", "ew_cagr", "sigma_cagr"):
        p.num(ev, k, "evidence", -1.0, nullable=True)
    if ev.get("beats_ew_after_costs") is not None and not isinstance(ev["beats_ew_after_costs"], bool):
        p.add("evidence.beats_ew_after_costs", "not a bool or null")
    if not is_iso_datetime(ev.get("computed_at")):
        p.add("evidence.computed_at", f"not ISO datetime: {ev.get('computed_at')!r}")
    if ev.get("data_asof") is not None and not is_date(ev["data_asof"]):
        p.add("evidence.data_asof", f"not a YYYY-MM-DD date: {ev['data_asof']!r}")
    if ev.get("current") is not None and not isinstance(ev["current"], bool):
        p.add("evidence.current", "not a bool")
    return p.items


def check_recommendations(d, horizon: int | None = None, top: int | None = None) -> list[str]:
    """Strict check of a ready ``/api/recommendations`` payload (DESIGN.md 6.3 + 9).

    The ordering contract: ``rank`` is ONE global rank, the position in ``all`` ordered by ``rank_signal`` (descending);
    ``buys`` / ``leaders`` best first, ``avoids`` / ``laggards`` weakest first, never by ``score`` / ``score_shrunk``.
    """
    p = _P()
    req = ("status", "asof", "asof_complete", "horizon", "computed_at", "universe_size", "n_scored", "market",
           "buys", "avoids", "all", "leaders", "laggards", "laggards_all_above_half", "n_buy_total", "rank_key",
           "rank_sigma_corr", "universe_sigma_median", "portfolio", "evidence", "notes")
    if not p.keys(d, "recommendations", req):
        return p.items
    if d.get("status") != "ready":
        p.add("status", f"expected ready, got {d.get('status')!r}")
    p.date(d, "asof", "recs")
    if not isinstance(d.get("asof_complete"), bool):
        p.add("asof_complete", "not a bool")
    if d.get("horizon") not in (1, 5, 20) or (horizon is not None and d.get("horizon") != horizon):
        p.add("horizon", f"unexpected {d.get('horizon')!r}")
    if not is_iso_datetime(d.get("computed_at")):
        p.add("computed_at", f"not ISO datetime: {d.get('computed_at')!r}")
    us, ns = d.get("universe_size"), d.get("n_scored")
    if not (isinstance(us, int) and isinstance(ns, int) and 0 <= ns <= us):
        p.add("universe_size/n_scored", f"bad counts {us!r}/{ns!r}")
    mk = d.get("market")
    if mk is not None:
        if p.keys(mk, "market", ("p_up_1", "p_up_5", "regime", "last_close", "chg_1d")):
            p.num(mk, "p_up_1", "market", 0.0, 1.0)
            p.num(mk, "p_up_5", "market", 0.0, 1.0)
            p.num(mk, "last_close", "market", 0.0)
            p.num(mk, "chg_1d", "market", -0.2, 0.2, nullable=True)
            p.text(mk, "regime", "market")
    if d.get("rank_key") != "rank_signal":
        p.add("rank_key", f"expected 'rank_signal', got {d.get('rank_key')!r}")
    p.num(d, "rank_sigma_corr", "recs", -1.0, 1.0, nullable=True)
    p.num(d, "universe_sigma_median", "recs", 0.0, nullable=True)
    if not isinstance(d.get("laggards_all_above_half"), bool):
        p.add("laggards_all_above_half", "not a bool")
    allr, buys, avoids = d.get("all"), d.get("buys"), d.get("avoids")
    leaders, laggards = d.get("leaders"), d.get("laggards")
    for key, lst in (("all", allr), ("buys", buys), ("avoids", avoids), ("leaders", leaders), ("laggards", laggards)):
        if not isinstance(lst, list):
            p.add(key, "not a list")
            return p.items
        for i, r in enumerate(lst):
            _check_rec(p, r, f"{key}[{i}]")
    p.items.extend(check_evidence(d.get("evidence"), d.get("horizon")))
    if p.items:
        return p.items
    n = len(allr)
    if ns is not None and n != ns:
        p.add("n_scored", f"{ns} != len(all) {n}")
    if len({r["code"] for r in allr}) != n:
        p.add("all", "duplicate codes")
    # ---- ordering: rank_signal descending (ties by p_up descending, then code); one GLOBAL rank
    rs = [r["rank_signal"] for r in allr]
    if any(b > a + 1e-12 for a, b in zip(rs, rs[1:])):
        p.add("all", "not sorted by rank_signal descending")
    for a, b in zip(allr, allr[1:]):
        if abs(a["rank_signal"] - b["rank_signal"]) <= 1e-12 and (b["p_up"] > a["p_up"] + 1e-12 or (
                abs(b["p_up"] - a["p_up"]) <= 1e-12 and b["code"] < a["code"])):
            p.add("all", f"tie between {a['code']} and {b['code']} not broken by p_up then code")
            break
    if [r["rank"] for r in allr] != list(range(1, n + 1)):
        p.add("all", "ranks are not 1..n in order")
    rank_of = {r["code"]: r["rank"] for r in allr}
    by_code = {r["code"]: r for r in allr}
    for key, lst in (("buys", buys), ("avoids", avoids), ("leaders", leaders), ("laggards", laggards)):
        if not {r["code"] for r in lst} <= set(rank_of):
            p.add(key, "names missing from `all`")
            continue
        if any(r["rank"] != rank_of[r["code"]] for r in lst):
            p.add(key, "rank differs from the global rank of the same name in `all`")
        ranks = [r["rank"] for r in lst]
        if key in ("buys", "leaders") and ranks != sorted(ranks):
            p.add(key, "not best-first (rank ascending)")
        if key in ("avoids", "laggards") and ranks != sorted(ranks, reverse=True):
            p.add(key, "not weakest-first (rank descending)")
        if any(by_code[r["code"]] != r for r in lst):
            p.add(key, "a row differs from the same name in `all`")
    if any(r["signal"] != "buy" for r in buys):
        p.add("buys", "contains a non-buy signal")
    if any(r["signal"] != "avoid" for r in avoids):
        p.add("avoids", "contains a non-avoid signal")
    nbuy_all = sum(1 for r in allr if r["signal"] == "buy")
    if d.get("n_buy_total") != nbuy_all:
        p.add("n_buy_total", f"{d.get('n_buy_total')!r} != number of buy signals {nbuy_all}")
    if top is not None:
        if len(buys) != min(top, nbuy_all):
            p.add("buys", f"expected min(top, #buy signals)={min(top, nbuy_all)}, got {len(buys)}")
        if len(avoids) > top:
            p.add("avoids", f"{len(avoids)} > top {top}")
        want_l = [r["code"] for r in allr if r["p_up"] > 0.5][:top]
        if [r["code"] for r in leaders] != want_l:
            p.add("leaders", f"expected the first {top} names of `all` with p_up > 0.5")
        if len(laggards) != min(top, n // 2):
            p.add("laggards", f"expected min(top, n_scored//2)={min(top, n // 2)} names, got {len(laggards)}")
    if [r["code"] for r in laggards] != [r["code"] for r in reversed(allr)][:len(laggards)]:
        p.add("laggards", "not the bottom of `all` (weakest first)")
    if d.get("laggards_all_above_half") != (bool(laggards) and all(r["p_up"] > 0.5 for r in laggards)):
        p.add("laggards_all_above_half", "inconsistent with the p_up of the laggards")
    if any(r["p_up"] <= 0.5 for r in leaders):
        p.add("leaders", "contains p_up <= 0.5")
    # ---- derived columns
    sg_med = d.get("universe_sigma_median")
    for r in allr:
        pct = (n - r["rank"]) / (n - 1) if n > 1 else 0.5
        if abs(r["score_rank_pct"] - pct) > 1e-9:
            p.add(f"all[{r['rank'] - 1}].score_rank_pct", f"{r['score_rank_pct']} != (n-rank)/(n-1) = {pct}")
            break
        lean = ("bullish" if (pct >= 0.8 and r["p_up"] > 0.5) else "bearish" if (pct <= 0.2 and r["p_up"] < 0.5) else "neutral")
        if r["lean"] != lean:
            p.add(f"all[{r['rank'] - 1}].lean", f"{r['lean']!r} != {lean!r} (pct {pct:.3f}, p_up {r['p_up']:.3f})")
            break
        if sg_med is not None and r["low_vol"] != (r["sigma"] < sg_med):
            p.add(f"all[{r['rank'] - 1}].low_vol", f"{r['low_vol']} with sigma {r['sigma']} vs median {sg_med}")
            break
    # ---- portfolio
    pf = d.get("portfolio")
    if p.keys(pf, "portfolio", ("cash_weight", "n", "note", "suppressed_reason")):
        cw = p.num(pf, "cash_weight", "portfolio", 0.0, 1.0)
        tot = sum(r["weight"] for r in allr)
        if cw is not None and abs(cw + tot - 1.0) > 1e-3:
            p.add("portfolio", f"cash_weight {cw} + sum(weights) {tot:.4f} != 1")
        mw = max([r["weight"] for r in allr] or [0.0])
        if mw > 0.15 + 1e-9:
            p.add("portfolio", f"a weight {mw} exceeds MAX_WEIGHT 0.15")
        nn = sum(1 for r in allr if r["weight"] > 0)
        if pf.get("n") != nn:
            p.add("portfolio.n", f"{pf.get('n')!r} != number of weighted names {nn}")
        p.text(pf, "note", "portfolio")
        if pf.get("suppressed_reason") not in SUPPRESSED:
            p.add("portfolio.suppressed_reason", f"unexpected {pf.get('suppressed_reason')!r}")
        elif pf.get("suppressed_reason") is not None:
            if cw is not None and abs(cw - 1.0) > 1e-9 or nn != 0:
                p.add("portfolio", "suppressed portfolio must be 100 % cash with no weighted name")
            if "雜訊" not in str(pf.get("note")) and ns:
                p.add("portfolio.note", "a suppressed portfolio must say that the cross-section carries no view")
        if any(r["weight"] > 0 and not r["edge_covers_cost"] for r in allr):
            p.add("portfolio", "a name whose net expected return is <= 0 carries a weight")
    notes = d.get("notes")
    if not (isinstance(notes, list) and notes and all(isinstance(x, str) for x in notes)):
        p.add("notes", "expected non-empty list of strings")
    return p.items


# ----------------------------------------------------------------------------------------------- 6.4
def check_backtest(d, horizon: int | None = None) -> list[str]:
    """Envelope + ``evaluation.report.validate_backtest_report`` + a few cross-checks on a ready ``/api/backtest``."""
    p = _P()
    if not isinstance(d, dict) or d.get("status") != "ready":
        return ["backtest: not a ready envelope"]
    rep = d.get("report")
    if not isinstance(rep, dict):
        return ["backtest: missing `report` object"]
    try:
        from backend.evaluation.report import validate_backtest_report
        p.items.extend(f"report: {m}" for m in validate_backtest_report(rep))
    except ImportError as exc:  # pragma: no cover - evaluation module missing
        p.add("report", f"validator unavailable: {exc}")
    if horizon is not None and rep.get("horizon") != horizon:
        p.add("report.horizon", f"{rep.get('horizon')!r} != {horizon}")
    cv = (rep.get("strategy") or {}).get("curve") or {}
    ds = cv.get("dates")
    if isinstance(ds, list) and ds:
        if len(ds) > 600:
            p.add("report.strategy.curve", f"{len(ds)} points > 600")
        if not all(is_date(x) for x in ds[:3] + ds[-3:]):
            p.add("report.strategy.curve.dates", "bad dates")
        for k in ("strategy", "buyhold", "position", "drawdown_strategy", "drawdown_buyhold"):
            if len(cv.get(k) or []) != len(ds):
                p.add(f"report.strategy.curve.{k}", "length differs from dates")
    per = rep.get("period") or {}
    if is_date(per.get("start")) and is_date(per.get("end")) and per["start"] > per["end"]:
        p.add("report.period", "start after end")
    v = rep.get("verdict") or {}
    if v.get("level") not in VERDICT_LEVELS:
        p.add("report.verdict.level", f"unexpected {v.get('level')!r}")
    if "schema_problems" in d:
        p.add("schema_problems", f"server reported {d['schema_problems']}")
    return p.items


# ----------------------------------------------------------------------------------------------- 6.5
def _check_stats(p: _P, st, path: str, nullable: bool = False) -> None:
    if st is None and nullable:
        return
    if not p.keys(st, path, STATS_KEYS):
        return
    for k in STATS_KEYS:
        if k == "sharpe_ci95":
            ci = st.get(k)
            if not (isinstance(ci, list) and len(ci) == 2 and all(x is None or is_num(x) for x in ci)):
                p.add(f"{path}.{k}", f"expected [lo, hi], got {ci!r}")
            elif all(is_num(x) for x in ci) and ci[0] > ci[1]:
                p.add(f"{path}.{k}", "lo > hi")
        elif k == "n_trades":
            if st.get(k) is not None and not (isinstance(st[k], (int, float)) and st[k] >= 0):
                p.add(f"{path}.{k}", f"bad value {st[k]!r}")
        else:
            p.num(st, k, path, nullable=True)
    mdd = st.get("max_dd")
    if is_num(mdd) and not (-1.0 <= mdd <= 1e-9):
        p.add(f"{path}.max_dd", f"{mdd} outside [-1, 0]")
    ex = st.get("exposure")
    if is_num(ex) and not (-1e-9 <= ex <= 1 + 1e-9):
        p.add(f"{path}.exposure", f"{ex} outside [0, 1]")
    for k in ("psr", "dsr"):
        if is_num(st.get(k)) and not (-1e-9 <= st[k] <= 1 + 1e-9):
            p.add(f"{path}.{k}", f"{st[k]} outside [0, 1]")


def check_universe_report(d, horizon: int | None = None, top_k: int | None = None) -> list[str]:
    """Strict check of the ``/api/backtest/universe`` result; ``d`` may be the ready envelope or the ``report``."""
    p = _P()
    if isinstance(d, dict) and "report" in d and isinstance(d["report"], dict):
        if d.get("status") != "ready":
            p.add("status", f"expected ready, got {d.get('status')!r}")
        if "schema_problems" in d:
            p.add("schema_problems", f"server reported {d['schema_problems']}")
        rep = d["report"]
    else:
        rep = d
    req = ("horizon", "top_k", "period", "ic", "picks", "top", "benchmark_equal_weight", "benchmark_taiex", "spread",
           "curve", "per_stock", "verdict", "notes")
    if not p.keys(rep, "report", req):
        return p.items
    if horizon is not None and rep.get("horizon") != horizon:
        p.add("horizon", f"{rep.get('horizon')!r} != {horizon}")
    if top_k is not None and rep.get("top_k") != top_k:
        p.add("top_k", f"{rep.get('top_k')!r} != {top_k}")
    per = rep.get("period")
    if p.keys(per, "period", ("start", "end", "n_days", "n_stocks")):
        s, e = p.date(per, "start", "period"), p.date(per, "end", "period")
        if s and e and s > e:
            p.add("period", "start after end")
        if not (isinstance(per.get("n_days"), int) and per["n_days"] >= 0):
            p.add("period.n_days", f"bad value {per.get('n_days')!r}")
        if not (isinstance(per.get("n_stocks"), int) and per["n_stocks"] >= 0):
            p.add("period.n_stocks", f"bad value {per.get('n_stocks')!r}")
    ic = rep.get("ic")
    if p.keys(ic, "ic", ("mean", "ir", "t", "pvalue", "series")):
        p.num(ic, "mean", "ic", -1.0, 1.0, nullable=True)
        p.num(ic, "ir", "ic", nullable=True)
        p.num(ic, "t", "ic", nullable=True)
        p.num(ic, "pvalue", "ic", 0.0, 1.0, nullable=True)
        se = ic.get("series")
        if p.keys(se, "ic.series", ("dates", "values")):
            ds, vs = se.get("dates"), se.get("values")
            if not (isinstance(ds, list) and isinstance(vs, list) and len(ds) == len(vs)):
                p.add("ic.series", "dates/values length mismatch")
            elif ds and not is_date(ds[0]):
                p.add("ic.series.dates", "bad dates")
    pk = rep.get("picks")
    if p.keys(pk, "picks", ("hit_rate", "base_rate", "n", "p_value", "mean_return")):
        p.num(pk, "hit_rate", "picks", 0.0, 1.0, nullable=True)
        p.num(pk, "base_rate", "picks", 0.0, 1.0, nullable=True)
        p.num(pk, "p_value", "picks", 0.0, 1.0, nullable=True)
        p.num(pk, "mean_return", "picks", -1.0, 1.0, nullable=True)
        if not (isinstance(pk.get("n"), (int, float)) and pk["n"] >= 0):
            p.add("picks.n", f"bad value {pk.get('n')!r}")
    _check_stats(p, rep.get("top"), "top")
    _check_stats(p, rep.get("benchmark_equal_weight"), "benchmark_equal_weight")
    _check_stats(p, rep.get("benchmark_taiex"), "benchmark_taiex", nullable=True)
    # ---- DESIGN.md 9.1 additive blocks (optional for old payloads; when present they must be well formed)
    _check_stats(p, rep.get("benchmark_top_sigma"), "benchmark_top_sigma", nullable=True)
    if "rank_key" in rep and rep["rank_key"] != "rank_signal":
        p.add("rank_key", f"expected 'rank_signal', got {rep['rank_key']!r}")
    if "only_buy_signals" in rep and not isinstance(rep["only_buy_signals"], bool):
        p.add("only_buy_signals", "not a bool")
    if "labels" in rep:
        lb = rep["labels"]
        if p.keys(lb, "labels", ("top", "benchmark_equal_weight", "benchmark_top_sigma", "benchmark_taiex")):
            for k in ("top", "benchmark_equal_weight", "benchmark_top_sigma", "benchmark_taiex"):
                p.text(lb, k, "labels")
    bk = rep.get("basket")
    if bk is not None:
        if p.keys(bk, "basket", ("k", "mean_sigma", "universe_mean_sigma", "sigma_ratio", "mean_beta", "universe_mean_beta",
                                 "beta_market", "topk_overlap", "corr_rank_sigma", "share_net_positive")):
            for k in ("mean_sigma", "universe_mean_sigma", "sigma_ratio"):
                p.num(bk, k, "basket", 0.0, nullable=True)
            for k in ("topk_overlap", "share_net_positive"):
                p.num(bk, k, "basket", 0.0, 1.0, nullable=True)
            p.num(bk, "corr_rank_sigma", "basket", -1.0, 1.0, nullable=True)
            for k in ("mean_beta", "universe_mean_beta"):
                p.num(bk, k, "basket", nullable=True)
            if bk.get("beta_market") not in ("taiex", "equal_weight"):
                p.add("basket.beta_market", f"unexpected {bk.get('beta_market')!r}")
    ex = rep.get("exposure")
    if ex is not None:
        if p.keys(ex, "exposure", ("mean_sigma_top", "mean_sigma_universe", "beta_top", "beta_universe")):
            for k in ("mean_sigma_top", "mean_sigma_universe"):
                p.num(ex, k, "exposure", 0.0, nullable=True)
            for k in ("beta_top", "beta_universe"):
                p.num(ex, k, "exposure", nullable=True)
    try:  # the evaluation module's own validator (covers every key of 6.5 including the additive ones)
        from backend.evaluation.universe_bt import validate_universe_report
        p.items.extend(f"report: {m}" for m in validate_universe_report(rep))
    except ImportError as exc:  # pragma: no cover - evaluation module missing
        p.add("report", f"validator unavailable: {exc}")
    sp = rep.get("spread")
    if p.keys(sp, "spread", ("mean", "t")):
        p.num(sp, "mean", "spread", nullable=True)
        p.num(sp, "t", "spread", nullable=True)
    cv = rep.get("curve")
    if p.keys(cv, "curve", ("dates", "top", "equal_weight", "taiex")):
        ds = cv.get("dates")
        if not (isinstance(ds, list) and ds and all(is_date(x) for x in ds[:3] + ds[-3:])):
            p.add("curve.dates", "empty or bad dates")
        else:
            if len(ds) > 600:
                p.add("curve", f"{len(ds)} points > 600")
            for k in ("top", "equal_weight"):
                v = cv.get(k)
                if not (isinstance(v, list) and len(v) == len(ds) and all(is_num(x) and x > 0 for x in v)):
                    p.add(f"curve.{k}", "expected positive numbers, same length as dates")
            tx = cv.get("taiex")
            if tx is not None and not (isinstance(tx, list) and len(tx) == len(ds)):
                p.add("curve.taiex", "expected null or same length as dates")
            ts_ = cv.get("top_sigma")
            if ts_ is not None and not (isinstance(ts_, list) and len(ts_) == len(ds)):
                p.add("curve.top_sigma", "expected null or same length as dates")
            if "top_sigma" in cv and (ts_ is None) != (rep.get("benchmark_top_sigma") is None):
                p.add("curve.top_sigma", "null-ness differs from benchmark_top_sigma")
            if (tx is None) != (rep.get("benchmark_taiex") is None):
                p.add("curve.taiex", "null-ness differs from benchmark_taiex")
    ps = rep.get("per_stock")
    if not isinstance(ps, list):
        p.add("per_stock", "not a list")
    else:
        for i, r in enumerate(ps):
            path = f"per_stock[{i}]"
            if p.keys(r, path, ("code", "name", "accuracy", "brier_skill", "ic", "n")):
                p.text(r, "code", path)
                nm = p.text(r, "name", path)
                if nm is not None and nm == r.get("code"):
                    p.add(f"{path}.name", "name equals the code")
                p.num(r, "accuracy", path, 0.0, 1.0, nullable=True)
                p.num(r, "brier_skill", path, nullable=True)
                p.num(r, "brier_skill_oracle", path, nullable=True)  # optional key (hindsight reference, DESIGN 6.4)
                p.num(r, "ic", path, -1.0, 1.0, nullable=True)
                if not (isinstance(r.get("n"), (int, float)) and r["n"] >= 0):
                    p.add(f"{path}.n", f"bad value {r.get('n')!r}")
        if isinstance(per, dict) and isinstance(per.get("n_stocks"), int) and per["n_stocks"] != len(ps) and ps:
            p.add("per_stock", f"{len(ps)} rows but period.n_stocks={per['n_stocks']}")
    v = rep.get("verdict")
    if p.keys(v, "verdict", ("level", "title", "text", "bullets")):
        if v.get("level") not in VERDICT_LEVELS:
            p.add("verdict.level", f"unexpected {v.get('level')!r}")
        p.text(v, "title", "verdict")
        p.text(v, "text", "verdict")
        if not (isinstance(v.get("bullets"), list) and all(isinstance(x, str) for x in v["bullets"])):
            p.add("verdict.bullets", "expected list of strings")
    notes = rep.get("notes")
    if not (isinstance(notes, list) and notes and all(isinstance(x, str) for x in notes)):
        p.add("notes", "expected non-empty list of strings")
    elif not any("存活者" in x for x in notes):
        p.add("notes", "survivorship-bias note (存活者偏差) missing")
    return p.items


if __name__ == "__main__":  # python tests/schema_check.py forecast file.json
    fns = {"status": check_status, "search": check_search, "history": check_history, "forecast": check_forecast,
           "recommendations": check_recommendations, "backtest": check_backtest, "universe": check_universe_report,
           "evidence": check_evidence}
    if len(sys.argv) != 3 or sys.argv[1] not in fns:
        print(__doc__)
        sys.exit(2)
    with open(sys.argv[2], "rb") as fh:
        problems = fns[sys.argv[1]](strict_loads(fh.read()))
    print("\n".join(problems) or "OK")
    sys.exit(1 if problems else 0)
