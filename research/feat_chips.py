"""Chip / valuation features C1-C9, V1-V3 (PLAN.md section 5, P1b).  Definitions are pre-registered and fixed.

====  ================  ============================================================================  =========
id    name              definition (``t`` = decision day, rows = panel trading days)                  expected
====  ================  ============================================================================  =========
C1    fi_net_5          sum_5 Foreign_Investor (buy-sell, shares) / sum_5 volume (shares)              +
C2    fi_net_20         same, 20 rows                                                                  +
C3    it_net_5          sum_5 Investment_Trust net / sum_5 volume                                      +
C4    it_net_20         same, 20 rows                                                                  +
C5    it_streak         consecutive Investment_Trust net-buy (+n) / net-sell (-n) days, cut at +-10    +
C6    dealer_net_5      sum_5 Dealer_self net (no hedging book) / sum_5 volume                         ?
C7    margin_chg_20     (margin balance[t] - [t-20]) * 1000 / (20 * vol20[t])                          -
C8    short_chg_20      (short balance[t] - [t-20]) * 1000 / (20 * vol20[t])                           ?
C9    fi_hold_chg_20    foreign holding ratio[t-1] - [t-21]  (percentage points, ONE EXTRA LAG)        +
V1    ey                1 / PER   (PER <= 0 or missing -> NaN)                                         ? (control)
V2    bp                1 / PBR   (PBR <= 0 or missing -> NaN)                                         ? (control)
V3    dy                dividend yield (%)                                                             ? (control)
====  ================  ============================================================================  =========

Units: FinMind institutional rows are in shares, margin / short balances are in lots (the parser ``finmind.parse_margin``
already multiplies by 1000, so every balance below is in SHARES).  ``vol20[t]`` = mean of the last 20 rows' volume (shares).

Point-in-time (PLAN.md section 3): institutional flows, margin balances and PER of day ``t`` are usable on day ``t``; the
foreign holding ratio only up to ``t-1`` (C9 lags the aligned level series by one extra row).  Every feature row ``t`` is a
function of source rows dated ``<= cal[t]`` (and panel rows ``<= t``) only -- nothing is ever filled from the future.

Alignment to ``panel.cal`` (explicit, conservative missing-value rules)
-----------------------------------------------------------------------
* FLOW series (institutional net buy/sell): exact date match only.  A day without a source row is NaN and is NOT forward
  filled -- a stale flow would be counted twice in a rolling sum.  (PLAN.md says "ffill at most 5 days"; that is applied to
  LEVEL series only.  Honest deviation, see the report.)
* LEVEL series (margin / short balance, foreign holding ratio, PER, PBR, dividend yield): the latest source row dated
  ``<= cal[t]`` is carried forward for at most ``MAX_FFILL = 5`` panel rows (rows, i.e. trading days of the panel calendar);
  older than that -> NaN.  Source dates before ``cal[0]`` are ignored.  Rows whose value is blank are skipped (the ffill
  then bridges them, within the same 5-row limit).
* ROLLING FLOW WINDOWS (C1-C4, C6, vol20): a window of ``w`` rows needs at least ``ceil(0.8 * w)`` rows in which BOTH the
  flow and the volume are finite (4 of 5, 16 of 20); the ratio is computed over exactly those rows (numerator and
  denominator use the same rows), otherwise NaN.  A window can only contain rows ``<= t``.
* C5 streak: a day with ``net > 0`` extends a positive run (or starts one at +1), ``net < 0`` extends a negative run (or
  starts one at -1), ``net == 0`` BREAKS the run (value 0), and a day with a missing flow breaks it as well (value NaN, the
  next valid day restarts from +-1 -- it never peeks at what came before the gap or after).  Output clipped to [-10, 10].
* C7/C8/C9 differences need both levels finite.  PER <= 0, PBR <= 0 (loss makers / missing) -> V1/V2 NaN.
* A stock without a FinMind file (or a dataset that is empty) has NaN for every feature of that dataset.

Cross-sectional ranks: :func:`rank_features` maps each feature, every day, to [-0.5, 0.5] inside the universe mask
(``harness.xs_rank``, missing -> 0).  Raw values and ranks are both available.
"""
from __future__ import annotations

import math

import numpy as np

from backend.altdata import finmind as FM

from .harness import xs_rank
from .panel import Panel

MAX_FFILL = 5            # rows
MIN_FRAC = 0.8           # share of a rolling window that must be observed
STREAK_CAP = 10

# (id, name, expected sign; 0 = "?" / no expectation)
FEATURES = [("C1", "fi_net_5", +1), ("C2", "fi_net_20", +1), ("C3", "it_net_5", +1), ("C4", "it_net_20", +1),
            ("C5", "it_streak", +1), ("C6", "dealer_net_5", 0), ("C7", "margin_chg_20", -1), ("C8", "short_chg_20", 0),
            ("C9", "fi_hold_chg_20", +1), ("V1", "ey", 0), ("V2", "bp", 0), ("V3", "dy", 0)]
NAMES = [n for _, n, _ in FEATURES]
DATASETS = ("inst", "margin", "hold", "per")
_PARSERS = {"inst": FM.parse_institutional, "margin": FM.parse_margin, "hold": FM.parse_shareholding, "per": FM.parse_per}


# ============================================================================= alignment
def _series(parsed: dict | None, key: str):
    """``(dates datetime64[D] ascending, values float)`` of the finite entries of ``parsed[date][key]`` (or two empty arrays)."""
    if not parsed:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float)
    ds, vs = [], []
    for d, rec in parsed.items():
        v = rec.get(key) if isinstance(rec, dict) else None
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        try:
            ds.append(np.datetime64(str(d)[:10], "D"))
        except ValueError:
            continue
        vs.append(v)
    if not ds:
        return np.array([], dtype="datetime64[D]"), np.array([], dtype=float)
    order = np.argsort(np.array(ds, dtype="datetime64[D]"), kind="stable")
    return np.array(ds, dtype="datetime64[D]")[order], np.array(vs, dtype=float)[order]


def align_flow(dates, vals, cal) -> np.ndarray:
    """Exact-date alignment onto ``cal`` -> ``(T,)``; days without a source row are NaN (no filling of any kind)."""
    cal = np.asarray(cal).astype("datetime64[D]")
    out = np.full(len(cal), np.nan)
    if len(dates) == 0 or len(cal) == 0:
        return out
    pos = np.searchsorted(cal, dates)
    ok = pos < len(cal)
    ok[ok] &= cal[pos[ok]] == dates[ok]
    out[pos[ok]] = np.asarray(vals, float)[ok]
    return out


def align_level(dates, vals, cal, max_ffill: int = MAX_FFILL) -> np.ndarray:
    """Level alignment onto ``cal`` -> ``(T,)``: latest source row dated ``<= cal[t]``, carried at most ``max_ffill`` rows.

    Backward-looking only: the source row used for row ``t`` has ``date <= cal[t]``.  Source rows before ``cal[0]`` are ignored.
    """
    cal = np.asarray(cal).astype("datetime64[D]")
    T = len(cal)
    out = np.full(T, np.nan)
    if len(dates) == 0 or T == 0:
        return out
    dates = np.asarray(dates).astype("datetime64[D]")
    vals = np.asarray(vals, float)
    keep = dates >= cal[0]
    dates, vals = dates[keep], vals[keep]
    if len(dates) == 0:
        return out
    idx = np.searchsorted(dates, cal, side="right") - 1                    # last source row with date <= cal[t]
    row_of_src = np.searchsorted(cal, dates, side="right") - 1             # last panel row with cal <= source date
    ok = idx >= 0
    gap = np.arange(T) - row_of_src[np.maximum(idx, 0)]
    ok &= gap <= int(max_ffill)
    out[ok] = vals[idx[ok]]
    return out


# ============================================================================= rolling helpers
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """``out[t] = x[t-k]`` (NaN for ``t < k``); ``k >= 0`` only -- never looks forward."""
    if k < 0:
        raise ValueError("negative shift would look into the future")
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:x.shape[0] - k] if k else x
    return out


def rolling_sum(x: np.ndarray, w: int):
    """``(sum, count)`` over rows ``t-w+1 .. t`` of the finite entries of ``x`` (exact, no cumsum drift)."""
    acc = np.zeros(x.shape)
    cnt = np.zeros(x.shape, dtype=int)
    for k in range(int(w)):
        xs = _shift(x, k)
        fin = np.isfinite(xs)
        acc += np.where(fin, xs, 0.0)
        cnt += fin
    return acc, cnt


def min_obs(w: int) -> int:
    return int(math.ceil(MIN_FRAC * w - 1e-9))


def flow_ratio(net: np.ndarray, vol: np.ndarray, w: int) -> np.ndarray:
    """``sum_w net / sum_w vol`` over the rows where both are finite; NaN if fewer than ``min_obs(w)`` such rows or sum vol <= 0."""
    both = np.isfinite(net) & np.isfinite(vol)
    sn, cnt = rolling_sum(np.where(both, net, np.nan), w)
    sv, _ = rolling_sum(np.where(both, vol, np.nan), w)
    with np.errstate(all="ignore"):
        r = sn / sv
    return np.where((cnt >= min_obs(w)) & (sv > 0) & np.isfinite(r), r, np.nan)


def vol_mean(vol: np.ndarray, w: int = 20) -> np.ndarray:
    """Trailing mean volume (shares) over ``w`` rows; NaN with fewer than ``min_obs(w)`` finite rows."""
    s, c = rolling_sum(vol, w)
    with np.errstate(all="ignore"):
        m = s / c
    return np.where(c >= min_obs(w), m, np.nan)


def streak(net: np.ndarray, cap: int = STREAK_CAP) -> np.ndarray:
    """Signed run length of the sign of ``net`` per column (see module docstring), clipped to ``[-cap, cap]``.

    ``net > 0`` extends/starts a buying run, ``net < 0`` a selling run, ``net == 0`` breaks the run (0), NaN breaks it
    (output NaN and the next valid day restarts at +-1).
    """
    T, S = net.shape
    out = np.full((T, S), np.nan)
    run = np.zeros(S)
    for t in range(T):
        x = net[t]
        fin = np.isfinite(x)
        pos, neg = fin & (x > 0), fin & (x < 0)
        run = np.where(pos, np.where(run > 0, run + 1, 1.0), np.where(neg, np.where(run < 0, run - 1, -1.0), 0.0))
        out[t] = np.where(fin, np.clip(run, -cap, cap), np.nan)
        run = np.minimum(np.maximum(run, -cap), cap)        # keep the state bounded; run lengths above the cap are equivalent
    return out


# ============================================================================= features
def chip_features(panel: Panel, raw: dict) -> dict:
    """``{name: (T, S)}`` raw (un-ranked) values of C1-C9, V1-V3.

    ``raw``: ``{code: {"inst": parsed, "margin": parsed, "hold": parsed, "per": parsed}}`` with the output of the
    ``finmind.parse_*`` functions (missing code / dataset allowed -> NaN columns).  Row ``t`` only depends on source rows
    dated ``<= cal[t]`` and on panel rows ``<= t``.
    """
    T, S = panel.T, panel.S
    cal = panel.cal
    vol = np.where(panel.has_bar, panel.volume, np.nan)
    nan = lambda: np.full((T, S), np.nan)    # noqa: E731
    fnet, tnet, dnet = nan(), nan(), nan()                       # flows (shares)
    mbal, sbal, hold, per, pbr, dyl = nan(), nan(), nan(), nan(), nan(), nan()
    for j, code in enumerate(panel.codes):
        r = raw.get(str(code)) or {}
        inst = r.get("inst")
        for dst, key in ((fnet, "foreign_net"), (tnet, "trust_net"), (dnet, "dealer_self_net")):
            dst[:, j] = align_flow(*_series(inst, key), cal)
        mg = r.get("margin")
        mbal[:, j] = align_level(*_series(mg, "margin_balance"), cal)
        sbal[:, j] = align_level(*_series(mg, "short_balance"), cal)
        hold[:, j] = align_level(*_series(r.get("hold"), "foreign_ratio"), cal)
        pr = r.get("per")
        per[:, j] = align_level(*_series(pr, "per"), cal)
        pbr[:, j] = align_level(*_series(pr, "pbr"), cal)
        dyl[:, j] = align_level(*_series(pr, "dividend_yield"), cal)

    out = {}
    out["fi_net_5"] = flow_ratio(fnet, vol, 5)
    out["fi_net_20"] = flow_ratio(fnet, vol, 20)
    out["it_net_5"] = flow_ratio(tnet, vol, 5)
    out["it_net_20"] = flow_ratio(tnet, vol, 20)
    out["it_streak"] = streak(tnet)
    out["dealer_net_5"] = flow_ratio(dnet, vol, 5)
    den = 20.0 * vol_mean(vol, 20)
    den = np.where(den > 0, den, np.nan)
    with np.errstate(all="ignore"):
        out["margin_chg_20"] = (mbal - _shift(mbal, 20)) / den
        out["short_chg_20"] = (sbal - _shift(sbal, 20)) / den
    h1 = _shift(hold, 1)                                         # the extra lag: level known at t-1, compared with t-21
    out["fi_hold_chg_20"] = h1 - _shift(h1, 20)
    with np.errstate(all="ignore"):
        out["ey"] = np.where(per > 0, 1.0 / per, np.nan)
        out["bp"] = np.where(pbr > 0, 1.0 / pbr, np.nan)
    out["dy"] = dyl
    for k, v in out.items():
        out[k] = np.where(np.isfinite(v), v, np.nan)
    return {n: out[n] for n in NAMES}


def rank_features(feats: dict, mask) -> dict:
    """Per-day cross-sectional rank of every feature to [-0.5, 0.5] inside ``mask`` (missing / outside the mask -> 0)."""
    return {k: xs_rank(v, mask, fill=0.0) for k, v in feats.items()}


# ============================================================================= loading / coverage
def load_raw(codes, datasets=DATASETS) -> dict:
    """Read ``data_cache/alt/finmind/<dataset>/<code>.json`` (read-only) and parse -> ``{code: {alias: parsed}}``.

    Missing / unreadable files are simply absent (the features of that stock are NaN).  Never downloads.
    """
    out = {}
    for c in codes:
        rec = {}
        for alias in datasets:
            try:
                payload = FM.load_dataset(alias, str(c))
            except (ValueError, OSError):
                payload = None
            if payload and payload.get("data"):
                parsed = _PARSERS[alias](payload["data"])
                if parsed:
                    rec[alias] = parsed
        out[str(c)] = rec
    return out


def coverage(feats: dict, panel: Panel, mask=None) -> dict:
    """Per feature: stocks with at least one finite value, first / last date with any value, and (``mask`` given) the mean
    share of the universe that has a value on a universe day."""
    res = {}
    for k, v in feats.items():
        fin = np.isfinite(v)
        any_t = fin.any(axis=1)
        d = {"n_stocks": int(fin.any(axis=0).sum()), "n_total": int(panel.S),
             "first": str(panel.cal[np.flatnonzero(any_t)[0]]) if any_t.any() else None,
             "last": str(panel.cal[np.flatnonzero(any_t)[-1]]) if any_t.any() else None}
        if mask is not None:
            m = np.asarray(mask, bool)
            rows = m.sum(axis=1) > 0
            share = (fin & m).sum(axis=1)[rows] / m.sum(axis=1)[rows] if rows.any() else np.array([])
            d["universe_share"] = float(share.mean()) if share.size else None
        res[k] = d
    return res
