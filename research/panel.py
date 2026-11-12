"""Aligned (T, S) price panel for cross-sectional research (PLAN.md sections 5-6, P0).

Everything is on the UNION trading calendar of the stocks.  A stock without a bar on a calendar day has ``has_bar == False``
and NaN prices there.  All derived objects are documented with the exact information set they use:

* ``universe_mask``      row ``t`` uses bars ``<= t`` only (prefix function; poisoning the future cannot change it).
* ``forward_open_return`` row ``t`` is the label ``ln(adjO[t+1+h] / adjO[t+1])`` -- it deliberately looks into the future
  (it is the target), so it must never be used as a feature.
* ``untradable_open``    row ``t`` uses the open of ``t+1`` (execution information: an order at the open that cannot be
  filled because the stock opened locked at the upper limit).

Persistence: ``save_panel`` / ``load_panel`` use ``np.savez_compressed`` at ``<cache_dir>/research/panel_v1.npz``
(``data_cache/`` is git-ignored).
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend import data as D

log = logging.getLogger(__name__)

PANEL_VERSION = 1
LIMIT_UP_GAP = 0.095     # open >= +9.5 % above the previous close: locked at the limit, a buy order is not filled


# ============================================================================= container
@dataclass
class Panel:
    """Stocks x days matrices.  ``(T, S)`` unless noted; prices/volumes are NaN where ``has_bar`` is False."""

    cal: np.ndarray                  # (T,) datetime64[D], ascending, unique (union trading calendar)
    codes: np.ndarray                # (S,) str
    adj_open: np.ndarray             # adjusted open  (PriceData.adj_open = open * adjclose / close)
    adj_close: np.ndarray            # adjusted close
    volume: np.ndarray               # shares
    value: np.ndarray                # traded value = raw close * volume (TWD)
    has_bar: np.ndarray              # bool
    bars_before: np.ndarray = None   # (S,) int: number of bars of each stock strictly before cal[0] (for the 500-bar rule)
    kinds: np.ndarray = None         # (S,) str, 'stock' | 'etf' (selects the transaction-cost schedule)

    def __post_init__(self):
        self.cal = np.asarray(self.cal).astype("datetime64[D]")
        self.codes = np.asarray(self.codes).astype(str)
        T, S = len(self.cal), len(self.codes)
        for name in ("adj_open", "adj_close", "volume", "value"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (T, S):
                raise ValueError(f"{name}: shape {a.shape} != {(T, S)}")
            setattr(self, name, a)
        self.has_bar = np.asarray(self.has_bar, dtype=bool)
        if self.has_bar.shape != (T, S):
            raise ValueError(f"has_bar: shape {self.has_bar.shape} != {(T, S)}")
        if T > 1 and not np.all(np.diff(self.cal.astype(np.int64)) > 0):
            raise ValueError("cal must be strictly ascending")
        self.bars_before = np.zeros(S, dtype=np.int64) if self.bars_before is None else np.asarray(self.bars_before, dtype=np.int64)
        self.kinds = np.full(S, "stock") if self.kinds is None else np.asarray(self.kinds).astype(str)
        if self.bars_before.shape != (S,) or self.kinds.shape != (S,):
            raise ValueError("bars_before / kinds must have shape (S,)")

    @property
    def T(self) -> int:
        return len(self.cal)

    @property
    def S(self) -> int:
        return len(self.codes)

    def copy(self) -> "Panel":
        return Panel(self.cal.copy(), self.codes.copy(), self.adj_open.copy(), self.adj_close.copy(), self.volume.copy(),
                     self.value.copy(), self.has_bar.copy(), self.bars_before.copy(), self.kinds.copy())

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_price_data(cls, datas, start: str = "2012-01-01") -> "Panel":
        """Align ``PriceData`` objects (full history each) onto the union calendar from ``start`` on.

        Bars before ``start`` are not stored, but they are counted in ``bars_before`` so that the "at least 500 bars"
        rule of the universe is not penalised by the choice of ``start`` (a count of past bars is point-in-time safe).
        """
        datas = list(datas)
        if not datas:
            raise ValueError("no price data")
        start64 = np.datetime64(start, "D")
        alld = np.unique(np.concatenate([np.asarray(d.dates).astype("datetime64[D]") for d in datas]))
        cal = alld[alld >= start64]
        T, S = len(cal), len(datas)
        ao, ac, vol, val = (np.full((T, S), np.nan) for _ in range(4))
        has = np.zeros((T, S), dtype=bool)
        before = np.zeros(S, dtype=np.int64)
        kinds = []
        for j, d in enumerate(datas):
            dates = np.asarray(d.dates).astype("datetime64[D]")
            before[j] = int((dates < start64).sum())
            kinds.append("etf" if getattr(d, "kind", "stock") == "etf" else "stock")
            if T == 0:
                continue
            pos = np.searchsorted(cal, dates)
            keep = (pos < T) & (dates >= start64)
            keep &= cal[np.minimum(pos, T - 1)] == dates
            r = np.flatnonzero(keep)
            p = pos[r]
            with np.errstate(all="ignore"):
                o_ = np.asarray(d.adj_open, float)[r]
                c_ = np.asarray(d.adjclose, float)[r]
                v_ = np.asarray(d.volume, float)[r]
                raw_c = np.asarray(d.close, float)[r]
            ok = np.isfinite(o_) & np.isfinite(c_) & (o_ > 0) & (c_ > 0)
            ao[p[ok], j], ac[p[ok], j] = o_[ok], c_[ok]
            vol[p[ok], j] = v_[ok]
            with np.errstate(all="ignore"):
                val[p[ok], j] = raw_c[ok] * v_[ok]
            has[p[ok], j] = True
        return cls(cal, np.array([str(d.symbol) for d in datas]), ao, ac, vol, val, has, before, np.array(kinds))


# ============================================================================= loading / building
def research_dir() -> Path:
    """``<cache_dir>/research`` (created on demand by the writers)."""
    return D.cache_dir() / "research"


def panel_path() -> Path:
    return research_dir() / f"panel_v{PANEL_VERSION}.npz"


def load_pool(path=None) -> list:
    """The candidate pool ``[{code, name, market, kind}, ...]`` (PLAN.md section 6: top-300 ordinary stocks by value)."""
    p = Path(path) if path is not None else research_dir() / "pool300.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_datas(codes, *, offline: bool = True) -> list:
    """``PriceData`` for ``codes`` with the still-running last bar removed (``backend.data.completed``).

    ``offline=True`` (default) reads the on-disk cache only (``peek_price_data``: reproducible, no network); codes that are
    not cached are skipped with a warning.  ``offline=False`` uses ``backend.data.get_price_data`` (may refresh / download).
    """
    out = []
    for c in codes:
        try:
            pd_ = D.peek_price_data(c) if offline else D.get_price_data(c)
        except D.DataError as exc:
            log.warning("%s: %s", c, exc)
            pd_ = None
        if pd_ is None or pd_.n == 0:
            log.warning("%s: no price data -- skipped", c)
            continue
        out.append(D.completed(pd_))
    return out


def build_panel(codes, start: str = "2012-01-01", *, offline: bool = True) -> Panel:
    """Panel of ``codes`` (see :func:`load_datas` for ``offline``) from ``start`` on, on the union calendar."""
    return Panel.from_price_data(load_datas(codes, offline=offline), start)


def save_panel(panel: Panel, path=None) -> Path:
    p = Path(path) if path is not None else panel_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, version=np.array(PANEL_VERSION), cal=panel.cal.astype("datetime64[D]").astype(np.int64), codes=panel.codes,
        adj_open=panel.adj_open, adj_close=panel.adj_close, volume=panel.volume, value=panel.value, has_bar=panel.has_bar,
        bars_before=panel.bars_before, kinds=panel.kinds)
    return p


def load_panel(path=None) -> Panel:
    p = Path(path) if path is not None else panel_path()
    with np.load(p, allow_pickle=False) as z:
        if int(z["version"]) != PANEL_VERSION:
            raise ValueError(f"panel version {int(z['version'])} != {PANEL_VERSION}")
        return Panel(z["cal"].astype("datetime64[D]"), z["codes"], z["adj_open"], z["adj_close"], z["volume"], z["value"],
                     z["has_bar"], z["bars_before"], z["kinds"])


def align_series(panel: Panel, data, series) -> np.ndarray:
    """Scatter a per-bar series of one stock (``len == data.n``) onto the panel calendar -> ``(T,)`` (NaN on days without a
    bar).  Used to align features computed on a stock's own bar index (e.g. ``build_features``) with the panel."""
    series = np.asarray(series, dtype=float)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    if len(series) != len(dates):
        raise ValueError("series length != number of bars")
    out = np.full(panel.T, np.nan)
    if panel.T == 0:
        return out
    pos = np.searchsorted(panel.cal, dates)
    keep = (pos < panel.T)
    keep &= panel.cal[np.minimum(pos, panel.T - 1)] == dates
    out[pos[keep]] = series[keep]
    return out


# ============================================================================= universe / labels / tradability
def _rolling_nanmedian(x: np.ndarray, w: int, min_obs: int, chunk: int = 256) -> np.ndarray:
    """Trailing-window (rows ``t-w+1 .. t``) median of the finite values; NaN when fewer than ``min_obs`` are finite."""
    T, S = x.shape
    out = np.full((T, S), np.nan)
    pad = np.vstack([np.full((w - 1, S), np.nan), x])
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        win = np.lib.stride_tricks.sliding_window_view(pad[a:b + w - 1], w, axis=0)   # (b-a, S, w)
        srt = np.sort(win, axis=2)                                                   # NaN sorts last
        k = np.isfinite(win).sum(axis=2)                                             # (b-a, S)
        lo = np.maximum((k - 1) // 2, 0)[..., None]
        hi = np.maximum(k // 2, 0)[..., None]
        med = 0.5 * (np.take_along_axis(srt, lo, axis=2)[..., 0] + np.take_along_axis(srt, hi, axis=2)[..., 0])
        out[a:b] = np.where(k >= min_obs, med, np.nan)
    return out


def universe_mask(panel: Panel, n: int = 150, lookback: int = 60, min_bars: int = 500, min_obs: int | None = None) -> np.ndarray:
    """``(T, S)`` bool: the day-``t`` universe of PLAN.md section 6.

    A stock is a member on day ``t`` iff (a) it has a bar on ``t``, (b) it has at least ``min_bars`` bars up to and including
    ``t`` (bars before the panel start are counted through ``bars_before``), (c) its median traded value (close x volume)
    over the last ``lookback`` calendar rows ``t-lookback+1 .. t`` -- computed from the bars it has there, at least
    ``min_obs`` (default ``lookback // 2``) of them -- is among the ``n`` largest of the eligible stocks of that day
    (ties: lower column index first).  Only data ``<= t`` is used.
    """
    if n < 1 or lookback < 1:
        raise ValueError("n and lookback must be >= 1")
    min_obs = lookback // 2 if min_obs is None else int(min_obs)
    cnt = panel.bars_before[None, :] + np.cumsum(panel.has_bar, axis=0)
    val = np.where(panel.has_bar, panel.value, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = _rolling_nanmedian(val, lookback, max(1, min_obs))
    elig = panel.has_bar & (cnt >= min_bars) & np.isfinite(med)
    score = np.where(elig, med, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(panel.S), order.shape), axis=1)
    return elig & (rank < n)


def forward_open_return(panel: Panel, h: int) -> np.ndarray:
    """``(T, S)`` label ``y_h[t] = ln(adjO[t+1+h] / adjO[t+1])`` (open-to-open from the entry open of day ``t+1``).

    NaN when either end has no bar (or a non-positive price) and for the last ``h + 1`` rows.  The start is NOT ``close[t]``:
    the overnight gap between ``t`` and ``t+1`` is not earnable by a decision taken after the ``t`` close (PLAN.md rule 3).
    """
    h = int(h)
    if h < 1:
        raise ValueError("h must be >= 1")
    T = panel.T
    y = np.full((T, panel.S), np.nan)
    if T > h + 1:
        o = np.where(panel.has_bar, panel.adj_open, np.nan)
        with np.errstate(all="ignore"):
            r = np.log(o[h + 1:] / o[1:T - h])
        y[:T - h - 1] = np.where(np.isfinite(r), r, np.nan)
    return y


def untradable_open(panel: Panel, threshold: float = LIMIT_UP_GAP) -> np.ndarray:
    """``(T, S)`` bool: the entry open of day ``t+1`` is ``>= +threshold`` above the last known close ``<= t``.

    Such a stock opened locked at the upper limit (the pre-2015 limit was 7 %: a lock below 9.5 % is NOT flagged -- the
    rule is the fixed one of PLAN.md section 5).  The long basket excludes these names.  The last row is always False.
    """
    T, S = panel.T, panel.S
    out = np.zeros((T, S), dtype=bool)
    if T < 2:
        return out
    c = np.where(panel.has_bar, panel.adj_close, np.nan)
    idx = np.where(np.isfinite(c), np.arange(T)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    last_c = np.where(idx >= 0, np.take_along_axis(c, np.maximum(idx, 0), axis=0), np.nan)
    o_next = np.where(panel.has_bar, panel.adj_open, np.nan)[1:]
    with np.errstate(all="ignore"):
        gap = o_next / last_c[:-1] - 1.0
    out[:-1] = np.isfinite(gap) & (gap >= threshold)
    return out
