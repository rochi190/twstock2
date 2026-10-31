"""Causal features for the dynamic linear model (every value at row ``i`` uses bars ``<= i`` only).

Design rules that guarantee the absence of look-ahead
-----------------------------------------------------
* Rolling sums are ``cumsum`` differences, smoothers are ``lfilter`` recursions, scalers are *expanding* statistics:
  all of them are prefix functions (the value at row ``i`` is identical whether or not later rows exist).  This is
  unit-tested by truncation.
* No centred windows, no global means/stds, no look-forward initial conditions.
* The market series is aligned to the stock calendar by "last market bar with date <= stock date" (forward fill from the
  past only).

Cleaning layer (real-data hygiene)
----------------------------------
Yahoo series occasionally contain unadjusted corporate actions (e.g. a -139 % log return in the adjusted close of 0050
on 2014-01-02) and zero-volume placeholder bars.  The TWSE/TPEx daily limit is +-10 % (+-7 % before 2015), so any
``|r| > 0.25`` is a data artefact and is set to 0; ``0.12 < |r| <= 0.25`` is clipped to +-0.12.  Models, features and
learning targets use the cleaned return path; the *realised* returns reported for evaluation are the raw ones.

Features (all in "volatility units" so that they are comparable across stocks and regimes, then expanding z-scored)
-------------------------------------------------------------------------------------------------------------------
``rev_1, rev_5``        1- and 5-day return / (sigma sqrt(w))                          short-term reversal
``mom_20, mom_60``      1- and 3-month return / (sigma sqrt(w))                        intermediate momentum
``mom_12_1``            return from t-252 to t-21 / (sigma sqrt(231))                  12-1 month momentum
``ma_gap``              ln(P / SMA20) / (sigma sqrt(w/3))                              distance from the 20-day average
``ma_cross``            ln(SMA20 / SMA60) / (sigma sqrt(20))                           trend direction
``rsi``                 Wilder RSI(14)                                                 oscillator (mean reversion)
``rvol_ratio``          0.5 ln(EWMA_0.8 / EWMA_0.98 of Parkinson variance)             range-volatility ratio
``vol_z``               ln-volume z-score vs its EWMA                                  abnormal volume
``mkt_ret, mkt_mom20``  market 1-day / 20-day return in market-vol units
``mkt_vol``             0.5 ln(EWMA_0.94 / EWMA_0.995 of market variance)             market volatility state
``idio_mom``            60-day sum of market-model residuals / (residual vol sqrt(60)) residual momentum
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

LIMIT_CLIP = 0.12
ARTIFACT_THRESHOLD = 0.25
PLACEHOLDER_TOL = 1e-4  # |ln(adj_t / adj_{t-1})| up to which a zero-volume, high == low bar still counts as "no price change"
Z_CLIP = 4.0
Z_PRIOR_COUNT = 60.0
VOL_PRIOR = 0.015 ** 2  # prior daily variance used to initialise the EWMAs
VOL_PRIOR_WEIGHT = 10.0

FEATURE_KEYS = ["rev_1", "rev_5", "mom_20", "mom_60", "mom_12_1", "ma_gap", "ma_cross", "rsi", "rvol_ratio",
                "vol_z", "mkt_ret", "mkt_mom20", "mkt_vol", "idio_mom"]
FEATURE_NAMES = {
    "rev_1": "1日反轉", "rev_5": "5日反轉", "mom_20": "1個月動能", "mom_60": "3個月動能",
    "mom_12_1": "12-1個月動能", "ma_gap": "均線乖離(20日)", "ma_cross": "均線多空(20/60日)",
    "rsi": "RSI(14)", "rvol_ratio": "區間波動比", "vol_z": "成交量異常", "mkt_ret": "大盤當日報酬",
    "mkt_mom20": "大盤1個月動能", "mkt_vol": "大盤波動狀態", "idio_mom": "特異動能(60日)"}
FEATURE_UNITS = {"rev_1": "pct", "rev_5": "pct", "mom_20": "pct", "mom_60": "pct", "mom_12_1": "pct", "ma_gap": "pct",
                 "ma_cross": "pct", "rsi": "level", "rvol_ratio": "ratio", "vol_z": "ratio", "mkt_ret": "pct",
                 "mkt_mom20": "pct", "mkt_vol": "pct", "idio_mom": "pct"}


# ----------------------------------------------------------------------------- helpers
def placeholder_mask(data) -> np.ndarray:
    """Bars that are NOT observations: trading halts and the filler bars Yahoo inserts on exchange holidays / data gaps.

    A placeholder has zero volume, ``high == low`` (hence ``open == close == high``) and an unchanged adjusted close
    (``|ln(adj_t / adj_{t-1})| <= PLACEHOLDER_TOL``: float noise from the dividend factor is tolerated, a real tick move
    is at least 2e-4).  Modelled as a real session such a bar is a zero return: the volatility filters decay although
    nothing was observed (after 60 halted sessions the one-day sigma of 2330 collapses from 1.7 % to 0.2 %) and the first
    traded bar afterwards carries the whole accumulated move.  The engine therefore removes these rows before fitting
    (``engine._drop_placeholders``).  Bars with volume but ``high == low`` (limit-locked sessions) and bars whose price
    moved are never placeholders.  Index series are exempt (TAIEX reports no volume).

    A prefix function: row ``t`` is decided from rows ``t`` and ``t - 1`` only, so truncating the data never changes it.
    """
    n = int(getattr(data, "n", len(data.adjclose)))
    mask = np.zeros(n, dtype=bool)
    if n < 2 or getattr(data, "kind", "") == "index":
        return mask
    adj = np.asarray(data.adjclose, float)
    with np.errstate(all="ignore"):
        flat = np.abs(np.log(adj[1:] / adj[:-1])) <= PLACEHOLDER_TOL
        no_trade = (np.asarray(data.volume, float)[1:] <= 0.0) & (np.asarray(data.high, float)[1:] == np.asarray(data.low, float)[1:])
    mask[1:] = flat & no_trade
    return mask


def next_bar_after_weekend(dates) -> np.ndarray:
    """``(n,)`` bool: row ``i`` is a Friday, i.e. by the weekly calendar the next bar is the first bar after a weekend.

    Calendar time matters for the variance of the NEXT bar (audit M-04): the volatility models count every bar as one unit
    of time, but a bar that follows a weekend carries more variance than the average bar (French & Roll 1986 found about
    10-20 % more for US stocks; here ``E[z^2]`` of the one-day forecast is 1.11 vs 0.96 on the development rows and 1.15 vs
    0.97 on the untouched test rows, with a stock-bootstrap interval of 1.15-1.23 for the ratio on 178k development rows).

    A pure function of the row's OWN date, so it is a prefix function and truncation cannot change it.  The real date of
    the next bar is deliberately never looked at: a typhoon closure is not known in advance, and a scheduled holiday would
    need an exchange calendar that the data layer cannot supply for past years (the TWSE schedule endpoint only returns the
    current year).  Skipped weekdays (holidays) are therefore NOT modelled; the Friday rule is the part of the calendar
    effect that is known with certainty at the forecast origin.
    """
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7 == 4  # 1970-01-01 (day 0) was a Thursday, so Monday = 0 and Friday = 4


def clean_returns(r: np.ndarray) -> np.ndarray:
    """Model return path: data artefacts (|r| > 0.25) -> 0, other outliers clipped to +-0.12; non-finite -> 0."""
    r = np.nan_to_num(np.asarray(r, float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(np.abs(r) > ARTIFACT_THRESHOLD, 0.0, np.clip(r, -LIMIT_CLIP, LIMIT_CLIP))


def ewma(x: np.ndarray, lam: float, init: float | None = None) -> np.ndarray:
    """y_t = lam y_{t-1} + (1 - lam) x_t with y_{-1} = init (default x_0 -> y_0 = x_0)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return x.copy()
    y0 = x[0] if init is None else init
    y, _ = lfilter([1.0 - lam], [1.0, -lam], x, zi=[lam * y0])
    return y


def ewma_variance(r: np.ndarray, lam: float, v_prior: float = VOL_PRIOR, weight: float = VOL_PRIOR_WEIGHT) -> np.ndarray:
    """Normalised EWMA of squared returns, ``v_i = sum_s lam^(i-s) r_s^2 / sum_s lam^(i-s)``, with a prior of ``weight``
    pseudo-observations at variance ``v_prior`` placed at row 0 (row 0 itself carries no return).

    ``v_i`` includes the return of row ``i`` and is therefore the forecast of the next day's variance made at ``i``.
    """
    n = len(r)
    num_in = np.asarray(r, float) ** 2
    den_in = np.ones(n)
    num_in[0] = weight * v_prior
    den_in[0] = weight
    num = lfilter([1.0], [1.0, -lam], num_in)
    den = lfilter([1.0], [1.0, -lam], den_in)
    return num / den


def ffill(x: np.ndarray, fill: float) -> np.ndarray:
    """Causal forward fill of NaNs (leading NaNs -> ``fill``)."""
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(len(x), fill)
    idx = np.where(ok, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, x[np.maximum(idx, 0)], fill)
    return out


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing window sum over rows i-w+1..i (NaN for i < w-1) via cumsum differences (prefix function)."""
    c = np.cumsum(np.asarray(x, float))
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = c[w - 1:]
        out[w:] -= c[:-w]
    return out


def lagged_diff(x: np.ndarray, w: int) -> np.ndarray:
    """x_i - x_{i-w} (NaN for i < w)."""
    out = np.full(len(x), np.nan)
    if len(x) > w:
        out[w:] = x[w:] - x[:-w]
    return out


def align_market_logprice(dates: np.ndarray, market) -> np.ndarray | None:
    """Log market close on the stock's calendar: last market bar dated <= the stock date (NaN before the market starts)."""
    if market is None or getattr(market, "n", 0) == 0:
        return None
    idx = np.searchsorted(market.dates, dates, side="right") - 1
    lm = np.full(len(dates), np.nan)
    ok = idx >= 0
    lm[ok] = np.log(market.adjclose[idx[ok]])
    return lm


def expanding_zscore(f: np.ndarray, prior_count: float = Z_PRIOR_COUNT, clip: float = Z_CLIP) -> np.ndarray:
    """Expanding (prefix) z-score with a (0, 1) prior of ``prior_count`` pseudo-observations; NaN -> 0."""
    f = np.asarray(f, float)
    ok = np.isfinite(f)
    x = np.where(ok, f, 0.0)
    n = np.cumsum(ok.astype(float))
    s1 = np.cumsum(x)
    s2 = np.cumsum(x * x)
    mean = s1 / (n + prior_count)
    m2 = (s2 + prior_count) / (n + prior_count)
    sd = np.sqrt(np.maximum(m2 - mean * mean, 1e-6))
    z = np.clip((x - mean) / sd, -clip, clip)
    return np.where(ok, z, 0.0)


@dataclass
class FeatureSet:
    """Everything the engine needs, aligned with the rows of the price data (all causal)."""

    keys: list
    X: np.ndarray  # (n, p) expanding z-scores (0 where unavailable)
    raw: np.ndarray  # (n, p) natural-unit display values (NaN where unavailable)
    rc: np.ndarray  # (n,) cleaned daily log returns (rc[0] = 0)
    cs: np.ndarray  # (n,) cumulative cleaned log return (cs[0] = 0)
    sigma_ref: np.ndarray  # (n,) reference daily vol: sqrt(0.5 EWMA_0.94 + 0.5 EWMA_0.99) of squared returns
    v97: np.ndarray  # (n,) EWMA_0.97 variance (naive component)
    drift: np.ndarray  # (n,) shrunk expanding mean of the daily log return
    logvol: np.ndarray  # (n,) log Parkinson range volatility (forward-filled, floored)
    has_market: bool

    @property
    def p(self) -> int:
        return len(self.keys)


def shrunk_drift(rc: np.ndarray, mu0: float = 0.0002, tau0: float = 0.0004) -> np.ndarray:
    """Posterior mean of the daily log drift: normal prior N(mu0, tau0^2), data variance = expanding mean of r^2.

    ``drift_i = (N0 mu0 + sum_{s<=i} r_s) / (N0 + i)`` with ``N0 = v_i / tau0^2`` pseudo-days (about 1400 for a typical
    stock), so a short or lucky history cannot create a large forecast drift.  The prior mean 0.0002 / day is about
    5 % a year and ``tau0 = 0.0004`` (about 10 % a year) is the cross-sectional spread of true expected returns one can
    plausibly believe a priori; both are fixed, nothing is tuned on a backtest.
    """
    n = len(rc)
    cnt = np.arange(n, dtype=float)  # rows 1..i carry returns
    s1 = np.cumsum(rc)
    s2 = np.cumsum(rc * rc)
    v = np.where(cnt > 0, s2 / np.maximum(cnt, 1.0), VOL_PRIOR)
    v = (VOL_PRIOR_WEIGHT * VOL_PRIOR + cnt * v) / (VOL_PRIOR_WEIGHT + cnt)
    n0 = v / (tau0 * tau0)
    return (n0 * mu0 + s1) / (n0 + cnt)


def parkinson_logvol(high: np.ndarray, low: np.ndarray, floor: float = 5e-4):
    """(log Parkinson daily volatility forward-filled, Parkinson variance forward-filled).

    ``sigma_P = ln(H/L) / (2 sqrt(ln 2))``.  Bars with ``H <= L`` (no trading / placeholder) are treated as missing.
    """
    hi = np.maximum(high, low)
    lo = np.minimum(high, low)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.log(hi / lo)
    sp = np.where(rng > 0.0, rng / (2.0 * np.sqrt(np.log(2.0))), np.nan)
    sp = np.where(np.isfinite(sp), np.maximum(sp, floor), np.nan)
    sp_f = ffill(sp, 0.01)
    return np.log(sp_f), sp_f ** 2


def build_features(data, market=None) -> FeatureSet:
    """Compute every causal feature for all rows of ``data`` (optionally using the ``market`` index)."""
    n = data.n
    adj = np.asarray(data.adjclose, float)
    r_raw = np.zeros(n)
    if n > 1:
        r_raw[1:] = np.diff(np.log(adj))
    rc = clean_returns(r_raw)
    rc[0] = 0.0
    cs = np.cumsum(rc)

    v94 = ewma_variance(rc, 0.94)
    v97 = ewma_variance(rc, 0.97)
    v99 = ewma_variance(rc, 0.99)
    sig = np.sqrt(0.5 * v94 + 0.5 * v99)
    drift = shrunk_drift(rc)

    price = np.exp(cs)  # cleaned price level (start = 1); cumulative |log return| is far below the overflow range

    f = {}
    raw = {}

    def vol_units(rr, w):
        return np.clip(rr / (sig * np.sqrt(w)), -5.0, 5.0)

    # --- returns based
    r1 = rc.copy()
    r1[0] = np.nan
    v94_lag = np.concatenate(([VOL_PRIOR], v94[:-1]))  # vol forecast made BEFORE the return was observed
    f["rev_1"] = np.clip(r1 / np.sqrt(np.maximum(v94_lag, 1e-12)), -5.0, 5.0)  # standardised surprise
    raw["rev_1"] = np.expm1(r1)
    for key, w in (("rev_5", 5), ("mom_20", 20), ("mom_60", 60)):
        rr = lagged_diff(cs, w)
        f[key] = vol_units(rr, w)
        raw[key] = np.expm1(rr)
    rr = np.full(n, np.nan)
    if n > 252:
        rr[252:] = cs[231:n - 21] - cs[:n - 252]  # cs[t-21] - cs[t-252] for t >= 252
    f["mom_12_1"] = vol_units(rr, 231)
    raw["mom_12_1"] = np.expm1(rr)

    # --- price-level based
    s20 = rolling_sum(price, 20) / 20.0
    s60 = rolling_sum(price, 60) / 60.0
    gap = np.log(price / s20)
    f["ma_gap"] = np.clip(gap / (sig * np.sqrt(20.0 / 3.0)), -5.0, 5.0)
    raw["ma_gap"] = price / s20 - 1.0
    cross = np.log(s20 / s60)
    f["ma_cross"] = np.clip(cross / (sig * np.sqrt(20.0)), -5.0, 5.0)
    raw["ma_cross"] = s20 / s60 - 1.0

    # --- RSI(14), Wilder smoothing
    up = np.maximum(rc, 0.0)
    dn = np.maximum(-rc, 0.0)
    a = 1.0 / 14.0
    init = 0.01
    avg_up, _ = lfilter([a], [1.0, -(1.0 - a)], up, zi=[(1.0 - a) * init])
    avg_dn, _ = lfilter([a], [1.0, -(1.0 - a)], dn, zi=[(1.0 - a) * init])
    rsi = 100.0 * avg_up / np.maximum(avg_up + avg_dn, 1e-12)
    rsi[:14] = np.nan
    f["rsi"] = (rsi - 50.0) / 25.0
    raw["rsi"] = rsi

    # --- range volatility ratio and abnormal volume
    logvol, pv = parkinson_logvol(np.asarray(data.high, float), np.asarray(data.low, float))
    short = ewma_pv(pv, 0.80)
    long_ = ewma_pv(pv, 0.98)
    ratio = 0.5 * np.log(short / long_)
    ratio[:30] = np.nan
    f["rvol_ratio"] = np.clip(ratio, -2.0, 2.0)
    raw["rvol_ratio"] = np.exp(2.0 * ratio)  # short/long variance ratio
    vol = np.asarray(data.volume, float)
    lv = np.where(vol > 0, np.log(np.maximum(vol, 1.0)), np.nan)
    if np.isfinite(lv).sum() >= 30:
        lvf = ffill(lv, np.nan)
        first = np.where(np.isfinite(lvf))[0][0]
        lvf[:first] = lvf[first]
        m = ewma(lvf, 0.97)
        s2 = ewma((lvf - m) ** 2, 0.97)
        z = (lvf - m) / np.sqrt(np.maximum(s2, 1e-4))
        z = np.where(np.isfinite(lv), z, 0.0)
        z[:30] = np.nan
        f["vol_z"] = np.clip(z, -4.0, 4.0)
        raw["vol_z"] = np.where(np.isfinite(lv), np.exp(lvf - m), np.nan)
    else:
        f["vol_z"] = np.full(n, np.nan)
        raw["vol_z"] = np.full(n, np.nan)

    # --- market features
    lm = align_market_logprice(data.dates, market)
    has_market = lm is not None and np.isfinite(lm).sum() > 60
    if has_market:
        rm_raw = np.zeros(n)
        d = np.diff(lm)
        rm_raw[1:] = np.where(np.isfinite(d), d, 0.0)
        rm = clean_returns(rm_raw)
        rm[0] = 0.0
        cm = np.cumsum(rm)
        mv94 = ewma_variance(rm, 0.94, v_prior=0.01 ** 2)
        mv99 = ewma_variance(rm, 0.99, v_prior=0.01 ** 2)
        mv995 = ewma_variance(rm, 0.995, v_prior=0.01 ** 2)
        msig = np.sqrt(0.5 * mv94 + 0.5 * mv99)
        m1 = rm.copy()
        m1[0] = np.nan
        mv94_lag = np.concatenate(([0.01 ** 2], mv94[:-1]))
        f["mkt_ret"] = np.clip(m1 / np.sqrt(np.maximum(mv94_lag, 1e-12)), -5.0, 5.0)
        raw["mkt_ret"] = np.expm1(m1)
        rr = lagged_diff(cm, 20)
        f["mkt_mom20"] = np.clip(rr / (msig * np.sqrt(20.0)), -5.0, 5.0)
        raw["mkt_mom20"] = np.expm1(rr)
        mvr = 0.5 * np.log(mv94 / mv995)
        mvr[:60] = np.nan
        f["mkt_vol"] = np.clip(mvr, -2.0, 2.0)
        raw["mkt_vol"] = np.sqrt(mv94 * 245.0)  # annualised market volatility (display)
        # idiosyncratic residual momentum: residuals use the beta known one row earlier (causal)
        cov = ewma(rc * rm, 0.99, init=0.0)
        mvar = ewma(rm * rm, 0.99, init=0.01 ** 2)
        beta = np.clip(cov / np.maximum(mvar, 1e-10), -1.0, 4.0)
        beta_lag = np.concatenate(([1.0], beta[:-1]))
        resid = rc - beta_lag * rm
        e2 = ewma(resid * resid, 0.97, init=VOL_PRIOR)
        rs = rolling_sum(resid, 60)
        im = rs / (np.sqrt(np.maximum(e2, 1e-12)) * np.sqrt(60.0))
        f["idio_mom"] = np.clip(im, -5.0, 5.0)
        raw["idio_mom"] = np.expm1(rs)
    else:
        for key in ("mkt_ret", "mkt_mom20", "mkt_vol", "idio_mom"):
            f[key] = np.full(n, np.nan)
            raw[key] = np.full(n, np.nan)

    X = np.column_stack([expanding_zscore(f[k]) for k in FEATURE_KEYS])
    R = np.column_stack([raw[k] for k in FEATURE_KEYS])
    return FeatureSet(keys=list(FEATURE_KEYS), X=X, raw=R, rc=rc, cs=cs, sigma_ref=sig, v97=v97, drift=drift,
                      logvol=logvol, has_market=bool(has_market))


def ewma_pv(pv: np.ndarray, lam: float) -> np.ndarray:
    """Normalised EWMA of the Parkinson variance (prefix function; weights sum to one from the first row)."""
    num = lfilter([1.0], [1.0, -lam], pv)
    den = lfilter([1.0], [1.0, -lam], np.ones(len(pv)))
    return num / den
