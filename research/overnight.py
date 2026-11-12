"""P3b -- US overnight returns -> opening-gap model -> pre-market update of the 1-day close probability (PLAN.md section 7).

Spec: ``research/overnight_spec.json`` (frozen; its sha256 is the OOS ``spec_hash``).  Time conventions (all dates are Taipei trading
days ``S``; "row k" = the k-th bar of one stock after no-trade placeholder bars are dropped):

* ``gap[k]      = ln(adjO[k] / adjC[k-1])``   -- the opening gap of day ``S = dates[k]``;
* ``x[k]        =`` sum of the US log returns of SOX / TSM / NVDA whose US date ``D`` lies in ``[dates[k-1], dates[k])`` (strictly
  ``D < S``; ``align_us_to_tw`` on the stock's own dates, empty window -> 0.0);
* ``yhat[k]``   = prediction of ``gap[k]`` from an EWMA (half-life 120 bars) least-squares fit on the pairs ``(gap[j], x[j])``,
  ``j < k`` only, evaluated at ``x[k]`` (known before the open of ``S``);
* ``cond[k] = yhat[k] - ybar[k]`` the US-driven part (``ybar`` = EWMA mean of the same sample);
* walk-forward h=1 outputs ``mu0/sigma0/p0`` are produced at the close of bar ``k-1`` for the close return
  ``R[k] = ln(adjC[k]/adjC[k-1])`` -> stored at index ``k`` (``mu0[k] = wf.mu[k-1]``);
* the pre-market update only ever uses the PREDICTED conditional gap ``cond[k]``, never the realised ``gap[k]``.

No function here reads the OOS period unless ``period="oos"`` is passed to :func:`evaluate` together with the spec hash (the
harness guard raises otherwise and writes the LOOKS.md row).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr

from backend import config
from backend import data as D
from backend.altdata import usmarket as US
from backend.quant import engine as E
from backend.quant import features as QF

from . import harness as H
from .panel import Panel, forward_open_return, research_dir

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = Path(__file__).with_name("overnight_spec.json")


# ============================================================================= spec
def load_spec(path=None) -> dict:
    with open(Path(path) if path is not None else SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def spec_sha256(path=None) -> str:
    """sha256 (hex) of the spec file's bytes -- the ``spec_hash`` of the OOS run."""
    return hashlib.sha256(Path(path if path is not None else SPEC_PATH).read_bytes()).hexdigest()


SPEC = load_spec()
REGRESSORS = tuple(SPEC["us_regressors"]["names"])
HALFLIFE = float(SPEC["gap_model"]["ewma_halflife_bars"])
MIN_OBS = int(SPEC["gap_model"]["min_obs"])
SHOCK = SPEC["tradability"]["shock"]
DEV_END = np.datetime64(SPEC["periods"]["dev"][1], "D")
DEV_START = np.datetime64(SPEC["periods"]["dev"][0], "D")
MIN_BARS_BEFORE_DEV = int(SPEC["stocks"]["min_bars_before_dev"])
R2_CLIP = 0.95


# ============================================================================= EWMA regression (causal)
def ewma_regression(y, X, halflife: float = HALFLIFE, min_obs: int = MIN_OBS, ridge: float = 1e-10) -> dict:
    """Rolling EWMA-weighted least squares of ``y`` on ``[1, X]`` where the coefficients at row ``k`` use rows ``j < k`` only.

    The weight of row ``j`` at prediction row ``k`` is ``0.5 ** ((k - 1 - j) / halflife)`` (most recent usable row = 1).  Rows
    with a non-finite ``y`` or ``X`` are skipped (their weight simply does not appear).  Implemented with causal IIR filters
    (``scipy.signal.lfilter``) over the sufficient statistics, so the outputs at row ``k`` are bit-identical whatever the data
    at rows ``>= k`` (apart from ``X[k]`` itself, which enters only ``yhat[k]``).

    Returns a dict of arrays: ``beta (n, q)`` (q = 1 + p, intercept first), ``yhat`` (``[1, X[k]] . beta[k]``), ``ybar``
    (EWMA mean of ``y``), ``r2`` (in-sample EWMA R2 on rows < k, adjusted for ``n_eff``, clipped to [0, 0.95]), ``n_eff``,
    ``count`` (number of usable rows before k) and ``ok`` (enough rows and finite ``X[k]``).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    q = p + 1
    obs = np.isfinite(y) & np.isfinite(X).all(axis=1)
    lam = 0.5 ** (1.0 / float(halflife))
    Xa = np.column_stack([np.ones(n), np.where(obs[:, None], np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0)])
    Xa = np.where(obs[:, None], Xa, 0.0)
    yy = np.where(obs, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    iu = [(i, j) for i in range(q) for j in range(i, q)]
    ch = np.column_stack([Xa[:, i] * Xa[:, j] for i, j in iu] + [Xa[:, i] * yy for i in range(q)] + [yy * yy])
    w = obs.astype(float)
    A = lfilter([1.0], [1.0, -lam], ch, axis=0)
    A2 = lfilter([1.0], [1.0, -lam * lam], w)
    cnt = np.cumsum(w)
    # statistics available when predicting row k: everything up to k-1
    shift = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), a[:-1]], axis=0)   # noqa: E731
    A, A2, cnt = shift(A), shift(A2), shift(cnt)
    XtX = np.zeros((n, q, q))
    for c, (i, j) in enumerate(iu):
        XtX[:, i, j] = A[:, c]
        XtX[:, j, i] = A[:, c]
    Xty = A[:, len(iu):len(iu) + q]
    Syy = A[:, -1]
    s1 = XtX[:, 0, 0]
    ok_fit = cnt >= int(min_obs)
    beta = np.full((n, q), np.nan)
    idx = np.flatnonzero(ok_fit)
    if len(idx):
        M = XtX[idx] + ridge * np.einsum("ni,ij->nij", np.diagonal(XtX[idx], axis1=1, axis2=2), np.eye(q))
        try:
            beta[idx] = np.linalg.solve(M, Xty[idx][..., None])[..., 0]
        except np.linalg.LinAlgError:
            for r in idx:
                beta[r] = np.linalg.lstsq(XtX[r], Xty[r], rcond=None)[0]
    with np.errstate(all="ignore"):
        ybar = np.where(ok_fit, Xty[:, 0] / s1, np.nan)
        ssr = Syy - 2.0 * np.einsum("ni,ni->n", beta, Xty) + np.einsum("ni,nij,nj->n", beta, XtX, beta)
        sst = Syy - Xty[:, 0] ** 2 / s1
        r2 = 1.0 - ssr / sst
        n_eff = s1 ** 2 / A2
        adj = 1.0 - (1.0 - r2) * (n_eff - 1.0) / (n_eff - q)
        adj = np.clip(np.where(np.isfinite(adj), adj, np.nan), 0.0, R2_CLIP)
        Xk = np.column_stack([np.ones(n), X])
        yhat = np.einsum("ni,ni->n", beta, Xk)
    ok = ok_fit & np.isfinite(yhat) & np.isfinite(adj) & np.isfinite(X).all(axis=1)
    return {"beta": beta, "yhat": np.where(ok, yhat, np.nan), "ybar": np.where(ok, ybar, np.nan), "r2": np.where(ok, adj, np.nan),
            "n_eff": n_eff, "count": cnt, "ok": ok}


def ewma_sigma(r, lam: float = 0.94, seed_n: int = 20, min_obs: int = 60) -> np.ndarray:
    """EWMA volatility forecast ``sigma[k]`` that uses ``r[j]`` for ``j < k`` only (NaNs skipped).

    Variance seed = mean of the first ``seed_n`` squared finite returns; afterwards ``v = lam v + (1-lam) r^2`` for every finite
    return; ``sigma[k]`` is NaN until ``min_obs`` finite returns have been seen.
    """
    r = np.asarray(r, float)
    out = np.full(len(r), np.nan)
    v, cnt, seed = None, 0, []
    for k in range(len(r)):
        if v is not None and cnt >= min_obs:
            out[k] = math.sqrt(v)
        x = r[k]
        if math.isfinite(x):
            cnt += 1
            if v is None:
                seed.append(x * x)
                if len(seed) >= seed_n:
                    v = float(np.mean(seed))
            else:
                v = lam * v + (1.0 - lam) * x * x
    return out


# ============================================================================= US data / alignment
def load_us(names=REGRESSORS, *, now=None, cache_dir=None) -> dict:
    """``{name: (dates, log_ret)}`` from the cached US series (``usmarket.daily_log_returns``)."""
    return {nm: US.daily_log_returns(nm, now=now, cache_dir=cache_dir) for nm in names}


def align_regressors(us: dict, dates, names=REGRESSORS, empty: float = 0.0) -> np.ndarray:
    """``(len(dates), len(names))`` US returns usable at the open of each date (``D < S``; empty window -> ``empty``)."""
    return np.column_stack([US.align_us_to_tw(us[nm][0], us[nm][1], dates, empty=empty) for nm in names])


# ============================================================================= per-stock series
@dataclass
class Stock:
    """One stock on its own placeholder-free bar index (all arrays length ``n``)."""
    code: str
    data: object                     # PriceData (placeholder-free)
    gap: np.ndarray
    intraday: np.ndarray
    R: np.ndarray                    # ln(adjC[k]/adjC[k-1])
    X: np.ndarray                    # (n, 3) aligned US returns
    reg: dict = field(default_factory=dict)
    mu0: np.ndarray = None           # walk-forward h=1 outputs of the origin k-1, stored at k
    sigma0: np.ndarray = None
    p0: np.ndarray = None

    @property
    def dates(self):
        return np.asarray(self.data.dates).astype("datetime64[D]")

    @property
    def cond(self) -> np.ndarray:
        return self.reg["yhat"] - self.reg["ybar"]


def clean_data(data):
    """``data`` without no-trade placeholder bars (``engine._drop_placeholders``) -> ``(real, keep_or_None)``."""
    return E._drop_placeholders(data)


def build_stock(data, us: dict, wf: dict | None = None, halflife: float = HALFLIFE, min_obs: int = MIN_OBS,
                names=REGRESSORS) -> Stock:
    """Gap model (and, when ``wf`` is given, the walk-forward outputs) of one placeholder-free ``PriceData``.

    ``wf`` = ``{"dates", "mu", "sigma", "p_up"}`` of the walk-forward on the same placeholder-free rows (origin indexed).
    """
    n = data.n
    o = np.asarray(data.adj_open, float)
    c = np.asarray(data.adjclose, float)
    dates = np.asarray(data.dates).astype("datetime64[D]")
    gap = np.full(n, np.nan)
    R = np.full(n, np.nan)
    with np.errstate(all="ignore"):
        gap[1:] = np.log(o[1:] / c[:-1])
        R[1:] = np.log(c[1:] / c[:-1])
        intr = np.log(c / o)
    bad = ~np.isfinite(gap)
    gap[bad] = np.nan
    R[~np.isfinite(R)] = np.nan
    intr[~np.isfinite(intr)] = np.nan
    X = align_regressors(us, dates, names)
    reg = ewma_regression(gap, X, halflife, min_obs)
    st = Stock(str(data.symbol), data, gap, intr, R, X, reg)
    if wf is not None:
        if len(wf["dates"]) != n or not np.array_equal(np.asarray(wf["dates"]).astype("datetime64[D]"), dates):
            raise ValueError(f"{data.symbol}: walk-forward rows do not match the placeholder-free bars")
        for src, dst in (("mu", "mu0"), ("sigma", "sigma0"), ("p_up", "p0")):
            a = np.full(n, np.nan)
            a[1:] = np.asarray(wf[src], float)[:-1]
            setattr(st, dst, a)
    return st


# ============================================================================= walk-forward h=1 cache
def wf_dir() -> Path:
    return research_dir() / "wf_h1"


def _wf_path(code: str, tag: str) -> Path:
    return wf_dir() / f"{code}_{tag}.npz"


def load_clean(code: str, end=None):
    """``(clean_data, full_data)``: cached completed bars cut at ``end`` (datetime64, inclusive) and the placeholder-free version."""
    d = D.peek_price_data(code)
    if d is None or d.n == 0:
        raise D.DataError(f"{code}: no cached price data")
    d = D.completed(d)
    if end is not None:
        d = d.slice(int((d.dates <= np.datetime64(end, "D")).sum()))
    real, _ = clean_data(d)
    return real, d


def load_market(end=None):
    m = D.peek_price_data(D.INDEX_SYMBOL)
    if m is None or m.n == 0:
        raise D.DataError("no cached market (TAIEX) data")
    m = D.completed(m)
    if end is not None:
        m = m.slice(int((m.dates <= np.datetime64(end, "D")).sum()))
    return m


def start_index(dates, first=DEV_START) -> int:
    """Origin row (of ``dates``) whose NEXT bar is the first bar on/after ``first``: the walk-forward must cover it."""
    return int((np.asarray(dates).astype("datetime64[D]") < np.datetime64(first, "D")).sum()) - 1


def compute_wf_h1(code: str, tag: str = "dev", refresh: bool = False) -> dict:
    """Walk-forward h=1 of one stock from the origin of the first 2013 bar, on placeholder-free rows, cached as
    ``<cache>/research/wf_h1/{code}_{tag}.npz``.  ``tag='dev'`` cuts data and market at 2022-12-31 (OOS prices never enter the
    DEV walk-forward); ``tag='full'`` uses everything (OOS run only).  Returns ``{dates, mu, sigma, p_up}`` (+ meta)."""
    end = DEV_END if tag == "dev" else None
    real, full = load_clean(code, end)
    keep = None if real.n == full.n else ~QF.placeholder_mask(full)
    cs = float(np.nansum(np.asarray(real.adjclose, float)))
    path = _wf_path(code, tag)
    if path.exists() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as z:
                if (int(z["n"]) == real.n and abs(float(z["cs"]) - cs) <= 1e-9 * max(1.0, abs(cs))
                        and str(z["version"]) == str(E.ENGINE_VERSION)):
                    return {"dates": z["dates"].astype("datetime64[D]"), "mu": z["mu"], "sigma": z["sigma"], "p_up": z["p_up"],
                            "start_idx": int(z["start_idx"]), "cached": True}
        except Exception as exc:  # noqa: BLE001 -- corrupt cache: recompute
            log.warning("%s: wf cache unreadable (%s); recomputing", code, exc)
    market = load_market(end)
    s_full = start_index(full.dates)
    if s_full < config.MIN_TRAIN:
        raise ValueError(f"{code}: only {s_full + 1} bars before {DEV_START}; need >= {config.MIN_TRAIN}")
    wf = E.walk_forward(full, market, horizons=(1,), min_train=config.MIN_TRAIN, refit_every=config.REFIT_EVERY, start_idx=s_full)
    hf = wf.horizons[1]
    sel = (lambda a: np.asarray(a, float)) if keep is None else (lambda a: np.asarray(a, float)[keep])
    out = {"dates": np.asarray(real.dates).astype("datetime64[D]"), "mu": sel(hf.mu), "sigma": sel(hf.sigma), "p_up": sel(hf.p_up),
           "start_idx": int(s_full), "cached": False}
    wf_dir().mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dates=out["dates"].astype(np.int64), mu=out["mu"], sigma=out["sigma"], p_up=out["p_up"],
                        n=np.array(real.n), cs=np.array(cs), version=np.array(str(E.ENGINE_VERSION)), start_idx=np.array(s_full))
    return out


def select_codes(pool, n: int | None = None, min_bars: int = MIN_BARS_BEFORE_DEV, loader=None) -> list:
    """First ``n`` pool codes (pool order = liquidity rank) with >= ``min_bars`` real bars strictly before 2013-01-01."""
    n = int(SPEC["stocks"]["n"]) if n is None else int(n)
    out = []
    for item in pool:
        code = item["code"] if isinstance(item, dict) else str(item)
        try:
            real = loader(code) if loader is not None else load_clean(code, DEV_END)[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: skipped (%s)", code, exc)
            continue
        if int((np.asarray(real.dates).astype("datetime64[D]") < DEV_START).sum()) >= int(min_bars):
            out.append(code)
        if len(out) >= n:
            break
    return out


# ============================================================================= matrices on the union calendar
def _scatter(cal, dates, vals):
    vals = np.asarray(vals)
    out = np.full((len(cal),) + vals.shape[1:], np.nan)
    pos = np.searchsorted(cal, dates)
    keep = pos < len(cal)
    keep &= cal[np.minimum(pos, len(cal) - 1)] == dates
    out[pos[keep]] = vals[keep]
    return out


def build_matrices(stocks, panel: Panel) -> dict:
    """``(T, S)`` matrices on ``panel.cal`` (column j = ``stocks[j]``, same order as ``panel.codes``)."""
    cal = panel.cal
    keys = ["gap", "intraday", "R", "yhat", "ybar", "r2", "mu0", "sigma0", "p0"]
    M = {k: np.full((panel.T, len(stocks)), np.nan) for k in keys}
    M["ok"] = np.zeros((panel.T, len(stocks)), bool)
    M["beta"] = np.full((panel.T, len(stocks), len(REGRESSORS) + 1), np.nan)
    for j, st in enumerate(stocks):
        d = st.dates
        for k in keys:
            src = {"gap": st.gap, "intraday": st.intraday, "R": st.R, "yhat": st.reg["yhat"], "ybar": st.reg["ybar"],
                   "r2": st.reg["r2"], "mu0": st.mu0, "sigma0": st.sigma0, "p0": st.p0}[k]
            if src is not None:
                M[k][:, j] = _scatter(cal, d, src)
        M["ok"][:, j] = _scatter(cal, d, st.reg["ok"].astype(float)) == 1.0
        M["beta"][:, j, :] = _scatter(cal, d, st.reg["beta"])
    M["cond"] = M["yhat"] - M["ybar"]
    return M


def date_mean(A, min_n: int = 5) -> np.ndarray:
    """Per-row mean over the finite entries (NaN when fewer than ``min_n``): the date-clustering step of every aggregate."""
    A = np.asarray(A, float)
    fin = np.isfinite(A)
    n = fin.sum(axis=1)
    s = np.where(fin, A, 0.0).sum(axis=1)
    with np.errstate(all="ignore"):
        m = s / n
    return np.where(n >= min_n, m, np.nan)


def _years(cal):
    return cal.astype("datetime64[Y]").astype(int) + 1970


def _by_year(series, rows, cal):
    yrs = _years(cal)
    ok = rows & np.isfinite(series)
    return {int(y): H._f(np.mean(series[ok & (yrs == y)])) for y in np.unique(yrs[ok])}


def _hac(series, h=1) -> dict:
    r = H.hac_mean_t(series, h)
    return {k: r[k] for k in ("n", "mean", "t", "p_two")}


# ============================================================================= item 1: gap model
def gap_report(M: dict, rows: np.ndarray, cal, codes) -> dict:
    """Out-of-sample R2 and direction hit rate of the gap forecast (pooled, per stock, date-clustered tests)."""
    valid = M["ok"] & np.isfinite(M["gap"]) & rows[:, None]
    g = np.where(valid, M["gap"], np.nan)
    yh = np.where(valid, M["yhat"], np.nan)
    yb = np.where(valid, M["ybar"], np.nan)
    em, eb, e0 = g - yh, g - yb, g
    sse = lambda e, ax=None: np.nansum(e ** 2, axis=ax)   # noqa: E731
    with np.errstate(all="ignore"):
        r2_zero = 1.0 - sse(em) / sse(e0)
        r2_mean = 1.0 - sse(em) / sse(eb)
        per_zero = 1.0 - sse(em, 0) / sse(e0, 0)
        per_mean = 1.0 - sse(em, 0) / sse(eb, 0)
    d_mean = date_mean(eb ** 2 - em ** 2)
    d_zero = date_mean(e0 ** 2 - em ** 2)
    yrs = _years(cal)
    by_year = {}
    for y in np.unique(yrs[rows]):
        sel = (yrs == y)[:, None] & valid
        if sel.any():
            by_year[int(y)] = {"r2_vs_mean": H._f(1.0 - np.sum(np.where(sel, em, 0) ** 2) / np.sum(np.where(sel, eb, 0) ** 2)),
                               "r2_vs_zero": H._f(1.0 - np.sum(np.where(sel, em, 0) ** 2) / np.sum(np.where(sel, e0, 0) ** 2))}
    nz = valid & (np.nan_to_num(g) != 0.0)
    sg = np.sign(g)
    hit_m = np.where(nz, (np.sign(yh) == sg).astype(float), np.nan)
    hit_b = np.where(nz, (np.sign(yb) == sg).astype(float), np.nan)
    pos_share = float(np.mean(g[nz] > 0)) if nz.any() else float("nan")
    base = max(pos_share, 1.0 - pos_share) if nz.any() else float("nan")
    hit_series = date_mean(hit_m)
    both = np.isfinite(g) & np.isfinite(yh)
    with np.errstate(all="ignore"):
        corr_y = float(np.corrcoef(yh[both], g[both])[0, 1]) if both.sum() > 3 else float("nan")
        cond = M["cond"]
        bc = both & np.isfinite(cond)
        corr_c = float(np.corrcoef(cond[bc], g[bc])[0, 1]) if bc.sum() > 3 else float("nan")
    betas = M["beta"][valid]
    per_stock = []
    for j, code in enumerate(codes):
        nzj = nz[:, j]
        per_stock.append({"code": str(code), "n": int(valid[:, j].sum()), "r2_vs_zero": H._f(per_zero[j]), "r2_vs_mean": H._f(per_mean[j]),
                          "hit": H._f(np.nanmean(hit_m[nzj, j])) if nzj.any() else None,
                          "hit_bench": H._f(np.nanmean(hit_b[nzj, j])) if nzj.any() else None})
    return {
        "n_obs": int(valid.sum()), "n_days": int(np.isfinite(d_mean).sum()),
        "r2_oos_vs_zero": H._f(r2_zero), "r2_oos_vs_mean": H._f(r2_mean),
        "loss_diff_vs_mean": _hac(d_mean), "loss_diff_vs_zero": _hac(d_zero),
        "n_stocks_r2_pos_vs_mean": int(np.sum(np.nan_to_num(per_mean, nan=-1.0) > 0)),
        "corr_yhat_gap": H._f(corr_y), "corr_cond_gap": H._f(corr_c),
        "gap_std": H._f(np.nanstd(g)), "yhat_std": H._f(np.nanstd(yh)),
        "hit_rate": H._f(np.nanmean(hit_m)), "hit_bench_causal_mean": H._f(np.nanmean(hit_b)), "share_gap_positive": H._f(pos_share),
        "hit_base_const": H._f(base), "hit_minus_base": _hac(hit_series - base),
        "hit_minus_causal": _hac(date_mean(hit_m - hit_b)),
        "beta_mean": {nm: H._f(np.nanmean(betas[:, i])) for i, nm in enumerate(["const"] + list(REGRESSORS))} if len(betas) else {},
        "by_year": by_year, "per_stock": per_stock}


# ============================================================================= item 2: pre-market update
def premarket_update(mu0, sigma0, cond, r2):
    """``(mu1, sigma1, p1, p1_mu_only, p_norm)`` -- the formulas of the spec, element-wise.

    ``mu1 = mu0 + cond``; ``sigma1^2 = sigma0^2 (1 - r2)``; ``p1 = Phi(mu1/sigma1)``; ``p1_mu_only = Phi(mu1/sigma0)``;
    ``p_norm = Phi(mu0/sigma0)`` (no update, same normal formula)."""
    mu0, sigma0, cond, r2 = (np.asarray(a, float) for a in (mu0, sigma0, cond, r2))
    mu1 = mu0 + cond
    sigma1 = sigma0 * np.sqrt(1.0 - r2)
    with np.errstate(all="ignore"):
        return mu1, sigma1, ndtr(mu1 / sigma1), ndtr(mu1 / sigma0), ndtr(mu0 / sigma0)


def pup_report(M: dict, rows: np.ndarray, cal, codes) -> dict:
    """Brier / hit-rate comparison of the engine's ``p_up``, the normal baseline and the pre-market updated probabilities."""
    R = M["R"]
    valid = (M["ok"] & rows[:, None] & np.isfinite(R) & (np.nan_to_num(R) != 0.0) & np.isfinite(M["mu0"]) & np.isfinite(M["sigma0"])
             & (np.nan_to_num(M["sigma0"]) > 0) & np.isfinite(M["p0"]) & np.isfinite(M["cond"]) & np.isfinite(M["r2"]))
    mu1, s1, p1, p1m, pn = premarket_update(M["mu0"], M["sigma0"], M["cond"], M["r2"])
    up = np.where(valid, (R > 0).astype(float), np.nan)
    P = {"engine": M["p0"], "norm": pn, "update": p1, "update_mu_only": p1m}
    P = {k: np.where(valid, v, np.nan) for k, v in P.items()}
    B = {k: (v - up) ** 2 for k, v in P.items()}
    Hh = {k: np.where(valid, ((v > 0.5).astype(float) == up).astype(float), np.nan) for k, v in P.items()}
    base = float(np.nanmean(up)) if valid.any() else float("nan")
    out = {"n_obs": int(valid.sum()), "n_days": int(np.isfinite(date_mean(B["engine"])).sum()), "up_share": H._f(base),
           "hit_base_const": H._f(max(base, 1.0 - base)) if valid.any() else None,
           "variants": {k: {"brier": H._f(np.nanmean(B[k])), "hit": H._f(np.nanmean(Hh[k])), "mean_p": H._f(np.nanmean(P[k]))} for k in P},
           "mean_cond": H._f(np.nanmean(np.where(valid, M["cond"], np.nan))), "std_cond": H._f(np.nanstd(np.where(valid, M["cond"], np.nan))),
           "mean_sigma_ratio": H._f(np.nanmean(np.where(valid, s1 / M["sigma0"], np.nan))),
           "mean_r2_rolling": H._f(np.nanmean(np.where(valid, M["r2"], np.nan))),
           "corr_cond_R": None, "comparisons": {}}
    c = valid & np.isfinite(M["cond"])
    if c.sum() > 3:
        out["corr_cond_R"] = H._f(np.corrcoef(M["cond"][c], R[c])[0, 1])
    for a, b in SPEC["premarket_update"]["comparisons"]:
        dB = B[b] - B[a]                         # positive = a has the lower Brier score
        dH = Hh[a] - Hh[b]                       # positive = a hits more often
        ds, dh = date_mean(dB), date_mean(dH)
        with np.errstate(all="ignore"):
            per_stock = np.nanmean(dB, axis=0)
        out["comparisons"][f"{a}_vs_{b}"] = {
            "d_brier": _hac(ds), "d_hit": _hac(dh), "brier_a": H._f(np.nanmean(B[a])), "brier_b": H._f(np.nanmean(B[b])),
            "brier_skill_pct": H._f(100.0 * (1.0 - np.nanmean(B[a]) / np.nanmean(B[b]))),
            "n_stocks_improved": int(np.sum(per_stock > 0)), "n_stocks": int(np.isfinite(per_stock).sum()),
            "by_year_d_brier": _by_year(ds, rows, cal),
            "per_stock_d_brier": {str(codes[j]): H._f(per_stock[j]) for j in range(len(codes))}}
    return out


# ============================================================================= item 3: tradability
def shock_z(x, lam: float = None, seed_n: int = None, min_obs: int = None) -> np.ndarray:
    """``z[S] = x[S] / sigma[S]`` with ``sigma`` from :func:`ewma_sigma` (returns of days < S only)."""
    lam = SHOCK["ewma_lambda"] if lam is None else lam
    seed_n = SHOCK["seed_n"] if seed_n is None else seed_n
    min_obs = SHOCK["min_obs"] if min_obs is None else min_obs
    x = np.asarray(x, float)
    with np.errstate(all="ignore"):
        return x / ewma_sigma(x, lam, seed_n, min_obs)


def shock_series(us: dict, cal) -> np.ndarray:
    """The aligned (usable at the S open) SOX return on ``cal``; days without a new US bar are NaN (not 0)."""
    nm = SHOCK["series"]
    return US.align_us_to_tw(us[nm][0], us[nm][1], cal)


def tradability_report(M: dict, rows: np.ndarray, cal, panel: Panel, x_shock: np.ndarray, period: str) -> dict:
    """Intraday IC with the predicted gap, sign strategy, and the 5-day open-to-open excess return after |z| >= threshold days."""
    ok = M["ok"] & np.isfinite(M["cond"]) & np.isfinite(M["intraday"])
    F = np.where(ok, M["cond"], np.nan)
    ic = H.ic_series(F, np.where(ok, M["intraday"], np.nan), ok, 20)
    ic_rep = H._series_summary(ic, rows, cal, 1)
    sel = ok & rows[:, None]
    prod = np.where(sel, M["cond"] * M["intraday"], np.nan)
    sq = np.where(sel, M["cond"] ** 2, np.nan)
    score = date_mean(prod)
    sign_ret = date_mean(np.where(sel, np.sign(M["cond"]) * M["intraday"], np.nan))
    gross = np.nanmean(np.where(sel, np.sign(M["cond"]) * M["intraday"], np.nan)) if sel.any() else np.nan
    slope = np.nanmean(score) / np.nanmean(date_mean(sq)) if np.isfinite(score).any() else float("nan")
    pooled = {"slope": H._f(slope), "slope_t": _hac(score)["t"], "n_days": int(np.isfinite(score).sum())}
    strat = {"mean_sign_x_intraday": H._f(gross), "mean_bps": H._f(1e4 * gross), "t": _hac(sign_ret)["t"],
             "pos_frac": H._f(float(np.mean(np.sign(M["cond"][sel]) * M["intraday"][sel] > 0))) if sel.any() else None,
             "round_trip_cost_plan": 0.0057}
    # ---- shock event study (entry = open of S+1, exit = open of S+6; see the spec)
    h = int(SHOCK["horizon"])
    z = shock_z(x_shock)
    y5 = forward_open_return(panel, h)                              # row S: ln(O[S+1+h] / O[S+1])
    y5ew = date_mean(y5, 20)
    rows5 = H.period_rows(cal, period, h) & np.isfinite(y5ew)
    drift = float(np.mean(y5ew[rows5])) if rows5.any() else float("nan")
    excess = np.where(rows5, y5ew - drift, np.nan)
    gap_ew, intr_ew = date_mean(M["gap"], 20), date_mean(M["intraday"], 20)
    thr = float(SHOCK["threshold"])
    ev_pos, ev_neg = rows5 & (z >= thr), rows5 & (z <= -thr)
    sgn = np.sign(z)

    def grp(sel_rows, series, sign_adjust=False):
        s = np.where(sel_rows, series * (sgn if sign_adjust else 1.0), np.nan)
        r = _hac(s, h)
        r["mean_bps"] = H._f(1e4 * r["mean"]) if r["mean"] is not None else None
        return r

    rows_ctx = rows & np.isfinite(z)
    intr_mean = np.nanmean(intr_ew[rows_ctx]) if np.isfinite(intr_ew[rows_ctx]).any() else np.nan
    intr_cont = np.where(rows_ctx & (np.abs(z) >= thr), sgn * (intr_ew - intr_mean), np.nan)
    event = {
        "threshold": thr, "horizon": h, "unconditional_5d_drift": H._f(drift), "n_rows": int(rows5.sum()),
        "n_events": int((ev_pos | ev_neg).sum()), "n_pos": int(ev_pos.sum()), "n_neg": int(ev_neg.sum()),
        "excess_pos": grp(ev_pos, excess), "excess_neg": grp(ev_neg, excess),
        "excess_signed": grp(ev_pos | ev_neg, excess, True),
        "event_day_gap_ew_bps": {"pos": H._f(1e4 * np.nanmean(gap_ew[ev_pos])) if ev_pos.any() else None,
                                 "neg": H._f(1e4 * np.nanmean(gap_ew[ev_neg])) if ev_neg.any() else None},
        "event_day_intraday_ew_bps": {"pos": H._f(1e4 * np.nanmean(intr_ew[ev_pos])) if ev_pos.any() else None,
                                      "neg": H._f(1e4 * np.nanmean(intr_ew[ev_neg])) if ev_neg.any() else None},
        "event_day_intraday_signed": _hac(intr_cont),
        "by_year_signed": _by_year(np.where(ev_pos | ev_neg, excess * sgn, np.nan), rows5, cal)}
    return {"intraday_ic": {k: ic_rep[k] for k in ("mean", "t", "n_days", "pos_frac", "by_year", "ir", "p_two")},
            "pooled_slope": pooled, "sign_strategy": strat, "shock": event}


# ============================================================================= evaluate (one period)
def evaluate(stocks, panel: Panel, x_shock, period: str, spec_hash=None) -> dict:
    """The three reports for ``period``.  ``period="oos"`` needs ``spec_hash`` (the harness guard raises BEFORE any computation) and
    appends a row to ``docs/research/LOOKS.md``.  ``stocks`` and ``panel`` columns must be in the same order."""
    H._check_period(period, spec_hash)
    if len(stocks) != panel.S:
        raise ValueError("stocks and panel columns differ")
    with H._Look(period, spec_hash, f"overnight P3b gap/pup/tradability n_stocks={panel.S}") as look:
        M = build_matrices(stocks, panel)
        rows = H.period_rows(panel.cal, period, 0)
        codes = list(panel.codes)
        res = {"period": period, "gap": gap_report(M, rows, panel.cal, codes), "pup": pup_report(M, rows, panel.cal, codes),
               "tradability": tradability_report(M, rows, panel.cal, panel, x_shock, period)}
        g, p, t = res["gap"], res["pup"]["comparisons"].get("update_vs_engine", {}), res["tradability"]
        look.result = (f"gap R2 vs mean={g['r2_oos_vs_mean']} hit={g['hit_rate']}; dBrier update-vs-engine="
                       f"{(p.get('d_brier') or {}).get('mean')} t={(p.get('d_brier') or {}).get('t')}; intraday IC={t['intraday_ic']['mean']}")
    return res


# ============================================================================= RESULTS.md
def _pct(x, d=2):
    return "-" if x is None else f"{100.0 * x:.{d}f}%"


def _num(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def render_section(res: dict, tag: str = "dev") -> str:
    """Markdown of the P3b section (without the BEGIN/END markers) from ``{"meta", "periods": {name: evaluate()}}``."""
    m = res["meta"]
    lines = [
        f"## P3b 美股隔夜與盤前更新（{'僅 DEV' if tag == 'dev' else 'OOS，只看一次'}）", "",
        f"- 產生：{m['generated_at_utc']} UTC，git `{m['git_commit']}`，spec sha256 `{m['spec_sha256'][:16]}…`（`research/overnight_spec.json`，已凍結）。"
        f"股票：流動性前 {m['n_stocks']} 檔（pool300 順序，且 2013-01-01 前已有 ≥ {m['min_bars_before_dev']} 根真實 K 棒）；"
        f"walk-forward h=1 從 2013 年第一根 K 棒的前一日起（`start_idx` = 該列，`min_train`={m['min_train']}，refit_every={m['refit_every']}）；"
        f"DEV 資料與大盤皆截斷於 2022-12-31。**{'OOS 未使用。' if tag == 'dev' else '這是 OOS 唯一一次查看。'}**",
        "- 跳空 `gap[S]=ln(adjO[S]/adjC[S−1])`；美股 `x[S]` = SOX／TSM／NVDA 在 `[前一根自身K棒日, S)` 內（嚴格 D<S）的對數報酬合計（美股假日 → 0）；"
        "EWMA（半衰期 120 根）最小平方，係數只用 `<S` 的列；`cond = ŷ − ȳ` 是美股驅動的部分。",
        "- 盤前更新：`mu'=mu+cond`、`σ'²=σ²(1−R²_滾動)`（調整後 R²）、`p'=Φ(mu'/σ')`；`mu、σ、p_up` 為 h=1 walk-forward 在前一日收盤的輸出，"
        "目標是收盤報酬 `R=ln(adjC[S]/adjC[S−1])`，平盤排除。只用**預測**跳空，不用已實現跳空。",
        "- 注意：`R²_滾動` 是**跳空**變異數的解釋比例，PLAN 公式把它套在**收盤報酬**變異數上，會高估 σ 的縮減；"
        "另列 `update_mu_only`（只移動 mu、σ 不變）與 `norm`（同公式不更新）以分離效果。", ""]
    for per, r in res["periods"].items():
        g, p, t = r["gap"], r["pup"], r["tradability"]
        lines += [f"### 期間 `{per}`", "",
                  f"**1. 跳空模型（預測更準嗎）** n={g['n_obs']} 股-日、{g['n_days']} 日", "",
                  H.md_table(["指標", "數值"], [
                      ["OOS R²（對「預測 0」）", _pct(g["r2_oos_vs_zero"], 2)],
                      ["OOS R²（對因果 EWMA 均值）", _pct(g["r2_oos_vs_mean"], 2)],
                      ["平方誤差差異（對均值）日均 / HAC t", f"{_num(g['loss_diff_vs_mean']['mean'], 8)} / {_num(g['loss_diff_vs_mean']['t'], 2)}"],
                      ["R²>0（對均值）的股票數", f"{g['n_stocks_r2_pos_vs_mean']} / {len(g['per_stock'])}"],
                      ["corr(ŷ, gap)", _num(g["corr_yhat_gap"], 3)],
                      ["跳空方向命中率", _pct(g["hit_rate"], 2)],
                      ["基準率：gap>0 比例 / 最佳常數命中", f"{_pct(g['share_gap_positive'], 2)} / {_pct(g['hit_base_const'], 2)}"],
                      ["因果均值方向命中", _pct(g["hit_bench_causal_mean"], 2)],
                      ["命中 − 基準率（日均 / HAC t）", f"{_pct(g['hit_minus_base']['mean'], 2)} / {_num(g['hit_minus_base']['t'], 2)}"],
                      ["平均係數", "，".join(f"{k}={_num(v, 3)}" for k, v in g["beta_mean"].items())]]), "",
                  f"**2. 盤前更新的收盤漲跌機率** n={p['n_obs']} 股-日、{p['n_days']} 日；上漲比例 {_pct(p['up_share'], 2)}；"
                  f"平均 σ'/σ = {_num(p['mean_sigma_ratio'], 4)}；平均 R²_滾動 = {_pct(p['mean_r2_rolling'], 2)}；corr(cond, R)={_num(p['corr_cond_R'], 3)}", "",
                  H.md_table(["變體", "Brier", "命中率", "平均 p"],
                             [[k, _num(v["brier"], 5), _pct(v["hit"], 2), _num(v["mean_p"], 4)] for k, v in p["variants"].items()]), "",
                  H.md_table(["比較 a vs b", "ΔBrier=b−a（日均）", "HAC t", "Brier skill %", "Δ命中率", "HAC t", "改善股票數"],
                             [[k, _num(c["d_brier"]["mean"], 6), _num(c["d_brier"]["t"], 2), _num(c["brier_skill_pct"], 2),
                               _pct(c["d_hit"]["mean"], 2), _num(c["d_hit"]["t"], 2), f"{c['n_stocks_improved']}/{c['n_stocks']}"]
                              for k, c in p["comparisons"].items()]), "",
                  "**3. 可交易性（開盤後還能賺嗎）**", "",
                  H.md_table(["檢驗", "數值"], [
                      ["盤中報酬 ln(C/O) 與預測跳空 cond 的橫斷面 IC（均值 / HAC t）", f"{_num(t['intraday_ic']['mean'], 4)} / {_num(t['intraday_ic']['t'], 2)}"],
                      ["合併斜率 intraday ~ cond（t）", f"{_num(t['pooled_slope']['slope'], 3)}（{_num(t['pooled_slope']['slope_t'], 2)}）"],
                      ["sign(cond)×盤中報酬（開盤買／收盤賣，毛，bps / t）", f"{_num(t['sign_strategy']['mean_bps'], 2)} / {_num(t['sign_strategy']['t'], 2)}"],
                      ["（對照：計畫的來回成本）", "約 57 bps"]]), ""]
        s = t["shock"]
        lines += [f"SOX 衝擊日（z 的絕對值 ≥ {s['threshold']:g}；正 {s['n_pos']}、負 {s['n_neg']}；進場 = S+1 開盤、出場 = S+6 開盤；超額 = 等權減期間平均 5 日漂移 {_pct(s['unconditional_5d_drift'], 3)}）：", "",
                  H.md_table(["群組", "n", "平均超額（bps）", "HAC t（h=5）"],
                             [["z ≥ 2", s["excess_pos"]["n"], _num(s["excess_pos"]["mean_bps"], 1), _num(s["excess_pos"]["t"], 2)],
                              ["z ≤ −2", s["excess_neg"]["n"], _num(s["excess_neg"]["mean_bps"], 1), _num(s["excess_neg"]["t"], 2)],
                              ["sign(z)×超額（延續為正）", s["excess_signed"]["n"], _num(s["excess_signed"]["mean_bps"], 1), _num(s["excess_signed"]["t"], 2)]]), "",
                  f"衝擊日 S 當天等權跳空 bps：正 {_num(s['event_day_gap_ew_bps']['pos'], 1)}／負 {_num(s['event_day_gap_ew_bps']['neg'], 1)}；"
                  f"當天盤中 bps：正 {_num(s['event_day_intraday_ew_bps']['pos'], 1)}／負 {_num(s['event_day_intraday_ew_bps']['neg'], 1)}。", ""]
    lines += ["**解讀（誠實區分）**：", "",
              "- 「盤前預測更準」是**預測準確度**：美股隔夜報酬在開盤前就已知，所以跳空模型的 R²／Brier 改善是真實的資訊，但它**不是**可交易的優勢——",
              "  跳空在開盤價就反映完，昨日收盤價買不到。",
              "- 「可交易的優勢」只看第 3 項：開盤後的盤中報酬、與衝擊日之後（S+1 開盤進場）的 5 日超額；請對照 57 bps 的來回成本。",
              "- 已知限制：股票是今天的流動性前 30 名（存活者偏差）；σ 縮減公式把跳空的 R² 套用在收盤報酬變異數上（見上）；"
              "股票間共同暴露於同一個美股因子，因此一律以日期分群（先對股票平均再做 HAC）。", ""]
    return "\n".join(lines).rstrip() + "\n"


def update_results_md(path, section: str, begin: str, end: str) -> None:
    """Replace the ``begin``/``end`` block of ``path`` (created with the standard header if missing; appended when absent)."""
    from .price_baseline import RESULTS_HEADER
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    block = f"{begin}\n{section}{end}\n"
    text = p.read_text(encoding="utf-8") if p.exists() else RESULTS_HEADER
    a, b = text.find(begin), text.find(end)
    if a != -1 and b != -1 and b > a:
        text = text[:a] + block + text[b + len(end):].lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    p.write_text(text, encoding="utf-8")

