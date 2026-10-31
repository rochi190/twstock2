"""Statistical metrics for out-of-sample forecast evaluation (module C).

Everything here is pure numpy + scipy and is *causal by construction*: baselines use only outcomes that
had matured (``j + h <= i``) at forecast origin ``i``.

Conventions
-----------
* ``R[i] = ln(adjclose[i+h] / adjclose[i])`` is the realised h-bar log return of origin ``i``.
* ``up = R > 0``.  Rows with ``R == 0`` carry no direction: the direction block always excludes (and counts) them; the
  probabilistic block scores the event the forecaster declares (``report.p_up_event``): ``P(R > 0 | R != 0)`` leaves them out,
  ``P(R > 0)`` counts them as "not up" (see ``expanding_up_rate(ties=...)``).
* A model *calls up* when ``p_up >= 0.5``.
* **Overlapping targets.**  For h > 1 consecutive targets share h-1 daily returns, so naive i.i.d.
  p-values are anti-conservative.  Binomial/Wilson/Kupiec/KS/Spearman inference therefore uses the
  effective sample size ``n_eff = n / h`` (exact i.i.d. formulas for h = 1; for h > 1 the variance of
  a mean of a persistent overlapping series is inflated by roughly h).  Diebold-Mariano uses a
  Newey-West HAC variance with lag h - 1 instead.
* Annualisation uses ``config.TRADING_DAYS_PER_YEAR`` (245).  Sharpe/Sortino are computed on daily simple
  net returns with a zero risk-free rate.

Formulas implemented (all cross-checked against independent computations in tests/test_eval_metrics.py)
-----------------------------------------------------------------------------------------------------
Wilson interval   centre = (p + z^2/2n) / (1 + z^2/n),  half = z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)
Brier             B = mean (p - y)^2,   skill = 1 - B / B_ref   (B_ref: causal expanding base rate, oracle reported apart)
Log loss          L = -mean [y ln p + (1-y) ln(1-p)]
ECE               sum_b (n_b / N) |mean_p_b - freq_b|  over equal-count bins
Spearman IC       rho = corr(rank(mu), rank(R)); p from t = rho sqrt((n_eff-2)/(1-rho^2)), df = n_eff-2
OOS R^2           1 - sum (R - f)^2 / sum (R - f_bench)^2          (Campbell & Thompson 2008)
Diebold-Mariano   d_t = L_ref,t - L_model,t ;  DM = mean(d) / sqrt(LRV / m),  LRV = g0 + 2 sum_{k<h} (1-k/h) g_k
Kupiec POF        LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]  ~ chi2(1)
Christoffersen    LR_ind from the 2x2 Markov transition counts of the hit series ~ chi2(1); LR_cc = LR_pof + LR_ind ~ chi2(2)
PIT               10-bin histogram + two-sided Kolmogorov-Smirnov vs U(0,1) (exact kstwo law)
Pinball           mean_tau mean_t max(tau (y - q), (tau - 1)(y - q))
PSR               Phi( (SR - SR*) sqrt(n-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) )     (Bailey & Lopez de Prado 2012)
DSR               PSR evaluated at SR0 = sqrt(V) [(1-gE) Phi^-1(1-1/N) + gE Phi^-1(1-1/(N e))]  (Bailey & Lopez de Prado 2014)
Stationary bootstrap  geometric blocks with mean length b, circular wrap (Politis & Romano 1994)
"""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
from scipy import stats

from .. import config

TRADING_DAYS = float(config.TRADING_DAYS_PER_YEAR)
EULER_GAMMA = 0.5772156649015329
DEFAULT_BUCKET_EDGES = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0)
MAX_CURVE_POINTS = 600
BOOTSTRAP_SEED = 20240607


# ============================================================================= JSON / array helpers
def finite_or_none(x):
    """``float(x)`` when it is a finite real number, else ``None``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def json_safe(obj, ndigits=None):
    """Recursively convert to strictly JSON-serialisable python objects.

    numpy scalars/arrays -> python, NaN/+-inf -> ``None``, datetime64/date -> ``YYYY-MM-DD``,
    dict keys -> ``str``.  ``ndigits`` optionally rounds every float.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return round(v, ndigits) if ndigits is not None else v
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else str(obj.astype("datetime64[D]"))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "M":
            return [json_safe(x) for x in obj]
        return json_safe(obj.tolist(), ndigits)
    if isinstance(obj, dict):
        return {str(k): json_safe(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, ndigits) for v in obj]
    return str(obj)


def date_str(d) -> str | None:
    """datetime64 / date -> 'YYYY-MM-DD' (None for NaT/None)."""
    if d is None:
        return None
    d = np.datetime64(d, "D")
    return None if np.isnat(d) else str(d)


def downsample_indices(n: int, max_points: int = MAX_CURVE_POINTS) -> np.ndarray:
    """Indices of at most ``max_points`` evenly spaced samples of ``range(n)`` keeping first and last."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.rint(np.linspace(0, n - 1, max_points)).astype(int))


# ============================================================================= realised returns & causal baselines
def realized_log_returns(adjclose, h: int) -> np.ndarray:
    """``R[i] = ln(adjclose[i+h] / adjclose[i])`` (NaN for the last ``h`` rows / invalid prices)."""
    a = np.asarray(adjclose, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    if 0 < h < n:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: n - h] = np.log(a[h:] / a[: n - h])
        out[~np.isfinite(out)] = np.nan
    return out


def expanding_up_rate(R, h: int, prior: float = 1.0, ties: str = "drop") -> np.ndarray:
    """Causal climatology of P(R > 0): Laplace-smoothed up-frequency of outcomes matured by origin i.

    ``c[i] = (U_i + prior) / (M_i + 2 prior)`` with ``U_i, M_i`` the numbers of up / counted outcomes among
    origins ``j <= i - h`` (their R_j is known at close i).  0.5 when nothing has matured yet.

    ``ties`` decides how flat outcomes (``R == 0``) are counted: ``"drop"`` (default) leaves them out of both
    numerator and denominator -- the climatology of P(up | not flat), the reference of the direction block and of a
    probabilistic block whose forecaster declares ``P(R > 0 | R != 0)`` --; ``"not_up"`` keeps them in the denominator --
    the climatology of the event ``R > 0`` itself, the reference of a probabilistic (Brier / log-loss) block whose
    forecaster declares ``P(R > 0)``.
    """
    if ties not in ("drop", "not_up"):
        raise ValueError(f"ties must be 'drop' or 'not_up', got {ties!r}")
    R = np.asarray(R, dtype=float)
    n = len(R)
    valid = np.isfinite(R) & (R != 0.0) if ties == "drop" else np.isfinite(R)
    cu = np.cumsum(valid & (R > 0))
    cn = np.cumsum(valid)
    out = np.full(n, 0.5)
    idx = np.arange(n) - int(h)
    ok = idx >= 0
    out[ok] = (cu[idx[ok]] + prior) / (cn[idx[ok]] + 2.0 * prior)
    return out


def expanding_mean_return(R, h: int) -> np.ndarray:
    """Causal historical-mean forecast: mean of finite R_j over origins ``j <= i - h`` (NaN if none)."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    fin = np.isfinite(R)
    cs = np.cumsum(np.where(fin, R, 0.0))
    cn = np.cumsum(fin)
    out = np.full(n, np.nan)
    idx = np.arange(n) - int(h)
    ok = idx >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.where(cn[idx[ok]] > 0, cs[idx[ok]] / np.maximum(cn[idx[ok]], 1), np.nan)
    return out


def trailing_sign_prediction(adjclose, h: int):
    """Momentum baseline: call 'up' iff ``ln(adjC[i] / adjC[i-h]) > 0``.  Returns ``(pred_up, valid)``."""
    a = np.asarray(adjclose, dtype=float)
    n = len(a)
    pred = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    if 0 < h < n:
        with np.errstate(divide="ignore", invalid="ignore"):
            tr = np.log(a[h:] / a[: n - h])
        ok = np.isfinite(tr)
        pred[h:] = ok & (tr > 0)
        valid[h:] = ok
    return pred, valid


# ============================================================================= direction
def effective_n(n: int, h: int) -> int:
    """Effective number of independent observations of an h-bar overlapping target: ``max(1, round(n/h))``."""
    return max(1, int(round(n / max(1, int(h))))) if n > 0 else 0


def wilson_interval(k: float, n: float, conf: float = 0.95):
    """Wilson score interval for a binomial proportion ``k / n`` -> ``(lo, hi)`` (NaNs if n <= 0)."""
    if not n or n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + conf / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_p_greater(k: int, n: int, p0: float) -> float:
    """Exact one-sided binomial p-value ``P(X >= k | n, p0)`` (``scipy.stats.binomtest``); NaN if undefined."""
    if n is None or n <= 0 or p0 is None or not (0.0 <= p0 <= 1.0) or not math.isfinite(p0):
        return float("nan")
    return float(stats.binomtest(int(k), int(n), float(p0), alternative="greater").pvalue)


def _scaled_counts(k: int, n: int, h: int):
    """(k_eff, n_eff) for overlapping targets; identity when h == 1."""
    if h <= 1 or n <= 0:
        return int(k), int(n)
    n_eff = effective_n(n, h)
    return int(round(k / n * n_eff)), n_eff


def direction_summary(p_up, up, baselines: dict, *, h: int = 1, conf: float = 0.95, high_conf: float = 0.6,
                      bucket_edges=DEFAULT_BUCKET_EDGES) -> dict:
    """Direction accuracy report (the ``direction`` block of DESIGN.md 6.4).

    Parameters
    ----------
    p_up : (m,) calibrated P(R > 0) of the scored rows.
    up : (m,) bool realised direction.
    baselines : ``{name: (pred_up (m,) bool, valid (m,) bool)}`` causal baseline calls on the same rows;
        names used by the schema: ``always_up``, ``prev_sign``, ``train_majority``.
    """
    p_up = np.asarray(p_up, dtype=float)
    up = np.asarray(up, dtype=bool)
    m = len(p_up)
    base_names = ("always_up", "prev_sign", "train_majority")
    out = {"accuracy": None, "ci95": [None, None], "n": int(m), "base_rate_up": None,
           "baselines": {b: None for b in base_names}, "best_baseline": None,
           "p_value_vs_best_baseline": None, "p_value_vs_coin": None, "by_confidence": [],
           "high_conf": {"threshold": float(high_conf), "n": 0, "accuracy": None, "p_value": None}}
    if m == 0:
        return out
    pred = p_up >= 0.5
    correct = pred == up
    k = int(correct.sum())
    acc = k / m
    k_eff, n_eff = _scaled_counts(k, m, h)
    lo, hi = wilson_interval(k_eff, n_eff, conf)
    base_rate = float(up.mean())
    accs = {}
    for name in base_names:
        if name in baselines:
            bp, bv = baselines[name]
            bp = np.asarray(bp, dtype=bool)
            bv = np.asarray(bv, dtype=bool)
            if bv.any():
                accs[name] = float((bp[bv] == up[bv]).mean())
    best = max(accs, key=lambda nme: (accs[nme], -base_names.index(nme))) if accs else None
    out.update(accuracy=acc, ci95=[lo, hi], base_rate_up=base_rate,
               baselines={b: accs.get(b) for b in base_names}, best_baseline=best,
               p_value_vs_best_baseline=binom_p_greater(k_eff, n_eff, accs[best]) if best else None,
               p_value_vs_coin=binom_p_greater(k_eff, n_eff, 0.5))
    # accuracy by predicted-probability bucket
    edges = list(bucket_edges)
    buckets = []
    for j in range(len(edges) - 1):
        lo_e, hi_e = edges[j], edges[j + 1]
        last = j == len(edges) - 2
        msk = (p_up >= lo_e) & ((p_up <= hi_e) if last else (p_up < hi_e))
        if msk.any():
            buckets.append({"bucket": f"[{lo_e:.2f},{hi_e:.2f}{']' if last else ')'}", "lo": float(lo_e),
                            "hi": float(hi_e), "n": int(msk.sum()), "accuracy": float(correct[msk].mean()),
                            "mean_p_up": float(p_up[msk].mean()), "actual_up_rate": float(up[msk].mean())})
    out["by_confidence"] = buckets
    # high-confidence calls: max(p, 1-p) >= threshold; null = a skill-less caller with the same call mix
    hc = np.maximum(p_up, 1.0 - p_up) >= high_conf
    n_hc = int(hc.sum())
    if n_hc > 0:
        k_hc = int(correct[hc].sum())
        p0 = float(np.where(pred[hc], base_rate, 1.0 - base_rate).mean())
        kh, nh = _scaled_counts(k_hc, n_hc, h)
        out["high_conf"] = {"threshold": float(high_conf), "n": n_hc, "accuracy": k_hc / n_hc,
                            "p_value": binom_p_greater(kh, nh, p0)}
    return out


# ============================================================================= probabilistic (binary P(up))
def brier_score(p, y) -> float:
    """Brier score ``mean (p - y)^2`` (NaN for empty input)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p, y, eps: float = 1e-12) -> float:
    """Binary log-loss ``-mean[y ln p + (1-y) ln(1-p)]`` with p clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    if not len(p):
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def reliability_table(p, y, n_bins: int = 10) -> list:
    """Equal-count reliability bins: rows sorted by p and split into ``n_bins`` chunks.

    Returns ``[{"p_mean", "freq", "n"}, ...]`` ordered by increasing p.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, max(1, min(int(n_bins), len(p)))):
        if len(chunk):
            rows.append({"p_mean": float(p[chunk].mean()), "freq": float(y[chunk].mean()), "n": int(len(chunk))})
    return rows


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) |mean_p_b - freq_b| over the equal-count bins of :func:`reliability_table`."""
    rows = reliability_table(p, y, n_bins)
    if not rows:
        return float("nan")
    tot = sum(r["n"] for r in rows)
    return float(sum(r["n"] / tot * abs(r["p_mean"] - r["freq"]) for r in rows))


ECE_NOISE_MULT = 1.5     # ~95th percentile of the null ECE over its mean (worst-case simulations: 1.40 - 1.43)


def ece_noise_floor(n: int, h: int = 1, n_bins: int = 10, mult: float = ECE_NOISE_MULT) -> float:
    """Level of the ECE that an *exactly calibrated* forecaster reaches by binning noise alone (95th percentile).

    With ``n`` scored rows split into ``n_bins`` equal-count bins every bin frequency carries sampling noise of sd
    ``0.5 sqrt(kappa / m)`` (``m = n / n_bins`` rows per bin, worst case p = 0.5).  ``kappa`` is the variance inflation of the
    mean of the overlapping h-bar indicators ``1[R_h > 0]``: for sums of independent daily returns the lag-k correlation of
    the indicator is ``(2/pi) arcsin(1 - k/h)``, so ``kappa = 1 + 2 sum_{k<h} (1 - k/m)(2/pi) arcsin(1 - k/h)``
    (= 1 for h = 1, about 3.8 for h = 5 and 14 for h = 20).  The mean absolute value of the bin errors is
    ``sqrt(2/pi) * 0.5 sqrt(kappa / m)``; ``mult`` scales it to the 95th percentile.  Simulated exactly calibrated forecasters
    (n = 1250, 10 bins, constant p, i.e. blocky bins = worst case) give mean ECE 3.6 % / 6.9 % / 13.3 % at h = 1 / 5 / 20, the
    formula gives 3.6 / 6.9 / 13.3 %; forecasters whose p varies over time mix the time blocks and sit below it.
    """
    if n is None or n <= 0 or n_bins is None or n_bins <= 0:
        return float("nan")
    h = max(1, int(h))
    m = float(n) / int(n_bins)
    k = np.arange(1, h, dtype=float)
    rho = (2.0 / math.pi) * np.arcsin(1.0 - k / h) if h > 1 else np.zeros(0)
    kappa = 1.0 + 2.0 * float(np.sum(np.clip(1.0 - k / m, 0.0, None) * rho))
    return float(mult * math.sqrt(2.0 / math.pi) * 0.5 * math.sqrt(max(kappa, 1.0) / m))


# Smallest rank correlation (IC) that the verdict machinery detects with ~80 % probability at n = 1250 scored forecasts
# (planted-signal simulations of the verdict rule, tests/test_eval_report.py::TestPower); scales with 1/sqrt(n_eff).
MIN_DETECTABLE_IC = {1: 0.12, 5: 0.25, 20: 0.40}
_MDIC_REF_N = 1250.0


def min_detectable_ic(h: int, n: int) -> float:
    """Approximate IC detectable with ~80 % power by the single-stock verdict rule for ``n`` scored forecasts.

    ``MIN_DETECTABLE_IC[h]`` (log-linear interpolation in h, flat beyond 20) times ``sqrt(1250 / n)``, capped at 0.99.
    Disclosure only: a ``no_edge`` verdict means "nothing detected at this sample size", not "no edge".
    """
    if n is None or n <= 0:
        return float("nan")
    hs = sorted(MIN_DETECTABLE_IC)
    base = float(np.interp(math.log(max(1, int(h))), [math.log(x) for x in hs], [MIN_DETECTABLE_IC[x] for x in hs]))
    return float(min(0.99, base * math.sqrt(_MDIC_REF_N / float(n))))


def probabilistic_summary(p, y, p_ref) -> dict:
    """The ``probabilistic`` block of DESIGN.md 6.4: Brier / log-loss of ``p`` and of the reference forecast ``p_ref``.

    ``p_ref`` is whatever reference the caller chooses; ``report.analyse`` passes the CAUSAL expanding-window base rate
    known at each origin (DESIGN.md 9(a)) for the headline numbers and computes the oracle (test-sample base rate)
    reference separately with the same function."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    out = {"brier": None, "brier_baseline": None, "brier_skill": None, "log_loss": None,
           "log_loss_baseline": None, "ece": None, "calibration": []}
    if not len(p):
        return out
    b, b0 = brier_score(p, y), brier_score(p_ref, y)
    n_bins = int(np.clip(len(p) // 30, 1, 10))
    out.update(brier=b, brier_baseline=b0, brier_skill=(1.0 - b / b0) if b0 > 0 else None,
               log_loss=log_loss(p, y), log_loss_baseline=log_loss(p_ref, y),
               ece=expected_calibration_error(p, y, n_bins), calibration=reliability_table(p, y, n_bins))
    return out


# ============================================================================= return-forecast skill
def _finite_rows(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def spearman_ic(x, y, h: int = 1):
    """Spearman rank correlation of forecast ``x`` with realised ``y`` -> ``(rho, p_two_sided, n)``.

    p-value from ``t = rho sqrt((n_eff-2)/(1-rho^2))`` with ``df = n_eff - 2`` and ``n_eff = n / h``
    (identical to ``scipy.stats.spearmanr`` for h = 1).  NaN when fewer than 3 effective points or
    one input is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = _finite_rows(x, y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    xs, ys = x[m], y[m]
    if np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return float("nan"), float("nan"), n
    rho = float(stats.spearmanr(xs, ys).statistic)
    n_eff = n / max(1, int(h))
    if n_eff < 3 or not math.isfinite(rho):
        return rho, float("nan"), n
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    df = n_eff - 2.0
    t = rho * math.sqrt(df / (1.0 - rho * rho))
    return rho, float(2.0 * stats.t.sf(abs(t), df)), n


def p_one_sided_positive(stat: float, p_two: float) -> float:
    """One-sided p-value for H1: effect > 0 from a symmetric two-sided p-value and the sign of the statistic."""
    if stat is None or p_two is None or not math.isfinite(stat) or not math.isfinite(p_two):
        return float("nan")
    return p_two / 2.0 if stat > 0 else 1.0 - p_two / 2.0


def oos_r2(y, f, f_bench) -> float:
    """Campbell-Thompson out-of-sample R^2 ``1 - SSE(f) / SSE(f_bench)`` on rows where all inputs are finite."""
    y, f, fb = (np.asarray(a, dtype=float) for a in (y, f, f_bench))
    m = _finite_rows(y, f, fb)
    if not m.any():
        return float("nan")
    sse_b = float(np.sum((y[m] - fb[m]) ** 2))
    if sse_b <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y[m] - f[m]) ** 2)) / sse_b


def newey_west_lrv(x, lag: int) -> float:
    """Bartlett-kernel long-run variance of the mean: ``g0 + 2 sum_{k=1..lag} (1 - k/(lag+1)) g_k`` (g_k /m)."""
    x = np.asarray(x, dtype=float)
    m = len(x)
    if m == 0:
        return float("nan")
    xc = x - x.mean()
    lrv = float(xc @ xc) / m
    for k in range(1, min(int(lag), m - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(xc[k:] @ xc[:-k]) / m
    return lrv


def dm_test(loss_ref, loss_model, h: int = 1):
    """Diebold-Mariano test with Newey-West HAC variance (lag h-1) -> ``(stat, p_two_sided)``.

    ``d_t = loss_ref - loss_model`` so a **positive statistic means the model is better** than the
    reference.  ``DM = mean(d) / sqrt(LRV/m)``, reference law Student-t with m-1 d.o.f. (conservative).
    Identical forecasts give ``(0, 1)``.
    """
    a = np.asarray(loss_ref, dtype=float)
    b = np.asarray(loss_model, dtype=float)
    m = _finite_rows(a, b)
    d = (a - b)[m]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    dbar = float(d.mean())
    lrv = newey_west_lrv(d, max(0, int(h) - 1))
    if not math.isfinite(lrv) or lrv <= 1e-300:
        return (0.0, 1.0) if abs(dbar) < 1e-15 else (float("nan"), float("nan"))
    stat = dbar / math.sqrt(lrv / n)
    return float(stat), float(2.0 * stats.t.sf(abs(stat), n - 1))


def interval_coverage(realized, q, qlevels, level: float):
    """Empirical coverage of the central ``level`` interval built from the quantile grid.

    Uses the columns at tau = (1-level)/2 and 1 - tau (must exist in ``qlevels``).  Returns
    ``(coverage, n, violations)`` where ``violations`` is the time-ordered bool series "outside interval".
    """
    realized = np.asarray(realized, dtype=float)
    q = np.asarray(q, dtype=float)
    ql = np.asarray(qlevels, dtype=float)
    tau = (1.0 - level) / 2.0
    lo_i = np.flatnonzero(np.isclose(ql, tau))
    hi_i = np.flatnonzero(np.isclose(ql, 1.0 - tau))
    if not len(lo_i) or not len(hi_i):
        raise KeyError(f"quantile grid has no levels {tau:.4f}/{1 - tau:.4f}")
    lo, hi = q[:, lo_i[0]], q[:, hi_i[0]]
    m = _finite_rows(realized, lo, hi)
    inside = (realized[m] >= lo[m]) & (realized[m] <= hi[m])
    if not len(inside):
        return float("nan"), 0, np.zeros(0, dtype=bool)
    return float(inside.mean()), int(len(inside)), ~inside


def _xlogy(x, y):
    from scipy.special import xlogy
    return xlogy(x, y)


def kupiec_pof(x: float, n: float, p: float):
    """Kupiec proportion-of-failures LR test: ``x`` violations in ``n`` trials, nominal rate ``p``.

    ``LR = -2 [ (n-x) ln(1-p) + x ln p - (n-x) ln(1-x/n) - x ln(x/n) ] ~ chi2(1)`` -> ``(LR, p_value)``.
    """
    if n is None or n <= 0 or not (0.0 <= p <= 1.0):
        return float("nan"), float("nan")
    pi = x / n
    ll0 = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1.0 - pi) + _xlogy(x, pi)
    lr = max(0.0, float(-2.0 * (ll0 - ll1))) if np.isfinite(ll0) else float("inf")
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits):
    """Christoffersen (1998) independence LR test on a bool violation series -> ``(LR_ind, p)`` (chi2(1))."""
    h = np.asarray(hits, dtype=bool)
    if len(h) < 2:
        return float("nan"), float("nan")
    a, b = h[:-1], h[1:]
    n00 = int(np.sum(~a & ~b))
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n11 = int(np.sum(a & b))
    tot = n00 + n01 + n10 + n11
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / tot
    ll0 = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11))
    lr = max(0.0, float(-2.0 * (ll0 - ll1)))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_cc(hits, p: float):
    """Conditional-coverage test ``LR_cc = LR_pof + LR_ind ~ chi2(2)`` -> ``(LR_cc, p)``."""
    h = np.asarray(hits, dtype=bool)
    lr_pof, _ = kupiec_pof(float(h.sum()), float(len(h)), p)
    lr_ind, _ = christoffersen_independence(h)
    lr = lr_pof + lr_ind
    return lr, float(stats.chi2.sf(lr, 2)) if math.isfinite(lr) else float("nan")


def coverage_summary(realized, q, qlevels, h: int = 1, levels=(0.50, 0.80, 0.90), headline: float = 0.90) -> dict:
    """Coverage per nominal level + Kupiec p-value at the ``headline`` level (n_eff-scaled for h > 1)."""
    cov, kup = {}, None
    for lev in levels:
        try:
            c, n, viol = interval_coverage(realized, q, qlevels, lev)
        except KeyError:
            c, n, viol = float("nan"), 0, np.zeros(0, dtype=bool)
        cov[f"{lev:.2f}"] = c
        if abs(lev - headline) < 1e-9 and n > 0:
            n_eff = effective_n(n, h) if h > 1 else n
            x_eff = int(round(viol.sum() / n * n_eff)) if h > 1 else int(viol.sum())
            kup = kupiec_pof(x_eff, n_eff, 1.0 - lev)[1]
    return {"coverage": cov, "kupiec_pvalue": kup}


def pit_histogram(pit, bins: int = 10) -> list:
    """Fractions of PIT values per equal-width bin on [0, 1] (each ~ 1/bins when calibrated)."""
    p = np.asarray(pit, dtype=float)
    p = np.clip(p[np.isfinite(p)], 0.0, 1.0)
    if not len(p):
        return [float("nan")] * bins
    return (np.histogram(p, bins=bins, range=(0.0, 1.0))[0] / len(p)).tolist()


def pit_ks_pvalue(pit, h: int = 1) -> float:
    """Two-sided Kolmogorov-Smirnov p-value of PIT vs U(0,1) using ``n_eff = n/h`` (exact kstwo law)."""
    p = np.asarray(pit, dtype=float)
    p = np.clip(p[np.isfinite(p)], 0.0, 1.0)
    n = len(p)
    if n < 2:
        return float("nan")
    d = float(stats.ks_1samp(p, stats.uniform.cdf).statistic)
    n_eff = effective_n(n, h)
    if n_eff <= 10000:
        return float(stats.kstwo.sf(d, n_eff))
    return float(stats.kstwobign.sf(d * math.sqrt(n_eff)))


def pinball_loss(y, q, qlevels) -> float:
    """Mean pinball (quantile) loss over rows and levels: ``max(tau (y - q), (tau - 1)(y - q))``."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    tau = np.asarray(qlevels, dtype=float)[None, :]
    m = np.isfinite(y) & np.all(np.isfinite(q), axis=1)
    if not m.any():
        return float("nan")
    diff = y[m][:, None] - q[m]
    return float(np.maximum(tau * diff, (tau - 1.0) * diff).mean())


def rolling_mean(x, window: int) -> np.ndarray:
    """Trailing-window mean for full windows only (length ``n - window + 1``)."""
    x = np.asarray(x, dtype=float)
    if window <= 0 or window > len(x):
        return np.zeros(0)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[window:] - cs[:-window]) / window


# ============================================================================= strategy / return-series statistics
def equity_from_returns(ret) -> np.ndarray:
    """Equity curve ``E_0 = 1, E_k = prod_{j<=k} (1 + r_j)`` (length ``len(ret) + 1``)."""
    r = np.asarray(ret, dtype=float)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def drawdown_series(equity) -> np.ndarray:
    """``E_t / max_{s<=t} E_s - 1`` (<= 0)."""
    e = np.asarray(equity, dtype=float)
    return e / np.maximum.accumulate(e) - 1.0


def max_drawdown(equity) -> float:
    """Most negative value of :func:`drawdown_series` (0.0 for a flat curve)."""
    e = np.asarray(equity, dtype=float)
    return float(drawdown_series(e).min()) if len(e) else float("nan")


def sharpe_ratio(ret, ppy: float = TRADING_DAYS) -> float:
    """Annualised Sharpe ``mean/std(ddof=1) * sqrt(ppy)`` (zero risk-free rate); NaN if degenerate."""
    r = np.asarray(ret, dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(r.mean()) / sd * math.sqrt(ppy)


def sortino_ratio(ret, ppy: float = TRADING_DAYS) -> float:
    """Annualised Sortino ``mean / sqrt(mean(min(r,0)^2)) * sqrt(ppy)`` (target 0); NaN if no downside."""
    r = np.asarray(ret, dtype=float)
    if len(r) < 2:
        return float("nan")
    dd = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    if dd < 1e-12:
        return float("nan")
    return float(r.mean()) / dd * math.sqrt(ppy)


def _sharpe_moments(r):
    """(per-period SR, skewness, raw kurtosis, n) used by PSR/DSR."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return float("nan"), float("nan"), float("nan"), n
    sr = float(r.mean()) / sd
    return sr, float(stats.skew(r, bias=True)), float(stats.kurtosis(r, fisher=False, bias=True)), n


def sharpe_standard_error(r) -> float:
    """Per-period standard error of the Sharpe estimate (Mertens 2002 / Bailey-Lopez de Prado):
    ``sqrt((1 - g3 SR + (g4-1)/4 SR^2) / (n-1))``."""
    sr, g3, g4, n = _sharpe_moments(r)
    if not math.isfinite(sr):
        return float("nan")
    v = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    return math.sqrt(v / (n - 1)) if v > 0 else float("nan")


def probabilistic_sharpe_ratio(r, sr_ref: float = 0.0) -> float:
    """PSR(SR*) = ``Phi((SR - SR*) sqrt(n-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2))`` (per-period SR, raw kurtosis)."""
    sr, g3, g4, n = _sharpe_moments(r)
    if not math.isfinite(sr):
        return float("nan")
    v = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if v <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_ref) * math.sqrt(n - 1) / math.sqrt(v)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """``SR0 = sqrt(V) [(1-gE) Phi^-1(1-1/N) + gE Phi^-1(1-1/(N e))]`` (per period); 0 for N <= 1."""
    if n_trials is None or n_trials <= 1 or not math.isfinite(var_sr) or var_sr < 0:
        return 0.0
    N = float(n_trials)
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * float(stats.norm.ppf(1.0 - 1.0 / N))
                                + EULER_GAMMA * float(stats.norm.ppf(1.0 - 1.0 / (N * math.e))))


def deflated_sharpe_ratio(r, n_trials: int = 1, var_sr: float | None = None) -> float:
    """Deflated Sharpe Ratio = PSR at the expected maximum Sharpe of ``n_trials`` skill-less trials.

    The cross-trial Sharpe variance is taken as the sampling variance of this series' own Sharpe
    estimate (:func:`sharpe_standard_error` squared) unless ``var_sr`` is supplied.  With one trial it
    equals ``PSR(0)``.
    """
    if var_sr is None:
        se = sharpe_standard_error(r)
        var_sr = se * se if math.isfinite(se) else float("nan")
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(r, sr0)


def default_block_length(n: int) -> float:
    """Mean block length for the stationary bootstrap: ``max(5, 2 n^(1/3))``."""
    return max(5.0, 2.0 * float(max(n, 1)) ** (1.0 / 3.0))


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n) index matrix of the Politis-Romano stationary bootstrap.

    At every position a new block starts with probability ``1/mean_block`` (uniform random start),
    otherwise the previous index + 1 (circular).  Block lengths are Geometric(1/mean_block).
    """
    p = 1.0 / max(1.0, float(mean_block))
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(new, pos[None, :], 0), axis=1)
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (pos[None, :] - last)) % n


def _sharpe_rows(R, ppy: float) -> np.ndarray:
    mu = R.mean(axis=1)
    sd = R.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mu / sd * math.sqrt(ppy), np.nan)


def _pct_ci(x, conf: float):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return [None, None]
    a = (1.0 - conf) / 2.0
    lo, hi = np.percentile(x, [100 * a, 100 * (1 - a)])
    return [float(lo), float(hi)]


def bootstrap_sharpe(r1, r2=None, *, ppy: float = TRADING_DAYS, n_boot: int = 1000, mean_block: float | None = None,
                     seed: int = BOOTSTRAP_SEED, conf: float = 0.95) -> dict:
    """Stationary-bootstrap inference for annualised Sharpe ratios (paired resampling of both series).

    Returns ``{"ci1", "ci2", "diff_ci", "diff_p"}``: percentile CIs of SR(r1), SR(r2), SR(r1)-SR(r2);
    ``diff_p`` is the one-sided p-value of H1: SR1 > SR2 from the re-centred bootstrap law
    ``P*(D* - D >= D)`` with +1 smoothing.  Deterministic for a given ``seed``.
    """
    r1 = np.asarray(r1, dtype=float)
    out = {"ci1": [None, None], "ci2": [None, None], "diff_ci": [None, None], "diff_p": None}
    n = len(r1)
    if n < 10 or not np.all(np.isfinite(r1)):
        return out
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block or default_block_length(n), n_boot, rng)
    s1 = _sharpe_rows(r1[idx], ppy)
    out["ci1"] = _pct_ci(s1, conf)
    if r2 is not None:
        r2 = np.asarray(r2, dtype=float)
        s2 = _sharpe_rows(r2[idx], ppy)
        out["ci2"] = _pct_ci(s2, conf)
        d_star = s1 - s2
        d_hat = sharpe_ratio(r1, ppy) - sharpe_ratio(r2, ppy)
        ok = np.isfinite(d_star)
        out["diff_ci"] = _pct_ci(d_star, conf)
        if ok.sum() >= 20 and math.isfinite(d_hat):
            out["diff_p"] = float((1 + np.sum(d_star[ok] - d_hat >= d_hat)) / (ok.sum() + 1))
    return out


def return_stats(ret, *, ppy: float = TRADING_DAYS, n_trials: int = 1, sharpe_ci=None, n_boot: int = 1000) -> dict:
    """Return-series half of the STATS block: total_return, cagr, vol, sharpe, sortino, max_dd, calmar,
    psr, dsr, sharpe_ci95.  Undefined values are ``None`` (e.g. Sharpe of a series that never moves)."""
    r = np.asarray(ret, dtype=float)
    out = {"total_return": None, "cagr": None, "vol": None, "sharpe": None, "sortino": None, "max_dd": None,
           "calmar": None, "psr": None, "dsr": None, "sharpe_ci95": [None, None]}
    n = len(r)
    if n == 0 or not np.all(np.isfinite(r)):
        return out
    eq = equity_from_returns(r)
    tot = float(eq[-1] - 1.0)
    years = n / ppy
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if eq[-1] > 0 else -1.0
    mdd = max_drawdown(eq)
    out.update(total_return=tot, cagr=cagr, vol=float(r.std(ddof=1) * math.sqrt(ppy)) if n > 1 else None,
               sharpe=finite_or_none(sharpe_ratio(r, ppy)), sortino=finite_or_none(sortino_ratio(r, ppy)),
               max_dd=mdd, calmar=(cagr / abs(mdd)) if mdd < -1e-12 else None,
               psr=finite_or_none(probabilistic_sharpe_ratio(r, 0.0)),
               dsr=finite_or_none(deflated_sharpe_ratio(r, n_trials)))
    out["sharpe_ci95"] = (list(sharpe_ci) if sharpe_ci is not None
                          else bootstrap_sharpe(r, None, ppy=ppy, n_boot=n_boot)["ci1"])
    return out
