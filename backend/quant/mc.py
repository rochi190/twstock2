"""Filtered-historical-simulation Monte Carlo: fan chart, VaR / CVaR, drawdown (live forecast only).

Filtered historical simulation (Barone-Adesi, Giannopoulos & Vosper 1999)
-------------------------------------------------------------------------
Paths follow the fitted GJR-GARCH recursion, but the shocks are *bootstrapped standardised residuals*
``z_t = eps_t / sigma_t`` of the fit window instead of parametric Student-t draws, so the simulated paths inherit the
empirical tail shape and skewness.  **Regime mixing**: every path carries a hidden regime that evolves with the fitted
HMM transition matrix (started from the *filtered* regime probabilities at the origin) and the residual of day ``s`` is
drawn from the pool re-weighted by the filtered probability of that regime (``w_k(t) = P(x_t = k | y_1..y_t)``), so a
path in the high-volatility regime draws the residual shapes observed in high-volatility spells.  The pool is re-centred
and re-scaled to zero mean / unit variance, so the regimes change the *shape* of the shocks while the drift is the
shrunk drift and the variance dynamics are GARCH.  The +-10 % daily limit is enforced on every simulated day
(the clipped return feeds back into the variance recursion).

Alignment with the exact ensemble
---------------------------------
The pooled ensemble predictive is exact at the forecast horizons.  So that the fan chart, VaR and the horizon cards
never contradict each other, the simulated cumulative returns are shifted by a level-dependent amount ``Delta_s(u)``
that equals (ensemble quantile - Monte-Carlo quantile) at the horizon days and is interpolated linearly in the day
number in between (constant outside); the shift is applied path-wise through each path's empirical level at that day.
At the horizon days the fan quantiles are then identical to the ensemble quantiles.
"""
from __future__ import annotations

import numpy as np

from . import garch as G


def fhs_simulate(p: G.GarchParams, sig2_next: float, z_pool: np.ndarray, pool_w: np.ndarray | None,
                 A: np.ndarray | None, pi_now: np.ndarray | None, drift: float, n_paths: int, n_steps: int,
                 rng: np.random.Generator, ret_bounds: tuple) -> np.ndarray:
    """Simulate cumulative log returns, shape (n_paths, n_steps).  See the module docstring."""
    z = np.asarray(z_pool, float)
    if not np.all(np.isfinite(z)) or len(z) < 20:
        raise ValueError("fhs: need >= 20 finite standardised residuals")
    T = len(z)
    wts = None
    if pool_w is not None and A is not None and pi_now is not None and np.shape(pool_w)[1] > 1:
        wts = np.asarray(pool_w, float)
        if wts.shape[0] != T:
            raise ValueError("pool_w must have one row per pooled residual")
    zs = (z - z.mean()) / max(z.std(), 1e-8)  # re-centre / re-scale the pool to mean 0, variance 1
    K = 1 if wts is None else wts.shape[1]
    if K > 1:
        cdfs = [np.cumsum(wts[:, k]) / max(wts[:, k].sum(), 1e-12) for k in range(K)]
        cumA = np.cumsum(A, axis=1)
        x = None
    s2 = np.full(n_paths, float(sig2_next))
    cum = np.zeros(n_paths)
    out = np.empty((n_paths, n_steps))
    lo, hi = ret_bounds
    for s in range(n_steps):
        u = rng.random(n_paths)
        if K > 1:
            if x is None:
                cum0 = np.cumsum(np.asarray(pi_now, float) @ A)
                x = np.minimum((rng.random(n_paths)[:, None] > cum0[None, :]).sum(axis=1), K - 1)
            else:
                x = np.minimum((rng.random(n_paths)[:, None] > cumA[x]).sum(axis=1), K - 1)
            idx = np.empty(n_paths, dtype=int)
            for k in range(K):
                m = x == k
                if m.any():
                    idx[m] = np.searchsorted(cdfs[k], u[m])
            zz = zs[np.minimum(idx, T - 1)]
        else:
            zz = zs[np.minimum((u * T).astype(int), T - 1)]
        eps = np.sqrt(s2) * zz
        r = np.clip(drift + eps, lo, hi)
        eps = r - drift
        cum = cum + r
        out[:, s] = cum
        s2 = p.omega + (p.alpha + p.gamma * (eps < 0.0)) * eps * eps + p.beta * s2
    return out


def align_to_anchors(cum: np.ndarray, levels, anchors: dict, ret_bounds: tuple | None = None):
    """Shift simulated cumulative log returns so that the quantiles at the anchor days equal the exact ones.

    ``cum`` (n_paths, n_steps); ``anchors`` maps day number (1-based) -> array of target log-return quantiles at
    ``levels``.  Returns ``(cum_aligned, shifts)`` with ``shifts`` of shape (n_steps, len(levels)).
    """
    levels = np.asarray(levels, float)
    n_paths, n_steps = cum.shape
    days = sorted(d for d in anchors if 1 <= d <= n_steps)
    if not days:
        return cum.copy(), np.zeros((n_steps, len(levels)))
    delta = {}
    for d in days:
        q_mc = np.quantile(cum[:, d - 1], levels)
        delta[d] = np.asarray(anchors[d], float) - q_mc
    shifts = np.zeros((n_steps, len(levels)))
    for s in range(1, n_steps + 1):
        if s <= days[0]:
            shifts[s - 1] = delta[days[0]]
        elif s >= days[-1]:
            shifts[s - 1] = delta[days[-1]]
        else:
            j = np.searchsorted(days, s, side="right") - 1
            d0, d1 = days[j], days[j + 1]
            lam = (s - d0) / (d1 - d0)
            shifts[s - 1] = (1.0 - lam) * delta[d0] + lam * delta[d1]
    adj = np.empty_like(cum)
    rank_den = n_paths + 1.0
    for s in range(n_steps):
        order = np.argsort(cum[:, s], kind="stable")
        u = np.empty(n_paths)
        u[order] = (np.arange(n_paths) + 1.0) / rank_den
        sh = np.interp(u, levels, shifts[s])  # constant beyond the first / last level
        shifted = cum[:, s] + sh
        # monotone rearrangement: the k-th smallest path keeps the k-th smallest adjusted value (the level-dependent
        # shift is only monotone if |d shift / du| < 1 / density; this guarantees it and keeps the adjusted marginal)
        adj[order, s] = np.sort(shifted)
    if ret_bounds is not None:  # cumulative return after one day is bounded by the daily limit
        adj[:, 0] = np.clip(adj[:, 0], ret_bounds[0], ret_bounds[1])
    return adj, shifts


def fan_from_paths(cum: np.ndarray, levels, last_close: float):
    """Fan-chart arrays: ``returns[level][day]`` (simple) and ``prices``; plus the log-return quantile matrix."""
    levels = np.asarray(levels, float)
    q = np.quantile(cum, levels, axis=0)  # (L, n_steps) log returns
    q = np.maximum.accumulate(q, axis=0)  # guard monotone-in-level
    ret = np.expm1(q)
    return {"days": list(range(1, cum.shape[1] + 1)), "levels": [float(v) for v in levels],
            "prices": (last_close * (1.0 + ret)).tolist(), "returns": ret.tolist()}, q


def var_cvar(simple_ret: np.ndarray, alpha: float = 0.05):
    """(VaR, CVaR) as (negative) returns: the alpha-quantile and the mean of the outcomes at or below it."""
    q = float(np.quantile(simple_ret, alpha))
    tail = simple_ret[simple_ret <= q]
    return q, float(tail.mean() if len(tail) else q)


def max_drawdown(cum_log: np.ndarray) -> np.ndarray:
    """Per-path maximum drawdown over the horizon (<= 0) of the price path ``exp(cum)`` started at 1."""
    price = np.exp(np.concatenate([np.zeros((cum_log.shape[0], 1)), cum_log], axis=1))
    run_max = np.maximum.accumulate(price, axis=1)
    return (price / run_max - 1.0).min(axis=1)


def risk_metrics(cum_aligned: np.ndarray, day_short: int = 1, day_long: int | None = None) -> dict:
    """VaR / CVaR at 95 % for the short and long day, median max drawdown and P(loss > 10 %) over the long horizon."""
    n_steps = cum_aligned.shape[1]
    day_long = n_steps if day_long is None else day_long
    r1 = np.expm1(cum_aligned[:, day_short - 1])
    rl = np.expm1(cum_aligned[:, day_long - 1])
    v1, c1 = var_cvar(r1)
    vl, cl = var_cvar(rl)
    dd = max_drawdown(cum_aligned[:, :day_long])
    return {"var95_1d": v1, "cvar95_1d": c1, f"var95_{day_long}d": vl, f"cvar95_{day_long}d": cl,
            f"median_max_drawdown_{day_long}d": float(np.median(dd)),
            f"prob_loss_gt_10pct_{day_long}d": float(np.mean(rl < -0.10))}
