"""Discount-factor dynamic linear model with Normal-Inverse-Gamma variance learning (West & Harrison).

Model for one horizon ``h``
---------------------------
``y_j = x_j' theta_t + v``,  ``v ~ N(0, V)``,  where ``y_j`` is the **vol-standardised, drift-adjusted h-day return of
origin j** and ``x_j`` the causal feature vector (with an intercept) at origin ``j``.  The coefficient vector drifts
slowly (``theta_t`` is only weakly tied to ``theta_{t-1}``), which is expressed through a *discount factor*
``delta``: before every update the information is discounted, ``R^{-1} = delta C^{-1} + (1-delta) Lambda_0``.  The
discount is taken **towards the prior precision** ``Lambda_0 = diag(1/tau^2)`` (an exponentially weighted ridge) rather
than towards zero as in the plain ``R = C / delta`` rule: a plain discount forgets the prior as well, so with 14
features the stationary coefficient noise (about ``sqrt(1-delta)`` each) would dominate any real signal of a few tenths
of a percent of explained variance.  The ridge prior ``theta ~ N(0, tau^2 I)`` encodes the a-priori belief that daily
equity returns are almost unpredictable (prior R^2 of order ``p tau^2`` ~ 0.5 %).

Matured outcomes only
---------------------
The target of origin ``j`` is only known at row ``j + h``.  The update performed at row ``t`` therefore uses the pair
``(x_{t-h}, y_{t-h})``; the forecast for origin ``t`` uses the coefficients after that update.  Nothing else is used.

Overlapping targets (tempering)
-------------------------------
Consecutive h-day targets share ``h-1`` daily returns, so they carry roughly ``1/h`` of the information of independent
observations.  The update is a *power likelihood* with exponent ``1/h``: the observation variance is ``h V``,
``Q = h + x' C x`` and, for the Inverse-Gamma variance posterior, ``n <- beta_v n + 1/h``, ``d <- beta_v d + e^2 / Q``
(``S = d / n`` estimates ``V``; ``beta_v`` is a variance discount).  Without tempering the learned variance would
re-inflate the information and undo the correction.

Recursions (``C = P^{-1}`` is the scale-free posterior covariance, ``theta | V ~ N(m, V C)``)::

    P <- delta P + (1 - delta) Lambda_0 ;  b <- delta b ;  m = P^{-1} b ;   (prior for the new pair)
    e = y - x'm ,  Q = h + x'Cx ,  A = C x / Q
    m <- m + A e ,  P <- P + x x' / h ,  b <- b + x y / h ,  C <- C - Q A A'

Forecast for origin ``t``: mean ``x_t' m``; parameter-uncertainty variance of that mean ``S x_t' C x_t``
(in standardised units).  The batch routine advances all horizons simultaneously with one batched ``solve`` per row.
"""
from __future__ import annotations

import numpy as np


def default_prior_sd(p: int, tau_slope: float = 0.025, tau_intercept: float = 0.03) -> np.ndarray:
    """Prior standard deviations of the coefficients: the last column is the intercept."""
    sd = np.full(p, tau_slope)
    sd[-1] = tau_intercept
    return sd


def run_dlm(X: np.ndarray, Y: np.ndarray, hs, delta: float = 0.998, beta_v: float = 0.998,
            prior_sd: np.ndarray | None = None, n0: float = 4.0, S0: float = 1.0):
    """Filter the discount-factor NIG DLM over all rows for several horizons at once.

    Parameters
    ----------
    X : (n, p) design at the forecast origins (include an intercept column; rows with non-finite entries are skipped).
    Y : (H, n) standardised targets by origin, ``Y[a, j]`` known at row ``j + hs[a]`` (NaN when not available).
    hs : the H horizons.

    Returns
    -------
    dict with ``f`` (H, n) forecast means ``x_t' m_t``, ``pu`` (H, n) parameter-uncertainty variances
    ``S_t x_t' C_t x_t``, ``S`` (H, n) learned noise variance, ``m`` (H, n, p) posterior means, ``n_obs`` (H, n) count
    of matured updates applied so far.  Row ``t`` only depends on rows ``<= t`` of ``X`` and on ``Y[a, j]`` with
    ``j <= t - hs[a]``.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    hs = np.asarray(hs, int)
    H = len(hs)
    sd = default_prior_sd(p) if prior_sd is None else np.asarray(prior_sd, float)
    lam0 = np.diag(1.0 / (sd * sd))[None, :, :]  # (1, p, p) prior precision
    k = hs.astype(float)  # observation-variance multiplier = tempering exponent inverse
    P = np.repeat(lam0, H, axis=0).copy()
    b = np.zeros((H, p))
    nn = np.full(H, n0)
    dd = np.full(H, n0 * S0)
    Xok = np.all(np.isfinite(X), axis=1)
    Xz = np.where(Xok[:, None], X, 0.0)
    # gathers for the matured pair of every row and horizon, done once outside the loop
    tt = np.arange(n)[:, None]
    jj_all = tt - hs[None, :]  # (n, H) matured origin
    has = jj_all >= 0
    jc = np.where(has, jj_all, 0)
    ya_all = Y[np.arange(H)[None, :], jc]  # (n, H)
    mask_all = has & Xok[jc] & np.isfinite(ya_all)
    xa_all = np.where(mask_all[:, :, None], Xz[jc], 0.0)  # (n, H, p)
    ya_all = np.where(mask_all, ya_all, 0.0)
    f_out = np.zeros((H, n))
    pu_out = np.zeros((H, n))
    s_out = np.zeros((H, n))
    m_out = np.zeros((H, n, p))
    cnt_out = np.zeros((H, n))
    cnt = np.zeros(H)
    one_minus = 1.0 - delta
    lam_blend = one_minus * lam0
    inv_k = 1.0 / k
    rhs = np.empty((H, p, 3))
    solve = np.linalg.solve
    einsum = np.einsum
    for t in range(n):
        xa = xa_all[t]
        ya = ya_all[t]
        mask = mask_all[t]
        xt = Xz[t]
        P *= delta
        P += lam_blend  # discount towards the prior precision
        b *= delta
        rhs[:, :, 0] = b
        rhs[:, :, 1] = xa
        rhs[:, :, 2] = xt
        sol = solve(P, rhs)
        m_prior = sol[:, :, 0]
        cxa = sol[:, :, 1]
        cxt = sol[:, :, 2]
        e = ya - einsum("hp,hp->h", xa, m_prior)
        q = k + einsum("hp,hp->h", xa, cxa)
        wmask = mask.astype(float)
        m_new = m_prior + cxa * (e * wmask / q)[:, None]
        # information update (masked rows have xa = 0, so they are no-ops)
        P += (xa[:, :, None] * xa[:, None, :]) * inv_k[:, None, None]
        b += xa * (ya * inv_k)[:, None]
        # tempered NIG variance learning
        nn = np.where(mask, beta_v * nn + inv_k, nn)
        dd = np.where(mask, beta_v * dd + e * e / q, dd)
        cnt += wmask
        # posterior covariance applied to x_t (Sherman-Morrison for the pair just absorbed)
        cax_xt = cxa @ xt
        quad = cxt @ xt - wmask * cax_xt * cax_xt / q
        S = dd / nn
        f_out[:, t] = m_new @ xt
        pu_out[:, t] = S * np.maximum(quad, 0.0)
        s_out[:, t] = S
        m_out[:, t, :] = m_new
        cnt_out[:, t] = cnt
    return {"f": f_out, "pu": pu_out, "S": s_out, "m": m_out, "n_obs": cnt_out}


def standardized_targets(cs: np.ndarray, drift: np.ndarray, sigma_ref: np.ndarray, hs, clip: float = 5.0) -> np.ndarray:
    """``Y[a, j] = (cs[j+h] - cs[j] - h drift[j]) / (sigma_ref[j] sqrt(h))`` winsorised at +-clip (NaN beyond the data).

    ``cs`` is the cumulative *cleaned* log return.  Everything on the right is known at origin ``j`` except
    ``cs[j+h]``, which is only used once it has matured.
    """
    n = len(cs)
    hs = list(hs)
    Y = np.full((len(hs), n), np.nan)
    for a, h in enumerate(hs):
        if n > h:
            R = cs[h:] - cs[:-h]
            y = (R - h * drift[: n - h]) / (sigma_ref[: n - h] * np.sqrt(h))
            Y[a, : n - h] = np.clip(y, -clip, clip)
    return Y
