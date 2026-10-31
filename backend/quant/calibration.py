"""Online Bayesian logistic (Platt) recalibration of ``P(R > 0)`` using matured outcomes only.

Model
-----
``P(up | p_raw) = sigmoid(a + b * logit(p_raw))``  with the Gaussian prior ``a ~ N(0, sigma_a^2)``,
``b ~ N(1, sigma_b^2)`` -- i.e. *regularised towards the identity map* (``a = 0``, ``b = 1``).  The posterior mode is
found by a few damped Newton steps on the exactly concave log posterior::

    log post = sum_t w_t [ y_t log p_t + (1 - y_t) log(1 - p_t) ] - (a/sigma_a)^2/2 - ((b-1)/sigma_b)^2/2

Matured outcomes only
---------------------
At origin ``i`` the pairs ``(p_raw_j, 1[R_j > 0])`` of origins ``j`` with ``j + h <= i`` are used (the outcome of
origin ``j`` is only known at row ``j + h``), restricted to the last ``window`` rows, weighted by an exponential
forgetting factor ``lam^{age}`` and *tempered* by ``1/h`` (overlapping h-day outcomes are strongly dependent, so ``h`` of
them carry about the information of one).  The result for origin ``i`` is therefore a function of a bounded history.

Ties carry no target
--------------------
An outcome whose indicator is NaN is skipped (``ok`` mask below).  The engine passes NaN for unchanged closes
(``R == 0`` exactly): the raw ensemble is a continuous mixture without an atom at 0, so the quantity it can predict is
``P(R > 0) = P(R > 0 | R != 0)``, and DESIGN section 4 scores direction with ties excluded as well.  Feeding ties in as
"not up" would instead teach the calibrator ``P(R > 0)`` with the flat mass counted as a down move (a bias of about
``0.3 x tie rate`` at h = 1, i.e. +2 pp on average and +7 pp for a 25 %-tie name).

Why such a tight prior: the raw ensemble probability sits in roughly [0.35, 0.65]; a recalibrator that moved freely
would add more noise than it removes.  With ``sigma_a = 0.12`` the intercept may absorb a persistent bias of a few
percentage points (e.g. a drift that the shrinkage prior is too slow to learn) but no more.
"""
from __future__ import annotations

import numpy as np

P_CLIP = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def recalibrate(p_raw: np.ndarray, up: np.ndarray, h: int, window: int = 400, lam: float = 0.995,
                sigma_a: float = 0.12, sigma_b: float = 0.35, temper: bool = True, newton_iters: int = 10):
    """Posterior-mode recalibration for every origin.

    Parameters
    ----------
    p_raw : (n,) raw ensemble ``P(R > 0)`` at each origin (NaN where no forecast exists).
    up : (n,) outcome indicator ``1[R > 0]`` of each origin (NaN until realised, and NaN for unchanged closes ``R == 0``
        which carry no direction; row j is only used from row j + h on).
    h : horizon of the outcomes.
    window : number of most recent *origins* eligible (``i - window <= j <= i - h``).

    Returns ``(p_cal, a, b)`` arrays of shape (n,).  ``p_cal`` is NaN where ``p_raw`` is NaN.
    """
    p_raw = np.asarray(p_raw, float)
    up = np.asarray(up, float)
    n = len(p_raw)
    x = logit(np.where(np.isfinite(p_raw), p_raw, 0.5))
    ok = np.isfinite(p_raw) & np.isfinite(up)
    # maturity-indexed arrays: entry t describes origin t - h and becomes known at row t
    xm = np.zeros(n)
    ym = np.zeros(n)
    vm = np.zeros(n)
    if n > h:
        xm[h:] = x[: n - h]
        ym[h:] = np.where(ok[: n - h], up[: n - h], 0.0)
        vm[h:] = ok[: n - h].astype(float)
    wc = max(window - h + 1, 1)  # maturities t in [i - wc + 1, i]  <=>  origins j in [i - window, i - h]
    pad = lambda a: np.concatenate([np.zeros(wc - 1), a])
    view = np.lib.stride_tricks.sliding_window_view
    xs, ys, vs = view(pad(xm), wc), view(pad(ym), wc), view(pad(vm), wc)  # (n, wc), column wc-1 = newest
    age = np.arange(wc - 1, -1, -1.0)
    w = vs * (lam ** age)[None, :] * ((1.0 / h) if temper else 1.0)
    la, lb = 1.0 / sigma_a ** 2, 1.0 / sigma_b ** 2
    a = np.zeros(n)
    b = np.ones(n)
    for _ in range(newton_iters):
        p = expit(a[:, None] + b[:, None] * xs)
        r = w * (ys - p)
        wp = w * p * (1.0 - p)
        g1 = r.sum(axis=1) - la * a
        g2 = (r * xs).sum(axis=1) - lb * (b - 1.0)
        h11 = wp.sum(axis=1) + la
        h12 = (wp * xs).sum(axis=1)
        h22 = (wp * xs * xs).sum(axis=1) + lb
        det = h11 * h22 - h12 * h12
        da = (h22 * g1 - h12 * g2) / det
        db = (-h12 * g1 + h11 * g2) / det
        a = a + np.clip(da, -1.0, 1.0)
        b = b + np.clip(db, -1.0, 1.0)
    p_cal = np.clip(expit(a + b * x), P_CLIP, 1.0 - P_CLIP)
    p_cal = np.where(np.isfinite(p_raw), p_cal, np.nan)
    return p_cal, a, b
