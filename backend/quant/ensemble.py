"""Predictive-distribution primitives, the exact linear opinion pool and dynamic model averaging.

Everything here is *closed form and vectorised over forecast origins*: CDF, density, quantile function, mean and
variance, so quantiles, P(R > 0), PIT values and log scores are cheap and exact (no simulation).

Primitives
----------
``Normal(mean, std)``
``SkewT(mean, std, nu, gamma)``  -- Fernandez-Steel (1998) two-piece Student-t, standardised to the requested mean
    and standard deviation.  With ``X = gamma |T|`` (probability gamma^2 / (1 + gamma^2)) or ``X = -|T| / gamma``
    (otherwise), ``T ~ t_nu``::

        F_X(x) = 2/(1+g^2) F_T(g x)                                  x < 0
        F_X(x) = 1/(1+g^2) + 2 g^2/(1+g^2) (F_T(x/g) - 1/2)          x >= 0
        E X^r  = E|T|^r (g^{r+2} + (-1)^r g^{-r}) / (1 + g^2)

    ``gamma = 1`` is the symmetric Student-t, ``gamma < 1`` is left-skewed.  ``SkewT.from_moments`` matches a target
    variance, excess kurtosis (through ``nu``) and skewness (through ``gamma``, solved by bisection).

Linear opinion pool (Stone 1961)
--------------------------------
``F(x) = sum_k w_k F_k(x)``.  The mixture quantile lies between the smallest and the largest component quantile at the
same level, so it is found by vectorised bisection on the exact mixture CDF (to machine precision).  An optional
truncation ``[lo, hi]`` implements the daily price limit: ``F_trunc(x) = (F(x) - F(lo)) / (F(hi) - F(lo))``.

Dynamic model averaging (Raftery, Karny & Ettler 2010)
------------------------------------------------------
With forgetting factor ``alpha`` the posterior model probabilities are
``w_k(t) proportional to prod_{s <= t} p_k(y_s | F_{s-h})^{alpha^{t-s}}``, i.e. a softmax of the exponentially
discounted cumulative log predictive likelihood of *matured* outcomes.  Because h-step outcomes overlap, the
increments are tempered by ``1/h`` (a power-likelihood that keeps the evidence of ``h`` overlapping observations
equal to that of about one independent observation) and only a finite window is used, so the weights at row ``i`` are
a function of a bounded history (this is what makes ``forecast_latest`` reproducible without replaying all history).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_EPS = 1e-300


def _lift(param: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Broadcast an (m,) parameter vector against ``x`` of shape (m,) or (m, L)."""
    return param if x.ndim <= param.ndim else param.reshape(param.shape + (1,) * (x.ndim - param.ndim))


class Normal:
    """Gaussian predictive N(mean, std^2)."""

    def __init__(self, mean, std):
        self.mean, self.std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
        self.var = self.std ** 2

    def cdf(self, x):
        x = np.asarray(x, float)
        return ndtr((x - _lift(self.mean, x)) / _lift(self.std, x))

    def logpdf(self, x):
        x = np.asarray(x, float)
        z = (x - _lift(self.mean, x)) / _lift(self.std, x)
        return -0.5 * z * z - LOG_SQRT_2PI - np.log(_lift(self.std, x))

    def ppf(self, u):
        u = np.asarray(u, float)
        return _lift(self.mean, u) + _lift(self.std, u) * ndtri(u)

    def subset(self, sel):
        return Normal(self.mean[sel], self.std[sel])


def _abs_moments(nu):
    """E|T|, E T^2, E|T|^3 of the (non-standardised) Student-t(nu); nu > 3."""
    lg = gammaln(0.5 * nu)
    e1 = 2.0 * np.sqrt(nu) * np.exp(gammaln(0.5 * (nu + 1.0)) - lg) / (np.sqrt(np.pi) * (nu - 1.0))
    e2 = nu / (nu - 2.0)
    e3 = nu ** 1.5 * np.exp(gammaln(0.5 * (nu - 3.0)) - lg) / np.sqrt(np.pi)
    return e1, e2, e3


def _fs_stats(nu, g, mom=None):
    """(mean m1, std sdX, skewness) of the unstandardised Fernandez-Steel variable X(nu, gamma)."""
    e1, e2, e3 = mom if mom is not None else _abs_moments(nu)
    m1 = e1 * (g - 1.0 / g)
    m2 = e2 * (g * g + g ** -2 - 1.0)
    m3 = e3 * (g ** 5 - g ** -3) / (1.0 + g * g)
    var = m2 - m1 * m1
    sd = np.sqrt(var)
    skew = (m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3) / var ** 1.5
    return m1, sd, skew


def skew_to_gamma(nu, skew, lo: float = 0.25, hi: float = 4.0, iters: int = 50):
    """gamma such that the Fernandez-Steel t(nu) has skewness ``skew`` (monotone increasing in gamma; clipped)."""
    nu = np.asarray(nu, float)
    skew = np.asarray(skew, float)
    nu, skew = np.broadcast_arrays(nu, skew)
    mom = _abs_moments(nu)
    a = np.full(nu.shape, lo)
    b = np.full(nu.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        s = _fs_stats(nu, mid, mom)[2]
        up = s < skew  # skewness too small -> need larger gamma
        a = np.where(up, mid, a)
        b = np.where(up, b, mid)
    return 0.5 * (a + b)


class SkewT:
    """Two-piece Student-t with prescribed mean and standard deviation (``gamma = 1``: symmetric Student-t)."""

    def __init__(self, mean, std, nu, gamma=1.0):
        mean, std, nu, gamma = np.broadcast_arrays(*(np.asarray(v, float) for v in (mean, std, nu, gamma)))
        self.mean, self.std, self.nu, self.gamma = mean, std, nu, gamma
        self.var = std ** 2
        self._m1, self._sdx, _ = _fs_stats(nu, gamma)
        self._a = self._sdx / std  # X = m1 + a (R - mean)
        self._c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)

    @classmethod
    def from_moments(cls, mean, var, kurt_ex, skew=0.0, nu_min: float = 4.5, nu_max: float = 200.0):
        """Match variance exactly, excess kurtosis through ``nu = 4 + 6/kurt_ex`` and skewness through ``gamma``."""
        k = np.asarray(kurt_ex, float)
        nu = np.where(k > 6.0 / (nu_max - 4.0), 4.0 + 6.0 / np.maximum(k, 1e-12), nu_max)
        nu = np.clip(nu, nu_min, nu_max)
        skew = np.asarray(skew, float)
        gamma = np.where(np.abs(skew) < 1e-12, 1.0, skew_to_gamma(nu, skew))
        return cls(mean, np.sqrt(np.asarray(var, float)), nu, gamma)

    # --- helpers working on the standardised X scale
    def _x(self, r):
        r = np.asarray(r, float)
        return _lift(self._m1, r) + _lift(self._a, r) * (r - _lift(self.mean, r))

    def cdf(self, r):
        x = self._x(r)
        g, nu = _lift(self.gamma, x), _lift(self.nu, x)
        g2 = g * g
        neg = x < 0.0
        tcdf = stdtr(nu, np.where(neg, g * x, x / g))  # one Student-t evaluation per point
        return np.where(neg, 2.0 / (1.0 + g2) * tcdf, 1.0 / (1.0 + g2) + 2.0 * g2 / (1.0 + g2) * (tcdf - 0.5))

    def logpdf(self, r):
        x = self._x(r)
        g, nu = _lift(self.gamma, x), _lift(self.nu, x)
        t = np.where(x >= 0.0, x / g, g * x)
        lf = _lift(self._c, x) - 0.5 * (nu + 1.0) * np.log1p(t * t / nu)
        return lf + np.log(2.0 / (g + 1.0 / g)) + np.log(_lift(self._a, x))

    def ppf(self, u):
        u = np.asarray(u, float)
        g, nu = _lift(self.gamma, u), _lift(self.nu, u)
        g2 = g * g
        pm = 1.0 / (1.0 + g2)
        left = stdtrit(nu, np.clip(u / pm * 0.5, 0.0, 0.5)) / g
        right = g * stdtrit(nu, np.clip(0.5 + (u - pm) / (1.0 - pm) * 0.5, 0.5, 1.0))
        x = np.where(u < pm, left, right)
        return _lift(self.mean, u) + (x - _lift(self._m1, u)) / _lift(self._a, u)

    def skewness(self):
        return _fs_stats(self.nu, self.gamma)[2]

    def subset(self, sel):
        return SkewT(self.mean[sel], self.std[sel], self.nu[sel], self.gamma[sel])


class Pool:
    """Linear opinion pool of primitives with weights ``w`` of shape (m, P) and optional truncation (lo, hi)."""

    def __init__(self, prims: list, weights: np.ndarray, trunc: tuple | None = None):
        self.prims = list(prims)
        w = np.asarray(weights, float)
        if w.ndim == 1:
            w = w[:, None] if len(self.prims) == 1 else w[None, :]
        if w.shape[1] != len(self.prims):
            raise ValueError("weights must have one column per primitive")
        self.w = w
        self.trunc = trunc
        self._flo = self._fhi = None
        if trunc is not None:
            m = w.shape[0]
            self._flo = self._cdf_raw(np.full(m, trunc[0]))
            self._fhi = self._cdf_raw(np.full(m, trunc[1]))

    @property
    def m(self) -> int:
        return self.w.shape[0]

    def _cdf_raw(self, x):
        x = np.asarray(x, float)
        out = 0.0
        for k, p in enumerate(self.prims):
            out = out + _lift(self.w[:, k], x) * p.cdf(x)
        return np.clip(out, 0.0, 1.0)

    def cdf(self, x):
        """Pooled (truncated) CDF; ``x`` of shape (m,) or (m, L)."""
        x = np.asarray(x, float)
        f = self._cdf_raw(x)
        if self.trunc is None:
            return f
        flo, fhi = _lift(self._flo, x), _lift(self._fhi, x)
        out = (f - flo) / np.maximum(fhi - flo, _EPS)
        out = np.where(x <= self.trunc[0], 0.0, np.where(x >= self.trunc[1], 1.0, out))
        return np.clip(out, 0.0, 1.0)

    def _pdf_raw(self, x):
        x = np.asarray(x, float)
        out = 0.0
        for k, p in enumerate(self.prims):
            out = out + _lift(self.w[:, k], x) * np.exp(p.logpdf(x))
        return out

    def ppf(self, u, max_iter: int = 60, tol: float = 1e-11):
        """Pooled quantile function.  ``u`` of shape (m,) or (m, L).

        Safeguarded Newton iteration on the exact mixture CDF: the root is bracketed by the smallest and largest
        component quantile at the same level (a mixture quantile lies between them), Newton steps use the exact
        mixture density and fall back to bisection whenever they leave the bracket.  Converges to ``|F(x) - u| < tol``.
        """
        u = np.asarray(u, float)
        if self.trunc is not None:
            flo, fhi = _lift(self._flo, u), _lift(self._fhi, u)
            ua = flo + u * (fhi - flo)
        else:
            ua = u
        ua = np.clip(ua, 1e-15, 1.0 - 1e-15)
        lo = np.full(u.shape, np.inf)
        hi = np.full(u.shape, -np.inf)
        x = np.zeros(u.shape)
        for k, p in enumerate(self.prims):
            q = p.ppf(ua)
            wk = _lift(self.w[:, k], u)
            active = wk > 0.0
            lo = np.where(active, np.minimum(lo, q), lo)
            hi = np.where(active, np.maximum(hi, q), hi)
            x = x + wk * q  # weighted mean of component quantiles = excellent starting point
        span = np.maximum(hi - lo, 1e-12)
        lo = lo - 1e-9 * span
        hi = hi + 1e-9 * span
        x = np.clip(x, lo, hi)
        for _ in range(max_iter):
            f = self._cdf_raw(x) - ua
            active = np.abs(f) >= tol
            if not active.any():
                break
            below = f < 0.0
            lo = np.where(active & below, x, lo)
            hi = np.where(active & ~below, x, hi)
            d = self._pdf_raw(x)
            xn = x - f / np.maximum(d, 1e-300)
            bad = ~np.isfinite(xn) | (xn < lo) | (xn > hi)
            xn = np.where(bad, 0.5 * (lo + hi), xn)
            x = np.where(active, xn, x)
        if self.trunc is not None:
            x = np.clip(x, self.trunc[0], self.trunc[1])
        return x

    def logpdf(self, x):
        """log density of the pooled (truncated) distribution; -inf outside the truncation interval."""
        x = np.asarray(x, float)
        lp = np.stack([np.log(np.maximum(_lift(self.w[:, k], x), _EPS)) + p.logpdf(x) for k, p in enumerate(self.prims)])
        mx = lp.max(axis=0)
        out = mx + np.log(np.exp(lp - mx).sum(axis=0))
        if self.trunc is not None:
            out = out - np.log(np.maximum(_lift(self._fhi - self._flo, x), _EPS))
            out = np.where((x < self.trunc[0]) | (x > self.trunc[1]), -np.inf, out)
        return out

    @property
    def mean(self):
        """Mean of the (untruncated) pool."""
        return sum(self.w[:, k] * p.mean for k, p in enumerate(self.prims))

    @property
    def var(self):
        """Variance of the (untruncated) pool: sum w (var + mean^2) - mean^2."""
        m1 = self.mean
        m2 = sum(self.w[:, k] * (p.var + p.mean ** 2) for k, p in enumerate(self.prims))
        return np.maximum(m2 - m1 * m1, 0.0)


def mixture_logpdf(x, weights, prims) -> np.ndarray:
    """log sum_k w_k f_k(x) computed stably (used to score the HMM component, a Gaussian mixture)."""
    x = np.asarray(x, float)
    lp = np.stack([np.log(np.maximum(w, _EPS)) + p.logpdf(x) for w, p in zip(weights, prims)], axis=0)
    mx = lp.max(axis=0)
    return mx + np.log(np.exp(lp - mx).sum(axis=0))


# ----------------------------------------------------------------------------- dynamic model averaging
def dma_weights(scores: np.ndarray, alpha: float, window: int, temper: float = 1.0) -> np.ndarray:
    """Posterior model weights from matured log predictive scores.

    ``scores[k, t]`` is the log predictive density of model ``k`` for the outcome that became known at time ``t``
    (NaN = nothing matured / not available).  Returns ``w[t, k]`` with
    ``log w[t, k] = temper * sum_{s=t-window+1..t} alpha^{t-s} scores[k, s]`` (NaN counted as 0), normalised over k
    (equal weights when nothing has matured).
    """
    sc = np.asarray(scores, float)
    k, n = sc.shape
    valid = np.isfinite(sc)
    x = np.where(valid, sc, 0.0)
    w = alpha ** np.arange(window - 1, -1, -1.0)  # oldest ... newest
    pad = np.concatenate([np.zeros((k, window - 1)), x], axis=1)
    view = np.lib.stride_tricks.sliding_window_view(pad, window, axis=1)  # (k, n, window)
    logw = temper * np.einsum("knw,w->nk", view, w)
    logw -= logw.max(axis=1, keepdims=True)
    e = np.exp(logw)
    return e / e.sum(axis=1, keepdims=True)
