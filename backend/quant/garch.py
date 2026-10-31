"""GJR-GARCH(1,1) with standardised Student-t innovations.

Model
-----
::

    r_t       = m + eps_t,            eps_t = sigma_t * z_t,      z_t ~ Student-t(nu) rescaled to unit variance
    sigma_t^2 = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma_{t-1}^2

* ``alpha, gamma, beta >= 0`` and persistence ``phi = alpha + beta + gamma / 2 < 1``.  (For a symmetric ``z``
  the expected news impact is ``(alpha + gamma/2) * sigma^2``, which is why ``gamma/2`` enters ``phi``.)
* **Variance targeting**: ``omega = (1 - phi) * v`` with ``v`` the sample second moment of ``eps`` over the
  fitting window.  This removes the weakly identified level parameter and makes the term structure mean-revert to
  the window variance.  The filter starts from ``sigma_0^2 = v``.
* The recursion is *linear in sigma^2 given the data*: with ``u_t = omega + (alpha + gamma*1[eps_{t-1}<0]) eps_{t-1}^2``
  (a function of the data only) ``sigma^2 = lfilter([1], [1, -beta], u)`` evaluates the whole filter in C.
* Estimation: exact Student-t log-likelihood, L-BFGS-B on an unconstrained re-parametrisation
  ``(phi, a/phi, gamma/(2a), nu)`` that makes positivity and stationarity hold by construction.

h-step predictive (closed form, no simulation)
----------------------------------------------
Let ``u_k = sigma^2_{i+k}`` and ``a = alpha + gamma/2``.  Conditional on the information at the forecast origin
``i`` (``u_1`` known), for a symmetric ``z``::

    E[u_{k+1}] = omega + phi E[u_k]                       (exact for GJR)
    E[u_{k+1}^2] = omega^2 + 2 omega phi E[u_k] + psi E[u_k^2],
    psi = kz (alpha^2 + alpha gamma + gamma^2/2) + 2 beta (alpha + gamma/2) + beta^2,   kz = E[z^4]

The cumulative return ``S = sum_{k=1..h} r_{i+k}`` has

* ``Var(S) = sum_k E[u_k]``  (exact; martingale-difference returns),
* ``E[S^4] = kz sum_k E[u_k^2] + 6 sum_{j<k} E[r_j^2 u_k]``  where
  ``E[r_j^2 u_k] = phi^{k-j-1} (omega E[u_j] + (a kz + beta) E[u_j^2]) + omega g(k-j-1) E[u_j]``,
  ``g(d) = sum_{i<d} phi^i``.

The leverage cross terms ``E[r_a^2 r_b r_c]`` (b < c < a) of the fourth moment are included to first order (see
``hstep_moments``; checked against Monte Carlo).  The **third** moment is purely a leverage effect and is available in
closed form:
``E[S^3] = -(3 gamma E|z|^3 / 2) sum_{j<h} E[sigma_j^3] g(h-j)``, so the multi-day distribution is negatively skewed
(about -0.2 at h = 5 and -0.4 at h = 20 for typical equity parameters).  A symmetric predictive would mis-state
P(R > 0) by 1-3 percentage points at those horizons, so the engine feeds (variance, excess kurtosis, skewness) to a
two-piece Student-t (``ensemble.SkewT``).  For ``h = 1`` the predictive is exactly the fitted symmetric innovation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

PHI_MAX = 0.9995  # persistence upper bound (strictly inside the stationary region)
NU_MIN = 4.5  # keeps the fourth moment of the innovations finite
NU_MAX = 100.0
_U_BOUNDS = ((-6.0, 9.0), (-8.0, 4.0), (-8.0, 8.0), (-3.0, float(np.log(NU_MAX - NU_MIN))))


@dataclass(frozen=True)
class GarchParams:
    """Fitted GJR-GARCH(1,1)-t parameters (``mean`` is the constant used to form the residuals)."""

    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float
    mean: float = 0.0
    loglik: float = float("nan")  # total log-likelihood over the fitting window
    n_obs: int = 0
    converged: bool = True

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def uncond_var(self) -> float:
        return self.omega / (1.0 - self.persistence)

    @property
    def kz(self) -> float:
        """E[z^4] of the standardised Student-t innovation (finite because nu > 4)."""
        return 3.0 + 6.0 / (self.nu - 4.0)

    def to_dict(self) -> dict:
        return {"omega": self.omega, "alpha": self.alpha, "gamma": self.gamma, "beta": self.beta, "nu": self.nu,
                "mean": self.mean, "persistence": self.persistence, "loglik": self.loglik, "n_obs": self.n_obs,
                "converged": self.converged}


# ----------------------------------------------------------------------------- parametrisation
def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _logit(p):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return float(np.log(p / (1.0 - p)))


def _unpack(u):
    """Unconstrained vector -> (phi, alpha, gamma, beta, nu)."""
    phi = PHI_MAX * _sigmoid(u[0])
    a = phi * _sigmoid(u[1])  # a = alpha + gamma/2  (ARCH share of the persistence)
    lam = _sigmoid(u[2])  # leverage share: gamma = 2 a lam, alpha = a (1 - lam)
    nu = NU_MIN + np.exp(u[3])
    return phi, a * (1.0 - lam), 2.0 * a * lam, phi - a, nu


def _pack(alpha: float, gamma: float, beta: float, nu: float):
    a = alpha + 0.5 * gamma
    phi = a + beta
    lam = 0.5 * gamma / a if a > 0 else 0.5
    u = [_logit(phi / PHI_MAX), _logit(a / phi if phi > 0 else 0.5), _logit(lam), float(np.log(max(nu - NU_MIN, 1e-3)))]
    return np.array([min(max(x, lo), hi) for x, (lo, hi) in zip(u, _U_BOUNDS)])


# ----------------------------------------------------------------------------- filter and likelihood
def variance_filter(eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, s2_0: float) -> np.ndarray:
    """Conditional variances ``sigma2[0..T]`` for residuals ``eps[0..T-1]``.

    ``sigma2[t]`` is Var(eps_t | eps_0..eps_{t-1}) with ``sigma2[0] = s2_0``; ``sigma2[T]`` is therefore the one-step
    ahead variance after the last residual.  Implemented with ``lfilter`` (the recursion is linear in sigma^2).
    """
    eps = np.asarray(eps, dtype=float)
    t = len(eps)
    u = omega + (alpha + gamma * (eps < 0.0)) * eps * eps  # news term feeding sigma2[t+1]
    s2 = np.empty(t + 1)
    s2[0] = s2_0
    if t:
        y, _ = lfilter([1.0], [1.0, -beta], u, zi=[beta * s2_0])
        s2[1:] = y
    return s2


def _t_const(nu: float) -> float:
    return gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))


def student_t_logpdf_unit(eps: np.ndarray, s2: np.ndarray, nu: float) -> np.ndarray:
    """log density of eps with variance s2 and standardised Student-t(nu) shape."""
    z2 = eps * eps / (s2 * (nu - 2.0))
    return _t_const(nu) - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(z2)


def loglik(eps: np.ndarray, p: GarchParams, s2_0: float | None = None) -> float:
    """Total log-likelihood of the residuals ``eps`` under ``p`` (filter started at ``s2_0``, default long-run)."""
    s2_0 = p.uncond_var if s2_0 is None else s2_0
    s2 = variance_filter(eps, p.omega, p.alpha, p.gamma, p.beta, s2_0)[:-1]
    return float(np.sum(student_t_logpdf_unit(eps, s2, p.nu)))


def _objective(u, eps, v):
    phi, alpha, gamma, beta, nu = _unpack(u)
    s2 = variance_filter(eps, (1.0 - phi) * v, alpha, gamma, beta, v)[:-1]
    z2 = eps * eps / (s2 * (nu - 2.0))
    ll = _t_const(nu) - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(z2)
    val = -float(np.mean(ll))
    return val if np.isfinite(val) else 1e10


def ew_variance(eps: np.ndarray, halflife: float | None) -> float:
    """Exponentially weighted mean of ``eps^2`` (weight 0.5 per ``halflife`` rows into the past; ``None`` = equal)."""
    e2 = np.asarray(eps, float) ** 2
    if halflife is None:
        return float(np.mean(e2))
    w = 0.5 ** (np.arange(len(e2) - 1, -1, -1.0) / float(halflife))
    return float(np.sum(w * e2) / np.sum(w))


def fit_gjr_garch_t(ret: np.ndarray, mean: float = 0.0, init: GarchParams | None = None,
                    max_iter: int = 200, target_var: float | None = None) -> GarchParams:
    """Maximum-likelihood fit of the GJR-GARCH(1,1)-t model (variance targeting, constant ``mean`` removed).

    The fit is a deterministic function of ``(ret, mean, init, target_var)``.  ``init`` defaults to the usual
    equity-return starting point (phi = 0.97, ARCH share 0.10, leverage share 0.5, nu = 7).  ``target_var`` is the
    long-run variance level imposed by variance targeting (default: the sample second moment of the window; the engine
    passes an exponentially weighted one so that the level can follow slow drifts of the volatility regime).
    """
    ret = np.asarray(ret, dtype=float)
    eps = ret - mean
    if len(eps) < 30 or not np.all(np.isfinite(eps)):
        raise ValueError("garch: need >= 30 finite returns")
    v = max(float(np.mean(eps * eps)) if target_var is None else float(target_var), 1e-12)
    if init is None:
        u0 = _pack(alpha=0.03, gamma=0.07, beta=0.90, nu=7.0)
    else:
        u0 = _pack(init.alpha, init.gamma, init.beta, init.nu)
    res = minimize(_objective, u0, args=(eps, v), method="L-BFGS-B", bounds=_U_BOUNDS,
                   options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6})
    x = res.x if (np.isfinite(res.fun) and res.fun < 1e9) else u0
    if res.fun >= _objective(u0, eps, v):  # never return something worse than the start
        x = u0
    phi, alpha, gamma, beta, nu = _unpack(x)
    p = GarchParams(omega=(1.0 - phi) * v, alpha=float(alpha), gamma=float(gamma), beta=float(beta), nu=float(nu),
                    mean=float(mean), n_obs=len(eps), converged=bool(res.success))
    ll = loglik(eps, p, s2_0=v)
    return GarchParams(**{**p.__dict__, "loglik": ll})


def standardized_residuals(eps: np.ndarray, p: GarchParams) -> np.ndarray:
    """eps_t / sigma_t for the filter started at the long-run variance (unit variance if the model is right)."""
    s2 = variance_filter(eps, p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)[:-1]
    return eps / np.sqrt(s2)


# ----------------------------------------------------------------------------- h-step term structure
def _geom_sums(phi: float, h: int):
    """g[n] = sum_{i<n} phi^i and hh[n] = sum_{d<n} g[d] for n = 0..h."""
    pw = phi ** np.arange(h + 1)
    g = np.concatenate(([0.0], np.cumsum(pw[:h])))
    hh = np.concatenate(([0.0], np.cumsum(g[:h])))
    return g, hh


def hstep_variance(u1, p: GarchParams, h: int):
    """Exact Var(sum_{k=1..h} r_{i+k} | F_i) = sum_k E[sigma^2_{i+k}] for next-day variance ``u1`` (array ok)."""
    u1 = np.asarray(u1, dtype=float)
    phi = p.persistence
    g, _ = _geom_sums(phi, h)
    sbar = p.uncond_var
    return h * sbar + (u1 - sbar) * g[h]


def abs_moment3_unit_t(nu: float) -> float:
    """E|z|^3 for the unit-variance Student-t(nu) innovation (nu > 3)."""
    return float(np.exp(1.5 * np.log(nu - 2.0) + gammaln(0.5 * (nu - 3.0)) - 0.5 * np.log(np.pi) - gammaln(0.5 * nu)))


def hstep_moments(u1, p: GarchParams, h: int):
    """(Var, excess kurtosis, skewness) of the h-day cumulative return given the next-day variance ``u1``.

    Vectorised over ``u1``.  The variance is exact.  With symmetric ``z`` only the leverage term makes the third moment
    non-zero, ``E[S^3] = 3 sum_{j<k} E[r_j u_k] = -(3 gamma E|z|^3 / 2) sum_{j<h} E[sigma_j^3] g(h-j)``, and the
    leverage also adds the cross terms ``12 sum_{b<c<a} E[r_a^2 r_b r_c]`` to ``E[S^4]``; with
    ``E[r_a^2 r_b r_c] = phi^(a-c-1) E[r_b r_c u_{c+1}]``, ``E[r_c u_{c+1} | F_{c-1}] = -(gamma E|z|^3 / 2) sigma_c^3`` and a
    first-order expansion of ``E[r_b sigma_c^3]`` this is
    ``4.5 gamma^2 (E|z|^3)^2 sum_b E[sigma_b^3] Q_b``, ``Q_b = sum_{c>b} sqrt(E u_c) g(h-c) phi^(c-b-1)``.
    ``E[sigma^3]`` uses a second-order Taylor expansion of ``u^{3/2}``.  Checked against Monte Carlo in the unit tests
    (skewness within a few hundredths, excess kurtosis within about 6 % on average).
    """
    u1 = np.atleast_1d(np.asarray(u1, dtype=float))
    om, phi, beta, kz = p.omega, p.persistence, p.beta, p.kz
    a = p.alpha + 0.5 * p.gamma
    psi = kz * (p.alpha ** 2 + p.alpha * p.gamma + 0.5 * p.gamma ** 2) + 2.0 * beta * a + beta ** 2
    g, hh = _geom_sums(phi, h)
    m = u1.copy()  # E[u_j]
    q = u1 * u1  # E[u_j^2]
    var = np.zeros_like(u1)
    sq = np.zeros_like(u1)
    cross = np.zeros_like(u1)
    s3 = np.zeros_like(u1)
    m_all = [None] * (h + 2)
    sig3_all = [None] * (h + 2)
    for j in range(1, h + 1):
        var += m
        sq += q
        d = h - j
        cross += (om * m + (a * kz + beta) * q) * g[d] + om * m * hh[d]
        # E[sigma_j^3] = E[u^(3/2)] ~ m^(3/2) + (3/8) m^(-1/2) (q - m^2)
        e_sig3 = m ** 1.5 + 0.375 * (q - m * m).clip(min=0.0) / np.sqrt(m)
        s3 += e_sig3 * g[d]
        m_all[j], sig3_all[j] = m, e_sig3
        q = om * om + 2.0 * om * phi * m + psi * q
        m = om + phi * m
    e4 = kz * sq + 6.0 * cross
    ez3 = abs_moment3_unit_t(p.nu)
    if h >= 3 and p.gamma > 0.0:  # leverage cross terms of the fourth moment
        qb = np.zeros_like(u1)  # Q_{h-1} = 0
        extra = np.zeros_like(u1)
        for b in range(h - 2, 0, -1):
            qb = np.sqrt(m_all[b + 1]) * g[h - b - 1] + phi * qb
            extra += sig3_all[b] * qb
        e4 = e4 + 4.5 * p.gamma ** 2 * ez3 ** 2 * extra
    e3 = -1.5 * p.gamma * ez3 * s3
    return var, e4 / (var * var) - 3.0, e3 / var ** 1.5


def hstep_variance_kurtosis(u1, p: GarchParams, h: int):
    """(Var, excess kurtosis) of the h-day cumulative return (see :func:`hstep_moments`)."""
    var, kurt, _ = hstep_moments(u1, p, h)
    return var, kurt


def student_t_params_from_moments(var, kurt_ex, nu_min: float = NU_MIN, nu_max: float = 200.0):
    """Scale and degrees of freedom of the Student-t with variance ``var`` and excess kurtosis ``kurt_ex``.

    ``kurt_ex = 6 / (nu - 4)``  =>  ``nu = 4 + 6 / kurt_ex`` (clipped to [nu_min, nu_max]);  ``scale^2 = var (nu-2)/nu``.
    """
    var = np.asarray(var, dtype=float)
    k = np.asarray(kurt_ex, dtype=float)
    nu = np.where(k > 6.0 / (nu_max - 4.0), 4.0 + 6.0 / np.maximum(k, 1e-12), nu_max)
    nu = np.clip(nu, nu_min, nu_max)
    return np.sqrt(var * (nu - 2.0) / nu), nu


def hstep_predictive(u1, p: GarchParams, h: int):
    """(variance, excess kurtosis, skewness) of the h-day cumulative return (zero mean).

    The engine turns these into a (skewed) Student-t predictive via ``ensemble.SkewT.from_moments``.
    """
    return hstep_moments(u1, p, h)


# ----------------------------------------------------------------------------- simulation
def standardized_t(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Student-t(nu) draws rescaled to unit variance."""
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


def simulate(p: GarchParams, n: int, rng: np.random.Generator, burn: int = 500) -> np.ndarray:
    """Simulate ``n`` residuals ``eps`` from the model (started at the long-run variance, ``burn`` discarded)."""
    z = standardized_t(rng, p.nu, n + burn)
    eps = np.empty(n + burn)
    s2 = p.uncond_var
    for t in range(n + burn):
        e = np.sqrt(s2) * z[t]
        eps[t] = e
        s2 = p.omega + (p.alpha + (p.gamma if e < 0 else 0.0)) * e * e + p.beta * s2
    return eps[burn:]


def simulate_paths(p: GarchParams, u1, z: np.ndarray, ret_bounds: tuple | None = None):
    """Forward-simulate residual paths from the next-day variance ``u1`` (scalar or per-path array).

    ``z`` is an ``(n_paths, steps)`` array of unit-variance shocks.  If ``ret_bounds = (lo, hi)`` is given the *return*
    ``p.mean + eps`` is clipped to it (daily price limit) and the clipped residual feeds back into the variance.
    Returns ``eps`` of shape ``(n_paths, steps)``.
    """
    n_paths, steps = z.shape
    s2 = np.broadcast_to(np.asarray(u1, dtype=float), (n_paths,)).copy()
    out = np.empty((n_paths, steps))
    for k in range(steps):
        e = np.sqrt(s2) * z[:, k]
        if ret_bounds is not None:
            e = np.clip(p.mean + e, ret_bounds[0], ret_bounds[1]) - p.mean
        out[:, k] = e
        s2 = p.omega + (p.alpha + p.gamma * (e < 0.0)) * e * e + p.beta * s2
    return out
