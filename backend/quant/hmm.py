"""Gaussian hidden Markov model with diagonal covariances (Baum-Welch / EM with scaling), numpy only.

Observation model
-----------------
``y_t = (daily log return, log range-based volatility)`` (any D >= 1 works; dimension 0 must be the return).  Given the
hidden state ``x_t = k``: ``y_t ~ N(mu_k, diag(var_k))``; ``P(x_{t+1} = l | x_t = k) = A[k, l]``.

Estimation (maximum a posteriori EM)
------------------------------------
* E-step: scaled forward/backward recursions (``alpha`` normalised at every step, ``c_t = p(y_t | y_<t)``), state
  posteriors ``gamma_t = alpha_t * beta_t`` and expected transition counts ``xi``.  The loops are written as unrolled
  scalar Python for K = 3 (CPython 3.11 does ~3 million float operations per second per core; about 0.35 ms per
  1000 observations) with a generic numpy fallback for other K.
* M-step with conjugate priors so that the fit is stable on short windows: the means shrink towards the window mean
  (``kappa`` pseudo-observations, 250 for the return and 5 for the log-volatility), the variances towards the window
  variance (3 pseudo-observations) and each row of ``A`` towards a persistent prior (0.95 on the diagonal, 10
  pseudo-transitions).
* Deterministic initialisations (no randomness): a 1-D Gaussian mixture and two skewed quantile splits of the smoothed
  log-volatility path compete for a few iterations (EM for HMMs has many local optima; equal-count terciles miss the
  rare high-volatility regime); the best is refined.  States are finally **sorted by return variance**, so
  state 0 is always the calmest regime and the columns of the filtered probabilities mean the same thing at every refit.

Filtering and prediction
------------------------
Only *filtered* probabilities ``P(x_t | y_1..y_t)`` are used (never smoothed).  For an h-step-ahead cumulative return
``S = sum_{s=1..h} r_{t+s}`` starting from state k the exact mean and variance are available from powers of ``A``::

    E[r_s | k]  = (A^s mu)_k
    Var(S | k)  = sum_s Var(r_s | k) + 2 sum_{s<u} Cov(r_s, r_u | k),
    Cov(r_s, r_u | k) = sum_j A^s_{kj} mu_j (A^{u-s} mu)_j - (A^s mu)_k (A^u mu)_k

so the predictive distribution is the mixture ``sum_k filtered_k * N(m_k(h), v_k(h))`` (regime-conditional drift and
volatility, including the extra variance caused by regime switching inside the horizon).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TINY = 1e-300


@dataclass
class HMMParams:
    """Fitted Gaussian HMM (states sorted by the variance of dimension 0)."""

    A: np.ndarray  # (K, K) row-stochastic transition matrix
    mu: np.ndarray  # (K, D)
    var: np.ndarray  # (K, D)
    lo: np.ndarray  # (D,) lower winsorisation bound applied to observations (-inf for the return dimension)
    hi: np.ndarray  # (D,) upper bound
    alpha_last: np.ndarray = field(default_factory=lambda: np.zeros(0))  # filtered state distribution at the last row
    loglik: float = float("nan")
    n_obs: int = 0
    n_iter: int = 0
    converged: bool = False

    @property
    def K(self) -> int:
        return self.A.shape[0]

    def clip(self, Y: np.ndarray) -> np.ndarray:
        return np.clip(Y, self.lo, self.hi)


# ----------------------------------------------------------------------------- basic building blocks
def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of the row-stochastic matrix ``A`` (solves pi (A - I) = 0, sum pi = 1)."""
    K = A.shape[0]
    M = A.T - np.eye(K)
    M[-1, :] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        pi = np.full(K, 1.0 / K)
    pi = np.clip(pi, 1e-12, None)
    return pi / pi.sum()


def log_emission(Y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(T, K) matrix of log N(y_t; mu_k, diag(var_k))."""
    d = Y[:, None, :] - mu[None, :, :]
    return -0.5 * np.sum(d * d / var[None, :, :] + np.log(2.0 * np.pi * var)[None, :, :], axis=2)


def _scale_rows(logB: np.ndarray):
    m = logB.max(axis=1)
    return np.exp(logB - m[:, None]), m


def _forward3(B, A, pi0):
    A00, A01, A02, A10, A11, A12, A20, A21, A22 = A.ravel().tolist()
    b0 = B[:, 0].tolist()
    b1 = B[:, 1].tolist()
    b2 = B[:, 2].tolist()
    T = len(b0)
    o0 = [0.0] * T
    o1 = [0.0] * T
    o2 = [0.0] * T
    cs = [0.0] * T
    a0, a1, a2 = (float(v) for v in pi0)
    n0 = a0 * b0[0]
    n1 = a1 * b1[0]
    n2 = a2 * b2[0]
    c = n0 + n1 + n2
    ic = 1.0 / c
    a0 = n0 * ic
    a1 = n1 * ic
    a2 = n2 * ic
    o0[0] = a0
    o1[0] = a1
    o2[0] = a2
    cs[0] = c
    for t in range(1, T):
        n0 = (a0 * A00 + a1 * A10 + a2 * A20) * b0[t]
        n1 = (a0 * A01 + a1 * A11 + a2 * A21) * b1[t]
        n2 = (a0 * A02 + a1 * A12 + a2 * A22) * b2[t]
        c = n0 + n1 + n2
        ic = 1.0 / c
        a0 = n0 * ic
        a1 = n1 * ic
        a2 = n2 * ic
        o0[t] = a0
        o1[t] = a1
        o2[t] = a2
        cs[t] = c
    return np.array([o0, o1, o2]).T, np.array(cs)


def _backward3(B, A, c):
    A00, A01, A02, A10, A11, A12, A20, A21, A22 = A.ravel().tolist()
    b0 = B[:, 0].tolist()
    b1 = B[:, 1].tolist()
    b2 = B[:, 2].tolist()
    cl = c.tolist()
    T = len(b0)
    o0 = [1.0] * T
    o1 = [1.0] * T
    o2 = [1.0] * T
    x0 = x1 = x2 = 1.0
    for t in range(T - 2, -1, -1):
        e0 = b0[t + 1] * x0
        e1 = b1[t + 1] * x1
        e2 = b2[t + 1] * x2
        ic = 1.0 / cl[t + 1]
        x0 = (A00 * e0 + A01 * e1 + A02 * e2) * ic
        x1 = (A10 * e0 + A11 * e1 + A12 * e2) * ic
        x2 = (A20 * e0 + A21 * e1 + A22 * e2) * ic
        o0[t] = x0
        o1[t] = x1
        o2[t] = x2
    return np.array([o0, o1, o2]).T


def forward(B: np.ndarray, A: np.ndarray, pi0: np.ndarray):
    """Scaled forward pass: returns filtered ``alpha`` (T, K) and the scales ``c`` (T,) (``B`` is row-scaled)."""
    T, K = B.shape
    if K == 3:
        return _forward3(B, A, pi0)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi0 * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def backward(B: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass matching :func:`forward` (so that ``gamma = alpha * beta`` sums to one)."""
    T, K = B.shape
    if K == 3:
        return _backward3(B, A, c)
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def filter_probs(params: HMMParams, Y: np.ndarray, alpha_prev: np.ndarray | None = None) -> np.ndarray:
    """Filtered state probabilities for the rows of ``Y`` continuing from ``alpha_prev`` (default: start of sample).

    If ``alpha_prev`` is ``None`` the first row uses the stationary distribution as prior, otherwise it is propagated
    through ``A`` first.  Returns an array of shape (len(Y), K).
    """
    Y = params.clip(np.asarray(Y, float).reshape(len(Y), -1))
    B, _ = _scale_rows(log_emission(Y, params.mu, params.var))
    A = params.A
    out = np.empty((len(Y), params.K))
    for t in range(len(Y)):
        if t > 0:
            prior = out[t - 1] @ A
        else:
            prior = stationary_distribution(A) if alpha_prev is None else np.asarray(alpha_prev, float) @ A
        a = prior * B[t]
        s = a.sum()
        out[t] = a / s if s > 0 else prior
    return out


# ----------------------------------------------------------------------------- estimation
def _default_prior_strength(D: int) -> np.ndarray:
    ks = np.full(D, 5.0)
    ks[0] = 250.0
    return ks


def _m_step(Y, gamma, xi_sum, prior):
    K = gamma.shape[1]
    w = gamma.sum(axis=0)  # (K,)
    gy = gamma.T @ Y  # (K, D)
    kappa = prior["kappa"][None, :]
    mu = (gy + kappa * prior["mean"][None, :]) / (w[:, None] + kappa)
    gy2 = gamma.T @ (Y * Y)
    ss = gy2 - 2.0 * mu * gy + (mu * mu) * w[:, None]
    nv = prior["nu0"]
    var = (ss + nv * prior["var"][None, :]) / (w[:, None] + nv)
    var = np.maximum(var, prior["var_floor"][None, :])
    P0 = np.full((K, K), (1.0 - prior["diag"]) / max(K - 1, 1))
    np.fill_diagonal(P0, prior["diag"])
    A = xi_sum + prior["a0"] * P0
    A = A / A.sum(axis=1, keepdims=True)
    A = np.maximum(A, 1e-6)
    A = A / A.sum(axis=1, keepdims=True)
    return A, mu, var


def _em(Y, A, mu, var, prior, max_iter, tol):
    """Run EM; returns (A, mu, var, loglik, n_iter, converged)."""
    T = len(Y)
    ll_prev = -np.inf
    converged = False
    ll = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        B, mx = _scale_rows(log_emission(Y, mu, var))
        pi0 = stationary_distribution(A)
        alpha, c = forward(B, A, pi0)
        ll = float(np.sum(np.log(c)) + mx.sum())
        if ll - ll_prev < tol * T and it > 1:
            converged = True
            break
        ll_prev = ll
        beta = backward(B, A, c)
        gamma = alpha * beta
        xi_sum = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        A, mu, var = _m_step(Y, gamma, xi_sum, prior)
    return A, mu, var, ll, it, converged


def _smooth(x: np.ndarray, span: int) -> np.ndarray:
    """Causal-within-window EWMA of a path (used only to build deterministic initial state assignments)."""
    lam = 1.0 - 2.0 / (span + 1.0)
    out = np.empty(len(x))
    s = x[0]
    for i, v in enumerate(x.tolist()):
        s = lam * s + (1.0 - lam) * v
        out[i] = s
    return out


def _init_from_labels(Y, labels, K, prior):
    w = np.zeros((len(Y), K))
    w[np.arange(len(Y)), labels] = 1.0
    cnt = np.zeros((K, K))
    np.add.at(cnt, (labels[:-1], labels[1:]), 1.0)
    A, mu, var = _m_step(Y, w, cnt, prior)
    return A, mu, var


def _gauss_logits(x, m, v, w):
    """(n, K) log(w_k) + log N(x_i; m_k, v_k) for a 1-D Gaussian mixture."""
    return np.log(w)[None, :] - 0.5 * np.log(2.0 * np.pi * v)[None, :] - 0.5 * (x[:, None] - m[None, :]) ** 2 / v[None, :]


def _gmm_labels(x: np.ndarray, K: int, iters: int = 60) -> np.ndarray:
    """Hard assignment from a 1-D Gaussian mixture fitted by EM (deterministic quantile start, vectorised).

    A 1-D mixture on the smoothed log-volatility isolates small high-volatility clusters that equal-count
    quantile splits miss (and which trap the HMM EM in poor local optima).
    """
    n = len(x)
    qs = np.quantile(x, [(i + 0.5) / K for i in range(K)])
    qs = np.sort(qs)
    m = qs.copy()
    v = np.full(K, max(np.var(x), 1e-12) / K)
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        lp = _gauss_logits(x, m, v, w)
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        m_new = (r * x[:, None]).sum(axis=0) / nk
        v = np.maximum((r * (x[:, None] - m_new[None, :]) ** 2).sum(axis=0) / nk, 1e-6 * max(np.var(x), 1e-12))
        w = np.maximum(nk / n, 1e-3)
        w /= w.sum()
        done = np.max(np.abs(m_new - m)) < 1e-5 * max(float(np.std(x)), 1e-12)
        m = m_new
        if done:
            break
    order = np.argsort(m)
    inv = np.empty(K, int)
    inv[order] = np.arange(K)
    return inv[np.argmax(_gauss_logits(x, m, v, w), axis=1)]


def _init_params(Y, K, kind, prior):
    D = Y.shape[1]
    score = _smooth(Y[:, 1] if D > 1 else np.log(np.abs(Y[:, 0] - np.median(Y[:, 0])) + 1e-4), 5)
    if kind == "gmm":
        labels = _gmm_labels(score, K)
    else:  # quantile splits: calm periods are the most frequent, crises the rarest
        base = {"skew": [0.55, 0.30, 0.15], "tail": [0.80, 0.15, 0.05]}[kind] if K == 3 else None
        if base is None:
            base = list(np.geomspace(1.0, 0.25 if kind == "tail" else 0.5, K))
        fr = np.cumsum(np.array(base) / np.sum(base))
        ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable") / len(Y)
        labels = np.minimum(np.searchsorted(fr, ranks, side="right"), K - 1)
    return _init_from_labels(Y, labels, K, prior)


def fit_hmm(Y: np.ndarray, K: int = 3, init: HMMParams | None = None, max_iter: int = 60, tol: float = 1e-5,
            prior_strength: np.ndarray | None = None, clip_q: tuple = (0.005, 0.995),
            prior_mean: np.ndarray | None = None) -> HMMParams:
    """Fit the Gaussian HMM by MAP-EM.  Deterministic function of ``(Y, K, init)``.

    ``prior_mean`` (D,) is the mean the state means shrink towards (default: the window mean); the engine passes the
    shrunk long-run drift for the return dimension so that no regime extrapolates a trailing trend.
    ``Y`` has shape (T, D) with the return in column 0.  Columns >= 1 are winsorised at the ``clip_q`` quantiles of the
    window (the bounds are stored in the result and re-applied when filtering new rows).  ``init`` warm-starts EM
    (single start); otherwise three deterministic initialisations (1-D Gaussian-mixture on the smoothed log-volatility,
    55/30/15 and 80/15/5 quantile splits) compete for 8 iterations and the winner is refined.
    """
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, D = Y.shape
    if T < 20 * K:
        raise ValueError("hmm: too few observations")
    lo = np.full(D, -np.inf)
    hi = np.full(D, np.inf)
    if D > 1:
        lo[1:] = np.quantile(Y[:, 1:], clip_q[0], axis=0)
        hi[1:] = np.quantile(Y[:, 1:], clip_q[1], axis=0)
    Yc = np.clip(Y, lo, hi)
    gm = Yc.mean(axis=0)
    gv = np.maximum(Yc.var(axis=0), 1e-12)
    prior = {"kappa": _default_prior_strength(D) if prior_strength is None else np.asarray(prior_strength, float),
             "mean": gm if prior_mean is None else np.asarray(prior_mean, float),
             "var": gv, "nu0": 3.0, "var_floor": 1e-4 * gv, "diag": 0.95, "a0": 10.0}
    if init is not None:
        A, mu, var = init.A.copy(), init.mu.copy(), np.maximum(init.var.copy(), prior["var_floor"])
        A, mu, var, ll, it, conv = _em(Yc, A, mu, var, prior, max_iter, tol)
    else:
        trials = []
        for kind in ("gmm", "skew", "tail"):
            A0, mu0, var0 = _init_params(Yc, K, kind, prior)
            A1, mu1, var1, ll1, it1, conv1 = _em(Yc, A0, mu0, var0, prior, 8, tol)
            trials.append((ll1, A1, mu1, var1))
        best = max(trials, key=lambda z: z[0])
        A, mu, var, ll, it, conv = _em(Yc, best[1], best[2], best[3], prior, max_iter, tol)
        it += 8
    order = np.argsort(var[:, 0], kind="stable")  # calmest regime first
    A, mu, var = A[np.ix_(order, order)], mu[order], var[order]
    B, mx = _scale_rows(log_emission(Yc, mu, var))
    alpha, c = forward(B, A, stationary_distribution(A))
    ll = float(np.sum(np.log(c)) + mx.sum())
    return HMMParams(A=A, mu=mu, var=var, lo=lo, hi=hi, alpha_last=alpha[-1].copy(), loglik=ll, n_obs=T, n_iter=it,
                     converged=bool(conv))


# ----------------------------------------------------------------------------- h-step predictive
def hstep_state_moments(params: HMMParams, H: int):
    """Exact mean and variance of ``S_h = r_{t+1} + ... + r_{t+h}`` (dimension 0) from every start state.

    Returns ``(mean, var)`` arrays of shape ``(H + 1, K)``; row h holds the moments for horizon h (row 0 is zero).
    """
    K = params.K
    A = params.A
    mu = params.mu[:, 0]
    m2 = params.var[:, 0] + mu * mu
    P = [np.eye(K)]
    for _ in range(H):
        P.append(P[-1] @ A)
    W = [Pd @ mu for Pd in P]  # W[d][j] = E[r_{+d} | start j]
    z = [mu * Wd for Wd in W]  # mu_j * (A^d mu)_j
    mean = np.zeros((H + 1, K))
    var = np.zeros((H + 1, K))
    for h in range(1, H + 1):
        var_h = P[h] @ m2 - W[h] ** 2  # Var(r_h | k)
        cov = np.zeros(K)
        for s in range(1, h):
            cov += P[s] @ z[h - s] - W[s] * W[h]
        mean[h] = mean[h - 1] + W[h]
        var[h] = var[h - 1] + var_h + 2.0 * cov
    return mean, np.maximum(var, 1e-18)


# ----------------------------------------------------------------------------- labels and simulation
_TIER_NAMES = {1: ["單一狀態"], 2: ["低波動", "高波動"], 3: ["低波動", "中波動", "高波動"],
               4: ["極低波動", "低波動", "高波動", "極高波動"]}


TREND_THRESHOLD = 0.10  # |mu_k / sigma_k| needed before a regime is called bullish / bearish (see regime_labels)


def regime_labels(params: HMMParams, trend_threshold: float = TREND_THRESHOLD) -> list:
    """Readable Chinese labels derived from the fitted state volatility rank and drift/volatility ratio.

    The drift descriptor uses ``mu_k / sigma_k`` (a daily Sharpe-like ratio): above ``+trend_threshold`` -> bullish,
    below ``-trend_threshold`` -> bearish, otherwise range-bound.  Nothing depends on the numeric state order, only on
    the fitted values (the states are sorted by variance, so index 0 is the calmest).

    Why 0.10 (audit M-06): the engine shrinks every regime mean towards the long-run drift with ``kappa = 500``
    pseudo-observations, so ``mu_k`` is the drift plus a small, noisy deviation and ``mu_k / sigma_k`` is dominated by
    ``drift / sigma_k`` -- largest for the calmest regime because its ``sigma_k`` is smallest.  On development rows (63
    stocks, 3,820 refits before the final 1250 bars) the calm regime's ratio had mean 0.046 and sd 0.035, i.e. it sat on the
    old threshold of 0.04 and the trend word (低波動盤整 / 低波動多頭) flipped between consecutive refits in 9.9 % of the cases;
    the middle regime had mean 0.022 / sd 0.034 and the high-volatility regime 0.017 / 0.018.  A threshold of 0.10 is the
    95th percentile of the calm regime's ratio (about 1.6 sd above its mean), so a trend word is only attached to a regime
    whose mean really stands out, and the label tuple then changed in 1.5 % of the refits.  The usual labels are therefore the
    volatility tiers with a neutral word (低波動盤整 / 中波動震盪 / 高波動震盪); nothing about the drift is claimed unless the
    fitted regime says so clearly.
    """
    K = params.K
    sd = np.sqrt(params.var[:, 0])
    ratio = params.mu[:, 0] / sd
    tiers = _TIER_NAMES.get(K, [f"波動第{i + 1}級" for i in range(K)])
    labels = []
    for k in range(K):
        trend = "多頭" if ratio[k] > trend_threshold else ("空頭" if ratio[k] < -trend_threshold else "盤整")
        tier = tiers[k]
        if K == 3 and k == 1:  # the middle regime: use the idiomatic names
            name = {"多頭": "多頭穩定", "盤整": "中波動震盪", "空頭": "空頭修正"}[trend]
        elif tier.endswith("高波動") and trend == "多頭":
            name = tier + "反彈"
        elif tier.endswith("低波動") and trend == "空頭":
            name = tier + "緩跌"
        elif trend == "盤整" and tier.endswith("高波動"):
            name = tier + "震盪"
        else:
            name = tier + trend
        labels.append(name)
    seen = {}
    for k, name in enumerate(labels):  # make labels unique
        if name in seen:
            labels[k] = f"{name}({k + 1})"
        seen[name] = k
    return labels


def simulate(params: HMMParams, T: int, rng: np.random.Generator):
    """Simulate ``(states, observations)`` from the fitted model (tests / synthetic data)."""
    K, D = params.mu.shape
    pi = stationary_distribution(params.A)
    x = np.empty(T, int)
    x[0] = rng.choice(K, p=pi)
    cum = np.cumsum(params.A, axis=1)
    u = rng.random(T)
    for t in range(1, T):
        x[t] = min(int(np.searchsorted(cum[x[t - 1]], u[t])), K - 1)
    Y = params.mu[x] + np.sqrt(params.var[x]) * rng.standard_normal((T, D))
    return x, Y
