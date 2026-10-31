"""Synthetic PriceData / WalkForward builders for the evaluation tests (no network, no quant engine).

Planted-edge world
------------------
A latent AR(1) signal ``m_t`` (stationary sd ``signal_sd``, persistence ``rho``) is known at close ``t`` and shifts the
expected log return of day ``t+1``::

    eps_{t+1} = drift + m_t + vol * z_{t+1}          (split into an overnight and an intraday leg)

so the exact predictive law of the h-bar log return R from origin t is Normal with

    mean   mu_h(t) = h drift + m_t (1 - rho^h) / (1 - rho)
    var    h vol^2 + s^2 (1 - rho^2) sum_{j=1}^{h-1} ((1 - rho^{h-j}) / (1 - rho))^2      (conditional on m_t)

``edge='edge'``   the model forecasts with the true conditional law   -> real, calibrated edge
``edge='none'``   the model knows only drift and the unconditional vol -> matches the base rate, calibrated
``edge='noise'``  the model conditions on an independent fake signal   -> over-confident, no skill
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from backend import config  # noqa: E402
from backend.types import HorizonForecasts, PriceData, WalkForward  # noqa: E402


def business_days(n: int, start: str = "2015-01-05") -> np.ndarray:
    """``n`` consecutive weekdays as datetime64[D]."""
    return np.busday_offset(np.datetime64(start), np.arange(n), roll="forward")


def ar1_latent(n: int, sd: float, rho: float, rng) -> np.ndarray:
    """Stationary AR(1) series with marginal sd ``sd``."""
    m = np.zeros(n)
    eta = rng.normal(0.0, 1.0, n)
    m[0] = sd * eta[0]
    k = sd * np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        m[t] = rho * m[t - 1] + k * eta[t]
    return m


def make_price_data(n: int = 1800, seed: int = 0, *, drift: float = 0.0004, vol: float = 0.015,
                    overnight_share: float = 0.4, latent=None, kind: str = "stock", symbol: str = "TEST",
                    name: str = "測試股", start: str = "2015-01-05", market_factor=None, beta: float = 0.0
                    ) -> PriceData:
    """Random-walk bars with distinct overnight/intraday legs.  ``latent[t]`` shifts the next day's mean."""
    rng = np.random.default_rng(seed)
    mean = np.full(n, drift)
    if latent is not None:
        mean[1:] += np.asarray(latent)[:-1]
    sd_o, sd_i = vol * np.sqrt(overnight_share), vol * np.sqrt(1.0 - overnight_share)
    a = mean * overnight_share + rng.normal(0.0, sd_o, n)
    b = mean * (1.0 - overnight_share) + rng.normal(0.0, sd_i, n)
    if market_factor is not None:
        a = a + beta * np.asarray(market_factor) * np.sqrt(overnight_share)
        b = b + beta * np.asarray(market_factor) * np.sqrt(1.0 - overnight_share)
    a[0] = 0.0
    b[0] = 0.0
    close = np.empty(n)
    opn = np.empty(n)
    close[0] = 100.0
    opn[0] = 100.0
    for t in range(1, n):
        opn[t] = close[t - 1] * np.exp(a[t])
        close[t] = opn[t] * np.exp(b[t])
    spread = np.exp(np.abs(rng.normal(0.0, 0.3 * vol, n)))
    high = np.maximum(opn, close) * spread
    low = np.minimum(opn, close) / spread
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return PriceData(symbol, name, "TWSE", kind, business_days(n, start), opn, high, low, close, close.copy(), volume)


def make_tiny_data(opens, closes, *, adj=None, kind: str = "stock", symbol: str = "TINY", start: str = "2020-01-06"
                   ) -> PriceData:
    """Hand-specified bars; ``adj`` optional adjclose (defaults to close, i.e. no dividends)."""
    o = np.asarray(opens, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    adjc = c.copy() if adj is None else np.asarray(adj, dtype=float)
    return PriceData(symbol, "微型", "TWSE", kind, business_days(n, start), o, np.maximum(o, c), np.minimum(o, c),
                     c, adjc, np.full(n, 1e6))


def realized_from_data(data: PriceData, h: int) -> np.ndarray:
    a = data.adjclose
    r = np.full(len(a), np.nan)
    if h < len(a):
        r[: len(a) - h] = np.log(a[h:] / a[: len(a) - h])
    return r


def make_hf(data: PriceData, h: int, first_idx: int, p_up, mu, sigma, *, qlevels=config.QLEVELS, pit=None,
            q=None, realized=None) -> HorizonForecasts:
    """HorizonForecasts from explicit arrays (NaN before ``first_idx``).  Normal quantiles/PIT unless given."""
    n = data.n
    p_up = np.broadcast_to(np.asarray(p_up, dtype=float), (n,)).copy()
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n,)).copy()
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
    for arr in (p_up, mu, sigma):
        arr[:first_idx] = np.nan
    R = realized_from_data(data, h) if realized is None else np.asarray(realized, dtype=float)
    z = stats.norm.ppf(np.asarray(qlevels))
    if q is None:
        with np.errstate(invalid="ignore"):
            q = mu[:, None] + sigma[:, None] * z[None, :]
    if pit is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            pit = stats.norm.cdf((R - mu) / sigma)
        pit[np.isnan(R)] = np.nan
    return HorizonForecasts(h=h, p_up=p_up, p_up_raw=p_up.copy(), mu=mu, sigma=sigma, q=q, pit=pit, realized=R,
                            comp_p_up={}, comp_mu={}, weights={})


def make_wf(data: PriceData, hfs: dict, first_idx: int, *, qlevels=config.QLEVELS) -> WalkForward:
    n = data.n
    return WalkForward(symbol=data.symbol, dates=data.dates.copy(), horizons=dict(hfs), first_idx=first_idx,
                       refit_every=21, min_train=config.MIN_TRAIN, model_names=["garch", "hmm", "dlm", "naive"],
                       regime_prob=np.full((n, 3), 1.0 / 3.0), regime_labels=["低波動盤整", "多頭穩定", "高波動空頭"],
                       vol1d=np.full(n, 0.015), config={}, qlevels=tuple(qlevels))


def _cond_var(h: int, vol: float, s: float, rho: float) -> float:
    """Var of R_h given m_t (see module docstring)."""
    if h == 1:
        return vol * vol
    j = np.arange(1, h)
    return h * vol * vol + s * s * (1 - rho * rho) * float(np.sum(((1 - rho ** (h - j)) / (1 - rho)) ** 2))


def _uncond_var(h: int, vol: float, s: float, rho: float) -> float:
    k = np.arange(1, h)
    return h * vol * vol + s * s * (h + 2.0 * float(np.sum((h - k) * rho ** k)))


def make_planted_world(n: int = 1800, seed: int = 0, *, h_list=(1, 5, 20), first_idx: int = 500, edge: str = "edge",
                       signal_sd: float = 0.003, rho: float = 0.9, vol: float = 0.015, drift: float = 0.0004,
                       kind: str = "stock", symbol: str = "TEST", name: str = "測試股", overconfidence: float = 1.0):
    """``(data, wf, latent)`` for the three scenarios described in the module docstring."""
    rng = np.random.default_rng(seed + 10_000)
    m = ar1_latent(n, signal_sd, rho, rng)
    data = make_price_data(n, seed, drift=drift, vol=vol, latent=m, kind=kind, symbol=symbol, name=name)
    fake = ar1_latent(n, signal_sd, rho, np.random.default_rng(seed + 20_000))
    hfs = {}
    for h in h_list:
        gain = (1 - rho ** h) / (1 - rho)
        if edge == "edge":
            mu = h * drift + m * gain
            sd = np.sqrt(_cond_var(h, vol, signal_sd, rho))
        elif edge == "none":
            mu = np.full(n, h * drift)
            sd = np.sqrt(_uncond_var(h, vol, signal_sd, rho))
        elif edge == "noise":
            mu = h * drift + overconfidence * fake * gain
            sd = np.sqrt(_cond_var(h, vol, signal_sd, rho))
        else:
            raise ValueError(edge)
        sigma = np.full(n, sd)
        p_up = stats.norm.cdf(mu / sigma)
        hfs[h] = make_hf(data, h, first_idx, p_up, mu, sigma)
    return data, make_wf(data, hfs, first_idx), m


def make_panel(n_stocks: int = 14, n: int = 900, seed: int = 0, *, edge: str = "edge", h_list=(1, 5),
               first_idx: int = 300, signal_sd: float = 0.003, rho: float = 0.9, vol: float = 0.015,
               drift: float = 0.0003):
    """Panel ``{code: (PriceData, WalkForward)}`` with a common market factor and per-stock planted signals."""
    rng = np.random.default_rng(seed + 777)
    market = rng.normal(0.0, 0.008, n)
    panel = {}
    for i in range(n_stocks):
        code = f"{1101 + i}"
        m = ar1_latent(n, signal_sd, rho, np.random.default_rng(seed * 1000 + i))
        data = make_price_data(n, seed * 1000 + 500 + i, drift=drift, vol=vol, latent=m, symbol=code,
                               name=f"股{i}", market_factor=market, beta=0.8 + 0.05 * i)
        fake = ar1_latent(n, signal_sd, rho, np.random.default_rng(seed * 1000 + 300 + i))
        total_vol = float(np.sqrt(vol ** 2 + (0.8 + 0.05 * i) ** 2 * 0.008 ** 2))
        hfs = {}
        for h in h_list:
            gain = (1 - rho ** h) / (1 - rho)
            if edge == "edge":
                mu = h * drift + m * gain
            elif edge == "noise":
                mu = h * drift + fake * gain
            else:
                mu = np.full(n, h * drift)
            sd = np.sqrt(_cond_var(h, total_vol, signal_sd, rho))
            sigma = np.full(n, sd)
            hfs[h] = make_hf(data, h, first_idx, stats.norm.cdf(mu / sigma), mu, sigma)
        panel[code] = (data, make_wf(data, hfs, first_idx))
    return panel


# ----------------------------------------------------------------------------- tick-size / bid-ask worlds (audit C-02, C-08)
def make_tick_data(n: int = 1500, seed: int = 0, *, tick: float = 0.5, vol: float = 0.012, drift: float = 0.0004,
                   symbol: str = "TICK", name: str = "跳動單位股") -> PriceData:
    """Random walk whose closes are rounded to a price tick: a good share of the h = 1 outcomes is EXACTLY zero
    (tick / (price * sigma) ~ 0.4 gives about 15 % unchanged closes), like the low-priced Taiwanese names of the real data."""
    rng = np.random.default_rng(seed)
    logp = np.log(100.0) + np.cumsum(drift + vol * rng.standard_normal(n))
    close = np.maximum(np.round(np.exp(logp) / tick) * tick, tick)
    opn = np.concatenate([[close[0]], close[:-1]])
    hi, lo = np.maximum(opn, close) * 1.002, np.minimum(opn, close) / 1.002
    return PriceData(symbol, name, "TWSE", "stock", business_days(n), opn, hi, lo, close, close.copy(), np.full(n, 5e6))


def make_bounce_data(n: int = 1800, seed: int = 0, *, vol: float = 0.008, bounce: float = 0.005, drift: float = 0.0004,
                     symbol: str = "BNC", name: str = "買賣價反彈股") -> PriceData:
    """Efficient log price = random walk; the close is the efficient price plus a random bid / ask half-spread
    (+-``bounce``), so daily returns have a negative lag-1 autocorrelation of about -bounce^2 / (vol^2 + 2 bounce^2)
    (-0.22 for the defaults) that a reversal feature can forecast (audit C-08: tick size ~ half a daily sigma)."""
    rng = np.random.default_rng(seed)
    eff = np.log(100.0) + np.cumsum(drift + vol * rng.standard_normal(n))
    close = np.exp(eff + bounce * rng.choice([-1.0, 1.0], n))
    opn = np.concatenate([[close[0]], close[:-1]])
    hi, lo = np.maximum(opn, close) * 1.002, np.minimum(opn, close) / 1.002
    return PriceData(symbol, name, "TWSE", "stock", business_days(n), opn, hi, lo, close, close.copy(), np.full(n, 5e6))


def make_reversal_world(n: int = 1800, seed: int = 0, first_idx: int = 500, *, p_const: float = 0.55, **kw):
    """``(data, wf, ac1)``: bounce prices + a forecaster whose MEAN uses the reversal (IC / MSE significant) while its
    probability ``p_up`` is a constant (no direction skill: it always calls 'up', i.e. it equals the always-up baseline)."""
    data = make_bounce_data(n, seed, **kw)
    r1 = np.concatenate([[0.0], np.diff(np.log(data.adjclose))])
    ac1 = float(np.corrcoef(r1[1:-1], r1[2:])[0, 1])
    mu = 0.0004 + ac1 * r1                               # E[r_{t+1} | r_t] = ac1 * r_t  (ac1 < 0: reversal)
    hf = make_hf(data, 1, first_idx, p_const, mu, 0.0095)
    return data, make_wf(data, {1: hf}, first_idx), ac1


# ----------------------------------------------------------------------------- heterogeneous-volatility universe (audit C-06 / S-05)
def make_vol_panel(n_stocks: int = 16, n: int = 900, seed: int = 0, *, h_list=(1, 5), first_idx: int = 300,
                   drift: float = 0.0003, vol_lo: float = 0.008, vol_hi: float = 0.030, view_sd: float = 0.012,
                   rho: float = 0.9, beta_slope: float = 0.0, market_drift: float = 0.0):
    """No-skill universe with HETEROGENEOUS volatility and a view that is independent of volatility.

    Stock i has daily vol linearly spaced in [vol_lo, vol_hi] (the stock index orders them) and i.i.d. returns (plus
    ``beta * market`` with ``beta = beta_slope * vol`` when ``beta_slope`` > 0: high-vol names are high-beta).  The model's
    *view* is an AR(1) information ratio z_i(t) (sd ``view_sd``) that is unrelated to the realised returns and to the vol:
    mu_h = z sigma_h (no common drift in the forecast mean, so ``rank_signal = mu / sigma = z`` is independent of the vol),
    sigma_h = vol sqrt(h), p_up = Phi(z).  Every net return is negative at h = 1 (and almost always at h = 5), so the
    cost-adjusted score ranks volatility (net / sigma ~ -cost / sigma + sigma / 2) while ``rank_signal`` does not.
    ``market_drift`` (per-day mean of the market factor, default 0) prices the beta: with ``beta_slope`` > 0 the high-vol names then
    earn ``beta * market_drift`` more -- a premium that a volatility ranking collects and a vol-independent ranking does not.
    """
    rng = np.random.default_rng(seed + 4242)
    market = rng.normal(market_drift, 0.008, n)                 # market_drift > 0: a PRICED market factor (beta x premium)
    vols = np.linspace(vol_lo, vol_hi, n_stocks)
    panel = {}
    for i in range(n_stocks):
        code = f"{2001 + i}"
        beta = beta_slope * vols[i]
        data = make_price_data(n, seed * 1000 + 700 + i, drift=drift, vol=vols[i], symbol=code, name=f"波動{i}",
                               market_factor=market if beta_slope else None, beta=beta)
        z = ar1_latent(n, view_sd, rho, np.random.default_rng(seed * 1000 + 900 + i))
        hfs = {}
        for h in h_list:
            sg = vols[i] * np.sqrt(h)
            if beta_slope:
                sg = float(np.sqrt(sg ** 2 + (beta * 0.008) ** 2 * h))
            mu = z * sg
            sigma = np.full(n, sg)
            hfs[h] = make_hf(data, h, first_idx, stats.norm.cdf(mu / sigma), mu, sigma)
        panel[code] = (data, make_wf(data, hfs, first_idx))
    return panel
