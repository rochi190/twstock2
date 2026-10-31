"""Causal position rule, execution simulator and buy-and-hold benchmark (module C).

Timeline (no look-ahead)
------------------------
A signal / target weight ``s_t`` is computed from the forecast of origin row ``t`` (information up to the
close of day ``t``).  It can only influence returns of days ``t+1`` and later:

``exec='next_open'`` (default, realistic)
    trade at the OPEN of day t+1.  Day ``d = t+1`` therefore earns the overnight gap ``co_d`` with the position
    decided one day earlier (``s_{d-2}``), pays the cost of the trade ``s_{d-1} - s_{d-2}`` at the open, and
    earns the intraday leg ``oc_d`` with the new position ``s_{d-1}``::

        1 + r_d = (1 + s_{d-2} co_d) * (1 - cost_d) * (1 + s_{d-1} oc_d)

    which equals ``r_{t+1} = s_{t-1} co_{t+1} + s_t oc_{t+1} - cost`` of DESIGN.md up to the cross terms
    (products of two sub-daily returns, < 1e-4 per day) and makes "always long" reproduce buy-and-hold
    *exactly* (``(1+co)(1+oc) = 1+cc``).

``exec='close'`` (optimistic: trade in the closing auction of day t)::

        1 + r_{t+1} = (1 - cost) * (1 + s_t cc_{t+1})

In both modes the cost of the trade ``|s_t - s_{t-1}|`` is booked on day t+1, so ``r_{t+1}`` depends on
``s_{<=t}`` only.  ``test_eval_strategy`` proves this by perturbing ``s_t`` and the data after ``t``.

Costs (``cost_d = sum_i buy_i * max(Delta_i, 0) + sell_i * max(-Delta_i, 0)``)
    buy  = commission_rate * discount + slippage
    sell = commission_rate * discount + slippage + transaction tax (0.3 % stock / 0.1 % ETF)
    buy + sell == config.roundtrip_cost(kind, discount, slippage_bps).

Returns use adjusted OHLC (``PriceData.adj_open`` / ``adjclose``): ``co_d = adjO_d / adjC_{d-1} - 1``,
``oc_d = adjC_d / adjO_d - 1``, ``cc_d = adjC_d / adjC_{d-1} - 1``.  Days with unusable opens fall back to
``co = cc, oc = 0`` (the position simply earns the close-to-close move).

Position rule (single stock, horizon h forecast)
    ``E_simple = exp(mu + sigma^2/2) - 1`` (log-normal moment of the predictive log return),
    ``net = E_simple - roundtrip_cost``.
    ``threshold`` sizing: ``s = 1`` if ``p_up >= thr and net > 0`` else 0.
    ``kelly`` sizing:    same gate, ``s = clip(KELLY_FRACTION * net / sigma^2, 0, 1)``.
    The position is re-decided every day from the then-current h-bar forecast (no fixed holding period).

Not modelled (disclosed in the report notes): price-limit lock-outs, NT$20 minimum commission, market impact,
dividend withholding, borrowing/short selling, interest on cash, liquidation costs of the final open position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..types import HorizonForecasts, PriceData
from . import metrics as M

EXEC_MODES = ("next_open", "close")
SIZINGS = ("threshold", "kelly")


# ============================================================================= costs & small helpers
def cost_kind(kind: str) -> str:
    """Map an instrument kind to the cost schedule ('etf' keeps the 0.1 % tax, everything else 0.3 %)."""
    return "etf" if kind == "etf" else "stock"


def side_costs(kind: str = "stock", commission_discount: float = config.COMMISSION_DISCOUNT,
               slippage_bps: float = config.SLIPPAGE_BPS):
    """``(buy_cost, sell_cost)`` as fractions of traded value (``buy + sell == roundtrip_cost``)."""
    tax = config.TAX_ETF if cost_kind(kind) == "etf" else config.TAX_STOCK
    base = config.COMMISSION_RATE * commission_discount + slippage_bps / 1e4
    return base, base + tax


def expected_simple_return(mu, sigma):
    """``E[exp(R) - 1] = exp(mu + sigma^2/2) - 1`` for a log return with mean ``mu`` and std ``sigma``."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.expm1(mu + 0.5 * sigma * sigma)


def rule_text(threshold: float, sizing: str = "threshold") -> str:
    """Plain-Chinese description of the position rule."""
    pct = f"{threshold * 100:g}"
    if sizing == "kelly":
        return (f"機率≥{pct}%且扣成本後期望報酬>0 時持有，部位依 {config.KELLY_FRACTION:g} 倍凱利公式"
                f"(上限100%)配置，否則空手")
    return f"機率≥{pct}%且扣成本後期望報酬>0 時持有，否則空手"


def leg_returns(data: PriceData):
    """Simple overnight / intraday / close-to-close returns from adjusted prices -> ``(co, oc, cc)``.

    Index 0 has no previous close and is 0.  ``(1 + co)(1 + oc) = 1 + cc`` holds exactly.
    """
    c = np.asarray(data.adjclose, dtype=float)
    o = np.asarray(data.adj_open, dtype=float)
    n = len(c)
    co = np.zeros(n)
    oc = np.zeros(n)
    cc = np.zeros(n)
    if n < 2:
        return co, oc, cc
    with np.errstate(divide="ignore", invalid="ignore"):
        cc[1:] = c[1:] / c[:-1] - 1.0
        co[1:] = o[1:] / c[:-1] - 1.0
        oc[1:] = c[1:] / o[1:] - 1.0
    bad_cc = ~np.isfinite(cc)
    cc[bad_cc] = 0.0
    bad_leg = bad_cc | ~np.isfinite(co) | ~np.isfinite(oc) | ~(o > 0)
    co[bad_leg] = cc[bad_leg]
    oc[bad_leg] = 0.0
    return co, oc, cc


# ============================================================================= position rule
def positions_from_forecast(hf: HorizonForecasts, *, kind: str = "stock", threshold: float = config.BUY_P_UP,
                            sizing: str = "threshold", commission_discount: float = config.COMMISSION_DISCOUNT,
                            slippage_bps: float = config.SLIPPAGE_BPS) -> np.ndarray:
    """Target weight ``s_t in [0, 1]`` per origin row (0 where no valid forecast).  Uses only row-t forecasts."""
    if sizing not in SIZINGS:
        raise ValueError(f"unknown sizing {sizing!r}")
    p = np.asarray(hf.p_up, dtype=float)
    mu = np.asarray(hf.mu, dtype=float)
    sg = np.asarray(hf.sigma, dtype=float)
    valid = np.isfinite(p) & np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
    cost = config.roundtrip_cost(cost_kind(kind), commission_discount, slippage_bps)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        net = expected_simple_return(mu, sg) - cost
        on = valid & (p >= threshold) & (net > 0)
        if sizing == "kelly":
            w = np.clip(config.KELLY_FRACTION * net / (sg * sg), 0.0, 1.0)
            s = np.where(on, w, 0.0)
        else:
            s = on.astype(float)
    return np.where(np.isfinite(s), s, 0.0)


# ============================================================================= simulator
def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def simulate_weights(co, oc, W, buy_cost, sell_cost, *, exec_mode: str = "next_open", first: int = 0) -> dict:
    """Portfolio simulation from target weights decided at each close (rows ``first .. n-2`` are used).

    Parameters
    ----------
    co, oc : (n,) or (n, S) simple overnight / intraday returns of row d (the legs of day d).
    W : (n,) or (n, S) target weight decided at the close of row t (NaN = 0).  Rows before ``first`` are
        treated as cash, so the first trade happens on day ``first + 1``.
    buy_cost, sell_cost : scalar or (S,) fractional costs per side.

    Returns arrays of length ``N = n - 1 - first`` (day ``first+1 .. n-1``): ``ret`` (net), ``gross``, ``cost``,
    ``turnover`` (sum |delta weight|), ``w_held`` (N, S) the weight decided the previous close, ``exposure``.
    """
    if exec_mode not in EXEC_MODES:
        raise ValueError(f"unknown exec_mode {exec_mode!r}")
    co2, oc2 = _as2d(co), _as2d(oc)
    W2 = np.nan_to_num(_as2d(W), nan=0.0, posinf=0.0, neginf=0.0)
    n, S = W2.shape
    N = n - 1 - int(first)
    empty = np.zeros(0)
    if N <= 0:
        return {"ret": empty, "gross": empty, "cost": empty, "turnover": empty, "w_held": np.zeros((0, S)),
                "exposure": empty, "first": int(first)}
    bc = np.broadcast_to(np.asarray(buy_cost, dtype=float), (S,))
    sc = np.broadcast_to(np.asarray(sell_cost, dtype=float), (S,))
    w1 = W2[first:n - 1]                                              # decided at close of day d-1
    w2 = np.vstack([np.zeros((1, S)), W2[first:n - 2]])               # decided at close of day d-2
    co_d, oc_d = np.nan_to_num(co2[first + 1:n]), np.nan_to_num(oc2[first + 1:n])
    delta = w1 - w2
    cost = (np.maximum(delta, 0.0) * bc + np.maximum(-delta, 0.0) * sc).sum(axis=1)
    if exec_mode == "next_open":
        g_co = (w2 * co_d).sum(axis=1)
        g_oc = (w1 * oc_d).sum(axis=1)
        gross = (1.0 + g_co) * (1.0 + g_oc) - 1.0
        ret = (1.0 + g_co) * (1.0 - cost) * (1.0 + g_oc) - 1.0
    else:
        cc_d = (1.0 + co_d) * (1.0 + oc_d) - 1.0
        gross = (w1 * cc_d).sum(axis=1)
        ret = (1.0 - cost) * (1.0 + gross) - 1.0
    return {"ret": ret, "gross": gross, "cost": cost, "turnover": np.abs(delta).sum(axis=1), "w_held": w1,
            "exposure": w1.sum(axis=1), "first": int(first)}


def trade_episode_returns(signal, ret) -> np.ndarray:
    """Net compounded return of every holding episode.

    ``signal[j] > 0`` marks a held target decided at close j; it first earns on day index j.  A run of
    signals ``a..b`` touches days ``a .. min(b+1, N-1)`` (the extra day carries the overnight leg / exit cost).
    """
    held = np.asarray(signal, dtype=float) > 0
    ret = np.asarray(ret, dtype=float)
    N = len(held)
    if N == 0 or not held.any():
        return np.zeros(0)
    d = np.diff(np.concatenate([[0], held.astype(int), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return np.array([np.prod(1.0 + ret[a:min(b + 1, N - 1) + 1]) - 1.0 for a, b in zip(starts, ends)])


def stats_block(ret, exposure, turnover, episode_returns, *, n_trials: int = 1, sharpe_ci=None,
                ppy: float = M.TRADING_DAYS) -> dict:
    """STATS dict of DESIGN.md 6.4 (key order as in the schema)."""
    st = M.return_stats(ret, ppy=ppy, n_trials=n_trials, sharpe_ci=sharpe_ci)
    n = len(ret)
    years = n / ppy if n else 0.0
    ep = np.asarray(episode_returns, dtype=float)
    return {"total_return": st["total_return"], "cagr": st["cagr"], "vol": st["vol"], "sharpe": st["sharpe"],
            "sortino": st["sortino"], "max_dd": st["max_dd"], "calmar": st["calmar"],
            "exposure": float(np.mean(exposure)) if n else None,
            "turnover_per_year": float(np.sum(turnover) / years) if years > 0 else None,
            "n_trades": int(len(ep)), "trade_hit_rate": float(np.mean(ep > 0)) if len(ep) else None,
            "psr": st["psr"], "dsr": st["dsr"], "sharpe_ci95": st["sharpe_ci95"]}


def empty_stats() -> dict:
    """All-null STATS block (used when a window is too short to simulate)."""
    return {"total_return": None, "cagr": None, "vol": None, "sharpe": None, "sortino": None, "max_dd": None,
            "calmar": None, "exposure": None, "turnover_per_year": None, "n_trades": 0, "trade_hit_rate": None,
            "psr": None, "dsr": None, "sharpe_ci95": [None, None]}


# ============================================================================= single-stock strategy
@dataclass
class StrategyResult:
    """Output of :func:`run_strategy`: the report ``strategy`` section plus numbers the verdict needs."""
    section: dict
    psr_vs_bh: float | None = None       # PSR of the net strategy vs the buy-and-hold Sharpe (per period)
    diff_p: float | None = None          # bootstrap one-sided p of SR_strategy > SR_buyhold
    sim: dict = field(default_factory=dict)
    s_signal: np.ndarray | None = None


def run_strategy(data: PriceData, hf: HorizonForecasts, first_idx: int, *, kind: str | None = None,
                 threshold: float = config.BUY_P_UP, exec_mode: str = "next_open",
                 commission_discount: float = config.COMMISSION_DISCOUNT, slippage_bps: float = config.SLIPPAGE_BPS,
                 sizing: str = "threshold", n_trials: int = 1, n_boot: int = 1000) -> StrategyResult:
    """Simulate the position rule on ``data`` from origin row ``first_idx`` and build the ``strategy`` section.

    Day ``first_idx + 1`` is the first day that can earn a return; buy-and-hold is bought at the close of
    ``first_idx`` (equity 1.0 on ``dates[first_idx]``) with **no** costs (the harder benchmark).  Under
    ``exec='next_open'`` the strategy cannot trade before the open of ``first_idx + 1`` and therefore (by construction,
    like any real investor) misses that first overnight gap, which buy-and-hold earns.
    """
    kind = cost_kind(kind if kind is not None else data.kind)
    n = data.n
    s = positions_from_forecast(hf, kind=kind, threshold=threshold, sizing=sizing,
                                commission_discount=commission_discount, slippage_bps=slippage_bps)
    s[:first_idx] = 0.0
    co, oc, cc = leg_returns(data)
    buy, sell = side_costs(kind, commission_discount, slippage_bps)
    sim = simulate_weights(co, oc, s, buy, sell, exec_mode=exec_mode, first=first_idx)
    ret = sim["ret"]
    N = len(ret)
    section = {"rule": rule_text(threshold, sizing), "stats": empty_stats(), "benchmark": empty_stats(),
               "sharpe_diff_ci95": [None, None],
               "curve": {"dates": [], "strategy": [], "buyhold": [], "position": [], "drawdown_strategy": [],
                         "drawdown_buyhold": []}}
    if N < 1:
        return StrategyResult(section, None, None, sim, s)
    bh = cc[first_idx + 1:n]
    eps = trade_episode_returns(sim["w_held"][:, 0], ret)
    boot = M.bootstrap_sharpe(ret, bh, n_boot=n_boot)
    stats_s = stats_block(ret, sim["exposure"], sim["turnover"], eps, n_trials=n_trials, sharpe_ci=boot["ci1"])
    bh_turn = np.zeros(N)
    bh_turn[0] = 1.0
    bh_eps = np.array([np.prod(1.0 + bh) - 1.0])
    stats_b = stats_block(bh, np.ones(N), bh_turn, bh_eps, n_trials=1, sharpe_ci=boot["ci2"])
    sd_b = float(np.std(bh, ddof=1)) if N > 1 else 0.0
    sr_bh = float(np.mean(bh)) / sd_b if sd_b > 1e-12 else float("nan")
    psr_vs_bh = M.finite_or_none(M.probabilistic_sharpe_ratio(ret, sr_bh)) if np.isfinite(sr_bh) else None
    eq_s, eq_b = M.equity_from_returns(ret), M.equity_from_returns(bh)
    pos_full = np.concatenate([[0.0], sim["exposure"]])
    ix = M.downsample_indices(N + 1)
    dates = np.concatenate([[data.dates[first_idx]], data.dates[first_idx + 1:n]])
    section.update(stats=stats_s, benchmark=stats_b, sharpe_diff_ci95=boot["diff_ci"])
    section["curve"] = {
        "dates": [M.date_str(d) for d in dates[ix]],
        "strategy": np.round(eq_s[ix], 6).tolist(), "buyhold": np.round(eq_b[ix], 6).tolist(),
        "position": np.round(pos_full[ix], 4).tolist(),
        "drawdown_strategy": np.round(M.drawdown_series(eq_s)[ix], 6).tolist(),
        "drawdown_buyhold": np.round(M.drawdown_series(eq_b)[ix], 6).tolist()}
    return StrategyResult(section, psr_vs_bh, boot["diff_p"], sim, s)
