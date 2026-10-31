"""Signal annotation and the daily recommendation list (DESIGN.md 6.3 + 9).

Two separate jobs (DESIGN.md 9.1)
---------------------------------
**Relative ranking** -- ``rank_signal = mu_log / sigma_log`` of the horizon forecast (log-return information ratio; falls back
to ``Phi^-1(p_up)`` when either is missing).  ``rank``, the order of ``all``, ``leaders``, ``laggards``, ``lean`` and
``score_rank_pct`` are based on it (descending, ties by ``p_up`` then code) and on nothing else.  It contains neither a trading
cost nor the convexity term ``sigma^2 / 2``: the cost-adjusted score below ranks VOLATILITY (every net return is negative at
h = 1, so ``net / sigma`` is largest for the most volatile names; Spearman(score, sigma) = 0.98 / 0.94 / 0.72 at h = 1 / 5 / 20 on
the real 60-name universe).  HONESTY CAVEAT: because forecast means are heavily shrunk towards a common drift ``rank_signal``
is itself a low-volatility-tilted ordering (mean daily Spearman with sigma -0.4 .. -0.6 on the real panel).  It is the model's
relative view, not proof of stock-picking skill; every row therefore carries ``sigma`` (and ``low_vol``).

**Tradability** -- ``score`` / ``score_shrunk`` (below) are kept ONLY for the strict ``signal`` gate (``buy`` / ``avoid``),
``edge_covers_cost`` and the Kelly weights.  A name with ``net_exp_return <= 0`` has no tradable edge and gets no weight; a
negative ``-cost / sigma`` never takes part in an ordering.

Score
-----
``net_h  = E[simple return_h] - roundtrip_cost(kind)``       (cost-adjusted expected return)
``score_h = net_h / sigma_h``                                   (Sharpe-like, per forecast horizon)

Winner's-curse control: cross-sectional empirical-Bayes (Efron-Morris / Lindley James-Stein) shrinkage
-----------------------------------------------------------------------------------------------------
A forecast mean is an *estimate* with sampling noise.  Treating the ``score`` of each name as
``score_i = theta_i + e_i`` with ``e_i ~ N(0, h / T)`` (T = ``SHRINK_BARS`` effective daily observations behind a
forecast mean, h-bar targets overlap so only T/h of them are independent), the standardised
``z_i = score_i * sqrt(T / h)`` has unit noise and the positive-part James-Stein estimator towards the
cross-sectional mean is::

    theta_hat_i = zbar + max(0, 1 - (k - 3) / sum_j (z_j - zbar)^2) * (z_i - zbar),         k = number of names

(k < 4: no shrinkage).  ``score_shrunk_i = theta_hat_i * sqrt(h / T)``.  The shrunk scores gate the weights: when the
retention ``B`` is 0 (cross-section no more dispersed than noise) every shrunk edge collapses to the same common value, which is
positive only because the market drift is, so NO weights are suggested (100 % cash, ``portfolio.suppressed_reason = "B0"``);
the same holds when no ``buy`` name has a positive shrunk net edge (``"no_positive_edge"``).  ``B`` measures the dispersion of the
cost-adjusted score: at short horizons that is mostly ``cost / sigma`` (a constant per name, not an estimate), so it is NOT a
statement about stock-picking signal and is only used as the weight gate.

Suggested weights: fractional Kelly of the shrunk edge
-----------------------------------------------------
For each ``buy`` name (at most ``top_k``): ``w_i = f * max(score_shrunk_i * sigma_i, 0) / sigma_i^2`` with
``f = KELLY_FRACTION``, capped at ``MAX_WEIGHT``, scaled down proportionally if the sum exceeds 1 (remainder
is cash).  With an optional matrix of recent returns the diagonal covariance is replaced by the Ledoit-Wolf
shrinkage covariance (``w = f * Sigma^-1 mu`` on the long-only active set).

Signals (per horizon; ``annotate_forecast``)
    buy   : p_up >= buy_p  and net_exp_return > 0  (and liquid, in ``build_recommendations``)   -> UI: 偏多候選
    avoid : p_up <= avoid_p and net_exp_return < 0                                              -> UI: 偏空／避開
    hold  : otherwise
Confidence 高/中/低 from |p_up - 0.5|, agreement of the model components and the volatility percentile.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.special import ndtri

from .. import config
from . import metrics as M
from .strategy import cost_kind

SHRINK_BARS = 250            # effective daily observations behind one forecast mean (see module docstring)
CONF_LABELS = ("低", "中", "高")
HORIZON_LABELS = {1: "明日", 5: "未來5日", 20: "未來20日"}
RANK_P_CLIP = 1e-6           # p_up clip of the Phi^-1 fallback of rank_signal
NO_VIEW_NOTE = "個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移，"
LEAN_TOP_PCT = 0.8           # lean: top quintile of rank_signal (score_rank_pct >= 0.8) and p_up > 0.5 -> bullish
LEAN_BOTTOM_PCT = 0.2        # bottom quintile (score_rank_pct <= 0.2) and p_up < 0.5 -> bearish


# ============================================================================= small helpers
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fin(x) -> bool:
    return math.isfinite(_f(x))


def _d(x) -> dict:
    """``x`` if it is a dict else ``{}`` (partial / corrupted payloads must degrade, not crash)."""
    return x if isinstance(x, dict) else {}


def _hz(fc: dict, h: int):
    hs = _d(_d(fc).get("horizons"))
    for key in (str(h), int(h)):
        if isinstance(hs.get(key), dict):
            return hs[key]
    return None


def plausible(p: float, er: float, sg: float) -> bool:
    """Finite numbers in a sane range: 0 <= p <= 1, |E[return]| < 500 %, 0 < sigma < 500 %."""
    return (math.isfinite(p) and math.isfinite(er) and math.isfinite(sg) and 0.0 <= p <= 1.0 and abs(er) < 5.0
            and 0.0 < sg < 5.0)


def rank_signal_value(mu_log, sigma_log, p_up) -> float:
    """Relative-strength key ``mu_log / sigma_log`` (log-return information ratio) of one horizon forecast.

    Falls back to ``Phi^-1(p_up)`` (the same quantity for a normal predictive) when the mean or the volatility is missing /
    not usable; NaN when neither is available.  No trading cost, no ``sigma^2 / 2`` convexity term (DESIGN.md 9.1).
    """
    mu, sg, p = _f(mu_log), _f(sigma_log), _f(p_up)
    if math.isfinite(mu) and math.isfinite(sg) and sg > 0:
        return mu / sg
    if math.isfinite(p):
        return float(ndtri(min(max(p, RANK_P_CLIP), 1.0 - RANK_P_CLIP)))
    return float("nan")


def rank_signal_array(mu, sigma, p_up) -> np.ndarray:
    """Vectorised :func:`rank_signal_value` (same fallback rule) for arrays of forecasts."""
    mu, sg, p = (np.asarray(a, dtype=float) for a in (mu, sigma, p_up))
    ok = np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = mu / sg
        fallback = ndtri(np.clip(p, RANK_P_CLIP, 1.0 - RANK_P_CLIP))
    return np.where(ok, ratio, fallback)


def _check_thresholds(buy_p: float, avoid_p: float) -> None:
    if not (0.0 < avoid_p < buy_p < 1.0):
        raise ValueError("need 0 < avoid_p < buy_p < 1")


def horizon_label(h: int) -> str:
    return HORIZON_LABELS.get(int(h), f"未來{int(h)}日")


def default_kind(fc: dict) -> str:
    """Instrument kind of a forecast dict ('etf' if declared or the code starts with 00)."""
    k = _d(fc).get("kind")
    if k in ("etf", "stock"):
        return k
    return "etf" if str(_d(fc).get("symbol", "")).startswith("00") else "stock"


# ============================================================================= confidence & signals
def agreement(p_up: float, hz: dict):
    """Fraction of model components whose direction (sign of p_up - 0.5) matches the combined forecast."""
    comps = _d(_d(hz).get("components"))
    sgn = np.sign(p_up - 0.5)
    votes = [np.sign(_f(c.get("p_up")) - 0.5) for c in comps.values() if isinstance(c, dict) and _fin(c.get("p_up"))]
    if not votes or sgn == 0:
        return None
    return float(np.mean([v == sgn for v in votes]))


def confidence_label(p_up: float, hz: dict, fc: dict | None = None) -> str:
    """高/中/低 confidence in the direction call.

    base level from the edge ``|p_up - 0.5|`` (>= 0.10 -> 2, >= 0.05 -> 1, else 0); -1 if fewer than half of the
    components agree, 2 -> 1 if fewer than 3/4 agree; -1 when current volatility is above its 90th percentile.
    """
    if not _fin(p_up):
        return "低"
    edge = round(abs(p_up - 0.5), 9)          # rounding: 60 % and 40 % must classify identically
    level = 2 if edge >= 0.10 else (1 if edge >= 0.05 else 0)
    ag = agreement(p_up, hz)
    if ag is not None:
        if ag < 0.5:
            level -= 1
        elif ag < 0.75 and level == 2:
            level = 1
    pct = _f(_d(_d(fc).get("vol")).get("percentile"))
    if math.isfinite(pct) and pct >= 0.90:
        level -= 1
    return CONF_LABELS[max(0, min(2, level))]


def classify_signal(p_up: float, net: float, buy_p: float, avoid_p: float, liquid: bool = True) -> str:
    if not (math.isfinite(p_up) and math.isfinite(net)):
        return "hold"
    if p_up >= buy_p and net > 0 and liquid:
        return "buy"
    if p_up <= avoid_p and net < 0:
        return "avoid"
    return "hold"


def score_horizon(hz: dict, fc: dict, cost: float, buy_p: float, avoid_p: float, liquid: bool = True) -> dict:
    """``{"net_exp_return", "score", "signal", "confidence"}`` for one horizon dict of a forecast."""
    p, er, sg = _f(hz.get("p_up")), _f(hz.get("exp_return")), _f(hz.get("sigma"))
    ok = plausible(p, er, sg)
    net = er - cost if math.isfinite(er) and abs(er) < 5.0 else float("nan")
    score = net / sg if ok else float("nan")
    return {"net_exp_return": net if math.isfinite(net) else None,
            "score": score if math.isfinite(score) else None,
            "signal": classify_signal(p, net, buy_p, avoid_p, liquid) if ok else "hold",
            "confidence": confidence_label(p, hz, fc) if ok else "低"}


def annotate_forecast(fc: dict, kind: str, *, buy_p: float = config.BUY_P_UP, avoid_p: float = config.AVOID_P_UP,
                      commission_discount: float = config.COMMISSION_DISCOUNT,
                      slippage_bps: float = config.SLIPPAGE_BPS) -> dict:
    """Return a **deep copy** of the forecast core ``fc`` with ``net_exp_return``, ``score``, ``signal`` and
    ``confidence`` added to every horizon (the input is not modified, so cached forecasts can be re-annotated
    with different thresholds concurrently).  Horizons with missing/invalid numbers become ``hold`` / ``低``.
    """
    _check_thresholds(buy_p, avoid_p)
    cost = config.roundtrip_cost(cost_kind(kind), commission_discount, slippage_bps)
    out = copy.deepcopy(fc)
    for hz in _d(out.get("horizons")).values():
        if isinstance(hz, dict):
            hz.update(score_horizon(hz, out, cost, buy_p, avoid_p, True))
    return out


# ============================================================================= empirical Bayes
def james_stein(z, noise_var: float = 1.0):
    """Positive-part James-Stein shrinkage of ``z`` towards its mean (unit-noise scale) -> ``(shrunk, B)``.

    ``B = max(0, 1 - (k-3) noise_var / sum (z - zbar)^2)`` is the retained fraction; ``B = 1`` for k < 4.
    """
    z = np.asarray(z, dtype=float)
    k = len(z)
    if k == 0:
        return z.copy(), 1.0
    zbar = float(z.mean())
    ss = float(np.sum((z - zbar) ** 2))
    if k < 4:
        return z.copy(), 1.0
    B = 0.0 if ss <= 0 else max(0.0, 1.0 - (k - 3) * noise_var / ss)
    return zbar + B * (z - zbar), B


def shrink_scores(scores, h: int, bars: int = SHRINK_BARS):
    """James-Stein on ``z = score * sqrt(bars/h)``; returns ``(score_shrunk, B)`` in score units."""
    s = np.asarray(scores, dtype=float)
    scale = math.sqrt(bars / max(1, int(h)))
    zs, B = james_stein(s * scale)
    return zs / scale, B


# ============================================================================= covariance & weights
def ledoit_wolf_cov(X):
    """Ledoit-Wolf (2004) shrinkage covariance towards ``mu I`` (``mu = tr(S)/N``) -> ``(Sigma, intensity)``.

    ``X`` is (T, N) returns.  ``Sigma = a mu I + (1-a) S`` with ``a = min(b2, d2) / d2``, ``d2 = |S - mu I|_F^2 / N``,
    ``b2 = (1/(N T^2)) sum_t |x_t x_t' - S|_F^2`` (X is demeaned first).
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = X.T @ X / T
    mu = float(np.trace(S) / N)
    d2 = float(np.sum((S - mu * np.eye(N)) ** 2) / N)
    if d2 <= 0:
        return S.copy(), 1.0
    # sum_t |x_t x_t' - S|_F^2 = sum_t |x_t|^4 - 2 sum_t x_t' S x_t + T |S|_F^2
    r2 = np.sum(X * X, axis=1)
    b2bar = float((np.sum(r2 ** 2) - 2.0 * np.sum((X @ S) * X) + T * np.sum(S * S)) / (N * T * T))
    a = min(b2bar, d2) / d2
    return a * mu * np.eye(N) + (1.0 - a) * S, float(a)


def kelly_weights(edge, sigma, *, fraction: float = config.KELLY_FRACTION, cap: float = config.MAX_WEIGHT,
                  cov=None) -> np.ndarray:
    """Long-only fractional-Kelly weights capped at ``cap`` with ``sum <= 1``.

    ``edge`` = expected (net, shrunk) return of each name over the horizon, ``sigma`` its std.  Without ``cov``
    names are independent: ``w_i = fraction * max(edge_i, 0) / sigma_i^2``.  With a covariance matrix of the same
    horizon, ``w = fraction * Sigma^-1 edge`` on the active set (names with negative weight are dropped, iteratively).
    """
    edge = np.asarray(edge, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(edge)
    w = np.zeros(n)
    if n == 0:
        return w
    if cov is None:
        ok = (sigma > 0) & np.isfinite(edge) & np.isfinite(sigma)
        w[ok] = fraction * np.maximum(edge[ok], 0.0) / sigma[ok] ** 2
    else:
        cov = np.asarray(cov, dtype=float)
        active = np.flatnonzero(np.isfinite(edge) & (edge > 0))
        while len(active):
            sub = cov[np.ix_(active, active)]
            try:
                x = fraction * np.linalg.solve(sub, edge[active])
            except np.linalg.LinAlgError:
                x = fraction * np.linalg.lstsq(sub, edge[active], rcond=None)[0]
            if np.all(x >= 0):
                w[active] = x
                break
            active = active[x > 0]
    w = np.minimum(w, cap)
    tot = float(w.sum())
    if tot > 1.0:
        w = w / tot
    return w


# ============================================================================= explanation text
def _driver_phrase(fc: dict) -> str:
    """Strongest driver of the DLM mean as a short Chinese clause ('' if none)."""
    best, best_abs = None, -1.0
    drivers = fc.get("drivers")
    for d in (drivers if isinstance(drivers, list) else []):
        if not isinstance(d, dict):
            continue
        c, z = _f(d.get("contribution")), _f(d.get("z"))
        a = abs(c) if math.isfinite(c) else (abs(z) if math.isfinite(z) else -1.0)
        if a > best_abs and d.get("name"):
            best, best_abs = d, a
    if best is None:
        return ""
    c, z = _f(best.get("contribution")), _f(best.get("z"))
    verb = "推升" if (math.isfinite(c) and c > 0) else ("壓低" if math.isfinite(c) and c < 0 else "影響")
    ztxt = f"（z={z:+.1f}）" if math.isfinite(z) else ""
    return f"{best['name']}{ztxt}{verb}預期報酬，"


def make_reason(fc: dict, h: int, rec: dict, *, buy_p: float, avoid_p: float, min_avg_value: float) -> str:
    """One Chinese sentence quoting the concrete numbers; never promises a profit."""
    hl = horizon_label(h)
    p, net, conf = rec["p_up"], rec["net_exp_return"], rec["confidence"]
    drv = _driver_phrase(fc)
    reg = f"目前處於「{rec['regime']}」，" if rec.get("regime") else ""
    sig = rec["signal"]
    if sig == "buy":
        return (f"{drv}{reg}模型估計{hl}上漲機率 {p * 100:.0f}%、扣除成本後預期報酬 {net * 100:+.2f}%"
                f"（信心{conf}）。")
    if sig == "avoid":
        return f"{drv}{reg}模型估計{hl}下跌機率 {(1 - p) * 100:.0f}%、扣除成本後預期報酬 {net * 100:+.2f}%，不宜追買。"
    if not rec["liquid"] and p >= buy_p and net > 0:
        avg = rec.get("avg_value_20d")
        return (f"模型估計{hl}上漲機率 {p * 100:.0f}%，但近20日平均成交值約 {avg / 1e8:.1f} 億元，低於門檻 "
                f"{min_avg_value / 1e8:.1f} 億元，流動性不足，不列入偏多候選。")
    if p >= buy_p:
        return (f"{drv}{reg}模型估計{hl}上漲機率 {p * 100:.0f}%，但扣除成本後預期報酬 {net * 100:+.2f}% 沒有優勢，"
                f"建議觀望。")
    if p <= avoid_p:
        return f"{drv}{reg}模型估計{hl}上漲機率只有 {p * 100:.0f}%，但預期報酬未明顯為負，建議觀望。"
    return f"{drv}{reg}模型估計{hl}上漲機率 {p * 100:.0f}%，未達偏多候選門檻 {buy_p * 100:.0f}%，方向不明確，建議觀望。"


# ============================================================================= recommendation list
def _portfolio_cov(codes, returns, h: int, window: int = 250):
    """Ledoit-Wolf covariance (scaled to horizon h) of the given codes from ``returns`` or ``None``."""
    if not returns or any(c not in returns for c in codes) or len(codes) < 2:
        return None
    cols = [np.asarray(returns[c], dtype=float)[-window:] for c in codes]
    T = min(len(c) for c in cols)
    if T < 30:
        return None
    X = np.column_stack([c[-T:] for c in cols])
    if not np.all(np.isfinite(X)):
        return None
    return ledoit_wolf_cov(X)[0] * h


def _evidence_note(evidence: dict, h: int) -> str | None:
    """One Chinese sentence from the server-supplied ``evidence`` block (None when it carries no verdict)."""
    from .universe_bt import UNIVERSE_TITLES          # lazy: universe_bt imports this module
    lvl = evidence.get("verdict_level")
    if evidence.get("source") != "universe_validation" or lvl not in UNIVERSE_TITLES:
        return ("尚未取得「推薦整體歷史驗證」結果：排名與偏多候選只是模型的相對看法，尚未經過歷史驗證。")
    nums = []
    ic, hit, base = _f(evidence.get("ic")), _f(evidence.get("hit_rate")), _f(evidence.get("base_rate"))
    if math.isfinite(ic):
        nums.append(f"橫斷面 IC {ic:+.3f}")
    if math.isfinite(hit) and math.isfinite(base):
        nums.append(f"精選股上漲比例 {hit * 100:.1f}% 對全體 {base * 100:.1f}%")
    # optional additive keys (universe_bt.evidence_from_report): the validated basket's return AFTER costs next to the universe's,
    # so a significant ranking statistic that cannot pay the trading costs is never presented as a usable edge
    net, ew = _f(evidence.get("net_cagr")), _f(evidence.get("ew_cagr"))
    both = math.isfinite(net) and math.isfinite(ew)
    if both:
        nums.append(f"扣成本後精選組合年化 {net * 100:+.1f}% 對全體等權重（不扣成本）{ew * 100:+.1f}%")
    extra = f"（{'、'.join(nums)}）" if nums else ""
    losing = (both and net < ew) or evidence.get("beats_ew_after_costs") is False
    if lvl in ("no_edge", "weak_edge"):
        tail = "「偏多候選」只代表模型的相對看法，不是買進建議。"
    elif losing:
        tail = ("但驗證中的精選組合扣成本後沒有勝過全體等權重：統計上的排序訊號不足以支付交易成本，"
                "「偏多候選」仍不是買進建議。")
    else:
        tail = "驗證名單含存活者偏差，仍不保證獲利。"
    return f"推薦整體歷史驗證（{horizon_label(h)}）結論：{UNIVERSE_TITLES[lvl]}{extra}。{tail}"


def build_recommendations(forecasts: list, horizon: int, *, top_k: int = 10, buy_p: float = config.BUY_P_UP,
                          avoid_p: float = config.AVOID_P_UP,
                          commission_discount: float = config.COMMISSION_DISCOUNT,
                          slippage_bps: float = config.SLIPPAGE_BPS, market_forecast: dict | None = None,
                          min_avg_value: float = config.MIN_AVG_VALUE_TWD,
                          kelly_fraction: float = config.KELLY_FRACTION, max_weight: float = config.MAX_WEIGHT,
                          shrink_bars: int = SHRINK_BARS, returns: dict | None = None, now=None,
                          evidence: dict | None = None) -> dict:
    """Rank the universe for ``horizon`` and build the DESIGN.md 6.3 payload (+ the section 9 additions).

    ``forecasts``: forecast dicts (6.1 schema, server fields ``name``/``kind`` filled; optional ``avg_value_20d`` =
    mean traded value in TWD over 20 sessions, names without it are assumed liquid).  ``market_forecast``: forecast
    dict of ^TWII for the ``market`` block.  ``returns``: optional ``{code: recent daily returns}`` enabling
    the Ledoit-Wolf covariance in the weights.  ``evidence``: the server's ``evidence`` block (cached universe validation
    of the same horizon), copied into the payload as given -- ``{"source": "none"}`` when omitted.  Output is strictly
    JSON-serialisable.

    Ordering (``rank``, ``all``, ``leaders``, ``laggards``, ``lean``, ``score_rank_pct``) uses ``rank_signal`` only;
    ``score`` / ``score_shrunk`` only gate ``signal``, ``edge_covers_cost`` and the weights.  ``rank`` is one GLOBAL rank
    (position in ``all``) in every list.
    """
    _check_thresholds(buy_p, avoid_p)
    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1")
    h = int(horizon)
    top_k = int(top_k)
    universe = [fc for fc in (forecasts or []) if isinstance(fc, dict)]
    rows = []
    for fc in universe:
        hz = _hz(fc, h)
        if hz is None:
            continue
        p, er, sg = _f(hz.get("p_up")), _f(hz.get("exp_return")), _f(hz.get("sigma"))
        if not plausible(p, er, sg):
            continue
        kind = default_kind(fc)
        net = er - config.roundtrip_cost(cost_kind(kind), commission_discount, slippage_bps)
        avg = _f(fc.get("avg_value_20d"))
        p_naive = _f(_d(_d(hz.get("components")).get("naive")).get("p_up"))
        rows.append({"fc": fc, "hz": hz, "kind": kind, "p": p, "er": er, "sg": sg, "net": net, "score": net / sg,
                     "rs": rank_signal_value(hz.get("mu_log"), hz.get("sigma_log"), p),
                     "p_naive": p_naive if math.isfinite(p_naive) else None,
                     "avg": avg if math.isfinite(avg) else None,
                     "liquid": True if not math.isfinite(avg) else avg >= min_avg_value})
    n_scored = len(rows)
    shrunk, B = shrink_scores([r["score"] for r in rows], h, shrink_bars) if rows else (np.zeros(0), 1.0)
    for r, sh in zip(rows, shrunk):
        r["shrunk"] = float(sh)
        r["signal"] = classify_signal(r["p"], r["net"], buy_p, avoid_p, r["liquid"])
        r["conf"] = confidence_label(r["p"], r["hz"], r["fc"])
    # ---- relative ordering: rank_signal only (never the cost-adjusted score), ties by p_up then code
    order = sorted(rows, key=lambda r: (-r["rs"], -r["p"], str(r["fc"].get("symbol", ""))))
    grank = {id(r): i + 1 for i, r in enumerate(order)}
    pct = {id(r): ((n_scored - grank[id(r)]) / (n_scored - 1) if n_scored > 1 else 0.5) for r in rows}
    sg_med = float(np.median([r["sg"] for r in rows])) if rows else float("nan")
    buys_rows = [r for r in order if r["signal"] == "buy"][:top_k]
    avoid_rows = [r for r in reversed(order) if r["signal"] == "avoid"][:top_k]
    n_buy_total = sum(1 for r in rows if r["signal"] == "buy")
    leader_rows = [r for r in order if r["p"] > 0.5][:top_k]
    lag_rows = list(reversed(order))[:min(top_k, n_scored // 2)]
    rs_sigma = float("nan")
    if n_scored >= 4:
        rs_a, sg_a = np.array([r["rs"] for r in rows]), np.array([r["sg"] for r in rows])
        if np.ptp(rs_a) > 0 and np.ptp(sg_a) > 0:
            rs_sigma = float(stats.spearmanr(rs_a, sg_a).statistic)
    # ---- suggested weights: fractional Kelly of the shrunk net edge, only with a positive shrunk edge to allocate
    weights = {}
    suppressed = None
    if rows and B <= 0.0 and n_scored >= 4:
        suppressed = "B0"                         # every shrunk edge is the same common (market-drift) value: no view
    elif not any(r["net"] > 0 and r["shrunk"] * r["sg"] > 0 for r in buys_rows):
        suppressed = "no_positive_edge"
    if suppressed is None:
        cov = _portfolio_cov([str(r["fc"].get("symbol")) for r in buys_rows], returns, h)
        edge = np.array([r["shrunk"] * r["sg"] if r["net"] > 0 else 0.0 for r in buys_rows])
        sig = np.array([r["sg"] for r in buys_rows])
        w = kelly_weights(edge, sig, fraction=kelly_fraction, cap=max_weight, cov=cov)
        weights = {id(r): float(x) for r, x in zip(buys_rows, w)}
    else:
        cov = None
    total_w = float(sum(weights.values()))
    n_hold = sum(1 for x in weights.values() if x > 0)

    def rec(r):
        fc, hz = r["fc"], r["hz"]
        by_h = {str(k): M.finite_or_none(_d(v).get("p_up")) for k, v in _d(fc.get("horizons")).items()}
        pc = pct[id(r)]
        lean = ("bullish" if (pc >= LEAN_TOP_PCT and r["p"] > 0.5)
                else "bearish" if (pc <= LEAN_BOTTOM_PCT and r["p"] < 0.5) else "neutral")
        out = {"rank": grank[id(r)], "code": str(fc.get("symbol", "")),
               "name": fc.get("name") or str(fc.get("symbol", "")),
               "kind": r["kind"], "last_close": fc.get("last_close"), "chg_1d": fc.get("chg_1d"),
               "p_up": r["p"], "exp_return": r["er"], "net_exp_return": r["net"], "sigma": r["sg"],
               "score": r["score"], "score_shrunk": r["shrunk"], "signal": r["signal"], "confidence": r["conf"],
               "weight": weights.get(id(r), 0.0), "reason": "", "p_up_by_h": by_h,
               "regime": _d(fc.get("regime")).get("current"), "avg_value_20d": r["avg"], "liquid": r["liquid"],
               "rank_signal": r["rs"], "score_rank_pct": pc, "lean": lean, "edge_covers_cost": bool(r["net"] > 0),
               "low_vol": bool(r["sg"] < sg_med),
               "edge_vs_baseline": (r["p"] - r["p_naive"]) if r["p_naive"] is not None else None}
        out["reason"] = make_reason(fc, h, out, buy_p=buy_p, avoid_p=avoid_p, min_avg_value=min_avg_value)
        return out

    buys = [rec(r) for r in buys_rows]
    avoids = [rec(r) for r in avoid_rows]
    allr = [rec(r) for r in order]
    leaders = [rec(r) for r in leader_rows]
    laggards = [rec(r) for r in lag_rows]
    # ---- header fields ----
    asofs = [fc.get("asof") for fc in universe if fc.get("asof")]
    asof = max(Counter(asofs).items(), key=lambda kv: (kv[1], kv[0]))[0] if asofs else None
    asof_complete = all(bool(fc.get("asof_complete", True)) for fc in universe) if universe else True
    mk = {"p_up_1": None, "p_up_5": None, "regime": None, "last_close": None, "chg_1d": None}
    if isinstance(market_forecast, dict):
        for k, key in (("p_up_1", 1), ("p_up_5", 5)):
            hzm = _hz(market_forecast, key)
            mk[k] = M.finite_or_none(_d(hzm).get("p_up"))
        mk.update(regime=_d(market_forecast.get("regime")).get("current"),
                  last_close=market_forecast.get("last_close"), chg_1d=market_forecast.get("chg_1d"))
    now = now or datetime.now().astimezone()
    cash = max(0.0, 1.0 - total_w)
    if not rows:
        pnote = "沒有可評分的標的，因此不配置權重，建議全數持有現金。"
    elif suppressed == "B0":
        pnote = f"{NO_VIEW_NOTE}因此不配置權重，建議全數持有現金。"
    elif suppressed == "no_positive_edge":
        pnote = ("目前沒有同時符合偏多候選條件、且經收縮後仍具正期望報酬的標的。"
                 f"{NO_VIEW_NOTE}因此不配置權重，建議全數持有現金。")
    else:
        pnote = (f"依 {kelly_fraction:g} 倍凱利與單檔上限 {max_weight * 100:.0f}% 配置 {n_hold} 檔，合計 "
                 f"{total_w * 100:.0f}%，其餘 {cash * 100:.0f}% 為現金。")
    c_stock = config.roundtrip_cost("stock", commission_discount, slippage_bps)
    c_etf = config.roundtrip_cost("etf", commission_discount, slippage_bps)
    tilt = ""
    if math.isfinite(rs_sigma):
        tilt = (f"今日全體 Spearman(相對強度, 波動度) = {rs_sigma:+.2f}，"
                + ("相對強度高者多為低波動個股" if rs_sigma <= -0.2 else
                   "相對強度高者多為高波動個股" if rs_sigma >= 0.2 else "排序與波動度的關聯不大")
                + "；")
    notes = [
        f"排名依「相對強度」= 預期對數報酬 ÷ 預期對數波動度（不含交易成本與波動凸性項）；{tilt}"
        f"預測均值被強烈收縮，此排序只代表模型對各檔的相對看法，並非選股能力的證明，請同時參考每檔的波動度。",
        f"評分 = (預期報酬 − 來回交易成本) ÷ 波動度；股票來回成本約 {c_stock * 100:.2f}%、ETF 約 {c_etf * 100:.2f}%"
        f"（手續費 {commission_discount * 10:g} 折、滑價每邊 {slippage_bps:g} bp）。此評分只用於偏多候選／偏空把關、"
        f"是否覆蓋成本與建議權重，不用於排名。",
        f"經驗貝氏(James-Stein)收縮只用於把關與建議權重（排名不使用收縮分數）：共 {n_scored} 檔，保留比例 B={B:.2f}"
        f"（B=0 表示各檔評分的差異不比雜訊大，此時不配置權重；B>0 並不等於有選股訊號——短天期的評分差異主要來自"
        f"「來回成本 ÷ 波動度」這個每檔固定的數字，而不是預測）。",
        f"偏多候選條件：上漲機率 ≥ {buy_p * 100:.0f}% 且扣成本後預期報酬 > 0 且近20日平均成交值 ≥ "
        f"{min_avg_value / 1e8:.1f} 億元；偏空／避開條件：上漲機率 ≤ {avoid_p * 100:.0f}% 且扣成本後預期報酬 < 0。",
        f"建議權重 = {kelly_fraction:g} 倍凱利公式、單檔上限 {max_weight * 100:.0f}%、其餘為現金；"
        + ("本次沒有可配置的標的，全數為現金。" if suppressed is not None else
           "已用 Ledoit-Wolf 收縮共變異數考慮各檔相關性。" if cov is not None else "假設各檔彼此獨立，未考慮相關性。"),
        f"預測期間為 {h} 個交易日；機率與報酬為統計模型估計，並非保證，本清單不構成投資建議。",
    ]
    if suppressed is not None and rows:
        notes.append(f"{NO_VIEW_NOTE}不配置權重。")
    if lag_rows and all(r["p"] > 0.5 for r in lag_rows):
        notes.append("相對最弱的名單，上漲機率仍高於 50%：那是整體市場漂移為正，不代表看空。")
    if not buys:
        why = ""
        if rows and all(r["net"] <= 0 for r in rows):
            why = (f"原因：今日所有標的扣除來回交易成本後的{horizon_label(h)}預期報酬都 ≤ 0（最高 "
                   f"{max(r['net'] for r in rows) * 100:+.2f}%），成本相對於預期報酬太大，嚴格的偏多候選條件不會觸發；"
                   f"這不代表模型看空。")
        notes.append("目前沒有任何標的同時滿足偏多候選條件：沒有訊號，本身也是模型的答案，建議空手或觀望。" + why)
    elif n_buy_total >= max(5, 0.25 * n_scored):
        notes.append(f"今日有 {n_buy_total} 檔（佔評分名單 {n_buy_total / n_scored * 100:.0f}%）同時達到偏多候選條件：當這麼多標的"
                     f"同時達標，通常反映整體市場的上漲漂移為正，而不是個股被特別挑出；請同時參考相對強度排名、每檔的波動度與"
                     f"「推薦整體歷史驗證」，不要把它當成選股訊號。")
    ev = copy.deepcopy(evidence) if isinstance(evidence, dict) else {"source": "none"}
    ev_note = _evidence_note(ev, h)
    if ev_note and rows:
        notes.append(ev_note)
    if n_scored < len(universe):
        notes.append(f"另有 {len(universe) - n_scored} 檔因資料不足或預測無效，未列入評分。")
    if not asof_complete:
        notes.append("部分資料為盤中未收盤，預測僅供參考，收盤後會更新。")
    return M.json_safe({
        "asof": asof, "asof_complete": asof_complete, "horizon": h, "computed_at": now.isoformat(timespec="seconds"),
        "universe_size": len(universe), "n_scored": n_scored, "market": mk, "buys": buys, "avoids": avoids,
        "all": allr, "leaders": leaders, "laggards": laggards,
        "laggards_all_above_half": bool(lag_rows) and all(r["p"] > 0.5 for r in lag_rows),
        "n_buy_total": n_buy_total, "rank_key": "rank_signal",
        "rank_sigma_corr": rs_sigma if math.isfinite(rs_sigma) else None,
        "universe_sigma_median": sg_med if math.isfinite(sg_med) else None,
        "portfolio": {"cash_weight": cash, "n": n_hold, "note": pnote, "suppressed_reason": suppressed},
        "evidence": ev, "notes": notes})
