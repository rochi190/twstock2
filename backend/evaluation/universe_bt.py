"""Universe-level validation of the daily recommender (DESIGN.md 6.5).

Procedure (every step uses only information available at that day's close)
-------------------------------------------------------------------------
1. All stocks are aligned on the union of their trading dates.  Day ``t`` ranks the stocks that have a forecast on ``t`` by
   the SAME key as the picks page, ``rank_signal = mu_log / sigma_log`` of the horizon forecast (DESIGN.md 9.1; the log-return
   information ratio, computed from that day's ``mu, sigma`` only, falling back to ``Phi^-1(p_up)`` when either is missing;
   ties are broken by ``p_up`` and then by code order).  The cost-adjusted score ``(E[simple R_h] - cost) / sigma_h`` of
   ``recommend.py`` is NOT used for ranking: it ranks volatility (every net return is negative at h = 1, so ``net / sigma`` is
   largest for the most volatile names; Spearman(score, sigma) = 0.98 / 0.94 / 0.72 at h = 1 / 5 / 20), and a top-5 basket
   of it is a high-sigma, high-beta basket that earns the market premium times its beta.  The score only gates the optional
   ``only_buy_signals`` restriction.  ``rank_signal`` itself is low-volatility tilted (forecast means are shrunk towards a
   common drift), so the report states the tilt instead of hiding it (step 6).
2. **Overlapping tranches** (chosen over "rebalance every h days" because it gives one observation per day and no
   dependence on the rebalancing phase): each day the top ``top_k`` names get an equal-weight tranche holding 1/h
   of the capital for ``h`` trading days.  The target weight vector after the close of day t is therefore the
   average of the pick vectors of days ``t-h+1 .. t`` (during the first h-1 days the average runs over the tranches
   that exist, so the portfolio is fully invested from the first day).
3. The target weights are executed with the same simulator as the single-stock strategy
   (``strategy.simulate_weights``): ``next_open`` -> trade at the next open, ``close`` -> trade at the same close;
   buys pay commission + slippage, sells pay commission + slippage + tax; returns use adjusted OHLC.
4. Benchmarks: equal-weight universe (daily-rebalanced mean of close-to-close returns of the stocks that had a
   forecast the previous day, no costs) and, if supplied, TAIEX (price index, no dividends).
5. Statistics: cross-sectional Spearman IC per origin day (mean, IR, HAC t-statistic), hit-rate of the picks
   against the same-day universe up-rate (day-level paired test), top-minus-bottom spread of the mean h-bar LOG returns (log
   returns so that a volatility sort does not earn the Jensen term sigma^2 / 2 for free), STATS blocks (PSR/DSR,
   stationary-bootstrap Sharpe CI), per-stock direction accuracy / Brier skill (causal base-rate reference, oracle apart) /
   time-series IC.
6. Exposure disclosure (so that an apparent outperformance can be told from a volatility / beta tilt): the report's ``basket``
   block gives the mean forecast sigma and the mean causal market beta (trailing 250-day OLS slope on TAIEX, equal-weight
   universe when TAIEX is missing) of the top-k picks next to the universe's, the day-to-day overlap of the top-k, the mean
   daily Spearman(rank_signal, sigma) and the share of name-days whose expected net return covers the round-trip cost;
   ``benchmark_top_sigma`` is the model-free basket "top-k by forecast sigma" run through the same tranche / cost simulator and
   ``benchmark_equal_weight`` the universe.  A model-free sigma sort that matches the picks means the picks add nothing.

Inference on daily series (IC, hit-rate difference, spread): scores and returns are persistent and the h-bar targets
overlap, so a plain t-test on the daily values is badly anti-conservative (measured: 2.9 % rejections at the 1 % level in
no-skill panels).  We use the Newey-West HAC variance with Bartlett weights and lag ``L = min(max(2h, ceil(3 n^(1/3))),
n // 5)``, the small-sample factor ``n / (n - 1)`` and a Student-t reference with ``df = max(2, n / (L + 1) - 1)``
(L = 0 is exactly the textbook one-sample t-test).  In simulations this gives 1.3 % / 5 % rejections at the 1 % / 5 %
levels in no-skill panels.  Because even these HAC statistics have heavier far tails than the t reference, the
verdict uses the stricter thresholds ``report.THRESH_UNIVERSE`` (family-wise 0.3 %, per-test 0.3 % / 0.03 %): over 2400
no-skill panels (12 stocks, ~800 origin days, h = 1 and 5) 0.33 % of runs left ``no_edge`` (0.25 % reached
``moderate_edge``; 1.08 % with the single-stock thresholds), while weakly planted per-stock signals (IC ~ 0.05) were
found in 78 to 86 % of runs.  Check of the ``rank_signal`` ranking (audit S-05; 16 stocks, 900 days, h = 1 and 5, 40 panels per
cell, no skill: volatility 0.5 - 4 % a day, beta rising with volatility, a market premium of 0.04 or 0.11 % a day, a view that is
independent of everything): 0 of 160 panels left ``no_edge`` and the one-sided IC was significant at 5 % in 5 - 7.5 % of them,
whereas the old cost-adjusted-score key (a volatility ranking that collects beta x premium) gave 7.5 / 12.5 / 40 / 47.5 %.

Survivorship bias: the panel is *today's* liquid universe; stocks delisted, merged or illiquid today are absent
and the very choice of names uses hindsight, so every figure here is optimistic (see the report notes).
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from .. import config
from ..types import PriceData
from . import metrics as M
from . import recommend as RC
from . import report as RP
from . import strategy as S

UNIVERSE_TITLES = {
    "no_edge": "沒有統計上顯著的選股優勢",
    "weak_edge": "有微弱的選股訊號，但證據有限",
    "moderate_edge": "有一定的選股能力，但需持續檢驗",
    "strong_edge": "歷史樣本外呈現顯著的選股能力",
}
IC_SERIES_WINDOW = 60
BETA_WINDOW = 250        # trailing window (days) of the causal market-beta estimate of the exposure block
BETA_MIN_OBS = 60        # fewer observations -> no beta yet
# Picks whose mean forecast sigma is at most this multiple of the universe's: the hit-rate leg is tilt-contaminated (see HIT_TILT_NOTE)
TILT_RATIO = 0.8
HIT_TILT_NOTE = ("命中率檢驗是拿精選股的上漲比例與同日全體相比，沒有控制波動度：低波動股票上漲的機率本來就較高（上漲機率約為 Φ(平均報酬 ÷ 波動度)），"
                 "所以精選股偏向低波動時這一項會偏向顯著，單純挑最低波動的股票也會如此，不算獨立的選股證據。")


# ============================================================================= panel preparation
class _Panel:
    """Stocks aligned on a common calendar; attributes are (T, S) matrices unless noted."""

    def __init__(self, panel: dict, h: int, commission_discount: float, slippage_bps: float):
        self.codes, self.datas, self.hfs, self.firsts, self.events = [], [], [], [], []
        for code in sorted(panel):
            data, wf = panel[code]
            hf = RP.hf_for(wf, h)
            data = RP.align_data(data, wf)
            chk = RP.check_realized_alignment(data, hf)
            if not chk["ok"]:
                raise ValueError(f"alignment error in {code}: realised returns differ from the price data "
                                 f"(max |diff| = {chk['max_abs_diff']:.3g})")
            self.codes.append(str(code))
            self.datas.append(data)
            self.hfs.append(hf)
            self.firsts.append(int(wf.first_idx))
            self.events.append(RP.p_up_event(wf))        # which event p_up forecasts (report.py): P(R>0) or P(R>0 | R!=0)
        self.h = h
        self.alerts = {c: RP.data_alerts(d, f) for c, d, f in zip(self.codes, self.datas, self.firsts)}
        self.alerts = {c: a for c, a in self.alerts.items() if a}
        S_ = len(self.codes)
        start = min((d.dates[f] for d, f in zip(self.datas, self.firsts) if f < d.n), default=None)
        self.empty = start is None
        if self.empty:
            self.cal = np.zeros(0, dtype="datetime64[D]")
            self.T, self.S = 0, S_
            return
        alld = np.unique(np.concatenate([d.dates for d in self.datas]))
        self.cal = alld[alld >= start]
        self.T, self.S = len(self.cal), S_
        T = self.T
        self.CO, self.OC, self.CC = (np.zeros((T, S_)) for _ in range(3))
        self.SC, self.PU, self.MU, self.SG, self.R = (np.full((T, S_), np.nan) for _ in range(5))
        self.RS = np.full((T, S_), np.nan)         # rank_signal = mu_log / sigma_log (the ranking key, DESIGN.md 9.1)
        self.CL = np.full((T, S_), np.nan)         # causal climatology of P(R > 0) known at each origin (own history)
        self.has_bar = np.zeros((T, S_), dtype=bool)
        self.buy_cost, self.sell_cost = np.zeros(S_), np.zeros(S_)
        for j, (d, hf, first) in enumerate(zip(self.datas, self.hfs, self.firsts)):
            kind = S.cost_kind(d.kind)
            self.buy_cost[j], self.sell_cost[j] = S.side_costs(kind, commission_discount, slippage_bps)
            co, oc, cc = S.leg_returns(d)
            pos = np.searchsorted(self.cal, d.dates)
            keep = (pos < T) & (self.cal[np.minimum(pos, T - 1)] == d.dates)
            p_, r_ = pos[keep], np.flatnonzero(keep)
            self.CO[p_, j], self.OC[p_, j], self.CC[p_, j] = co[r_], oc[r_], cc[r_]
            self.has_bar[p_, j] = True
            fc_rows = r_[r_ >= first]
            pf = pos[fc_rows]
            p_up, mu, sg = (np.asarray(a, dtype=float)[fc_rows] for a in (hf.p_up, hf.mu, hf.sigma))
            ok = np.isfinite(p_up) & np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
            cost = config.roundtrip_cost(kind, commission_discount, slippage_bps)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                score = (S.expected_simple_return(mu, sg) - cost) / sg
            self.SC[pf[ok], j] = score[ok]
            self.PU[pf[ok], j] = np.clip(p_up[ok], 0.0, 1.0)
            self.MU[pf[ok], j], self.SG[pf[ok], j] = mu[ok], sg[ok]
            self.RS[pf[ok], j] = RC.rank_signal_array(mu, sg, p_up)[ok]
            Rfull = M.realized_log_returns(d.adjclose, h)
            self.R[pf[ok], j] = Rfull[fc_rows][ok]
            self.CL[pf[ok], j] = M.expanding_up_rate(
                Rfull, h, ties="drop" if self.events[j] == "up_given_move" else "not_up")[fc_rows][ok]
        self.avail = np.isfinite(self.SC)
        self.net = np.where(self.avail, self.SC * self.SG, np.nan)         # net expected return of each score
        with np.errstate(invalid="ignore"):
            self.Rs = np.expm1(self.R)                                      # realised simple h-bar returns

    # ------------------------------------------------------------------ picks / weights
    def _order(self, key: np.ndarray, elig: np.ndarray, descending: bool = True) -> np.ndarray:
        """Per-day name order by ``key`` (ineligible names last), ties by ``p_up`` then by column (code) order."""
        pu = np.nan_to_num(self.PU, nan=0.0)
        idx = np.broadcast_to(np.arange(self.S), (self.T, self.S))
        if descending:
            return np.lexsort((idx, -pu, -np.where(elig, key, -np.inf)), axis=1)
        return np.lexsort((idx, pu, np.where(elig, key, np.inf)), axis=1)

    def picks(self, top_k: int, only_buy: bool = False, buy_p: float = config.BUY_P_UP, key=None) -> np.ndarray:
        """(T, S) tranche matrix: 1/k for each of the day's top-k names (rows may be all zero).

        Names are ranked by ``key`` (default :attr:`RS`, the ``rank_signal`` of DESIGN.md 9.1; ``SG`` gives the model-free
        sigma benchmark).  ``only_buy`` restricts the candidates to the strict buy signal (``p_up >= buy_p`` and net
        expected return > 0, the only place where the cost-adjusted score still matters).
        """
        key = self.RS if key is None else key
        elig = self.avail & np.isfinite(key)
        if only_buy:
            with np.errstate(invalid="ignore"):
                elig &= (self.PU >= buy_p) & (self.net > 0)
        order = self._order(key, elig)
        kk = np.minimum(top_k, elig.sum(axis=1))
        P = np.zeros((self.T, self.S))
        for t in np.flatnonzero(kk > 0):
            P[t, order[t, :kk[t]]] = 1.0 / kk[t]
        return P

    def bottoms(self, k_cap: int) -> np.ndarray:
        """(T, S) bool mask of the lowest-ranked names (k = min(k_cap, n_avail // 2) per day)."""
        elig = self.avail & np.isfinite(self.RS)
        order = self._order(self.RS, elig, descending=False)
        kk = np.minimum(k_cap, elig.sum(axis=1) // 2)
        mask = np.zeros((self.T, self.S), dtype=bool)
        for t in np.flatnonzero(kk > 0):
            mask[t, order[t, :kk[t]]] = True
        return mask

    def tranche_weights(self, P: np.ndarray) -> tuple:
        """Average of the last h pick vectors (expanding at the start) -> ``(W, t0)``; ``t0`` = first day with picks."""
        has = P.sum(axis=1) > 0
        if not has.any():
            return np.zeros_like(P), None
        t0 = int(np.argmax(has))
        h = self.h
        C = np.vstack([np.zeros((1, self.S)), np.cumsum(P, axis=0)])
        t = np.arange(self.T)
        lo = np.maximum(t - h + 1, t0)
        m = np.where(t >= t0, t - lo + 1, 1)
        W = (C[t + 1] - C[lo]) / m[:, None]
        W[:t0] = 0.0
        return W, t0


# ============================================================================= statistics
def hac_lag(n: int, h: int) -> int:
    """Newey-West lag used for the daily-series tests: ``min(max(2h, ceil(3 n^(1/3))), n // 5)``."""
    if n <= 0:
        return 0
    return int(min(max(2 * int(h), math.ceil(3.0 * n ** (1.0 / 3.0))), max(0, n // 5)))


def _t_stats(x, h: int):
    """Mean, IR (mean/std), HAC t-statistic and its p-values for a daily series (None-safe).

    ``t = mean / sqrt(LRV n / (n-1) / n)`` with the Bartlett/Newey-West long-run variance at :func:`hac_lag`;
    p-values from Student-t with ``df = max(2, n / (L + 1) - 1)``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"n": n, "mean": None, "ir": None, "t": None, "p_two": None, "p_one": None, "lag": 0}
    if n == 0:
        return out
    out["mean"] = float(x.mean())
    if n < 3:
        return out
    sd = float(x.std(ddof=1))
    if sd < 1e-15:
        return out
    L = hac_lag(n, h)
    lrv = M.newey_west_lrv(x, L) * n / (n - 1.0)
    if not math.isfinite(lrv) or lrv <= 1e-30:
        return out
    t = float(x.mean() / math.sqrt(lrv / n))
    df = max(2.0, n / (L + 1.0) - 1.0)
    p_two = float(2.0 * stats.t.sf(abs(t), df))
    out.update(ir=float(x.mean() / sd), t=t, p_two=p_two, p_one=M.p_one_sided_positive(t, p_two), lag=int(L))
    return out


def ic_series(pn: _Panel, min_xs: int, key=None) -> np.ndarray:
    """Daily cross-sectional Spearman IC of the ranking key (default ``rank_signal``) vs the realised h-bar return."""
    key = pn.RS if key is None else key
    ics = np.full(pn.T, np.nan)
    ok = pn.avail & np.isfinite(pn.R) & np.isfinite(key)
    for t in np.flatnonzero(ok.sum(axis=1) >= min_xs):
        m = ok[t]
        ics[t] = M.spearman_ic(key[t, m], pn.R[t, m], 1)[0]
    return ics


def picks_stats(pn: _Panel, P: np.ndarray, h: int) -> dict:
    """Hit-rate of the picks vs the same-day universe up-rate (day-level paired one-sided HAC test)."""
    picked = P > 0
    valid = picked & np.isfinite(pn.R) & (pn.R != 0)
    nk = valid.sum(axis=1)
    uv = pn.avail & np.isfinite(pn.R) & (pn.R != 0)
    nu = uv.sum(axis=1)
    days = (nk > 0) & (nu > 0)
    out = {"hit_rate": None, "base_rate": None, "n": int(nk[days].sum()), "p_value": None, "mean_return": None}
    fin = picked & np.isfinite(pn.R)
    if fin.any():
        out["mean_return"] = float(pn.Rs[fin].mean())
    if not days.any():
        return out
    hit_d = ((pn.R > 0) & valid).sum(axis=1)[days] / nk[days]
    up_d = ((pn.R > 0) & uv).sum(axis=1)[days] / nu[days]
    w = nk[days]
    out["hit_rate"] = float((hit_d * w).sum() / w.sum())
    out["base_rate"] = float((up_d * w).sum() / w.sum())
    ts = _t_stats(hit_d - up_d, h)
    out["p_value"] = ts["p_one"]
    return out


def spread_stats(pn: _Panel, P: np.ndarray, top_k: int, min_xs: int, h: int) -> dict:
    """Mean top-minus-bottom realised h-bar LOG return per origin day (bottom = lowest ranks) and its t-statistic.

    Log returns, not simple ones: the mean of simple returns contains ``sigma^2 / 2``, so any ranking that sorts on volatility
    would earn a spread (Jensen term) even at equal log drift (audit S-05).
    """
    bot = pn.bottoms(top_k)
    top = P > 0
    fin = np.isfinite(pn.R)
    ok = ((pn.avail.sum(axis=1) >= min_xs) & (top.sum(axis=1) > 0) & (bot.sum(axis=1) > 0)
          & ((top & ~fin).sum(axis=1) == 0) & ((bot & ~fin).sum(axis=1) == 0))
    R0 = np.where(fin, pn.R, 0.0)
    top_m = (R0 * P).sum(axis=1)
    bot_m = (R0 * bot).sum(axis=1) / np.maximum(bot.sum(axis=1), 1)
    d = np.where(ok, top_m - bot_m, np.nan)
    ts = _t_stats(d, h)
    return {"mean": ts["mean"], "t": ts["t"], "p_one": ts["p_one"], "n": ts["n"]}


def per_stock_table(pn: _Panel) -> list:
    """Per-stock table: direction accuracy (non-flat days), Brier skill, time-series IC.

    ``brier_skill`` scores ``p_up`` as the event the stock's forecaster declares (``report.p_up_event``): ``P(R > 0 | R != 0)``
    on the moving days only, or ``P(R > 0)`` on every evaluated forecast (flat = not up), against the CAUSAL expanding base
    rate of the stock for the same event (DESIGN.md 9(a)); ``brier_skill_oracle`` uses the stock's test-sample up-rate on the
    same rows (hindsight, harder, reported apart).
    """
    rows = []
    for j, code in enumerate(pn.codes):
        ev = pn.avail[:, j] & np.isfinite(pn.R[:, j])
        sc = ev & (pn.R[:, j] != 0)
        n = int(sc.sum())
        rec = {"code": code, "name": pn.datas[j].name, "accuracy": None, "brier_skill": None, "ic": None, "n": n,
               "brier_skill_oracle": None}
        if n:
            p, up = pn.PU[sc, j], pn.R[sc, j] > 0
            rec["accuracy"] = float(((p >= 0.5) == up).mean())
        pr_rows = sc if pn.events[j] == "up_given_move" else ev          # rows on which p_up is scored
        if pr_rows.any():
            p, y = pn.PU[pr_rows, j], (pn.R[pr_rows, j] > 0).astype(float)
            b, b_causal = M.brier_score(p, y), M.brier_score(pn.CL[pr_rows, j], y)
            if b_causal > 0:
                rec["brier_skill"] = 1.0 - b / b_causal
            u = float(y.mean())
            if u * (1 - u) > 0:
                rec["brier_skill_oracle"] = 1.0 - b / (u * (1 - u))
        if ev.sum() >= 10:
            rec["ic"] = M.finite_or_none(M.spearman_ic(pn.MU[ev, j], pn.R[ev, j], pn.h)[0])
        rows.append(rec)
    return rows


def equal_weight_returns(pn: _Panel, t0: int) -> np.ndarray:
    """Daily-rebalanced equal-weight close-to-close return of the stocks that had a forecast the previous close."""
    d = np.arange(t0 + 1, pn.T)
    elig = pn.avail[d - 1] & pn.has_bar[d]
    cnt = elig.sum(axis=1)
    tot = (pn.CC[d] * elig).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)


def index_cc(taiex: PriceData, cal: np.ndarray) -> np.ndarray:
    """Close-to-close returns of an index/price series on the whole master calendar (0 on dates it did not trade)."""
    cc = np.zeros(len(cal))
    c = np.asarray(taiex.adjclose, dtype=float)
    if taiex.n >= 2:
        r = np.zeros(taiex.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = c[1:] / c[:-1] - 1.0
        r[~np.isfinite(r)] = 0.0
        pos = np.searchsorted(cal, taiex.dates)
        keep = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == taiex.dates)
        cc[pos[keep]] = r[keep]
    return cc


def index_returns(taiex: PriceData, cal: np.ndarray, t0: int) -> np.ndarray:
    """:func:`index_cc` restricted to the days after ``t0`` (the strategy's return days)."""
    return index_cc(taiex, cal)[t0 + 1:]


def causal_betas(x: np.ndarray, valid: np.ndarray, m: np.ndarray, window: int = BETA_WINDOW,
                 min_obs: int = BETA_MIN_OBS) -> np.ndarray:
    """(T, S) trailing-window OLS beta of the daily returns ``x`` (T, S) on the market return ``m`` (T,).

    Row t uses only days ``t - window + 1 .. t`` (returns known at the close of t) and only days where ``valid`` is true;
    NaN with fewer than ``min_obs`` observations or a flat market.  Computed from cumulative sums.
    """
    T, S_ = x.shape
    v = valid.astype(float)
    xv = np.where(valid, x, 0.0)
    mm = m[:, None] * v

    def cs(a):
        return np.vstack([np.zeros((1, S_)), np.cumsum(a, axis=0)])

    t = np.arange(T)
    lo = np.maximum(t + 1 - int(window), 0)

    def win(c):
        return c[t + 1] - c[lo]

    N, SX, SY, SXY, SYY = (win(cs(a)) for a in (v, xv, mm, xv * mm, mm * mm))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov, var = SXY - SX * SY / N, SYY - SY * SY / N
        return np.where((N >= min_obs) & (var > 1e-12), cov / var, np.nan)


def _pooled_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    """Weighted mean over the entries where both are finite and the weight is positive (None when empty)."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else None


def topk_overlap(P: np.ndarray) -> float | None:
    """Mean share of a day's picks that were also picked the day before (1 = the list never changes)."""
    pk = P > 0
    n_prev, n_cur = pk[:-1].sum(axis=1), pk[1:].sum(axis=1)
    both = (pk[:-1] & pk[1:]).sum(axis=1)
    ok = (n_prev > 0) & (n_cur > 0)
    return float(np.mean(both[ok] / n_cur[ok])) if ok.any() else None


def exposure_stats(pn: _Panel, P: np.ndarray, beta: np.ndarray | None, t0: int) -> dict:
    """Mean forecast sigma and mean causal market beta of the picks ``P`` at selection time (pick-weighted)."""
    out = {"mean_sigma": _pooled_mean(pn.SG, P), "mean_beta": None}
    if beta is not None:
        out["mean_beta"] = _pooled_mean(beta, P)
    return out


def universe_exposure(pn: _Panel, beta: np.ndarray | None, t0: int) -> dict:
    """Mean forecast sigma / causal beta over every available name-day from the first pick day on."""
    live = pn.avail.copy()
    live[:t0] = False
    out = {"mean_sigma": _pooled_mean(pn.SG, live.astype(float)), "mean_beta": None}
    if beta is not None:
        out["mean_beta"] = _pooled_mean(beta, live.astype(float))
    return out


def rank_sigma_corr(pn: _Panel, min_xs: int, t0: int) -> float | None:
    """Mean daily cross-sectional Spearman(rank_signal, forecast sigma) over the evaluation days."""
    ok = pn.avail & np.isfinite(pn.RS) & np.isfinite(pn.SG)
    vals = []
    for t in np.flatnonzero(ok.sum(axis=1) >= min_xs):
        if t >= t0:
            r = M.spearman_ic(pn.RS[t, ok[t]], pn.SG[t, ok[t]], 1)[0]
            if math.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else None


# ============================================================================= verdict & notes
def _exposure_bullets(rep: dict, *, h: int, top_k: int) -> list:
    """Bullets that put the picks' volatility / beta exposure and the model-free sigma basket next to the result."""
    bk, hl = rep.get("basket"), RP.horizon_label(h)
    if not bk:
        return []
    top, sg_b = rep["top"], rep.get("benchmark_top_sigma")
    out = []
    ratio = bk.get("sigma_ratio")
    beta_txt = ""
    if bk.get("mean_beta") is not None and bk.get("universe_mean_beta") is not None:
        beta_txt = (f"、平均市場 β {RP.num(bk['mean_beta'])} 對全體 {RP.num(bk['universe_mean_beta'])}"
                    f"（相對{'加權指數' if bk.get('beta_market') == 'taiex' else '全體等權重'}，滾動 {BETA_WINDOW} 日估計）")
    corr = bk.get("corr_rank_sigma")
    tilt = ""
    if corr is not None:
        tilt = (f"；相對強度與預測波動度的平均 Spearman 相關為 {RP.num(corr, 2, True)}，"
                + ("排序偏向低波動股票" if corr <= -0.2 else "排序偏向高波動股票" if corr >= 0.2 else "排序與波動度關聯不大"))
    out.append(f"風險暴露：精選股的平均預測{hl}波動度 {RP.pct(bk.get('mean_sigma'))}，全體為 {RP.pct(bk.get('universe_mean_sigma'))}"
               + (f"（{ratio:.2f} 倍）" if ratio is not None else "") + beta_txt + tilt
               + f"；前 {top_k} 名名單的日對日重疊率 {RP.pct(bk.get('topk_overlap'), 0)}。"
               + ("警示：精選股平均波動度／β 明顯高於全體，報酬可能只是高波動、高 β 的曝險，而不是選股能力。"
                  if (ratio is not None and ratio >= 1.25) or (bk.get("mean_beta") is not None
                                                                and bk.get("universe_mean_beta")
                                                                and bk["mean_beta"] >= 1.25 * bk["universe_mean_beta"])
                  else "精選股與全體的波動度／β 差異愈大，報酬差異愈可能只是曝險不同，而不是選股能力。"))
    if sg_b:
        better = (sg_b.get("sharpe") is not None and top.get("sharpe") is not None and sg_b["sharpe"] >= top["sharpe"])
        out.append(f"無預測對照組（每天只挑預測波動度最高的前 {top_k} 名，同樣的成本與持有方式）：總報酬 {RP.pct(sg_b['total_return'])}、"
                   f"年化 {RP.pct(sg_b['cagr'])}、夏普 {RP.num(sg_b['sharpe'])}；模型精選股：總報酬 {RP.pct(top['total_return'])}、"
                   f"年化 {RP.pct(top['cagr'])}、夏普 {RP.num(top['sharpe'])}。"
                   + ("模型精選股沒有勝過這個不使用任何預測的波動度排序，報酬不能歸功於預測。" if better else
                      "模型精選股的夏普高於此對照組，但仍須與上方風險暴露一併判讀。"))
    share = bk.get("share_net_positive")
    if share is not None and share < 0.02:
        out.append(f"提醒：{hl}預測中，只有 {RP.pct(share, 1)} 的「名次—日期」預期報酬能覆蓋來回交易成本，"
                   f"嚴格的偏多候選條件（扣成本後預期報酬 > 0）幾乎不會觸發。")
    if ratio is not None and ratio <= TILT_RATIO:
        out.append(f"提醒：精選股的平均波動度只有全體的 {ratio:.2f} 倍。{HIT_TILT_NOTE}")
    return out


def make_universe_verdict(rep: dict, ass: dict, ev: dict, *, h: int, top_k: int, exec_mode: str) -> dict:
    """Plain-Chinese verdict for the universe backtest quoting the concrete numbers."""
    ic, pk, sp = rep["ic"], rep["picks"], rep["spread"]
    top, ew, tx = rep["top"], rep["benchmark_equal_weight"], rep["benchmark_taiex"]
    per = rep["period"]
    hl = RP.horizon_label(h)
    level = ass["level"]
    if per["n_days"] == 0 or ic["mean"] is None:
        text = "可用的樣本外資料不足，無法評估每日推薦的整體表現，因此不宣稱有選股優勢。" + RP.DISCLAIM
        return {"level": "no_edge", "title": UNIVERSE_TITLES["no_edge"], "text": text,
                "bullets": ["資料不足：至少需要數十檔股票、數百個交易日的前推式預測。"]}
    head = (f"在 {per['start']} 至 {per['end']}、{per['n_stocks']} 檔股票共 {per['n_days']:,} 個交易日中，每天挑選{hl}"
            f"相對強度（預期對數報酬 ÷ 預期對數波動度，與推薦頁排名相同）最高的前 {top_k} 名（等權重、重疊分批持有）："
            f"橫斷面預測排序相關(IC)平均 {RP.num(ic['mean'], 3, True)}"
            f"（IR {RP.num(ic['ir'])}、t={RP.num(ic['t'])}、雙尾 {RP.pval(ic['pvalue'])}）；精選股上漲比例 "
            f"{RP.pct(pk['hit_rate'])}，同日全體為 {RP.pct(pk['base_rate'])}（{RP.pval(pk['p_value'])}）；"
            f"前段與後段的對數報酬差平均 {RP.pct(sp['mean'], 2, True)}（t={RP.num(sp['t'])}）。")
    names = {"ic": "橫斷面IC", "picks": "精選股命中率", "spread": "多空報酬差"}
    loss = (f"但扣除交易成本後策略反而虧損（年化 {RP.pct(top['cagr'])}，總報酬 {RP.pct(top['total_return'])}），訊號太弱，"
            f"不足以支付每日換股的成本，無法據此獲利") if (top["cagr"] is not None and top["cagr"] < 0) else ""
    thr = RP.THRESH_UNIVERSE
    cl = f"{thr['confirm'] * 100:g}%"
    sig = [f"{names[k]} {RP.pval(v)}" for k, v in sorted(ev["p"].items(), key=lambda kv: kv[1]) if v < thr["confirm"]]
    ratio = (rep.get("basket") or {}).get("sigma_ratio")
    tilt = (f"其中精選股命中率的比較基準沒有控制波動度，而精選股的平均波動度只有全體的 {ratio:.2f} 倍，這一項會偏向顯著，"
            f"不算獨立證據。" if (ev["p"].get("picks", 1.0) < thr["confirm"] and ratio is not None and ratio <= TILT_RATIO) else "")
    if level == "no_edge":
        if ass["reason"] == "small_sample":
            tail = f"有效樣本只有約 {ev['n_eff']} 筆獨立觀察，樣本太少，無法得出可靠結論，因此不宣稱每日推薦有選股優勢。"
        elif ass["reason"] == "no_tests":
            tail = "可用於檢驗的資料不足，因此不宣稱每日推薦有選股優勢。"
        elif ass["reason"] == "not_consistent":
            lone = [f"{names[k]} {RP.pval(v)}" for k, v in ev["p"].items() if v < thr["agree"]]
            tail = f"只有 1 項檢驗（{'、'.join(lone)}）達到顯著，其餘檢驗沒有同步支持，較可能是偶然，因此不宣稱每日推薦有選股優勢。"
        else:
            tail = (f"IC、命中率與多空報酬差經多重檢定校正後（最小調整後 {RP.pval(ass['p_fw'])}）都未達顯著標準，"
                    f"因此不能說每日推薦的排序有預測優勢。")
        text = head + tail + RP.DISCLAIM
    elif level == "weak_edge":
        text = (head + f"{len(sig)} 項檢驗達到 {cl} 顯著（{'、'.join(sig) or '—'}），校正後最小調整後 {RP.pval(ass['p_fw'])}。{tilt}"
                f"排序可能帶有微弱訊號，但效果小、證據有限，且受存活者偏差影響，實際表現可能更差。"
                + (f"此外，{loss}。" if loss else "") + RP.DISCLAIM)
    elif level == "moderate_edge":
        trade = ("且扣成本後的夏普明顯高於等權重全體" if ass["tradable"] else loss or "但扣除交易成本後，尚未能穩定勝過等權重持有全體")
        text = (head + f"{len(sig)} 項檢驗同時達到 {cl} 顯著（{'、'.join(sig)}），{trade}。{tilt}此結果含存活者偏差，"
                f"仍需持續追蹤。" + RP.DISCLAIM)
    else:
        text = (head + f"{len(sig)} 項檢驗達到 {cl} 顯著（{'、'.join(sig)}），且扣成本後夏普顯著高於等權重全體"
                f"（PSR {RP.pct(ev.get('psr_vs_bh'), 1)}）。但驗證名單為今日仍存活的流動性佳股票，存在存活者偏差，"
                f"實際結果仍可能與回測不同，並不保證獲利。" + RP.DISCLAIM)
    b = [f"排序能力：橫斷面 IC 平均 {RP.num(ic['mean'], 3, True)}、IR {RP.num(ic['ir'])}（t={RP.num(ic['t'])}，"
         f"雙尾 {RP.pval(ic['pvalue'])}）；IC 為正代表相對強度高的股票平均報酬較高。",
         f"命中率：共 {pk['n']:,} 次精選，上漲 {RP.pct(pk['hit_rate'])} 對同日全體 {RP.pct(pk['base_rate'])}"
         f"（單尾 {RP.pval(pk['p_value'])}），平均{hl}報酬 {RP.pct(pk['mean_return'], 2, True)}。",
         f"多空報酬差：前 {top_k} 名減後段的平均{hl}對數報酬差 {RP.pct(sp['mean'], 2, True)}（t={RP.num(sp['t'])}；"
         f"採對數報酬，避免波動度造成的 Jensen 項讓高波動排序白賺報酬差）。",
         f"策略（已扣成本，{RP.exec_label(exec_mode)}）：總報酬 {RP.pct(top['total_return'])}、年化 {RP.pct(top['cagr'])}、"
         f"夏普 {RP.num(top['sharpe'])}、最大回撤 {RP.pct(top['max_dd'])}、換手率 {RP.num(top['turnover_per_year'], 1)} 倍/年；"
         f"等權重全體：總報酬 {RP.pct(ew['total_return'])}、夏普 {RP.num(ew['sharpe'])}、最大回撤 {RP.pct(ew['max_dd'])}。"]
    if tx is not None:
        b.append(f"加權指數（價格指數，不含股利）：總報酬 {RP.pct(tx['total_return'])}、夏普 {RP.num(tx['sharpe'])}、"
                 f"最大回撤 {RP.pct(tx['max_dd'])}。")
    b.extend(_exposure_bullets(rep, h=h, top_k=top_k))
    if ic["mean"] < 0 and ic["pvalue"] is not None and ic["pvalue"] < 0.01:
        b.append(f"警示：橫斷面 IC 顯著為負（平均 {RP.num(ic['mean'], 3, True)}、雙尾 {RP.pval(ic['pvalue'])}）：排序方向與實際報酬"
                 f"相反；上列檢定皆為單尾「優於」，不會把這種情況標示出來，請不要依此排序操作。")
    if sp["t"] is not None and sp["t"] < -2.58:
        b.append(f"警示：前段減後段的報酬差顯著為負（t={RP.num(sp['t'])}），排名較高的股票平均報酬反而較低。")
    if (ev["p"].get("ic", 1.0) < RP.ALPHA_AGREE and top["cagr"] is not None and ew["cagr"] is not None
            and top["cagr"] < ew["cagr"]):
        b.append(f"提醒：排序相關為正，但扣除交易成本後精選股的年化報酬（{RP.pct(top['cagr'])}）低於等權重全體"
                 f"（{RP.pct(ew['cagr'])}），訊號不足以支付每日換股的成本。")
    if ev.get("psr_vs_bh") is not None:
        b.append(f"策略夏普高於等權重全體的機率(PSR) {RP.pct(ev['psr_vs_bh'], 1)}，自助法單尾 {RP.pval(ev.get('diff_p'))}。")
    b.append(f"t 值與 p 值採 Newey-West 異質變異數自相關穩健(HAC)標準誤（落後 {rep.get('_lag', 0)} 期）並以 t 分配判斷，"
             f"已考慮分數與報酬的持續性{'及預測期間重疊' if h > 1 else ''}，比一般 t 檢定保守。")
    b.append("存活者偏差：名單只含今日仍存活且流動性佳的股票，歷史結果偏樂觀。")
    return {"level": level, "title": UNIVERSE_TITLES[level], "text": text, "bullets": b}


def _flat_note(events) -> str:
    """How the per-stock Brier skill treats unchanged closes (depends on the event each stock's forecaster declares)."""
    kinds = set(events or ())
    if kinds == {"up_given_move"}:
        return "平盤（報酬恰為 0）既不是漲也不是跌，不計入機率評分，因為模型預測的是「價格有變動時上漲的機率」。"
    if kinds <= {"up"}:
        return "平盤視為「未上漲」，與模型的預測目標 P(報酬>0) 一致。"
    return "平盤的處理依各檔預測所宣告的事件而定（P(報酬>0｜報酬≠0) 不計入平盤；P(報酬>0) 視平盤為「未上漲」）。"


def make_universe_notes(*, h: int, top_k: int, exec_mode: str, n_stocks: int, commission_discount: float,
                        slippage_bps: float, only_buy: bool, ic_window: int, taiex_note: str | None,
                        alerts: dict | None = None, events=None) -> list:
    notes = [
        f"存活者偏差：驗證對象是「今天」流動性最佳的 {n_stocks} 檔股票；已下市、合併或今日已不在名單內的股票沒有納入，"
        f"名單本身也用到了事後資訊（今日仍存活且成交熱絡的公司，過去平均表現較好），因此歷史報酬與準確率很可能被高估。",
        f"選股方式：每日依「相對強度」（預期對數報酬 ÷ 預期對數波動度，與今日推薦頁的排名相同，不含交易成本與波動凸性項）排序"
        + ("，只在符合偏多候選條件（上漲機率達門檻且扣成本後預期報酬 > 0）的股票中" if only_buy
           else "（全部股票都參與排名，不論是否符合偏多候選條件）")
        + f"取前 {top_k} 名等權重；採 {h} 日重疊分批持有，每天更換 1/{h} 的部位。",
        f"已計入手續費（牌告 {config.COMMISSION_RATE * 100:.4f}% × {commission_discount * 10:g} 折）、證交稅"
        f"（股票 {config.TAX_STOCK * 100:.1f}%／ETF {config.TAX_ETF * 100:.1f}%）與滑價（每邊 {slippage_bps:g} bp）；"
        f"執行假設：{RP.exec_label(exec_mode)}；未計入漲跌停鎖死、最低手續費與市場衝擊。",
        "等權重基準為每日再平衡、不扣成本的全體平均；已將當日有預測的股票視為可投資範圍。",
        f"對照組：(a) 全體等權重；(b) 每天只依預測波動度挑前 {top_k} 名（不使用任何預測，同樣的成本與持有方式）。"
        f"精選股的平均波動度、平均市場 β 與全體並列於「風險暴露」：相對強度排序偏向低波動股票，若精選股與全體的曝險差異大，"
        f"報酬差異可能只是曝險不同，不一定是選股能力。",
        f"IC、命中率與報酬差的 t 值／p 值採 Newey-West HAC 標準誤（落後 max(2×{h}, 3×天數^(1/3)) 期）與 t 分配，"
        f"以處理分數與報酬的持續性{'及預測期間重疊' if h > 1 else ''}，結論較保守。",
        "未對多組參數（持股數、持有期間）做多重比較校正；反覆調整參數挑出最佳組合會高估績效。",
        "每檔股票資訊（準確率、Brier 技能、IC）取自其自身前推預測；Brier 技能的基準為「因果氣候基準」（每個預測日只用當時已知的"
        f"上漲比例），含事後資訊的全期上漲比例另列為事後基準；{_flat_note(events)}",
    ]
    if ic_window > 1:
        notes.append(f"IC 序列圖為 {ic_window} 日滾動平均；平均值、IR 與 t 值以每日 IC 計算。")
    for code, al in list((alerts or {}).items())[:3]:
        notes.insert(0, RP.data_alert_text(al, f"{code} "))
    if alerts and len(alerts) > 3:
        notes.insert(3, f"另有 {len(alerts) - 3} 檔股票也有類似的資料品質警示。")
    if taiex_note:
        notes.append(taiex_note)
    return notes


# ============================================================================= public API
LABELS = {"top": "模型精選：每日相對強度最高的前 k 名", "benchmark_equal_weight": "全體等權重（不扣成本）",
          "benchmark_top_sigma": "無預測對照：每日預測波動度最高的前 k 名",
          "benchmark_taiex": "加權指數（價格指數，不含股利）"}


def _empty_report(h: int, top_k: int, n_stocks: int, reason: str, notes: list | None = None) -> dict:
    ev = {"n_scored": 0, "n_eff": 0, "p": {}, "psr_vs_bh": None, "diff_p": None}
    rep = {"horizon": int(h), "top_k": int(top_k),
           "period": {"start": None, "end": None, "n_days": 0, "n_stocks": int(n_stocks)},
           "ic": {"mean": None, "ir": None, "t": None, "pvalue": None, "series": {"dates": [], "values": []}},
           "picks": {"hit_rate": None, "base_rate": None, "n": 0, "p_value": None, "mean_return": None},
           "top": S.empty_stats(), "benchmark_equal_weight": S.empty_stats(), "benchmark_taiex": None,
           "benchmark_top_sigma": None, "basket": None, "exposure": None, "rank_key": "rank_signal",
           "only_buy_signals": False, "labels": _labels(top_k),
           "spread": {"mean": None, "t": None},
           "curve": {"dates": [], "top": [], "equal_weight": [], "taiex": None, "top_sigma": None},
           "per_stock": [], "notes": notes or []}
    rep["verdict"] = make_universe_verdict(rep, {"level": "no_edge", "reason": "no_tests", "p_fw": None,
                                                 "tradable": False}, ev, h=h, top_k=top_k, exec_mode="next_open")
    rep["verdict"]["bullets"].insert(0, f"原因：{reason}")
    rep["notes"] = (notes or []) + [f"存活者偏差：驗證對象為今日仍存活的流動性佳股票；本次因「{reason}」而未能產生統計。"]
    return M.json_safe(rep)


def _labels(top_k: int) -> dict:
    return {k: v.replace("前 k 名", f"前 {top_k} 名") for k, v in LABELS.items()}


def evaluate_universe(panel: dict, horizon: int, *, top_k: int = 5,
                      commission_discount: float = config.COMMISSION_DISCOUNT,
                      slippage_bps: float = config.SLIPPAGE_BPS, exec_mode: str = "next_open",
                      taiex: PriceData | None = None, only_buy_signals: bool = False,
                      buy_p: float = config.BUY_P_UP, n_boot: int = 1000) -> dict:
    """Walk-forward validation of the daily top-k recommender over a panel ``{code: (PriceData, WalkForward)}``.

    Returns the DESIGN.md 6.5 ``report`` object (strictly JSON-serialisable).  The picks are ranked by ``rank_signal`` (the
    key of the picks page, DESIGN.md 9.1).  ``taiex`` (optional PriceData of ^TWII) fills ``benchmark_taiex`` /
    ``curve.taiex`` (else ``null``) and is the market of the beta estimate.  ``only_buy_signals=True`` restricts the picks
    to names that also satisfy the buy rule (p_up >= buy_p and net expected return > 0); the default ranks every name, which
    tests ranking skill with far more observations.  Raises ``ValueError`` on misaligned data.
    """
    if exec_mode not in S.EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {S.EXEC_MODES}, got {exec_mode!r}")
    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1")
    h, top_k = int(horizon), int(top_k)
    pn = _Panel(panel or {}, h, commission_discount, slippage_bps) if panel else None
    if pn is None or pn.S == 0 or pn.empty:
        return _empty_report(h, top_k, 0 if pn is None else pn.S, "沒有可用的股票或前推式預測")
    min_xs = min(max(6, 2 * top_k), pn.S)
    P = pn.picks(top_k, only_buy_signals, buy_p)
    W, t0 = pn.tranche_weights(P)
    if t0 is None or pn.T - 1 - t0 < 2:
        return _empty_report(h, top_k, pn.S, "回測期間太短")
    sim = S.simulate_weights(pn.CO, pn.OC, W, pn.buy_cost, pn.sell_cost, exec_mode=exec_mode, first=t0)
    ret, N = sim["ret"], len(sim["ret"])
    ew = equal_weight_returns(pn, t0)
    boot = M.bootstrap_sharpe(ret, ew, n_boot=n_boot)
    # ---- picks as "trades": net h-bar return after the round-trip cost of the stock's kind
    rt = np.array([config.roundtrip_cost(S.cost_kind(d.kind), commission_discount, slippage_bps) for d in pn.datas])
    picked = (P > 0) & np.isfinite(pn.R)
    pick_net = (pn.Rs - rt[None, :])[picked]
    top = S.stats_block(ret, sim["exposure"], sim["turnover"], pick_net, n_trials=1, sharpe_ci=boot["ci1"])
    unit = np.zeros(N)
    unit[0] = 1.0
    ew_stats = S.stats_block(ew, np.ones(N), unit, np.array([np.prod(1 + ew) - 1.0]), sharpe_ci=boot["ci2"])
    taiex_note, tx_stats, tx_eq = None, None, None
    tx = None
    if taiex is not None and taiex.n >= 2:
        tx = index_returns(taiex, pn.cal, t0)
        covered = float(np.mean(np.isin(pn.cal[t0 + 1:], taiex.dates)))
        if covered < 0.5:
            tx, taiex_note = None, "加權指數資料與股票日期重疊不足，未提供指數基準。"
        else:
            tx_stats = S.stats_block(tx, np.ones(N), unit, np.array([np.prod(1 + tx) - 1.0]),
                                     sharpe_ci=M.bootstrap_sharpe(tx, None, n_boot=n_boot)["ci1"])
            taiex_note = "加權指數(^TWII)為價格指數，不含股利，與含息的股票報酬比較時對指數略不利。"
    # ---- model-free benchmark: the top-k by forecast sigma (same tranches, costs and execution)
    sig_stats, ret_sig, P_sig = None, None, None
    P_sig = pn.picks(top_k, False, key=pn.SG)
    W_sig, t0_sig = pn.tranche_weights(P_sig)
    if t0_sig is not None:
        sim_sig = S.simulate_weights(pn.CO, pn.OC, W_sig, pn.buy_cost, pn.sell_cost, exec_mode=exec_mode, first=t0)
        ret_sig = sim_sig["ret"]
        net_sig = (pn.Rs - rt[None, :])[(P_sig > 0) & np.isfinite(pn.R)]
        sig_stats = S.stats_block(ret_sig, sim_sig["exposure"], sim_sig["turnover"], net_sig, n_trials=1,
                                  sharpe_ci=M.bootstrap_sharpe(ret_sig, None, n_boot=n_boot)["ci1"])
    # ---- exposure: mean forecast sigma and causal market beta of the picks vs the universe
    use_taiex = tx is not None
    mkt = index_cc(taiex, pn.cal) if use_taiex else np.concatenate([[0.0], equal_weight_returns(pn, 0)])
    beta = causal_betas(pn.CC, pn.has_bar, mkt)
    ex_top, ex_all = exposure_stats(pn, P, beta, t0), universe_exposure(pn, beta, t0)
    live = pn.avail.copy()
    live[:t0] = False
    basket = {"k": top_k, "mean_sigma": ex_top["mean_sigma"], "universe_mean_sigma": ex_all["mean_sigma"],
              "sigma_ratio": (ex_top["mean_sigma"] / ex_all["mean_sigma"]
                              if ex_top["mean_sigma"] is not None and ex_all["mean_sigma"] else None),
              "mean_beta": ex_top["mean_beta"], "universe_mean_beta": ex_all["mean_beta"],
              "beta_market": "taiex" if use_taiex else "equal_weight", "topk_overlap": topk_overlap(P),
              "corr_rank_sigma": rank_sigma_corr(pn, min_xs, t0),
              "share_net_positive": float(np.mean(pn.net[live] > 0)) if live.any() else None}
    if P_sig is not None and t0_sig is not None:
        basket["sigma_basket"] = exposure_stats(pn, P_sig, beta, t0)
    exposure = {"mean_sigma_top": basket["mean_sigma"], "mean_sigma_universe": basket["universe_mean_sigma"],
                "beta_top": basket["mean_beta"], "beta_universe": basket["universe_mean_beta"],
                "benchmark_top_sigma": sig_stats}
    # ---- cross-sectional statistics (all origin days from t0 on)
    ics = ic_series(pn, min_xs)
    ts_ic = _t_stats(ics[t0:], h)
    valid_ic = np.flatnonzero(np.isfinite(ics) & (np.arange(pn.T) >= t0))
    win = int(min(IC_SERIES_WINDOW, max(1, len(valid_ic) // 4)))
    vals = M.rolling_mean(ics[valid_ic], win) if len(valid_ic) else np.zeros(0)
    dts = pn.cal[valid_ic][win - 1:] if len(valid_ic) else pn.cal[:0]
    ix = M.downsample_indices(len(vals))
    pk = picks_stats(pn, P, h)
    sp = spread_stats(pn, P, top_k, min_xs, h)
    # ---- curves
    cal_dates = np.concatenate([[pn.cal[t0]], pn.cal[t0 + 1:]])
    cix = M.downsample_indices(N + 1)
    eq_top, eq_ew = M.equity_from_returns(ret), M.equity_from_returns(ew)
    curve = {"dates": [M.date_str(d) for d in cal_dates[cix]], "top": np.round(eq_top[cix], 6).tolist(),
             "equal_weight": np.round(eq_ew[cix], 6).tolist(), "taiex": None, "top_sigma": None}
    if tx is not None:
        curve["taiex"] = np.round(M.equity_from_returns(tx)[cix], 6).tolist()
    if ret_sig is not None:
        curve["top_sigma"] = np.round(M.equity_from_returns(ret_sig)[cix], 6).tolist()
    rep = {"horizon": h, "top_k": top_k,
           "period": {"start": M.date_str(pn.cal[t0]), "end": M.date_str(pn.cal[-1]), "n_days": int(N),
                      "n_stocks": int(pn.S)},
           "ic": {"mean": ts_ic["mean"], "ir": ts_ic["ir"], "t": ts_ic["t"], "pvalue": ts_ic["p_two"],
                  "series": {"dates": [M.date_str(d) for d in dts[ix]], "values": np.round(vals[ix], 5).tolist()}},
           "picks": pk, "top": top, "benchmark_equal_weight": ew_stats, "benchmark_taiex": tx_stats,
           "benchmark_top_sigma": sig_stats, "basket": basket, "exposure": exposure, "rank_key": "rank_signal",
           "only_buy_signals": bool(only_buy_signals), "labels": _labels(top_k),
           "spread": {"mean": sp["mean"], "t": sp["t"]}, "curve": curve, "per_stock": per_stock_table(pn)}
    # ---- verdict
    sd_e = float(np.std(ew, ddof=1)) if N > 1 else 0.0
    sr_ew = float(np.mean(ew)) / sd_e if sd_e > 1e-12 else float("nan")
    psr = M.finite_or_none(M.probabilistic_sharpe_ratio(ret, sr_ew)) if math.isfinite(sr_ew) else None
    ps = {"ic": ts_ic["p_one"], "picks": pk["p_value"], "spread": sp["p_one"]}
    ev = {"n_scored": int(ts_ic["n"]), "n_eff": M.effective_n(int(ts_ic["n"]), h) if ts_ic["n"] else 0,
          "p": {k: v for k, v in ps.items() if v is not None and math.isfinite(v)},
          "psr_vs_bh": psr, "diff_p": boot["diff_p"]}
    ass = RP.assess_evidence(ev, RP.THRESH_UNIVERSE)
    rep["_lag"] = ts_ic["lag"]
    rep["verdict"] = make_universe_verdict(rep, ass, ev, h=h, top_k=top_k, exec_mode=exec_mode)
    rep.pop("_lag")
    rep["notes"] = make_universe_notes(h=h, top_k=top_k, exec_mode=exec_mode, n_stocks=pn.S,
                                       commission_discount=commission_discount, slippage_bps=slippage_bps,
                                       only_buy=only_buy_signals, ic_window=win, taiex_note=taiex_note,
                                       alerts=pn.alerts, events=pn.events)
    return M.json_safe(rep)


def evidence_from_report(report, horizon: int, computed_at: str | None = None) -> dict:
    """The ``evidence`` block of the picks payload (DESIGN.md 9(4)) built from a universe report of the same horizon.

    ``{"horizon", "source": "universe_validation", "verdict_level", "ic", "ic_pvalue", "hit_rate", "base_rate", "computed_at"}``
    plus the additive after-cost numbers ``net_cagr`` (the validated top-k basket, net of costs), ``ew_cagr`` (equal-weight
    universe, no costs), ``sigma_cagr`` (model-free top-k by sigma, net of costs) and ``beats_ew_after_costs`` (bool or None):
    a significant ranking statistic that cannot pay the trading costs (the real h = 1 result: IC t = 3.6 but the basket loses
    most of its capital after costs) must not unlock buy-advice wording.  ``report`` may also be the cached job payload
    ``{"report": ..., "computed_at": ...}``.  The same keys with ``source = "none"`` and null values when it is not a usable
    report (missing, malformed, or without any evaluated day); never raises.
    """
    none = {"horizon": int(horizon), "source": "none", "verdict_level": None, "ic": None, "ic_pvalue": None,
            "hit_rate": None, "base_rate": None, "computed_at": None}
    try:
        rep = report if isinstance(report, dict) else {}
        if isinstance(rep.get("report"), dict):                  # the cached job payload {"report", "computed_at", ...}
            computed_at = computed_at or rep.get("computed_at")
            rep = rep["report"]
        lvl = (rep.get("verdict") or {}).get("level")
        per = rep.get("period") or {}
        if lvl not in UNIVERSE_TITLES or not isinstance(rep.get("ic"), dict) or not (per.get("n_days") or 0) > 0:
            return none
        top, ew = rep.get("top") or {}, rep.get("benchmark_equal_weight") or {}
        sg = rep.get("benchmark_top_sigma") or {}
        pk = rep.get("picks") or {}
        net, eqw = M.finite_or_none(top.get("cagr")), M.finite_or_none(ew.get("cagr"))
        return {"horizon": int(horizon), "source": "universe_validation", "verdict_level": lvl,
                "ic": M.finite_or_none(rep["ic"].get("mean")), "ic_pvalue": M.finite_or_none(rep["ic"].get("pvalue")),
                "hit_rate": M.finite_or_none(pk.get("hit_rate")), "base_rate": M.finite_or_none(pk.get("base_rate")),
                "computed_at": computed_at, "net_cagr": net, "ew_cagr": eqw,
                "sigma_cagr": M.finite_or_none(sg.get("cagr")),
                "beats_ew_after_costs": (net > eqw) if (net is not None and eqw is not None) else None}
    except Exception:  # noqa: BLE001 - a malformed cached report must degrade to "no evidence", never break the picks page
        return none


# ----------------------------------------------------------------------------- schema validation
_BASKET_SPEC = {"k": "int", "mean_sigma": "num?", "universe_mean_sigma": "num?", "sigma_ratio": "num?",
                "mean_beta": "num?", "universe_mean_beta": "num?", "beta_market": ("enum", ("taiex", "equal_weight")),
                "topk_overlap": "num?", "corr_rank_sigma": "num?", "share_net_positive": "num?",
                "sigma_basket?": {"mean_sigma": "num?", "mean_beta": "num?"}}
# keys ending in "?" are optional (additive extensions of 6.5; absent in reports of older versions)
UNIVERSE_SPEC = {
    "horizon": "int", "top_k": "int",
    "period": {"start": "str?", "end": "str?", "n_days": "int", "n_stocks": "int"},
    "ic": {"mean": "num?", "ir": "num?", "t": "num?", "pvalue": "num?",
           "series": {"dates": ("list", "str"), "values": ("list", "num?")}},
    "picks": {"hit_rate": "num?", "base_rate": "num?", "n": "int", "p_value": "num?", "mean_return": "num?"},
    "top": RP._STATS_SPEC, "benchmark_equal_weight": RP._STATS_SPEC, "benchmark_taiex": ("opt", RP._STATS_SPEC),
    "benchmark_top_sigma?": ("opt", RP._STATS_SPEC), "basket?": ("opt", _BASKET_SPEC),
    "exposure?": ("opt", {"mean_sigma_top": "num?", "mean_sigma_universe": "num?", "beta_top": "num?",
                          "beta_universe": "num?", "benchmark_top_sigma": ("opt", RP._STATS_SPEC)}),
    "rank_key?": ("enum", ("rank_signal",)), "only_buy_signals?": "bool",
    "labels?": {"top": "str", "benchmark_equal_weight": "str", "benchmark_top_sigma": "str", "benchmark_taiex": "str"},
    "spread": {"mean": "num?", "t": "num?"},
    "curve": {"dates": ("list", "str"), "top": ("list", "num?"), "equal_weight": ("list", "num?"),
              "taiex": ("opt", ("list", "num?")), "top_sigma?": ("opt", ("list", "num?"))},
    "per_stock": ("list", {"code": "str", "name": "str", "accuracy": "num?", "brier_skill": "num?", "ic": "num?",
                           "n": "int", "brier_skill_oracle?": "num?"}),
    "verdict": {"level": ("enum", RP.LEVELS), "title": "str", "text": "str", "bullets": ("list", "str")},
    "notes": ("list", "str"),
}


def validate_universe_report(report: dict) -> list:
    """Schema problems of a universe report (``[]`` when it satisfies DESIGN.md 6.5)."""
    if not isinstance(report, dict):
        return [f"report: expected object, got {type(report).__name__}"]
    probs: list = []
    RP.check_schema(UNIVERSE_SPEC, report, "", probs)
    cur = report.get("curve")
    if isinstance(cur, dict) and isinstance(cur.get("dates"), list):
        for k in ("top", "equal_weight", "taiex", "top_sigma"):
            if isinstance(cur.get(k), list) and len(cur[k]) != len(cur["dates"]):
                probs.append(f"curve.{k}: length {len(cur[k])} != dates {len(cur['dates'])}")
        if len(cur["dates"]) > M.MAX_CURVE_POINTS:
            probs.append(f"curve: {len(cur['dates'])} points exceeds {M.MAX_CURVE_POINTS}")
    ser = (report.get("ic") or {}).get("series") if isinstance(report.get("ic"), dict) else None
    if isinstance(ser, dict) and isinstance(ser.get("dates"), list) and isinstance(ser.get("values"), list):
        if len(ser["dates"]) != len(ser["values"]):
            probs.append("ic.series: dates/values differ in length")
        if len(ser["dates"]) > M.MAX_CURVE_POINTS:
            probs.append(f"ic.series: {len(ser['dates'])} points exceeds {M.MAX_CURVE_POINTS}")
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        probs.append(f"report: not strictly JSON serialisable ({exc})")
    return probs
