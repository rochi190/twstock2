"""Evaluation harness (PLAN.md section 6, P0): cross-sectional rank IC, long-basket and residual-IC reports.

Conventions
-----------
* Everything is ``(T, S)``: row ``t`` = decision day (after the close of ``t``), column = stock.  The label is
  ``panel.forward_open_return(panel, h)`` (open of ``t+1`` to open of ``t+1+h``), the trade happens at the open of ``t+1``.
* ``mask`` is the day-``t`` universe (``panel.universe_mask``).  Entries outside the mask, or non-finite, are ignored.
* Periods (decision dates, inclusive): ``dev`` 2013-01-01..2022-12-31, ``dev_a`` 2013..2017, ``dev_b`` 2018..2022,
  ``oos`` 2023-01-01 onward.  For the closed periods a decision row also needs its label window to end inside the period
  (``cal[t+1+h] <= end``): the last ``h+1`` decision days of DEV are dropped, so no DEV statistic contains OOS prices.
* ``period="oos"`` demands a ``spec_hash`` (the hash of the frozen composite spec) and appends one row to
  ``docs/research/LOOKS.md`` -- an OOS look is never silent.  Without ``spec_hash`` the function raises BEFORE computing.
* Inference on daily series (IC, excess return) reuses ``backend.evaluation.universe_bt._t_stats`` (Newey-West HAC with the
  lag rule of that module, which accounts for the h-day overlap of the targets).

Public helpers for later stages: :func:`xs_rank`, :func:`ic_series`, :func:`hac_mean_t`, :func:`composite`,
:func:`screen_dev`, :func:`tranche_weights`, :func:`spec_hash_of`, :func:`md_table`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

from backend import config
from backend.evaluation import strategy as ST
from backend.evaluation import universe_bt as UB

from .panel import Panel, forward_open_return, untradable_open

REPO_ROOT = Path(__file__).resolve().parent.parent
LOOKS_PATH = REPO_ROOT / "docs" / "research" / "LOOKS.md"     # tests monkeypatch this module attribute
DAYS_PER_YEAR = 245                                           # Taiwan trading days per year (annualisation)

PERIODS = {
    "dev": ("2013-01-01", "2022-12-31"),
    "dev_a": ("2013-01-01", "2017-12-31"),
    "dev_b": ("2018-01-01", "2022-12-31"),
    "oos": ("2023-01-01", None),
}
SCREEN_T = 3.0           # PLAN.md section 5: |HAC t| of the mean rank IC on DEV
LOOKS_HEADER = ("# OOS 查看紀錄 (LOOKS)\n\n每次以 `period=\"oos\"` 呼叫 `research.harness` 都會自動附加一列。OOS 只能看一次（PLAN.md §0 規則 4、§6）。\n\n"
                "| 日期 (UTC) | git commit | 看了什麼 | 結果 |\n|---|---|---|---|\n")


class OOSGuardError(ValueError):
    """Raised when ``period="oos"`` is requested without a ``spec_hash``."""


# ============================================================================= periods
def _check_period(period: str, spec_hash) -> None:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {sorted(PERIODS)}, got {period!r}")
    if period == "oos" and not (isinstance(spec_hash, str) and spec_hash.strip()):
        raise OOSGuardError('period="oos" requires a non-empty spec_hash (hash of the frozen composite spec)')


def period_bounds(period: str):
    """``(start, end)`` as ``datetime64[D]``; ``end`` is None for the open-ended ``oos``."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {sorted(PERIODS)}, got {period!r}")
    s, e = PERIODS[period]
    return np.datetime64(s, "D"), (None if e is None else np.datetime64(e, "D"))


def period_rows(cal, period: str, h: int = 0) -> np.ndarray:
    """``(T,)`` bool of the decision days of ``period``; for a closed period the label window (``t+1+h``) must end by the
    period's last date (purging, see module docstring).  ``h = 0`` gives the plain date filter (used for realised-return days).
    """
    cal = np.asarray(cal).astype("datetime64[D]")
    start, end = period_bounds(period)
    rows = cal >= start
    if end is not None:
        rows &= cal <= end
        if h > 0:
            n_in = int(np.searchsorted(cal, end, side="right"))      # number of calendar days <= end
            rows &= np.arange(len(cal)) < n_in - 1 - int(h)
    return rows


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        s = r.stdout.strip()
        if r.returncode == 0 and s:
            d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=10)
            return s + ("+dirty" if d.stdout.strip() else "")
    except Exception:  # noqa: BLE001 -- git missing / not a repo: still log the look
        pass
    return "unknown"


def _log_look(what: str, result: str, path=None) -> None:
    """Append one row to LOOKS.md (creating it with a header if missing)."""
    p = Path(path) if path is not None else Path(LOOKS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = lambda s: str(s).replace("|", "/").replace("\n", " ").strip()   # noqa: E731
    row = f"| {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} | {_git_commit()} | {clean(what)} | {clean(result)} |\n"
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", encoding="utf-8") as fh:
        if new:
            fh.write(LOOKS_HEADER)
        fh.write(row)


class _Look:
    """Context manager: for ``oos`` writes the LOOKS row when the computation ends (also on error); no-op otherwise."""

    def __init__(self, period, spec_hash, what):
        self.on = period == "oos"
        self.what = f"{what} spec_hash={spec_hash}"
        self.result = "(no result)"

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if self.on:
            _log_look(self.what, f"ERROR {et.__name__}: {ev}" if et else self.result)
        return False


# ============================================================================= small tools
def _f(x):
    """JSON-safe float (None for NaN/inf/None)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def hac_mean_t(x, h: int = 1) -> dict:
    """HAC t-test of the mean of a daily series ``x`` (NaNs ignored) -> ``{n, mean, ir, t, p_two, p_one, lag, se}``.

    Reuses ``universe_bt._t_stats`` (Newey-West, Bartlett kernel, lag ``min(max(2h, ceil(3 n^(1/3))), n//5)``, Student-t
    reference).  ``se = mean / t`` (None when ``t`` is None or 0).  ``h`` is the overlap of the targets.
    """
    r = UB._t_stats(x, int(h))
    out = {k: _f(v) if k not in ("n", "lag") else int(v) for k, v in r.items()}
    out["se"] = _f(r["mean"] / r["t"]) if r.get("t") not in (None, 0) and r.get("mean") is not None else None
    return out


def _row_rank(A: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) of ``A`` among the ``valid`` entries of each row; NaN elsewhere."""
    out = np.full(A.shape, np.nan)
    for t in np.flatnonzero(valid.sum(axis=1) >= 1):
        v = valid[t]
        out[t, v] = rankdata(A[t, v])
    return out


def xs_rank(F, mask=None, fill: float = np.nan) -> np.ndarray:
    """Cross-sectional rank of ``F`` per row, mapped to ``[-0.5, 0.5]`` (ties share the average rank).

    Ranks are taken among entries that are finite and inside ``mask`` (default: all).  With ``n`` such entries the value is
    ``(rank - 1) / (n - 1) - 0.5`` (the minimum maps to -0.5, the maximum to +0.5; a single entry maps to 0).  Entries
    outside the mask / non-finite get ``fill`` (default NaN; PLAN.md section 5 uses ``fill=0.0`` -- "missing counts as 0" --
    before features are combined).
    """
    F = np.asarray(F, dtype=float)
    valid = np.isfinite(F) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(F))
    r = _row_rank(F, valid)
    n = valid.sum(axis=1)[:, None]
    with np.errstate(all="ignore"):
        out = np.where(n > 1, (r - 1.0) / np.maximum(n - 1.0, 1.0) - 0.5, 0.0)
    return np.where(valid, out, fill)


def ic_series(F, y, mask, min_xs: int = 20) -> np.ndarray:
    """``(T,)`` daily cross-sectional Spearman correlation of ``F`` with ``y`` over entries that are in ``mask`` and finite in
    both; NaN on days with fewer than ``min_xs`` such entries."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    valid = np.asarray(mask, bool) & np.isfinite(F) & np.isfinite(y)
    return _spearman_rows(F, y, valid, min_xs)


def _spearman_rows(A, B, valid, min_xs):
    n = valid.sum(axis=1)
    ra, rb = _row_rank(A, valid), _row_rank(B, valid)
    mid = ((n + 1.0) / 2.0)[:, None]
    a = np.where(valid, ra - mid, 0.0)
    b = np.where(valid, rb - mid, 0.0)
    with np.errstate(all="ignore"):
        ic = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.where((n >= min_xs) & np.isfinite(ic), ic, np.nan)


def _as_cal(cal):
    return np.asarray(getattr(cal, "cal", cal)).astype("datetime64[D]")


def _series_summary(x: np.ndarray, rows: np.ndarray, cal, h: int) -> dict:
    """HAC summary of a daily series restricted to ``rows``, plus per-calendar-year means and the positive-day share."""
    xs = np.where(rows, x, np.nan)
    st = hac_mean_t(xs, h)
    fin = np.isfinite(xs)
    years = cal.astype("datetime64[Y]").astype(int) + 1970
    by_year = {int(yy): _f(np.nanmean(xs[fin & (years == yy)])) for yy in np.unique(years[fin])}
    st.update(n_days=int(fin.sum()), pos_frac=_f((xs[fin] > 0).mean()) if fin.any() else None, by_year=by_year)
    return st


# ============================================================================= IC
def ic_report(F, y, mask, h: int, period: str, cal, spec_hash=None, min_xs: int = 20) -> dict:
    """Mean cross-sectional Spearman IC of ``F`` vs the label ``y`` (= ``forward_open_return(panel, h)``) in ``period``.

    ``cal`` is the panel calendar (array or :class:`Panel`).  Returns ``{period, h, mean, t, ir, p_two, lag, se, n_days,
    pos_frac, by_year, avg_xs}`` where ``t`` is the HAC t of the mean IC.  ``period="oos"`` needs ``spec_hash`` and is logged.
    """
    _check_period(period, spec_hash)
    cal = _as_cal(cal)
    with _Look(period, spec_hash, f"ic_report h={h}") as look:
        ic = ic_series(F, y, mask, min_xs)
        rows = period_rows(cal, period, h)
        rep = _series_summary(ic, rows, cal, h)
        used = rows & np.isfinite(ic)
        n_xs = (np.asarray(mask, bool) & np.isfinite(F) & np.isfinite(y)).sum(axis=1)
        rep.update(period=period, h=int(h), avg_xs=_f(n_xs[used].mean()) if used.any() else None)
        look.result = f"mean IC={rep['mean']} HAC t={rep['t']} n_days={rep['n_days']}"
    return rep


def screen_dev(F, y, mask, h: int, cal, expected_sign: int = 0, min_xs: int = 20) -> dict:
    """PLAN.md section 5 DEV screen (DEV only): ``|HAC t| >= 3`` on ``dev``, the same IC sign in ``dev_a`` and ``dev_b``, and
    that sign equal to ``expected_sign`` (+1/-1; 0 = no expectation).  Returns the three compact reports and the verdicts."""
    reps = {p: ic_report(F, y, mask, h, p, cal, min_xs=min_xs) for p in ("dev", "dev_a", "dev_b")}
    m = {p: reps[p]["mean"] for p in reps}
    t = reps["dev"]["t"]
    same = all(v is not None for v in m.values()) and np.sign(m["dev_a"]) == np.sign(m["dev_b"]) != 0
    sign = float(np.sign(m["dev"])) if m["dev"] is not None else 0.0
    exp_ok = (not expected_sign) or sign == float(np.sign(expected_sign))
    return {"dev": reps["dev"], "dev_a": reps["dev_a"], "dev_b": reps["dev_b"], "t_ok": bool(t is not None and abs(t) >= SCREEN_T),
            "halves_same_sign": bool(same), "sign_as_expected": bool(exp_ok),
            "passes": bool(t is not None and abs(t) >= SCREEN_T and same and exp_ok)}


# ============================================================================= residual IC
def residual_ic(F, controls, y, mask, h: int, period: str, cal, spec_hash=None, min_xs: int = 20) -> dict:
    """IC of the part of ``F`` not explained by ``controls``.

    Each day ``F`` and every control are cross-sectionally ranked (``xs_rank``) among the stocks that have all of them and
    ``y``; the ranked ``F`` is regressed (OLS with intercept) on the ranked controls and the Spearman correlation of the
    residual with ``y`` is the day's residual IC.  The projection is linear in ranks, so a feature that is a noisy copy of a
    control keeps a small nonlinear leakage of it (a few percent of the raw IC in simulations): read the residual IC as
    "what is left after the linear rank effect of the controls", not as an exact orthogonalisation.  ``controls``: dict ``name -> (T, S)`` or a sequence of arrays (PLAN.md:
    ``neg_sigma``, ``mom_12_1``, ``rev_5``).  Returns the :func:`ic_report` fields plus ``raw_mean`` / ``raw_t`` (plain IC on
    the very same rows) and the control names.
    """
    _check_period(period, spec_hash)
    cal = _as_cal(cal)
    if isinstance(controls, dict):
        names = list(controls.keys())
        C = [np.asarray(controls[k], float) for k in names]
    else:
        C = [np.asarray(c, float) for c in controls]
        names = [f"c{i}" for i in range(len(C))]
    F, y = np.asarray(F, float), np.asarray(y, float)
    with _Look(period, spec_hash, f"residual_ic h={h}") as look:
        valid = np.asarray(mask, bool) & np.isfinite(F) & np.isfinite(y)
        for c in C:
            valid &= np.isfinite(c)
        rF = xs_rank(F, valid)
        rC = [xs_rank(c, valid) for c in C]
        ry = _row_rank(y, valid)
        n = valid.sum(axis=1)
        need = max(int(min_xs), len(C) + 5)
        res_ic = np.full(F.shape[0], np.nan)
        for t in np.flatnonzero((n >= need) & period_rows(cal, period, h)):
            v = valid[t]
            X = np.column_stack([np.ones(int(v.sum()))] + [c[t, v] for c in rC])
            beta, *_ = np.linalg.lstsq(X, rF[t, v], rcond=None)
            res = rF[t, v] - X @ beta
            a = rankdata(res) - (n[t] + 1.0) / 2.0
            b = ry[t, v] - (n[t] + 1.0) / 2.0
            den = math.sqrt(float(a @ a) * float(b @ b))
            res_ic[t] = float(a @ b) / den if den > 0 else np.nan
        raw_ic = _spearman_rows(F, y, valid, need)
        rows = period_rows(cal, period, h)
        rep = _series_summary(res_ic, rows, cal, h)
        raw = hac_mean_t(np.where(rows & np.isfinite(res_ic), raw_ic, np.nan), h)
        rep.update(period=period, h=int(h), controls=names, raw_mean=raw["mean"], raw_t=raw["t"])
        look.result = f"residual IC={rep['mean']} HAC t={rep['t']} n_days={rep['n_days']}"
    return rep


# ============================================================================= long basket
def tranche_weights(P: np.ndarray, h: int):
    """Overlapping-tranche weights ``(W, t0)`` from the pick matrix ``P`` (rows = decision days): the average of the last ``h``
    pick vectors (expanding at the start).  Calls ``universe_bt._Panel.tranche_weights`` (same method as the product's
    universe validation) on a minimal stand-in carrying ``h``, ``T`` and ``S``."""
    T, S = P.shape
    return UB._Panel.tranche_weights(SimpleNamespace(h=int(h), T=T, S=S), P)


def leg_matrices(panel: Panel):
    """``(co, oc)`` overnight / intraday simple-return matrices ``(T, S)`` on the panel calendar.

    Per stock the legs come from ``strategy.leg_returns`` over the stock's OWN consecutive bars (so a stock's first bar after a
    gap carries the whole gap in ``co``); days without a bar are 0/0 (a held position earns nothing and costs nothing)."""
    co = np.zeros((panel.T, panel.S))
    oc = np.zeros((panel.T, panel.S))
    for j in range(panel.S):
        r = np.flatnonzero(panel.has_bar[:, j])
        if len(r) < 2:
            continue
        a, b, _ = ST.leg_returns(SimpleNamespace(adjclose=panel.adj_close[r, j], adj_open=panel.adj_open[r, j]))
        co[r, j], oc[r, j] = a, b
    return co, oc


def _costs(panel: Panel, commission_discount, slippage_bps):
    bc, sc = np.zeros(panel.S), np.zeros(panel.S)
    for j, k in enumerate(panel.kinds):
        bc[j], sc[j] = ST.side_costs(ST.cost_kind(str(k)), commission_discount, slippage_bps)
    return bc, sc


def _max_drawdown(ret: np.ndarray) -> float | None:
    if len(ret) == 0:
        return None
    eq = np.cumprod(1.0 + ret)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _ann_geo(ret: np.ndarray) -> float | None:
    if len(ret) == 0:
        return None
    with np.errstate(all="ignore"):
        return float(np.exp(np.log1p(ret).sum() * DAYS_PER_YEAR / len(ret)) - 1.0)


def quintile_report(F, panel: Panel, mask, h: int, period: str, spec_hash=None, top_frac: float = 0.2,
                    commission_discount: float = config.COMMISSION_DISCOUNT, slippage_bps: float = config.SLIPPAGE_BPS,
                    min_xs: int = 20, direction: int = 1) -> dict:
    """Long basket of the top ``top_frac`` (default 20 %) of ``direction * F`` versus the equal-weight candidate set.

    Candidates on day ``t``: ``mask & finite(F) & has_bar[t] & ~untradable_open[t]`` (names whose entry open is locked at the
    limit-up are excluded); days with fewer than ``min_xs`` candidates hold nothing.  The basket is the top
    ``ceil(top_frac * n_candidates)`` names, equal weight (ties broken by column order).  Holding ``h`` days uses overlapping
    tranches (``tranche_weights``) and ``strategy.simulate_weights(exec_mode="next_open")`` with ``strategy.side_costs``
    (buy: commission+slippage, sell: +tax).  The benchmark is the equal-weight of ALL candidates run through the identical
    tranche/cost machine, so the excess isolates the ranking.  Statistics use the realised return days inside ``period``
    on which both portfolios are invested; HAC t of the daily excess via ``universe_bt._t_stats``.

    Returned (floats; annualised with 245 days): ``ann_net`` / ``ann_bench`` (geometric), ``ann_excess`` (mean daily net excess
    x 245), ``excess_t``, ``ann_excess_gross``, ``mdd``, ``ann_turnover`` (mean daily sum|dw| x 245: buy+sell notional / capital),
    ``ann_cost`` (basket), ``sigma_ratio`` and ``beta`` (basket vs benchmark, daily net returns), ``mean_k``, plus
    ``quintile_y``: mean ``y_h`` by quintile of ``direction * F`` (Q1 lowest .. Q5 highest, gross, no costs, untradable names
    included) with ``q5_minus_q1`` and its HAC t.
    """
    _check_period(period, spec_hash)
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must be in (0, 1]")
    cal = panel.cal
    T, S = panel.T, panel.S
    F = np.asarray(F, float) * (1.0 if direction >= 0 else -1.0)
    with _Look(period, spec_hash, f"quintile_report h={h}") as look:
        cand = np.asarray(mask, bool) & np.isfinite(F) & panel.has_bar & ~untradable_open(panel)
        nc = cand.sum(axis=1)
        k = np.where(nc >= min_xs, np.ceil(top_frac * nc), 0).astype(int)
        order = np.argsort(-np.where(cand, F, -np.inf), axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(S), order.shape), axis=1)
        sel = cand & (rank < k[:, None])
        P = sel / np.maximum(k, 1)[:, None]
        B = (cand & (k > 0)[:, None]) / np.maximum(nc, 1)[:, None]
        W, _ = tranche_weights(P, h)
        WB, _ = tranche_weights(B, h)
        co, oc = leg_matrices(panel)
        bc, sc = _costs(panel, commission_discount, slippage_bps)
        sim = ST.simulate_weights(co, oc, W, bc, sc, exec_mode="next_open")
        simb = ST.simulate_weights(co, oc, WB, bc, sc, exec_mode="next_open")
        d_idx = np.arange(1, T)                                  # realised return day of sim[...][i] is calendar row i + 1
        use = period_rows(cal, period, 0)[d_idx] & (sim["exposure"] > 0) & (simb["exposure"] > 0)
        net, bench = sim["ret"][use], simb["ret"][use]
        excess = net - bench
        n = int(use.sum())
        rep = {"period": period, "h": int(h), "n_days": n, "mean_k": _f(k[(k > 0) & period_rows(cal, period, 0)].mean()) if ((k > 0) & period_rows(cal, period, 0)).any() else None,
               "ann_net": _f(_ann_geo(net)), "ann_bench": _f(_ann_geo(bench)),
               "ann_excess": _f(excess.mean() * DAYS_PER_YEAR) if n else None, "excess_t": None, "excess_p_two": None,
               "ann_excess_gross": _f((sim["gross"][use] - simb["gross"][use]).mean() * DAYS_PER_YEAR) if n else None,
               "mdd": _f(_max_drawdown(net)), "mdd_bench": _f(_max_drawdown(bench)),
               "ann_turnover": _f(sim["turnover"][use].mean() * DAYS_PER_YEAR) if n else None,
               "ann_cost": _f(sim["cost"][use].mean() * DAYS_PER_YEAR) if n else None,
               "sigma_ratio": None, "beta": None}
        if n >= 3:
            st = hac_mean_t(excess, h)
            rep["excess_t"], rep["excess_p_two"] = st["t"], st["p_two"]
            if bench.std(ddof=1) > 0:
                rep["sigma_ratio"] = _f(net.std(ddof=1) / bench.std(ddof=1))
                rep["beta"] = _f(np.cov(net, bench)[0, 1] / bench.var(ddof=1))
        rep["quintile_y"] = _quintile_y(F, forward_open_return(panel, h), np.asarray(mask, bool), cal, h, period, min_xs)
        look.result = f"ann_excess={rep['ann_excess']} t={rep['excess_t']} n_days={n}"
    return rep


def _quintile_y(F, y, mask, cal, h, period, min_xs) -> dict:
    valid = mask & np.isfinite(F) & np.isfinite(y)
    r = _row_rank(F, valid)
    n = valid.sum(axis=1)
    rows = period_rows(cal, period, h) & (n >= max(min_xs, 5))
    with np.errstate(all="ignore"):
        bucket = np.where(valid, np.minimum(np.nan_to_num((r - 1.0) * 5.0 / np.maximum(n, 1)[:, None]).astype(int), 4), -1)
    means = np.full((len(cal), 5), np.nan)
    for q in range(5):
        sel = valid & (bucket == q)
        with np.errstate(all="ignore"):
            means[:, q] = np.where(sel, y, 0.0).sum(axis=1) / sel.sum(axis=1)
    means[~rows] = np.nan
    spread = means[:, 4] - means[:, 0]
    st = hac_mean_t(spread, h)
    return {"mean_y": [_f(np.nanmean(means[:, q])) if np.isfinite(means[:, q]).any() else None for q in range(5)],
            "q5_minus_q1": st["mean"], "q5_minus_q1_t": st["t"], "n_days": st["n"]}


# ============================================================================= composite / utilities
def composite(feats, signs, mask) -> np.ndarray:
    """Equal-weight composite signal (PLAN.md section 6): every feature is cross-sectionally ranked (``xs_rank``, missing = 0),
    multiplied by its expected-direction sign and the results are averaged (no fitted weights).  ``feats``: dict
    ``name -> (T, S)``; ``signs``: dict ``name -> +1/-1``.  NaN outside ``mask``."""
    names = list(feats)
    if not names:
        raise ValueError("no features")
    mask = np.asarray(mask, bool)
    acc = np.zeros(mask.shape)
    for k in names:
        s = float(np.sign(signs[k]))
        if s == 0:
            raise ValueError(f"sign of {k} must be +1 or -1")
        acc += s * xs_rank(feats[k], mask, fill=0.0)
    return np.where(mask, acc / len(names), np.nan)


def spec_hash_of(obj) -> str:
    """Short hash (12 hex) of a JSON-serialisable spec (canonical: sorted keys, compact separators)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def md_table(headers, rows) -> str:
    """GitHub-markdown table; ``None`` renders as ``-``."""
    cell = lambda v: "-" if v is None else str(v).replace("|", "/")   # noqa: E731
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)
