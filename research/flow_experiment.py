"""Pre-registered experiment flow-1 (research/PROTOCOL.md): do TWSE institutional flows (三大法人買賣超) predict the
cross-section of 5- / 20-day returns, strongly enough to beat the equal-weight universe after costs?

    python3 -m research.flow_experiment                 # downloads what is missing (first run: ~1 h of T86 pages)
    python3 -m research.flow_experiment --offline       # cached data only

Writes ``research/results/flow1_<date>.json`` and ``.md`` and prints the verdict.  Nothing here is used by the app.
The rules (signals, horizons, splits, thresholds) are fixed in PROTOCOL.md and in the constants below; changing them after a
run invalidates the experiment (start a flow-2 instead).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

from backend import config
from backend.evaluation import metrics as M
from backend.evaluation import strategy as S
from backend.evaluation.universe_bt import _t_stats

# ---------------------------------------------------------------- pre-registered constants (PROTOCOL.md) -- do not edit
EXPERIMENT = "flow-1"
SIGNALS = {                       # name -> (flow key, window in trading days)
    "foreign_20": ("foreign", 20),
    "trust_20": ("trust", 20),
    "inst_5": ("total", 5),
}
HORIZONS = (5, 20)
LAG = 1                           # flows up to D-1 for a decision at the close of D
START = date(2021, 1, 4)
SPLIT_DATE = date(2024, 7, 1)     # dev: < this date; test B: >= this date
DEV_RANKS = (1, 60)
HOLD_RANKS = (61, 120)
MIN_XS = 20                       # stocks needed for a daily cross-sectional IC
N_TESTS = len(SIGNALS) * len(HORIZONS)
ALPHA = 0.05 / N_TESTS            # Bonferroni, one-sided
TOP_K = 5
TRADING_DAYS = 252
MAX_MISSING_SHARE = 0.02          # refuse to report when more T86 days than this are missing (blocked download)

RESULTS = Path(__file__).resolve().parent / "results"


# ================================================================================================ panel
class Panel:
    """(T, S) matrices on the union calendar of the given price series."""

    def __init__(self, datas: list, codes: list):
        self.codes = list(codes)
        self.cal = np.unique(np.concatenate([d.dates for d in datas])) if datas else np.zeros(0, "datetime64[D]")
        T, S_ = len(self.cal), len(datas)
        self.T, self.S = T, S_
        self.has_bar = np.zeros((T, S_), dtype=bool)
        self.vol = np.full((T, S_), np.nan)
        self.CO, self.OC, self.CC = (np.zeros((T, S_)) for _ in range(3))
        self.adj = np.full((T, S_), np.nan)
        self.buy_cost, self.sell_cost = np.zeros(S_), np.zeros(S_)
        for j, d in enumerate(datas):
            pos = np.searchsorted(self.cal, d.dates)
            co, oc, cc = S.leg_returns(d)
            self.has_bar[pos, j] = True
            self.vol[pos, j] = np.asarray(d.volume, dtype=float)
            self.CO[pos, j], self.OC[pos, j], self.CC[pos, j] = co, oc, cc
            self.adj[pos, j] = np.asarray(d.adjclose, dtype=float)
            self.buy_cost[j], self.sell_cost[j] = S.side_costs(S.cost_kind(d.kind))

    def realized(self, h: int) -> np.ndarray:
        """R[t, j] = ln(adj[t+h] / adj[t]) counted in the stock's OWN trading rows (NaN where unknown)."""
        R = np.full((self.T, self.S), np.nan)
        for j in range(self.S):
            rows = np.flatnonzero(self.has_bar[:, j])
            R[rows, j] = M.realized_log_returns(self.adj[rows, j], h)
        return R


def flow_signal(panel: Panel, flow: np.ndarray, window: int, lag: int = LAG) -> np.ndarray:
    """sum(flow over `window` own rows ending `lag` rows before t) / sum(volume over the same rows); NaN when unknown.

    Strictly causal: the value at row t only reads rows <= t - lag of the same stock.
    """
    out = np.full((panel.T, panel.S), np.nan)
    for j in range(panel.S):
        rows = np.flatnonzero(panel.has_bar[:, j])
        n = len(rows)
        if n < window + lag:
            continue
        f = flow[rows, j]
        v = panel.vol[rows, j]
        bad = ~np.isfinite(f) | ~np.isfinite(v)
        f0, v0 = np.where(bad, 0.0, f), np.where(bad, 0.0, v)
        cf = np.concatenate([[0.0], np.cumsum(f0)])
        cv = np.concatenate([[0.0], np.cumsum(v0)])
        cb = np.concatenate([[0], np.cumsum(bad)])
        i = np.arange(n)
        end = i - lag + 1                      # window = own rows [end - window, end)
        start = end - window
        ok = start >= 0
        e, s = end[ok], start[ok]
        sf, sv, sb = cf[e] - cf[s], cv[e] - cv[s], cb[e] - cb[s]
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where((sb == 0) & (sv > 0), sf / sv, np.nan)
        out[rows[ok], j] = val
    return out


# ================================================================================================ statistics
def ic_series(sig: np.ndarray, R: np.ndarray, rows: np.ndarray, cols: np.ndarray, min_xs: int = MIN_XS) -> np.ndarray:
    """Daily cross-sectional Spearman IC on the given row mask (T,) and column mask (S,)."""
    out = np.full(len(rows), np.nan)
    ok = np.isfinite(sig) & np.isfinite(R) & cols[None, :]
    for t in np.flatnonzero(rows & (ok.sum(axis=1) >= min_xs)):
        m = ok[t]
        out[t] = M.spearman_ic(sig[t, m], R[t, m], 1)[0]
    return out


def ic_test(sig, R, rows, cols, h: int) -> dict:
    ics = ic_series(sig, R, rows, cols)
    st = _t_stats(ics, h)
    return {"n_days": st["n"], "ic_mean": st["mean"], "ic_ir": st["ir"], "t": st["t"], "p_one": st["p_one"],
            "pass": bool(st["mean"] is not None and st["mean"] > 0 and st["p_one"] is not None and st["p_one"] < ALPHA)}


def _perf(ret: np.ndarray) -> dict:
    ret = np.asarray(ret, dtype=float)
    ret = ret[np.isfinite(ret)]
    if len(ret) < 20:
        return {"n_days": len(ret), "cagr": None, "sharpe": None, "max_dd": None, "total": None}
    eq = np.cumprod(1.0 + ret)
    sd = float(ret.std(ddof=1))
    return {"n_days": len(ret), "total": float(eq[-1] - 1.0), "cagr": float(eq[-1] ** (TRADING_DAYS / len(ret)) - 1.0),
            "sharpe": float(ret.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else None,
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1.0).min())}


def basket_test(panel: Panel, sig: np.ndarray, rows: np.ndarray, cols: np.ndarray, h: int, top_k: int = TOP_K) -> dict:
    """Top-k by signal, equal weight, overlapping tranches held h days, next-open execution with costs, vs the
    equal-weight universe (no costs) of the same columns, over the contiguous row range of ``rows``."""
    idx = np.flatnonzero(rows)
    if len(idx) < 2 * h + 20:
        return {"pass": False, "reason": "too few days"}
    a, b = int(idx[0]), int(idx[-1]) + 1
    C = np.flatnonzero(cols)
    sg = sig[a:b][:, C]
    T, S_ = sg.shape
    elig = np.isfinite(sg) & panel.has_bar[a:b][:, C]
    P = np.zeros((T, S_))
    for t in range(T):
        e = np.flatnonzero(elig[t])
        if len(e) >= max(top_k * 2, MIN_XS):
            top = e[np.lexsort((e, -sg[t, e]))[:top_k]]
            P[t, top] = 1.0 / top_k
    has = P.sum(axis=1) > 0
    if not has.any():
        return {"pass": False, "reason": "no picks"}
    t0 = int(np.argmax(has))
    Cs = np.vstack([np.zeros((1, S_)), np.cumsum(P, axis=0)])
    tt = np.arange(T)
    lo = np.maximum(tt - h + 1, t0)
    m = np.where(tt >= t0, tt - lo + 1, 1)
    W = (Cs[tt + 1] - Cs[lo]) / m[:, None]
    W[:t0] = 0.0
    sim = S.simulate_weights(panel.CO[a:b][:, C], panel.OC[a:b][:, C], W, panel.buy_cost[C], panel.sell_cost[C],
                             exec_mode="next_open", first=t0)
    prev_ok = elig[t0:T - 1]
    cc = panel.CC[a:b][:, C][t0 + 1:T]
    with np.errstate(invalid="ignore"):
        ew = np.where(prev_ok.sum(axis=1) > 0, (np.where(prev_ok, cc, 0.0)).sum(axis=1) / np.maximum(prev_ok.sum(axis=1), 1), 0.0)
    top, bench = _perf(sim["ret"]), _perf(ew)
    ok = (top["cagr"] is not None and bench["cagr"] is not None and top["sharpe"] is not None and bench["sharpe"] is not None
          and top["cagr"] > bench["cagr"] and top["sharpe"] > bench["sharpe"])
    return {"pass": bool(ok), "top": top, "equal_weight": bench,
            "turnover_per_year": float(np.mean(sim["turnover"]) * TRADING_DAYS) if len(sim["turnover"]) else None}


# ================================================================================================ experiment
def run(panel: Panel, flows: dict, dev_cols: np.ndarray, hold_cols: np.ndarray) -> dict:
    """Every pre-registered test on an assembled panel.  ``flows`` = {flow key: (T, S) net shares}."""
    split = np.datetime64(SPLIT_DATE)
    dev_rows = panel.cal < split
    late_rows = panel.cal >= split
    all_rows = np.ones(panel.T, dtype=bool)
    results = []
    for name, (key, window) in SIGNALS.items():
        sig = flow_signal(panel, flows[key], window)
        for h in HORIZONS:
            R = panel.realized(h)
            r = {"signal": name, "h": h,
                 "dev": ic_test(sig, R, dev_rows, dev_cols, h),
                 "test_a": ic_test(sig, R, all_rows, hold_cols, h),
                 "test_b": ic_test(sig, R, late_rows, dev_cols, h)}
            r["signal_holds"] = r["test_a"]["pass"] and r["test_b"]["pass"]
            if r["signal_holds"]:
                r["basket_a"] = basket_test(panel, sig, all_rows, hold_cols, h)
                r["basket_b"] = basket_test(panel, sig, late_rows, dev_cols, h)
                r["profitable"] = r["basket_a"]["pass"] and r["basket_b"]["pass"]
            else:
                r["profitable"] = False
            results.append(r)
    winners = [f"{r['signal']} h={r['h']}" for r in results if r["profitable"]]
    return {"experiment": EXPERIMENT, "alpha_per_test": ALPHA, "n_tests": N_TESTS, "results": results,
            "n_dev_stocks": int(dev_cols.sum()), "n_hold_stocks": int(hold_cols.sum()),
            "period": [str(panel.cal[0]) if panel.T else None, str(panel.cal[-1]) if panel.T else None],
            "verdict": "pass" if winners else "fail", "winners": winners}


def _f(x, fmt="{:+.4f}"):
    return "—" if x is None else fmt.format(x)


def to_markdown(rep: dict) -> str:
    lines = [f"# 實驗 {rep['experiment']} 結果", "",
             f"期間 {rep['period'][0]} ~ {rep['period'][1]}；開發組 {rep['n_dev_stocks']} 檔、保留組 {rep['n_hold_stocks']} 檔；"
             f"單一檢定門檻 p < {rep['alpha_per_test']:.4f}（{rep['n_tests']} 個檢定，Bonferroni）。", "",
             "| 訊號 | h | 開發期 IC (p) | 測試A 換股票 IC (p) | 測試B 換時間 IC (p) | 訊號成立 | 扣成本後勝過等權重 |",
             "|---|---|---|---|---|---|---|"]
    for r in rep["results"]:
        cell = lambda x: f"{_f(x['ic_mean'])} ({_f(x['p_one'], '{:.3g}')})"
        lines.append(f"| {r['signal']} | {r['h']} | {cell(r['dev'])} | {cell(r['test_a'])} | {cell(r['test_b'])} | "
                     f"{'是' if r['signal_holds'] else '否'} | {'是' if r['profitable'] else '否'} |")
    for r in rep["results"]:
        if r.get("basket_a"):
            for k, lab in (("basket_a", "測試A"), ("basket_b", "測試B")):
                b = r[k]
                if "top" in b:
                    lines.append(f"- {r['signal']} h={r['h']} {lab}：前 {TOP_K} 名扣成本後年化 {_f(b['top']['cagr'], '{:.1%}')}、夏普 "
                                 f"{_f(b['top']['sharpe'], '{:.2f}')}；等權重年化 {_f(b['equal_weight']['cagr'], '{:.1%}')}、夏普 "
                                 f"{_f(b['equal_weight']['sharpe'], '{:.2f}')}；年換手 {_f(b['turnover_per_year'], '{:.0f}')} 倍")
    lines += ["", "## 結論", ""]
    if rep["verdict"] == "pass":
        lines.append("通過預先登記的兩道門檻：" + "、".join(rep["winners"]) + "。下一步是把它整合進模型，並用之後的新資料"
                     "（前瞻紀錄）再驗證一次；過去有效不保證未來，也不等於可以穩定獲利。")
    else:
        lines.append("沒有任何訊號同時通過「換股票」與「換時間」兩個測試，並在扣成本後勝過全部股票等權重。依預先登記的規則，"
                     "結論是：這些三大法人訊號沒有可用的證據；不得在同一份資料上改參數重試。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from backend import data as D
    from research import twse_flow as TF

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="use cached prices / T86 pages only")
    args = ap.parse_args(argv)

    uni = D.get_universe(HOLD_RANKS[1])
    ranked = [(i + 1, u) for i, u in enumerate(uni) if u.get("market") == "TWSE"]
    datas, codes, rank_of = [], [], {}
    for rank, u in ranked:
        try:
            d = D.completed(D.get_price_data(u["code"], refresh=not args.offline))
        except D.DataError as exc:
            print(f"skip {u['code']}: {exc}", file=sys.stderr)
            continue
        keep = d.dates >= np.datetime64(START)
        if keep.sum() < 300:
            continue
        datas.append(d.slice(d.n) if keep.all() else _tail(d, int(np.argmax(keep))))
        codes.append(u["code"])
        rank_of[u["code"]] = rank
    if len(codes) < 2 * MIN_XS:
        print("not enough price series", file=sys.stderr)
        return 2
    panel = Panel(datas, codes)
    days = {}
    for d64 in panel.cal:
        d = d64.astype(object)
        try:
            days[d] = TF.fetch_day(d, network=not args.offline)
        except TF.FlowUnavailable as exc:
            print(f"T86 {d}: {exc}", file=sys.stderr)
            days[d] = None
    missing = [d for d, v in days.items() if not (isinstance(v, dict) and v.get("status") == "ok")]
    if len(missing) > MAX_MISSING_SHARE * len(days):
        print(f"{len(missing)} / {len(days)} 個交易日的三大法人資料沒有下載成功（多半是證交所暫時封鎖太頻繁的請求）。"
              "已下載的部分都存在快取裡，請過一段時間（例如一小時）再執行同一個指令，會從中斷處繼續。", file=sys.stderr)
        return 3
    flows = TF.flow_matrices(days, codes, panel.cal)
    ranks = np.array([rank_of[c] for c in codes])
    dev_cols = (ranks >= DEV_RANKS[0]) & (ranks <= DEV_RANKS[1])
    hold_cols = (ranks >= HOLD_RANKS[0]) & (ranks <= HOLD_RANKS[1])
    rep = run(panel, flows, dev_cols, hold_cols)
    rep["generated_at"] = datetime.now().isoformat(timespec="seconds")
    rep["t86_days_missing"] = len(missing)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"flow1_{date.today():%Y%m%d}"
    stem.with_suffix(".json").write_text(json.dumps(M.json_safe(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(rep)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _tail(d, i0: int):
    """Rows [i0:] of a PriceData (PriceData.slice only cuts the end)."""
    from backend.types import PriceData
    return PriceData(d.symbol, d.name, d.market, d.kind, d.dates[i0:].copy(), d.open[i0:].copy(), d.high[i0:].copy(),
                     d.low[i0:].copy(), d.close[i0:].copy(), d.adjclose[i0:].copy(), d.volume[i0:].copy(),
                     [x for x in d.dividends if x[0] >= d.dates[i0]], d.last_bar_complete, d.fetched_at, d.stale)


if __name__ == "__main__":
    sys.exit(main())
