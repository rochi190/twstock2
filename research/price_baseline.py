"""Price-only control features P1-P4 on DEV (PLAN.md section 5 "價格控制組", section 7 P0 item 3).

Features (taken from the product's ``backend.quant.features.build_features`` -- imported, never modified):

====  ==============  ===========================================================================================
P1    ``rev_5``       ``raw['rev_5']``    = exp(5-bar cleaned log return) - 1
P2    ``mom_12_1``    ``raw['mom_12_1']`` = exp(cs[t-21] - cs[t-252]) - 1   (needs 253 bars)
P3    ``idio_mom``    ``raw['idio_mom']`` = exp(60-bar sum of market-model residuals) - 1   (needs TAIEX)
P4    ``neg_sigma``   ``-sigma_ref``      = -sqrt(0.5 EWMA_0.94 + 0.5 EWMA_0.99) of squared cleaned returns (low vol)
====  ==============  ===========================================================================================

Every one of them is a prefix function of the stock's own bars (value at bar ``t`` uses bars ``<= t``; the TAIEX is aligned as
"last index bar dated <= the stock date").  Features are computed on each stock's FULL history and then scattered onto the
panel calendar, so lookbacks are not truncated by the panel start.

Run ``python -m research.price_baseline`` (real cached data, offline) or ``--synthetic`` (random-walk smoke test; never writes
RESULTS.md unless ``--results-md`` is given).  Output: ``research/out/price_baseline_dev.json`` and section 1 of
``docs/research/RESULTS.md`` (idempotent: the section between the BEGIN/END markers is replaced).  Only DEV is used; OOS is
never touched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

from backend import data as D
from backend.quant import features as QF
from backend.types import PriceData

from . import harness as H
from .panel import Panel, align_series, forward_open_return, load_datas, load_pool, panel_path, save_panel, universe_mask

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "research" / "out" / "price_baseline_dev.json"
RESULTS_MD = REPO_ROOT / "docs" / "research" / "RESULTS.md"
SECTION_BEGIN = "<!-- BEGIN price_baseline -->"
SECTION_END = "<!-- END price_baseline -->"
RESULTS_HEADER = ("# RESULTS — 模型輸入強化實驗結果\n\n"
                  "每一節由對應的 `research/run_*.py` / `research/price_baseline.py` 自動產生（標記之間的內容會被覆寫）。"
                  "結論為「沒有優勢」也是合格交付（PLAN.md §0 規則 6）。\n\n")

FEATURES = [("P1", "rev_5"), ("P2", "mom_12_1"), ("P3", "idio_mom"), ("P4", "neg_sigma")]
CONTROL_NAMES = ["neg_sigma", "mom_12_1", "rev_5"]          # PLAN.md section 6 residual-IC controls
HORIZONS = (5, 20)                                          # primary h = 5, secondary h = 20


# ============================================================================= features
def price_features(datas, market, panel: Panel) -> dict:
    """``{name: (T, S)}`` for P1-P4, column ``j`` from ``datas[j]`` (same order as ``panel.codes``)."""
    if len(datas) != panel.S:
        raise ValueError("datas and panel columns differ")
    out = {name: np.full((panel.T, panel.S), np.nan) for _, name in FEATURES}
    for j, d in enumerate(datas):
        fs = QF.build_features(d, market)
        col = {k: i for i, k in enumerate(fs.keys)}
        out["rev_5"][:, j] = align_series(panel, d, fs.raw[:, col["rev_5"]])
        out["mom_12_1"][:, j] = align_series(panel, d, fs.raw[:, col["mom_12_1"]])
        out["idio_mom"][:, j] = align_series(panel, d, fs.raw[:, col["idio_mom"]])
        out["neg_sigma"][:, j] = align_series(panel, d, -np.asarray(fs.sigma_ref, float))
    return out


# ============================================================================= run
def _compact(rep: dict, drop=("by_year",)) -> dict:
    return {k: v for k, v in rep.items() if k not in drop}


def run(datas, market=None, *, start: str = "2012-01-01", n: int = 150, out_json=OUT_JSON, results_md=RESULTS_MD,
        panel_out=None, meta=None, min_bars: int = 500) -> dict:
    """Evaluate P1-P4 on DEV for h = 5 and 20 and write the JSON (+ RESULTS.md section 1 when ``results_md`` is not None)."""
    panel = Panel.from_price_data(datas, start)
    if panel_out is not None:
        save_panel(panel, panel_out)
    mask = universe_mask(panel, n=n, min_bars=min_bars)
    feats = price_features(datas, market, panel)
    dev_rows = H.period_rows(panel.cal, "dev", 0)
    in_dev = np.flatnonzero(dev_rows & (mask.sum(axis=1) > 0))
    res = {"meta": {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": H._git_commit(),
        "n_stocks": int(panel.S), "n_days": int(panel.T), "panel_start": str(panel.cal[0]) if panel.T else None,
        "panel_end": str(panel.cal[-1]) if panel.T else None, "universe_n": int(n), "min_bars": int(min_bars),
        "dev_first_universe_day": str(panel.cal[in_dev[0]]) if len(in_dev) else None,
        "dev_mean_universe_size": H._f(mask[in_dev].sum(axis=1).mean()) if len(in_dev) else None,
        "has_market": market is not None and getattr(market, "n", 0) > 0, "period": "dev (2013-01-01..2022-12-31)",
        "oos_touched": False, **(meta or {})}, "features": {}}
    for fid, name in FEATURES:
        F = feats[name]
        ctrl = {k: feats[k] for k in CONTROL_NAMES if k != name}
        per_h = {}
        for h in HORIZONS:
            y = forward_open_return(panel, h)
            scr = H.screen_dev(F, y, mask, h, panel.cal)
            per_h[f"h{h}"] = {
                "screen": {"dev": scr["dev"], "dev_a": _compact(scr["dev_a"]), "dev_b": _compact(scr["dev_b"]),
                           "t_ok": scr["t_ok"], "halves_same_sign": scr["halves_same_sign"], "passes": scr["passes"]},
                "residual": H.residual_ic(F, ctrl, y, mask, h, "dev", panel.cal),
                "basket": H.quintile_report(F, panel, mask, h, "dev")}
        res["features"][name] = {"id": fid, **per_h}
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if results_md is not None:
        update_results_md(results_md, render_section(res))
    return res


# ============================================================================= RESULTS.md
def _pct(x, d=2):
    return "-" if x is None else f"{100.0 * x:.{d}f}%"


def _num(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def render_section(res: dict) -> str:
    """Markdown of section 1 (without the BEGIN/END markers)."""
    m = res["meta"]
    lines = [
        "## 1. 價格基準（P0，僅 DEV）", "",
        f"- 產生：{m['generated_at_utc']} UTC，git `{m['git_commit']}`；面板 {m['n_stocks']} 檔、{m['panel_start']} ~ {m['panel_end']}；"
        f"DEV = 2013-01-01 ~ 2022-12-31（每日宇宙取過去 60 日成交值中位數前 {m['universe_n']} 名、已有 ≥ {m['min_bars']} 根 K 棒）。"
        f"DEV 內首個有宇宙的交易日 {m['dev_first_universe_day']}，平均宇宙 {_num(m['dev_mean_universe_size'], 1)} 檔。**OOS 未使用。**",
        "- 標籤：`y_h[t] = ln(adjO[t+1+h] / adjO[t+1])`（開盤到開盤，t+1 開盤進場）；IC = 每日橫斷面 Spearman，t 為 Newey-West HAC"
        "（重用 `universe_bt._t_stats`）；DEV 最後 h+1 個決策日因標籤跨入 2023 而剔除。",
        "- 多頭 = 特徵最高 20% 等權（排除 t+1 開盤漲停鎖死者）、h 日重疊批次、`next_open` 成交、含手續費／滑價／證交稅；"
        "基準 = 同一候選集等權（同一套批次與成本）。超額為淨報酬差，年化 = 日均 × 245；年換手 = 日均 Σ|Δw| × 245（買賣合計）。",
        "- 殘差 IC：每日把特徵（橫斷面排名）對 `neg_sigma`、`mom_12_1`、`rev_5`（扣掉特徵自己）做線性（排名空間）OLS 取殘差再算 IC。",
        "- P1–P4 是**控制組**，沒有預先登記的方向；方向以 DEV 實測符號報告，不做 §5 篩選的「預期方向」檢查。"
        "P3 `idio_mom` 需要大盤" + ("" if m["has_market"] else "，**本次沒有取得大盤資料，P3 全為缺值**") + "。",
        "- 已知限制（PLAN.md §6）：候選池用了今天的資訊、已下市股票不在內 → 絕對報酬偏高；相對指標（IC、多頭減全體）受影響較小。", ""]
    for h in HORIZONS:
        key = f"h{h}"
        lines += [f"### h = {h}（{'主要' if h == 5 else '次要'}）", ""]
        rows = []
        for fid, name in FEATURES:
            f = res["features"][name][key]
            s, r, b = f["screen"], f["residual"], f["basket"]
            rows.append([f"{fid} `{name}`", _num(s["dev"]["mean"], 4), _num(s["dev"]["t"], 2), _num(s["dev_a"]["mean"], 4),
                         _num(s["dev_b"]["mean"], 4), "Y" if s["passes"] else "N", _num(r["mean"], 4), _num(r["t"], 2),
                         _pct(b["ann_net"]), _pct(b["ann_bench"]), _pct(b["ann_excess"]), _num(b["excess_t"], 2),
                         _num(b["ann_turnover"], 1), _num(b["sigma_ratio"], 2), _num(b["beta"], 2), _pct(b["mdd"], 1)])
        lines += [H.md_table(["特徵", "平均 IC", "HAC t", "前半 IC (2013–17)", "後半 IC (2018–22)", "過 §5 閘*", "殘差 IC", "殘差 t",
                              "多頭年化(淨)", "基準年化(淨)", "年化超額", "超額 t", "年換手", "σ 比", "β", "多頭 MDD"], rows), "",
                  "\\* 過 §5 閘 = |t| ≥ 3 且前後半同號（不含方向預期；僅供對照，控制組不參與篩選）。", ""]
        rows = []
        for fid, name in FEATURES:
            q = res["features"][name][key]["basket"]["quintile_y"]
            rows.append([f"{fid} `{name}`"] + [_pct(v, 3) for v in q["mean_y"]] + [_pct(q["q5_minus_q1"], 3), _num(q["q5_minus_q1_t"], 2)])
        lines += [f"分位平均 y（每 {h} 日開盤到開盤的對數報酬，毛，未扣成本；Q1 最低 .. Q5 最高）：", "",
                  H.md_table(["特徵", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5−Q1", "Q5−Q1 HAC t"], rows), ""]
    return "\n".join(lines).rstrip() + "\n"


def update_results_md(path, section: str) -> None:
    """Replace the BEGIN/END block of this section in ``path`` (created with a header if missing; appended if no block yet)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    block = f"{SECTION_BEGIN}\n{section}{SECTION_END}\n"
    text = p.read_text(encoding="utf-8") if p.exists() else RESULTS_HEADER
    a, b = text.find(SECTION_BEGIN), text.find(SECTION_END)
    if a != -1 and b != -1 and b > a:
        text = text[:a] + block + text[b + len(SECTION_END):].lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    p.write_text(text, encoding="utf-8")


# ============================================================================= synthetic data
def synthetic_price_data(n_stocks: int = 24, start: str = "2011-01-03", end: str = "2022-12-30", seed: int = 1):
    """Random-walk PriceData (no skill by construction) + a synthetic index -> ``(datas, market)``.  Weekdays only."""
    rng = np.random.default_rng(seed)
    d0, d1 = np.datetime64(start, "D"), np.datetime64(end, "D")
    days = np.arange(d0, d1 + 1)
    dates = days[((days.astype(np.int64) + 3) % 7) < 5]
    T = len(dates)

    def series(code, vol, kind="stock", with_volume=True):
        r = rng.normal(0.0003, vol, T)
        close = 50.0 * np.exp(np.cumsum(r))
        gap = rng.normal(0.0, 0.004, T)
        op = np.empty(T)
        op[0] = close[0]
        op[1:] = close[:-1] * np.exp(gap[1:])
        hi = np.maximum(op, close) * (1.0 + np.abs(rng.normal(0, 0.004, T)))
        lo = np.minimum(op, close) * (1.0 - np.abs(rng.normal(0, 0.004, T)))
        v = np.exp(rng.normal(13.0, 0.4, T)) if with_volume else np.zeros(T)
        return PriceData(symbol=code, name=code, market="INDEX" if kind == "index" else "TWSE", kind=kind, dates=dates.copy(), open=op,
                         high=hi, low=lo, close=close, adjclose=close.copy(), volume=v)

    datas = [series(f"S{j:03d}", float(rng.uniform(0.012, 0.03))) for j in range(n_stocks)]
    return datas, series("^TWII", 0.009, kind="index", with_volume=False)


# ============================================================================= CLI
def _market(offline: bool):
    """TAIEX for P3: the cache first; if it is not cached, ONE download of the index (cached afterwards) even in offline mode --
    the stock prices stay cache-only.  Without it P3 is missing and the report says so."""
    m = None
    try:
        m = D.peek_price_data(D.INDEX_SYMBOL)
        if m is None or not offline:
            m = D.get_market_data()
    except D.DataError as exc:
        log.warning("market data unavailable: %s -- P3 idio_mom will be missing", exc)
    return D.completed(m) if m is not None else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0 price-baseline features on DEV")
    ap.add_argument("--codes", help="comma-separated codes (default: pool300.json)")
    ap.add_argument("--limit", type=int, help="use only the first N codes of the pool")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--online", action="store_true", help="allow backend.data to download/refresh (default: cache only)")
    ap.add_argument("--synthetic", action="store_true", help="random-walk smoke test (writes only research/out/*_synthetic.json)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--results-md", default=None)
    ap.add_argument("--no-results", action="store_true", help="do not touch RESULTS.md")
    ap.add_argument("--no-save-panel", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.synthetic:
        datas, market = synthetic_price_data()
        out = a.out_json or str(OUT_JSON.with_name("price_baseline_synthetic.json"))
        res = run(datas, market, start=a.start, out_json=out, results_md=a.results_md if not a.no_results else None,
                  meta={"synthetic": True})
    else:
        codes = a.codes.split(",") if a.codes else [r["code"] for r in load_pool()]
        if a.limit:
            codes = codes[:a.limit]
        datas = load_datas(codes, offline=not a.online)
        if not datas:
            log.error("no price data loaded")
            return 2
        log.info("loaded %d/%d stocks", len(datas), len(codes))
        market = _market(not a.online)
        res = run(datas, market, start=a.start, out_json=a.out_json or OUT_JSON,
                  results_md=None if a.no_results else (a.results_md or RESULTS_MD),
                  panel_out=None if a.no_save_panel else panel_path(), meta={"n_requested": len(codes)})
    for fid, name in FEATURES:
        for h in HORIZONS:
            f = res["features"][name][f"h{h}"]
            log.info("%s %-9s h=%-2d IC=%s t=%s | resid t=%s | excess=%s t=%s", fid, name, h, _num(f["screen"]["dev"]["mean"], 4),
                     _num(f["screen"]["dev"]["t"], 2), _num(f["residual"]["t"], 2), _pct(f["basket"]["ann_excess"]),
                     _num(f["basket"]["excess_t"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
