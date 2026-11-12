"""P2 monthly revenue on DEV (PLAN.md section 5 R1-R4, section 7 P2): ``python -m research.run_revenue``.

For R1-R4 (``research.feat_revenue``) and h = 5 (primary) / 20 (secondary):
  * DEV rank-IC report with HAC t, first/second half (2013-17 / 2018-22), residual IC against ``neg_sigma``, ``mom_12_1``,
    ``rev_5`` (price controls computed from the cached prices exactly as in ``research.price_baseline``), long-basket report,
  * the PLAN section 5 screen: |HAC t| >= 3.0, same IC sign in both halves, sign equal to the expected (+) direction.
Plus an EVENT-TIME table: cumulative excess return over the first 1..20 trading days after the revenue availability day, by
R1 quintile.  Only DEV is used (the harness refuses everything else; this module never passes ``period="oos"``).

Output: ``research/out/revenue_dev.json`` and section 3 of ``docs/research/RESULTS.md`` (the block between the
``<!-- BEGIN revenue -->`` / ``<!-- END revenue -->`` markers is replaced on every run, so reruns are idempotent).

Event-study definition (so the numbers can be audited)
------------------------------------------------------
For a revenue month M let ``a`` be its availability row (``feat_revenue.availability_row``: the 2nd trading day strictly after
the 10th of M+1).  The decision is taken after the close of ``a`` (R1 of month M is first usable on day ``a``), the entry is the
OPEN of day ``a+1`` (PLAN rule 3: never the close of ``a``).  Event day ``k`` (k = 1..20) is the holding day from the open of
``a+k`` to the open of ``a+k+1``; the cumulative return through event day k is therefore
``y_k[a, i] = ln(adjO[a+1+k] / adjO[a+1])`` = ``panel.forward_open_return(panel, k)[a, i]`` (the same label as the IC study).
Excess = ``y_k[a, i]`` minus the equal-weight mean of ``y_k[a, .]`` over the whole universe of day ``a`` (``universe_mask[a]``,
finite ``y_k``) -- the same entry and horizon.  Cohort of an event = universe members that own a revenue row for month M itself
(a carried-forward older month does not count as an event) with a finite R1 and a finite ``y_20``.  R1 quintiles are cut inside
the cohort of that month (Q1 lowest .. Q5 highest, rank ties averaged).  Only events whose whole 20-day label window ends inside
DEV are used (``harness.period_rows(cal, "dev", 20)``).  The Q5-Q1 t-statistic is the plain t of the monthly spread series
(events are one per month; the 20-day windows do not overlap across months except at the very edge).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from backend import data as D
from backend.quant import features as QF

from . import feat_revenue as FR
from . import harness as H
from .panel import Panel, align_series, forward_open_return, load_datas, load_panel, panel_path, universe_mask
from .price_baseline import RESULTS_HEADER

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "research" / "out" / "revenue_dev.json"
RESULTS_MD = REPO_ROOT / "docs" / "research" / "RESULTS.md"
SECTION_BEGIN = "<!-- BEGIN revenue -->"
SECTION_END = "<!-- END revenue -->"

HORIZONS = (5, 20)
CONTROL_NAMES = ["neg_sigma", "mom_12_1", "rev_5"]       # PLAN section 6
KMAX = 20
MIN_COHORT = 25                                          # events with fewer usable stocks are skipped
TABLE_KS = (1, 2, 3, 5, 10, 15, 20)                      # columns shown in RESULTS.md (all 20 are in the JSON)


# ============================================================================= controls
def price_controls(datas, panel: Panel) -> dict:
    """``{neg_sigma, mom_12_1, rev_5}`` ``(T, S)`` from the stocks' own bars (as ``price_baseline.price_features``; no market
    needed).  A panel column without a ``PriceData`` object stays NaN."""
    by_sym = {str(d.symbol): d for d in datas}
    out = {k: np.full((panel.T, panel.S), np.nan) for k in CONTROL_NAMES}
    for j, code in enumerate(panel.codes):
        d = by_sym.get(str(code))
        if d is None:
            continue
        fs = QF.build_features(d, None)
        col = {k: i for i, k in enumerate(fs.keys)}
        out["rev_5"][:, j] = align_series(panel, d, fs.raw[:, col["rev_5"]])
        out["mom_12_1"][:, j] = align_series(panel, d, fs.raw[:, col["mom_12_1"]])
        out["neg_sigma"][:, j] = align_series(panel, d, -np.asarray(fs.sigma_ref, float))
    return out


# ============================================================================= event study
def event_months(cal, rows_by_code: dict):
    """Sorted ``[(year, month, availability_row)]`` over the revenue months present in any file (rows beyond the calendar end are
    dropped)."""
    months = set()
    for rows in rows_by_code.values():
        for r in rows or ():
            months.add((int(r["revenue_year"]), int(r["revenue_month"])))
    out = []
    for (y, m) in sorted(months):
        a = FR.availability_row(y, m, cal)
        if a is not None:
            out.append((y, m, a))
    return out


def event_study(panel: Panel, mask, rows_by_code: dict, kmax: int = KMAX, min_cohort: int = MIN_COHORT) -> dict:
    """Cumulative excess return by R1 quintile after the availability day (definition: module docstring).  DEV only."""
    cal = panel.cal
    Y = np.stack([forward_open_return(panel, k) for k in range(1, kmax + 1)])           # (K, T, S)
    dev_ok = H.period_rows(cal, "dev", kmax)
    mask = np.asarray(mask, bool)
    cohort = []                                          # (y, m, a, idx of members, quintile per member)
    tables = [FR.monthly_feature_table(rows_by_code.get(str(code)) or []) for code in panel.codes]
    for (y, m, a) in event_months(cal, rows_by_code):
        if not dev_ok[a]:
            continue
        r1 = np.full(panel.S, np.nan)
        for j, tab in enumerate(tables):
            if (y, m) in tab:                         # the stock owns a row for month M itself
                r1[j] = tab[(y, m)][0]
        ok = mask[a] & np.isfinite(r1) & np.isfinite(Y[kmax - 1, a])
        n = int(ok.sum())
        if n < min_cohort:
            continue
        idx = np.flatnonzero(ok)
        rk = rankdata(r1[idx])
        q = np.minimum(np.floor((rk - 1.0) * 5.0 / n).astype(int), 4)
        cohort.append((y, m, a, idx, q))
    E = len(cohort)
    cum = np.full((E, 5, kmax), np.nan)                  # mean excess log return of each quintile, per event and k
    sizes = np.zeros((E, 5))
    for e, (y, m, a, idx, q) in enumerate(cohort):
        for k in range(kmax):
            yk = Y[k, a]
            uni = mask[a] & np.isfinite(yk)
            bench = yk[uni].mean()
            ex = yk[idx] - bench
            for b in range(5):
                sel = q == b
                if sel.any():
                    cum[e, b, k] = np.nanmean(ex[sel]) if np.isfinite(ex[sel]).any() else np.nan
        sizes[e] = [(q == b).sum() for b in range(5)]
    res = {"n_events": E, "kmax": kmax, "first_event": f"{cohort[0][0]}-{cohort[0][1]:02d}" if E else None,
           "last_event": f"{cohort[-1][0]}-{cohort[-1][1]:02d}" if E else None,
           "mean_cohort_size": H._f(sizes.sum(axis=1).mean()) if E else None, "quantiles": {}}
    for b in range(5):
        res["quantiles"][f"Q{b + 1}"] = [H._f(np.nanmean(cum[:, b, k])) if E and np.isfinite(cum[:, b, k]).any() else None
                                         for k in range(kmax)]
    spread = cum[:, 4, :] - cum[:, 0, :] if E else np.zeros((0, kmax))
    sp_mean, sp_t = [], []
    for k in range(kmax):
        s = spread[:, k][np.isfinite(spread[:, k])] if E else np.array([])
        if len(s) >= 3 and s.std(ddof=1) > 0:
            sp_mean.append(H._f(s.mean()))
            sp_t.append(H._f(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))))
        else:
            sp_mean.append(H._f(s.mean()) if len(s) else None)
            sp_t.append(None)
    res["q5_minus_q1"], res["q5_minus_q1_t"] = sp_mean, sp_t
    return res


# ============================================================================= run
def _compact(rep: dict, drop=("by_year",)) -> dict:
    return {k: v for k, v in rep.items() if k not in drop}


def run(panel: Panel, rows_by_code: dict, controls: dict, *, n: int = 150, min_bars: int = 500, out_json=OUT_JSON,
        results_md=RESULTS_MD, meta=None, mask=None) -> dict:
    """Evaluate R1-R4 on DEV and write the JSON (+ RESULTS.md section 3 when ``results_md`` is not None).

    ``rows_by_code``: ``{code: parsed revenue rows | None}`` (None / absent -> NaN feature column).
    """
    if mask is None:
        mask = universe_mask(panel, n=n, min_bars=min_bars)
    feats, cov = FR.revenue_features(list(panel.codes), panel.cal, loader=lambda c: rows_by_code.get(c))
    dev_rows = H.period_rows(panel.cal, "dev", 0)
    in_dev = np.flatnonzero(dev_rows & (mask.sum(axis=1) > 0))
    have = np.isin(panel.codes, cov["codes_with_file"])
    res = {"meta": {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": H._git_commit(),
        "n_stocks": int(panel.S), "n_days": int(panel.T), "panel_start": str(panel.cal[0]) if panel.T else None,
        "panel_end": str(panel.cal[-1]) if panel.T else None, "universe_n": int(n), "min_bars": int(min_bars),
        "n_with_revenue_file": cov["n_with_file"], "period": "dev (2013-01-01..2022-12-31)", "oos_touched": False,
        "dev_mean_universe_size": H._f(mask[in_dev].sum(axis=1).mean()) if len(in_dev) else None,
        # share of the day-t universe that owns a file (members without a file are NaN and drop out of every statistic)
        "dev_file_coverage": H._f((mask[in_dev] & have[None, :]).sum() / max(1, mask[in_dev].sum())) if len(in_dev) else None,
        **(meta or {})}, "features": {}}
    for name in FR.NAMES:
        F = feats[name]
        fin = np.isfinite(F) & mask
        res["meta"].setdefault("dev_feature_coverage", {})[name] = (
            H._f(fin[in_dev].sum() / max(1, mask[in_dev].sum())) if len(in_dev) else None)
        ctrl = {k: controls[k] for k in CONTROL_NAMES}
        per_h = {}
        for h in HORIZONS:
            y = forward_open_return(panel, h)
            scr = H.screen_dev(F, y, mask, h, panel.cal, expected_sign=FR.EXPECTED_SIGN[name])
            per_h[f"h{h}"] = {
                "screen": {"dev": scr["dev"], "dev_a": _compact(scr["dev_a"]), "dev_b": _compact(scr["dev_b"]),
                           "t_ok": scr["t_ok"], "halves_same_sign": scr["halves_same_sign"],
                           "sign_as_expected": scr["sign_as_expected"], "passes": scr["passes"]},
                "residual": H.residual_ic(F, ctrl, y, mask, h, "dev", panel.cal),
                "basket": H.quintile_report(F, panel, mask, h, "dev")}
        res["features"][name] = {"id": FR.IDS[name], "expected_sign": FR.EXPECTED_SIGN[name], **per_h}
    res["event_study"] = event_study(panel, mask, rows_by_code)
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
    """Markdown of section 3 (without the BEGIN/END markers)."""
    m = res["meta"]
    cov = m.get("dev_feature_coverage", {})
    lines = [
        "## 3. 月營收（P2，僅 DEV）", "",
        f"- 產生：{m['generated_at_utc']} UTC，git `{m['git_commit']}`；面板 {m['n_stocks']} 檔（其中 **{m['n_with_revenue_file']} 檔有營收檔**，"
        f"DEV 期間宇宙成員有營收檔的比例 {_pct(m['dev_file_coverage'], 1)}；沒有檔案的股票特徵為 NaN、不進任何統計）、"
        f"{m['panel_start']} ~ {m['panel_end']}；DEV = 2013-01-01 ~ 2022-12-31，每日宇宙取前 {m['universe_n']} 名，平均 "
        f"{_num(m['dev_mean_universe_size'], 1)} 檔。**OOS 未使用。**",
        "- 可得日（PLAN §3，保守）：M 月營收自「M+1 月 10 日之後的第 2 個交易日」起可用；不使用 `create_time`；可得日之前沿用上一個月的值。"
        "R1 截斷於 [−0.9, 3]；R2 用同樣截斷後的 YoY；R2–R4 需要窗口內每個月都有資料，否則 NaN；R4 = 1／0。",
        "- 標籤、IC、HAC、殘差 IC、多頭籃子的定義與第 1 節相同（`y_h[t] = ln(adjO[t+1+h]/adjO[t+1])`，t+1 開盤進場）。"
        "§5 篩選 = |HAC t| ≥ 3.0 **且** 前後半（2013–17／2018–22）IC 同號 **且** 符號與預期方向（+）一致；主要 h = 5，次要 h = 20。",
        "- R4 為二元特徵，多頭籃子在同分者之間以欄位順序決定（確定性但任意），僅供參考；看 IC 與殘差 IC 較有意義。",
        "- 已知限制（PLAN §6）：候選池用了今天的資訊、已下市股票不在內 → 絕對報酬偏高；相對指標受影響較小。", "",
        "DEV 內特徵覆蓋率（宇宙成員中特徵有值的比例）：" + "、".join(f"`{k}` {_pct(v, 1)}" for k, v in cov.items()), ""]
    for h in HORIZONS:
        key = f"h{h}"
        lines += [f"### h = {h}（{'主要' if h == 5 else '次要'}）", ""]
        rows = []
        for name in FR.NAMES:
            f = res["features"][name]
            s, r, b = f[key]["screen"], f[key]["residual"], f[key]["basket"]
            rows.append([f"{f['id']} `{name}`", _num(s["dev"]["mean"], 4), _num(s["dev"]["t"], 2), _num(s["dev_a"]["mean"], 4),
                         _num(s["dev_b"]["mean"], 4), "Y" if s["halves_same_sign"] else "N", "Y" if s["sign_as_expected"] else "N",
                         "**Y**" if s["passes"] else "N", _num(r["mean"], 4), _num(r["t"], 2), _pct(b["ann_net"]),
                         _pct(b["ann_bench"]), _pct(b["ann_excess"]), _num(b["excess_t"], 2), _num(b["ann_turnover"], 1),
                         _num(b["sigma_ratio"], 2), _num(b["beta"], 2), _pct(b["mdd"], 1)])
        lines += [H.md_table(["特徵", "平均 IC", "HAC t", "前半 IC", "後半 IC", "前後半同號", "符號如預期(+)", "過 §5 篩選", "殘差 IC",
                              "殘差 t", "多頭年化(淨)", "基準年化(淨)", "年化超額", "超額 t", "年換手", "σ 比", "β", "多頭 MDD"], rows), "",
                  f"分位平均 y（每 {h} 日開盤到開盤的對數報酬，毛，未扣成本；Q1 最低 .. Q5 最高）：", ""]
        rows = []
        for name in FR.NAMES:
            q = res["features"][name][key]["basket"]["quintile_y"]
            rows.append([f"{res['features'][name]['id']} `{name}`"] + [_pct(v, 3) for v in q["mean_y"]]
                        + [_pct(q["q5_minus_q1"], 3), _num(q["q5_minus_q1_t"], 2)])
        lines += [H.md_table(["特徵", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5−Q1", "Q5−Q1 HAC t"], rows), ""]
    ev = res["event_study"]
    lines += ["### 事件時間：可得日後累積超額報酬（依 R1 分五組）", "",
              f"定義見 `research/run_revenue.py` 開頭：決策在可得日 a 收盤後，**a+1 開盤進場**；事件日 k = 從 a+k 開盤到 a+k+1 開盤，"
              f"累積到第 k 日 = `ln(O[a+1+k]/O[a+1])`；超額 = 減去同日宇宙等權的同期報酬（對數差）。五組在每月的事件群內依 R1 切分。"
              f"事件數 {ev['n_events']}（{ev['first_event']} ~ {ev['last_event']} 營收月；每月平均群大小 {_num(ev['mean_cohort_size'], 1)} 檔）。"
              "Q5−Q1 t = 逐月價差序列的簡單 t。", ""]
    ks = [k for k in TABLE_KS if k <= ev["kmax"]]
    rows = [[q] + [_pct(ev["quantiles"][q][k - 1], 3) for k in ks] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")]
    rows.append(["Q5−Q1"] + [_pct(ev["q5_minus_q1"][k - 1], 3) for k in ks])
    rows.append(["Q5−Q1 t"] + [_num(ev["q5_minus_q1_t"][k - 1], 2) for k in ks])
    lines += [H.md_table(["組別＼可得日後第 k 日"] + [str(k) for k in ks], rows), ""]
    return "\n".join(lines).rstrip() + "\n"


def update_results_md(path, section: str, begin: str = SECTION_BEGIN, end: str = SECTION_END) -> None:
    """Replace the BEGIN/END block of this section in ``path`` (created with a header if missing; appended if no block yet)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    block = f"{begin}\n{section}{end}\n"
    text = p.read_text(encoding="utf-8") if p.exists() else RESULTS_HEADER
    a, b = text.find(begin), text.find(end)
    if a != -1 and b != -1 and b > a:
        text = text[:a] + block + text[b + len(end):].lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    p.write_text(text, encoding="utf-8")


# ============================================================================= CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2 monthly revenue R1-R4 on DEV (cache only, no network)")
    ap.add_argument("--codes", help="comma-separated codes: restrict the panel to these columns (smoke test; changes the universe)")
    ap.add_argument("--panel", default=None, help="panel npz (default data_cache/research/panel_v1.npz)")
    ap.add_argument("--universe-n", type=int, default=150)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--results-md", default=None)
    ap.add_argument("--no-results", action="store_true", help="do not touch RESULTS.md")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = load_panel(a.panel or panel_path())
    if a.codes:
        want = [c.strip() for c in a.codes.split(",") if c.strip()]
        keep = np.flatnonzero(np.isin(panel.codes, want))
        if not len(keep):
            log.error("none of --codes is in the panel")
            return 2
        panel = Panel(panel.cal, panel.codes[keep], panel.adj_open[:, keep], panel.adj_close[:, keep], panel.volume[:, keep],
                      panel.value[:, keep], panel.has_bar[:, keep], panel.bars_before[keep], panel.kinds[keep])
    rows_by_code = {str(c): FR.load_rows(str(c)) for c in panel.codes}
    n_files = sum(v is not None for v in rows_by_code.values())
    log.info("panel %d stocks x %d days; revenue files: %d/%d", panel.S, panel.T, n_files, panel.S)
    if n_files == 0:
        log.error("no revenue file found under %s", D.cache_dir() / "alt" / "finmind")
        return 2
    datas = load_datas([str(c) for c in panel.codes], offline=True)
    controls = price_controls(datas, panel)
    res = run(panel, rows_by_code, controls, n=a.universe_n, out_json=a.out_json or OUT_JSON,
              results_md=None if a.no_results else (a.results_md or RESULTS_MD))
    log.info("DEV file coverage of the universe: %s", _pct(res["meta"]["dev_file_coverage"], 1))
    for name in FR.NAMES:
        for h in HORIZONS:
            f = res["features"][name][f"h{h}"]
            log.info("%s %-13s h=%-2d IC=%s t=%s pass=%s | resid t=%s | excess=%s t=%s", res["features"][name]["id"], name, h,
                     _num(f["screen"]["dev"]["mean"], 4), _num(f["screen"]["dev"]["t"], 2), f["screen"]["passes"],
                     _num(f["residual"]["t"], 2), _pct(f["basket"]["ann_excess"]), _num(f["basket"]["excess_t"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
