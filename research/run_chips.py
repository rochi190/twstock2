"""P1c: evaluate the 12 chip / valuation features C1-C9, V1-V3 on DEV (PLAN.md sections 5-7).  DEV only; OOS is never used.

``python -m research.run_chips``            real cached prices + whatever FinMind files exist (read-only, never downloads)
``python -m research.run_chips --synthetic`` random-walk prices + random chip data (smoke test; no RESULTS.md write by default)

For every feature and h in (5, 20): HAC-t rank IC on DEV, the DEV halves, the section-5 screen (|t| >= 3, same sign in both
halves, sign = expected direction; "?" = any sign), residual IC against ``neg_sigma``/``mom_12_1``/``rev_5``, and the long
basket (top 20 %).  Output: ``research/out/chips_dev.json`` and section 2 of ``docs/research/RESULTS.md`` (between
``<!-- BEGIN chips -->`` / ``<!-- END chips -->``).

Partial data: stocks whose FinMind files are missing have NaN features and drop out of the cross-section (``harness`` ignores
non-finite entries).  The IC/basket numbers are then computed on the covered subset of the universe; the report lists the
coverage of every feature so this is visible.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

from . import feat_chips as FC
from . import harness as H
from . import price_baseline as PB
from .panel import Panel, forward_open_return, load_datas, load_pool, universe_mask

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "research" / "out" / "chips_dev.json"
RESULTS_MD = REPO_ROOT / "docs" / "research" / "RESULTS.md"
SECTION_BEGIN = "<!-- BEGIN chips -->"
SECTION_END = "<!-- END chips -->"
HORIZONS = PB.HORIZONS
CONTROL_NAMES = PB.CONTROL_NAMES


# ============================================================================= run
def run(panel: Panel, feats: dict, price_ctrl: dict, *, n: int = 150, min_bars: int = 500, out_json=OUT_JSON, results_md=RESULTS_MD,
        meta=None, mask=None) -> dict:
    """Evaluate ``feats`` (``feat_chips.chip_features`` output) on DEV; ``price_ctrl`` holds ``neg_sigma``, ``mom_12_1``, ``rev_5``."""
    mask = universe_mask(panel, n=n, min_bars=min_bars) if mask is None else mask
    dev_rows = H.period_rows(panel.cal, "dev", 0)
    in_dev = np.flatnonzero(dev_rows & (mask.sum(axis=1) > 0))
    res = {"meta": {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": H._git_commit(),
        "n_stocks": int(panel.S), "n_days": int(panel.T), "panel_start": str(panel.cal[0]) if panel.T else None,
        "panel_end": str(panel.cal[-1]) if panel.T else None, "universe_n": int(n), "min_bars": int(min_bars),
        "dev_first_universe_day": str(panel.cal[in_dev[0]]) if len(in_dev) else None,
        "dev_mean_universe_size": H._f(mask[in_dev].sum(axis=1).mean()) if len(in_dev) else None,
        "period": "dev (2013-01-01..2022-12-31)", "oos_touched": False, **(meta or {})},
        "coverage": FC.coverage(feats, panel, mask), "features": {}}
    for fid, name, sign in FC.FEATURES:
        F = feats[name]
        ctrl = {k: price_ctrl[k] for k in CONTROL_NAMES}
        per_h = {}
        for h in HORIZONS:
            y = forward_open_return(panel, h)
            scr = H.screen_dev(F, y, mask, h, panel.cal, expected_sign=sign)
            m = scr["dev"]["mean"]
            # basket direction: the pre-registered sign; for "?" features the sign of the DEV mean IC (descriptive, DEV only)
            direction = sign if sign else (1 if (m is None or m >= 0) else -1)
            per_h[f"h{h}"] = {
                "screen": {"dev": scr["dev"], "dev_a": PB._compact(scr["dev_a"]), "dev_b": PB._compact(scr["dev_b"]),
                           "t_ok": scr["t_ok"], "halves_same_sign": scr["halves_same_sign"],
                           "sign_as_expected": scr["sign_as_expected"], "passes": scr["passes"]},
                "residual": H.residual_ic(F, ctrl, y, mask, h, "dev", panel.cal),
                "basket": H.quintile_report(F, panel, mask, h, "dev", direction=direction), "basket_direction": int(direction)}
        res["features"][name] = {"id": fid, "expected_sign": int(sign), **per_h}
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if results_md is not None:
        update_results_md(results_md, render_section(res))
    return res


# ============================================================================= RESULTS.md
_pct, _num = PB._pct, PB._num
_SIGN = {1: "+", -1: "−", 0: "?"}


def render_section(res: dict) -> str:
    """Markdown of section 2 (without the BEGIN/END markers)."""
    m = res["meta"]
    lines = [
        "## 2. 籌碼與評價（P1，僅 DEV）", "",
        f"- 產生：{m['generated_at_utc']} UTC，git `{m['git_commit']}`；面板 {m['n_stocks']} 檔、{m['panel_start']} ~ {m['panel_end']}；"
        f"DEV = 2013-01-01 ~ 2022-12-31，宇宙同第 1 節（前 {m['universe_n']} 名、≥ {m['min_bars']} 根 K 棒），"
        f"DEV 平均宇宙 {_num(m['dev_mean_universe_size'], 1)} 檔。**OOS 未使用。**",
        "- 特徵 C1–C9、V1–V3 定義見 PLAN.md §5 與 `research/feat_chips.py`（點時間：三大法人／融資券／PER 用 t 日值，外資持股比多落後 1 日；"
        "水準類資料往過去 ffill 最多 5 列，流量類不補值；視窗內觀測 < 80% → NaN）。評估方法同第 1 節（HAC t、前後半、殘差 IC、多頭前 20%）。",
        "- **篩選判定**（PLAN.md §5）：DEV 平均 rank IC 的 HAC |t| ≥ 3.0 且前後半同號且符號與預期方向一致（預期 `?` 者符號不限）。"
        "V1–V3 在 PLAN 中屬評價控制變數、未登記方向，視為 `?`。",
        "- 多頭方向：預期方向為 + / − 者依預期；`?` 者取 DEV 平均 IC 的符號（僅描述用）。",
        "- **覆蓋率警告**：FinMind 資料缺檔的股票其特徵為 NaN，IC／多頭只在「有資料」的子集上計算（見下表覆蓋率）；"
        "覆蓋不全時結果不代表完整宇宙。"
        + (f" 本次 meta：{m['coverage_note']}" if m.get("coverage_note") else ""), "",
        "### 覆蓋率", ""]
    rows = []
    for fid, name, sign in FC.FEATURES:
        c = res["coverage"][name]
        rows.append([f"{fid} `{name}`", f"{c['n_stocks']}/{c['n_total']}", c["first"], c["last"], _pct(c.get("universe_share"), 1)])
    lines += [H.md_table(["特徵", "有資料檔數", "首個有值日", "最後有值日", "宇宙日平均覆蓋"], rows), ""]
    for h in HORIZONS:
        key = f"h{h}"
        lines += [f"### h = {h}（{'主要' if h == 5 else '次要'}）", ""]
        rows = []
        for fid, name, sign in FC.FEATURES:
            f = res["features"][name][key]
            s, r, b = f["screen"], f["residual"], f["basket"]
            rows.append([f"{fid} `{name}`", _SIGN[sign], _num(s["dev"]["mean"], 4), _num(s["dev"]["t"], 2), _num(s["dev_a"]["mean"], 4),
                         _num(s["dev_b"]["mean"], 4), "Y" if s["passes"] else "N", _num(r["mean"], 4), _num(r["t"], 2),
                         _num(s["dev"].get("avg_xs"), 0), _pct(b["ann_net"]), _pct(b["ann_bench"]), _pct(b["ann_excess"]),
                         _num(b["excess_t"], 2), _num(b["ann_turnover"], 1), _num(b["sigma_ratio"], 2), _num(b["beta"], 2),
                         _pct(b["mdd"], 1)])
        lines += [H.md_table(["特徵", "預期", "平均 IC", "HAC t", "前半 IC (2013–17)", "後半 IC (2018–22)", "過 §5 篩選", "殘差 IC", "殘差 t",
                              "平均截面檔數", "多頭年化(淨)", "基準年化(淨)", "年化超額", "超額 t", "年換手", "σ 比", "β", "多頭 MDD"], rows), "",
                  "多頭 = 方向調整後最高 20%；基準 = 同一候選集（有該特徵值者）等權。", ""]
        rows = []
        for fid, name, sign in FC.FEATURES:
            q = res["features"][name][key]["basket"]["quintile_y"]
            rows.append([f"{fid} `{name}`"] + [_pct(v, 3) for v in q["mean_y"]] + [_pct(q["q5_minus_q1"], 3), _num(q["q5_minus_q1_t"], 2)])
        lines += [f"分位平均 y（每 {h} 日開盤到開盤對數報酬，毛，未扣成本；Q1 最低 .. Q5 最高，依「多頭方向 × 特徵值」排序）：", "",
                  H.md_table(["特徵", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5−Q1", "Q5−Q1 HAC t"], rows), ""]
    passed = [f"{res['features'][nm]['id']} `{nm}`" for _, nm, _ in FC.FEATURES if res["features"][nm]["h5"]["screen"]["passes"]]
    lines += ["### §5 篩選結論（h = 5）", "",
              "通過：" + ("、".join(passed) if passed else "**無**（沒有優勢也是合格結果）") + "。", ""]
    return "\n".join(lines).rstrip() + "\n"


def update_results_md(path, section: str) -> None:
    """Replace this section's BEGIN/END block in ``path`` (header created if missing; appended if the block is new)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    block = f"{SECTION_BEGIN}\n{section}{SECTION_END}\n"
    text = p.read_text(encoding="utf-8") if p.exists() else PB.RESULTS_HEADER
    a, b = text.find(SECTION_BEGIN), text.find(SECTION_END)
    if a != -1 and b != -1 and b > a:
        text = text[:a] + block + text[b + len(SECTION_END):].lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    p.write_text(text, encoding="utf-8")


# ============================================================================= synthetic data
def synthetic_raw(panel: Panel, seed: int = 3, frac: float = 0.8) -> dict:
    """Random (no-skill) FinMind-shaped parsed data for ``frac`` of the stocks (the rest get no files)."""
    rng = np.random.default_rng(seed)
    out = {}
    dates = [str(d) for d in panel.cal]
    for j, c in enumerate(panel.codes):
        if rng.random() > frac:
            continue
        inst, margin, hold, per = {}, {}, {}, {}
        mb, sb, hr = 1000.0, 200.0, 30.0
        for t, d in enumerate(dates):
            v = panel.volume[t, j]
            if not np.isfinite(v) or d < "2012-05-02":
                continue
            inst[d] = {"foreign_net": float(rng.normal(0, 0.02) * v), "trust_net": float(rng.normal(0, 0.005) * v),
                       "dealer_self_net": float(rng.normal(0, 0.003) * v) if d >= "2014-12-01" else None}
            mb = max(0.0, mb + rng.normal(0, 20))
            sb = max(0.0, sb + rng.normal(0, 5))
            hr = float(np.clip(hr + rng.normal(0, 0.05), 0, 100))
            margin[d] = {"margin_balance": mb * 1000.0, "short_balance": sb * 1000.0}
            hold[d] = {"foreign_ratio": hr}
            per[d] = {"per": float(rng.uniform(-5, 40)), "pbr": float(rng.uniform(0.5, 5)), "dividend_yield": float(rng.uniform(0, 8))}
        out[str(c)] = {"inst": inst, "margin": margin, "hold": hold, "per": per}
    return out


# ============================================================================= CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1c chip / valuation features on DEV")
    ap.add_argument("--codes", help="comma-separated codes (default: pool300.json)")
    ap.add_argument("--limit", type=int, help="use only the first N codes of the pool")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--online", action="store_true", help="allow backend.data to refresh PRICES (never downloads FinMind)")
    ap.add_argument("--synthetic", action="store_true", help="random-walk smoke test (no RESULTS.md unless --results-md)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--results-md", default=None)
    ap.add_argument("--no-results", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.synthetic:
        datas, market = PB.synthetic_price_data()
        panel = Panel.from_price_data(datas, a.start)
        raw = synthetic_raw(panel)
        meta = {"synthetic": True}
        out = a.out_json or str(OUT_JSON.with_name("chips_synthetic.json"))
        md = a.results_md if not a.no_results else None
    else:
        codes = a.codes.split(",") if a.codes else [r["code"] for r in load_pool()]
        if a.limit:
            codes = codes[:a.limit]
        datas = load_datas(codes, offline=not a.online)
        if not datas:
            log.error("no price data loaded")
            return 2
        market = PB._market(not a.online)
        panel = Panel.from_price_data(datas, a.start)
        raw = FC.load_raw(panel.codes)
        meta = {"n_requested": len(codes)}
        out = a.out_json or OUT_JSON
        md = None if a.no_results else (a.results_md or RESULTS_MD)
    n_inst = sum(1 for r in raw.values() if r.get("inst"))
    meta["coverage_note"] = (f"{len(panel.codes)} 檔價格面板中，inst {n_inst}、margin {sum(1 for r in raw.values() if r.get('margin'))}、"
                             f"hold {sum(1 for r in raw.values() if r.get('hold'))}、per {sum(1 for r in raw.values() if r.get('per'))} 檔有資料檔。")
    log.info(meta["coverage_note"])
    feats = FC.chip_features(panel, raw)
    price_ctrl = PB.price_features(datas, market, panel)
    res = run(panel, feats, price_ctrl, out_json=out, results_md=md, meta=meta)
    for fid, name, _ in FC.FEATURES:
        c = res["coverage"][name]
        for h in HORIZONS:
            f = res["features"][name][f"h{h}"]
            log.info("%s %-15s h=%-2d cov=%3d/%d IC=%s t=%s pass=%s | resid t=%s | excess=%s t=%s", fid, name, h, c["n_stocks"], c["n_total"],
                     _num(f["screen"]["dev"]["mean"], 4), _num(f["screen"]["dev"]["t"], 2), "Y" if f["screen"]["passes"] else "N",
                     _num(f["residual"]["t"], 2), _pct(f["basket"]["ann_excess"]), _num(f["basket"]["excess_t"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
