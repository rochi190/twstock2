"""Deterministic generator for the frontend test fixtures (tests/fixtures/*.json).

The fixtures are built STRICTLY from the DESIGN.md section 6 schemas with realistic values (2330 near
NT$2385, 20-point fan arrays, 10-bin PIT histogram, <=600-point curves).  They contain no look-ahead or
model logic -- they are plain shape/value fixtures for the renderer.  Pure stdlib so it runs anywhere:

    python3 tests/fixtures/make_front_fixtures.py
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

OUT = Path(__file__).resolve().parent
ND = NormalDist()
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
DISCLAIMER = ("本工具為統計模型之研究用途，非投資建議。預測以機率表示，過去表現不代表未來結果，"
              "投資人應自行評估風險並自負盈虧。")


def tick(p: float, kind: str = "stock") -> float:
    if kind == "etf":
        return 0.01 if p < 50 else 0.05
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.1
    if p < 500:
        return 0.5
    if p < 1000:
        return 1.0
    return 5.0


def snap(p: float, kind: str = "stock") -> float:
    t = tick(p, kind)
    return round(round(p / t) * t, 2)


def weekdays_back(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


# ----------------------------------------------------------------------------- history
def make_history(code: str, name: str, market: str, kind: str, last_close: float, n: int, seed: int,
                 start_ratio: float = 0.55, asof: date = date(2026, 10, 22), daily_vol: float = 0.016) -> dict:
    rnd = random.Random(seed)
    days = weekdays_back(asof, n)
    # log-price random walk, then rescale so that the final close is exactly `last_close`
    lp = [0.0]
    drift = math.log(1 / start_ratio) / n
    for _ in range(n - 1):
        lp.append(lp[-1] + drift + rnd.gauss(0, daily_vol))
    scale = last_close / math.exp(lp[-1])
    closes = [math.exp(x) * scale for x in lp]
    bars = []
    prev = closes[0] * (1 - 0.003)
    for d, c in zip(days, closes):
        o = prev * (1 + rnd.gauss(0, daily_vol * 0.35))
        hi = max(o, c) * (1 + abs(rnd.gauss(0, daily_vol * 0.4)))
        lo = min(o, c) * (1 - abs(rnd.gauss(0, daily_vol * 0.4)))
        base_v = 28e6 if kind == "stock" else 60e6
        v = int(base_v * math.exp(rnd.gauss(0, 0.35)) * (1 + 6 * abs(c / prev - 1)))
        so, sh, sl, sc = snap(o, kind), snap(hi, kind), snap(lo, kind), snap(c, kind)
        sh = max(sh, so, sc)
        sl = min(sl, so, sc)
        bars.append({"d": d.isoformat(), "o": so, "h": sh, "l": sl, "c": sc, "v": v})
        prev = c
    bars[-1]["c"] = last_close
    bars[-1]["h"] = max(bars[-1]["h"], last_close, bars[-1]["o"])
    bars[-1]["l"] = min(bars[-1]["l"], last_close, bars[-1]["o"])
    divs = [{"d": (asof - timedelta(days=127)).isoformat(), "amt": 5.0 if kind == "stock" else 0.9},       # ascending, like the real API
            {"d": (asof - timedelta(days=36)).isoformat(), "amt": 6.0 if kind == "stock" else 1.0}]
    return {"status": "ready", "symbol": code, "name": name, "market": market, "kind": kind,
            "asof": asof.isoformat(), "last_bar_complete": True, "stale": False, "bars": bars, "dividends": divs}


# ----------------------------------------------------------------------------- forecast
def make_forecast(code="2330", name="台積電", market="TWSE", kind="stock", last_close=2385.0, chg_1d=-0.0145,
                  seed=7, sigma_d=0.0165, n_bars=6669, asof="2026-10-22", p_ups=(0.52, 0.565, 0.585),
                  signals=("hold", "buy", "buy"), confs=("低", "中", "中")) -> dict:
    rnd = random.Random(seed)
    horizons = {}
    names = {"garch": "GJR-GARCH-t", "hmm": "隱馬可夫regime", "dlm": "動態線性模型(貝氏)", "naive": "隨機漫步基準"}
    rt_cost = 0.0057 if kind == "stock" else 0.0037
    for h, p_up, sig, conf in zip((1, 5, 20), p_ups, signals, confs):
        s = sigma_d * math.sqrt(h) * (1 + 0.04 * rnd.random())
        mu_log = ND.inv_cdf(p_up) * s              # so that P(R>0) ~ p_up under a normal predictive
        q = {}
        for lv in LEVELS:
            z = ND.inv_cdf(lv)
            skew = -0.06 * s if lv < 0.5 else 0.0   # slight negative skew
            q[str(lv)] = math.expm1(mu_log + z * s * (1.04 if z < 0 else 0.97) + skew * abs(z))
        exp_ret = math.expm1(mu_log + 0.5 * s * s)
        med = q["0.5"]
        horizons[str(h)] = {
            "h": h, "days_label": {1: "明日", 5: "未來5日", 20: "未來20日"}[h],
            "p_up": r(p_up), "p_up_raw": r(p_up + rnd.uniform(-0.03, 0.04)),
            "exp_return": r(exp_ret, 5), "median_return": r(med, 5), "sigma": r(s, 5),
            "quantiles": {k: r(v, 5) for k, v in q.items()},
            "price_quantiles": {k: r(last_close * (1 + v), 2) for k, v in q.items()},
            "mu_log": r(mu_log, 5), "sigma_log": r(s, 5),
            "signal": sig, "score": r((exp_ret - rt_cost) / s, 4), "net_exp_return": r(exp_ret - rt_cost, 5),
            "confidence": conf,
            "components": {
                "garch": {"p_up": r(0.5 + rnd.uniform(-0.02, 0.03)), "mu": r(rnd.uniform(-0.0002, 0.0006) * h, 5)},
                "hmm": {"p_up": r(p_up + rnd.uniform(-0.04, 0.05)), "mu": r(rnd.uniform(-0.0003, 0.0009) * h, 5)},
                "dlm": {"p_up": r(p_up + rnd.uniform(-0.03, 0.06)), "mu": r(rnd.uniform(0.0, 0.0012) * h, 5)},
                "naive": {"p_up": r(0.5 + rnd.uniform(0.0, 0.03)), "mu": r(0.0003 * h, 5)},
            },
        }
    # fan chart: log-return quantiles growing like sqrt(t) with a mild fat-tail widening
    days = list(range(1, 21))
    mu1 = ND.inv_cdf(p_ups[0]) * sigma_d * 0.4
    prices, rets = [], []
    for lv in LEVELS:
        z = ND.inv_cdf(lv)
        row_p, row_r = [], []
        for t in days:
            s_t = sigma_d * math.sqrt(t) * (1 + 0.002 * t)
            lr = mu1 * t * 0.5 + z * s_t
            row_r.append(r(math.expm1(lr), 5))
            row_p.append(r(last_close * math.exp(lr), 2))
        prices.append(row_p)
        rets.append(row_r)
    weights = {}
    for h, w in zip(("1", "5", "20"), ((0.34, 0.16, 0.18, 0.32), (0.30, 0.20, 0.30, 0.20), (0.22, 0.20, 0.38, 0.20))):
        weights[h] = dict(zip(("garch", "hmm", "dlm", "naive"), w))
    drivers = [
        {"name": "12-1個月動能", "key": "mom_12_1", "value": 0.31, "z": 1.2, "contribution": 0.0003},
        {"name": "短期反轉(5日)", "key": "rev_5", "value": -0.031, "z": -1.1, "contribution": 0.00021},
        {"name": "月動能(21日)", "key": "mom_21", "value": 0.046, "z": 0.7, "contribution": 0.00012},
        {"name": "均線乖離(60日)", "key": "ma_gap_60", "value": 0.052, "z": 0.9, "contribution": -0.00008},
        {"name": "RSI(14)", "key": "rsi_14", "value": 46.0, "z": -0.3, "contribution": -0.00003},
        {"name": "波動率比(近5日/60日)", "key": "vol_ratio", "value": 1.12, "z": 0.4, "contribution": -0.00005},
        {"name": "成交量Z分數", "key": "vol_z", "value": 0.2, "z": 0.2, "contribution": 0.00002},
        {"name": "大盤報酬(5日)", "key": "mkt_ret_5", "value": -0.012, "z": -0.6, "contribution": -0.00011},
        {"name": "個股特異動能", "key": "idio_mom", "value": 0.08, "z": 0.5, "contribution": 0.00009},
    ]
    nxt = next_weekday(date.fromisoformat(asof)).isoformat()
    return {
        "status": "ready", "symbol": code, "name": name, "market": market, "kind": kind,
        "asof": asof, "asof_complete": True, "next_trading_day": nxt, "last_close": last_close,
        "chg_1d": chg_1d, "n_bars": n_bars, "short_history": False,
        "horizons": horizons,
        "fan": {"days": days, "levels": LEVELS, "prices": prices, "returns": rets},
        "regime": {"labels": ["低波動盤整", "多頭穩定", "高波動空頭"], "probs": [0.58, 0.31, 0.11],
                   "current": "低波動盤整", "confident": True,
                   "transition": [[0.965, 0.025, 0.010], [0.030, 0.960, 0.010], [0.050, 0.030, 0.920]]},
        "vol": {"sigma_daily": sigma_d, "sigma_annual": r(sigma_d * math.sqrt(245), 4), "percentile": 0.7,
                "regime_note": "目前波動度高於歷史約 70% 的時間，屬於偏高但未達極端的水準。"},
        "risk": {"var95_1d": -0.0252, "cvar95_1d": -0.0341, "var95_20d": -0.108, "cvar95_20d": -0.149,
                 "median_max_drawdown_20d": -0.052, "prob_loss_gt_10pct_20d": 0.043},
        "models": {"weights": weights, "names": names},
        "engine": {"version": "1.1", "refit_every": 21, "last_refit_row": 6657, "state_window": 700, "has_market": True,
                   "placeholder_bars_dropped": 0, "p_up_event": "up_given_move", "mc": {"paths": 5000, "pool": 1005, "regime_mixing": True}},
        "drivers": drivers,
        "notes": ["資料截至 2026-10-22 收盤，預測目標為下一個交易日。",
                  "預測以還原權息價計算，價格區間以最新未還原收盤價換算。"],
        "disclaimer": DISCLAIMER,
    }


def make_forecast_nulls() -> dict:
    """Same shape, but many numbers are null / blocks missing: the UI must degrade gracefully."""
    fc = make_forecast(code="6488", name="環球晶", market="TPEx", seed=3)
    fc["chg_1d"] = None
    fc["short_history"] = True
    fc["asof_complete"] = False
    h = fc["horizons"]
    h["1"]["p_up"] = None
    h["1"]["exp_return"] = None
    h["1"]["quantiles"] = {"0.05": None, "0.25": None, "0.5": None, "0.75": None, "0.95": None}
    h["1"]["price_quantiles"] = {"0.05": None, "0.25": None, "0.5": None, "0.75": None, "0.95": None}
    h["1"]["signal"] = None
    h["1"]["confidence"] = None
    h["1"]["components"] = {"garch": {"p_up": None, "mu": None}}
    h["5"]["score"] = None
    h["5"]["net_exp_return"] = None
    h["5"]["sigma"] = None
    del h["20"]["components"]
    del h["20"]["price_quantiles"]
    fc["fan"]["prices"][2][5] = None
    fc["fan"]["prices"][0][19] = None
    fc["regime"]["probs"] = [None, 0.4, 0.6]
    fc["regime"]["transition"] = None
    fc["vol"] = {"sigma_daily": None, "sigma_annual": 0.41, "percentile": None, "regime_note": None}
    fc["risk"] = {"var95_1d": None, "cvar95_1d": -0.04}
    fc["models"]["weights"] = {"1": {"garch": None, "hmm": 0.5, "dlm": 0.5, "naive": 0.0}}
    fc["drivers"][0]["value"] = None
    fc["drivers"][1]["z"] = None
    fc["drivers"][2]["contribution"] = None
    fc["notes"] = []
    del fc["disclaimer"]
    return fc


def make_forecast_sparse() -> dict:
    """Minimal payload: only the keys the server is guaranteed to send; everything else absent."""
    return {"status": "ready", "symbol": "9999", "name": "測試", "asof": "2026-10-22", "last_close": 12.34,
            "horizons": {"1": {"h": 1, "p_up": 0.5}}}


# ----------------------------------------------------------------------------- recommendations
NAMES = [
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"), ("2382", "廣達"),
    ("2412", "中華電"), ("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金"),
    ("1301", "台塑"), ("1303", "南亞"), ("2002", "中鋼"), ("3711", "日月光投控"), ("2603", "長榮"),
    ("2609", "陽明"), ("3034", "聯詠"), ("2379", "瑞昱"), ("2303", "聯電"), ("2357", "華碩"),
    ("2345", "智邦"), ("2884", "玉山金"), ("2885", "元大金"), ("2892", "第一金"), ("5880", "合庫金"),
    ("2880", "華南金"), ("2883", "開發金"), ("2887", "台新金"), ("1216", "統一"), ("1101", "台泥"),
    ("2207", "和泰車"), ("2301", "光寶科"), ("2327", "國巨"), ("2353", "宏碁"), ("2356", "英業達"),
    ("2395", "研華"), ("2408", "南亞科"), ("2474", "可成"), ("3008", "大立光"), ("3045", "台灣大"),
    ("3231", "緯創"), ("3443", "創意"), ("3661", "世芯-KY"), ("4904", "遠傳"), ("5871", "中租-KY"),
    ("5876", "上海商銀"), ("6446", "藥華藥"), ("6505", "台塑化"), ("6669", "緯穎"), ("9910", "豐泰"),
    ("1590", "亞德客-KY"), ("2049", "上銀"), ("2059", "川湖"), ("2618", "長榮航"), ("2610", "華航"),
    ("2615", "萬海"), ("3017", "奇鋐"), ("3037", "欣興"),
]


def spearman(x, y):
    """Spearman rank correlation (average ranks), pure stdlib."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def make_recs(h=5, seed=11, evidence=None) -> dict:
    """The §6.3 / §9 payload: ranked by rank_signal = mu_log / sigma_log (NOT by the cost-adjusted score), strict buy / avoid lists,
    leaders / laggards, lean / low_vol / edge_covers_cost, portfolio with ``suppressed_reason`` and the ``evidence`` block.  Every REC
    carries a non-empty ``reason`` (the real server never sends an empty one)."""
    rnd = random.Random(seed + h)
    regimes = ["低波動盤整", "中波動震盪", "高波動震盪"]
    rt_cost = 0.0057
    raw = []
    for code, name in NAMES:
        base = rnd.uniform(25, 2400)
        price = snap(base * (1 if code in ("2330", "3008", "3661", "6669") else 0.3 + rnd.random()))
        if code == "2330":
            price = 2385.0
        u = rnd.uniform(0.8, 1.6)                          # volatility scale: calmer names get a (slightly) higher p -> rank_signal is low-vol tilted
        s = 0.017 * math.sqrt(h) * u
        if h >= 20:                                         # a month: every drift is positive (the real h = 20 picture)
            p = min(0.66, max(0.51, 0.57 + rnd.gauss(0, 0.04) - 0.1 * (u - 1.2)))
        else:
            p = min(0.68, max(0.34, 0.5 + rnd.gauss(0, 0.06) - 0.12 * (u - 1.2)))
        rs = ND.inv_cdf(p) * 0.9                          # relative strength = mu_log / sigma_log (calibration pulls p a little off Phi(rs))
        mu_log = rs * s
        exp_r = math.expm1(mu_log + 0.5 * s * s)
        net = exp_r - rt_cost
        score = net / s
        sig = "buy" if p >= 0.55 and net > 0 else ("avoid" if p <= 0.45 and net < 0 else "hold")
        conf = "高" if abs(p - 0.5) > 0.1 else ("中" if abs(p - 0.5) > 0.05 else "低")
        raw.append({
            "code": code, "name": name, "kind": "stock", "last_close": price,
            "chg_1d": r(rnd.gauss(0, 0.014), 4), "p_up": r(p), "exp_return": r(exp_r, 5), "net_exp_return": r(net, 5),
            "sigma": r(s, 5), "score": r(score, 4), "score_shrunk": r(score * 0.55, 4), "signal": sig, "confidence": conf,
            "weight": 0.0, "reason": "",
            "p_up_by_h": {"1": r(0.5 + (p - 0.5) * 0.5), "5": r(p), "20": r(0.5 + (p - 0.5) * 1.1)},
            "regime": regimes[rnd.choices([0, 1, 2], [0.5, 0.35, 0.15])[0]],
            "avg_value_20d": r(10 ** rnd.uniform(8.2, 10.9), 0), "liquid": rnd.random() > 0.04,
            "rank_signal": r(rs, 4), "edge_covers_cost": net > 0,
            "edge_vs_baseline": r(p - (0.5 + rnd.uniform(0.0, 0.03)), 4),
        })
    raw.sort(key=lambda x: (-x["rank_signal"], -x["p_up"], x["code"]))          # ties by p_up, then code
    n = len(raw)
    med = median([x["sigma"] for x in raw])
    days_label = {1: "明日", 5: "未來5日", 20: "未來20日"}[h]
    for i, rec in enumerate(raw, 1):
        rec["rank"] = i
        pct = (n - i) / (n - 1) if n > 1 else 0.5
        rec["score_rank_pct"] = r(pct, 4)
        rec["lean"] = "bullish" if pct >= 0.8 and rec["p_up"] > 0.5 else ("bearish" if pct <= 0.2 and rec["p_up"] < 0.5 else "neutral")
        rec["low_vol"] = rec["sigma"] < med
        if rec["signal"] == "buy":
            rec["reason"] = f"{rec['regime']}狀態、動能與反轉因子偏多，模型估計{days_label}上漲機率 {rec['p_up'] * 100:.0f}%、扣除成本後預期報酬 {rec['net_exp_return'] * 100:+.2f}%。"
        elif rec["signal"] == "avoid":
            rec["reason"] = "短線動能轉弱且波動偏高，扣除成本後預期報酬為負，建議避開（本工具不提供放空建議）。"
        else:
            rec["reason"] = f"目前處於「{rec['regime']}」，模型估計上漲機率 {rec['p_up'] * 100:.0f}%，未達偏多候選門檻 55%，方向不明確，建議觀望。"
    buys_all = [x for x in raw if x["signal"] == "buy"]
    buys = buys_all[:10]
    avoids = sorted([x for x in raw if x["signal"] == "avoid"], key=lambda x: x["rank"], reverse=True)[:10]
    leaders = [x for x in raw if x["p_up"] > 0.5][:10]
    laggards = raw[-min(10, n // 2):]
    wsum = 0.0
    for b in buys[:5]:
        b["weight"] = r(min(0.15, 0.05 + 0.4 * max(0, b["score_shrunk"])), 3)
        wsum += b["weight"]
    if wsum > 0:
        portfolio = {"cash_weight": r(1 - wsum, 3), "n": min(5, len(buys)), "suppressed_reason": None,
                     "note": "權重為分數收縮後之 1/4 凱利比例，單一標的上限 15%，其餘持有現金。"}
    else:
        portfolio = {"cash_weight": 1.0, "n": 0, "suppressed_reason": "no_positive_edge" if h == 1 else "B0",
                     "note": "個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移，因此不配置權重，建議全數持有現金。"}
    return {
        "status": "ready", "asof": "2026-10-22", "asof_complete": True, "horizon": h,
        "computed_at": "2026-10-22T18:31:07+08:00", "universe_size": 60, "n_scored": n,
        "market": {"p_up_1": 0.51, "p_up_5": 0.53, "regime": "中波動震盪", "last_close": 46041.4, "chg_1d": -0.011},
        "buys": buys, "avoids": avoids, "all": raw,
        "leaders": leaders, "laggards": laggards, "laggards_all_above_half": all(x["p_up"] > 0.5 for x in laggards),
        "n_buy_total": len(buys_all), "rank_key": "rank_signal",
        "rank_sigma_corr": r(spearman([x["rank_signal"] for x in raw], [x["sigma"] for x in raw]), 4),
        "universe_sigma_median": r(med, 5),
        "portfolio": portfolio,
        "evidence": evidence if evidence is not None else {"source": "none"},
        "notes": ["排名依「相對強度」= 預期對數報酬 ÷ 預期對數波動度（不含交易成本與波動凸性項）；預測均值被強烈收縮，此排序只代表模型對各檔的相對看法，並非選股能力的證明。",
                  "評分 = (預期報酬 − 來回交易成本) ÷ 波動度；此評分只用於偏多候選／偏空把關、是否覆蓋成本與建議權重，不用於排名；全部機率皆為模型估計，非保證。"],
        "skipped": [], "stale": False,
    }


# ----------------------------------------------------------------------------- backtest
def downsample_dates(n=600):
    ds = weekdays_back(date(2026, 10, 21), 1240)
    step = (len(ds) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [ds[i].isoformat() for i in idx]


def make_stats(eq, days_per_year=118.4, exposure=1.0, n_trades=0, turnover=0.0, hit=None, sharpe_ci=None, psr=None, dsr=None,
               rets=None):
    n = len(eq)
    total = eq[-1] / eq[0] - 1
    yrs = max(1e-9, (n - 1) / days_per_year)
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    if rets is None:
        rets = [eq[i] / eq[i - 1] - 1 for i in range(1, n)]
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
    dsd = math.sqrt(sum(min(0, x) ** 2 for x in rets) / len(rets))
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    sharpe = m / sd * math.sqrt(days_per_year)
    return {"total_return": r(total), "cagr": r(cagr), "vol": r(sd * math.sqrt(days_per_year)), "sharpe": r(sharpe, 3),
            "sortino": r(m / dsd * math.sqrt(days_per_year) if dsd else None, 3), "max_dd": r(mdd),
            "calmar": r(cagr / abs(mdd) if mdd else None, 3), "exposure": r(exposure, 3),
            "turnover_per_year": r(turnover, 2), "n_trades": n_trades, "trade_hit_rate": r(hit) if hit is not None else None,
            "psr": r(psr, 3) if psr is not None else None, "dsr": r(dsr, 3) if dsr is not None else None,
            "sharpe_ci95": sharpe_ci}


def drawdown(eq):
    peak, out = eq[0], []
    for v in eq:
        peak = max(peak, v)
        out.append(r(v / peak - 1, 5))
    return out


def make_backtest(h=1, seed=21, level="no_edge", thr=0.55) -> dict:
    rnd = random.Random(seed + h)
    dates = downsample_dates(600)
    n = len(dates)
    bh, st, pos = [1.0], [1.0], []
    state, hold = 0, 0
    for i in range(1, n):
        if hold <= 0:
            state = 1 if rnd.random() < 0.58 else 0
            hold = rnd.randint(4, 30)
        hold -= 1
        pos.append(state)
        ret = rnd.gauss(0.0011, 0.0237)
        bh.append(bh[-1] * (1 + ret))
        st.append(st[-1] * (1 + (ret * state if state else 0.0) - (0.0006 if (i > 1 and pos[-1] != pos[-2]) else 0)))
    pos = [pos[0]] + pos
    rets_bh = [bh[i] / bh[i - 1] - 1 for i in range(1, n)]
    rets_st = [st[i] / st[i - 1] - 1 for i in range(1, n)]
    # rolling accuracy / base rate (250-sample window on a ~1236 sample series, downsampled)
    rolling_dates = dates[150:]
    acc, base = [], []
    for i, _ in enumerate(rolling_dates):
        a = 0.535 + 0.05 * math.sin(i / 55.0) + rnd.gauss(0, 0.008)
        b = 0.548 + 0.04 * math.sin(i / 70.0 + 1.0) + rnd.gauss(0, 0.006)
        acc.append(r(a))
        base.append(r(b))
    edge_acc = {"no_edge": 0.536, "weak_edge": 0.572, "moderate_edge": 0.60, "strong_edge": 0.64}[level]
    pv = {"no_edge": 0.31, "weak_edge": 0.048, "moderate_edge": 0.008, "strong_edge": 0.0004}[level]
    titles = {"no_edge": "沒有統計上顯著的預測優勢", "weak_edge": "有微弱的統計優勢，但仍須審慎看待",
              "moderate_edge": "具有中等程度的預測優勢", "strong_edge": "具有顯著的預測優勢"}
    cal = []
    for p_mean, freq, nn in [(0.40, 0.43, 41), (0.44, 0.46, 118), (0.48, 0.49, 232), (0.52, 0.55, 301),
                             (0.56, 0.54, 266), (0.60, 0.58, 160), (0.64, 0.66, 78), (0.69, 0.63, 40)]:
        cal.append({"p_mean": p_mean, "freq": freq, "n": nn})
    report = {
        "symbol": "2330", "name": "台積電", "kind": "stock", "horizon": h,
        "params": {"threshold": thr, "exec": "next_open", "commission_discount": 0.6, "slippage_bps": 5, "sizing": "threshold",
                   "n_trials": 1, "roundtrip_cost": 0.0057, "min_train": 500, "refit_every": 21},
        "period": {"start": "2021-10-25", "end": "2026-10-21", "n_forecasts": 1240, "n_scored": 1236, "n_tie_excluded": 4},
        "direction": {
            "accuracy": edge_acc, "ci95": [r(edge_acc - 0.028), r(edge_acc + 0.028)], "n": 1236, "base_rate_up": 0.55,
            "baselines": {"always_up": 0.55, "prev_sign": 0.49, "train_majority": 0.55}, "best_baseline": "always_up",
            "p_value_vs_best_baseline": pv, "p_value_vs_coin": 0.01,
            "by_confidence": [
                {"bucket": "[0.40,0.45)", "lo": 0.40, "hi": 0.45, "n": 62, "accuracy": 0.55, "mean_p_up": 0.43, "actual_up_rate": 0.45},
                {"bucket": "[0.45,0.50)", "lo": 0.45, "hi": 0.50, "n": 250, "accuracy": 0.52, "mean_p_up": 0.48, "actual_up_rate": 0.48},
                {"bucket": "[0.50,0.55)", "lo": 0.50, "hi": 0.55, "n": 540, "accuracy": 0.54, "mean_p_up": 0.52, "actual_up_rate": 0.54},
                {"bucket": "[0.55,0.60)", "lo": 0.55, "hi": 0.60, "n": 290, "accuracy": 0.56, "mean_p_up": 0.57, "actual_up_rate": 0.56},
                {"bucket": "[0.60,0.65)", "lo": 0.60, "hi": 0.65, "n": 77, "accuracy": 0.60, "mean_p_up": 0.62, "actual_up_rate": 0.60},
                {"bucket": "[0.65,0.70)", "lo": 0.65, "hi": 0.70, "n": 17, "accuracy": 0.59, "mean_p_up": 0.66, "actual_up_rate": 0.59},
            ],
            "high_conf": {"threshold": 0.6, "n": 94, "accuracy": 0.6, "p_value": 0.1}},
        "probabilistic": {"brier": 0.2501, "brier_baseline": 0.2499, "brier_skill": -0.0008, "log_loss": 0.6931,
                          "log_loss_baseline": 0.6925, "ece": 0.021, "calibration": cal,
                          "log_loss_skill": -0.0009, "brier_baseline_oracle": 0.2497, "brier_skill_oracle": -0.0016,
                          "log_loss_baseline_oracle": 0.6921, "log_loss_skill_oracle": -0.0014,
                          "ece_noise_floor": {1: 0.054, 5: 0.104, 20: 0.2}[h], "p_up_event": "up_given_move", "n": 1236},
        "returns": {"ic": 0.031, "ic_pvalue": 0.27, "oos_r2": -0.002, "rmse_model": 0.0161, "rmse_naive": 0.0160,
                    "dm_stat": 0.4, "dm_pvalue": 0.69, "coverage": {"0.50": 0.51, "0.80": 0.79, "0.90": 0.89},
                    "kupiec_pvalue": 0.61, "pit_hist": [0.108, 0.097, 0.101, 0.095, 0.104, 0.098, 0.103, 0.099, 0.101, 0.094],
                    "pit_ks_pvalue": 0.42, "pinball_loss": 0.0051},
        "strategy": {
            "rule": f"機率≥{int(thr * 100)}%且扣成本後期望報酬>0 時持有，否則空手",
            "stats": make_stats(st, exposure=sum(pos) / len(pos), n_trades=84, turnover=7.4, hit=0.52, sharpe_ci=[-0.31, 0.86],
                                psr=0.74, dsr=0.55, rets=rets_st),
            "benchmark": make_stats(bh, exposure=1.0, n_trades=1, turnover=0.0, sharpe_ci=[-0.2, 0.9], psr=0.71, dsr=None, rets=rets_bh),
            "sharpe_diff_ci95": [-0.4, 0.6],
            "curve": {"dates": dates, "strategy": [r(v, 5) for v in st], "buyhold": [r(v, 5) for v in bh], "position": pos,
                      "drawdown_strategy": drawdown(st), "drawdown_buyhold": drawdown(bh)}},
        "rolling": {"window": 250, "dates": rolling_dates, "accuracy": acc, "base_rate": base},
        "verdict": {"level": level, "title": titles[level],
                    "text": (f"樣本外 {1236} 次預測中，方向準確率 {edge_acc * 100:.1f}%，最佳基準(永遠看漲) 55.0%，"
                             f"二項檢定 p={pv}。Brier 技能分數 −0.0008，等同於氣候基準；資訊係數 0.031 (p=0.27)。"),
                    "bullets": ["方向準確率未顯著高於「永遠看漲」基準。", "校準誤差(ECE) 2.1%，機率大致可信但沒有額外鑑別力。",
                                "策略夏普 0.38 對買進持有 0.52，扣除成本後沒有優勢。"]},
        "notes": ["已計入手續費(6折)、證交稅 0.3% 與滑價 5bp；次日開盤價成交。",
                  "未計入：股利再投資之稅負、融資利息、盤中無法成交之風險。",
                  "樣本內選股存在存活者偏差：本檔為今日仍為大型權值股。"]}
    return {"status": "ready", "report": report}


def make_universe(h=5) -> dict:
    rnd = random.Random(99)
    dates = downsample_dates(500)
    n = len(dates)
    top, ew, tx = [1.0], [1.0], [1.0]
    for _ in range(n - 1):
        m = rnd.gauss(0.0007, 0.0105)
        ew.append(ew[-1] * (1 + m))
        tx.append(tx[-1] * (1 + m * 0.9 + rnd.gauss(0, 0.002)))
        top.append(top[-1] * (1 + m * 1.1 + rnd.gauss(0.00005, 0.003)))
    ic_dates = dates[::5]
    ic_vals = [r(rnd.gauss(0.012, 0.11), 4) for _ in ic_dates]
    stats_top = make_stats(top, exposure=0.92, n_trades=1230, turnover=14.2, hit=0.53, sharpe_ci=[-0.1, 0.9], psr=0.8, dsr=0.6)
    stats_ew = make_stats(ew, exposure=1.0, n_trades=60, turnover=0.4, sharpe_ci=[0.0, 0.95], psr=0.9, dsr=None)
    stats_tx = make_stats(tx, exposure=1.0, n_trades=1, turnover=0.0, sharpe_ci=[-0.1, 0.8], psr=0.82, dsr=None)
    ts = [1.0]
    for i in range(1, n):                              # model-free "top-k by sigma" basket: higher beta, higher vol
        ts.append(ts[-1] * (1 + (ew[i] / ew[i - 1] - 1) * 1.5 + rnd.gauss(0.0004, 0.004)))
    stats_ts = make_stats(ts, exposure=1.0, n_trades=1100, turnover=22.0, hit=0.51, sharpe_ci=[0.2, 1.4], psr=0.97, dsr=None)
    per_stock = []
    for code, name in NAMES[:40]:
        bs = rnd.gauss(-0.002, 0.004)
        per_stock.append({"code": code, "name": name, "accuracy": r(rnd.uniform(0.47, 0.58)), "brier_skill": r(bs, 4),
                          "brier_skill_oracle": r(bs - 0.001, 4), "ic": r(rnd.gauss(0.01, 0.04), 3), "n": 1200})
    report = {
        "horizon": h, "top_k": 5,
        "period": {"start": "2021-10-25", "end": "2026-10-21", "n_days": 1100, "n_stocks": 58},
        "ic": {"mean": 0.012, "ir": 0.2, "t": 2.1, "pvalue": 0.03, "series": {"dates": ic_dates, "values": ic_vals}},
        "picks": {"hit_rate": 0.53, "base_rate": 0.52, "n": 5400, "p_value": 0.12, "mean_return": 0.003},
        "top": stats_top, "benchmark_equal_weight": stats_ew, "benchmark_taiex": stats_tx,
        "benchmark_top_sigma": stats_ts,
        "basket": {"k": 5, "mean_sigma": 0.025, "universe_mean_sigma": 0.046, "sigma_ratio": 0.54, "mean_beta": 0.42, "universe_mean_beta": 0.84,
                   "beta_market": "taiex", "topk_overlap": 0.84, "corr_rank_sigma": -0.56, "share_net_positive": 0.12,
                   "sigma_basket": {"mean_sigma": 0.088, "mean_beta": 1.42}},
        "exposure": {"mean_sigma_top": 0.025, "mean_sigma_universe": 0.046, "beta_top": 0.42, "beta_universe": 0.84, "benchmark_top_sigma": stats_ts},
        "rank_key": "rank_signal", "only_buy_signals": False,
        "labels": {"top": "模型精選：每日相對強度最高的前 5 名", "benchmark_equal_weight": "全體等權重（不扣成本）",
                   "benchmark_top_sigma": "無預測對照：每日預測波動度最高的前 5 名", "benchmark_taiex": "加權指數（價格指數，不含股利）"},
        "spread": {"mean": 0.0004, "t": 1.1},
        "curve": {"dates": dates, "top": [r(v, 5) for v in top], "equal_weight": [r(v, 5) for v in ew], "taiex": [r(v, 5) for v in tx],
                  "top_sigma": [r(v, 5) for v in ts]},
        "per_stock": per_stock,
        "verdict": {"level": "weak_edge", "title": "橫斷面排序有微弱但不穩定的資訊", "text": "平均橫斷面 IC 為 0.012 (t=2.1)，前 5 名組合報酬扣除成本後與等權重基準差距不顯著。",
                    "bullets": ["IC 的 t 值達 2.1，但資訊比率僅 0.2。", "選股命中率 53% 對基準 52%，p=0.12。"]},
        "notes": ["存活者偏差：宇宙為今日流動性最高的股票，過去已下市或衰退的標的不在其中，績效可能被高估。",
                  "每 5 日再平衡，等權重持有前 5 名；已計入交易成本。"]}
    return {"status": "ready", "report": report}


def main() -> None:
    def dump(name, obj):
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    dump("status.json", {"status": "ready", "server_time": "2026-10-22T10:31:07+00:00", "taipei_time": "2026-10-22T18:31:07+08:00",
                         "market_open": False, "last_trading_date": "2026-10-22", "version": "1.0"})
    dump("search.json", {"status": "ready", "results": [
        {"code": "2330", "name": "台積電", "market": "TWSE", "kind": "stock"},
        {"code": "2331", "name": "精英", "market": "TWSE", "kind": "stock"},
        {"code": "6488", "name": "環球晶", "market": "TPEx", "kind": "stock"},
        {"code": "0050", "name": "元大台灣50", "market": "TWSE", "kind": "etf"}]})
    dump("search_empty.json", {"status": "ready", "results": [{"code": c, "name": n, "market": "TWSE", "kind": "stock"} for c, n in NAMES[:12]]
                               + [{"code": "0050", "name": "元大台灣50", "market": "TWSE", "kind": "etf"},
                                  {"code": "0056", "name": "元大高股息", "market": "TWSE", "kind": "etf"},
                                  {"code": "00878", "name": "國泰永續高股息", "market": "TWSE", "kind": "etf"}]})
    dump("history_2330.json", make_history("2330", "台積電", "TWSE", "stock", 2385.0, 750, seed=5))
    dump("history_0050.json", make_history("0050", "元大台灣50", "TWSE", "etf", 60.35, 400, seed=9, start_ratio=0.7, daily_vol=0.011))
    dump("forecast_2330.json", make_forecast())
    dump("forecast_0050.json", make_forecast(code="0050", name="元大台灣50", kind="etf", last_close=60.35, chg_1d=0.0067, seed=12,
                                             sigma_d=0.0105, p_ups=(0.505, 0.47, 0.44), signals=("hold", "hold", "avoid"),
                                             confs=("低", "低", "中")))
    dump("forecast_nulls.json", make_forecast_nulls())
    dump("forecast_sparse.json", make_forecast_sparse())
    dump("recommendations_h5.json", make_recs(5))
    dump("recommendations_h1.json", make_recs(1, evidence={
        "horizon": 1, "source": "universe_validation", "verdict_level": "no_edge", "ic": 0.006, "ic_pvalue": 0.66, "hit_rate": 0.565, "base_rate": 0.537,
        "computed_at": "2026-10-22T18:40:00+08:00", "net_cagr": 0.097, "ew_cagr": 0.304, "sigma_cagr": 0.706, "beats_ew_after_costs": False}))
    dump("recommendations_h20.json", make_recs(20))
    empty_recs = make_recs(5)
    empty_recs.update({"buys": [], "avoids": [], "all": [], "leaders": [], "laggards": [], "laggards_all_above_half": False, "n_scored": 0,
                       "n_buy_total": 0, "rank_sigma_corr": None, "universe_sigma_median": None,
                       "portfolio": {"cash_weight": 1.0, "n": 0, "suppressed_reason": "no_positive_edge",
                                     "note": "今日沒有符合條件的偏多候選，不配置權重，建議全數持有現金。"}})
    dump("recommendations_empty.json", empty_recs)
    dump("backtest_2330_h1.json", make_backtest(1, level="no_edge"))
    dump("backtest_2330_h5.json", make_backtest(5, level="weak_edge", thr=0.55))
    dump("backtest_2330_h20.json", make_backtest(20, level="no_edge"))
    bn = make_backtest(1, level="moderate_edge")
    rep = bn["report"]
    rep["direction"]["ci95"] = [None, None]
    rep["direction"]["by_confidence"] = []
    rep["probabilistic"]["calibration"] = []
    rep["probabilistic"]["ece"] = None
    rep["returns"]["pit_hist"] = None
    rep["returns"]["coverage"] = {"0.50": None, "0.80": 0.8}
    rep["strategy"]["stats"]["dsr"] = None
    rep["strategy"]["stats"]["sharpe_ci95"] = [None, None]
    rep["strategy"]["sharpe_diff_ci95"] = None
    rep["strategy"]["curve"]["drawdown_strategy"] = None
    rep["strategy"]["curve"]["strategy"][10:14] = [None] * 4
    rep["rolling"] = {"window": 250, "dates": [], "accuracy": [], "base_rate": []}
    rep["verdict"]["bullets"] = []
    rep["notes"] = []
    dump("backtest_nulls.json", bn)
    dump("backtest_universe.json", make_universe(5))
    nu = make_universe(5)
    nu["report"]["benchmark_taiex"] = None
    for k in ("benchmark_top_sigma", "basket", "exposure", "rank_key", "only_buy_signals", "labels"):
        nu["report"].pop(k, None)                      # a report of an older server: none of the §9.1 blocks
    nu["report"]["curve"].pop("top_sigma", None)
    nu["report"]["curve"]["taiex"] = None
    nu["report"]["ic"]["series"] = {"dates": [], "values": []}
    nu["report"]["per_stock"] = []
    nu["report"]["spread"] = {"mean": None, "t": None}
    dump("backtest_universe_nulls.json", nu)
    dump("computing.json", {"status": "computing", "progress": {"done": 12, "total": 60, "message": "計算 2330 …"}})
    dump("computing_nototal.json", {"status": "computing", "progress": {"done": 0, "total": 0, "message": ""}})
    dump("error.json", {"status": "error", "error": {"code": "unknown_symbol", "message": "找不到這檔股票：1234"}})
    dump("error_insufficient.json", {"status": "error", "error": {"code": "insufficient_history", "message": "歷史資料不足 250 個交易日，無法預測。"}})


if __name__ == "__main__":
    main()
