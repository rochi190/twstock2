"""Synthetic check of the flow-1 protocol: how often does it pass when there is NO signal (false positives), and how often
does it find a planted one (power)?  Run:  python3 -m research.simulate [--reps 40]

The synthetic market: S stocks on business days from 2021-01-04, daily log returns = common market factor + idiosyncratic
noise (2 % a day), constant volume, institutional flows ~ N(0, 10 % of volume) a day, independent across stocks.  With
``strength > 0`` the foreign-flow signal (foreign_20, lagged as the protocol says) adds ``strength * z`` to every later daily
return, ``z`` its cross-sectional z-score.
"""
from __future__ import annotations

import argparse

import numpy as np

from backend.types import PriceData
from research import flow_experiment as FE
from research import twse_flow as TF


def make_world(n_stocks: int = 80, n_days: int = 1150, strength: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = np.busday_offset(np.datetime64(FE.START), np.arange(n_days), roll="forward").astype("datetime64[D]")
    T, S = n_days, n_stocks
    vol = np.full((T, S), 1e6)
    flows = {k: rng.normal(0.0, 1e5, (T, S)) for k in TF.FLOW_KEYS}
    flows["total"] = flows["foreign"] + flows["trust"] + flows["dealer"]
    proto = _datas(dates, np.zeros((T, S)), vol)
    panel0 = FE.Panel(proto, [f"{1000 + j}" for j in range(S)])
    sig = FE.flow_signal(panel0, flows["foreign"], 20)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # the first rows have no signal yet (all NaN)
        mu_cs = np.nanmean(sig, axis=1, keepdims=True)
        sd_cs = np.nanstd(sig, axis=1, keepdims=True)
        z = np.nan_to_num((sig - mu_cs) / np.where(sd_cs > 0, sd_cs, 1.0))
    mkt = rng.normal(0.0003, 0.01, (T, 1))
    r = mkt + rng.normal(0.0, 0.02, (T, S))
    r[1:] += strength * z[:-1]                                  # signal known at t drives the return of t+1
    datas = _datas(dates, r, vol)
    codes = [f"{1000 + j}" for j in range(S)]
    panel = FE.Panel(datas, codes)
    half = S // 2
    dev = np.arange(S) < half
    return panel, flows, dev, ~dev


def _datas(dates, r, vol):
    T, S = r.shape
    out = []
    for j in range(S):
        c = 100.0 * np.exp(np.cumsum(r[:, j]))
        o = np.concatenate([[100.0], c[:-1]])                   # open = previous close (no overnight gap)
        hi, lo = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
        out.append(PriceData(f"{1000 + j}", f"S{j}", "TWSE", "stock", dates.copy(), o, hi, lo, c, c.copy(), vol[:, j].copy()))
    return out


def rates(strength: float, reps: int, seed0: int = 1000) -> dict:
    holds = {f"{s} h={h}": 0 for s in FE.SIGNALS for h in FE.HORIZONS}
    any_hold = any_profit = 0
    for k in range(reps):
        panel, flows, dev, hold = make_world(strength=strength, seed=seed0 + k)
        rep = FE.run(panel, flows, dev, hold)
        hit = False
        for r in rep["results"]:
            if r["signal_holds"]:
                holds[f"{r['signal']} h={r['h']}"] += 1
                hit = True
        any_hold += hit
        any_profit += rep["verdict"] == "pass"
    return {"strength": strength, "reps": reps, "any_signal_holds": any_hold / reps, "verdict_pass": any_profit / reps,
            "per_test": {k: v / reps for k, v in holds.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args(argv)
    for s in (0.0, 0.0005, 0.001, 0.002):
        print(rates(s, args.reps), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
