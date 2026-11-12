"""P3b runner: US overnight -> gap model -> pre-market update -> tradability (PLAN.md section 7).

DEV (default)::

    python -m research.run_overnight [--limit N | --codes 2330,2317] [--workers 6] [--no-results]

writes ``research/out/overnight_dev.json`` and the marked block ``overnight`` of ``docs/research/RESULTS.md``.  Prices and market are cut at
2022-12-31, the walk-forward is cached per stock in ``<cache>/research/wf_h1/{code}_dev.npz``.  Only data caches are read: no
network, no FinMind.

OOS (ONE look, after review; do not run before the coordinator says so)::

    python -m research.run_overnight --oos --spec-hash <sha256 of research/overnight_spec.json>

Checks that the hash equals the file's sha256, then the harness guard (``period="oos"``) appends the row to ``docs/research/LOOKS.md``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from backend import config
from backend.quant import engine as E

from . import harness as H
from . import overnight as O
from .panel import Panel, load_pool

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DEV = REPO_ROOT / "research" / "out" / "overnight_dev.json"
OUT_OOS = REPO_ROOT / "research" / "out" / "overnight_oos.json"
RESULTS_MD = REPO_ROOT / "docs" / "research" / "RESULTS.md"
MARKERS = {"dev": ("<!-- BEGIN overnight -->", "<!-- END overnight -->"),
           "oos": ("<!-- BEGIN overnight_oos -->", "<!-- END overnight_oos -->")}
DEV_PERIODS = ("dev", "dev_a", "dev_b")


def _wf_job(args):
    code, tag, refresh = args
    try:
        r = O.compute_wf_h1(code, tag, refresh)
        return code, {"ok": True, "cached": r["cached"], "start_idx": r["start_idx"]}, r
    except Exception as exc:  # noqa: BLE001 -- one failing stock must not stop the run
        return code, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None


def compute_all_wf(codes, tag: str, workers: int = 1, refresh: bool = False) -> dict:
    """``{code: wf dict}`` for the codes that succeeded (+ ``info`` per code)."""
    jobs = [(c, tag, refresh) for c in codes]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_wf_job, jobs))
    else:
        res = [_wf_job(j) for j in jobs]
    wfs, info = {}, {}
    for code, inf, r in res:
        info[code] = inf
        if r is not None:
            wfs[code] = r
        else:
            log.warning("%s: walk-forward failed: %s", code, inf.get("error"))
    return wfs, info


def run(period_set: str = "dev", codes=None, limit: int | None = None, workers: int = 1, refresh: bool = False,
        spec_hash: str | None = None, out_json=None, results_md=RESULTS_MD) -> dict:
    """``period_set='dev'`` evaluates dev / dev_a / dev_b; ``'oos'`` evaluates the OOS period (hash-guarded, logged)."""
    oos = period_set == "oos"
    sha = O.spec_sha256()
    if oos:
        if not spec_hash or spec_hash.strip().lower() != sha:
            raise H.OOSGuardError(f"--spec-hash must equal the sha256 of research/overnight_spec.json ({sha})")
        H._check_period("oos", spec_hash)
    end = None if oos else O.DEV_END
    tag = "full" if oos else "dev"
    pool = load_pool()
    codes = list(codes) if codes else O.select_codes(pool, n=limit)
    log.info("stocks (%d): %s", len(codes), ",".join(codes))
    wfs, wf_info = compute_all_wf(codes, tag, workers, refresh)
    us = O.load_us()
    stocks = []
    for code in codes:
        if code not in wfs:
            continue
        real, _ = O.load_clean(code, end)
        stocks.append(O.build_stock(real, us, wfs[code]))
    if not stocks:
        raise RuntimeError("no stock with a walk-forward")
    panel = Panel.from_price_data([s.data for s in stocks], "2012-01-01")
    x_shock = O.shock_series(us, panel.cal)
    meta = {"generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": H._git_commit(),
            "spec_sha256": sha, "n_stocks": len(stocks), "codes": [s.code for s in stocks], "n_requested": len(codes),
            "failed": {c: i for c, i in wf_info.items() if not i["ok"]}, "min_bars_before_dev": O.MIN_BARS_BEFORE_DEV,
            "min_train": config.MIN_TRAIN, "refit_every": config.REFIT_EVERY, "engine_version": str(E.ENGINE_VERSION),
            "panel_start": str(panel.cal[0]), "panel_end": str(panel.cal[-1]), "oos_touched": oos,
            "us_last_date": str(us["SOX"][0][-1]), "tag": tag}
    periods = ("oos",) if oos else DEV_PERIODS
    res = {"meta": meta, "periods": {}}
    for per in periods:
        res["periods"][per] = O.evaluate(stocks, panel, x_shock, per, spec_hash if oos else None)
    out_json = Path(out_json) if out_json else (OUT_OOS if oos else OUT_DEV)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    if results_md is not None:
        b, e = MARKERS["oos" if oos else "dev"]
        O.update_results_md(results_md, O.render_section(res, "oos" if oos else "dev"), b, e)
    return res


def _summary(res: dict) -> None:
    for per, r in res["periods"].items():
        g, p, t = r["gap"], r["pup"], r["tradability"]
        c = p["comparisons"]
        log.info("[%s] gap: R2 vs0=%s vsMean=%s hit=%s (base %s) | pup: Brier engine=%s norm=%s update=%s | dBrier upd-vs-eng=%s t=%s | "
                 "intraday IC=%s t=%s | shock signed=%s bps t=%s", per, g["r2_oos_vs_zero"], g["r2_oos_vs_mean"], g["hit_rate"],
                 g["hit_base_const"], p["variants"]["engine"]["brier"], p["variants"]["norm"]["brier"], p["variants"]["update"]["brier"],
                 c["update_vs_engine"]["d_brier"]["mean"], c["update_vs_engine"]["d_brier"]["t"], t["intraday_ic"]["mean"],
                 t["intraday_ic"]["t"], t["shock"]["excess_signed"]["mean_bps"], t["shock"]["excess_signed"]["t"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3b: US overnight -> gap -> pre-market update (DEV; OOS only with --oos --spec-hash)")
    ap.add_argument("--codes", help="comma-separated codes (default: first 30 of pool300 passing the history rule)")
    ap.add_argument("--limit", type=int, help="number of stocks (default: spec n = 30)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--refresh-wf", action="store_true", help="ignore the walk-forward cache")
    ap.add_argument("--oos", action="store_true", help="evaluate the OOS period (ONE look; needs --spec-hash)")
    ap.add_argument("--spec-hash", default=None, help="sha256 of research/overnight_spec.json (required with --oos)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--results-md", default=None)
    ap.add_argument("--no-results", action="store_true", help="do not touch RESULTS.md")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.oos and not a.spec_hash:
        ap.error("--oos requires --spec-hash")
    res = run("oos" if a.oos else "dev", a.codes.split(",") if a.codes else None, a.limit, a.workers, a.refresh_wf, a.spec_hash,
              a.out_json, None if a.no_results else (a.results_md or RESULTS_MD))
    _summary(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
