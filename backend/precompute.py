"""Warm the caches so the first page load is fast (thin wrapper around :class:`backend.jobs.Compute`).

    python -m backend.precompute                      # universe prices + today's forecasts
    python -m backend.precompute --universe-bt --h 5  # also the universe walk-forwards + validation report
    python -m backend.precompute --symbols 2330,0050 --backtest   # extra symbols: forecast (+ default backtest)

Steps (each can be selected; default = ``--prices --forecasts``):

* ``--prices``        download / refresh daily bars of the TAIEX, the universe and ``--symbols``
* ``--forecasts``     forecast core of every universe name (process pool) and the recommendation list
* ``--universe-bt``   walk-forward of every universe name (about 1250 out-of-sample bars) and the
                      ``/api/backtest/universe`` report for each ``--h`` (alias ``--validate``); the first run takes
                      minutes (about 2 minutes for 60 names on 4 cores)
* ``--backtest``      default single-stock backtest (h=1, 5 years) for each of ``--symbols``

Everything is cached under ``data_cache/`` exactly like the server does, so the server finds it afterwards.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from . import config, data, jobs


def _printer():
    last = [0.0, ""]
    tty = sys.stderr.isatty()

    def progress(done, total, message=None):
        now = time.time()
        msg = f"[{done}/{total}] {message or ''}"
        if tty:
            sys.stderr.write("\r" + msg[:100].ljust(100))
            sys.stderr.flush()
        elif now - last[0] > 5 and msg != last[1]:
            print(msg, file=sys.stderr, flush=True)
        last[0], last[1] = now, msg

    def finish():
        if tty:
            sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.flush()

    return progress, finish


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.precompute", description="預先計算並快取行情、預測與驗證結果。")
    ap.add_argument("--n", type=int, default=config.UNIVERSE_SIZE, help="推薦宇宙的股票檔數（預設 %(default)s）")
    ap.add_argument("--symbols", default="", help="額外代號，逗號分隔，例如 2330,0050")
    ap.add_argument("--prices", action="store_true", help="下載/更新行情")
    ap.add_argument("--forecasts", action="store_true", help="計算預測與今日推薦")
    ap.add_argument("--universe-bt", "--validate", dest="validate", action="store_true",
                    help="計算整體驗證（每檔前推式回測＋推薦名單歷史績效，第一次要數分鐘）")
    ap.add_argument("--backtest", action="store_true", help="為 --symbols 計算預設單檔回測")
    ap.add_argument("--h", default="5", help="整體驗證/推薦的預測天數，逗號分隔（預設 5）")
    ap.add_argument("--top", type=int, default=5, help="整體驗證的 top-k（預設 5）")
    ap.add_argument("--refresh", action="store_true", help="強制重新下載行情（忽略快取新鮮度）")
    ap.add_argument("--workers", type=int, default=None, help="平行行程數（預設 CPU 數 − 1）")
    ap.add_argument("--no-processes", action="store_true", help="以執行緒代替行程（除錯用）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    steps = {"prices": args.prices, "forecasts": args.forecasts, "validate": args.validate, "backtest": args.backtest}
    if not any(steps.values()):
        steps["prices"] = steps["forecasts"] = True
    try:
        horizons = [jobs.validate_horizon(x) for x in args.h.split(",") if x.strip()]
    except jobs.ApiError as exc:
        print(exc.message, file=sys.stderr)
        return 2
    try:
        extra = [data.resolve_symbol(s) for s in args.symbols.split(",") if s.strip()]
    except data.BadSymbol as exc:
        print(f"代號格式不正確：{exc}", file=sys.stderr)
        return 2
    comp = jobs.Compute(workers=args.workers, use_processes=not args.no_processes)
    comp.universe_size = max(1, args.n)
    progress, finish = _printer()
    t0 = time.time()
    try:
        data.warm_up()
        universe = data.get_universe(args.n)
        print(f"宇宙 {len(universe)} 檔：{', '.join(u['code'] for u in universe[:12])} …")
        if steps["prices"]:
            items = universe + [{"code": c, "name": c} for c in extra if c not in {u["code"] for u in universe}]
            if args.refresh:
                data.get_market_data(refresh=True)
            ok, bad = comp.load_universe_series(items, progress, 0, len(items), refresh=args.refresh)
            comp.market_series()
            finish()
            print(f"行情：成功 {len(ok)} 檔，失敗 {len(bad)} 檔 {sorted(bad)[:10] if bad else ''}  ({time.time() - t0:.1f}s)")
        if steps["forecasts"]:
            t1 = time.time()
            res = None
            for h in horizons:
                res = comp._run_recommendations(h, 10, progress)
                finish()
                print(f"預測 h={h}：評分 {res.get('n_scored')} 檔，買進候選 {len(res.get('buys', []))}，"
                      f"避開 {len(res.get('avoids', []))}，略過 {len(res.get('skipped', []))}  ({time.time() - t1:.1f}s)")
            for c in extra:
                comp.forecast_payload(c)
                print(f"預測 {c} 完成")
        if steps["backtest"]:
            for c in extra:
                t1 = time.time()
                p = jobs.build_backtest_params(c)
                comp._run_backtest(p, progress)
                finish()
                print(f"回測 {c} 完成 ({time.time() - t1:.1f}s)")
        if steps["validate"]:
            for h in horizons:
                t1 = time.time()
                rep = comp._run_universe_backtest(h, args.top, "next_open", config.COMMISSION_DISCOUNT,
                                                  config.SLIPPAGE_BPS, progress)
                finish()
                v = (rep.get("report") or {}).get("verdict") or {}
                print(f"整體驗證 h={h}：{v.get('title', '')}  ({time.time() - t1:.1f}s)")
        print(f"全部完成，用時 {time.time() - t0:.1f} 秒。")
        return 0
    except KeyboardInterrupt:
        finish()
        print("\n已中斷。", file=sys.stderr)
        return 130
    except jobs.ApiError as exc:
        finish()
        print(f"失敗：{exc.message}", file=sys.stderr)
        return 1
    finally:
        comp.close()


if __name__ == "__main__":
    sys.exit(main())
