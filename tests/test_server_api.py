"""HTTP API: every endpoint against a real in-process server with a fake network and a fake quant engine."""
import json
import os
import sys
import threading
import time
import unittest
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from server_fakes import ServerCase, price_series, series_dates  # noqa: E402
from data_fakes import TAIPEI  # noqa: E402

import schema_check as sc  # noqa: E402
from backend import config, data, jobs  # noqa: E402


class Status(ServerCase):
    def test_status_ready_and_schema(self):
        status, d, hdr = self.json("/api/status", 200)
        self.assertEqual(sc.check_status(d), [])
        self.assertEqual(d["version"], "1.0")
        self.assertEqual(d["last_trading_date"], "2026-10-22")  # Thursday after the close
        self.assertFalse(d["market_open"])
        self.assertEqual(hdr["content-type"], "application/json; charset=utf-8")
        self.assertEqual(hdr["cache-control"], "no-store")
        self.assertTrue(d["taipei_time"].endswith("+08:00"))

    def test_market_open_flag_follows_the_clock(self):
        self.clock.dt = datetime(2026, 10, 22, 10, 30, tzinfo=TAIPEI)
        self.assertTrue(self.json("/api/status")[1]["market_open"])
        self.clock.dt = datetime(2026, 10, 24, 10, 30, tzinfo=TAIPEI)  # Saturday
        d = self.json("/api/status")[1]
        self.assertFalse(d["market_open"])
        self.assertEqual(d["last_trading_date"], "2026-10-23")  # Friday

    def test_trailing_slash_and_head(self):
        self.assertEqual(self.get("/api/status/")[0], 200)
        st, hdr, body = self.request("/api/status", "HEAD")
        self.assertEqual((st, body), (200, b""))
        self.assertGreater(int(hdr["content-length"]), 100)


class Search(ServerCase):
    def test_chinese_name_substring(self):
        d = self.json("/api/search?q=%E5%8F%B0%E7%A9%8D", 200)[1]  # 台積
        self.assertEqual(sc.check_search(d, "2330"), [])
        self.assertEqual(d["results"][0], {"code": "2330", "name": "台積電", "market": "TWSE", "kind": "stock"})

    def test_raw_utf8_query_like_curl_sends_it(self):
        raw = self.raw_exchange("GET /api/search?q=台積 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode("utf-8"))
        body = raw.split(b"\r\n\r\n", 1)[1]
        d = sc.strict_loads(body)
        self.assertEqual([r["code"] for r in d["results"]], ["2330"])

    def test_code_prefix_and_etf_and_otc(self):
        d = self.json("/api/search?q=00", 200)[1]
        self.assertEqual([r["code"] for r in d["results"]], ["0050"])
        self.assertEqual(d["results"][0]["kind"], "etf")
        d = self.json("/api/search?q=6488", 200)[1]
        self.assertEqual(d["results"][0]["market"], "TPEx")
        self.assertEqual(self.json("/api/search?q=zzzz", 200)[1]["results"], [])

    def test_empty_query_returns_popular_list(self):
        d = self.json("/api/search", 200)[1]
        self.assertEqual(sc.check_search(d), [])
        codes = [r["code"] for r in d["results"]]
        self.assertEqual(codes[:5], ["1101", "1102", "1216", "2002", "2330"])  # the (patched) universe first
        self.assertIn("0050", codes)
        self.assertEqual(self.json("/api/search?q=", 200)[1]["results"], d["results"])

    def test_limit_and_bad_params(self):
        d = self.json("/api/search?q=1&limit=2", 200)[1]
        self.assertLessEqual(len(d["results"]), 2)
        for bad in ("limit=0", "limit=abc", "limit=999", "q=" + "x" * 100):
            st, d, _ = self.json("/api/search?" + bad)
            self.assertEqual((st, d["error"]["code"]), (400, "bad_request"), bad)
            self.assertEqual(sc.check_error(d), [])


class History(ServerCase):
    def test_history_happy_path(self):
        d = self.json("/api/history?symbol=2330&limit=120", 200)[1]
        self.assertEqual(sc.check_history(d, 120), [])
        self.assertEqual(len(d["bars"]), 120)
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertEqual((d["symbol"], d["name"], d["market"], d["kind"]), ("2330", "台積電", "TWSE", "stock"))
        self.assertTrue(d["last_bar_complete"])
        self.assertFalse(d["stale"])
        self.assertEqual(d["n_total"], self.N_BARS)
        # raw unadjusted OHLC exactly as served by (fake) Yahoo
        want = price_series(self.N_BARS, seed=10 + 4, start=50 + 40 * 4)
        self.assertAlmostEqual(d["bars"][-1]["c"], float(want[-1]), places=4)

    def test_default_limit_is_750(self):
        d = self.json("/api/history?symbol=2330", 200)[1]
        self.assertEqual(len(d["bars"]), 750)

    def test_dividends_are_listed(self):
        ds = series_dates(400)
        self.route_symbol("2330.TW", ds, price_series(400, 1), divs=[(ds[-50], 6.0), (ds[-300], 3.0)])
        d = self.json("/api/history?symbol=2330&limit=100", 200)[1]
        self.assertEqual(d["dividends"], [{"d": str(ds[-50]), "amt": 6.0}])  # only dividends inside the window
        self.assertEqual(sc.check_history(d, 100), [])

    def test_otc_and_etf_and_index_and_name_lookup(self):
        self.assertEqual(self.json("/api/history?symbol=6488&limit=10", 200)[1]["market"], "TPEx")
        self.assertEqual(self.json("/api/history?symbol=0050&limit=10", 200)[1]["kind"], "etf")
        d = self.json("/api/history?symbol=%5ETWII&limit=10", 200)[1]
        self.assertEqual((d["symbol"], d["market"], d["kind"]), ("^TWII", "INDEX", "index"))
        self.assertEqual(self.json("/api/history?symbol=2330.TW&limit=5", 200)[1]["symbol"], "2330")
        self.assertEqual(self.json("/api/history?symbol=%E5%8F%B0%E7%A9%8D%E9%9B%BB&limit=5", 200)[1]["symbol"], "2330")  # 台積電

    def test_intraday_bar_is_flagged(self):
        self.clock.dt = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)
        d = self.json("/api/history?symbol=2330&limit=10", 200)[1]
        self.assertFalse(d["last_bar_complete"])
        self.assertEqual(d["asof"], "2026-10-22")

    def test_errors(self):
        self.route_world(extra_codes=("9999",))
        for path, status, code in (("/api/history?symbol=9999", 404, "unknown_symbol"),
                                   ("/api/history?symbol=..%2Fetc", 400, "bad_request"),
                                   ("/api/history?symbol=abc", 400, "bad_request"),
                                   ("/api/history", 400, "bad_request"),
                                   ("/api/history?symbol=2330&limit=0", 400, "bad_request"),
                                   ("/api/history?symbol=2330&limit=x", 400, "bad_request"),
                                   ("/api/history?symbol=2330&limit=99999999", 400, "bad_request")):
            st, d, _ = self.json(path)
            self.assertEqual((st, d.get("error", {}).get("code")), (status, code), path)
            self.assertEqual(sc.check_error(d), [], path)

    def test_network_down_without_cache_is_data_unavailable(self):
        self.net.route("/chart/2412.TW?", data.DataUnavailable("down"))
        st, d, hdr = self.json("/api/history?symbol=2412")
        self.assertEqual((st, d["error"]["code"]), (503, "data_unavailable"))
        self.assertEqual(hdr.get("retry-after"), "5")


class Forecast(ServerCase):
    def test_forecast_happy_path_and_schema(self):
        st, d, hdr = self.json("/api/forecast?symbol=2330", 200)
        self.assertEqual(sc.check_forecast(d), [], sc.check_forecast(d))
        self.assertEqual((d["symbol"], d["name"], d["market"], d["kind"]), ("2330", "台積電", "TWSE", "stock"))
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertEqual(d["next_trading_day"], "2026-10-23")  # Friday
        self.assertTrue(d["asof_complete"])
        self.assertFalse(d["short_history"])
        self.assertFalse(d["stale"])
        for hz in d["horizons"].values():
            self.assertIn(hz["signal"], ("buy", "hold", "avoid"))  # added by evaluation.recommend.annotate_forecast
            self.assertIn(hz["confidence"], ("高", "中", "低"))
        self.assertEqual(self.engine.calls["forecast:2330"], 1)

    def test_next_trading_day_skips_the_holiday_weekend(self):
        # asof Friday 2026-10-23, then Sat/Sun and the TWSE holiday Monday 10-26 -> next trading day is Tue 10-27
        end = datetime(2026, 10, 23).date()
        ds = series_dates(900, end)
        self.clock.dt = datetime(2026, 10, 23, 16, 0, tzinfo=TAIPEI)
        self.route_symbol("2330.TW", ds, price_series(900, 5))
        self.route_symbol("%5ETWII", ds, price_series(900, 6, 15000), volume=[0] * 900)
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-23")
        self.assertEqual(d["next_trading_day"], "2026-10-27")

    def test_forecast_is_cached_per_symbol_and_last_bar(self):
        self.json("/api/forecast?symbol=2330", 200)
        n = self.engine.calls["forecast:2330"]
        for _ in range(3):
            self.json("/api/forecast?symbol=2330", 200)
        self.assertEqual(self.engine.calls["forecast:2330"], n)  # served from the cache
        # a new bar appears next day -> the cache key changes -> recompute
        self.clock.dt = datetime(2026, 10, 23, 16, 0, tzinfo=TAIPEI)
        self.clock.mono += 3600
        ds = series_dates(901, datetime(2026, 10, 23).date())
        self.route_symbol("2330.TW", ds, price_series(901, 10 + 4, start=50 + 40 * 4))
        self.route_symbol("%5ETWII", ds, price_series(901, 99, 15000), volume=[0] * 901)
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-23")
        self.assertEqual(self.engine.calls["forecast:2330"], n + 1)

    def test_forecast_cache_survives_a_restart_via_disk(self):
        self.json("/api/forecast?symbol=2330", 200)
        n = self.engine.calls["forecast:2330"]
        self.compute._forecast_mem.clear()  # forget the memory copy: the JSON file under data_cache/forecast remains
        self.assertTrue(os.listdir(os.path.join(self.tmp, "forecast")))
        self.json("/api/forecast?symbol=2330", 200)
        self.assertEqual(self.engine.calls["forecast:2330"], n)

    def test_concurrent_identical_forecasts_compute_once(self):
        self.engine.gate = threading.Event()
        out = []

        def go():
            out.append(self.json("/api/forecast?symbol=2330")[0])

        ts = [threading.Thread(target=go) for _ in range(8)]
        [t.start() for t in ts]
        time.sleep(0.4)
        self.engine.gate.set()
        [t.join(30) for t in ts]
        self.assertEqual(out, [200] * 8)
        self.assertEqual(self.engine.calls["forecast:2330"], 1)

    def test_unknown_symbol_and_bad_symbol(self):
        self.route_world(extra_codes=("9999",))
        st, d, _ = self.json("/api/forecast?symbol=9999")
        self.assertEqual((st, d["error"]["code"]), (404, "unknown_symbol"))
        self.assertIn("9999", d["error"]["message"])
        for bad in ("", "abc", "12", "../../etc/passwd", "2330;DROP", "%00"):
            st, d, _ = self.json("/api/forecast?symbol=" + bad)
            self.assertEqual((st, d["error"]["code"]), (400, "bad_request"), bad)

    def test_insufficient_history(self):
        self.route_short("3000", 150)
        st, d, _ = self.json("/api/forecast?symbol=3000")
        self.assertEqual((st, d["error"]["code"]), (422, "insufficient_history"))
        self.assertIn("150", d["error"]["message"])
        self.assertEqual(self.engine.calls["forecast"], 0)  # refused before any model work

    def test_short_history_flag(self):
        self.route_short("3001", 600)
        d = self.json("/api/forecast?symbol=3001", 200)[1]
        self.assertTrue(d["short_history"])
        self.assertEqual(sc.check_forecast(d), [])
        self.assertTrue(any("歷史資料" in x for x in d["notes"]))

    def test_intraday_bar_is_dropped_from_the_model(self):
        self.clock.dt = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)  # session still running
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-21")
        self.assertTrue(d["intraday_bar_dropped"])
        self.assertEqual(d["data_through"], "2026-10-22")
        self.assertEqual(d["n_bars"], self.N_BARS - 1)
        self.assertTrue(any("盤中" in x for x in d["notes"]))
        self.assertEqual(d["next_trading_day"], "2026-10-22")

    def test_stale_cache_flag_when_network_fails(self):
        self.json("/api/forecast?symbol=2330", 200)
        # next trading day: Yahoo is down, the cache is one bar behind
        self.clock.dt = datetime(2026, 10, 23, 16, 0, tzinfo=TAIPEI)
        self.clock.mono += 3600
        self.net.down("/chart/")
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertTrue(d["stale"])
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertTrue(any("快取" in x for x in d["notes"]))
        self.assertEqual(sc.check_forecast(d), [])
        h = self.json("/api/history?symbol=2330&limit=5", 200)[1]
        self.assertTrue(h["stale"])

    def test_market_data_down_still_forecasts_without_market(self):
        self.net.route("/chart/%5ETWII?", data.DataUnavailable("down"))
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertTrue(any("大盤" in x for x in d["notes"]))

    def test_nan_and_inf_become_null(self):
        self.engine.nan_driver = True
        st, hdr, raw = self.get("/api/forecast?symbol=2330")
        self.assertEqual(st, 200)
        self.assertNotIn(b"NaN", raw)
        self.assertNotIn(b"Infinity", raw)
        d = sc.strict_loads(raw)  # strict parser: would raise on NaN / Infinity tokens
        self.assertIsNone(d["drivers"][0]["value"])
        self.assertIsNone(d["vol"]["percentile"])

    def test_engine_failure_is_a_clean_500(self):
        self.engine.fail_for.add("2330")
        st, d, _ = self.json("/api/forecast?symbol=2330")
        self.assertEqual((st, d["error"]["code"]), (500, "internal"))
        self.assertEqual(sc.check_error(d), [])
        self.assertNotIn("boom", d["error"]["message"])  # internals do not leak into the user-facing message
        self.assertEqual(self.json("/api/status", 200)[0], 200)  # and the server lives on

    def test_index_forecast(self):
        d = self.json("/api/forecast?symbol=%5ETWII", 200)[1]
        self.assertEqual((d["symbol"], d["kind"]), ("^TWII", "index"))


class Recommendations(ServerCase):
    def test_computing_then_ready_with_progress(self):
        self.engine.gate = threading.Event()
        st, d, _ = self.json("/api/recommendations?h=5", 200)
        self.assertEqual(d["status"], "computing")
        self.assertEqual(sc.check_envelope(d), [])
        self.assertIn("progress", d)
        time.sleep(0.3)
        d1 = self.json("/api/recommendations?h=5")[1]
        self.assertEqual(d1["status"], "computing")
        self.assertGreater(d1["progress"]["total"], 0)
        self.assertGreater(d1["progress"]["done"], 0)  # downloads advanced while the engine is paused
        self.engine.gate.set()
        st, d, trace = self.poll("/api/recommendations?h=5")
        self.assertEqual(st, 200)
        self.assertEqual(sc.check_recommendations(d, 5, 10), [], sc.check_recommendations(d, 5, 10))
        self.assertEqual(d["universe_size"], 5)
        self.assertEqual(d["n_scored"], 5)
        self.assertEqual(d["horizon"], 5)
        self.assertEqual(self.engine.calls["forecast"], 5 + 1)  # 5 stocks + the market block

    def test_progress_advances_per_stock(self):
        self.engine.delay = 0.05
        seen = set()
        deadline = time.time() + 30
        while time.time() < deadline:
            d = self.json("/api/recommendations?h=1")[1]
            if d["status"] != "computing":
                break
            seen.add(d["progress"]["done"])
            time.sleep(0.01)
        self.assertEqual(d["status"], "ready")
        self.assertGreaterEqual(len(seen), 3, f"progress barely moved: {sorted(seen)}")

    def test_top_parameter_and_other_horizons_reuse_forecast_cores(self):
        st, d, _ = self.poll("/api/recommendations?h=5&top=2")
        self.assertEqual(sc.check_recommendations(d, 5, 2), [])
        n = self.engine.calls["forecast"]
        st, d1, _ = self.poll("/api/recommendations?h=1&top=3")
        st, d20, _ = self.poll("/api/recommendations?h=20")
        self.assertEqual((d1["horizon"], d20["horizon"]), (1, 20))
        self.assertEqual(self.engine.calls["forecast"], n)  # all horizons live in one cached forecast core

    def test_identical_concurrent_requests_share_one_job(self):
        self.engine.gate = threading.Event()
        bodies = []

        def go():
            bodies.append(self.json("/api/recommendations?h=5")[1]["status"])

        ts = [threading.Thread(target=go) for _ in range(10)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(bodies, ["computing"] * 10)
        self.assertEqual(self.compute.jobs.stats()["computing"], 1)  # ONE job, not ten
        self.engine.gate.set()
        self.poll("/api/recommendations?h=5")
        self.assertEqual(self.engine.calls["forecast"], 6)

    def test_ready_result_is_served_from_memory(self):
        self.poll("/api/recommendations?h=5")
        n = self.engine.calls["forecast"]
        t0 = time.time()
        st, d, _ = self.json("/api/recommendations?h=5", 200)
        self.assertEqual(d["status"], "ready")
        self.assertLess(time.time() - t0, 0.5)
        self.assertEqual(self.engine.calls["forecast"], n)

    def test_bad_parameters(self):
        for bad in ("h=7", "h=abc", "h=0", "top=0", "top=999", "top=x"):
            st, d, _ = self.json("/api/recommendations?" + bad)
            self.assertEqual((st, d["error"]["code"]), (400, "bad_request"), bad)

    def test_a_failing_stock_is_skipped_not_fatal(self):
        self.engine.fail_for.add("1216")
        st, d, _ = self.poll("/api/recommendations?h=5")
        self.assertEqual(st, 200)
        self.assertEqual(d["n_scored"], 4)
        self.assertEqual([s["code"] for s in d["skipped"]], ["1216"])
        self.assertTrue(any("略過" in n for n in d["notes"]))
        self.assertEqual(sc.check_recommendations(d, 5, 10), [])

    def test_laggard_with_old_last_bar_is_skipped(self):
        ds = series_dates(899, datetime(2026, 10, 21).date())  # this stock stops one day early
        self.route_symbol("2002.TW", ds, price_series(899, 77))
        st, d, _ = self.poll("/api/recommendations?h=5")
        self.assertEqual([s["code"] for s in d["skipped"]], ["2002"])
        self.assertIn("未與其他標的同步", d["skipped"][0]["reason"])

    def test_stale_flag_propagates(self):
        self.poll("/api/recommendations?h=5")
        self.clock.dt = datetime(2026, 10, 23, 16, 0, tzinfo=TAIPEI)
        self.clock.mono += 3600
        self.net.down("/chart/")
        st, d, _ = self.poll("/api/recommendations?h=5")  # new epoch -> new job, network dead -> stale caches
        self.assertEqual(st, 200)
        self.assertTrue(d["stale"])
        self.assertTrue(any("快取" in n for n in d["notes"]))

    def test_all_downloads_failing_is_data_unavailable(self):
        self.net.routes.clear()
        st, d, _ = self.poll("/api/recommendations?h=5")
        self.assertEqual((st, d["error"]["code"]), (503, "data_unavailable"))


class Backtest(ServerCase):
    def test_backtest_happy_path(self):
        st, d, trace = self.poll("/api/backtest?symbol=2330&h=1")
        self.assertEqual(st, 200)
        self.assertEqual(sc.check_backtest(d, 1), [], sc.check_backtest(d, 1))
        rep = d["report"]
        self.assertEqual((rep["symbol"], rep["name"], rep["horizon"]), ("2330", "台積電", 1))
        self.assertEqual(rep["params"]["threshold"], 0.55)
        self.assertEqual(rep["params"]["exec"], "next_open")
        self.assertEqual(rep["params"]["commission_discount"], 0.6)
        self.assertEqual(rep["params"]["slippage_bps"], 5.0)
        self.assertAlmostEqual(rep["params"]["roundtrip_cost"], config.roundtrip_cost("stock"), places=9)
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertEqual(self.engine.calls["wf"], 1)

    def test_walk_forward_reused_across_horizons_and_params(self):
        self.poll("/api/backtest?symbol=2330&h=1")
        for path in ("/api/backtest?symbol=2330&h=5", "/api/backtest?symbol=2330&h=20", "/api/backtest?symbol=2330&h=1&thr=0.6",
                     "/api/backtest?symbol=2330&h=1&exec=close", "/api/backtest?symbol=2330&h=5&sizing=kelly",
                     "/api/backtest?symbol=2330&h=1&disc=1&slip=20"):
            st, d, _ = self.poll(path)
            self.assertEqual(st, 200, path)
            self.assertEqual(sc.check_backtest(d), [], path)
        self.assertEqual(self.engine.calls["wf"], 1)  # ONE walk-forward served seven reports

    def test_params_are_echoed(self):
        st, d, _ = self.poll("/api/backtest?symbol=2330&h=5&thr=0.6&exec=close&disc=0.3&slip=10&sizing=kelly")
        p = d["report"]["params"]
        self.assertEqual((p["threshold"], p["exec"], p["commission_discount"], p["slippage_bps"], p["sizing"]),
                         (0.6, "close", 0.3, 10.0, "kelly"))

    def test_walk_forward_pickle_survives_a_restart(self):
        self.poll("/api/backtest?symbol=2330&h=1")
        self.assertTrue(any(f.endswith(".pkl.gz") for f in os.listdir(os.path.join(self.tmp, "wf"))))
        self.compute._wf_mem.clear()
        self.compute.jobs._jobs.clear()
        self.poll("/api/backtest?symbol=2330&h=5")
        self.assertEqual(self.engine.calls["wf"], 1)

    def test_finished_report_is_cached_on_disk(self):
        self.poll("/api/backtest?symbol=2330&h=1")
        self.compute.jobs._jobs.clear()
        self.compute._wf_mem.clear()
        # remove the pickle: only the finished-report cache can answer without the engine
        for f in os.listdir(os.path.join(self.tmp, "wf")):
            os.remove(os.path.join(self.tmp, "wf", f))
        st, d, _ = self.poll("/api/backtest?symbol=2330&h=1")
        self.assertEqual(st, 200)
        self.assertEqual(self.engine.calls["wf"], 1)

    def test_cached_backtest_is_answered_by_the_first_request(self):
        self.poll("/api/backtest?symbol=2330&h=1")
        self.compute.jobs._jobs.clear()
        st, d, _ = self.json("/api/backtest?symbol=2330&h=1", 200)  # job restarts but the disk cache answers within first_wait
        self.assertEqual(d["status"], "ready")

    def test_identical_requests_share_one_job(self):
        self.engine.gate = threading.Event()
        res = []

        def go():
            res.append(self.json("/api/backtest?symbol=2330&h=1")[1]["status"])

        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(res, ["computing"] * 6)
        self.engine.gate.set()
        self.poll("/api/backtest?symbol=2330&h=1")
        self.assertEqual(self.engine.calls["wf"], 1)

    def test_progress_reports_walk_forward_steps(self):
        self.engine.delay = 0.9  # longer than one 0.4 s polling round of the pool, so a progress read is guaranteed
        st, d, trace = self.poll("/api/backtest?symbol=2330&h=1", every=0.01)
        msgs = {t["message"] for t in trace}
        self.assertTrue(len(msgs) >= 2, msgs)
        self.assertTrue(any("前推式" in m for m in msgs))

    def test_errors(self):
        self.route_world(extra_codes=("9999",))
        self.route_short("3000", 300)
        cases = [("/api/backtest?symbol=9999&h=1", 404, "unknown_symbol"),
                 ("/api/backtest?symbol=3000&h=1", 422, "insufficient_history"),
                 ("/api/backtest?symbol=../x", 400, "bad_request"),
                 ("/api/backtest", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&h=3", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&h=x", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&thr=2", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&thr=nan", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&thr=abc", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&exec=magic", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&sizing=yolo", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&disc=-1", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&slip=1000", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&years=0", 400, "bad_request"),
                 ("/api/backtest?symbol=2330&years=99", 400, "bad_request")]
        for path, status, code in cases:
            st, d, _ = self.poll(path)
            self.assertEqual((st, d.get("error", {}).get("code")), (status, code), path)
            self.assertEqual(sc.check_error(d), [], path)
        self.assertEqual(self.engine.calls["wf"], 0)

    def test_error_results_are_not_cached_forever(self):
        self.engine.fail_for.add("2330")
        st, d, _ = self.poll("/api/backtest?symbol=2330&h=1")
        self.assertEqual((st, d["error"]["code"]), (500, "internal"))
        self.engine.fail_for.clear()
        self.compute.jobs.error_ttl = 0.0  # as if the 15 s error memory had elapsed
        st, d, _ = self.poll("/api/backtest?symbol=2330&h=1")
        self.assertEqual(st, 200)

    def test_years_changes_the_test_window(self):
        st, d5, _ = self.poll("/api/backtest?symbol=2330&h=1&years=5")
        st, d1, _ = self.poll("/api/backtest?symbol=2330&h=1&years=1")
        self.assertGreater(d5["report"]["period"]["n_forecasts"], d1["report"]["period"]["n_forecasts"])
        self.assertEqual(self.engine.calls["wf"], 2)

    def test_etf_uses_the_etf_tax(self):
        st, d, _ = self.poll("/api/backtest?symbol=0050&h=1")
        self.assertAlmostEqual(d["report"]["params"]["roundtrip_cost"], config.roundtrip_cost("etf"), places=9)


class UniverseBacktest(ServerCase):
    def test_universe_happy_path(self):
        st, d, trace = self.poll("/api/backtest/universe?h=5&top=2")
        self.assertEqual(st, 200)
        self.assertEqual(sc.check_universe_report(d, 5, 2), [], sc.check_universe_report(d, 5, 2))
        self.assertEqual(d["report"]["period"]["n_stocks"], 5)
        self.assertEqual(self.engine.calls["wf"], 5)
        self.assertEqual(sorted(self.engine.wf_symbols), ["1101", "1102", "1216", "2002", "2330"])

    def test_progress_is_reported_per_stock(self):
        self.engine.delay = 0.05
        st, d, trace = self.poll("/api/backtest/universe?h=5", every=0.01)
        steps = sorted({t["done"] for t in trace})
        self.assertGreaterEqual(len(steps), 4, steps)
        self.assertTrue(any("前推式預測" in t["message"] for t in trace))
        dones = [t["done"] for t in trace]
        self.assertEqual(dones, sorted(dones))  # never goes backwards

    def test_shares_walk_forward_pickles_with_the_single_stock_backtest(self):
        self.poll("/api/backtest?symbol=2330&h=1")  # default 5 years -> the same test window as the universe validation
        self.assertEqual(self.engine.calls["wf:2330"], 1)
        st, d, _ = self.poll("/api/backtest/universe?h=5")
        self.assertEqual(st, 200)
        self.assertEqual(self.engine.calls["wf:2330"], 1)  # reused, not recomputed
        self.assertEqual(self.engine.calls["wf"], 5)
        # and the other way round: a second horizon / parameter set needs no new walk-forward at all
        self.poll("/api/backtest/universe?h=1&top=3")
        self.assertEqual(self.engine.calls["wf"], 5)

    def test_identical_requests_share_one_job(self):
        self.engine.gate = threading.Event()
        res = []

        def go():
            res.append(self.json("/api/backtest/universe?h=5")[1]["status"])

        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(res, ["computing"] * 6)
        self.engine.gate.set()
        self.poll("/api/backtest/universe?h=5")
        self.assertEqual(self.engine.calls["wf"], 5)

    def test_failing_stocks_are_reported_not_fatal(self):
        self.universe.append({"code": "2303", "name": "聯電", "market": "TWSE", "kind": "stock"})
        self.compute.universe_size = 6
        self.route_symbol("2303.TW", self.ds, price_series(self.N_BARS, 31))
        self.engine.fail_for.add("2303")
        st, d, _ = self.poll("/api/backtest/universe?h=5")
        self.assertEqual(st, 200)
        self.assertEqual([s["code"] for s in d["skipped"]], ["2303"])
        self.assertEqual(d["report"]["period"]["n_stocks"], 5)
        self.assertTrue(any("2303" in n for n in d["report"]["notes"]))

    def test_too_few_stocks_is_an_error(self):
        self.compute.universe_size = 3
        st, d, _ = self.poll("/api/backtest/universe?h=5")
        self.assertEqual((st, d["error"]["code"]), (503, "data_unavailable"))

    def test_bad_parameters(self):
        for bad in ("h=2", "top=0", "top=99", "exec=x", "disc=7", "slip=-1", "disc=nan"):
            st, d, _ = self.json("/api/backtest/universe?" + bad)
            self.assertEqual((st, d["error"]["code"]), (400, "bad_request"), bad)

    def test_exec_and_cost_parameters_change_the_report_key(self):
        a = self.poll("/api/backtest/universe?h=5&exec=next_open")[1]
        b = self.poll("/api/backtest/universe?h=5&exec=close&disc=1&slip=0")[1]
        self.assertEqual(sc.check_universe_report(b), [])
        self.assertNotEqual(a["report"]["top"]["total_return"], b["report"]["top"]["total_return"])
        self.assertEqual(self.engine.calls["wf"], 5)


class Routing(ServerCase):
    def test_unknown_api_path(self):
        st, d, _ = self.json("/api/nope")
        self.assertEqual(st, 404)
        self.assertEqual(sc.check_error(d), [])
        self.assertEqual(self.json("/api")[0], 404)
        self.assertEqual(self.json("/api/")[0], 404)

    def test_methods(self):
        for m in ("POST", "PUT", "DELETE", "PATCH"):
            st, hdr, raw = self.request("/api/status", m, body=b"x=1" if m != "DELETE" else None)
            self.assertEqual(st, 405, m)
            self.assertEqual(sc.check_error(sc.strict_loads(raw)), [])
            self.assertIn("GET", hdr["allow"])
        st, hdr, raw = self.request("/api/status", "OPTIONS")
        self.assertEqual(st, 204)

    def test_json_content_type_and_utf8(self):
        st, hdr, raw = self.get("/api/search?q=%E5%8F%B0%E7%A9%8D")
        self.assertEqual(hdr["content-type"], "application/json; charset=utf-8")
        self.assertIn("台積電".encode("utf-8"), raw)  # real UTF-8, not \\u escapes

    def test_security_headers(self):
        st, hdr, raw = self.get("/api/status")
        self.assertEqual(hdr["x-content-type-options"], "nosniff")
        self.assertIn("default-src 'self'", hdr["content-security-policy"])

    def test_keep_alive_connection_serves_several_requests(self):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            for _ in range(5):
                conn.request("GET", "/api/status")
                r = conn.getresponse()
                self.assertEqual(r.status, 200)
                r.read()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
