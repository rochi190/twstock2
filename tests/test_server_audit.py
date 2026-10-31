"""Audit fixes of the server layer, end to end through the real in-process HTTP server (fake network, fake engine, real
evaluation package): C-04 (lost last bar -> as-of reference), C-05 (cache keys), P-01 (closure -> as-of / next trading
day), P-14 (degenerate / malformed upstream data), the evidence block of ``/api/recommendations`` (DESIGN 9.4) and the
complete popular list of ``/api/search``."""
import copy
import json
import os
import sys
import time
import unittest
from datetime import date, datetime
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from server_fakes import ServerCase, UNIVERSE, price_series, series_dates  # noqa: E402
from data_fakes import TAIPEI, bdays, ts_of, yahoo_payload  # noqa: E402

import schema_check as sc  # noqa: E402
from backend import config, data, jobs  # noqa: E402


# ============================================================================================ search (mandatory task 2)
class PopularSearch(ServerCase):
    CODES = [c for c, _ in UNIVERSE]

    def test_empty_query_is_complete_for_every_limit(self):
        want = None
        for q in ("/api/search", "/api/search?q=", "/api/search?limit=1", "/api/search?limit=3", "/api/search?q=&limit=5",
                  "/api/search?limit=14", "/api/search?limit=20", "/api/search?limit=50", "/api/search?q=%20%20&limit=2"):
            d = self.json(q, 200)[1]
            self.assertEqual(sc.check_popular(d, self.CODES), [], q)
            want = want or d["results"]
            self.assertEqual(d["results"], want, q)  # the same list whatever limit the client asked for
        codes = [r["code"] for r in want]
        self.assertEqual(codes, self.CODES + ["0050", "0056", "00878"])  # 5 (patched) universe names + the 3 ETFs

    def test_real_universe_gives_twelve_plus_three(self):
        # the unpatched data layer: 12 curated names first, then the three ETFs -> 15 rows even for limit=1
        with mock.patch.object(data, "get_universe", lambda n=config.UNIVERSE_SIZE, refresh=False: [
                {"code": c, "name": nm, "market": "TWSE", "kind": "stock"} for c, nm in data.FALLBACK_UNIVERSE[:n]]):
            d = self.json("/api/search?limit=1", 200)[1]
        codes = [r["code"] for r in d["results"]]
        self.assertEqual(len(codes), 15)
        self.assertEqual(codes[:12], [c for c, _ in data.FALLBACK_UNIVERSE[:12]])
        self.assertEqual(codes[12:], ["0050", "0056", "00878"])
        self.assertEqual(sc.check_search(d), [])

    def test_a_real_query_still_honours_the_limit(self):
        d = self.json("/api/search?q=1&limit=2", 200)[1]
        self.assertLessEqual(len(d["results"]), 2)
        d = self.json("/api/search?q=%E5%8F%B0&limit=1", 200)[1]  # 台
        self.assertEqual(len(d["results"]), 1)


# ============================================================================================ evidence (mandatory task 1)
class Evidence(ServerCase):
    def recs(self, h=5, top=10, expect_ready=True):
        st, d, _ = self.poll(f"/api/recommendations?h={h}&top={top}")
        self.assertEqual(st, 200)
        if expect_ready:
            self.assertEqual(sc.check_recommendations(d, h, top), [], sc.check_recommendations(d, h, top))
        return d

    def validate(self, h=5, extra=""):
        st, d, _ = self.poll(f"/api/backtest/universe?h={h}&top=5{extra}")
        self.assertEqual(st, 200)
        self.assertEqual(sc.check_universe_report(d, h, 5), [], sc.check_universe_report(d, h, 5))
        return d

    def test_no_validation_means_source_none_and_the_heavy_job_is_not_started(self):
        d = self.recs(5)
        self.assertEqual(d["evidence"]["source"], "none")
        self.assertEqual(d["evidence"]["horizon"], 5)
        self.assertEqual(self.engine.calls["wf"], 0)  # the picks never trigger the universe validation
        self.assertEqual(self.compute.jobs.stats()["computing"], 0)
        self.assertTrue(any("尚未取得" in n for n in d["notes"]))

    def test_validation_of_the_same_horizon_becomes_the_evidence(self):
        before = self.recs(5)
        self.assertEqual(before["evidence"]["source"], "none")
        rep = self.validate(5)["report"]
        n_fc = self.engine.calls["forecast"]
        d = self.recs(5)  # a validation finished AFTER the list was built: the next request shows it (new job, cached cores)
        ev = d["evidence"]
        self.assertEqual(sc.check_evidence(ev, 5), [])
        self.assertEqual(ev["source"], "universe_validation")
        self.assertEqual(ev["horizon"], 5)
        self.assertEqual(ev["verdict_level"], rep["verdict"]["level"])
        self.assertAlmostEqual(ev["ic"], rep["ic"]["mean"], places=9)
        self.assertEqual(ev["hit_rate"], rep["picks"]["hit_rate"])
        self.assertEqual(ev["base_rate"], rep["picks"]["base_rate"])
        self.assertEqual(ev["data_asof"], rep["period"]["end"])
        self.assertEqual(ev["top_k"], 5)
        self.assertTrue(ev["current"])  # validated on the same last bar as the list
        for k in ("net_cagr", "ew_cagr", "sigma_cagr", "beats_ew_after_costs"):  # evaluation.universe_bt.evidence_from_report
            self.assertIn(k, ev)
        self.assertTrue(any("推薦整體歷史驗證（" in n for n in d["notes"]), d["notes"])
        self.assertEqual(self.engine.calls["forecast"], n_fc)  # no model work for the new list: every core was cached
        self.assertEqual(self.engine.calls["wf"], 5)  # and still no extra walk-forward

    def test_other_horizons_are_not_evidence(self):
        self.validate(5)
        self.assertEqual(self.recs(1)["evidence"]["source"], "none")
        self.assertEqual(self.recs(20)["evidence"]["source"], "none")
        self.validate(1)
        d1 = self.recs(1)
        self.assertEqual((d1["evidence"]["source"], d1["evidence"]["horizon"]), ("universe_validation", 1))
        self.assertEqual(self.recs(5)["evidence"]["horizon"], 5)

    def test_non_default_costs_or_execution_are_not_evidence(self):
        self.validate(5, "&disc=1&slip=0")
        self.assertEqual(self.recs(5)["evidence"]["source"], "none")
        self.validate(5, "&exec=close")
        self.assertEqual(self.recs(5)["evidence"]["source"], "none")
        self.validate(5)
        self.assertEqual(self.recs(5)["evidence"]["source"], "universe_validation")

    def test_a_validation_from_another_model_version_is_ignored(self):
        self.validate(5)
        self.assertEqual(self.recs(5)["evidence"]["source"], "universe_validation")
        self.compute.jobs._jobs.clear()
        with mock.patch.object(jobs, "_fp_all", return_value="0123456789"):  # as if quant/ or config.py changed
            self.assertEqual(self.compute.universe_evidence(5)["source"], "none")

    def test_an_old_validation_is_ignored(self):
        self.validate(5)
        for f in os.listdir(os.path.join(self.tmp, "results")):
            if f.startswith("universe_backtest_"):
                old = time.time() - (jobs.EVIDENCE_MAX_AGE_DAYS + 1) * 86400
                os.utime(os.path.join(self.tmp, "results", f), (old, old))
        self.assertEqual(self.compute.universe_evidence(5)["source"], "none")

    def test_a_corrupt_result_file_is_not_evidence_and_not_an_error(self):
        self.validate(5)
        for f in os.listdir(os.path.join(self.tmp, "results")):
            if f.startswith("universe_backtest_"):
                with open(os.path.join(self.tmp, "results", f), "w") as fh:
                    fh.write('{"report": [')
        self.compute._evidence_memo.clear()
        self.assertEqual(self.compute.universe_evidence(5)["source"], "none")
        self.assertEqual(self.recs(5)["evidence"]["source"], "none")

    def test_the_newest_of_several_validations_wins(self):
        self.validate(5)
        first = self.compute.universe_evidence(5)
        time.sleep(1.1)  # computed_at has a one second resolution
        self.compute.jobs._jobs.clear()
        self.clock.advance(minutes=5)
        self.validate(5, "&top=3")  # a second default-cost validation with another k (separate result file)
        newer = self.compute.universe_evidence(5)
        self.assertGreaterEqual(newer["computed_at"], first["computed_at"])

    def test_evidence_survives_a_server_restart(self):
        self.validate(5)
        want = self.compute.universe_evidence(5)
        self.compute._evidence_memo.clear()
        self.compute.jobs._jobs.clear()
        self.assertEqual(self.compute.universe_evidence(5), want)

    def test_cached_validation_payload_keeps_its_api_shape(self):
        # the bookkeeping `_meta` of the cache file never reaches the client; `params` (additive) does
        d = self.validate(5)
        self.assertNotIn("_meta", d)
        self.assertEqual(d["params"], {"h": 5, "top": 5, "exec": "next_open", "disc": 0.6, "slip": 5.0})
        self.compute.jobs._jobs.clear()
        d2 = self.poll("/api/backtest/universe?h=5&top=5")[1]
        self.assertNotIn("_meta", d2)
        self.assertEqual(d2["report"], d["report"])


# ============================================================================================ recommendations: new contract keys
class RecommendationContract(ServerCase):
    def test_new_keys_reach_the_client(self):
        st, d, _ = self.poll("/api/recommendations?h=5&top=3")
        self.assertEqual(sc.check_recommendations(d, 5, 3), [])
        for k in ("leaders", "laggards", "laggards_all_above_half", "n_buy_total", "rank_key", "rank_sigma_corr",
                  "universe_sigma_median", "evidence"):
            self.assertIn(k, d)
        self.assertEqual(d["rank_key"], "rank_signal")
        r = d["all"][0]
        for k in ("rank_signal", "score_rank_pct", "lean", "edge_covers_cost", "low_vol", "sigma"):
            self.assertIn(k, r)
        self.assertEqual(d["portfolio"]["suppressed_reason"] in (None, "B0", "no_positive_edge"), True)
        self.assertLessEqual(len(d["leaders"]), 3)
        self.assertEqual(len(d["laggards"]), min(3, d["n_scored"] // 2))

    def test_ordering_is_rank_signal_not_cost_adjusted_score(self):
        d = self.poll("/api/recommendations?h=1")[1]
        rs = [r["rank_signal"] for r in d["all"]]
        self.assertEqual(rs, sorted(rs, reverse=True))
        self.assertEqual([r["rank"] for r in d["all"]], list(range(1, len(rs) + 1)))


# ============================================================================================ C-04 asof reference
class AsofReference(ServerCase):
    def test_pick_asof_ref_prefers_the_newest_well_supported_date(self):
        d1, d2 = date(2026, 10, 21), date(2026, 10, 22)
        p = jobs.pick_asof_ref
        self.assertEqual(p([d2] * 40 + [d1] * 20), d2)
        self.assertEqual(p([d1] * 40 + [d2] * 20), d2)  # 1/3 of the names carry the newer bar: it is the reference
        self.assertEqual(p([d1] * 55 + [d2] * 5), d1)  # a few early birds (8 %) do not move the reference
        self.assertEqual(p([d1] * 9 + [d2]), d1)  # a single newer name never does
        self.assertEqual(p([d2]), d2)
        self.assertEqual(p([d1, d2]), d2)  # no date reaches two names: the most common one, ties -> the newest
        self.assertEqual(p([d1] * 3), d1)
        self.assertEqual(p([d1, d1, d2, d2]), d2)

    def test_regressed_names_are_skipped_not_the_correct_newer_ones(self):
        # three of five names lost their last bar (Yahoo's null row): the OLD rule (most common date) kept those three
        # and skipped the two correct, newer names; now the up-to-date names are scored and the laggards are skipped
        ds = series_dates(899, date(2026, 10, 21))
        for code in ("1101", "1102", "1216"):
            self.route_symbol(f"{code}.TW", ds, price_series(899, 77), volume=[1_000_000] * 899)
        st, d, _ = self.poll("/api/recommendations?h=5")
        self.assertEqual(st, 200)
        self.assertEqual(d["n_scored"], 2)
        self.assertEqual(sorted(s["code"] for s in d["skipped"]), ["1101", "1102", "1216"])
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertTrue(all("未與其他標的同步" in s["reason"] for s in d["skipped"]))


# ============================================================================================ C-05 cache keys
class CacheKeys(ServerCase):
    def load(self, code="2330"):
        full, dc = self.compute.stock_series(code)
        return dc, self.compute.market_series()

    def test_every_price_column_is_part_of_the_forecast_and_walk_forward_keys(self):
        dc, mc = self.load()
        k0, w0 = self.compute._forecast_key(dc, mc), self.compute._wf_key(dc, mc, 400)
        for field in ("open", "high", "low", "volume", "close", "adjclose"):
            d2 = copy.deepcopy(dc)
            getattr(d2, field)[-5] *= 1.002
            self.assertNotEqual(self.compute._forecast_key(d2, mc), k0, field)
            self.assertNotEqual(self.compute._wf_key(d2, mc, 400), w0, field)
        m2 = copy.deepcopy(mc)
        m2.volume[-1] += 1.0
        self.assertNotEqual(self.compute._forecast_key(dc, m2), k0)  # the market series counts too
        self.assertEqual(self.compute._forecast_key(copy.deepcopy(dc), mc), k0)  # and an identical copy does not

    def test_a_revised_volume_recomputes_the_forecast(self):
        self.json("/api/forecast?symbol=2330", 200)
        n = self.engine.calls["forecast:2330"]
        self.json("/api/forecast?symbol=2330", 200)
        self.assertEqual(self.engine.calls["forecast:2330"], n)
        # Yahoo revises the volume of the last bars after the first fetch (audit C-13): same closes, new volumes
        vol = [2_000_000 + 1000 * j for j in range(self.N_BARS)]
        vol[-3:] = [v * 3 for v in vol[-3:]]
        self.route_symbol("2330.TW", self.ds, price_series(self.N_BARS, seed=10 + 4, start=50 + 40 * 4), volume=vol)
        self.clock.advance(hours=20)  # a new day, so that the cache is refreshed from the (revised) network answer
        self.clock.dt = datetime(2026, 10, 22, 17, 0, tzinfo=TAIPEI)
        data.get_price_data("2330", refresh=True)
        self.json("/api/forecast?symbol=2330", 200)
        self.assertEqual(self.engine.calls["forecast:2330"], n + 1)

    def test_configuration_constants_change_the_code_fingerprint(self):
        base = jobs.code_fingerprint("quant", "evaluation")
        jobs._FP.clear()
        self.assertEqual(jobs.code_fingerprint("quant", "evaluation"), base)  # deterministic
        for name, val in (("REFIT_EVERY", 10), ("TAX_STOCK", 0.006), ("DAILY_LIMIT", 0.07), ("HORIZONS", (1, 5, 10)),
                          ("MIN_TRAIN", 400), ("QLEVELS", (0.05, 0.5, 0.95)), ("COMMISSION_RATE", 0.002)):
            with mock.patch.object(config, name, val):
                jobs._FP.clear()
                self.assertNotEqual(jobs.code_fingerprint("quant"), jobs.code_fingerprint("quant") and base, name)
                self.assertNotEqual(jobs.code_fingerprint("quant", "evaluation"), base, name)
            jobs._FP.clear()
        self.assertEqual(jobs.code_fingerprint("quant", "evaluation"), base)

    def test_moving_the_project_does_not_invalidate_caches(self):
        base = jobs.config_fingerprint()
        with mock.patch.object(config, "ROOT", config.ROOT / "elsewhere"), \
                mock.patch.object(config, "CACHE_DIR", config.CACHE_DIR / "elsewhere"):
            self.assertEqual(jobs.config_fingerprint(), base)

    def test_keys_of_a_changed_configuration_differ(self):
        dc, mc = self.load()
        k0, w0 = self.compute._forecast_key(dc, mc), self.compute._wf_key(dc, mc, 400)
        with mock.patch.object(config, "REFIT_EVERY", 10):
            jobs._FP.clear()
            self.assertNotEqual(self.compute._wf_key(dc, mc, 400), w0)
            self.assertNotEqual(self.compute._forecast_key(dc, mc), k0)
            r_a = self.compute._result_key("backtest", params="x")
        jobs._FP.clear()
        self.assertNotEqual(self.compute._result_key("backtest", params="x"), r_a)
        self.assertEqual(self.compute._wf_key(dc, mc, 400), w0)

    def test_wf_key_names_the_walk_forward_configuration(self):
        dc, mc = self.load()
        w0 = self.compute._wf_key(dc, mc, 400)
        for name, val in (("HORIZONS", (1, 5)), ("MIN_TRAIN", 321), ("REFIT_EVERY", 7)):
            with mock.patch.object(config, name, val):
                self.assertNotEqual(self.compute._wf_key(dc, mc, 400), w0, name)  # even with the fingerprint memoised
            jobs._FP.clear()


# ============================================================================================ P-14 degenerate / malformed data
class DegenerateData(ServerCase):
    def test_a_frozen_price_series_is_not_forecast(self):
        self.engine.zero_sigma = True  # what the engine reports for a series that never moves
        st, d, _ = self.json("/api/forecast?symbol=2330")
        self.assertEqual((st, d["error"]["code"]), (422, "insufficient_history"))
        self.assertIn("沒有變動", d["error"]["message"])
        self.assertEqual(sc.check_error(d), [])

    def test_check_degenerate_unit(self):
        dc = self.compute.stock_series("2330")[1]
        ok = {"horizons": {"1": {"sigma": 0.01}, "5": {"sigma": 0.03}}}
        self.compute.check_degenerate(ok, dc)
        for bad in (0.0, -0.1, None, float("nan"), float("inf"), "x", True):
            with self.assertRaises(jobs.ApiError, msg=repr(bad)):
                self.compute.check_degenerate({"horizons": {"1": {"sigma": 0.01}, "5": {"sigma": bad}}}, dc)

    def test_timestamps_in_milliseconds_are_data_unavailable_not_a_500(self):
        j = json.loads(yahoo_payload(self.ds, price_series(self.N_BARS, 5)))
        j["chart"]["result"][0]["timestamp"] = [t * 1000 for t in j["chart"]["result"][0]["timestamp"]]
        self.net.route("/chart/3002.TW?", json.dumps(j).encode())
        for path in ("/api/forecast?symbol=3002", "/api/history?symbol=3002"):
            st, d, hdr = self.json(path)
            self.assertEqual((st, d["error"]["code"]), (503, "data_unavailable"), path)
            self.assertEqual(hdr.get("retry-after"), "5")

    def test_malformed_answer_is_not_remembered_as_an_unknown_symbol(self):
        j = json.loads(yahoo_payload(self.ds, price_series(self.N_BARS, 6)))
        del j["chart"]["result"][0]["indicators"]["quote"]
        self.net.route("/chart/3003.TW?", json.dumps(j).encode())
        self.net.route("/chart/3003.TWO?", data.NotFoundError("404"))  # the OTC probe really says "unknown"
        st, d, _ = self.json("/api/forecast?symbol=3003")
        self.assertEqual((st, d["error"]["code"]), (503, "data_unavailable"))
        # Yahoo recovers: the code works at once (the old behaviour: 404 unknown_symbol for UNKNOWN_TTL = 10 minutes)
        self.net.route("/chart/3003.TW?", yahoo_payload(self.ds, price_series(self.N_BARS, 6), volume=[1_000_000] * self.N_BARS))
        self.clock.mono += data.FAIL_BACKOFF + 1
        st, d, _ = self.json("/api/forecast?symbol=3003")
        self.assertEqual(st, 200)

    def test_a_move_beyond_the_daily_limit_is_flagged(self):
        c = price_series(self.N_BARS, 8)
        c[-1] = c[-2] * 1.15
        self.route_symbol("2330.TW", self.ds, c, volume=[2_000_000] * self.N_BARS)
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertTrue(any("資料可能有誤" in n and "+15.0%" in n for n in d["notes"]), d["notes"])
        self.assertEqual(sc.check_forecast(d), [])
        d2 = self.json("/api/forecast?symbol=1101", 200)[1]  # an ordinary series gets no such note
        self.assertFalse(any("資料可能有誤" in n for n in d2["notes"]))

    def test_data_that_ends_before_the_newest_session_is_flagged(self):
        ds = series_dates(899, date(2026, 10, 20))  # the stock stopped two sessions ago (suspended / lagging feed)
        self.route_symbol("2330.TW", ds, price_series(899, 9), volume=[2_000_000] * 899)
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-20")
        self.assertFalse(d["stale"])  # the meaning of `stale` is unchanged (network failure only)
        self.assertTrue(any("更舊" in n and "2026-10-22" in n for n in d["notes"]), d["notes"])
        self.assertEqual(sc.check_forecast(d), [])
        d2 = self.json("/api/forecast?symbol=1101", 200)[1]
        self.assertFalse(any("更舊" in n for n in d2["notes"]))

    def test_no_lag_note_while_the_session_is_running(self):
        self.clock.dt = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)  # today's bar is dropped in-session: not "old data"
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-21")
        self.assertFalse(any("更舊" in n for n in d["notes"]), d["notes"])


# ============================================================================================ P-01 closure through the API
class UnannouncedClosure(ServerCase):
    """2026-10-23 (Friday): Yahoo serves flat zero-volume bars for every stock and a null row for the TAIEX."""
    N = 900

    def setUp(self):
        super().setUp()
        self.clock.dt = datetime(2026, 10, 23, 15, 0, tzinfo=TAIPEI)
        ds = series_dates(self.N, date(2026, 10, 23))
        self.dsc = ds
        for i, (code, _) in enumerate(UNIVERSE + [("0050", "元大台灣50")]):
            close = price_series(self.N, seed=10 + i, start=50 + 40 * i)
            o, h, l = close.copy(), close * 1.01, close * 0.99
            vol = [2_000_000] * self.N
            close[-1] = close[-2]
            o[-1] = h[-1] = l[-1] = close[-1]
            vol[-1] = 0
            self.route_symbol(f"{code}.TW", ds, close, open_=o, high=h, low=l, volume=vol)
        self.route_symbol("%5ETWII", ds, price_series(self.N, 99, 15000), volume=[0] * self.N, itype="INDEX", nulls=(self.N - 1,))

    def test_asof_and_next_trading_day_follow_the_real_sessions(self):
        d = self.json("/api/forecast?symbol=2330", 200)[1]
        self.assertEqual(d["asof"], "2026-10-22")  # not the phantom 10-23
        self.assertNotEqual(d["chg_1d"], 0.0)
        self.assertEqual(d["next_trading_day"], "2026-10-27")  # 10-23 closed, weekend, scheduled holiday Mon 10-26
        self.assertEqual(sc.check_forecast(d), [])

    def test_history_has_no_phantom_bar_and_is_not_stale(self):
        d = self.json("/api/history?symbol=2330&limit=5", 200)[1]
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertEqual(d["bars"][-1]["d"], "2026-10-22")
        self.assertGreater(d["bars"][-1]["v"], 0)
        self.assertFalse(d["stale"])

    def test_status_reports_the_last_real_trading_day_once_the_closure_is_learned(self):
        for code in ("2330", "1101", "1102"):
            self.json(f"/api/history?symbol={code}&limit=2", 200)
        st = self.json("/api/status", 200)[1]
        self.assertEqual(st["last_trading_date"], "2026-10-22")
        self.assertFalse(st["market_open"])

    def test_recommendations_use_the_last_real_session(self):
        st, d, _ = self.poll("/api/recommendations?h=5")
        self.assertEqual(st, 200)
        self.assertEqual(d["asof"], "2026-10-22")
        self.assertEqual(d["n_scored"], 5)
        self.assertEqual(d["skipped"], [])
        self.assertTrue(all(r["chg_1d"] != 0.0 for r in d["all"]))


if __name__ == "__main__":
    unittest.main()
