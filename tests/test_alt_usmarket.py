"""Tests for backend/altdata/usmarket.py (PLAN.md P3a).  Hermetic: no network, fixtures in tests/fixtures/alt/.

The fixtures ``us_chart_*.json`` are small *real* Yahoo chart answers sampled on 2026-11-12 ~16:16 UTC (the last
bar of each is the still-forming 2026-11-12 session); ``us_cache_nvda.json`` is a slice of a real cache file."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from backend import data  # noqa: E402
from backend.altdata import usmarket as U  # noqa: E402

FIX = HERE / "fixtures" / "alt"
D = np.datetime64


def utc(y, mo, d, h=0, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def load_chart(name):
    return data.parse_yahoo_chart((FIX / name).read_bytes())


def brute_align(us_dates, us_ret, tw_cal):
    """Reference: explicit loops over calendar days; same definition, no vectorisation."""
    out = np.full(len(tw_cal), np.nan)
    for j in range(1, len(tw_cal)):
        s, s_prev = tw_cal[j], tw_cal[j - 1]
        acc, n, bad = 0.0, 0, False
        for d, r in zip(us_dates, us_ret):
            if s_prev <= d < s:
                n += 1
                if np.isnan(r):
                    bad = True
                else:
                    acc += r
        if n > 0 and not bad:
            out[j] = acc
    return out


class TestIncompleteBars(unittest.TestCase):
    def test_boundary_2130_utc(self):
        dates = np.array(["2026-11-10", "2026-11-11"], dtype="datetime64[D]")
        # one second before 21:30 UTC on D: incomplete
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 11, 21, 29, 59)).tolist(), [True, False])
        # exactly 21:30:00 UTC: complete (incomplete only while now < D 21:30)
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 11, 21, 30, 0)).tolist(), [True, True])
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 11, 21, 30, 1)).tolist(), [True, True])
        # a day earlier at 21:30 exactly: bar of the 10th complete, 11th not
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 10, 21, 30, 0)).tolist(), [True, False])
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 10, 21, 29, 59)).tolist(), [False, False])

    def test_timezone_aware_now_is_converted(self):
        dates = np.array(["2026-11-11"], dtype="datetime64[D]")
        tz_taipei = timezone(timedelta(hours=8))
        # 21:30 UTC == 05:30 next day Taipei
        self.assertEqual(U.complete_mask(dates, datetime(2026, 11, 12, 5, 30, tzinfo=tz_taipei)).tolist(), [True])
        self.assertEqual(U.complete_mask(dates, datetime(2026, 11, 12, 5, 29, 59, tzinfo=tz_taipei)).tolist(),
                         [False])
        # naive datetimes and ISO strings are read as UTC
        self.assertEqual(U.complete_mask(dates, datetime(2026, 11, 11, 21, 30)).tolist(), [True])
        self.assertEqual(U.complete_mask(dates, "2026-11-11T21:29:59Z").tolist(), [False])

    def test_currency_bar_is_final_only_at_end_of_utc_day(self):
        dates = np.array(["2026-11-11"], dtype="datetime64[D]")
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 11, 23, 59, 59), 24 * 3600).tolist(), [False])
        self.assertEqual(U.complete_mask(dates, utc(2026, 11, 12, 0, 0, 0), 24 * 3600).tolist(), [True])

    def test_drop_incomplete_multiple_arrays(self):
        dates = np.array(["2026-11-10", "2026-11-11"], dtype="datetime64[D]")
        d, (a, b) = U.drop_incomplete(dates, [np.array([1.0, 2.0]), np.array([3.0, 4.0])], utc(2026, 11, 11, 12))
        self.assertEqual(d.tolist(), [D("2026-11-10")])
        self.assertEqual((a.tolist(), b.tolist()), ([1.0], [3.0]))


class TestAlignment(unittest.TestCase):
    TW = np.array(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"], dtype="datetime64[D]")

    def test_same_day_not_usable_next_day_usable(self):
        us_d = np.array(["2026-11-09"], dtype="datetime64[D]")
        us_r = np.array([0.0123])
        out = U.align_us_to_tw(us_d, us_r, self.TW)
        # S = D (Mon 11-09) must NOT see the Monday US bar; S = D+1 (Tue 11-10) does
        self.assertTrue(np.isnan(out[1]))
        self.assertAlmostEqual(out[2], 0.0123)
        self.assertTrue(np.isnan(out[3]))  # nothing new for 11-11 (no US bar on 11-10)

    def test_first_calendar_day_nan_and_no_reuse_of_stale_bar(self):
        # bar 11-05 was usable at S=11-06 (first calendar day -> NaN by definition); it must not resurface at the
        # next trading days: S=11-09 has window [11-06, 11-09) which holds no bar
        us_d = np.array(["2026-11-05"], dtype="datetime64[D]")
        out = U.align_us_to_tw(us_d, np.array([0.01]), self.TW)
        self.assertTrue(np.all(np.isnan(out)))

    def test_taiwan_holiday_gap_is_cumulative(self):
        # Taipei trades Fri 11-06 then Tue 11-10 (Mon 11-09 closed): US sessions Fri 11-06 and Mon 11-09 are both new
        tw = np.array(["2026-11-05", "2026-11-06", "2026-11-10", "2026-11-11"], dtype="datetime64[D]")
        us_d = np.array(["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"], dtype="datetime64[D]")
        us_r = np.array([0.001, 0.002, 0.004, 0.008, 0.016])
        out = U.align_us_to_tw(us_d, us_r, tw)
        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], 0.002)            # S=11-06: window [11-05, 11-06) -> bar 11-05
        self.assertAlmostEqual(out[2], 0.004 + 0.008)    # S=11-10: bars 11-06 and 11-09 (log returns add)
        self.assertAlmostEqual(out[3], 0.016)            # S=11-11: bar 11-10

    def test_us_holiday_gives_nan(self):
        tw = np.array(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"], dtype="datetime64[D]")
        # US closed Mon 2026-09-07 (Labor Day): Taipei S=09-08 has no new bar
        us_d = np.array(["2026-09-02", "2026-09-03", "2026-09-04"], dtype="datetime64[D]")
        us_r = np.array([0.1, 0.2, 0.3])
        out = U.align_us_to_tw(us_d, us_r, tw)
        self.assertAlmostEqual(out[1], 0.2)
        self.assertAlmostEqual(out[2], 0.3)   # S=Mon 09-07 sees Fri 09-04
        self.assertTrue(np.isnan(out[3]))     # S=Tue 09-08: window [09-07, 09-08) is empty
        self.assertEqual(U.align_us_to_tw(us_d, us_r, tw, empty=0.0)[3], 0.0)

    def test_nan_return_in_window_gives_nan(self):
        tw = np.array(["2026-11-05", "2026-11-10"], dtype="datetime64[D]")
        us_d = np.array(["2026-11-05", "2026-11-06", "2026-11-09"], dtype="datetime64[D]")
        out = U.align_us_to_tw(us_d, np.array([0.1, np.nan, 0.3]), tw)
        self.assertTrue(np.isnan(out[1]))

    def test_no_data_before_series_start(self):
        tw = np.array(["2026-11-05", "2026-11-06", "2026-11-09"], dtype="datetime64[D]")
        us_d = np.array(["2026-11-10"], dtype="datetime64[D]")
        self.assertTrue(np.all(np.isnan(U.align_us_to_tw(us_d, np.array([0.5]), tw))))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            U.align_us_to_tw(np.array(["2026-11-05"], dtype="datetime64[D]"), np.array([0.1, 0.2]), self.TW)
        with self.assertRaises(ValueError):
            U.align_us_to_tw(np.array(["2026-11-05", "2026-11-05"], dtype="datetime64[D]"), np.array([0.1, 0.2]),
                             self.TW)
        with self.assertRaises(ValueError):
            U.align_us_to_tw(np.array(["2026-11-05"], dtype="datetime64[D]"), np.array([0.1]), self.TW[::-1])
        self.assertEqual(U.align_us_to_tw(np.array([], dtype="datetime64[D]"), np.zeros(0),
                                          np.array([], dtype="datetime64[D]")).shape, (0,))

    def test_unsorted_us_input_is_sorted(self):
        us_d = np.array(["2026-11-09", "2026-11-05"], dtype="datetime64[D]")
        a = U.align_us_to_tw(us_d, np.array([0.2, 0.1]), self.TW)
        b = U.align_us_to_tw(us_d[::-1], np.array([0.1, 0.2]), self.TW)
        np.testing.assert_array_equal(a, b)

    @staticmethod
    def _random_case(rng):
        base = D("2020-01-01")
        # US: business days with random holidays; TW: business days with different random holidays
        days = base + np.arange(0, 400)
        wd = (days.astype("int64") + 3) % 7  # 1970-01-01 was Thursday -> 0=Mon
        biz = days[wd < 5]
        us_d = biz[rng.random(len(biz)) > 0.04]
        tw = biz[rng.random(len(biz)) > 0.06]
        us_r = rng.normal(0, 0.01, len(us_d))
        us_r[rng.random(len(us_d)) < 0.02] = np.nan
        return us_d, us_r, tw

    def test_matches_brute_force(self):
        for seed in range(25):
            rng = np.random.default_rng(seed)
            us_d, us_r, tw = self._random_case(rng)
            np.testing.assert_allclose(U.align_us_to_tw(us_d, us_r, tw), brute_align(us_d, us_r, tw),
                                       rtol=0, atol=1e-12, equal_nan=True, err_msg=f"seed {seed}")

    def test_poisoned_future_does_not_change_past(self):
        for seed in range(10):
            rng = np.random.default_rng(100 + seed)
            us_d, us_r, tw = self._random_case(rng)
            d0 = us_d[len(us_d) // 2]
            base = U.align_us_to_tw(us_d, us_r, tw)
            bad = us_r.copy()
            fut = us_d > d0
            bad[fut] = rng.normal(5.0, 3.0, int(fut.sum()))  # garbage after D0
            poisoned = U.align_us_to_tw(us_d, bad, tw)
            # S uses only bars with D < S, so every S <= D0 + 1 is unaffected
            safe = tw <= d0 + 1
            self.assertTrue(safe.any())
            np.testing.assert_array_equal(base[safe], poisoned[safe])
            self.assertFalse(np.allclose(base[~safe], poisoned[~safe], equal_nan=True))  # the poison is effective

    def test_value_at_S_depends_only_on_strictly_earlier_bars(self):
        rng = np.random.default_rng(7)
        us_d, us_r, tw = self._random_case(rng)
        full = U.align_us_to_tw(us_d, us_r, tw)
        for j in (50, 120, 200, 260):
            s = tw[j]
            keep = us_d < s
            part = U.align_us_to_tw(us_d[keep], us_r[keep], tw[: j + 1])
            np.testing.assert_array_equal(full[: j + 1], part)


class TestFixturesAndFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="twq_us_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_urls_are_percent_encoded(self):
        self.assertIn("/%5ESOX?", data._yahoo_url("^SOX", 0, 1))
        self.assertIn("/TWD%3DX?", data._yahoo_url("TWD=X", 0, 1))

    def test_yahoo_dates_are_exchange_local(self):
        sox = load_chart("us_chart_sox.json")
        self.assertEqual(sox["meta"]["type"], "INDEX")
        last = np.datetime64(int(sox["d"][-1]), "D")
        ny = datetime.fromtimestamp(sox["meta"]["regular_market_time"], timezone.utc) - timedelta(hours=5)
        # the forming bar is stamped with the New York date of the request (not the Taipei date)
        self.assertEqual(str(last), ny.strftime("%Y-%m-%d"))
        wd = (sox["d"] + 3) % 7
        self.assertTrue(np.all(wd < 5))  # index sessions on weekdays only

    def fake_fetcher(self, calls):
        charts = {"^SOX": "us_chart_sox.json", "TWD=X": "us_chart_twd.json", "NVDA": "us_chart_sox.json"}

        def fetch(ysym, period1, *a, **k):
            calls.append((ysym, period1))
            if ysym == "BOOM":
                raise data.DataUnavailable("boom")
            return load_chart(charts[ysym])
        return fetch

    def test_fetch_all_drops_forming_bar_and_writes_cache(self):
        calls, sleeps = [], []
        rep = U.fetch_all(["SOX", "TWDUSD"], now=utc(2026, 11, 12, 16, 20), fetcher=self.fake_fetcher(calls),
                          sleep=sleeps.append, cache_dir=self.tmp)
        self.assertEqual(calls, [("^SOX", 0), ("TWD=X", 0)])
        self.assertEqual(sleeps, [1.0])  # >= 1 s between requests, none before the first
        self.assertTrue(rep["SOX"]["ok"] and rep["TWDUSD"]["ok"])
        self.assertEqual(rep["SOX"]["last"], "2026-11-11")      # 2026-11-12 session still forming at 16:20 UTC
        self.assertEqual(rep["SOX"]["dropped_incomplete"], 1)
        self.assertEqual(rep["TWDUSD"]["last"], "2026-11-11")   # 2026-11-12 London-dated bar not final either
        rec = json.loads((Path(self.tmp) / "alt" / "us" / "SOX.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["fetched_at"], "2026-11-12T16:20:00Z")
        self.assertEqual(rec["yahoo_symbol"], "^SOX")
        self.assertEqual(len(rec["dates"]), rep["SOX"]["n"])
        for k in ("open", "high", "low", "close", "adjclose", "volume"):
            self.assertEqual(len(rec[k]), len(rec["dates"]))

    def test_fetch_all_boundary_keeps_bar_at_2130_exactly(self):
        for now, expect_last in ((utc(2026, 11, 12, 21, 29, 59), "2026-11-11"), (utc(2026, 11, 12, 21, 30, 0), "2026-11-12")):
            rep = U.fetch_all(["SOX"], now=now, fetcher=self.fake_fetcher([]), sleep=lambda s: None,
                              cache_dir=self.tmp)
            self.assertEqual(rep["SOX"]["last"], expect_last, str(now))
        # TWD=X bar of 2026-11-12 only becomes final on 2026-11-13 00:00 UTC
        rep = U.fetch_all(["TWDUSD"], now=utc(2026, 11, 12, 21, 30), fetcher=self.fake_fetcher([]),
                          sleep=lambda s: None, cache_dir=self.tmp)
        self.assertEqual(rep["TWDUSD"]["last"], "2026-11-11")
        rep = U.fetch_all(["TWDUSD"], now=utc(2026, 11, 13, 0, 0), fetcher=self.fake_fetcher([]),
                          sleep=lambda s: None, cache_dir=self.tmp)
        self.assertEqual(rep["TWDUSD"]["last"], "2026-11-12")

    def test_one_failure_does_not_stop_the_others(self):
        saved = dict(U.SERIES)
        self.addCleanup(lambda: (U.SERIES.clear(), U.SERIES.update(saved)))
        U.SERIES["BAD"] = {"ysym": "BOOM", "basis": "close", "final_after_s": 77400}
        rep = U.fetch_all(["BAD", "SOX"], now=utc(2026, 11, 13), fetcher=self.fake_fetcher([]),
                          sleep=lambda s: None, cache_dir=self.tmp)
        self.assertFalse(rep["BAD"]["ok"])
        self.assertIn("boom", rep["BAD"]["error"])
        self.assertTrue(rep["SOX"]["ok"])
        self.assertFalse((Path(self.tmp) / "alt" / "us" / "BAD.json").exists())

    def test_unknown_name_and_missing_cache(self):
        with self.assertRaises(KeyError):
            U.fetch_all(["NOPE"], now=utc(2026, 11, 13), fetcher=self.fake_fetcher([]), sleep=lambda s: None,
                        cache_dir=self.tmp)
        with self.assertRaises(FileNotFoundError):
            U.daily_log_returns("SOX", cache_dir=self.tmp)

    def test_cache_roundtrip_and_returns_match_hand_computation(self):
        U.fetch_all(["SOX"], now=utc(2026, 11, 13), fetcher=self.fake_fetcher([]), sleep=lambda s: None,
                    cache_dir=self.tmp)
        rec = U.load_series("SOX", self.tmp)
        dates, ret = U.daily_log_returns("SOX", cache_dir=self.tmp)
        close = np.array(rec["close"])
        self.assertEqual(len(dates), len(close) - 1)
        self.assertEqual(dates.tolist(), [D(x) for x in rec["dates"][1:]])
        for k in range(len(ret)):
            self.assertAlmostEqual(ret[k], np.log(close[k + 1] / close[k]), places=12)
        # last bar (2026-11-12) final at 2026-11-13 -> present
        self.assertEqual(str(dates[-1]), "2026-11-12")

    def test_read_time_drop_uses_fetched_at_by_default(self):
        # a cache written at 16:16 UTC on 2026-11-12 that (wrongly) contains the forming 2026-11-12 bar
        rec = json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["dates"][-1], "2026-11-11")
        rec2 = copy.deepcopy(rec)
        rec2["dates"].append("2026-11-12")
        for k in ("open", "high", "low", "close", "adjclose", "volume"):
            rec2[k].append(rec2[k][-1] * 2)
        U.save_series("NVDA", rec2, self.tmp)
        d, r = U.daily_log_returns("NVDA", cache_dir=self.tmp)           # now = fetched_at (16:16Z) -> dropped
        self.assertEqual(str(d[-1]), "2026-11-11")
        d, r = U.daily_log_returns("NVDA", cache_dir=self.tmp, now=utc(2026, 11, 12, 21, 30))
        self.assertEqual(str(d[-1]), "2026-11-12")
        self.assertAlmostEqual(r[-1], np.log(2.0), places=9)
        d, r = U.daily_log_returns("NVDA", cache_dir=self.tmp, now=utc(2026, 11, 12, 21, 29, 59))
        self.assertEqual(str(d[-1]), "2026-11-11")

    def test_equities_use_adjclose_indices_use_close(self):
        rec = json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["basis"], "adjclose")
        rec["adjclose"] = [x * (1.0 + 0.01 * i) for i, x in enumerate(rec["close"])]
        U.save_series("NVDA", rec, self.tmp)
        _, r = U.daily_log_returns("NVDA", cache_dir=self.tmp)
        adj = np.array(rec["adjclose"])
        np.testing.assert_allclose(r, np.log(adj[1:] / adj[:-1]), atol=1e-12)
        self.assertEqual(U.SERIES["SOX"]["basis"], "close")
        self.assertEqual(U.SERIES["TSM"]["basis"], "adjclose")

    def test_bad_price_gives_nan_return(self):
        rec = json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8"))
        rec["adjclose"][5] = None
        U.save_series("NVDA", rec, self.tmp)
        _, r = U.daily_log_returns("NVDA", cache_dir=self.tmp)
        self.assertTrue(np.isnan(r[4]) and np.isnan(r[5]))
        self.assertEqual(int(np.isnan(r).sum()), 2)

    def test_one_bar_bad_print_is_bridged_but_real_moves_are_kept(self):
        rec = json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8"))
        n = len(rec["dates"])
        base = [100.0 + i for i in range(n)]
        base[10] = 6.0  # glitch: 100 -> 6 -> back to ~110
        base[20] = base[19] * 0.5   # genuine crash that stays down (not a spike)
        for i in range(21, n):
            base[i] = base[20] * (1 + 0.001 * (i - 20))
        rec["adjclose"] = base
        U.save_series("NVDA", rec, self.tmp)
        d, r = U.daily_log_returns("NVDA", cache_dir=self.tmp)
        self.assertEqual(len(d), n - 2)  # bad bar removed -> one bar fewer than the raw n-1 returns
        self.assertNotIn(rec["dates"][10], [str(x) for x in d])
        ds = [str(x) for x in d]
        self.assertAlmostEqual(r[ds.index(rec["dates"][11])], np.log(base[11] / base[9]), places=12)
        self.assertAlmostEqual(r[ds.index(rec["dates"][20])], np.log(0.5 * base[19] / base[19]), places=12)
        d2, r2 = U.daily_log_returns("NVDA", cache_dir=self.tmp, despike=False)
        self.assertEqual(len(d2), n - 1)
        self.assertEqual(int((np.abs(r2) > 1.0).sum()), 1 + 1)  # 100 -> 6 and 6 -> 110
        self.assertFalse(U.spike_mask(np.array([1.0, 2.0, 2.0, 2.1])).any())
        self.assertTrue(U.spike_mask(np.array([1.0, 3.0, 1.01, 1.0]))[1])
        self.assertFalse(U.spike_mask(np.array([1.0, 1.0, 3.0]))[2])  # last bar never flagged

    def test_vix_is_never_despiked(self):
        rec = json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8"))
        rec.update(name="VIX", basis="close")
        rec["close"] = [20.0] * len(rec["dates"])
        rec["close"][10] = 40.0  # genuine volatility spike that reverses next day
        U.save_series("VIX", rec, self.tmp)
        d, r = U.daily_log_returns("VIX", cache_dir=self.tmp)
        self.assertEqual(len(d), len(rec["dates"]) - 1)
        self.assertAlmostEqual(float(np.nanmax(r)), np.log(2.0), places=12)

    def test_end_to_end_alignment_on_fixture(self):
        U.save_series("NVDA", json.loads((FIX / "us_cache_nvda.json").read_text(encoding="utf-8")), self.tmp)
        dates, ret = U.daily_log_returns("NVDA", cache_dir=self.tmp)
        # Taipei calendar: Tue 11-10 .. Fri 11-13 plus the following Mon
        tw = np.array(["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"],
                      dtype="datetime64[D]")
        out = U.aligned_overnight("NVDA", tw, cache_dir=self.tmp)
        by_date = {str(d): r for d, r in zip(dates, ret)}
        self.assertAlmostEqual(out[1], by_date["2026-11-09"])
        self.assertAlmostEqual(out[2], by_date["2026-11-10"])
        self.assertAlmostEqual(out[3], by_date["2026-11-11"])   # last cached bar -> usable on Thu 11-12
        self.assertTrue(np.isnan(out[4]))  # no cached 11-12 bar yet
        np.testing.assert_allclose(out, brute_align(dates, ret, tw), atol=1e-12, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
