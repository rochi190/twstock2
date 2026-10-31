"""Data layer: price cache, refresh policy, .TW/.TWO fallback, stale fallback, incremental refresh, incomplete bars,
concurrency and atomic writes.  All network traffic is faked."""
import json
import os
import sys
import threading
import time
import unittest
from datetime import date, datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from data_fakes import TAIPEI, DataCase, bdays, random_walk, yahoo_error, yahoo_payload  # noqa: E402

from backend import data  # noqa: E402

# 300 business days ending Thursday 2026-10-22
END = date(2026, 10, 22)


def history(n=300, end=END):
    ds = bdays(end - timedelta(days=int(n * 1.6) + 10), 10_000)
    ds = [d for d in ds if d <= end][-n:]
    assert ds[-1] == end and len(ds) == n
    return ds


class PriceCache(DataCase):
    def setUp(self):
        super().setUp()
        self.ds = history()
        self.close = random_walk(len(self.ds), seed=1)
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)])

    def test_first_fetch_builds_pricedata_and_cache_file(self):
        self.route_symbol("2330.TW", self.ds, self.close, volume=[5000] * len(self.ds))
        pd = data.get_price_data("2330")
        self.assertEqual((pd.symbol, pd.name, pd.market, pd.kind), ("2330", "台積電", "TWSE", "stock"))
        self.assertEqual(pd.n, 300)
        self.assertEqual(pd.dates.dtype, np.dtype("datetime64[D]"))
        self.assertEqual(str(pd.dates[-1]), "2026-10-22")
        np.testing.assert_allclose(pd.close, self.close)
        self.assertTrue(pd.last_bar_complete)
        self.assertFalse(pd.stale)
        p = os.path.join(self.tmp, "prices", "2330.json")
        self.assertTrue(os.path.isfile(p))
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "prices"))), ["2330.json"])  # no temp leftovers
        # URL contract: period1=0, interval=1d, events, never range=max
        url = [u for u in self.net.calls if "/chart/" in u][0]
        self.assertIn("/v8/finance/chart/2330.TW?period1=0&period2=", url)
        self.assertIn("interval=1d", url)
        self.assertIn("events=div%2Csplits", url)
        self.assertNotIn("range=max", url)

    def test_second_call_uses_cache_without_network(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        data.get_price_data("2330")
        n = len(self.net.calls)
        for _ in range(3):
            data.get_price_data("2330")
        self.assertEqual(len(self.net.calls), n)

    def test_cache_survives_process_restart(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        data.get_price_data("2330")
        n = len(self.net.calls)
        self.reset_memory()  # simulate a new process: only the file remains
        pd = data.get_price_data("2330")
        self.assertEqual(len(self.net.calls), n)
        self.assertEqual(pd.n, 300)

    def test_returned_arrays_are_independent_copies(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        a = data.get_price_data("2330")
        a.close[:] = -1
        b = data.get_price_data("2330")
        self.assertTrue(np.all(b.close > 0))

    def test_tw_then_two_fallback_is_remembered(self):
        self.route_names(twse=[], tpex=[])  # no hint from the name tables
        self.net.route("/chart/6488.TW?", data.NotFoundError("404"))
        self.route_symbol("6488.TWO", self.ds, self.close)
        pd = data.get_price_data("6488")
        self.assertEqual(pd.market, "TPEx")
        self.assertEqual(self.net.count("6488.TW?"), 1)
        self.assertEqual(self.net.count("6488.TWO?"), 1)
        self.assertEqual(json.load(open(os.path.join(self.tmp, "prices", "6488.json")))["yahoo_symbol"], "6488.TWO")
        # next refresh goes straight to .TWO
        self.net.reset()
        self.clock.advance(days=1, hours=-7)  # Fri 09:00+ -> a new bar is expected
        data.get_price_data("6488")
        self.assertEqual(self.net.count("6488.TW?"), 0)
        self.assertGreaterEqual(self.net.count("6488.TWO?"), 1)

    def test_name_table_market_hint_avoids_wasted_request(self):
        self.route_names(twse=[], tpex=[("6488", "環球晶", 7 * 10 ** 9, 400)])
        self.net.route("/chart/6488.TW?", data.NotFoundError("404"))
        self.route_symbol("6488.TWO", self.ds, self.close)
        pd = data.get_price_data("6488")
        self.assertEqual(pd.name, "環球晶")
        self.assertEqual(self.net.count("6488.TW?"), 0)  # TPEx known -> .TWO first

    def test_unknown_symbol(self):
        self.net.route("/chart/9999.TW?", data.NotFoundError("404"))
        self.net.route("/chart/9999.TWO?", data.NotFoundError("404"))
        with self.assertRaises(data.UnknownSymbol):
            data.get_price_data("9999")
        self.assertTrue(issubclass(data.UnknownSymbol, KeyError))
        n = len(self.net.calls)
        with self.assertRaises(KeyError):  # remembered for a while: no new traffic
            data.get_price_data("9999")
        self.assertEqual(len(self.net.calls), n)
        self.clock.advance(minutes=11)
        with self.assertRaises(KeyError):
            data.get_price_data("9999")
        self.assertGreater(len(self.net.calls), n)

    def test_bad_symbol(self):
        with self.assertRaises(data.BadSymbol):
            data.get_price_data("../etc")

    def test_network_failure_without_cache(self):
        self.net.route("/chart/2330.TW?", data.DataUnavailable("down"))
        with self.assertRaises(data.DataUnavailable):
            data.get_price_data("2330")

    def test_stale_cache_served_on_network_failure(self):
        self.route_symbol("2330.TW", self.ds[:-2], self.close[:-2])  # cache is two bars behind
        data.get_price_data("2330")
        self.clock.advance(minutes=20)
        self.net.route("/chart/2330.TW?", data.DataUnavailable("down"))
        pd = data.get_price_data("2330")
        self.assertTrue(pd.stale)
        self.assertEqual(pd.n, 298)
        with self.assertRaises(data.DataUnavailable):
            data.get_price_data("2330", allow_stale=False, refresh=True)

    def test_failure_backoff_prevents_hammering(self):
        self.route_symbol("2330.TW", self.ds[:-1], self.close[:-1])
        data.get_price_data("2330")
        self.clock.advance(minutes=20)
        self.net.route("/chart/2330.TW?", data.DataUnavailable("down"))
        data.get_price_data("2330")
        n = self.net.count("/chart/")
        pd = data.get_price_data("2330")  # within FAIL_BACKOFF -> no network, still flagged stale
        self.assertEqual(self.net.count("/chart/"), n)
        self.assertTrue(pd.stale)
        self.clock.advance(seconds=data.FAIL_BACKOFF + 1)
        data.get_price_data("2330")
        self.assertGreater(self.net.count("/chart/"), n)

    def test_refresh_throttle_when_new_bar_expected(self):
        self.route_symbol("2330.TW", self.ds[:-1], self.close[:-1])  # Wed's bar is the newest we know
        data.get_price_data("2330")
        n = self.net.count("/chart/")
        # Thursday's bar is expected (clock is Thu 16:00) but the cache was fetched just now -> no new request
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n)
        self.clock.advance(minutes=11)
        self.route_symbol("2330.TW", self.ds, self.close)
        pd = data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n + 1)
        self.assertEqual(pd.n, 300)

    def test_force_refresh_ignores_throttle(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        data.get_price_data("2330")
        n = self.net.count("/chart/")
        data.get_price_data("2330", refresh=True)
        self.assertEqual(self.net.count("/chart/"), n + 1)

    def test_no_refresh_on_weekend_and_before_open(self):
        # Friday close fetched Friday 15:00; Saturday and Monday 08:00 need nothing
        self.clock.dt = datetime(2026, 10, 23, 15, 0, tzinfo=TAIPEI)
        ds = history(300, date(2026, 10, 23))
        self.route_symbol("2330.TW", ds, random_walk(300, 3))
        data.get_price_data("2330")
        n = self.net.count("/chart/")
        for when in (datetime(2026, 10, 24, 11, 0), datetime(2026, 10, 25, 11, 0), datetime(2026, 10, 26, 10, 0),  # Sat, Sun, holiday Mon
                     datetime(2026, 10, 27, 8, 30)):  # Tue before the open
            self.clock.dt = when.replace(tzinfo=TAIPEI)
            self.clock.mono += 7200
            data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n)
        # Tuesday 09:30 a new bar is expected -> refresh
        self.clock.dt = datetime(2026, 10, 27, 9, 30, tzinfo=TAIPEI)
        self.clock.mono += 7200
        data.get_price_data("2330")
        self.assertGreater(self.net.count("/chart/"), n)

    def test_corrupt_cache_file_is_refetched(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        data.get_price_data("2330")
        with open(os.path.join(self.tmp, "prices", "2330.json"), "w") as fh:
            fh.write('{"v": 2, "dates": [')  # truncated JSON
        self.reset_memory()
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 300)
        json.load(open(os.path.join(self.tmp, "prices", "2330.json")))  # valid again

    def test_old_cache_version_is_ignored(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        data.get_price_data("2330")
        p = os.path.join(self.tmp, "prices", "2330.json")
        j = json.load(open(p))
        j["v"] = 1
        json.dump(j, open(p, "w"))
        self.reset_memory()
        n = self.net.count("/chart/")
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n + 1)

    def test_market_data_drops_null_rows(self):
        ds = history(400)
        c = random_walk(400, 7, 15000)
        self.route_symbol("%5ETWII", ds, c, nulls=(5, 6, 100, 399), itype="INDEX", volume=[0] * 400)
        m = data.get_market_data()
        self.assertEqual((m.symbol, m.name, m.market, m.kind), ("^TWII", "加權指數", "INDEX", "index"))
        self.assertEqual(m.n, 396)
        self.assertTrue(np.all(np.isfinite(m.close)))
        self.assertEqual(float(m.volume[-1]), 0.0)
        self.assertIn("/chart/%5ETWII?", self.net.calls[-1])

    def test_adjclose_and_dividends_roundtrip(self):
        adj = self.close * np.linspace(0.8, 1.0, 300)
        self.route_symbol("2330.TW", self.ds, self.close, adj=adj, divs=[(self.ds[100], 3.0)])
        pd = data.get_price_data("2330")
        np.testing.assert_allclose(pd.adjclose, adj)
        self.assertEqual(len(pd.dividends), 1)
        self.assertEqual(pd.dividends[0][0], np.datetime64(self.ds[100], "D"))
        self.assertEqual(pd.dividends[0][1], 3.0)
        self.reset_memory()
        pd2 = data.get_price_data("2330")  # from disk
        np.testing.assert_allclose(pd2.adjclose, adj)
        self.assertEqual(pd2.dividends, pd.dividends)

    def test_name_falls_back_to_curated_then_code(self):
        self.route_names(twse=[], tpex=[])
        self.route_symbol("2317.TW", self.ds, self.close, name="HON HAI")
        self.assertEqual(data.get_price_data("2317").name, "鴻海")  # curated fallback name
        self.route_symbol("8888.TW", self.ds, self.close, name="Obscure Co")
        self.assertEqual(data.get_price_data("8888").name, "Obscure Co")


class IncrementalRefresh(DataCase):
    def setUp(self):
        super().setUp()
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)])
        self.ds = history(300)
        self.close = random_walk(300, 11)
        # cache created Wed 2026-10-21 17:00 with data up to Wed
        self.clock.dt = datetime(2026, 10, 21, 17, 0, tzinfo=TAIPEI)
        self.route_symbol("2330.TW", self.ds[:-1], self.close[:-1])
        self.first = data.get_price_data("2330")
        self.assertEqual(self.first.n, 299)
        self.net.reset()
        self.clock.dt = datetime(2026, 10, 22, 16, 0, tzinfo=TAIPEI)  # next day after close
        self.clock.mono += 86400

    def window_payload(self, ds, close, **kw):
        """What Yahoo returns for an incremental window: only the trailing rows."""
        return yahoo_payload(ds, close, **kw)

    def test_incremental_appends_without_full_download(self):
        self.net.route("/chart/2330.TW?", lambda url: self.window_payload(self.ds[-15:], self.close[-15:]))
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 300)
        np.testing.assert_allclose(pd.close, self.close)
        self.assertEqual(self.net.count("/chart/"), 1)
        url = self.net.calls[-1]
        p1 = int(url.split("period1=")[1].split("&")[0])
        self.assertGreater(p1, 0)  # incremental: not period1=0
        # taipei-date of period1 is ~25 days before the last cached bar
        self.assertAlmostEqual((self.ds[-2] - date.fromtimestamp(p1 + 8 * 3600)).days, data.INCREMENTAL_DAYS, delta=1)

    def test_incremental_equals_full_download(self):
        # one underlying history; Yahoo serves the same numbers whether the request is a window or the full range
        vol = [1_000_000 + 1000 * i for i in range(300)]
        self.clock.dt = datetime(2026, 10, 21, 17, 0, tzinfo=TAIPEI)  # re-create the cache with this history
        data.set_cache_dir(self.tmp + "_inc")
        self.addCleanup(__import__("shutil").rmtree, self.tmp + "_inc", True)
        self.load_holidays()
        self.route_symbol("2330.TW", self.ds[:-1], self.close[:-1], volume=vol[:-1])
        data.get_price_data("2330")
        self.clock.dt = datetime(2026, 10, 22, 16, 0, tzinfo=TAIPEI)
        self.clock.mono += 86400
        self.net.reset()
        self.net.route("/chart/2330.TW?", lambda url: self.window_payload(self.ds[-15:], self.close[-15:],
                                                                         volume=vol[-15:]))
        inc = data.get_price_data("2330")
        self.assertNotIn("period1=0&", self.net.calls[-1])  # really incremental
        data.set_cache_dir(self.tmp + "_full")
        self.addCleanup(__import__("shutil").rmtree, self.tmp + "_full", True)
        self.load_holidays()
        self.route_symbol("2330.TW", self.ds, self.close, volume=vol)
        full = data.get_price_data("2330")
        for f in ("dates", "open", "high", "low", "close", "adjclose", "volume"):
            np.testing.assert_array_equal(getattr(inc, f), getattr(full, f), err_msg=f)

    def test_dividend_in_window_triggers_full_redownload(self):
        # a new dividend rescales the whole adjusted history: ratio adj/close changes in the overlap
        new_adj = self.close * 0.97
        calls = []

        def route(url):
            calls.append(url)
            if "period1=0&" in url:
                return yahoo_payload(self.ds, self.close, adj=new_adj, divs=[(self.ds[-1], 29.0)])
            return yahoo_payload(self.ds[-15:], self.close[-15:], adj=new_adj[-15:], divs=[(self.ds[-1], 29.0)])

        self.net.route("/chart/2330.TW?", route)
        pd = data.get_price_data("2330")
        self.assertEqual(len(calls), 2)  # window first, then the full history
        self.assertIn("period1=0&", calls[1])
        np.testing.assert_allclose(pd.adjclose, new_adj)  # the OLD history was rescaled too
        self.assertEqual(len(pd.dividends), 1)

    def test_split_adjustment_triggers_full_redownload(self):
        halved = self.close / 2.0  # 2:1 split: Yahoo split-adjusts every historical close

        def route(url):
            if "period1=0&" in url:
                return yahoo_payload(self.ds, halved)
            return yahoo_payload(self.ds[-15:], halved[-15:])

        self.net.route("/chart/2330.TW?", route)
        pd = data.get_price_data("2330")
        np.testing.assert_allclose(pd.close, halved)

    def test_empty_or_404_window_falls_back_to_full(self):
        def route(url):
            if "period1=0&" in url:
                return yahoo_payload(self.ds, self.close)
            raise data.NotFoundError("404")

        self.net.route("/chart/2330.TW?", route)
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 300)

    def test_full_redownload_when_history_is_old(self):
        self.clock.advance(days=data.FULL_REFRESH_DAYS + 1)
        calls = []

        def route(url):
            calls.append(url)
            return yahoo_payload(self.ds, self.close)

        self.net.route("/chart/2330.TW?", route)
        data.get_price_data("2330")
        self.assertEqual(len(calls), 1)
        self.assertIn("period1=0&", calls[0])

    def test_intraday_revision_of_last_cached_bar_is_not_a_conflict(self):
        # cache holds an intraday bar for today; the final bar has a different close but history is unchanged
        data.set_cache_dir(self.tmp + "_b")
        self.addCleanup(__import__("shutil").rmtree, self.tmp + "_b", True)
        self.clock.dt = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)
        mid = self.close.copy()
        mid[-1] = self.close[-1] * 1.03
        self.route_symbol("2330.TW", self.ds, mid)
        a = data.get_price_data("2330")
        self.assertFalse(a.last_bar_complete)
        self.clock.dt = datetime(2026, 10, 22, 14, 5, tzinfo=TAIPEI)
        self.clock.mono += 5000
        calls = []

        def route(url):
            calls.append(url)
            return yahoo_payload(self.ds[-15:], self.close[-15:]) if "period1=0&" not in url else yahoo_payload(self.ds, self.close)

        self.net.route("/chart/2330.TW?", route)
        b = data.get_price_data("2330")
        self.assertEqual(len(calls), 1)  # incremental was enough
        self.assertNotIn("period1=0&", calls[0])
        self.assertAlmostEqual(float(b.close[-1]), float(self.close[-1]))
        self.assertTrue(b.last_bar_complete)


def null_rows(ds, close, null_idx, **kw):
    """A Yahoo reply whose rows ``null_idx`` are all-null (the intermittent null row of a completed session)."""
    return yahoo_payload(ds, close, nulls=tuple(null_idx), **kw)


class NullLastRow(DataCase):
    """Audit C-04: a refresh must never delete a valid final bar because Yahoo served an all-null row for it."""

    def setUp(self):
        super().setUp()
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)])
        self.ds = history(300)
        self.close = random_walk(300, 21)
        self.vol = [1_000_000 + 1000 * i for i in range(300)]
        self.route_symbol("2330.TW", self.ds, self.close, volume=self.vol)  # cached Thu 16:00: bars up to Thu 10-22, final
        self.first = data.get_price_data("2330")
        self.assertEqual((self.first.n, str(self.first.dates[-1])), (300, "2026-10-22"))
        self.net.reset()
        self.clock.dt = datetime(2026, 10, 23, 11, 30, tzinfo=TAIPEI)  # Friday, session running: a new bar is expected
        self.clock.mono += 3600 * 20

    def window(self, null_idx, k=15, extra_today_null=True):
        """Incremental window: the last ``k`` bars (rows ``null_idx`` null) + a null row for today."""
        ds = self.ds[-k:]
        close = self.close[-k:]
        nulls = [i - (300 - k) for i in null_idx]
        body = yahoo_payload(ds, close, nulls=tuple(nulls), volume=self.vol[-k:])
        if not extra_today_null:
            return body
        import json
        j = json.loads(body)
        r = j["chart"]["result"][0]
        r["timestamp"].append(__import__("data_fakes").ts_of(date(2026, 10, 23)))
        for key in ("open", "high", "low", "close", "volume"):
            r["indicators"]["quote"][0][key].append(None)
        r["indicators"]["adjclose"][0]["adjclose"].append(None)
        return json.dumps(j).encode()

    def test_incremental_refresh_keeps_the_last_final_bar(self):
        self.net.route("/chart/2330.TW?", lambda url: self.window([299]))
        pd = data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), 1)
        self.assertNotIn("period1=0&", self.net.calls[-1])  # really the incremental path
        self.assertEqual((pd.n, str(pd.dates[-1])), (300, "2026-10-22"))
        np.testing.assert_array_equal(pd.close, self.first.close)
        np.testing.assert_array_equal(pd.volume, self.first.volume)
        np.testing.assert_array_equal(pd.adjclose, self.first.adjclose)
        self.assertFalse(pd.stale)
        self.assertTrue(pd.last_bar_complete)
        # the refresh time is the new one: the 10-minute throttle applies, no request storm while Yahoo has no row
        self.assertEqual(pd.fetched_at, data._iso(self.clock.dt))
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), 1)
        self.clock.advance(minutes=11)
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), 2)

    def test_full_download_keeps_the_last_final_bar(self):
        self.clock.advance(days=data.FULL_REFRESH_DAYS + 1)  # record older than 30 days: whole history again
        self.net.route("/chart/2330.TW?", lambda url: null_rows(self.ds, self.close, [299], volume=self.vol))
        pd = data.get_price_data("2330")
        self.assertIn("period1=0&", self.net.calls[-1])
        self.assertEqual((pd.n, str(pd.dates[-1])), (300, "2026-10-22"))
        np.testing.assert_array_equal(pd.close, self.first.close)

    def test_it_heals_when_yahoo_serves_the_row_again_with_a_revision(self):
        self.net.route("/chart/2330.TW?", lambda url: self.window([299]))
        data.get_price_data("2330")
        self.clock.advance(minutes=11)
        revised = self.close.copy()
        revised[-1] *= 1.004  # Yahoo's final number differs a little from what we cached
        # a revised FINAL bar is a conflict of the overlap check -> the whole history is downloaded again (and wins)
        self.net.route("/chart/2330.TW?", lambda url: yahoo_payload(self.ds, revised, volume=self.vol)
                       if "period1=0&" in url else yahoo_payload(self.ds[-15:], revised[-15:], volume=self.vol[-15:]))
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 300)
        self.assertAlmostEqual(float(pd.close[-1]), float(revised[-1]), places=3)  # the served row wins over the kept one

    def test_two_vanished_bars_are_kept_and_a_hole_in_the_middle_is_closed(self):
        self.net.route("/chart/2330.TW?", lambda url: self.window([297, 299]))  # a middle bar and the last one are null
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 300)
        np.testing.assert_array_equal(pd.dates, self.first.dates)
        np.testing.assert_array_equal(pd.close, self.first.close)

    def test_a_provisional_last_bar_is_not_kept(self):
        # the cached last bar was fetched while its session ran: it is provisional, the refreshed history replaces it
        data.set_cache_dir(self.tmp + "_p")
        self.addCleanup(__import__("shutil").rmtree, self.tmp + "_p", True)
        self.clock.dt = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)
        self.route_symbol("2330.TW", self.ds, self.close, volume=self.vol)
        a = data.get_price_data("2330")
        self.assertFalse(a.last_bar_complete)
        self.clock.dt = datetime(2026, 10, 22, 14, 30, tzinfo=TAIPEI)
        self.clock.mono += 5000
        self.net.route("/chart/2330.TW?", lambda url: self.window([299], extra_today_null=False))
        b = data.get_price_data("2330")
        self.assertEqual((b.n, str(b.dates[-1])), (299, "2026-10-21"))  # the provisional 10-22 bar is gone

    def test_dividend_adjustment_and_null_last_row_in_a_full_download(self):
        # the full re-download carries a new dividend: the whole adjusted history is rescaled, the kept tail with it
        self.clock.advance(days=data.FULL_REFRESH_DAYS + 1)
        new_adj = self.close * 0.97
        self.net.route("/chart/2330.TW?", lambda url: null_rows(self.ds, self.close, [299], adj=new_adj, volume=self.vol,
                                                                divs=[(self.ds[-3], 3.0)]))
        pd = data.get_price_data("2330")
        self.assertEqual((pd.n, str(pd.dates[-1])), (300, "2026-10-22"))
        np.testing.assert_allclose(pd.adjclose, new_adj)  # also the re-attached last bar (rescaled by 0.97)
        self.assertEqual(len(pd.dividends), 1)

    def test_split_and_null_last_row_cannot_be_reattached(self):
        self.clock.advance(days=data.FULL_REFRESH_DAYS + 1)
        halved = self.close / 2.0  # 2:1 split: every historical close is rescaled, the cached tail is on the old scale
        self.net.route("/chart/2330.TW?", lambda url: null_rows(self.ds, halved, [299], volume=self.vol))
        pd = data.get_price_data("2330")
        self.assertEqual((pd.n, str(pd.dates[-1])), (299, "2026-10-21"))
        np.testing.assert_allclose(pd.close, halved[:-1])

    def test_empty_incremental_window_then_full_download_with_missing_tail_rows(self):
        # the symbol stopped trading: the incremental window is empty (no timestamps at all), the full download ends
        # earlier than the cache only because its last rows are null -> the cached final bars stay
        def route(url):
            if "period1=0&" in url:
                return null_rows(self.ds, self.close, [296, 297, 298, 299], volume=self.vol)
            return yahoo_payload([], [])

        self.net.route("/chart/2330.TW?", route)
        pd = data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), 2)  # window first, then the full history
        self.assertEqual((pd.n, str(pd.dates[-1])), (300, "2026-10-22"))

    def test_no_usable_rows_at_all_is_data_unavailable_and_not_remembered_as_unknown(self):
        # audit P-14 for the first download of a never-cached code: a well formed answer without a single usable row
        self.route_names(twse=[], tpex=[])
        self.net.route("/chart/7777.TW?", yahoo_payload(self.ds[-5:], self.close[-5:], nulls=(0, 1, 2, 3, 4)))
        self.net.route("/chart/7777.TWO?", data.NotFoundError("404"))
        with self.assertRaises(data.DataUnavailable):
            data.get_price_data("7777")
        self.net.route("/chart/7777.TW?", yahoo_payload(self.ds, self.close))
        self.clock.mono += data.FAIL_BACKOFF + 1
        pd = data.get_price_data("7777")  # the transient fault did not poison the symbol for UNKNOWN_TTL
        self.assertEqual(pd.n, 300)

    def test_malformed_first_answer_is_data_unavailable_not_unknown_symbol(self):
        import json
        j = json.loads(yahoo_payload(self.ds[-5:], self.close[-5:]))
        del j["chart"]["result"][0]["indicators"]["quote"]
        self.net.route("/chart/7778.TW?", json.dumps(j).encode())
        self.net.route("/chart/7778.TWO?", data.NotFoundError("404"))
        with self.assertRaises(data.DataUnavailable):
            data.get_price_data("7778")
        with self.assertRaises(data.DataUnavailable):  # and not remembered as "no such symbol"
            data.get_price_data("7778", refresh=True)


class LegacyCache(DataCase):
    """Audit P-01 / C-05: a v2 cache record (phantom bars inside) is cleaned on load and re-downloaded in full next."""

    def test_v2_record_loses_its_phantom_bars_and_forces_a_full_download(self):
        ds = history(300)
        close = random_walk(300, 5)
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)])
        self.route_symbol("2330.TW", ds, close)
        data.get_price_data("2330")
        p = os.path.join(self.tmp, "prices", "2330.json")
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        # append the 2026-10-23 phantom as the old code stored it, and relabel the record as version 2
        j["v"] = 2
        j["dates"].append("2026-10-23")
        for k in ("open", "high", "low", "close", "adjclose"):
            j[k].append(j["close"][-1])
        j["volume"].append(0.0)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(j, fh)
        self.reset_memory()
        self.clock.dt = datetime(2026, 10, 23, 15, 0, tzinfo=TAIPEI)  # the day of the phantom, after the close
        self.net.reset()
        peek = data.peek_price_data("2330")  # cache only, no network: the phantom is already gone
        self.assertEqual((peek.n, str(peek.dates[-1])), (300, "2026-10-22"))
        self.assertEqual(self.net.count("/chart/"), 0)
        pd = data.get_price_data("2330")  # Friday 15:00: the 10-23 bar is expected -> refresh, and it is a FULL download
        self.assertEqual(self.net.count("/chart/"), 1)
        self.assertIn("period1=0&", self.net.calls[-1])  # no merge into a record that was not re-parsed
        self.assertEqual((pd.n, str(pd.dates[-1])), (300, "2026-10-22"))
        with open(p, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["v"], data.CACHE_VERSION)


class IncompleteBar(DataCase):
    START = datetime(2026, 10, 22, 11, 0, tzinfo=TAIPEI)  # Thursday, mid-session

    def setUp(self):
        super().setUp()
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)])
        self.ds = history(300)
        self.close = random_walk(300, 5)
        self.route_symbol("2330.TW", self.ds, self.close)

    def test_today_bar_before_1400_is_incomplete(self):
        pd = data.get_price_data("2330")
        self.assertFalse(pd.last_bar_complete)
        self.assertEqual(str(pd.dates[-1]), "2026-10-22")
        dc = data.completed(pd)
        self.assertEqual(dc.n, pd.n - 1)
        self.assertEqual(str(dc.dates[-1]), "2026-10-21")
        self.assertTrue(dc.last_bar_complete)
        self.assertEqual(pd.n, 300)  # the original is untouched
        np.testing.assert_array_equal(dc.close, pd.close[:-1])

    def test_completed_is_identity_when_final(self):
        self.clock.dt = datetime(2026, 10, 22, 14, 0, tzinfo=TAIPEI)
        pd = data.get_price_data("2330")
        self.assertTrue(pd.last_bar_complete)
        self.assertIs(data.completed(pd), pd)

    def test_boundary_1400_exact(self):
        self.clock.dt = datetime(2026, 10, 22, 13, 59, 59, tzinfo=TAIPEI)
        self.assertFalse(data.get_price_data("2330").last_bar_complete)
        self.reset_memory()
        os.remove(os.path.join(self.tmp, "prices", "2330.json"))
        self.clock.dt = datetime(2026, 10, 22, 14, 0, 0, tzinfo=TAIPEI)
        self.assertTrue(data.get_price_data("2330").last_bar_complete)

    def test_previous_days_bar_fetched_intraday_is_incomplete_until_refreshed(self):
        # cache created Tuesday 10:00 (bar for Tuesday intraday); now it is Thursday and nothing changed upstream
        self.clock.dt = datetime(2026, 10, 20, 10, 0, tzinfo=TAIPEI)
        ds = [d for d in self.ds if d <= date(2026, 10, 20)]
        self.route_symbol("2330.TW", ds, self.close[: len(ds)])
        a = data.get_price_data("2330")
        self.assertFalse(a.last_bar_complete)
        self.clock.dt = datetime(2026, 10, 22, 16, 0, tzinfo=TAIPEI)  # no throttle for this state transition
        self.clock.mono += 30
        self.route_symbol("2330.TW", self.ds, self.close)
        b = data.get_price_data("2330")
        self.assertTrue(b.last_bar_complete)
        self.assertEqual(str(b.dates[-1]), "2026-10-22")

    def test_cache_upgraded_when_session_ends_without_throttle(self):
        a = data.get_price_data("2330")
        self.assertFalse(a.last_bar_complete)
        n = self.net.count("/chart/")
        self.clock.advance(minutes=30)  # 11:30 still in session, throttle (10 min) elapsed -> intraday refresh allowed
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n + 1)
        self.clock.advance(minutes=5)  # 11:35: throttled
        data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n + 1)
        self.clock.dt = datetime(2026, 10, 22, 14, 1, tzinfo=TAIPEI)  # session final: refresh regardless of throttle
        self.clock.mono += 10
        b = data.get_price_data("2330")
        self.assertEqual(self.net.count("/chart/"), n + 2)
        self.assertTrue(b.last_bar_complete)
        data.get_price_data("2330")  # final bar: never refreshed again
        self.assertEqual(self.net.count("/chart/"), n + 2)

    def test_fetch_failed_after_close_keeps_incomplete_flag(self):
        a = data.get_price_data("2330")
        self.assertFalse(a.last_bar_complete)
        self.clock.dt = datetime(2026, 10, 22, 15, 0, tzinfo=TAIPEI)
        self.clock.mono += 10
        self.net.route("/chart/2330.TW?", data.DataUnavailable("down"))
        b = data.get_price_data("2330")
        self.assertTrue(b.stale)
        self.assertFalse(b.last_bar_complete)  # we only ever saw the intraday bar
        self.assertEqual(data.completed(b).n, b.n - 1)


class Concurrency(DataCase):
    def setUp(self):
        super().setUp()
        self.ds = history(300)
        self.close = random_walk(300, 2)
        self.route_names(twse=[])

    def test_same_symbol_downloads_once(self):
        self.route_symbol("2330.TW", self.ds, self.close)
        self.net.delay = 0.1
        out, errs = [], []

        def go():
            try:
                out.append(data.get_price_data("2330"))
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        ts = [threading.Thread(target=go) for _ in range(12)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])
        self.assertEqual(len(out), 12)
        self.assertEqual(self.net.count("/chart/2330.TW"), 1)

    def test_parallel_symbols_never_exceed_six_requests(self):
        codes = [f"{1100 + i}" for i in range(24)]
        for c in codes:
            self.route_symbol(f"{c}.TW", self.ds, self.close)
        self.net.delay = 0.05
        errs = []

        def go(c):
            try:
                data.get_price_data(c)
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        ts = [threading.Thread(target=go, args=(c,)) for c in codes]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])
        self.assertLessEqual(self.net.peak_chart, data.YAHOO_CONCURRENCY)  # Yahoo is capped ...
        self.assertGreaterEqual(self.net.peak_chart, 2)  # ... but really parallel
        self.assertEqual(self.net.count("/chart/"), 24)


class AtomicWrites(DataCase):
    def test_readers_never_see_partial_files(self):
        path = os.path.join(self.tmp, "x", "blob.json")
        from pathlib import Path
        stop = threading.Event()
        bad = []

        def reader():
            while not stop.is_set():
                try:
                    with open(path, "rb") as fh:
                        raw = fh.read()
                except FileNotFoundError:
                    continue
                try:
                    j = json.loads(raw)
                    if len(j["a"]) != 20000 or len(set(j["a"])) != 1:
                        bad.append("inconsistent")
                except ValueError:
                    bad.append("partial")

        ts = [threading.Thread(target=reader) for _ in range(3)]
        [t.start() for t in ts]
        for i in range(150):
            data.atomic_write_json(Path(path), {"a": [i] * 20000})
        stop.set()
        [t.join() for t in ts]
        self.assertEqual(bad, [])
        self.assertEqual([f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")], [])

    def test_failed_write_leaves_no_temp_file(self):
        from pathlib import Path
        # (a) the payload cannot be serialised: nothing is created at all
        p = Path(self.tmp) / "y" / "z.json"
        with self.assertRaises(TypeError):
            data.atomic_write_json(p, {"bad": object()})
        self.assertFalse(p.exists())
        self.assertEqual([f for f in os.listdir(p.parent) if f.endswith(".tmp")] if p.parent.exists() else [], [])
        # (b) the final rename fails after the temp file was written: the temp file is removed, the old file survives
        q = Path(self.tmp) / "w" / "old.json"
        data.atomic_write_json(q, {"v": 1})
        from unittest import mock
        with mock.patch.object(data.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                data.atomic_write_json(q, {"v": 2})
        self.assertEqual(json.load(open(q)), {"v": 1})
        self.assertEqual(sorted(os.listdir(q.parent)), ["old.json"])
        # (c) writing the bytes fails half way
        with mock.patch.object(data.os, "replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                data.atomic_write_bytes(q, b"x")
        self.assertEqual(sorted(os.listdir(q.parent)), ["old.json"])

    def test_helpers(self):
        self.assertTrue(data.price_signature.__doc__)


class Signature(DataCase):
    def test_price_signature_detects_any_revision(self):
        ds = history(260)
        c = random_walk(260, 4)
        self.route_symbol("2330.TW", ds, c)
        self.route_names(twse=[])
        a = data.get_price_data("2330")
        s1 = data.price_signature(a)
        self.assertEqual(s1, data.price_signature(a))
        b = data.get_price_data("2330")
        self.assertEqual(s1, data.price_signature(b))
        b.adjclose[10] *= 1.0000001
        self.assertNotEqual(s1, data.price_signature(b))
        self.assertNotEqual(s1, data.price_signature(a.slice(250)))

    def test_price_signature_covers_every_model_input(self):
        # audit C-05: the models read open / high / low / volume as well (range volatility, volume z-score)
        ds = history(260)
        self.route_symbol("2330.TW", ds, random_walk(260, 4))
        self.route_names(twse=[])
        a = data.get_price_data("2330")
        s0 = data.price_signature(a)
        for field in ("open", "high", "low", "volume", "close", "adjclose"):
            b = data.get_price_data("2330")
            getattr(b, field)[-3] *= 1.001
            self.assertNotEqual(s0, data.price_signature(b), field)
        b = data.get_price_data("2330")
        b.dates = b.dates.copy()
        b.dates[10] = b.dates[10] + 1
        self.assertNotEqual(s0, data.price_signature(b), "dates")
        self.assertEqual(s0, data.price_signature(data.get_price_data("2330")))  # and it is deterministic

    def test_avg_traded_value(self):
        ds = history(260)
        self.route_symbol("2330.TW", ds, [100.0] * 260, volume=[2000] * 260)
        self.route_names(twse=[])
        pd = data.get_price_data("2330")
        self.assertAlmostEqual(data.avg_traded_value(pd), 200000.0)
        self.route_symbol("2331.TW", ds, [100.0] * 260, volume=[0] * 260)
        self.assertIsNone(data.avg_traded_value(data.get_price_data("2331")))


if __name__ == "__main__":
    unittest.main()
