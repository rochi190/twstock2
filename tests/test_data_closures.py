"""Audit P-01: days on which the exchange was closed without a schedule entry (typhoon days, 2026-10-23) arrive from
Yahoo as flat zero-volume placeholder bars.  The data layer drops those bars AND learns the closure for the calendar
(``is_trading_day`` / ``last_trading_date_expected`` / ``next_trading_day``), otherwise every symbol would look stale
(last bar 10-22 < "expected" 10-23) and the as-of chip would blame the wrong thing.  All network traffic is faked."""
import json
import os
import sys
import unittest
from datetime import date, datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from data_fakes import TAIPEI, DataCase, bdays, random_walk, ts_of, yahoo_payload  # noqa: E402

from backend import data  # noqa: E402

CLOSED = date(2026, 10, 23)  # Friday, no TWSE schedule entry, Yahoo serves placeholders
LAST_REAL = date(2026, 10, 22)


def bday_history(n, end):
    ds = [d for d in bdays(end - timedelta(days=int(n * 1.6) + 10), 10_000) if d <= end][-n:]
    assert len(ds) == n and ds[-1] == end
    return ds


class ClosureCase(DataCase):
    START = datetime(2026, 10, 23, 15, 0, tzinfo=TAIPEI)  # Friday after the close: the day Yahoo served the placeholders

    CODES = ("2330", "2317", "2454", "2308", "2382")

    def setUp(self):
        super().setUp()
        self.ds = bday_history(300, CLOSED)  # ends with the closure day
        self.route_names(twse=[(c, f"名稱{c}", 10 ** 10, 100) for c in self.CODES])

    # ---- payload builders
    def series(self, seed, placeholder=True):
        """300 rows ending on CLOSED; the last row is Yahoo's filler (flat, volume 0, the previous close) unless disabled."""
        close = random_walk(300, seed)
        o, h, l = close.copy(), close * 1.01, close * 0.99
        vol = [1_000_000 + 1000 * i for i in range(300)]
        if placeholder:
            close[-1] = close[-2]
            o[-1] = h[-1] = l[-1] = close[-1]
            vol[-1] = 0
        return close, o, h, l, vol

    def route_stock(self, code, placeholder=True, seed=None):
        close, o, h, l, vol = self.series(seed or int(code), placeholder)
        self.net.route(f"/chart/{code}.TW?", yahoo_payload(self.ds, close, open_=o, high=h, low=l, volume=vol))

    def route_index(self, null_closed=True):
        close = random_walk(300, 99, 15000)
        nulls = (299,) if null_closed else ()
        self.net.route("/chart/%5ETWII?", yahoo_payload(self.ds, close, volume=[0] * 300, itype="INDEX", nulls=nulls))

    def closed(self):
        return CLOSED in data.extra_closed_days()


class Quorum(ClosureCase):
    def test_placeholder_bar_is_dropped_and_asof_is_the_last_real_session(self):
        self.route_stock("2330")
        pd = data.get_price_data("2330")
        self.assertEqual((pd.n, str(pd.dates[-1])), (299, "2026-10-22"))
        self.assertGreater(float(pd.volume[-1]), 0.0)

    def test_three_stocks_with_a_placeholder_declare_the_closure(self):
        for code in self.CODES[:2]:
            self.route_stock(code)
            data.get_price_data(code)
        self.assertFalse(self.closed())
        self.assertTrue(data.is_trading_day(CLOSED))  # two stocks are not a quorum yet
        self.route_stock(self.CODES[2])
        data.get_price_data(self.CODES[2])
        self.assertTrue(self.closed())
        self.assertFalse(data.is_trading_day(CLOSED))
        # the calendar follows: today (Fri after the open) is not a trading day, so the newest bar that can exist is Thu's
        self.assertEqual(data.last_trading_date_expected(), LAST_REAL)
        self.assertFalse(data.market_is_open(datetime(2026, 10, 23, 10, 0, tzinfo=TAIPEI)))
        # next trading day after Thursday skips the closure, the weekend and the scheduled holiday Mon 10-26
        self.assertEqual(data.next_trading_day(LAST_REAL), date(2026, 10, 27))

    def test_one_stock_is_not_a_closure(self):
        self.route_stock("2330")
        data.get_price_data("2330")
        self.assertFalse(self.closed())
        self.assertEqual(data.last_trading_date_expected(), CLOSED)

    def test_halted_stocks_are_not_a_closure(self):
        # 20 stocks, 3 of them halted (placeholder), 17 traded: 15 % is not "nearly all"
        for i in range(20):
            code = f"{1100 + i}"
            self.route_stock(code, placeholder=i < 3, seed=i + 1)
            data.get_price_data(code)
        self.assertFalse(self.closed())

    def test_the_majority_rule_follows_the_number_of_real_bars(self):
        # 3 placeholders and 1 real bar: 75 % < 80 % -> no closure; a second placeholder-only stock tips it to 4/5 = 80 %
        for i, ph in enumerate((True, True, True, False)):
            code = f"{1100 + i}"
            self.route_stock(code, placeholder=ph, seed=i + 1)
            data.get_price_data(code)
        self.assertFalse(self.closed())
        self.route_stock("1199")
        data.get_price_data("1199")
        self.assertTrue(self.closed())

    def test_stock_refresh_stops_once_the_closure_is_known(self):
        for code in self.CODES[:3]:
            self.route_stock(code)
            data.get_price_data(code)
        self.assertTrue(self.closed())
        n = self.net.count("/chart/")
        self.clock.advance(minutes=30)  # throttle long over; the cache ends 10-22 == the newest bar that can exist
        for code in self.CODES[:3]:
            data.get_price_data(code)
        self.assertEqual(self.net.count("/chart/"), n)

    def test_known_closure_survives_a_restart(self):
        for code in self.CODES[:3]:
            self.route_stock(code)
            data.get_price_data(code)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "closures.json")))
        self.reset_memory()
        self.assertTrue(self.closed())

    def test_nothing_is_learned_before_the_session_is_final(self):
        self.clock.dt = datetime(2026, 10, 23, 12, 0, tzinfo=TAIPEI)  # still "in session": Yahoo's pre-open filler
        for code in self.CODES[:4]:
            self.route_stock(code)
            data.get_price_data(code)
        self.assertFalse(self.closed())
        self.assertTrue(data.is_trading_day(CLOSED))
        self.clock.dt = datetime(2026, 10, 23, 14, 1, tzinfo=TAIPEI)
        self.assertTrue(self.closed())  # the same evidence counts once the day is over

    def test_old_filler_does_not_enter_the_tally(self):
        # 25 years of history contain holiday filler runs; only the last weeks are evidence
        ds = bday_history(300, LAST_REAL)
        close = random_walk(300, 1)
        o, h, l, vol = close.copy(), close * 1.01, close * 0.99, [1_000_000] * 300
        for i in (100, 101, 102):  # three old flat zero-volume rows
            close[i] = close[99]
            o[i] = h[i] = l[i] = close[99]
            vol[i] = 0
        self.net.route("/chart/2330.TW?", yahoo_payload(ds, close, open_=o, high=h, low=l, volume=vol))
        pd = data.get_price_data("2330")
        self.assertEqual(pd.n, 297)  # dropped
        with open(os.path.join(self.tmp, "closures.json"), encoding="utf-8") as fh:
            days = json.load(fh)["days"]
        self.assertTrue(all(date.fromisoformat(k) >= date(2026, 10, 1) for k in days), sorted(days))
        self.assertTrue(all(not e["ph"] for e in days.values()))


class IndexEvidence(ClosureCase):
    def test_index_null_row_plus_a_stock_placeholder_is_a_closure(self):
        self.route_stock("2330")
        self.route_index(null_closed=True)
        data.get_price_data("2330")
        data.get_market_data()
        self.assertTrue(self.closed())

    def test_index_without_a_null_row_does_not_count(self):
        self.route_stock("2330")
        self.route_index(null_closed=False)
        data.get_price_data("2330")
        data.get_market_data()
        self.assertFalse(self.closed())

    def test_a_real_stock_bar_takes_the_closure_back(self):
        self.route_stock("2330")
        self.route_index(null_closed=True)
        data.get_price_data("2330")
        data.get_market_data()
        self.assertTrue(self.closed())
        self.route_stock("2317", placeholder=False)  # somebody traded that day after all
        data.get_price_data("2317")
        self.assertFalse(self.closed())

    def test_index_series_keeps_flat_zero_volume_bars(self):
        # an index has no volume: its bars are never placeholders (only the null rows are missing days)
        close = random_walk(300, 99, 15000)
        self.net.route("/chart/%5ETWII?", yahoo_payload(self.ds, close, volume=[0] * 300, itype="INDEX"))
        m = data.get_market_data()
        self.assertEqual(m.n, 300)


class CalendarFallsBackToTheSchedule(ClosureCase):
    def test_without_any_evidence_the_calendar_is_the_published_schedule(self):
        self.assertEqual(data.extra_closed_days(), frozenset())
        self.assertTrue(data.is_trading_day(CLOSED))
        self.assertFalse(data.is_trading_day(date(2026, 10, 26)))  # the scheduled holiday is still excluded


if __name__ == "__main__":
    unittest.main()
