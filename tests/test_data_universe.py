"""Data layer: trading calendar, name tables / search, universe, upstream circuit breaker.  All traffic is faked."""
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
from data_fakes import (HOLIDAYS_2026, TAIPEI, DataCase, holiday_json, random_walk, tpex_day, twse_company,  # noqa: E402
                        twse_day_all)
from test_data_cache import history  # noqa: E402

from backend import config, data  # noqa: E402


def at(y, m, d, hh=12, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=TAIPEI)


class Calendar(DataCase):
    def test_trading_days(self):
        t = data.is_trading_day
        self.assertTrue(t(date(2026, 10, 22)))  # Thursday
        self.assertFalse(t(date(2026, 10, 24)))  # Saturday
        self.assertFalse(t(date(2026, 10, 25)))  # Sunday
        self.assertFalse(t(date(2026, 10, 9)))  # National Day (holiday on a Friday)
        self.assertFalse(t(date(2026, 10, 26)))  # Retrocession Day (Monday)
        self.assertFalse(t(date(2026, 2, 12)))  # "market closed, settlement only"
        self.assertTrue(t(date(2026, 1, 2)))  # "開始交易日" marker rows are TRADING days
        self.assertTrue(t(date(2026, 2, 11)))  # "最後交易日" marker row
        self.assertTrue(t(date(2026, 2, 23)))  # "春節後開始交易日"

    def test_next_and_previous_trading_day_skip_weekends_and_holidays(self):
        self.assertEqual(data.next_trading_day(date(2026, 10, 22)), date(2026, 10, 23))
        self.assertEqual(data.next_trading_day(date(2026, 10, 23)), date(2026, 10, 27))  # Sat, Sun, holiday Mon
        self.assertEqual(data.next_trading_day(date(2026, 10, 8)), date(2026, 10, 12))  # Fri 10-09 + weekend... 10-10 Sat
        self.assertEqual(data.next_trading_day(date(2026, 2, 11)), date(2026, 2, 23))  # Lunar New Year closure
        self.assertEqual(data.prev_trading_day(date(2026, 10, 27)), date(2026, 10, 23))
        self.assertEqual(data.prev_trading_day(date(2026, 2, 23)), date(2026, 2, 11))
        for d in (date(2026, 10, 23), date(2026, 2, 11), date(2026, 4, 2)):
            self.assertTrue(data.is_trading_day(data.next_trading_day(d)))

    def test_last_trading_date_expected(self):
        f = data.last_trading_date_expected
        self.assertEqual(f(at(2026, 10, 22, 16)), date(2026, 10, 22))
        self.assertEqual(f(at(2026, 10, 22, 8, 59)), date(2026, 10, 21))  # before the open: yesterday's bar
        self.assertEqual(f(at(2026, 10, 22, 9, 0)), date(2026, 10, 22))
        self.assertEqual(f(at(2026, 10, 24, 12)), date(2026, 10, 23))  # Saturday
        self.assertEqual(f(at(2026, 10, 26, 10)), date(2026, 10, 23))  # holiday Monday
        self.assertEqual(f(at(2026, 10, 27, 8)), date(2026, 10, 23))
        self.assertEqual(f(at(2026, 10, 27, 9, 30)), date(2026, 10, 27))
        self.assertEqual(f(None), date(2026, 10, 22))  # uses the (patched) clock
        # naive datetimes are Taipei local, other zones are converted
        self.assertEqual(f(datetime(2026, 10, 22, 16, 0)), date(2026, 10, 22))
        from datetime import timezone
        self.assertEqual(f(datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)), date(2026, 10, 22))  # 07:00 Taipei Fri: before the open
        self.assertEqual(f(datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc)), date(2026, 10, 23))

    def test_market_is_open(self):
        o = data.market_is_open
        self.assertFalse(o(at(2026, 10, 22, 8, 59)))
        self.assertTrue(o(at(2026, 10, 22, 9, 0)))
        self.assertTrue(o(at(2026, 10, 22, 13, 29)))
        self.assertFalse(o(at(2026, 10, 22, 13, 30)))
        self.assertFalse(o(at(2026, 10, 24, 10)))
        self.assertFalse(o(at(2026, 10, 9, 10)))  # holiday
        self.assertFalse(o(at(2026, 10, 22, 20)))

    def test_unknown_year_falls_back_to_weekends_only(self):
        self.assertTrue(data.is_trading_day(date(2027, 1, 4)))
        self.assertFalse(data.is_trading_day(date(2027, 1, 2)))  # Saturday

    def test_bad_now_type(self):
        with self.assertRaises(TypeError):
            data.taipei_now("noon")

    def test_final_time_is_1400(self):
        self.assertEqual(data.final_time(date(2026, 10, 22)), at(2026, 10, 22, 14))


class HolidayFetching(DataCase):
    HOLIDAYS = False

    def setUp(self):
        super().setUp()
        self.net.route("holidaySchedule?response=json&queryYear=2026", holiday_json(2026, HOLIDAYS_2026))
        self.net.route("holidaySchedule?response=json", data.NotFoundError("404"))

    def test_fetch_persist_and_reuse_from_disk(self):
        self.assertFalse(data.is_trading_day(date(2026, 10, 9)))
        n = self.net.count("holidaySchedule")
        self.assertGreaterEqual(n, 1)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "holidays_2026.json")))
        self.assertFalse(data.is_trading_day(date(2026, 10, 10)))
        self.assertEqual(self.net.count("holidaySchedule"), n)  # memoised
        data.reset_memory_caches()  # a restart: the file answers
        self.assertFalse(data.is_trading_day(date(2026, 10, 9)))
        self.assertEqual(self.net.count("holidaySchedule"), n)

    def test_stale_copy_survives_a_failed_refresh_and_failures_are_throttled(self):
        data.holidays(2026)
        self.clock.advance(days=4)  # older than HOLIDAY_TTL (3 days) -> a refresh is attempted
        data.reset_memory_caches()
        self.net.down("holidaySchedule")
        self.assertFalse(data.is_trading_day(date(2026, 10, 9)))  # old copy still used
        n = self.net.count("holidaySchedule")
        for _ in range(5):
            data.is_trading_day(date(2026, 10, 9))
        self.assertEqual(self.net.count("holidaySchedule"), n)  # backoff: no hammering
        self.clock.advance(minutes=11)
        data.is_trading_day(date(2026, 10, 9))
        self.assertGreater(self.net.count("holidaySchedule"), n)

    def test_unreachable_schedule_means_weekdays_only(self):
        self.net.down("holidaySchedule")
        self.assertTrue(data.is_trading_day(date(2026, 10, 9)))  # unknowable offline
        self.assertFalse(data.is_trading_day(date(2026, 10, 10)))  # Saturday is always closed

    def test_concurrent_callers_download_once(self):
        self.net.delay = 0.1
        ts = [threading.Thread(target=data.is_trading_day, args=(date(2026, 10, 9),)) for _ in range(10)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(self.net.count("holidaySchedule?response=json&queryYear=2026"), 1)

    def test_parse_formats(self):
        self.assertIsNone(data.parse_holiday_schedule({"stat": "很抱歉，沒有符合條件的資料!"}, 2026))
        self.assertIsNone(data.parse_holiday_schedule("junk", 2026))
        self.assertIsNone(data.parse_holiday_schedule({"stat": "OK", "data": [["bad", "x", "y"]]}, 2026))
        openapi = [{"Date": "1151009", "Name": "國慶日", "Description": ""}, {"Date": "1150102", "Name": "國曆新年開始交易日", "Description": ""},
                   {"Date": "1141225", "Name": "行憲紀念日", "Description": ""}]
        got = data.parse_holiday_schedule(openapi, 2026)
        self.assertEqual(got, frozenset({date(2026, 10, 9)}))  # trading-marker rows and other years are dropped


class Search(DataCase):
    def setUp(self):
        super().setUp()
        self.route_names(
            twse=[("2330", "台積電", 100 * 10 ** 9, 1000), ("2331", "精英", 10 ** 9, 20), ("2317", "鴻海", 50 * 10 ** 9, 200),
                  ("3711", "日月光投控", 8 * 10 ** 9, 150), ("0050", "元大台灣50", 9 * 10 ** 9, 100),
                  ("00878", "國泰永續高股息", 7 * 10 ** 9, 20), ("2915", "潤泰全", 10 ** 8, 60), ("5871", "中租-KY", 3 * 10 ** 9, 100),
                  ("2427", "三商電", 10 ** 8, 30), ("2905", "三商", 2 * 10 ** 8, 30), ("9904", "寶成", 3 * 10 ** 8, 30),
                  ("2834", "臺企銀", 10 ** 8, 12), ("3045", "台灣大", 10 ** 9, 100)],
            tpex=[("6488", "環球晶", 7 * 10 ** 9, 400), ("8069", "元太", 6 * 10 ** 9, 200)])

    def codes(self, q, **kw):
        return [r["code"] for r in data.search_instruments(q, **kw)]

    def test_code_prefix_and_exact(self):
        self.assertEqual(self.codes("2330"), ["2330"])
        self.assertEqual(self.codes("233"), ["2330", "2331"])  # prefix, ties by traded value
        self.assertEqual(self.codes("00")[:2], ["0050", "00878"])
        self.assertEqual(self.codes("0050")[0], "0050")

    def test_chinese_name_substring_and_ranking(self):
        self.assertEqual(self.codes("台積"), ["2330"])
        self.assertEqual(self.codes("三商"), ["2905", "2427"])  # name-prefix hit on both; liquid first
        self.assertEqual(self.codes("電"), ["2330", "2427", ])  # substring hits (both end in 電)
        self.assertEqual(self.codes("元")[:2], ["0050", "8069"])  # both are name-prefix hits: the more liquid first

    def test_rank_exact_code_before_name_hits(self):
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1), ("9999", "2330概念股", 10 ** 12, 1)])
        data._names_table(refresh=True)
        self.assertEqual(self.codes("2330"), ["2330", "9999"])

    def test_tai_variants_fold(self):
        self.assertEqual(self.codes("臺灣大"), ["3045"])
        self.assertEqual(self.codes("台企銀"), ["2834"])
        self.assertEqual(self.codes("臺積"), ["2330"])

    def test_case_and_width_insensitive(self):
        self.assertEqual(self.codes("ky"), ["5871"])
        self.assertEqual(self.codes("ＫＹ"), ["5871"])
        self.assertEqual(self.codes(" ２３３０ "), ["2330"])

    def test_otc_etf_fields(self):
        r = data.search_instruments("6488")[0]
        self.assertEqual(r, {"code": "6488", "name": "環球晶", "market": "TPEx", "kind": "stock"})
        r = data.search_instruments("00878")[0]
        self.assertEqual((r["kind"], r["market"]), ("etf", "TWSE"))
        self.assertEqual(set(r), {"code", "name", "market", "kind"})  # no internal fields leak

    def test_limit(self):
        self.assertEqual(len(data.search_instruments("2", limit=3)), 3)
        self.assertEqual(len(data.search_instruments("2", limit=0)), 1)  # clamped to >= 1

    def test_no_match(self):
        self.assertEqual(data.search_instruments("不存在的名稱"), [])

    def test_instrument_info(self):
        self.assertEqual(data.instrument_info("2330"), {"code": "2330", "name": "台積電", "market": "TWSE", "kind": "stock"})
        self.assertEqual(data.instrument_info("6488")["market"], "TPEx")
        self.assertEqual(data.instrument_info("2330.TW")["code"], "2330")
        self.assertEqual(data.instrument_info("^TWII"), {"code": "^TWII", "name": "加權指數", "market": "INDEX", "kind": "index"})
        for bad in ("9999", "abc", "", None):
            with self.assertRaises(KeyError):
                data.instrument_info(bad)

    def test_resolve_symbol_by_exact_name(self):
        self.assertEqual(data.resolve_symbol("台積電"), "2330")
        self.assertEqual(data.resolve_symbol("臺企銀"), "2834")
        self.assertEqual(data.resolve_symbol("2330"), "2330")
        with self.assertRaises(data.BadSymbol):
            data.resolve_symbol("台積")  # partial names are for search, not for resolving
        with self.assertRaises(data.BadSymbol):
            data.resolve_symbol("")


class SearchOffline(DataCase):
    """No name table reachable at all: the curated lists keep the app usable."""

    def test_fallback_names(self):
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        self.assertEqual([r["code"] for r in data.search_instruments("台積")], ["2330"])
        self.assertEqual(data.instrument_info("2317")["name"], "鴻海")
        self.assertEqual(data.instrument_info("0050")["kind"], "etf")
        self.assertEqual([r["code"] for r in data.search_instruments("00878")][:1], ["00878"])
        pop = data.search_instruments("")
        self.assertEqual(len(pop), 15)
        calls = len(self.net.calls)
        data.search_instruments("鴻海")
        self.assertEqual(len(self.net.calls), calls)  # failure is remembered for a few minutes

    def test_any_otc_code_still_works_by_price_cache(self):
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        ds = history(260)
        self.net.route("/chart/6488.TW?", data.NotFoundError("404"))
        self.route_symbol("6488.TWO", ds, random_walk(260, 3), name="GlobalWafers")
        pd = data.get_price_data("6488")
        self.assertEqual((pd.market, pd.name), ("TPEx", "GlobalWafers"))  # no Chinese name known, English fallback
        self.assertEqual(data.instrument_info("6488")["market"], "TPEx")


class NameRefresh(DataCase):
    def setUp(self):
        super().setUp()
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000)], tpex=[("6488", "環球晶", 10 ** 9, 400)])

    def test_table_is_cached_on_disk_for_a_day(self):
        data.search_instruments("台積")
        n = len(self.net.calls)
        self.assertEqual(n, 3)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "names.json")))
        data.reset_memory_caches()
        data.search_instruments("台積")
        self.assertEqual(len(self.net.calls), n)

    def test_expired_table_is_served_at_once_and_refreshed_in_the_background(self):
        from unittest import mock
        p = mock.patch.object(data, "NAMES_BACKGROUND_REFRESH", True)
        p.start()
        self.addCleanup(p.stop)
        data.search_instruments("台積")
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000), ("2454", "聯發科", 10 ** 10, 1000)])
        self.clock.advance(hours=25)
        self.net.delay = 0.3
        t0 = time.time()
        res = data.search_instruments("聯發")
        self.assertLess(time.time() - t0, 0.25)  # did not wait for the download
        self.assertEqual(res, [])  # the old table does not know the name yet
        th = data._NAMES_BG["thread"]
        self.assertIsNotNone(th)
        th.join(10)
        self.net.delay = 0
        self.assertEqual([r["code"] for r in data.search_instruments("聯發")], ["2454"])

    def test_synchronous_refresh_when_background_is_off(self):
        data.search_instruments("台積")
        self.route_names(twse=[("2330", "台積電", 10 ** 10, 1000), ("2454", "聯發科", 10 ** 10, 1000)])
        self.clock.advance(hours=25)
        self.assertEqual([r["code"] for r in data.search_instruments("聯發")], ["2454"])

    def test_failed_refresh_keeps_serving_the_old_table_and_backs_off(self):
        data.search_instruments("台積")
        self.clock.advance(hours=25)
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        self.assertEqual([r["code"] for r in data.search_instruments("台積")], ["2330"])
        n = len(self.net.calls)
        for _ in range(5):
            data.search_instruments("台積")
        self.assertEqual(len(self.net.calls), n)  # 5 minute back-off

    def test_flaky_tpex_does_not_forget_known_otc_names(self):
        data.search_instruments("環球")
        self.net.down("tpex_mainboard")
        self.clock.advance(hours=25)
        self.assertEqual([r["code"] for r in data.search_instruments("環球")], ["6488"])
        self.assertEqual(data.instrument_info("6488")["market"], "TPEx")
        # an incomplete table expires after 30 minutes only, so TPEx is retried soon
        tbl = data._names_table(network=False)
        self.assertLess(tbl["expires_ts"] - tbl["fetched_ts"], 3600)

    def test_thundering_herd_on_a_dead_source_makes_one_attempt(self):
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        self.net.delay = 0.05
        ts = [threading.Thread(target=data.search_instruments, args=("台積",)) for _ in range(12)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(self.net.count("STOCK_DAY_ALL"), 1)  # not 12 sequential attempts

    def test_market_hint_for_first_symbol_request(self):
        ds = history(260)
        self.net.route("/chart/6488.TW?", data.NotFoundError("404"))
        self.route_symbol("6488.TWO", ds, random_walk(260, 1))
        data.get_price_data("6488")
        self.assertEqual(self.net.count("6488.TW?"), 0)  # went straight to .TWO thanks to the TPEx table


class Universe(DataCase):
    def setUp(self):
        super().setUp()
        cur = data.FALLBACK_UNIVERSE
        rows = [(c, nm, 10 ** 11 - i * 10 ** 8, 100) for i, (c, nm) in enumerate(cur)]
        rows += [("0050", "元大台灣50", 10 ** 12, 100), ("00878", "國泰永續高股息", 10 ** 11, 20)]  # ETFs must never enter
        rows += [(f"{7000 + i}", f"高成交{i}", 5 * 10 ** 9 - i * 10 ** 6, 50) for i in range(30)]  # the top-up pool
        rows += [("7999", "無成交價", 9 * 10 ** 10, 0), ("8001", "無成交值", 0, 10)]
        self.rows = rows
        self.route_names(twse=rows)

    def test_default_universe_has_60_liquid_stocks_and_no_etfs(self):
        u = data.get_universe()
        self.assertEqual(len(u), config.UNIVERSE_SIZE)
        self.assertEqual(set(u[0]), {"code", "name", "market", "kind"})
        self.assertTrue(all(i["kind"] == "stock" and len(i["code"]) == 4 for i in u))
        self.assertEqual(len({i["code"] for i in u}), len(u))
        self.assertEqual([i["code"] for i in u[:3]], ["2330", "2317", "2454"])
        self.assertNotIn("0050", [i["code"] for i in u])

    def test_curated_first_then_top_traded_value(self):
        u = data.get_universe(70)
        codes = [i["code"] for i in u]
        n_cur = len(data.FALLBACK_UNIVERSE)
        self.assertEqual(codes[:n_cur], [c for c, _ in data.FALLBACK_UNIVERSE])
        self.assertEqual(codes[n_cur:], ["7000", "7001", "7002", "7003", "7004", "7005", "7006"][: 70 - n_cur])  # by traded value
        self.assertNotIn("7999", codes)  # no price
        self.assertNotIn("8001", codes)  # no traded value

    def test_live_names_override_curated_names_and_vanished_codes_are_replaced(self):
        rows = [r for r in self.rows if r[0] != "2002"]  # China Steel "delisted"
        rows = [("2330", "台灣積體製造", 10 ** 11, 100) if r[0] == "2330" else r for r in rows]
        self.route_names(twse=rows)
        u = data.get_universe(62, refresh=True)
        codes = [i["code"] for i in u]
        self.assertNotIn("2002", codes)
        self.assertEqual(u[0]["name"], "台灣積體製造")
        self.assertEqual(len(u), 62)

    def test_cached_for_a_week_and_stable(self):
        a = data.get_universe()
        n = len(self.net.calls)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "universe.json")))
        self.route_names(twse=[("2330", "台積電", 1, 1)])  # the market "changes" ...
        self.clock.advance(days=3)
        self.assertEqual(data.get_universe(), a)  # ... the list does not
        self.assertEqual(len(self.net.calls), n)
        data.reset_memory_caches()
        self.assertEqual(data.get_universe(), a)  # survives restarts
        self.clock.advance(days=5)  # > 7 days since creation
        data.reset_memory_caches()
        os.remove(os.path.join(self.tmp, "names.json"))
        self.route_names(twse=self.rows)
        self.assertEqual(len(data.get_universe()), 60)
        self.assertGreater(len(self.net.calls), n)

    def test_asking_for_more_than_cached_rebuilds(self):
        data.get_universe(10)
        eligible = len(data.FALLBACK_UNIVERSE) + 30  # curated + the 30-name top-up pool of this fixture
        self.assertEqual(len(data.get_universe(100)), min(100, eligible))
        self.assertEqual(len(data.get_universe(5)), 5)

    def test_offline_uses_curated_list_and_never_overwrites_a_live_ranking(self):
        live = data.get_universe()
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        data.reset_memory_caches()
        os.remove(os.path.join(self.tmp, "names.json"))
        self.assertEqual(data.get_universe(refresh=True), live)  # live ranking preserved when the refresh degrades

    def test_offline_cold_start_gives_curated_60(self):
        self.net.down("openapi.twse")
        self.net.down("tpex.org")
        u = data.get_universe()
        self.assertEqual(len(u), 60)
        self.assertEqual(u[0], {"code": "2330", "name": "台積電", "market": "TWSE", "kind": "stock"})
        with open(os.path.join(self.tmp, "universe.json")) as fh:
            self.assertEqual(json.load(fh)["source"], "fallback")
        # a degraded list is retried after an hour
        self.net.routes.pop("openapi.twse")
        self.net.routes.pop("tpex.org")
        self.clock.advance(minutes=61)
        data.reset_memory_caches()
        data.get_universe()
        with open(os.path.join(self.tmp, "universe.json")) as fh:
            self.assertEqual(json.load(fh)["source"], "live")

    def test_popular_list_for_empty_query(self):
        pop = data.search_instruments("")
        codes = [r["code"] for r in pop]
        self.assertEqual(len(codes), 15)  # first 12 of the universe + 0050 / 0056 / 00878
        self.assertEqual(codes[:12], [i["code"] for i in data.get_universe()[:12]])
        self.assertEqual(codes[12:], ["0050", "0056", "00878"])  # 0056 comes from the curated ETF list
        self.assertEqual(data.search_instruments("   "), pop)
        self.assertEqual(pop[0]["name"], "台積電")
        self.assertEqual(data.search_instruments("", limit=5), pop[:5])

    def test_build_universe_pure_function(self):
        items = data.parse_name_tables([{"Code": "2330", "Name": "台積電", "TradeValue": "9", "ClosingPrice": "1"},
                                        {"Code": "7001", "Name": "乙", "TradeValue": "5", "ClosingPrice": "1"},
                                        {"Code": "7002", "Name": "丙", "TradeValue": "7", "ClosingPrice": "1"},
                                        {"Code": "0050", "Name": "ETF", "TradeValue": "99", "ClosingPrice": "1"}], None, None)
        u = data.build_universe(items, 5)
        self.assertEqual([i["code"] for i in u], ["2330", "7002", "7001"])
        self.assertEqual([i["code"] for i in data.build_universe(None, 3)], ["2330", "2317", "2454"])


class CircuitBreaker(DataCase):
    def setUp(self):
        super().setUp()
        self.route_names(twse=[])
        self.ds = history(260)
        self.codes = [f"{1300 + i}" for i in range(8)]
        for c in self.codes:
            self.route_symbol(f"{c}.TW", self.ds, random_walk(260, int(c)))

    def test_opens_after_three_failures_then_fails_fast_then_recovers(self):
        self.net.down("/chart/")
        for c in self.codes[:3]:
            with self.assertRaises(data.DataUnavailable):
                data.get_price_data(c)
        n = self.net.count("/chart/")
        self.assertEqual(n, 3)
        for c in self.codes[3:6]:
            with self.assertRaises(data.DataUnavailable) as cm:
                data.get_price_data(c)
            self.assertIn("recently", str(cm.exception))
        self.assertEqual(self.net.count("/chart/"), n)  # no traffic while open
        self.clock.advance(seconds=data.CIRCUIT_OPEN_SECONDS + 1)
        self.net.routes.pop("/chart/")
        self.assertEqual(data.get_price_data(self.codes[6]).n, 260)

    def test_stale_cache_is_served_instantly_while_open(self):
        for c in self.codes[:2]:
            data.get_price_data(c)
        self.clock.advance(days=1, hours=-6)  # next morning 10:00: a new bar is expected
        self.clock.mono += 3600
        self.net.down("/chart/")
        for c in self.codes[2:5]:
            with self.assertRaises(data.DataUnavailable):
                data.get_price_data(c)
        n = self.net.count("/chart/")
        pd = data.get_price_data(self.codes[0])
        self.assertTrue(pd.stale)
        self.assertEqual(self.net.count("/chart/"), n)  # served without trying

    def test_not_found_does_not_count_as_an_outage(self):
        for i in range(10):
            self.net.route(f"/chart/{9000 + i}.TW?", data.NotFoundError("404"))
            self.net.route(f"/chart/{9000 + i}.TWO?", data.NotFoundError("404"))
            with self.assertRaises(data.UnknownSymbol):
                data.get_price_data(f"{9000 + i}")
        self.assertEqual(data.get_price_data(self.codes[0]).n, 260)

    def test_success_resets_the_failure_streak(self):
        self.net.down("/chart/")
        for c in self.codes[:2]:
            with self.assertRaises(data.DataUnavailable):
                data.get_price_data(c)
        self.net.routes.pop("/chart/")
        data.get_price_data(self.codes[2])  # success
        self.net.down("/chart/")
        for c in self.codes[3:5]:
            with self.assertRaises(data.DataUnavailable):
                data.get_price_data(c)
        self.net.routes.pop("/chart/")
        self.assertEqual(data.get_price_data(self.codes[5]).n, 260)  # only 2 consecutive failures: still closed

    def test_fewer_retries_when_a_cache_exists(self):
        seen = []
        orig = self.net.__call__

        def spy(url, **kw):
            if "/chart/" in url:
                seen.append(kw.get("retries"))
            return orig(url, **kw)

        data.get_price_data(self.codes[0])
        self.clock.advance(days=1, hours=-6)
        self.clock.mono += 3600
        self.net.down("/chart/")
        from unittest import mock
        with mock.patch.object(data, "_http_get", spy):
            data.get_price_data(self.codes[0])  # stale fallback available
            with self.assertRaises(data.DataUnavailable):
                data.get_price_data(self.codes[1])  # nothing to fall back on
        self.assertEqual(seen, [2, data.HTTP_RETRIES])


class OfflineSwitch(unittest.TestCase):
    """TWQ_OFFLINE=1 / TWQ_YAHOO_CHART are the documented ways to simulate a dead network."""

    def test_offline_env_makes_every_request_fail_at_once(self):
        os.environ["TWQ_OFFLINE"] = "1"
        try:
            t0 = time.time()
            with self.assertRaises(data.DataUnavailable):
                data._http_get("http://127.0.0.1:1/x")
            self.assertLess(time.time() - t0, 0.5)
        finally:
            del os.environ["TWQ_OFFLINE"]

    def test_cache_dir_env(self):
        os.environ["TWQ_CACHE_DIR"] = "/tmp/twq_x_cache"
        try:
            self.assertEqual(str(data.cache_dir()), "/tmp/twq_x_cache")
        finally:
            del os.environ["TWQ_CACHE_DIR"]
        self.assertEqual(data.cache_dir(), config.CACHE_DIR)


if __name__ == "__main__":
    unittest.main()
