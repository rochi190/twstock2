"""Data layer: Yahoo payload parsing, symbol normalisation, name-table parsing (no network)."""
import os
import sys
import unittest
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from data_fakes import bdays, ts_of, yahoo_error, yahoo_payload  # noqa: E402

from backend import data  # noqa: E402


def day_ints(ds):
    return np.array([(d - date(1970, 1, 1)).days for d in ds])


class ParseYahoo(unittest.TestCase):
    def test_basic_columns_and_dates(self):
        ds = bdays(date(2026, 10, 5), 6)
        close = [10, 11, 12, 11.5, 12.5, 13]
        b = data.parse_yahoo_chart(yahoo_payload(ds, close, volume=[100, 200, 300, 400, 500, 600]))
        np.testing.assert_array_equal(b["d"], day_ints(ds))
        np.testing.assert_allclose(b["c"], close)
        np.testing.assert_allclose(b["adj"], close)
        np.testing.assert_allclose(b["v"], [100, 200, 300, 400, 500, 600])
        self.assertTrue(np.all(b["h"] >= np.maximum(b["o"], b["c"])))
        self.assertTrue(np.all(b["l"] <= np.minimum(b["o"], b["c"])))

    def test_float32_noise_is_rounded_away(self):
        # Yahoo serves float32 representations such as 106.9000015258789 for a price of 106.9
        ds = bdays(date(2026, 10, 5), 3)
        noisy = [106.9000015258789, 27.950000762939453, 61.900001525878906]
        b = data.parse_yahoo_chart(yahoo_payload(ds, noisy))
        np.testing.assert_array_equal(b["c"], [106.9, 27.95, 61.9])
        np.testing.assert_array_equal(b["o"], [106.9, 27.95, 61.9])
        self.assertTrue(np.all(b["h"] >= np.maximum(b["o"], b["c"])))
        self.assertTrue(np.all(b["l"] <= np.minimum(b["o"], b["c"])))
        self.assertAlmostEqual(float(b["adj"][0]), 106.900002, places=6)  # adjusted series keeps 6 decimals

    def test_gmtoffset_shifts_to_taipei_date(self):
        # 16:30 UTC on Oct 5 is 00:30 Taipei on Oct 6 -> the bar belongs to Oct 6
        late = 1_790_000_000
        from datetime import datetime, timezone
        t = datetime.fromtimestamp(late, timezone.utc)
        payload = yahoo_payload([date(2026, 10, 5)], [10.0])
        import json
        j = json.loads(payload)
        j["chart"]["result"][0]["timestamp"] = [int(datetime(2026, 10, 5, 16, 30, tzinfo=timezone.utc).timestamp())]
        b = data.parse_yahoo_chart(j)
        self.assertEqual(int(b["d"][0]), (date(2026, 10, 6) - date(1970, 1, 1)).days)
        # exactly 09:00 Taipei (01:00 UTC) stays on its own day
        j["chart"]["result"][0]["timestamp"] = [ts_of(date(2026, 10, 5))]
        b = data.parse_yahoo_chart(j)
        self.assertEqual(int(b["d"][0]), (date(2026, 10, 5) - date(1970, 1, 1)).days)
        del t

    def test_null_rows_dropped(self):
        ds = bdays(date(2026, 10, 5), 8)
        close = [10 + i for i in range(8)]
        b = data.parse_yahoo_chart(yahoo_payload(ds, close, nulls=(2, 5)))
        kept = [d for i, d in enumerate(ds) if i not in (2, 5)]
        np.testing.assert_array_equal(b["d"], day_ints(kept))
        self.assertEqual(len(b["c"]), 6)
        for a in (b["o"], b["h"], b["l"], b["c"], b["adj"], b["v"]):
            self.assertTrue(np.all(np.isfinite(a)))

    def test_partial_nulls_and_non_positive_dropped(self):
        import json
        ds = bdays(date(2026, 10, 5), 6)
        j = json.loads(yahoo_payload(ds, [10, 11, 12, 13, 14, 15]))
        q = j["chart"]["result"][0]["indicators"]["quote"][0]
        q["open"][1] = None  # only the open is null -> the whole row is unusable
        q["high"][2] = 0.0  # non-positive price
        q["low"][3] = -1.0
        q["volume"][4] = None  # null volume is fine -> 0
        b = data.parse_yahoo_chart(j)
        np.testing.assert_array_equal(b["d"], day_ints([ds[0], ds[4], ds[5]]))
        self.assertEqual(b["v"][1], 0.0)

    def test_duplicates_last_wins_and_sorted(self):
        import json
        ds = bdays(date(2026, 10, 5), 4)
        j = json.loads(yahoo_payload(ds, [10, 11, 12, 13]))
        r = j["chart"]["result"][0]
        # append a duplicate of ds[1] (a later, revised bar) and swap two rows to break the ordering
        r["timestamp"].append(ts_of(ds[1], hour_utc=5))
        for k in ("open", "high", "low", "close"):
            r["indicators"]["quote"][0][k].append(21.0)
        r["indicators"]["quote"][0]["volume"].append(999)
        r["indicators"]["adjclose"][0]["adjclose"].append(21.0)
        for arr in (r["timestamp"], *[r["indicators"]["quote"][0][k] for k in ("open", "high", "low", "close", "volume")],
                    r["indicators"]["adjclose"][0]["adjclose"]):
            arr[0], arr[2] = arr[2], arr[0]
        b = data.parse_yahoo_chart(j)
        np.testing.assert_array_equal(b["d"], day_ints(ds))
        self.assertTrue(np.all(np.diff(b["d"]) > 0))
        self.assertEqual(b["c"][1], 21.0)  # the later duplicate won
        self.assertEqual(b["v"][1], 999)
        self.assertEqual(b["c"][0], 10.0)

    def test_high_low_repaired(self):
        ds = bdays(date(2026, 10, 5), 3)
        b = data.parse_yahoo_chart(yahoo_payload(ds, [10, 10, 10], open_=[10, 12, 8], high=[9, 9, 9], low=[11, 11, 11]))
        self.assertTrue(np.all(b["h"] >= np.maximum(b["o"], b["c"])))
        self.assertTrue(np.all(b["l"] <= np.minimum(b["o"], b["c"])))
        self.assertTrue(np.all(b["h"] >= b["l"]))

    def test_trailing_null_adjclose_keeps_ratio(self):
        import json
        ds = bdays(date(2026, 10, 5), 4)
        j = json.loads(yahoo_payload(ds, [10, 10, 10, 10], adj=[9, 9, 9, 9]))
        j["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][3] = None
        b = data.parse_yahoo_chart(j)
        self.assertEqual(len(b["d"]), 4)
        self.assertAlmostEqual(b["adj"][3] / b["c"][3], 0.9)
        # a null adjclose in the *middle* is not repaired (row dropped)
        j["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][1] = None
        b = data.parse_yahoo_chart(j)
        self.assertEqual(len(b["d"]), 3)

    def test_missing_adjclose_uses_close(self):
        import json
        ds = bdays(date(2026, 10, 5), 3)
        j = json.loads(yahoo_payload(ds, [10, 11, 12]))
        del j["chart"]["result"][0]["indicators"]["adjclose"]
        b = data.parse_yahoo_chart(j)
        np.testing.assert_allclose(b["adj"], b["c"])

    def test_dividends(self):
        ds = bdays(date(2026, 10, 5), 5)
        b = data.parse_yahoo_chart(yahoo_payload(ds, [10] * 5, divs=[(ds[2], 0.5), (ds[4], 1.25)]))
        self.assertEqual(b["div"], [(int(day_ints([ds[2]])[0]), 0.5), (int(day_ints([ds[4]])[0]), 1.25)])

    def test_errors(self):
        with self.assertRaises(data.NotFoundError):
            data.parse_yahoo_chart(yahoo_error("Not Found"))
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(yahoo_error("Internal Server Error"))
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(b"<html>rate limited</html>")
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(b'{"foo": 1}')
        # result present but no timestamps: a malformed answer (audit P-14) -- an empty INCREMENTAL window is the only
        # place where it is legitimate (allow_empty), and it then yields empty bars
        import json
        j = json.loads(yahoo_payload(bdays(date(2026, 10, 5), 1), [10]))
        del j["chart"]["result"][0]["timestamp"]
        j["chart"]["result"][0]["indicators"] = {"quote": [{}]}
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)
        b = data.parse_yahoo_chart(j, allow_empty=True)
        self.assertEqual(len(b["d"]), 0)

    def test_column_length_mismatch_is_malformed(self):
        import json
        j = json.loads(yahoo_payload(bdays(date(2026, 10, 5), 4), [10, 11, 12, 13]))
        j["chart"]["result"][0]["indicators"]["quote"][0]["close"].pop()
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)


def flat_row_payload(volume_last=0, close_last=2385.0, itype="EQUITY", n_flat=1):
    """Five real sessions, then ``n_flat`` rows with o=h=l=c (equal to the previous close unless ``close_last`` says
    otherwise) and ``volume_last`` shares: Yahoo's filler for a day the exchange was closed (audit P-01)."""
    ds = bdays(date(2026, 10, 14), 5 + n_flat)
    close = [2420.0, 2405.0, 2420.0, 2395.0, 2385.0] + [close_last] * n_flat
    o = [2415.0, 2400.0, 2425.0, 2400.0, 2390.0] + [close_last] * n_flat
    hi = [2425.0, 2415.0, 2435.0, 2405.0, 2395.0] + [close_last] * n_flat
    lo = [2410.0, 2395.0, 2400.0, 2385.0, 2380.0] + [close_last] * n_flat
    vol = [23_000_000, 17_000_000, 22_000_000, 21_000_000, 19_000_000] + [volume_last] * n_flat
    return ds, yahoo_payload(ds, close, open_=o, high=hi, low=lo, volume=vol, itype=itype)


class PlaceholderBars(unittest.TestCase):
    """Audit P-01: Yahoo serves flat zero-volume bars for days the exchange was closed (live 2026-10-23: 63 of 63 stocks);
    they are not trading days and must not become bars."""

    def test_closure_placeholder_is_not_a_bar(self):
        ds, payload = flat_row_payload()
        b = data.parse_yahoo_chart(payload)
        np.testing.assert_array_equal(b["d"], day_ints(ds[:5]))  # the phantom 6th bar is gone
        self.assertEqual(float(b["c"][-1]), 2385.0)
        np.testing.assert_array_equal(b["ph_days"], day_ints(ds[5:]))
        np.testing.assert_array_equal(b["real_days"], day_ints(ds[:5]))

    def test_run_of_placeholders_is_removed_as_a_whole(self):
        ds, payload = flat_row_payload(n_flat=4)  # a holiday run, as in the 2000-2005 Lunar New Year fillers
        b = data.parse_yahoo_chart(payload)
        np.testing.assert_array_equal(b["d"], day_ints(ds[:5]))
        self.assertEqual(len(b["ph_days"]), 4)

    def test_traded_flat_bar_is_kept(self):
        # volume > 0: a real session at an unchanged price (illiquid stock), not a placeholder
        ds, payload = flat_row_payload(volume_last=1000)
        b = data.parse_yahoo_chart(payload)
        self.assertEqual(len(b["d"]), 6)
        self.assertEqual(len(b["ph_days"]), 0)

    def test_flat_zero_volume_bar_at_a_new_price_is_kept(self):
        # not equal to the previous close: that is not "nothing happened", keep it
        ds, payload = flat_row_payload(close_last=2400.0)
        b = data.parse_yahoo_chart(payload)
        self.assertEqual(len(b["d"]), 6)

    def test_first_row_is_never_a_placeholder(self):
        ds = bdays(date(2026, 10, 5), 3)
        payload = yahoo_payload(ds, [10.0, 10.0, 10.0], open_=[10.0] * 3, high=[10.0] * 3, low=[10.0] * 3, volume=[0, 0, 0])
        b = data.parse_yahoo_chart(payload)
        np.testing.assert_array_equal(b["d"], day_ints(ds[:1]))  # row 0 stays (no previous close), rows 1-2 repeat it

    def test_index_is_exempt(self):
        # an index has no volume; a flat index bar is an index bar (and the tests' ^TWII fixtures have volume 0 everywhere)
        ds, payload = flat_row_payload(itype="INDEX")
        b = data.parse_yahoo_chart(payload)
        self.assertEqual(len(b["d"]), 6)
        ds2 = bdays(date(2026, 10, 14), 6)
        flat = yahoo_payload(ds2, [100.0] * 6, open_=[100.0] * 6, high=[100.0] * 6, low=[100.0] * 6, volume=[0] * 6,
                             meta={"symbol": "^TWII"}, itype="EQUITY")
        self.assertEqual(len(data.parse_yahoo_chart(flat)["d"]), 6)  # recognised by its symbol as well

    def test_null_rows_are_reported_not_counted_as_placeholders(self):
        ds = bdays(date(2026, 10, 14), 6)
        payload = yahoo_payload(ds, [10, 11, 12, 13, 14, 15], nulls=(5,))
        b = data.parse_yahoo_chart(payload)
        np.testing.assert_array_equal(b["null_days"], day_ints(ds[5:]))
        self.assertEqual(len(b["ph_days"]), 0)
        np.testing.assert_array_equal(b["d"], day_ints(ds[:5]))

    def test_legacy_record_migration_matches_the_parser(self):
        # the same placeholder rule is applied to old (v2) cache records without a re-download
        ds, payload = flat_row_payload(n_flat=2)
        full = data.parse_yahoo_chart(payload)
        import json
        j = json.loads(payload)
        # rebuild the raw columns of the whole payload (placeholders included) as a v2 record
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        rec = {"v": 2, "symbol": "2330", "yahoo_symbol": "2330.TW", "market": "TWSE", "name_hint": "x",
               "instrument_type": "EQUITY", "fetched_at": "2026-10-22T22:00:00+08:00",
               "full_fetched_at": "2026-10-22T22:00:00+08:00", "dates": [str(d) for d in ds],
               "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
               "adjclose": r["indicators"]["adjclose"][0]["adjclose"], "dividends": []}
        out = data._rec_from_json(rec)
        np.testing.assert_array_equal(out["d"], full["d"])
        np.testing.assert_array_equal(out["c"], full["c"])
        self.assertEqual(out["v"], data.CACHE_VERSION)
        self.assertEqual(out["full_fetched_at"], "")  # the next refresh re-downloads the whole history


class MalformedAnswers(unittest.TestCase):
    """Audit P-14: a transient malformed answer is DataUnavailable (503, retry), never "no such symbol" or an HTTP 500."""

    def payload(self):
        import json
        return json.loads(yahoo_payload(bdays(date(2026, 10, 5), 6), [10, 11, 12, 13, 14, 15]))

    def test_no_quote_block(self):
        j = self.payload()
        del j["chart"]["result"][0]["indicators"]["quote"]
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)
        j["chart"]["result"][0]["indicators"] = {}
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)

    def test_empty_timestamps(self):
        j = self.payload()
        j["chart"]["result"][0]["timestamp"] = []
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)

    def test_millisecond_timestamps_are_rejected_not_a_500(self):
        j = self.payload()
        j["chart"]["result"][0]["timestamp"] = [t * 1000 for t in j["chart"]["result"][0]["timestamp"]]
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)

    def test_negative_and_non_numeric_timestamps(self):
        j = self.payload()
        j["chart"]["result"][0]["timestamp"][2] = -5
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)
        j = self.payload()
        j["chart"]["result"][0]["timestamp"][2] = None
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)
        j = self.payload()
        j["chart"]["result"][0]["timestamp"][2] = "soon"
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)

    def test_absurd_gmtoffset(self):
        j = self.payload()
        j["chart"]["result"][0]["meta"]["gmtoffset"] = 10 ** 12
        with self.assertRaises(data.DataUnavailable):
            data.parse_yahoo_chart(j)

    def test_absurd_dividend_dates_are_ignored(self):
        j = self.payload()
        j["chart"]["result"][0]["events"] = {"dividends": {"1": {"amount": 1.0, "date": 10 ** 15}}}
        self.assertEqual(data.parse_yahoo_chart(j)["div"], [])

    def test_well_formed_but_all_rows_null_is_empty_not_an_error_here(self):
        # parse level: rows that are all null give empty bars; get_price_data turns that into DataUnavailable (see test_data_cache)
        ds = bdays(date(2026, 10, 5), 3)
        b = data.parse_yahoo_chart(yahoo_payload(ds, [10, 11, 12], nulls=(0, 1, 2)))
        self.assertEqual(len(b["d"]), 0)


class Symbols(unittest.TestCase):
    def test_normalize(self):
        n = data.normalize_symbol
        self.assertEqual(n("2330"), "2330")
        self.assertEqual(n(" 2330.tw "), "2330")
        self.assertEqual(n("6488.TWO"), "6488")
        self.assertEqual(n("００５０"), "0050")  # full-width digits
        self.assertEqual(n("00878"), "00878")
        self.assertEqual(n("00400a"), "00400A")
        self.assertEqual(n("00679B"), "00679B")
        for alias in ("^TWII", "^twii", "TWII", "taiex", "加權指數", "大盤"):
            self.assertEqual(n(alias), "^TWII", alias)

    def test_normalize_rejects(self):
        for bad in ("", "  ", None, "abc", "23", "2330/../x", "23 30", "1234567", "../../etc/passwd", "2330;rm", "台積電"):
            with self.assertRaises(data.BadSymbol, msg=repr(bad)):
                data.normalize_symbol(bad)
        self.assertTrue(issubclass(data.BadSymbol, ValueError))

    def test_kind(self):
        self.assertEqual(data.kind_of("2330"), "stock")
        self.assertEqual(data.kind_of("0050"), "etf")
        self.assertEqual(data.kind_of("00878"), "etf")
        self.assertEqual(data.kind_of("00400A"), "etf")
        self.assertEqual(data.kind_of("^TWII"), "index")


class NameTables(unittest.TestCase):
    def test_parse_filters_and_merges(self):
        twse = [{"Code": "2330", "Name": "台積電", "TradeValue": "100", "ClosingPrice": "1000.00"},
                {"Code": "0050", "Name": "元大台灣50", "TradeValue": "50", "ClosingPrice": "100"},
                {"Code": "00400A", "Name": "主動國泰動能高息", "TradeValue": "5", "ClosingPrice": "14.8"},
                {"Code": "910322", "Name": "康師傅-DR", "TradeValue": "5", "ClosingPrice": "5"},  # TDR -> skipped
                {"Code": "03001P", "Name": "權證", "TradeValue": "5", "ClosingPrice": "5"},
                {"Code": "2317", "Name": "", "TradeValue": "5", "ClosingPrice": "5"},  # nameless -> skipped
                {"Code": "1101", "Name": "台泥", "TradeValue": "", "ClosingPrice": ""}]
        company = [{"公司代號": "2330", "公司簡稱": "不應覆蓋"}, {"公司代號": "9999", "公司簡稱": "停牌股"},
                   {"公司代號": "1101", "公司簡稱": "不應覆蓋"}]
        tpex = [{"SecuritiesCompanyCode": "6488", "CompanyName": "環球晶", "Close": "400", "TransactionAmount": "70"},
                {"SecuritiesCompanyCode": "030001", "CompanyName": "權證", "Close": "1", "TransactionAmount": "1"},
                {"SecuritiesCompanyCode": "00679B", "CompanyName": "元大美債20年", "Close": "30", "TransactionAmount": "9"},
                {"SecuritiesCompanyCode": "2330", "CompanyName": "冒名", "Close": "1", "TransactionAmount": "1"}]
        it = data.parse_name_tables(twse, company, tpex)
        self.assertEqual(sorted(it), ["00400A", "0050", "00679B", "1101", "2330", "6488", "9999"])
        self.assertEqual(it["2330"]["name"], "台積電")
        self.assertEqual(it["2330"]["market"], "TWSE")
        self.assertEqual(it["2330"]["tv"], 100.0)
        self.assertEqual(it["0050"]["kind"], "etf")
        self.assertEqual(it["00400A"]["kind"], "etf")
        self.assertEqual(it["6488"]["market"], "TPEx")
        self.assertEqual(it["6488"]["kind"], "stock")
        self.assertEqual(it["00679B"]["market"], "TPEx")
        self.assertEqual(it["00679B"]["kind"], "etf")
        self.assertEqual(it["9999"]["name"], "停牌股")
        self.assertIsNone(it["9999"]["tv"])
        self.assertIsNone(it["1101"]["tv"])
        self.assertEqual(it["1101"]["name"], "台泥")

    def test_parse_handles_none_sources(self):
        self.assertEqual(data.parse_name_tables(None, None, None), {})

    def test_fallback_list_integrity(self):
        codes = [c for c, _ in data.FALLBACK_UNIVERSE]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertGreaterEqual(len(codes), 60)
        for c, nm in data.FALLBACK_UNIVERSE:
            self.assertRegex(c, r"^\d{4}$")
            self.assertTrue(nm)
        self.assertEqual(codes[:12], ["2330", "2317", "2454", "2308", "2382", "2412", "2881", "2882", "2891", "2886",
                                      "1301", "1303"])
        for c, _ in data.FALLBACK_ETFS:
            self.assertTrue(c.startswith("00"))


if __name__ == "__main__":
    unittest.main()
