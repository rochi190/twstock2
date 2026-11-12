"""P4a news archiver (backend/altdata/news.py): parsing, de-duplication, code mapping, ROC dates, failure isolation.

Hermetic: every response comes from tests/fixtures/alt/news_* (real responses of 2026-11-12, trimmed to a few items
with descriptions / bodies removed).  No network, no sleeping.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backend import data  # noqa: E402
from backend.altdata import news  # noqa: E402

FIX = Path(HERE) / "fixtures" / "alt"

_GUARDS = []


def _no_network(*a, **k):
    raise AssertionError("test_alt_news must be hermetic: a real network call was attempted")


def setUpModule():
    """Belt and braces: any accidental real HTTP request fails the test instead of reaching the network."""
    import urllib.request
    for target, name in ((data, "_http_get"), (urllib.request, "urlopen")):
        p = mock.patch.object(target, name, _no_network)
        p.start()
        _GUARDS.append(p)


def tearDownModule():
    while _GUARDS:
        _GUARDS.pop().stop()


def fx_bytes(name):
    return (FIX / name).read_bytes()


def fx_json(name):
    return json.loads(fx_bytes(name).decode("utf-8"))


def fixture_names():
    return news.NameMatcher.from_items(fx_json("news_names.json")["items"])


def fake_get(fail=(), seen=None):
    """URL -> fixture bytes; hosts / substrings listed in ``fail`` raise DataUnavailable."""
    def get(url, **kw):
        if seen is not None:
            seen.append(url)
        if any(f in url for f in fail):
            raise data.DataUnavailable(f"boom {url}")
        if "t187ap04_L" in url:
            return fx_bytes("news_twse.json")
        if "mopsfin_t187ap04_O" in url:
            return fx_bytes("news_tpex.json")
        if "api.cnyes.com" in url:
            return fx_bytes("news_cnyes.json")
        if "feedburner" in url:
            return fx_bytes("news_cna.xml")
        if "yahoo.com/rss" in url:
            return fx_bytes("news_yahoo.xml")
        if "news.google.com" in url:
            return fx_bytes("news_google.xml")
        raise AssertionError(f"unexpected URL {url}")
    return get


def fetcher(**kw):
    return news.Fetcher(fake_get(**kw), min_interval=0.0, sleep=lambda s: None)


def row(source="cna", link="https://x/1", title="t", codes=(), fetched_at="2026-11-12T10:00:00+08:00"):
    return {"source": source, "title": title, "link": link, "published_at": None, "codes": list(codes),
            "fetched_at": fetched_at}


class DateTests(unittest.TestCase):
    def test_roc_date(self):
        self.assertEqual(news.roc_date("1151112"), date(2026, 11, 12))
        self.assertEqual(news.roc_date(1151112), date(2026, 11, 12))
        self.assertEqual(news.roc_date("991231"), date(2010, 12, 31))  # 6 digits: year 99
        self.assertEqual(news.roc_date("20261112"), date(2026, 11, 12))
        for bad in ("", None, "1151332", "abc", "11511"):
            self.assertIsNone(news.roc_date(bad), bad)

    def test_roc_datetime_left_pads_time(self):
        self.assertEqual(news.roc_datetime("1151112", "10405"), "2026-11-12T01:04:05+08:00")
        self.assertEqual(news.roc_datetime("1151111", "70003"), "2026-11-11T07:00:03+08:00")
        self.assertEqual(news.roc_datetime("1151112", "133000"), "2026-11-12T13:30:00+08:00")
        self.assertEqual(news.roc_datetime("1151112", "5"), "2026-11-12T00:00:05+08:00")
        for bad in ("246000", "1234567", "", None, "126100"):
            self.assertIsNone(news.roc_datetime("1151112", bad), bad)
        self.assertIsNone(news.roc_datetime("bad", "10405"))

    def test_rfc822_and_epoch_become_taipei(self):
        self.assertEqual(news.rfc822_to_iso("Thu, 12 Nov 2026 15:14:19 GMT"), "2026-11-12T23:14:19+08:00")
        self.assertEqual(news.rfc822_to_iso("Thu, 12 Nov 2026 22:40:08 +0800"), "2026-11-12T22:40:08+08:00")
        self.assertIsNone(news.rfc822_to_iso("not a date"))
        self.assertIsNone(news.rfc822_to_iso(None))
        self.assertEqual(news.epoch_to_iso(1794495870), "2026-11-12T23:04:30+08:00")  # 15:04:30 UTC
        self.assertIsNone(news.epoch_to_iso("x"))


class MopsParseTests(unittest.TestCase):
    def test_fixture_really_has_trailing_space_key(self):
        self.assertIn("主旨 ", fx_json("news_twse.json")[0])  # guards the fixture, not the parser
        self.assertIn("主旨", fx_json("news_tpex.json")[0])

    def test_twse_rows(self):
        rows = news.parse_twse(fx_json("news_twse.json"))
        self.assertEqual(len(rows), 4)
        r = rows[0]
        self.assertEqual(r["source"], "twse")
        self.assertEqual(r["codes"], ["6869"])
        self.assertEqual(r["title"], "公告本公司董事會通過115年第三季合併財務報告")
        self.assertEqual(r["published_at"], "2026-11-11T01:04:05+08:00")  # 發言日期 1151111, 發言時間 10405
        self.assertTrue(r["link"].startswith(news.MOPS_PAGE + "#6869-"))
        self.assertEqual(set(r), {"source", "title", "link", "published_at", "codes"})
        self.assertEqual(len({x["link"] for x in rows}), 4)

    def test_trailing_space_subject_key_and_plain_key_both_work(self):
        base = {"發言日期": "1151112", "發言時間": "10405", "公司代號": "2330", "符合條款": "第1款", "事實發生日": "1151112"}
        a = news.parse_twse([{**base, "主旨 ": "公告A"}])
        b = news.parse_twse([{**base, "主旨": "公告A"}])
        self.assertEqual(a[0]["title"], "公告A")
        self.assertEqual(a, b)

    def test_tpex_rows_collapse_crlf(self):
        rows = news.parse_tpex(fx_json("news_tpex.json"))
        self.assertEqual(len(rows), 4)
        r = rows[0]
        self.assertEqual(r["source"], "tpex")
        self.assertEqual(r["codes"], ["3489"])  # SecuritiesCompanyCode
        self.assertNotIn("\n", r["title"])
        self.assertNotIn("\r", r["title"])
        self.assertNotIn("  ", r["title"])
        self.assertEqual(r["published_at"], "2026-11-11T07:00:03+08:00")

    def test_rows_without_code_or_subject_are_skipped_and_bad_payload_raises(self):
        self.assertEqual(news.parse_twse([{"公司代號": "", "主旨 ": "x"}, {"公司代號": "2330", "主旨 ": " "}, "junk"]), [])
        with self.assertRaises(ValueError):
            news.parse_twse({"not": "a list"})


class RssJsonParseTests(unittest.TestCase):
    def test_yahoo_uses_queried_code_and_no_body(self):
        rows = news.parse_rss(fx_bytes("news_yahoo.xml"), "yahoo", None, ["2330"])
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertEqual(r["codes"], ["2330"])
            self.assertEqual(set(r), {"source", "title", "link", "published_at", "codes"})
        self.assertEqual(rows[0]["published_at"], "2026-11-12T23:14:19+08:00")  # pubDate is GMT
        self.assertNotIn("　", rows[0]["title"])  # full-width space collapsed

    def test_fixtures_hold_no_article_bodies(self):
        for name in ("news_yahoo.xml", "news_cna.xml", "news_google.xml"):
            self.assertNotIn(b"<description><![CDATA[", fx_bytes(name))
            self.assertLess(len(fx_bytes(name)), 6000)

    def test_cna_and_google(self):
        cna = news.parse_rss(fx_bytes("news_cna.xml"), "cna")
        self.assertEqual(len(cna), 4)
        self.assertEqual(cna[0]["published_at"], "2026-11-12T22:40:08+08:00")
        self.assertTrue(cna[0]["link"].startswith("https://www.cna.com.tw/news/"))
        g = news.parse_rss(fx_bytes("news_google.xml"), "google")
        self.assertEqual(len(g), 4)
        self.assertTrue(all(r["link"].startswith("https://news.google.com/rss/articles/") for r in g))

    def test_item_without_link_or_title_skipped(self):
        xml = b"<rss><channel><item><title>a</title></item><item><link>http://x</link></item>" \
              b"<item><title>ok</title><link>http://y</link></item></channel></rss>"
        rows = news.parse_rss(xml, "cna")
        self.assertEqual([r["link"] for r in rows], ["http://y"])
        self.assertIsNone(rows[0]["published_at"])

    def test_malformed_xml_raises(self):
        with self.assertRaises(Exception):
            news.parse_rss(b"<rss><channel><item>", "cna")

    def test_cnyes(self):
        m = fixture_names()
        rows = news.parse_cnyes(fx_json("news_cnyes.json"), m)
        self.assertEqual(len(rows), 5)
        r = rows[0]
        self.assertEqual(r["source"], "cnyes")
        self.assertEqual(r["link"], "https://news.cnyes.com/news/id/6660936")
        self.assertEqual(r["published_at"], "2026-11-12T23:04:30+08:00")
        self.assertEqual(r["codes"], [])
        self.assertEqual(rows[2]["codes"], ["3227"])  # name 原相 in the title AND stock list: once

    def test_cnyes_stock_list_adds_codes_but_ignores_junk(self):
        obj = {"items": {"data": [{"newsId": 1, "title": "無關標題", "publishAt": 1794495870,
                                   "stock": ["2330", "abc", " 00878 "]}]}}
        self.assertEqual(news.parse_cnyes(obj, news.NameMatcher({}))[0]["codes"], ["00878", "2330"])


class CodeMappingTests(unittest.TestCase):
    def test_names_of_two_chars_or_fewer_are_not_matched(self):
        m = news.NameMatcher({"2317": "鴻海", "1301": "台塑", "2330": "台積電", "3008": "大立光", "9999": "王"})
        self.assertEqual(m.codes_for("鴻海與台塑今天公告"), [])
        self.assertEqual(m.codes_for("台積電法說會 鴻海"), ["2330"])
        self.assertEqual(m.codes_for("大立光與台積電"), ["2330", "3008"])
        self.assertEqual(m.codes_for(""), [])

    def test_fixture_names_table_skips_short_names(self):
        raw = fx_json("news_names.json")["items"]
        short = [it["name"] for it in raw.values() if len(it["name"]) <= 2]
        self.assertTrue(short, "fixture should contain 2-character names")
        m = fixture_names()
        for name in short:
            self.assertEqual(m.codes_for(f"{name}今天大漲"), [], name)
        self.assertEqual(m.codes_for("台積電：發現部分員工資訊"), ["2330"])

    def test_fold_taiwan_variant_char(self):
        m = news.NameMatcher({"2891": "臺灣銀行股"})
        self.assertEqual(m.codes_for("台灣銀行股公告"), ["2891"])

    def test_etfs_excluded_from_matcher(self):
        items = {"0050": {"code": "0050", "name": "元大台灣50", "kind": "etf"},
                 "2330": {"code": "2330", "name": "台積電", "kind": "stock"}}
        m = news.NameMatcher.from_items(items)
        self.assertEqual(m.codes_for("元大台灣50 台積電"), ["2330"])

    def test_rss_with_matcher_unions_fixed_codes(self):
        rows = news.parse_rss(fx_bytes("news_yahoo.xml"), "yahoo", fixture_names(), ["2317"])
        tsmc = [r for r in rows if r["title"].startswith("台積電")][0]
        self.assertEqual(tsmc["codes"], ["2317", "2330"])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = news.NewsStore(Path(self._td.name) / "news")

    def test_record_shape_and_id(self):
        n = self.store.add([row(link="https://x/1", title="標題", codes=["2330", "2330", "1101"])])
        self.assertEqual(n, 1)
        lines = (self.store.root / "2026-11-12.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(list(rec), ["id", "source", "title", "link", "published_at", "fetched_at", "codes"])
        self.assertEqual(rec["id"], hashlib.sha1("cnahttps://x/1".encode()).hexdigest())
        self.assertEqual(rec["codes"], ["1101", "2330"])
        self.assertEqual(rec["fetched_at"], "2026-11-12T10:00:00+08:00")

    def test_file_date_is_taipei_date_of_fetched_at(self):
        self.store.add([row(link="a", fetched_at="2026-11-12T23:59:00+08:00"),
                        row(link="b", fetched_at="2026-11-12T16:30:00+00:00")])  # = 11-13 00:30 Taipei
        self.assertEqual(sorted(p.name for p in self.store.root.glob("*.jsonl")),
                         ["2026-11-12.jsonl", "2026-11-13.jsonl"])

    def test_dedup_same_batch_merges_codes(self):
        n = self.store.add([row(link="u", codes=["2330"]), row(link="u", codes=["2317"]), row(link="v")])
        self.assertEqual(n, 2)
        recs = self.store.read_day(date(2026, 11, 12))
        self.assertEqual([r["codes"] for r in recs if r["link"] == "u"], [["2317", "2330"]])

    def test_dedup_across_runs_and_days(self):
        self.assertEqual(self.store.add([row(link="u")]), 1)
        self.assertEqual(self.store.add([row(link="u")]), 0)  # same day
        self.assertEqual(self.store.add([row(link="u", fetched_at="2026-11-13T09:00:00+08:00")]), 0)  # next day
        self.assertEqual(self.store.add([row(link="u", fetched_at="2026-11-14T09:00:00+08:00")]), 0)  # 2 days later
        # feeds keep returning old items for days: the id stays known no matter how old the first sighting is
        self.assertEqual(self.store.add([row(link="u", fetched_at="2026-11-15T09:00:00+08:00")]), 0)
        self.assertEqual(self.store.add([row(link="u", fetched_at="2026-12-30T09:00:00+08:00")]), 0)
        self.assertEqual(self.store.add([row(link="u", source="google")]), 1)  # source is part of the id
        self.assertEqual(self.store.add([row(link="w", fetched_at="2026-12-30T09:00:00+08:00")]), 1)
        # a fresh store object (a new process) sees the same history
        self.assertEqual(news.NewsStore(self.store.root).add([row(link="u"), row(link="w")]), 0)

    def test_torn_last_line_is_survived(self):
        self.store.root.mkdir(parents=True)
        p = self.store.root / "2026-11-12.jsonl"
        good = json.dumps({"id": "old", "source": "cna", "title": "t", "link": "l", "published_at": None,
                           "fetched_at": "2026-11-12T01:00:00+08:00", "codes": []})
        p.write_bytes((good + "\n" + '{"id": "torn", "sou').encode("utf-8"))
        self.assertEqual(self.store.add([row(link="new")]), 1)
        recs = self.store.read_day(date(2026, 11, 12))
        self.assertEqual([r["id"] for r in recs][0], "old")
        self.assertEqual(len(recs), 2)  # the torn fragment is skipped, the new record is on its own line
        self.assertEqual(self.store.add([row(link="new")]), 0)

    def test_empty_batch_creates_nothing(self):
        self.assertEqual(self.store.add([]), 0)
        self.assertFalse(self.store.root.exists())

    def test_lock_file_removed_after_write(self):
        self.store.add([row(link="u")])
        self.assertFalse((self.store.root / ".lock").exists())


class FetcherTests(unittest.TestCase):
    def test_same_host_spaced_other_host_not(self):
        t = [100.0]
        sleeps = []

        def sleep(s):
            sleeps.append(s)
            t[0] += s

        f = news.Fetcher(lambda url: b"x", min_interval=3.0, sleep=sleep, clock=lambda: t[0])
        f.get("https://a.example/1")
        f.get("https://b.example/1")  # other host: no wait
        self.assertEqual(sleeps, [])
        t[0] += 1.0
        f.get("https://a.example/2")  # only 1 s since the last request to a.example
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 2.0)
        t[0] += 10.0
        f.get("https://a.example/3")  # enough time passed: no sleep
        self.assertEqual(len(sleeps), 1)

    def test_failure_still_counts_for_spacing(self):
        t = [0.0]
        sleeps = []

        def boom(url):
            raise data.DataUnavailable("x")

        f = news.Fetcher(boom, min_interval=3.0, sleep=lambda s: sleeps.append(s), clock=lambda: t[0])
        with self.assertRaises(data.DataUnavailable):
            f.get("https://a.example/1")
        with self.assertRaises(data.DataUnavailable):
            f.get("https://a.example/2")
        self.assertEqual(sleeps, [3.0])

    def test_default_minimum_interval_is_three_seconds(self):
        self.assertGreaterEqual(news.Fetcher().min_interval, 3.0)
        self.assertGreaterEqual(news.MIN_INTERVAL, 3.0)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.store = news.NewsStore(Path(self._td.name) / "news")
        self.now = lambda: news.datetime(2026, 11, 12, 23, 0, 0, tzinfo=news.TAIPEI)
        self.kw = dict(store=self.store, now=self.now, matcher=fixture_names(),
                       codes=["2330", "2317"], queries=["輝達", "台積電"])

    def run_once(self, sources=None, **kw):
        opts = {**self.kw, **kw}
        opts.setdefault("fetcher", fetcher())
        return news.run_once(sources, **opts)

    def test_all_sources_then_second_run_adds_nothing(self):
        r1 = self.run_once()
        self.assertEqual(r1["errors"], {})
        self.assertEqual(r1["added"]["twse"], 4)
        self.assertEqual(r1["added"]["tpex"], 4)
        self.assertEqual(r1["added"]["cnyes"], 5)
        self.assertEqual(r1["added"]["cna"], 4)
        self.assertEqual(r1["added"]["yahoo"], 4)  # same 4 links for both codes -> merged
        self.assertEqual(r1["added"]["google"], 4)
        r2 = self.run_once()
        self.assertEqual(sum(r2["added"].values()), 0)
        self.assertEqual(r2["fetched"]["twse"], 4)

    def test_yahoo_codes_merged_when_same_link_under_two_queries(self):
        self.run_once(["yahoo"])
        recs = [r for r in self.store.read_day(date(2026, 11, 12)) if r["source"] == "yahoo"]
        self.assertEqual(len(recs), 4)
        tsmc = [r for r in recs if r["title"].startswith("台積電")][0]
        self.assertEqual(tsmc["codes"], ["2317", "2330"])

    def test_stored_rows_have_only_headline_fields(self):
        self.run_once()
        recs = self.store.read_day(date(2026, 11, 12))
        self.assertEqual(len(recs), 4 + 4 + 5 + 4 + 4 + 4)
        for r in recs:
            self.assertEqual(set(r), {"id", "source", "title", "link", "published_at", "fetched_at", "codes"})
            self.assertEqual(r["fetched_at"], "2026-11-12T23:00:00+08:00")  # the only trusted time
        self.assertEqual(len({r["id"] for r in recs}), len(recs))

    def test_one_failing_source_does_not_stop_the_others(self):
        res = self.run_once(fetcher=fetcher(fail=("t187ap04_L", "api.cnyes.com")))
        self.assertEqual(set(res["errors"]), {"twse", "cnyes"})
        self.assertEqual(res["added"]["twse"], 0)
        self.assertEqual(res["added"]["tpex"], 4)
        self.assertEqual(res["added"]["cna"], 4)
        self.assertEqual(res["added"]["google"], 4)
        self.assertTrue(res["errors"]["twse"].startswith("DataUnavailable"))

    def test_malformed_payload_is_a_source_failure_not_a_crash(self):
        def get(url, **kw):
            return b"<html>not what we expected" if "feedburner" in url else fake_get()(url)

        res = self.run_once(fetcher=news.Fetcher(get, min_interval=0.0, sleep=lambda s: None))
        self.assertEqual(set(res["errors"]), {"cna"})
        self.assertEqual(res["added"]["tpex"], 4)

    def test_per_request_failures_in_yahoo_and_google(self):
        res = self.run_once(["yahoo", "google"], fetcher=fetcher(fail=("s=2317",)))
        self.assertEqual(res["errors"], {})  # 1 of 2 yahoo requests failed: tolerated
        self.assertEqual(res["added"]["yahoo"], 4)
        res = self.run_once(["yahoo", "google"], fetcher=fetcher(fail=("yahoo.com/rss", "news.google.com")))
        self.assertEqual(set(res["errors"]), {"yahoo", "google"})

    def test_google_url_is_encoded_with_when_1d(self):
        seen = []
        self.run_once(["google"], fetcher=fetcher(seen=seen), queries=["美元 台幣"])
        self.assertEqual(len(seen), 1)
        self.assertIn("q=%E7%BE%8E%E5%85%83%20%E5%8F%B0%E5%B9%A3+when:1d", seen[0])
        self.assertIn("hl=zh-TW&gl=TW&ceid=TW:zh-Hant", seen[0])

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            news.run_once(["nope"], **self.kw)


class QueriesAndUniverseTests(unittest.TestCase):
    def test_queries_file_has_at_least_ten_keywords(self):
        qs = news.load_queries()
        self.assertGreaterEqual(len(qs), 10)
        for kw in ("輝達", "台積電", "聯準會", "關稅", "費城半導體"):
            self.assertIn(kw, qs)
        self.assertEqual(len(qs), len(set(qs)))

    def test_load_queries_accepts_plain_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "q.json"
            p.write_text(json.dumps(["a", " b ", "", 3]), encoding="utf-8")
            self.assertEqual(news.load_queries(p), ["a", "b"])

    def test_universe_codes_prefers_pool300(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "research").mkdir()
            pool = [{"code": f"{1000 + i}", "name": "n", "market": "TWSE", "kind": "stock"} for i in range(100)]
            (Path(td) / "research" / "pool300.json").write_text(json.dumps(pool), encoding="utf-8")
            with mock.patch.object(data, "cache_dir", return_value=Path(td)):
                codes = news.universe_codes(60)
        self.assertEqual(len(codes), 60)
        self.assertEqual(codes[0], "1000")
        self.assertEqual(codes[-1], "1059")

    def test_universe_codes_falls_back_to_get_universe(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(data, "cache_dir", return_value=Path(td)), \
                    mock.patch.object(data, "get_universe", return_value=[{"code": "2330"}, {"code": "2317"}]) as gu:
                self.assertEqual(news.universe_codes(60), ["2330", "2317"])
            gu.assert_called_once_with(60)


class CliTests(unittest.TestCase):
    def test_no_network_once_prints_zero_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"TWQ_CACHE_DIR": td}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(data, "_CACHE_DIR_OVERRIDE", None):
                rc = news.main(["--once", "--no-network", "--sources", "twse,tpex"])
            out = buf.getvalue()
            self.assertEqual(rc, 1)  # every selected source failed
            self.assertIn("twse: +0 new", out)
            self.assertIn("tpex: +0 new", out)
            self.assertIn("FAILED", out)
            self.assertFalse((Path(td) / "alt" / "news").exists())

    def test_once_is_required_and_bad_source_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                news.main([])
            with self.assertRaises(SystemExit):
                news.main(["--once", "--sources", "bogus", "--no-network"])


if __name__ == "__main__":
    unittest.main()
