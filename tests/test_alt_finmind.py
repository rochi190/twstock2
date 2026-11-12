"""Tests for backend/altdata/finmind.py (PLAN.md P1a).  Hermetic: no network, no real sleeping.

Fixtures ``tests/fixtures/alt/finmind_*.json`` are small slices of *real* FinMind responses sampled 2026-11-12
(2330 = TWSE, 6488 = TPEx); the envelope ``{msg, status, data}`` is kept as served."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backend import data  # noqa: E402
from backend.altdata import finmind as F  # noqa: E402

FIX = HERE / "fixtures" / "alt"
INST = "TaiwanStockInstitutionalInvestorsBuySell"
REV = "TaiwanStockMonthRevenue"


def fixture(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def body(rows, msg="success", status=200) -> bytes:
    return json.dumps({"msg": msg, "status": status, "data": rows}).encode("utf-8")


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b""))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="twq_fm_")
        data.set_cache_dir(self.tmp)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(data.set_cache_dir, None)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("FINMIND_TOKEN", None)
        os.environ.pop("TWQ_OFFLINE", None)  # the real _http_get path test patches urlopen itself
        self.sleeps = []
        p = mock.patch.object(F, "_sleep", lambda s: self.sleeps.append(s))
        p.start()
        self.addCleanup(p.stop)


class UrlTests(Base):
    def test_url_without_token(self):
        url = F.build_url("inst", "2330", "2012-05-01")
        self.assertEqual(url, "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell"
                              "&data_id=2330&start_date=2012-05-01")
        self.assertNotIn("token", url)

    def test_url_with_token_from_env_only(self):
        os.environ["FINMIND_TOKEN"] = "abc.def"
        url = F.build_url(REV, "6488", "2010-01-01")
        self.assertTrue(url.endswith("&token=abc.def"), url)
        self.assertIn("dataset=TaiwanStockMonthRevenue", url)

    def test_aliases_and_defaults(self):
        self.assertEqual(F.resolve_dataset("rev"), REV)
        self.assertEqual(F.resolve_dataset(REV), REV)
        self.assertEqual({a: F.default_start(a) for a in F.ALIASES},
                         {"inst": "2012-05-01", "margin": "2012-01-01", "hold": "2012-01-01",
                          "per": "2012-01-01", "rev": "2010-01-01"})
        with self.assertRaises(ValueError):
            F.resolve_dataset("nope")

    def test_bad_code_rejected(self):
        with self.assertRaises(ValueError):
            F.dataset_path("inst", "../x")


class FetchTests(Base):
    def test_success_returns_rows_and_uses_http_get(self):
        rows = fixture("finmind_inst_2330.json")["data"]
        with mock.patch.object(data, "_http_get", return_value=body(rows)) as g:
            out = F.fetch_dataset("inst", "2330", "2012-05-01")
        self.assertEqual(out, rows)
        self.assertIn("data_id=2330", g.call_args[0][0])
        self.assertEqual(g.call_args[1].get("retries"), 1)  # never hammer a rate limit

    def test_msg_not_success_is_rate_limited(self):
        with mock.patch.object(data, "_http_get", return_value=body([], msg="Requests reach the upper limit", status=402)):
            with self.assertRaises(F.RateLimited):
                F.fetch_dataset("inst", "2330", "2012-05-01")
        with mock.patch.object(data, "_http_get", return_value=body([], msg="other", status=200)):
            with self.assertRaises(F.RateLimited):
                F.fetch_dataset("inst", "2330", "2012-05-01")

    def test_body_status_402_with_success_msg_is_rate_limited(self):
        with mock.patch.object(data, "_http_get", return_value=body([], msg="success", status=402)):
            with self.assertRaises(F.RateLimited):
                F.fetch_dataset("per", "2330", "2012-01-01")

    def test_http_402_429_chained_from_http_get(self):
        for code in (402, 429):
            exc = data.DataUnavailable(f"HTTP {code} for https://x")
            exc.__cause__ = http_error(code)
            with mock.patch.object(data, "_http_get", side_effect=exc) as g:
                with self.assertRaises(F.RateLimited):
                    F.fetch_dataset("inst", "2330", "2012-05-01")
            self.assertEqual(g.call_count, 1)  # no retry on a rate limit

    def test_real_http_get_path_429_and_402(self):
        """Through the real ``_http_get`` (urlopen patched): 429 is retried by it only if retries>1 -- we pass 1."""
        for code in (402, 429):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code)) as uo, \
                    mock.patch.object(data, "_sleep"):
                with self.assertRaises(F.RateLimited):
                    F.fetch_dataset("inst", "2330", "2012-05-01")
            self.assertEqual(uo.call_count, 1)

    def test_transient_failure_retried_then_ok(self):
        calls = [data.DataUnavailable("boom"), b"not json", body([{"date": "2020-01-02"}])]
        with mock.patch.object(data, "_http_get", side_effect=calls):
            out = F.fetch_dataset("per", "2330", "2012-01-01")
        self.assertEqual(out, [{"date": "2020-01-02"}])
        self.assertEqual(len(self.sleeps), 2)

    def test_persistent_failure_raises_and_hides_token(self):
        os.environ["FINMIND_TOKEN"] = "SECRET-TOKEN"
        with mock.patch.object(data, "_http_get", side_effect=data.DataUnavailable("x SECRET-TOKEN y")):
            with self.assertRaises(data.DataUnavailable) as cm:
                F.fetch_dataset("per", "2330", "2012-01-01")
        self.assertNotIn("SECRET-TOKEN", str(cm.exception))

    def test_null_data_is_empty_list(self):
        payload = json.dumps({"msg": "success", "status": 200, "data": None}).encode()
        with mock.patch.object(data, "_http_get", return_value=payload):
            self.assertEqual(F.fetch_dataset("per", "9999", "2012-01-01"), [])


class BackfillTests(Base):
    def _fake(self, rows_by_code=None, calls=None):
        rows_by_code = rows_by_code or {}

        def fake(url, **kw):
            if calls is not None:
                calls.append(url)
            code = url.split("data_id=")[1].split("&")[0]
            return body(rows_by_code.get(code, [{"date": "2020-01-02", "stock_id": code}]))
        return fake

    def test_writes_files_with_metadata(self):
        with mock.patch.object(data, "_http_get", self._fake()):
            res = F.backfill(["inst", "rev"], ["2330", "6488"])
        self.assertEqual((res["fetched"], res["skipped"], res["requests"], res["rate_limited"]), (4, 0, 4, False))
        p = Path(self.tmp) / "alt" / "finmind" / INST / "2330.json"
        self.assertTrue(p.exists())
        obj = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual((obj["dataset"], obj["stock_id"], obj["start_date"], obj["empty"]),
                         (INST, "2330", "2012-05-01", False))
        self.assertTrue(obj["fetched_at"])
        self.assertEqual(obj["data"], [{"date": "2020-01-02", "stock_id": "2330"}])
        self.assertEqual(json.loads((Path(self.tmp) / "alt" / "finmind" / REV / "6488.json").read_text("utf-8"))
                         ["start_date"], "2010-01-01")
        self.assertEqual(F.load_dataset("inst", "2330")["stock_id"], "2330")  # alias works on the reader
        self.assertIsNone(F.load_dataset("inst", "1101"))

    def test_resume_skips_existing_without_requests_or_sleep(self):
        calls = []
        with mock.patch.object(data, "_http_get", self._fake(calls=calls)):
            F.backfill(["inst"], ["2330"])
            self.assertEqual(len(calls), 1)
            self.sleeps.clear()
            res = F.backfill(["inst"], ["2330", "6488"])
        self.assertEqual(len(calls), 2)  # only 6488 fetched on the second run
        self.assertEqual((res["skipped"], res["fetched"]), (1, 1))
        self.assertEqual(self.sleeps, [])  # first (and only) request of the run is not preceded by a sleep

    def test_corrupt_file_is_refetched(self):
        p = Path(self.tmp) / "alt" / "finmind" / INST / "2330.json"
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
        with mock.patch.object(data, "_http_get", self._fake()) as g:
            res = F.backfill(["inst"], ["2330"])
        self.assertEqual((res["fetched"], res["skipped"]), (1, 0))
        self.assertIsNotNone(F.load_dataset("inst", "2330"))

    def test_empty_data_saved_and_flagged(self):
        with mock.patch.object(data, "_http_get", self._fake({"9999": []})):
            res = F.backfill(["per"], ["9999"])
        self.assertEqual(res["empty"], 1)
        obj = F.load_dataset("per", "9999")
        self.assertIs(obj["empty"], True)
        self.assertEqual(obj["data"], [])
        with mock.patch.object(data, "_http_get", side_effect=AssertionError("should not refetch")):
            res2 = F.backfill(["per"], ["9999"])
        self.assertEqual(res2["skipped"], 1)

    def test_interval_without_and_with_token(self):
        with mock.patch.object(data, "_http_get", self._fake()):
            F.backfill(["inst"], ["2330", "6488", "2317"])
        self.assertEqual(self.sleeps, [12.0, 12.0])
        self.assertGreaterEqual(F.min_interval(""), 12)
        self.sleeps.clear()
        os.environ["FINMIND_TOKEN"] = "tok"
        with mock.patch.object(data, "_http_get", self._fake()):
            F.backfill(["margin"], ["2330", "6488", "2317"])
        self.assertEqual(self.sleeps, [1.0, 1.0])

    def test_rate_limit_stops_immediately_and_reports(self):
        seq = [body([{"date": "d"}]), body([], msg="limit", status=402)]
        with mock.patch.object(data, "_http_get", side_effect=seq) as g:
            res = F.backfill(["inst"], ["2330", "6488", "2317", "2454"])
        self.assertEqual(g.call_count, 2)
        self.assertTrue(res["rate_limited"])
        self.assertEqual((res["fetched"], res["requests"]), (1, 2))
        self.assertEqual(res["remaining"], [(INST, "6488"), (INST, "2317"), (INST, "2454")])
        self.assertIsNotNone(F.load_dataset("inst", "2330"))
        self.assertIsNone(F.load_dataset("inst", "6488"))  # nothing half-written for the refused one

    def test_consecutive_failures_stop(self):
        codes = [str(2000 + i) for i in range(10)]
        with mock.patch.object(data, "_http_get", side_effect=data.DataUnavailable("down")) as g:
            res = F.backfill(["inst"], codes)
        self.assertEqual(len(res["failed"]), F.MAX_CONSECUTIVE_FAILURES)
        self.assertTrue(res["message"])
        self.assertEqual(len(res["remaining"]), 10 - F.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(g.call_count, F.MAX_CONSECUTIVE_FAILURES * F.FETCH_ATTEMPTS)

    def test_atomic_write_leaves_no_partial_file(self):
        with mock.patch.object(data, "_http_get", self._fake()):
            with mock.patch("os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    F.backfill(["inst"], ["2330"])
        d = Path(self.tmp) / "alt" / "finmind" / INST
        self.assertEqual(list(d.glob("*")), [])  # neither the target nor a stray temp file
        # a good run replaces atomically and leaves only the final file
        with mock.patch.object(data, "_http_get", self._fake()):
            F.backfill(["inst"], ["2330"])
        self.assertEqual([p.name for p in d.glob("*")], ["2330.json"])


class ParserTests(unittest.TestCase):
    def test_institutional_five_names_in_shares(self):
        out = F.parse_institutional(fixture("finmind_inst_2330.json")["data"])
        r = out["2026-11-12"]
        self.assertEqual(set(r), {"foreign_net", "foreign_dealer_self_net", "trust_net", "dealer_self_net",
                                  "dealer_hedge_net", "dealer_net"})
        raw = {x["name"]: x for x in fixture("finmind_inst_2330.json")["data"] if x["date"] == "2026-11-12"}
        self.assertEqual(len(raw), 5)
        self.assertEqual(r["trust_net"], raw["Investment_Trust"]["buy"] - raw["Investment_Trust"]["sell"])
        self.assertEqual(r["foreign_net"], raw["Foreign_Investor"]["buy"] - raw["Foreign_Investor"]["sell"])
        self.assertEqual(r["dealer_self_net"], raw["Dealer_self"]["buy"] - raw["Dealer_self"]["sell"])
        self.assertAlmostEqual(r["dealer_net"], r["dealer_self_net"] + r["dealer_hedge_net"])

    def test_institutional_legacy_and_partial_periods(self):
        out = F.parse_institutional(fixture("finmind_inst_2330.json")["data"])
        # 2012-05-02: only Foreign_Investor (38841300 - 32897643), Investment_Trust, legacy combined Dealer
        r = out["2012-05-02"]
        self.assertEqual(r["foreign_net"], 38841300 - 32897643)
        self.assertEqual(r["dealer_net"], 1774000 - 1043000)
        self.assertIsNone(r["dealer_self_net"])
        self.assertIsNone(r["foreign_dealer_self_net"])
        # 2014-12-01: split dealers exist, Foreign_Dealer_Self does not yet
        r = out["2014-12-01"]
        self.assertIsNotNone(r["dealer_self_net"])
        self.assertIsNone(r["foreign_dealer_self_net"])
        self.assertIsNotNone(out["2017-12-18"]["foreign_dealer_self_net"])
        self.assertEqual(list(out), sorted(out))

    def test_institutional_tpex_stock_and_junk_rows(self):
        out = F.parse_institutional(fixture("finmind_inst_6488.json")["data"])
        self.assertEqual(out["2014-11-04"]["foreign_net"], 5000)
        junk = [{"date": "2020-01-02", "name": "Mystery", "buy": 1, "sell": 0},
                {"date": "2020-01-02", "name": "Investment_Trust", "buy": None, "sell": 0},
                {"date": "2020-01-03", "name": "Investment_Trust", "buy": 10, "sell": 4}]
        self.assertEqual(list(F.parse_institutional(junk)), ["2020-01-03"])

    def test_institutional_duplicates_and_nonfinite(self):
        rows = [{"date": "2020-01-02", "name": "Investment_Trust", "buy": 10, "sell": 4},
                {"date": "2020-01-02", "name": "Investment_Trust", "buy": 10, "sell": 4},   # exact repeat -> dropped
                {"date": "2020-01-02", "name": "Investment_Trust", "buy": 3, "sell": 0},    # different row -> summed
                {"date": "2020-01-03", "name": "Investment_Trust", "buy": "nan", "sell": 0},
                {"date": "2020-01-06", "name": "Investment_Trust", "buy": float("inf"), "sell": 0}]
        out = F.parse_institutional(rows)
        self.assertEqual(out, {"2020-01-02": {**out["2020-01-02"], "trust_net": 9.0}})
        self.assertEqual(list(out), ["2020-01-02"])
        self.assertIsNone(F.parse_per([{"date": "2020-01-02", "PER": "nan"}])["2020-01-02"]["per"])
        self.assertIsNone(F.parse_margin([{"date": "2020-01-02", "MarginPurchaseTodayBalance": float("nan")}])
                          ["2020-01-02"]["margin_balance"])

    def test_margin_lots_to_shares(self):
        rows = fixture("finmind_margin_2330.json")["data"]
        out = F.parse_margin(rows)
        first = rows[0]
        r = out[first["date"]]
        self.assertEqual(r["margin_balance"], first["MarginPurchaseTodayBalance"] * 1000)
        self.assertEqual(r["margin_balance"], 11087 * 1000)
        self.assertEqual(r["short_balance"], 7550 * 1000)
        self.assertEqual(r["margin_balance_prev"], 10771 * 1000)
        last = rows[-1]
        self.assertEqual(out[last["date"]]["margin_balance"], last["MarginPurchaseTodayBalance"] * 1000)
        self.assertEqual(out[last["date"]]["short_balance"], last["ShortSaleTodayBalance"] * 1000)
        out6 = F.parse_margin(fixture("finmind_margin_6488.json")["data"])
        self.assertEqual(out6["2016-03-28"]["margin_balance"], 0.0)

    def test_margin_blank_fields_are_none(self):
        out = F.parse_margin([{"date": "2020-01-02", "MarginPurchaseTodayBalance": 5}])
        self.assertEqual(out["2020-01-02"]["margin_balance"], 5000)
        self.assertIsNone(out["2020-01-02"]["short_balance"])

    def test_per_and_shareholding(self):
        per = F.parse_per(fixture("finmind_per_2330.json")["data"])
        self.assertEqual(per["2012-01-02"], {"per": 13.56, "pbr": 3.25, "dividend_yield": 3.99})
        hold = F.parse_shareholding(fixture("finmind_hold_2330.json")["data"])
        self.assertEqual(hold["2012-01-02"]["foreign_ratio"], 74.89)

    def test_monthly_revenue(self):
        rows = fixture("finmind_rev_2330.json")["data"]
        out = F.parse_monthly_revenue(rows)
        keys = [(r["revenue_year"], r["revenue_month"]) for r in out]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(keys[0], (2009, 12))  # date 2010-01-01 holds December 2009
        self.assertEqual(out[0]["date"], "2010-01-01")
        self.assertEqual(out[0]["revenue"], 30466359000)
        self.assertIsNone(out[0]["create_time"])  # '' on old rows -> None
        self.assertEqual(out[-1]["create_time"], "2026-11-10")
        self.assertEqual((out[-1]["revenue_year"], out[-1]["revenue_month"]), (2026, 10))
        self.assertTrue(any(r["create_time"] is None for r in out))

    def test_monthly_revenue_drops_bad_rows_and_dedupes(self):
        rows = [{"revenue_year": 2020, "revenue_month": 1, "revenue": 5, "date": "2020-02-01"},
                {"revenue_year": 2020, "revenue_month": 1, "revenue": 7, "date": "2020-02-01"},
                {"revenue_year": 2020, "revenue_month": 13, "revenue": 1},
                {"revenue_year": None, "revenue_month": 2, "revenue": 1},
                {"revenue_year": 2020, "revenue_month": 2, "revenue": None}]
        out = F.parse_monthly_revenue(rows)
        self.assertEqual([(r["revenue_month"], r["revenue"]) for r in out], [(1, 7.0)])


class CliTests(Base):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = F.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_makes_no_requests(self):
        with mock.patch.object(data, "_http_get", side_effect=AssertionError("no network in dry-run")):
            code, out, _ = self._run(["--datasets", "rev,inst", "--codes", "2330,2317", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("requests=4", out)
        self.assertIn("interval=12s", out)
        self.assertFalse((Path(self.tmp) / "alt").exists())

    def test_universe_from_pool_file_and_exit_codes(self):
        pool = Path(self.tmp) / "research" / "pool300.json"
        pool.parent.mkdir(parents=True)
        pool.write_text(json.dumps([{"code": c, "name": "x"} for c in ("2330", "2317", "2454")]), encoding="utf-8")
        self.assertEqual(F.load_universe_codes(2), ["2330", "2317"])
        with mock.patch.object(data, "_http_get", side_effect=[body([{"date": "d"}]), body([], "limit", 402)]):
            code, out, err = self._run(["--datasets", "inst", "--universe", "3"])
        self.assertEqual(code, 2)
        self.assertIn("rate limit", err)
        self.assertIn("2 files remain", err)
        with mock.patch.object(data, "_http_get", return_value=body([{"date": "d"}])):
            code, out, err = self._run(["--datasets", "inst", "--universe", "3"])  # resume the rest
        self.assertEqual(code, 0)
        self.assertIn("skipped=1", out)

    def test_missing_pool_builds_and_writes_back(self):
        items = {"2330": {"code": "2330", "name": "TSMC", "market": "TWSE", "kind": "stock", "tv": 9, "close": 1}}
        with mock.patch.object(data, "_names_table", return_value={"items": items}) as nt, \
                mock.patch.object(data, "build_universe", return_value=[{"code": "2330"}, {"code": "2317"}]) as bu:
            codes = F.load_universe_codes(5)
        self.assertEqual(codes, ["2330", "2317"])
        nt.assert_called_once_with(network=True)
        self.assertEqual(bu.call_args[1]["size"], 300)
        saved = json.loads((Path(self.tmp) / "research" / "pool300.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, [{"code": "2330"}, {"code": "2317"}])

    def test_bad_dataset_exit_1(self):
        code, _, err = self._run(["--datasets", "zzz", "--codes", "2330"])
        self.assertEqual(code, 1)
        self.assertIn("unknown FinMind dataset", err)


if __name__ == "__main__":
    unittest.main()
