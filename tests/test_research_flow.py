"""research/ (experiment flow-1): T86 parsing, causal flow signals, the pre-registered tests on synthetic markets."""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np

from research import flow_experiment as FE
from research import simulate as SM
from research import twse_flow as TF

MODERN_FIELDS = ["證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
                 "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數", "投信買進股數", "投信賣出股數", "投信買賣超股數",
                 "自營商買賣超股數", "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
                 "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數"]


def modern_row(code, foreign, trust, dealer, total):
    r = [code, "名稱"] + ["0"] * 17
    r[4], r[10], r[11], r[18] = foreign, trust, dealer, total
    return r


class ParseT86(unittest.TestCase):
    def test_modern_layout_by_name_with_thousands_separators(self):
        p = {"stat": "OK", "fields": MODERN_FIELDS,
             "data": [modern_row("2330", "12,345,678", "-1,000", "500", "12,345,178"), modern_row("2317", "-3", "0", "--", "-3")]}
        rec = TF.parse_t86(p)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["rows"]["2330"], [12345678.0, -1000.0, 500.0, 12345178.0])
        self.assertEqual(rec["rows"]["2317"], [-3.0, 0.0, 0.0, -3.0])

    def test_old_layout_before_the_foreign_dealer_split(self):
        fields = ["證券代號", "證券名稱", "外資買進股數", "外資賣出股數", "外資買賣超股數", "投信買進股數", "投信賣出股數", "投信買賣超股數",
                  "自營商買賣超股數", "三大法人買賣超股數"]
        rec = TF.parse_t86({"stat": "OK", "fields": fields, "data": [["1101", "台泥", "1", "2", "-1", "0", "0", "7", "2", "8"]]})
        self.assertEqual(rec["rows"]["1101"], [-1.0, 7.0, 2.0, 8.0])

    def test_closed_day_and_bad_answers(self):
        self.assertEqual(TF.parse_t86({"stat": "很抱歉，沒有符合條件的資料!"})["status"], "closed")
        for bad in (None, "html", {"stat": "OK", "fields": ["x"], "data": []}, {"stat": "系統忙碌"},
                    {"stat": "OK", "fields": MODERN_FIELDS, "data": []}):
            with self.assertRaises(TF.FlowUnavailable):
                TF.parse_t86(bad)

    def test_malformed_rows_are_skipped(self):
        p = {"stat": "OK", "fields": MODERN_FIELDS, "data": [["2330"], modern_row("2412", "x", "0", "0", "0"), modern_row("1101", "1", "2", "3", "6")]}
        self.assertEqual(list(TF.parse_t86(p)["rows"]), ["1101"])


class FlowMatrices(unittest.TestCase):
    def test_missing_day_is_unknown_absent_stock_is_zero(self):
        cal = np.array(["2026-01-05", "2026-01-06", "2026-01-07"], dtype="datetime64[D]")
        days = {date(2026, 1, 5): {"status": "ok", "rows": {"A": [1, 2, 3, 6]}},
                date(2026, 1, 6): None,
                date(2026, 1, 7): {"status": "closed", "rows": {}}}
        m = TF.flow_matrices(days, ["A", "B"], cal)
        self.assertEqual(m["foreign"][0].tolist(), [1.0, 0.0])
        self.assertEqual(m["total"][0].tolist(), [6.0, 0.0])
        self.assertTrue(np.isnan(m["trust"][1:]).all())


class FlowSignal(unittest.TestCase):
    def setUp(self):
        self.panel, self.flows, _, _ = SM.make_world(n_stocks=6, n_days=80, seed=3)

    def test_value_is_the_lagged_window_ratio(self):
        f, v = self.flows["foreign"], self.panel.vol
        sig = FE.flow_signal(self.panel, f, 5, lag=1)
        t, j = 30, 2
        self.assertAlmostEqual(sig[t, j], f[t - 5:t, j].sum() / v[t - 5:t, j].sum(), places=12)
        self.assertTrue(np.isnan(sig[:5]).all())                   # 5-row window + 1 lag: first value at row 5
        self.assertTrue(np.isfinite(sig[5:]).all())

    def test_strictly_causal_poisoning_the_future_changes_nothing_before_it(self):
        base = FE.flow_signal(self.panel, self.flows["trust"], 20)
        for cut in (25, 40, 79):
            f = self.flows["trust"].copy()
            f[cut:] = np.nan if cut % 2 else 1e12
            v = self.panel.vol.copy()
            self.panel.vol[cut:] = 7.0
            try:
                got = FE.flow_signal(self.panel, f, 20)
            finally:
                self.panel.vol[:] = v
            np.testing.assert_array_equal(got[:cut + 1], base[:cut + 1])     # row `cut` uses rows <= cut - 1 only

    def test_unknown_flow_day_blanks_every_window_that_contains_it(self):
        f = self.flows["foreign"].copy()
        f[40, 0] = np.nan
        sig = FE.flow_signal(self.panel, f, 5, lag=1)
        self.assertTrue(np.isnan(sig[41:46, 0]).all())
        self.assertTrue(np.isfinite(sig[40, 0]) and np.isfinite(sig[46, 0]))


class Protocol(unittest.TestCase):
    """The full pre-registered run on synthetic markets (fixed seeds)."""

    def test_constants_match_the_registration(self):
        self.assertEqual(FE.N_TESTS, 6)
        self.assertAlmostEqual(FE.ALPHA, 0.05 / 6)
        self.assertEqual(FE.LAG, 1)
        self.assertEqual((FE.DEV_RANKS, FE.HOLD_RANKS), ((1, 60), (61, 120)))
        self.assertEqual(FE.HORIZONS, (5, 20))

    def test_no_signal_world_fails(self):
        panel, flows, dev, hold = SM.make_world(strength=0.0, seed=11)
        rep = FE.run(panel, flows, dev, hold)
        self.assertEqual(rep["verdict"], "fail")
        self.assertEqual(rep["winners"], [])
        self.assertEqual(len(rep["results"]), 6)
        self.assertIn("沒有可用的證據", FE.to_markdown(rep))

    def test_planted_foreign_signal_is_found_and_the_others_are_not(self):
        panel, flows, dev, hold = SM.make_world(strength=0.002, seed=12)
        rep = FE.run(panel, flows, dev, hold)
        by = {(r["signal"], r["h"]): r for r in rep["results"]}
        self.assertTrue(by[("foreign_20", 20)]["signal_holds"])
        self.assertTrue(by[("foreign_20", 20)]["profitable"])
        self.assertFalse(by[("trust_20", 20)]["signal_holds"])     # independent flows carry no information
        self.assertEqual(rep["verdict"], "pass")
        b = by[("foreign_20", 20)]["basket_a"]
        self.assertGreater(b["top"]["cagr"], b["equal_weight"]["cagr"])

    def test_costs_are_charged(self):
        panel, flows, dev, hold = SM.make_world(strength=0.0, seed=13)
        sig = FE.flow_signal(panel, flows["foreign"], 20)
        rows = np.ones(panel.T, dtype=bool)
        b = FE.basket_test(panel, sig, rows, hold, 5)
        self.assertGreater(b["turnover_per_year"], 1.0)
        # with no signal, a costly basket must not beat the cost-free equal-weight universe by a wide margin
        self.assertLess(b["top"]["cagr"], b["equal_weight"]["cagr"] + 0.15)


class CommandLine(unittest.TestCase):
    """main() end to end with the network replaced by a synthetic market (what the user runs on a connected machine)."""

    def test_main_writes_the_report(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from backend import data as D

        panel, flows, _, _ = SM.make_world(n_stocks=100, n_days=1150, strength=0.0, seed=21)
        datas = SM._datas(panel.cal, np.diff(np.log(np.vstack([np.full((1, panel.S), 100.0), panel.adj])), axis=0), panel.vol)
        by_code = {d.symbol: d for d in datas}
        uni = [{"code": d.symbol, "name": d.name, "market": "TWSE" if i % 10 else "TPEx", "kind": "stock"} for i, d in enumerate(datas)]
        col = {c: j for j, c in enumerate(panel.codes)}

        def fake_day(d, network=True):
            t = int(np.searchsorted(panel.cal, np.datetime64(d)))
            return {"status": "ok", "rows": {c: [flows[k][t, j] for k in TF.FLOW_KEYS] for c, j in col.items()}}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(D, "get_universe", lambda n: uni[:n]), \
                mock.patch.object(D, "get_price_data", lambda code, refresh=False: by_code[code]), \
                mock.patch.object(TF, "fetch_day", fake_day), \
                mock.patch.object(FE, "RESULTS", Path(tmp)):
            rc = FE.main(["--offline"])
            self.assertEqual(rc, 0)
            files = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(sorted(Path(f).suffix for f in files), [".json", ".md"])
            import json
            rep = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(rep["n_dev_stocks"], 54)          # ranks 1-60 minus the 6 TPEx names (every 10th)
        self.assertEqual(rep["n_hold_stocks"], 36)         # ranks 61-100 minus 4 TPEx names (the fake universe has 100)
        self.assertEqual(rep["t86_days_missing"], 0)
        self.assertEqual(rep["verdict"], "fail")

    def test_main_refuses_a_verdict_when_the_download_was_blocked(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from backend import data as D

        panel, flows, _, _ = SM.make_world(n_stocks=50, n_days=400, seed=22)
        datas = SM._datas(panel.cal, np.zeros((panel.T, panel.S)), panel.vol)
        by_code = {d.symbol: d for d in datas}
        uni = [{"code": d.symbol, "name": d.name, "market": "TWSE", "kind": "stock"} for d in datas]

        def blocked(d, network=True):
            raise TF.FlowUnavailable("因為安全性考量")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(D, "get_universe", lambda n: uni[:n]), \
                mock.patch.object(D, "get_price_data", lambda code, refresh=False: by_code[code]), \
                mock.patch.object(TF, "fetch_day", blocked), \
                mock.patch.object(FE, "RESULTS", Path(tmp)):
            self.assertEqual(FE.main(["--offline"]), 3)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
