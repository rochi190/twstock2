"""format.js: number / date / semantic formatters.  Contract: null, undefined, NaN, +-Infinity and junk give "—", nothing throws."""
from __future__ import annotations

import json
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp

NA = "—"
M = "−"          # typographic minus U+2212


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Format(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js"], fixtures=[])

    def f(self, name, *args):
        js = "TQ.fmt." + name + "(" + ",".join(json.dumps(a) if a is not ... else "undefined" for a in args) + ")"
        return self.app.json(js)

    def test_every_formatter_survives_junk(self):
        junk = ["null", "undefined", "NaN", "Infinity", "-Infinity", "''", "'abc'", "{}", "[]", "true", "function(){}"]
        names = ["fmtNum", "fmtInt", "fmtPct", "fmtPP", "fmtBps", "fmtSigned", "fmtTWD", "fmtPrice", "fmtBigTWD", "fmtLots", "fmtCompact", "fmtP",
                 "fmtCI", "fmtCIn", "fmtDate", "fmtMD", "fmtWeekday", "fmtDateW", "fmtYM", "fmtDateTime", "fmtChange", "tickSize", "decimalsForStep"]
        r = self.app.json("(function () { var bad = []; var junk = [" + ",".join(junk) + "]; var names = " + json.dumps(names) + ";"
                          "names.forEach(function (n) { junk.forEach(function (j, i) { try { var v = TQ.fmt[n](j, j, j); "
                          "if (typeof v === 'string' && /NaN|undefined|Infinity|null|\\[object/.test(v)) bad.push(n + ' ' + i + ' -> ' + v); } catch (e) { bad.push(n + ' ' + i + ' threw ' + e); } }); });"
                          "return bad; })()")
        self.assertEqual(r, [])

    def test_numbers(self):
        self.assertEqual(self.f("fmtNum", 1234567.891, 2), "1,234,567.89")
        self.assertEqual(self.f("fmtNum", -0.004, 2), "0.00")                 # no "-0.00"
        self.assertEqual(self.f("fmtNum", -1234.5, 1), f"{M}1,234.5")
        self.assertEqual(self.f("fmtNum", 0.5, 0, True), "+1")
        self.assertEqual(self.f("fmtNum", "12.5", 1), "12.5")                 # numeric strings accepted
        self.assertEqual(self.f("fmtNum", "", 1), NA)
        self.assertEqual(self.f("fmtNum", True), NA)
        self.assertEqual(self.f("fmtInt", 1234.6), "1,235")
        self.assertEqual(self.f("fmtSigned", 0.5, 2), "+0.50")
        self.assertEqual(self.f("fmtSigned", -0.5, 2), f"{M}0.50")
        self.assertEqual(self.f("fmtSigned", 0, 2), "0.00")

    def test_percent_family(self):
        self.assertEqual(self.f("fmtPct", 0.0123), "1.2%")
        self.assertEqual(self.f("fmtPct", 0.0123, 2, True), "+1.23%")
        self.assertEqual(self.f("fmtPct", -0.0145, 2, True), f"{M}1.45%")
        self.assertEqual(self.f("fmtPct", 0.5164, 1), "51.6%")
        self.assertEqual(self.f("fmtPct", 0, 1, True), "0.0%")
        self.assertEqual(self.f("fmtPct", 1e-9, 1, True), "0.0%")
        self.assertEqual(self.f("fmtPP", 0.012), "+1.2pp")
        self.assertEqual(self.f("fmtBps", 0.0012), "12bp")
        self.assertEqual(self.f("fmtBps", -0.00027, 1, True), f"{M}2.7bp")

    def test_change_glyph_follows_the_rounded_value(self):
        self.assertEqual(self.f("fmtChange", 0.0123), "▲ +1.23%")
        self.assertEqual(self.f("fmtChange", -0.0145), f"▼ {M}1.45%")
        self.assertEqual(self.f("fmtChange", 0), "▬ 0.00%")
        self.assertEqual(self.f("fmtChange", 0.00001), "▬ 0.00%")
        self.assertEqual(self.f("fmtChange", None), NA)

    def test_prices_follow_twse_ticks(self):
        self.assertEqual(self.f("tickSize", 9.99), 0.01)
        self.assertEqual(self.f("tickSize", 10), 0.05)
        self.assertEqual(self.f("tickSize", 49.95), 0.05)
        self.assertEqual(self.f("tickSize", 50), 0.1)
        self.assertEqual(self.f("tickSize", 100), 0.5)
        self.assertEqual(self.f("tickSize", 500), 1)
        self.assertEqual(self.f("tickSize", 1000), 5)
        self.assertEqual(self.f("tickSize", 49.9, "etf"), 0.01)
        self.assertEqual(self.f("tickSize", 50, "etf"), 0.05)
        self.assertEqual(self.f("fmtPrice", 2385), "2,385")
        self.assertEqual(self.f("fmtPrice", 301.0), "301.0")
        self.assertEqual(self.f("fmtPrice", 42.1), "42.10")
        self.assertEqual(self.f("fmtPrice", 106.9, "etf"), "106.90")
        self.assertEqual(self.f("fmtPrice", 7.05), "7.05")
        self.assertEqual(self.f("fmtPrice", None), NA)
        self.assertEqual(self.f("fmtTWD", 2385), "NT$2,385")
        self.assertEqual(self.f("fmtTWD", -50), f"{M}NT$50")

    def test_taiwanese_units(self):
        self.assertEqual(self.f("fmtBigTWD", 6.5e10), "650億")
        self.assertEqual(self.f("fmtBigTWD", 3.2e12), "3.20兆")
        self.assertEqual(self.f("fmtBigTWD", 5e7), "5,000萬")
        self.assertEqual(self.f("fmtBigTWD", 1.5e8), "1.50億")
        self.assertEqual(self.f("fmtBigTWD", 12.3e8), "12.3億")
        self.assertEqual(self.f("fmtBigTWD", -2e8), f"{M}2.00億")
        self.assertEqual(self.f("fmtLots", 8360404), "8,360張")
        self.assertEqual(self.f("fmtLots", 27360404), "2.74萬張")
        self.assertEqual(self.f("fmtLots", 131498869), "13.15萬張")
        self.assertEqual(self.f("fmtCompact", 1234567), "123萬")
        self.assertEqual(self.f("fmtCompact", 86000), "8.6萬")
        self.assertEqual(self.f("fmtCompact", 12), "12.0")

    def test_p_values_and_intervals(self):
        self.assertEqual(self.f("fmtP", 0.0004), "<0.001")
        self.assertEqual(self.f("fmtP", 0.031), "0.031")
        self.assertEqual(self.f("fmtP", 0.31), "0.31")
        self.assertEqual(self.f("fmtP", 1), "1.00")
        self.assertEqual(self.f("fmtCI", [0.508, 0.564]), "[50.8%, 56.4%]")
        self.assertEqual(self.f("fmtCI", [None, 0.5]), NA)
        self.assertEqual(self.f("fmtCI", None), NA)
        self.assertEqual(self.f("fmtCIn", [-0.31, 0.86]), f"[{M}0.31, 0.86]")

    def test_dates(self):
        self.assertEqual(self.f("fmtDate", "2026-10-22"), "2026-10-22")
        self.assertEqual(self.f("fmtDate", "2026-10-22T18:31:07+08:00"), "2026-10-22")
        self.assertEqual(self.f("fmtDate", "2026-02-31"), NA)
        self.assertEqual(self.f("fmtDate", "2026-13-01"), NA)
        self.assertEqual(self.f("fmtDate", 20261022), NA)
        self.assertEqual(self.f("fmtMD", "2026-10-02"), "10/2")
        self.assertEqual(self.f("fmtWeekday", "2026-10-22"), "週四")
        self.assertEqual(self.f("fmtWeekday", "2026-10-25"), "週日")
        self.assertEqual(self.f("fmtDateW", "2026-10-22"), "2026-10-22 (週四)")
        self.assertEqual(self.f("fmtYM", "2026-10-22"), "2026-10")
        self.assertEqual(self.f("fmtDateTime", "2026-10-22T22:03:15+08:00"), "2026-10-22 22:03")
        self.assertEqual(self.f("fmtDateTime", "2026-10-22"), "2026-10-22")
        self.assertEqual(self.f("weekdaysAfter", "2026-10-22", 3), ["2026-10-23", "2026-10-26", "2026-10-27"])
        self.assertEqual(self.f("weekdaysAfter", "garbage", 3), [])

    def test_semantics(self):
        self.assertEqual([self.f("dirOf", 1), self.f("dirOf", -1), self.f("dirOf", 0), self.f("dirOf", None), self.f("dirOf", 0.001, 0.01)], ["up", "down", "flat", "na", "flat"])
        self.assertEqual([self.f("arrow", "up"), self.f("arrow", "down"), self.f("arrow", "flat"), self.f("arrow", "na")], ["▲", "▼", "▬", ""])
        # Changed on purpose (HANDOFF 2.8 / audit P-10): the default buy label is the statistical screen 偏多候選; the advice wording 偏多·買進
        # appears only when the caller passes unlocked === true (evidence >= moderate_edge and the basket not losing to the universe after costs).
        self.assertEqual(self.f("signalInfo", "buy"), {"label": "偏多候選", "short": "偏多", "dir": "up", "icon": "▲"})
        self.assertEqual(self.f("signalInfo", "buy", False)["label"], "偏多候選")
        self.assertEqual(self.f("signalInfo", "buy", "yes")["label"], "偏多候選")             # only the boolean true unlocks
        self.assertEqual(self.f("signalInfo", "buy", True), {"label": "偏多·買進", "short": "買進", "dir": "up", "icon": "▲"})
        self.assertEqual(self.f("signalInfo", "hold", True)["label"], "中性觀望")                 # nothing else changes with the gate
        self.assertEqual(self.f("signalInfo", "avoid", True)["label"], "偏空／避開")
        self.assertEqual([self.f("levelUnlocksBuy", x) for x in ("no_edge", "weak_edge", "moderate_edge", "strong_edge", None)], [False, False, True, True, False])
        self.assertEqual(self.f("signalInfo", "avoid")["dir"], "down")
        self.assertEqual(self.f("signalInfo", "hold")["dir"], "flat")
        self.assertEqual(self.f("signalInfo", "zzz"), {"label": NA, "short": NA, "dir": "na", "icon": ""})
        self.assertEqual(self.f("signalInfo", "toString")["dir"], "na")          # prototype keys are not signals
        self.assertEqual(self.f("verdictInfo", "no_edge")["label"], "無預測優勢")
        self.assertEqual(self.f("verdictInfo", "constructor")["label"], "未判定")
        self.assertEqual(self.f("baselineName", "always_up"), "永遠看漲")
        self.assertEqual(self.f("baselineName", "toString"), "toString")
        self.assertEqual(self.f("baselineName", None), NA)
        self.assertEqual(self.f("marketName", "constructor"), "constructor")
        self.assertEqual(self.f("marketName", "TWSE"), "上市")
        self.assertEqual(self.f("marketName", "TPEx"), "上櫃")
        self.assertEqual(self.f("kindName", "etf"), "ETF")
        self.assertEqual(self.f("execName", "next_open"), "次日開盤價成交")

    def test_util(self):
        a = self.app
        self.assertEqual(a.json("TQ.util.clamp(5, 0, 3)"), 3)
        self.assertEqual(a.json("TQ.util.extent([3, NaN, 1, null, 7, 'x'])"), [1, 7])
        self.assertIsNone(a.json("TQ.util.extent([])"))
        self.assertEqual(a.json("TQ.util.get({a: {b: [{c: 5}]}}, 'a.b.0.c')"), 5)
        self.assertIsNone(a.json("TQ.util.get({a: null}, 'a.b.c')"))
        w = a.json("TQ.util.wilson(50, 100)")
        self.assertAlmostEqual(w[0], 0.4038, places=3)
        self.assertAlmostEqual(w[1], 0.5962, places=3)
        self.assertIsNone(a.json("TQ.util.wilson(1, 0)"))


if __name__ == "__main__":
    unittest.main()
