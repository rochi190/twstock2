"""Integration / smoke tests for the evaluation module.

* ``TestEngineIntegration``: the real ``quant.engine.walk_forward`` on synthetic bars (no network); skipped when the
  engine cannot be imported.  Checks that a real ``WalkForward`` passes the alignment guards and yields schema-valid
  reports and that the verdict on a pure random walk is ``no_edge``.
* ``TestRealYahoo``: real 2330 / 0050 / ^TWII prices from Yahoo (clearly marked, skipped gracefully when offline).
"""
import json
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import eval_helpers as H  # noqa: E402
from backend.evaluation import report as RP  # noqa: E402
from backend.evaluation import strategy as S  # noqa: E402
from backend.evaluation import universe_bt as U  # noqa: E402
from backend.types import PriceData  # noqa: E402

try:
    from backend.quant import engine as ENGINE
except Exception:  # pragma: no cover - engine missing/broken
    ENGINE = None


def fetch_yahoo(symbol: str, timeout: float = 20.0):
    """Minimal Yahoo daily-bar fetch -> PriceData (only for these smoke tests); ``None`` if offline/unparseable."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=0&period2={int(time.time())}"
           f"&interval=1d&events=div%2Csplits")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())["chart"]["result"][0]
        ts = np.array(res["timestamp"], dtype=np.int64) + int(res["meta"].get("gmtoffset", 28800))
        q = res["indicators"]["quote"][0]
        adj = np.array([np.nan if v is None else v for v in res["indicators"]["adjclose"][0]["adjclose"]], dtype=float)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    cols = {k: np.array([np.nan if v is None else v for v in q[k]], dtype=float)
            for k in ("open", "high", "low", "close", "volume")}
    ok = np.all([np.isfinite(cols[k]) for k in ("open", "high", "low", "close")], axis=0) & np.isfinite(adj)
    dates = (ts // 86400).astype("datetime64[D]")[ok]
    sel = np.flatnonzero(ok)[np.concatenate([[True], np.diff(dates.astype(np.int64)) > 0])]
    dates = (ts // 86400).astype("datetime64[D]")[sel]
    code = symbol.split(".")[0].replace("%5E", "^")
    kind = "index" if code.startswith("^") else ("etf" if code.startswith("00") else "stock")
    return PriceData(code, code, "INDEX" if kind == "index" else "TWSE", kind, dates, cols["open"][sel],
                     cols["high"][sel], cols["low"][sel], cols["close"][sel], adj[sel],
                     np.nan_to_num(cols["volume"][sel]))


@unittest.skipIf(ENGINE is None, "quant engine not importable")
class TestEngineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = H.make_price_data(1300, seed=3, drift=0.0004, vol=0.015, symbol="SYN", name="合成")
        cls.wf = ENGINE.walk_forward(cls.data, None, start_idx=950)

    def test_real_walkforward_passes_guards_and_schema(self):
        for h in (1, 5, 20):
            chk = RP.check_realized_alignment(self.data, self.wf.horizons[h])
            self.assertTrue(chk["ok"], chk)
            rep = RP.build_backtest_report(self.data, self.wf, h, n_boot=200)
            self.assertEqual(RP.validate_backtest_report(rep), [], h)
            json.dumps(rep, allow_nan=False)
            self.assertEqual(rep["period"]["start"], str(self.data.dates[self.wf.first_idx]))
            # a random walk has no exploitable structure: the engine cannot show a statistically solid edge
            self.assertEqual(rep["verdict"]["level"], "no_edge", (h, rep["verdict"]["text"]))
            self.assertTrue(0.8 < rep["returns"]["coverage"]["0.90"] / 0.9 < 1.1 or h == 20)

    def test_causality_of_the_whole_pipeline_with_real_forecasts(self):
        # strategy returns up to a day t must not change if the data after t is replaced
        h = 5
        hf = self.wf.horizons[h]
        a = S.run_strategy(self.data, hf, self.wf.first_idx, n_boot=50)
        cut = 1100
        d2 = self.data.slice(self.data.n)
        rng = np.random.default_rng(0)
        for arr in (d2.open, d2.high, d2.low, d2.close, d2.adjclose):
            arr[cut + 1:] *= np.exp(rng.normal(0, 0.03, len(arr) - cut - 1))
        b = S.run_strategy(d2, hf, self.wf.first_idx, n_boot=50)
        k = cut - self.wf.first_idx
        np.testing.assert_array_equal(a.sim["ret"][:k], b.sim["ret"][:k])

    def test_universe_with_real_walkforwards(self):
        panel = {}
        for i in range(4):
            d = H.make_price_data(1000, seed=20 + i, symbol=f"{2000 + i}", name=f"合成{i}")
            panel[d.symbol] = (d, ENGINE.walk_forward(d, None, start_idx=800))
        rep = U.evaluate_universe(panel, 5, top_k=2, n_boot=100)
        self.assertEqual(U.validate_universe_report(rep), [])
        self.assertEqual(rep["period"]["n_stocks"], 4)
        self.assertEqual(rep["verdict"]["level"], "no_edge")


class TestRealYahoo(unittest.TestCase):
    """Network smoke test (skipped when Yahoo is unreachable or the engine is missing)."""

    @classmethod
    def setUpClass(cls):
        if ENGINE is None:
            raise unittest.SkipTest("quant engine not importable")
        cls.px = {}
        for sym in ("2330.TW", "0050.TW", "%5ETWII"):
            d = fetch_yahoo(sym)
            if d is None or d.n < 1500:
                raise unittest.SkipTest("Yahoo Finance not reachable (offline smoke test skipped)")
            cls.px[d.symbol] = d
        for k, d in list(cls.px.items()):
            if str(d.dates[-1]) >= time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600)):
                cls.px[k] = d.slice(d.n - 1)           # drop a possibly unfinished bar of today (Taipei)

    def test_leg_returns_are_sane_on_real_bars(self):
        for code, d in self.px.items():
            co, oc, cc = S.leg_returns(d)
            self.assertTrue(np.all(np.isfinite(co)) and np.all(np.isfinite(oc)))
            np.testing.assert_allclose((1 + co) * (1 + oc), 1 + cc, atol=1e-12)
            # evaluation windows (last 1250 bars) must be free of split artefacts; Yahoo's 0050 history contains an
            # unadjusted 4:1 split on 2014-01-02 which the data-quality alert must catch when it is in the window
            self.assertEqual(RP.data_alerts(d, d.n - 1250), [], code)
            if code == "0050":
                full = RP.data_alerts(d, 0)
                if full:
                    self.assertTrue(any(a[0] == "2014-01-02" and a[1] < -0.7 for a in full), full)

    def test_single_stock_reports_on_real_data(self):
        for code in ("2330", "0050"):
            d = self.px[code]
            wf = ENGINE.walk_forward(d, None, start_idx=d.n - 450)
            for h in (1, 5, 20):
                rep = RP.build_backtest_report(d, wf, h, n_boot=200)
                self.assertEqual(RP.validate_backtest_report(rep), [], (code, h))
                json.dumps(rep, allow_nan=False)
                self.assertGreater(rep["period"]["n_scored"], 380 - 20)
                self.assertEqual(rep["kind"], "etf" if code == "0050" else "stock")
                self.assertIn(rep["verdict"]["level"], RP.LEVELS)
                # honesty: no stock/horizon of this size should be declared a 'strong' edge by chance
                self.assertNotEqual(rep["verdict"]["level"], "strong_edge")
            rep = RP.build_backtest_report(d, wf, 1)
            self.assertAlmostEqual(rep["params"]["roundtrip_cost"],
                                   S.config.roundtrip_cost("etf" if code == "0050" else "stock"))

    def test_universe_pipeline_with_taiex_on_real_data(self):
        panel = {}
        for code in ("2330", "0050"):
            d = self.px[code]
            panel[code] = (d, ENGINE.walk_forward(d, None, start_idx=d.n - 350))
        rep = U.evaluate_universe(panel, 5, top_k=1, taiex=self.px["^TWII"], n_boot=100)
        self.assertEqual(U.validate_universe_report(rep), [])
        self.assertIsNotNone(rep["benchmark_taiex"])
        self.assertEqual(len(rep["curve"]["taiex"]), len(rep["curve"]["dates"]))
        self.assertIn("存活者偏差", rep["notes"][0])


if __name__ == "__main__":
    unittest.main()
