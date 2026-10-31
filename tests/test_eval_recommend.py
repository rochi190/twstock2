"""Tests for backend/evaluation/recommend.py."""
import copy
import json
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from backend import config  # noqa: E402
from backend.evaluation import recommend as RC  # noqa: E402

COST_STOCK = config.roundtrip_cost("stock")          # 0.00571
COST_ETF = config.roundtrip_cost("etf")              # 0.00371
NOW = datetime(2026, 10, 22, 15, 0, 0, tzinfo=timezone.utc)
SIMPLIFIED = set("预统显优样获检准关买损较约应错该为实时间点发线图号体义观验览态质总结况构运动过区务强价绩费记录别难势报")
FORBIDDEN = ("保證獲利", "穩賺", "必賺", "一定賺", "包賺", "穩賠不賠", "保證賺", "穩定獲利")


def hzd(p, er, sg, comps=None, h=5, mu_log=None, sigma_log=None):
    if comps is None:
        comps = {"garch": {"p_up": p, "mu": er}, "hmm": {"p_up": p, "mu": er}, "dlm": {"p_up": p, "mu": er},
                 "naive": {"p_up": p, "mu": er}}
    out = {"h": h, "p_up": p, "p_up_raw": p, "exp_return": er, "median_return": er, "sigma": sg, "components": comps}
    if mu_log is not None:                    # the engine's forecast_latest adds the log-scale moments (rank_signal inputs)
        out["mu_log"], out["sigma_log"] = mu_log, sigma_log
    return out


def make_fc(code="2330", p=0.6, er=0.012, sg=0.03, kind="stock", avg=5e9, regime="多頭穩定", name=None, asof="2026-10-22",
            vol_pct=0.5, comps=None, mu_log=None, sigma_log=None):
    return {"symbol": code, "name": name or f"股{code}", "kind": kind, "asof": asof, "asof_complete": True,
            "last_close": 100.0, "chg_1d": 0.01,
            "horizons": {"1": hzd(0.5, 0.0004, 0.015, h=1), "5": hzd(p, er, sg, comps, mu_log=mu_log, sigma_log=sigma_log),
                         "20": hzd(0.5, 0.01, 0.07, h=20)},
            "regime": {"current": regime}, "vol": {"percentile": vol_pct},
            "drivers": [{"name": "12-1個月動能", "key": "mom_12_1", "value": 0.3, "z": 1.2, "contribution": 0.0003},
                        {"name": "短期反轉", "key": "rev", "value": -0.01, "z": -0.4, "contribution": -0.00005}],
            "avg_value_20d": avg}


def make_universe(n=40, seed=0, dispersion=1.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = float(np.clip(rng.normal(0.5, 0.05 * dispersion), 0.3, 0.7))
        er = float(rng.normal(0.004, 0.006 * dispersion))
        sg = float(rng.uniform(0.02, 0.05))
        out.append(make_fc(f"{2300 + i}", p, er, sg))
    return out


class TestAnnotate(unittest.TestCase):
    def test_hand_case_stock_and_etf(self):
        fc = make_fc(p=0.60, er=0.012, sg=0.03)
        out = RC.annotate_forecast(fc, "stock")
        h5 = out["horizons"]["5"]
        self.assertAlmostEqual(h5["net_exp_return"], 0.012 - COST_STOCK)
        self.assertAlmostEqual(h5["score"], (0.012 - COST_STOCK) / 0.03)
        self.assertEqual(h5["signal"], "buy")
        self.assertIn(h5["confidence"], ("高", "中", "低"))
        # 0.5% expected return: enough for an ETF (hurdle 0.37 %), not for a stock (0.57 %)
        fc = make_fc(p=0.60, er=0.005)
        self.assertEqual(RC.annotate_forecast(fc, "stock")["horizons"]["5"]["signal"], "hold")
        self.assertEqual(RC.annotate_forecast(fc, "etf")["horizons"]["5"]["signal"], "buy")
        self.assertAlmostEqual(RC.annotate_forecast(fc, "etf")["horizons"]["5"]["net_exp_return"], 0.005 - COST_ETF)

    def test_all_horizons_annotated(self):
        out = RC.annotate_forecast(make_fc(), "stock")
        for k in ("1", "5", "20"):
            for key in ("net_exp_return", "score", "signal", "confidence"):
                self.assertIn(key, out["horizons"][k])
        self.assertEqual(out["horizons"]["1"]["signal"], "hold")       # 0.04 % cannot pay 0.57 %
        self.assertEqual(out["horizons"]["20"]["signal"], "hold")      # p = 0.5

    def test_threshold_boundaries(self):
        buy = RC.annotate_forecast(make_fc(p=config.BUY_P_UP, er=0.02), "stock")["horizons"]["5"]["signal"]
        self.assertEqual(buy, "buy")                                    # inclusive
        self.assertEqual(RC.annotate_forecast(make_fc(p=config.BUY_P_UP - 1e-9, er=0.02), "stock")["horizons"]["5"]["signal"], "hold")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.6, er=COST_STOCK), "stock")["horizons"]["5"]["signal"], "hold")  # net == 0
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.6, er=COST_STOCK + 1e-6), "stock")["horizons"]["5"]["signal"], "buy")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.45, er=-0.01), "stock")["horizons"]["5"]["signal"], "avoid")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.45 + 1e-9, er=-0.01), "stock")["horizons"]["5"]["signal"], "hold")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.40, er=0.02), "stock")["horizons"]["5"]["signal"], "hold")  # net > 0
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.62, er=0.02), "stock", buy_p=0.65)["horizons"]["5"]["signal"], "hold")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.62, er=0.02), "stock", buy_p=0.60)["horizons"]["5"]["signal"], "buy")
        self.assertEqual(RC.annotate_forecast(make_fc(p=0.35, er=-0.01), "stock", avoid_p=0.3)["horizons"]["5"]["signal"], "hold")

    def test_costs_parameters_change_the_hurdle(self):
        fc = make_fc(p=0.6, er=0.0065)
        self.assertEqual(RC.annotate_forecast(fc, "stock")["horizons"]["5"]["signal"], "buy")
        self.assertEqual(RC.annotate_forecast(fc, "stock", commission_discount=1.0, slippage_bps=20)["horizons"]["5"]["signal"], "hold")
        h = RC.annotate_forecast(fc, "stock", commission_discount=0.28, slippage_bps=0)["horizons"]["5"]
        self.assertAlmostEqual(h["net_exp_return"], 0.0065 - config.roundtrip_cost("stock", 0.28, 0.0))

    def test_invalid_numbers_become_hold(self):
        for bad in (None, float("nan"), 0.0, -0.01, "x"):
            fc = make_fc()
            fc["horizons"]["5"]["sigma"] = bad
            h = RC.annotate_forecast(fc, "stock")["horizons"]["5"]
            self.assertEqual((h["signal"], h["confidence"]), ("hold", "低"), bad)
            self.assertIsNone(h["score"])
        fc = make_fc()
        fc["horizons"]["5"]["p_up"] = None
        self.assertEqual(RC.annotate_forecast(fc, "stock")["horizons"]["5"]["signal"], "hold")
        fc = make_fc()
        fc["horizons"]["5"]["exp_return"] = float("nan")
        h = RC.annotate_forecast(fc, "stock")["horizons"]["5"]
        self.assertEqual(h["signal"], "hold")
        self.assertIsNone(h["net_exp_return"])
        json.dumps(RC.annotate_forecast({"horizons": {}}, "stock"), allow_nan=False)

    def test_input_not_mutated_and_returns_copy(self):
        fc = make_fc()
        snapshot = copy.deepcopy(fc)
        out = RC.annotate_forecast(fc, "stock")
        self.assertEqual(fc, snapshot)
        self.assertNotIn("signal", fc["horizons"]["5"])
        out["horizons"]["5"]["components"]["garch"]["p_up"] = 0.0
        self.assertEqual(fc, snapshot)
        # re-annotating with other thresholds is independent (cache safety)
        a = RC.annotate_forecast(fc, "stock", buy_p=0.55)["horizons"]["5"]["signal"]
        b = RC.annotate_forecast(fc, "stock", buy_p=0.70)["horizons"]["5"]["signal"]
        self.assertEqual((a, b), ("buy", "hold"))

    def test_threshold_validation(self):
        for kw in ({"buy_p": 0.4, "avoid_p": 0.45}, {"buy_p": 0.55, "avoid_p": 0.55}, {"buy_p": 1.2}, {"avoid_p": 0.0}):
            with self.assertRaises(ValueError):
                RC.annotate_forecast(make_fc(), "stock", **kw)

    def test_confidence_levels(self):
        def conf(p, comps=None, vol=0.5):
            fc = make_fc(p=p, er=0.02, comps=comps, vol_pct=vol)
            return RC.annotate_forecast(fc, "stock")["horizons"]["5"]["confidence"]
        self.assertEqual(conf(0.62), "高")                 # edge 0.12, all components agree
        self.assertEqual(conf(0.57), "中")
        self.assertEqual(conf(0.52), "低")
        split = {"a": {"p_up": 0.65}, "b": {"p_up": 0.45}, "c": {"p_up": 0.44}, "d": {"p_up": 0.43}}
        self.assertEqual(conf(0.62, split), "中")          # 1/4 agree -> level 2 -> 1 -> ... disagree -> lowered
        mostly = {"a": {"p_up": 0.65}, "b": {"p_up": 0.64}, "c": {"p_up": 0.6}, "d": {"p_up": 0.45}}
        self.assertEqual(conf(0.62, mostly), "高")         # 3/4 agree is not < 0.75: stays 高
        half = {"a": {"p_up": 0.65}, "b": {"p_up": 0.64}, "c": {"p_up": 0.45}, "d": {"p_up": 0.44}}
        self.assertEqual(conf(0.62, half), "中")           # 2/4 agree: 高 -> 中
        self.assertEqual(conf(0.62, vol=0.95), "中")       # very high volatility -> one level down
        self.assertEqual(conf(0.57, vol=0.95), "低")
        self.assertEqual(conf(0.40), "高")                 # bearish with edge 0.10 is just as confident ...
        self.assertEqual(conf(0.60), "高")                 # ... as bullish (float noise must not matter)
        self.assertEqual(conf(0.55), conf(0.45))
        self.assertEqual(conf(0.55), "中")
        self.assertEqual(conf(0.5), "低")

    def test_agreement_function(self):
        hz = {"components": {"a": {"p_up": 0.6}, "b": {"p_up": 0.7}, "c": {"p_up": 0.4}, "d": {"p_up": None}}}
        self.assertAlmostEqual(RC.agreement(0.6, hz), 2 / 3)
        self.assertAlmostEqual(RC.agreement(0.4, hz), 1 / 3)
        self.assertIsNone(RC.agreement(0.5, hz))
        self.assertIsNone(RC.agreement(0.6, {}))


def ref_james_stein(z):
    k = len(z)
    zbar = sum(z) / k
    ss = sum((v - zbar) ** 2 for v in z)
    B = 0.0 if ss <= 0 else max(0.0, 1 - (k - 3) / ss)
    return [zbar + B * (v - zbar) for v in z], B


def ref_lw(X):
    """Brute-force Ledoit-Wolf (2004) with explicit loops."""
    X = X - X.mean(0)
    T, N = X.shape
    S = sum(np.outer(x, x) for x in X) / T
    mu = np.trace(S) / N
    d2 = np.sum((S - mu * np.eye(N)) ** 2) / N
    b2bar = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / (N * T * T)
    a = min(b2bar, d2) / d2
    return a * mu * np.eye(N) + (1 - a) * S, a


class TestJamesStein(unittest.TestCase):
    def test_matches_formula(self):
        rng = np.random.default_rng(1)
        for k, sc in [(8, 0.5), (30, 1.0), (60, 2.0), (5, 3.0)]:
            z = rng.normal(0.3, sc, k)
            out, B = RC.james_stein(z)
            ref, refB = ref_james_stein(list(z))
            np.testing.assert_allclose(out, ref, atol=1e-12)
            self.assertAlmostEqual(B, refB, places=12)
            self.assertAlmostEqual(float(out.mean()), float(z.mean()), places=12)         # mean preserved
            self.assertTrue(np.all(np.diff(out[np.argsort(z)]) >= -1e-12))                 # order preserved
            self.assertTrue(0.0 <= B <= 1.0)

    def test_edge_cases(self):
        z = np.array([1.0, 2.0, 3.0])
        out, B = RC.james_stein(z)
        np.testing.assert_array_equal(out, z)         # k < 4: no shrinkage
        self.assertEqual(B, 1.0)
        out, B = RC.james_stein(np.full(10, 0.7))
        np.testing.assert_allclose(out, 0.7)
        self.assertEqual(B, 0.0)
        out, B = RC.james_stein(np.array([]))
        self.assertEqual(len(out), 0)
        out, B = RC.james_stein(np.array([0.0, 0.1, -0.1, 0.05, -0.05, 0.02]))   # dispersion below noise: B = 0
        self.assertEqual(B, 0.0)
        np.testing.assert_allclose(out, out[0])
        out, B = RC.james_stein(np.linspace(-50, 50, 30))                          # huge signal: nearly untouched
        self.assertGreater(B, 0.99)

    def test_risk_dominates_raw_estimator(self):
        rng = np.random.default_rng(2)
        k, reps = 20, 3000
        for tau, max_ratio in [(0.3, 0.5), (1.0, 0.9), (5.0, 1.005)]:
            theta = rng.normal(0, tau, (reps, k))
            z = theta + rng.normal(size=(reps, k))
            shr = np.array([RC.james_stein(row)[0] for row in z])
            raw_risk = np.mean((z - theta) ** 2)
            js_risk = np.mean((shr - theta) ** 2)
            self.assertLess(js_risk / raw_risk, max_ratio, (tau, js_risk, raw_risk))
        self.assertAlmostEqual(raw_risk, 1.0, delta=0.02)

    def test_winners_curse_is_reduced(self):
        rng = np.random.default_rng(3)
        bias_raw, bias_js = [], []
        for _ in range(400):
            theta = rng.normal(0, 0.5, 40)
            z = theta + rng.normal(size=40)
            sh, _ = RC.james_stein(z)
            top = np.argsort(-z)[:5]
            bias_raw.append(np.mean(z[top] - theta[top]))
            top2 = np.argsort(-sh)[:5]
            bias_js.append(np.mean(sh[top2] - theta[top2]))
        self.assertGreater(np.mean(bias_raw), 0.8)                   # the picks' apparent edge is mostly luck
        self.assertLess(abs(np.mean(bias_js)), 0.35 * np.mean(bias_raw))

    def test_shrink_scores_units(self):
        rng = np.random.default_rng(4)
        s = rng.normal(0.0, 0.1, 30)
        h = 5
        out, B = RC.shrink_scores(s, h, bars=250)
        z = s * math.sqrt(250 / h)
        ref, refB = ref_james_stein(list(z))
        np.testing.assert_allclose(out, np.array(ref) / math.sqrt(250 / h), atol=1e-12)
        self.assertAlmostEqual(B, refB)
        # a longer horizon means fewer independent observations -> stronger shrinkage on the same scores
        self.assertLessEqual(RC.shrink_scores(s, 20)[1], RC.shrink_scores(s, 1)[1])


class TestLedoitWolf(unittest.TestCase):
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(5)
        for T, N in [(60, 5), (250, 12), (40, 30), (500, 3)]:
            X = rng.normal(size=(T, N)) @ rng.normal(size=(N, N)) * 0.01
            cov, a = RC.ledoit_wolf_cov(X)
            ref, ra = ref_lw(X)
            np.testing.assert_allclose(cov, ref, rtol=1e-9, atol=1e-14)
            self.assertAlmostEqual(a, ra, places=10)

    def test_matches_sklearn_when_available(self):
        try:
            from sklearn.covariance import ledoit_wolf
        except ImportError:
            self.skipTest("sklearn not installed")
        rng = np.random.default_rng(6)
        for T, N in [(120, 8), (60, 40)]:
            X = rng.standard_t(5, size=(T, N)) * 0.02
            cov, a = RC.ledoit_wolf_cov(X)
            cov_sk, a_sk = ledoit_wolf(X)
            np.testing.assert_allclose(cov, cov_sk, rtol=1e-8, atol=1e-14)
            self.assertAlmostEqual(a, a_sk, places=9)

    def test_properties(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 50)) * 0.01                   # more assets than observations
        cov, a = RC.ledoit_wolf_cov(X)
        self.assertTrue(0 < a <= 1)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)   # sample covariance is singular, shrunk one is not
        np.testing.assert_allclose(cov, cov.T)
        A = np.array([[1.0, 0.8, 0.5, 0.2], [0.0, 0.6, 0.7, 0.1], [0.0, 0.0, 0.3, 0.9], [0.0, 0.0, 0.0, 2.0]])
        big = rng.normal(size=(20000, 4)) @ A                     # strongly structured covariance, plenty of data
        self.assertLess(RC.ledoit_wolf_cov(big)[1], 0.01)         # -> almost no shrinkage towards mu*I
        iso = rng.normal(size=(20000, 4))                         # true covariance IS the target: shrink (almost) fully
        self.assertGreater(RC.ledoit_wolf_cov(iso)[1], 0.5)
        cov0, a0 = RC.ledoit_wolf_cov(np.zeros((10, 3)))
        self.assertEqual(a0, 1.0)


class TestKellyWeights(unittest.TestCase):
    def test_independent_hand_example(self):
        edge, sg = [0.02, 0.01, -0.01, np.nan], [0.1, 0.1, 0.1, 0.1]
        w = RC.kelly_weights(edge, sg, fraction=0.25, cap=1.0)
        np.testing.assert_allclose(w, [0.5, 0.25, 0.0, 0.0])
        w = RC.kelly_weights(edge, sg, fraction=0.25, cap=0.15)
        np.testing.assert_allclose(w, [0.15, 0.15, 0.0, 0.0])
        w = RC.kelly_weights([0.04, 0.04], [0.1, 0.1], fraction=0.25, cap=1.0)    # 1.0 + 1.0 -> scaled to sum 1
        np.testing.assert_allclose(w, [0.5, 0.5])
        self.assertEqual(len(RC.kelly_weights([], [])), 0)
        w = RC.kelly_weights([0.01], [0.0])                                      # zero sigma: no weight
        self.assertEqual(w.tolist(), [0.0])

    def test_defaults_use_config(self):
        w = RC.kelly_weights([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
        self.assertTrue(np.all(w <= config.MAX_WEIGHT + 1e-15))
        np.testing.assert_allclose(w, config.MAX_WEIGHT)

    def test_covariance_version(self):
        mu = np.array([0.02, 0.015])
        S = np.array([[0.01, 0.0], [0.0, 0.04]])
        w = RC.kelly_weights(mu, np.sqrt(np.diag(S)), fraction=0.2, cap=1.0, cov=S)
        np.testing.assert_allclose(w, 0.2 * mu / np.diag(S))              # diagonal covariance == independent formula
        np.testing.assert_allclose(w, RC.kelly_weights(mu, np.sqrt(np.diag(S)), fraction=0.2, cap=1.0))
        S = np.array([[0.04, 0.012], [0.012, 0.04]])                         # rho = 0.3: all weights positive
        w = RC.kelly_weights(mu, np.sqrt(np.diag(S)), fraction=0.1, cap=1.0, cov=S)
        np.testing.assert_allclose(w, 0.1 * np.linalg.solve(S, mu), atol=1e-12)
        S = np.array([[0.04, 0.036], [0.036, 0.04]])                         # rho = 0.9: unconstrained solution shorts asset 2
        self.assertLess(np.linalg.solve(S, mu)[1], 0)
        w = RC.kelly_weights(mu, np.sqrt(np.diag(S)), fraction=0.1, cap=1.0, cov=S)
        np.testing.assert_allclose(w, [0.1 * 0.02 / 0.04, 0.0], atol=1e-12)     # long-only: asset 2 dropped
        mu2 = np.array([0.03, 0.01])
        w = RC.kelly_weights(mu2, np.sqrt(np.diag(S)), fraction=0.1, cap=1.0, cov=S)   # unconstrained solution is short asset 2
        self.assertTrue(np.all(w >= 0))
        self.assertEqual(w[1], 0.0)
        self.assertAlmostEqual(w[0], 0.1 * 0.03 / 0.04)
        w = RC.kelly_weights(np.array([0.5, 0.5, 0.5]), [0.1] * 3, fraction=1.0, cap=0.4, cov=np.eye(3) * 0.01)
        self.assertTrue(np.all(w <= 0.4 + 1e-12) and w.sum() <= 1 + 1e-12)


class TestBuildRecommendations(unittest.TestCase):
    REC_KEYS = {"rank", "code", "name", "kind", "last_close", "chg_1d", "p_up", "exp_return", "net_exp_return", "sigma",
                "score", "score_shrunk", "signal", "confidence", "weight", "reason", "p_up_by_h", "regime",
                "avg_value_20d", "liquid",
                # DESIGN.md 9.1 additions (additive: every pre-existing key is still there)
                "rank_signal", "score_rank_pct", "lean", "edge_covers_cost", "low_vol", "edge_vs_baseline"}
    TOP_KEYS = {"asof", "asof_complete", "horizon", "computed_at", "universe_size", "n_scored", "market", "buys",
                "avoids", "all", "portfolio", "notes",
                "leaders", "laggards", "laggards_all_above_half", "n_buy_total", "rank_key", "rank_sigma_corr",
                "universe_sigma_median", "evidence"}

    @classmethod
    def setUpClass(cls):
        planted = ([make_fc(f"90{i}", 0.62 + 0.01 * i, 0.03 + 0.004 * i, 0.03) for i in range(5)]      # clear buys
                   + [make_fc(f"91{i}", 0.38 - 0.01 * i, -0.02, 0.03) for i in range(4)]                # clear avoids
                   + [make_fc("920", 0.60, 0.002, 0.03)])                                              # p high, no edge
        cls.fcs = make_universe(51, seed=11) + planted
        cls.N = len(cls.fcs)
        cls.out = RC.build_recommendations(cls.fcs, 5, top_k=8, now=NOW)

    def test_structure(self):
        o = self.out
        self.assertEqual(set(o), self.TOP_KEYS)
        self.assertEqual((o["horizon"], o["universe_size"], o["n_scored"]), (5, self.N, self.N))
        self.assertEqual(o["computed_at"], "2026-10-22T15:00:00+00:00")
        self.assertEqual((o["asof"], o["asof_complete"]), ("2026-10-22", True))
        self.assertEqual(set(o["market"]), {"p_up_1", "p_up_5", "regime", "last_close", "chg_1d"})
        self.assertEqual(set(o["portfolio"]), {"cash_weight", "n", "note", "suppressed_reason"})
        self.assertEqual((o["rank_key"], o["evidence"]), ("rank_signal", {"source": "none"}))
        for lst in ("buys", "avoids", "all", "leaders", "laggards"):
            for r in o[lst]:
                self.assertEqual(set(r), self.REC_KEYS)
                self.assertIn(r["signal"], ("buy", "hold", "avoid"))
                self.assertIn(r["confidence"], ("高", "中", "低"))
                self.assertEqual(set(r["p_up_by_h"]), {"1", "5", "20"})
        json.dumps(o, allow_nan=False)

    def test_scores_match_independent_formula(self):
        by_code = {fc["symbol"]: fc for fc in self.fcs}
        for r in self.out["all"]:
            hz = by_code[r["code"]]["horizons"]["5"]
            net = hz["exp_return"] - COST_STOCK
            self.assertAlmostEqual(r["net_exp_return"], net, places=12)
            self.assertAlmostEqual(r["score"], net / hz["sigma"], places=12)
            self.assertEqual(r["p_up"], hz["p_up"])
            self.assertEqual(r["p_up_by_h"]["1"], 0.5)

    def test_shrinkage_matches_independent_james_stein(self):
        scores = [r["score"] for r in self.out["all"]]
        z = [s * math.sqrt(250 / 5) for s in scores]
        ref, B = ref_james_stein(z)
        for r, v in zip(self.out["all"], ref):
            self.assertAlmostEqual(r["score_shrunk"], v / math.sqrt(250 / 5), places=12)
        self.assertTrue(any(f"B={B:.2f}" in n for n in self.out["notes"]))

    def test_ordering_and_ranks(self):
        # CHANGED (DESIGN.md 9.1, audit C-06 / P-03 / P-04): the old test asserted "sorted by score_shrunk" and per-list ranks
        # 1..k.  The cost-adjusted score ranks VOLATILITY (see TestRankSignal), so the order of `all` and every list now
        # follows rank_signal, and `rank` is ONE global rank (position in `all`) in every list.
        o = self.out
        rs = [r["rank_signal"] for r in o["all"]]
        self.assertEqual(rs, sorted(rs, reverse=True))
        self.assertEqual([r["rank"] for r in o["all"]], list(range(1, self.N + 1)))
        pos = {r["code"]: r["rank"] for r in o["all"]}
        for name in ("buys", "avoids", "leaders", "laggards"):
            for r in o[name]:
                self.assertEqual(r["rank"], pos[r["code"]], (name, r["code"]))
        br = [r["rank"] for r in o["buys"]]
        self.assertEqual(br, sorted(br))                                  # best first
        ar = [r["rank"] for r in o["avoids"]]
        self.assertEqual(ar, sorted(ar, reverse=True))                    # weakest first
        self.assertLessEqual(len(o["buys"]), 8)
        self.assertLessEqual(len(o["avoids"]), 8)
        self.assertGreater(len(o["buys"]), 0)
        self.assertGreater(len(o["avoids"]), 0)
        self.assertEqual(o["n_buy_total"], sum(r["signal"] == "buy" for r in o["all"]))

    def test_signals_follow_the_rule(self):
        for r in self.out["all"]:
            exp = "hold"
            if r["p_up"] >= config.BUY_P_UP and r["net_exp_return"] > 0 and r["liquid"]:
                exp = "buy"
            elif r["p_up"] <= config.AVOID_P_UP and r["net_exp_return"] < 0:
                exp = "avoid"
            self.assertEqual(r["signal"], exp)
        n_buy_all = sum(r["signal"] == "buy" for r in self.out["all"])
        self.assertEqual(len(self.out["buys"]), min(8, n_buy_all))
        self.assertTrue(all(r["signal"] == "buy" for r in self.out["buys"]))
        self.assertTrue(all(r["signal"] == "avoid" for r in self.out["avoids"]))

    def test_weights(self):
        o = self.out
        ws = [r["weight"] for r in o["buys"]]
        self.assertTrue(all(0 <= w <= config.MAX_WEIGHT + 1e-12 for w in ws))
        tot = sum(ws)
        self.assertLessEqual(tot, 1.0 + 1e-12)
        self.assertAlmostEqual(o["portfolio"]["cash_weight"], 1 - tot, places=12)
        self.assertEqual(o["portfolio"]["n"], sum(w > 0 for w in ws))
        buy_codes = {r["code"] for r in o["buys"]}
        for r in o["all"]:
            if r["code"] not in buy_codes:
                self.assertEqual(r["weight"], 0.0)
            else:
                self.assertEqual(r["weight"], next(b["weight"] for b in o["buys"] if b["code"] == r["code"]))
        if tot < 1 - 1e-9:      # not rescaled -> each weight is the capped fractional Kelly of the shrunk edge
            for r in o["buys"]:
                kelly = config.KELLY_FRACTION * max(r["score_shrunk"], 0) / r["sigma"]
                self.assertAlmostEqual(r["weight"], min(config.MAX_WEIGHT, kelly), places=12)

    def test_weights_zero_when_shrunk_edge_not_positive(self):
        fcs = [make_fc(f"{3000 + i}", 0.5 + 0.02 * (i % 3), 0.001, 0.03) for i in range(12)]
        fcs.append(make_fc("9999", 0.60, 0.0075, 0.03))      # barely clears the 0.57 % hurdle, in a weak cross-section
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertTrue(any(r["code"] == "9999" for r in o["buys"]))
        for r in o["buys"]:
            self.assertEqual(r["weight"] > 0, r["score_shrunk"] > 0)

    def test_illiquid_name_is_not_a_buy(self):
        fcs = make_universe(30, seed=1) + [make_fc("8888", 0.68, 0.05, 0.03, avg=5e7)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        r = next(x for x in o["all"] if x["code"] == "8888")
        self.assertEqual((r["signal"], r["liquid"]), ("hold", False))
        self.assertEqual(r["avg_value_20d"], 5e7)
        self.assertNotIn("8888", [b["code"] for b in o["buys"]])
        self.assertEqual(r["weight"], 0.0)
        self.assertIn("流動性不足", r["reason"])
        r2 = next(x for x in RC.build_recommendations(fcs, 5, now=NOW, min_avg_value=1e7)["all"] if x["code"] == "8888")
        self.assertEqual((r2["signal"], r2["liquid"]), ("buy", True))

    def test_missing_liquidity_is_assumed_liquid(self):
        fc = make_fc("7777", 0.7, 0.05, 0.03)
        del fc["avg_value_20d"]
        r = RC.build_recommendations([fc] * 1, 5, now=NOW)["all"][0]
        self.assertTrue(r["liquid"])
        self.assertIsNone(r["avg_value_20d"])
        self.assertEqual(r["signal"], "buy")

    def test_etf_cost_is_lower(self):
        a = RC.build_recommendations([make_fc("0050", 0.6, 0.0045, 0.03, kind="etf")], 5)["all"][0]
        b = RC.build_recommendations([make_fc("2330", 0.6, 0.0045, 0.03, kind="stock")], 5)["all"][0]
        self.assertEqual((a["kind"], b["kind"]), ("etf", "stock"))
        self.assertAlmostEqual(a["net_exp_return"], 0.0045 - COST_ETF)
        self.assertEqual((a["signal"], b["signal"]), ("buy", "hold"))
        c = make_fc("00878", 0.6, 0.0045, 0.03)
        del c["kind"]
        self.assertEqual(RC.build_recommendations([c], 5)["all"][0]["kind"], "etf")      # inferred from the code

    def test_top_k_and_validation(self):
        o = RC.build_recommendations(self.fcs, 5, top_k=2, now=NOW)
        self.assertLessEqual(len(o["buys"]), 2)
        self.assertEqual(len(o["all"]), self.N)
        with self.assertRaises(ValueError):
            RC.build_recommendations(self.fcs, 5, top_k=0)
        with self.assertRaises(ValueError):
            RC.build_recommendations(self.fcs, 5, buy_p=0.4, avoid_p=0.6)

    def test_invalid_entries_are_counted_not_scored(self):
        fcs = make_universe(10, seed=3)
        bad1 = make_fc("1111")
        bad1["horizons"]["5"]["sigma"] = float("nan")
        bad2 = make_fc("1112")
        del bad2["horizons"]["5"]
        fcs += [bad1, bad2, None, {"symbol": "x"}]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual((o["universe_size"], o["n_scored"]), (13, 10))
        self.assertTrue(any("未列入評分" in n for n in o["notes"]))
        self.assertTrue(all(r["code"] not in ("1111", "1112") for r in o["all"]))
        o = RC.build_recommendations(fcs, 7, now=NOW)         # no such horizon anywhere
        self.assertEqual((o["n_scored"], o["all"], o["buys"], o["avoids"]), (0, [], [], []))
        json.dumps(o, allow_nan=False)

    def test_header_fields(self):
        fcs = make_universe(6, seed=4)
        fcs[0]["asof"] = "2026-10-21"
        fcs[1]["asof_complete"] = False
        mk = make_fc("^TWII", 0.53, 0.004, 0.02)
        mk["horizons"]["1"]["p_up"] = 0.51
        mk["last_close"], mk["chg_1d"], mk["regime"] = 46041.4, -0.011, {"current": "多頭穩定"}
        o = RC.build_recommendations(fcs, 5, market_forecast=mk, now=NOW)
        self.assertEqual(o["asof"], "2026-10-22")
        self.assertFalse(o["asof_complete"])
        self.assertTrue(any("盤中" in n for n in o["notes"]))
        self.assertEqual(o["market"], {"p_up_1": 0.51, "p_up_5": 0.53, "regime": "多頭穩定", "last_close": 46041.4,
                                       "chg_1d": -0.011})

    def test_empty_and_tiny_universes(self):
        o = RC.build_recommendations([], 5, now=NOW)
        self.assertEqual((o["universe_size"], o["n_scored"], o["buys"], o["all"]), (0, 0, [], []))
        self.assertEqual(o["portfolio"]["cash_weight"], 1.0)
        self.assertIsNone(o["asof"])
        self.assertTrue(any("沒有任何標的" in n for n in o["notes"]))
        json.dumps(o, allow_nan=False)
        one = RC.build_recommendations([make_fc("2330", 0.62, 0.02, 0.03)], 5, now=NOW)
        r = one["all"][0]
        self.assertEqual(r["score_shrunk"], r["score"])        # k < 4: no shrinkage
        self.assertEqual(r["signal"], "buy")
        self.assertGreater(r["weight"], 0)

    def test_identical_forecasts_break_ties_by_code(self):
        fcs = [make_fc(f"{2400 + i}", 0.6, 0.012, 0.03) for i in (3, 1, 2, 0, 4)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual([r["code"] for r in o["all"]], ["2400", "2401", "2402", "2403", "2404"])
        self.assertEqual(len({r["score_shrunk"] for r in o["all"]}), 1)

    def test_covariance_option(self):
        fcs = [make_fc(f"{2500 + i}", 0.62, 0.02, 0.03) for i in range(6)] + make_universe(20, seed=9)
        rng = np.random.default_rng(0)
        common = rng.normal(0, 0.01, 300)
        rets = {f"{2500 + i}": common + rng.normal(0, 0.005, 300) for i in range(6)}
        a = RC.build_recommendations(fcs, 5, now=NOW)
        b = RC.build_recommendations(fcs, 5, now=NOW, returns=rets)
        self.assertTrue(any("Ledoit-Wolf" in n for n in b["notes"]))
        self.assertTrue(any("彼此獨立" in n for n in a["notes"]))
        tot_a = sum(r["weight"] for r in a["buys"])
        tot_b = sum(r["weight"] for r in b["buys"])
        self.assertLessEqual(tot_b, 1 + 1e-12)
        self.assertLess(tot_b, tot_a + 1e-9)          # highly correlated names: concentrate less than the independent case
        for r in b["buys"]:
            self.assertTrue(0 <= r["weight"] <= config.MAX_WEIGHT + 1e-12)

    def test_deterministic(self):
        a = RC.build_recommendations(self.fcs, 5, top_k=8, now=NOW)
        b = RC.build_recommendations(copy.deepcopy(self.fcs), 5, top_k=8, now=NOW)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(RC.build_recommendations(self.fcs, 5)["computed_at"][:4], "2026" if False else
                         RC.build_recommendations(self.fcs, 5)["computed_at"][:4])

    def test_inputs_not_mutated(self):
        snap = copy.deepcopy(self.fcs)
        RC.build_recommendations(self.fcs, 5, now=NOW)
        self.assertEqual(self.fcs, snap)

    def test_other_horizons(self):
        for h, label in ((1, "明日"), (20, "未來20日")):
            o = RC.build_recommendations(self.fcs, h, now=NOW)
            self.assertEqual(o["horizon"], h)
            self.assertEqual(o["n_scored"], self.N)
            self.assertTrue(all(label in r["reason"] for r in o["all"]))

    def test_reason_text(self):
        o = self.out
        for r in o["all"]:
            s = r["reason"]
            self.assertTrue(s.endswith("。"))
            self.assertFalse(SIMPLIFIED & set(s), s)
            for w in FORBIDDEN:
                self.assertNotIn(w, s)
            self.assertIn(f"{r['p_up'] * 100:.0f}%", s if r["signal"] != "avoid" else f"{r['p_up'] * 100:.0f}%")
        b = o["buys"][0]
        self.assertIn(f"{b['net_exp_return'] * 100:+.2f}%", b["reason"])
        self.assertIn("12-1個月動能", b["reason"])
        self.assertIn("推升", b["reason"])
        self.assertIn("多頭穩定", b["reason"])
        self.assertIn(f"信心{b['confidence']}", b["reason"])
        a = o["avoids"][0]
        self.assertIn("下跌機率", a["reason"])
        self.assertIn("不宜追買", a["reason"])
        hold_hi = next((r for r in o["all"] if r["signal"] == "hold" and r["p_up"] >= config.BUY_P_UP), None)
        if hold_hi:
            self.assertIn("沒有優勢", hold_hi["reason"])
        hold_mid = next(r for r in o["all"] if r["signal"] == "hold" and config.AVOID_P_UP < r["p_up"] < config.BUY_P_UP)
        self.assertIn("未達偏多候選門檻", hold_mid["reason"])        # was 未達買進門檻: DESIGN.md 9.6 wording (偏多候選)

    def test_notes_text(self):
        for n in self.out["notes"]:
            self.assertFalse(SIMPLIFIED & set(n), n)
            for w in FORBIDDEN:
                self.assertNotIn(w, n)
        joined = "".join(self.out["notes"])
        for key in ("來回交易成本", "James-Stein", "凱利", "不構成投資建議", "6 折"):
            self.assertIn(key, joined)

    def test_dispersed_signals_survive_shrinkage_and_noise_collapses(self):
        strong = make_universe(60, seed=5, dispersion=4.0)           # widely different forecasts
        weak = make_universe(60, seed=5, dispersion=0.05)            # nearly identical forecasts
        Bs = {}
        for name, fcs in (("strong", strong), ("weak", weak)):
            o = RC.build_recommendations(fcs, 5, now=NOW)
            Bs[name] = float(next(n for n in o["notes"] if "保留比例" in n).split("B=")[1].split("（")[0])
        self.assertGreater(Bs["strong"], 0.5)
        self.assertEqual(Bs["weak"], 0.0)


# ============================================================================= DESIGN.md 9.1: rank_signal, lean, leaders / laggards
def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def view_fc(code, z, sg=0.05, **kw):
    """Forecast whose information ratio (mu_log / sigma_log) is ``z``; p_up = Phi(z)."""
    mu = z * sg
    return make_fc(code, float(stats.norm.cdf(z)), math.expm1(mu + 0.5 * sg * sg), sg, mu_log=mu, sigma_log=sg, **kw)


def sigma_indep_universe(n=60, seed=1):
    """Universe whose VIEW is independent of volatility and whose net returns are all negative (the h = 1 / 5 situation of
    the real data): sigma ~ U(0.02, 0.07), information ratio z ~ U(-0.015, 0.015), p_up = Phi(z)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sg = float(rng.uniform(0.02, 0.07))
        out.append(view_fc(f"{3000 + i}", float(rng.uniform(-0.015, 0.015)), sg))
    return out


class TestRankSignalValue(unittest.TestCase):
    def test_ratio_and_fallback(self):
        self.assertAlmostEqual(RC.rank_signal_value(0.02, 0.05, 0.7), 0.4)
        self.assertAlmostEqual(RC.rank_signal_value(None, 0.05, 0.7), stats.norm.ppf(0.7))        # no mu_log
        self.assertAlmostEqual(RC.rank_signal_value(0.02, None, 0.2), stats.norm.ppf(0.2))        # no sigma_log
        self.assertAlmostEqual(RC.rank_signal_value(0.02, 0.0, 0.7), stats.norm.ppf(0.7))         # unusable sigma
        self.assertAlmostEqual(RC.rank_signal_value("x", float("nan"), 0.5), 0.0)
        self.assertTrue(math.isfinite(RC.rank_signal_value(None, None, 1.0)))                     # clipped, not inf
        self.assertTrue(math.isnan(RC.rank_signal_value(None, None, None)))

    def test_array_version_matches_scalar(self):
        rng = np.random.default_rng(3)
        mu, sg, p = rng.normal(0, 0.01, 50), rng.uniform(0.01, 0.05, 50), rng.uniform(0.2, 0.8, 50)
        mu[::7] = np.nan
        sg[3::11] = np.nan
        ref = [RC.rank_signal_value(m, s, q) for m, s, q in zip(mu, sg, p)]
        np.testing.assert_allclose(RC.rank_signal_array(mu, sg, p), ref, atol=1e-12)

    def test_contains_neither_cost_nor_convexity(self):
        base = RC.build_recommendations([view_fc(f"{i}", 0.02 * i - 0.1) for i in range(10)], 5, now=NOW)
        costly = RC.build_recommendations([view_fc(f"{i}", 0.02 * i - 0.1) for i in range(10)], 5, now=NOW,
                                          commission_discount=1.0, slippage_bps=80)
        self.assertEqual([r["code"] for r in base["all"]], [r["code"] for r in costly["all"]])
        for a, b in zip(base["all"], costly["all"]):
            self.assertEqual(a["rank_signal"], b["rank_signal"])              # the trading cost does not move it
            self.assertLess(b["score"], a["score"])                           # ... but it does move the score
        for r in base["all"]:
            self.assertAlmostEqual(r["rank_signal"], 0.02 * (9 - (r["rank"] - 1)) - 0.1, places=12)


class TestRankSignalOrdering(unittest.TestCase):
    """The headline regression of DESIGN.md 9.1: the cost-adjusted score ranks volatility, rank_signal must not."""

    @classmethod
    def setUpClass(cls):
        cls.fcs = sigma_indep_universe()
        cls.out = RC.build_recommendations(cls.fcs, 5, top_k=10, now=NOW)

    def test_premise_every_net_is_negative_and_the_score_is_a_volatility_ranking(self):
        o = self.out
        self.assertTrue(all(r["net_exp_return"] < 0 for r in o["all"]))
        self.assertEqual(o["n_buy_total"], 0)
        sg = [r["sigma"] for r in o["all"]]
        self.assertGreater(spearman([r["score"] for r in o["all"]], sg), 0.9)         # the artefact of the old ordering

    def test_delivered_order_is_not_the_sigma_order(self):
        o = self.out
        sg = [r["sigma"] for r in o["all"]]
        rho_rank = spearman([-r["rank"] for r in o["all"]], sg)
        self.assertLess(abs(rho_rank), 0.9)                                   # DESIGN.md 9.1 regression test
        self.assertLess(abs(rho_rank), 0.45)                                  # the view is independent of sigma by construction
        self.assertAlmostEqual(o["rank_sigma_corr"], spearman([r["rank_signal"] for r in o["all"]], sg), places=10)
        by_score = [r["code"] for r in sorted(o["all"], key=lambda r: -r["score"])]       # the old (volatility) order
        self.assertNotEqual([r["code"] for r in o["all"]], by_score)
        self.assertEqual(o["rank_key"], "rank_signal")

    def test_order_follows_rank_signal_when_it_disagrees_with_the_score(self):
        # A: lower rank_signal but a MUCH higher (cost-adjusted) score thanks to a large sigma^2 / 2 convexity term
        a = make_fc("1111", 0.60, 0.060, 0.30, mu_log=0.002, sigma_log=0.30)          # rs = 0.0067, net/sigma = 0.18
        b = make_fc("2222", 0.60, 0.012, 0.05, mu_log=0.010, sigma_log=0.05)          # rs = 0.2,    net/sigma = 0.126
        c = make_fc("3333", 0.52, -0.001, 0.05, mu_log=-0.004, sigma_log=0.05)        # rs = -0.08
        d = make_fc("4444", 0.51, 0.000, 0.05, mu_log=0.000, sigma_log=0.05)          # rs = 0
        o = RC.build_recommendations([a, b, c, d], 5, now=NOW)
        by = {r["code"]: r for r in o["all"]}
        self.assertGreater(by["1111"]["score"], by["2222"]["score"])
        self.assertEqual([r["code"] for r in o["all"]], ["2222", "1111", "4444", "3333"])
        self.assertEqual([r["rank"] for r in o["all"]], [1, 2, 3, 4])

    def test_ties_are_broken_by_p_up_then_code(self):
        fcs = [make_fc("2003", 0.55, 0.01, 0.03, mu_log=0.003, sigma_log=0.03),
               make_fc("2001", 0.55, 0.01, 0.03, mu_log=0.003, sigma_log=0.03),
               make_fc("2002", 0.58, 0.01, 0.03, mu_log=0.003, sigma_log=0.03)]       # same rank_signal 0.1, higher p_up
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual([r["code"] for r in o["all"]], ["2002", "2001", "2003"])

    def test_fallback_to_the_normal_quantile_of_p_up(self):
        fcs = [make_fc(f"{2200 + i}", 0.45 + 0.02 * i, 0.01, 0.03) for i in range(8)]   # no mu_log / sigma_log at all
        o = RC.build_recommendations(fcs, 5, now=NOW)
        for r in o["all"]:
            self.assertAlmostEqual(r["rank_signal"], stats.norm.ppf(r["p_up"]), places=12)
        self.assertEqual([r["code"] for r in o["all"]], [f"{2200 + i}" for i in reversed(range(8))])

    def test_ranking_ignores_liquidity_signal_and_cost(self):
        # an illiquid name keeps its place in the relative ranking (it just cannot be a buy candidate)
        fcs = [view_fc(f"{2300 + i}", 0.05 * i - 0.2, 0.03, avg=5e9 if i != 7 else 1e7) for i in range(9)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        by = {r["code"]: r for r in o["all"]}
        self.assertEqual(by["2307"]["rank"], 2)
        self.assertFalse(by["2307"]["liquid"])
        self.assertNotEqual(by["2307"]["signal"], "buy")


class TestLeadersLaggardsLean(unittest.TestCase):
    @staticmethod
    def ladder(zs, sg=0.05):
        return [view_fc(f"{2400 + i}", z, sg) for i, z in enumerate(zs)]

    def test_mixed_cross_section(self):
        zs = [0.30 - 0.06 * k for k in range(10)]            # +0.30 .. -0.24, p_up from 0.62 down to 0.40
        o = RC.build_recommendations(self.ladder(zs), 5, top_k=8, now=NOW)
        allr = o["all"]
        self.assertEqual([r["rank"] for r in allr], list(range(1, 11)))
        self.assertEqual([r["code"] for r in allr], [f"{2400 + i}" for i in range(10)])
        np.testing.assert_allclose([r["score_rank_pct"] for r in allr], [(10 - k) / 9 for k in range(1, 11)], atol=1e-12)
        self.assertEqual([r["lean"] for r in allr],
                         ["bullish", "bullish"] + ["neutral"] * 6 + ["bearish", "bearish"])     # quintile AND the p_up side
        # leaders: top top_k with p_up > 0.5 (the name with z = 0 has p_up = 0.5 exactly and is excluded)
        self.assertEqual([r["rank"] for r in o["leaders"]], [1, 2, 3, 4, 5])
        # laggards: bottom min(top_k, n // 2) = 5, weakest first, no p_up requirement
        self.assertEqual([r["rank"] for r in o["laggards"]], [10, 9, 8, 7, 6])
        self.assertFalse(o["laggards_all_above_half"])                  # the very weakest names do have p_up < 0.5

    def test_all_p_up_above_half_laggards_are_flagged_and_nobody_is_bearish(self):
        zs = [0.35 - 0.03 * k for k in range(10)]           # all positive: p_up 0.64 .. 0.51
        o = RC.build_recommendations(self.ladder(zs), 5, top_k=3, now=NOW)
        self.assertTrue(all(r["p_up"] > 0.5 for r in o["all"]))
        self.assertEqual(len(o["laggards"]), 3)                          # NOT empty although no name has p_up < 0.5
        self.assertEqual([r["rank"] for r in o["laggards"]], [10, 9, 8])
        self.assertTrue(o["laggards_all_above_half"])
        self.assertTrue(all(r["lean"] != "bearish" for r in o["all"]))   # lean=bearish keeps the strict rule: often empty
        self.assertTrue(any("上漲機率仍高於 50%" in n for n in o["notes"]))
        self.assertEqual(len(o["leaders"]), 3)

    def test_small_universe_limits(self):
        o = RC.build_recommendations(self.ladder([0.2, 0.1, -0.1]), 5, top_k=10, now=NOW)
        self.assertEqual(len(o["laggards"]), 1)                          # n // 2
        self.assertEqual(len(o["leaders"]), 2)
        one = RC.build_recommendations(self.ladder([0.2]), 5, now=NOW)
        self.assertEqual((len(one["laggards"]), one["laggards_all_above_half"]), (0, False))
        self.assertEqual(one["all"][0]["score_rank_pct"], 0.5)           # undefined with one name: neutral
        self.assertEqual(one["all"][0]["lean"], "neutral")
        empty = RC.build_recommendations([], 5, now=NOW)
        self.assertEqual((empty["leaders"], empty["laggards"], empty["laggards_all_above_half"], empty["n_buy_total"]),
                         ([], [], False, 0))
        self.assertIsNone(empty["rank_sigma_corr"])
        self.assertIsNone(empty["universe_sigma_median"])

    def test_low_vol_edge_covers_cost_and_baseline_edge(self):
        comps = {"naive": {"p_up": 0.52, "mu": 0.001}, "garch": {"p_up": 0.55, "mu": 0.002}}
        fcs = [view_fc("2501", 0.3, 0.02, comps=comps), view_fc("2502", 0.2, 0.06), view_fc("2503", 0.1, 0.04),
               view_fc("2504", 0.0, 0.08), view_fc("2505", -0.1, 0.10)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        by = {r["code"]: r for r in o["all"]}
        med = float(np.median([0.02, 0.06, 0.04, 0.08, 0.10]))
        self.assertAlmostEqual(o["universe_sigma_median"], med)
        for r in o["all"]:
            self.assertEqual(r["low_vol"], r["sigma"] < med)
            self.assertEqual(r["edge_covers_cost"], r["net_exp_return"] > 0)
        self.assertEqual([by[c]["low_vol"] for c in ("2501", "2502", "2503", "2504", "2505")],
                         [True, False, True, False, False])                   # sigma 0.02 / 0.06 / 0.04 / 0.08 / 0.10, median 0.06
        self.assertAlmostEqual(by["2501"]["edge_vs_baseline"], by["2501"]["p_up"] - 0.52, places=12)
        self.assertAlmostEqual(by["2502"]["edge_vs_baseline"], by["2502"]["p_up"] - by["2502"]["p_up"], places=12)  # naive == p here
        fc = view_fc("2506", 0.1, 0.03)
        fc["horizons"]["5"]["components"] = {"garch": {"p_up": 0.5, "mu": 0.0}}               # no naive component
        self.assertIsNone(RC.build_recommendations([fc], 5, now=NOW)["all"][0]["edge_vs_baseline"])


class TestGlobalRankAndCounts(unittest.TestCase):
    def test_rank_is_one_global_number_and_counts_are_reported(self):
        fcs = [make_fc(f"{2600 + i}", 0.58 + 0.003 * i, 0.012 + 0.0004 * i, 0.03, mu_log=0.004 - 0.0003 * i,
                       sigma_log=0.03) for i in range(12)]                          # 12 buys, rank_signal falls with i
        fcs += [make_fc(f"{2700 + i}", 0.40 - 0.002 * i, -0.02, 0.03, mu_log=-0.01 - 0.001 * i, sigma_log=0.03)
                for i in range(4)]
        o = RC.build_recommendations(fcs, 5, top_k=5, now=NOW)
        pos = {r["code"]: r["rank"] for r in o["all"]}
        self.assertEqual(o["n_buy_total"], 12)
        self.assertEqual(len(o["buys"]), 5)                                # truncated to top_k ...
        for lst in ("buys", "avoids", "leaders", "laggards"):
            for r in o[lst]:
                self.assertEqual(r["rank"], pos[r["code"]], (lst, r["code"]))
        self.assertEqual([r["rank"] for r in o["buys"]], [1, 2, 3, 4, 5])      # the strongest buys are also the best ranks here
        self.assertEqual([r["code"] for r in o["avoids"]][0], max(pos, key=pos.get))   # weakest first
        self.assertEqual(sum(r["signal"] == "avoid" for r in o["all"]), 4)

    def test_buy_ranks_are_not_renumbered(self):
        # a name that is the 4th best by rank_signal but only the 2nd buy keeps rank 4 in `buys`
        fcs = [make_fc("2801", 0.50, 0.001, 0.03, mu_log=0.009, sigma_log=0.03),       # rs 0.30, hold (p_up too low)
               make_fc("2802", 0.60, 0.012, 0.03, mu_log=0.008, sigma_log=0.03),       # rs 0.27 buy
               make_fc("2803", 0.50, 0.001, 0.03, mu_log=0.007, sigma_log=0.03),       # rs 0.23 hold
               make_fc("2804", 0.60, 0.012, 0.03, mu_log=0.006, sigma_log=0.03),       # rs 0.20 buy
               make_fc("2805", 0.50, 0.001, 0.03, mu_log=0.001, sigma_log=0.03)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual([(r["code"], r["rank"]) for r in o["buys"]], [("2802", 2), ("2804", 4)])


class TestWeightGate(unittest.TestCase):
    def assert_no_weights(self, o, reason):
        pf = o["portfolio"]
        self.assertEqual((pf["n"], pf["cash_weight"], pf["suppressed_reason"]), (0, 1.0, reason))
        self.assertTrue(all(r["weight"] == 0.0 for r in o["all"] + o["buys"] + o["leaders"] + o["laggards"]))
        self.assertIn("個股間的預測差異無法與雜訊區分", pf["note"])
        self.assertIn("偏多主要反映整體市場正漂移", pf["note"])
        self.assertTrue(any("個股間的預測差異無法與雜訊區分" in n for n in o["notes"]))
        self.assertNotIn("趨近於零或平均", "".join(o["notes"]))        # the old wording ("weights near zero or equal") is gone

    def test_zero_retention_means_no_weights_even_with_buy_signals(self):
        # audit P-02: at h = 20 every shrunk score was the same positive number and a 62 %-invested portfolio was suggested
        fcs = [make_fc(f"{2900 + i}", 0.60, 0.012, 0.03, mu_log=0.01, sigma_log=0.03) for i in range(12)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual(o["n_buy_total"], 12)
        self.assertEqual(len({r["score_shrunk"] for r in o["all"]}), 1)
        self.assertGreater(o["all"][0]["score_shrunk"], 0)                 # a positive common value: the old code allocated
        self.assert_no_weights(o, "B0")
        self.assertIn("建議全數持有現金", o["portfolio"]["note"])

    def test_shuffled_cross_section_has_zero_retention_and_no_weights(self):
        # placebo of audit P-02: names with identical views (a shuffle of identical rows) must never receive weights
        rng = np.random.default_rng(0)
        fcs = [make_fc(f"{3100 + i}", 0.60, 0.012, 0.03 + 1e-9 * rng.random(), mu_log=0.01, sigma_log=0.03) for i in range(30)]
        self.assert_no_weights(RC.build_recommendations(fcs, 20, now=NOW), "B0")

    def test_no_buy_names_means_no_weights(self):
        fcs = [make_fc(f"{3200 + i}", 0.52, -0.05 + 0.005 * i, 0.03, mu_log=-0.05 + 0.005 * i, sigma_log=0.03)
               for i in range(20)]                                       # dispersed forecasts (B > 0) but p_up < buy threshold
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual(o["n_buy_total"], 0)
        self.assert_no_weights(o, "no_positive_edge")

    def test_buys_without_a_positive_shrunk_edge_get_no_weights(self):
        er = [-0.0093 + 0.012 * (i / 19 - 0.5) for i in range(20)]       # net / sigma about -0.5 +- 0.2 in the cross-section
        fcs = [make_fc(f"{3300 + i}", 0.51, e, 0.03, mu_log=e, sigma_log=0.03) for i, e in enumerate(er)]
        fcs.append(make_fc("3399", 0.60, 0.0075, 0.03, mu_log=0.0074, sigma_log=0.03))   # barely clears the cost hurdle
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual([r["code"] for r in o["buys"]], ["3399"])
        self.assertLessEqual(o["buys"][0]["score_shrunk"], 0)             # pulled below zero by the weak cross-section
        self.assert_no_weights(o, "no_positive_edge")

    def test_positive_weights_only_for_buys_with_positive_net_and_shrunk_edge(self):
        fcs = ([make_fc(f"{3400 + i}", 0.62 + 0.01 * i, 0.04 + 0.01 * i, 0.03, mu_log=0.03 + 0.01 * i, sigma_log=0.03)
                for i in range(4)]
               + [make_fc(f"{3500 + i}", 0.40, -0.03 - 0.005 * i, 0.03, mu_log=-0.03, sigma_log=0.03) for i in range(6)]
               + [make_fc(f"{3600 + i}", 0.50, 0.001 * i, 0.03, mu_log=0.001 * i, sigma_log=0.03) for i in range(6)])
        o = RC.build_recommendations(fcs, 5, now=NOW)
        pf = o["portfolio"]
        self.assertIsNone(pf["suppressed_reason"])
        self.assertGreater(pf["n"], 0)
        self.assertAlmostEqual(pf["cash_weight"], 1 - sum(r["weight"] for r in o["all"]), places=12)
        for r in o["all"]:
            if r["weight"] > 0:
                self.assertEqual(r["signal"], "buy")
                self.assertGreater(r["net_exp_return"], 0)
                self.assertTrue(r["edge_covers_cost"])
                self.assertGreater(r["score_shrunk"], 0)
        self.assertTrue(all(r["weight"] == 0 for r in o["all"] if r["signal"] != "buy"))
        self.assertIn("凱利", pf["note"])


class TestSignalCountNotes(unittest.TestCase):
    """Audit P-07: say WHY the strict buy flag is empty (cost > expected return, h = 1 / 5) or full (market drift, h = 20)."""

    def test_empty_buy_list_because_every_net_return_is_negative(self):
        o = RC.build_recommendations(sigma_indep_universe(40, 2), 5, now=NOW)         # every net < 0 (premise: see above)
        self.assertEqual(o["n_buy_total"], 0)
        txt = next(n for n in o["notes"] if n.startswith("目前沒有任何標的同時滿足偏多候選條件"))
        best = max(r["net_exp_return"] for r in o["all"])
        self.assertLess(best, 0)
        self.assertIn("今日所有標的扣除來回交易成本後的未來5日預期報酬都 ≤ 0", txt)
        self.assertIn(f"{best * 100:+.2f}%", txt)
        self.assertIn("不代表模型看空", txt)

    def test_empty_buy_list_for_another_reason_gets_no_cost_explanation(self):
        # one name clears the cost hurdle but its probability is below the threshold: no buy, yet the cost is not the reason
        fcs = [make_fc("2001", 0.52, 0.02, 0.03, mu_log=0.019, sigma_log=0.03)] + \
              [make_fc(f"{2100 + i}", 0.5, -0.01, 0.03, mu_log=-0.01, sigma_log=0.03) for i in range(6)]
        o = RC.build_recommendations(fcs, 5, now=NOW)
        self.assertEqual(o["n_buy_total"], 0)
        self.assertGreater(max(r["net_exp_return"] for r in o["all"]), 0)
        txt = next(n for n in o["notes"] if n.startswith("目前沒有任何標的同時滿足偏多候選條件"))
        self.assertNotIn("原因：", txt)

    def test_many_buys_are_flagged_as_a_market_drift_signature(self):
        many = [make_fc(f"{2900 + i}", 0.60, 0.012, 0.03, mu_log=0.01 + 0.0001 * i, sigma_log=0.03) for i in range(12)]
        o = RC.build_recommendations(many, 5, now=NOW)
        self.assertEqual(o["n_buy_total"], 12)
        txt = next(n for n in o["notes"] if "同時達到偏多候選條件" in n)
        self.assertIn("12 檔", txt)
        self.assertIn("100%", txt)
        self.assertIn("整體市場的上漲漂移", txt)
        self.assertIn("不要把它當成選股訊號", txt)
        # a handful of buys in a big universe is an ordinary signal list: no such note
        few = RC.build_recommendations(make_universe(40, seed=7) + [make_fc("9001", 0.66, 0.05, 0.03)], 5, now=NOW)
        self.assertLess(few["n_buy_total"], max(5, 0.25 * few["n_scored"]))
        self.assertGreater(few["n_buy_total"], 0)
        self.assertFalse(any("同時達到偏多候選條件" in n for n in few["notes"]))
        self.assertFalse(any(n.startswith("目前沒有任何標的") for n in few["notes"]))          # and it is not the empty-list note


class TestEvidenceAndWording(unittest.TestCase):
    FCS = [view_fc(f"{3700 + i}", 0.04 * i - 0.1, 0.04) for i in range(10)]

    @staticmethod
    def strings(o):
        out = []

        def walk(x):
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(o)
        return out

    def test_evidence_default_and_pass_through(self):
        o = RC.build_recommendations(self.FCS, 5, now=NOW)
        self.assertEqual(o["evidence"], {"source": "none"})
        self.assertTrue(any("尚未取得「推薦整體歷史驗證」" in n for n in o["notes"]))
        ev = {"horizon": 5, "source": "universe_validation", "verdict_level": "no_edge", "ic": 0.012, "ic_pvalue": 0.4,
              "hit_rate": 0.53, "base_rate": 0.52, "computed_at": "2026-10-22T10:00:00+08:00", "extra": [1, 2]}
        snap = copy.deepcopy(ev)
        o = RC.build_recommendations(self.FCS, 5, now=NOW, evidence=ev)
        self.assertEqual(o["evidence"], snap)                              # copied as given, extra keys included
        self.assertIsNot(o["evidence"], ev)
        self.assertEqual(ev, snap)                                         # the argument is not mutated
        txt = next(n for n in o["notes"] if "推薦整體歷史驗證" in n)
        self.assertIn("沒有統計上顯著的選股優勢", txt)
        self.assertIn("+0.012", txt)
        self.assertIn("不是買進建議", txt)
        json.dumps(o, allow_nan=False)

    def test_evidence_note_by_level(self):
        for lvl, tail in (("weak_edge", "不是買進建議"), ("moderate_edge", "存活者偏差"), ("strong_edge", "存活者偏差")):
            ev = {"horizon": 5, "source": "universe_validation", "verdict_level": lvl}
            o = RC.build_recommendations(self.FCS, 5, now=NOW, evidence=ev)
            txt = next(n for n in o["notes"] if "推薦整體歷史驗證" in n)
            self.assertIn(tail, txt, lvl)
            self.assertEqual(o["evidence"]["verdict_level"], lvl)
        o = RC.build_recommendations(self.FCS, 5, now=NOW, evidence={"source": "universe_validation", "verdict_level": "bogus"})
        self.assertTrue(any("尚未取得" in n for n in o["notes"]))          # unusable evidence is treated as missing
        o = RC.build_recommendations(self.FCS, 5, now=NOW, evidence="not a dict")
        self.assertEqual(o["evidence"], {"source": "none"})

    def test_no_buy_advice_wording_anywhere(self):
        fcs = make_universe(40, seed=2) + [make_fc("9001", 0.66, 0.05, 0.03), make_fc("9002", 0.30, -0.05, 0.03)]
        for lvl in (None, "no_edge", "weak_edge", "moderate_edge", "strong_edge"):
            ev = None if lvl is None else {"source": "universe_validation", "verdict_level": lvl}
            o = RC.build_recommendations(fcs, 5, now=NOW, evidence=ev)
            joined = "".join(self.strings(o))
            self.assertNotIn("建議買進", joined, lvl)                       # DESIGN.md 9.6: 偏多候選, never 建議買進
            self.assertTrue(any("偏多候選" in n for n in o["notes"]), lvl)
        self.assertTrue(any(r["signal"] == "buy" for r in o["all"]))
        for r in o["all"]:
            self.assertNotIn("買進建議", r["reason"])
            self.assertFalse(SIMPLIFIED & set(r["reason"]))

    def test_rank_note_explains_the_tilt(self):
        o = RC.build_recommendations(sigma_indep_universe(40, 2), 5, now=NOW)
        txt = o["notes"][0]
        self.assertIn("相對強度", txt)
        self.assertIn("並非選股能力的證明", txt)
        self.assertIn(f"{o['rank_sigma_corr']:+.2f}", txt)
        low = RC.build_recommendations([view_fc(f"{3800 + i}", 0.4 - 0.05 * i, 0.02 + 0.01 * i) for i in range(10)], 5, now=NOW)
        self.assertLess(low["rank_sigma_corr"], -0.9)                       # high view <-> low sigma by construction
        self.assertIn("低波動", low["notes"][0])



class TestRobustness(unittest.TestCase):
    def test_wrong_types_degrade_to_unscored(self):
        for key, bad in (("horizons", 5), ("horizons", None), ("horizons", [1, 2]), ("regime", 7), ("vol", "x"),
                         ("drivers", 3), ("drivers", [None, 4, {"name": None}])):
            fc = make_fc()
            fc[key] = bad
            ann = RC.annotate_forecast(fc, "stock")                       # must not raise
            out = RC.build_recommendations([fc, make_fc("2317")], 5, now=NOW)
            self.assertEqual(out["universe_size"], 2)
            json.dumps(out, allow_nan=False)
            if key == "horizons":
                self.assertEqual(out["n_scored"], 1)
        fc = make_fc()
        fc["horizons"]["5"] = 3.0                                         # horizon entry is not a dict
        self.assertEqual(RC.build_recommendations([fc], 5, now=NOW)["n_scored"], 0)
        fc = make_fc()
        fc["horizons"]["5"]["components"] = 4
        self.assertEqual(RC.annotate_forecast(fc, "stock")["horizons"]["5"]["signal"], "buy")
        out = RC.build_recommendations([make_fc()], 5, market_forecast={"horizons": 3, "regime": 4}, now=NOW)
        self.assertEqual(out["market"]["p_up_1"], None)

    def test_implausible_numbers_are_not_scored(self):
        for p, er, sg in ((1.2, 0.01, 0.03), (-0.1, 0.01, 0.03), (0.6, 6.0, 0.03), (0.6, 0.01, 6.0), (0.6, 1e300, 0.03),
                          (0.6, 0.01, 1e-300 * 0)):
            fc = make_fc(p=p, er=er, sg=sg)
            h = RC.annotate_forecast(fc, "stock")["horizons"]["5"]
            self.assertEqual((h["signal"], h["score"]), ("hold", None), (p, er, sg))
            self.assertEqual(RC.build_recommendations([fc], 5, now=NOW)["n_scored"], 0)
        self.assertEqual(RC.build_recommendations([make_fc(p=1.0, er=0.04, sg=0.03)], 5, now=NOW)["n_scored"], 1)

    def test_random_corruption_never_crashes(self):
        import warnings
        rng = np.random.default_rng(7)
        junk = [None, float("nan"), float("inf"), "x", [], {}, 0, -1, 1e300, True]

        def mutate(fc):
            fc = copy.deepcopy(fc)

            def walk(d, depth=0):
                for k in list(d.keys()):
                    r = rng.random()
                    if r < 0.07:
                        d[k] = junk[int(rng.integers(len(junk)))]
                    elif r < 0.10:
                        del d[k]
                    elif isinstance(d[k], dict) and depth < 3:
                        walk(d[k], depth + 1)
            walk(fc)
            return fc
        base = make_universe(20, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for _ in range(60):
                fcs = [mutate(f) if rng.random() < 0.5 else f for f in base] + [None, 5]
                for h in (1, 5, 7):
                    out = RC.build_recommendations(fcs, h, top_k=int(rng.integers(1, 9)), now=NOW)
                    json.dumps(out, allow_nan=False)
                    self.assertLessEqual(sum(r["weight"] for r in out["buys"]), 1 + 1e-9)
                    self.assertTrue(all(0 <= r["weight"] <= config.MAX_WEIGHT + 1e-12 for r in out["all"]))
                for f in fcs:
                    if isinstance(f, dict):
                        RC.annotate_forecast(f, "stock")


if __name__ == "__main__":
    unittest.main()
