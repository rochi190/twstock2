"""Tests for backend/evaluation/strategy.py: tiny hand-computed examples, costs, benchmark, causality."""
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import eval_helpers as H  # noqa: E402
from backend import config  # noqa: E402
from backend.evaluation import metrics as M  # noqa: E402
from backend.evaluation import strategy as S  # noqa: E402

# six bars with easy arithmetic
OPENS = [100, 101, 102, 102, 105, 104]
CLOSES = [100, 102, 101, 103, 104, 102]
CO = [0, 101 / 100 - 1, 102 / 102 - 1, 102 / 101 - 1, 105 / 103 - 1, 104 / 104 - 1]   # overnight leg of row d
OC = [0, 102 / 101 - 1, 101 / 102 - 1, 103 / 102 - 1, 104 / 105 - 1, 102 / 104 - 1]   # intraday leg of row d
CC = [0, 102 / 100 - 1, 101 / 102 - 1, 103 / 101 - 1, 104 / 103 - 1, 102 / 104 - 1]
W = [0, 1, 1, 0, 1, 0]   # decided at the close of each row (row 5 is never tradable)
B, SELL = 0.001, 0.004


class TestLegs(unittest.TestCase):
    def test_legs_hand_example(self):
        d = H.make_tiny_data(OPENS, CLOSES)
        co, oc, cc = S.leg_returns(d)
        np.testing.assert_allclose(co, CO, atol=1e-15)
        np.testing.assert_allclose(oc, OC, atol=1e-15)
        np.testing.assert_allclose(cc, CC, atol=1e-15)
        np.testing.assert_allclose((1 + co) * (1 + oc), 1 + cc, rtol=1e-15)

    def test_adjusted_prices_remove_dividend_gap(self):
        # 5 TWD dividend, ex-date at row 2: raw price drops 100 -> 95 but the holder lost nothing
        d = H.make_tiny_data([100, 100, 95, 96], [100, 100, 95, 96], adj=[95, 95, 95, 96])
        co, oc, cc = S.leg_returns(d)
        self.assertAlmostEqual(co[2], 0.0, places=12)
        self.assertAlmostEqual(cc[2], 0.0, places=12)
        self.assertAlmostEqual(oc[2], 0.0, places=12)
        raw_gap = d.close[2] / d.close[1] - 1
        self.assertAlmostEqual(raw_gap, -0.05)   # the raw series would show a fake -5 % loss
        np.testing.assert_allclose(cc[3], 96 / 95 - 1)

    def test_invalid_open_falls_back_to_close_to_close(self):
        d = H.make_tiny_data([100, 0.0, 102, np.nan], [100, 102, 101, 103])
        co, oc, cc = S.leg_returns(d)
        self.assertAlmostEqual(co[1], 0.02)
        self.assertEqual(oc[1], 0.0)
        self.assertAlmostEqual(co[3], 103 / 101 - 1)
        self.assertEqual(oc[3], 0.0)
        self.assertTrue(np.all(np.isfinite(co)) and np.all(np.isfinite(oc)))

    def test_single_bar(self):
        d = H.make_tiny_data([100], [100])
        for a in S.leg_returns(d):
            self.assertEqual(a.tolist(), [0.0])


class TestCosts(unittest.TestCase):
    def test_side_costs_sum_to_roundtrip(self):
        for kind in ("stock", "etf", "index"):
            for disc in (0.28, 0.6, 1.0):
                for slip in (0.0, 5.0, 20.0):
                    b, s = S.side_costs(kind, disc, slip)
                    self.assertAlmostEqual(b + s, config.roundtrip_cost(S.cost_kind(kind), disc, slip), places=15)
        b, s = S.side_costs("stock", 0.6, 5.0)
        self.assertAlmostEqual(b, 0.001425 * 0.6 + 0.0005)             # commission + slippage, no tax
        self.assertAlmostEqual(s, 0.001425 * 0.6 + 0.0005 + 0.003)    # + 0.3 % tax
        self.assertAlmostEqual(S.side_costs("etf", 0.6, 5.0)[1] - S.side_costs("etf", 0.6, 5.0)[0], 0.001)

    def test_expected_simple_return(self):
        self.assertAlmostEqual(float(S.expected_simple_return(0.01, 0.1)), math.exp(0.01 + 0.005) - 1)
        mc = np.random.default_rng(1).normal(0.02, 0.1, 400000)
        self.assertAlmostEqual(float(S.expected_simple_return(0.02, 0.1)), float(np.mean(np.exp(mc) - 1)), places=3)


class TestSimulatorHandExamples(unittest.TestCase):
    def sim(self, mode, buy=0.0, sell=0.0, w=W, first=0):
        co, oc, cc = S.leg_returns(H.make_tiny_data(OPENS, CLOSES))
        return S.simulate_weights(co, oc, np.array(w, float), buy, sell, exec_mode=mode, first=first)

    def test_next_open_no_cost(self):
        r = self.sim("next_open")["ret"]
        # day d earns  s_{d-2} * overnight_d  and  s_{d-1} * intraday_d  (compounded within the day)
        exp = [0.0,
               1 * OC[2],                                     # d=2: old position 0 overnight, new 1 intraday
               (1 + CO[3]) * (1 + OC[3]) - 1,                 # d=3: held through the whole day = close-to-close
               (1 + CO[4]) - 1,                                # d=4: held overnight, sold at the open
               OC[5]]                                          # d=5: bought at the open, intraday only
        np.testing.assert_allclose(r, exp, atol=1e-15)
        # additive contract formula agrees to O(co*oc)
        add = [0, OC[2], CO[3] + OC[3], CO[4], OC[5]]
        np.testing.assert_allclose(r, add, atol=1.2e-4)
        # chain of the 1st holding episode equals open-to-open price ratio 105/102
        self.assertAlmostEqual((1 + r[1]) * (1 + r[2]) * (1 + r[3]) - 1, 105 / 102 - 1, places=14)

    def test_next_open_with_costs(self):
        r = self.sim("next_open", B, SELL)["ret"]
        exp = [0.0,
               (1 - B) * (1 + OC[2]) - 1,                      # buy at the open of day 2
               (1 + CO[3]) * (1 + OC[3]) - 1,                  # no trade
               (1 + CO[4]) * (1 - SELL) - 1,                    # sell at the open of day 4: commission+slip+tax
               (1 - B) * (1 + OC[5]) - 1]                       # buy again at the open of day 5
        np.testing.assert_allclose(r, exp, atol=1e-15)

    def test_close_mode_no_cost_and_cost(self):
        r = self.sim("close")["ret"]
        np.testing.assert_allclose(r, [0.0, CC[2], CC[3], 0.0, CC[5]], atol=1e-15)
        rc = self.sim("close", B, SELL)["ret"]
        exp = [0.0, (1 - B) * (1 + CC[2]) - 1, CC[3], (1 - SELL) - 1, (1 - B) * (1 + CC[5]) - 1]
        np.testing.assert_allclose(rc, exp, atol=1e-15)

    def test_close_mode_signal_earns_next_close_to_close_only(self):
        w = [0, 0, 1, 0, 0, 0]
        r = self.sim("close", w=w)["ret"]
        np.testing.assert_allclose(r, [0, 0, CC[3], 0, 0], atol=1e-15)   # s_2 earns cc_3 and nothing else

    def test_next_open_signal_earns_intraday_then_overnight(self):
        w = [0, 0, 1, 0, 0, 0]
        r = self.sim("next_open", w=w)["ret"]       # days 1..5
        np.testing.assert_allclose(r, [0, 0, OC[3], CO[4], 0], atol=1e-15)   # s_2: bought open(3), sold open(4)

    def test_roundtrip_cost_matches_config(self):
        b, s = S.side_costs("stock", 0.6, 5.0)
        co = oc = np.zeros(6)
        sim = S.simulate_weights(co, oc, np.array([0, 1, 0, 0, 0, 0.0]), b, s, exec_mode="next_open", first=0)
        total = float(np.prod(1 + sim["ret"]) - 1)
        self.assertAlmostEqual(total, (1 - b) * (1 - s) - 1, places=15)
        # equals the additive config cost up to the second-order product b*s
        self.assertAlmostEqual(-total, config.roundtrip_cost("stock", 0.6, 5.0), delta=1.01 * b * s)

    def test_first_index_and_horizon(self):
        for first in (0, 2, 4):
            out = self.sim("next_open", first=first)
            self.assertEqual(len(out["ret"]), 6 - 1 - first)
            self.assertEqual(out["w_held"].shape, (6 - 1 - first, 1))
        self.assertEqual(len(self.sim("next_open", first=5)["ret"]), 0)
        self.assertEqual(len(self.sim("next_open", first=9)["ret"]), 0)
        with self.assertRaises(ValueError):
            self.sim("vwap")

    def test_rows_before_first_are_cash(self):
        a = self.sim("next_open", first=2, w=[1, 1, 0, 1, 0, 0])
        b = self.sim("next_open", first=2, w=[0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(a["ret"], b["ret"])

    def test_nan_weights_are_cash(self):
        a = self.sim("next_open", w=[0, np.nan, 1, 0, np.nan, 0])
        b = self.sim("next_open", w=[0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(a["ret"], b["ret"])

    def test_two_assets_hand_example(self):
        co = np.array([[0, 0], [0.01, -0.02], [0.0, 0.01]])
        oc = np.array([[0, 0], [0.02, 0.00], [0.01, -0.01]])
        Wm = np.array([[0.5, 0.5], [1.0, 0.0], [0, 0]])
        out = S.simulate_weights(co, oc, Wm, [0.001, 0.002], [0.004, 0.005], exec_mode="next_open", first=0)
        # d=1: w2 = 0, w1 = (.5,.5): cost = .5*.001 + .5*.002 ; intraday = .5*.02 + .5*0
        c1 = 0.5 * 0.001 + 0.5 * 0.002
        e1 = (1 - c1) * (1 + 0.5 * 0.02) - 1
        # d=2: w2 = (.5,.5), w1 = (1,0): overnight = .5*0 + .5*.01 ; trade +.5 asset0 (buy .001), -.5 asset1 (sell .005)
        c2 = 0.5 * 0.001 + 0.5 * 0.005
        e2 = (1 + 0.005) * (1 - c2) * (1 + 1.0 * 0.01) - 1
        np.testing.assert_allclose(out["ret"], [e1, e2], atol=1e-15)
        np.testing.assert_allclose(out["turnover"], [1.0, 1.0])
        np.testing.assert_allclose(out["exposure"], [1.0, 1.0])


class TestContractFormula(unittest.TestCase):
    """The simulator compounds within the day; DESIGN.md writes the day return additively.  They must agree to second
    order (products of two sub-daily returns / cost x return), on random data and random positions."""

    def test_agrees_with_additive_formula_to_second_order(self):
        data = H.make_price_data(2000, seed=77, vol=0.02)
        co, oc, cc = S.leg_returns(data)
        rng = np.random.default_rng(1)
        W = np.where(rng.random(2000) < 0.5, 1.0, 0.0) * rng.uniform(0.2, 1.0, 2000)     # fractional weights too
        b, s = S.side_costs("stock", 0.6, 5.0)
        sim = S.simulate_weights(co, oc, W, b, s, exec_mode="next_open", first=10)
        r = sim["ret"]
        N = len(r)
        w1, w2 = W[10:10 + N], np.concatenate([[0.0], W[10:10 + N - 1]])
        delta = w1 - w2
        cost = np.maximum(delta, 0) * b + np.maximum(-delta, 0) * s
        a, bb = w2 * co[11:11 + N], w1 * oc[11:11 + N]
        additive = a + bb - cost
        # exact algebra: (1+a)(1-c)(1+b) - 1 = a + b - c + ab - c(a+b) - abc
        np.testing.assert_allclose(r, additive + a * bb - cost * (a + bb) - cost * a * bb, atol=1e-14)
        self.assertLess(np.max(np.abs(r - additive)), 1e-3)           # second order: tiny next to the 1st-order terms
        self.assertGreater(np.max(np.abs(r - additive)), 0.0)         # ... but it is not the same formula
        self.assertLess(np.mean(np.abs(r - additive)), 0.02 * np.mean(np.abs(additive)))
        # close mode: r = s_t * cc_{t+1} - cost
        sim_c = S.simulate_weights(co, oc, W, b, s, exec_mode="close", first=10)["ret"]
        add_c = w1 * cc[11:11 + N] - cost
        self.assertLess(np.max(np.abs(sim_c - add_c)), 1e-3)
        # total return over 1990 days differs by far less than the effect of costs themselves
        self.assertLess(abs(np.prod(1 + r) - np.prod(1 + additive)), 0.05 * abs(np.prod(1 + additive)))


class TestCausality(unittest.TestCase):
    """A decision at close t can only change returns of day t+1 and later."""

    def setUp(self):
        d = H.make_price_data(300, seed=4)
        self.co, self.oc, self.cc = S.leg_returns(d)
        self.rng = np.random.default_rng(0)
        self.W = (self.rng.random(300) < 0.5).astype(float)
        self.b, self.s = S.side_costs("stock", 0.6, 5.0)

    def run_sim(self, W, mode, co=None, oc=None, first=20):
        return S.simulate_weights(self.co if co is None else co, self.oc if oc is None else oc, W, self.b, self.s,
                                  exec_mode=mode, first=first)["ret"]

    def test_perturbing_signal_t_never_changes_earlier_days(self):
        for mode in S.EXEC_MODES:
            base = self.run_sim(self.W, mode)
            for t in (25, 100, 190, 250):
                W2 = self.W.copy()
                W2[t] = 1.0 - W2[t]
                new = self.run_sim(W2, mode)
                k = t - 20                       # day index of day t+1 is k = t - first
                np.testing.assert_array_equal(base[:k], new[:k])          # days <= t : identical
                self.assertFalse(np.array_equal(base[k:], new[k:]))      # day t+1 on: changed (cost at least)

    def test_perturbing_future_prices_never_changes_earlier_days(self):
        for mode in S.EXEC_MODES:
            base = self.run_sim(self.W, mode)
            t = 150
            co2, oc2 = self.co.copy(), self.oc.copy()
            co2[t + 1:] = self.rng.normal(0, 0.05, 300 - t - 1)
            oc2[t + 1:] = self.rng.normal(0, 0.05, 300 - t - 1)
            new = self.run_sim(self.W, mode, co2, oc2)
            k = t - 20                                # day index k is day t+1; days <= t are indices 0..k-1
            np.testing.assert_array_equal(base[:k], new[:k])
            self.assertFalse(np.array_equal(base[k:], new[k:]))

    def test_lone_signal_earns_nothing_before_next_day(self):
        for mode in S.EXEC_MODES:
            for t in (30, 120):
                W = np.zeros(300)
                W[t] = 1.0
                r = S.simulate_weights(self.co, self.oc, W, 0.0, 0.0, exec_mode=mode, first=20)["ret"]
                k = t - 20
                self.assertTrue(np.all(r[:k] == 0.0), mode)            # nothing before day t+1
                if mode == "close":
                    self.assertAlmostEqual(r[k], self.cc[t + 1], places=15)   # held from close t: earns cc_{t+1}
                    self.assertEqual(r[k + 1], 0.0)                            # flat again after one day
                else:
                    self.assertAlmostEqual(r[k], self.oc[t + 1], places=15)   # day t+1: intraday leg only
                    self.assertAlmostEqual(r[k + 1], self.co[t + 2], places=15)   # day t+2: overnight leg, then sold
                    self.assertEqual(r[k + 2], 0.0)

    def test_close_mode_alignment_is_next_day(self):
        W = np.zeros(300)
        W[100] = 1.0
        r = S.simulate_weights(self.co, self.oc, W, 0.0, 0.0, exec_mode="close", first=20)["ret"]
        # s_100 is held from close(100): earns cc_101 on day 101 (index 100-20 = 80 -> day 101)
        self.assertAlmostEqual(r[80], self.cc[101], places=15)
        self.assertTrue(np.all(r[:80] == 0) and np.all(r[81:] == 0))

    def test_positions_use_only_row_t_forecast(self):
        data, wf, _ = H.make_planted_world(900, 5, h_list=(5,), first_idx=300)
        hf = wf.horizons[5]
        s = S.positions_from_forecast(hf, kind="stock", threshold=0.55)
        for t in (320, 500, 700):
            hf2 = H.make_hf(data, 5, 300, hf.p_up.copy(), hf.mu.copy(), hf.sigma.copy())
            hf2.p_up[t + 1:] = 0.99
            hf2.mu[t + 1:] = 0.5
            hf2.sigma[t + 1:] = 0.01
            hf2.realized[:] = np.random.default_rng(1).normal(size=900)   # look-ahead fields must be ignored
            hf2.pit[:] = 0.5
            s2 = S.positions_from_forecast(hf2, kind="stock", threshold=0.55)
            np.testing.assert_array_equal(s[:t + 1], s2[:t + 1])

    def test_run_strategy_ignores_realized_and_pit(self):
        data, wf, _ = H.make_planted_world(900, 6, h_list=(1,), first_idx=300)
        hf = wf.horizons[1]
        a = S.run_strategy(data, hf, 300, n_boot=100)
        hf.realized = np.random.default_rng(2).normal(size=900)
        hf.pit = np.full(900, 0.5)
        b = S.run_strategy(data, hf, 300, n_boot=100)
        self.assertEqual(a.section, b.section)


class TestPositionRule(unittest.TestCase):
    def setUp(self):
        self.data = H.make_tiny_data([100] * 6, [100] * 6)

    def hf(self, p, mu, sg, h=5):
        n = 6
        return H.make_hf(self.data, h, 0, p, mu, sg)

    def test_threshold_rule_hand(self):
        cost = 2 * 0.001425 * 0.6 + 0.003 + 2 * 5 / 1e4
        self.assertAlmostEqual(cost, config.roundtrip_cost("stock", 0.6, 5.0))
        p = [0.60, 0.56, 0.50, 0.70, 0.9, np.nan]
        mu = [0.010, 0.004, 0.020, 0.020, np.nan, 0.02]
        sg = [0.03] * 6
        s = S.positions_from_forecast(self.hf(p, mu, sg), kind="stock", threshold=0.55)
        net = [math.exp(m + 0.5 * 0.03 ** 2) - 1 - cost if not math.isnan(m) else float("nan") for m in mu]
        self.assertGreater(net[0], 0)
        self.assertLess(net[1], 0)          # p high enough but expected return does not cover the cost
        self.assertEqual(s.tolist(), [1, 0, 0, 1, 0, 0])   # p<thr / NaN / net<=0 -> cash
        # ETF: lower tax lowers the hurdle
        mu2 = [0.0045] * 6     # E[simple] ~ 0.50 %: below the stock hurdle (0.57 %) but above the ETF hurdle (0.37 %)
        self.assertEqual(S.positions_from_forecast(self.hf([0.6] * 6, mu2, sg), kind="stock").sum(), 0)
        self.assertEqual(S.positions_from_forecast(self.hf([0.6] * 6, mu2, sg), kind="etf").sum(), 6)

    def test_threshold_is_inclusive_and_commission_matters(self):
        s = S.positions_from_forecast(self.hf([0.55] * 6, [0.02] * 6, [0.03] * 6), threshold=0.55)
        self.assertEqual(s.sum(), 6)
        hi = S.positions_from_forecast(self.hf([0.6] * 6, [0.0062] * 6, [0.03] * 6), commission_discount=1.0,
                                       slippage_bps=30)
        lo = S.positions_from_forecast(self.hf([0.6] * 6, [0.0062] * 6, [0.03] * 6), commission_discount=0.3,
                                       slippage_bps=0)
        self.assertEqual((hi.sum(), lo.sum()), (0, 6))

    def test_kelly_sizing_hand(self):
        sg = 0.1
        cost = config.roundtrip_cost("stock")
        mu = [0.01, 0.0, 0.5, 0.01, 0.01, 0.01]
        p = [0.6, 0.6, 0.6, 0.5, 0.6, 0.6]
        s = S.positions_from_forecast(self.hf(p, mu, [sg] * 6), sizing="kelly")
        net0 = math.exp(0.01 + 0.5 * sg * sg) - 1 - cost
        self.assertAlmostEqual(s[0], config.KELLY_FRACTION * net0 / sg ** 2, places=12)
        self.assertLess(s[0], 1.0)
        self.assertEqual(s[1], 0.0)    # no edge
        self.assertEqual(s[2], 1.0)    # capped at 100 %
        self.assertEqual(s[3], 0.0)    # probability gate
        self.assertTrue(np.all((s >= 0) & (s <= 1)))
        with self.assertRaises(ValueError):
            S.positions_from_forecast(self.hf(p, mu, [sg] * 6), sizing="martingale")

    def test_zero_or_negative_sigma_is_cash(self):
        s = S.positions_from_forecast(self.hf([0.9] * 6, [0.05] * 6, [0.0, -1, 0.02, 0.02, 0.02, 0.02]))
        self.assertEqual(s.tolist(), [0, 0, 1, 1, 1, 1])

    def test_rule_text(self):
        self.assertEqual(S.rule_text(0.55), "機率≥55%且扣成本後期望報酬>0 時持有，否則空手")
        self.assertIn("57.5%", S.rule_text(0.575))
        self.assertIn("凱利", S.rule_text(0.55, "kelly"))


class TestEpisodesAndStats(unittest.TestCase):
    def test_episode_returns_hand(self):
        r = np.array([0.0, -0.0098, 0.0198, 0.0194, -0.0192])
        eps = S.trade_episode_returns([0, 1, 1, 0, 1], r)
        np.testing.assert_allclose(eps, [(1 - 0.0098) * 1.0198 * 1.0194 - 1, -0.0192])
        self.assertEqual(len(S.trade_episode_returns([0, 0, 0], np.zeros(3))), 0)
        self.assertEqual(len(S.trade_episode_returns([], [])), 0)
        # an episode that runs to the very end is closed at the last day (mark to market)
        np.testing.assert_allclose(S.trade_episode_returns([0, 1, 1], np.array([0, 0.01, 0.02])), [1.01 * 1.02 - 1])
        # two episodes separated by a single flat day do not share days
        e = S.trade_episode_returns([1, 0, 1, 0, 0], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        np.testing.assert_allclose(e, [1.1 * 1.2 - 1, 1.3 * 1.4 - 1])

    def test_stats_block_hand(self):
        sim = S.simulate_weights(*S.leg_returns(H.make_tiny_data(OPENS, CLOSES))[:2], np.array(W, float), 0, 0,
                                 exec_mode="next_open", first=0)
        eps = S.trade_episode_returns(sim["w_held"][:, 0], sim["ret"])
        st = S.stats_block(sim["ret"], sim["exposure"], sim["turnover"], eps, n_boot=None) if False else \
            S.stats_block(sim["ret"], sim["exposure"], sim["turnover"], eps, sharpe_ci=[0, 1])
        self.assertEqual(st["n_trades"], 2)
        self.assertAlmostEqual(st["trade_hit_rate"], 0.5)
        self.assertAlmostEqual(st["exposure"], 0.6)                 # held weights 0,1,1,0,1
        self.assertAlmostEqual(st["turnover_per_year"], 3 / (5 / 245))   # |delta| = 0,1,0,1,1
        self.assertEqual(list(st), ["total_return", "cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "exposure",
                                    "turnover_per_year", "n_trades", "trade_hit_rate", "psr", "dsr", "sharpe_ci95"])

    def test_empty_stats_schema(self):
        e = S.empty_stats()
        self.assertEqual(e["n_trades"], 0)
        json.dumps(e, allow_nan=False)


class TestRunStrategy(unittest.TestCase):
    def test_always_long_equals_buy_and_hold_exactly(self):
        data = H.make_price_data(700, seed=8)
        n = data.n
        hf = H.make_hf(data, 1, 300, np.full(n, 0.99), np.full(n, 0.5), np.full(n, 0.02))
        res = S.run_strategy(data, hf, 300, commission_discount=0.0, slippage_bps=0.0, n_boot=100)
        c = res.section["curve"]
        co, oc, cc = S.leg_returns(data)
        self.assertEqual(c["dates"][0], str(data.dates[300]))
        self.assertEqual(c["dates"][-1], str(data.dates[-1]))
        self.assertEqual(c["position"][0], 0.0)
        # signal at close 300 is executed at the open of 301: the first overnight gap is (by design) not earned,
        # from the second day on "always long" reproduces close-to-close buy-and-hold exactly
        ret = res.sim["ret"]
        self.assertAlmostEqual(ret[0], oc[301], places=14)
        np.testing.assert_allclose(ret[1:], cc[302:], atol=1e-14)
        np.testing.assert_allclose(c["buyhold"][-1], data.adjclose[-1] / data.adjclose[300], atol=1e-6)
        bh_eq = np.array(c["buyhold"])
        st_eq = np.array(c["strategy"])
        np.testing.assert_allclose(st_eq[1:] / st_eq[1], bh_eq[1:] / bh_eq[1], atol=2e-5)  # same shape after day 1
        # close-of-day execution has no such gap: the curves coincide
        res_c = S.run_strategy(data, hf, 300, exec_mode="close", commission_discount=0.0, slippage_bps=0.0,
                               n_boot=100)
        np.testing.assert_allclose(res_c.section["curve"]["strategy"], res_c.section["curve"]["buyhold"], atol=1e-6)
        np.testing.assert_allclose(res_c.sim["ret"], cc[301:], atol=1e-14)
        stc, bmc = res_c.section["stats"], res_c.section["benchmark"]
        self.assertAlmostEqual(stc["total_return"], bmc["total_return"], places=10)
        self.assertAlmostEqual(stc["max_dd"], bmc["max_dd"], places=10)

    def test_costs_reduce_return_and_close_beats_next_open_on_trending_signal(self):
        data, wf, _ = H.make_planted_world(1200, 3, h_list=(1,), first_idx=400, signal_sd=0.006)
        hf = wf.horizons[1]
        free = S.run_strategy(data, hf, 400, commission_discount=0.0, slippage_bps=0.0, n_boot=50)
        paid = S.run_strategy(data, hf, 400, commission_discount=1.0, slippage_bps=20.0, n_boot=50)
        self.assertLess(paid.section["stats"]["total_return"], free.section["stats"]["total_return"])
        # the cost hurdle is part of the position rule: higher costs can only remove trades
        self.assertLessEqual(paid.section["stats"]["n_trades"], free.section["stats"]["n_trades"])

    def test_section_structure_downsampling_and_json(self):
        data = H.make_price_data(3000, seed=9)
        n = data.n
        p = 0.5 + 0.1 * np.sin(np.arange(n) / 7.0)
        hf = H.make_hf(data, 1, 300, p, 0.01 * np.sin(np.arange(n) / 7.0) + 0.0002, 0.02)
        res = S.run_strategy(data, hf, 300, n_boot=100)
        sec = res.section
        self.assertEqual(set(sec), {"rule", "stats", "benchmark", "sharpe_diff_ci95", "curve"})
        c = sec["curve"]
        lens = {k: len(v) for k, v in c.items()}
        self.assertEqual(len(set(lens.values())), 1)
        self.assertLessEqual(lens["dates"], 600)
        self.assertEqual(c["dates"][-1], str(data.dates[-1]))
        self.assertAlmostEqual(c["strategy"][0], 1.0)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in c["position"]))
        self.assertTrue(all(x <= 1e-12 for x in c["drawdown_strategy"]))
        json.dumps(M.json_safe(sec), allow_nan=False)

    def test_no_signals_all_cash(self):
        data = H.make_price_data(700, seed=10)
        hf = H.make_hf(data, 1, 300, np.full(700, 0.4), np.full(700, -0.001), np.full(700, 0.02))
        res = S.run_strategy(data, hf, 300, n_boot=100)
        st = res.section["stats"]
        self.assertEqual((st["n_trades"], st["total_return"], st["exposure"]), (0, 0.0, 0.0))
        self.assertIsNone(st["sharpe"])
        self.assertIsNone(st["trade_hit_rate"])
        self.assertIsNone(res.psr_vs_bh)
        self.assertIsNotNone(res.section["benchmark"]["sharpe"])
        json.dumps(M.json_safe(res.section), allow_nan=False)

    def test_tiny_windows(self):
        data = H.make_price_data(310, seed=11)
        hf = H.make_hf(data, 1, 300, np.full(310, 0.9), np.full(310, 0.05), np.full(310, 0.02))
        for first in (300, 308, 309, 400):
            res = S.run_strategy(data, hf, first, n_boot=50)
            json.dumps(M.json_safe(res.section), allow_nan=False)
        res = S.run_strategy(data, hf, 309, n_boot=50)
        self.assertEqual(res.section["curve"]["dates"], [])
        res = S.run_strategy(data, hf, 308, n_boot=50)
        self.assertEqual(len(res.section["curve"]["dates"]), 2)

    def test_constant_prices(self):
        data = H.make_tiny_data([50.0] * 400, [50.0] * 400)
        hf = H.make_hf(data, 1, 300, np.full(400, 0.9), np.full(400, 0.05), np.full(400, 0.02))
        res = S.run_strategy(data, hf, 300, n_boot=50)
        st, bm = res.section["stats"], res.section["benchmark"]
        self.assertIsNone(bm["sharpe"])
        self.assertAlmostEqual(bm["total_return"], 0.0)
        self.assertLess(st["total_return"], 0.0)       # pays costs on a flat market
        json.dumps(M.json_safe(res.section), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
