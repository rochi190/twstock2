"""Filtered historical simulation, alignment to exact quantiles, VaR / CVaR / drawdown."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401

from backend import config  # noqa: E402
from backend.quant import garch as G  # noqa: E402
from backend.quant import mc as MC  # noqa: E402

LIMIT = (np.log(0.9), np.log(1.1))
V = 0.016 ** 2


def gp(**kw):
    a = dict(alpha=0.03, gamma=0.08, beta=0.90, nu=6.0, mean=0.0)
    a.update(kw)
    phi = a["alpha"] + a["beta"] + a["gamma"] / 2
    return G.GarchParams(omega=(1 - phi) * V, **a)


def t_pool(n=30000, nu=6.0, seed=1):
    return G.standardized_t(np.random.default_rng(seed), nu, n)


class TestSimulation(unittest.TestCase):
    def test_shape_deterministic_and_limit(self):
        p = gp()
        args = dict(p=p, sig2_next=V, z_pool=t_pool(), pool_w=None, A=None, pi_now=None, drift=0.0003,
                    n_paths=3000, n_steps=20)
        a = MC.fhs_simulate(rng=np.random.default_rng(5), ret_bounds=LIMIT, **args)
        b = MC.fhs_simulate(rng=np.random.default_rng(5), ret_bounds=LIMIT, **args)
        self.assertEqual(a.shape, (3000, 20))
        np.testing.assert_array_equal(a, b)
        # extreme volatility: every simulated daily return is clipped to the +-10 % price limit
        wild = dict(args, sig2_next=0.2 ** 2)
        c = MC.fhs_simulate(rng=np.random.default_rng(6), ret_bounds=LIMIT, **wild)
        daily = np.diff(np.concatenate([np.zeros((3000, 1)), c], axis=1), axis=1)
        self.assertLessEqual(daily.max(), LIMIT[1] + 1e-12)
        self.assertGreaterEqual(daily.min(), LIMIT[0] - 1e-12)
        at_limit = (daily <= LIMIT[0] + 1e-12) | (daily >= LIMIT[1] - 1e-12)
        self.assertGreater(float(at_limit.mean()), 0.3)  # the limit really binds in this stress case

    def test_variance_follows_the_garch_term_structure(self):
        p = gp()
        u1 = 2.0 * p.uncond_var
        cum = MC.fhs_simulate(p, u1, t_pool(60000, seed=2), None, None, None, 0.0, 40000, 20, np.random.default_rng(7),
                              (-5.0, 5.0))
        for h in (1, 5, 20):
            self.assertAlmostEqual(float(cum[:, h - 1].var()) / float(G.hstep_variance(u1, p, h)), 1.0, delta=0.05)

    def test_drift_and_empirical_shape_are_inherited(self):
        rng = np.random.default_rng(3)
        pool = -np.abs(rng.standard_normal(40000)) * 1.0  # strongly negatively skewed residual pool
        p = gp(gamma=0.0, alpha=0.0, beta=0.0)
        cum = MC.fhs_simulate(p, V, pool, None, None, None, 0.001, 50000, 1, np.random.default_rng(8), (-5.0, 5.0))
        r = cum[:, 0]
        self.assertAlmostEqual(float(r.mean()), 0.001, delta=3e-4)  # pool re-centred: drift is the supplied drift
        z = (r - r.mean()) / r.std()
        self.assertLess(float((z ** 3).mean()), -0.7)  # the half-normal's skewness (-0.995) survives

    def test_regime_mixing_changes_the_shocks_not_the_drift(self):
        rng = np.random.default_rng(4)
        z = rng.standard_normal(20000)
        # regime 0 draws the lower half of the residuals, regime 1 the upper half
        w = np.column_stack([(z < 0).astype(float) + 1e-9, (z >= 0).astype(float) + 1e-9])
        A = np.array([[0.97, 0.03], [0.03, 0.97]])
        p = gp()
        low = MC.fhs_simulate(p, V, z, w, A, np.array([1.0, 0.0]), 0.0, 20000, 5, np.random.default_rng(9), (-5, 5))
        high = MC.fhs_simulate(p, V, z, w, A, np.array([0.0, 1.0]), 0.0, 20000, 5, np.random.default_rng(9), (-5, 5))
        self.assertLess(float(low[:, 0].mean()), -0.5 * np.sqrt(V))  # starts in the "down" regime
        self.assertGreater(float(high[:, 0].mean()), 0.5 * np.sqrt(V))
        self.assertGreater(float(high[:, 4].mean()), float(low[:, 4].mean()))  # persistence carries it forward
        # a mixture over regimes (stationary start) is centred
        mix = MC.fhs_simulate(p, V, z, w, A, np.array([0.5, 0.5]), 0.0, 20000, 5, np.random.default_rng(10), (-5, 5))
        self.assertAlmostEqual(float(mix[:, 0].mean()), 0.0, delta=0.05 * np.sqrt(V) * 3)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            MC.fhs_simulate(gp(), V, np.array([0.1, np.nan]), None, None, None, 0.0, 10, 3, np.random.default_rng(1), LIMIT)


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.levels = np.array(config.FAN_LEVELS)
        rng = np.random.default_rng(11)
        self.cum = np.cumsum(0.015 * rng.standard_normal((6000, 20)), axis=1)

    def test_quantiles_at_anchor_days_equal_the_targets(self):
        anchors = {1: np.array([-0.025, -0.009, 0.001, 0.011, 0.027]),
                   5: np.array([-0.05, -0.02, 0.004, 0.027, 0.06]),
                   20: np.array([-0.10, -0.03, 0.017, 0.066, 0.15])}
        adj, shifts = MC.align_to_anchors(self.cum, self.levels, anchors)
        for d, q in anchors.items():
            np.testing.assert_allclose(np.quantile(adj[:, d - 1], self.levels), q, atol=2e-4)
        self.assertEqual(shifts.shape, (20, 5))
        # in-between days interpolate the shifts linearly
        np.testing.assert_allclose(shifts[2], 0.5 * (shifts[0] + shifts[4]) if False else shifts[2])
        lam = (3 - 1) / (5 - 1)
        np.testing.assert_allclose(shifts[2], (1 - lam) * shifts[0] + lam * shifts[4])

    def test_ranks_are_preserved_and_noop_when_targets_match(self):
        q_mc = np.quantile(self.cum[:, 4], self.levels)
        adj, shifts = MC.align_to_anchors(self.cum, self.levels, {5: q_mc})
        np.testing.assert_allclose(shifts[4], 0.0, atol=1e-12)
        anchors = {1: np.array([-0.03, -0.01, 0.0, 0.01, 0.03]), 20: np.array([-0.1, -0.03, 0.01, 0.05, 0.12])}
        adj2, _ = MC.align_to_anchors(self.cum, self.levels, anchors)
        for d in (0, 7, 19):
            self.assertTrue(np.array_equal(np.argsort(self.cum[:, d], kind="stable"), np.argsort(adj2[:, d], kind="stable")))

    def test_one_day_limit_is_enforced(self):
        anchors = {1: np.array([-0.5, -0.3, 0.0, 0.3, 0.5])}
        adj, _ = MC.align_to_anchors(self.cum, self.levels, anchors, LIMIT)
        self.assertLessEqual(adj[:, 0].max(), LIMIT[1] + 1e-12)
        self.assertGreaterEqual(adj[:, 0].min(), LIMIT[0] - 1e-12)

    def test_no_anchor_in_range_is_a_noop(self):
        adj, shifts = MC.align_to_anchors(self.cum, self.levels, {60: np.zeros(5)})
        np.testing.assert_array_equal(adj, self.cum)
        np.testing.assert_array_equal(shifts, 0.0)


class TestRisk(unittest.TestCase):
    def test_var_cvar_known_sample(self):
        x = np.linspace(-0.2, 0.2, 401)
        v, c = MC.var_cvar(x, 0.05)
        self.assertAlmostEqual(v, np.quantile(x, 0.05))
        self.assertAlmostEqual(c, float(x[x <= v].mean()))
        self.assertLess(c, v)

    def test_max_drawdown_hand_example(self):
        cum = np.log(np.array([[1.1, 0.99, 1.2, 0.9], [1.05, 1.1, 1.2, 1.3]]))
        dd = MC.max_drawdown(cum)
        self.assertAlmostEqual(float(dd[0]), 0.9 / 1.2 - 1.0, places=12)  # -25 %
        self.assertAlmostEqual(float(dd[1]), 0.0, places=12)
        self.assertTrue(np.all(MC.max_drawdown(np.zeros((3, 5))) == 0.0))

    def test_risk_metrics_keys_and_values(self):
        rng = np.random.default_rng(12)
        cum = np.cumsum(0.02 * rng.standard_normal((20000, 20)), axis=1)
        r = MC.risk_metrics(cum, 1, 20)
        self.assertEqual(set(r), {"var95_1d", "cvar95_1d", "var95_20d", "cvar95_20d", "median_max_drawdown_20d",
                                  "prob_loss_gt_10pct_20d"})
        self.assertLess(r["var95_1d"], 0)
        self.assertLess(r["cvar95_1d"], r["var95_1d"])
        self.assertLess(r["var95_20d"], r["var95_1d"])
        self.assertLessEqual(r["median_max_drawdown_20d"], 0)
        self.assertAlmostEqual(r["var95_1d"], np.expm1(-1.645 * 0.02), delta=0.002)
        self.assertAlmostEqual(r["prob_loss_gt_10pct_20d"], float(np.mean(np.expm1(cum[:, -1]) < -0.1)))

    def test_fan_from_paths(self):
        rng = np.random.default_rng(13)
        cum = np.cumsum(0.01 * rng.standard_normal((5000, 20)), axis=1)
        fan, q = MC.fan_from_paths(cum, config.FAN_LEVELS, 100.0)
        pr, rt = np.array(fan["prices"]), np.array(fan["returns"])
        self.assertEqual(pr.shape, (5, 20))
        self.assertEqual(fan["days"], list(range(1, 21)))
        np.testing.assert_allclose(pr, 100.0 * (1.0 + rt))
        self.assertTrue(np.all(np.diff(pr, axis=0) >= 0))
        self.assertTrue(np.all(np.diff(pr[-1] - pr[0]) > -1e-9))  # the fan widens with the horizon


if __name__ == "__main__":
    unittest.main()
