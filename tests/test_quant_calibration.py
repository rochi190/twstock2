"""Online Bayesian logistic recalibration: identity prior, MAP optimality, matured-only, bounded window."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401

from backend.quant import calibration as C  # noqa: E402


def raw_probs(n, seed=1, sd=0.12):
    return np.clip(0.5 + sd * np.random.default_rng(seed).standard_normal(n), 0.3, 0.7)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


class TestIdentityAndLearning(unittest.TestCase):
    def test_identity_when_nothing_has_matured(self):
        p = raw_probs(30)
        up = np.full(30, np.nan)
        pc, a, b = C.recalibrate(p, up, 1)
        np.testing.assert_allclose(pc, p, atol=1e-9)
        np.testing.assert_allclose(a, 0.0, atol=1e-12)
        np.testing.assert_allclose(b, 1.0, atol=1e-12)
        self.assertAlmostEqual(float(pc[0]), float(p[0]), places=9)

    def test_stays_near_identity_when_the_forecasts_are_calibrated(self):
        n = 3000
        p = raw_probs(n, 2)
        up = (np.random.default_rng(3).random(n) < p).astype(float)
        pc, a, b = C.recalibrate(p, up, 1)
        self.assertLess(float(np.abs(pc - p)[500:].mean()), 0.02)
        self.assertAlmostEqual(float(b[500:].mean()), 1.0, delta=0.15)

    def test_corrects_overconfidence(self):
        n = 4000
        p = raw_probs(n, 4)
        truth = C.expit(0.5 * C.logit(p))  # true slope 0.5 on the logit scale
        up = (np.random.default_rng(5).random(n) < truth).astype(float)
        pc, a, b = C.recalibrate(p, up, 1)
        self.assertLess(brier(pc[800:], up[800:]), brier(p[800:], up[800:]))
        self.assertTrue(0.45 < b[800:].mean() < 0.9)  # moves from the prior (1) towards 0.5 but is regularised

    def test_corrects_a_persistent_bias_partially(self):
        n = 4000
        p = raw_probs(n, 6)
        up = (np.random.default_rng(7).random(n) < np.clip(p + 0.05, 0, 1)).astype(float)
        pc, a, b = C.recalibrate(p, up, 1)
        shift = float((pc - p)[800:].mean())
        self.assertTrue(0.01 < shift < 0.05)
        self.assertLess(brier(pc[800:], up[800:]), brier(p[800:], up[800:]))

    def test_longer_horizons_are_tempered(self):
        n = 3000
        p = raw_probs(n, 8)
        up = (np.random.default_rng(9).random(n) < np.clip(p + 0.05, 0, 1)).astype(float)
        s = {h: float((C.recalibrate(p, up, h)[0] - p)[800:].mean()) for h in (1, 5, 20)}
        self.assertGreater(s[1], s[5])
        self.assertGreater(s[5], s[20])
        self.assertLess(s[20], 0.01)

    def test_output_is_a_probability(self):
        p = np.concatenate([np.linspace(1e-9, 1 - 1e-9, 200), np.full(100, np.nan)])
        up = (np.random.default_rng(10).random(300) < 0.5).astype(float)
        pc, a, b = C.recalibrate(p, up, 5)
        self.assertTrue(np.all(np.isnan(pc[200:])))
        self.assertTrue(np.all((pc[:200] > 0) & (pc[:200] < 1)))


class TestTiesCarryNoTarget(unittest.TestCase):
    """Audit C-03 / M-01: an unchanged close (``R == 0``, NaN target) is neither a success nor a failure of "up"."""

    def test_a_nan_target_is_ignored_whatever_its_raw_probability(self):
        n, h = 600, 1
        p = raw_probs(n, 21)
        up = (np.random.default_rng(22).random(n) < p).astype(float)
        tie = np.zeros(n, bool)
        tie[[250, 251, 400, 520]] = True
        up_t = np.where(tie, np.nan, up)
        base = C.recalibrate(p, up_t, h)[0]
        p2 = p.copy()
        p2[tie] = 0.9  # the raw probability of a row without a target cannot enter the fit
        pert = C.recalibrate(p2, up_t, h)[0]
        np.testing.assert_array_equal(base[~tie], pert[~tie])
        self.assertTrue(np.all(np.abs(base[tie] - pert[tie]) > 0.1))  # (only the tie rows' own output follows their input)
        down = C.recalibrate(p, np.where(tie, 0.0, up), h)[0]  # the pre-audit convention: a tie is a failure -> it matters
        self.assertGreater(float(np.abs(base[300:] - down[300:]).max()), 1e-4)

    def test_counting_ties_as_failures_biases_down_leaving_them_out_does_not(self):
        """P(up | the price moved) = p, 20 % of the outcomes are ties.  Left out: unbiased.  Counted as 'not up' the target
        frequency is 0.8 p and the Platt fit drifts down by the amount its prior allows (about 4 pp here)."""
        n = 6000
        p = raw_probs(n, 23)
        rng = np.random.default_rng(24)
        tie = rng.random(n) < 0.2
        up = (rng.random(n) < p).astype(float)
        left_out = C.recalibrate(p, np.where(tie, np.nan, up), 1)[0]
        as_failure = C.recalibrate(p, np.where(tie, 0.0, up), 1)[0]
        late = slice(1000, None)
        self.assertLess(abs(float((left_out - p)[late].mean())), 0.01)
        self.assertLess(float((as_failure - p)[late].mean()), -0.025)

    def test_a_long_run_of_ties_leaves_the_forecast_at_the_prior(self):
        n = 400
        p = raw_probs(n, 25)
        up = np.full(n, np.nan)  # every outcome is a tie
        pc, a, b = C.recalibrate(p, up, 1)
        np.testing.assert_allclose(pc, p, atol=1e-9)
        np.testing.assert_allclose(a, 0.0, atol=1e-12)
        np.testing.assert_allclose(b, 1.0, atol=1e-12)


class TestMAPAndCausality(unittest.TestCase):
    def test_solution_satisfies_the_first_order_condition(self):
        n, h, window, lam = 700, 5, 300, 0.995
        p = raw_probs(n, 11)
        up = (np.random.default_rng(12).random(n) < np.clip(p + 0.04, 0, 1)).astype(float)
        pc, a, b = C.recalibrate(p, up, h, window=window, lam=lam)
        i = 600
        # reconstruct the window: origins j in [i - window, i - h] (matured), age = i - h - j
        j = np.arange(i - window, i - h + 1)
        j = j[j >= 0]
        w = lam ** (i - h - j) / h
        x = C.logit(p[j])
        pr = C.expit(a[i] + b[i] * x)
        g_a = np.sum(w * (up[j] - pr)) - a[i] / 0.12 ** 2
        g_b = np.sum(w * (up[j] - pr) * x) - (b[i] - 1.0) / 0.35 ** 2
        self.assertAlmostEqual(g_a, 0.0, places=8)
        self.assertAlmostEqual(g_b, 0.0, places=8)

    def test_only_matured_outcomes_are_used(self):
        n, h = 500, 5
        p = raw_probs(n, 13)
        up = (np.random.default_rng(14).random(n) < p).astype(float)
        base = C.recalibrate(p, up, h)[0]
        i = 300
        up2 = up.copy()
        up2[i - h + 1:] = 1.0 - up2[i - h + 1:]  # outcomes of origins > i - h are unknown at row i
        pert = C.recalibrate(p, up2, h)[0]
        np.testing.assert_array_equal(base[: i + 1], pert[: i + 1])
        up3 = up.copy()
        up3[i - h] = 1.0 - up3[i - h]  # the newest matured outcome does matter
        self.assertNotEqual(C.recalibrate(p, up3, h)[0][i], base[i])

    def test_result_depends_only_on_the_window(self):
        n, window = 900, 250
        p = raw_probs(n, 15)
        up = (np.random.default_rng(16).random(n) < p).astype(float)
        full = C.recalibrate(p, up, 1, window=window)[0]
        i = 700
        lo = i - window
        part = C.recalibrate(p[lo: i + 1], up[lo: i + 1], 1, window=window)[0]
        self.assertAlmostEqual(float(full[i]), float(part[-1]), places=12)
        up2 = up.copy()
        up2[:lo] = 1.0 - up2[:lo]  # everything older than the window is irrelevant
        self.assertAlmostEqual(float(C.recalibrate(p, up2, 1, window=window)[0][i]), float(full[i]), places=12)


if __name__ == "__main__":
    unittest.main()
