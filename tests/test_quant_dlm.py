"""Discount-factor NIG dynamic linear model: reference cross-check, matured-only updates, learning, shrinkage."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401

from backend.quant import dlm as D  # noqa: E402


def reference_dlm(X, Y, hs, delta, beta_v, prior_sd, n0=4.0, S0=1.0):
    """Plain per-horizon loop implementation (textbook Kalman form) of the same recursions."""
    n, p = X.shape
    out = {k: np.zeros((len(hs), n)) for k in ("f", "pu", "S")}
    out["m"] = np.zeros((len(hs), n, p))
    lam0 = np.diag(1.0 / prior_sd ** 2)
    for a, h in enumerate(hs):
        k = float(h)
        Pinv = lam0.copy()
        m = np.zeros(p)
        nn, dd = n0, n0 * S0
        for t in range(n):
            b = Pinv @ m
            Pinv = delta * Pinv + (1.0 - delta) * lam0  # discount the information towards the prior information
            m = np.linalg.solve(Pinv, delta * b)
            j = t - h
            if j >= 0 and np.isfinite(Y[a, j]) and np.all(np.isfinite(X[j])):
                x, y = X[j], Y[a, j]
                C = np.linalg.inv(Pinv)
                e = y - x @ m
                Q = k + x @ C @ x
                m = m + C @ x * e / Q
                Pinv = Pinv + np.outer(x, x) / k
                nn = beta_v * nn + 1.0 / k
                dd = beta_v * dd + e * e / Q
            xt = X[t]
            C = np.linalg.inv(Pinv)
            S = dd / nn
            out["f"][a, t] = xt @ m
            out["pu"][a, t] = S * (xt @ C @ xt)
            out["S"][a, t] = S
            out["m"][a, t] = m
    return out


def overlapping_targets(shocks, hs, n):
    """Standardised h-day targets built from iid unit-variance daily shocks: Y[a, j] known at j + h."""
    cs = np.cumsum(shocks)
    Y = np.full((len(hs), n), np.nan)
    for a, h in enumerate(hs):
        Y[a, : n - h] = (cs[h: n] - cs[: n - h]) / np.sqrt(h)
    return Y


class TestAgainstReference(unittest.TestCase):
    def test_batched_information_form_matches_loop_reference(self):
        rng = np.random.default_rng(1)
        n, p = 260, 5
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        Y = overlapping_targets(rng.standard_normal(n + 30), (1, 5, 20), n)
        Y[0] += 0.2 * X[:, 0]  # some signal
        sd = D.default_prior_sd(p)
        out = D.run_dlm(X, Y, (1, 5, 20), delta=0.99, beta_v=0.995, prior_sd=sd)
        ref = reference_dlm(X, Y, (1, 5, 20), 0.99, 0.995, sd)
        for key in ("f", "pu", "S"):
            np.testing.assert_allclose(out[key], ref[key], rtol=1e-8, atol=1e-12, err_msg=key)
        np.testing.assert_allclose(out["m"], ref["m"], rtol=1e-8, atol=1e-12)

    def test_nonfinite_rows_are_skipped_not_propagated(self):
        rng = np.random.default_rng(2)
        n, p = 120, 3
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        Y = overlapping_targets(rng.standard_normal(n + 5), (1,), n)
        X[40, 0] = np.nan
        Y[0, 60] = np.nan
        out = D.run_dlm(X, Y, (1,))
        self.assertTrue(np.all(np.isfinite(out["f"])) and np.all(np.isfinite(out["pu"])))


class TestNoLookAhead(unittest.TestCase):
    def test_only_matured_pairs_are_used(self):
        rng = np.random.default_rng(3)
        n, p, hs = 400, 4, (1, 5, 20)
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        Y = overlapping_targets(rng.standard_normal(n + 30), hs, n)
        base = D.run_dlm(X, Y, hs)
        t = 250
        Y2 = Y.copy()
        for a, h in enumerate(hs):
            Y2[a, t - h + 1:] = 123.0  # outcomes of origins > t - h are not yet known at row t
        X2 = X.copy()
        X2[t + 1:] = -55.0  # future features
        pert = D.run_dlm(X2, Y2, hs)
        for key in ("f", "pu", "S"):
            np.testing.assert_array_equal(base[key][:, : t + 1], pert[key][:, : t + 1])
        np.testing.assert_array_equal(base["m"][:, : t + 1], pert["m"][:, : t + 1])
        # ... and the update at row t does use the pair of origin t - h
        Y3 = Y.copy()
        Y3[1, t - 5] += 1.0
        self.assertGreater(np.abs(D.run_dlm(X, Y3, hs)["m"][1, t] - base["m"][1, t]).max(), 1e-6)
        np.testing.assert_array_equal(D.run_dlm(X, Y3, hs)["m"][1, t - 1], base["m"][1, t - 1])

    def test_truncated_run_equals_prefix(self):
        rng = np.random.default_rng(4)
        n, p, hs = 300, 4, (1, 5)
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        Y = overlapping_targets(rng.standard_normal(n + 10), hs, n)
        full = D.run_dlm(X, Y, hs)
        m = 220
        Yt = Y[:, :m].copy()  # targets of truncated data are NaN for the last h origins
        for a, h in enumerate(hs):
            Yt[a, m - h:] = np.nan
        part = D.run_dlm(X[:m], Yt, hs)
        np.testing.assert_allclose(part["f"], full["f"][:, :m], atol=1e-12)


class TestLearning(unittest.TestCase):
    def test_noise_variance_is_learned_for_overlapping_targets(self):
        """Tempered NIG: S -> true variance V for every horizon (an untempered filter would learn V / h)."""
        rng = np.random.default_rng(5)
        n, p, hs = 4000, 6, (1, 5, 20)
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        sd_true = 0.8
        Y = overlapping_targets(sd_true * rng.standard_normal(n + 30), hs, n)
        out = D.run_dlm(X, Y, hs)
        for a, h in enumerate(hs):
            self.assertAlmostEqual(float(out["S"][a, 1500:].mean()), sd_true ** 2, delta=0.1, msg=f"h={h}")

    def test_no_signal_gives_tiny_forecasts_and_honest_uncertainty(self):
        rng = np.random.default_rng(6)
        n, p, hs = 5000, 15, (1, 5, 20)
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        Y = overlapping_targets(rng.standard_normal(n + 30), hs, n)
        out = D.run_dlm(X, Y, hs)
        for a, h in enumerate(hs):
            f2 = float(np.mean(out["f"][a, 1000:] ** 2))
            pu = float(np.mean(out["pu"][a, 1000:]))
            self.assertLess(f2, 0.003, f"h={h}: forecast noise too large")  # prior R^2 ~ 0.5 %
            self.assertLessEqual(f2, 1.2 * pu, f"h={h}: forecasts noisier than the reported parameter uncertainty")

    def test_recovers_constant_coefficient_with_shrinkage(self):
        rng = np.random.default_rng(7)
        n, p = 24000, 5  # long sample: the average of an estimate with a ~500-row correlation time is stable
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        beta = 0.08
        noise = rng.standard_normal(n)
        y = beta * X[:, 0] + noise
        Y = np.full((1, n), np.nan)
        Y[0, : n - 1] = y[: n - 1]
        out = D.run_dlm(X, Y, (1,))
        mh = out["m"][0, 2000:, 0]
        # exponentially weighted ridge: n_eff = 1 / (1 - delta) = 500 effective rows against a prior precision 1 / 0.025^2
        shrunk = beta * 500.0 / (500.0 + 1.0 / 0.025 ** 2)  # about 0.019
        self.assertAlmostEqual(float(mh.mean()), shrunk, delta=0.008)
        self.assertTrue(0.01 < mh.mean() < beta)  # biased towards 0 (ridge) but clearly learning
        self.assertLess(abs(float(out["m"][0, 2000:, 1].mean())), 0.02)  # irrelevant feature stays near 0

    def test_tracks_a_drifting_coefficient(self):
        rng = np.random.default_rng(8)
        n, p = 6000, 4
        X = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
        beta = 0.15 + 0.10 * np.sin(np.arange(n) / 900.0)
        y = beta * X[:, 0] + rng.standard_normal(n)
        Y = np.full((1, n), np.nan)
        Y[0, : n - 1] = y[: n - 1]
        out = D.run_dlm(X, Y, (1,))
        self.assertGreater(np.corrcoef(out["m"][0, 1500:, 0], beta[1500:])[0, 1], 0.7)

    def test_prior_only_at_the_start(self):
        X = np.column_stack([np.ones((5, 2)), np.ones(5)])
        Y = np.full((1, 5), np.nan)
        out = D.run_dlm(X, Y, (1,), n0=4.0, S0=1.0)
        np.testing.assert_array_equal(out["f"], 0.0)
        np.testing.assert_allclose(out["S"], 1.0)
        # parameter uncertainty of the mean = x' Sigma0 x = sum tau_j^2 for x = 1
        sd = D.default_prior_sd(3)
        np.testing.assert_allclose(out["pu"][0, 0], float(np.sum(sd ** 2)), rtol=1e-3)


class TestStandardizedTargets(unittest.TestCase):
    def test_hand_computed_example(self):
        cs = np.array([0.0, 0.01, 0.03, 0.02, 0.06])
        drift = np.full(5, 0.001)
        sig = np.full(5, 0.02)
        Y = D.standardized_targets(cs, drift, sig, (1, 2))
        np.testing.assert_allclose(Y[0, :4], (np.diff(cs) - 0.001) / 0.02)
        self.assertTrue(np.isnan(Y[0, 4]))
        np.testing.assert_allclose(Y[1, :3], (cs[2:] - cs[:3] - 0.002) / (0.02 * np.sqrt(2)))
        self.assertTrue(np.all(np.isnan(Y[1, 3:])))

    def test_winsorised_and_uses_origin_scale_only(self):
        cs = np.array([0.0, 0.5, 0.5, 0.5])
        Y = D.standardized_targets(cs, np.zeros(4), np.full(4, 0.01), (1,), clip=5.0)
        self.assertEqual(Y[0, 0], 5.0)  # a +50 % jump is winsorised at 5 sigma
        sig = np.array([0.01, 0.01, 0.01, 99.0])  # the scale of a LATER row must not matter
        Y2 = D.standardized_targets(cs, np.zeros(4), sig, (1,), clip=5.0)
        np.testing.assert_array_equal(Y[0, :3], Y2[0, :3])


if __name__ == "__main__":
    unittest.main()
