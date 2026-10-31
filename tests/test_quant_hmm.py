"""Gaussian HMM: exact forward-backward, EM recovery, filtering causality, h-step regime moments, labels."""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401

from backend.quant import hmm as H  # noqa: E402

A3 = np.array([[.985, .012, .003], [.02, .968, .012], [.01, .04, .95]])
MU3 = np.array([[0.0007, np.log(0.007)], [0.0002, np.log(0.012)], [-0.0015, np.log(0.025)]])
VAR3 = np.array([[0.008 ** 2, .25 ** 2], [0.012 ** 2, .25 ** 2], [0.025 ** 2, .3 ** 2]])


def truth(A=A3, mu=MU3, var=VAR3):
    D = mu.shape[1]
    return H.HMMParams(A=A, mu=mu, var=var, lo=np.full(D, -np.inf), hi=np.full(D, np.inf))


class TestForwardBackward(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        _, self.Y = H.simulate(truth(), 6, rng)
        self.p = truth()

    def brute(self):
        T, K = len(self.Y), 3
        lb = H.log_emission(self.Y, self.p.mu, self.p.var)
        pi0 = H.stationary_distribution(self.p.A)
        tot = 0.0
        post = np.zeros((T, K))
        xi = np.zeros((K, K))
        filt_last = np.zeros(K)
        for path in itertools.product(range(K), repeat=T):
            lp = np.log(pi0[path[0]]) + sum(np.log(self.p.A[path[t - 1], path[t]]) for t in range(1, T)) + \
                sum(lb[t, path[t]] for t in range(T))
            w = np.exp(lp)
            tot += w
            for t in range(T):
                post[t, path[t]] += w
            for t in range(T - 1):
                xi[path[t], path[t + 1]] += w
        return tot, post / tot, xi / tot

    def test_loglik_gamma_xi_match_enumeration(self):
        tot, post, xi = self.brute()
        B, mx = H._scale_rows(H.log_emission(self.Y, self.p.mu, self.p.var))
        pi0 = H.stationary_distribution(self.p.A)
        alpha, c = H.forward(B, self.p.A, pi0)
        beta = H.backward(B, self.p.A, c)
        self.assertAlmostEqual(float(np.sum(np.log(c)) + mx.sum()), float(np.log(tot)), places=10)
        np.testing.assert_allclose(alpha * beta, post, atol=1e-12)
        xi_sum = self.p.A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        np.testing.assert_allclose(xi_sum, xi, atol=1e-12)
        np.testing.assert_allclose(alpha.sum(axis=1), 1.0)

    def test_unrolled_k3_equals_generic(self):
        rng = np.random.default_rng(11)
        _, Y = H.simulate(truth(), 300, rng)
        B, _ = H._scale_rows(H.log_emission(Y, MU3, VAR3))
        pi0 = H.stationary_distribution(A3)
        a3, c3 = H._forward3(B, A3, pi0)
        ag = np.empty((300, 3))
        cg = np.empty(300)
        a = pi0 * B[0]
        cg[0] = a.sum()
        ag[0] = a / cg[0]
        for t in range(1, 300):
            a = (ag[t - 1] @ A3) * B[t]
            cg[t] = a.sum()
            ag[t] = a / cg[t]
        np.testing.assert_allclose(a3, ag, atol=1e-13)
        np.testing.assert_allclose(c3, cg, rtol=1e-13)
        b3 = H._backward3(B, A3, c3)
        bg = np.ones((300, 3))
        for t in range(298, -1, -1):
            bg[t] = (A3 @ (B[t + 1] * bg[t + 1])) / c3[t + 1]
        np.testing.assert_allclose(b3, bg, atol=1e-10, rtol=1e-10)

    def test_stationary_distribution(self):
        pi = H.stationary_distribution(A3)
        np.testing.assert_allclose(pi @ A3, pi, atol=1e-12)
        self.assertAlmostEqual(pi.sum(), 1.0)


class TestEM(unittest.TestCase):
    def test_two_state_sticky_recovery(self):
        A = np.array([[0.98, 0.02], [0.04, 0.96]])
        mu = np.array([[0.0006, np.log(0.008)], [-0.001, np.log(0.022)]])
        var = np.array([[0.008 ** 2, .25 ** 2], [0.022 ** 2, .25 ** 2]])
        tp = truth(A, mu, var)
        x, Y = H.simulate(tp, 3000, np.random.default_rng(21))
        fit = H.fit_hmm(Y[:2000], K=2)
        self.assertEqual(fit.K, 2)
        np.testing.assert_allclose(np.sqrt(fit.var[:, 0]), [0.008, 0.022], rtol=0.15)
        np.testing.assert_allclose(np.exp(fit.mu[:, 1]), [0.008 * 0.85 / 0.85, 0.022 * 1.0], rtol=0.2)
        self.assertGreater(fit.A[0, 0], 0.95)
        self.assertGreater(fit.A[1, 1], 0.92)
        self.assertLess(np.sqrt(fit.var[0, 0]), np.sqrt(fit.var[1, 0]))  # sorted by variance
        fp = H.filter_probs(fit, Y[2000:], fit.alpha_last)
        self.assertGreater(float(np.mean(fp.argmax(axis=1) == x[2000:])), 0.88)

    def test_three_state_reaches_the_truth_optimum(self):
        for seed in (4, 5, 6):
            x, Y = H.simulate(truth(), 1000, np.random.default_rng(seed))
            fit = H.fit_hmm(Y)
            ref = H.fit_hmm(Y, init=truth())  # EM started from the truth
            self.assertGreater(fit.loglik, ref.loglik - 3.0, f"seed {seed}: cold fit stuck in a poor local optimum")
            self.assertTrue(np.all(np.diff(fit.var[:, 0]) > 0))  # sorted by return variance
            np.testing.assert_allclose(fit.A.sum(axis=1), 1.0)

    def test_fit_is_deterministic_and_warm_start_consistent(self):
        _, Y = H.simulate(truth(), 900, np.random.default_rng(8))
        a, b = H.fit_hmm(Y), H.fit_hmm(Y)
        np.testing.assert_array_equal(a.A, b.A)
        np.testing.assert_array_equal(a.mu, b.mu)
        warm = H.fit_hmm(Y, init=a)
        self.assertAlmostEqual(warm.loglik, a.loglik, delta=0.5)

    def test_log_vol_winsorised_and_return_not(self):
        _, Y = H.simulate(truth(), 800, np.random.default_rng(9))
        Y[5, 1] = -50.0  # absurd log-vol outlier (e.g. zero-range bar)
        Y[7, 0] = 0.3
        fit = H.fit_hmm(Y)
        self.assertTrue(np.isfinite(fit.lo[1]) and fit.lo[1] > -50)
        self.assertTrue(np.isinf(fit.lo[0]) and np.isinf(fit.hi[0]))
        self.assertTrue(np.all(np.isfinite(fit.mu)) and np.all(np.isfinite(fit.var)))

    def test_too_few_observations(self):
        with self.assertRaises(ValueError):
            H.fit_hmm(np.zeros((30, 2)), K=3)


class TestFilteringIsCausal(unittest.TestCase):
    def test_filtered_probabilities_ignore_the_future(self):
        x, Y = H.simulate(truth(), 500, np.random.default_rng(12))
        p = truth()
        full = H.filter_probs(p, Y)
        Y2 = Y.copy()
        Y2[300:] = Y2[300:] * 3.0 + 0.01  # corrupt everything after row 299
        corrupted = H.filter_probs(p, Y2)
        np.testing.assert_array_equal(full[:300], corrupted[:300])
        self.assertGreater(np.abs(full[300:] - corrupted[300:]).max(), 1e-3)

    def test_continuation_equals_one_pass(self):
        _, Y = H.simulate(truth(), 400, np.random.default_rng(13))
        p = truth()
        one = H.filter_probs(p, Y)
        head = H.filter_probs(p, Y[:250])
        tail = H.filter_probs(p, Y[250:], alpha_prev=head[-1])
        np.testing.assert_allclose(np.vstack([head, tail]), one, atol=1e-12)

    def test_alpha_last_matches_filter(self):
        _, Y = H.simulate(truth(), 700, np.random.default_rng(14))
        fit = H.fit_hmm(Y[:500])
        fp = H.filter_probs(fit, Y[:500])
        np.testing.assert_allclose(fit.alpha_last, fp[-1], atol=1e-9)


class TestHorizonMoments(unittest.TestCase):
    def test_exact_vs_path_enumeration(self):
        p = truth(A=np.array([[.8, .2], [.3, .7]]), mu=np.array([[0.002, 0.0], [-0.003, 0.0]]),
                  var=np.array([[0.01 ** 2, 1.0], [0.03 ** 2, 1.0]]))
        mean, var = H.hstep_state_moments(p, 4)
        A = p.A
        for start in range(2):
            for h in (1, 2, 4):
                m1 = m2 = 0.0
                for path in itertools.product(range(2), repeat=h):
                    pr, prev = 1.0, start
                    for s in path:
                        pr *= A[prev, s]
                        prev = s
                    mu_sum = sum(p.mu[s, 0] for s in path)
                    v_sum = sum(p.var[s, 0] for s in path)
                    m1 += pr * mu_sum
                    m2 += pr * (v_sum + mu_sum ** 2)
                self.assertAlmostEqual(mean[h, start], m1, places=12)
                self.assertAlmostEqual(var[h, start], m2 - m1 ** 2, places=12)

    def test_vs_monte_carlo_three_states(self):
        p = truth()
        mean, var = H.hstep_state_moments(p, 20)
        rng = np.random.default_rng(15)
        n = 150000
        cum = np.cumsum(p.A, axis=1)
        for start in (0, 2):
            st = np.full(n, start)
            S = np.zeros(n)
            for _ in range(20):
                st = np.minimum((rng.random(n)[:, None] > cum[st]).sum(1), 2)
                S += p.mu[st, 0] + np.sqrt(p.var[st, 0]) * rng.standard_normal(n)
            self.assertAlmostEqual(var[20, start] / S.var(), 1.0, delta=0.03)
            self.assertAlmostEqual(mean[20, start], S.mean(), delta=4 * S.std() / np.sqrt(n))


class TestLabels(unittest.TestCase):
    SD = np.array([0.01, 0.015, 0.03])
    VAR = np.array([[0.01 ** 2, 1.0], [0.015 ** 2, 1.0], [0.03 ** 2, 1.0]])
    NEUTRAL = ["低波動盤整", "中波動震盪", "高波動震盪"]

    def with_ratios(self, ratios):
        """3-state HMM whose regime k has drift / volatility ``mu_k / sigma_k == ratios[k]`` (return dimension)."""
        mu = np.column_stack([np.asarray(ratios, float) * self.SD, np.zeros(3)])
        return truth(np.full((3, 3), 1 / 3), mu, self.VAR)

    def test_labels_follow_fitted_values_not_indices(self):
        """The fixtures used mu / sigma = 0.05 .. 0.10 (a "trend" at the pre-audit threshold of 0.04).  Audit M-06 moved the
        default threshold to 0.10 because the engine's shrunk regime means produce exactly that range as noise, so the
        bull / bear fixtures now carry a drift that clearly stands out (0.14 .. 0.16 per unit of volatility)."""
        bull = self.with_ratios([0.15, 0.14, 0.16])
        bear = self.with_ratios([-0.15, -0.14, -0.16])
        flat = self.with_ratios([0.0, 0.0, 0.0])
        lb, le, lf = H.regime_labels(bull), H.regime_labels(bear), H.regime_labels(flat)
        self.assertEqual(lb, ["低波動多頭", "多頭穩定", "高波動反彈"])
        self.assertEqual(le, ["低波動緩跌", "空頭修正", "高波動空頭"])
        self.assertEqual(lf, self.NEUTRAL)
        for lab in (lb, le, lf):
            self.assertEqual(len(set(lab)), 3)

    def test_shrunk_regime_means_keep_the_neutral_words(self):
        """Audit M-06: with ``hmm_mean_kappa = 500`` every regime mean is the shrunk drift plus a small deviation, so the calm
        regime's ratio moves between about -0.02 and +0.09 from refit to refit (63 stocks, 3,820 development refits: mean
        0.046, sd 0.035, 95th percentile 0.10).  The old threshold 0.04 sat in the middle of that range and the label flipped
        between 低波動盤整 and 低波動多頭 on noise; nothing in that range may change a word any more."""
        for r0 in (-0.05, -0.02, 0.0, 0.03, 0.039, 0.041, 0.046, 0.06, 0.09):
            p = self.with_ratios([r0, 0.5 * r0, 0.4 * r0])
            self.assertEqual(H.regime_labels(p), self.NEUTRAL, f"calm-regime ratio {r0}")

    def test_trend_word_needs_a_ratio_above_the_threshold(self):
        self.assertGreater(H.TREND_THRESHOLD, 0.09)  # above the shrunk range ...
        self.assertLess(H.TREND_THRESHOLD, 0.13)  # ... but a regime that really stands out is still named
        p = self.with_ratios([0.13, -0.13, 0.13])
        self.assertEqual(H.regime_labels(p), ["低波動多頭", "空頭修正", "高波動反彈"])
        near = self.with_ratios([0.06, 0.03, 0.02])
        self.assertEqual(H.regime_labels(near)[0], "低波動盤整")
        self.assertEqual(H.regime_labels(near, trend_threshold=0.04)[0], "低波動多頭")  # the pre-audit setting is still reachable

    def test_labels_unique_and_two_state_names(self):
        two = H.HMMParams(A=np.full((2, 2), 0.5), mu=np.zeros((2, 2)), var=self.VAR[[0, 2]], lo=np.full(2, -np.inf),
                          hi=np.full(2, np.inf))
        self.assertEqual(H.regime_labels(two), ["低波動盤整", "高波動震盪"])

    def test_labels_from_a_real_fit(self):
        _, Y = H.simulate(truth(), 1200, np.random.default_rng(16))
        labels = H.regime_labels(H.fit_hmm(Y))
        self.assertEqual(len(labels), 3)
        self.assertTrue(labels[0].startswith("低波動") and labels[2].startswith("高波動"))


class TestInitRobustness(unittest.TestCase):
    def test_rare_high_volatility_regime_is_found(self):
        """Equal-count terciles trapped EM in a poor optimum when the crisis state is rare; the GMM start finds it."""
        A = np.array([[0.99, 0.008, 0.002], [0.03, 0.96, 0.01], [0.02, 0.05, 0.93]])
        tp = truth(A=A)
        gaps = []
        for seed in (3, 22, 41):
            x, Y = H.simulate(tp, 1000, np.random.default_rng(seed))
            fit = H.fit_hmm(Y)
            gaps.append(H.fit_hmm(Y, init=tp).loglik - fit.loglik)
        self.assertLess(max(gaps), 3.0)


if __name__ == "__main__":
    unittest.main()
