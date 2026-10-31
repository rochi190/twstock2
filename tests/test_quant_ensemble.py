"""Predictive primitives, exact linear opinion pool (quantile inversion, truncation) and dynamic model averaging."""
import os
import sys
import unittest

import numpy as np
from scipy import integrate, optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401

from backend.quant import ensemble as E  # noqa: E402

LIMIT = (np.log(0.9), np.log(1.1))


class TestNormal(unittest.TestCase):
    def test_matches_scipy(self):
        n = E.Normal(np.array([0.01, -0.02]), np.array([0.02, 0.05]))
        x = np.array([0.0, 0.03])
        np.testing.assert_allclose(n.cdf(x), stats.norm.cdf(x, [0.01, -0.02], [0.02, 0.05]))
        np.testing.assert_allclose(n.logpdf(x), stats.norm.logpdf(x, [0.01, -0.02], [0.02, 0.05]))
        u = np.array([0.1, 0.9])
        np.testing.assert_allclose(n.ppf(u), stats.norm.ppf(u, [0.01, -0.02], [0.02, 0.05]))
        np.testing.assert_allclose(n.cdf(np.array([[0.0, 0.01, 0.02], [0.0, 0.01, 0.02]])).shape, (2, 3))


class TestSkewT(unittest.TestCase):
    def test_symmetric_case_is_the_scaled_student_t(self):
        st = E.SkewT(0.001, 0.02, 6.0, 1.0)
        scale = 0.02 * np.sqrt(4.0 / 6.0)
        x = np.array([-0.05, -0.01, 0.001, 0.03])
        np.testing.assert_allclose(st.cdf(x), stats.t.cdf((x - 0.001) / scale, 6.0), atol=1e-12)
        np.testing.assert_allclose(np.exp(st.logpdf(x)), stats.t.pdf((x - 0.001) / scale, 6.0) / scale, rtol=1e-10)
        u = np.array([0.01, 0.3, 0.5, 0.99])
        np.testing.assert_allclose(st.ppf(u), 0.001 + scale * stats.t.ppf(u, 6.0), atol=1e-12)

    def test_prescribed_mean_std_skewness_by_quadrature(self):
        mean = np.array([0.001, -0.02, 0.0])
        var = np.array([0.0004, 0.01, 0.0025])
        st = E.SkewT.from_moments(mean, var, np.array([3.0, 1.5, 0.3]), np.array([0.0, -0.5, 0.3]))
        for i in range(3):
            f = lambda r: float(np.exp(st.logpdf(np.full(3, r))[i]))
            s = np.sqrt(var[i])
            lo, hi, pts = mean[i] - 80 * s, mean[i] + 80 * s, [mean[i]]
            m0 = integrate.quad(f, lo, hi, limit=500, points=pts)[0]
            m1 = integrate.quad(lambda r: r * f(r), lo, hi, limit=500, points=pts)[0]
            m2 = integrate.quad(lambda r: (r - m1) ** 2 * f(r), lo, hi, limit=500, points=pts)[0]
            m3 = integrate.quad(lambda r: (r - m1) ** 3 * f(r), lo, hi, limit=500, points=pts)[0]
            self.assertAlmostEqual(m0, 1.0, places=6)
            self.assertAlmostEqual(m1, mean[i], places=7)
            self.assertAlmostEqual(m2 / var[i], 1.0, places=5)
            self.assertAlmostEqual(m3 / m2 ** 1.5, [0.0, -0.5, 0.3][i], places=4)
        np.testing.assert_allclose(st.nu[:2], [6.0, 8.0])  # nu = 4 + 6 / excess kurtosis

    def test_cdf_ppf_roundtrip_and_pdf_is_derivative(self):
        st = E.SkewT(np.array([0.0, 0.01]), np.array([0.02, 0.03]), np.array([5.0, 12.0]), np.array([0.8, 1.2]))
        u = np.tile(np.array([1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-4]), (2, 1))
        np.testing.assert_allclose(st.cdf(st.ppf(u)), u, atol=1e-10)
        x = np.array([0.012, 0.02])
        h = 1e-7
        np.testing.assert_allclose(np.exp(st.logpdf(x)), (st.cdf(x + h) - st.cdf(x - h)) / (2 * h), rtol=1e-5)
        self.assertTrue(np.all(np.diff(st.ppf(u), axis=1) > 0))

    def test_skew_to_gamma_is_monotone_and_inverts(self):
        nu = np.full(5, 7.0)
        sk = np.array([-0.8, -0.3, 0.0, 0.3, 0.8])
        g = E.skew_to_gamma(nu, sk)
        self.assertTrue(np.all(np.diff(g) > 0))
        self.assertAlmostEqual(float(g[2]), 1.0, places=8)
        np.testing.assert_allclose(E._fs_stats(nu, g)[2], sk, atol=1e-8)

    def test_negative_skew_puts_the_median_above_the_mean(self):
        st = E.SkewT.from_moments(0.0, 1e-4, 1.0, -0.5)
        self.assertGreater(float(st.ppf(np.array([0.5]))[0]), 0.0)
        self.assertGreater(float(1.0 - st.cdf(np.zeros(1))[0]), 0.5)

    def test_subset(self):
        mean, std, nu, gam = np.arange(4.0), np.ones(4), np.full(4, 8.0), np.linspace(0.8, 1.2, 4)
        st = E.SkewT(mean, std, nu, gam)
        sel = np.array([True, False, True, False])
        sub = st.subset(sel)
        ref = E.SkewT(mean[sel], std[sel], nu[sel], gam[sel])
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(sub.cdf(x), ref.cdf(x), rtol=1e-14)
        np.testing.assert_allclose(sub.cdf(x), st.cdf(np.array([1.0, 0.0, 2.0, 0.0]))[[0, 2]], rtol=1e-14)  # same rows
        self.assertEqual(sub.mean.shape, (2,))
        nrm = E.Normal(mean, std).subset(sel)
        np.testing.assert_allclose(nrm.cdf(x), E.Normal(mean[sel], std[sel]).cdf(x))


def random_pool(m, seed, trunc=None):
    rng = np.random.default_rng(seed)
    prims = [E.SkewT(rng.normal(0, .002, m), rng.uniform(.01, .03, m), rng.uniform(5, 20, m), rng.uniform(.8, 1.1, m)),
             E.SkewT(rng.normal(0, .002, m), rng.uniform(.01, .03, m), rng.uniform(5, 20, m), 1.0),
             E.Normal(rng.normal(0, .002, m), rng.uniform(.01, .03, m)),
             E.Normal(rng.normal(0, .002, m), rng.uniform(.01, .03, m))]
    w = rng.dirichlet(np.ones(4), m)
    w[0] = [0, 1, 0, 0]
    w[1] = [0.5, 0, 0.5, 0]  # exact zero weights must be ignored
    return E.Pool(prims, w, trunc), prims, w


class TestPool(unittest.TestCase):
    def test_quantile_roundtrip_monotone_and_extreme_levels(self):
        pool, _, _ = random_pool(300, 1)
        u = np.tile(np.array([1e-5, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1 - 1e-5]), (300, 1))
        q = pool.ppf(u)
        np.testing.assert_allclose(pool.cdf(q), u, atol=5e-11)
        self.assertTrue(np.all(np.diff(q, axis=1) > 0))

    def test_agrees_with_an_independent_root_finder(self):
        pool, prims, w = random_pool(40, 2)
        u = np.array([0.03, 0.5, 0.91])
        q = pool.ppf(np.tile(u, (40, 1)))
        for i in range(0, 40, 7):
            for j, uu in enumerate(u):
                f = lambda x: sum(w[i, k] * float(prims[k].cdf(np.full(40, x))[i]) for k in range(4)) - uu
                ref = optimize.brentq(f, -1.0, 1.0, xtol=1e-14)
                self.assertAlmostEqual(q[i, j], ref, places=9)

    def test_mean_and_variance_formulas(self):
        pool, prims, w = random_pool(5, 3)
        for i in range(5):
            lo, hi = -2.0, 2.0
            pdf = lambda x: sum(w[i, k] * float(np.exp(prims[k].logpdf(np.full(5, x))[i])) for k in range(4))
            m1 = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=500, points=[0.0])[0]
            m2 = integrate.quad(lambda x: (x - m1) ** 2 * pdf(x), lo, hi, limit=500, points=[0.0])[0]
            self.assertAlmostEqual(pool.mean[i], m1, places=7)
            self.assertAlmostEqual(pool.var[i], m2, places=7)

    def test_logpdf_integrates_to_one_with_and_without_truncation(self):
        for trunc in (None, LIMIT):
            pool, _, _ = random_pool(4, 4, trunc)
            for i in range(4):
                f = lambda x: float(np.exp(pool.logpdf(np.full(4, x))[i]))
                lo, hi = (LIMIT if trunc else (-3.0, 3.0))
                self.assertAlmostEqual(integrate.quad(f, lo, hi, limit=500, points=[0.0])[0], 1.0, places=6)

    def test_truncation_equals_closed_form_truncated_normal(self):
        sig = 0.06
        pool = E.Pool([E.Normal(np.zeros(1), np.full(1, sig))], np.ones((1, 1)), LIMIT)
        a, b = LIMIT
        den = stats.norm.cdf(b / sig) - stats.norm.cdf(a / sig)
        x = np.array([-0.08, -0.01, 0.0, 0.02, 0.09])
        for xi in x:
            ref = (stats.norm.cdf(xi / sig) - stats.norm.cdf(a / sig)) / den
            self.assertAlmostEqual(float(pool.cdf(np.array([xi]))[0]), ref, places=12)
        self.assertEqual(float(pool.cdf(np.array([a - 0.01]))[0]), 0.0)
        self.assertEqual(float(pool.cdf(np.array([b + 0.01]))[0]), 1.0)
        q = pool.ppf(np.array([[0.0001, 0.5, 0.9999]]))
        self.assertTrue(np.all((q >= a) & (q <= b)))
        p_up = 1.0 - float(pool.cdf(np.zeros(1))[0])
        self.assertAlmostEqual(p_up, (stats.norm.cdf(b / sig) - 0.5) / den, places=12)  # asymmetric limits: slightly below 1/2
        self.assertLess(p_up, 0.5)

    def test_truncated_quantiles_roundtrip_in_a_random_pool(self):
        pool, _, _ = random_pool(200, 5, LIMIT)
        u = np.tile(np.array([0.01, 0.1, 0.5, 0.9, 0.99]), (200, 1))
        q = pool.ppf(u)
        np.testing.assert_allclose(pool.cdf(q), u, atol=5e-11)
        self.assertTrue(np.all((q >= LIMIT[0]) & (q <= LIMIT[1])))

    def test_single_primitive_pool_equals_the_primitive(self):
        sk = E.SkewT(np.array([0.001]), np.array([0.02]), np.array([6.0]), np.array([0.85]))
        pool = E.Pool([sk], np.ones((1, 1)))
        u = np.array([[0.05, 0.5, 0.95]])
        np.testing.assert_allclose(pool.ppf(u), sk.ppf(u), atol=1e-10)

    def test_mixture_logpdf_matches_direct(self):
        prims = [E.Normal(np.zeros(3), np.full(3, 0.01)), E.Normal(np.zeros(3), np.full(3, 0.03))]
        w = [np.array([0.7, 0.2, 0.5]), np.array([0.3, 0.8, 0.5])]
        x = np.array([0.0, 0.02, -0.15])
        direct = np.log(w[0] * np.exp(prims[0].logpdf(x)) + w[1] * np.exp(prims[1].logpdf(x)))
        np.testing.assert_allclose(E.mixture_logpdf(x, w, prims), direct, rtol=1e-12)


def direct_dma(scores, alpha, window, temper):
    k, n = scores.shape
    out = np.zeros((n, k))
    for t in range(n):
        logw = np.zeros(k)
        for s in range(max(0, t - window + 1), t + 1):
            for m in range(k):
                if np.isfinite(scores[m, s]):
                    logw[m] += temper * alpha ** (t - s) * scores[m, s]
        w = np.exp(logw - logw.max())
        out[t] = w / w.sum()
    return out


class TestDMA(unittest.TestCase):
    def test_matches_the_direct_formula(self):
        rng = np.random.default_rng(1)
        sc = rng.normal(0, 0.3, (4, 80))
        sc[:, :7] = np.nan
        sc[2, 30:35] = np.nan
        for alpha, window, temper in ((0.99, 30, 1.0), (0.9, 10, 0.25), (0.995, 200, 1 / 5)):
            np.testing.assert_allclose(E.dma_weights(sc, alpha, window, temper), direct_dma(sc, alpha, window, temper),
                                       rtol=1e-10, atol=1e-14)

    def test_equal_weights_without_evidence_and_normalised(self):
        sc = np.full((4, 10), np.nan)
        w = E.dma_weights(sc, 0.99, 5)
        np.testing.assert_allclose(w, 0.25)
        rng = np.random.default_rng(2)
        w2 = E.dma_weights(rng.normal(size=(3, 50)), 0.98, 20)
        np.testing.assert_allclose(w2.sum(axis=1), 1.0)
        self.assertTrue(np.all(w2 > 0))

    def test_weight_shifts_to_the_better_model_and_back(self):
        n = 400
        sc = np.zeros((2, n))
        sc[0, :200] = 0.2  # model 0 better first (log-odds saturate at 0.2 / (1 - 0.97) = 6.7)
        sc[1, 200:] = 0.2  # then model 1
        w = E.dma_weights(sc, 0.97, 100, temper=1.0)
        self.assertGreater(w[199, 0], 0.9)
        self.assertGreater(w[399, 1], 0.9)  # forgetting allows the weight to move
        self.assertAlmostEqual(w[199, 0] + w[199, 1], 1.0)

    def test_window_is_hard_and_future_cannot_matter(self):
        rng = np.random.default_rng(3)
        sc = rng.normal(size=(3, 120))
        full = E.dma_weights(sc, 0.99, 25)
        sc2 = sc.copy()
        sc2[:, :40] = 100.0  # older than the window for t >= 65
        sc2[:, 100:] += 5.0  # the future of t = 90
        pert = E.dma_weights(sc2, 0.99, 25)
        np.testing.assert_allclose(full[65:100], pert[65:100], atol=1e-12)  # window [t-24, t] excludes rows < 40 and > t

    def test_temper_scales_the_evidence(self):
        sc = np.zeros((2, 60))
        sc[0] = 0.1
        w1 = E.dma_weights(sc, 1.0, 60, temper=1.0)[-1, 0]
        w2 = E.dma_weights(sc, 1.0, 60, temper=0.25)[-1, 0]
        self.assertAlmostEqual(np.log(w1 / (1 - w1)), 6.0, places=9)  # 60 * 0.1
        self.assertAlmostEqual(np.log(w2 / (1 - w2)), 1.5, places=9)


if __name__ == "__main__":
    unittest.main()
