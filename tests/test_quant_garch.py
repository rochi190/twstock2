"""GJR-GARCH(1,1)-t: filter, MLE recovery, closed-form h-step moments vs Monte Carlo, two-piece-t predictive."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ROOT  # noqa: F401  (adds the project root to sys.path)

from backend.quant import ensemble as E  # noqa: E402
from backend.quant import garch as G  # noqa: E402

V = 0.016 ** 2


def params(alpha=0.03, gamma=0.08, beta=0.90, nu=6.0, v=V, mean=0.0):
    phi = alpha + beta + gamma / 2
    return G.GarchParams(omega=(1 - phi) * v, alpha=alpha, gamma=gamma, beta=beta, nu=nu, mean=mean)


class TestFilter(unittest.TestCase):
    def test_lfilter_equals_explicit_loop(self):
        rng = np.random.default_rng(1)
        p = params()
        eps = 0.015 * rng.standard_t(5, 700)
        s2 = G.variance_filter(eps, p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)
        ref = np.empty(len(eps) + 1)
        ref[0] = p.uncond_var
        for t, e in enumerate(eps):
            ref[t + 1] = p.omega + (p.alpha + (p.gamma if e < 0 else 0.0)) * e * e + p.beta * ref[t]
        np.testing.assert_allclose(s2, ref, rtol=1e-12, atol=0)

    def test_filter_is_causal(self):
        rng = np.random.default_rng(2)
        p = params()
        eps = 0.015 * rng.standard_normal(400)
        full = G.variance_filter(eps, p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)
        short = G.variance_filter(eps[:250], p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)
        np.testing.assert_array_equal(full[:251], short)

    def test_leverage_raises_variance_after_negative_shock(self):
        p = params(gamma=0.1)
        down = G.variance_filter(np.array([-0.03]), p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)[1]
        up = G.variance_filter(np.array([0.03]), p.omega, p.alpha, p.gamma, p.beta, p.uncond_var)[1]
        self.assertAlmostEqual(down - up, p.gamma * 0.03 ** 2, places=15)


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.true = params(alpha=0.03, gamma=0.08, beta=0.90, nu=6.0)
        cls.eps = [G.simulate(cls.true, 5000, np.random.default_rng(100 + k)) for k in range(4)]

    def test_parameter_recovery(self):
        est = np.array([[f.persistence, f.nu, f.gamma, f.alpha + f.gamma / 2, f.uncond_var]
                        for f in (G.fit_gjr_garch_t(e) for e in self.eps)])
        m = est.mean(axis=0)
        self.assertAlmostEqual(m[0], self.true.persistence, delta=0.012)  # persistence 0.97
        self.assertAlmostEqual(m[1], 6.0, delta=1.2)  # degrees of freedom
        self.assertGreater(m[2], 0.03)  # leverage detected
        self.assertAlmostEqual(m[3], self.true.alpha + self.true.gamma / 2, delta=0.02)
        self.assertAlmostEqual(m[4] / V, 1.0, delta=0.2)  # variance targeting level = sample variance

    def test_constraints_and_variance_targeting(self):
        for e in self.eps:
            p = G.fit_gjr_garch_t(e)
            self.assertGreaterEqual(p.alpha, 0)
            self.assertGreaterEqual(p.gamma, 0)
            self.assertGreaterEqual(p.beta, 0)
            self.assertLess(p.persistence, 1.0)
            self.assertGreaterEqual(p.nu, G.NU_MIN)
            self.assertAlmostEqual(p.uncond_var, float(np.mean(e * e)), delta=1e-12)

    def test_deterministic_and_target_override(self):
        a = G.fit_gjr_garch_t(self.eps[0], mean=0.0002)
        b = G.fit_gjr_garch_t(self.eps[0], mean=0.0002)
        self.assertEqual(a, b)
        c = G.fit_gjr_garch_t(self.eps[0], mean=0.0002, target_var=2e-4)
        self.assertAlmostEqual(c.uncond_var, 2e-4, delta=1e-15)

    def test_loglik_not_worse_than_start(self):
        e = self.eps[1]
        p = G.fit_gjr_garch_t(e)
        start = G.GarchParams(omega=(1 - 0.97) * np.mean(e * e), alpha=0.03, gamma=0.07, beta=0.90, nu=7.0)
        self.assertGreaterEqual(p.loglik, G.loglik(e, start, s2_0=np.mean(e * e)) - 1e-6)

    def test_ew_variance(self):
        e = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(G.ew_variance(e, None), np.mean(e ** 2))
        w = 0.5 ** (np.array([2.0, 1.0, 0.0]) / 1.0)
        self.assertAlmostEqual(G.ew_variance(e, 1.0), np.sum(w * e ** 2) / w.sum())

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            G.fit_gjr_garch_t(np.zeros(10))


class TestHstep(unittest.TestCase):
    def test_variance_matches_brute_force_recursion(self):
        p = params()
        for mult in (0.4, 1.0, 3.0):
            u1 = p.uncond_var * mult
            for h in (1, 2, 5, 20):
                u, tot = u1, 0.0
                for _ in range(h):  # E[u_{k+1}] = omega + phi E[u_k]
                    tot += u
                    u = p.omega + p.persistence * u
                self.assertAlmostEqual(float(G.hstep_variance(u1, p, h)) / tot, 1.0, places=12)
                self.assertAlmostEqual(float(G.hstep_moments(u1, p, h)[0][0]) / tot, 1.0, places=12)

    def test_horizon_one_is_the_fitted_innovation(self):
        p = params(nu=7.0)
        var, kurt, skew = G.hstep_moments(np.array([2e-4]), p, 1)
        self.assertAlmostEqual(var[0], 2e-4)
        self.assertAlmostEqual(kurt[0], 6.0 / (7.0 - 4.0), places=12)
        self.assertAlmostEqual(skew[0], 0.0, places=12)

    def test_iid_limit_kurtosis_falls_like_one_over_h(self):
        p = G.GarchParams(omega=2.5e-4, alpha=0.0, gamma=0.0, beta=0.0, nu=7.0)  # no dynamics
        k1 = 6.0 / 3.0
        for h in (2, 5, 20):
            _, k, s = G.hstep_moments(np.array([2.5e-4]), p, h)
            self.assertAlmostEqual(k[0], k1 / h, places=10)
            self.assertAlmostEqual(s[0], 0.0, places=12)

    def test_moments_vs_monte_carlo(self):
        """Variance (exact), skewness and excess kurtosis (closed forms) against simulation of the model itself."""
        n = 200000
        for nu, rel in ((10.0, 0.20), (6.0, 0.35)):  # sample kurtosis of heavy-tailed sums is noisy: relative tolerance
            p = params(nu=nu)
            for mult in (0.6, 2.5):
                u1 = p.uncond_var * mult
                for h in (5, 20):
                    k_mc, s_mc, v_mc = [], [], []
                    for seed in (7, 8, 9):
                        z = G.standardized_t(np.random.default_rng(seed), nu, (n, h))
                        S = G.simulate_paths(p, u1, z).sum(axis=1)
                        zc = (S - S.mean()) / S.std()
                        k_mc.append((zc ** 4).mean() - 3)
                        s_mc.append((zc ** 3).mean())
                        v_mc.append(S.var())
                    var, kurt, skew = (a[0] for a in G.hstep_moments(u1, p, h))
                    self.assertAlmostEqual(var / np.mean(v_mc), 1.0, delta=0.012)
                    self.assertAlmostEqual(skew, float(np.mean(s_mc)), delta=0.05)
                    self.assertLess(skew, 0.0)  # leverage -> negative skew
                    self.assertLess(abs(kurt - np.mean(k_mc)), 0.1 + rel * abs(np.mean(k_mc)),
                                    f"nu={nu} mult={mult} h={h}: analytic {kurt:.3f} vs MC {np.mean(k_mc):.3f}")

    def test_skewed_t_predictive_matches_mc_centre_and_p_up(self):
        """The damped two-piece t (engine setting) reproduces P(S>0) and the 10-90 % quantiles of the true aggregate."""
        n = 300000
        damp = 0.6
        for nu in (6.0, 10.0):
            p = params(nu=nu)
            for h in (5, 20):
                u1 = p.uncond_var
                z = G.standardized_t(np.random.default_rng(3), nu, (n, h))
                S = G.simulate_paths(p, u1, z).sum(axis=1)
                var, kurt, skew = G.hstep_moments(u1, p, h)
                st = E.SkewT.from_moments(0.0, var, kurt, damp * skew)
                lv = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
                qa = st.ppf(lv[None, :])[0] / np.sqrt(var[0])
                qm = np.quantile(S, lv) / np.sqrt(var[0])
                self.assertLess(np.abs(qa - qm).max(), 0.035)
                p_up = 1.0 - st.cdf(np.zeros(1))[0]
                self.assertAlmostEqual(p_up, float(np.mean(S > 0)), delta=0.004)
                # a symmetric predictive would be biased by 0.8-2 percentage points here
                self.assertGreater(abs(0.5 - float(np.mean(S > 0))), 0.006)

    def test_vectorised_over_origins(self):
        p = params()
        u1 = np.array([1e-4, 2e-4, 5e-4])
        v, k, s = G.hstep_moments(u1, p, 5)
        for i in range(3):
            vi, ki, si = G.hstep_moments(u1[i:i + 1], p, 5)
            self.assertEqual((v[i], k[i], s[i]), (vi[0], ki[0], si[0]))

    def test_student_t_params_from_moments(self):
        var = np.array([1e-4, 1e-4, 1e-4])
        scale, nu = G.student_t_params_from_moments(var, np.array([3.0, 0.5, -1.0]))
        np.testing.assert_allclose(nu[:2], [6.0, 16.0])
        self.assertEqual(nu[2], 200.0)  # non-positive excess kurtosis -> (almost) Gaussian
        np.testing.assert_allclose(scale ** 2 * nu / (nu - 2), var)  # variance preserved


class TestSimulation(unittest.TestCase):
    def test_simulated_series_has_target_moments(self):
        p = params()
        e = G.simulate(p, 60000, np.random.default_rng(5))
        self.assertAlmostEqual(float(np.var(e)) / V, 1.0, delta=0.12)
        z = e / e.std()
        self.assertGreater((z ** 4).mean() - 3, 1.0)  # fat tails

    def test_paths_respect_return_bounds_and_feedback(self):
        p = params()
        z = G.standardized_t(np.random.default_rng(6), p.nu, (4000, 10)) * 8.0  # absurd shocks
        eps = G.simulate_paths(p, V, z, ret_bounds=(np.log(0.9), np.log(1.1)))
        r = p.mean + eps
        self.assertLessEqual(r.max(), np.log(1.1) + 1e-12)
        self.assertGreaterEqual(r.min(), np.log(0.9) - 1e-12)


if __name__ == "__main__":
    unittest.main()
