"""Tests for backend/evaluation/metrics.py.  Every statistic is compared with an independent computation
(scipy reference, explicit loops, or a hand-worked example)."""
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
from scipy import stats  # noqa: E402

from backend import config  # noqa: E402
from backend.evaluation import metrics as M  # noqa: E402


# ----------------------------------------------------------------------------- independent reference code
def ref_binom_sf(k, n, p):
    """P(X >= k) by direct summation of the binomial pmf (log-space, no scipy)."""
    if p <= 0.0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k <= n else 0.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k, n + 1))


def ref_nw_lrv(d, lag):
    """Newey-West long-run variance with explicit double loops."""
    m = len(d)
    mean = sum(d) / m
    x = [v - mean for v in d]
    g = [sum(x[t] * x[t - k] for t in range(k, m)) / m for k in range(lag + 1)]
    return g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))


def ref_kupiec(x, n, p):
    def ll(pp):
        t = 0.0
        if n - x > 0:
            t += (n - x) * math.log(1 - pp)
        if x > 0:
            t += x * math.log(pp)
        return t
    return -2 * (ll(p) - ll(x / n))


class TestHelpers(unittest.TestCase):
    def test_json_safe_strict(self):
        obj = {"a": np.float64("nan"), "b": [np.inf, -np.inf, 1.5, np.int64(3)], "c": np.array([1.0, np.nan]),
               "d": np.datetime64("2024-03-05"), 7: np.bool_(True), "e": np.array(["2020-01-01"], dtype="datetime64[D]")}
        out = M.json_safe(obj)
        text = json.dumps(out, allow_nan=False)  # must not raise
        self.assertIsNone(out["a"])
        self.assertEqual(out["b"], [None, None, 1.5, 3])
        self.assertEqual(out["c"], [1.0, None])
        self.assertEqual(out["d"], "2024-03-05")
        self.assertEqual(out["7"], True)
        self.assertEqual(out["e"], ["2020-01-01"])
        self.assertIn("2024-03-05", text)
        self.assertEqual(M.json_safe(1.23456789, 3), 1.235)

    def test_finite_or_none(self):
        self.assertIsNone(M.finite_or_none(float("nan")))
        self.assertIsNone(M.finite_or_none(None))
        self.assertIsNone(M.finite_or_none("x"))
        self.assertEqual(M.finite_or_none(np.float32(2.0)), 2.0)

    def test_downsample_keeps_first_last_and_cap(self):
        for n in (0, 1, 2, 599, 600, 601, 1250, 5000):
            ix = M.downsample_indices(n)
            self.assertLessEqual(len(ix), 600)
            if n:
                self.assertEqual(ix[0], 0)
                self.assertEqual(ix[-1], n - 1)
                self.assertTrue(np.all(np.diff(ix) > 0))
            else:
                self.assertEqual(len(ix), 0)
        self.assertEqual(len(M.downsample_indices(600)), 600)
        self.assertEqual(len(M.downsample_indices(3000)), 600)

    def test_date_str(self):
        self.assertEqual(M.date_str(np.datetime64("2026-10-22")), "2026-10-22")
        self.assertIsNone(M.date_str(np.datetime64("NaT")))
        self.assertIsNone(M.date_str(None))


class TestBaselines(unittest.TestCase):
    def test_realized_log_returns(self):
        a = np.array([100.0, 101.0, 99.0, 102.0, 104.0])
        r = M.realized_log_returns(a, 2)
        np.testing.assert_allclose(r[:3], [math.log(99 / 100), math.log(102 / 101), math.log(104 / 99)])
        self.assertTrue(np.all(np.isnan(r[3:])))
        self.assertTrue(np.all(np.isnan(M.realized_log_returns(a, 9))))
        bad = M.realized_log_returns(np.array([1.0, 0.0, 2.0]), 1)
        self.assertTrue(np.isnan(bad[0]) and np.isfinite(bad[1]) is np.False_ or np.isnan(bad[1]))

    def test_expanding_up_rate_hand_example(self):
        R = np.array([0.1, -0.2, 0.0, 0.3, 0.1, -0.1, np.nan, np.nan])
        c = M.expanding_up_rate(R, h=1)
        # row i uses outcomes j <= i-1 with R != 0 : (ups + 1) / (n + 2)
        expect = [0.5, (1 + 1) / (1 + 2), (1 + 1) / (2 + 2), (1 + 1) / (2 + 2), (2 + 1) / (3 + 2),
                  (3 + 1) / (4 + 2), (3 + 1) / (5 + 2), (3 + 1) / (5 + 2)]
        np.testing.assert_allclose(c, expect)

    def test_expanding_up_rate_flat_outcomes_as_not_up(self):
        R = np.array([0.1, -0.2, 0.0, 0.3, 0.1, -0.1, np.nan, np.nan])
        c = M.expanding_up_rate(R, h=1, ties="not_up")
        # row i: origins j <= i-1 with a finite outcome, the flat one counts as a trial that is not up: (ups + 1) / (n + 2)
        expect = [0.5, (1 + 1) / (1 + 2), (1 + 1) / (2 + 2), (1 + 1) / (3 + 2), (2 + 1) / (4 + 2),
                  (3 + 1) / (5 + 2), (3 + 1) / (6 + 2), (3 + 1) / (6 + 2)]
        np.testing.assert_allclose(c, expect)
        np.testing.assert_allclose(M.expanding_up_rate(R, h=1, ties="drop"), M.expanding_up_rate(R, h=1))   # default
        self.assertFalse(np.allclose(c, M.expanding_up_rate(R, h=1)))
        # no flat outcomes: the two definitions coincide
        R2 = np.array([0.1, -0.2, 0.3, 0.1, -0.1, 0.2])
        np.testing.assert_allclose(M.expanding_up_rate(R2, 2, ties="not_up"), M.expanding_up_rate(R2, 2))
        with self.assertRaises(ValueError):
            M.expanding_up_rate(R, 1, ties="maybe")

    def test_expanding_up_rate_not_up_is_causal(self):
        rng = np.random.default_rng(11)
        a = 100 * np.exp(np.cumsum(np.round(rng.normal(0, 0.01, 400), 2)))        # rounded steps -> some exact ties
        for h in (1, 5):
            R = M.realized_log_returns(a, h)
            base = M.expanding_up_rate(R, h, ties="not_up")
            for i in (60, 150, 300):
                R2 = M.realized_log_returns(a[: i + 1], h)                         # data after origin i does not exist
                np.testing.assert_allclose(M.expanding_up_rate(R2, h, ties="not_up")[i], base[i])
            # poisoning the future: replace every price after i by garbage, the reference at i is unchanged
            b = a.copy()
            b[201:] = rng.uniform(50, 150, len(b) - 201)
            Rb = M.realized_log_returns(b, h)
            np.testing.assert_allclose(M.expanding_up_rate(Rb, h, ties="not_up")[:201], base[:201])

    def test_expanding_baselines_are_causal(self):
        rng = np.random.default_rng(1)
        a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        for h in (1, 5):
            R = M.realized_log_returns(a, h)
            base_up, base_mean = M.expanding_up_rate(R, h), M.expanding_mean_return(R, h)
            for i in (50, 120, 300):
                # the value at origin i may only depend on prices up to bar i
                R2 = M.realized_log_returns(a[: i + 1], h)
                np.testing.assert_allclose(M.expanding_up_rate(R2, h)[i], base_up[i])
                np.testing.assert_allclose(M.expanding_mean_return(R2, h)[i], base_mean[i])

    def test_expanding_mean_return(self):
        R = np.array([0.1, 0.3, -0.2, np.nan])
        m = M.expanding_mean_return(R, 1)
        self.assertTrue(np.isnan(m[0]))
        np.testing.assert_allclose(m[1:], [0.1, 0.2, (0.1 + 0.3 - 0.2) / 3])

    def test_trailing_sign(self):
        a = np.array([10.0, 11.0, 10.5, 10.5, 12.0])
        pred, valid = M.trailing_sign_prediction(a, 1)
        self.assertEqual(valid.tolist(), [False, True, True, True, True])
        self.assertEqual(pred.tolist(), [False, True, False, False, True])  # flat -> not up
        pred2, valid2 = M.trailing_sign_prediction(a, 2)
        self.assertEqual(valid2.tolist(), [False, False, True, True, True])
        self.assertEqual(pred2.tolist(), [False, False, True, False, True])


class TestDirection(unittest.TestCase):
    P = np.array([0.62, 0.58, 0.55, 0.52, 0.51, 0.49, 0.45, 0.41, 0.35, 0.65])
    UP = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1], dtype=bool)
    PREV = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)

    def summary(self, **kw):
        n = len(self.P)
        base = {"always_up": (np.ones(n, bool), np.ones(n, bool)), "prev_sign": (self.PREV, np.ones(n, bool)),
                "train_majority": (np.ones(n, bool), np.ones(n, bool))}
        return M.direction_summary(self.P, self.UP, base, **kw)

    def test_wilson_matches_scipy(self):
        for k, n in [(0, 10), (10, 10), (7, 10), (536, 1000), (1, 3), (55, 100), (2, 2000)]:
            lo, hi = M.wilson_interval(k, n)
            ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            self.assertAlmostEqual(lo, ref.low, places=10)
            self.assertAlmostEqual(hi, ref.high, places=10)
        lo, hi = M.wilson_interval(8, 10, 0.9)
        ref = stats.binomtest(8, 10).proportion_ci(confidence_level=0.9, method="wilson")
        self.assertAlmostEqual(lo, ref.low, places=10)
        self.assertTrue(all(math.isnan(v) for v in M.wilson_interval(0, 0)))

    def test_binom_p_matches_direct_sum(self):
        for k, n, p in [(7, 10, 0.5), (60, 100, 0.55), (0, 5, 0.3), (5, 5, 0.9), (663, 1236, 0.55)]:
            self.assertAlmostEqual(M.binom_p_greater(k, n, p), ref_binom_sf(k, n, p), places=10)
        self.assertTrue(math.isnan(M.binom_p_greater(1, 0, 0.5)))
        self.assertTrue(math.isnan(M.binom_p_greater(1, 5, 1.5)))

    def test_hand_example(self):
        s = self.summary(h=1)
        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["accuracy"], 0.7)
        self.assertAlmostEqual(s["base_rate_up"], 0.5)
        self.assertAlmostEqual(s["baselines"]["always_up"], 0.5)
        self.assertAlmostEqual(s["baselines"]["prev_sign"], 0.2)
        self.assertAlmostEqual(s["baselines"]["train_majority"], 0.5)
        self.assertEqual(s["best_baseline"], "always_up")  # tie -> first in canonical order
        self.assertAlmostEqual(s["p_value_vs_best_baseline"], 176 / 1024)  # P(X>=7 | 10, .5)
        self.assertAlmostEqual(s["p_value_vs_coin"], 176 / 1024)
        lo, hi = s["ci95"]
        ref = stats.binomtest(7, 10).proportion_ci(method="wilson")
        self.assertAlmostEqual(lo, ref.low)
        self.assertAlmostEqual(hi, ref.high)

    def test_buckets_hand_example(self):
        s = self.summary(h=1)
        by = {b["bucket"]: b for b in s["by_confidence"]}
        b = by["[0.55,0.60)"]
        self.assertEqual((b["n"], b["accuracy"], b["actual_up_rate"]), (2, 0.5, 0.5))
        self.assertAlmostEqual(b["mean_p_up"], 0.565)
        b = by["[0.50,0.55)"]
        self.assertEqual((b["n"], b["accuracy"]), (2, 0.5))
        self.assertAlmostEqual(b["mean_p_up"], 0.515)
        b = by["[0.40,0.45)"]
        self.assertEqual((b["n"], b["accuracy"], b["actual_up_rate"]), (1, 0.0, 1.0))
        self.assertEqual(sum(b["n"] for b in s["by_confidence"]), 10)
        self.assertEqual([b["lo"] for b in s["by_confidence"]], sorted(b["lo"] for b in s["by_confidence"]))

    def test_high_conf_hand_example(self):
        hc = self.summary(h=1)["high_conf"]
        self.assertEqual(hc["n"], 3)  # p = 0.62, 0.35, 0.65
        self.assertAlmostEqual(hc["accuracy"], 1.0)
        self.assertAlmostEqual(hc["p_value"], 0.125)  # P(X>=3 | 3, .5), null mix = base rate 0.5

    def test_high_conf_null_uses_call_mix(self):
        # all 'up' calls, base rate 0.8 -> null accuracy is 0.8, not 0.5
        p = np.full(50, 0.7)
        up = np.array([1] * 42 + [0] * 8, dtype=bool)
        s = M.direction_summary(p, up, {"always_up": (np.ones(50, bool), np.ones(50, bool))}, h=1, high_conf=0.6)
        self.assertEqual(s["high_conf"]["n"], 50)
        self.assertAlmostEqual(s["high_conf"]["p_value"], ref_binom_sf(42, 50, 0.84), places=10)

    def test_overlap_uses_effective_sample(self):
        rng = np.random.default_rng(3)
        n = 100
        p = rng.uniform(0.3, 0.7, n)
        up = rng.random(n) < p
        one = {"always_up": (np.ones(n, bool), np.ones(n, bool))}
        s1 = M.direction_summary(p, up, one, h=1)
        s5 = M.direction_summary(p, up, one, h=5)
        k = int(((p >= 0.5) == up).sum())
        self.assertAlmostEqual(s1["p_value_vs_coin"], ref_binom_sf(k, n, 0.5), places=10)
        k_eff, n_eff = int(round(k / n * 20)), 20
        self.assertAlmostEqual(s5["p_value_vs_coin"], ref_binom_sf(k_eff, n_eff, 0.5), places=10)
        self.assertEqual(s5["n"], n)  # reported n stays the raw count
        w1, w5 = s1["ci95"][1] - s1["ci95"][0], s5["ci95"][1] - s5["ci95"][0]
        self.assertGreater(w5, w1)  # fewer effective observations -> wider interval

    def test_empty_and_missing_baselines(self):
        s = M.direction_summary(np.array([]), np.array([], dtype=bool), {}, h=1)
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["accuracy"])
        self.assertEqual(s["by_confidence"], [])
        s = M.direction_summary(np.array([0.6, 0.4]), np.array([True, False]), {}, h=1)
        self.assertIsNone(s["best_baseline"])
        self.assertIsNone(s["p_value_vs_best_baseline"])
        self.assertAlmostEqual(s["accuracy"], 1.0)

    def test_matching_base_rate_has_no_significance(self):
        # always forecasting 'up' when 55 % of outcomes are up: identical to the always_up baseline
        up = np.array([True] * 55 + [False] * 45)
        s = M.direction_summary(np.full(100, 0.55), up, {"always_up": (np.ones(100, bool), np.ones(100, bool))}, h=1)
        self.assertAlmostEqual(s["accuracy"], 0.55)
        self.assertGreater(s["p_value_vs_best_baseline"], 0.4)


class TestProbabilistic(unittest.TestCase):
    def test_brier_logloss_hand(self):
        p = np.array([0.1, 0.4, 0.6, 0.9])
        y = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(M.brier_score(p, y), 0.135)
        ref = -(math.log(0.9) + math.log(0.4) + math.log(0.6) + math.log(0.9)) / 4
        self.assertAlmostEqual(M.log_loss(p, y), ref)
        self.assertTrue(math.isfinite(M.log_loss(np.array([0.0, 1.0]), np.array([1, 0]))))  # clipped
        self.assertTrue(math.isnan(M.brier_score(np.array([]), np.array([]))))

    def test_reliability_and_ece_hand(self):
        p = np.array([0.9, 0.1, 0.6, 0.4])
        y = np.array([1, 0, 1, 1])
        rows = M.reliability_table(p, y, 2)
        self.assertEqual([r["n"] for r in rows], [2, 2])
        self.assertAlmostEqual(rows[0]["p_mean"], 0.25)
        self.assertAlmostEqual(rows[0]["freq"], 0.5)
        self.assertAlmostEqual(rows[1]["p_mean"], 0.75)
        self.assertAlmostEqual(rows[1]["freq"], 1.0)
        self.assertAlmostEqual(M.expected_calibration_error(p, y, 2), 0.25)
        self.assertEqual(sum(r["n"] for r in M.reliability_table(np.random.rand(103), np.zeros(103), 10)), 103)

    def test_calibrated_forecaster_has_small_ece_and_skill(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.2, 0.8, 20000)
        y = (rng.random(20000) < p).astype(float)
        self.assertLess(M.expected_calibration_error(p, y, 10), 0.02)
        s = M.probabilistic_summary(p, y, np.full(len(y), y.mean()))
        self.assertGreater(s["brier_skill"], 0.05)
        self.assertAlmostEqual(s["brier_skill"], 1 - s["brier"] / s["brier_baseline"])
        self.assertEqual(len(s["calibration"]), 10)

    def test_summary_keys_and_empty(self):
        s = M.probabilistic_summary(np.array([]), np.array([]), np.array([]))
        self.assertEqual(set(s), {"brier", "brier_baseline", "brier_skill", "log_loss", "log_loss_baseline", "ece",
                                  "calibration"})
        self.assertEqual(s["calibration"], [])
        s = M.probabilistic_summary(np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        self.assertIsNone(s["brier_skill"])  # zero-variance baseline -> undefined, not inf


class TestEceNoiseFloor(unittest.TestCase):
    def test_formula_hand_values(self):
        # h = 1: no overlap -> kappa = 1, floor = mult * sqrt(2/pi) * 0.5 / sqrt(m), m = n / bins
        self.assertAlmostEqual(M.ece_noise_floor(1250, 1, 10, mult=1.0), math.sqrt(2 / math.pi) * 0.5 / math.sqrt(125), places=12)
        self.assertAlmostEqual(M.ece_noise_floor(1250, 1, 10), M.ECE_NOISE_MULT * M.ece_noise_floor(1250, 1, 10, mult=1.0))
        # h = 5: kappa = 1 + 2 sum_k (1 - k/m) (2/pi) arcsin(1 - k/5)
        m = 125.0
        kappa = 1 + 2 * sum((1 - k / m) * (2 / math.pi) * math.asin(1 - k / 5) for k in range(1, 5))
        self.assertAlmostEqual(M.ece_noise_floor(1250, 5, 10, mult=1.0),
                               math.sqrt(2 / math.pi) * 0.5 * math.sqrt(kappa / m), places=12)
        # the hand value of kappa: 3.78 in the large-bin limit (m -> infinity), 3.737 with the (1 - k/m) edge correction of
        # 125 rows per bin (the earlier literal 3.78 +- 0.02 compared the finite-m formula with the limit)
        kappa_inf = 1 + 2 * sum((2 / math.pi) * math.asin(1 - k / 5) for k in range(1, 5))
        self.assertAlmostEqual(kappa_inf, 3.78, delta=0.01)
        self.assertAlmostEqual(kappa, 3.737, delta=0.002)
        self.assertLess(kappa, kappa_inf)

    def test_monotone_and_degenerate(self):
        f = [M.ece_noise_floor(1250, h, 10) for h in (1, 5, 20, 60)]
        self.assertTrue(all(a < b for a, b in zip(f, f[1:])), f)               # longer horizon -> more overlap -> more noise
        g = [M.ece_noise_floor(n, 5, 10) for n in (200, 500, 1250, 5000)]
        self.assertTrue(all(a > b for a, b in zip(g, g[1:])), g)               # more data -> less noise
        self.assertGreater(M.ece_noise_floor(1250, 5, 5), M.ece_noise_floor(1250, 5, 10) * 0.7)
        for bad in (0, -5, None):
            self.assertTrue(math.isnan(M.ece_noise_floor(bad, 1, 10)))
        self.assertTrue(math.isnan(M.ece_noise_floor(100, 1, 0)))
        self.assertTrue(math.isfinite(M.ece_noise_floor(12, 20, 1)))          # tiny sample, huge horizon: still finite

    def test_calibrated_constant_forecaster_rarely_exceeds_the_floor(self):
        # worst case (constant p -> bins are contiguous time blocks): iid returns with drift, overlapping h-day outcomes
        rng = np.random.default_rng(2024)
        for h in (1, 5, 20):
            p_true = float(stats.norm.cdf(0.0004 * h / (0.012 * math.sqrt(h))))
            over, mean_ece = 0, []
            reps = 120
            for _ in range(reps):
                r = 0.0004 + 0.012 * rng.standard_normal(1250 + h)
                cs = np.concatenate([[0.0], np.cumsum(r)])
                y = (cs[h:h + 1250] - cs[:1250] > 0).astype(float)
                e = M.expected_calibration_error(np.full(1250, p_true), y, 10)
                mean_ece.append(e)
                over += e > M.ece_noise_floor(1250, h, 10)
            self.assertLessEqual(over / reps, 0.10, (h, over / reps))          # nominal ~5 %
            self.assertAlmostEqual(np.mean(mean_ece), M.ece_noise_floor(1250, h, 10, mult=1.0), delta=0.12 * np.mean(mean_ece))

    def test_miscalibrated_forecaster_is_above_the_floor(self):
        # CHANGED: the old assertion "ECE > 2 x floor at every horizon" cannot hold at h = 20, where binning noise alone reaches
        # ~ 20 % (an exactly calibrated forecaster scores 13 % on average): a forecaster that says 80 % when the truth is 50 %
        # (ECE ~ 0.28) is far above the floor at h = 1 / 5, above it at h = 20, and a 97 % forecaster is far above it there too
        rng = np.random.default_rng(5)
        y = (rng.random(1250) < 0.5).astype(float)
        bad = M.expected_calibration_error(np.full(1250, 0.8), y, 10)
        for h in (1, 5):
            self.assertGreater(bad, 2 * M.ece_noise_floor(1250, h, 10), h)
        self.assertGreater(bad, M.ece_noise_floor(1250, 20, 10))
        self.assertLess(bad, 2 * M.ece_noise_floor(1250, 20, 10))                 # (and that is all one can say at h = 20)
        worse = M.expected_calibration_error(np.full(1250, 0.97), y, 10)
        self.assertGreater(worse, 2 * M.ece_noise_floor(1250, 20, 10))


class TestMinDetectableIc(unittest.TestCase):
    def test_table_scaling_and_interpolation(self):
        for h, v in M.MIN_DETECTABLE_IC.items():
            self.assertAlmostEqual(M.min_detectable_ic(h, 1250), v, places=12)
        self.assertAlmostEqual(M.min_detectable_ic(5, 312.5), M.MIN_DETECTABLE_IC[5] * 2, places=9)      # sqrt(1250 / n)
        vals = [M.min_detectable_ic(h, 1250) for h in (1, 2, 3, 5, 10, 20)]
        self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])), vals)                                 # grows with h
        self.assertEqual(M.min_detectable_ic(20, 1250), M.min_detectable_ic(60, 1250))                    # flat beyond 20
        self.assertLessEqual(M.min_detectable_ic(20, 20), 0.99)
        for bad in (0, -1, None):
            self.assertTrue(math.isnan(M.min_detectable_ic(5, bad)))


class TestReturnForecastStats(unittest.TestCase):
    def test_spearman_matches_scipy_h1(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=300)
        y = 0.2 * x + rng.normal(size=300)
        x[:20] = np.round(x[:20], 1)  # ties
        rho, p, n = M.spearman_ic(x, y, 1)
        ref = stats.spearmanr(x, y)
        self.assertAlmostEqual(rho, ref.statistic, places=12)
        self.assertAlmostEqual(p, ref.pvalue, places=10)
        self.assertEqual(n, 300)

    def test_spearman_overlap_adjustment(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=400)
        y = 0.1 * x + rng.normal(size=400)
        rho, p5, _ = M.spearman_ic(x, y, 5)
        n_eff = 400 / 5
        t = rho * math.sqrt((n_eff - 2) / (1 - rho ** 2))
        self.assertAlmostEqual(p5, 2 * stats.t.sf(abs(t), n_eff - 2), places=12)
        self.assertGreater(p5, M.spearman_ic(x, y, 1)[1])

    def test_spearman_degenerate(self):
        self.assertTrue(math.isnan(M.spearman_ic(np.ones(50), np.arange(50.0), 1)[0]))
        self.assertTrue(math.isnan(M.spearman_ic(np.arange(2.0), np.arange(2.0), 1)[0]))
        x = np.array([1, 2, 3, np.nan, 5.0])
        self.assertEqual(M.spearman_ic(x, x, 1)[2], 4)
        self.assertLess(M.spearman_ic(np.arange(10.0), np.arange(10.0), 1)[1], 1e-12)

    def test_p_one_sided(self):
        self.assertAlmostEqual(M.p_one_sided_positive(2.0, 0.04), 0.02)
        self.assertAlmostEqual(M.p_one_sided_positive(-2.0, 0.04), 0.98)
        self.assertTrue(math.isnan(M.p_one_sided_positive(float("nan"), 0.5)))

    def test_oos_r2_hand(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0, 4.0])
        b = np.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(M.oos_r2(y, f, b), 1 - 1 / 14)
        self.assertAlmostEqual(M.oos_r2(y, y, b), 1.0)
        self.assertLess(M.oos_r2(y, -y, b), 0)
        self.assertTrue(math.isnan(M.oos_r2(y, f, y)))  # zero benchmark SSE

    def test_newey_west_matches_loops(self):
        rng = np.random.default_rng(7)
        d = np.convolve(rng.normal(size=210), np.ones(5) / 5, mode="valid")
        for lag in (0, 1, 4, 10):
            self.assertAlmostEqual(M.newey_west_lrv(d, lag), ref_nw_lrv(list(d), lag), places=12)
        self.assertAlmostEqual(M.newey_west_lrv(d, 0), float(np.var(d)))  # lag 0 = plain variance (ddof 0)

    def test_dm_matches_manual_and_sign(self):
        rng = np.random.default_rng(8)
        a = rng.chisquare(1, 600) * 1e-4          # reference loss
        b = a - 2e-5 + rng.normal(0, 3e-5, 600)   # model loss, smaller on average
        for h in (1, 5):
            stat, p = M.dm_test(a, b, h)
            d = a - b
            ref = d.mean() / math.sqrt(ref_nw_lrv(list(d), h - 1) / len(d))
            self.assertAlmostEqual(stat, ref, places=9)
            self.assertAlmostEqual(p, 2 * stats.t.sf(abs(ref), len(d) - 1), places=12)
            self.assertGreater(stat, 0)  # model better -> positive
            self.assertLess(M.dm_test(b, a, h)[0], 0)
        self.assertEqual(M.dm_test(a, a, 3), (0.0, 1.0))
        self.assertTrue(math.isnan(M.dm_test(a[:2], b[:2], 1)[0]))

    def test_dm_size_and_power(self):
        rng = np.random.default_rng(9)
        rej = 0
        reps = 400
        for _ in range(reps):
            e1, e2 = rng.normal(size=300), rng.normal(size=300)
            if M.dm_test(e1 ** 2, e2 ** 2, 1)[1] < 0.05:
                rej += 1
        self.assertTrue(0.02 <= rej / reps <= 0.085, rej / reps)
        rej = 0
        for _ in range(100):
            e1, e2 = rng.normal(size=300), 0.8 * rng.normal(size=300)
            if M.dm_test(e1 ** 2, e2 ** 2, 1)[1] < 0.05:
                rej += 1
        self.assertGreater(rej / 100, 0.6)


class TestIntervals(unittest.TestCase):
    QL = (0.25, 0.75)

    def test_coverage_hand_example(self):
        q = np.array([[-1, 1], [0.5, 2], [2.5, 3], [0, 1.0]])
        r = np.array([0.0, 1.0, 2.0, 3.0])
        cov, n, viol = M.interval_coverage(r, q, self.QL, 0.5)
        self.assertEqual((cov, n), (0.5, 4))
        self.assertEqual(viol.tolist(), [False, False, True, True])
        with self.assertRaises(KeyError):
            M.interval_coverage(r, q, self.QL, 0.9)

    def test_coverage_drops_nan_rows(self):
        q = np.array([[-1, 1], [np.nan, 2], [-1, 1.0]])
        r = np.array([0.0, 0.0, np.nan])
        cov, n, viol = M.interval_coverage(r, q, self.QL, 0.5)
        self.assertEqual((cov, n, len(viol)), (1.0, 1, 1))

    def test_true_distribution_covers_nominally(self):
        rng = np.random.default_rng(11)
        r = rng.normal(size=40000)
        q = np.tile(stats.norm.ppf(config.QLEVELS), (len(r), 1))
        for lev in (0.5, 0.8, 0.9):
            cov, n, _ = M.interval_coverage(r, q, config.QLEVELS, lev)
            self.assertAlmostEqual(cov, lev, delta=0.01)
        cs = M.coverage_summary(r, q, config.QLEVELS, 1)
        self.assertEqual(set(cs["coverage"]), {"0.50", "0.80", "0.90"})
        self.assertGreater(cs["kupiec_pvalue"], 0.01)

    def test_kupiec_matches_formula(self):
        for x, n, p in [(10, 100, 0.05), (0, 100, 0.05), (100, 100, 0.9), (5, 100, 0.05), (37, 500, 0.1)]:
            lr, pv = M.kupiec_pof(x, n, p)
            self.assertAlmostEqual(lr, max(0.0, ref_kupiec(x, n, p)), places=9)
            self.assertAlmostEqual(pv, stats.chi2.sf(max(0.0, ref_kupiec(x, n, p)), 1), places=9)
        self.assertAlmostEqual(M.kupiec_pof(5, 100, 0.05)[1], 1.0)
        self.assertAlmostEqual(M.kupiec_pof(0, 100, 0.05)[0], -2 * 100 * math.log(0.95), places=9)
        self.assertLess(M.kupiec_pof(20, 100, 0.05)[1], 1e-5)

    def test_kupiec_size(self):
        rng = np.random.default_rng(12)
        pv = [M.kupiec_pof(int((rng.random(250) < 0.1).sum()), 250, 0.1)[1] for _ in range(1500)]
        self.assertTrue(0.03 < np.mean(np.array(pv) < 0.05) < 0.08)

    def test_coverage_summary_overlap_scaling(self):
        rng = np.random.default_rng(13)
        r = rng.normal(size=1000)
        q = np.tile(stats.norm.ppf(config.QLEVELS), (1000, 1))
        q[:, 1] += 0.5  # shrink the 90% interval from below -> too many violations
        cs1 = M.coverage_summary(r, q, config.QLEVELS, 1)
        cs5 = M.coverage_summary(r, q, config.QLEVELS, 5)
        _, n, viol = M.interval_coverage(r, q, config.QLEVELS, 0.9)
        self.assertAlmostEqual(cs1["kupiec_pvalue"], M.kupiec_pof(int(viol.sum()), n, 0.1)[1])
        self.assertAlmostEqual(cs5["kupiec_pvalue"],
                               M.kupiec_pof(int(round(viol.sum() / n * 200)), 200, 0.1)[1])
        self.assertGreater(cs5["kupiec_pvalue"], cs1["kupiec_pvalue"])

    def test_christoffersen_independence_manual(self):
        hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        a, b = hits[:-1].astype(int), hits[1:].astype(int)
        n00 = int(((a == 0) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        n11 = int(((a == 1) & (b == 1)).sum())
        pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        ll0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
        ll1 = (n00 * math.log(1 - pi01) + n01 * math.log(pi01) + n10 * math.log(1 - pi11) + n11 * math.log(pi11))
        lr, pv = M.christoffersen_independence(hits)
        self.assertAlmostEqual(lr, -2 * (ll0 - ll1), places=10)
        self.assertAlmostEqual(pv, stats.chi2.sf(-2 * (ll0 - ll1), 1), places=10)
        lr_cc, p_cc = M.christoffersen_cc(hits, 0.3)
        self.assertAlmostEqual(lr_cc, lr + M.kupiec_pof(float(hits.sum()), 14.0, 0.3)[0], places=10)
        self.assertAlmostEqual(p_cc, stats.chi2.sf(lr_cc, 2), places=10)

    def test_christoffersen_detects_clustering(self):
        rng = np.random.default_rng(14)
        iid = rng.random(2000) < 0.1
        clustered = np.zeros(2000, dtype=bool)
        for s in rng.integers(0, 1980, 25):
            clustered[s:s + 8] = True
        self.assertGreater(M.christoffersen_independence(iid)[1], 0.01)
        self.assertLess(M.christoffersen_independence(clustered)[1], 1e-6)

    def test_pit_histogram_hand(self):
        h = M.pit_histogram(np.array([0.05, 0.15, 0.15, 0.95, 1.0, np.nan]))
        np.testing.assert_allclose(h, [0.2, 0.4, 0, 0, 0, 0, 0, 0, 0, 0.4])
        self.assertAlmostEqual(sum(h), 1.0)

    def test_pit_ks_matches_scipy(self):
        rng = np.random.default_rng(15)
        for pit in (rng.random(500), rng.beta(2, 2, 500), rng.random(60)):
            ref = stats.ks_1samp(pit, stats.uniform.cdf)
            self.assertAlmostEqual(M.pit_ks_pvalue(pit, 1), ref.pvalue, places=10)
            d = ref.statistic
            self.assertAlmostEqual(M.pit_ks_pvalue(pit, 5), stats.kstwo.sf(d, int(round(len(pit) / 5))), places=10)
        self.assertGreater(M.pit_ks_pvalue(rng.random(1000), 1), 0.01)
        self.assertLess(M.pit_ks_pvalue(rng.beta(1, 3, 1000), 1), 1e-6)

    def test_pinball_hand_and_optimality(self):
        self.assertAlmostEqual(M.pinball_loss(np.array([2.0, 0.5]), np.array([[1.0], [1.0]]), [0.9]), 0.475)
        rng = np.random.default_rng(16)
        y = rng.normal(size=20000)
        lv = np.array(config.QLEVELS)
        q_true = np.tile(stats.norm.ppf(lv), (len(y), 1))
        base = M.pinball_loss(y, q_true, lv)
        self.assertLess(base, M.pinball_loss(y, q_true + 0.2, lv))
        self.assertLess(base, M.pinball_loss(y, q_true * 1.5, lv))
        self.assertLess(base, M.pinball_loss(y, q_true * 0.6, lv))
        self.assertTrue(math.isnan(M.pinball_loss(np.array([np.nan]), np.zeros((1, 9)), lv)))

    def test_rolling_mean(self):
        x = np.random.default_rng(17).random(50)
        ref = np.convolve(x, np.ones(7) / 7, mode="valid")
        np.testing.assert_allclose(M.rolling_mean(x, 7), ref)
        self.assertEqual(len(M.rolling_mean(x, 51)), 0)
        self.assertEqual(len(M.rolling_mean(x, 0)), 0)


def ref_psr(r, sr_ref):
    """PSR from explicit moments (no scipy.stats.skew/kurtosis)."""
    n = len(r)
    mean = sum(r) / n
    m2 = sum((x - mean) ** 2 for x in r) / n
    m3 = sum((x - mean) ** 3 for x in r) / n
    m4 = sum((x - mean) ** 4 for x in r) / n
    skew, kurt = m3 / m2 ** 1.5, m4 / m2 ** 2
    sr = mean / math.sqrt(m2 * n / (n - 1))
    z = (sr - sr_ref) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class TestStrategyStats(unittest.TestCase):
    def test_hand_example(self):
        r = np.array([0.10, -0.05, 0.02])
        eq = M.equity_from_returns(r)
        np.testing.assert_allclose(eq, [1.0, 1.1, 1.045, 1.0659])
        np.testing.assert_allclose(M.drawdown_series(eq), [0, 0, 1.045 / 1.1 - 1, 1.0659 / 1.1 - 1])
        self.assertAlmostEqual(M.max_drawdown(eq), 1.045 / 1.1 - 1)
        mean = sum(r) / 3
        sd = math.sqrt(sum((x - mean) ** 2 for x in r) / 2)
        self.assertAlmostEqual(M.sharpe_ratio(r), mean / sd * math.sqrt(245))
        dd = math.sqrt((0.05 ** 2) / 3)
        self.assertAlmostEqual(M.sortino_ratio(r), mean / dd * math.sqrt(245))
        st = M.return_stats(r, sharpe_ci=[0.0, 1.0])
        self.assertAlmostEqual(st["total_return"], 0.0659)
        self.assertAlmostEqual(st["cagr"], 1.0659 ** (245 / 3) - 1)
        self.assertAlmostEqual(st["vol"], sd * math.sqrt(245))
        self.assertAlmostEqual(st["calmar"], st["cagr"] / 0.05, places=9)
        self.assertEqual(st["sharpe_ci95"], [0.0, 1.0])

    def test_degenerate_series_are_none(self):
        for r in (np.zeros(100), np.array([]), np.full(10, np.nan), np.full(50, 0.001)):
            st = M.return_stats(r, n_boot=50)
            self.assertEqual(set(st), {"total_return", "cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "psr",
                                       "dsr", "sharpe_ci95"})
            json.dumps(M.json_safe(st), allow_nan=False)
            self.assertIsNone(st["sharpe"])
            self.assertIsNone(st["psr"])
        z = M.return_stats(np.zeros(100), n_boot=50)
        self.assertEqual((z["total_return"], z["max_dd"]), (0.0, 0.0))
        self.assertIsNone(z["calmar"])
        self.assertIsNone(M.return_stats(np.array([]))["total_return"])

    def test_total_loss_equity(self):
        st = M.return_stats(np.array([0.1, -1.0, 0.1]), n_boot=20)
        self.assertEqual(st["cagr"], -1.0)
        self.assertAlmostEqual(st["max_dd"], -1.0)

    def test_psr_matches_independent_formula(self):
        rng = np.random.default_rng(21)
        for r in (rng.normal(0.001, 0.01, 800), rng.standard_t(4, 600) * 0.01 + 0.0006,
                  np.where(rng.random(500) < 0.1, -0.03, 0.004)):
            for ref in (0.0, 0.03, -0.02):
                self.assertAlmostEqual(M.probabilistic_sharpe_ratio(r, ref), ref_psr(list(r), ref), places=9)

    def test_psr_properties(self):
        rng = np.random.default_rng(22)
        r = rng.normal(0.001, 0.01, 1000)
        self.assertGreater(M.probabilistic_sharpe_ratio(r, 0.0), 0.5)
        self.assertGreater(M.probabilistic_sharpe_ratio(r, 0.0), M.probabilistic_sharpe_ratio(r, 0.05))
        # normal data and large n: PSR ~ Phi(SR sqrt(n-1) / sqrt(1 + SR^2/2))
        sr = r.mean() / r.std(ddof=1)
        approx = stats.norm.cdf(sr * math.sqrt(999) / math.sqrt(1 + sr ** 2 / 2))
        self.assertAlmostEqual(M.probabilistic_sharpe_ratio(r, 0.0), approx, delta=0.02)
        self.assertTrue(math.isnan(M.probabilistic_sharpe_ratio(np.zeros(50), 0.0)))
        self.assertTrue(math.isnan(M.probabilistic_sharpe_ratio(np.array([0.1, 0.2]), 0.0)))

    def test_expected_max_sharpe_formula(self):
        nd = __import__("statistics").NormalDist()
        g = 0.5772156649015329
        for N, V in [(10, 0.01), (100, 0.04), (1000, 0.0004)]:
            ref = math.sqrt(V) * ((1 - g) * nd.inv_cdf(1 - 1 / N) + g * nd.inv_cdf(1 - 1 / (N * math.e)))
            self.assertAlmostEqual(M.expected_max_sharpe(N, V), ref, places=12)
        self.assertEqual(M.expected_max_sharpe(1, 0.01), 0.0)
        # it approximates E[max of N iid N(0, V)]
        rng = np.random.default_rng(23)
        emp = rng.normal(0, 0.2, (4000, 100)).max(axis=1).mean()
        self.assertAlmostEqual(M.expected_max_sharpe(100, 0.04), emp, delta=0.03 * emp)

    def test_dsr(self):
        rng = np.random.default_rng(24)
        r = rng.normal(0.0012, 0.01, 1250)
        self.assertAlmostEqual(M.deflated_sharpe_ratio(r, 1), M.probabilistic_sharpe_ratio(r, 0.0))
        d = [M.deflated_sharpe_ratio(r, n) for n in (1, 5, 50, 500)]
        self.assertTrue(all(a > b for a, b in zip(d, d[1:])), d)
        se = M.sharpe_standard_error(r)
        sr0 = M.expected_max_sharpe(50, se ** 2)
        self.assertAlmostEqual(M.deflated_sharpe_ratio(r, 50), ref_psr(list(r), sr0), places=9)
        self.assertAlmostEqual(M.deflated_sharpe_ratio(r, 50, var_sr=0.0004),
                               ref_psr(list(r), M.expected_max_sharpe(50, 0.0004)), places=9)

    def test_sharpe_standard_error_normal_case(self):
        rng = np.random.default_rng(25)
        r = rng.normal(0.0005, 0.01, 2000)
        sr = r.mean() / r.std(ddof=1)
        self.assertAlmostEqual(M.sharpe_standard_error(r), math.sqrt((1 + sr ** 2 / 2) / 1999), delta=2e-3)


class TestBootstrap(unittest.TestCase):
    def test_indices_valid_and_block_length(self):
        rng = np.random.default_rng(31)
        n, b = 2000, 10.0
        idx = M.stationary_bootstrap_indices(n, b, 60, rng)
        self.assertEqual(idx.shape, (60, n))
        self.assertTrue(idx.min() >= 0 and idx.max() < n)
        breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % n).sum(axis=1) + 1
        self.assertAlmostEqual(float(np.mean(n / breaks)), b, delta=0.8)
        self.assertAlmostEqual(float(idx.mean()), (n - 1) / 2, delta=40)  # uniform marginal

    def test_bootstrap_mean_variance_matches_theory(self):
        # AR(1): the stationary bootstrap variance of the mean is gamma0/n * (1+q)/(1-q), q = (1-1/b) phi
        rng = np.random.default_rng(32)
        n, phi, b = 4000, 0.7, 20.0
        e = rng.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        idx = M.stationary_bootstrap_indices(n, b, 3000, rng)
        sd_boot = x[idx].mean(axis=1).std()
        q = (1 - 1 / b) * phi
        theory = math.sqrt(x.var() / n * (1 + q) / (1 - q))
        self.assertAlmostEqual(sd_boot / theory, 1.0, delta=0.12)

    def test_sharpe_ci_coverage_iid(self):
        rng = np.random.default_rng(33)
        true_sr = 0.0008 / 0.01 * math.sqrt(245)
        hit, reps = 0, 120
        for i in range(reps):
            r = rng.normal(0.0008, 0.01, 500)
            lo, hi = M.bootstrap_sharpe(r, None, n_boot=300, seed=i)["ci1"]
            hit += lo <= true_sr <= hi
        self.assertGreater(hit / reps, 0.85)

    def test_determinism_and_degenerate(self):
        r = np.random.default_rng(34).normal(0.0005, 0.01, 300)
        a = M.bootstrap_sharpe(r, r * 0.5, n_boot=200, seed=5)
        b = M.bootstrap_sharpe(r, r * 0.5, n_boot=200, seed=5)
        self.assertEqual(a, b)
        c = M.bootstrap_sharpe(r, r * 0.5, n_boot=200, seed=6)
        self.assertNotEqual(a["ci1"], c["ci1"])
        z = M.bootstrap_sharpe(np.zeros(100), np.zeros(100), n_boot=100)
        self.assertEqual(z["ci1"], [None, None])
        self.assertIsNone(z["diff_p"])
        self.assertEqual(M.bootstrap_sharpe(np.zeros(5), None)["ci1"], [None, None])

    def test_sharpe_difference_inference(self):
        rng = np.random.default_rng(35)
        r = rng.normal(0.0004, 0.01, 1000)
        same = M.bootstrap_sharpe(r, r.copy(), n_boot=400, seed=1)
        self.assertEqual(same["diff_ci"], [0.0, 0.0])  # identical series: difference is exactly 0
        better = r + 0.0015                             # strictly higher mean, same vol
        res = M.bootstrap_sharpe(better, r, n_boot=600, seed=1)
        self.assertLess(res["diff_p"], 0.05)
        self.assertGreater(res["diff_ci"][0], 0.0)
        worse = M.bootstrap_sharpe(r, better, n_boot=600, seed=1)
        self.assertGreater(worse["diff_p"], 0.95)
        indep = M.bootstrap_sharpe(rng.normal(0.0004, 0.01, 1000), r, n_boot=600, seed=1)
        self.assertGreater(indep["diff_p"], 0.02)


if __name__ == "__main__":
    unittest.main()
