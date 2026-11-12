"""Tests for research/panel.py, research/harness.py and research/price_baseline.py.  Synthetic data only, no network.

Run: python -m unittest discover -s tests -p "test_research_harness.py"
"""
import copy
import json
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from research import harness as H  # noqa: E402
from research import panel as PN  # noqa: E402
from research import price_baseline as PB  # noqa: E402
from research.panel import Panel  # noqa: E402

NAN = np.nan


def weekdays(start, end):
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    return days[((days.astype(np.int64) + 3) % 7) < 5]


def make_panel(S=60, start="2011-01-03", end="2023-06-30", seed=0, vol=0.02):
    """Random-walk panel (no skill).  Opens carry a small overnight gap."""
    rng = np.random.default_rng(seed)
    cal = weekdays(start, end)
    T = len(cal)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, vol, (T, S)), axis=0))
    op = np.empty_like(close)
    op[0] = close[0]
    op[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, (T - 1, S)))
    vol_ = np.exp(rng.normal(13, 0.4, (T, S)))
    return Panel(cal, [f"S{j:03d}" for j in range(S)], op, close, vol_, close * vol_, np.ones((T, S), bool))


def tiny_panel(opens, closes=None, volume=None, cal=None, bars_before=None):
    """Panel from small matrices; NaN open/close = no bar."""
    opens = np.asarray(opens, float)
    T, S = opens.shape
    closes = opens.copy() if closes is None else np.asarray(closes, float)
    has = np.isfinite(opens) & np.isfinite(closes)
    vol = np.where(has, 1.0 if volume is None else np.asarray(volume, float), NAN)
    cal = weekdays("2014-03-03", "2014-12-31")[:T] if cal is None else cal
    return Panel(cal, [f"C{j}" for j in range(S)], opens, closes, vol, np.where(has, closes * vol, NAN), has, bars_before)


def planted(y, rho, seed):
    """F with Pearson correlation ~rho to y per row: z(y) * rho + sqrt(1 - rho^2) * noise (NaN where y is NaN)."""
    rng = np.random.default_rng(seed)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = (y - np.nanmean(y, axis=1, keepdims=True)) / np.nanstd(y, axis=1, keepdims=True)
    return rho * z + math.sqrt(1 - rho * rho) * rng.standard_normal(y.shape)


# ============================================================================= panel: hand-computed cases
class ForwardOpenReturn(unittest.TestCase):
    def setUp(self):
        self.opens = np.array([[10, 20], [11, 20], [12, NAN], [13, 22], [14, 24], [15, 26]], float)
        self.p = tiny_panel(self.opens)

    def test_h1_hand_computed(self):
        y = PN.forward_open_return(self.p, 1)
        self.assertAlmostEqual(y[0, 0], math.log(12 / 11))
        self.assertAlmostEqual(y[2, 0], math.log(14 / 13))
        self.assertAlmostEqual(y[3, 0], math.log(15 / 14))
        self.assertTrue(np.isnan(y[4:, 0]).all())            # t+1+h beyond the sample
        self.assertTrue(np.isnan(y[0, 1]))                    # entry open o[1] fine, exit open o[2] missing
        self.assertTrue(np.isnan(y[1, 1]))                    # entry open o[2] missing
        self.assertAlmostEqual(y[2, 1], math.log(24 / 22))

    def test_h2_hand_computed_and_not_close_based(self):
        y = PN.forward_open_return(self.p, 2)
        self.assertAlmostEqual(y[0, 0], math.log(13 / 11))
        self.assertAlmostEqual(y[0, 1], math.log(22 / 20))
        self.assertTrue(np.isnan(y[1, 1]))                    # exit o[4]=24 fine but entry o[2] missing
        self.assertTrue(np.isnan(y[3:, 0]).all())
        # the start is the NEXT open, not close[t]: change close[0] and nothing moves
        p2 = tiny_panel(self.opens, closes=self.opens * 1.5)
        np.testing.assert_array_equal(PN.forward_open_return(p2, 2), y)

    def test_h_must_be_positive(self):
        with self.assertRaises(ValueError):
            PN.forward_open_return(self.p, 0)


class UniverseMaskHand(unittest.TestCase):
    def setUp(self):
        value = np.array([[10, 30, 20], [10, 30, 20], [10, 30, 20], [50, 5, NAN], [50, 5, 25], [50, 5, 25]], float)
        has = np.isfinite(value)
        T, S = value.shape
        self.p = Panel(weekdays("2014-03-03", "2014-12-31")[:T], ["A", "B", "C"], np.ones((T, S)), np.ones((T, S)),
                       np.ones((T, S)), value, has)

    def test_top2_by_trailing_median_value(self):
        m = PN.universe_mask(self.p, n=2, lookback=3, min_bars=3, min_obs=2)
        expect = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]], bool)
        np.testing.assert_array_equal(m, expect)

    def test_min_bars_counts_bars_up_to_t(self):
        m = PN.universe_mask(self.p, n=2, lookback=3, min_bars=4, min_obs=2)
        self.assertFalse(m[2].any())                           # only 3 bars so far
        np.testing.assert_array_equal(m[3], [True, True, False])

    def test_bars_before_start_are_counted(self):
        p = self.p.copy()
        p.bars_before = np.array([10, 10, 10])
        m = PN.universe_mask(p, n=2, lookback=3, min_bars=12, min_obs=2)
        self.assertFalse(m[0].any())                           # 11 bars at t=0
        self.assertTrue(m[1].any())                            # 12 bars at t=1 (but window has only 2 obs -> min_obs ok)

    def test_only_members_with_a_bar_and_never_more_than_n(self):
        p = make_panel(S=30, start="2012-01-02", end="2014-12-31", seed=3)
        m = PN.universe_mask(p, n=10, lookback=20, min_bars=100)
        self.assertLessEqual(m.sum(axis=1).max(), 10)
        self.assertTrue((~m | p.has_bar).all())
        self.assertEqual(m[:99].sum(), 0)                      # nobody has 100 bars before row 99


class UntradableOpen(unittest.TestCase):
    def test_limit_up_open_flagged_using_last_known_close(self):
        o = np.array([[100, 100, 100], [100, 100, NAN], [110, 109, 111], [111, 109, 111]], float)
        c = np.array([[100, 100, 100], [100, 100, NAN], [110, 109, 111], [111, 109, 111]], float)
        u = PN.untradable_open(tiny_panel(o, c))
        self.assertTrue(u[1, 0])                                # open[2] = +10 % over close[1]
        self.assertFalse(u[1, 1])                               # +9 % is below 9.5 %
        self.assertTrue(u[1, 2])                                # no bar at t=1: last close <= t is close[0] = 100 -> +11 %
        self.assertFalse(u[0, 2])                               # open[1] of stock 2 does not exist
        self.assertFalse(u[-1].any())                           # last row has no t+1

    def test_save_load_roundtrip(self):
        p = make_panel(S=4, start="2014-01-02", end="2014-03-31", seed=5)
        p.bars_before = np.array([1, 2, 3, 4])
        with tempfile.TemporaryDirectory() as d:
            path = PN.save_panel(p, Path(d) / "p.npz")
            q = PN.load_panel(path)
        np.testing.assert_array_equal(q.cal, p.cal)
        np.testing.assert_array_equal(q.codes, p.codes)
        np.testing.assert_array_equal(q.adj_open, p.adj_open)
        np.testing.assert_array_equal(q.has_bar, p.has_bar)
        np.testing.assert_array_equal(q.bars_before, p.bars_before)


# ============================================================================= harness: periods and the OOS guard
class Periods(unittest.TestCase):
    def setUp(self):
        self.cal = weekdays("2012-06-01", "2023-03-31")

    def test_boundaries(self):
        c = self.cal
        d = lambda s: np.datetime64(s, "D")   # noqa: E731
        dev = H.period_rows(c, "dev")
        self.assertFalse(dev[c == d("2012-12-31")][0])
        self.assertTrue(dev[c == d("2013-01-01")][0])
        self.assertTrue(dev[c == d("2022-12-30")][0])
        self.assertFalse(dev[c == d("2023-01-02")][0])
        a, b, o = H.period_rows(c, "dev_a"), H.period_rows(c, "dev_b"), H.period_rows(c, "oos")
        self.assertTrue(a[c == d("2017-12-29")][0] and not a[c == d("2018-01-01")][0])
        self.assertTrue(b[c == d("2018-01-01")][0] and not b[c == d("2017-12-29")][0])
        np.testing.assert_array_equal(a | b, dev)
        self.assertFalse((a & b).any())
        self.assertTrue(o[c == d("2023-01-02")][0] and not o[c == d("2022-12-30")][0])
        self.assertFalse((dev & o).any())

    def test_label_window_is_purged_at_the_end_of_dev(self):
        c, h = self.cal, 5
        rows = H.period_rows(c, "dev", h)
        last = np.flatnonzero(rows).max()
        self.assertLessEqual(c[last + 1 + h], np.datetime64("2022-12-31"))      # label window ends inside DEV
        self.assertGreater(c[last + 2 + h], np.datetime64("2022-12-31"))        # and the next decision day would not
        self.assertEqual(rows.sum(), H.period_rows(c, "dev", 0).sum() - (h + 1))
        # OOS is not purged at the end (open-ended)
        self.assertEqual(H.period_rows(c, "oos", h).sum(), H.period_rows(c, "oos", 0).sum())

    def test_unknown_period_raises(self):
        for p in ("DEV", "test", "", "all"):
            with self.assertRaises(ValueError):
                H.period_bounds(p)
            with self.assertRaises(ValueError):
                H.ic_report(np.zeros((3, 3)), np.zeros((3, 3)), np.ones((3, 3), bool), 5, p, np.arange(3).astype("datetime64[D]"))


class OosGuard(unittest.TestCase):
    def setUp(self):
        self.p = make_panel(S=30, start="2022-06-01", end="2023-06-30", seed=2)
        self.mask = np.ones((self.p.T, self.p.S), bool)
        self.y = PN.forward_open_return(self.p, 5)
        self.F = planted(self.y, 0.1, 1)
        self._orig = H.LOOKS_PATH
        self.tmp = tempfile.TemporaryDirectory()
        H.LOOKS_PATH = Path(self.tmp.name) / "LOOKS.md"

    def tearDown(self):
        H.LOOKS_PATH = self._orig
        self.tmp.cleanup()

    def test_oos_without_spec_hash_raises_and_logs_nothing(self):
        for bad in (None, "", "   ", 123):
            with self.assertRaises(ValueError):
                H.ic_report(self.F, self.y, self.mask, 5, "oos", self.p.cal, spec_hash=bad)
        with self.assertRaises(ValueError):
            H.ic_report(self.F, self.y, self.mask, 5, "oos", self.p.cal)
        with self.assertRaises(ValueError):
            H.quintile_report(self.F, self.p, self.mask, 5, "oos")
        with self.assertRaises(ValueError):
            H.residual_ic(self.F, [self.F * 0 + 1], self.y, self.mask, 5, "oos", self.p.cal)
        self.assertFalse(H.LOOKS_PATH.exists())

    def test_oos_with_spec_hash_works_and_appends_one_row_per_look(self):
        r = H.ic_report(self.F, self.y, self.mask, 5, "oos", self.p, spec_hash="abc123def456")
        self.assertGreater(r["n_days"], 50)
        text = H.LOOKS_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("abc123def456"), 1)
        self.assertIn("ic_report", text)
        H.quintile_report(self.F, self.p, self.mask, 5, "oos", spec_hash="abc123def456")
        rows = [ln for ln in H.LOOKS_PATH.read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]
        self.assertEqual(len(rows), 2)

    def test_dev_periods_never_log(self):
        H.ic_report(self.F, self.y, self.mask, 5, "dev", self.p.cal)
        self.assertFalse(H.LOOKS_PATH.exists())


# ============================================================================= harness: small tools
class SmallTools(unittest.TestCase):
    def test_xs_rank_range_ties_and_fill(self):
        F = np.array([[3, 1, 2, NAN, 5], [1, 1, 2, 7, 7]], float)
        r = H.xs_rank(F)
        np.testing.assert_allclose(r[0, [0, 1, 2, 4]], [1 / 6, -0.5, -1 / 6, 0.5])
        self.assertTrue(np.isnan(r[0, 3]))
        np.testing.assert_allclose(r[1], [(rk - 1) / 4 - 0.5 for rk in (1.5, 1.5, 3.0, 4.5, 4.5)])    # ties share the average rank
        self.assertEqual(H.xs_rank(F, fill=0.0)[0, 3], 0.0)
        m = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], bool)
        rm = H.xs_rank(F, m)
        self.assertTrue(np.isnan(rm[0, 4]))
        self.assertAlmostEqual(np.nanmin(rm[0]), -0.5)
        self.assertAlmostEqual(np.nanmax(rm[0]), 0.5)

    def test_xs_rank_single_entry_is_zero(self):
        self.assertEqual(H.xs_rank(np.array([[NAN, 4.0, NAN]]))[0, 1], 0.0)

    def test_hac_mean_t_on_iid_noise_and_on_a_shifted_series(self):
        rng = np.random.default_rng(0)
        r = H.hac_mean_t(rng.standard_normal(2000), 1)
        self.assertLess(abs(r["t"]), 3)
        self.assertEqual(r["n"], 2000)
        r2 = H.hac_mean_t(rng.standard_normal(2000) * 0.1 + 0.05, 5)
        self.assertGreater(r2["t"], 4)
        self.assertAlmostEqual(r2["se"], r2["mean"] / r2["t"])

    def test_composite_equal_weights_signs_and_missing_as_zero(self):
        mask = np.ones((1, 4), bool)
        a = np.array([[1.0, 2, 3, 4]])
        b = np.array([[4.0, 3, 2, NAN]])
        c = H.composite({"a": a, "b": b}, {"a": 1, "b": -1}, mask)
        ra = H.xs_rank(a, mask)
        rb = H.xs_rank(b, mask, fill=0.0)
        np.testing.assert_allclose(c, (ra - rb) / 2)
        with self.assertRaises(ValueError):
            H.composite({"a": a}, {"a": 0}, mask)

    def test_spec_hash_is_deterministic_and_order_free(self):
        self.assertEqual(H.spec_hash_of({"a": 1, "b": [1, 2]}), H.spec_hash_of({"b": [1, 2], "a": 1}))
        self.assertNotEqual(H.spec_hash_of({"a": 1}), H.spec_hash_of({"a": 2}))

    def test_tranche_weights_is_the_trailing_h_average(self):
        P = np.zeros((6, 3))
        P[0, 0] = P[1, 1] = P[2, 2] = P[3, 0] = P[4, 1] = P[5, 2] = 1.0
        W, t0 = H.tranche_weights(P, 3)
        self.assertEqual(t0, 0)
        np.testing.assert_allclose(W[0], [1, 0, 0])
        np.testing.assert_allclose(W[1], [0.5, 0.5, 0])
        np.testing.assert_allclose(W[2], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(W[4], [1 / 3, 1 / 3, 1 / 3])


# ============================================================================= the four statistical tests (PLAN P0 item 4)
class StatisticalSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = make_panel(S=60, seed=11)
        cls.mask = PN.universe_mask(cls.p, n=150, min_bars=500)
        cls.y5 = PN.forward_open_return(cls.p, 5)

    def test_placebo_shuffled_feature_has_no_ic(self):
        """Shuffling the feature across stocks every day destroys any relation: |mean IC| < 3 SE."""
        strong = planted(self.y5, 0.3, 3)                       # a feature that WOULD have a big IC ...
        rng = np.random.default_rng(4)
        shuffled = strong.copy()
        for t in range(self.p.T):                               # ... shuffled independently across stocks each day
            shuffled[t] = strong[t, rng.permutation(self.p.S)]
        ic = H.ic_series(shuffled, self.y5, self.mask)
        rows = H.period_rows(self.p.cal, "dev", 5) & np.isfinite(ic)
        x = ic[rows]
        self.assertGreater(len(x), 2000)
        se = x.std(ddof=1) / math.sqrt(len(x))
        self.assertLess(abs(x.mean()), 3 * se)
        self.assertLess(abs(H.ic_report(shuffled, self.y5, self.mask, 5, "dev", self.p)["t"]), 3)
        # and the un-shuffled feature really is strong (the test would fail if shuffling did nothing)
        self.assertGreater(H.ic_report(strong, self.y5, self.mask, 5, "dev", self.p)["t"], 10)

    def test_planted_signal_is_detected(self):
        """F = y + noise with IC ~ 0.05 -> HAC t > 4."""
        F = planted(self.y5, 0.05, 5)
        r = H.ic_report(F, self.y5, self.mask, 5, "dev", self.p)
        self.assertGreater(r["mean"], 0.03)
        self.assertLess(r["mean"], 0.08)
        self.assertGreater(r["t"], 4)
        s = H.screen_dev(F, self.y5, self.mask, 5, self.p, expected_sign=+1)
        self.assertTrue(s["passes"])
        self.assertFalse(H.screen_dev(F, self.y5, self.mask, 5, self.p, expected_sign=-1)["passes"])

    def test_no_peeking_poisoning_the_future_changes_nothing_before_t0(self):
        h = 5
        base = self.p
        t0 = int(np.flatnonzero(base.cal >= np.datetime64("2016-06-30"))[0])
        pz = base.copy()
        rng = np.random.default_rng(99)
        shape = pz.adj_open[t0 + 1:].shape
        pz.adj_open[t0 + 1:] = 10 * np.exp(rng.normal(0, 1, shape))
        pz.adj_close[t0 + 1:] = 10 * np.exp(rng.normal(0, 1, shape))
        pz.volume[t0 + 1:] = np.exp(rng.normal(8, 3, shape))
        pz.value[t0 + 1:] = pz.adj_close[t0 + 1:] * pz.volume[t0 + 1:]
        pz.has_bar[t0 + 1:] = rng.random(shape) < 0.5
        # universe: bitwise identical up to and including t0
        np.testing.assert_array_equal(PN.universe_mask(base)[:t0 + 1], PN.universe_mask(pz)[:t0 + 1])
        # a causal feature built from prices (20-day momentum on the adjusted close, ranked) is identical up to t0
        feat = lambda pn: H.xs_rank(np.log(pn.adj_close / np.roll(pn.adj_close, 20, axis=0)), PN.universe_mask(pn))   # noqa: E731
        np.testing.assert_array_equal(feat(base)[20:t0 + 1], feat(pz)[20:t0 + 1])
        # the label: rows <= t0-h-2 never see the poisoned prices
        yb, yp = PN.forward_open_return(base, h), PN.forward_open_return(pz, h)
        np.testing.assert_array_equal(yb[:t0 - h - 1], yp[:t0 - h - 1])
        # (row t0-h-1 still ends exactly at the open of t0, and the later rows DO change: the poison is effective)
        np.testing.assert_array_equal(yb[t0 - h - 1], yp[t0 - h - 1])
        self.assertFalse(np.array_equal(yb[t0 - h:t0 - h + 3], yp[t0 - h:t0 - h + 3], equal_nan=True))
        # untradable flags of rows < t0 never look past t0 (row t0 uses the open of t0+1 by design)
        np.testing.assert_array_equal(PN.untradable_open(base)[:t0], PN.untradable_open(pz)[:t0])

    def test_residual_ic_removes_a_proxy_but_keeps_independent_information(self):
        # The residual regression is linear in cross-sectional ranks, so a small nonlinear leakage of the control remains
        # (a few percent of the raw IC; documented in residual_ic).  The test therefore checks "almost everything is gone",
        # not exact zero.
        rng = np.random.default_rng(8)
        fin = np.isfinite(self.y5)
        ctrl = rng.standard_normal(self.y5.shape)
        indep = rng.standard_normal(self.y5.shape)
        noise = rng.standard_normal(self.y5.shape)
        lab = np.where(fin, H.xs_rank(ctrl, self.mask, fill=0.0) + noise, np.nan)
        proxy = ctrl + 0.5 * rng.standard_normal(ctrl.shape)    # a feature that is the control in disguise
        r = H.residual_ic(proxy, {"ctrl": ctrl}, lab, self.mask, 5, "dev", self.p)
        self.assertGreater(r["raw_t"], 10)                      # raw IC is strong ...
        self.assertLess(abs(r["mean"]), 0.15 * r["raw_mean"])   # ... but >85 % of it is gone after residualising
        lab2 = np.where(fin, lab + H.xs_rank(indep, self.mask, fill=0.0), np.nan)    # add independent information
        both = H.xs_rank(ctrl, self.mask, fill=0.0) + 2 * H.xs_rank(indep, self.mask, fill=0.0)
        r2 = H.residual_ic(both, {"ctrl": ctrl}, lab2, self.mask, 5, "dev", self.p)
        self.assertGreater(r2["t"], 4)


class QuintileReport(unittest.TestCase):
    def test_timing_and_gross_excess_of_a_constant_ranking(self):
        """Top-5 of 25 stocks earn +1 % intraday every day (open = previous close): gross basket 1 %, benchmark 0.2 %."""
        T, S = 60, 25
        cal = weekdays("2014-03-03", "2014-12-31")[:T]
        close = np.full((T, S), 100.0)
        opn = np.full((T, S), 100.0)
        for t in range(T):
            if t > 0:
                opn[t, 20:] = close[t - 1, 20:]
            close[t, 20:] = opn[t, 20:] * 1.01
        p = Panel(cal, [f"X{j}" for j in range(S)], opn, close, np.ones((T, S)), close, np.ones((T, S), bool))
        F = np.tile(np.arange(S, dtype=float), (T, 1))
        mask = np.ones((T, S), bool)
        r = H.quintile_report(F, p, mask, 5, "dev")
        self.assertEqual(r["mean_k"], 5.0)
        self.assertAlmostEqual(r["ann_excess_gross"], (0.01 - 0.2 * 0.01) * H.DAYS_PER_YEAR, places=6)
        self.assertGreater(r["ann_excess"], 0)
        self.assertGreater(r["ann_cost"], 0)                    # the initial purchase is charged
        q = r["quintile_y"]
        self.assertEqual(len(q["mean_y"]), 5)
        self.assertGreater(q["mean_y"][4], q["mean_y"][0])
        # an inverted ranking (long the flat stocks) earns nothing gross
        r2 = H.quintile_report(F, p, mask, 5, "dev", direction=-1)
        self.assertAlmostEqual(r2["ann_excess_gross"], (0.0 - 0.2 * 0.01) * H.DAYS_PER_YEAR, places=6)

    def test_locked_limit_up_names_are_excluded_from_the_basket(self):
        T, S = 40, 26
        cal = weekdays("2014-03-03", "2014-12-31")[:T]
        opn = np.full((T, S), 100.0)
        close = np.full((T, S), 100.0)
        opn[5:, 25] = 111.0          # the best-ranked name opens +11 % over its previous close on day 5 only
        close[5:, 25] = 111.0
        p = Panel(cal, [f"X{j}" for j in range(S)], opn, close, np.ones((T, S)), close, np.ones((T, S), bool))
        un = PN.untradable_open(p)
        self.assertEqual(int(un.sum()), 1)
        self.assertTrue(un[4, 25])
        F = np.tile(np.arange(S, dtype=float), (T, 1))
        mask = np.ones((T, S), bool)
        r = H.quintile_report(F, p, mask, 1, "dev", min_xs=20)
        # 26 candidates -> ceil(0.2 * 26) = 6 names every day, except on day 4: 25 candidates -> 5 names
        self.assertAlmostEqual(r["mean_k"], (6 * (T - 1) + 5) / T)

    def test_planted_signal_gives_positive_gross_excess_and_noise_does_not(self):
        p = make_panel(S=60, start="2012-01-02", end="2019-12-31", seed=21)
        mask = PN.universe_mask(p, n=150, min_bars=100)
        y = PN.forward_open_return(p, 5)
        good = H.quintile_report(planted(y, 0.3, 1), p, mask, 5, "dev")
        self.assertGreater(good["ann_excess_gross"], 0.05)
        self.assertEqual(good["mean_k"], 12.0)
        self.assertGreater(good["quintile_y"]["q5_minus_q1"], 0)
        noise = H.quintile_report(np.random.default_rng(3).standard_normal(y.shape), p, mask, 5, "dev")
        self.assertLess(abs(noise["ann_excess_gross"]), 0.05)
        self.assertGreater(noise["ann_turnover"], 10)           # random ranks churn the basket


# ============================================================================= price_baseline
class PriceBaselineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datas, cls.market = PB.synthetic_price_data(n_stocks=22, start="2011-01-03", end="2019-12-31", seed=4)

    def test_features_are_point_in_time(self):
        """Replacing every price/volume after t0 by noise leaves the P1-P4 rows <= t0 bit-identical."""
        panel = Panel.from_price_data(self.datas, "2012-01-01")
        t0 = int(np.flatnonzero(panel.cal >= np.datetime64("2016-03-31"))[0])
        cut = panel.cal[t0]
        rng = np.random.default_rng(1)
        poisoned = []
        for d in self.datas:
            e = copy.deepcopy(d)
            k = int((d.dates > cut).sum())
            for name in ("open", "high", "low", "close", "adjclose", "volume"):
                a = getattr(e, name)
                a[len(a) - k:] = np.exp(rng.normal(3, 1, k))
            poisoned.append(e)
        m2 = copy.deepcopy(self.market)
        k = int((m2.dates > cut).sum())
        m2.adjclose[len(m2.adjclose) - k:] = np.exp(rng.normal(3, 1, k))
        f1 = PB.price_features(self.datas, self.market, panel)
        f2 = PB.price_features(poisoned, m2, Panel.from_price_data(poisoned, "2012-01-01"))
        for name, A in f1.items():
            self.assertTrue(np.isfinite(A[:t0 + 1]).any(), name)
            np.testing.assert_array_equal(A[:t0 + 1], f2[name][:t0 + 1], err_msg=name)
            self.assertFalse(np.array_equal(A[t0 + 1:], f2[name][t0 + 1:], equal_nan=True), name)

    def test_run_writes_json_and_an_idempotent_results_section(self):
        with tempfile.TemporaryDirectory() as d:
            out, md = Path(d) / "o.json", Path(d) / "RESULTS.md"
            md.write_text("# RESULTS\n\nexisting text\n", encoding="utf-8")
            res = PB.run(self.datas, self.market, out_json=out, results_md=md, meta={"synthetic": True})
            res2 = PB.run(self.datas, self.market, out_json=out, results_md=md, meta={"synthetic": True})
            j = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(j["features"]), {"rev_5", "mom_12_1", "idio_mom", "neg_sigma"})
            self.assertFalse(j["meta"]["oos_touched"])
            for name, f in j["features"].items():
                for h in (5, 20):
                    blk = f[f"h{h}"]
                    self.assertIsNotNone(blk["screen"]["dev"]["t"], (name, h))
                    self.assertIsNotNone(blk["residual"]["t"], (name, h))
                    self.assertIsNotNone(blk["basket"]["ann_excess"], (name, h))
                    # DEV rows only: nothing from 2023 on appears in the per-year table
                    self.assertTrue(all(int(y) <= 2022 for y in blk["screen"]["dev"]["by_year"]))
            text = md.read_text(encoding="utf-8")
            self.assertEqual(text.count(PB.SECTION_BEGIN), 1)
            self.assertEqual(text.count(PB.SECTION_END), 1)
            self.assertIn("existing text", text)
            self.assertIn("## 1. 價格基準", text)
            for fid in ("P1", "P2", "P3", "P4"):
                self.assertIn(fid, text)
            self.assertEqual(res["features"]["rev_5"]["h5"]["screen"]["dev"]["mean"],
                             res2["features"]["rev_5"]["h5"]["screen"]["dev"]["mean"])    # deterministic


if __name__ == "__main__":
    unittest.main()
