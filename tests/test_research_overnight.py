"""Tests for research/overnight.py and research/run_overnight.py (P3b).  Synthetic data only: no network, no data_cache prices.

Run: python -m unittest discover -s tests -p "test_research_overnight.py"
"""
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
from scipy.special import ndtr  # noqa: E402

from backend.types import PriceData  # noqa: E402
from research import harness as H  # noqa: E402
from research import overnight as O  # noqa: E402
from research import run_overnight as RO  # noqa: E402
from research.panel import Panel  # noqa: E402

NAN = np.nan
NAMES = ("SOX", "TSM", "NVDA")


def weekdays(start, end):
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    return days[((days.astype(np.int64) + 3) % 7) < 5]


def mk_price(code, dates, op, cl, volume=None, high=None, low=None):
    n = len(dates)
    op, cl = np.asarray(op, float), np.asarray(cl, float)
    return PriceData(symbol=code, name=code, market="TWSE", kind="stock", dates=np.asarray(dates).astype("datetime64[D]"), open=op,
                     high=np.maximum(op, cl) * 1.001 if high is None else np.asarray(high, float),
                     low=np.minimum(op, cl) * 0.999 if low is None else np.asarray(low, float), close=cl, adjclose=cl.copy(),
                     volume=np.full(n, 1e6) if volume is None else np.asarray(volume, float))


def make_world(S=8, start="2010-01-04", end="2023-06-30", beta=0.8, gap_noise=0.002, intr_noise=0.003, seed=0):
    """Weekday world.  The US session dated D is usable at the next weekday S (x[S] = us_ret at the previous weekday), and
    ``gap[S] = beta_j * r_SOX(prev weekday) + noise`` so the true relation is known.  Returns (datas, us, days)."""
    rng = np.random.default_rng(seed)
    days = weekdays(start, end)
    T = len(days)
    f = rng.normal(0, 0.012, T)
    us = {"SOX": (days.copy(), f), "TSM": (days.copy(), 0.8 * f + rng.normal(0, 0.006, T)),
          "NVDA": (days.copy(), 0.9 * f + rng.normal(0, 0.01, T))}
    x_sox = np.concatenate([[0.0], f[:-1]])
    datas = []
    for j in range(S):
        gap = beta * (0.8 + 0.05 * j) * x_sox + rng.normal(0, gap_noise, T)
        intr = rng.normal(0, intr_noise, T)
        cl = np.empty(T)
        op = np.empty(T)
        cl[0] = op[0] = 50.0
        for k in range(1, T):
            op[k] = cl[k - 1] * math.exp(gap[k])
            cl[k] = op[k] * math.exp(intr[k])
        datas.append(mk_price(f"W{j:02d}", days, op, cl, volume=np.exp(rng.normal(13, 0.3, T))))
    return datas, us, days


def fake_wf(data, sigma=0.012):
    n = data.n
    return {"dates": np.asarray(data.dates).astype("datetime64[D]"), "mu": np.full(n, 0.0001), "sigma": np.full(n, sigma),
            "p_up": np.full(n, 0.5)}


def build_all(datas, us, sigma=0.012):
    stocks = [O.build_stock(d, us, fake_wf(d, sigma)) for d in datas]
    panel = Panel.from_price_data([s.data for s in stocks], "2012-01-01")
    x_shock = O.shock_series(us, panel.cal)
    return stocks, panel, x_shock


def weighted_ls(X, y, w):
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


# ============================================================================= spec
class SpecTests(unittest.TestCase):
    def test_spec_contents_and_consistency(self):
        s = O.load_spec()
        self.assertEqual(s["us_regressors"]["names"], ["SOX", "TSM", "NVDA"])
        self.assertEqual(s["gap_model"]["ewma_halflife_bars"], 120)
        self.assertEqual(s["tradability"]["shock"]["ewma_lambda"], 0.94)
        self.assertEqual(s["tradability"]["shock"]["threshold"], 2.0)
        self.assertEqual(s["stocks"]["n"], 30)
        for per in ("dev", "dev_a", "dev_b", "oos"):                   # same bounds as the harness
            a, b = H.PERIODS[per]
            self.assertEqual(s["periods"][per], [a, b])
        for key in ("premarket_update", "gap_metrics", "tradability", "oos_protocol", "frozen_note"):
            self.assertIn(key, s)
        self.assertEqual(O.HALFLIFE, 120.0)
        self.assertEqual(O.REGRESSORS, NAMES)

    def test_sha256_is_of_the_file_bytes(self):
        import hashlib
        self.assertEqual(O.spec_sha256(), hashlib.sha256(O.SPEC_PATH.read_bytes()).hexdigest())
        self.assertEqual(len(O.spec_sha256()), 64)


# ============================================================================= EWMA regression
class EwmaRegression(unittest.TestCase):
    def test_hand_computed_weights(self):
        rng = np.random.default_rng(1)
        n, hl, k = 60, 5.0, 40
        X = rng.normal(size=(n, 2))
        y = 0.3 + X @ np.array([1.0, -2.0]) + rng.normal(0, 0.5, n)
        r = O.ewma_regression(y, X, halflife=hl, min_obs=10)
        j = np.arange(k)
        w = 0.5 ** ((k - 1 - j) / hl)                       # most recent usable row (k-1) has weight 1
        beta = weighted_ls(X[:k], y[:k], w)
        np.testing.assert_allclose(r["beta"][k], beta, atol=1e-6)
        self.assertAlmostEqual(r["ybar"][k], float(np.sum(w * y[:k]) / w.sum()), places=9)
        self.assertAlmostEqual(r["yhat"][k], float(np.r_[1.0, X[k]] @ beta), places=6)
        A = np.column_stack([np.ones(k), X[:k]])
        res = y[:k] - A @ beta
        ybar = np.sum(w * y[:k]) / w.sum()
        r2 = 1 - np.sum(w * res ** 2) / np.sum(w * (y[:k] - ybar) ** 2)
        n_eff = w.sum() ** 2 / np.sum(w ** 2)
        adj = min(max(1 - (1 - r2) * (n_eff - 1) / (n_eff - 3), 0.0), O.R2_CLIP)
        self.assertAlmostEqual(r["r2"][k], adj, places=6)
        self.assertAlmostEqual(r["n_eff"][k], n_eff, places=6)

    def test_half_life_semantics(self):
        """An observation `halflife` bars old carries half the weight of the newest usable one."""
        n, hl = 30, 4.0
        y = np.zeros(n)
        y[10] = 1.0
        r = O.ewma_regression(y, np.zeros((n, 1)) + 1e-3 * np.arange(n)[:, None], halflife=hl, min_obs=5, ridge=1e-6)
        w = 0.5 ** ((20 - 1 - np.arange(20)) / hl)
        self.assertAlmostEqual(r["ybar"][20], w[10] / w.sum(), places=9)
        self.assertAlmostEqual(w[19 - int(hl)] / w[19], 0.5, places=12)

    def test_coefficients_use_only_rows_before_k(self):
        rng = np.random.default_rng(2)
        n, K = 400, 250
        X = rng.normal(size=(n, 3))
        y = X @ np.array([0.5, 0.2, -0.1]) + rng.normal(0, 0.3, n)
        ref = O.ewma_regression(y, X, 60.0, 30)
        for poison in (1e6, np.nan, -1e9):
            y2, X2 = y.copy(), X.copy()
            y2[K:], X2[K:] = poison, poison
            r = O.ewma_regression(y2, X2, 60.0, 30)
            for key in ("beta", "n_eff", "count"):                  # fitted from rows < k only: unchanged up to and including row K
                self.assertTrue(np.array_equal(r[key][:K + 1], ref[key][:K + 1], equal_nan=True), (poison, key))
            for key in ("yhat", "ybar", "r2"):                       # these are masked by "X[k] is finite": compare the rows before K
                self.assertTrue(np.array_equal(r[key][:K], ref[key][:K], equal_nan=True), (poison, key))

    def test_prediction_at_k_uses_x_of_k_only_through_the_dot_product(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(120, 3))
        y = X @ np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.1, 120)
        base = O.ewma_regression(y, X, 40.0, 30)
        X2 = X.copy()
        X2[100] += 5.0
        r = O.ewma_regression(y, X2, 40.0, 30)
        np.testing.assert_array_equal(r["beta"][:101], base["beta"][:101])
        self.assertNotEqual(r["yhat"][100], base["yhat"][100])

    def test_recovers_true_coefficients(self):
        rng = np.random.default_rng(4)
        n = 3000
        X = rng.normal(0, 0.01, size=(n, 3))
        y = 0.0002 + X @ np.array([0.8, 0.0, 0.3]) + rng.normal(0, 0.002, n)
        r = O.ewma_regression(y, X, 120.0, 60)
        np.testing.assert_allclose(r["beta"][-1, 1:], [0.8, 0.0, 0.3], atol=0.12)
        self.assertGreater(float(np.nanmean(r["r2"][500:])), 0.8)

    def test_min_obs_and_nonfinite_rows(self):
        rng = np.random.default_rng(5)
        n = 200
        X = rng.normal(size=(n, 3))
        y = X[:, 0] + rng.normal(0, 0.1, n)
        y[50:60] = np.nan
        X[70, 1] = np.nan
        r = O.ewma_regression(y, X, 50.0, 60)
        self.assertFalse(r["ok"][:60].any())                  # fewer than 60 usable rows before row 60 (10 are NaN)
        self.assertTrue(r["ok"][100])
        self.assertFalse(r["ok"][70])                          # X[k] itself is NaN -> no prediction
        self.assertEqual(r["count"][100], 100 - 10 - 1)       # NaN rows are not counted

    def test_ewma_sigma_hand_and_causal(self):
        r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        s = O.ewma_sigma(r, lam=0.9, seed_n=3, min_obs=3)
        v = np.mean([0.01, 0.04, 0.09])
        self.assertTrue(np.isnan(s[:3]).all())
        self.assertAlmostEqual(s[3], math.sqrt(v), places=12)
        v = 0.9 * v + 0.1 * 0.16
        self.assertAlmostEqual(s[4], math.sqrt(v), places=12)
        r2 = r.copy()
        r2[4:] = [np.nan, 1e6]
        s2 = O.ewma_sigma(r2, lam=0.9, seed_n=3, min_obs=3)
        self.assertTrue(np.array_equal(s2[:5], s[:5], equal_nan=True))      # sigma[k] depends on r[<k] only


# ============================================================================= availability of the US return
class Alignment(unittest.TestCase):
    def us_one(self, d, v):
        z = (np.array([np.datetime64(d, "D")]), np.array([v]))
        return {nm: z for nm in NAMES}

    def test_same_day_us_bar_is_not_usable_and_next_day_is(self):
        tw = weekdays("2020-03-02", "2020-03-06")                                  # Mon..Fri
        x = O.align_regressors(self.us_one("2020-03-04", 0.05), tw)               # US session dated Wed
        self.assertTrue(np.isnan(x[0]).all())                                      # first row has no bounded window
        np.testing.assert_array_equal(x[1:, 0], [0.0, 0.0, 0.05, 0.0])             # Tue 0, Wed 0 (D == S not usable), Thu 0.05, Fri 0
        self.assertEqual(x.shape, (5, 3))

    def test_monday_uses_friday_and_a_taiwan_holiday_accumulates(self):
        us_dates = weekdays("2020-03-02", "2020-03-13")
        ret = np.arange(1, len(us_dates) + 1) * 0.01
        us = {nm: (us_dates, ret) for nm in NAMES}
        tw = np.array(["2020-03-06", "2020-03-09", "2020-03-11"], dtype="datetime64[D]")   # Fri, Mon, Wed (Tue closed)
        x = O.align_regressors(us, tw)[:, 0]
        self.assertAlmostEqual(x[1], ret[4], places=12)                           # Mon <- Fri session only (Sat/Sun have no bar)
        self.assertAlmostEqual(x[2], ret[5] + ret[6], places=12)                  # Wed <- Mon + Tue sessions
        self.assertTrue(np.isnan(x[0]))

    def test_nan_return_in_window_stays_nan(self):
        us_dates = weekdays("2020-03-02", "2020-03-06")
        ret = np.array([0.01, np.nan, 0.02, 0.03, 0.04])
        us = {nm: (us_dates, ret) for nm in NAMES}
        x = O.align_regressors(us, weekdays("2020-03-02", "2020-03-06"))[:, 0]
        self.assertTrue(np.isnan(x[2]))                                            # Wed's window holds Tue's NaN
        self.assertAlmostEqual(x[3], 0.02)

    def test_stock_x_equals_previous_weekday_us_return_in_the_world(self):
        datas, us, days = make_world(S=1, end="2011-06-30")
        st = O.build_stock(datas[0], us)
        np.testing.assert_allclose(st.X[1:, 0], us["SOX"][1][:-1], atol=0)
        # and the gap of the world really is driven by it (sanity of the generator)
        self.assertGreater(np.corrcoef(st.gap[1:], st.X[1:, 0])[0, 1], 0.9)


# ============================================================================= gap series / stock build
class StockBuild(unittest.TestCase):
    def test_gap_intraday_and_close_return_use_adjusted_prices_of_consecutive_bars(self):
        dates = weekdays("2020-03-02", "2020-03-06")
        op = [10.0, 10.5, 10.2, 9.0, 9.6]
        cl = [10.2, 10.0, 10.0, 9.5, 9.9]
        d = mk_price("T", dates, op, cl)
        d.adjclose = d.close * np.array([0.9, 0.9, 1.0, 1.0, 1.0])               # an ex-dividend between bar 1 and 2
        us = {nm: (dates, np.zeros(5)) for nm in NAMES}
        st = O.build_stock(d, us)
        ao = d.open * d.adjclose / d.close
        self.assertTrue(np.isnan(st.gap[0]))
        for k in range(1, 5):
            self.assertAlmostEqual(st.gap[k], math.log(ao[k] / d.adjclose[k - 1]), places=12)
            self.assertAlmostEqual(st.R[k], math.log(d.adjclose[k] / d.adjclose[k - 1]), places=12)
        for k in range(5):
            self.assertAlmostEqual(st.intraday[k], math.log(d.adjclose[k] / ao[k]), places=12)

    def test_placeholder_bars_are_dropped_before_the_gap_is_defined(self):
        dates = weekdays("2020-03-02", "2020-03-06")
        cl = np.array([10.0, 10.2, 10.2, 10.4, 10.5])
        op = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
        vol = np.array([1e6, 1e6, 0.0, 1e6, 1e6])
        hi = np.maximum(op, cl) * 1.001
        lo = np.minimum(op, cl) * 0.999
        hi[2] = lo[2] = 10.2
        d = mk_price("T", dates, op, cl, volume=vol, high=hi, low=lo)
        real, keep = O.clean_data(d)
        self.assertEqual(real.n, 4)
        self.assertEqual(list(keep), [True, True, False, True, True])
        st = O.build_stock(real, {nm: (dates, np.zeros(5)) for nm in NAMES})
        self.assertAlmostEqual(st.gap[2], math.log(10.3 / 10.2), places=12)       # previous REAL close, not the filler bar

    def test_walk_forward_outputs_are_shifted_to_the_target_day(self):
        datas, us, _ = make_world(S=1, end="2010-12-31")
        d = datas[0]
        wf = {"dates": d.dates, "mu": np.arange(d.n) * 1e-5, "sigma": np.full(d.n, 0.01), "p_up": np.linspace(0.4, 0.6, d.n)}
        st = O.build_stock(d, us, wf)
        self.assertTrue(np.isnan(st.mu0[0]))
        np.testing.assert_array_equal(st.mu0[1:], wf["mu"][:-1])                  # mu at the origin k-1 targets R[k]
        np.testing.assert_array_equal(st.p0[1:], wf["p_up"][:-1])
        bad = dict(wf, dates=d.dates + np.timedelta64(1, "D"))
        with self.assertRaises(ValueError):
            O.build_stock(d, us, bad)

    def test_no_lookahead_end_to_end(self):
        """Truncating (or poisoning) the stock's data after bar K leaves everything at rows <= K bit-identical."""
        datas, us, _ = make_world(S=1, end="2012-06-29")
        d = datas[0]
        K = 400
        full = O.build_stock(d, us)
        cut = O.build_stock(d.slice(K + 1), us)
        poisoned = mk_price("P", d.dates, np.r_[d.open[:K + 1], np.full(d.n - K - 1, 1e5)], np.r_[d.close[:K + 1], np.full(d.n - K - 1, 3.0)])
        pois = O.build_stock(poisoned, us)
        for other in (cut, pois):
            for key in ("yhat", "ybar", "r2"):
                self.assertTrue(np.array_equal(other.reg[key][:K + 1], full.reg[key][:K + 1], equal_nan=True), key)
            self.assertTrue(np.array_equal(other.cond[:K + 1], full.cond[:K + 1], equal_nan=True))
            self.assertTrue(np.array_equal(other.gap[:K + 1], full.gap[:K + 1], equal_nan=True))

    def test_select_codes_rule(self):
        d_old = mk_price("A", weekdays("2009-01-05", "2014-12-31"), np.full(1, 1.0).repeat(len(weekdays("2009-01-05", "2014-12-31"))),
                         np.full(len(weekdays("2009-01-05", "2014-12-31")), 1.0))
        d_new = mk_price("B", weekdays("2012-06-01", "2014-12-31"), np.ones(len(weekdays("2012-06-01", "2014-12-31"))),
                         np.ones(len(weekdays("2012-06-01", "2014-12-31"))))
        got = O.select_codes([{"code": "B"}, {"code": "A"}, {"code": "C"}], n=5, loader=lambda c: {"A": d_old, "B": d_new}[c])
        self.assertEqual(got, ["A"])                                               # B too short, C fails to load


# ============================================================================= item 1: gap metrics
class GapMetrics(unittest.TestCase):
    def test_hand_computed_r2_and_hit_rate(self):
        T, S = 6, 6
        cal = weekdays("2014-03-03", "2014-12-31")[:T]
        gap = np.tile(np.array([[0.02, -0.01, 0.03, -0.02, 0.01, 0.0]]).T, (1, S))
        yhat = np.tile(np.array([[0.01, -0.01, 0.02, 0.01, 0.01, 0.01]]).T, (1, S))
        ybar = np.full((T, S), 0.002)
        M = {"gap": gap, "yhat": yhat, "ybar": ybar, "ok": np.ones((T, S), bool), "cond": yhat - ybar,
             "beta": np.full((T, S, 4), 0.1)}
        rep = O.gap_report(M, np.ones(T, bool), cal, list("abcdef"))
        g, y = gap[:, 0], yhat[:, 0]
        self.assertAlmostEqual(rep["r2_oos_vs_zero"], 1 - np.sum((g - y) ** 2) / np.sum(g ** 2), places=12)
        self.assertAlmostEqual(rep["r2_oos_vs_mean"], 1 - np.sum((g - y) ** 2) / np.sum((g - 0.002) ** 2), places=12)
        nz = g != 0
        self.assertAlmostEqual(rep["hit_rate"], np.mean(np.sign(y[nz]) == np.sign(g[nz])), places=12)    # 4/5 (the 0.0 gap is excluded)
        self.assertAlmostEqual(rep["hit_rate"], 0.8, places=12)
        self.assertAlmostEqual(rep["share_gap_positive"], 3 / 5, places=12)
        self.assertAlmostEqual(rep["hit_base_const"], 3 / 5, places=12)
        self.assertEqual(rep["n_obs"], T * S)

    def test_informative_world_has_positive_oos_r2_and_beats_the_base_rate(self):
        datas, us, _ = make_world(S=8, end="2016-12-30", beta=0.8)
        stocks, panel, xs = build_all(datas, us)
        M = O.build_matrices(stocks, panel)
        rows = H.period_rows(panel.cal, "dev", 0)
        rep = O.gap_report(M, rows, panel.cal, list(panel.codes))
        self.assertGreater(rep["r2_oos_vs_mean"], 0.7)
        self.assertGreater(rep["loss_diff_vs_mean"]["t"], 5)
        self.assertGreater(rep["hit_rate"], rep["hit_base_const"] + 0.15)
        self.assertGreater(rep["hit_minus_base"]["t"], 5)
        self.assertEqual(rep["n_stocks_r2_pos_vs_mean"], 8)
        self.assertAlmostEqual(rep["beta_mean"]["SOX"] + rep["beta_mean"]["TSM"] + rep["beta_mean"]["NVDA"], 0.8 * 0.95, delta=0.25)

    def test_no_skill_world_has_no_r2(self):
        datas, us, _ = make_world(S=8, end="2016-12-30", beta=0.0, gap_noise=0.006)
        stocks, panel, xs = build_all(datas, us)
        M = O.build_matrices(stocks, panel)
        rep = O.gap_report(M, H.period_rows(panel.cal, "dev", 0), panel.cal, list(panel.codes))
        self.assertLess(rep["r2_oos_vs_mean"], 0.01)
        self.assertLess(rep["loss_diff_vs_mean"]["t"], 2.0)          # fitting noise costs accuracy; it never helps
        self.assertLess(abs(rep["hit_rate"] - rep["hit_bench_causal_mean"]), 0.03)

    def test_date_mean_needs_min_n(self):
        A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, NAN], [1.0, NAN, NAN, NAN, NAN, NAN]])
        m = O.date_mean(A, 5)
        self.assertAlmostEqual(m[0], 3.0)
        self.assertTrue(np.isnan(m[1]))


# ============================================================================= item 2: pre-market update
class PremarketUpdate(unittest.TestCase):
    def test_formula_by_hand(self):
        mu1, s1, p1, p1m, pn = O.premarket_update(0.001, 0.02, 0.01, 0.36)
        self.assertAlmostEqual(mu1, 0.011, places=15)
        self.assertAlmostEqual(s1, 0.02 * math.sqrt(0.64), places=15)             # sigma'^2 = sigma^2 (1 - R2)  ->  0.016
        self.assertAlmostEqual(s1, 0.016, places=15)
        self.assertAlmostEqual(p1, float(ndtr(0.011 / 0.016)), places=15)
        self.assertAlmostEqual(p1m, float(ndtr(0.011 / 0.02)), places=15)
        self.assertAlmostEqual(pn, float(ndtr(0.001 / 0.02)), places=15)
        arr = O.premarket_update(np.array([0.0, 0.0]), np.array([0.02, 0.02]), np.array([0.02, -0.02]), np.array([0.75, 0.75]))
        self.assertAlmostEqual(arr[2][0], float(ndtr(2.0)), places=12)
        self.assertAlmostEqual(arr[2][1], float(ndtr(-2.0)), places=12)

    def matrices(self, R, cond, mu0=0.0, sigma0=0.02, r2=0.75, p0=0.5):
        T, S = R.shape
        full = lambda v: np.full((T, S), v, float)   # noqa: E731
        return {"R": R, "cond": cond, "mu0": full(mu0), "sigma0": full(sigma0), "r2": full(r2), "p0": full(p0),
                "ok": np.ones((T, S), bool), "gap": np.full((T, S), 123.0)}

    def test_brier_and_hit_by_hand_and_flat_days_excluded(self):
        S = 6
        cal = weekdays("2014-03-03", "2014-12-31")[:3]
        R = np.array([[0.01] * S, [-0.01] * S, [0.0] * S])                         # up, down, flat
        M = self.matrices(R, np.full((3, S), 0.02))
        rep = O.pup_report(M, np.ones(3, bool), cal, list("abcdef"))
        self.assertEqual(rep["n_obs"], 2 * S)                                      # the flat day is out
        p1 = float(ndtr(2.0))
        b_up, b_dn = (1 - p1) ** 2, p1 ** 2
        self.assertAlmostEqual(rep["variants"]["update"]["brier"], (b_up + b_dn) / 2, places=12)
        self.assertAlmostEqual(rep["variants"]["engine"]["brier"], 0.25, places=12)
        self.assertAlmostEqual(rep["variants"]["update"]["hit"], 0.5, places=12)   # right on the up day, wrong on the down day
        c = rep["comparisons"]["update_vs_engine"]
        self.assertAlmostEqual(c["d_brier"]["mean"], 0.25 - (b_up + b_dn) / 2, places=12)
        self.assertEqual(c["n_stocks_improved"], 0)                                # mean over the two days is negative for every stock
        # the update only moves mu: mean-only variant
        p1m = float(ndtr(1.0))
        self.assertAlmostEqual(rep["variants"]["update_mu_only"]["brier"], ((1 - p1m) ** 2 + p1m ** 2) / 2, places=12)

    def test_realised_gap_is_never_used(self):
        rng = np.random.default_rng(0)
        T, S = 50, 6
        cal = weekdays("2014-03-03", "2014-12-31")[:T]
        M = self.matrices(rng.normal(0, 0.02, (T, S)), rng.normal(0, 0.01, (T, S)))
        a = O.pup_report(M, np.ones(T, bool), cal, list("abcdef"))
        M2 = dict(M, gap=rng.normal(size=(T, S)))
        b = O.pup_report(M2, np.ones(T, bool), cal, list("abcdef"))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_informative_cond_improves_brier_with_a_significant_hac_t(self):
        datas, us, _ = make_world(S=8, end="2016-12-30", beta=0.8)
        stocks, panel, xs = build_all(datas, us, sigma=0.0105)
        M = O.build_matrices(stocks, panel)
        rep = O.pup_report(M, H.period_rows(panel.cal, "dev", 0), panel.cal, list(panel.codes))
        c = rep["comparisons"]["update_vs_engine"]
        self.assertGreater(c["d_brier"]["mean"], 0.02)
        self.assertGreater(c["d_brier"]["t"], 5)
        self.assertGreater(c["d_hit"]["mean"], 0.1)
        self.assertEqual(c["n_stocks_improved"], 8)
        self.assertLess(rep["variants"]["update"]["brier"], rep["variants"]["norm"]["brier"])
        self.assertLess(rep["mean_sigma_ratio"], 1.0)
        self.assertGreater(rep["corr_cond_R"], 0.7)
        # the comparison against the normal baseline isolates the update
        self.assertGreater(rep["comparisons"]["update_vs_norm"]["d_brier"]["mean"], 0.02)

    def test_noise_cond_does_not_help(self):
        datas, us, _ = make_world(S=8, end="2016-12-30", beta=0.0, gap_noise=0.006)
        stocks, panel, xs = build_all(datas, us, sigma=0.0105)
        M = O.build_matrices(stocks, panel)
        rep = O.pup_report(M, H.period_rows(panel.cal, "dev", 0), panel.cal, list(panel.codes))
        self.assertLess(rep["comparisons"]["update_vs_norm"]["d_brier"]["mean"], 0.002)
        self.assertLess(abs(rep["comparisons"]["update_mu_only_vs_norm"]["d_brier"]["mean"]), 0.002)


# ============================================================================= item 3: shock / tradability
class Tradability(unittest.TestCase):
    def test_shock_z_is_causal_and_thresholded(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.01, 300)
        x[200] = 0.05
        z = O.shock_z(x)
        self.assertGreater(abs(z[200]), 2.0)
        self.assertTrue(np.isnan(z[:60]).all())
        x2 = x.copy()
        x2[200:] = 9.0                                        # poison the day itself and the future
        z2 = O.shock_z(x2)
        self.assertTrue(np.array_equal(z[:200], z2[:200], equal_nan=True))
        sigma = O.ewma_sigma(x, 0.94, 20, 60)
        self.assertAlmostEqual(z[200], x[200] / sigma[200], places=12)           # the shock day is scored against the PAST volatility
        self.assertEqual(sigma[200], O.ewma_sigma(x2, 0.94, 20, 60)[200])

    def test_entry_is_the_open_after_the_shock_day_and_exit_five_days_later(self):
        cal = weekdays("2013-03-01", "2014-12-31")
        T, S = len(cal), 25
        S0 = 150
        opens = np.full((T, S), 100.0)
        opens[S0 + 6:] = 110.0                                 # the price steps up exactly at the open of S0+6
        closes = opens.copy()
        has = np.ones((T, S), bool)
        vol = np.full((T, S), 1e6)
        panel = Panel(cal, [f"C{j}" for j in range(S)], opens, closes, vol, closes * vol, has)
        rng = np.random.default_rng(1)
        x = 0.01 * rng.choice([-1.0, 1.0], T)                  # constant magnitude -> |z| = 1 on ordinary days
        x[S0] = 0.08                                           # |z| >> 2 on S0 (only)
        M = {"ok": np.ones((T, S), bool), "cond": rng.normal(0, 0.003, (T, S)), "intraday": rng.normal(0, 0.01, (T, S)),
             "gap": rng.normal(0, 0.01, (T, S))}
        rep = O.tradability_report(M, H.period_rows(cal, "dev", 0), cal, panel, x, "dev")["shock"]
        self.assertEqual((rep["n_pos"], rep["n_neg"]), (1, 0))
        ev = rep["excess_pos"]
        drift = rep["unconditional_5d_drift"]
        # y5[S0] = ln(O[S0+6]/O[S0+1]) = ln(1.1); entering at the S0 open itself (the gapped price) would have given 0
        self.assertAlmostEqual(ev["mean"], math.log(1.1) - drift, places=12)
        self.assertGreater(ev["mean"], 0.09)
        self.assertEqual(ev["n"], 1)
        # rows whose window [r+1, r+6] straddles the step are r = S0 .. S0+4 (five of them); r = S0-1 ends at O[S0+5], before it
        y5 = O.forward_open_return(panel, 5)
        self.assertEqual(int(np.sum(np.abs(y5[:, 0]) > 1e-12)), 5)
        self.assertAlmostEqual(y5[S0, 0], math.log(1.1), places=12)
        self.assertAlmostEqual(y5[S0 - 1, 0], 0.0, places=12)                     # ln(O[S0+5]/O[S0]) = 0 -- the step is not yet inside

    def test_signed_group_flips_negative_shocks(self):
        cal = weekdays("2013-03-01", "2014-12-31")
        T, S = len(cal), 25
        opens = np.full((T, S), 100.0)
        opens[100 + 6:] *= 1.1                                 # continuation after a positive shock at 100
        opens[200 + 6:] *= 1 / 1.1                             # ... and after a negative shock at 200
        panel = Panel(cal, [f"C{j}" for j in range(S)], opens, opens.copy(), np.full((T, S), 1e6), opens * 1e6, np.ones((T, S), bool))
        rng = np.random.default_rng(2)
        x = 0.01 * rng.choice([-1.0, 1.0], T)
        x[100], x[200] = 0.08, -0.08
        M = {"ok": np.ones((T, S), bool), "cond": rng.normal(0, 0.003, (T, S)), "intraday": rng.normal(0, 0.01, (T, S)),
             "gap": rng.normal(0, 0.01, (T, S))}
        rep = O.tradability_report(M, H.period_rows(cal, "dev", 0), cal, panel, x, "dev")["shock"]
        self.assertEqual((rep["n_pos"], rep["n_neg"]), (1, 1))
        self.assertGreater(rep["excess_pos"]["mean"], 0.05)
        self.assertLess(rep["excess_neg"]["mean"], -0.05)
        self.assertGreater(rep["excess_signed"]["mean"], 0.05)                     # both continue -> sign-adjusted mean is positive
        self.assertEqual(rep["excess_signed"]["n"], 2)

    def test_intraday_ic_and_sign_strategy(self):
        datas, us, _ = make_world(S=24, end="2015-12-31")
        stocks, panel, xs = build_all(datas, us)
        M = O.build_matrices(stocks, panel)
        rows = H.period_rows(panel.cal, "dev", 0)
        t = O.tradability_report(M, rows, panel.cal, panel, xs, "dev")
        # intraday noise is independent of the predicted gap in the world -> no tradable edge
        self.assertLess(abs(t["intraday_ic"]["t"]), 3.5)
        self.assertLess(abs(t["sign_strategy"]["t"]), 3.5)
        self.assertEqual(t["sign_strategy"]["round_trip_cost_plan"], 0.0057)


# ============================================================================= DEV / OOS boundary and guards
class Boundaries(unittest.TestCase):
    def test_period_rows_split_at_2022_12_31(self):
        cal = np.array(["2022-12-29", "2022-12-30", "2023-01-02", "2023-01-03"], dtype="datetime64[D]")
        np.testing.assert_array_equal(H.period_rows(cal, "dev", 0), [True, True, False, False])
        np.testing.assert_array_equal(H.period_rows(cal, "oos", 0), [False, False, True, True])
        self.assertEqual(O.DEV_END, np.datetime64("2022-12-31"))
        self.assertEqual(O.DEV_START, np.datetime64("2013-01-01"))

    def test_start_index_covers_the_first_2013_bar(self):
        dates = np.array(["2012-12-27", "2012-12-28", "2013-01-02", "2013-01-03"], dtype="datetime64[D]")
        s = O.start_index(dates)
        self.assertEqual(s, 1)                                                     # origin 2012-12-28 -> target = the first 2013 bar
        self.assertEqual(str(dates[s + 1]), "2013-01-02")

    def test_dev_report_is_identical_when_everything_from_2023_is_poisoned(self):
        datas, us, days = make_world(S=8, start="2010-01-04", end="2023-06-30", beta=0.8, seed=5)
        cut = np.datetime64("2023-01-01")

        def poison(d):
            m = d.dates >= cut
            op, cl = d.open.copy(), d.close.copy()
            op[m], cl[m] = 7777.0, 0.01
            return mk_price(d.symbol, d.dates, op, cl, volume=d.volume)

        us2 = {nm: (dd, np.where(dd >= cut, 5.0, r)) for nm, (dd, r) in us.items()}
        a = build_all(datas, us)
        b = build_all([poison(d) for d in datas], us2)
        ra = O.evaluate(*a, "dev")
        rb = O.evaluate(*b, "dev")
        self.assertEqual(json.dumps(ra, sort_keys=True), json.dumps(rb, sort_keys=True))
        self.assertGreater(ra["gap"]["n_obs"], 5000)
        # while the OOS reports of the two worlds are of course different
        with tempfile.TemporaryDirectory() as tmp:
            old = H.LOOKS_PATH
            H.LOOKS_PATH = Path(tmp) / "LOOKS.md"
            try:
                oa = O.evaluate(*a, "oos", spec_hash="x" * 64)
                ob = O.evaluate(*b, "oos", spec_hash="x" * 64)
            finally:
                H.LOOKS_PATH = old
        self.assertNotEqual(json.dumps(oa, sort_keys=True), json.dumps(ob, sort_keys=True))

    def test_dev_halves_partition_dev(self):
        datas, us, _ = make_world(S=8, end="2022-12-30", beta=0.8, seed=6)
        a = build_all(datas, us)
        full = O.evaluate(*a, "dev")["gap"]["n_obs"]
        pa = O.evaluate(*a, "dev_a")["gap"]["n_obs"]
        pb = O.evaluate(*a, "dev_b")["gap"]["n_obs"]
        self.assertEqual(full, pa + pb)

    def test_oos_requires_the_spec_hash_and_is_logged_once(self):
        datas, us, _ = make_world(S=6, end="2023-03-31", seed=7)
        a = build_all(datas, us)
        with tempfile.TemporaryDirectory() as tmp:
            looks = Path(tmp) / "LOOKS.md"
            old = H.LOOKS_PATH
            H.LOOKS_PATH = looks
            try:
                with self.assertRaises(H.OOSGuardError):
                    O.evaluate(*a, "oos")                                          # no hash -> raises before computing
                self.assertFalse(looks.exists())
                O.evaluate(*a, "dev")                                              # DEV never writes a look
                self.assertFalse(looks.exists())
                O.evaluate(*a, "oos", spec_hash=O.spec_sha256())
                text = looks.read_text(encoding="utf-8")
                self.assertEqual(text.count("overnight P3b"), 1)
                self.assertIn(O.spec_sha256(), text)
            finally:
                H.LOOKS_PATH = old

    def test_runner_checks_the_hash_before_touching_any_data(self):
        for bad in (None, "", "deadbeef", O.spec_sha256()[:-1] + "0" if not O.spec_sha256().endswith("0") else O.spec_sha256()[:-1] + "1"):
            with self.assertRaises(H.OOSGuardError):
                RO.run("oos", spec_hash=bad, results_md=None)

    def test_columns_must_match(self):
        datas, us, _ = make_world(S=6, end="2014-03-31", seed=8)
        stocks, panel, xs = build_all(datas, us)
        with self.assertRaises(ValueError):
            O.evaluate(stocks[:-1], panel, xs, "dev")


class RenderAndResults(unittest.TestCase):
    def test_render_and_idempotent_block_update(self):
        datas, us, _ = make_world(S=8, end="2016-12-30", seed=9)
        a = build_all(datas, us)
        res = {"meta": {"generated_at_utc": "2026-01-01 00:00", "git_commit": "abc", "spec_sha256": O.spec_sha256(), "n_stocks": 8,
                        "min_bars_before_dev": 500, "min_train": 500, "refit_every": 21},
               "periods": {"dev": O.evaluate(*a, "dev")}}
        text = O.render_section(res, "dev")
        for needle in ("P3b", "OOS R²", "Brier", "盤前預測更準", "可交易的優勢", "57 bps", "S+1 開盤"):
            self.assertIn(needle, text)
        json.dumps(res)                                                            # JSON-serialisable
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "RESULTS.md"
            p.write_text("# head\n\n<!-- BEGIN price_baseline -->\nkeep me\n<!-- END price_baseline -->\n", encoding="utf-8")
            b, e = RO.MARKERS["dev"]
            O.update_results_md(p, text, b, e)
            O.update_results_md(p, text, b, e)
            out = p.read_text(encoding="utf-8")
            self.assertEqual(out.count(b), 1)
            self.assertIn("keep me", out)


if __name__ == "__main__":
    unittest.main()

