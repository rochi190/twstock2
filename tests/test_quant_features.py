"""Causal features: every value at row i uses bars <= i only; market alignment; cleaning of data artefacts."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import garch_returns, iid_returns, make_pricedata  # noqa: E402

from backend.quant import features as F  # noqa: E402


def make_pair(n=900, seed=0):
    stock = make_pricedata(garch_returns(n, seed=seed + 1), "STK", seed=seed)
    mkt = make_pricedata(garch_returns(n, seed=seed + 2, v=0.011 ** 2, drift=0.0002), "MKT", seed=seed + 5, kind="index")
    return stock, mkt


ARRAYS = ("X", "raw", "rc", "cs", "sigma_ref", "v97", "drift", "logvol")


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


class TestCausality(unittest.TestCase):
    def test_features_are_prefix_functions(self):
        stock, mkt = make_pair()
        full = F.build_features(stock, mkt)
        for cut in (120, 301, 555, 899):
            part = F.build_features(stock.slice(cut), mkt)
            for name in ARRAYS:
                self.assertTrue(same(getattr(full, name)[:cut], getattr(part, name)), f"{name} differs at cut {cut}")

    def test_changing_the_future_does_not_change_the_past(self):
        stock, mkt = make_pair(seed=3)
        base = F.build_features(stock, mkt)
        i = 500
        s2 = make_pricedata(garch_returns(899, seed=4), "STK", seed=3)
        # same first 501 rows, different afterwards
        for name in ("open", "high", "low", "close", "adjclose", "volume"):
            getattr(s2, name)[: i + 1] = getattr(stock, name)[: i + 1]
        pert = F.build_features(s2, mkt)
        for name in ARRAYS:
            self.assertTrue(same(getattr(base, name)[: i + 1], getattr(pert, name)[: i + 1]), name)
        self.assertFalse(same(base.X[i + 1:], pert.X[i + 1:]))

    def test_market_bars_after_a_stock_date_do_not_leak(self):
        stock, mkt = make_pair(seed=5)
        base = F.build_features(stock, mkt)
        i = 600
        m2 = make_pricedata(garch_returns(899, seed=9), "MKT", seed=5, kind="index")
        k = int(np.searchsorted(mkt.dates, stock.dates[i], side="right"))  # market bars dated <= date_i
        for name in ("open", "high", "low", "close", "adjclose", "volume"):
            getattr(m2, name)[:k] = getattr(mkt, name)[:k]
        pert = F.build_features(stock, m2)
        for name in ARRAYS:
            self.assertTrue(same(getattr(base, name)[: i + 1], getattr(pert, name)[: i + 1]), name)


class TestMarketAlignment(unittest.TestCase):
    def test_forward_fill_from_the_past_only(self):
        stock = make_pricedata(iid_returns(30, seed=1), "A")
        mkt = make_pricedata(iid_returns(30, seed=2), "M", kind="index")
        keep = np.ones(mkt.n, bool)
        keep[[5, 6, 12]] = False  # market has no bars on these dates
        sub = type(mkt)(mkt.symbol, mkt.name, mkt.market, mkt.kind, mkt.dates[keep], mkt.open[keep], mkt.high[keep],
                        mkt.low[keep], mkt.close[keep], mkt.adjclose[keep], mkt.volume[keep])
        lm = F.align_market_logprice(stock.dates, sub)
        for i, d in enumerate(stock.dates):
            j = np.searchsorted(sub.dates, d, side="right") - 1
            self.assertAlmostEqual(lm[i], np.log(sub.adjclose[j]))
        self.assertAlmostEqual(lm[5], lm[4])  # carried forward, never backward
        self.assertAlmostEqual(lm[6], lm[4])

    def test_stock_dates_before_the_market_starts_are_nan_and_future_market_ignored(self):
        stock = make_pricedata(iid_returns(40, seed=3), "A", start="2005-01-03")
        mkt = make_pricedata(iid_returns(40, seed=4), "M", kind="index", start="2005-01-17")  # starts later
        lm = F.align_market_logprice(stock.dates, mkt)
        self.assertTrue(np.all(np.isnan(lm[stock.dates < mkt.dates[0]])))
        self.assertTrue(np.all(np.isfinite(lm[stock.dates >= mkt.dates[0]])))
        short = mkt.slice(10)  # market ends earlier than the stock: last known value carried forward
        lm2 = F.align_market_logprice(stock.dates, short)
        self.assertAlmostEqual(lm2[-1], np.log(short.adjclose[-1]))

    def test_no_market_disables_market_features(self):
        stock, _ = make_pair(seed=6)
        fs = F.build_features(stock, None)
        self.assertFalse(fs.has_market)
        for key in ("mkt_ret", "mkt_mom20", "mkt_vol", "idio_mom"):
            self.assertTrue(np.all(fs.X[:, fs.keys.index(key)] == 0.0))
        fs2 = F.build_features(stock, make_pair(seed=6)[1])
        self.assertTrue(fs2.has_market)
        self.assertGreater(np.abs(fs2.X[:, fs2.keys.index("mkt_mom20")]).max(), 0.5)


class TestCleaning(unittest.TestCase):
    def test_clean_returns_thresholds(self):
        r = np.array([0.0, 0.05, -0.12, 0.13, -0.2, 0.25, 0.2501, -1.39, np.nan, np.inf])
        np.testing.assert_allclose(F.clean_returns(r), [0.0, 0.05, -0.12, 0.12, -0.12, 0.12, 0.0, 0.0, 0.0, 0.0])

    def test_unadjusted_split_glitch_is_neutralised(self):
        n = 800
        ret = garch_returns(n, seed=7)
        ret[300] = -1.389  # e.g. 0050 on 2014-01-02
        d = make_pricedata(ret, "SPLIT")
        fs = F.build_features(d, None)
        self.assertEqual(fs.rc[301], 0.0)  # row 301 carries return ret[300]
        self.assertTrue(np.all(np.isfinite(fs.X)) and np.abs(fs.X).max() <= F.Z_CLIP)
        self.assertLess(fs.sigma_ref.max(), 0.2)  # no volatility explosion from the glitch
        self.assertTrue(np.all(np.isfinite(fs.sigma_ref)))
        clean = F.build_features(make_pricedata(np.where(np.arange(n) == 300, 0.0, ret), "SPLIT"), None)
        np.testing.assert_allclose(fs.rc, clean.rc)


class TestBuildingBlocks(unittest.TestCase):
    def test_ewma_variance_matches_loop(self):
        rng = np.random.default_rng(1)
        r = 0.01 * rng.standard_normal(200)
        r[0] = 0.0
        lam, v0, w = 0.94, 2e-4, 10.0
        v = F.ewma_variance(r, lam, v0, w)
        num, den = w * v0, w
        ref = [num / den]
        for t in range(1, len(r)):
            num, den = lam * num + r[t] ** 2, lam * den + 1.0
            ref.append(num / den)
        np.testing.assert_allclose(v, ref, rtol=1e-12)

    def test_rolling_sum_lagged_diff_ffill(self):
        x = np.arange(1.0, 11.0)
        rs = F.rolling_sum(x, 3)
        self.assertTrue(np.all(np.isnan(rs[:2])))
        np.testing.assert_allclose(rs[2:], [6, 9, 12, 15, 18, 21, 24, 27])
        ld = F.lagged_diff(x, 4)
        self.assertTrue(np.all(np.isnan(ld[:4])))
        np.testing.assert_allclose(ld[4:], 4.0)
        np.testing.assert_allclose(F.ffill(np.array([np.nan, 1.0, np.nan, np.nan, 2.0, np.nan]), -9.0), [-9, 1, 1, 1, 2, 2])

    def test_expanding_zscore_formula_and_prefix(self):
        rng = np.random.default_rng(2)
        f = rng.standard_normal(300) * 2 + 1
        f[:40] = np.nan
        z = F.expanding_zscore(f, prior_count=60.0, clip=4.0)
        for i in (45, 100, 299):
            v = f[: i + 1][np.isfinite(f[: i + 1])]
            n = len(v)
            mean = v.sum() / (n + 60.0)
            m2 = ((v ** 2).sum() + 60.0) / (n + 60.0)
            self.assertAlmostEqual(z[i], np.clip((f[i] - mean) / np.sqrt(m2 - mean ** 2), -4, 4), places=12)
        self.assertTrue(np.all(z[:40] == 0.0))
        np.testing.assert_array_equal(F.expanding_zscore(f[:150], 60.0, 4.0), z[:150])

    def test_shrunk_drift(self):
        rng = np.random.default_rng(3)
        rc = 0.015 * rng.standard_normal(2000) + 0.001
        rc[0] = 0.0
        d = F.shrunk_drift(rc, mu0=0.0002, tau0=0.0006)
        i = 1500
        v = np.mean(rc[1: i + 1] ** 2)
        v = (F.VOL_PRIOR_WEIGHT * F.VOL_PRIOR + i * v) / (F.VOL_PRIOR_WEIGHT + i)
        n0 = v / 0.0006 ** 2
        self.assertAlmostEqual(d[i], (n0 * 0.0002 + rc[1: i + 1].sum()) / (n0 + i), places=12)
        self.assertLess(abs(d[30] - 0.0002), abs(d[1500] - 0.0002) + 1e-4)  # short history stays near the prior
        sample_mean = rc[1:2000].mean()
        self.assertTrue(min(0.0002, sample_mean) < d[1999] < max(0.0002, sample_mean))  # strictly between prior and data

    def test_parkinson_handles_degenerate_bars(self):
        high = np.array([10.0, 10.0, 10.5, 10.0, 11.0])
        low = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # H == L on rows 0, 1, 3 (placeholder bars)
        lv, pv = F.parkinson_logvol(high, low)
        sp = np.log(1.05) / (2 * np.sqrt(np.log(2)))
        self.assertAlmostEqual(lv[2], np.log(sp), places=12)
        self.assertAlmostEqual(lv[3], lv[2], places=12)  # forward-filled
        self.assertAlmostEqual(lv[1], np.log(0.01), places=12)  # leading bars: fill value
        self.assertTrue(np.all(np.isfinite(lv)) and np.all(pv > 0))


class TestCalendarFlag(unittest.TestCase):
    """Audit M-04: the weekly calendar is the part of calendar time that is known at the origin."""

    def test_fridays_are_flagged_and_nothing_else(self):
        d = np.arange(np.datetime64("2005-01-03"), np.datetime64("2005-01-24"))  # Mon 3 Jan .. Sun 23 Jan (every calendar day)
        got = F.next_bar_after_weekend(d)
        names = [x.astype(object).strftime("%a") for x in d]
        self.assertEqual([n for n, g in zip(names, got) if g], ["Fri", "Fri", "Fri"])
        self.assertEqual(int(got.sum()), 3)
        self.assertFalse(F.next_bar_after_weekend(np.array([np.datetime64("2005-01-01")]))[0])  # a Saturday
        self.assertTrue(F.next_bar_after_weekend(np.array([np.datetime64("1970-01-02")]))[0])  # day 1 of the epoch is a Friday
        self.assertTrue(F.next_bar_after_weekend(np.array([np.datetime64("2026-10-23")]))[0])  # the Friday of the real data

    def test_business_day_series_has_one_flag_per_week_and_is_a_prefix_function(self):
        stock = make_pricedata(garch_returns(299, seed=2), "STK")
        got = F.next_bar_after_weekend(stock.dates)
        self.assertEqual(int(got.sum()), 60)  # 300 Monday-Friday bars: 60 Fridays
        self.assertTrue(np.all(got[4::5]) and not np.any(got[0::5] | got[1::5] | got[2::5] | got[3::5]))
        for cut in (1, 4, 5, 77, 299):
            np.testing.assert_array_equal(F.next_bar_after_weekend(stock.slice(cut).dates), got[:cut])


class TestFeatureSet(unittest.TestCase):
    def test_shapes_bounds_and_finiteness(self):
        stock, mkt = make_pair(n=1200, seed=8)
        fs = F.build_features(stock, mkt)
        self.assertEqual(fs.X.shape, (stock.n, len(F.FEATURE_KEYS)))
        self.assertTrue(np.all(np.isfinite(fs.X)))
        self.assertLessEqual(float(np.abs(fs.X).max()), F.Z_CLIP)
        rsi = fs.raw[:, fs.keys.index("rsi")]
        self.assertTrue(np.nanmin(rsi) >= 0 and np.nanmax(rsi) <= 100)
        self.assertTrue(np.all(fs.sigma_ref > 0) and np.all(fs.v97 > 0))
        self.assertEqual(set(F.FEATURE_NAMES), set(F.FEATURE_KEYS))
        self.assertEqual(set(F.FEATURE_UNITS), set(F.FEATURE_KEYS))
        # late rows: every feature is populated (not all zeros)
        self.assertTrue(np.all(np.abs(fs.X[-300:]).max(axis=0) > 0))

    def test_momentum_definitions(self):
        stock, _ = make_pair(n=600, seed=9)
        fs = F.build_features(stock, None)
        i = 500
        r20 = fs.cs[i] - fs.cs[i - 20]
        self.assertAlmostEqual(fs.raw[i, fs.keys.index("mom_20")], np.expm1(r20), places=12)
        r121 = fs.cs[i - 21] - fs.cs[i - 252]
        self.assertAlmostEqual(fs.raw[i, fs.keys.index("mom_12_1")], np.expm1(r121), places=12)
        self.assertAlmostEqual(fs.raw[i, fs.keys.index("rev_1")], np.expm1(fs.rc[i]), places=12)
        self.assertTrue(np.isnan(fs.raw[100, fs.keys.index("mom_12_1")]))  # needs 252 rows

    def test_zero_volume_series_is_handled(self):
        stock, _ = make_pair(n=500, seed=10)
        idx = make_pricedata(garch_returns(499, seed=3), "IDX", kind="index")
        idx.volume[:] = 0.0
        fs = F.build_features(idx, None)
        self.assertTrue(np.all(fs.X[:, fs.keys.index("vol_z")] == 0.0))
        self.assertTrue(np.all(np.isfinite(fs.X)))


if __name__ == "__main__":
    unittest.main()
