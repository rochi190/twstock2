"""Engine: contract/invariants, NO LOOK-AHEAD (truncation invariance), forecast_latest == walk-forward last row,
honesty on i.i.d. noise, planted signal, data hygiene and (skippable) real-data smoke tests with the speed targets."""
import json
import os
import sys
import time
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_helpers import ar1_returns, garch_returns, iid_returns, load_real_pricedata, make_pricedata  # noqa: E402

from backend import config  # noqa: E402
from backend.quant import engine as EN  # noqa: E402

QL = np.array(config.QLEVELS)
HS = (1, 5, 20)
FORECAST_FIELDS = ("p_up", "p_up_raw", "mu", "sigma", "q")


def make_stock_and_market(n=1500, seed=21):
    stock = make_pricedata(garch_returns(n, seed=seed), "GAR", seed=seed)
    mkt = make_pricedata(garch_returns(n, seed=seed + 1, v=0.011 ** 2, drift=0.0002), "MKT", seed=seed + 2, kind="index")
    return stock, mkt


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, o


def pinball(q, R):
    d = R[:, None] - q
    return np.mean(np.maximum(QL * d, (QL - 1) * d), axis=1)


def first_origin(ctx, start, refit_every=21):
    """First origin the engine itself pre-rolls for reported rows from ``start`` (``walk_forward``: ``max(g0, start - L_state)``).

    A refit block before the first absolute grid row ``g0`` would fit on fewer than ``min_fit_bars`` rows (an empty window for
    ``start - L_state = 0``: "Mean of empty slice" warnings and a meaningless fallback fit), so tests that drive the private
    ``_compute_components`` / ``_combine`` pair must start where the engine does."""
    return max(EN._grid_first(refit_every, int(ctx.cfg["min_fit_bars"])), start - EN._state_window(ctx.cfg))


class Shared(unittest.TestCase):
    """One walk-forward and one live forecast on synthetic GJR-GARCH data with a market series (computed once)."""

    @classmethod
    def setUpClass(cls):
        cls.START = 700
        cls.data, cls.mkt = make_stock_and_market()
        cls.wf = EN.walk_forward(cls.data, cls.mkt, start_idx=cls.START)
        cls.fc = EN.forecast_latest(cls.data, cls.mkt)


class TestWalkForwardContract(Shared):
    def test_container(self):
        wf, n = self.wf, self.data.n
        self.assertEqual(wf.symbol, "GAR")
        self.assertEqual(len(wf.dates), n)
        self.assertEqual(sorted(wf.horizons), list(HS))
        self.assertEqual((wf.first_idx, wf.refit_every, wf.min_train), (self.START, 21, config.MIN_TRAIN))
        self.assertEqual(wf.model_names, ["garch", "hmm", "dlm", "naive"])
        self.assertEqual(wf.regime_prob.shape, (n, 3))
        self.assertEqual(len(wf.regime_labels), 3)
        self.assertEqual(tuple(wf.qlevels), tuple(config.QLEVELS))
        self.assertEqual(wf.vol1d.shape, (n,))
        json.dumps(wf.config)  # serialisable
        self.assertTrue(np.all(np.isnan(wf.regime_prob[: self.START])))
        np.testing.assert_allclose(wf.regime_prob[self.START:].sum(axis=1), 1.0)
        self.assertTrue(np.all(wf.vol1d[self.START:] > 0) and np.all(np.isnan(wf.vol1d[: self.START])))

    def test_arrays_nan_structure_and_invariants(self):
        wf, n = self.wf, self.data.n
        after = np.arange(n) >= wf.first_idx
        lp = np.log(self.data.adjclose)
        for h, hf in wf.horizons.items():
            for name in ("p_up", "p_up_raw", "mu", "sigma", "pit", "realized"):
                self.assertEqual(getattr(hf, name).shape, (n,), name)
            self.assertEqual(hf.q.shape, (n, len(QL)))
            for name in ("p_up", "p_up_raw", "mu", "sigma"):
                a = getattr(hf, name)
                self.assertTrue(np.all(np.isfinite(a[after])), f"h={h} {name} has NaN after first_idx")
                self.assertTrue(np.all(np.isnan(a[~after])))
            self.assertTrue(np.all(np.isfinite(hf.q[after])) and np.all(np.isnan(hf.q[~after])))
            realized = np.full(n, np.nan)
            realized[:-h] = lp[h:] - lp[:-h]
            np.testing.assert_allclose(hf.realized, realized, equal_nan=True, rtol=0, atol=1e-12)  # independent recomputation
            np.testing.assert_array_equal(np.isfinite(hf.pit), after & np.isfinite(realized))
            self.assertTrue(np.all((hf.pit[np.isfinite(hf.pit)] >= 0) & (hf.pit[np.isfinite(hf.pit)] <= 1)))
            # hard invariants
            self.assertTrue(np.all((hf.p_up[after] > 0) & (hf.p_up[after] < 1)))
            self.assertTrue(np.all((hf.p_up_raw[after] > 0) & (hf.p_up_raw[after] < 1)))
            self.assertTrue(np.all(hf.sigma[after] > 0))
            self.assertTrue(np.all(np.diff(hf.q[after], axis=1) > 0), "quantiles must be strictly monotone in level")
            if h == 1:  # the daily limit
                self.assertGreaterEqual(float(hf.q[after].min()), np.log(0.9) - 1e-12)
                self.assertLessEqual(float(hf.q[after].max()), np.log(1.1) + 1e-12)
            for name in wf.model_names:
                for d in (hf.comp_p_up, hf.comp_mu, hf.weights):
                    self.assertIn(name, d)
                    self.assertEqual(d[name].shape, (n,))
                self.assertTrue(np.all((hf.comp_p_up[name][after] > 0) & (hf.comp_p_up[name][after] < 1)))
                self.assertTrue(np.all(hf.weights[name][after] >= 0))
            wsum = sum(hf.weights[nm][after] for nm in wf.model_names)
            np.testing.assert_allclose(wsum, 1.0, atol=1e-12)

    def test_p_up_raw_is_one_minus_cdf_at_zero_of_the_same_predictive(self):
        wf = self.wf
        rows = np.arange(wf.first_idx, self.data.n)
        for h, hf in wf.horizons.items():
            q = hf.q[rows]
            cdf0 = 1.0 - hf.p_up_raw[rows]
            below = np.where(q < 0, QL[None, :], 0.0).max(axis=1)  # largest level whose quantile is below zero
            above = np.where(q > 0, QL[None, :], 1.0).min(axis=1)
            self.assertTrue(np.all(cdf0 >= below - 1e-9) and np.all(cdf0 <= above + 1e-9), f"h={h}")

    def test_pit_is_the_randomised_predictive_cdf_at_the_realised_value(self):
        """Audit M-05: PIT = F(lo) + U (F(hi) - F(lo)) with [lo, hi] the tick cell of the realised terminal close.

        (The old test pinned PIT == F(R) up to 2 mismatches at the 5 % / 95 % quantiles; that point estimate has a point
        mass at F(0) for rounded prices and is intentionally gone.)  The cell is reconstructed independently from the
        predictive pool of the reported rows."""
        ctx = EN._Ctx(self.data, self.mkt, HS, EN._merge_cfg(None))
        comp = EN._compute_components(ctx, first_origin(ctx, self.START), self.data.n - 1, 21)
        res = EN._combine(ctx, comp, out_lo=self.START)
        rows = np.arange(self.START, self.data.n)
        for a, h in enumerate(HS):
            hf = self.wf.horizons[h]
            R = hf.realized[rows]
            ok = np.isfinite(R)
            term = np.full(len(rows), np.nan)
            term[ok] = self.data.close[rows[ok] + h]
            half = 0.5 * EN.tick_size(term, "stock") / term
            x = np.where(ok, R, 0.0)
            pool = res[a]["pool"]
            f_lo, f_pt, f_hi = (pool.cdf(x + np.log1p(s * np.where(ok, half, 0.0))) for s in (-1.0, 0.0, 1.0))
            pit = hf.pit[rows]
            self.assertTrue(np.all(pit[ok] >= f_lo[ok] - 1e-12) and np.all(pit[ok] <= f_hi[ok] + 1e-12), f"h={h}")
            self.assertTrue(np.all(f_lo[ok] <= f_pt[ok] + 1e-12) and np.all(f_pt[ok] <= f_hi[ok] + 1e-12))
            self.assertLess(float(np.max((f_hi - f_lo)[ok])), 0.2, f"h={h}: the tick cell is a small part of the predictive")
            self.assertGreater(float(np.mean(np.abs(pit[ok] - f_pt[ok]) > 1e-6)), 0.9, f"h={h}: PIT must be randomised")
            for k in (1, 7):  # the 5 % and 95 % quantiles: rows far from the cell are classified like the plain CDF
                far = ok & ((f_pt - QL[k]) ** 2 > (f_hi - f_lo) ** 2)
                inside = hf.realized[rows][far] < hf.q[rows][far, k]
                self.assertEqual(int(np.sum(inside != (pit[far] < QL[k]))), 0, f"h={h} level {QL[k]}")

    def test_calibrated_probability_stays_close_to_raw(self):
        for hf in self.wf.horizons.values():
            a = np.arange(self.data.n) >= self.wf.first_idx
            self.assertLess(float(np.abs(hf.p_up[a] - hf.p_up_raw[a]).max()), 0.12)


class TestLiveForecastSchema(Shared):
    def test_server_fields_are_not_produced_here(self):
        fc = self.fc
        for key in ("name", "market", "kind", "short_history", "next_trading_day", "asof_complete", "disclaimer", "notes"):
            self.assertNotIn(key, fc)
        for h in fc["horizons"].values():
            for key in ("signal", "score", "net_exp_return", "confidence"):
                self.assertNotIn(key, h)

    def test_top_level_and_horizon_keys(self):
        fc = self.fc
        for key in ("status", "symbol", "asof", "last_close", "chg_1d", "n_bars", "horizons", "fan", "regime", "vol",
                    "risk", "models", "drivers"):
            self.assertIn(key, fc)
        self.assertEqual((fc["status"], fc["symbol"], fc["n_bars"]), ("ready", "GAR", self.data.n))
        self.assertEqual(fc["asof"], str(self.data.dates[-1]))
        self.assertAlmostEqual(fc["last_close"], float(self.data.close[-1]))
        self.assertAlmostEqual(fc["chg_1d"], float(self.data.adjclose[-1] / self.data.adjclose[-2] - 1))
        self.assertEqual(set(fc["horizons"]), {"1", "5", "20"})
        for hs, h in fc["horizons"].items():
            for key in ("h", "days_label", "p_up", "p_up_raw", "exp_return", "median_return", "sigma", "quantiles",
                        "price_quantiles", "components"):
                self.assertIn(key, h)
            self.assertEqual(h["h"], int(hs))
            self.assertEqual(set(h["quantiles"]), {"0.05", "0.25", "0.5", "0.75", "0.95"})
            self.assertEqual(set(h["price_quantiles"]), set(h["quantiles"]))
            self.assertEqual(set(h["components"]), {"garch", "hmm", "dlm", "naive"})
            for comp in h["components"].values():
                self.assertEqual(set(comp), {"p_up", "mu"})
        self.assertEqual(fc["horizons"]["1"]["days_label"], "明日")
        self.assertEqual(fc["horizons"]["5"]["days_label"], "未來5日")
        self.assertEqual(fc["horizons"]["20"]["days_label"], "未來20日")

    def test_everything_is_finite_and_json_serialisable(self):
        bad = [p for p, v in walk(self.fc) if isinstance(v, float) and not np.isfinite(v)]
        self.assertEqual(bad, [])
        json.dumps(self.fc, ensure_ascii=False)  # only native python types

    def test_horizon_numbers(self):
        last = self.fc["last_close"]
        for h in self.fc["horizons"].values():
            q = np.array([h["quantiles"][k] for k in ("0.05", "0.25", "0.5", "0.75", "0.95")])
            self.assertTrue(np.all(np.diff(q) > 0))
            self.assertTrue(0 < h["p_up"] < 1 and 0 < h["p_up_raw"] < 1 and h["sigma"] > 0)
            self.assertEqual(h["median_return"], h["quantiles"]["0.5"])
            for k, v in h["quantiles"].items():
                self.assertAlmostEqual(h["price_quantiles"][k], last * (1 + v), places=8)
            self.assertTrue(q[0] < h["exp_return"] < q[-1])
            self.assertTrue(-1 < q[0])
        h1 = self.fc["horizons"]["1"]
        self.assertGreater(h1["quantiles"]["0.05"], -0.10)
        self.assertLess(h1["quantiles"]["0.95"], 0.10)

    def test_fan_regime_vol_risk_models_drivers(self):
        fc = self.fc
        fan = fc["fan"]
        self.assertEqual(fan["days"], list(range(1, 21)))
        self.assertEqual(fan["levels"], list(config.FAN_LEVELS))
        pr, rt = np.array(fan["prices"]), np.array(fan["returns"])
        self.assertEqual(pr.shape, (5, 20))
        np.testing.assert_allclose(pr, fc["last_close"] * (1 + rt), rtol=1e-12)
        self.assertTrue(np.all(np.diff(pr, axis=0) >= 0))
        self.assertTrue(np.all(np.diff(pr[-1] - pr[0]) > 0))  # widening
        for h in (1, 5, 20):  # the fan is exactly the horizon-card quantiles on the horizon days
            for li, lv in enumerate(("0.05", "0.25", "0.5", "0.75", "0.95")):
                self.assertAlmostEqual(pr[li, h - 1] / fc["horizons"][str(h)]["price_quantiles"][lv], 1.0, places=9)
        rg = fc["regime"]
        self.assertEqual(len(rg["labels"]), 3)
        self.assertAlmostEqual(sum(rg["probs"]), 1.0, places=9)
        self.assertIn(rg["current"], rg["labels"])
        self.assertEqual(rg["current"], rg["labels"][int(np.argmax(rg["probs"]))])
        np.testing.assert_allclose(np.array(rg["transition"]).sum(axis=1), 1.0)
        v = fc["vol"]
        self.assertAlmostEqual(v["sigma_annual"], v["sigma_daily"] * np.sqrt(config.TRADING_DAYS_PER_YEAR), places=12)
        self.assertTrue(0.0 <= v["percentile"] <= 1.0 and v["sigma_daily"] > 0)
        self.assertIn("百分位", v["regime_note"])
        rk = fc["risk"]
        self.assertEqual(set(rk), {"var95_1d", "cvar95_1d", "var95_20d", "cvar95_20d", "median_max_drawdown_20d",
                                   "prob_loss_gt_10pct_20d"})
        self.assertLess(rk["cvar95_1d"], rk["var95_1d"] + 1e-12)
        self.assertLess(rk["cvar95_20d"], rk["var95_20d"] + 1e-12)
        self.assertLessEqual(rk["median_max_drawdown_20d"], 0.0)
        self.assertTrue(0.0 <= rk["prob_loss_gt_10pct_20d"] <= 1.0)
        # VaR is the 5 % quantile of the same (aligned) distribution as the horizon card
        self.assertAlmostEqual(rk["var95_1d"], fc["horizons"]["1"]["quantiles"]["0.05"], delta=2e-3)
        self.assertAlmostEqual(rk["var95_20d"], fc["horizons"]["20"]["quantiles"]["0.05"], delta=4e-3)
        m = fc["models"]
        self.assertEqual(m["names"], {"garch": "GJR-GARCH-t", "hmm": "隱馬可夫regime", "dlm": "動態線性模型(貝氏)",
                                      "naive": "隨機漫步基準"})
        for hs, w in m["weights"].items():
            self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
        keys = [d["key"] for d in fc["drivers"]]
        self.assertEqual(sorted(keys), sorted(EN.F.FEATURE_KEYS))
        for d in fc["drivers"]:
            for k in ("name", "key", "value", "z", "contribution"):
                self.assertIn(k, d)
        contrib = [abs(d["contribution"]) for d in fc["drivers"]]
        self.assertEqual(contrib, sorted(contrib, reverse=True))
        self.assertLessEqual(max(abs(d["z"]) for d in fc["drivers"]), 4.0)

    def test_monte_carlo_uses_enough_paths_and_regime_mixing(self):
        mc = self.fc["engine"]["mc"]
        self.assertGreaterEqual(mc["paths"], 4000)
        self.assertTrue(mc["regime_mixing"])
        self.assertGreaterEqual(mc["pool"], 20)
        self.assertEqual(self.fc["engine"]["has_market"], True)

    def test_the_event_of_p_up_is_declared(self):
        """Audit C-03 / M-01: p_up is P(R>0 | R != 0); the live forecast and the walk-forward say so (a scorer must skip flats)."""
        self.assertEqual(EN.P_UP_EVENT, "up_given_move")
        self.assertEqual(self.fc["engine"]["p_up_event"], EN.P_UP_EVENT)
        self.assertEqual(self.wf.config["p_up_event"], EN.P_UP_EVENT)
        self.assertEqual(self.fc["engine"]["version"], EN.ENGINE_VERSION)
        self.assertEqual(self.wf.config["version"], EN.ENGINE_VERSION)

    def test_regime_note_names_a_state_only_when_it_is_clear(self):
        """Audit M-06: the filtered regime probabilities move quickly (fitted persistence ~0.75-0.8; the arg-max regime changed on
        32 % of the days), so the live note must not present a 52 % arg-max as 'the' state."""
        from unittest import mock
        cur, pmax = self.fc["regime"]["current"], max(self.fc["regime"]["probs"])
        with mock.patch.object(EN, "REGIME_CONFIDENT", 1.01):
            unsure = EN.forecast_latest(self.data, self.mkt)["vol"]["regime_note"]
        with mock.patch.object(EN, "REGIME_CONFIDENT", 0.0):
            sure = EN.forecast_latest(self.data, self.mkt)["vol"]["regime_note"]
        self.assertIn("判斷不明確", unsure)
        self.assertNotIn("最可能處於", unsure)
        self.assertIn(f"「{cur}」", unsure)
        self.assertIn(f"{pmax * 100:.0f}%", unsure)
        self.assertIn("最可能處於", sure)
        self.assertNotIn("判斷不明確", sure)
        self.assertIn(f"（機率 {pmax * 100:.0f}%）", sure)
        self.assertEqual("判斷不明確" in self.fc["vol"]["regime_note"], pmax < EN.REGIME_CONFIDENT)  # the default picks by probability
        self.assertIn("百分位", unsure)  # the volatility part of the note is unchanged
        self.assertIs(self.fc["regime"]["confident"], bool(pmax >= EN.REGIME_CONFIDENT))  # additive flag for the UI

    def test_deterministic(self):
        again = EN.forecast_latest(self.data, self.mkt)
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.fc, sort_keys=True))


class TestNoLookAhead(Shared):
    def assert_last_row_equals(self, wf_small, i):
        for h in HS:
            a, b = self.wf.horizons[h], wf_small.horizons[h]
            for f in FORECAST_FIELDS:
                np.testing.assert_allclose(getattr(a, f)[i], getattr(b, f)[-1], rtol=1e-9, atol=1e-12, err_msg=f"h={h} {f}")
            for name in self.wf.model_names:
                for d in ("comp_p_up", "comp_mu", "weights"):
                    np.testing.assert_allclose(getattr(a, d)[name][i], getattr(b, d)[name][-1], rtol=1e-9, atol=1e-12,
                                               err_msg=f"h={h} {d}.{name}")
            if np.isfinite(a.pit[i]) and np.isfinite(b.pit[-1]):
                self.assertAlmostEqual(a.pit[i], b.pit[-1], places=9)
        np.testing.assert_allclose(self.wf.regime_prob[i], wf_small.regime_prob[-1], rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(self.wf.vol1d[i], wf_small.vol1d[-1], places=12)

    def test_truncation_invariance(self):
        """DESIGN.md: walk_forward(data.slice(i+1)) last row == walk_forward(data) row i, all horizons."""
        for i in (700, 701, 744, 905, 1111, 1499):  # first origin, next row, on / off the refit grid, last row
            small = EN.walk_forward(self.data.slice(i + 1), self.mkt, start_idx=self.START)
            self.assertEqual(small.horizons[1].p_up.shape[0], i + 1)
            self.assert_last_row_equals(small, i)

    def test_randomised_pit_is_a_function_of_its_own_row_only(self):
        """The PIT jitter is keyed by (symbol, origin date, horizon): identical whether or not later rows exist."""
        i = 1111
        small = EN.walk_forward(self.data.slice(i + 1), self.mkt, start_idx=self.START)
        for h in HS:
            j = np.arange(self.START, i + 1 - h)  # realised inside the truncated data
            self.assertTrue(np.all(np.isfinite(small.horizons[h].pit[j])))
            np.testing.assert_allclose(small.horizons[h].pit[j], self.wf.horizons[h].pit[j], rtol=0, atol=1e-12)

    def test_forecast_latest_equals_last_row_of_walk_forward(self):
        for i in (903, 904, 1260):  # 904 is a Friday: the one-day predictive is the weekend-widened one (audit M-04)
            live = EN.forecast_latest(self.data.slice(i + 1), self.mkt)
            for h in HS:
                j, hf = live["horizons"][str(h)], self.wf.horizons[h]
                self.assertAlmostEqual(j["p_up"], hf.p_up[i], places=12)
                self.assertAlmostEqual(j["p_up_raw"], hf.p_up_raw[i], places=12)
                self.assertAlmostEqual(j["mu_log"], hf.mu[i], places=12)
                self.assertAlmostEqual(j["sigma_log"], hf.sigma[i], places=12)
                for lv, col in (("0.05", 1), ("0.25", 3), ("0.5", 4), ("0.75", 5), ("0.95", 7)):
                    self.assertAlmostEqual(np.log1p(j["quantiles"][lv]), hf.q[i, col], places=10)
                for name in self.wf.model_names:
                    self.assertAlmostEqual(live["models"]["weights"][str(h)][name], hf.weights[name][i], places=12)
                    self.assertAlmostEqual(j["components"][name]["p_up"], hf.comp_p_up[name][i], places=12)
            self.assertAlmostEqual(live["vol"]["sigma_daily"], self.wf.vol1d[i], places=12)
            np.testing.assert_allclose(live["regime"]["probs"], self.wf.regime_prob[i], atol=1e-12)

    def test_garbage_after_row_i_cannot_change_the_past(self):
        i = 1000
        rng = np.random.default_rng(5)
        d2, m2 = make_stock_and_market()
        for arr in (d2.open, d2.high, d2.low, d2.close, d2.adjclose):
            arr[i + 1:] = arr[i] * np.exp(np.cumsum(rng.normal(0, 0.2, len(arr) - i - 1)))  # wild future prices
        d2.volume[i + 1:] = 0.0
        m2.adjclose[i + 1:] *= 3.0
        wf2 = EN.walk_forward(d2, m2, start_idx=self.START)
        for h in HS:
            for f in FORECAST_FIELDS:  # a leak would show up at >> 1e-12; identical code paths agree to rounding
                np.testing.assert_allclose(getattr(self.wf.horizons[h], f)[: i + 1], getattr(wf2.horizons[h], f)[: i + 1],
                                           rtol=1e-12, atol=1e-14, err_msg=f"h={h} {f}")
            for name in self.wf.model_names:
                np.testing.assert_allclose(self.wf.horizons[h].weights[name][: i + 1], wf2.horizons[h].weights[name][: i + 1],
                                           rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(self.wf.regime_prob[: i + 1], wf2.regime_prob[: i + 1], rtol=1e-12, atol=1e-14)

    def test_refit_schedule_is_absolute_start_idx_does_not_matter(self):
        other = EN.walk_forward(self.data, self.mkt, start_idx=950)
        for h in HS:
            for f in FORECAST_FIELDS:
                np.testing.assert_allclose(getattr(self.wf.horizons[h], f)[950:], getattr(other.horizons[h], f)[950:],
                                           rtol=1e-9, atol=1e-12, err_msg=f"h={h} {f}")

    def test_truncation_invariance_without_market_and_iid_data(self):
        data = make_pricedata(iid_returns(1000, seed=3), "IID")
        full = EN.walk_forward(data, None, start_idx=560)
        for i in (560, 650, 999):
            part = EN.walk_forward(data.slice(i + 1), None, start_idx=560)
            for h in HS:
                for f in FORECAST_FIELDS:
                    np.testing.assert_allclose(getattr(full.horizons[h], f)[i], getattr(part.horizons[h], f)[-1], rtol=1e-9,
                                               atol=1e-12)


class TestHonestyOnNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_pricedata(iid_returns(2300, sd=0.015, nu=5.0, drift=0.0002, seed=31), "IID")
        cls.wf = EN.walk_forward(cls.data, None, start_idx=800)

    def test_no_edge_calibrated_probabilities_coverage_and_pit(self):
        wf = self.wf
        for h in HS:
            hf = wf.horizons[h]
            idx = np.arange(wf.first_idx, self.data.n)
            ok = np.isfinite(hf.realized[idx]) & (hf.realized[idx] != 0)
            idx = idx[ok]
            R, p = hf.realized[idx], hf.p_up[idx]
            up = (R > 0).astype(float)
            base = up.mean()
            # calibration in the large against the TRUE probability of the i.i.d. process (drift = the prior mean 0.0002);
            # the tolerance is the sampling error of the estimated drift (cross-seed sd 0.8 / 1.3 / 2.3 pp at h = 1 / 5 / 20)
            true_p = float(stats.norm.cdf(h * 0.0002 / (0.015 * np.sqrt(h))))
            self.assertLess(abs(p.mean() - true_p), {1: 0.02, 5: 0.03, 20: 0.05}[h], f"h={h}: calibration in the large")
            self.assertLess(p.std(), 0.04, f"h={h}: p_up must hardly move on noise")
            acc = float(np.mean((p > 0.5) == (up == 1)))
            self.assertLess(abs(acc - max(base, 1 - base)), 0.04, f"h={h}: no systematic accuracy edge")
            k = int(round(acc * len(idx)))
            pval = stats.binomtest(k, len(idx), max(base, 1 - base), alternative="greater").pvalue
            self.assertGreater(pval, 0.01)
            brier = np.mean((p - up) ** 2)
            brier_clim = np.mean((base - up) ** 2)
            self.assertLess(abs(1 - brier / brier_clim), 0.01, f"h={h}: Brier skill vs climatology ~ 0")
            allr = np.arange(wf.first_idx, self.data.n)
            allr = allr[np.isfinite(hf.realized[allr])]
            Rall = hf.realized[allr]
            tol = 0.03 if h == 1 else 0.05  # overlapping outcomes at h > 1 -> larger sampling error
            for lo, hi, nominal in ((1, 7, 0.90), (3, 5, 0.50)):
                cov = float(np.mean((Rall > hf.q[allr, lo]) & (Rall < hf.q[allr, hi])))
                self.assertAlmostEqual(cov, nominal, delta=tol, msg=f"h={h} nominal {nominal}")
            sub = np.arange(0, len(allr), h)  # non-overlapping subsample for the uniformity test
            ks = stats.kstest(hf.pit[allr][sub], "uniform")
            self.assertGreater(ks.pvalue, 0.01, f"h={h}: PIT not uniform")
            self.assertAlmostEqual(float(np.mean(hf.pit[allr])), 0.5, delta=0.04)
        self.assertLess(abs(float(np.mean(wf.horizons[1].p_up[wf.first_idx:])) - 0.5), 0.03)  # spec: ~0.5 +- 0.03 at h=1

    def test_ensemble_weights_stay_diversified_on_noise(self):
        for h in HS:
            w = np.array([self.wf.horizons[h].weights[n][self.wf.first_idx:] for n in self.wf.model_names])
            self.assertTrue(np.all(w.mean(axis=1) > 0.08), f"h={h}: {w.mean(axis=1)}")


class TestCalibrationOnModelData(unittest.TestCase):
    """Data generated from the model class itself (GJR-GARCH-t with leverage): every horizon must be well calibrated."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_pricedata(garch_returns(3200, seed=51, alpha=0.04, gamma=0.08, beta=0.88, drift=0.0003), "GJR")
        cls.wf = EN.walk_forward(cls.data, None, start_idx=1000)

    def test_interval_coverage_pit_and_zscores(self):
        for h in HS:
            hf = self.wf.horizons[h]
            ia = np.arange(1000, self.data.n)
            ia = ia[np.isfinite(hf.realized[ia])]
            R, q = hf.realized[ia], hf.q[ia]
            tol = 0.035 if h == 1 else 0.05
            for lo, hi, nominal in ((3, 5, 0.50), (2, 6, 0.80), (1, 7, 0.90), (0, 8, 0.98)):
                cov = float(np.mean((R > q[:, lo]) & (R < q[:, hi])))
                self.assertAlmostEqual(cov, nominal, delta=tol if nominal < 0.95 else 0.02, msg=f"h={h} {nominal}")
            z = stats.norm.ppf(np.clip(hf.pit[ia], 1e-6, 1 - 1e-6))
            self.assertAlmostEqual(float(z.std()), 1.0, delta=0.08, msg=f"h={h}: predictive spread")
            self.assertLess(abs(float(z.mean())), 0.2, f"h={h}: predictive location")
            sub = np.arange(0, len(ia), h)
            self.assertGreater(stats.kstest(hf.pit[ia][sub], "uniform").pvalue, 0.01, f"h={h}")


class TestPlantedSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_pricedata(ar1_returns(2300, rho=0.30, sd=0.015, seed=5), "AR")
        cls.wf = EN.walk_forward(cls.data, None, start_idx=800)
        cls.off = EN.walk_forward(cls.data, None, start_idx=800, cfg={"dlm_tau": 1e-6, "dlm_tau_intercept": 1e-6})

    def test_planted_signal_gives_measurably_better_scores(self):
        a, b = self.wf.horizons[1], self.off.horizons[1]
        idx = np.arange(self.wf.first_idx, self.data.n)
        idx = idx[np.isfinite(a.realized[idx])]
        R = a.realized[idx]
        diff = pinball(b.q[idx], R) - pinball(a.q[idx], R)  # positive = the model with the DLM is better
        t = diff.mean() / (diff.std() / np.sqrt(len(diff)))
        self.assertGreater(diff.mean(), 0.0)
        self.assertGreater(t, 3.0, f"pinball gain t-stat {t:.2f}")
        up = (R > 0)
        acc = float(np.mean((a.p_up[idx] > 0.5) == up))
        self.assertGreater(acc, max(up.mean(), 1 - up.mean()) + 0.015)
        self.assertGreater(float(np.corrcoef(a.mu[idx], R)[0, 1]), 0.1)  # forecast mean is informative

    def test_planted_signal_gives_a_better_log_score(self):
        """Paired comparison of the pooled predictive log density with / without the regression component."""
        def pooled_log_scores(cfg):
            ctx = EN._Ctx(self.data, None, (1,), EN._merge_cfg(cfg))
            start = 800
            comp = EN._compute_components(ctx, first_origin(ctx, start), self.data.n - 1, 21)
            res = EN._combine(ctx, comp, out_lo=start, keep=True)
            org = np.arange(comp.o_lo, comp.o_hi + 1)
            R = ctx.Rraw[0][org][org >= start]
            return res[0]["pool"].logpdf(np.where(np.isfinite(R), R, 0.0))[np.isfinite(R)]
        full = pooled_log_scores({})
        off = pooled_log_scores({"dlm_tau": 1e-6, "dlm_tau_intercept": 1e-6})
        diff = full - off
        t = diff.mean() / (diff.std() / np.sqrt(len(diff)))
        self.assertGreater(diff.mean(), 0.005, "log-score gain (nats per forecast)")
        self.assertGreater(t, 4.0)

    def test_probabilities_are_not_overconfident_when_a_signal_exists(self):
        """Reliability at h=1 with a real predictable component: p_up moves the right way and never exaggerates."""
        a = self.wf.horizons[1]
        idx = np.arange(self.wf.first_idx, self.data.n)
        idx = idx[np.isfinite(a.realized[idx])]
        p, up = a.p_up[idx], (a.realized[idx] > 0).astype(float)
        order = np.argsort(p)
        bins = np.array_split(order, 5)
        freq = np.array([up[b].mean() for b in bins])
        pm = np.array([p[b].mean() for b in bins])
        self.assertGreater(freq[-1], freq[0] + 0.05)  # realised up-frequency rises with the forecast
        self.assertLess(float(np.abs(pm - 0.5).max()), float(np.abs(freq - 0.5).max()) + 0.02)  # no exaggeration
        x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        design = np.column_stack([np.ones_like(x), x])
        beta = np.zeros(2)
        for _ in range(25):  # logistic regression of the outcome on logit(p_up) by Newton's method
            pr = 1 / (1 + np.exp(-design @ beta))
            beta += np.linalg.solve(design.T @ (design * (pr * (1 - pr))[:, None]) + 1e-9 * np.eye(2), design.T @ (up - pr))
        self.assertGreater(beta[1], 0.8, "reliability slope: model must not be over-confident")

    def test_ensemble_weight_shifts_to_the_better_component(self):
        w = {n: float(np.mean(self.wf.horizons[1].weights[n][self.wf.first_idx:])) for n in self.wf.model_names}
        self.assertGreater(w["dlm"], 0.30)
        self.assertGreater(w["dlm"], w["naive"] + 0.10)
        self.assertGreater(w["dlm"], w["garch"])
        off = {n: float(np.mean(self.off.horizons[1].weights[n][self.off.first_idx:])) for n in self.off.model_names}
        self.assertGreater(w["dlm"], off["dlm"] + 0.03)

    def test_vol_clustering_favours_garch_over_the_ewma_baseline(self):
        # fast mean reversion of the variance (phi = 0.85): a non-reverting EWMA is clearly mis-specified
        data = make_pricedata(garch_returns(2300, seed=41, alpha=0.20, gamma=0.10, beta=0.60, drift=0.0), "VOLC")
        wf = EN.walk_forward(data, None, start_idx=800)
        w = {n: float(np.mean(wf.horizons[1].weights[n][wf.first_idx:])) for n in wf.model_names}
        self.assertGreater(w["garch"], w["naive"] + 0.05)


class TestHygieneAndErrors(unittest.TestCase):
    def test_insufficient_history_and_minimum_size(self):
        data = make_pricedata(garch_returns(400, seed=1), "S")
        with self.assertRaises(ValueError) as cm:
            EN.forecast_latest(data.slice(config.MIN_BARS_FORECAST - 1), None)
        self.assertEqual(str(cm.exception), "insufficient_history")
        fc = EN.forecast_latest(data.slice(config.MIN_BARS_FORECAST), None)  # exactly the minimum works
        self.assertEqual(fc["n_bars"], config.MIN_BARS_FORECAST)
        self.assertTrue(all(np.isfinite(v) for p, v in walk(fc) if isinstance(v, float)))
        with self.assertRaises(ValueError):
            EN.walk_forward(data, None, start_idx=401)

    def test_bad_inputs(self):
        data = make_pricedata(garch_returns(400, seed=2), "S")
        with self.assertRaises(ValueError):
            EN.walk_forward(data, None, cfg={"nonsense": 1})
        bad = data.slice(400)
        bad.adjclose[100] = np.nan
        with self.assertRaises(ValueError):
            EN.forecast_latest(bad, None)
        bad2 = data.slice(400)
        bad2.adjclose[50] = -3.0
        with self.assertRaises(ValueError):
            EN.walk_forward(bad2, None, start_idx=300)
        with self.assertRaises(ValueError):
            EN.walk_forward(data, None, horizons=(0, 5))
        with self.assertRaises(ValueError):
            EN.walk_forward(data, None, refit_every=0)

    def test_progress_callback(self):
        data = make_pricedata(garch_returns(600, seed=3), "S")
        calls = []
        EN.walk_forward(data, None, start_idx=400, progress=lambda d, t: calls.append((d, t)))
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(calls[-1][0], calls[-1][1])
        self.assertTrue(all(b[0] >= a[0] for a, b in zip(calls, calls[1:])))
        self.assertEqual(len({t for _, t in calls}), 1)

    def test_other_horizons_and_refit_every(self):
        data = make_pricedata(garch_returns(700, seed=4), "S")
        wf = EN.walk_forward(data, None, horizons=(2, 10), refit_every=10, start_idx=450)
        self.assertEqual(sorted(wf.horizons), [2, 10])
        self.assertEqual(wf.refit_every, 10)
        self.assertTrue(np.all(np.isfinite(wf.horizons[10].p_up[450:])))
        fc = EN.forecast_latest(data, None, horizons=(2, 10))
        self.assertEqual(set(fc["horizons"]), {"2", "10"})
        self.assertEqual(fc["horizons"]["2"]["days_label"], "未來2日")
        small = EN.walk_forward(data.slice(600), None, horizons=(2, 10), refit_every=10, start_idx=450)
        np.testing.assert_allclose(small.horizons[10].p_up[-1], wf.horizons[10].p_up[599], rtol=1e-9)

    def test_start_idx_before_first_possible_refit_is_clamped(self):
        data = make_pricedata(garch_returns(300, seed=5), "S")
        wf = EN.walk_forward(data, None, min_train=50)
        self.assertGreaterEqual(wf.first_idx, 119)
        self.assertTrue(np.all(np.isfinite(wf.horizons[1].p_up[wf.first_idx:])))

    def test_unadjusted_split_glitch_and_ghost_bars_do_not_break_anything(self):
        ret = garch_returns(1300, seed=7)
        clean = make_pricedata(ret, "G")
        glitch = make_pricedata(ret, "G")
        glitch.adjclose[501:] *= np.exp(-1.389)  # 0050-style unadjusted 4:1 split in the adjusted close
        glitch.close[501:] *= np.exp(-1.389)
        glitch.volume[[300, 301, 302, 900]] = 0.0  # placeholder bars
        glitch.high[[300, 301]] = glitch.low[[300, 301]]
        wf_g = EN.walk_forward(glitch, None, start_idx=700)
        wf_c = EN.walk_forward(clean, None, start_idx=700)
        for h in HS:
            self.assertTrue(np.all(np.isfinite(wf_g.horizons[h].p_up[700:])))
            self.assertTrue(np.all(np.isfinite(wf_g.horizons[h].q[700:])))
            self.assertLess(float(np.abs(wf_g.horizons[h].p_up[700:] - wf_c.horizons[h].p_up[700:]).max()), 0.03)
        self.assertLess(float(np.abs(wf_g.vol1d[700:] / wf_c.vol1d[700:] - 1).max()), 0.25)
        fc = EN.forecast_latest(glitch, None)
        self.assertTrue(all(np.isfinite(v) for p, v in walk(fc) if isinstance(v, float)))

    def test_two_regimes_configuration(self):
        data = make_pricedata(garch_returns(600, seed=9), "K2")
        wf = EN.walk_forward(data, None, start_idx=450, cfg={"hmm_states": 2})
        self.assertEqual(wf.regime_prob.shape, (data.n, 2))
        self.assertEqual(len(wf.regime_labels), 2)
        np.testing.assert_allclose(wf.regime_prob[450:].sum(axis=1), 1.0)
        fc = EN.forecast_latest(data, None, cfg={"hmm_states": 2})
        self.assertEqual(len(fc["regime"]["labels"]), 2)
        self.assertEqual(np.array(fc["regime"]["transition"]).shape, (2, 2))

    def test_awkward_series_stay_finite_and_valid(self):
        rng = np.random.default_rng(1)
        cases = {
            "low-vol bond ETF": make_pricedata(iid_returns(500, sd=0.0025, nu=8, drift=0.0001, seed=1), "B", kind="etf"),
            "very high vol": make_pricedata(np.clip(iid_returns(500, sd=0.035, nu=4.5, seed=2), -0.1, 0.095), "W"),
        }
        r = garch_returns(500, seed=3, v=0.02 ** 2)
        r[250:260], r[300:306] = 0.095, -0.1  # limit-up and limit-down streaks
        cases["limit streaks"] = make_pricedata(np.clip(r, -0.105, 0.095), "L")
        flat = make_pricedata(garch_returns(500, seed=5), "F")
        flat.adjclose[200:260] = flat.adjclose[200]
        flat.close[200:260] = flat.close[200]
        cases["suspended (flat) days"] = flat
        zr = make_pricedata(garch_returns(500, seed=4), "Z", p0=3.0)
        zr.high[::7] = zr.low[::7]
        cases["zero-range bars"] = zr
        import warnings
        for name, d in cases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no numerical warnings at all
                fc = EN.forecast_latest(d, None)
            self.assertEqual([p for p, v in walk(fc) if isinstance(v, float) and not np.isfinite(v)], [], name)
            for h in fc["horizons"].values():
                q = np.array(list(h["quantiles"].values()))
                self.assertTrue(np.all(np.diff(q) > 0) and 0 < h["p_up"] < 1, name)
        # a series with (almost) no noise at all
        trend = make_pricedata(np.full(400, 0.0005) + 1e-5 * rng.standard_normal(400), "T")
        fc = EN.forecast_latest(trend, None)
        self.assertTrue(all(0 < h["p_up"] < 1 for h in fc["horizons"].values()))

    def test_zero_volume_index_like_series(self):
        data = make_pricedata(garch_returns(500, seed=8), "IDX", kind="index")
        data.volume[:] = 0.0
        fc = EN.forecast_latest(data, None)
        self.assertTrue(all(np.isfinite(v) for p, v in walk(fc) if isinstance(v, float)))

    def test_missing_market_dates_and_short_market(self):
        stock, mkt = make_stock_and_market(n=700)
        short = mkt.slice(400)  # the market ends before the stock: last value carried forward
        a = EN.walk_forward(stock, short, start_idx=450)
        self.assertTrue(np.all(np.isfinite(a.horizons[5].p_up[450:])))


# ----------------------------------------------------------------------------- audit fixes: ties, tick cells, placeholders
def tick_rounded_world(seed, n=3300, sd=0.008, p0=158.0):
    """Latent GJR-GARCH-t path (the model's home world) whose closes are rounded to the Taiwan tick grid (15-19 % ties)."""
    r = garch_returns(n - 1, seed=seed, drift=0.0, v=sd ** 2)
    lat = p0 * np.exp(np.r_[0.0, np.cumsum(r)])
    tick = np.where(lat < 10, 0.01, np.where(lat < 50, 0.05, np.where(lat < 100, 0.1, np.where(lat < 500, 0.5,
                                                                                               np.where(lat < 1000, 1.0, 5.0)))))
    close = np.round(lat / tick) * tick
    d = make_pricedata(r, seed=seed, p0=p0)
    d.close, d.adjclose = close.copy(), close.copy()
    d.open = np.r_[close[0], close[:-1]]
    d.high, d.low = np.maximum(d.open, close) * 1.003, np.minimum(d.open, close) * 0.997
    return d


def causal_up_rate(R, h):
    """Laplace-smoothed up-frequency of the non-zero outcomes matured at each origin (the app's causal base rate)."""
    valid = np.isfinite(R) & (R != 0.0)
    cu, cn = np.cumsum(valid & (R > 0)), np.cumsum(valid)
    out = np.full(len(R), 0.5)
    idx = np.arange(len(R)) - h
    ok = idx >= 0
    out[ok] = (cu[idx[ok]] + 1.0) / (cn[idx[ok]] + 2.0)
    return out


class TestTiesAndTickCells(unittest.TestCase):
    """Audit C-03 / M-01 (unchanged closes are not a down move) and M-05 (randomised PIT for rounded prices)."""

    @classmethod
    def setUpClass(cls):
        cls.worlds = {}
        for seed in (0, 1):
            d = tick_rounded_world(seed)
            cls.worlds[seed] = (d, EN.walk_forward(d, None, start_idx=d.n - 1250))

    def test_ties_carry_no_calibration_target(self):
        d = make_pricedata(garch_returns(400, seed=12), "T")
        d.adjclose[201] = d.adjclose[200]  # one-day tie, origin 200
        d.close[201] = d.close[200]
        d.adjclose[155] = d.adjclose[150]  # five-day tie, origin 150
        d.close[155] = d.close[150]
        ctx = EN._Ctx(d, None, HS, EN._merge_cfg(None))
        lp = np.log(d.adjclose)
        for a, h in enumerate(HS):
            R = np.full(d.n, np.nan)
            R[:-h] = lp[h:] - lp[:-h]
            up = ctx.up[a]
            self.assertTrue(np.all(np.isnan(up[R == 0.0])), f"h={h}: ties must have no target")
            self.assertTrue(np.all(np.isnan(up[np.isnan(R)])))
            mv = np.isfinite(R) & (R != 0.0)
            np.testing.assert_array_equal(up[mv], (R[mv] > 0).astype(float))
        self.assertTrue(np.isnan(ctx.up[0][200]) and np.isnan(ctx.up[1][150]))  # the planted ties are really in there

    def test_cleaned_away_artefacts_carry_no_target_either(self):
        """A data artefact (|r| > 0.25, e.g. an unadjusted split) is cleaned to a zero return: that zero is not a 'flat close',
        and counting it as 'not up' would bias the calibrator downwards.  Longer windows that merely CONTAIN the artefact keep a
        direction (their cleaned return is not zero)."""
        d = make_pricedata(garch_returns(400, seed=13), "T")
        d.adjclose[301:] *= 2.0  # the one-day log return of row 301 is ln 2 = 0.69
        ctx = EN._Ctx(d, None, HS, EN._merge_cfg(None))
        self.assertEqual(float(ctx.Rc[0][300]), 0.0)  # cleaned to zero ...
        self.assertTrue(abs(float(ctx.Rraw[0][300]) - np.log(2.0)) < 0.2)  # ... although the raw return is huge
        self.assertTrue(np.isnan(ctx.up[0][300]))
        self.assertTrue(np.all(np.isfinite(ctx.up[0][[299, 301]])))  # its neighbours have a target
        self.assertTrue(np.all(np.isfinite(ctx.up[1][296:301])), "5-day windows containing the artefact keep a direction")

    def test_p_up_matches_the_non_tie_up_frequency_and_the_causal_base(self):
        """Before the fix the calibrator learned P(R>0) with the flat mass counted as 'down': mean p_up was 3.3 pp below
        the frequency of up moves among moving closes and the Brier score was worse than the causal base rate."""
        bias, bss = [], []
        for d, wf in self.worlds.values():
            hf = wf.horizons[1]
            R = np.full(d.n, np.nan)
            R[:-1] = np.log(d.adjclose[1:] / d.adjclose[:-1])
            rows = np.arange(wf.first_idx, d.n - 1)
            tie = R[rows] == 0.0
            self.assertGreater(float(tie.mean()), 0.08, "the world must contain many ties")
            rows = rows[~tie]
            up = (R[rows] > 0).astype(float)
            bias.append(float(hf.p_up[rows].mean() - up.mean()))
            base = causal_up_rate(R, 1)[rows]
            bss.append(1.0 - np.mean((hf.p_up[rows] - up) ** 2) / np.mean((base - up) ** 2))
        self.assertLess(abs(float(np.mean(bias))), 0.015, f"mean bias per world {bias}")
        self.assertGreater(float(np.mean(bss)), -0.002, f"Brier skill vs the causal base per world {bss}")

    def test_pit_of_rounded_prices_is_uniform(self):
        """The plain F(R) has a point mass at F(0) when a close is unchanged: KS p-values were 9e-5 and 3e-8 here."""
        for seed, (d, wf) in self.worlds.items():
            hf = wf.horizons[1]
            pit = hf.pit[np.isfinite(hf.pit)]
            self.assertGreater(len(pit), 1200)
            self.assertGreater(stats.kstest(pit, "uniform").pvalue, 0.01, f"seed {seed}")
            central = float(np.mean((pit > 0.4) & (pit < 0.6)))
            self.assertAlmostEqual(central, 0.2, delta=0.04, msg=f"seed {seed}: no excess mass around the median")

    def test_tick_size_table(self):
        px = np.array([0.5, 9.99, 10.0, 49.95, 50.0, 99.9, 100.0, 499.5, 500.0, 999.0, 1000.0, 2600.0])
        np.testing.assert_allclose(EN.tick_size(px, "stock"), [0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1, 1, 5, 5])
        np.testing.assert_allclose(EN.tick_size(px, "etf"), np.where(px < 50, 0.01, 0.05))
        np.testing.assert_array_equal(EN.tick_size(px, "index"), 0.0)

    def test_row_uniform_is_deterministic_keyed_and_uniform(self):
        dates = np.arange(np.datetime64("2010-01-04"), np.datetime64("2010-01-04") + 6000)
        u = EN._row_uniform("2330", dates, 1)
        self.assertTrue(np.all((u > 0) & (u < 1)))
        self.assertGreater(stats.kstest(u, "uniform").pvalue, 0.01)
        np.testing.assert_array_equal(EN._row_uniform("2330", dates[:50], 1), u[:50])  # a pure function of the row
        self.assertFalse(np.allclose(EN._row_uniform("2317", dates[:50], 1), u[:50]))  # different symbol
        self.assertFalse(np.allclose(EN._row_uniform("2330", dates[:50], 5), u[:50]))  # different horizon
        self.assertLess(abs(float(np.corrcoef(u[:-1], u[1:])[0, 1])), 0.05)

    def test_randomised_pit_is_uniform_where_the_plain_pit_is_not(self):
        """Unit test of the cell construction: latent N(0, 1 %) return, terminal close rounded to a 0.5 tick at 158."""
        rng = np.random.default_rng(7)
        m, c0, sd = 6000, 158.0, 0.01
        lat = rng.normal(0.0, sd, m)
        c1 = np.round(c0 * np.exp(lat) / 0.5) * 0.5
        R = np.log(c1 / c0)
        self.assertGreater(float(np.mean(R == 0.0)), 0.1)
        pool = EN.E.Pool([EN.E.Normal(np.zeros(m), np.full(m, sd))], np.ones((m, 1)))
        dates = np.arange(np.datetime64("2000-01-03"), np.datetime64("2000-01-03") + m)
        pit = EN._randomised_pit(pool, None, R, c1, "stock", EN._row_uniform("X", dates, 1))
        plain = pool.cdf(R)
        self.assertGreater(stats.kstest(pit, "uniform").pvalue, 0.01)
        self.assertLess(stats.kstest(plain, "uniform").pvalue, 1e-6)
        self.assertTrue(np.all(np.isnan(EN._randomised_pit(pool, None, np.full(m, np.nan), c1, "stock", pit))))
        idx = EN._randomised_pit(pool, np.arange(m)[::-1], R[::-1], c1[::-1], "stock", EN._row_uniform("X", dates, 1)[::-1])
        np.testing.assert_allclose(idx[::-1], pit, atol=1e-12)  # ``sel`` picks the pool row of every output row
        np.testing.assert_allclose(EN._randomised_pit(pool, None, R, c1, "index", pit), plain, atol=1e-12)  # no tick grid
        # an ETF at the same price trades on the finer 0.05 grid (stocks: 0.5 at 158); a low-volatility ETF path has ~4 % ties
        sd_e = 0.003
        c1e = np.round(c0 * np.exp(rng.normal(0.0, sd_e, m)) / 0.05) * 0.05
        Re = np.log(c1e / c0)
        self.assertGreater(float(np.mean(Re == 0.0)), 0.03)
        pool_e = EN.E.Pool([EN.E.Normal(np.zeros(m), np.full(m, sd_e))], np.ones((m, 1)))
        u = EN._row_uniform("ETF", dates, 1)
        pit_e = EN._randomised_pit(pool_e, None, Re, c1e, "etf", u)
        self.assertGreater(stats.kstest(pit_e, "uniform").pvalue, 0.01)
        self.assertLess(stats.kstest(pool_e.cdf(Re), "uniform").pvalue, 1e-3)  # the plain PIT piles up at F(0)
        half = 0.5 * 0.05 / c1e
        self.assertTrue(np.all(pit_e >= pool_e.cdf(Re + np.log1p(-half)) - 1e-12))  # inside the cell of the ETF tick
        self.assertTrue(np.all(pit_e <= pool_e.cdf(Re + np.log1p(half)) + 1e-12))


def with_placeholders(data, rows):
    """Copy of ``data`` whose rows ``rows`` are no-trade placeholder bars (flat at the previous close, zero volume)."""
    d = data.slice(data.n)
    for r in sorted(rows):
        for arr in (d.open, d.high, d.low, d.close):
            arr[r] = d.close[r - 1]
        d.adjclose[r] = d.adjclose[r - 1]
        d.volume[r] = 0.0
    return d


def only_rows(data, keep):
    """The rows of ``data`` selected by the boolean mask ``keep`` (what the engine's models are supposed to see)."""
    return type(data)(data.symbol, data.name, data.market, data.kind, data.dates[keep], data.open[keep], data.high[keep],
                      data.low[keep], data.close[keep], data.adjclose[keep], data.volume[keep])


def append_placeholders(data, k):
    """``data`` followed by ``k`` flat zero-volume bars on the next business days (a halt in progress)."""
    dates = np.concatenate([data.dates, np.busday_offset(data.dates[-1], np.arange(1, k + 1), roll="forward")])
    ext = lambda a, v: np.concatenate([a, np.full(k, v)])
    c = data.close[-1]  # open == high == low == close == the last close
    return type(data)(data.symbol, data.name, data.market, data.kind, dates, ext(data.open, c), ext(data.high, c),
                      ext(data.low, c), ext(data.close, c), ext(data.adjclose, data.adjclose[-1]), ext(data.volume, 0.0))


class TestPlaceholderBars(unittest.TestCase):
    """Audit C-09: halts / Yahoo filler bars (zero volume, flat) are not observations."""
    START = 640
    PH = [300, 301, 650, 651, 652, 653, 654, 800, 900, 901]  # a 5-bar halt, single filler bars and two before the start

    @classmethod
    def setUpClass(cls):
        cls.base, cls.mkt = make_stock_and_market(n=1100, seed=33)
        cls.d = with_placeholders(cls.base, cls.PH)
        cls.keep = np.ones(cls.d.n, bool)
        cls.keep[cls.PH] = False
        cls.pos = np.flatnonzero(cls.keep)
        cls.ci = np.cumsum(cls.keep) - 1  # modelled row of the last real bar <= each original row
        cls.d_del = only_rows(cls.d, cls.keep)
        s_c = int(cls.keep[: cls.START].sum())
        cls.wf = EN.walk_forward(cls.d, cls.mkt, start_idx=cls.START)
        cls.wf_del = EN.walk_forward(cls.d_del, cls.mkt, start_idx=s_c)

    def test_mask_definition(self):
        d = make_pricedata(garch_returns(60, seed=2), "M")
        d = with_placeholders(d, [10, 11, 12])
        d.volume[20] = 0.0  # zero volume but a real range -> real bar
        d.high[30] = d.low[30] = d.close[30]  # H == L but traded (limit-locked) -> real bar
        d.volume[40] = 0.0
        d.high[40] = d.low[40] = d.open[40] = d.close[40]  # no trade but the price MOVED (a data glitch, not a flat bar)
        d.volume[50] = 0.0
        d.high[50] = d.low[50] = d.open[50] = d.close[50] = d.close[49]
        d.adjclose[50] = d.adjclose[49] * (1.0 + 3e-7)  # flat up to float noise of the dividend factor -> placeholder
        got = np.flatnonzero(EN.F.placeholder_mask(d))
        np.testing.assert_array_equal(got, [10, 11, 12, 50])
        idx = make_pricedata(garch_returns(60, seed=3), "I", kind="index")
        idx = with_placeholders(idx, [10, 11])
        self.assertFalse(EN.F.placeholder_mask(idx).any(), "index series (no volume) are exempt")
        self.assertFalse(EN.F.placeholder_mask(d.slice(1))[0])  # the first row is never a placeholder
        self.assertFalse(EN.F.placeholder_mask(d.slice(0)).any())
        for cut in (11, 13, 51):  # prefix function
            np.testing.assert_array_equal(EN.F.placeholder_mask(d.slice(cut)), EN.F.placeholder_mask(d)[:cut])

    def test_walk_forward_equals_the_walk_forward_on_the_real_bars_only(self):
        wf, wd, ci = self.wf, self.wf_del, self.ci
        self.assertEqual(wf.config["placeholder_bars_dropped"], len(self.PH))
        self.assertEqual(wf.first_idx, int(self.pos[wd.first_idx]))
        rows = np.arange(wf.first_idx, self.d.n)
        self.assertEqual(len(wf.dates), self.d.n)
        for h in HS:
            a, b = wf.horizons[h], wd.horizons[h]
            for f in FORECAST_FIELDS:
                np.testing.assert_allclose(getattr(a, f)[rows], getattr(b, f)[ci[rows]], rtol=1e-12, atol=1e-14,
                                           err_msg=f"h={h} {f}")
            for name in wf.model_names:
                for dct in ("comp_p_up", "comp_mu", "weights"):
                    np.testing.assert_allclose(getattr(a, dct)[name][rows], getattr(b, dct)[name][ci[rows]], rtol=1e-12,
                                               atol=1e-14)
            self.assertTrue(np.all(np.isnan(a.p_up[: wf.first_idx])))
        np.testing.assert_allclose(wf.regime_prob[rows], wd.regime_prob[ci[rows]], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(wf.vol1d[rows], wd.vol1d[ci[rows]], rtol=1e-12, atol=1e-14)

    def test_a_placeholder_row_repeats_the_forecast_of_the_last_real_bar(self):
        for h in HS:
            hf = self.wf.horizons[h]
            for r in (650, 654, 800, 901):
                for f in FORECAST_FIELDS:
                    np.testing.assert_array_equal(getattr(hf, f)[r], getattr(hf, f)[self.pos[self.ci[r]]], err_msg=f"{h} {f}")

    def test_realised_returns_are_the_raw_returns_of_the_original_rows(self):
        lp = np.log(self.d.adjclose)
        after = np.arange(self.wf.first_idx, self.d.n)
        for h in HS:
            hf = self.wf.horizons[h]
            exp = np.full(self.d.n, np.nan)
            exp[:-h] = lp[h:] - lp[:-h]
            np.testing.assert_allclose(hf.realized, exp, equal_nan=True, rtol=0, atol=1e-12)  # what an evaluation recomputes
            np.testing.assert_array_equal(np.isfinite(hf.pit[after]), np.isfinite(exp[after]))
        self.assertEqual(self.wf.horizons[1].realized[649], 0.0)  # the (flat) halt bar follows row 649

    def test_pit_of_untouched_windows_equals_the_one_without_placeholders(self):
        clean = np.ones(self.d.n, bool)
        for h in HS:
            win = np.ones(self.d.n, bool)  # rows whose window (i, i+h] holds no placeholder and which are real bars
            for r in self.PH:
                win[max(r - h, 0): r + 1] = False
            rows = np.flatnonzero(win & clean)
            rows = rows[(rows >= self.wf.first_idx) & (rows + h < self.d.n)]
            np.testing.assert_allclose(self.wf.horizons[h].pit[rows], self.wf_del.horizons[h].pit[self.ci[rows]], atol=1e-12)

    def test_truncation_invariance_with_placeholders(self):
        for i in (652, 655, 801, 1050):  # inside the halt, the first real bar after it, after a single filler, late
            small = EN.walk_forward(self.d.slice(i + 1), self.mkt, start_idx=self.START)
            for h in HS:
                for f in FORECAST_FIELDS:
                    np.testing.assert_allclose(getattr(self.wf.horizons[h], f)[i], getattr(small.horizons[h], f)[-1],
                                               rtol=1e-9, atol=1e-12, err_msg=f"i={i} h={h} {f}")
            np.testing.assert_allclose(self.wf.regime_prob[i], small.regime_prob[-1], rtol=1e-9, atol=1e-12)
            self.assertAlmostEqual(self.wf.vol1d[i], small.vol1d[-1], places=12)

    def test_start_idx_on_a_placeholder_starts_at_the_next_real_bar(self):
        wf = EN.walk_forward(self.d, self.mkt, start_idx=651)
        self.assertEqual(wf.first_idx, 655)
        np.testing.assert_allclose(wf.horizons[5].p_up[655:], self.wf.horizons[5].p_up[655:], rtol=1e-9)

    def test_live_forecast_ignores_a_halt_in_progress(self):
        """k flat zero-volume bars appended: the one-day sigma used to collapse (1.66 % -> 1.04 % after 20 bars on 2330)."""
        fc0 = EN.forecast_latest(self.d, self.mkt)
        for k in (5, 40):
            ext = append_placeholders(self.d, k)
            fc1 = EN.forecast_latest(ext, self.mkt)
            for key in ("horizons", "fan", "regime", "vol", "risk", "models", "drivers"):
                self.assertEqual(json.dumps(fc0[key], sort_keys=True), json.dumps(fc1[key], sort_keys=True), key)
            self.assertEqual(fc1["asof"], str(ext.dates[-1]))
            self.assertEqual(fc1["n_bars"], self.d.n + k)
            self.assertEqual(fc1["engine"]["placeholder_bars_dropped"], len(self.PH) + k)
            self.assertAlmostEqual(fc1["last_close"], float(self.d.close[-1]))
            self.assertEqual(fc1["chg_1d"], 0.0)  # the quoted 1-day change of the halted name is genuinely 0
        wf = EN.walk_forward(append_placeholders(self.d, 5), self.mkt, start_idx=self.START)
        for h in HS:
            np.testing.assert_array_equal(wf.horizons[h].p_up[-5:], np.full(5, self.wf.horizons[h].p_up[-1]))

    def test_live_forecast_equals_the_last_row_of_the_walk_forward(self):
        live = EN.forecast_latest(self.d, self.mkt)
        for h in HS:
            self.assertAlmostEqual(live["horizons"][str(h)]["p_up"], self.wf.horizons[h].p_up[-1], places=12)
            self.assertAlmostEqual(live["horizons"][str(h)]["sigma_log"], self.wf.horizons[h].sigma[-1], places=12)

    def test_index_series_and_volume_bars_are_never_dropped(self):
        idx = make_pricedata(garch_returns(500, seed=8), "IDX", kind="index")
        idx = with_placeholders(idx, [100, 101, 102])
        wf = EN.walk_forward(idx, None, start_idx=300)
        self.assertEqual(wf.config["placeholder_bars_dropped"], 0)
        halted_with_volume = make_pricedata(garch_returns(500, seed=9), "V")
        halted_with_volume.adjclose[200:230] = halted_with_volume.adjclose[200]  # flat but traded -> real zero returns
        self.assertFalse(EN.F.placeholder_mask(halted_with_volume).any())


class TestTiesAndHaltsTogether(unittest.TestCase):
    """A tick-rounded world (many unchanged closes) that also contains halts: the two audit fixes must not interfere."""
    START = 1400
    HALT = [1500, 1501, 1502, 1503, 1504, 1505, 1506, 1800, 2100, 2101]  # a 7-bar halt and single filler bars, all in the window

    @classmethod
    def setUpClass(cls):
        cls.d = with_placeholders(tick_rounded_world(2, n=2600), cls.HALT)
        cls.wf = EN.walk_forward(cls.d, None, start_idx=cls.START)

    def test_halt_rows_are_dropped_and_the_rest_is_aligned(self):
        wf, d = self.wf, self.d
        self.assertEqual(wf.config["placeholder_bars_dropped"], len(self.HALT))
        self.assertEqual(wf.first_idx, self.START)  # a real bar
        lp = np.log(d.adjclose)
        for h in HS:
            hf = wf.horizons[h]
            self.assertTrue(np.all(np.isfinite(hf.p_up[self.START:])) and np.all(np.isfinite(hf.q[self.START:])))
            exp = np.full(d.n, np.nan)
            exp[:-h] = lp[h:] - lp[:-h]
            np.testing.assert_allclose(hf.realized, exp, equal_nan=True, atol=1e-12)
        self.assertEqual(float(wf.horizons[1].realized[1505]), 0.0)  # inside the halt the next row is flat as well
        self.assertTrue(np.all(wf.horizons[1].p_up[1500:1507] == wf.horizons[1].p_up[1499]))  # nothing new was observed

    def test_p_up_is_calibrated_to_the_moving_closes_and_the_pit_is_uniform(self):
        hf, d = self.wf.horizons[1], self.d
        rows = np.arange(self.START, d.n - 1)
        R = hf.realized[rows]
        self.assertGreater(float(np.mean(R == 0.0)), 0.08, "many ties (genuine unchanged closes and halted rows)")
        mv = R != 0.0
        bias = float(hf.p_up[rows][mv].mean() - np.mean(R[mv] > 0))
        self.assertLess(abs(bias), 0.03, f"p_up vs the up-frequency of moving closes: {bias:+.3f}")
        pit = hf.pit[rows][np.isfinite(hf.pit[rows])]
        self.assertGreater(len(pit), 1100)
        self.assertGreater(stats.kstest(pit, "uniform").pvalue, 0.01)


def weekend_world(n=2400, seed=61, extra=0.6, v=0.012 ** 2):
    """Business-day GJR-GARCH-t path in which every bar that follows a weekend carries ``1 + extra`` times the variance."""
    r = garch_returns(n - 1, seed=seed, drift=0.0, v=v)
    monday = (np.arange(1, n) % 5) == 0  # row j of the helper's Monday-Friday calendar is a Monday iff j % 5 == 0
    return make_pricedata(r * np.where(monday, np.sqrt(1.0 + extra), 1.0), seed=seed)


class TestCalendarTime(unittest.TestCase):
    """Audit M-04: the one-day predictive is wider before a weekend (Friday origins); nothing else moves."""
    START = 700
    OFF = {"weekend_var_uplift": 0.0}

    @classmethod
    def setUpClass(cls):
        cls.data, cls.mkt = make_stock_and_market()
        cls.on = EN.walk_forward(cls.data, cls.mkt, start_idx=cls.START)
        cls.off = EN.walk_forward(cls.data, cls.mkt, start_idx=cls.START, cfg=cls.OFF)
        cls.rows = np.arange(cls.START, cls.data.n)
        cls.fri = EN.F.next_bar_after_weekend(cls.data.dates)[cls.rows]
        cls.world = weekend_world()
        w0 = 800
        cls.w_rows = np.arange(w0, cls.world.n - 1)
        cls.w_fri = EN.F.next_bar_after_weekend(cls.world.dates)[cls.w_rows]
        cls.w_on = EN.walk_forward(cls.world, None, horizons=(1,), start_idx=w0)
        cls.w_off = EN.walk_forward(cls.world, None, horizons=(1,), start_idx=w0, cfg=cls.OFF)

    def test_the_uplift_is_a_documented_default(self):
        self.assertTrue(0.10 < EN.DEFAULT_CFG["weekend_var_uplift"] < 0.25)  # development rows: 0.15 - 0.23
        self.assertEqual(self.on.config["cfg"]["weekend_var_uplift"], EN.DEFAULT_CFG["weekend_var_uplift"])
        self.assertEqual(self.off.config["cfg"]["weekend_var_uplift"], 0.0)
        self.assertEqual(int(self.fri.sum()), len(self.rows) // 5)  # the test calendar is Monday-Friday

    def test_longer_horizons_do_not_change_at_all(self):
        """For h >= 5 every window holds exactly one weekend bar: the effect is a constant, and the development rows show
        no gain from widening there, so the multiplier is applied to h = 1 only (and the other horizons are bit-identical)."""
        for h in (5, 20):
            a, b = self.on.horizons[h], self.off.horizons[h]
            for f in FORECAST_FIELDS + ("pit",):
                np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f"h={h} {f}")
            for name in self.on.model_names:
                for d in ("comp_p_up", "comp_mu", "weights"):
                    np.testing.assert_array_equal(getattr(a, d)[name], getattr(b, d)[name], err_msg=f"h={h} {d}.{name}")
        np.testing.assert_array_equal(self.on.regime_prob, self.off.regime_prob)
        np.testing.assert_array_equal(self.on.vol1d, self.off.vol1d)

    def test_one_day_interval_is_wider_only_before_a_weekend(self):
        on, off, rows, fri = self.on.horizons[1], self.off.horizons[1], self.rows, self.fri
        width = lambda hf: hf.q[rows, 7] - hf.q[rows, 1]  # the 90 % interval
        ratio = width(on) / width(off)
        sd_ratio = np.sqrt(1.0 + EN.DEFAULT_CFG["weekend_var_uplift"])
        self.assertTrue(np.all(ratio[fri] > 1.0 + 0.4 * (sd_ratio - 1.0)), float(ratio[fri].min()))
        self.assertTrue(np.all(ratio[fri] < 1.0 + 1.1 * (sd_ratio - 1.0)), float(ratio[fri].max()))
        self.assertAlmostEqual(float(np.mean(ratio[fri])), sd_ratio, delta=0.012)
        # other origins only move through the ensemble weights / calibrator state (they learn from the wider Friday scores)
        self.assertLess(float(np.max(np.abs(ratio[~fri] - 1.0))), 0.02)
        self.assertAlmostEqual(float(np.mean(ratio[~fri])), 1.0, delta=0.003)
        # a wider spread with the same centre: the mean forecast is untouched up to those weight effects
        scale = float(np.median(off.sigma[rows]))
        self.assertLess(float(np.max(np.abs(on.mu[rows] - off.mu[rows]))), 0.05 * scale)

    def test_p_up_moves_towards_one_half_before_a_weekend(self):
        on, off, rows, fri = self.on.horizons[1], self.off.horizons[1], self.rows, self.fri
        d_on, d_off = np.abs(on.p_up_raw[rows] - 0.5), np.abs(off.p_up_raw[rows] - 0.5)
        self.assertLess(float(d_on[fri].mean()), float(d_off[fri].mean()))  # same mean, more spread
        self.assertAlmostEqual(float(d_on[~fri].mean()), float(d_off[~fri].mean()), delta=0.002)

    def test_planted_weekend_variance_is_covered_better(self):
        """World in which the bar after a weekend has 1.6 x the variance (the audit's real data: 1.19 x).  The uplift cannot
        fix all of it, but Friday-origin coverage must move towards the nominal level and everything else must not."""
        R = self.w_on.horizons[1].realized[self.w_rows]
        fri = self.w_fri

        def cover(wf, lo=1, hi=7):
            q = wf.horizons[1].q[self.w_rows]
            return (R >= q[:, lo]) & (R <= q[:, hi])

        c_on, c_off = cover(self.w_on), cover(self.w_off)
        gain = float(c_on[fri].mean() - c_off[fri].mean())
        self.assertGreater(gain, 0.015, "Friday-origin 90 % coverage must improve")
        self.assertLess(float(c_off[fri].mean()), float(c_off[~fri].mean()) - 0.04, "the planted effect is really there")
        self.assertLess(abs(float(c_on[~fri].mean() - c_off[~fri].mean())), 0.006, "other origins are not affected")
        self.assertGreater(float(c_on[fri].mean()), float(c_off[fri].mean()))
        q_on, q_off = self.w_on.horizons[1].q[self.w_rows], self.w_off.horizons[1].q[self.w_rows]
        QLV = np.array(config.QLEVELS)

        def pin(q):
            d = R[:, None] - q
            return np.maximum(QLV * d, (QLV - 1.0) * d).mean(axis=1)

        self.assertLess(float(pin(q_on)[fri].mean()), float(pin(q_off)[fri].mean()), "pinball loss improves on Friday origins")

    def test_friday_origin_truncation_invariance_and_live_forecast(self):
        """The multiplier is a function of the origin's own date: a Friday forecast cannot depend on later rows."""
        i = 904  # a Friday of the helper calendar
        self.assertTrue(EN.F.next_bar_after_weekend(self.data.dates)[i])
        small = EN.walk_forward(self.data.slice(i + 1), self.mkt, start_idx=self.START)
        for h in HS:
            for f in FORECAST_FIELDS:
                np.testing.assert_allclose(getattr(self.on.horizons[h], f)[i], getattr(small.horizons[h], f)[-1], rtol=1e-9,
                                           atol=1e-12, err_msg=f"h={h} {f}")
        d = self.data.slice(i + 1)
        live_on = EN.forecast_latest(d, self.mkt)["horizons"]["1"]
        live_off = EN.forecast_latest(d, self.mkt, cfg=self.OFF)["horizons"]["1"]
        w_on = np.log1p(live_on["quantiles"]["0.95"]) - np.log1p(live_on["quantiles"]["0.05"])
        w_off = np.log1p(live_off["quantiles"]["0.95"]) - np.log1p(live_off["quantiles"]["0.05"])
        self.assertAlmostEqual(w_on / w_off, np.sqrt(1.0 + EN.DEFAULT_CFG["weekend_var_uplift"]), delta=0.02)
        self.assertAlmostEqual(live_on["sigma_log"], self.on.horizons[1].sigma[i], places=12)


class TestRealData(unittest.TestCase):
    """Smoke tests on real Yahoo data (skipped when offline)."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("TWQ_OFFLINE"):
            raise unittest.SkipTest("TWQ_OFFLINE set")
        cls.mkt = load_real_pricedata("^TWII", "%5ETWII", kind="index")
        cls.d = {s: load_real_pricedata(s, f"{s}.TW", kind="etf" if s.startswith("00") else "stock")
                 for s in ("2330", "0050")}
        if cls.mkt is None or any(v is None for v in cls.d.values()):
            raise unittest.SkipTest("real data unavailable (offline)")

    def check_finite(self, fc):
        self.assertEqual([p for p, v in walk(fc) if isinstance(v, float) and not np.isfinite(v)], [])

    def test_forecast_latest_speed_invariants(self):
        for sym, data in self.d.items():
            t0 = time.perf_counter()
            fc = EN.forecast_latest(data, self.mkt)
            dt = time.perf_counter() - t0
            self.assertLess(dt, 5.0, f"{sym}: forecast_latest took {dt:.1f}s (target <= 5 s on 25 years)")
            self.check_finite(fc)
            for h in fc["horizons"].values():
                q = np.array(list(h["quantiles"].values()))
                self.assertTrue(np.all(np.diff(q) > 0) and 0 < h["p_up"] < 1)

    def test_walk_forward_speed_and_live_equals_last_row(self):
        data = self.d["2330"]
        n = data.n
        s = n - config.TEST_BARS_DEFAULT
        t0 = time.perf_counter()
        wf = EN.walk_forward(data, self.mkt, start_idx=s)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 40.0, f"walk_forward took {dt:.1f}s for {n - s} rows x 3 horizons (target <= 40 s)")
        fc = EN.forecast_latest(data, self.mkt)
        for h in HS:
            self.assertAlmostEqual(fc["horizons"][str(h)]["p_up"], wf.horizons[h].p_up[-1], places=10)
            self.assertAlmostEqual(fc["horizons"][str(h)]["mu_log"], wf.horizons[h].mu[-1], places=10)
        for h in HS:  # out-of-sample bookkeeping on real data
            hf = wf.horizons[h]
            self.assertTrue(np.all(np.isfinite(hf.p_up[s:])) and np.all(np.diff(hf.q[s:], axis=1) > 0))
        # truncation invariance on real data at one cut
        cut = n - 37
        small = EN.walk_forward(data.slice(cut), self.mkt, start_idx=s)
        for h in HS:
            np.testing.assert_allclose(wf.horizons[h].p_up[cut - 1], small.horizons[h].p_up[-1], rtol=1e-9)
            np.testing.assert_allclose(wf.horizons[h].q[cut - 1], small.horizons[h].q[-1], rtol=1e-9)

    def test_etf_with_the_unadjusted_split_in_its_history(self):
        data = self.d["0050"]
        fc = EN.forecast_latest(data, self.mkt)
        self.check_finite(fc)
        self.assertLess(fc["vol"]["sigma_daily"], 0.05)


if __name__ == "__main__":
    unittest.main()
