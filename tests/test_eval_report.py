"""Tests for backend/evaluation/report.py: numbers vs independent recomputation, alignment guards, schema validator,
verdict logic (no-edge vs planted edge), degenerate inputs and Chinese text."""
import copy
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

import eval_helpers as H  # noqa: E402
from backend import config  # noqa: E402
from backend.evaluation import metrics as M  # noqa: E402
from backend.evaluation import report as RP  # noqa: E402


# ----------------------------------------------------------------------------- independent recomputation
def indep(data, wf, h):
    """Re-derive the evaluated rows and headline numbers with plain python loops (no evaluation code)."""
    hf = wf.horizons[h]
    a = data.adjclose
    n = len(a)
    R = [float("nan")] * n
    for i in range(n - h):
        R[i] = math.log(a[i + h] / a[i])
    rows = [i for i in range(wf.first_idx, n) if not math.isnan(hf.p_up[i]) and not math.isnan(R[i])]
    ties = [i for i in rows if R[i] == 0.0]
    scored = [i for i in rows if R[i] != 0.0]
    up = {i: R[i] > 0 for i in scored}
    p = {i: float(hf.p_up[i]) for i in scored}
    # expanding majority (matured outcomes j <= i-h, ties excluded; U >= M/2 -> up)
    maj = {}
    for i in scored:
        u = m = 0
        for j in range(0, i - h + 1):
            if not math.isnan(R[j]) and R[j] != 0.0:
                m += 1
                u += R[j] > 0
        maj[i] = (u >= m / 2.0)
    prev = {i: (math.log(a[i] / a[i - h]) > 0) if i - h >= 0 else False for i in scored}
    out = {"R": R, "rows": rows, "ties": ties, "scored": scored}
    ns = len(scored)
    out["acc"] = sum((p[i] >= 0.5) == up[i] for i in scored) / ns
    out["base"] = sum(up.values()) / ns
    out["prev_acc"] = sum(prev[i] == up[i] for i in scored) / ns
    out["maj_acc"] = sum(maj[i] == up[i] for i in scored) / ns
    # probabilistic block (DESIGN.md 4 / 9(a)): EVERY evaluated row, y = 1[R > 0] (a flat outcome is "not up"); references =
    # CAUSAL expanding base rate (outcomes matured by the origin, flat = not up, Laplace +1 / +2) and the ORACLE sample base rate
    ye = {i: R[i] > 0 for i in rows}
    ref = {}
    for i in rows:
        u = m = 0
        for j in range(0, i - h + 1):
            if not math.isnan(R[j]):
                m += 1
                u += R[j] > 0
        ref[i] = (u + 1) / (m + 2)
    ne = len(rows)
    b_e = sum(ye.values()) / ne

    def ll(pp, yy):
        return -math.log(pp) if yy else -math.log(1 - pp)

    pe = {i: float(hf.p_up[i]) for i in rows}
    out["n_eval"] = ne
    out["brier"] = sum((pe[i] - ye[i]) ** 2 for i in rows) / ne
    out["brier_clim"] = sum((ref[i] - ye[i]) ** 2 for i in rows) / ne
    out["brier_oracle"] = b_e * (1 - b_e)
    out["logloss"] = sum(ll(pe[i], ye[i]) for i in rows) / ne
    out["logloss_clim"] = sum(ll(ref[i], ye[i]) for i in rows) / ne
    out["logloss_oracle"] = sum(ll(b_e, ye[i]) for i in rows) / ne
    out["ref"] = ref
    return out


def planted(seed=0, edge="edge", **kw):
    kw.setdefault("n", 1700)
    return H.make_planted_world(seed=seed, edge=edge, **kw)


class TestReportNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.wf, _ = planted(1, "edge", signal_sd=0.004, h_list=(1, 5, 20))
        cls.reports = {h: RP.build_backtest_report(cls.data, cls.wf, h, n_boot=200) for h in (1, 5, 20)}

    def test_schema_valid_all_horizons(self):
        for h, rep in self.reports.items():
            self.assertEqual(RP.validate_backtest_report(rep), [], h)
            json.dumps(rep, allow_nan=False)
            self.assertEqual(rep["horizon"], h)

    def test_period_counts(self):
        for h, rep in self.reports.items():
            ix = indep(self.data, self.wf, h)
            per = rep["period"]
            self.assertEqual(per["n_forecasts"], len(ix["rows"]))
            self.assertEqual(per["n_scored"], len(ix["scored"]))
            self.assertEqual(per["n_tie_excluded"], len(ix["ties"]))
            self.assertEqual(per["start"], str(self.data.dates[ix["rows"][0]]))
            self.assertEqual(per["end"], str(self.data.dates[ix["rows"][-1]]))
        # for h=1 the last origin with a realised outcome is the second-to-last bar
        self.assertEqual(self.reports[1]["period"]["end"], str(self.data.dates[-2]))
        self.assertEqual(self.reports[20]["period"]["end"], str(self.data.dates[-21]))

    def test_direction_numbers(self):
        for h, rep in self.reports.items():
            ix = indep(self.data, self.wf, h)
            d = rep["direction"]
            self.assertAlmostEqual(d["accuracy"], ix["acc"], places=12)
            self.assertAlmostEqual(d["base_rate_up"], ix["base"], places=12)
            self.assertAlmostEqual(d["baselines"]["always_up"], ix["base"], places=12)
            self.assertAlmostEqual(d["baselines"]["prev_sign"], ix["prev_acc"], places=12)
            self.assertAlmostEqual(d["baselines"]["train_majority"], ix["maj_acc"], places=12)
            best = max(("always_up", "prev_sign", "train_majority"), key=lambda k: (d["baselines"][k],))
            self.assertEqual(d["baselines"][d["best_baseline"]], d["baselines"][best])
            n = len(ix["scored"])
            k = round(ix["acc"] * n)
            n_eff = n if h == 1 else int(round(n / h))
            k_eff = k if h == 1 else int(round(k / n * n_eff))
            ref = stats.binomtest(k_eff, n_eff, d["baselines"][d["best_baseline"]], alternative="greater").pvalue
            self.assertAlmostEqual(d["p_value_vs_best_baseline"], ref, places=10)
            self.assertAlmostEqual(d["p_value_vs_coin"], stats.binomtest(k_eff, n_eff, 0.5, "greater").pvalue,
                                   places=10)
            lo, hi = stats.binomtest(k_eff, n_eff).proportion_ci(method="wilson")
            self.assertAlmostEqual(d["ci95"][0], lo, places=10)
            self.assertAlmostEqual(d["ci95"][1], hi, places=10)
            self.assertEqual(sum(b["n"] for b in d["by_confidence"]), n)
            self.assertEqual(d["n"], n)

    def test_probabilistic_numbers(self):
        # CHANGED (audit C-01 / M-02 / S-01 / C-02): the headline baseline is the CAUSAL expanding base rate, the hindsight
        # (test-sample) base rate b (1 - b) the old test compared with is now the separate `*_oracle` keys, and the block is
        # scored on every evaluated row (flat = not up).
        for h, rep in self.reports.items():
            ix = indep(self.data, self.wf, h)
            pr = rep["probabilistic"]
            self.assertAlmostEqual(pr["brier"], ix["brier"], places=12)
            self.assertAlmostEqual(pr["brier_baseline"], ix["brier_clim"], places=12)
            self.assertAlmostEqual(pr["brier_skill"], 1 - ix["brier"] / ix["brier_clim"], places=12)
            self.assertAlmostEqual(pr["brier_baseline_oracle"], ix["brier_oracle"], places=12)
            self.assertAlmostEqual(pr["brier_skill_oracle"], 1 - ix["brier"] / ix["brier_oracle"], places=12)
            self.assertAlmostEqual(pr["log_loss"], ix["logloss"], places=12)
            self.assertAlmostEqual(pr["log_loss_baseline"], ix["logloss_clim"], places=12)
            self.assertAlmostEqual(pr["log_loss_skill"], 1 - ix["logloss"] / ix["logloss_clim"], places=12)
            self.assertAlmostEqual(pr["log_loss_baseline_oracle"], ix["logloss_oracle"], places=12)
            self.assertAlmostEqual(pr["log_loss_skill_oracle"], 1 - ix["logloss"] / ix["logloss_oracle"], places=12)
            self.assertEqual(sum(r["n"] for r in pr["calibration"]), ix["n_eval"])
            ece = sum(r["n"] * abs(r["p_mean"] - r["freq"]) for r in pr["calibration"]) / ix["n_eval"]
            self.assertAlmostEqual(pr["ece"], ece, places=12)
            self.assertAlmostEqual(pr["ece_noise_floor"], M.ece_noise_floor(ix["n_eval"], h, len(pr["calibration"])),
                                   places=12)

    def test_return_numbers(self):
        for h, rep in self.reports.items():
            ix = indep(self.data, self.wf, h)
            hf = self.wf.horizons[h]
            rows = ix["rows"]
            mu = np.array([hf.mu[i] for i in rows])
            R = np.array([ix["R"][i] for i in rows])
            rt = rep["returns"]
            rho = stats.spearmanr(mu, R).statistic
            self.assertAlmostEqual(rt["ic"], rho, places=12)
            n_eff = len(rows) / h
            t = rho * math.sqrt((n_eff - 2) / (1 - rho ** 2))
            self.assertAlmostEqual(rt["ic_pvalue"], 2 * stats.t.sf(abs(t), n_eff - 2), places=10)
            rmse = math.sqrt(float(np.mean((R - mu) ** 2)))
            self.assertAlmostEqual(rt["rmse_model"], rmse, places=12)
            # naive = better of {zero, expanding mean of matured outcomes}
            hist = []
            for i in rows:
                past = [ix["R"][j] for j in range(0, i - h + 1)]
                hist.append(sum(past) / len(past))
            hist = np.array(hist)
            naive = np.zeros(len(rows)) if np.sum(R ** 2) <= np.sum((R - hist) ** 2) else hist
            self.assertAlmostEqual(rt["rmse_naive"], math.sqrt(float(np.mean((R - naive) ** 2))), places=12)
            self.assertAlmostEqual(rt["oos_r2"], 1 - np.sum((R - mu) ** 2) / np.sum((R - naive) ** 2), places=12)
            d = (R - naive) ** 2 - (R - mu) ** 2
            lrv = _nw(d, h - 1)
            stat = d.mean() / math.sqrt(lrv / len(d))
            self.assertAlmostEqual(rt["dm_stat"], stat, places=9)
            self.assertAlmostEqual(rt["dm_pvalue"], 2 * stats.t.sf(abs(stat), len(d) - 1), places=10)
            ql = list(self.wf.qlevels)
            for lev, (a, b) in {"0.50": (0.25, 0.75), "0.80": (0.10, 0.90), "0.90": (0.05, 0.95)}.items():
                lo = np.array([hf.q[i, ql.index(a)] for i in rows])
                hi = np.array([hf.q[i, ql.index(b)] for i in rows])
                self.assertAlmostEqual(rt["coverage"][lev], float(np.mean((R >= lo) & (R <= hi))), places=12)
            self.assertEqual(len(rt["pit_hist"]), 10)
            self.assertAlmostEqual(sum(rt["pit_hist"]), 1.0)
            pit = np.array([hf.pit[i] for i in rows])
            self.assertAlmostEqual(rt["pit_ks_pvalue"],
                                   stats.kstwo.sf(stats.ks_1samp(pit, stats.uniform.cdf).statistic,
                                                  len(pit) if h == 1 else int(round(len(pit) / h))), places=10)
            tau = np.array(ql)[None, :]
            diff = R[:, None] - np.array([hf.q[i] for i in rows])
            self.assertAlmostEqual(rt["pinball_loss"], float(np.maximum(tau * diff, (tau - 1) * diff).mean()),
                                   places=12)

    def test_rolling_window(self):
        rep = self.reports[1]
        ro = rep["rolling"]
        ix = indep(self.data, self.wf, 1)
        hf = self.wf.horizons[1]
        corr = np.array([(hf.p_up[i] >= 0.5) == (ix["R"][i] > 0) for i in ix["scored"]], dtype=float)
        ups = np.array([ix["R"][i] > 0 for i in ix["scored"]], dtype=float)
        self.assertEqual(ro["window"], 250)
        full = np.convolve(corr, np.ones(250) / 250, mode="valid")
        fullb = np.convolve(ups, np.ones(250) / 250, mode="valid")
        keep = M.downsample_indices(len(full))                  # > 600 windows -> downsampled, first/last kept
        self.assertEqual(len(ro["accuracy"]), len(keep))
        self.assertLessEqual(len(keep), 600)
        np.testing.assert_allclose(ro["accuracy"], full[keep], atol=1e-5)
        np.testing.assert_allclose(ro["base_rate"], fullb[keep], atol=1e-5)
        win_end_dates = [str(self.data.dates[ix["scored"][249 + j]]) for j in keep]
        self.assertEqual(ro["dates"], win_end_dates)
        self.assertEqual(ro["dates"][0], str(self.data.dates[ix["scored"][249]]))
        self.assertEqual(ro["dates"][-1], str(self.data.dates[ix["scored"][-1]]))

    def test_params_and_costs(self):
        rep = self.reports[5]
        p = rep["params"]
        self.assertEqual(p["exec"], "next_open")
        self.assertEqual(p["sizing"], "threshold")
        self.assertEqual((p["threshold"], p["n_trials"]), (config.BUY_P_UP, 1))
        self.assertAlmostEqual(p["roundtrip_cost"], config.roundtrip_cost("stock"))
        self.assertEqual((p["min_train"], p["refit_every"]), (config.MIN_TRAIN, 21))
        etf, wf2, _ = H.make_planted_world(1300, 2, h_list=(1,), kind="etf", first_idx=500)
        r2 = RP.build_backtest_report(etf, wf2, 1, commission_discount=0.28, slippage_bps=2, n_boot=50)
        self.assertAlmostEqual(r2["params"]["roundtrip_cost"], config.roundtrip_cost("etf", 0.28, 2))
        self.assertEqual(r2["kind"], "etf")
        self.assertEqual(r2["symbol"], "TEST")

    def test_curve_cap_first_last(self):
        data, wf, _ = H.make_planted_world(3200, 3, h_list=(1,), first_idx=500)
        rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
        c = rep["strategy"]["curve"]
        self.assertEqual(len(c["dates"]), 600)
        self.assertEqual(c["dates"][0], str(data.dates[500]))
        self.assertEqual(c["dates"][-1], str(data.dates[-1]))
        self.assertAlmostEqual(c["strategy"][0], 1.0)
        self.assertAlmostEqual(c["buyhold"][0], 1.0)
        self.assertAlmostEqual(c["buyhold"][-1], data.adjclose[-1] / data.adjclose[500], places=5)
        self.assertLessEqual(len(rep["rolling"]["dates"]), 600)
        self.assertEqual(rep["rolling"]["dates"][-1], rep["period"]["end"])
        self.assertEqual(RP.validate_backtest_report(rep), [])

    def test_exec_modes_and_sizing_valid(self):
        for mode in ("next_open", "close"):
            for sizing in ("threshold", "kelly"):
                rep = RP.build_backtest_report(self.data, self.wf, 5, exec_mode=mode, sizing=sizing, threshold=0.6,
                                               n_trials=7, n_boot=50)
                self.assertEqual(RP.validate_backtest_report(rep), [], (mode, sizing))
                self.assertEqual(rep["params"]["exec"], mode)
                self.assertEqual(rep["params"]["n_trials"], 7)
        a = RP.build_backtest_report(self.data, self.wf, 5, n_trials=1, n_boot=50)["strategy"]["stats"]
        b = RP.build_backtest_report(self.data, self.wf, 5, n_trials=50, n_boot=50)["strategy"]["stats"]
        self.assertLess(b["dsr"], a["dsr"])
        self.assertAlmostEqual(a["dsr"], a["psr"], places=9)

    def test_invalid_parameters(self):
        for kw in ({"exec_mode": "vwap"}, {"sizing": "all-in"}, {"threshold": 1.2}, {"threshold": 0.0},
                   {"n_trials": 0}):
            with self.assertRaises(ValueError):
                RP.build_backtest_report(self.data, self.wf, 1, n_boot=20, **kw)
        with self.assertRaises(ValueError):
            RP.build_backtest_report(self.data, self.wf, 3, n_boot=20)   # horizon not in the walk-forward


def _nw(d, lag):
    d = np.asarray(d, dtype=float)
    m = len(d)
    x = d - d.mean()
    g0 = float(np.sum(x * x)) / m
    return g0 + 2 * sum((1 - k / (lag + 1)) * float(np.sum(x[k:] * x[:m - k])) / m for k in range(1, lag + 1))


class TestAlignmentGuards(unittest.TestCase):
    def setUp(self):
        self.data, self.wf, _ = planted(2, "edge", n=900, h_list=(1, 5), first_idx=300)

    def test_ok(self):
        chk = RP.check_realized_alignment(self.data, self.wf.horizons[5])
        self.assertTrue(chk["ok"])
        self.assertLess(chk["max_abs_diff"], 1e-12)
        self.assertEqual(chk["n_checked"], 900 - 5)

    def test_shifted_realized_raises(self):
        hf = self.wf.horizons[1]
        hf.realized = np.roll(hf.realized, 1)
        with self.assertRaises(ValueError) as cm:
            RP.build_backtest_report(self.data, self.wf, 1, n_boot=20)
        self.assertIn("alignment", str(cm.exception))

    def test_tiny_perturbation_raises(self):
        hf = self.wf.horizons[5]
        hf.realized = hf.realized.copy()
        hf.realized[400] += 1e-6
        with self.assertRaises(ValueError):
            RP.build_backtest_report(self.data, self.wf, 5, n_boot=20)

    def test_realized_claimed_where_data_has_none(self):
        hf = self.wf.horizons[1]
        hf.realized = hf.realized.copy()
        hf.realized[-1] = 0.01
        self.assertFalse(RP.check_realized_alignment(self.data, hf)["ok"])
        with self.assertRaises(ValueError):
            RP.build_backtest_report(self.data, self.wf, 1, n_boot=20)

    def test_wrong_dates_raise(self):
        wf = self.wf
        wf.dates = wf.dates + np.timedelta64(1, "D")
        with self.assertRaises(ValueError):
            RP.build_backtest_report(self.data, wf, 1, n_boot=20)

    def test_data_longer_than_walkforward_is_truncated(self):
        data2, wf2, _ = H.make_planted_world(900, 2, h_list=(1,), first_idx=300)
        short = data2.slice(850)
        data_long = data2
        wf_short = H.make_wf(short, {1: H.make_hf(short, 1, 300, wf2.horizons[1].p_up[:850], wf2.horizons[1].mu[:850],
                                                   wf2.horizons[1].sigma[:850])}, 300)
        rep = RP.build_backtest_report(data_long, wf_short, 1, n_boot=20)
        self.assertEqual(rep["period"]["end"], str(short.dates[-2]))
        self.assertEqual(RP.validate_backtest_report(rep), [])

    def test_str_keys_accepted(self):
        wf = self.wf
        wf.horizons = {str(k): v for k, v in wf.horizons.items()}
        rep = RP.build_backtest_report(self.data, wf, 5, n_boot=20)
        self.assertEqual(rep["horizon"], 5)


class TestValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, wf, _ = planted(4, "edge", n=1200, h_list=(1,), first_idx=500)
        cls.rep = RP.build_backtest_report(data, wf, 1, n_boot=50)

    def mutated(self, fn):
        r = copy.deepcopy(self.rep)
        fn(r)
        return RP.validate_backtest_report(r)

    def test_valid(self):
        self.assertEqual(RP.validate_backtest_report(self.rep), [])

    def test_missing_keys(self):
        for path in (["symbol"], ["direction"], ["direction", "ci95"], ["returns", "coverage", "0.80"],
                     ["strategy", "stats", "psr"], ["strategy", "curve", "position"], ["verdict", "level"],
                     ["params", "roundtrip_cost"], ["rolling", "window"], ["notes"], ["period", "n_scored"]):
            def f(r, path=path):
                d = r
                for k in path[:-1]:
                    d = d[k]
                del d[path[-1]]
            probs = self.mutated(f)
            self.assertTrue(any(path[-1] in p for p in probs), (path, probs))

    def test_mistyped_values(self):
        cases = {
            ("direction", "accuracy"): "0.5", ("direction", "n"): 1.5, ("direction", "ci95"): [0.1],
            ("returns", "pit_hist"): [0.1] * 9, ("verdict", "level"): "great_edge", ("verdict", "bullets"): "x",
            ("strategy", "stats", "n_trades"): True, ("strategy", "stats", "sharpe_ci95"): [None, "a"],
            ("params", "exec"): "twap", ("horizon",): "1", ("returns", "ic"): float("nan"),
            ("probabilistic", "calibration"): [{"p_mean": 0.5, "freq": 0.5}], ("notes",): [1, 2],
            ("direction", "best_baseline"): "magic", ("period", "start"): 5,
        }
        for path, bad in cases.items():
            def f(r, path=path, bad=bad):
                d = r
                for k in path[:-1]:
                    d = d[k]
                d[path[-1]] = bad
            probs = self.mutated(f)
            self.assertTrue(probs, (path, bad))
            self.assertTrue(any(path[-1] in p for p in probs), (path, probs))

    def test_cross_field_checks(self):
        probs = self.mutated(lambda r: r["strategy"]["curve"]["strategy"].pop())
        self.assertTrue(any("differ in length" in p for p in probs), probs)
        probs = self.mutated(lambda r: r["rolling"]["accuracy"].pop())
        self.assertTrue(any("rolling" in p for p in probs), probs)

        def too_long(r):
            for k in r["strategy"]["curve"]:
                r["strategy"]["curve"][k] = r["strategy"]["curve"][k][:1] * 601 if k != "dates" else ["2020-01-01"] * 601
        probs = self.mutated(too_long)
        self.assertTrue(any("601" in p for p in probs), probs)

    def test_nan_and_non_json(self):
        probs = self.mutated(lambda r: r["strategy"]["stats"].__setitem__("sharpe", float("inf")))
        self.assertTrue(probs)
        probs = self.mutated(lambda r: r.__setitem__("extra_ok", {1, 2}))
        self.assertTrue(any("JSON" in p for p in probs), probs)
        self.assertNotEqual(RP.validate_backtest_report([]), [])
        self.assertNotEqual(RP.validate_backtest_report(None), [])
        self.assertTrue(RP.validate_backtest_report({}))

    def test_extra_keys_tolerated(self):
        r = copy.deepcopy(self.rep)
        r["future_field"] = {"a": 1}
        r["direction"]["extra"] = 3
        self.assertEqual(RP.validate_backtest_report(r), [])

    def test_null_allowed_for_metrics_not_for_structure(self):
        r = copy.deepcopy(self.rep)
        r["returns"]["ic"] = None
        r["strategy"]["stats"]["sharpe"] = None
        self.assertEqual(RP.validate_backtest_report(r), [])
        r["direction"]["n"] = None
        self.assertTrue(RP.validate_backtest_report(r))


def ev(p, n=1200, n_eff=None, psr=None, dp=None):
    return {"n_scored": n, "n_eff": n if n_eff is None else n_eff, "p": dict(p), "effect": {}, "psr_vs_bh": psr,
            "diff_p": dp}


class TestAssessEvidence(unittest.TestCase):
    def lvl(self, *a, **k):
        return RP.assess_evidence(ev(*a, **k))

    def test_levels(self):
        none = {"direction": 0.5, "brier": 0.5, "ic": 0.5, "mse": 0.5}
        self.assertEqual(self.lvl(none)["level"], "no_edge")
        lone = dict(none, direction=1e-9)                      # one lone significant test = artefact
        r = self.lvl(lone)
        self.assertEqual((r["level"], r["reason"]), ("no_edge", "not_consistent"))
        weak = dict(none, direction=0.0004, ic=0.03)
        r = self.lvl(weak)
        self.assertEqual(r["level"], "weak_edge")
        self.assertAlmostEqual(r["p_fw"], 4 * 0.0004)
        mod = dict(none, direction=0.004, brier=0.008, ic=0.4)
        self.assertEqual(self.lvl(dict(mod, direction=0.0004))["level"], "moderate_edge")
        strong_p = {"direction": 1e-5, "brier": 5e-4, "ic": 1e-4, "mse": 0.2}
        self.assertEqual(self.lvl(strong_p)["level"], "moderate_edge")                       # not tradable
        self.assertEqual(self.lvl(strong_p, psr=0.99, dp=0.01)["level"], "strong_edge")
        self.assertEqual(self.lvl(strong_p, n=400, psr=0.99, dp=0.01)["level"], "moderate_edge")  # n < 500
        self.assertEqual(self.lvl(dict(strong_p, brier=0.002), psr=0.99, dp=0.01)["level"], "moderate_edge")
        self.assertEqual(self.lvl(strong_p, psr=0.9, dp=0.01)["level"], "moderate_edge")
        self.assertEqual(self.lvl(strong_p, psr=0.99, dp=0.2)["level"], "moderate_edge")

    def test_bonferroni_boundary(self):
        base = {"direction": 0.5, "brier": 0.5, "ic": 0.04, "mse": 0.5}
        self.assertEqual(self.lvl(dict(base, direction=0.0024))["level"], "weak_edge")     # 4 * 0.0024 < 0.01
        self.assertEqual(self.lvl(dict(base, direction=0.0026))["level"], "no_edge")       # 4 * 0.0026 > 0.01
        # fewer tests -> smaller multiplier
        three = {"direction": 0.003, "ic": 0.04, "mse": 0.5}
        self.assertEqual(self.lvl(three)["level"], "weak_edge")

    def test_small_samples_never_claim_an_edge(self):
        great = {"direction": 1e-9, "brier": 1e-9, "ic": 1e-9, "mse": 1e-9}
        r = self.lvl(great, n=100)
        self.assertEqual((r["level"], r["reason"]), ("no_edge", "small_sample"))
        r = self.lvl(great, n=1200, n_eff=20)
        self.assertEqual((r["level"], r["reason"]), ("no_edge", "small_sample"))
        self.assertEqual(self.lvl(great, n=150, n_eff=150)["level"], "moderate_edge")
        r = self.lvl({})
        self.assertEqual((r["level"], r["reason"]), ("no_edge", "no_tests"))

    def test_universe_thresholds_are_stricter(self):
        for k in RP.THRESH_STOCK:
            self.assertLessEqual(RP.THRESH_UNIVERSE[k], RP.THRESH_STOCK[k])
        p = {"ic": 0.0017, "picks": 0.03, "spread": 0.2}
        self.assertEqual(RP.assess_evidence(ev(p))["level"], "weak_edge")                        # 3 * 0.0017 < 0.01
        self.assertEqual(RP.assess_evidence(ev(p), RP.THRESH_UNIVERSE)["level"], "no_edge")      # 0.0051 >= 0.003
        p2 = {"ic": 0.0005, "picks": 0.03, "spread": 0.2}
        self.assertEqual(RP.assess_evidence(ev(p2), RP.THRESH_UNIVERSE)["level"], "weak_edge")   # 0.0015 < 0.003
        p3 = {"ic": 0.0005, "picks": 0.002, "spread": 0.2}
        r3 = RP.assess_evidence(ev(p3), RP.THRESH_UNIVERSE)                                     # two tests < 0.3 %
        self.assertEqual((r3["level"], r3["n_confirm"]), ("moderate_edge", 2))
        p4 = {"ic": 0.0005, "picks": 0.004, "spread": 0.2}                                      # 2nd test 0.4 % > 0.3 %
        r4 = RP.assess_evidence(ev(p4), RP.THRESH_UNIVERSE)
        self.assertEqual((r4["level"], r4["n_confirm"]), ("weak_edge", 1))

    def test_tradable_flag(self):
        self.assertTrue(self.lvl({"direction": 0.5}, psr=0.95, dp=0.049)["tradable"])
        self.assertFalse(self.lvl({"direction": 0.5}, psr=0.949, dp=0.01)["tradable"])
        self.assertFalse(self.lvl({"direction": 0.5}, psr=0.99, dp=0.05)["tradable"])
        self.assertFalse(self.lvl({"direction": 0.5}, psr=None, dp=None)["tradable"])


class TestVerdictSimulations(unittest.TestCase):
    """The headline requirement: matching the base rate is never an 'edge'; a planted real edge is found."""

    def levels(self, edge, seeds, hs=(1, 5, 20), **kw):
        out = []
        for seed in seeds:
            data, wf, _ = H.make_planted_world(1700, seed, edge=edge, h_list=hs, first_idx=500, **kw)
            for h in hs:
                a = RP.analyse(data, wf, h, with_strategy=False)
                out.append((seed, h, RP.assess_evidence(a["evidence"])["level"]))
        return out

    def test_base_rate_model_is_no_edge(self):
        res = self.levels("none", range(30))
        self.assertTrue(all(lv == "no_edge" for _, _, lv in res), [r for r in res if r[2] != "no_edge"])

    def test_uninformative_overconfident_model_is_no_edge(self):
        for oc in (1.0, 4.0):
            res = self.levels("noise", range(30), overconfidence=oc)
            self.assertTrue(all(lv == "no_edge" for _, _, lv in res), (oc, [r for r in res if r[2] != "no_edge"]))

    def test_end_to_end_no_edge_text(self):
        for seed in (0, 1, 2):
            data, wf, _ = H.make_planted_world(1700, seed, edge="none", h_list=(1, 5), first_idx=500)
            for h in (1, 5):
                rep = RP.build_backtest_report(data, wf, h, n_boot=100)
                self.assertEqual(rep["verdict"]["level"], "no_edge")
                self.assertEqual(rep["verdict"]["title"], "沒有統計上顯著的預測優勢")
                self.assertEqual(RP.validate_backtest_report(rep), [])

    def test_model_equal_to_the_baseline_is_no_edge(self):
        # the model *is* the causal climatology: identical to the 'train_majority' baseline by construction
        data, _, _ = H.make_planted_world(1500, 3, edge="none", h_list=(1,), first_idx=500)
        for h in (1, 5):
            R = M.realized_log_returns(data.adjclose, h)
            p = M.expanding_up_rate(R, h)
            hf = H.make_hf(data, h, 500, p, 0.0, 0.015 * math.sqrt(h))
            wf = H.make_wf(data, {h: hf}, 500)
            rep = RP.build_backtest_report(data, wf, h, n_boot=100)
            d = rep["direction"]
            self.assertEqual(rep["verdict"]["level"], "no_edge")
            self.assertAlmostEqual(d["accuracy"], d["baselines"]["train_majority"], places=12)
            self.assertGreater(d["p_value_vs_best_baseline"], 0.05)
            self.assertLessEqual(rep["probabilistic"]["brier_skill"], 0.0005)

    def test_inverted_edge_is_no_edge_and_negative_skill(self):
        data, wf, _ = H.make_planted_world(1700, 5, edge="edge", signal_sd=0.004, h_list=(1,), first_idx=500)
        hf = wf.horizons[1]
        inv = H.make_hf(data, 1, 500, 1 - hf.p_up, -hf.mu, hf.sigma)
        rep = RP.build_backtest_report(data, H.make_wf(data, {1: inv}, 500), 1, n_boot=100)
        self.assertEqual(rep["verdict"]["level"], "no_edge")
        self.assertLess(rep["direction"]["accuracy"], 0.5)
        self.assertLess(rep["probabilistic"]["brier_skill"], 0)
        self.assertGreater(rep["direction"]["p_value_vs_best_baseline"], 0.99)
        self.assertLess(rep["returns"]["ic"], 0)
        self.assertLess(rep["returns"]["oos_r2"], 0)

    def test_always_up_model_on_drifting_stock_is_no_edge(self):
        data = H.make_price_data(1700, seed=21, drift=0.001, vol=0.012)
        hf = H.make_hf(data, 1, 500, 0.6, 0.0008, 0.012)
        rep = RP.build_backtest_report(data, H.make_wf(data, {1: hf}, 500), 1, n_boot=100)
        self.assertEqual(rep["verdict"]["level"], "no_edge")
        self.assertAlmostEqual(rep["direction"]["accuracy"], rep["direction"]["base_rate_up"])

    def test_planted_edge_detected(self):
        for seed in range(6):
            data, wf, _ = H.make_planted_world(1700, seed, edge="edge", signal_sd=0.006, h_list=(1, 5), first_idx=500)
            for h in (1, 5):
                rep = RP.build_backtest_report(data, wf, h, n_boot=200)
                self.assertIn(rep["verdict"]["level"], ("moderate_edge", "strong_edge"), (seed, h))
                self.assertEqual(RP.validate_backtest_report(rep), [])
                d = rep["direction"]
                self.assertGreater(d["accuracy"], d["baselines"][d["best_baseline"]] + 0.03)
                if h == 1:   # (h=5 uses the effective sample n/5: its direction test is far less powerful by design)
                    self.assertLess(d["p_value_vs_best_baseline"], 1e-3)
                self.assertGreater(rep["probabilistic"]["brier_skill"], 0)
                self.assertGreater(rep["returns"]["ic"], 0.2)
                self.assertGreater(rep["returns"]["oos_r2"], 0)

    def test_planted_edge_strong_when_tradable(self):
        data, wf, _ = H.make_planted_world(1700, 2, edge="edge", signal_sd=0.01, h_list=(5,), first_idx=500)
        rep = RP.build_backtest_report(data, wf, 5, n_boot=300)
        self.assertEqual(rep["verdict"]["level"], "strong_edge")
        st, bm = rep["strategy"]["stats"], rep["strategy"]["benchmark"]
        self.assertGreater(st["sharpe"], bm["sharpe"])
        self.assertGreater(rep["strategy"]["sharpe_diff_ci95"][0], 0)
        self.assertIn("不保證獲利", rep["verdict"]["text"])

    def test_weak_planted_edge_is_not_over_claimed(self):
        # true IC ~ 0.03 (accuracy ~ 51 %): usually invisible; a lucky sample may reach 'moderate' but never 'strong'
        res = self.levels("edge", range(30), hs=(1,), signal_sd=0.0005)
        lv = [r[2] for r in res]
        self.assertGreaterEqual(lv.count("no_edge"), 24, lv)
        self.assertNotIn("strong_edge", lv)

    def test_determinism(self):
        data, wf, _ = H.make_planted_world(1500, 7, edge="edge", signal_sd=0.004, h_list=(1,), first_idx=500)
        a = RP.build_backtest_report(data, wf, 1)
        b = RP.build_backtest_report(data, wf, 1)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_runtime(self):
        import time
        data, wf, _ = H.make_planted_world(1750, 1, h_list=(1, 5, 20), first_idx=500)
        t0 = time.time()
        for h in (1, 5, 20):
            RP.build_backtest_report(data, wf, h)
        self.assertLess(time.time() - t0, 6.0)


class TestDegenerateInputs(unittest.TestCase):
    def check(self, data, wf, h=1, **kw):
        rep = RP.build_backtest_report(data, wf, h, n_boot=50, **kw)
        self.assertEqual(RP.validate_backtest_report(rep), [])
        json.dumps(rep, allow_nan=False)
        self.assertEqual(rep["verdict"]["level"], "no_edge")
        return rep

    def test_no_forecasts_at_all(self):
        data = H.make_price_data(800, 1)
        n = data.n
        hf = H.make_hf(data, 1, 500, np.nan, np.nan, np.nan)
        rep = self.check(data, H.make_wf(data, {1: hf}, 500))
        self.assertEqual(rep["period"], {"start": None, "end": None, "n_forecasts": 0, "n_scored": 0,
                                         "n_tie_excluded": 0})
        self.assertIsNone(rep["direction"]["accuracy"])
        self.assertIsNone(rep["probabilistic"]["brier"])
        self.assertIsNone(rep["returns"]["ic"])
        self.assertEqual(rep["rolling"]["dates"], [])
        self.assertEqual(rep["strategy"]["stats"]["n_trades"], 0)

    def test_constant_prices_all_ties(self):
        data = H.make_tiny_data([50.0] * 700, [50.0] * 700)
        hf = H.make_hf(data, 1, 500, 0.6, 0.001, 0.01)
        rep = self.check(data, H.make_wf(data, {1: hf}, 500))
        self.assertEqual(rep["period"]["n_scored"], 0)
        self.assertEqual(rep["period"]["n_tie_excluded"], rep["period"]["n_forecasts"])
        self.assertGreater(rep["period"]["n_tie_excluded"], 150)
        self.assertIsNone(rep["direction"]["accuracy"])
        self.assertIsNone(rep["strategy"]["benchmark"]["sharpe"])

    def test_tiny_sample(self):
        data, wf, _ = H.make_planted_world(800, 4, edge="edge", signal_sd=0.02, h_list=(1, 5), first_idx=788)
        for h in (1, 5):
            rep = self.check(data, wf, h)
            self.assertLessEqual(rep["period"]["n_scored"], 11)
            self.assertIn("樣本", rep["verdict"]["text"])
        wf2 = H.make_wf(data, wf.horizons, 799)       # only the last row has a forecast and no outcome yet
        rep = self.check(data, wf2, 1)
        self.assertEqual(rep["period"]["n_forecasts"], 0)

    def test_ties_are_excluded_and_counted(self):
        data = H.make_price_data(1200, 8)
        c = data.close.copy()
        for i in range(550, 1100, 10):          # flat days: close repeats
            c[i] = c[i - 1]
        data.close, data.adjclose = c, c.copy()
        data.open = np.where(np.arange(1200) % 10 == 0, data.open, data.open)
        hf = H.make_hf(data, 1, 500, np.random.default_rng(1).uniform(0.4, 0.6, 1200), 0.0003, 0.015)
        wf = H.make_wf(data, {1: hf}, 500)
        rep = self.check(data, wf)
        ix = indep(data, wf, 1)
        self.assertEqual(rep["period"]["n_tie_excluded"], len(ix["ties"]))
        self.assertGreater(len(ix["ties"]), 40)
        self.assertEqual(rep["period"]["n_scored"], len(ix["scored"]))
        self.assertAlmostEqual(rep["direction"]["accuracy"], ix["acc"], places=12)
        self.assertEqual(rep["period"]["n_forecasts"], rep["period"]["n_scored"] + rep["period"]["n_tie_excluded"])

    def test_no_trades_text(self):
        data, wf, _ = H.make_planted_world(1200, 9, edge="none", h_list=(1,), first_idx=500)
        rep = self.check(data, wf, 1)
        self.assertEqual(rep["strategy"]["stats"]["n_trades"], 0)       # 1-day edge never covers the 0.57 % cost
        self.assertTrue(any("全程空手" in b for b in rep["verdict"]["bullets"]))

    def test_nan_pit_and_quantiles(self):
        data, wf, _ = H.make_planted_world(1000, 10, edge="edge", signal_sd=0.004, h_list=(1,), first_idx=500)
        hf = wf.horizons[1]
        hf.pit[600:620] = np.nan
        hf.q[700:710, :] = np.nan
        rep = self.check(data, wf) if False else RP.build_backtest_report(data, wf, 1, n_boot=50)
        self.assertEqual(RP.validate_backtest_report(rep), [])

    def test_probabilities_at_and_beyond_bounds(self):
        data = H.make_price_data(900, 11)
        p = np.random.default_rng(0).choice([0.0, 1.0, 1.2, -0.3, 0.5], 900)
        hf = H.make_hf(data, 1, 400, p, 0.0, 0.015)
        rep = RP.build_backtest_report(data, H.make_wf(data, {1: hf}, 400), 1, n_boot=50)
        self.assertEqual(RP.validate_backtest_report(rep), [])
        self.assertTrue(math.isfinite(rep["probabilistic"]["log_loss"]))

    def test_nan_sigma_rows_are_skipped(self):
        data, wf, _ = H.make_planted_world(1000, 12, edge="edge", signal_sd=0.004, h_list=(1,), first_idx=500)
        wf.horizons[1].sigma[700:720] = np.nan
        rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
        self.assertEqual(RP.validate_backtest_report(rep), [])
        self.assertEqual(rep["period"]["n_forecasts"], (999 - 500) - 20)      # origins 500..998 minus 20 NaN rows

    def test_horizon_longer_than_window(self):
        data, wf, _ = H.make_planted_world(560, 13, edge="edge", h_list=(20,), first_idx=500)
        rep = self.check(data, wf, 20)
        self.assertEqual(rep["period"]["n_forecasts"], 40)

    def test_outputs_do_not_depend_on_wf_array_dtype(self):
        data, wf, _ = H.make_planted_world(900, 14, edge="edge", h_list=(1,), first_idx=400)
        hf = wf.horizons[1]
        hf.p_up = hf.p_up.astype(np.float32).astype(np.float64)
        rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
        self.assertEqual(RP.validate_backtest_report(rep), [])


def with_split_artifact(seed=3, row=700, factor=0.25, first=500, n=1200):
    """World whose prices contain an unadjusted split at ``row`` (all later prices x ``factor``)."""
    data, wf, _ = H.make_planted_world(n, seed, edge="none", h_list=(1,), first_idx=first)
    for arr in (data.open, data.high, data.low, data.close, data.adjclose):
        arr[row:] *= factor
    hf = wf.horizons[1]
    wf2 = H.make_wf(data, {1: H.make_hf(data, 1, first, hf.p_up.copy(), hf.mu.copy(), hf.sigma.copy())}, first)
    return data, wf2


class TestDataAlerts(unittest.TestCase):
    def test_clean_data_and_limit_moves_do_not_alert(self):
        data, wf, _ = H.make_planted_world(1200, 3, edge="none", h_list=(1,), first_idx=500)
        self.assertEqual(RP.data_alerts(data, 500), [])
        for arr in (data.open, data.high, data.low, data.close, data.adjclose):
            arr[800:] *= 1.10                                   # a +10 % limit-up day is legitimate
        self.assertEqual(RP.data_alerts(data, 500), [])

    def test_split_artifact_is_reported_in_notes_and_bullets(self):
        data, wf = with_split_artifact()
        al = RP.data_alerts(data, 500)
        self.assertEqual(len(al), 1)
        self.assertEqual(al[0][0], str(data.dates[700]))
        self.assertAlmostEqual(al[0][1], -0.75, delta=0.05)         # x0.25 on top of that day's own random move
        rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
        self.assertEqual(RP.validate_backtest_report(rep), [])
        self.assertTrue(rep["notes"][0].startswith("資料品質警示"))
        self.assertTrue(rep["verdict"]["bullets"][0].startswith("資料品質警示"))
        self.assertIn(str(data.dates[700]), rep["notes"][0])
        self.assertRegex(rep["notes"][0], r"-7[0-9]\.[0-9]%")
        self.assertIn("不可靠", rep["notes"][0])
        self.assertFalse(SIMPLIFIED & set(rep["notes"][0]))
        clean = RP.build_backtest_report(*H.make_planted_world(1200, 3, edge="none", h_list=(1,), first_idx=500)[:2], 1,
                                         n_boot=50)
        self.assertFalse(any("資料品質警示" in n for n in clean["notes"]))

    def test_artifact_before_the_evaluation_window_is_ignored(self):
        data, wf = with_split_artifact(row=300, first=500)
        self.assertEqual(RP.data_alerts(data, 500), [])
        rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
        self.assertFalse(any("資料品質警示" in n for n in rep["notes"]))

    def test_largest_first_and_limit(self):
        data, _, _ = H.make_planted_world(1200, 3, edge="none", h_list=(1,), first_idx=500)
        for row, f in ((600, 0.5), (700, 0.2), (800, 2.0), (900, 0.6)):
            for arr in (data.open, data.high, data.low, data.close, data.adjclose):
                arr[row:] *= f
        al = RP.data_alerts(data, 500)
        self.assertEqual(len(al), 3)
        # |+100 %| (row 800) > |-80 %| (row 700) > |-50 %| (row 600) > |-40 %| (row 900, dropped by the limit)
        self.assertEqual([a[0] for a in al], [str(data.dates[800]), str(data.dates[700]), str(data.dates[600])])
        self.assertGreater(al[0][1], 0.9)
        self.assertEqual(len(RP.data_alerts(data, 500, limit=10)), 4)


SIMPLIFIED = set("预统显优样获检准关买损较约应错该为实时间点发线图号体义观验览态质总结况构运动过区务强价绩费记录别难势报")
TRAD_ONLY_CHECK = ("預測", "統計", "顯著")
FORBIDDEN = ("保證獲利", "穩賺", "必賺", "一定賺", "包賺", "穩賠不賠", "保證賺", "穩定獲利")


class TestChineseText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {}
        data, wf, _ = H.make_planted_world(1700, 3, edge="noise", h_list=(1, 5), first_idx=500)
        cls.reports["no"] = RP.build_backtest_report(data, wf, 5, n_boot=100)
        d2, w2, _ = H.make_planted_world(1700, 3, edge="edge", signal_sd=0.01, h_list=(1, 5), first_idx=500)
        cls.reports["edge"] = RP.build_backtest_report(d2, w2, 5, n_boot=200)
        cls.reports["edge1"] = RP.build_backtest_report(d2, w2, 1, n_boot=200)
        cls.data, cls.wf = data, wf

    def strings(self, rep):
        v = rep["verdict"]
        return [v["title"], v["text"], *v["bullets"], *rep["notes"], rep["strategy"]["rule"]]

    def test_no_simplified_characters_and_no_promises(self):
        for name, rep in self.reports.items():
            for s in self.strings(rep):
                bad = SIMPLIFIED & set(s)
                self.assertFalse(bad, (name, bad, s))
                cleaned = s.replace("不保證獲利", "").replace("不保證", "")
                for w in FORBIDDEN:
                    self.assertNotIn(w, cleaned, (name, w))

    def test_every_level_title_and_disclaimer(self):
        titles = {k: RP.LEVEL_TITLES[k] for k in RP.LEVELS}
        self.assertEqual(len(set(titles.values())), 4)
        a = RP.analyse(self.data, self.wf, 5, n_boot=50)
        for level in RP.LEVELS:
            ass = {"level": level, "p_fw": 0.001, "n_tests": 4, "n_confirm": 3, "n_strong": 3, "tradable": level == "strong_edge",
                   "reason": "ok"}
            v = RP.make_verdict(a, ass, h=5, exec_mode="next_open")
            self.assertEqual(v["level"], level)
            self.assertEqual(v["title"], titles[level])
            self.assertIn("不構成投資建議", v["text"])
            self.assertTrue(v["bullets"])
            cleaned = v["text"].replace("不保證獲利", "").replace("不保證", "")
            for w in FORBIDDEN:
                self.assertNotIn(w, cleaned)
        v = RP.make_verdict(a, {"level": "no_edge", "p_fw": 0.5, "n_tests": 4, "n_confirm": 0, "n_strong": 0,
                                "tradable": False, "reason": "small_sample"}, h=5, exec_mode="next_open")
        self.assertIn("樣本太少", v["text"])

    def test_text_quotes_the_concrete_numbers(self):
        rep = self.reports["no"]
        v, d, pr = rep["verdict"], rep["direction"], rep["probabilistic"]
        self.assertIn(f"{d['accuracy'] * 100:.1f}%", v["text"])
        self.assertIn(f"{rep['period']['n_scored']:,}", v["text"])
        self.assertIn(rep["period"]["start"], v["text"])
        self.assertIn(rep["period"]["end"], v["text"])
        self.assertIn(RP.BASELINE_LABELS[d["best_baseline"]], v["text"])
        self.assertIn(f"{d['ci95'][0] * 100:.1f}%", v["text"])
        joined = "".join(v["bullets"])
        self.assertIn(f"{pr['brier']:.4f}", joined)
        self.assertIn(f"{rep['returns']['ic']:+.3f}", joined)
        st = rep["strategy"]["stats"]
        self.assertIn(f"{st['sharpe']:.2f}" if st["sharpe"] is not None else "全程空手", joined)
        self.assertIn("未來5日", v["text"])
        self.assertIn("重疊", joined)                         # overlap adjustment disclosed for h = 5

    def test_edge_text_lists_significant_tests(self):
        rep = self.reports["edge"]
        self.assertIn(rep["verdict"]["level"], ("moderate_edge", "strong_edge"))
        t = rep["verdict"]["text"]
        self.assertIn("方向準確率", t)
        self.assertIn("p<0.001", t)
        self.assertIn("1% 顯著", t)

    def test_notes_disclose_assumptions(self):
        notes = "".join(self.reports["no"]["notes"])
        for key in ("手續費", "證交稅", "滑價", "漲跌停", "存活", "選擇偏誤") if False else ("手續費", "證交稅", "滑價", "漲跌停",
                                                                              "還原權息", "樣本外"):
            self.assertIn(key, notes)
        self.assertIn("隔日開盤", notes)
        data, wf, _ = H.make_planted_world(900, 3, h_list=(1,), first_idx=500)
        rc = RP.build_backtest_report(data, wf, 1, exec_mode="close", n_boot=50)
        self.assertIn("收盤價成交", "".join(rc["notes"]))
        self.assertTrue(any("尚未到期" in n for n in self.reports["no"]["notes"]))

    def test_lone_significant_test_wording(self):
        a = RP.analyse(self.data, self.wf, 5, n_boot=50)
        ev_ = copy.deepcopy(a["evidence"])
        ev_["p"] = {"direction": 1e-6, "brier": 0.4, "ic": 0.5, "mse": 0.6}
        ass = RP.assess_evidence(ev_)
        self.assertEqual((ass["level"], ass["reason"]), ("no_edge", "not_consistent"))
        v = RP.make_verdict(a, ass, h=5, exec_mode="next_open")
        self.assertIn("只有 1 項檢驗", v["text"])
        self.assertIn("偶然", v["text"])
        self.assertNotIn("都未達到", v["text"])

    def test_phrasing_has_no_redundant_p_prefix_and_marks_two_sided(self):
        for rep in self.reports.values():
            t = rep["verdict"]["text"] + "".join(rep["verdict"]["bullets"])
            self.assertNotIn("p 值 p=", t)
            self.assertIn("雙尾", t)
            self.assertIn("單尾", t)
        bullets = "".join(self.reports["no"]["verdict"]["bullets"])
        self.assertTrue("模型誤差較大" in bullets or "模型誤差較小" in bullets)

    def test_formatters(self):
        self.assertEqual(RP.pct(0.5364), "53.6%")
        self.assertEqual(RP.pct(0.0123, 2, True), "+1.23%")
        self.assertEqual(RP.pct(None), "—")
        self.assertEqual(RP.pct(float("nan")), "—")
        self.assertEqual(RP.pval(0.0004), "p<0.001")
        self.assertEqual(RP.pval(0.004), "p=0.004")
        self.assertEqual(RP.pval(0.31), "p=0.31")
        self.assertEqual(RP.pval(None), "p=—")
        self.assertEqual(RP.horizon_label(1), "明日")
        self.assertEqual(RP.horizon_label(5), "未來5日")
        self.assertEqual(RP.horizon_label(20), "未來20日")
        self.assertEqual(RP.horizon_label(60), "未來60日")


# ============================================================================= audit C-02: what the probabilistic block scores
class TestProbabilisticPopulation(unittest.TestCase):
    """The probabilistic block scores the event R > 0 on EVERY evaluated forecast (a flat outcome is 'not up', exactly like
    the engine's calibrator target); only the direction block drops flat days (it is undefined when the price does not move)."""

    @classmethod
    def setUpClass(cls):
        cls.data = H.make_tick_data(1500, 4)                       # ~12 % unchanged closes at h = 1
        R = M.realized_log_returns(cls.data.adjclose, 1)
        cls.R = R
        cls.E = np.arange(500, 1499)
        cls.y = (R[cls.E] > 0).astype(float)
        cls.b = float(cls.y.mean())
        hf = H.make_hf(cls.data, 1, 500, cls.b, 0.0, 0.012)         # constant forecast = the up-rate of the event R > 0
        cls.wf = H.make_wf(cls.data, {1: hf}, 500)
        cls.rep = RP.build_backtest_report(cls.data, cls.wf, 1, n_boot=50)

    def test_flat_days_exist_are_counted_and_only_leave_the_direction_block(self):
        per = self.rep["period"]
        self.assertGreater(per["n_tie_excluded"], 80)
        self.assertEqual(per["n_forecasts"], per["n_scored"] + per["n_tie_excluded"])
        self.assertEqual(self.rep["direction"]["n"], per["n_scored"])
        self.assertEqual(RP.validate_backtest_report(self.rep), [])

    def test_probabilistic_block_scores_every_evaluated_row(self):
        pr, per = self.rep["probabilistic"], self.rep["period"]
        self.assertEqual(sum(r["n"] for r in pr["calibration"]), per["n_forecasts"])
        pooled = sum(r["n"] * r["freq"] for r in pr["calibration"]) / per["n_forecasts"]
        self.assertAlmostEqual(pooled, self.b, places=12)               # up-rate of R > 0 INCLUDING the flat days
        self.assertGreater(self.rep["direction"]["base_rate_up"], self.b + 0.03)   # P(up | not flat): the old, inflated one
        self.assertAlmostEqual(pr["brier"], self.b * (1 - self.b), places=12)

    def test_a_forecaster_calibrated_for_the_event_has_exactly_zero_oracle_skill(self):
        pr = self.rep["probabilistic"]
        self.assertAlmostEqual(pr["brier_skill_oracle"], 0.0, places=12)
        self.assertAlmostEqual(pr["log_loss_skill_oracle"], 0.0, places=12)
        # Before the fix the flat rows were dropped: the scored up-rate was ~6 pp higher and the very same, perfectly
        # calibrated constant forecast looked clearly miscalibrated (negative skill).
        nf = self.R[self.E][self.R[self.E] != 0] > 0
        b_nf = float(nf.mean())
        old_skill = 1 - float(np.mean((self.b - nf) ** 2)) / (b_nf * (1 - b_nf))
        self.assertLess(old_skill, -0.01)

    def test_direction_block_is_unchanged_by_flat_days(self):
        ix = indep(self.data, self.wf, 1)
        d = self.rep["direction"]
        self.assertAlmostEqual(d["accuracy"], ix["acc"], places=12)
        self.assertAlmostEqual(d["base_rate_up"], ix["base"], places=12)
        self.assertAlmostEqual(d["baselines"]["train_majority"], ix["maj_acc"], places=12)
        self.assertTrue(any("平盤" in n and "未上漲" in n for n in self.rep["notes"]))
        self.assertEqual((self.rep["probabilistic"]["p_up_event"], self.rep["probabilistic"]["n"]),
                         ("up", self.rep["period"]["n_forecasts"]))                # undeclared = P(R > 0)

    def test_declared_up_given_move_leaves_flat_rows_out_of_the_probabilistic_block(self):
        # the quant engine declares p_up = P(R > 0 | R != 0) (wf.config["p_up_event"]): the scorer must follow that, otherwise a
        # perfectly calibrated forecast looks biased high by 1 / (1 - tie rate)
        nf = self.R[self.E][self.R[self.E] != 0] > 0
        b_nf = float(nf.mean())
        hf = H.make_hf(self.data, 1, 500, b_nf, 0.0, 0.012)               # constant = the up-rate among MOVING days
        wf = H.make_wf(self.data, {1: hf}, 500)
        wf.config = {"p_up_event": "up_given_move"}
        rep = RP.build_backtest_report(self.data, wf, 1, n_boot=50)
        pr, per = rep["probabilistic"], rep["period"]
        self.assertEqual((pr["p_up_event"], pr["n"]), ("up_given_move", per["n_scored"]))
        self.assertLess(per["n_scored"], per["n_forecasts"])
        self.assertEqual(sum(r["n"] for r in pr["calibration"]), per["n_scored"])
        self.assertEqual(pr["n"], rep["direction"]["n"])
        self.assertAlmostEqual(pr["brier"], b_nf * (1 - b_nf), places=12)
        self.assertAlmostEqual(pr["brier_skill_oracle"], 0.0, places=12)
        self.assertAlmostEqual(pr["log_loss_skill_oracle"], 0.0, places=12)
        self.assertTrue(any("平盤" in n and "不計入" in n and "P(報酬>0｜報酬≠0)" in n for n in rep["notes"]))
        self.assertEqual(RP.validate_backtest_report(rep), [])
        # the very same forecasts without the declaration are scored as P(R > 0) on every row and look clearly miscalibrated
        wf0 = H.make_wf(self.data, {1: hf}, 500)
        pr0 = RP.build_backtest_report(self.data, wf0, 1, n_boot=50)["probabilistic"]
        self.assertEqual((pr0["p_up_event"], pr0["n"]), ("up", per["n_forecasts"]))
        self.assertLess(pr0["brier_skill_oracle"], -0.01)
        self.assertGreater(pr0["ece"], 0.05)

    def test_event_helper(self):
        wf = H.make_wf(self.data, {1: H.make_hf(self.data, 1, 500, 0.5, 0.0, 0.012)}, 500)
        self.assertEqual(RP.p_up_event(wf), "up")
        for value, want in (("up_given_move", "up_given_move"), ("up", "up"), ("whatever", "up"), (None, "up")):
            wf.config = {"p_up_event": value}
            self.assertEqual(RP.p_up_event(wf), want)
        wf.config = None
        self.assertEqual(RP.p_up_event(wf), "up")


# ============================================================================= audit C-01 / M-02 / S-01: Brier references
class TestBrierReferences(unittest.TestCase):
    """DESIGN.md 9(a): headline Brier / log-loss skill against the CAUSAL expanding base rate, the hindsight (oracle) base rate
    reported apart, and a verdict leg that has to beat BOTH."""

    @classmethod
    def setUpClass(cls):
        n = 1700
        latent = np.where(np.arange(n) < n // 2, -0.002, 0.002)       # the up-rate changes over time (44 % -> 56 %)
        cls.data = H.make_price_data(n, seed=8, drift=0.0, vol=0.012, latent=latent)

    def test_model_equal_to_the_causal_reference_has_zero_skill_and_p_one_half(self):
        for h in (1, 5):
            R = M.realized_log_returns(self.data.adjclose, h)
            p = M.expanding_up_rate(R, h, ties="not_up")
            wf = H.make_wf(self.data, {h: H.make_hf(self.data, h, 500, p, 0.0, 0.012 * math.sqrt(h))}, 500)
            a = RP.analyse(self.data, wf, h, with_strategy=False)
            pr, ev = a["probabilistic"], a["evidence"]
            self.assertAlmostEqual(pr["brier_skill"], 0.0, places=12)
            self.assertAlmostEqual(pr["log_loss_skill"], 0.0, places=12)
            self.assertAlmostEqual(ev["p_brier_causal"], 0.5, places=12)               # identical forecasts: DM stat 0
            self.assertNotAlmostEqual(pr["brier_skill_oracle"], 0.0, places=4)        # the hindsight reference is another one
            self.assertAlmostEqual(ev["p"]["brier"], max(ev["p_brier_causal"], ev["p_brier_oracle"]), places=15)

    def test_declared_event_selects_the_matching_causal_reference(self):
        # a forecaster equal to the causal base rate of the SAME event has exactly zero skill in either convention; paired with
        # the other convention's reference it is visibly off (flat closes are ~12 % of the h = 1 outcomes of this world)
        data = H.make_tick_data(1500, 4)
        for h in (1, 5):
            R = M.realized_log_returns(data.adjclose, h)
            skills = {}
            for event, ties in (("up", "not_up"), ("up_given_move", "drop")):
                for model_ties in ("not_up", "drop"):
                    p = M.expanding_up_rate(R, h, ties=model_ties)
                    wf = H.make_wf(data, {h: H.make_hf(data, h, 500, p, 0.0, 0.012 * math.sqrt(h))}, 500)
                    wf.config = {"p_up_event": event}
                    a = RP.analyse(data, wf, h, with_strategy=False)
                    self.assertEqual(a["probabilistic"]["p_up_event"], event)
                    skills[event, model_ties] = a["probabilistic"]["brier_skill"]
            self.assertAlmostEqual(skills["up", "not_up"], 0.0, places=12)
            self.assertAlmostEqual(skills["up_given_move", "drop"], 0.0, places=12)
            if h == 1:
                self.assertLess(skills["up", "drop"], -0.005)                # P(up | moving) scored as P(R > 0): too high
                self.assertLess(skills["up_given_move", "not_up"], -0.005)   # P(R > 0) scored as P(up | moving): too low

    def test_headline_is_causal_oracle_is_apart_and_labelled(self):
        data, wf, _ = H.make_planted_world(1700, 3, edge="none", h_list=(1, 5, 20), first_idx=500)
        for h in (1, 5, 20):
            pr = RP.analyse(data, wf, h, with_strategy=False)["probabilistic"]
            self.assertNotAlmostEqual(pr["brier_baseline"], pr["brier_baseline_oracle"], places=7)
            self.assertAlmostEqual(pr["brier_skill"], 1 - pr["brier"] / pr["brier_baseline"], places=12)
            self.assertAlmostEqual(pr["brier_skill_oracle"], 1 - pr["brier"] / pr["brier_baseline_oracle"], places=12)
            rep = RP.build_backtest_report(data, wf, h, n_boot=50)
            txt = rep["verdict"]["text"] + "".join(rep["verdict"]["bullets"]) + "".join(rep["notes"])
            self.assertIn("因果氣候基準", txt)
            self.assertIn("事後基準", txt)
            self.assertIn("含事後資訊", txt)

    def test_a_constant_forecaster_can_never_beat_the_oracle_reference(self):
        # exact identity: mean (p0 - y)^2 - mean (b - y)^2 = (p0 - b)^2  ->  oracle skill = -(p0 - b)^2 / (b (1 - b)) <= 0
        data, _, _ = H.make_planted_world(1700, 4, edge="none", h_list=(1,), first_idx=500)
        for h in (1, 5, 20):
            R = M.realized_log_returns(data.adjclose, h)
            E = np.arange(500, data.n - h)
            b = float(np.mean(R[E] > 0))
            for p0 in (0.3, 0.5, b, 0.62):
                wf = H.make_wf(data, {h: H.make_hf(data, h, 500, p0, 0.0, 0.015 * math.sqrt(h))}, 500)
                pr = RP.analyse(data, wf, h, with_strategy=False)["probabilistic"]
                self.assertLessEqual(pr["brier_skill_oracle"], 1e-12, (h, p0))
                self.assertAlmostEqual(pr["brier_skill_oracle"], -(p0 - b) ** 2 / (b * (1 - b)), places=12)
                self.assertLessEqual(pr["log_loss_skill_oracle"], 1e-12)

    def test_reference_uses_only_outcomes_matured_at_the_origin(self):
        # a report built on data truncated at `cut` (it cannot see anything later) must reproduce the Brier baseline that the
        # full-data loops give for the very same rows -> no future outcome enters the reference
        data, wf, _ = H.make_planted_world(1700, 6, edge="edge", signal_sd=0.004, h_list=(5,), first_idx=500)
        ix = indep(data, wf, 5)
        cut = 1300
        hf = wf.horizons[5]
        ds = data.slice(cut)
        wfs = H.make_wf(ds, {5: H.make_hf(ds, 5, 500, hf.p_up[:cut], hf.mu[:cut], hf.sigma[:cut])}, 500)
        a = RP.analyse(ds, wfs, 5, with_strategy=False)
        rows = [i for i in ix["rows"] if i <= cut - 1 - 5]
        ye = {i: ix["R"][i] > 0 for i in rows}
        ref = float(np.mean([(ix["ref"][i] - ye[i]) ** 2 for i in rows]))
        self.assertEqual(a["period"]["n_forecasts"], len(rows))
        self.assertAlmostEqual(a["probabilistic"]["brier_baseline"], ref, places=12)

    def test_zero_information_forecaster_size_of_the_brier_leg(self):
        # a forecaster that merely KNOWS the base rate (true P(up), constant) has no information beyond it: the Brier leg of the
        # verdict must not reject.  Against the (noisy) causal reference alone it rejects far above 5 % (audit S-01); against the
        # hindsight reference never; the verdict uses the larger p-value, so the leg is size-correct.
        rng = np.random.default_rng(2024)
        rej = {h: {"causal": 0, "oracle": 0, "max": 0} for h in (1, 5, 20)}
        skill = {h: [] for h in (1, 5, 20)}
        reps = 300
        for _ in range(reps):
            T = 500 + 1250 + 25
            cs = np.concatenate([[0.0], np.cumsum(0.0004 + 0.012 * rng.standard_normal(T))])
            for h in (1, 5, 20):
                R = np.full(T, np.nan)
                R[:T - h] = cs[h:T] - cs[:T - h]
                E = np.arange(500, T - h)
                y = (R[E] > 0).astype(float)
                p = np.full(len(E), float(stats.norm.cdf(0.0004 * h / (0.012 * math.sqrt(h)))))
                ref_c, ref_o = M.expanding_up_rate(R, h, ties="not_up")[E], np.full(len(E), y.mean())
                sq = (p - y) ** 2
                pc = M.p_one_sided_positive(*M.dm_test((ref_c - y) ** 2, sq, h))
                po = M.p_one_sided_positive(*M.dm_test((ref_o - y) ** 2, sq, h))
                rej[h]["causal"] += pc < 0.05
                rej[h]["oracle"] += po < 0.05
                rej[h]["max"] += max(pc, po) < 0.05
                skill[h].append(1 - sq.mean() / np.mean((ref_c - y) ** 2))
        for h, floor in ((1, 0.07), (5, 0.12), (20, 0.12)):
            self.assertGreater(rej[h]["causal"] / reps, floor, (h, rej[h]))           # the hazard of a causal-only test
            self.assertLessEqual(rej[h]["oracle"] / reps, 0.03, (h, rej[h]))
            self.assertLessEqual(rej[h]["max"] / reps, 0.02, (h, rej[h]))
            self.assertGreater(float(np.mean(skill[h])), 0.0, h)                        # positive 'skill' vs the noisy reference

    def test_evidence_exposes_both_p_values(self):
        data, wf, _ = H.make_planted_world(1700, 2, edge="edge", signal_sd=0.006, h_list=(1,), first_idx=500)
        a = RP.analyse(data, wf, 1, with_strategy=False)
        ev = a["evidence"]
        self.assertEqual(ev["p"]["brier"], max(ev["p_brier_causal"], ev["p_brier_oracle"]))
        self.assertLess(ev["p"]["brier"], 1e-3)                             # a real edge beats both references
        self.assertEqual(len(ev["brier_dm"]), 2)
        self.assertAlmostEqual(ev["effect"]["brier"], a["probabilistic"]["brier_skill"], places=12)


# ============================================================================= audit C-08 / S-07: conflicting evidence
class TestAdverseEvidence(unittest.TestCase):
    def test_adverse_legs_rules(self):
        self.assertEqual(RP.adverse_legs({"direction": 0.99, "brier": 0.2, "ic": 0.001}), ["direction"])      # hard
        self.assertEqual(RP.adverse_legs({"direction": 0.3, "brier": 0.96}), ["brier"])
        self.assertEqual(RP.adverse_legs({"direction": 0.97, "brier": 0.98}), ["direction", "brier"])
        self.assertEqual(RP.adverse_legs({"direction": 0.6, "brier": 0.7}), ["direction", "brier"])           # soft: neither helps
        self.assertEqual(RP.adverse_legs({"direction": 0.6, "brier": 0.4}), [])
        self.assertEqual(RP.adverse_legs({"direction": 0.6}), [])                                             # one leg only
        self.assertEqual(RP.adverse_legs({"direction": 0.01, "brier": 0.01, "ic": 0.5}), [])
        self.assertEqual(RP.adverse_legs({"ic": 0.99, "mse": 0.99}), [])                                      # mean legs never veto
        self.assertEqual(RP.adverse_legs({}), [])

    def lvl(self, p, **kw):
        return RP.assess_evidence(ev(p, **kw))

    def test_two_mean_legs_cannot_outvote_worse_than_baseline_probability_legs(self):
        # the three real h = 1 reports of the audit (2412, 3045, 4904): IC and MSE significant, direction / Brier not
        for p in ({"direction": 1.0, "brier": 0.959, "ic": 0.000036, "mse": 0.0061},
                  {"direction": 0.96, "brier": 0.69, "ic": 0.00001, "mse": 0.00075},
                  {"direction": 0.79, "brier": 0.69, "ic": 0.0009, "mse": 0.0024}):
            r = self.lvl(p)
            self.assertEqual((r["level"], r["reason"]), ("weak_edge", "conflicting_evidence"), p)
            self.assertTrue(r["conflict"])
            self.assertTrue(set(r["conflict"]) <= {"direction", "brier"})
            without = {k: v for k, v in p.items() if k in ("ic", "mse")}
            self.assertEqual(self.lvl(without)["level"], "moderate_edge")       # what the old rule would have said
        # a probability leg that merely is not significant (and not worse than the baseline) does not veto
        r = self.lvl({"direction": 0.3, "brier": 0.2, "ic": 0.0004, "mse": 0.002})
        self.assertEqual((r["level"], r["conflict"]), ("moderate_edge", []))
        # real edges (probability legs agree) are untouched
        self.assertEqual(self.lvl({"direction": 1e-5, "brier": 5e-4, "ic": 1e-4, "mse": 0.2})["level"], "moderate_edge")
        # the veto only lowers the level to weak_edge, never below, and never raises
        self.assertEqual(self.lvl({"direction": 0.99, "brier": 0.99, "ic": 0.3, "mse": 0.4})["level"], "no_edge")
        strong = {"direction": 0.97, "brier": 5e-4, "ic": 1e-4, "mse": 1e-5}
        self.assertEqual(self.lvl(strong, psr=0.99, dp=0.01)["level"], "weak_edge")     # would have been strong
        self.assertEqual(RP.assess_evidence(ev({"ic": 1e-5, "mse": 1e-6}))["level"], "moderate_edge")   # no probability legs

    def test_universe_evidence_has_no_probability_legs_and_is_unaffected(self):
        p = {"ic": 0.0005, "picks": 0.002, "spread": 0.2}
        r = RP.assess_evidence(ev(p), RP.THRESH_UNIVERSE)
        self.assertEqual((r["level"], r["conflict"]), ("moderate_edge", []))

    def test_end_to_end_bid_ask_bounce_world(self):
        # prices bounce between bid and ask (lag-1 autocorrelation about -0.2): a reversal feature forecasts the mean (IC and
        # MSE significant at 1 %), but the probabilistic forecast is the constant base rate (no direction skill)
        for seed in range(4):
            data, wf, ac1 = H.make_reversal_world(seed=seed)
            self.assertLess(ac1, -0.15)
            rep = RP.build_backtest_report(data, wf, 1, n_boot=50)
            a = RP.analyse(data, wf, 1, with_strategy=False)
            p = a["evidence"]["p"]
            self.assertLess(p["ic"], 0.01, (seed, p))
            self.assertLess(p["mse"], 0.01, (seed, p))                          # the old rule: moderate_edge
            self.assertGreater(p["brier"], 0.5)
            ass = RP.assess_evidence(a["evidence"])
            self.assertEqual((ass["level"], ass["reason"]), ("weak_edge", "conflicting_evidence"), (seed, p))
            v = rep["verdict"]
            self.assertEqual((v["level"], v["title"]), ("weak_edge", RP.LEVEL_TITLES["weak_edge"]))
            self.assertIn("互相矛盾", v["text"])
            self.assertIn("最小跳動單位", v["text"])
            self.assertIn("不宣稱有可用的預測優勢", v["text"])
            self.assertNotIn("統計證據相當一致", v["text"])
            self.assertEqual(RP.validate_backtest_report(rep), [])
            for s_ in [v["text"], *v["bullets"]]:
                self.assertFalse(SIMPLIFIED & set(s_), s_)

    def test_moderate_text_does_not_claim_consistency_when_only_mean_legs_confirm(self):
        data, wf, _ = H.make_planted_world(1700, 3, edge="none", h_list=(5,), first_idx=500)
        a = RP.analyse(data, wf, 5, n_boot=50)
        e = copy.deepcopy(a["evidence"])
        e["p"] = {"direction": 0.3, "brier": 0.2, "ic": 0.0004, "mse": 0.002}
        ass = RP.assess_evidence(e)
        self.assertEqual(ass["level"], "moderate_edge")
        t = RP.make_verdict(dict(a, evidence=e), ass, h=5, exec_mode="next_open")["text"]
        self.assertIn("同為預測均值的函數", t)
        self.assertNotIn("統計證據相當一致", t)
        self.assertIn(RP.BOUNCE_HINT, t)                      # the tick-size / bid-ask-bounce reading aid (audit C-08)
        self.assertTrue(t.rstrip().endswith("不構成投資建議。"))
        e2 = dict(copy.deepcopy(a["evidence"]), p={"direction": 0.0004, "brier": 0.002, "ic": 0.4, "mse": 0.5})
        t = RP.make_verdict(dict(a, evidence=e2), RP.assess_evidence(e2), h=5, exec_mode="next_open")["text"]
        self.assertIn("統計證據相當一致", t)
        self.assertNotIn("同為預測均值的函數", t)
        self.assertNotIn(RP.BOUNCE_HINT, t)                   # genuine edges (probability legs agree) get no such hint
        # weak edge carried by the two mean legs only (5 % level, probability legs silent): same reading aid
        e3 = dict(copy.deepcopy(a["evidence"]), p={"direction": 0.4, "brier": 0.3, "ic": 0.002, "mse": 0.03})
        ass3 = RP.assess_evidence(e3)
        self.assertEqual(ass3["level"], "weak_edge")
        self.assertIn(RP.BOUNCE_HINT, RP.make_verdict(dict(a, evidence=e3), ass3, h=5, exec_mode="next_open")["text"])
        e4 = dict(copy.deepcopy(a["evidence"]), p={"direction": 0.002, "brier": 0.4, "ic": 0.03, "mse": 0.5})
        ass4 = RP.assess_evidence(e4)
        self.assertEqual(ass4["level"], "weak_edge")
        self.assertNotIn(RP.BOUNCE_HINT, RP.make_verdict(dict(a, evidence=e4), ass4, h=5, exec_mode="next_open")["text"])
        self.assertTrue(RP.mean_legs_only({"p": {"ic": 0.001, "mse": 0.2}}, 0.01))
        self.assertFalse(RP.mean_legs_only({"p": {"ic": 0.001, "brier": 0.002}}, 0.01))
        self.assertFalse(RP.mean_legs_only({"p": {"ic": 0.5}}, 0.01))                  # nothing significant: not "mean only"

    def test_planted_edge_is_not_vetoed(self):
        for seed in range(4):
            data, wf, _ = H.make_planted_world(1700, seed, edge="edge", signal_sd=0.006, h_list=(1, 5), first_idx=500)
            for h in (1, 5):
                a = RP.analyse(data, wf, h, with_strategy=False)
                ass = RP.assess_evidence(a["evidence"])
                self.assertEqual(ass["conflict"], [], (seed, h))
                self.assertIn(ass["level"], ("moderate_edge", "strong_edge"), (seed, h))


# ============================================================================= audit S-03 / S-04 / S-02: honest wording
class TestEceWarning(unittest.TestCase):
    BULLET = "機率校準偏差偏大"

    def test_calibrated_models_do_not_get_the_warning_at_long_horizons(self):
        old_rule_would_warn = 0
        for seed in range(6):
            data, wf, _ = H.make_planted_world(1700, seed, edge="none", h_list=(5, 20), first_idx=500)
            for h in (5, 20):
                rep = RP.build_backtest_report(data, wf, h, n_boot=50)
                pr = rep["probabilistic"]
                old_rule_would_warn += pr["ece"] > 0.05                                   # the old fixed trigger
                self.assertFalse(any(self.BULLET in b for b in rep["verdict"]["bullets"]), (seed, h, pr["ece"]))
                self.assertGreater(pr["ece_noise_floor"], 0.05)
                self.assertAlmostEqual(pr["ece_noise_floor"], M.ece_noise_floor(rep["period"]["n_forecasts"], h, 10),
                                       places=12)
                self.assertTrue(any("純雜訊水準約" in b for b in rep["verdict"]["bullets"]))
        self.assertGreaterEqual(old_rule_would_warn, 9, old_rule_would_warn)               # the old rule flagged these calibrated models

    def test_a_clearly_miscalibrated_model_is_still_warned(self):
        data, _, _ = H.make_planted_world(1700, 2, edge="none", h_list=(1,), first_idx=500)
        for h, p0 in ((1, 0.85), (5, 0.97)):
            wf = H.make_wf(data, {h: H.make_hf(data, h, 500, p0, 0.0, 0.015 * math.sqrt(h))}, 500)
            rep = RP.build_backtest_report(data, wf, h, n_boot=50)
            pr = rep["probabilistic"]
            self.assertGreater(pr["ece"], pr["ece_noise_floor"])
            bullet = next(b for b in rep["verdict"]["bullets"] if self.BULLET in b)
            self.assertIn(f"{pr['ece'] * 100:.1f}%", bullet)
            self.assertIn(f"{pr['ece_noise_floor'] * 100:.1f}%", bullet)


class TestPowerDisclosureAndIcWording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.wf, _ = H.make_planted_world(1700, 3, edge="none", h_list=(1, 5, 20), first_idx=500)
        cls.reps = {h: RP.build_backtest_report(cls.data, cls.wf, h, n_boot=50) for h in (1, 5, 20)}

    def test_no_edge_text_states_what_could_not_have_been_detected(self):
        xs = []
        for h, rep in self.reports_items():
            self.assertEqual(rep["verdict"]["level"], "no_edge")
            n_scored = rep["period"]["n_scored"]
            x = M.min_detectable_ic(h, n_scored)
            xs.append(x)
            t = rep["verdict"]["text"]
            self.assertIn("並不證明沒有優勢", t)
            self.assertIn(f"{x:.2f}", t)
            self.assertIn(f"約 {M.effective_n(n_scored, h)} 筆獨立觀察", t)
            self.assertTrue(t.rstrip().endswith("不構成投資建議。"))                    # disclaimer stays last
            self.assertTrue(any("檢定力" in n and f"{x:.2f}" in n and "不等於「沒有優勢」" in n for n in rep["notes"]))
        self.assertLess(xs[0], xs[1])
        self.assertLess(xs[1], xs[2])                                                      # harder to detect at longer horizons

    def reports_items(self):
        return sorted(self.reps.items())

    def test_power_sentence_helper(self):
        self.assertEqual(RP.power_sentence({"n_scored": 0, "n_eff": 0}, 5), "")
        s = RP.power_sentence({"n_scored": 1250, "n_eff": 250}, 5)
        self.assertIn("0.25", s)
        self.assertIn("250", s)

    def test_edge_verdicts_do_not_carry_the_no_edge_disclosure(self):
        data, wf, _ = H.make_planted_world(1700, 3, edge="edge", signal_sd=0.01, h_list=(5,), first_idx=500)
        rep = RP.build_backtest_report(data, wf, 5, n_boot=100)
        self.assertIn(rep["verdict"]["level"], ("moderate_edge", "strong_edge"))
        self.assertNotIn("並不證明沒有優勢", rep["verdict"]["text"])

    def test_two_sided_ic_p_value_is_marked_uncalibrated(self):
        for rep in self.reps.values():
            self.assertIn("p 值未校準", rep["verdict"]["text"])
            self.assertTrue(any("p 值未校準僅供參考" in b and "虛無分配略偏負" in b for b in rep["verdict"]["bullets"]))


class TestPower(unittest.TestCase):
    """The detectable-IC table of ``metrics.MIN_DETECTABLE_IC`` (disclosure only) against planted-signal simulations of the
    verdict rule: at the tabulated IC the single-stock verdict leaves 'no_edge' in most runs, far below it rarely, and at
    h = 20 a moderate IC is mostly missed."""

    def rate(self, h, sd, seeds):
        hit, ics = 0, []
        for seed in seeds:
            data, wf, _ = H.make_planted_world(1700, 5000 + seed, edge="edge", signal_sd=sd, h_list=(h,), first_idx=500)
            a = RP.analyse(data, wf, h, with_strategy=False)
            hit += RP.assess_evidence(a["evidence"])["level"] != "no_edge"
            ics.append(a["returns"]["ic"])
        return hit / len(seeds), float(np.mean(ics))

    def test_detection_rates_match_the_disclosed_table(self):
        seeds = range(30)
        r1, ic1 = self.rate(1, 0.002, seeds)               # IC about 0.12 = MIN_DETECTABLE_IC[1]: ~80 % power
        self.assertAlmostEqual(ic1, M.MIN_DETECTABLE_IC[1], delta=0.03)
        self.assertGreaterEqual(r1, 0.55)
        lo1, _ = self.rate(1, 0.0005, seeds)               # IC about 0.03: invisible
        self.assertLessEqual(lo1, 0.30)
        r20, ic20 = self.rate(20, 0.003, seeds)            # IC about 0.29: the rule misses it most of the time at h = 20
        self.assertGreater(ic20, 0.2)
        self.assertLess(ic20, M.MIN_DETECTABLE_IC[20])
        self.assertLessEqual(r20, 0.55)
        big20, ic_big = self.rate(20, 0.009, seeds)        # IC about 0.49 > MIN_DETECTABLE_IC[20]
        self.assertGreater(ic_big, M.MIN_DETECTABLE_IC[20])
        self.assertGreaterEqual(big20, 0.8)


# ============================================================================= schema: additive keys
class TestAdditiveSchemaKeys(unittest.TestCase):
    NEW = ("log_loss_skill", "brier_baseline_oracle", "brier_skill_oracle", "log_loss_baseline_oracle",
           "log_loss_skill_oracle", "ece_noise_floor")

    @classmethod
    def setUpClass(cls):
        data, wf, _ = planted(4, "edge", n=1200, h_list=(1,), first_idx=500)
        cls.rep = RP.build_backtest_report(data, wf, 1, n_boot=50)

    def test_new_probabilistic_keys_are_present_and_numeric(self):
        pr = self.rep["probabilistic"]
        for k in self.NEW:
            self.assertIn(k, pr)
            self.assertTrue(pr[k] is None or isinstance(pr[k], float), k)
        self.assertEqual(RP.validate_backtest_report(self.rep), [])

    def test_old_reports_without_the_new_keys_still_validate(self):
        r = copy.deepcopy(self.rep)
        for k in self.NEW:
            del r["probabilistic"][k]
        self.assertEqual(RP.validate_backtest_report(r), [])

    def test_new_keys_are_type_checked(self):
        for k in self.NEW:
            r = copy.deepcopy(self.rep)
            r["probabilistic"][k] = "x"
            probs = RP.validate_backtest_report(r)
            self.assertTrue(any(k in p for p in probs), (k, probs))
            r["probabilistic"][k] = None                                  # null is allowed (undefined skill)
            self.assertEqual(RP.validate_backtest_report(r), [])

    def test_event_keys(self):
        r = copy.deepcopy(self.rep)
        self.assertIn(r["probabilistic"]["p_up_event"], RP.P_UP_EVENTS)
        self.assertEqual(r["probabilistic"]["n"], r["period"]["n_forecasts"])        # undeclared event: every evaluated row
        r["probabilistic"]["p_up_event"] = "maybe"
        self.assertTrue(any("p_up_event" in p for p in RP.validate_backtest_report(r)))
        r["probabilistic"]["p_up_event"] = "up"
        r["probabilistic"]["n"] = 1.5
        self.assertTrue(any("probabilistic.n" in p for p in RP.validate_backtest_report(r)))
        del r["probabilistic"]["n"], r["probabilistic"]["p_up_event"]
        self.assertEqual(RP.validate_backtest_report(r), [])

    def test_check_schema_optional_keys_and_bool(self):
        probs = []
        RP.check_schema({"a": "int", "b?": "num?", "c?": "bool"}, {"a": 1}, "", probs)
        self.assertEqual(probs, [])
        RP.check_schema({"a": "int", "b?": "num?", "c?": "bool"}, {"a": 1, "b": "x", "c": 1}, "", probs)
        self.assertEqual(len(probs), 2, probs)
        probs = []
        RP.check_schema({"a": "int"}, {}, "", probs)
        self.assertEqual(probs, ["a: missing key"])


if __name__ == "__main__":
    unittest.main()
