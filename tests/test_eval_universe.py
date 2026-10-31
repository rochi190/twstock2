"""Tests for backend/evaluation/universe_bt.py."""
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
from backend.evaluation import strategy as S  # noqa: E402
from backend.evaluation import universe_bt as U  # noqa: E402

SIMPLIFIED = set("预统显优样获检准关买损较约应错该为实时间点发线图号体义观验览态质总结况构运动过区务强价绩费记录别难势报")
FORBIDDEN = ("保證獲利", "穩賺", "必賺", "一定賺", "包賺", "穩賠不賠", "保證賺", "穩定獲利")


# ----------------------------------------------------------------------------- hand-computable micro universe
def tiny_stock(code, closes, kind="stock"):
    """Open = previous close (no overnight gap): co = 0 and oc = cc."""
    c = np.array(closes, dtype=float)
    return H.make_tiny_data(np.concatenate([[c[0]], c[:-1]]), c, symbol=code, kind=kind)


CLOSES = {"A": [100, 100, 100, 102, 102, 104.04, 104.04],
          "B": [50, 50, 50, 50, 51, 51, 52.02],
          "C": [20, 20, 20, 19.6, 19.6, 20.384, 20.384]}
# forecast mean of the h=1 log return per origin row 2..6 (sigma 0.01 for all -> ranking == ranking of mu)
MUS = {"A": [0.003, 0.001, 0.001, 0.000, 0.009], "B": [0.002, 0.004, 0.003, 0.001, 0.001],
       "C": [0.001, 0.002, 0.000, 0.005, 0.002]}
FIRST = 2


def micro_panel(h=1, kinds=None):
    panel = {}
    for code in "ABC":
        d = tiny_stock(code, CLOSES[code], (kinds or {}).get(code, "stock"))
        mu = np.concatenate([[np.nan] * FIRST, MUS[code]])
        hf = H.make_hf(d, h, FIRST, 0.5, mu, 0.01)
        panel[code] = (d, H.make_wf(d, {h: hf}, FIRST))
    return panel


def legs(code):
    c = np.array(CLOSES[code], dtype=float)
    cc = np.zeros(len(c))
    cc[1:] = c[1:] / c[:-1] - 1
    return cc


class TestMicroUniverse(unittest.TestCase):
    """h = 1, top_k = 1, zero commission/slippage (only the 0.3 % sell tax remains), hand-derived expectations."""

    @classmethod
    def setUpClass(cls):
        cls.rep = U.evaluate_universe(micro_panel(), 1, top_k=1, commission_discount=0.0, slippage_bps=0.0, n_boot=50)

    def expected_returns(self, mode):
        picks = ["A", "B", "B", "C"]        # best mu on origin rows 2, 3, 4, 5
        sell = config.TAX_STOCK
        w_prev = {"A": 0.0, "B": 0.0, "C": 0.0}
        w_prev2 = dict(w_prev)
        rets = []
        for k, code in enumerate(picks):
            d = 3 + k                                   # day earning the return
            w1 = {c: 1.0 if c == code else 0.0 for c in "ABC"}
            w2 = dict(w_prev)
            cost = sum(sell * max(w2[c] - w1[c], 0.0) for c in "ABC")      # buys are free here
            cc = {c: legs(c)[d] for c in "ABC"}
            if mode == "close":
                g = sum(w1[c] * cc[c] for c in "ABC")
                rets.append((1 - cost) * (1 + g) - 1)
            else:                                        # open = previous close -> co = 0, oc = cc
                g_oc = sum(w1[c] * cc[c] for c in "ABC")
                rets.append((1 + 0.0) * (1 - cost) * (1 + g_oc) - 1)
            w_prev = w1
        return np.array(rets)

    def test_curve_matches_hand_simulation_next_open(self):
        ret = self.expected_returns("next_open")
        eq = np.concatenate([[1.0], np.cumprod(1 + ret)])
        np.testing.assert_allclose(self.rep["curve"]["top"], eq, atol=1e-6)
        self.assertEqual(self.rep["period"]["n_days"], 4)
        np.testing.assert_allclose(self.rep["top"]["total_return"], eq[-1] - 1, atol=1e-6)
        # day 4: A is sold at the open (cost 0.3 %) and B bought: (1 - 0.003) * (1 + 0.02) - 1
        self.assertAlmostEqual(ret[1], (1 - 0.003) * 1.02 - 1)

    def test_curve_matches_hand_simulation_close(self):
        rep = U.evaluate_universe(micro_panel(), 1, top_k=1, commission_discount=0.0, slippage_bps=0.0,
                                  exec_mode="close", n_boot=50)
        eq = np.concatenate([[1.0], np.cumprod(1 + self.expected_returns("close"))])
        np.testing.assert_allclose(rep["curve"]["top"], eq, atol=1e-6)

    def test_equal_weight_benchmark(self):
        ew = np.array([np.mean([legs(c)[d] for c in "ABC"]) for d in range(3, 7)])
        eq = np.concatenate([[1.0], np.cumprod(1 + ew)])
        np.testing.assert_allclose(self.rep["curve"]["equal_weight"], eq, atol=1e-6)
        self.assertAlmostEqual(self.rep["benchmark_equal_weight"]["total_return"], eq[-1] - 1, places=6)
        self.assertIsNone(self.rep["curve"]["taiex"])
        self.assertIsNone(self.rep["benchmark_taiex"])

    def test_dates_and_period(self):
        d = micro_panel()["A"][0].dates
        self.assertEqual(self.rep["curve"]["dates"], [str(x) for x in d[FIRST:]])
        self.assertEqual(self.rep["period"], {"start": str(d[FIRST]), "end": str(d[-1]), "n_days": 4, "n_stocks": 3})

    def test_ic_series_hand(self):
        ics = []
        for k in range(4):                       # origin rows 2..5
            sc = [MUS[c][k] for c in "ABC"]
            R = [math.log(CLOSES[c][FIRST + k + 1] / CLOSES[c][FIRST + k]) for c in "ABC"]
            ics.append(stats.spearmanr(sc, R).statistic)
        self.assertAlmostEqual(ics[0], 1.0)
        self.assertAlmostEqual(ics[1], 0.8660254, places=6)
        self.assertAlmostEqual(self.rep["ic"]["mean"], float(np.mean(ics)), places=9)
        n_eff = 4.0
        t = np.mean(ics) / (np.std(ics, ddof=1) / math.sqrt(n_eff))
        self.assertAlmostEqual(self.rep["ic"]["t"], t, places=9)
        self.assertAlmostEqual(self.rep["ic"]["ir"], np.mean(ics) / np.std(ics, ddof=1), places=9)
        self.assertAlmostEqual(self.rep["ic"]["pvalue"], 2 * stats.t.sf(abs(t), n_eff - 1), places=9)

    def test_picks_hand(self):
        pk = self.rep["picks"]
        # picks A (+2 %), B (+2 %), B (0 -> tie), C (0 -> tie): two decisive picks, both up
        self.assertEqual(pk["n"], 2)
        self.assertAlmostEqual(pk["hit_rate"], 1.0)
        # same-day universe up-rates among non-zero outcomes: day of row 2: A up, B flat, C down -> 1/2 ; row 3: only B -> 1
        self.assertAlmostEqual(pk["base_rate"], (0.5 + 1.0) / 2)
        self.assertAlmostEqual(pk["mean_return"], np.mean([0.02, 0.02, 0.0, 0.0]), places=12)

    def test_trade_stats_count_picks_net_of_cost(self):
        top = self.rep["top"]
        self.assertEqual(top["n_trades"], 4)
        self.assertAlmostEqual(top["trade_hit_rate"], 0.5)          # +2 %, +2 % beat the 0.3 % cost; the two 0 % picks do not

    def test_spread_hand(self):
        # top = best mu, bottom = lowest mu (k = min(1, 3 // 2) = 1).  CHANGED (audit S-05): the spread is the mean of h-bar LOG
        # returns, not of simple returns, so that a ranking on volatility does not earn the Jensen term sigma^2 / 2 for free.
        sp, sp_simple = [], []
        for k in range(4):
            mus = {c: MUS[c][k] for c in "ABC"}
            best, worst = max(mus, key=mus.get), min(mus, key=mus.get)
            lr = lambda c: math.log(CLOSES[c][FIRST + k + 1] / CLOSES[c][FIRST + k])
            sr = lambda c: CLOSES[c][FIRST + k + 1] / CLOSES[c][FIRST + k] - 1
            sp.append(lr(best) - lr(worst))
            sp_simple.append(sr(best) - sr(worst))
        self.assertAlmostEqual(self.rep["spread"]["mean"], float(np.mean(sp)), places=12)
        self.assertNotAlmostEqual(float(np.mean(sp)), float(np.mean(sp_simple)), places=5)     # the two definitions do differ

    def test_per_stock_hand(self):
        rows = {r["code"]: r for r in self.rep["per_stock"]}
        self.assertEqual(set(rows), {"A", "B", "C"})
        for c in "ABC":
            R = np.array([math.log(CLOSES[c][FIRST + k + 1] / CLOSES[c][FIRST + k]) for k in range(4)])
            nz = R != 0
            self.assertEqual(rows[c]["n"], int(nz.sum()))
        self.assertEqual([rows[c]["n"] for c in "ABC"], [2, 2, 2])      # two non-flat outcomes each
        self.assertEqual(rows["A"]["name"], "微型")
        self.assertAlmostEqual(rows["A"]["accuracy"], 1.0 if False else rows["A"]["accuracy"])

    def test_valid_and_no_edge(self):
        self.assertEqual(U.validate_universe_report(self.rep), [])
        self.assertEqual(self.rep["verdict"]["level"], "no_edge")
        self.assertIn("存活者偏差", self.rep["notes"][0])

    def test_sigma_benchmark_and_basket_hand(self):
        # all names have sigma 0.01 -> the top-by-sigma basket is decided by the tie-breaks (p_up equal, then code): always 'A'.
        # Buy A at the open of row 3 (open = previous close, no costs): +2 %, 0, +2 %, 0 over the 4 return days.
        sg = self.rep["benchmark_top_sigma"]
        self.assertAlmostEqual(sg["total_return"], 1.02 * 1.02 - 1, places=9)
        self.assertEqual(len(self.rep["curve"]["top_sigma"]), len(self.rep["curve"]["dates"]))
        self.assertAlmostEqual(self.rep["curve"]["top_sigma"][-1], 1.02 * 1.02, places=6)
        bk = self.rep["basket"]
        self.assertAlmostEqual(bk["mean_sigma"], 0.01)
        self.assertAlmostEqual(bk["universe_mean_sigma"], 0.01)
        self.assertAlmostEqual(bk["sigma_ratio"], 1.0)
        self.assertIsNone(bk["mean_beta"])                                  # fewer than 60 observations: no beta yet
        self.assertEqual(bk["beta_market"], "equal_weight")
        # name-days whose expected net return covers the 0.3 % sell tax (the only cost left): A 0.003, 0.009; B 0.004, 0.003;
        # C 0.005  ->  5 of 15 (mu + sigma^2 / 2 > 0.003 needs mu >= 0.003)
        self.assertAlmostEqual(bk["share_net_positive"], 5 / 15)
        # picks on the five forecast days: A, B, B, C, A -> the day-to-day overlap of a top-1 list: only B -> B, i.e. 1 of 4
        self.assertAlmostEqual(bk["topk_overlap"], 1 / 4)


class TestTrancheLogic(unittest.TestCase):
    def pn(self, h, T=8, S_=3):
        pn = U._Panel.__new__(U._Panel)
        pn.h, pn.T, pn.S = h, T, S_
        return pn

    def test_overlapping_tranche_average(self):
        P = np.zeros((6, 3))
        for t, j in enumerate([0, 1, 2, 0, 1, 2]):
            P[t, j] = 1.0
        W, t0 = self.pn(3, 6).tranche_weights(P)
        self.assertEqual(t0, 0)
        np.testing.assert_allclose(W[0], [1, 0, 0])
        np.testing.assert_allclose(W[1], [0.5, 0.5, 0])                     # expanding start: fully invested
        np.testing.assert_allclose(W[2], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(W[3], (P[1] + P[2] + P[3]) / 3)         # window of the last h = 3 tranches
        np.testing.assert_allclose(W[5], (P[3] + P[4] + P[5]) / 3)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)

    def test_h1_is_the_pick_vector_and_late_start(self):
        P = np.zeros((6, 2))
        P[2:, 0] = 1.0
        W, t0 = self.pn(1, 6, 2).tranche_weights(P)
        self.assertEqual(t0, 2)
        np.testing.assert_allclose(W, P)
        W, t0 = self.pn(4, 6, 2).tranche_weights(np.zeros((6, 2)))
        self.assertIsNone(t0)
        self.assertEqual(W.sum(), 0)

    def test_gap_day_lowers_invested_fraction(self):
        P = np.zeros((5, 2))
        P[0, 0] = P[1, 1] = P[3, 0] = P[4, 1] = 1.0           # day 2 has no pick
        W, _ = self.pn(2, 5, 2).tranche_weights(P)
        np.testing.assert_allclose(W[2], [0, 0.5])             # window {day1, day2}: half invested
        np.testing.assert_allclose(W[3], [0.5, 0.0])


class TestPicksAndRanking(unittest.TestCase):
    def test_top_k_by_rank_signal_with_deterministic_ties(self):
        # CHANGED (DESIGN.md 9.1, audit C-06 / S-05): the picks rank by rank_signal = mu / sigma (the key of the picks page),
        # not by the cost-adjusted score (which ranks volatility); ties are broken by p_up and then by code order.
        panel = H.make_panel(10, 600, 3, edge="none", h_list=(1,), first_idx=300)   # constant forecasts per stock
        pn = U._Panel(panel, 1, 0.6, 5.0)
        P = pn.picks(3)
        self.assertTrue(np.all(np.isclose(P.sum(axis=1)[pn.avail.any(axis=1)], 1.0)))
        self.assertTrue(np.all((P > 0).sum(axis=1)[pn.avail.any(axis=1)] == 3))
        row = pn.T - 1
        np.testing.assert_allclose(pn.RS[row][pn.avail[row]], (pn.MU[row] / pn.SG[row])[pn.avail[row]])
        best = np.argsort(-pn.RS[row], kind="stable")[:3]
        self.assertEqual(sorted(np.flatnonzero(P[row] > 0)), sorted(best))
        # identical keys -> the higher p_up wins, then the lowest code index
        pn.RS[:] = np.where(pn.avail, 0.5, np.nan)
        pn.PU[:] = np.where(pn.avail, 0.6, np.nan)
        self.assertEqual(np.flatnonzero(pn.picks(2)[row] > 0).tolist(), [0, 1])
        pn.PU[row, 5] = 0.9
        self.assertEqual(np.flatnonzero(pn.picks(2)[row] > 0).tolist(), [0, 5])
        self.assertEqual(np.flatnonzero(pn.bottoms(2)[row]).tolist(), [0, 1])      # bottom: lowest key, then lowest p_up

    def test_ranking_key_can_be_replaced_for_the_model_free_benchmarks(self):
        panel = H.make_panel(10, 600, 3, edge="none", h_list=(1,), first_idx=300)
        pn = U._Panel(panel, 1, 0.6, 5.0)
        row = pn.T - 1
        by_sigma = pn.picks(3, key=pn.SG)
        np.testing.assert_array_equal(np.flatnonzero(by_sigma[row] > 0), sorted(np.argsort(-pn.SG[row], kind="stable")[:3]))
        by_score = pn.picks(3, key=pn.SC)
        np.testing.assert_array_equal(np.flatnonzero(by_score[row] > 0), sorted(np.argsort(-pn.SC[row], kind="stable")[:3]))
        self.assertFalse(np.array_equal(by_score, pn.picks(3)))                     # the old key picks different names
        # only_buy still gates on p_up and net > 0 whatever the key is
        picked = pn.picks(9, only_buy=True, buy_p=0.5, key=pn.SG) > 0
        self.assertTrue(np.all(pn.net[picked] > 0))

    def test_top_k_larger_than_universe_and_buy_only(self):
        panel = H.make_panel(4, 500, 4, edge="edge", h_list=(1,), first_idx=300)
        pn = U._Panel(panel, 1, 0.6, 5.0)
        P = pn.picks(9)
        np.testing.assert_allclose(P[pn.avail.any(axis=1)].sum(axis=1), 1.0)
        Pb = pn.picks(9, only_buy=True, buy_p=0.55)
        picked = Pb > 0
        self.assertTrue(np.all(pn.PU[picked] >= 0.55))
        self.assertTrue(np.all(pn.net[picked] > 0))
        self.assertLess(picked.sum(), (P > 0).sum())


def ref_hac(x, h):
    """Textbook HAC t-statistic of a mean with explicit loops: Bartlett lag min(max(2h, ceil(3 n^(1/3))), n//5),
    factor n/(n-1), Student-t df = max(2, n/(L+1) - 1).  Returns (t, df, L)."""
    x = [float(v) for v in x]
    n = len(x)
    L = min(max(2 * h, math.ceil(3 * n ** (1 / 3))), n // 5)
    m = sum(x) / n
    d = [v - m for v in x]
    g = [sum(d[t] * d[t - k] for t in range(k, n)) / n for k in range(L + 1)]
    lrv = (g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))) * n / (n - 1)
    return m / math.sqrt(lrv / n), max(2.0, n / (L + 1) - 1), L


def indep_universe_stats(panel, h, top_k):
    """Plain-loop recomputation of the IC / picks statistics from the raw walk-forward arrays.

    The ranking key is ``rank_signal = mu / sigma`` (log-return information ratio, DESIGN.md 9.1), ties broken by p_up and
    then by code order -- NOT the cost-adjusted score (which ranks volatility)."""
    codes = sorted(panel)
    n = panel[codes[0]][0].n
    first = panel[codes[0]][1].first_idx
    ics, hits, bases, nks, rets = [], [], [], [], []
    min_xs = min(max(6, 2 * top_k), len(codes))
    for t in range(first, n - h):
        sc, R, pu = {}, {}, {}
        for c in codes:
            hf = panel[c][1].horizons[h]
            mu, sg = hf.mu[t], hf.sigma[t]
            sc[c] = mu / sg
            pu[c] = hf.p_up[t]
            R[c] = hf.realized[t]
        if len(codes) >= min_xs:
            ics.append(stats.spearmanr([sc[c] for c in codes], [R[c] for c in codes]).statistic)
        order = sorted(codes, key=lambda c: (-sc[c], -pu[c], codes.index(c)))[:top_k]
        nz = [c for c in order if R[c] != 0]
        allnz = [c for c in codes if R[c] != 0]
        if nz and allnz:
            hits.append(sum(R[c] > 0 for c in nz) / len(nz))
            bases.append(sum(R[c] > 0 for c in allnz) / len(allnz))
            nks.append(len(nz))
        rets.extend(math.expm1(R[c]) for c in order)
    return {"ic": ics, "hit": hits, "base": bases, "nk": nks, "ret": rets}


class TestUniverseNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = H.make_panel(12, 800, 5, edge="edge", h_list=(1, 5), first_idx=300)
        cls.reps = {h: U.evaluate_universe(cls.panel, h, top_k=4, n_boot=100) for h in (1, 5)}

    def test_schema_valid(self):
        for h, rep in self.reps.items():
            self.assertEqual(U.validate_universe_report(rep), [], h)
            json.dumps(rep, allow_nan=False)
            self.assertEqual((rep["horizon"], rep["top_k"]), (h, 4))

    def test_ic_matches_plain_loops(self):
        for h, rep in self.reps.items():
            ref = indep_universe_stats(self.panel, h, 4)
            ics = np.array(ref["ic"])
            self.assertAlmostEqual(rep["ic"]["mean"], ics.mean(), places=10)
            self.assertAlmostEqual(rep["ic"]["ir"], ics.mean() / ics.std(ddof=1), places=10)
            t, df, L = ref_hac(ics, h)
            self.assertAlmostEqual(rep["ic"]["t"], t, places=9)
            self.assertAlmostEqual(rep["ic"]["pvalue"], 2 * stats.t.sf(abs(t), df), places=9)
            self.assertEqual(U.hac_lag(len(ics), h), L)

    def test_picks_match_plain_loops(self):
        for h, rep in self.reps.items():
            ref = indep_universe_stats(self.panel, h, 4)
            w = np.array(ref["nk"], dtype=float)
            hit, base = np.array(ref["hit"]), np.array(ref["base"])
            pk = rep["picks"]
            self.assertEqual(pk["n"], int(w.sum()))
            self.assertAlmostEqual(pk["hit_rate"], float((hit * w).sum() / w.sum()), places=10)
            self.assertAlmostEqual(pk["base_rate"], float((base * w).sum() / w.sum()), places=10)
            t, df, _ = ref_hac(hit - base, h)
            self.assertAlmostEqual(pk["p_value"], stats.t.sf(t, df), places=9)
            self.assertAlmostEqual(pk["mean_return"], float(np.mean(ref["ret"])), places=10)

    def test_per_stock_matches(self):
        rep = self.reps[5]
        for row in rep["per_stock"]:
            d, wf = self.panel[row["code"]]
            hf = wf.horizons[5]
            ev = [i for i in range(wf.first_idx, d.n - 5)]                      # every evaluated forecast
            idx = [i for i in ev if hf.realized[i] != 0]                         # direction: non-flat outcomes only
            p = np.array([hf.p_up[i] for i in idx])
            up = np.array([hf.realized[i] > 0 for i in idx])
            self.assertEqual(row["n"], len(idx))
            self.assertAlmostEqual(row["accuracy"], float(((p >= 0.5) == up).mean()), places=12)
            # CHANGED (audit C-01 / C-02): Brier skill of P(R > 0) on every evaluated row (flat = not up) against the CAUSAL
            # expanding base rate of the stock (independent loops over the outcomes matured at each origin); the old
            # test-sample reference b (1 - b) is now `brier_skill_oracle`
            pe = np.array([hf.p_up[i] for i in ev])
            y = np.array([hf.realized[i] > 0 for i in ev], dtype=float)
            ref = []
            for i in ev:
                js = [j for j in range(0, i - 5 + 1) if np.isfinite(hf.realized[j])]
                ref.append((sum(hf.realized[j] > 0 for j in js) + 1) / (len(js) + 2))
            ref = np.array(ref)
            self.assertAlmostEqual(row["brier_skill"], 1 - np.mean((pe - y) ** 2) / np.mean((ref - y) ** 2), places=12)
            b = y.mean()
            self.assertAlmostEqual(row["brier_skill_oracle"], 1 - np.mean((pe - y) ** 2) / (b * (1 - b)), places=12)
            rho = stats.spearmanr([hf.mu[i] for i in ev], [hf.realized[i] for i in ev]).statistic
            self.assertAlmostEqual(row["ic"], rho, places=10)
        self.assertEqual([r["code"] for r in rep["per_stock"]], sorted(self.panel))

    def test_curves_and_stats_consistent(self):
        rep = self.reps[5]
        c = rep["curve"]
        self.assertEqual(len(c["dates"]), rep["period"]["n_days"] + 1)
        self.assertAlmostEqual(c["top"][0], 1.0)
        self.assertAlmostEqual(c["top"][-1] - 1, rep["top"]["total_return"], places=5)
        self.assertAlmostEqual(c["equal_weight"][-1] - 1, rep["benchmark_equal_weight"]["total_return"], places=5)
        self.assertEqual(rep["benchmark_equal_weight"]["n_trades"], 1)
        self.assertTrue(0 < rep["top"]["exposure"] <= 1.0 + 1e-9)
        self.assertGreater(rep["top"]["turnover_per_year"], 0)
        self.assertGreater(rep["top"]["sharpe"], rep["benchmark_equal_weight"]["sharpe"])   # planted edge

    def test_planted_edge_verdict(self):
        for h, rep in self.reps.items():
            self.assertIn(rep["verdict"]["level"], ("moderate_edge", "strong_edge"), h)
            self.assertGreater(rep["ic"]["mean"], 0.1)
            self.assertLess(rep["ic"]["pvalue"], 1e-6)
            self.assertGreater(rep["picks"]["hit_rate"], rep["picks"]["base_rate"] + 0.03)
            self.assertGreater(rep["spread"]["mean"], 0)

    def test_hac_is_more_conservative_than_naive_for_persistent_series(self):
        for h in (1, 5):
            rep = self.reps[h]
            ics = np.array(indep_universe_stats(self.panel, h, 4)["ic"])
            naive_t = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))
            self.assertLess(rep["ic"]["t"], naive_t)            # planted AR(1) signals -> positively autocorrelated IC
            self.assertGreater(rep["ic"]["t"], 0.2 * naive_t)
        # the HAC lag grows with the horizon and is capped at n // 5
        self.assertEqual((U.hac_lag(1000, 1), U.hac_lag(1000, 5), U.hac_lag(1000, 20)), (30, 30, 40))
        self.assertEqual(U.hac_lag(4, 1), 0)
        self.assertEqual(U.hac_lag(50, 20), 10)
        self.assertEqual(U.hac_lag(0, 3), 0)

    def test_t_stats_lag_zero_is_textbook_t_test(self):
        x = np.random.default_rng(1).normal(0.1, 1.0, 4)               # n // 5 == 0 -> lag 0
        r = U._t_stats(x, 1)
        ref = stats.ttest_1samp(x, 0.0)
        self.assertAlmostEqual(r["t"], ref.statistic, places=12)
        self.assertAlmostEqual(r["p_two"], ref.pvalue, places=12)
        self.assertAlmostEqual(r["p_one"], ref.pvalue / 2 if ref.statistic > 0 else 1 - ref.pvalue / 2, places=12)
        e = U._t_stats([0.1, np.nan], 1)
        self.assertEqual((e["n"], e["t"]), (1, None))
        self.assertIsNone(U._t_stats([0.3] * 30, 1)["t"])            # no variance -> undefined

    def test_ic_series_is_rolling_mean(self):
        rep = self.reps[1]
        ics = np.array(indep_universe_stats(self.panel, 1, 4)["ic"])
        win = int(min(60, max(1, len(ics) // 4)))
        roll = np.convolve(ics, np.ones(win) / win, mode="valid")
        ser = rep["ic"]["series"]
        keep = M.downsample_indices(len(roll))
        np.testing.assert_allclose(ser["values"], roll[keep], atol=1e-5)
        self.assertEqual(len(ser["dates"]), len(ser["values"]))
        self.assertTrue(any(f"{win} 日滾動平均" in n for n in rep["notes"]))


# ============================================================================= DESIGN.md 9.1: rank_signal picks, exposure block, benchmarks
def daily_spearman(A, B, mask, min_n=6):
    """Mean over days of the cross-sectional Spearman correlation of two (T, S) matrices on the masked entries."""
    vals = [stats.spearmanr(A[t, mask[t]], B[t, mask[t]]).statistic for t in range(A.shape[0]) if mask[t].sum() >= min_n]
    return float(np.mean(vals))


class TestRankSignalPicks(unittest.TestCase):
    """Audit C-06 / S-05 / P-05: the validated top-k must be a rank_signal basket (not a volatility basket), and the report
    must publish what a volatility / beta tilt looks like next to it."""

    @classmethod
    def setUpClass(cls):
        # no-skill universe: volatility 0.5 % .. 4 % a day, high-vol names are high-beta, the view is independent of volatility
        cls.panel = H.make_vol_panel(16, 900, 3, h_list=(1, 5), vol_lo=0.005, vol_hi=0.04, beta_slope=30.0)
        cls.pn = {h: U._Panel(cls.panel, h, 0.6, 5.0) for h in (1, 5)}
        cls.rep = {h: U.evaluate_universe(cls.panel, h, top_k=3, n_boot=100) for h in (1, 5)}

    def test_premise_the_cost_adjusted_score_is_a_volatility_key(self):
        pn = self.pn[1]
        self.assertEqual(float(np.mean(pn.net[pn.avail] > 0)), 0.0)                         # h = 1: no name covers the cost
        for h, pn in self.pn.items():
            self.assertGreater(daily_spearman(pn.SC, pn.SG, pn.avail), 0.9, h)             # the artefact of the old key
            self.assertLess(abs(daily_spearman(pn.RS, pn.SG, pn.avail)), 0.3, h)           # rank_signal: not a volatility key

    def test_rank_signal_basket_is_not_a_volatility_basket(self):
        for h, pn in self.pn.items():
            t0 = int(np.argmax(pn.avail.any(axis=1)))
            uni = U.universe_exposure(pn, None, t0)["mean_sigma"]

            def ratio(P):
                return U.exposure_stats(pn, P, None, t0)["mean_sigma"] / uni
            r_rs, r_sc, r_sg = ratio(pn.picks(3)), ratio(pn.picks(3, key=pn.SC)), ratio(pn.picks(3, key=pn.SG))
            self.assertLess(abs(r_rs - 1), 0.4, (h, r_rs))
            self.assertGreater(r_sc, 1.5, (h, r_sc))                 # picks by the old score = the most volatile names
            self.assertGreaterEqual(r_sg, r_sc - 1e-9)
            self.assertGreater(r_sg, 1.5)

    def test_report_publishes_exposure_next_to_the_universe(self):
        for h, rep in self.rep.items():
            self.assertEqual(U.validate_universe_report(rep), [], h)
            bk, ex = rep["basket"], rep["exposure"]
            self.assertEqual((rep["rank_key"], rep["only_buy_signals"], bk["k"]), ("rank_signal", False, 3))
            self.assertEqual(set(rep["labels"]), {"top", "benchmark_equal_weight", "benchmark_top_sigma", "benchmark_taiex"})
            self.assertIn("前 3 名", rep["labels"]["top"])
            self.assertLess(abs(bk["sigma_ratio"] - 1), 0.4)
            self.assertAlmostEqual(bk["sigma_ratio"], bk["mean_sigma"] / bk["universe_mean_sigma"], places=12)
            self.assertGreater(bk["sigma_basket"]["mean_sigma"], 1.5 * bk["universe_mean_sigma"])
            self.assertGreater(bk["sigma_basket"]["mean_beta"], bk["universe_mean_beta"])   # high vol <-> high beta here
            self.assertEqual(bk["beta_market"], "equal_weight")                                # no TAIEX supplied
            self.assertLess(abs(bk["corr_rank_sigma"]), 0.3)
            self.assertTrue(0.0 <= bk["topk_overlap"] <= 1.0)
            self.assertEqual(ex["mean_sigma_top"], bk["mean_sigma"])
            self.assertEqual(ex["mean_sigma_universe"], bk["universe_mean_sigma"])
            self.assertEqual((ex["beta_top"], ex["beta_universe"]), (bk["mean_beta"], bk["universe_mean_beta"]))
            self.assertEqual(ex["benchmark_top_sigma"], rep["benchmark_top_sigma"])
            self.assertEqual(len(rep["curve"]["top_sigma"]), len(rep["curve"]["dates"]))
            self.assertAlmostEqual(rep["curve"]["top_sigma"][-1] - 1, rep["benchmark_top_sigma"]["total_return"], places=5)
        self.assertEqual(self.rep[1]["basket"]["share_net_positive"], 0.0)

    def test_report_ranks_by_rank_signal_not_by_score(self):
        pn, rep = self.pn[1], self.rep[1]
        P = pn.picks(3)
        W, t0 = pn.tranche_weights(P)
        sim = S.simulate_weights(pn.CO, pn.OC, W, pn.buy_cost, pn.sell_cost, first=t0)
        eq = np.concatenate([[1.0], np.cumprod(1 + sim["ret"])])
        np.testing.assert_allclose(rep["curve"]["top"], eq[M.downsample_indices(len(eq))], atol=1e-5)
        P_old = pn.picks(3, key=pn.SC)
        W_old, _ = pn.tranche_weights(P_old)
        sim_old = S.simulate_weights(pn.CO, pn.OC, W_old, pn.buy_cost, pn.sell_cost, first=t0)
        self.assertFalse(np.allclose(sim["ret"], sim_old["ret"]))                           # the old key built another basket
        old_sigma = U.exposure_stats(pn, P_old, None, t0)["mean_sigma"]
        self.assertGreater(old_sigma, 1.5 * rep["basket"]["mean_sigma"])

    def test_top_sigma_benchmark_is_model_free(self):
        panel2 = {}
        for code, (d, wf) in self.panel.items():
            hf = wf.horizons[1]
            mu2 = 2.0 * hf.mu + 0.0005                                 # another view, same volatility
            panel2[code] = (d, H.make_wf(d, {1: H.make_hf(d, 1, 300, stats.norm.cdf(mu2 / hf.sigma), mu2,
                                                          hf.sigma.copy())}, 300))
        rep2 = U.evaluate_universe(panel2, 1, top_k=3, n_boot=100)
        rep1 = self.rep[1]
        self.assertEqual(rep1["benchmark_top_sigma"], rep2["benchmark_top_sigma"])
        self.assertEqual(rep1["curve"]["top_sigma"], rep2["curve"]["top_sigma"])
        self.assertNotEqual(rep1["top"]["total_return"], rep2["top"]["total_return"])      # the model basket does move

    def test_taiex_is_the_market_of_the_beta_estimate(self):
        idx = H.make_price_data(900, 77, symbol="^TWII", kind="index", drift=0.0003, vol=0.01)
        rep = U.evaluate_universe(self.panel, 5, top_k=3, taiex=idx, n_boot=50)
        self.assertEqual(rep["basket"]["beta_market"], "taiex")
        self.assertIsInstance(rep["basket"]["mean_beta"], float)
        self.assertIsInstance(rep["basket"]["universe_mean_beta"], float)
        self.assertTrue(any("加權指數" in b and "β" in b for b in rep["verdict"]["bullets"]))
        self.assertEqual(U.validate_universe_report(rep), [])

    def test_verdict_and_notes_describe_the_new_ranking(self):
        rep = self.rep[5]
        txt = rep["verdict"]["text"] + "".join(rep["verdict"]["bullets"]) + "".join(rep["notes"])
        self.assertIn("相對強度", rep["verdict"]["text"])
        self.assertNotIn("評分最高", txt)
        self.assertIn("風險暴露", "".join(rep["verdict"]["bullets"]))
        self.assertIn("無預測對照組", "".join(rep["verdict"]["bullets"]))
        self.assertIn("日對日重疊率", "".join(rep["verdict"]["bullets"]))
        self.assertIn("對照組", "".join(rep["notes"]))
        self.assertTrue(any("對數報酬" in b and "Jensen" in b for b in rep["verdict"]["bullets"]))
        for s_ in [rep["verdict"]["title"], rep["verdict"]["text"], *rep["verdict"]["bullets"], *rep["notes"]]:
            self.assertFalse(SIMPLIFIED & set(s_), s_)


class TestPricedBetaNull(unittest.TestCase):
    """Audit S-05 end to end: a no-skill universe whose high-volatility names are high-beta in a market with a premium.  The old
    key (cost-adjusted score = a volatility ranking) collects beta x premium and shows a 'significant' IC in many panels; the
    rank_signal ranking (independent of volatility here) must stay at its nominal rate and the verdict must stay no_edge."""

    def test_rank_signal_does_not_turn_a_priced_beta_into_an_edge(self):
        h, old_sig, new_sig = 5, 0, 0
        for seed in range(10):
            panel = H.make_vol_panel(12, 800, seed, h_list=(h,), vol_lo=0.005, vol_hi=0.04, drift=0.0, beta_slope=30.0,
                                     market_drift=0.0011)
            rep = U.evaluate_universe(panel, h, top_k=3, n_boot=20)
            self.assertEqual(rep["verdict"]["level"], "no_edge", seed)
            pn = U._Panel(panel, h, 0.6, 5.0)
            t0 = int(np.argmax(pn.avail.any(axis=1)))
            old_sig += U._t_stats(U.ic_series(pn, 6, key=pn.SC)[t0:], h)["p_one"] < 0.05
            new_sig += U._t_stats(U.ic_series(pn, 6, key=pn.RS)[t0:], h)["p_one"] < 0.05
        self.assertGreaterEqual(old_sig, 4, old_sig)          # the premium is real: the volatility key finds it (6 / 10 here)
        self.assertLessEqual(new_sig, 2, new_sig)             # nominal 5 % (1 / 10 here)

    def test_default_market_drift_keeps_the_old_panels(self):
        a = H.make_vol_panel(4, 300, 3, h_list=(1,), beta_slope=30.0)
        b = H.make_vol_panel(4, 300, 3, h_list=(1,), beta_slope=30.0, market_drift=0.0)
        for code in a:
            np.testing.assert_array_equal(a[code][0].adjclose, b[code][0].adjclose)
        c = H.make_vol_panel(4, 300, 3, h_list=(1,), beta_slope=30.0, market_drift=0.002)
        self.assertFalse(np.array_equal(a["2004"][0].adjclose, c["2004"][0].adjclose))


class TestExposureHelpers(unittest.TestCase):
    def test_topk_overlap_hand(self):
        P = np.zeros((4, 5))
        for t, js in enumerate([(0, 1), (0, 1), (1, 2), (3, 4)]):
            P[t, list(js)] = 0.5
        self.assertAlmostEqual(U.topk_overlap(P), (1.0 + 0.5 + 0.0) / 3)
        self.assertIsNone(U.topk_overlap(np.zeros((3, 4))))
        P2 = P.copy()
        P2[1] = 0                                   # a day without picks drops both pairs it belongs to
        self.assertAlmostEqual(U.topk_overlap(P2), 0.0)

    def test_pooled_mean(self):
        v, w = np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 0.0, 1.0, 3.0])
        self.assertAlmostEqual(U._pooled_mean(v, w), (1 * 1 + 4 * 3) / 4)           # NaN value and zero weight are dropped
        self.assertIsNone(U._pooled_mean(np.array([np.nan]), np.array([1.0])))
        self.assertIsNone(U._pooled_mean(np.array([1.0]), np.array([0.0])))

    def test_causal_betas_match_ols_and_never_look_ahead(self):
        rng = np.random.default_rng(5)
        T, S_ = 400, 3
        m = rng.normal(0, 0.01, T)
        true = np.array([0.5, 1.0, 1.5])
        x = m[:, None] * true + rng.normal(0, 0.01, (T, S_))
        valid = np.ones((T, S_), bool)
        valid[:100, 2] = False
        valid[200:220, 0] = False
        b = U.causal_betas(x, valid, m, window=120, min_obs=60)
        checked_nan = checked_val = 0
        for t, j in ((59, 0), (150, 1), (250, 0), (399, 2), (130, 2), (99, 2)):
            rows = [i for i in range(max(0, t - 119), t + 1) if valid[i, j]]
            if len(rows) < 60:
                self.assertTrue(np.isnan(b[t, j]), (t, j))
                checked_nan += 1
                continue
            self.assertAlmostEqual(b[t, j], np.polyfit(m[rows], x[rows, j], 1)[0], places=10)
            checked_val += 1
        self.assertGreaterEqual((checked_nan, checked_val), (2, 4))
        # causality: whatever happens after day 300 leaves the estimates up to day 300 untouched
        x2, m2 = x.copy(), m.copy()
        x2[301:] += rng.normal(0, 0.05, (T - 301, S_))
        m2[301:] *= 3
        b2 = U.causal_betas(x2, valid, m2, window=120, min_obs=60)
        np.testing.assert_array_equal(b[:301], b2[:301])
        self.assertFalse(np.array_equal(b[301:], b2[301:]))
        np.testing.assert_allclose(b[300], true, atol=0.45)                              # ~ 5 standard errors
        self.assertTrue(np.all(np.isnan(U.causal_betas(x, valid, np.zeros(T)))))        # flat market: undefined

    def test_rank_sigma_corr_and_universe_exposure_match_plain_loops(self):
        pn = U._Panel(H.make_vol_panel(10, 600, 4, h_list=(1,)), 1, 0.6, 5.0)
        t0 = int(np.argmax(pn.avail.any(axis=1)))
        vals = [stats.spearmanr(pn.RS[t, pn.avail[t]], pn.SG[t, pn.avail[t]]).statistic
                for t in range(t0, pn.T) if pn.avail[t].sum() >= 6]
        self.assertAlmostEqual(U.rank_sigma_corr(pn, 6, t0), float(np.mean(vals)), places=12)
        t1 = t0 + 40
        live = pn.avail.copy()
        live[:t1] = False
        self.assertAlmostEqual(U.universe_exposure(pn, None, t1)["mean_sigma"], float(np.mean(pn.SG[live])), places=12)
        beta = np.where(pn.avail, 1.0 + np.arange(pn.S)[None, :] * 0.1, np.nan)
        self.assertAlmostEqual(U.universe_exposure(pn, beta, t1)["mean_beta"], float(np.mean(beta[live])), places=12)
        P = pn.picks(3)
        self.assertAlmostEqual(U.exposure_stats(pn, P, beta, t0)["mean_beta"],
                               float((beta * P).sum() / P.sum()), places=12)


class TestExposureBullets(unittest.TestCase):
    def rep(self, ratio, sg_sharpe, share=0.5, betas=(1.8, 1.0)):
        return {"top": {"total_return": 0.5, "cagr": 0.1, "sharpe": 0.8},
                "benchmark_top_sigma": {"total_return": 0.6, "cagr": 0.12, "sharpe": sg_sharpe},
                "basket": {"mean_sigma": 0.05 * ratio, "universe_mean_sigma": 0.05, "sigma_ratio": ratio,
                           "mean_beta": betas[0], "universe_mean_beta": betas[1], "beta_market": "taiex",
                           "topk_overlap": 0.8, "corr_rank_sigma": -0.5, "share_net_positive": share}}

    def test_high_vol_high_beta_basket_is_flagged(self):
        b = U._exposure_bullets(self.rep(1.6, 1.4), h=5, top_k=5)
        self.assertEqual(len(b), 2)
        self.assertIn("警示：精選股平均波動度／β 明顯高於全體", b[0])
        self.assertIn("1.60 倍", b[0])
        self.assertIn("低波動", b[0])                                        # corr -0.5: the ranking is low-vol tilted
        self.assertIn("沒有勝過這個不使用任何預測的波動度排序", b[1])
        self.assertIn("報酬不能歸功於預測", b[1])
        # beta alone triggers the warning, a similar basket does not
        self.assertIn("警示", U._exposure_bullets(self.rep(1.0, 0.3, betas=(1.5, 1.0)), h=5, top_k=5)[0])
        calm = U._exposure_bullets(self.rep(0.7, 0.3, betas=(0.9, 1.0)), h=5, top_k=5)
        self.assertNotIn("警示", calm[0])
        self.assertIn("夏普高於此對照組", calm[1])

    def test_no_cost_coverage_reminder_and_missing_blocks(self):
        b = U._exposure_bullets(self.rep(1.0, 0.3, share=0.0), h=1, top_k=5)
        self.assertTrue(any("提醒" in x and "覆蓋來回交易成本" in x and "明日" in x for x in b))
        self.assertFalse(any("覆蓋來回交易成本" in x for x in U._exposure_bullets(self.rep(1.0, 0.3, share=0.4), h=1, top_k=5)))
        self.assertEqual(U._exposure_bullets({"top": {}, "basket": None}, h=5, top_k=5), [])
        r = self.rep(1.0, 0.3)
        r["benchmark_top_sigma"] = None
        r["basket"]["mean_beta"] = r["basket"]["universe_mean_beta"] = None
        self.assertEqual(len(U._exposure_bullets(r, h=5, top_k=5)), 1)                  # no beta / no benchmark: still fine


class TestHitRateTiltDisclosure(unittest.TestCase):
    """The picks hit-rate leg compares with the same-day universe up-rate and does not control for volatility: P(up) is higher
    for low-volatility names, so a model-free low-vol basket also 'beats' it (real data: +1.9 / +3.0 / +4.4 pp at h = 1 / 5 / 20
    against +2.65 / +2.83 / +3.42 pp for the rank_signal basket).  The wording must say so (verdict level logic is unchanged)."""

    @staticmethod
    def stub(ratio, p_picks):
        stats_ = {"total_return": 0.5, "cagr": 0.08, "sharpe": 0.7, "max_dd": -0.2, "turnover_per_year": 20.0}
        rep = {"ic": {"mean": 0.02, "ir": 0.1, "t": 3.3, "pvalue": 0.001},
               "picks": {"hit_rate": 0.54, "base_rate": 0.51, "n": 5000, "p_value": p_picks, "mean_return": 0.002},
               "spread": {"mean": 0.001, "t": 1.5}, "top": stats_, "benchmark_equal_weight": dict(stats_, cagr=0.07),
               "benchmark_taiex": None, "benchmark_top_sigma": None,
               "period": {"start": "2021-01-04", "end": "2026-01-02", "n_days": 1200, "n_stocks": 60},
               "basket": {"mean_sigma": 0.05 * ratio, "universe_mean_sigma": 0.05, "sigma_ratio": ratio, "mean_beta": None,
                          "universe_mean_beta": None, "beta_market": "taiex", "topk_overlap": 0.8, "corr_rank_sigma": -0.4,
                          "share_net_positive": 0.5}}
        ev = {"p": {"ic": 0.0004, "picks": p_picks, "spread": 0.1}, "n_eff": 1200, "n_scored": 1200, "psr_vs_bh": 0.5,
              "diff_p": 0.4}
        return rep, ev

    def verdict(self, ratio, p_picks, p=None):
        rep, ev = self.stub(ratio, p_picks)
        if p is not None:
            ev["p"] = p
        ass = RP.assess_evidence(ev, RP.THRESH_UNIVERSE)
        return ass, U.make_universe_verdict(rep, ass, ev, h=1, top_k=5, exec_mode="next_open")

    def test_clause_appears_only_when_the_picks_leg_confirms_and_the_basket_is_low_vol(self):
        ass, v = self.verdict(0.6, 0.0005)
        self.assertEqual(ass["level"], "moderate_edge")
        self.assertIn("命中率的比較基準沒有控制波動度", v["text"])
        self.assertIn("0.60 倍", v["text"])
        self.assertIn("不算獨立證據", v["text"])
        self.assertTrue(any(U.HIT_TILT_NOTE in b and "0.60 倍" in b for b in v["bullets"]))
        for ratio, p_picks in ((1.0, 0.0005), (1.3, 0.0005), (0.6, 0.2)):
            _, v2 = self.verdict(ratio, p_picks)
            self.assertNotIn("命中率的比較基準沒有控制波動度", v2["text"], (ratio, p_picks))
        _, v3 = self.verdict(1.0, 0.0005)
        self.assertFalse(any(U.HIT_TILT_NOTE in b for b in v3["bullets"]))
        _, v4 = self.verdict(0.6, 0.2)                       # the bullet describes the basket, whatever this verdict's legs are
        self.assertTrue(any(U.HIT_TILT_NOTE in b for b in v4["bullets"]))

    def test_level_logic_is_not_touched(self):
        for ratio in (0.6, 1.0):
            ass, v = self.verdict(ratio, 0.0005)
            self.assertEqual((ass["level"], v["level"]), ("moderate_edge", "moderate_edge"))

    def test_weak_edge_text_carries_the_clause_too(self):
        # picks below the per-test level (0.3 %), IC only below 5 %: weak_edge with the picks leg among the significant tests
        ass, v = self.verdict(0.5, 0.0005, p={"ic": 0.02, "picks": 0.0005, "spread": 0.3})
        self.assertEqual(ass["level"], "weak_edge")
        self.assertIn("命中率的比較基準沒有控制波動度", v["text"])
        self.assertIn("0.50 倍", v["text"])
        ass, v = self.verdict(1.0, 0.0005, p={"ic": 0.02, "picks": 0.0005, "spread": 0.3})
        self.assertEqual(ass["level"], "weak_edge")
        self.assertNotIn("命中率的比較基準沒有控制波動度", v["text"])


class TestPerStockWithFlatDays(unittest.TestCase):
    def test_flat_days_are_scored_as_not_up_in_the_brier_skill_and_dropped_from_accuracy(self):
        panel = {}
        for i in range(6):
            d = H.make_tick_data(1200, 20 + i, symbol=f"{4000 + i}")
            R = M.realized_log_returns(d.adjclose, 1)
            b = float(np.mean(R[400:1199] > 0))
            panel[d.symbol] = (d, H.make_wf(d, {1: H.make_hf(d, 1, 400, b, 0.0, 0.012)}, 400))   # constant, event-calibrated
        rep = U.evaluate_universe(panel, 1, top_k=2, n_boot=50)
        self.assertEqual(U.validate_universe_report(rep), [])
        for row in rep["per_stock"]:
            self.assertAlmostEqual(row["brier_skill_oracle"], 0.0, places=12)       # exact: b is the up-rate of R > 0 incl. flats
            self.assertLess(row["n"], 799 - 60)                                      # accuracy counts the non-flat days only
            self.assertIsNotNone(row["brier_skill"])
        self.assertTrue(any("平盤視為「未上漲」" in n for n in rep["notes"]))

    def test_declared_up_given_move_leaves_flat_days_out_of_the_brier_skill(self):
        # the quant engine declares p_up = P(R > 0 | R != 0): the per-stock Brier skill follows (same rule as report.py)
        panel = {}
        for i in range(6):
            d = H.make_tick_data(1200, 20 + i, symbol=f"{4000 + i}")
            R = M.realized_log_returns(d.adjclose, 1)
            seg = R[400:1199]
            b_nf = float(np.mean(seg[seg != 0] > 0))
            wf = H.make_wf(d, {1: H.make_hf(d, 1, 400, b_nf, 0.0, 0.012)}, 400)    # constant = up-rate among MOVING days
            wf.config = {"p_up_event": "up_given_move"}
            panel[d.symbol] = (d, wf)
        rep = U.evaluate_universe(panel, 1, top_k=2, n_boot=50)
        self.assertEqual(U.validate_universe_report(rep), [])
        for row in rep["per_stock"]:
            self.assertAlmostEqual(row["brier_skill_oracle"], 0.0, places=12)       # exact on the moving rows
        self.assertTrue(any("不計入機率評分" in n for n in rep["notes"]))
        pn = U._Panel(panel, 1, 0.6, 5.0)
        self.assertEqual(set(pn.events), {"up_given_move"})
        # undeclared: the same constants are scored as P(R > 0) on every row and look biased high
        for code, (d, wf) in panel.items():
            wf.config = {}
        rep0 = U.evaluate_universe(panel, 1, top_k=2, n_boot=50)
        self.assertTrue(all(r["brier_skill_oracle"] < -0.005 for r in rep0["per_stock"]))
        self.assertEqual(set(U._Panel(panel, 1, 0.6, 5.0).events), {"up"})

    def test_flat_note_helper(self):
        self.assertIn("不計入機率評分", U._flat_note(["up_given_move", "up_given_move"]))
        self.assertIn("未上漲", U._flat_note(["up", "up"]))
        self.assertIn("未上漲", U._flat_note(None))
        self.assertIn("宣告", U._flat_note(["up", "up_given_move"]))


class TestUniverseVerdictTexts(unittest.TestCase):
    def test_positive_ic_but_losing_after_costs_is_called_out(self):
        panel = H.make_panel(12, 900, 5, edge="edge", h_list=(1,), first_idx=300)
        rep = U.evaluate_universe(panel, 1, top_k=4, commission_discount=1.0, slippage_bps=150.0, n_boot=100)
        self.assertGreater(rep["ic"]["mean"], 0.1)
        self.assertLess(rep["top"]["cagr"], 0.0)
        v = rep["verdict"]
        self.assertIn(v["level"], ("weak_edge", "moderate_edge"))
        self.assertIn("反而虧損", v["text"])
        self.assertIn("無法據此獲利", v["text"])
        self.assertTrue(any("訊號不足以支付每日換股的成本" in b for b in v["bullets"]))
        self.assertEqual(U.validate_universe_report(rep), [])

    def test_significantly_negative_ic_is_flagged_even_though_all_tests_are_one_sided(self):
        panel = H.make_panel(12, 900, 3, edge="edge", h_list=(1,), first_idx=300)
        inv = {}
        for code, (d, wf) in panel.items():
            hf = wf.horizons[1]
            inv[code] = (d, H.make_wf(d, {1: H.make_hf(d, 1, 300, 1 - hf.p_up, -hf.mu, hf.sigma)}, 300))
        rep = U.evaluate_universe(inv, 1, top_k=4, n_boot=100)
        self.assertEqual(rep["verdict"]["level"], "no_edge")
        self.assertLess(rep["ic"]["mean"], -0.1)
        self.assertTrue(any("橫斷面 IC 顯著為負" in b and "請不要依此排序操作" in b for b in rep["verdict"]["bullets"]))
        good = U.evaluate_universe(panel, 1, top_k=4, n_boot=100)
        self.assertFalse(any("顯著為負" in b for b in good["verdict"]["bullets"]))


class TestUniverseAdditiveKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = U.evaluate_universe(H.make_vol_panel(8, 700, 3, h_list=(1,)), 1, top_k=3, n_boot=50)

    def mutate(self, fn):
        r = copy.deepcopy(self.rep)
        fn(r)
        return U.validate_universe_report(r)

    def test_old_reports_without_the_new_keys_still_validate(self):
        def strip(r):
            for k in ("basket", "exposure", "benchmark_top_sigma", "rank_key", "only_buy_signals", "labels"):
                del r[k]
            del r["curve"]["top_sigma"]
            for row in r["per_stock"]:
                del row["brier_skill_oracle"]
        self.assertEqual(self.mutate(strip), [])
        self.assertEqual(U.validate_universe_report(self.rep), [])

    def test_new_keys_are_checked_when_present(self):
        self.assertTrue(self.mutate(lambda r: r["basket"].__setitem__("mean_sigma", "x")))
        self.assertTrue(self.mutate(lambda r: r["basket"].__setitem__("beta_market", "nasdaq")))
        self.assertTrue(self.mutate(lambda r: r["basket"].pop("k")))
        self.assertTrue(self.mutate(lambda r: r.__setitem__("rank_key", "score")))
        self.assertTrue(self.mutate(lambda r: r.__setitem__("only_buy_signals", 1)))
        self.assertTrue(self.mutate(lambda r: r["labels"].pop("top")))
        self.assertTrue(self.mutate(lambda r: r["curve"].__setitem__("top_sigma", r["curve"]["top_sigma"][:-1])))
        self.assertTrue(self.mutate(lambda r: r["exposure"].__setitem__("beta_top", "x")))
        self.assertTrue(self.mutate(lambda r: r["benchmark_top_sigma"].pop("psr")))
        self.assertTrue(self.mutate(lambda r: r["per_stock"][0].__setitem__("brier_skill_oracle", "x")))
        self.assertEqual(self.mutate(lambda r: r.__setitem__("benchmark_top_sigma", None)), [])     # optional block may be null


class TestUniverseNull(unittest.TestCase):
    def test_no_skill_universe_is_no_edge(self):
        for edge in ("none", "noise"):
            for seed in range(4):
                panel = H.make_panel(12, 1100, seed, edge=edge, h_list=(1, 5), first_idx=300)
                for h in (1, 5):
                    rep = U.evaluate_universe(panel, h, top_k=4, n_boot=100)
                    self.assertEqual(rep["verdict"]["level"], "no_edge", (edge, seed, h))
                    self.assertEqual(U.validate_universe_report(rep), [])
                    self.assertEqual(rep["verdict"]["title"], "沒有統計上顯著的選股優勢")

    def test_inverted_signal_is_no_edge(self):
        panel = H.make_panel(12, 900, 3, edge="edge", h_list=(1,), first_idx=300)
        inv = {}
        for code, (d, wf) in panel.items():
            hf = wf.horizons[1]
            hf2 = H.make_hf(d, 1, 300, 1 - hf.p_up, -hf.mu, hf.sigma)
            inv[code] = (d, H.make_wf(d, {1: hf2}, 300))
        rep = U.evaluate_universe(inv, 1, top_k=4, n_boot=100)
        self.assertEqual(rep["verdict"]["level"], "no_edge")
        self.assertLess(rep["ic"]["mean"], -0.1)
        self.assertLess(rep["top"]["sharpe"], rep["benchmark_equal_weight"]["sharpe"])


class TestUniverseCausalityAndOptions(unittest.TestCase):
    def test_future_data_never_changes_past_weights_or_returns(self):
        panel = H.make_panel(8, 700, 6, edge="edge", h_list=(1,), first_idx=300)
        t_cut = 500
        panel2 = {}
        rng = np.random.default_rng(0)
        for code, (d, wf) in panel.items():
            d2 = copy.deepcopy(d)
            for arr in (d2.open, d2.high, d2.low, d2.close, d2.adjclose):
                arr[t_cut + 1:] *= np.exp(rng.normal(0, 0.05, len(arr) - t_cut - 1))
            hf = wf.horizons[1]
            hf2 = H.make_hf(d2, 1, 300, hf.p_up.copy(), hf.mu.copy(), hf.sigma.copy())
            hf2.p_up[t_cut + 1:] = np.nan                       # forecasts after the cut are altered as well
            hf2.mu[t_cut + 1:] = rng.normal(0, 0.01, len(hf2.mu) - t_cut - 1)
            panel2[code] = (d2, H.make_wf(d2, {1: hf2}, 300))
        a, b = U._Panel(panel, 1, 0.6, 5.0), U._Panel(panel2, 1, 0.6, 5.0)
        Pa, Pb = a.picks(3), b.picks(3)
        row = t_cut - 300                                       # master row of the last unchanged signal day
        np.testing.assert_array_equal(Pa[:row + 1], Pb[:row + 1])
        Wa, t0 = a.tranche_weights(Pa)
        Wb, _ = b.tranche_weights(Pb)
        np.testing.assert_allclose(Wa[:row + 1], Wb[:row + 1])
        sa = S.simulate_weights(a.CO, a.OC, Wa, a.buy_cost, a.sell_cost, first=t0)["ret"]
        sb = S.simulate_weights(b.CO, b.OC, Wb, b.buy_cost, b.sell_cost, first=t0)["ret"]
        np.testing.assert_array_equal(sa[:row], sb[:row])     # days <= t_cut are untouched
        self.assertFalse(np.array_equal(sa[row:], sb[row:]))

    def test_only_buy_signals_option(self):
        panel = H.make_panel(10, 800, 7, edge="edge", h_list=(5,), first_idx=300)
        rep_all = U.evaluate_universe(panel, 5, top_k=4, n_boot=50)
        rep_buy = U.evaluate_universe(panel, 5, top_k=4, n_boot=50, only_buy_signals=True)
        self.assertEqual(U.validate_universe_report(rep_buy), [])
        self.assertLessEqual(rep_buy["top"]["exposure"], rep_all["top"]["exposure"] + 1e-12)
        self.assertLess(rep_buy["top"]["n_trades"], rep_all["top"]["n_trades"])
        self.assertTrue(any("偏多候選條件" in n for n in rep_buy["notes"]))       # was 買進條件: DESIGN.md 9.6 wording
        self.assertTrue(rep_buy["only_buy_signals"])
        self.assertFalse(rep_all["only_buy_signals"])
        self.assertTrue(any("全部股票都參與排名" in n for n in rep_all["notes"]))

    def test_exec_modes_and_costs(self):
        panel = H.make_panel(10, 800, 8, edge="edge", h_list=(1,), first_idx=300)
        free = U.evaluate_universe(panel, 1, top_k=3, commission_discount=0.0, slippage_bps=0.0, n_boot=50)
        paid = U.evaluate_universe(panel, 1, top_k=3, commission_discount=1.0, slippage_bps=30.0, n_boot=50)
        self.assertLess(paid["top"]["total_return"], free["top"]["total_return"])
        close = U.evaluate_universe(panel, 1, top_k=3, exec_mode="close", n_boot=50)
        self.assertEqual(U.validate_universe_report(close), [])
        with self.assertRaises(ValueError):
            U.evaluate_universe(panel, 1, exec_mode="vwap")
        with self.assertRaises(ValueError):
            U.evaluate_universe(panel, 1, top_k=0)

    def test_taiex_benchmark(self):
        panel = H.make_panel(10, 800, 9, edge="none", h_list=(1,), first_idx=300)
        idx = H.make_price_data(800, 99, symbol="^TWII", kind="index", drift=0.0003, vol=0.01)
        rep = U.evaluate_universe(panel, 1, top_k=3, taiex=idx, n_boot=50)
        self.assertEqual(U.validate_universe_report(rep), [])
        c = rep["curve"]
        self.assertEqual(len(c["taiex"]), len(c["dates"]))
        d0 = panel["1101"][0].dates
        first_date = rep["curve"]["dates"][0]
        i0 = int(np.flatnonzero(d0 == np.datetime64(first_date))[0])
        self.assertAlmostEqual(c["taiex"][-1], idx.adjclose[-1] / idx.adjclose[i0], places=4)
        self.assertAlmostEqual(rep["benchmark_taiex"]["total_return"], c["taiex"][-1] - 1, places=5)
        self.assertTrue(any("價格指數" in n for n in rep["notes"]))
        old = H.make_price_data(100, 5, symbol="^TWII", kind="index", start="1990-01-01")      # no date overlap
        rep2 = U.evaluate_universe(panel, 1, top_k=3, taiex=old, n_boot=50)
        self.assertIsNone(rep2["benchmark_taiex"])
        self.assertIsNone(rep2["curve"]["taiex"])
        self.assertTrue(any("重疊不足" in n for n in rep2["notes"]))

    def test_curve_cap(self):
        panel = H.make_panel(6, 2600, 10, edge="none", h_list=(1,), first_idx=300)
        rep = U.evaluate_universe(panel, 1, top_k=2, n_boot=50)
        self.assertEqual(len(rep["curve"]["dates"]), 600)
        self.assertEqual(rep["curve"]["dates"][-1], str(panel["1101"][0].dates[-1]))
        self.assertLessEqual(len(rep["ic"]["series"]["dates"]), 600)
        self.assertEqual(U.validate_universe_report(rep), [])


class TestUniverseDegenerate(unittest.TestCase):
    def ok(self, rep, no_edge=True):
        self.assertEqual(U.validate_universe_report(rep), [])
        json.dumps(rep, allow_nan=False)
        if no_edge:
            self.assertEqual(rep["verdict"]["level"], "no_edge")
        self.assertIn("存活者偏差", " ".join(rep["notes"]))

    def test_empty_panels(self):
        for panel in ({}, None):
            rep = U.evaluate_universe(panel, 5)
            self.ok(rep)
            self.assertEqual(rep["period"]["n_days"], 0)
            self.assertEqual(rep["per_stock"], [])
            self.assertIsNone(rep["ic"]["mean"])

    def test_single_stock_and_tiny_panels(self):
        panel = H.make_panel(1, 600, 1, edge="edge", h_list=(1,), first_idx=300)
        rep = U.evaluate_universe(panel, 1, top_k=5, n_boot=50)
        self.ok(rep)
        self.assertEqual(rep["period"]["n_stocks"], 1)
        panel2 = H.make_panel(3, 500, 1, edge="edge", h_list=(1, 5), first_idx=300)   # planted edge, only 3 names
        for h in (1, 5):
            self.ok(U.evaluate_universe(panel2, h, top_k=5, n_boot=50), no_edge=False)

    def test_no_forecasts_and_too_short(self):
        d = H.make_price_data(500, 1)
        hf = H.make_hf(d, 1, 300, np.nan, np.nan, np.nan)
        rep = U.evaluate_universe({"X": (d, H.make_wf(d, {1: hf}, 300))}, 1, n_boot=50)
        self.ok(rep)
        d2, wf2, _ = H.make_planted_world(500, 1, h_list=(1,), first_idx=498)
        rep = U.evaluate_universe({"X": (d2, wf2), "Y": (d2, wf2)}, 1, n_boot=50)
        self.ok(rep)

    def test_missing_horizon_and_misalignment_raise(self):
        panel = H.make_panel(4, 500, 2, edge="edge", h_list=(1,), first_idx=300)
        with self.assertRaises(ValueError):
            U.evaluate_universe(panel, 5)
        d, wf = panel["1101"]
        wf.horizons[1].realized = np.roll(wf.horizons[1].realized, 1)
        with self.assertRaises(ValueError) as cm:
            U.evaluate_universe(panel, 1)
        self.assertIn("1101", str(cm.exception))

    def test_stocks_with_different_calendars_and_nan_forecasts(self):
        panel = H.make_panel(8, 800, 11, edge="edge", h_list=(1,), first_idx=300)
        d, wf = panel["1103"]
        wf.horizons[1].p_up[400:450] = np.nan                       # forecast gap
        # a later-listed stock: shorter history, first forecast later
        late = H.make_price_data(560, 77, symbol="9999", name="新股", start=str(panel["1101"][0].dates[240]))
        hf = H.make_hf(late, 1, 400, 0.55, 0.002, 0.015)
        panel["9999"] = (late, H.make_wf(late, {1: hf}, 400))
        # a stock that stops trading early (suspension / delisting)
        dead = H.make_price_data(600, 78, symbol="8888", name="下市股", start=str(panel["1101"][0].dates[0]))
        hf = H.make_hf(dead, 1, 300, 0.6, 0.004, 0.015)
        panel["8888"] = (dead, H.make_wf(dead, {1: hf}, 300))
        rep = U.evaluate_universe(panel, 1, top_k=3, n_boot=50)
        self.ok(rep, no_edge=False)
        self.assertEqual(rep["period"]["n_stocks"], 10)
        self.assertEqual(len(rep["per_stock"]), 10)
        n = {r["code"]: r["n"] for r in rep["per_stock"]}
        self.assertLess(n["9999"], n["1101"])
        self.assertLess(n["8888"], n["1101"])

    def test_constant_prices_universe(self):
        panel = {}
        for i in range(8):
            d = H.make_tiny_data([50.0 + i] * 500, [50.0 + i] * 500, symbol=f"{3000 + i}")
            hf = H.make_hf(d, 1, 300, 0.6, 0.002 + 0.0001 * i, 0.01)
            panel[d.symbol] = (d, H.make_wf(d, {1: hf}, 300))
        rep = U.evaluate_universe(panel, 1, top_k=3, n_boot=50)
        self.assertEqual(U.validate_universe_report(rep), [])
        self.assertIsNone(rep["ic"]["mean"])
        self.assertEqual(rep["picks"]["n"], 0)
        self.assertIsNone(rep["benchmark_equal_weight"]["sharpe"])


class TestUniverseDataAlerts(unittest.TestCase):
    def test_bad_stock_is_named_in_the_notes(self):
        panel = H.make_panel(8, 700, 3, edge="none", h_list=(1,), first_idx=300)
        d, wf = panel["1103"]
        for arr in (d.open, d.high, d.low, d.close, d.adjclose):
            arr[500:] *= 0.25
        hf = wf.horizons[1]
        panel["1103"] = (d, H.make_wf(d, {1: H.make_hf(d, 1, 300, hf.p_up.copy(), hf.mu.copy(), hf.sigma.copy())}, 300))
        rep = U.evaluate_universe(panel, 1, top_k=3, n_boot=50)
        self.assertEqual(U.validate_universe_report(rep), [])
        self.assertTrue(rep["notes"][0].startswith("資料品質警示：1103"))
        self.assertRegex(rep["notes"][0], r"-7[0-9]\.[0-9]%")
        clean = U.evaluate_universe(H.make_panel(8, 700, 3, edge="none", h_list=(1,), first_idx=300), 1, top_k=3,
                                    n_boot=50)
        self.assertFalse(any("資料品質警示" in n for n in clean["notes"]))


class TestUniverseText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edge = U.evaluate_universe(H.make_panel(12, 900, 5, edge="edge", h_list=(5,), first_idx=300), 5, top_k=4,
                                       n_boot=100)
        cls.none = U.evaluate_universe(H.make_panel(12, 900, 5, edge="noise", h_list=(5,), first_idx=300), 5, top_k=4,
                                       n_boot=100)

    def strings(self, rep):
        v = rep["verdict"]
        return [v["title"], v["text"], *v["bullets"], *rep["notes"]]

    def test_chinese_and_no_promises(self):
        for rep in (self.edge, self.none):
            for s in self.strings(rep):
                self.assertFalse(SIMPLIFIED & set(s), s)
                cleaned = s.replace("不保證獲利", "").replace("不保證", "")
                for w in FORBIDDEN:
                    self.assertNotIn(w, cleaned)
            self.assertIn("不構成投資建議", rep["verdict"]["text"])

    def test_explicit_survivorship_note(self):
        for rep in (self.edge, self.none):
            self.assertTrue(rep["notes"][0].startswith("存活者偏差"))
            self.assertIn("下市", rep["notes"][0])
            self.assertIn("高估", rep["notes"][0])
            self.assertTrue(any("存活者偏差" in b for b in rep["verdict"]["bullets"]))

    def test_text_quotes_numbers(self):
        rep = self.edge
        v, ic, pk = rep["verdict"], rep["ic"], rep["picks"]
        self.assertIn(f"{ic['mean']:+.3f}", v["text"])
        self.assertIn(f"{pk['hit_rate'] * 100:.1f}%", v["text"])
        self.assertIn(f"{rep['period']['n_days']:,}", v["text"])
        self.assertIn(str(rep["period"]["n_stocks"]), v["text"])
        self.assertIn("未來5日", v["text"])
        self.assertTrue(any("重疊" in b for b in v["bullets"]))
        self.assertIn(rep["period"]["start"], v["text"])

    def test_uses_the_stricter_universe_threshold_in_text(self):
        t = self.edge["verdict"]["text"]
        if self.edge["verdict"]["level"] in ("moderate_edge", "strong_edge"):
            self.assertIn("0.3% 顯著", t)
        self.assertIn("Newey-West", "".join(self.edge["verdict"]["bullets"]))
        self.assertIn("HAC", "".join(self.edge["notes"]))

    def test_titles(self):
        self.assertEqual(self.none["verdict"]["title"], U.UNIVERSE_TITLES["no_edge"])
        self.assertIn(self.edge["verdict"]["title"], (U.UNIVERSE_TITLES["moderate_edge"], U.UNIVERSE_TITLES["strong_edge"]))


class TestUniverseValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = U.evaluate_universe(H.make_panel(8, 700, 3, edge="edge", h_list=(1,), first_idx=300), 1, top_k=3,
                                      n_boot=50)

    def mutate(self, fn):
        r = copy.deepcopy(self.rep)
        fn(r)
        return U.validate_universe_report(r)

    def test_valid_and_failures(self):
        self.assertEqual(U.validate_universe_report(self.rep), [])
        self.assertTrue(self.mutate(lambda r: r.pop("picks")))
        self.assertTrue(self.mutate(lambda r: r["ic"].pop("series")))
        self.assertTrue(self.mutate(lambda r: r["top"].pop("psr")))
        self.assertTrue(self.mutate(lambda r: r["curve"].pop("taiex")))
        self.assertTrue(self.mutate(lambda r: r["ic"].__setitem__("mean", "x")))
        self.assertTrue(self.mutate(lambda r: r["period"].__setitem__("n_days", 1.5)))
        self.assertTrue(self.mutate(lambda r: r["verdict"].__setitem__("level", "huge")))
        self.assertTrue(self.mutate(lambda r: r["curve"]["top"].pop()))
        self.assertTrue(self.mutate(lambda r: r["per_stock"][0].pop("code")))
        self.assertTrue(self.mutate(lambda r: r["ic"]["series"]["values"].pop()))
        self.assertTrue(self.mutate(lambda r: r["spread"].__setitem__("t", float("nan"))))
        self.assertTrue(U.validate_universe_report([]))
        self.assertTrue(U.validate_universe_report(None))

    def test_optional_taiex_blocks(self):
        r = copy.deepcopy(self.rep)
        r["benchmark_taiex"] = copy.deepcopy(r["top"])
        r["curve"]["taiex"] = list(r["curve"]["top"])
        self.assertEqual(U.validate_universe_report(r), [])
        r["curve"]["taiex"] = r["curve"]["taiex"][:-1]
        self.assertTrue(U.validate_universe_report(r))
        r["curve"]["taiex"] = list(r["curve"]["top"])
        r["benchmark_taiex"].pop("sharpe")
        self.assertTrue(U.validate_universe_report(r))


class TestEvidenceFromReport(unittest.TestCase):
    """The picks payload's ``evidence`` block (DESIGN.md 9(4)) built from a universe report, with the after-cost numbers that
    keep a significant but untradable ranking statistic from unlocking buy-advice wording (real h = 1: IC t = 3.6, basket -79 %)."""
    KEYS = {"horizon", "source", "verdict_level", "ic", "ic_pvalue", "hit_rate", "base_rate", "computed_at"}
    EXTRA = {"net_cagr", "ew_cagr", "sigma_cagr", "beats_ew_after_costs"}

    @classmethod
    def setUpClass(cls):
        panel = H.make_panel(12, 900, 5, edge="edge", h_list=(1,), first_idx=300)
        cls.good = U.evaluate_universe(panel, 1, top_k=4, n_boot=100)
        cls.costly = U.evaluate_universe(panel, 1, top_k=4, commission_discount=1.0, slippage_bps=150.0, n_boot=100)

    def test_values_come_from_the_report(self):
        for rep in (self.good, self.costly):
            ev = U.evidence_from_report(rep, 1, "2026-10-22T10:00:00+08:00")
            self.assertEqual(set(ev), self.KEYS | self.EXTRA)
            self.assertEqual((ev["horizon"], ev["source"], ev["computed_at"]), (1, "universe_validation",
                                                                                  "2026-10-22T10:00:00+08:00"))
            self.assertEqual(ev["verdict_level"], rep["verdict"]["level"])
            self.assertEqual((ev["ic"], ev["ic_pvalue"]), (rep["ic"]["mean"], rep["ic"]["pvalue"]))
            self.assertEqual((ev["hit_rate"], ev["base_rate"]), (rep["picks"]["hit_rate"], rep["picks"]["base_rate"]))
            self.assertEqual((ev["net_cagr"], ev["ew_cagr"]), (rep["top"]["cagr"], rep["benchmark_equal_weight"]["cagr"]))
            self.assertEqual(ev["sigma_cagr"], rep["benchmark_top_sigma"]["cagr"])
            self.assertEqual(ev["beats_ew_after_costs"], rep["top"]["cagr"] > rep["benchmark_equal_weight"]["cagr"])
            json.dumps(ev, allow_nan=False)
        # costs that eat the signal: the verdict machinery still sees the statistical edge, the evidence says it loses money
        self.assertLess(self.costly["top"]["cagr"], 0.0)
        self.assertIs(U.evidence_from_report(self.costly, 1)["beats_ew_after_costs"], False)

    def test_the_cached_job_payload_is_accepted_as_is(self):
        payload = {"report": self.good, "computed_at": "2026-10-22T10:00:00+08:00", "skipped": []}
        ev = U.evidence_from_report(payload, 1)
        self.assertEqual(ev, U.evidence_from_report(self.good, 1, "2026-10-22T10:00:00+08:00"))
        self.assertEqual(U.evidence_from_report(payload, 1, "other")["computed_at"], "other")      # explicit argument wins

    def test_unusable_reports_degrade_to_no_evidence(self):
        empty = U.evaluate_universe({}, 5)
        broken = copy.deepcopy(self.good)
        broken["verdict"]["level"] = "huge"
        no_ic = {k: v for k, v in self.good.items() if k != "ic"}
        for bad in (None, "x", [], {}, empty, broken, no_ic, {"verdict": {"level": "no_edge"}, "ic": {}, "period": {}}):
            ev = U.evidence_from_report(bad, 5, "t")
            self.assertEqual(ev, {"horizon": 5, "source": "none", "verdict_level": None, "ic": None, "ic_pvalue": None,
                                  "hit_rate": None, "base_rate": None, "computed_at": None}, bad)
        scrambled = copy.deepcopy(self.good)
        scrambled["top"] = "oops"
        scrambled["benchmark_top_sigma"] = 3
        ev = U.evidence_from_report(scrambled, 1)                 # a malformed block never raises
        self.assertIn(ev["source"], ("none", "universe_validation"))

    def test_after_cost_numbers_reach_the_picks_note(self):
        from backend.evaluation import recommend as RC
        fcs = [make_rec_fc(f"{3900 + i}", 0.03 * i - 0.1) for i in range(8)]
        for rep, losing in ((self.good, self.good["top"]["cagr"] < self.good["benchmark_equal_weight"]["cagr"]),
                            (self.costly, True)):
            ev = U.evidence_from_report(rep, 5, "t")
            ev["verdict_level"] = "moderate_edge"                  # what the real h = 1 report says
            o = RC.build_recommendations(fcs, 5, evidence=ev)
            txt = next(n for n in o["notes"] if "推薦整體歷史驗證" in n)
            self.assertIn(f"{ev['net_cagr'] * 100:+.1f}%", txt)
            self.assertIn("扣成本後精選組合年化", txt)
            self.assertEqual("沒有勝過全體等權重" in txt, losing)
            self.assertEqual("存活者偏差" in txt, not losing)
            self.assertNotIn("建議買進", txt)


def make_rec_fc(code, z, sg=0.04):
    """Minimal forecast dict for the evidence-note test (information ratio z at h = 5)."""
    mu = z * sg
    hz = {"h": 5, "p_up": float(stats.norm.cdf(z)), "p_up_raw": 0.5, "exp_return": mu, "median_return": mu, "sigma": sg,
          "mu_log": mu, "sigma_log": sg, "components": {}}
    return {"symbol": code, "name": f"股{code}", "kind": "stock", "asof": "2026-10-22", "asof_complete": True,
            "last_close": 100.0, "chg_1d": 0.0, "horizons": {"5": hz}, "regime": {"current": "x"}, "vol": {"percentile": 0.5},
            "drivers": [], "avg_value_20d": 5e9}


class TestPerformance(unittest.TestCase):
    def test_sixty_stocks_five_years(self):
        import time
        panel = H.make_panel(60, 1800, 12, edge="none", h_list=(5,), first_idx=550)
        t0 = time.time()
        rep = U.evaluate_universe(panel, 5, top_k=5)
        dt = time.time() - t0
        self.assertEqual(U.validate_universe_report(rep), [])
        self.assertLess(dt, 15.0, dt)


if __name__ == "__main__":
    unittest.main()
