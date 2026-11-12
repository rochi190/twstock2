"""Tests for research/feat_chips.py and research/run_chips.py (PLAN.md P1b/P1c).  Hermetic: synthetic data + tests/fixtures/alt.

Run: python -m unittest discover -s tests -p "test_research_chips.py"
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from backend import data  # noqa: E402
from backend.altdata import finmind as FM  # noqa: E402
from research import feat_chips as FC  # noqa: E402
from research import harness as H  # noqa: E402
from research import price_baseline as PB  # noqa: E402
from research import run_chips as RC  # noqa: E402
from research.panel import Panel, universe_mask  # noqa: E402

FIX = HERE / "fixtures" / "alt"
NAN = np.nan


def weekdays(start, end):
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    return days[((days.astype(np.int64) + 3) % 7) < 5]


def small_panel(T=80, S=2, volume=1000.0, start="2020-01-06", seed=0, random=False):
    cal = weekdays(start, "2030-01-01")[:T]
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, S)), axis=0)) if random else np.full((T, S), 50.0)
    vol = np.exp(rng.normal(8, 0.3, (T, S))) if random else np.full((T, S), float(volume))
    return Panel(cal, [f"S{j}" for j in range(S)], close.copy(), close, vol, close * vol, np.ones((T, S), bool))


def inst_rows(cal, net_by_row, code="S0", name="Foreign_Investor"):
    """FinMind-shaped long rows: buy = max(net, 0) + 500, sell = buy - net (so net is exact)."""
    rows = []
    for t, net in net_by_row.items():
        buy = max(net, 0) + 500
        rows.append({"date": str(cal[t]), "stock_id": code, "buy": buy, "sell": buy - net, "name": name})
    return rows


def margin_rows(cal, bal_by_row, field="MarginPurchaseTodayBalance", code="S0"):
    return [{"date": str(cal[t]), "stock_id": code, field: v} for t, v in bal_by_row.items()]


def hold_rows(cal, ratio_by_row, code="S0"):
    return [{"date": str(cal[t]), "stock_id": code, "ForeignInvestmentSharesRatio": v} for t, v in ratio_by_row.items()]


def per_rows(cal, by_row, code="S0"):
    return [{"date": str(cal[t]), "stock_id": code, "PER": v[0], "PBR": v[1], "dividend_yield": v[2]} for t, v in by_row.items()]


def feats_of(panel, **datasets):
    """chip_features for column 0 with raw rows given as ``inst=rows, margin=rows, ...``."""
    parsers = {"inst": FM.parse_institutional, "margin": FM.parse_margin, "hold": FM.parse_shareholding, "per": FM.parse_per}
    raw = {"S0": {k: parsers[k](v) for k, v in datasets.items()}}
    return FC.chip_features(panel, raw)


# ============================================================================= alignment
class AlignTests(unittest.TestCase):
    def test_level_ffill_limit_and_no_future(self):
        cal = weekdays("2020-01-06", "2020-03-31")
        src_d = np.array([cal[10]], dtype="datetime64[D]")
        out = FC.align_level(src_d, np.array([7.0]), cal)
        self.assertTrue(np.all(np.isnan(out[:10])), "nothing may be filled backward in time (from the future)")
        self.assertTrue(np.all(out[10:16] == 7.0), "source row + 5 rows of carry")
        self.assertTrue(np.all(np.isnan(out[16:])), "carry stops after 5 rows")

    def test_level_uses_latest_source_not_after(self):
        cal = weekdays("2020-01-06", "2020-03-31")
        src_d = np.array([cal[3], cal[5]], dtype="datetime64[D]")
        out = FC.align_level(src_d, np.array([1.0, 2.0]), cal)
        self.assertEqual(list(out[3:6]), [1.0, 1.0, 2.0])
        self.assertTrue(np.isnan(out[2]))

    def test_flow_exact_match_only(self):
        cal = weekdays("2020-01-06", "2020-03-31")
        out = FC.align_flow(np.array([cal[4], cal[6]], dtype="datetime64[D]"), np.array([1.0, 2.0]), cal)
        self.assertEqual(out[4], 1.0)
        self.assertEqual(out[6], 2.0)
        self.assertEqual(int(np.isfinite(out).sum()), 2, "flows are never forward filled")

    def test_source_dates_not_in_calendar_are_dropped_for_flows(self):
        cal = weekdays("2020-01-06", "2020-03-31")
        sat = np.datetime64("2020-01-11", "D")
        self.assertNotIn(sat, cal)
        out = FC.align_flow(np.array([sat], dtype="datetime64[D]"), np.array([1.0]), cal)
        self.assertFalse(np.isfinite(out).any())

    def test_shift_rejects_negative(self):
        with self.assertRaises(ValueError):
            FC._shift(np.zeros((3, 1)), -1)


# ============================================================================= units and hand calculations
class FormulaTests(unittest.TestCase):
    def test_flow_ratio_hand_calc_and_window_rule(self):
        P = small_panel(T=40, volume=1000.0)
        net = {t: 100 for t in range(40)}
        f = feats_of(P, inst=inst_rows(P.cal, net))
        c1, c2 = f["fi_net_5"][:, 0], f["fi_net_20"][:, 0]
        self.assertAlmostEqual(c1[10], 100.0 * 5 / (1000.0 * 5))        # 0.1
        self.assertAlmostEqual(c2[30], 0.1)
        # truncated window at the panel start: rows 0..3 = 4 observed of 5 -> allowed; 3 observed -> NaN
        self.assertTrue(np.isnan(c1[2]))
        self.assertAlmostEqual(c1[3], 0.1)
        self.assertTrue(np.isnan(c2[14]))
        self.assertAlmostEqual(c2[15], 0.1)

    def test_missing_rows_in_window(self):
        P = small_panel(T=40, volume=1000.0)
        net = {t: 100 for t in range(40) if t not in (20, 22)}          # two holes inside the 5-row window ending at 22
        f = feats_of(P, inst=inst_rows(P.cal, net))
        c1 = f["fi_net_5"][:, 0]
        self.assertTrue(np.isnan(c1[22]), "3 of 5 observed -> NaN")
        self.assertAlmostEqual(c1[21], 0.1)                              # rows 17..21: 4 observed -> value
        # one hole: numerator and denominator use the same 4 rows
        net2 = {t: (100 if t != 20 else 999999) for t in range(40) if t != 21}
        f2 = feats_of(P, inst=inst_rows(P.cal, net2))
        self.assertAlmostEqual(f2["fi_net_5"][22, 0], (100 + 999999 + 100 + 100) / 4000.0)   # rows 18,19,20,22

    def test_trust_and_dealer_use_their_own_names(self):
        P = small_panel(T=30, volume=1000.0)
        rows = (inst_rows(P.cal, {t: 100 for t in range(30)}, name="Foreign_Investor")
                + inst_rows(P.cal, {t: 30 for t in range(30)}, name="Investment_Trust")
                + inst_rows(P.cal, {t: -20 for t in range(30)}, name="Dealer_self")
                + inst_rows(P.cal, {t: 9999 for t in range(30)}, name="Dealer_Hedging"))
        f = feats_of(P, inst=rows)
        self.assertAlmostEqual(f["fi_net_5"][10, 0], 0.1)
        self.assertAlmostEqual(f["it_net_5"][10, 0], 0.03)
        self.assertAlmostEqual(f["it_net_20"][25, 0], 0.03)
        self.assertAlmostEqual(f["dealer_net_5"][10, 0], -0.02, msg="Dealer_self only, hedging book excluded")

    def test_legacy_combined_dealer_gives_nan_dealer_self(self):
        P = small_panel(T=30)
        f = feats_of(P, inst=inst_rows(P.cal, {t: 50 for t in range(30)}, name="Dealer"))
        self.assertTrue(np.isnan(f["dealer_net_5"][:, 0]).all())

    def test_margin_units_lots_to_shares(self):
        P = small_panel(T=60, volume=1000.0)
        bal = {t: 100 + t for t in range(60)}                                   # lots, +1 lot per day
        f = feats_of(P, margin=margin_rows(P.cal, bal))
        # (balance[t]-balance[t-20]) = 20 lots = 20000 shares; denominator 20 * vol20 = 20 * 1000 = 20000 -> 1.0
        self.assertAlmostEqual(f["margin_chg_20"][40, 0], 1.0)
        self.assertTrue(np.isnan(f["margin_chg_20"][19, 0]))
        self.assertTrue(np.isnan(f["short_chg_20"][:, 0]).all(), "no short balance given")

    def test_short_balance(self):
        P = small_panel(T=60, volume=2000.0)
        bal = {t: 10 * t for t in range(60)}                                    # +10 lots per day -> 200 lots over 20 rows
        f = feats_of(P, margin=margin_rows(P.cal, bal, "ShortSaleTodayBalance"))
        self.assertAlmostEqual(f["short_chg_20"][50, 0], 200 * 1000.0 / (20 * 2000.0))   # 5.0

    def test_margin_ffill_five_rows_only(self):
        P = small_panel(T=80, volume=1000.0)
        bal = {t: 100 + t for t in range(80) if not (45 <= t <= 52)}            # an 8-row hole
        f = feats_of(P, margin=margin_rows(P.cal, bal))
        c = f["margin_chg_20"][:, 0]
        self.assertTrue(np.isfinite(c[44]))
        self.assertTrue(np.isfinite(c[49]))          # level carried from row 44, gap 5
        self.assertTrue(np.isnan(c[50]))             # gap 6 -> NaN
        self.assertTrue(np.isfinite(c[53]))

    def test_streak_values_cap_zero_break_and_nan(self):
        net = np.array([[1, 1, 1, -1, -1, 0, 5, np.nan, 3, 3] + [2] * 15], float).T       # (T, 1)
        s = FC.streak(net)[:, 0]
        self.assertEqual(list(s[:7]), [1, 2, 3, -1, -2, 0, 1])
        self.assertTrue(np.isnan(s[7]))
        self.assertEqual(list(s[8:10]), [1, 2], "a missing day restarts the run (no look beyond the gap)")
        self.assertEqual(s[-1], 10, "capped at +10")
        self.assertEqual(float(np.nanmax(s)), 10.0)
        neg = FC.streak(-np.ones((15, 1)))[:, 0]
        self.assertEqual(neg[-1], -10)

    def test_streak_feature_uses_trust_flow(self):
        P = small_panel(T=30)
        net = {t: (10 if t < 20 else -5) for t in range(30)}
        f = feats_of(P, inst=inst_rows(P.cal, net, name="Investment_Trust"))
        s = f["it_streak"][:, 0]
        self.assertEqual(s[3], 4)
        self.assertEqual(s[19], 10)
        self.assertEqual(s[20], -1)
        self.assertEqual(s[29], -10)

    def test_per_nonpositive_is_nan(self):
        P = small_panel(T=12)
        by = {0: (10.0, 2.0, 3.0), 1: (0.0, 0.0, 0.0), 2: (-5.0, -1.0, 1.5), 3: (None, None, None), 4: (20.0, 4.0, 0.0)}
        f = feats_of(P, per=per_rows(P.cal, by))
        ey, bp, dy = f["ey"][:, 0], f["bp"][:, 0], f["dy"][:, 0]
        self.assertAlmostEqual(ey[0], 0.1)
        self.assertAlmostEqual(bp[0], 0.5)
        self.assertEqual(dy[0], 3.0)
        self.assertTrue(np.isnan(ey[1]) and np.isnan(bp[1]))
        self.assertEqual(dy[1], 0.0, "a zero dividend yield is a value, not missing")
        self.assertTrue(np.isnan(ey[2]) and np.isnan(bp[2]))
        self.assertEqual(dy[2], 1.5)
        # row 3 is blank -> skipped by the parser path; ffill (<= 5 rows) carries row 2, i.e. still the negative PER -> NaN
        self.assertTrue(np.isnan(ey[3]))
        self.assertAlmostEqual(ey[4], 0.05)

    def test_missing_files_give_nan(self):
        P = small_panel(T=40, S=3)
        raw = {"S0": {}, "S1": None}                       # S2 not in raw at all
        f = FC.chip_features(P, raw)
        self.assertEqual(sorted(f), sorted(FC.NAMES))
        for k, v in f.items():
            self.assertEqual(v.shape, (40, 3))
            self.assertTrue(np.isnan(v).all(), k)
        cov = FC.coverage(f, P)
        self.assertTrue(all(c["n_stocks"] == 0 and c["first"] is None for c in cov.values()))

    def test_missing_dataset_only_affects_its_features(self):
        P = small_panel(T=40)
        f = feats_of(P, inst=inst_rows(P.cal, {t: 100 for t in range(40)}))
        self.assertTrue(np.isfinite(f["fi_net_5"][10, 0]))
        self.assertTrue(np.isnan(f["margin_chg_20"][:, 0]).all())
        self.assertTrue(np.isnan(f["ey"][:, 0]).all())

    def test_volume_missing_day_without_bar(self):
        P = small_panel(T=40, volume=1000.0)
        P.has_bar[20, 0] = False
        P.volume[20, 0] = NAN
        f = feats_of(P, inst=inst_rows(P.cal, {t: 100 for t in range(40)}))
        self.assertAlmostEqual(f["fi_net_5"][22, 0], 0.1, msg="the barless day drops out of numerator and denominator")


# ============================================================================= C9 extra lag
class C9LagTests(unittest.TestCase):
    def _c9(self, ratios):
        P = small_panel(T=60)
        return feats_of(P, hold=hold_rows(P.cal, ratios))["fi_hold_chg_20"][:, 0]

    def test_hand_calc(self):
        base = {t: 50.0 + 0.1 * t for t in range(60)}
        c9 = self._c9(base)
        # C9[t] = ratio[t-1] - ratio[t-21] = 20 rows * 0.1
        self.assertAlmostEqual(c9[40], 2.0)
        self.assertTrue(np.isnan(c9[20]), "needs ratio[t-21]")
        self.assertAlmostEqual(c9[21], 2.0)

    def test_day_t_input_does_not_affect_day_t(self):
        base = {t: 50.0 + 0.1 * t for t in range(60)}
        c9a = self._c9(base)
        t0 = 40
        for tt in (t0, t0 - 1, t0 - 21):
            mod = dict(base)
            mod[tt] += 7.0
            c9b = self._c9(mod)
            if tt == t0:
                self.assertEqual(c9b[t0], c9a[t0], "the ratio of day t must not reach C9[t]")
                self.assertNotEqual(c9b[t0 + 1], c9a[t0 + 1], "but it does reach C9[t+1]")
            elif tt == t0 - 1:
                self.assertNotEqual(c9b[t0], c9a[t0], "the ratio of day t-1 is used by C9[t]")
            else:
                self.assertNotEqual(c9b[t0], c9a[t0], "the ratio of day t-21 is the comparison base")

    def test_lag_is_after_ffill(self):
        # level only observed on row 30; ffill covers rows 30..35; C9 compares t-1 with t-21 -> needs both
        ratios = {10: 40.0, 30: 45.0}
        c9 = self._c9(ratios)
        self.assertAlmostEqual(c9[31], 5.0)        # ratio[30] (fresh) - ratio[10] (fresh)
        self.assertTrue(np.isnan(c9[37]), "ratio[36] is older than 5 rows -> NaN")


# ============================================================================= point in time
class PoisonFutureTests(unittest.TestCase):
    def test_future_poison_leaves_past_bitwise_identical(self):
        S, T = 6, 150
        P = small_panel(T=T, S=S, random=True, seed=4)
        rng = np.random.default_rng(1)
        cal = P.cal

        def build_raw(poison_after=None):
            raw = {}
            for j in range(S):
                inst_r, mg_r, hold_r, per_r = [], [], [], []
                for t in range(T):
                    poisoned = poison_after is not None and t > poison_after
                    r = np.random.default_rng(1000 * j + t)
                    scale = 1e3 if poisoned else 1.0
                    for nm in ("Foreign_Investor", "Investment_Trust", "Dealer_self"):
                        buy = float(r.integers(100, 5000)) * scale
                        inst_r.append({"date": str(cal[t]), "name": nm, "buy": buy, "sell": float(r.integers(100, 5000))})
                    mg_r.append({"date": str(cal[t]), "MarginPurchaseTodayBalance": float(r.integers(100, 900)) * scale,
                                 "ShortSaleTodayBalance": float(r.integers(10, 90)) * scale})
                    hold_r.append({"date": str(cal[t]), "ForeignInvestmentSharesRatio": float(r.uniform(20, 60)) * (-1 if poisoned else 1)})
                    per_r.append({"date": str(cal[t]), "PER": float(r.uniform(5, 30)) * scale, "PBR": float(r.uniform(1, 4)) * (0.01 if poisoned else 1.0),
                                  "dividend_yield": float(r.uniform(0, 6)) * scale})
                raw[f"S{j}"] = {"inst": FM.parse_institutional(inst_r), "margin": FM.parse_margin(mg_r),
                                "hold": FM.parse_shareholding(hold_r), "per": FM.parse_per(per_r)}
            return raw

        t0 = 90
        a = FC.chip_features(P, build_raw())
        Q = P.copy()
        Q.adj_open[t0 + 1:] = rng.uniform(1, 500, Q.adj_open[t0 + 1:].shape)
        Q.adj_close[t0 + 1:] = rng.uniform(1, 500, Q.adj_close[t0 + 1:].shape)
        Q.volume[t0 + 1:] = rng.uniform(1, 1e9, Q.volume[t0 + 1:].shape)
        Q.value[t0 + 1:] = rng.uniform(1, 1e12, Q.value[t0 + 1:].shape)
        b = FC.chip_features(Q, build_raw(poison_after=t0))
        for k in FC.NAMES:
            self.assertTrue(np.isfinite(a[k][:t0 + 1]).any(), f"{k} has no finite value to compare")
            self.assertTrue(np.array_equal(a[k][:t0 + 1], b[k][:t0 + 1], equal_nan=True), f"{k} changed in the past")
            self.assertFalse(np.array_equal(a[k][t0 + 1:], b[k][t0 + 1:], equal_nan=True), f"{k}: poison should move the future")
        # ranks inside the (prefix-only) universe too
        ma, mb = universe_mask(P, n=4, lookback=20, min_bars=30), universe_mask(Q, n=4, lookback=20, min_bars=30)
        self.assertTrue(np.array_equal(ma[:t0 + 1], mb[:t0 + 1]))
        ra, rb = FC.rank_features(a, ma), FC.rank_features(b, mb)
        for k in FC.NAMES:
            self.assertTrue(np.array_equal(ra[k][:t0 + 1], rb[k][:t0 + 1]), f"rank of {k} changed in the past")


class RankTests(unittest.TestCase):
    def test_rank_range_and_missing_zero(self):
        P = small_panel(T=30, S=5)
        F = np.tile(np.array([3.0, 1.0, 2.0, NAN, 5.0]), (30, 1))
        mask = np.ones((30, 5), bool)
        mask[:, 4] = False
        r = FC.rank_features({"x": F}, mask)["x"]
        self.assertEqual(list(r[0]), [0.5, -0.5, 0.0, 0.0, 0.0])      # NaN and out-of-mask -> 0
        self.assertTrue(((r >= -0.5) & (r <= 0.5)).all())


# ============================================================================= loading real-shaped fixtures
class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="twq_chips_")
        data.set_cache_dir(self.tmp)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(data.set_cache_dir, None)

    def _install(self, alias, code, fixture):
        obj = json.loads((FIX / fixture).read_text(encoding="utf-8"))
        path = FM.dataset_path(alias, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": "x", "dataset": FM.resolve_dataset(alias), "stock_id": code, "start_date": "x",
                                    "empty": False, "data": obj["data"]}), encoding="utf-8")

    def test_load_raw_fixtures_and_missing_files(self):
        for alias in ("inst", "margin", "hold", "per"):
            self._install(alias, "2330", f"finmind_{alias}_2330.json")
        raw = FC.load_raw(["2330", "9999"])
        self.assertEqual(sorted(raw["2330"]), ["hold", "inst", "margin", "per"])
        self.assertEqual(raw["9999"], {})
        inst = raw["2330"]["inst"]["2012-05-02"]
        self.assertEqual(inst["foreign_net"], 38841300 - 32897643)
        self.assertEqual(raw["2330"]["margin"]["2012-01-02"]["margin_balance"], 11087 * 1000.0)

    def test_fixture_aligns_to_calendar_and_values(self):
        for alias in ("inst", "margin", "hold", "per"):
            self._install(alias, "2330", f"finmind_{alias}_2330.json")
        raw = FC.load_raw(["2330"])
        cal = weekdays("2012-01-02", "2012-06-29")
        T = len(cal)
        close = np.full((T, 1), 100.0)
        vol = np.full((T, 1), 1e7)
        P = Panel(cal, ["2330"], close, close.copy(), vol, close * vol, np.ones((T, 1), bool))
        f = FC.chip_features(P, raw)
        row = int(np.flatnonzero(cal == np.datetime64("2012-01-02"))[0])
        per = raw["2330"]["per"]["2012-01-02"]
        self.assertAlmostEqual(f["ey"][row, 0], 1.0 / per["per"])
        self.assertAlmostEqual(f["dy"][row, 0], per["dividend_yield"])
        # only two inst dates in 2012 -> windows of 5 rows have < 4 observations -> NaN
        self.assertTrue(np.isnan(f["fi_net_5"][:, 0]).all())
        cov = FC.coverage(f, P)
        self.assertEqual(cov["ey"]["n_stocks"], 1)
        self.assertEqual(cov["fi_net_5"]["n_stocks"], 0)


# ============================================================================= run_chips smoke (synthetic, no OOS)
class RunSmokeTests(unittest.TestCase):
    def test_synthetic_run_writes_json_and_markdown_without_oos(self):
        tmp = Path(tempfile.mkdtemp(prefix="twq_runchips_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        looks = tmp / "LOOKS.md"
        datas, market = PB.synthetic_price_data(n_stocks=20, start="2011-06-01", end="2023-03-31", seed=2)
        panel = Panel.from_price_data(datas, "2012-01-01")
        raw = RC.synthetic_raw(panel, seed=5, frac=0.8)
        feats = FC.chip_features(panel, raw)
        ctrl = PB.price_features(datas, market, panel)
        out_json, md = tmp / "chips.json", tmp / "RESULTS.md"
        with mock.patch.object(H, "LOOKS_PATH", looks):
            res = RC.run(panel, feats, ctrl, n=12, min_bars=100, out_json=out_json, results_md=md, meta={"synthetic": True})
        self.assertFalse(looks.exists(), "an OOS look must never happen")
        self.assertFalse(res["meta"]["oos_touched"])
        self.assertEqual([v["id"] for v in res["features"].values()], [i for i, _, _ in FC.FEATURES])
        for name in FC.NAMES:
            for h in (5, 20):
                f = res["features"][name][f"h{h}"]
                self.assertEqual(f["screen"]["dev"]["period"], "dev")
                self.assertIn("passes", f["screen"])
                self.assertIn("residual", f)
                self.assertIn("basket", f)
        self.assertGreater(res["coverage"]["fi_net_5"]["n_stocks"], 0)
        self.assertLess(res["coverage"]["fi_net_5"]["n_stocks"], panel.S + 1)
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(sorted(saved["features"]), sorted(FC.NAMES))
        text = md.read_text(encoding="utf-8")
        self.assertIn(RC.SECTION_BEGIN, text)
        self.assertIn("## 2. 籌碼與評價", text)
        self.assertIn("覆蓋率", text)
        RC.update_results_md(md, RC.render_section(res))                 # idempotent
        self.assertEqual(md.read_text(encoding="utf-8").count(RC.SECTION_BEGIN), 1)

    def test_no_data_files_all_nan_still_runs(self):
        tmp = Path(tempfile.mkdtemp(prefix="twq_runchips_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        datas, market = PB.synthetic_price_data(n_stocks=12, start="2011-06-01", end="2023-03-31", seed=3)
        panel = Panel.from_price_data(datas, "2012-01-01")
        feats = FC.chip_features(panel, {})
        res = RC.run(panel, feats, PB.price_features(datas, market, panel), n=8, min_bars=100, out_json=tmp / "o.json",
                     results_md=tmp / "R.md")
        self.assertEqual(res["coverage"]["ey"]["n_stocks"], 0)
        self.assertFalse(res["features"]["ey"]["h5"]["screen"]["passes"])
        self.assertIn("無", (tmp / "R.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
