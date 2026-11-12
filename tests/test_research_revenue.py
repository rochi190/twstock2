"""Tests for research/feat_revenue.py and research/run_revenue.py (PLAN.md P2).  Synthetic / fixture data only, no network.

Run: python -m unittest discover -s tests -p "test_research_revenue.py"
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from backend import data  # noqa: E402
from backend.altdata import finmind as FM  # noqa: E402
from research import feat_revenue as FR  # noqa: E402
from research import harness as H  # noqa: E402
from research import price_baseline as PB  # noqa: E402
from research import run_revenue as RR  # noqa: E402
from research.panel import Panel  # noqa: E402

FIX = HERE / "fixtures" / "alt"
NAN = np.nan


def weekdays(start, end):
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    return days[((days.astype(np.int64) + 3) % 7) < 5]


def load_fixture(name):
    with open(FIX / name, encoding="utf-8") as fh:
        return json.load(fh)


def D64(s):
    return np.datetime64(s, "D")


def eq(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def rows_of(rev):
    """{(y, m): revenue} -> parsed rows like finmind.parse_monthly_revenue."""
    return [{"revenue_year": y, "revenue_month": m, "revenue": float(v), "date": None, "create_time": None}
            for (y, m), v in sorted(rev.items())]


def hand_rev():
    """2012: 100 every month; 2013: 110..200 with a dip in November and a spike in December."""
    rev = {(2012, m): 100.0 for m in range(1, 13)}
    for m, v in zip(range(1, 13), [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 150, 300]):
        rev[(2013, m)] = float(v)
    return rev


# ============================================================================= availability rule
class AvailabilityTests(unittest.TestCase):
    def test_tenth_is_a_trading_day_is_not_counted(self):
        # 2014-03-10 is a Monday: the 1st trading day AFTER it is Tue 03-11, the 2nd is Wed 03-12.
        cal = weekdays("2014-01-01", "2014-12-31")
        self.assertEqual(FR.revenue_available_day(2014, 2, cal), D64("2014-03-12"))

    def test_tenth_is_saturday(self):
        # 2012-03-10 is a Saturday: after it Mon 03-12 (1st), Tue 03-13 (2nd)
        cal = weekdays("2012-01-02", "2012-12-31")
        self.assertEqual(FR.revenue_available_day(2012, 2, cal), D64("2012-03-13"))

    def test_tenth_is_sunday(self):
        # 2014-08-10 is a Sunday
        cal = weekdays("2014-01-01", "2014-12-31")
        self.assertEqual(FR.revenue_available_day(2014, 7, cal), D64("2014-08-12"))

    def test_tenth_inside_a_holiday_block(self):
        # Mon 2014-02-10 .. Wed 02-12 closed: the trading days after the 10th are Thu 02-13 (1st), Fri 02-14 (2nd)
        cal = weekdays("2014-01-01", "2014-12-31")
        cal = cal[~((cal >= D64("2014-02-10")) & (cal <= D64("2014-02-12")))]
        self.assertEqual(FR.revenue_available_day(2014, 1, cal), D64("2014-02-14"))
        # long holiday that swallows the whole week: next trading days are 02-17 and 02-18
        cal2 = cal[~((cal >= D64("2014-02-13")) & (cal <= D64("2014-02-14")))]
        self.assertEqual(FR.revenue_available_day(2014, 1, cal2), D64("2014-02-18"))

    def test_second_day_skips_a_holiday(self):
        # 10th = Mon 03-10 (trading); Tue 03-11 trading (1st); Wed 03-12 closed -> 2nd is Thu 03-13
        cal = weekdays("2014-01-01", "2014-12-31")
        cal = cal[cal != D64("2014-03-12")]
        self.assertEqual(FR.revenue_available_day(2014, 2, cal), D64("2014-03-13"))

    def test_december_revenue_crosses_the_year(self):
        # December 2014 revenue is due 2015-01-10 (a Saturday): Mon 01-12 (1st), Tue 01-13 (2nd)
        cal = weekdays("2014-06-01", "2015-12-31")
        self.assertEqual(FR.revenue_available_day(2014, 12, cal), D64("2015-01-13"))
        self.assertEqual(FR.availability_row(2014, 12, cal), int(np.searchsorted(cal, D64("2015-01-13"))))

    def test_calendar_end_is_never_extrapolated(self):
        cal = weekdays("2014-01-01", "2014-03-11")          # ends on the 1st trading day after the 10th
        self.assertIsNone(FR.revenue_available_day(2014, 2, cal))
        self.assertIsNone(FR.availability_row(2014, 2, cal))
        cal = weekdays("2014-01-01", "2014-03-12")          # now the 2nd trading day is inside the calendar
        self.assertEqual(FR.revenue_available_day(2014, 2, cal), D64("2014-03-12"))
        self.assertIsNone(FR.revenue_available_day(2014, 3, cal))     # 10th of April beyond the end
        self.assertIsNone(FR.revenue_available_day(2014, 2, weekdays("2014-01-01", "2014-02-20")))
        self.assertIsNone(FR.revenue_available_day(2014, 2, np.array([], dtype="datetime64[D]")))

    def test_monotone_in_the_month(self):
        cal = weekdays("2010-01-01", "2020-12-31")
        prev = None
        for y in range(2010, 2020):
            for m in range(1, 13):
                d = FR.revenue_available_day(y, m, cal)
                if prev is not None:
                    self.assertGreater(d, prev)
                prev = d

    def test_bad_month(self):
        with self.assertRaises(ValueError):
            FR.revenue_available_day(2014, 13, weekdays("2014-01-01", "2014-12-31"))


# ============================================================================= features
class FeatureTests(unittest.TestCase):
    def test_hand_computed_values(self):
        tab = FR.monthly_feature_table(rows_of(hand_rev()))
        r1, r2, r3, r4 = tab[(2013, 12)]
        self.assertAlmostEqual(r1, 2.0)
        self.assertAlmostEqual(r2, 2.0 - np.mean([150 / 100 - 1, 200 / 100 - 1, 190 / 100 - 1]))       # 2.0 - 0.8
        self.assertAlmostEqual(r3, (300 + 150 + 200) / 300 - 1)
        self.assertEqual(r4, 1.0)
        r1, r2, r3, r4 = tab[(2013, 11)]
        self.assertAlmostEqual(r1, 0.5)
        self.assertEqual(r4, 0.0)                    # 150 is below the 200 of October
        self.assertAlmostEqual(r2, 0.5 - np.mean([1.0, 0.9, 0.8]))
        self.assertAlmostEqual(tab[(2013, 3)][0], 0.3)

    def test_missing_last_year_is_nan(self):
        tab = FR.monthly_feature_table(rows_of(hand_rev()))
        for m in range(1, 13):                       # 2012 has no 2011 -> no YoY at all
            r = tab[(2012, m)]
            self.assertTrue(np.isnan(r[0]) and np.isnan(r[1]) and np.isnan(r[2]))
        self.assertTrue(np.isnan(tab[(2013, 1)][1]))     # R2 needs YoY of 2012-12..2012-10 (undefined)
        self.assertTrue(np.isnan(tab[(2013, 2)][1]))
        self.assertTrue(np.isnan(tab[(2013, 3)][1]))
        self.assertFalse(np.isnan(tab[(2013, 4)][1]))   # YoY of 2013-03..01 exist now
        self.assertTrue(np.isnan(tab[(2012, 6)][3]))     # fewer than 12 months -> R4 NaN
        self.assertEqual(tab[(2012, 12)][3], 1.0)        # all equal: ties count as the 12-month high
        # a hole in the data -> NaN only where the window needs it
        rev = hand_rev()
        del rev[(2012, 7)]
        t2 = FR.monthly_feature_table(rows_of(rev))
        self.assertTrue(np.isnan(t2[(2013, 7)][0]))      # last-year same month missing
        self.assertFalse(np.isnan(t2[(2013, 8)][0]))
        self.assertFalse(np.isnan(t2[(2013, 6)][2]))     # 3m-yoy of 2013-06 needs 2012-04..06 (all present)
        for m in (7, 8, 9):                              # 2012-07 sits in the year-ago window of 2013-07..09
            self.assertTrue(np.isnan(t2[(2013, m)][2]), m)
        self.assertFalse(np.isnan(t2[(2013, 10)][2]))
        self.assertFalse(np.isnan(t2[(2013, 12)][1]))    # R2 of 2013-12 uses the YoY of 2013-09..11 only

    def test_clip_and_nonpositive_base(self):
        rev = {(2012, 1): 100.0, (2013, 1): 1000.0, (2012, 2): 100.0, (2013, 2): 5.0, (2012, 3): 0.0, (2013, 3): 50.0,
               (2012, 4): 100.0, (2013, 4): 0.0}
        tab = FR.monthly_feature_table(rows_of(rev))
        self.assertEqual(tab[(2013, 1)][0], 3.0)                       # +900 % -> 3
        self.assertEqual(tab[(2013, 2)][0], -0.9)                      # -95 % -> -0.9
        self.assertTrue(np.isnan(tab[(2013, 3)][0]))                   # zero base
        self.assertEqual(tab[(2013, 4)][0], -0.9)                      # revenue 0 -> -100 % -> clipped

    def test_feature_switches_on_at_the_availability_day_and_carries_forward(self):
        cal = weekdays("2013-01-01", "2014-12-31")
        f = FR.stock_features(rows_of(hand_rev()), cal)
        a_mar = FR.availability_row(2013, 3, cal)          # April-10 + 2 trading days
        a_apr = FR.availability_row(2013, 4, cal)
        self.assertTrue(a_mar < a_apr)
        self.assertAlmostEqual(f[a_mar, 0], 0.3)           # R1 of March from the availability day on
        self.assertAlmostEqual(f[a_apr - 1, 0], 0.3)       # carried forward until April becomes available
        self.assertAlmostEqual(f[a_apr, 0], 0.4)
        self.assertTrue(np.isnan(f[a_mar - 1, 0]) or abs(f[a_mar - 1, 0] - 0.2) < 1e-12)   # the day before: February (0.2)
        self.assertAlmostEqual(f[a_mar - 1, 0], 0.2)
        # before any 2013 month is available the 2012 values are NaN (no 2011 data)
        self.assertTrue(np.isnan(f[0, 0]))
        # the last rows carry December 2013 (R1 = 2.0, R4 = 1)
        self.assertAlmostEqual(f[-1, 0], 2.0)
        self.assertEqual(f[-1, 3], 1.0)

    def test_fixture_field_semantics_and_values(self):
        raw = load_fixture("finmind_rev_2330.json")["data"]
        # `date` is the 1st of the month AFTER the revenue month; revenue_year / revenue_month are the revenue month
        for r in raw:
            y, m = r["revenue_year"], r["revenue_month"]
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            self.assertEqual(r["date"], f"{ny:04d}-{nm:02d}-01")
        rows = FM.parse_monthly_revenue(raw)
        tab = FR.monthly_feature_table(rows)
        self.assertTrue(all(np.isnan(v[0]) for v in tab.values()))       # fixture has no year-ago partner at all
        cal = weekdays("2026-01-01", "2026-12-31")
        # 2026-10 revenue: 2026-11-10 is a Tuesday -> 11-11 (1st), 11-12 (2nd)
        self.assertEqual(FR.revenue_available_day(2026, 10, cal), D64("2026-11-12"))
        # ...and the real publication (create_time 2026-11-10) is before that day: the rule is conservative
        self.assertLessEqual("2026-11-10", "2026-11-12")
        # the rule ignores create_time: wipe it and nothing changes
        for r in raw:
            r["create_time"] = "9999-01-01"
        self.assertEqual(FR.monthly_feature_table(FM.parse_monthly_revenue(raw)).keys(), tab.keys())

    def test_unsorted_and_duplicate_rows_are_handled_by_the_parser(self):
        raw = [{"revenue_year": 2013, "revenue_month": 1, "revenue": 110, "date": "2013-02-01", "create_time": ""},
               {"revenue_year": 2012, "revenue_month": 1, "revenue": 100, "date": "2012-02-01", "create_time": ""}]
        tab = FR.monthly_feature_table(FM.parse_monthly_revenue(raw))
        self.assertAlmostEqual(tab[(2013, 1)][0], 0.1)


# ============================================================================= panel level
class PanelLevelTests(unittest.TestCase):
    def test_missing_file_is_nan_and_coverage(self):
        cal = weekdays("2013-01-01", "2014-12-31")
        rows = {"A": rows_of(hand_rev()), "B": None, "C": rows_of(hand_rev())}
        feats, cov = FR.revenue_features(["A", "B", "C"], cal, loader=rows.get)
        self.assertEqual(cov, {"n_codes": 3, "n_with_file": 2, "codes_with_file": ["A", "C"]})
        for n in FR.NAMES:
            self.assertEqual(feats[n].shape, (len(cal), 3))
            self.assertTrue(np.isnan(feats[n][:, 1]).all())
            self.assertTrue(eq(feats[n][:, 0], feats[n][:, 2]))
        self.assertAlmostEqual(feats["rev_yoy"][-1, 0], 2.0)

    def test_load_rows_reads_the_cache_and_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            data.set_cache_dir(tmp)
            try:
                self.assertIsNone(FR.load_rows("2330"))
                FM._store("rev", "2330", "2010-01-01", load_fixture("finmind_rev_2330.json")["data"])
                rows = FR.load_rows("2330")
                self.assertEqual(len(rows), 17)
                self.assertEqual((rows[-1]["revenue_year"], rows[-1]["revenue_month"]), (2026, 10))
            finally:
                data.set_cache_dir(None)

    def test_poisoned_future_leaves_the_past_bit_identical(self):
        """Replace every revenue row that is not yet available at t0 (and append fake future months) and rewrite the future
        prices: features on rows <= t0 must be bitwise identical.  Also: truncating the calendar at t0 changes nothing <= t0."""
        rng = np.random.default_rng(3)
        cal = weekdays("2011-01-03", "2016-12-30")
        codes = [f"S{j}" for j in range(5)]
        rows = {}
        for c in codes:
            rev = {(y, m): float(np.exp(rng.normal(10, 0.3))) for y in range(2010, 2017) for m in range(1, 13)}
            rows[c] = rows_of(rev)
        feats, _ = FR.revenue_features(codes, cal, loader=rows.get)
        t0 = int(np.searchsorted(cal, D64("2014-06-17")))
        poisoned = {}
        for c in codes:
            new = []
            for r in rows[c]:
                a = FR.availability_row(r["revenue_year"], r["revenue_month"], cal)
                if a is None or a > t0:
                    r = dict(r, revenue=float(rng.uniform(1, 1e6)))      # future (or never available) -> garbage
                new.append(r)
            new += [{"revenue_year": 2017 + k, "revenue_month": 1 + k, "revenue": 1e9, "date": None, "create_time": None}
                    for k in range(3)]
            poisoned[c] = new
        pf, _ = FR.revenue_features(codes, cal, loader=poisoned.get)
        # prices are not an input of the features at all; build a panel with garbage future prices to make the point explicit
        T = len(cal)
        px = np.full((T, 5), 50.0)
        px2 = px.copy()
        px2[t0 + 1:] = rng.uniform(1, 500, (T - t0 - 1, 5))
        p1 = Panel(cal, codes, px, px, np.ones((T, 5)), px, np.ones((T, 5), bool))
        p2 = Panel(cal, codes, px2, px2, np.ones((T, 5)), px2, np.ones((T, 5), bool))
        f1, _ = FR.revenue_features(list(p1.codes), p1.cal, loader=rows.get)
        f2, _ = FR.revenue_features(list(p2.codes), p2.cal, loader=poisoned.get)
        for n in FR.NAMES:
            self.assertTrue(eq(feats[n][:t0 + 1], pf[n][:t0 + 1]), n)
            self.assertTrue(eq(f1[n][:t0 + 1], f2[n][:t0 + 1]), n)
            self.assertFalse(eq(feats[n][t0 + 1:], pf[n][t0 + 1:]), n)      # the poison really changed the future
        short, _ = FR.revenue_features(codes, cal[:t0 + 1], loader=rows.get)   # calendar ends at t0
        for n in FR.NAMES:
            self.assertTrue(eq(feats[n][:t0 + 1], short[n]), n)

    def test_section_markers_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "RESULTS.md"
            PB.update_results_md(p, "## 1. 價格基準\nA\n")
            RR.update_results_md(p, "## 3. 月營收\nfirst\n")
            t1 = p.read_text(encoding="utf-8")
            RR.update_results_md(p, "## 3. 月營收\nfirst\n")
            self.assertEqual(p.read_text(encoding="utf-8"), t1)                      # same section twice: no change
            RR.update_results_md(p, "## 3. 月營收\nsecond\n")
            t2 = p.read_text(encoding="utf-8")
            self.assertIn("second", t2)
            self.assertNotIn("first", t2)
            self.assertEqual(t2.count(RR.SECTION_BEGIN), 1)
            self.assertEqual(t2.count(RR.SECTION_END), 1)
            self.assertIn("A\n", t2)                                                 # the price-baseline section survives
            self.assertEqual(t2.count(PB.SECTION_BEGIN), 1)
            PB.update_results_md(p, "## 1. 價格基準\nB\n")                            # and the other way round
            t3 = p.read_text(encoding="utf-8")
            self.assertIn("second", t3)
            self.assertIn("B\n", t3)
            self.assertNotIn("\nA\n", t3)


# ============================================================================= event study / run
def synthetic_rows(codes, missing, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for c in codes:
        if c in missing:
            out[c] = None
            continue
        base = np.exp(rng.normal(10, 0.5))
        out[c] = rows_of({(y, m): base * np.exp(rng.normal(0, 0.35) + 0.01 * (y - 2010) * 12 + 0.05 * np.sin(m))
                          for y in range(2010, 2024) for m in range(1, 13)})
    return out


def make_panel(S, start="2011-01-03", end="2023-06-30", seed=0):
    rng = np.random.default_rng(seed)
    cal = weekdays(start, end)
    T = len(cal)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (T, S)), axis=0))
    op = np.empty_like(close)
    op[0] = close[0]
    op[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, (T - 1, S)))
    vol = np.exp(rng.normal(13, 0.4, (T, S)))
    return Panel(cal, [f"S{j:03d}" for j in range(S)], op, close, vol, close * vol, np.ones((T, S), bool))


class EventStudyTests(unittest.TestCase):
    def test_planted_drift_after_the_availability_day_is_recovered(self):
        S = 40
        panel = make_panel(S, seed=11)
        codes = [str(c) for c in panel.codes]
        rows = synthetic_rows(codes, missing=set(), seed=7)
        mask = np.ones((panel.T, S), bool)
        delta, span = 0.003, 10
        logo = np.log(panel.adj_open)
        for (y, m, a) in RR.event_months(panel.cal, rows):
            if a + span + 2 >= panel.T:
                continue
            r1 = np.array([FR.monthly_feature_table(rows[c])[(y, m)][0] for c in codes])
            if not np.isfinite(r1).any():
                continue
            high = np.isfinite(r1) & (r1 >= np.nanmedian(r1))
            # holding days a+1 -> a+2 ... : every high-R1 stock gains +delta per day for `span` days after the entry open
            for k in range(1, span + 1):
                logo[a + 1 + k:, high] += delta
        planted = Panel(panel.cal, panel.codes, np.exp(logo), panel.adj_close, panel.volume, panel.value, panel.has_bar)
        ev = RR.event_study(planted, mask, rows)
        self.assertGreater(ev["n_events"], 100)
        self.assertEqual(ev["kmax"], 20)
        # through day k <= span the spread between the top and the bottom quintile is ~ k * delta, flat afterwards
        for k in (1, 5, 10):
            self.assertAlmostEqual(ev["q5_minus_q1"][k - 1], k * delta, delta=0.0015 * k + 0.002)
        self.assertGreater(ev["q5_minus_q1_t"][9], 8.0)
        self.assertAlmostEqual(ev["q5_minus_q1"][19], span * delta, delta=0.006)
        # excess of the five groups sums to ~ 0 (the benchmark is the universe mean)
        tot = [sum(ev["quantiles"][q][9] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")) for _ in (0,)]
        self.assertLess(abs(tot[0]), 0.01)

    def test_no_event_without_own_revenue_row_and_dev_only(self):
        panel = make_panel(40, seed=2)
        codes = [str(c) for c in panel.codes]
        rows = synthetic_rows(codes, missing=set(codes[:10]))
        ev = RR.event_study(panel, np.ones((panel.T, 40), bool), rows)
        self.assertAlmostEqual(ev["mean_cohort_size"], 30.0, delta=0.01)           # the 10 stocks without a file never enter
        # last event month must have its whole 20-day window inside DEV (decision day <= 2022-12-30 minus 21 rows)
        self.assertLessEqual(ev["last_event"], "2022-11")
        self.assertGreaterEqual(ev["first_event"], "2012-12")


class RunTests(unittest.TestCase):
    def test_run_on_synthetic_panel_with_missing_files(self):
        datas, _ = PB.synthetic_price_data(n_stocks=40, start="2011-01-03", end="2023-06-30", seed=4)
        panel = Panel.from_price_data(datas, "2012-01-01")
        codes = [str(c) for c in panel.codes]
        missing = set(codes[:4])
        rows = synthetic_rows(codes, missing)
        controls = RR.price_controls(datas, panel)
        for k in RR.CONTROL_NAMES:
            self.assertEqual(controls[k].shape, (panel.T, panel.S))
        with tempfile.TemporaryDirectory() as tmp:
            md, js = Path(tmp) / "RESULTS.md", Path(tmp) / "revenue_dev.json"
            res = RR.run(panel, rows, controls, n=150, min_bars=200, out_json=js, results_md=md)
            self.assertEqual(res["meta"]["n_with_revenue_file"], 36)
            self.assertFalse(res["meta"]["oos_touched"])
            self.assertAlmostEqual(res["meta"]["dev_file_coverage"], 36 / 40, delta=0.03)
            for name in FR.NAMES:
                for h in (5, 20):
                    f = res["features"][name][f"h{h}"]
                    self.assertEqual(f["screen"]["dev"]["period"], "dev")
                    self.assertEqual(f["residual"]["controls"], RR.CONTROL_NAMES)
                    self.assertIn("passes", f["screen"])
                self.assertGreater(res["features"][name]["h5"]["screen"]["dev"]["n_days"], 1000)
            # no skill by construction: the primary-horizon R1 IC is small
            self.assertLess(abs(res["features"]["rev_yoy"]["h5"]["screen"]["dev"]["mean"]), 0.05)
            parsed = json.loads(js.read_text(encoding="utf-8"))
            self.assertEqual(parsed["event_study"]["kmax"], 20)
            text = md.read_text(encoding="utf-8")
            self.assertEqual(text.count(RR.SECTION_BEGIN), 1)
            self.assertIn("## 3. 月營收", text)
            self.assertIn("R1 `rev_yoy`", text)
            RR.run(panel, rows, controls, n=150, min_bars=200, out_json=None, results_md=md)     # rerun: still one block
            self.assertEqual(md.read_text(encoding="utf-8").count(RR.SECTION_BEGIN), 1)
            # a stock without a file stays NaN in the feature matrices
            feats, _ = FR.revenue_features(codes, panel.cal, loader=lambda c: rows.get(c))
            self.assertTrue(np.isnan(feats["rev_yoy"][:, :4]).all())

    def test_oos_is_never_requested(self):
        src = (HERE.parent / "research" / "run_revenue.py").read_text(encoding="utf-8")
        body = src.split('"""', 2)[2]                       # code after the module docstring
        self.assertNotIn("oos", body.replace("oos_touched", ""))
        self.assertNotIn("spec_hash", body)


if __name__ == "__main__":
    unittest.main()
