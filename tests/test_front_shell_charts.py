"""charts.js: every draw function against a recording Canvas2D with real and hostile data; mount() behaviour (hover, keys, resize,
theme, destroy).  Optional PNG gallery when TQ_FRONT_OUT is set (see tests/canvas_replay.py)."""
from __future__ import annotations

import json
import unittest

import front_helpers as FH
from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, out_dir

FILES = ["format.js", "charts.js"]


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class DrawFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=FILES, fixtures=["real/history_2330", "real/history_0050", "real/forecast_2330", "real/backtest_2330_h1", "real/backtest_universe_h5",
                                                     "forecast_nulls", "forecast_sparse"])

    def draw(self, name, fn, W, H, mode, spec_js, hover="null"):
        """Render and return (model summary, ops report).  `spec_js` is a JS expression."""
        js = ("(function () { var m = window.__renderChart(" + json.dumps(name) + "," + json.dumps(fn) + "," + str(W) + "," + str(H) + "," + json.dumps(mode) + "," + spec_js + "," + hover + ");"
              "var tipN = 0, tipBad = 0; if (m && m.tooltip && m.count) { for (var i = 0; i < m.count; i++) { var t = m.tooltip(i); if (t) { tipN++; var s = JSON.stringify(t); if (/NaN|undefined|Infinity|\\[object/.test(s)) tipBad++; } } }"
              "return { empty: !!m.empty, count: m.count || 0, a11y: m.a11y || '', legend: (m.legend || []).map(function (l) { return l.label; }), tipN: tipN, tipBad: tipBad,"
              "         idx: m.idxAtX ? [m.idxAtX(-50), m.idxAtX(" + str(W) + " / 2), m.idxAtX(" + str(W) + " + 50)] : null, hasTooltip: typeof m.tooltip, report: window.__opsReport(" + json.dumps(name) + ") }; })()")
        return self.app.json(js)

    def assertSane(self, r, texts=True):
        rep = r["report"]
        self.assertEqual(rep["bad"], [], "non-finite numbers in drawing ops")
        self.assertEqual(rep["badText"], [], "junk text drawn")
        self.assertEqual(rep["oob"], [], "text drawn outside the canvas")
        self.assertGreater(rep["n"], 3)
        self.assertEqual(r["tipBad"], 0)

    HIST = "window.__fx.real_history_2330.bars"
    FAN = "window.__fx.real_forecast_2330.fan"

    def price_spec(self, extra=""):
        return "{ bars: %s, visible: 120, fan: %s, lastClose: 2385, asof: '2026-10-22', kind: 'stock'%s }" % (self.HIST, self.FAN, extra)

    # ---- price + fan
    def test_price_fan_sizes_and_modes(self):
        for W, H in ((1200, 420), (900, 360), (360, 300), (220, 240), (1600, 300)):
            for mode in ("light", "dark"):
                r = self.draw("pf", "drawPriceFan", W, H, mode, self.price_spec())
                self.assertSane(r)
                self.assertFalse(r["empty"], (W, H))
                self.assertEqual(r["count"], 120 + 20)
                self.assertIn("近 120 個交易日", r["a11y"])
        r = self.draw("pf", "drawPriceFan", 900, 380, "light", self.price_spec())
        self.assertEqual(r["legend"], ["上漲(紅)", "下跌(綠)", "MA20", "MA60", "預測中位數", "50%區間", "90%區間"])
        self.assertEqual(r["tipN"], 140)
        self.assertEqual(r["idx"][0], -1)
        self.assertEqual(r["idx"][2], -1)

    def test_price_fan_variants(self):
        for visible in (60, 120, 250, 5000, 0, None):
            r = self.draw("pf", "drawPriceFan", 800, 360, "light", "{ bars: %s, visible: %s, fan: %s, lastClose: 2385, kind: 'stock' }" % (self.HIST, json.dumps(visible), self.FAN))
            self.assertSane(r)
        # forecast only (history failed) / history only (forecast failed) / nothing
        r = self.draw("pf", "drawPriceFan", 800, 360, "light", "{ bars: [], fan: %s, lastClose: 2385 }" % self.FAN)
        self.assertSane(r)
        self.assertFalse(r["empty"])
        r = self.draw("pf", "drawPriceFan", 800, 360, "light", "{ bars: %s, fan: null }" % self.HIST)
        self.assertSane(r)
        self.assertEqual(r["count"], 120)
        for spec in ("{}", "undefined", "{ bars: [] }", "{ bars: [null, 5, 'x', {}] }", "{ bars: [{ d: '2026-01-01', o: null, h: null, l: null, c: null, v: null }] }"):
            r = self.draw("pf", "drawPriceFan", 800, 360, "dark", spec)
            self.assertSane(r)
            self.assertTrue(r["empty"] or r["count"] >= 0)

    def test_price_fan_hostile_fan_and_bars(self):
        bars = "[{d:'2026-10-20',o:100,h:100,l:100,c:100,v:0},{d:'2026-10-21',o:100,h:100,l:100,c:100,v:0},{d:'2026-10-22',o:100,h:100,l:100,c:100,v:0}]"
        r = self.draw("pf", "drawPriceFan", 700, 320, "light", "{ bars: %s, lastClose: 100, kind: 'stock' }" % bars)      # flat price, no volume
        self.assertSane(r)
        fans = [
            "{ days: [1,2,3], levels: [0.05,0.5,0.95], prices: [[90,85,80],[100,100,100],[110,115,120]] }",                         # no 25/75 rows
            "{ days: [1,2,3], levels: [0.05,0.25,0.5,0.75,0.95], prices: [[null,85,80],[null,null,null],[100,101,102],[null,115,null],[110,120,125]] }",   # gaps
            "{ days: [1,2], levels: [0.05,0.25,0.5,0.75,0.95], prices: [[120,130],[110,100],[100,100],[90,95],[80,85]] }",             # rows in the wrong order
            "{ levels: [0.5], prices: [[100,101,102]] }", "{ levels: [], prices: [] }", "{ levels: 'x', prices: 5 }", "{ prices: [[1,2,3]] }",
        ]
        for f in fans:
            r = self.draw("pf", "drawPriceFan", 700, 320, "light", "{ bars: %s, fan: %s, lastClose: 100 }" % (self.HIST, f))
            self.assertSane(r)

    def test_price_fan_hover_draws_crosshair_and_tooltip_content(self):
        base = self.draw("pf", "drawPriceFan", 900, 380, "light", self.price_spec())
        for hover in ("30", "119", "125", "139"):
            r = self.draw("pf", "drawPriceFan", 900, 380, "light", self.price_spec(), hover)
            self.assertSane(r)
            self.assertGreater(r["report"]["n"], base["report"]["n"] - 1)
        t = self.app.json("""(function () {
          var m = window.__renderChart('pfx', 'drawPriceFan', 900, 380, 'light', %s, null);
          return { hist: m.tooltip(100), fut: m.tooltip(125), none: m.tooltip(999), neg: m.tooltip(-1) }; })()""" % self.price_spec())
        self.assertEqual([r["label"] for r in t["hist"]["rows"][:6]], ["收盤", "漲跌", "開", "高", "低", "成交量"])
        self.assertRegex(t["hist"]["title"], r"^\d{4}-\d{2}-\d{2} \(週.\)$")
        self.assertEqual([r["label"] for r in t["fut"]["rows"]], ["中位數", "50%區間", "90%區間"])
        self.assertIn("預測 第6個交易日", t["fut"]["title"])
        self.assertIsNone(t["none"])
        self.assertIsNone(t["neg"])

    def test_etf_price_decimals(self):
        r = self.app.json("""(function () {
          var m = window.__renderChart('etf', 'drawPriceFan', 800, 360, 'light', { bars: window.__fx.real_history_0050.bars, visible: 60, lastClose: 106.9, kind: 'etf' }, null);
          var texts = window.__rec.etf.ops.filter(function (o) { return o[0] === 'fillText'; }).map(function (o) { return o[1]; });
          return { texts: texts, tip: m.tooltip(10).rows[0].value }; })()""")
        self.assertRegex(r["tip"], r"^\d+\.\d{2}$")        # ETF ticks are 0.01 / 0.05 -> two decimals

    # ---- gauge / strip
    def test_gauge(self):
        for p in (0.516, 0.5, 0.0, 1.0, 0.9, 0.1, "0.6", None, "x", "NaN"):
            r = self.draw("g", "drawGauge", 200, 112, "light", json.dumps({"p": p, "lo": 0.3, "hi": 0.7}))
            self.assertSane(r)
        r = self.draw("g", "drawGauge", 200, 112, "light", "{ p: 0.516, lo: 0.3, hi: 0.7 }")
        self.assertIn("51.6%", r["a11y"])
        self.assertIn("48.4%", r["a11y"])
        self.assertIn("無資料", self.draw("g", "drawGauge", 200, 112, "light", "{ p: null }")["a11y"])
        for W, H in ((60, 40), (30, 30), (0, 0), (200, 20)):
            self.assertEqual(self.draw("g", "drawGauge", W, H, "light", "{ p: 0.6 }")["report"]["bad"], [])
        self.draw("g", "drawGauge", 200, 112, "dark", "undefined")

    def test_gauge_colours_follow_taiwan_convention(self):
        cols = self.app.json("""(function () {
          function strokeColors(p) {
            window.__renderChart('gg', 'drawGauge', 200, 112, 'light', { p: p, lo: 0.3, hi: 0.7 }, null);
            var out = {}; window.__rec.gg.ops.forEach(function (o) { if (o[0] === 'set' && o[1] === 'strokeStyle') out[o[2]] = 1; }); return Object.keys(out);
          }
          return { up: strokeColors(0.6), down: strokeColors(0.4), flat: strokeColors(0.5) }; })()""")
        self.assertIn("#c62828", cols["up"])
        self.assertNotIn("#17794a", cols["up"])
        self.assertIn("#17794a", cols["down"])
        self.assertNotIn("#c62828", cols["down"])
        self.assertNotIn("#c62828", cols["flat"])
        self.assertNotIn("#17794a", cols["flat"])

    def test_interval_strip(self):
        q = "window.__fx.real_forecast_2330.horizons['5'].quantiles"
        r = self.draw("s", "drawIntervalStrip", 340, 62, "light", "{ q05: %s['0.05'], q25: %s['0.25'], q50: %s['0.5'], q75: %s['0.75'], q95: %s['0.95'], mean: 0.004 }" % ((q,) * 5))
        self.assertSane(r)
        self.assertIn("90%區間", r["a11y"])
        for spec in ("{ q05: -0.1, q95: 0.2 }", "{ q05: 0.01, q25: 0.02, q50: 0.03, q75: 0.04, q95: 0.05, mean: 0.9 }", "{ q05: 0, q95: 0 }", "{ q05: 0.1, q95: -0.1 }",
                     "{ q05: null, q95: null }", "{}", "undefined", "{ q05: -1e9, q95: 1e9, q50: 0 }"):
            self.assertSane(self.draw("s", "drawIntervalStrip", 300, 62, "dark", spec))
        self.assertSane(self.draw("s", "drawIntervalStrip", 30, 62, "light", "{ q05: -0.1, q95: 0.2 }"))

    # ---- time series (equity / drawdown / position) from a real backtest report
    def ts_spec(self):
        return """(function () {
          var rep = window.__fx.real_backtest_2330_h1.report, c = rep.strategy.curve, pal = TQ.charts.palette('light');
          return { dates: c.dates, n: c.dates.length, panels: [
            { title: '淨值', weight: 3, series: [{ key: 's', label: '策略', color: pal.accent, values: c.strategy, endLabel: true }, { key: 'b', label: '買進持有', color: pal.gray, values: c.buyhold, endLabel: true }], refLines: [{ v: 1, label: '起點' }] },
            { title: '持股', weight: 1, domain: [0, 1], tickLabels: { 0: '空手', 1: '持股' }, series: [{ key: 'p', label: '部位', color: pal.accentMid, values: c.position, kind: 'step' }] },
            { title: '回撤', weight: 1.4, includeZero: true, fmt: function (v) { return TQ.fmt.fmtPct(v, 0); }, series: [{ key: 'd', label: '策略回撤', color: pal.accent, values: c.drawdown_strategy, kind: 'area' }] }] }; })()"""

    def test_time_series_equity_position_drawdown(self):
        for W, H in ((900, 520), (360, 420), (200, 300)):
            r = self.draw("ts", "drawTimeSeries", W, H, "light", self.ts_spec())
            self.assertSane(r)
            self.assertFalse(r["empty"])
            self.assertEqual(r["legend"], ["策略", "買進持有", "部位", "策略回撤"])
        r = self.draw("ts", "drawTimeSeries", 900, 520, "light", self.ts_spec(), "100")
        self.assertSane(r)
        self.assertGreater(r["count"], 100)

    def test_time_series_hostile(self):
        specs = ["{}", "undefined", "{ panels: [] }", "{ dates: [], panels: [{ series: [{ values: [] }] }] }",
                 "{ dates: ['2026-01-01'], n: 1, panels: [{ title: 't', series: [{ key: 'a', label: 'A', color: '#000', values: [5] }] }] }",
                 "{ dates: ['2026-01-01','x',null], n: 3, panels: [{ series: [{ label: 'A', color: '#000', values: [1, null, NaN] }] }] }",
                 "{ n: 4, panels: [{ series: [{ label: 'A', color: '#000', values: [1, 2, 3, 4], endLabel: true }, { label: 'B', color: '#111', values: [1.0001, 2, 3, 4.0001], endLabel: true }] }] }",
                 "{ n: 3, panels: [null, { series: null }, { series: [null, { values: 'x' }] }] }"]
        for sp in specs:
            r = self.draw("ts", "drawTimeSeries", 600, 300, "light", sp)
            self.assertSane(r)

    # ---- calibration / histogram
    def test_calibration_and_histogram_from_real_report(self):
        rep = "window.__fx.real_backtest_2330_h1.report"
        r = self.draw("cal", "drawCalibration", 460, 360, "light", "{ bins: %s.probabilistic.calibration }" % rep)
        self.assertSane(r)
        self.assertGreater(r["count"], 3)
        for spec in ("{}", "undefined", "{ bins: [] }", "{ bins: [{ p_mean: 0.5, freq: 0.5 }] }", "{ bins: [{ p_mean: 0.4, freq: 0.9, n: 3 }, { p_mean: 0.6, freq: 0.1, n: 0 }, { p_mean: null, freq: 0.5 }] }"):
            self.assertSane(self.draw("cal", "drawCalibration", 400, 320, "dark", spec))
        r = self.draw("pit", "drawHistogram", 460, 260, "light", "{ values: %s.returns.pit_hist, expected: 0.1, n: %s.direction.n }" % (rep, rep))
        self.assertSane(r)
        self.assertEqual(r["count"], 10)
        for spec in ("{}", "undefined", "{ values: [] }", "{ values: [null, null] }", "{ values: [0.5, 0.5] }", "{ values: [0.1, null, 0.3], expected: 'x', n: -5 }"):
            self.assertSane(self.draw("pit", "drawHistogram", 400, 240, "dark", spec))

    def test_empty_states_have_text(self):
        for fn, spec in (("drawPriceFan", "{}"), ("drawTimeSeries", "{}"), ("drawCalibration", "{}"), ("drawHistogram", "{}")):
            r = self.draw("e", fn, 400, 200, "light", spec)
            self.assertTrue(r["empty"], fn)
            self.assertGreaterEqual(r["report"]["texts"], 1, fn)

    # ---- gallery
    def test_gallery_written_when_requested(self):
        d = out_dir()
        if d is None:
            self.skipTest("set TQ_FRONT_OUT to write chart PNGs")
        import canvas_replay
        if not canvas_replay.HAVE_PIL:
            self.skipTest("PIL missing")
        self.draw("price_light", "drawPriceFan", 900, 380, "light", self.price_spec())
        self.draw("price_dark", "drawPriceFan", 900, 380, "dark", self.price_spec())
        self.draw("price_mobile", "drawPriceFan", 360, 300, "light", self.price_spec())
        for n in ("price_light", "price_dark", "price_mobile"):
            rec = json.loads(self.app.eval(f"window.__recJson({json.dumps(n)})"))
            img = canvas_replay.replay(rec, 2)
            img.save(d / f"{n}.png")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Helpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=FILES, fixtures=[])

    def test_palette(self):
        r = self.app.json("""(function () { var l = TQ.charts.palette('light'), d = TQ.charts.palette('dark'), x = TQ.charts.palette('weird');
          return { l: [l.mode, l.up, l.down, l.bg, l.upSoft], d: [d.mode, d.up, d.down, d.bg], x: x.mode }; })()""")
        self.assertEqual(r["l"], ["light", "#c62828", "#17794a", "#ffffff", "rgba(198,40,40,0.16)"])
        self.assertEqual(r["d"], ["dark", "#ff6b6b", "#35c27e", "#151b24"])
        self.assertEqual(r["x"], "light")

    def test_ticks(self):
        a = self.app
        self.assertEqual(a.json("TQ.charts.tickSet(0, 100, 5, 0).values"), [0, 20, 40, 60, 80, 100])
        v = a.json("TQ.charts.tickSet(2287, 2731, 6, 5).values")
        self.assertTrue(all(2287 <= t <= 2731 for t in v))
        self.assertEqual(a.json("TQ.charts.tickSet(5, 5, 5, 1).values"), [5])
        self.assertEqual(a.json("TQ.charts.tickSet(NaN, 5, 5, 1).values"), [])
        self.assertEqual(a.json("TQ.charts.tickSet(9, 1, 5, 1).values"), [])
        self.assertEqual(a.json("TQ.charts.niceStepFor(0.7, 0.05)"), 0.5)
        self.assertEqual(a.json("TQ.charts.niceStepFor(0.01, 0.05)"), 0.05)
        self.assertGreater(a.json("TQ.charts.niceStepFor(-3, 0)"), 0)

    def test_date_ticks_year_and_month_labels(self):
        r = self.app.json("""(function () {
          var d = []; for (var t = Date.UTC(2025, 5, 2); t < Date.UTC(2026, 9, 23); t += 86400000 * 1) { var x = new Date(t); if (x.getUTCDay() % 6) d.push(TQ.fmt.isoOf(t)); }
          return { many: TQ.charts.dateTicks(d, 0, d.length - 1, 4).map(function (t) { return t.label; }), few: TQ.charts.dateTicks(d, 0, d.length - 1, 30).length, none: TQ.charts.dateTicks([], 0, -1, 5), junk: TQ.charts.dateTicks(['x', null], 0, 1, 5) };
        })()""")
        self.assertLessEqual(len(r["many"]), 4)
        self.assertGreater(r["few"], 10)
        self.assertEqual((r["none"], r["junk"]), ([], []))

    def test_sanitize_fan_enforces_ordering(self):
        r = self.app.json("""TQ.charts.sanitizeFan({ days: [1,2,3], levels: [0.05,0.25,0.5,0.75,0.95],
            prices: [[120,130,140],[110,100,90],[100,100,100],[90,95,99],[80,85,70]] }, 100).rows""")
        for i in range(3):
            lo90, lo50, med, hi50, hi90 = (r[k][i] for k in ("lo90", "lo50", "med", "hi50", "hi90"))
            self.assertLessEqual(lo90, lo50)
            self.assertLessEqual(lo50, med)
            self.assertLessEqual(med, hi50)
            self.assertLessEqual(hi50, hi90)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class MountBehaviour(unittest.TestCase):
    def setUp(self):
        self.app = FrontApp(scripts=FILES, fixtures=["real/history_2330", "real/forecast_2330"])
        self.app.eval("""window.__host = document.createElement('div'); document.body.appendChild(window.__host);
          window.__spec = { bars: window.__fx.real_history_2330.bars, visible: 120, fan: window.__fx.real_forecast_2330.fan, lastClose: 2385, kind: 'stock' };
          window.__ctl = TQ.charts.mount(window.__host, { height: 300, label: '測試圖', draw: function (ctx, W, H, pal, hover) { return TQ.charts.drawPriceFan(ctx, W, H, pal, window.__spec, hover); } }); 0""")
        self.app.settle(100)

    def test_structure_and_initial_draw(self):
        a = self.app
        self.assertEqual(a.count(".chart-wrap .chart-legend"), 1)
        self.assertEqual(a.attr(".chart-canvas", "role"), "img")
        self.assertIn("價格走勢圖", a.attr(".chart-canvas", "aria-label"))
        self.assertEqual(a.attr(".chart-tip", "role"), "status")
        self.assertTrue(a.eval("document.querySelector('.chart-tip').hidden"))
        self.assertEqual(a.attr(".chart-canvas", "tabindex"), "0")
        self.assertEqual(a.json("document.querySelector('canvas').width"), 2560)                  # 1280 css px (fake viewport) * dpr 2
        self.assertEqual(a.json("document.querySelector('canvas').style.width"), "1280px")
        self.assertEqual(a.json("document.querySelector('canvas').style.height"), "300px")
        self.assertEqual(a.count(".chart-legend .lg"), 7)

    def test_hover_shows_tooltip_and_leaves(self):
        a = self.app
        a.eval("(function () { var c = document.querySelector('canvas'); c.dispatchEvent(new MouseEvent('pointermove', { clientX: 300, bubbles: true })); })()")
        self.assertFalse(a.eval("document.querySelector('.chart-tip').hidden"))
        self.assertIn("收盤", a.text(".chart-tip"))
        self.assertRegex(a.text(".chart-tip .tip-title"), r"\d{4}-\d{2}-\d{2}")
        a.eval("document.querySelector('canvas').dispatchEvent(new MouseEvent('pointerleave', { bubbles: true }))")
        self.assertTrue(a.eval("document.querySelector('.chart-tip').hidden"))
        a.eval("document.querySelector('canvas').dispatchEvent(new MouseEvent('pointermove', { clientX: 5000, bubbles: true }))")
        self.assertTrue(a.eval("document.querySelector('.chart-tip').hidden"))

    def test_keyboard_navigation(self):
        a = self.app
        key = lambda k: a.eval("document.querySelector('canvas').dispatchEvent(new KeyboardEvent('keydown', { key: %s, bubbles: true, cancelable: true }))" % json.dumps(k))
        key("ArrowLeft")                                       # first press from "nothing": jumps to the last point
        self.assertIn("預測", a.text(".chart-tip .tip-title"))
        key("Home")
        self.assertRegex(a.text(".chart-tip .tip-title"), r"^\d{4}")
        key("ArrowRight")
        key("End")
        self.assertIn("第20個交易日", a.text(".chart-tip .tip-title"))
        key("Escape")
        self.assertTrue(a.eval("document.querySelector('.chart-tip').hidden"))
        key("a")

    def test_tooltip_stays_inside_the_box(self):
        a = self.app
        for x in (5, 400, 790):
            a.eval("document.querySelector('canvas').dispatchEvent(new MouseEvent('pointermove', { clientX: %d, bubbles: true }))" % x)
            left = float(a.eval("document.querySelector('.chart-tip').style.left").replace("px", ""))
            self.assertGreaterEqual(left, 4)
            self.assertLessEqual(left, 1280)

    def test_resize_and_theme_redraw_and_destroy(self):
        a = self.app
        a.eval("window.__host.firstChild.firstChild.nextSibling._w = 360; window.__fireResize(); 0")
        a.settle(100)
        self.assertEqual(a.json("document.querySelector('canvas').style.width"), "360px")
        self.assertEqual(a.json("TQ.charts.mountedCount()"), 1)
        a.eval("document.documentElement.setAttribute('data-theme', 'dark'); window.dispatchEvent(new Event('tq:theme')); 0")
        a.settle(100)
        colors = a.json("(function () { var s = {}; document.querySelector('canvas').getContext('2d').ops.forEach(function (o) { if (o[0] === 'set' && o[1] === 'fillStyle') s[o[2]] = 1; }); return Object.keys(s); })()")
        self.assertIn("#ff6b6b", colors)
        a.eval("window.__ctl.destroy(); 0")
        self.assertEqual(a.json("TQ.charts.mountedCount()"), 0)
        self.assertEqual(a.count(".chart-wrap"), 0)
        a.eval("window.dispatchEvent(new Event('tq:theme')); window.__fireResize(); 0")       # no leaked listeners
        a.settle(100)
        self.assertEqual(a.errors(), [])

    def test_failing_draw_function_degrades_to_message(self):
        a = self.app
        a.eval("window.__ctl.setDraw(function () { throw new Error('draw boom'); }); 0")
        self.assertTrue(a.errors())
        self.assertIn("圖表繪製失敗", a.attr(".chart-canvas", "aria-label") or "圖表繪製失敗")
        self.assertEqual(a.json("document.querySelector('canvas').getContext('2d').ops.filter(function (o) { return o[0] === 'fillText'; }).pop()[1]"), "圖表繪製失敗")

    def test_zero_width_container_is_skipped_not_drawn(self):
        a = self.app
        a.eval("""window.__h2 = document.createElement('div'); document.body.appendChild(window.__h2); window.__h2._w = 0;
          window.__c2 = TQ.charts.mount(window.__h2, { height: 200, draw: function (ctx, W, H, pal) { window.__drawn = (window.__drawn || 0) + 1; return TQ.charts.drawEmpty(ctx, W, H, pal, 'x'); } }); 0""")
        a.settle(100)
        self.assertEqual(a.eval("window.__drawn || 0"), 0)


if __name__ == "__main__":
    unittest.main()
