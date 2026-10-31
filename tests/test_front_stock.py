"""個股預測 view (#/stock/:code): every state (full / ETF / OTC / nulls / sparse / computing / error / aborted) in the fake browser.

Fixtures: REAL backend output (tests/fixtures/real/*) for the happy paths, schema-built ones (tests/fixtures/*.json) for the
null / sparse / empty variants.
"""
from __future__ import annotations

import copy
import json
import unittest

from front_helpers import (FIX, HAVE_RACER, SKIP_MSG, FrontApp, assert_clean, audit, bad_text_hits, default_resp, default_rules,
                           fixture)

OK = lambda body: {"status": 200, "body": body}


def stock(code="2330", settle=4000, extra=(), real=True, **kw):
    app = FrontApp(hash=f"#/stock/{code}", real=real, boot=False, **kw)
    app.set_rules(list(extra) + default_rules(real))
    app.boot()
    app.settle(settle)
    return app


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class FullPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = stock("2330", 6000)
        cls.text = cls.app.text("#view")

    def test_clean_and_accessible(self):
        assert_clean(self, self.app)
        self.assertEqual(audit(self.app)["h1"], 1)

    def test_header(self):
        t = self.text
        for needle in ("台積電", "2330", "上市", "2,385", "▼ −1.45%", "資料截至 2026-10-22 (週四) 收盤", "已收盤"):
            self.assertIn(needle, t)
        self.assertEqual(self.app.eval("document.title"), "台積電 2330 個股預測 · 台股量化實驗室")
        self.assertEqual(self.app.attr(".sh-chg", "class"), "sh-chg down")           # falling = green, never red
        self.assertEqual(json.loads(self.app.eval("window.__ls['tq.lastStock']")), "2330")

    def test_three_horizon_cards_with_probabilities_and_meaning(self):
        t = self.text
        for needle in ("明日", "未來5日", "未來20日", "51.6%", "54.7%", "58.4%", "上漲機率", "模型估計明日收盤高於今日的機率",
                       "模型估計 5 個交易日後收盤高於今日的機率", "模型估計 20 個交易日後收盤高於今日的機率"):
            self.assertIn(needle, t)
        self.assertEqual(self.app.count(".hz-card"), 3)
        self.assertEqual(self.app.texts(".hz-card .gauge-val"), ["上漲機率 51.6%", "上漲機率 54.7%", "上漲機率 58.4%"])
        # signal chips: neutral, neutral, bullish (red up-chip), with text AND glyph
        chips = self.app.texts(".hz-card .hz-chips .chip")
        self.assertEqual(chips, ["◆中性·觀望", "信心 低", "◆中性·觀望", "信心 低", "▲偏多·買進", "信心 中"])      # signal chip + confidence chip
        self.assertEqual(self.app.count(".hz-card .hz-chips .chip-up"), 1)
        # a neutral signal must not paint the probability red or green
        self.assertEqual(self.app.attr(".hz-card .gauge-val", "class"), "gauge-val flat")
        self.assertEqual(self.app.json("Array.prototype.map.call(document.querySelectorAll('.hz-card .gauge-val'), function (e) { return e.className; })"),
                         ["gauge-val flat", "gauge-val flat", "gauge-val up"])

    def test_intervals_returns_components_confidence(self):
        t = self.text
        for needle in ("90% 區間", "50% 區間", "中位數", "2,324 ~ 2,452", "−2.6% ~ +2.8%", "+0.10%", "−0.47%", "1.66%", "信心 低",
                       "GJR-GARCH-t", "隱馬可夫regime", "動態線性模型(貝氏)", "隨機漫步基準", "綜合（已校準）", "51.7%"):
            self.assertIn(needle, t)

    def test_regime_vol_risk_weights_drivers(self):
        t = self.text
        for needle in ("低波動多頭", "多頭穩定", "高波動震盪", "目前最可能", "40.1%", "52.1%", "7.8%", "狀態轉移機率",
                       "22.4%", "高於歷史 32% 的時間", "波動中段", "VaR 95%", "CVaR 95%", "−2.55%", "−10.5%", "−3.59%", "−15.1%", "5.8%",
                       "模型權重", "24%", "12-1個月動能", "+73.5%", "+1.68", "+2.7bp", "RSI(14)", "37.9"):
            self.assertIn(needle, t)
        self.assertEqual(self.app.count(".regime-row.current"), 1)
        # weights: three stacked bars, each segment list sums to 100 %
        for w in self.app.json("Array.prototype.map.call(document.querySelectorAll('.w-row .stack'), function (s) { return s.getAttribute('aria-label'); })"):
            self.assertIn("GJR-GARCH-t", w)
        self.assertEqual(self.app.count(".w-row"), 3)

    def test_drivers_show_all_toggle(self):
        self.assertEqual(self.app.json("document.querySelectorAll('.table-wrap[aria-label=\"預測驅動因子\"] tbody tr').length"), 8)
        self.app.click(".table-more button")
        self.assertEqual(self.app.json("document.querySelectorAll('.table-wrap[aria-label=\"預測驅動因子\"] tbody tr').length"), 14)
        self.assertEqual(self.app.attr(".table-more button", "aria-expanded"), "true")
        self.app.click(".table-more button")
        self.assertEqual(self.app.json("document.querySelectorAll('.table-wrap[aria-label=\"預測驅動因子\"] tbody tr').length"), 8)

    def test_backtest_summary_loaded_lazily_after_forecast_and_sequentially(self):
        f = self.app.fetched()
        self.assertEqual(f[:2], ["/api/history?symbol=2330&limit=250", "/api/forecast?symbol=2330"])
        self.assertEqual([x for x in f if "backtest" in x], ["/api/backtest?symbol=2330&h=1", "/api/backtest?symbol=2330&h=5"])
        t = self.text
        for needle in ("歷史驗證摘要", "沒有統計上顯著的預測優勢", "方向準確率", "52.0%", "[49.1%, 54.8%]", "最佳基準：永遠看漲", "對最佳基準的 p 值", "0.51",
                       "Brier 技能分數", "−0.17%", "1,164 次", "無預測優勢", "54.0%", "[47.6%, 60.0%]", "0.61"):
            self.assertIn(needle, t)
        hrefs = self.app.json("Array.prototype.map.call(document.querySelectorAll('.bt-card a'), function (a) { return a.getAttribute('href'); })")
        self.assertIn("#/backtest/2330", hrefs)
        self.assertEqual(set(hrefs), {"#/backtest/2330", "#/backtest/2330?h=1", "#/backtest/2330?h=5"})      # each column opens the full validation of ITS horizon

    def test_notes_and_disclaimer(self):
        self.assertIn("注意事項與免責聲明", self.text)
        self.assertIn("非投資建議", self.text)
        self.assertIn("資料截至 2026-10-22 收盤", self.text)
        self.assertEqual(self.app.count(".disclaimer"), 1)
        self.assertEqual(self.app.count(".disclaimer-strip"), 1)       # the slim banner at the top

    def test_charts_are_drawn_and_summarised(self):
        n = self.app.json("TQ.charts.mountedCount()")
        self.assertEqual(n, 7)                                          # price + 3 gauges + 3 interval strips
        ops = self.app.json("Array.prototype.map.call(document.querySelectorAll('canvas'), function (c) { return c.getContext('2d').ops.length; })")
        self.assertTrue(all(o > 10 for o in ops), ops)
        price = self.app.attr("canvas", "aria-label")
        self.assertIn("近 120 個交易日", price)
        self.assertIn("延伸 20 個交易日的預測區間", price)
        self.assertIn("2,385", price)
        # only the hover-capable chart is a tab stop (gauges / strips are static images)
        self.assertEqual(self.app.count("canvas[tabindex=\"0\"]"), 1)
        # gauges and strips duplicate text that is already in the DOM: hidden from assistive technology
        self.assertEqual(self.app.count("canvas[aria-hidden=\"true\"]"), 6)

    def test_range_selector_persists_and_redraws(self):
        app = stock("2330", 4000)
        app.click(".card-actions .seg-btn", 0)        # 60日
        app.settle(100)
        self.assertIn("近 60 個交易日", app.attr("canvas", "aria-label"))
        self.assertEqual(json.loads(app.eval("window.__ls['tq.chartBars']")), 60)
        self.assertEqual(app.attr(".card-actions .seg-btn", "aria-checked"), "true")
        app2 = stock("2330", 4000, storage={"tq.chartBars": "250"})
        self.assertIn("近 250 個交易日", app2.attr("canvas", "aria-label"))
        app3 = stock("2330", 4000, storage={"tq.chartBars": "7"})          # invalid stored value falls back to 120
        self.assertIn("近 120 個交易日", app3.attr("canvas", "aria-label"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class OtherInstruments(unittest.TestCase):
    def test_etf_prices_use_two_decimals_and_chip(self):
        app = stock("0050", 5000)
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("元大台灣50", t)
        self.assertIn("ETF", t)
        self.assertEqual(app.texts(".sh-last"), ["106.90"])                # ETF tick size 0.05 -> two decimals
        self.assertIn("▼ −1.16%", t)
        self.assertIn("104.29 ~ 109.75", t)                                # 90 % price interval, same formatting

    def test_tpex_stock(self):
        app = stock("6488", 5000)
        assert_clean(self, app)
        self.assertIn("上櫃", app.text("#view"))
        self.assertIn("環球晶", app.text("#view"))

    def test_index_symbol_with_caret(self):
        fc = fixture("real/forecast_2330.json")
        fc.update(symbol="^TWII", name="加權指數", market="INDEX", kind="index", last_close=46041.43)
        hist = fixture("real/history_2330.json")
        hist.update(symbol="^TWII", name="加權指數", market="INDEX", kind="index")
        for b in hist["bars"]:
            b["v"] = 0                                                   # the index has no volume
        app = stock("%5ETWII", 4000, extra=[{"re": r"^/api/forecast\?symbol=%5ETWII", "resp": OK(fc)}, {"re": r"^/api/history\?symbol=%5ETWII", "resp": OK(hist)}])
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("加權指數", t)
        self.assertIn("指數", app.text(".sh-title"))
        self.assertIn("46,041", t)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/%5ETWII")
        self.assertEqual(app.json("JSON.parse(window.__ls['tq.lastStock'])"), "^TWII")
        self.assertEqual(app.count(".hz-card"), 3)

    def test_code_in_url_is_normalised(self):
        app = stock("00400a", 500)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/00400A")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class PartialPayloads(unittest.TestCase):
    def test_nulls_everywhere(self):
        app = stock("6488", 5000, real=False)               # schema-built forecast_nulls + history of 0050
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("環球晶", t)
        self.assertIn("—", t)
        self.assertIn("僅有", t)                            # short_history banner
        self.assertEqual(app.count(".hz-card"), 3)
        # a null p_up still gets a gauge box and an em dash
        self.assertIn("上漲機率 —", app.texts(".hz-card .gauge-val")[0])
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 7)

    def test_sparse_payload(self):
        app = stock("9999", 4000, real=False)
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("測試", t)
        self.assertIn("12.34", t)
        self.assertIn("上漲機率 50.0%", t)                  # the only number in the payload
        self.assertIn("沒有市場狀態資料", t)
        self.assertIn("沒有因子資料", t)

    def test_garbage_types_do_not_throw(self):
        fc = fixture("real/forecast_2330.json")
        fc["horizons"] = {"1": None, "5": "x", "20": {"p_up": "0.6", "quantiles": [], "components": 5, "signal": 3}}
        fc["regime"] = {"labels": "abc", "probs": 5}
        fc["vol"] = None
        fc["risk"] = []
        fc["models"] = {"weights": [], "names": 4}
        fc["drivers"] = [None, 5, {"name": "x"}, {"name": "y", "value": "abc", "z": {}, "contribution": "1e-4"}]
        fc["notes"] = [None, 5, "ok note"]
        fc["fan"] = {"days": None, "levels": "x", "prices": 5}
        app = stock("2330", 4000, extra=[{"re": r"^/api/forecast\?symbol=2330", "resp": OK(fc)}])
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("ok note", t)
        self.assertIn("60.0%", t)                            # numeric strings are accepted

    def test_backtest_payload_variants(self):
        nulls = fixture("backtest_nulls.json")
        app = stock("2330", 8000, extra=[{"re": r"^/api/backtest\?symbol=2330&h=1", "resp": OK(nulls)},
                                          {"re": r"^/api/backtest\?symbol=2330&h=5", "resp": OK({"status": "ready"})}])
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("尚無結論", t)
        self.assertIn("中等優勢", t)

    def test_intraday_and_stale_flags(self):
        fc = fixture("real/forecast_2330.json")
        fc["intraday_bar_dropped"] = True
        fc["stale"] = True
        app = stock("2330", 3000, extra=[{"re": r"^/api/forecast\?symbol=2330", "resp": OK(fc)},
                                          {"re": r"^/api/status", "resp": OK({"status": "ready", "market_open": True, "last_trading_date": "2026-10-23"})}])
        assert_clean(self, app)
        t = app.text("#view")
        for needle in ("目前為盤中", "未完成的當日 K 棒", "本機快取", "開盤中", "資料可能過期", "尚未更新至"):
            if needle == "尚未更新至":      # only shown when the market is closed
                self.assertNotIn(needle, t)
            else:
                self.assertIn(needle, t)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ErrorsAndLoading(unittest.TestCase):
    def test_unknown_symbol_shows_friendly_page_and_does_not_start_backtests(self):
        app = stock("1234", 3000)
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("找不到這檔股票", t)
        self.assertIn("搜尋框", t)
        self.assertEqual(audit(app)["h1"], 1)                                  # the error page still has a level-1 heading
        self.assertEqual([x for x in app.fetched() if "backtest" in x], [])
        self.assertEqual(app.eval("typeof window.__ls['tq.lastStock']"), "undefined")      # a bad code is never remembered
        self.assertIn("找不到 1234", app.eval("document.title"))

    def test_forecast_error_keeps_chart_and_retry_recovers(self):
        err = {"status": 422, "body": fixture("error_insufficient.json")}
        ok = default_resp(True, r"^/api/forecast\?symbol=0050")
        app = stock("2330", 3000, extra=[{"re": r"^/api/forecast\?symbol=2330", "seq": [err, ok], "resp": err}])
        t = app.text("#view")
        self.assertIn("歷史資料不足", t)
        self.assertIn("insufficient_history", t)
        self.assertEqual(app.count(".hz-card"), 0)
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 1)           # the price chart (from /api/history) is still there
        self.assertIn("台積電", t)
        self.assertEqual([x for x in app.fetched() if "backtest" in x], [])
        app.click(".slot-fc .error-box button")
        app.settle(4000)
        self.assertEqual(app.count(".hz-card"), 3)
        assert_clean(self, app)

    def test_forecast_that_answers_computing_is_polled_until_ready(self):
        comp = {"status": 200, "body": {"status": "computing", "progress": {"done": 1, "total": 3, "message": "x"}}}
        ready = default_resp(True, r"^/api/forecast\?symbol=0050")
        app = stock("2330", 1000, extra=[{"re": r"^/api/forecast\?symbol=2330", "seq": [comp, comp, ready], "resp": ready}])
        self.assertEqual(app.count(".hz-card"), 0)
        app.settle(4000)
        self.assertEqual(app.count(".hz-card"), 3)
        assert_clean(self, app)

    def test_network_error_is_retried_with_feedback_then_reported_and_retry_recovers(self):
        ok = default_resp(True, r"^/api/forecast\?symbol=0050")
        app = stock("2330", 1800, extra=[{"re": r"^/api/forecast\?symbol=2330", "seq": [{"networkError": True}] * 4 + [ok], "resp": ok}])
        self.assertIn("連線不穩，重新嘗試中（1/3）", app.text(".fc-wait"))
        app.settle(15000)
        self.assertIn("無法連線到伺服器", app.text("#view"))
        self.assertEqual(app.count(".hz-card"), 0)
        app.click(".slot-fc .error-box button")
        app.settle(3000)
        self.assertEqual(app.count(".hz-card"), 3)

    def test_history_failure_still_renders_forecast_with_fan_only(self):
        app = stock("2330", 4000, extra=[{"re": r"^/api/history\?symbol=2330", "resp": {"networkError": True}}])
        t = app.text("#view")
        self.assertIn("價格歷史載入失敗", t)
        self.assertEqual(app.count(".hz-card"), 3)
        self.assertIn("51.6%", t)
        assert_clean(self, app, allow_errors=False)

    def test_history_failure_alone_offers_retry(self):
        err = {"networkError": True}
        app = stock("2330", 800, extra=[{"re": r"^/api/history\?symbol=2330", "seq": [err, default_resp(True, r"^/api/history\?symbol=2330")], "resp": err},
                                         {"re": r"^/api/forecast\?symbol=2330", "resp": {"networkError": True}}])
        self.assertEqual(app.count(".slot-chart .error-box"), 1)
        app.click(".slot-chart .error-box button")
        app.settle(500)
        self.assertEqual(app.count(".slot-chart .error-box"), 0)

    def test_skeletons_while_loading_then_content(self):
        app = FrontApp(hash="#/stock/2330", real=True, boot=False, delay=400)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(100)
        self.assertGreaterEqual(app.count(".skeleton"), 2)
        self.assertEqual(app.count(".hz-card"), 0)
        self.assertIn("2330", app.text(".stock-head"))             # the header exists right away (code only)
        app.settle(1500)
        self.assertEqual(app.count(".hz-card"), 3)

    def test_backtest_progress_then_result_and_per_column_error(self):
        comp = lambda d, t, m: {"status": 200, "body": {"status": "computing", "progress": {"done": d, "total": t, "message": m}}}
        ready1 = default_resp(True, r"^/api/backtest\?symbol=2330&h=1")
        err5 = {"status": 500, "body": {"status": "error", "error": {"code": "internal", "message": "前推式預測失敗"}}}
        ready5 = default_resp(True, r"^/api/backtest\?symbol=2330&h=5")
        app = stock("2330", 900, extra=[
            {"re": r"^/api/backtest\?symbol=2330&h=1", "seq": [comp(120, 1250, "前推式預測 120/1250"), comp(900, 1250, "前推式預測 900/1250"), ready1], "resp": ready1},
            {"re": r"^/api/backtest\?symbol=2330&h=5", "seq": [err5, ready5], "resp": err5}])
        t = app.text(".bt-card")
        self.assertIn("前推式預測 120/1250", t)
        self.assertIn("120 / 1,250", t)
        self.assertEqual(app.attr(".bt-card .progress", "aria-valuenow"), "10")
        self.assertIn("排隊中", t)                                   # second column waits for the first (they share a walk-forward)
        self.assertEqual([x for x in app.fetched() if "backtest" in x], ["/api/backtest?symbol=2330&h=1"])
        app.settle(6000)
        t = app.text(".bt-card")
        self.assertIn("沒有統計上顯著的預測優勢", t)
        self.assertIn("前推式預測失敗", t)                            # h=5 failed independently
        self.assertEqual(app.count(".bt-card .error-box"), 1)
        app.click(".bt-card .error-box button")
        app.settle(2000)
        self.assertEqual(app.count(".bt-card .error-box"), 0)
        self.assertIn("54.0%", app.text(".bt-card"))
        assert_clean(self, app, allow_errors=False)

    def test_route_change_aborts_everything_and_leaves_no_stragglers(self):
        app = FrontApp(hash="#/stock/2330", real=True, boot=False, delay=600)
        app.set_rules(default_rules(True) + [])
        app.boot()
        app.settle(200)
        app.eval("window.__sig = window.__log = []; 0")
        app.goto("#/method", 100)
        n = len(app.fetched())
        app.settle(10000)
        self.assertEqual(len(app.fetched()), n, "no new requests after the view was torn down (backtests must not start)")
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 0)
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.count(".hz-card"), 0)

    def test_leaving_during_backtest_polling_stops_polling(self):
        comp = {"status": 200, "body": {"status": "computing", "progress": {"done": 1, "total": 10, "message": "x"}}}
        app = stock("2330", 2500, extra=[{"re": r"^/api/backtest", "resp": comp}])
        n_before = len([x for x in app.fetched() if "backtest" in x])
        self.assertGreaterEqual(n_before, 1)
        app.goto("#/picks", 100)
        n = len(app.fetched())
        app.settle(20000)
        self.assertEqual([x for x in app.fetched()[n:] if "backtest" in x], [])


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ThemeAndViewport(unittest.TestCase):
    def test_dark_theme_draws_with_dark_palette(self):
        app = stock("2330", 4000, dark=True)
        self.assertEqual(app.eval("document.documentElement.getAttribute('data-theme')"), "dark")
        colors = set(app.json("(function () { var s = {}; Array.prototype.forEach.call(document.querySelectorAll('canvas'), function (c) { c.getContext('2d').ops.forEach(function (o) { if (o[0] === 'set' && o[1] === 'fillStyle') s[o[2]] = 1; }); }); return Object.keys(s); })()"))
        self.assertIn("#ff6b6b", colors)           # dark "up" red
        self.assertNotIn("#c62828", colors)

    def test_theme_switch_redraws_charts(self):
        app = stock("2330", 4000)
        before = app.json("document.querySelector('canvas').getContext('2d').ops.length")
        app.click("#theme-toggle")                 # auto -> light (same palette) ; second click -> dark
        app.click("#theme-toggle")
        app.settle(100)
        colors = app.json("(function () { var s = {}; document.querySelector('canvas').getContext('2d').ops.forEach(function (o) { if (o[0] === 'set' && o[1] === 'fillStyle') s[o[2]] = 1; }); return Object.keys(s); })()")
        self.assertIn("#ff6b6b", colors)

    def test_phone_width(self):
        app = stock("2330", 5000, viewport=360)
        assert_clean(self, app)
        self.assertEqual(app.count(".hz-card"), 3)
        h = app.json("document.querySelector('canvas').style.height")
        self.assertEqual(h, "300px")                # chart height rule for narrow screens
        app = stock("2330", 5000, viewport=1280)
        self.assertEqual(app.json("document.querySelector('canvas').style.height"), "420px")


if __name__ == "__main__":
    unittest.main()
