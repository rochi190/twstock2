"""歷史驗證 view (#/backtest/:code?h=&thr=&exec=&disc=&years=): URL <-> settings, controls, polling with progress / cancel, results
rendered from REAL backend payloads, null / partial payloads, errors, teardown.  Runs in V8 (py_mini_racer) with the fake DOM."""
from __future__ import annotations

import json
import re
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, assert_clean, audit, bad_text_hits, default_rules

OK = lambda body: {"status": 200, "body": body}
COMPUTING = lambda d, t, m: OK({"status": "computing", "progress": {"done": d, "total": t, "message": m}})
ERR = lambda status, code, msg: {"status": status, "body": {"status": "error", "error": {"code": code, "message": msg}}}
BT_RE = r"^/api/backtest\?"
DEFAULT_QS = "h=1&thr=0.55&exec=next_open&disc=0.6&years=5"


def bt(hash_="#/backtest/2330", settle=3000, extra=(), real=True, **kw):
    app = FrontApp(hash=hash_, real=real, boot=False, **kw)
    app.set_rules(list(extra) + default_rules(real))
    app.boot()
    app.settle(settle)
    return app


def bt_fetches(app):
    return [x for x in app.fetched() if x.startswith("/api/backtest")]


def seg_values(app, sel):
    """aria-checked flags of the radio buttons of the .seg matched by `sel`."""
    return app.json("Array.prototype.map.call(document.querySelectorAll(" + json.dumps(sel + " .seg-btn") + "), function (b) { return b.getAttribute('aria-checked'); })")


class Base(unittest.TestCase):
    def has(self, text, needle):
        """assertIn without dumping the whole page into the failure message."""
        self.assertTrue(needle in text, f"missing {needle!r}")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class IdleAndDefaults(Base):
    @classmethod
    def setUpClass(cls):
        cls.app = bt("#/backtest/2330", 1500)

    def test_idle_page_does_not_compute_anything(self):
        self.assertEqual(bt_fetches(self.app), [])
        t = self.app.text("#view")
        for needle in ("歷史驗證", "驗證設定", "尚未執行驗證", "執行驗證", "還原預設值", "10–60 秒", "沒有統計上顯著的預測優勢"):
            self.has(t, needle)
        self.assertEqual(self.app.count(".bt-progress-card"), 0)
        self.assertEqual(self.app.count(".bt-head-card"), 0)
        assert_clean(self, self.app)
        self.assertEqual(audit(self.app)["h1"], 1)
        self.assertEqual(self.app.eval("document.title"), "歷史驗證 2330 · 台股量化實驗室")

    def test_controls_show_the_documented_defaults(self):
        a = self.app
        self.assertEqual(seg_values(a, ".bt-f-h"), ["true", "false", "false"])              # 明日
        self.assertEqual(seg_values(a, ".bt-f-exec"), ["true", "false"])                      # 下一個開盤價成交
        self.assertEqual(seg_values(a, ".bt-f-years"), ["false", "false", "true", "false"])   # 5 年
        self.assertEqual(a.texts(".bt-f-h .seg-btn"), ["明日", "一週", "一個月"])
        self.assertEqual(a.texts(".bt-f-exec .seg-btn"), ["下一個開盤價成交", "收盤成交(樂觀)"])
        self.assertEqual(a.texts(".bt-f-years .seg-btn"), ["2 年", "3 年", "5 年", "8 年"])
        slider = a.json("(function () { var s = document.querySelector('#bt-thr'); return [s.getAttribute('type'), s.getAttribute('min'), s.getAttribute('max'), s.getAttribute('step'), s.value]; })()")
        self.assertEqual(slider, ["range", "0.5", "0.7", "0.01", "0.55"])
        self.assertEqual(a.text(".bt-thr-out"), "55%")
        self.assertEqual(a.json("document.querySelector('#bt-disc').value"), "0.6")
        opts = a.texts("#bt-disc option")
        self.assertEqual(opts[0], "2.8 折")
        self.assertIn("6 折（預設）", opts)
        self.assertEqual(opts[-1], "不打折（牌告價）")
        self.assertIn("預估來回交易成本約 0.57%", a.text(".bt-cost-hint"))
        self.assertEqual(a.json("document.querySelector('.bt-dirty').hidden"), True)

    def test_stock_search_is_the_reusable_combobox_with_a_label(self):
        a = self.app
        self.assertEqual(a.attr("#bt-stock", "role"), "combobox")
        self.assertEqual(a.count("label[for=bt-stock]"), 2)         # visible field label + the component's own screen-reader label
        self.assertTrue(a.attr("#bt-stock", "aria-describedby"))
        self.assertIn("2330", a.text(".bt-cur"))
        self.assertEqual(a.attr(".bt-stock-link", "href"), "#/stock/2330")

    def test_nothing_is_written_into_the_url_until_a_run(self):
        self.assertEqual(self.app.eval("location.hash"), "#/backtest/2330")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class QueryRestore(Base):
    def test_full_query_restores_every_control_and_runs(self):
        app = bt("#/backtest/2330?h=5&thr=0.6&exec=close&disc=0.35&years=3")
        self.assertEqual(bt_fetches(app), ["/api/backtest?symbol=2330&h=5&thr=0.6&exec=close&disc=0.35&years=3"])
        self.assertEqual(seg_values(app, ".bt-f-h"), ["false", "true", "false"])
        self.assertEqual(seg_values(app, ".bt-f-exec"), ["false", "true"])
        self.assertEqual(seg_values(app, ".bt-f-years"), ["false", "true", "false", "false"])
        self.assertEqual(app.json("document.querySelector('#bt-thr').value"), "0.60")
        self.assertEqual(app.text(".bt-thr-out"), "60%")
        self.assertEqual(app.json("document.querySelector('#bt-disc').value"), "0.35")
        self.assertIn("3.5 折", app.texts("#bt-disc option"))
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=5&thr=0.6&exec=close&disc=0.35&years=3")        # kept, canonical order
        self.assertEqual(app.count(".bt-head-card"), 1)
        assert_clean(self, app)

    def test_partial_query_fills_in_defaults_and_rewrites_the_url(self):
        app = bt("#/backtest/2330?thr=0.52")
        self.assertEqual(bt_fetches(app), ["/api/backtest?symbol=2330&h=1&thr=0.52&exec=next_open&disc=0.6&years=5"])
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=1&thr=0.52&exec=next_open&disc=0.6&years=5")

    def test_invalid_values_are_normalised_and_reported(self):
        app = bt("#/backtest/2330?h=7&thr=0.9&exec=banana&disc=2&years=10")
        self.assertEqual(bt_fetches(app), ["/api/backtest?symbol=2330&h=1&thr=0.7&exec=next_open&disc=0.6&years=8"])
        t = app.text(".slot-notice")
        for needle in ("網址中的部分設定已調整", "h=7", "thr=0.9", "70%", "exec=banana", "disc=2", "years=10", "最接近的 8 年"):
            self.has(t, needle)
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=1&thr=0.7&exec=next_open&disc=0.6&years=8")

    def test_garbage_is_harmless(self):
        for q in ("h=&thr=&exec=&disc=&years=", "thr=abc&years=-3&disc=0", "h=1&h=5", "__proto__=x&constructor=y&thr=0.5", "%E0%A4%A=1&h=20"):
            app = bt("#/backtest/2330?" + q, 1500)
            self.assertEqual(app.errors(), [], q)
            self.assertEqual(len(bt_fetches(app)), 1, q)

    def test_unknown_query_keys_alone_do_not_start_a_run(self):
        app = bt("#/backtest/2330?utm_source=x", 1000)
        self.assertEqual(bt_fetches(app), [])
        self.assertIn("尚未執行驗證", app.text("#view"))

    def test_code_is_upper_cased_and_validated_by_the_router_query_survives(self):
        app = bt("#/backtest/00878?h=20", 1500)
        self.assertEqual(app.eval("location.hash"), "#/backtest/00878?h=20&thr=0.55&exec=next_open&disc=0.6&years=5")
        self.assertEqual(bt_fetches(app)[0], "/api/backtest?symbol=00878&h=20&thr=0.55&exec=next_open&disc=0.6&years=5")
        app = bt("#/backtest/00400a?h=5", 500)
        self.assertTrue(app.eval("location.hash").startswith("#/backtest/00400A?h=5"))

    def test_internals_read_query(self):
        app = FrontApp(scripts=["format.js", "charts.js", "core.js", "view_backtest.js"], fixtures=[])
        r = app.json("TQ.views.backtest._internals.readQuery({ h: '5', thr: '0.5', exec: 'close', disc: '1', years: '2' })")
        self.assertEqual(r["p"], {"h": 5, "thr": 0.5, "exec": "close", "disc": 1, "years": 2})
        self.assertEqual((r["given"], r["notes"]), (True, []))
        r = app.json("TQ.views.backtest._internals.readQuery({})")
        self.assertEqual((r["given"], r["p"]), (False, {"h": 1, "thr": 0.55, "exec": "next_open", "disc": 0.6, "years": 5}))
        self.assertEqual(app.json("TQ.views.backtest._internals.readQuery(null).given"), False)
        self.assertEqual(app.json("TQ.views.backtest._internals.readQuery({ years: '4' }).p.years"), 3)        # nearest option (ties go to the smaller)
        self.assertEqual(app.json("TQ.views.backtest._internals.readQuery({ thr: '0.555' }).p.thr"), 0.56)
        self.assertEqual(app.json("TQ.views.backtest._internals.queryOf({ h: 20, thr: 0.6, exec: 'close', disc: 0.28, years: 8 })"), "h=20&thr=0.6&exec=close&disc=0.28&years=8")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class RealResultOneDay(Base):
    """The real backend's answer for 2330 / 明日: no edge, and the strategy never traded at the 55% threshold."""

    @classmethod
    def setUpClass(cls):
        cls.app = bt("#/backtest/2330?h=1", 3000)
        cls.text = cls.app.text("#view")

    def test_clean_and_one_h1(self):
        assert_clean(self, self.app)
        self.assertEqual(audit(self.app)["h1"], 1)
        self.assertEqual(self.app.eval("document.title"), "台積電 2330 歷史驗證 · 台股量化實驗室")
        self.assertIn("歷史驗證完成：無預測優勢，沒有統計上顯著的預測優勢", self.app.text("#status"))

    def test_verdict_comes_first_and_is_neutral_not_red_or_green(self):
        a = self.app
        self.assertEqual(a.count(".verdict"), 1)
        self.assertEqual(a.attr(".verdict", "role"), "group")
        cls = a.attr(".verdict", "class")
        self.assertIn("verdict-no_edge", cls)
        self.assertIn("tone-neutral", cls)
        self.assertNotRegex(cls, r"\b(up|down|chip-up|chip-down)\b")
        for needle in ("沒有統計上顯著的預測優勢", "無預測優勢", "單尾二項檢定 p=0.51", "結論等級怎麼判定？", "微弱優勢", "中等優勢", "顯著優勢"):
            self.has(self.text, needle)
        # the header chip of the result carries the same level (icon + words, not only colour)
        self.assertIn("⊘無預測優勢", a.text(".bt-res-title"))
        titles = a.texts("h2.card-title")
        want = ["驗證設定", "驗證結果", "驗證結論", "預測品質", "假想策略 vs 買進持有", "資產曲線、持倉與回撤", "校準圖：預測機率 vs 實際頻率", "滾動方向準確率 vs 基準",
                "PIT 直方圖：預測分布是否校準", "報酬預測與區間涵蓋", "信心分組：機率越高，真的越準嗎？", "參數與交易成本", "假設、限制與注意事項"]
        self.assertEqual([re.sub(r"ⓘ.*$", "", x) for x in titles], want)

    def test_header_meta(self):
        t = self.app.text(".bt-head-card")
        for needle in ("台積電", "2330", "明日預測", "2021-08-27 ~ 2026-10-21", "1,249 次", "1,164 次（持平排除 85）", "2026-10-22", "計算於 2026-10-22 22:03"):
            self.has(t, needle)

    def test_quality_kpis_have_values_baselines_and_explanations(self):
        a = self.app
        kpis = a.texts(".bt-quality-card .kpi")
        self.assertEqual(len(kpis), 3)
        acc, brier, ic = kpis
        for needle in ("方向準確率", "52.0%", "最佳基準「永遠看漲」52.0%", "95% 信賴區間 [49.1%, 54.8%]", "對最佳基準 p = 0.51", "對丟銅板 p = 0.094", "不顯著"):
            self.has(acc, needle)
        for needle in ("Brier 技能分數", "−0.17%", "Brier 0.2500 vs 基準 0.2496", "ECE 3.1%"):
            self.has(brier, needle)
        for needle in ("預測排序相關 IC", "−0.013", "雙尾 p = 0.65"):
            self.has(ic, needle)
        self.assertEqual(a.count(".bt-quality-card .kpi .itip"), 3)           # every KPI has an ⓘ
        tips = a.texts(".bt-quality-card .kpi .itip-pop")
        self.assertIn("p 值是「若模型其實沒有優勢", tips[0])
        self.assertIn("1 − 模型 Brier ÷ 基準 Brier", tips[1])
        self.assertIn("Spearman", tips[2])

    def test_baseline_table_marks_the_best_baseline(self):
        t = self.app.text(".bt-quality-card table")
        for needle in ("模型（上漲機率 ≥ 50% 即預測上漲）", "永遠看漲", "最佳基準", "沿用前一期方向", "48.7%", "+3.3pp", "訓練期多數方向"):
            self.has(t, needle)
        self.assertIn("驗證期間內一次也沒有出現", self.text)             # no high-confidence call at all: said plainly

    def test_strategy_that_never_traded_is_explained_not_hidden(self):
        a = self.app
        t = a.text(".bt-strategy-card")
        for needle in ("策略全程空手", "從未進場", "0 次", "沒有進出場，不產生交易成本", "0%", "買進持有 1.11", "策略全程空手，沒有夏普比率", "買進持有 −44.8%", "+337.3%", "績效統計比較"):
            self.has(t, needle)
        self.assertEqual(a.count(".bt-strategy-card .banner-warn"), 1)
        self.assertEqual(a.count(".bt-strategy-card .kpi"), 4)
        self.assertEqual(bad_text_hits(t), [])
        # stats table: strategy column dashes, benchmark values
        rows = a.texts(".bt-strategy-card tbody tr")
        sharpe = [r for r in rows if r.startswith("夏普比率")][0]
        self.assertIn("—", sharpe)
        self.assertIn("1.11", sharpe)
        self.assertEqual([r for r in rows if r.startswith("交易勝率")][0].count("—"), 2)        # no win rate for 0 trades or a single buy-and-hold trade

    def test_four_charts_are_mounted_with_text_alternatives(self):
        a = self.app
        self.assertEqual(a.count("canvas"), 4)
        labels = a.texts(".chart-summary")
        self.assertEqual(len(labels), 4)
        self.assertTrue(any("資產曲線圖" in x for x in labels))
        self.assertTrue(any("校準圖" in x for x in labels))
        self.assertTrue(any("滾動250日方向準確率" in x for x in labels))
        self.assertTrue(any("PIT" in x for x in labels))
        for i in range(4):
            n = a.json(f"document.querySelectorAll('canvas')[{i}].getContext('2d').ops.length")
            self.assertGreater(n, 10, f"chart {i} drew nothing")
            self.assertEqual(a.attr(f"canvas", "role"), "img")
        self.assertIn("策略全程空手，所以藍線貼在 1.00", a.text(".chart-note"))

    def test_coverage_and_bucket_tables(self):
        a = self.app
        t = a.text(".bt-returns-card")
        for needle in ("50% 區間", "48.8%", "−1.2pp", "接近預期", "80% 區間", "79.7%", "90% 區間", "88.6%", "Kupiec", "0.11", "KS", "0.34", "−0.13%", "1.927% / 1.926%", "−1.69（0.091）"):
            self.has(t, needle)
        b = a.text(".bt-bucket-card")
        for needle in ("45% – 50%", "450", "48.7%", "50% – 55%", "714", "51.2%"):
            self.has(b, needle)

    def test_parameters_and_cost_breakdown_add_up(self):
        a = self.app
        t = a.text(".bt-params-card")
        for needle in ("股票", "明日（下一個交易日）", "55%", "下一個開盤價成交", "6 折", "5.0 bp", "0.57%", "500 個交易日", "每 21 個交易日", "1,164", "成本拆解",
                       "0.171%", "0.300%", "0.100%", "0.571%", "牌告 0.1425% × 6 折 × 2 次", "股票賣出 0.3%"):
            self.has(t, needle)
        n = a.text(".notes-card")
        for needle in ("已計入手續費", "非投資建議", "挑選偏誤"):
            self.has(n, needle)

    def test_url_keeps_the_settings(self):
        self.assertEqual(self.app.eval("location.hash"), "#/backtest/2330?" + DEFAULT_QS)
        self.assertEqual(self.app.attr('#nav a[data-route="backtest"]', "href"), "#/backtest/2330")        # last stock is remembered for the nav


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class RealResultOneWeekAndTradingStrategy(Base):
    def test_real_five_day_report(self):
        app = bt("#/backtest/2330?h=5&thr=0.55")
        assert_clean(self, app)
        t = app.text("#view")
        for needle in ("一週預測", "未來 5 個交易日", "沒有統計上顯著的預測優勢", "預測期間大於 1 天時，相鄰預測高度重疊"):
            self.has(t, needle)
        self.assertEqual(app.count("canvas"), 4)

    def test_strategy_with_trades_shows_sharpe_comparison_and_psr_dsr(self):
        rules = [{"re": BT_RE, "resp": OK("backtest_2330_h5")}]          # schema fixture: weak_edge with 84 trades
        app = bt("#/backtest/2330?h=5", 3000, extra=rules, real=False)
        assert_clean(self, app)
        a = app
        t = a.text(".bt-strategy-card")
        self.assertEqual(a.count(".bt-strategy-card .banner"), 0)
        for needle in ("−0.04", "買進持有 0.71", "兩者差的 95% 信賴區間 [−0.40, 0.60]", "PSR 74%\u3000DSR 55%", "84 次", "年換手率 7.4 倍", "交易勝率 52.0%", "64%", "通縮夏普 (DSR)", "機率夏普 (PSR)",
                       "策略夏普 95% 信賴區間", "[−0.31, 0.86]", "−34.2%", "買進持有 −32.1%"):
            self.has(t, needle)
        self.assertIn("verdict-weak_edge", a.attr(".verdict", "class"))
        self.assertIn("tone-warn", a.attr(".verdict", "class"))
        self.assertEqual(a.count(".bt-strategy-card tbody tr"), 13)

    def test_dark_theme_charts_use_dark_palette(self):
        app = bt("#/backtest/2330?h=1", 3000, dark=True)
        self.assertEqual(app.eval("document.documentElement.getAttribute('data-theme')"), "dark")
        self.assertEqual(app.errors(), [])
        ops = app.json("document.querySelectorAll('canvas')[0].getContext('2d').ops.filter(function (o) { return o[0] === 'set' && o[1] === 'strokeStyle'; }).map(function (o) { return o[2]; })")
        self.assertIn("#6b9bff", ops)            # dark accent for the strategy line
        self.assertNotIn("#2457c5", ops)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class RealResultTwentyDayFixedBackend(Base):
    """Real h = 20 payload of the fixed backend: causal + oracle baselines, ECE noise floor, p_up_event = up_given_move, a NEGATIVE IC."""

    @classmethod
    def setUpClass(cls):
        cls.app = bt("#/backtest/2330?h=20", 3000)
        cls.text = cls.app.text("#view")

    def test_clean(self):
        assert_clean(self, self.app)

    def test_brier_card_shows_causal_baseline_oracle_baseline_and_ece_noise_floor(self):
        brier = self.app.texts(".bt-quality-card .kpi")[1]
        for needle in ("Brier 技能分數", "−1.61%", "對事後基準 −2.75%", "ECE 9.5%", "雜訊底限約 20%"):
            self.has(brier, needle)
        tip = self.app.texts(".bt-quality-card .kpi .itip-pop")[1]
        self.assertIn("因果氣候基準", tip)

    def test_negative_ic_never_gets_a_significance_chip(self):
        """S-02: the p-value is two-sided; a (significantly) negative IC must not read as ranking skill."""
        ic = self.app.texts(".bt-quality-card .kpi")[2]
        self.has(ic, "−0.073")
        self.has(ic, "無正向排序資訊")
        self.assertNotIn("單項檢定顯著", ic)

    def test_sample_note_follows_p_up_event(self):
        self.has(self.text, "排除收盤價不變（平盤）的日子")
        self.has(self.text, "評分樣本 1,219 次")

    def test_legacy_payload_without_p_up_event_has_no_flat_day_note(self):
        app = bt("#/backtest/2330?h=1", 3000)
        self.assertNotIn("排除收盤價不變（平盤）的日子", app.text("#view"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class NullsAndPartialPayloads(Base):
    def check(self, body, settle=3000, expect=None):
        app = bt("#/backtest/2330?h=1", settle, extra=[{"re": BT_RE, "resp": OK(body)}], real=False)
        assert_clean(self, app)
        t = app.text("#view")
        for needle in (expect or ()):
            self.has(t, needle)
        return app

    def test_fixture_with_nulls_everywhere(self):
        app = bt("#/backtest/2330?h=1", 3000, extra=[{"re": BT_RE, "resp": {"status": 200, "body": "backtest_nulls"}}], real=False)
        assert_clean(self, app)
        t = app.text("#view")
        self.assertIn("驗證結論", t)
        self.assertIn("沒有足夠的資料可分組", t)            # empty by_confidence
        self.assertEqual(app.count("canvas"), 4)

    def test_every_top_level_block_missing_one_at_a_time(self):
        base = json.loads(FrontApp(scripts=["format.js"], fixtures=["real/backtest_2330_h1"]).eval("JSON.stringify(window.__fx.real_backtest_2330_h1)"))
        for key in ("direction", "probabilistic", "returns", "strategy", "rolling", "verdict", "params", "period", "notes"):
            body = json.loads(json.dumps(base))
            body["report"][key] = None
            self.check(body, 2500)
        # wrong types / empty containers
        body = json.loads(json.dumps(base))
        r = body["report"]
        r["strategy"]["curve"] = {"dates": [], "strategy": [], "buyhold": []}
        r["rolling"] = {"window": None, "dates": None, "accuracy": None, "base_rate": None}
        r["returns"]["coverage"] = {}
        r["returns"]["pit_hist"] = []
        r["probabilistic"]["calibration"] = None
        r["direction"]["baselines"] = None
        r["direction"]["best_baseline"] = "toString"
        r["direction"]["ci95"] = None
        r["verdict"] = {"level": "constructor", "bullets": [1, None, "ok"]}
        r["notes"] = [None, 3, "只有這則是字串"]
        app = self.check(body, 2500, ["只有這則是字串"])
        self.assertEqual(app.count(".verdict-bullets li"), 1)

    def test_ready_without_report_is_a_bad_response_not_a_crash(self):
        app = bt("#/backtest/2330?h=1", 2500, extra=[{"re": BT_RE, "resp": OK({"status": "ready"})}])
        self.assertEqual(app.errors(), [])
        self.assertIn("伺服器回應中沒有驗證結果", app.text("#view"))
        self.assertEqual(app.count(".error-box button"), 1)           # retry offered

    def test_report_that_is_not_an_object(self):
        for body in ({"status": "ready", "report": "oops"}, {"status": "ready", "report": []}, {"status": "ready", "report": 5}):
            app = bt("#/backtest/2330?h=1", 2000, extra=[{"re": BT_RE, "resp": OK(body)}])
            self.assertEqual(app.errors(), [])
            self.assertIn("沒有驗證結果", app.text("#view"))

    def test_short_history_notice(self):
        base = json.loads(FrontApp(scripts=["format.js"], fixtures=["real/backtest_2330_h1"]).eval("JSON.stringify(window.__fx.real_backtest_2330_h1)"))
        base["report"]["period"].update({"n_forecasts": 400, "n_scored": 180})
        base["report"]["direction"]["n"] = 180
        app = bt("#/backtest/2330?h=1&years=5", 3000, extra=[{"re": BT_RE, "resp": OK(base)}])
        t = app.text("#view")
        for needle in ("驗證期間比預期短", "約 1.6 年", "評分樣本只有 180 次"):
            self.has(t, needle)
        assert_clean(self, app)

    def test_stale_flag_is_shown(self):
        base = json.loads(FrontApp(scripts=["format.js"], fixtures=["real/backtest_2330_h1"]).eval("JSON.stringify(window.__fx.real_backtest_2330_h1)"))
        base["stale"] = True
        app = bt("#/backtest/2330?h=1", 3000, extra=[{"re": BT_RE, "resp": OK(base)}])
        self.assertIn("資料可能過期", app.text(".bt-head-card"))
        self.assertIn("本機快取", app.text("#view"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class PollingProgressCancel(Base):
    def test_progress_bar_message_and_elapsed_time_then_result(self):
        ready = {"status": 200, "body": "real_backtest_2330_h1"}
        rules = [{"re": BT_RE, "seq": [COMPUTING(0, 0, "排隊中…"), COMPUTING(300, 1250, "前推式預測 300/1250"), COMPUTING(900, 1250, "前推式預測 900/1250"), ready], "resp": ready}]
        app = bt("#/backtest/2330?h=1", 100, extra=rules)
        self.assertEqual(app.count(".bt-progress-card"), 1)
        self.assertEqual(app.attr(".progress", "role"), "progressbar")
        self.assertEqual(app.count(".bt-cancel button"), 1)
        self.assertEqual(app.count(".bt-head-card"), 0)
        app.settle(1600)
        self.assertIn("前推式預測 300/1250", app.text(".progress-msg"))
        self.assertIn("300 / 1,250（24%）", app.text(".progress-msg"))
        self.assertEqual(app.attr(".progress", "aria-valuenow"), "24")
        app.settle(1500)
        self.assertIn("900 / 1,250（72%）", app.text(".progress-msg"))
        self.assertIn("已等待", app.text(".progress-msg"))
        app.settle(1700)
        self.assertEqual(app.count(".bt-progress-card"), 0)
        self.assertEqual(app.count(".bt-head-card"), 1)
        assert_clean(self, app)
        self.assertEqual(len(bt_fetches(app)), 4)
        self.assertEqual(app.pending_timers() <= 12, True)

    def test_cancel_stops_polling_and_restores_the_idle_state(self):
        rules = [{"re": BT_RE, "resp": COMPUTING(10, 1250, "前推式預測 10/1250")}]
        app = bt("#/backtest/2330?h=1", 1700, extra=rules)
        n = len(bt_fetches(app))
        self.assertGreaterEqual(n, 2)
        app.click(".bt-cancel button")
        app.settle(6000)
        self.assertEqual(len(bt_fetches(app)), n, "no further polling after cancel")
        t = app.text("#view")
        self.assertIn("已取消", t)
        self.assertIn("伺服器端的計算可能仍會完成", t)
        self.assertEqual(app.count(".bt-progress-card"), 0)
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.json("document.querySelector('.bt-run').disabled"), False)
        self.assertEqual(app.text("#status"), "已取消歷史驗證")

    def test_cancel_after_a_result_keeps_the_old_result(self):
        app = bt("#/backtest/2330?h=1", 3000)
        self.assertEqual(app.count(".bt-head-card"), 1)
        app.prepend_rule({"re": BT_RE, "resp": COMPUTING(1, 100, "x")})
        app.click(".bt-f-h .seg-btn", 1)
        app.click(".bt-run")
        app.settle(300)
        self.assertEqual(app.count(".bt-progress-card"), 1)
        app.click(".bt-cancel button")
        app.settle(300)
        self.assertEqual(app.count(".bt-head-card"), 1)
        self.assertIn("已取消。目前顯示的是上一次的結果。", app.text(".slot-status"))

    def test_old_result_is_dimmed_while_recomputing_and_url_follows_what_is_shown(self):
        app = bt("#/backtest/2330?h=1", 3000)
        self.assertEqual(app.count(".slot-result.bt-stale"), 0)
        app.prepend_rule({"re": BT_RE, "resp": COMPUTING(1, 100, "x")})
        app.click(".bt-f-h .seg-btn", 1)
        app.click(".bt-run")
        app.settle(300)
        self.assertEqual(app.count(".slot-result.bt-stale"), 1)
        self.assertEqual(app.attr(".slot-result", "aria-busy"), "true")
        self.assertEqual(app.count(".bt-head-card"), 1)
        self.assertIn("h=5", app.eval("location.hash"), "while running, the URL describes the run")
        app.click(".bt-cancel button")
        app.settle(200)
        self.assertEqual(app.count(".slot-result.bt-stale"), 0)
        self.assertIsNone(app.attr(".slot-result", "aria-busy"))
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?" + DEFAULT_QS, "cancelled: the URL goes back to the result that is still shown")
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), False, "the controls still say 一週, the result says 明日")

    def test_failed_rerun_keeps_the_old_result_marks_it_and_restores_the_url(self):
        app = bt("#/backtest/2330?h=1", 3000)
        app.prepend_rule({"re": BT_RE, "resp": ERR(500, "internal", "壞了")})
        app.click(".bt-f-h .seg-btn", 1)
        app.click(".bt-run")
        app.settle(500)
        self.assertIn("壞了", app.text(".bt-error-card"))
        self.assertEqual(app.count(".bt-head-card"), 1)
        self.assertEqual(app.count(".slot-result.bt-stale"), 0)
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?" + DEFAULT_QS)
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), False)
        self.assertEqual(app.errors(), [])

    def test_leaving_the_page_stops_polling(self):
        rules = [{"re": BT_RE, "resp": COMPUTING(10, 1250, "x")}]
        app = bt("#/backtest/2330?h=1", 1700, extra=rules)
        n = len(bt_fetches(app))
        app.goto("#/method", 200)
        app.settle(8000)
        self.assertEqual(len(bt_fetches(app)), n)
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.count(".bt-progress-card"), 0)

    def test_transient_network_failures_are_tolerated(self):
        ready = {"status": 200, "body": "real_backtest_2330_h1"}
        rules = [{"re": BT_RE, "seq": [{"networkError": True}, {"networkError": True}, ready], "resp": ready}]
        app = bt("#/backtest/2330?h=1", 100, extra=rules)
        app.settle(1700)
        self.assertIn("連線不穩", app.text(".progress-msg"))
        app.settle(6000)
        self.assertEqual(app.count(".bt-head-card"), 1)
        self.assertEqual(app.errors(), [])


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Errors(Base):
    def test_unknown_symbol_has_no_retry_and_explains_what_to_do(self):
        rules = [{"re": BT_RE, "resp": ERR(404, "unknown_symbol", "找不到代號 9999。")}]
        app = bt("#/backtest/9999?h=1", 1500, extra=rules)
        t = app.text(".bt-error-card")
        for needle in ("找不到這檔股票", "找不到代號 9999。", "unknown_symbol", "請在上方「股票」欄位搜尋"):
            self.has(t, needle)
        self.assertEqual(app.count(".bt-error-card button"), 0)
        self.assertEqual(app.errors(), [])
        self.assertEqual(len(bt_fetches(app)), 1)             # a definitive answer is not polled again
        self.assertEqual(app.attr('#nav a[data-route="backtest"]', "href"), "#/backtest/2330")       # an invalid code is never remembered

    def test_insufficient_history_explanation(self):
        rules = [{"re": BT_RE, "resp": ERR(422, "insufficient_history", "歷史資料不足。")}]
        app = bt("#/backtest/2330?h=1", 1500, extra=rules)
        t = app.text(".bt-error-card")
        self.assertIn("歷史資料不足", t)
        self.assertIn("新上市的標的暫時無法驗證", t)

    def test_server_error_then_retry_succeeds(self):
        ready = {"status": 200, "body": "real_backtest_2330_h1"}
        rules = [{"re": BT_RE, "seq": [ERR(500, "internal", "伺服器發生錯誤"), ready], "resp": ready}]
        app = bt("#/backtest/2330?h=1", 1500, extra=rules)
        self.assertIn("伺服器發生錯誤", app.text(".bt-error-card"))
        self.assertEqual(app.count(".bt-error-card .btn"), 1)
        app.click(".bt-error-card .btn")
        app.settle(2000)
        self.assertEqual(app.count(".bt-error-card"), 0)
        self.assertEqual(app.count(".bt-head-card"), 1)
        self.assertEqual(app.errors(), [])

    def test_network_down_after_retries(self):
        rules = [{"re": BT_RE, "resp": {"networkError": True}}]
        app = bt("#/backtest/2330?h=1", 20000, extra=rules)
        self.assertIn("無法連線到伺服器", app.text(".bt-error-card"))
        self.assertEqual(app.errors(), [])

    def test_bad_request_has_no_retry(self):
        rules = [{"re": BT_RE, "resp": ERR(400, "bad_request", "參數 thr 必須介於 0.5 與 0.95 之間。")}]
        app = bt("#/backtest/2330?h=1", 1500, extra=rules)
        self.assertIn("請求內容有誤", app.text(".bt-error-card"))
        self.assertEqual(app.count(".bt-error-card button"), 0)

    def test_render_exception_in_one_block_degrades_only_that_block(self):
        # a getter that throws while building one card must not take the page down: the error is logged and the card replaced
        base = json.loads(FrontApp(scripts=["format.js"], fixtures=["real/backtest_2330_h1"]).eval("JSON.stringify(window.__fx.real_backtest_2330_h1)"))
        app = FrontApp(hash="#/backtest/2330?h=1", boot=False)
        app.set_rules([{"re": BT_RE, "resp": OK(base)}] + default_rules(True))
        app.boot()
        app.eval("""(function () { var orig = TQ.chartCard; TQ.chartCard = function (title) { if (title.indexOf('PIT') === 0) throw new Error('boom'); return orig.apply(this, arguments); }; })()""")
        app.goto("#/backtest/2330?h=5", 3000)
        t = app.text("#view")
        self.assertIn("驗證結論", t)
        self.assertIn("預測品質", t)
        self.assertTrue(any("boom" in e for e in app.errors()))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ControlInteractions(Base):
    def test_changing_controls_marks_the_result_stale_until_run(self):
        app = bt("#/backtest/2330?h=1", 3000)
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), True)
        n = len(bt_fetches(app))
        app.click(".bt-f-h .seg-btn", 1)                           # 一週
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), False)
        self.assertIn("設定已變更，尚未套用", app.text(".bt-dirty"))
        self.assertEqual(len(bt_fetches(app)), n, "changing a control must not start a computation")
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?" + DEFAULT_QS, "the URL describes the displayed result until it is re-run")
        app.click(".bt-f-h .seg-btn", 0)                           # back to what is displayed
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), True)

    def test_run_uses_the_new_settings_and_updates_the_url(self):
        app = bt("#/backtest/2330?h=1", 3000)
        app.click(".bt-f-h .seg-btn", 1)
        app.click(".bt-f-exec .seg-btn", 1)
        app.click(".bt-f-years .seg-btn", 3)
        app.eval("(function () { var s = document.querySelector('#bt-thr'); s.value = '0.62'; s.dispatchEvent(new Event('input', { bubbles: true })); var d = document.querySelector('#bt-disc'); d.value = '0.28'; d.dispatchEvent(new Event('change', { bubbles: true })); })()")
        self.assertEqual(app.text(".bt-thr-out"), "62%")
        self.assertEqual(app.attr("#bt-thr", "aria-valuetext"), "62%")
        self.assertIn("預估來回交易成本約 0.48%", app.text(".bt-cost-hint"))
        app.click(".bt-run")
        app.settle(3000)
        self.assertEqual(bt_fetches(app)[-1], "/api/backtest?symbol=2330&h=5&thr=0.62&exec=close&disc=0.28&years=8")
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=5&thr=0.62&exec=close&disc=0.28&years=8")
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), True)
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.count(".bt-head-card"), 1)

    def test_run_from_idle_then_dirty_state_logic(self):
        app = bt("#/backtest/2330", 800)
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), True)
        app.click(".bt-f-years .seg-btn", 0)
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), True, "nothing displayed yet: nothing can be out of date")
        app.click(".bt-run")
        app.settle(3000)
        self.assertEqual(bt_fetches(app), ["/api/backtest?symbol=2330&h=1&thr=0.55&exec=next_open&disc=0.6&years=2"])
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=1&thr=0.55&exec=next_open&disc=0.6&years=2")

    def test_form_submit_runs_once(self):
        app = bt("#/backtest/2330", 800)
        app.eval("document.querySelector('form.bt-form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))")
        app.settle(2500)
        self.assertEqual(len(bt_fetches(app)), 1)

    def test_run_twice_quickly_aborts_the_first_request(self):
        rules = [{"re": BT_RE, "resp": COMPUTING(1, 10, "x")}]
        app = bt("#/backtest/2330", 300, extra=rules)
        app.click(".bt-run")
        app.settle(100)
        app.click(".bt-f-h .seg-btn", 1)
        app.click(".bt-run")
        app.settle(100)
        self.assertEqual(app.count(".bt-progress-card"), 1)
        self.assertEqual(app.errors(), [])
        self.assertTrue(bt_fetches(app)[-1].startswith("/api/backtest?symbol=2330&h=5"))

    def test_reset_restores_defaults_everywhere(self):
        app = bt("#/backtest/2330?h=5&thr=0.65&exec=close&disc=0.45&years=2", 3000)
        self.assertIn("4.5 折", app.texts("#bt-disc option"))
        app.click(".bt-reset")
        self.assertEqual(seg_values(app, ".bt-f-h"), ["true", "false", "false"])
        self.assertEqual(seg_values(app, ".bt-f-exec"), ["true", "false"])
        self.assertEqual(seg_values(app, ".bt-f-years"), ["false", "false", "true", "false"])
        self.assertEqual(app.json("document.querySelector('#bt-thr').value"), "0.55")
        self.assertEqual(app.text(".bt-thr-out"), "55%")
        self.assertEqual(app.json("document.querySelector('#bt-disc').value"), "0.6")
        self.assertNotIn("4.5 折", app.texts("#bt-disc option"))          # the custom discount from the URL is gone from the list
        self.assertEqual(app.json("document.querySelector('.bt-dirty').hidden"), False)

    def test_custom_discount_from_url_is_selectable(self):
        app = bt("#/backtest/2330?disc=0.45", 1500)
        self.assertEqual(app.json("document.querySelector('#bt-disc').value"), "0.45")
        self.assertIn("4.5 折", app.texts("#bt-disc option"))

    def test_slider_is_clamped(self):
        app = bt("#/backtest/2330", 500)
        app.eval("(function () { var s = document.querySelector('#bt-thr'); s.value = '0.99'; s.dispatchEvent(new Event('input', { bubbles: true })); })()")
        self.assertEqual(app.text(".bt-thr-out"), "70%")
        app.eval("(function () { var s = document.querySelector('#bt-thr'); s.value = 'abc'; s.dispatchEvent(new Event('input', { bubbles: true })); })()")
        self.assertEqual(app.text(".bt-thr-out"), "70%")

    def test_search_selection_keeps_settings_only_when_a_result_is_shown(self):
        hit = {"re": r"^/api/search\?q=0050", "resp": OK({"status": "ready", "results": [{"code": "0050", "name": "元大台灣50", "market": "TWSE", "kind": "etf"}]})}
        app = bt("#/backtest/2330?h=5&thr=0.6", 3000, extra=[hit])
        app.type("#bt-stock", "0050")
        app.settle(400)
        app.press("#bt-stock", "Enter")
        app.settle(3000)
        self.assertTrue(app.eval("location.hash").startswith("#/backtest/"))
        code = app.eval("location.hash").split("?")[0].split("/")[-1]
        self.assertEqual(code, "0050")
        self.assertIn("h=5&thr=0.6", app.eval("location.hash"))
        self.assertTrue(bt_fetches(app)[-1].startswith("/api/backtest?symbol=0050&h=5&thr=0.6"))
        # idle page: just the new code
        app = bt("#/backtest/2330", 800, extra=[hit])
        app.type("#bt-stock", "0050")
        app.settle(400)
        app.press("#bt-stock", "Enter")
        app.settle(500)
        self.assertEqual(app.eval("location.hash"), "#/backtest/0050")
        self.assertEqual(bt_fetches(app), [])

    def test_copy_link_button_only_when_clipboard_exists(self):
        app = bt("#/backtest/2330?h=1", 3000)
        self.assertEqual(app.count(".bt-copy"), 0)
        app = FrontApp(hash="#/backtest/2330?h=1", boot=False)
        app.set_rules(default_rules(True))
        app.eval("window.__copied = null; window.navigator.clipboard = { writeText: function (t) { window.__copied = t; return Promise.resolve(); } };")
        app.boot()
        app.settle(3000)
        self.assertEqual(app.count(".bt-copy"), 1)
        app.click(".bt-copy")
        app.settle(100)
        self.assertEqual(app.eval("window.__copied"), "http://localhost:8000/#/backtest/2330?" + DEFAULT_QS)
        self.assertIn("已複製", app.text(".bt-copy"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class StockPageLinks(Base):
    def test_stock_summary_links_carry_the_horizon(self):
        app = FrontApp(hash="#/stock/2330", boot=False)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(8000)
        hrefs = app.json("Array.prototype.map.call(document.querySelectorAll('a[href^=\"#/backtest\"]'), function (a) { return a.getAttribute('href'); })")
        self.assertIn("#/backtest/2330?h=1", hrefs)
        self.assertIn("#/backtest/2330?h=5", hrefs)
        self.assertIn("#/backtest/2330", hrefs)
        app.goto("#/backtest/2330?h=5", 3000)
        self.assertEqual(app.errors(), [])
        self.assertEqual(seg_values(app, ".bt-f-h"), ["false", "true", "false"])
        self.assertEqual(app.count(".bt-head-card"), 1)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Internals(Base):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js", "charts.js", "core.js", "view_backtest.js"], fixtures=[])

    def test_cost_breakdown_matches_config_roundtrip_cost(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend import config
        for kind in ("stock", "etf"):
            for disc in (0.28, 0.35, 0.5, 0.6, 0.85, 1.0):
                for slip in (0, 5, 12.5):
                    js = self.app.json(f"TQ.views.backtest._internals.costBreakdown({disc}, {slip}, {json.dumps(kind)}).total")
                    self.assertAlmostEqual(js, config.roundtrip_cost(kind, disc, slip), places=12, msg=(kind, disc, slip))

    def test_constants_mirror_config(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend import config
        c = self.app.json("TQ.views.backtest._internals.COST")
        self.assertEqual((c["commission"], c["taxStock"], c["taxEtf"], c["slippageBps"]), (config.COMMISSION_RATE, config.TAX_STOCK, config.TAX_ETF, config.SLIPPAGE_BPS))
        d = self.app.json("TQ.views.backtest._internals.DEFAULTS")
        self.assertEqual((d["thr"], d["disc"]), (config.BUY_P_UP, config.COMMISSION_DISCOUNT))
        self.assertEqual(d["years"] * config.TRADING_DAYS_PER_YEAR, 1225)           # the server special-cases years == 5 -> TEST_BARS_DEFAULT
        self.assertEqual(config.TEST_BARS_DEFAULT, 1250)

    def test_discount_labels_and_coverage_judgement(self):
        i = "TQ.views.backtest._internals."
        self.assertEqual(self.app.json(i + "discLabel(0.28)"), "2.8 折")
        self.assertEqual(self.app.json(i + "discLabel(0.6)"), "6 折")
        self.assertEqual(self.app.json(i + "discLabel(1)"), "不打折（牌告價）")
        self.assertEqual(self.app.json(i + "discLabel(null)"), "—")
        self.assertEqual(self.app.json(i + "coverageJudge(0.9, 0.89)"), "接近預期")
        self.assertIn("偏低", self.app.json(i + "coverageJudge(0.9, 0.8)"))
        self.assertIn("偏高", self.app.json(i + "coverageJudge(0.5, 0.6)"))
        self.assertEqual(self.app.json(i + "coverageJudge(0.5, null)"), "—")
        self.assertEqual(self.app.json(i + "coverageOf({ '0.50': 0.4, '0.90': 0.9 }, 0.5)"), 0.4)
        self.assertIsNone(self.app.json(i + "coverageOf({ '0.80': 0.8 }, 0.5)"))
        self.assertIsNone(self.app.json(i + "coverageOf(null, 0.5)"))


if __name__ == "__main__":
    unittest.main()
