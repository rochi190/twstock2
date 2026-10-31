"""今日模型觀察 view (#/picks): horizon control, progress, candidates / avoids / full table, portfolio, universe-validation panel."""
from __future__ import annotations

import json
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, assert_clean, audit, default_resp, default_rules, fixture

OK = lambda body: {"status": 200, "body": body}
COMPUTING = lambda d, t, m: OK({"status": "computing", "progress": {"done": d, "total": t, "message": m}})


def picks(settle=3000, extra=(), real=True, **kw):
    app = FrontApp(hash="#/picks", real=real, boot=False, **kw)
    app.set_rules(list(extra) + default_rules(real))
    app.boot()
    app.settle(settle)
    return app


def rec_fetches(app):
    return [x for x in app.fetched() if x.startswith("/api/recommendations")]


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class RealOneWeek(unittest.TestCase):
    """The real backend answers 'no candidates' for 明日 / 一週 -- the common case must read well."""

    @classmethod
    def setUpClass(cls):
        cls.app = picks(3000)
        cls.text = cls.app.text("#view")

    def test_clean(self):
        assert_clean(self, self.app)
        self.assertEqual(audit(self.app)["h1"], 1)
        self.assertEqual(self.app.eval("document.title"), "今日模型觀察 · 台股量化實驗室")

    def test_banner_market_strip_and_empty_buys(self):
        t = self.text
        # Rewritten for the fixed backend (HANDOFF 4A): the real h = 5 payload is dated 2026-10-30, ranked by rank_signal, carries a real
        # `evidence` block (no_edge, the basket loses to the universe after costs) and an all-cash portfolio with suppressed_reason B0.
        for needle in ("資料截至 2026-10-30 (週五) 收盤", "非即時報價", "本次掃描 60 / 60 檔", "2026-10-30 21:08", "目前市場已收盤",
                       "加權指數", "45,212", "▼ −2.49%", "51.9%", "52.8%", "高波動震盪",
                       "偏多候選（嚴格條件）", "目前沒有同時符合偏多候選條件的標的", "沒有訊號本身就是模型的答案",
                       "模型配置範例", "持股檔數", "全數持有現金", "為什麼沒有權重", "偏空／避開（嚴格條件）", "目前沒有明確偏空"):
            self.assertIn(needle, t)
        self.assertEqual(self.app.count(".pick-card"), 0)
        # data-asof banner + market-closed banner + the evidence banner (all three are .banner)
        self.assertEqual(self.app.count(".banner"), 3)

    def test_evidence_banner_is_amber_and_never_says_buy(self):
        """DESIGN 9.4 / audit P-10: with no_edge (and a basket that loses to the universe after costs) the page must not read as advice."""
        t = self.text
        self.assertIn("沒有統計上顯著的預測力", t)
        self.assertIn("IC +0.006，p=0.66", t)
        self.assertIn("扣成本後前 5 名組合年化 9.7%", t)
        self.assertEqual(self.app.count(".ev-host .banner-warn"), 1)
        self.assertEqual(self.app.count(".ev-host .ev-open"), 1)
        for forbidden in ("建議買進", "偏多·買進", "買進候選"):
            self.assertNotIn(forbidden, t)

    def test_relative_strength_lists_show_sigma_and_low_vol_and_cost_marks(self):
        t = self.app.text(".lead-card")
        for needle in ("今日相對強度排名", "相對偏多 Top 10", "相對最弱（上漲機率仍高於 50%）", "不代表看空", "波動 σ", "低波動", "未覆蓋成本",
                       "相對強度 +0.088", "Spearman 相關為 −0.50"):
            self.assertIn(needle, t)
        self.assertEqual(self.app.count(".lead-leaders .lead-item"), 10)
        self.assertEqual(self.app.count(".lead-laggards .lead-item"), 10)
        self.assertEqual(self.app.json("document.querySelector('.lead-leaders .lead-item').querySelectorAll('.chip').length"), 3)   # 相對偏多 + 低波動 + 未覆蓋成本
        # every leader / laggard row carries its sigma
        self.assertEqual(self.app.json("[].slice.call(document.querySelectorAll('.lead-item')).every(function (e) { return e.textContent.indexOf('波動 σ') >= 0; })"), True)
        self.assertNotIn("相對偏空 Top", t)         # all laggards have p_up > 0.5: the heading must not claim a bearish view

    def test_constant_cost_score_column_is_hidden_when_shrinkage_collapses_it(self):
        """B = 0 makes score_shrunk identical for all 60 names: a column of one number is noise (HANDOFF 4A.2)."""
        head = self.app.text(".all-card thead")
        self.assertNotIn("成本後分數", head)
        for needle in ("相對強度", "波動 σ", "淨預期報酬", "模型一致度"):
            self.assertIn(needle, head)

    def test_horizon_control_defaults_to_one_week(self):
        self.assertEqual(self.app.attr(".page-head .seg", "role"), "radiogroup")
        self.assertEqual(self.app.attr(".page-head .seg", "aria-label"), "預測期間")
        self.assertEqual(self.app.texts(".page-head .seg-btn"), ["明日", "一週", "一個月"])
        self.assertEqual([self.app.attr(".page-head .seg-btn", "aria-checked")] + [self.app.json(f"document.querySelectorAll('.page-head .seg-btn')[{i}].getAttribute('aria-checked')") for i in (1, 2)], ["false", "true", "false"])
        self.assertEqual(rec_fetches(self.app), ["/api/recommendations?h=5&top=10"])

    def test_full_table_has_every_scored_name_and_is_sortable(self):
        app = picks(2000)
        self.assertEqual(app.json("document.querySelectorAll('.all-card tbody tr').length"), 60)
        first = app.json("document.querySelector('.all-card tbody tr').textContent")
        self.assertIn("台積電", first)               # rank 1 by rank_signal (+0.088)
        self.assertIn("▼ −1.86%", first)
        self.assertIn("+0.088", first)
        # sort by 1-day change: the first click on a numeric column sorts descending (largest first)
        app.click(".all-card thead .th-btn", 3)
        self.assertEqual(app.json("document.querySelectorAll('.all-card thead th')[3].getAttribute('aria-sort')"), "descending")
        top = app.json("document.querySelector('.all-card tbody tr').textContent")
        self.assertIn("長榮", top)
        self.assertIn("▲ +4.02%", top)             # 長榮 is the biggest gainer of the day
        app.click(".all-card thead .th-btn", 3)
        self.assertEqual(app.json("document.querySelectorAll('.all-card thead th')[3].getAttribute('aria-sort')"), "ascending")
        self.assertIn("▼ −9.80%", app.json("document.querySelector('.all-card tbody tr').textContent"))
        # names link to the stock page
        self.assertEqual(app.json("document.querySelector('.all-card tbody a').getAttribute('href')").startswith("#/stock/"), True)
        assert_clean(self, app)

    def test_signal_filter(self):
        app = picks(2000)
        app.click(".all-card .seg-btn", 1)         # 買進
        self.assertEqual(app.json("document.querySelectorAll('.all-card tbody tr').length"), 1)
        self.assertIn("沒有符合條件的標的", app.text(".all-card tbody"))
        self.assertIn("顯示 0 / 60 檔", app.text(".all-card .filter-count"))
        app.click(".all-card .seg-btn", 2)         # 觀望
        self.assertEqual(app.json("document.querySelectorAll('.all-card tbody tr').length"), 60)
        app.click(".all-card .seg-btn", 0)
        self.assertIn("顯示 60 / 60 檔", app.text(".all-card .filter-count"))

    def test_notes(self):
        for needle in ("如何解讀這份名單", "收縮（shrinkage）", "贏家詛咒", "存活者偏差", "機率不是保證", "計算方式與假設", "非投資建議"):
            self.assertIn(needle, self.text)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class HorizonSwitching(unittest.TestCase):
    def test_one_month_has_candidates_cards_with_all_fields(self):
        app = picks(2000)
        app.click(".page-head .seg-btn", 2)
        app.settle(1000)
        assert_clean(self, app)
        self.assertEqual(rec_fetches(app), ["/api/recommendations?h=5&top=10", "/api/recommendations?h=20&top=10"])
        self.assertEqual(json.loads(app.eval("window.__ls['tq.picksH']")), 20)
        self.assertEqual(app.count(".pick-card"), 10)
        card = app.text(".pick-card")
        # Rewritten for the fixed backend: the real h = 20 payload ranks 中華電 first (rank_signal +0.173), has 27 names passing the strict
        # screen (10 shown), no validation yet (evidence.source = none) and an all-cash portfolio (B = 0, so every weight is 0).
        for needle in ("中華電", "2412", "160.0", "▲ +1.27%", "上漲機率", "57.8%", "淨預期報酬", "▲ +0.12%", "模型配置權重", "模型一致度 低",
                       "相對強度 +0.173", "預測波動 σ 3.63%", "低波動", "偏多候選", "均線多空", "未來 20 個交易日", "共 27 檔符合"):
            self.assertIn(needle, card + app.text(".buys-card"))
        self.assertEqual(app.attr(".pick-card .stock-link", "href"), "#/stock/2412")
        t = app.text("#view")
        for needle in ("持股檔數", "0 檔", "股票部位", "0%", "現金比例", "100%", "為什麼沒有權重", "全數持有現金", "尚未取得「一個月」的歷史驗證"):
            self.assertIn(needle, t)
        self.assertEqual(app.count(".pick-card.pick-up"), 10)           # bullish cards are marked (red top border), not by colour alone: they carry the 偏多 chip
        self.assertEqual(app.count(".pick-card .chip-up"), 10)
        # no validation => the advice wording stays locked, everywhere (cards, table, chips)
        for forbidden in ("建議買進", "偏多·買進"):
            self.assertNotIn(forbidden, t)
        self.assertEqual(app.count(".ev-host .banner-neutral"), 1)
        self.assertEqual(app.count(".ev-host .ev-run"), 1)
        # the portfolio is empty: no weight table, and no row of the full table carries a weight
        self.assertEqual(app.count(".pf-card .stacked-bar"), 0)
        self.assertEqual(app.json("document.querySelector('.all-card tbody tr').textContent").count("%"), 4)       # 1日, 上漲機率, 淨預期, σ -- no weight

    def test_persisted_horizon_is_used_on_load(self):
        app = picks(2000, storage={"tq.picksH": "1"})
        self.assertEqual(rec_fetches(app), ["/api/recommendations?h=1&top=10"])
        self.assertIn("下一個交易日", app.text("#view"))
        app = picks(2000, storage={"tq.picksH": "99"})
        self.assertEqual(rec_fetches(app), ["/api/recommendations?h=5&top=10"])

    def test_switching_back_uses_the_in_memory_result(self):
        app = picks(2000)
        app.click(".page-head .seg-btn", 0)
        app.settle(500)
        app.click(".page-head .seg-btn", 1)
        app.settle(500)
        self.assertEqual(rec_fetches(app), ["/api/recommendations?h=5&top=10", "/api/recommendations?h=1&top=10"])
        self.assertIn("未來 5 個交易日", app.text(".buys-card"))

    def test_switch_while_computing_aborts_the_old_poll(self):
        slow = COMPUTING(10, 122, "預測 10/60：2330 台積電")
        app = picks(1000, extra=[{"re": r"^/api/recommendations\?h=5&", "resp": slow}])
        self.assertIn("正在計算今日模型觀察", app.text("#view"))
        self.assertIn("預測 10/60：2330 台積電", app.text("#view"))
        self.assertEqual(app.attr(".progress-card .progress", "aria-valuenow"), "8")
        n5 = len([x for x in rec_fetches(app) if "h=5" in x])
        app.click(".page-head .seg-btn", 0)
        app.settle(6000)
        self.assertEqual(len([x for x in rec_fetches(app) if "h=5" in x]), n5, "the h=5 poll must stop after switching")
        self.assertEqual(app.count(".progress-card"), 0)
        self.assertIn("下一個交易日", app.text(".buys-card"))
        assert_clean(self, app)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class StatesAndVariants(unittest.TestCase):
    def test_progress_bar_then_result(self):
        seq = [COMPUTING(0, 122, "載入大盤與股票行情…"), COMPUTING(61, 122, "預測 20/60：2454 聯發科"), default_resp(True, r"^/api/recommendations\?h=5&")]
        app = picks(300, extra=[{"re": r"^/api/recommendations\?h=5&", "seq": seq, "resp": seq[0]}])
        self.assertEqual(app.attr(".progress", "role"), "progressbar")
        self.assertEqual(app.attr(".progress", "aria-valuenow"), "0")
        self.assertIn("載入大盤與股票行情…", app.text(".progress"))
        app.settle(1500)
        self.assertEqual(app.attr(".progress", "aria-valuenow"), "50")
        self.assertIn("61 / 122", app.text(".progress"))
        app.settle(1700)
        self.assertEqual(app.count(".progress"), 0)
        self.assertEqual(app.count(".all-card"), 1)
        assert_clean(self, app)

    def test_indeterminate_progress_without_total(self):
        app = picks(500, extra=[{"re": r"^/api/recommendations", "resp": OK(fixture("computing_nototal.json"))}])
        self.assertIn("progress-indeterminate", app.attr(".progress", "class"))
        self.assertIsNone(app.attr(".progress", "aria-valuenow"))
        assert_clean(self, app)

    def test_server_error_retry_and_network_error(self):
        err = {"status": 500, "body": {"status": "error", "error": {"code": "internal", "message": "伺服器內部發生錯誤，請稍後再試。"}}}
        ok = default_resp(True, r"^/api/recommendations\?h=5&")
        app = picks(1500, extra=[{"re": r"^/api/recommendations\?h=5&", "seq": [err, ok], "resp": err}])
        self.assertIn("伺服器內部發生錯誤", app.text("#view"))
        self.assertEqual(app.count(".pick-grid"), 0)
        app.click(".error-box button")
        app.settle(1500)
        self.assertEqual(app.count(".all-card"), 1)
        app = picks(1800, extra=[{"re": r"^/api/recommendations", "resp": {"networkError": True}}])
        self.assertIn("連線不穩，重新嘗試中（1/3）", app.text("#view"))     # transient failures are retried, with feedback
        app.settle(15000)
        self.assertIn("無法連線到伺服器", app.text("#view"))
        self.assertEqual(len(rec_fetches(app)), 4)

    def test_schema_payload_with_buys_and_avoids(self):
        app = picks(2000, real=False)
        assert_clean(self, app)
        rec = fixture("recommendations_h5.json")
        self.assertEqual(app.count(".pick-card"), len(rec["buys"]))
        self.assertEqual(app.count(".avoid-item"), len(rec["avoids"]))
        self.assertGreater(len(rec["buys"]), 0)
        b = rec["buys"][0]
        self.assertIn(b["name"], app.text(".pick-card"))
        self.assertIn(b["reason"], app.text(".pick-card"))
        if rec["avoids"]:
            a = rec["avoids"][0]
            avoid = app.text(".avoid-item")
            self.assertIn(a["name"], avoid)
            self.assertIn("偏空／避開", avoid)
            self.assertEqual(app.count(".avoid-item .chip-down"), len(rec["avoids"]))       # avoid = green chip with ▼ + text
        self.assertEqual(app.json("document.querySelectorAll('.all-card tbody tr').length"), len(rec["all"]))

    def test_empty_universe_and_missing_blocks(self):
        app = picks(2000, extra=[{"re": r"^/api/recommendations\?h=5&", "resp": OK(fixture("recommendations_empty.json"))}])
        assert_clean(self, app)
        self.assertIn("沒有符合條件的標的", app.text(".all-card"))
        self.assertIn("今日沒有符合條件的偏多候選", app.text(".pf-card"))
        bare = {"status": "ready"}
        app = picks(2000, extra=[{"re": r"^/api/recommendations\?h=5&", "resp": OK(bare)}])
        assert_clean(self, app)
        junk = {"status": "ready", "asof": 5, "buys": "x", "avoids": [None, 3, {"code": "1"}], "all": [None, {"code": "9", "p_up": "bad", "signal": 7}],
                "market": [], "portfolio": 5, "notes": [None, 3, "ok"], "skipped": 4}
        app = picks(2000, extra=[{"re": r"^/api/recommendations\?h=5&", "resp": OK(junk)}])
        assert_clean(self, app)
        self.assertIn("ok", app.text("#view"))

    def test_stale_and_intraday_banners(self):
        rec = fixture("real/recommendations_h5.json")
        rec["stale"] = True
        rec["asof_complete"] = False
        rec["skipped"] = [{"code": "1"}]
        app = picks(2000, extra=[{"re": r"^/api/recommendations\?h=5&", "resp": OK(rec)},
                                  {"re": r"^/api/status", "resp": OK({"status": "ready", "market_open": True, "last_trading_date": "2026-10-23"})}])
        assert_clean(self, app)
        t = app.text("#view")
        for needle in ("盤中資料", "目前為盤中", "本機快取", "略過"):
            self.assertIn(needle, t)

    def test_leaving_the_view_stops_everything(self):
        app = picks(1000, extra=[{"re": r"^/api/recommendations", "resp": COMPUTING(1, 10, "x")}])
        app.goto("#/method", 100)
        n = len(app.fetched())
        app.settle(20000)
        self.assertEqual(len(app.fetched()), n)
        self.assertEqual(app.errors(), [])

    def test_phone_width(self):
        app = picks(2500, viewport=360)
        assert_clean(self, app)
        app.click(".page-head .seg-btn", 2)
        app.settle(1000)
        self.assertEqual(app.count(".pick-card"), 10)
        assert_clean(self, app)

    def test_dark_theme_clean(self):
        app = picks(2500, dark=True)
        assert_clean(self, app)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class UniversePanel(unittest.TestCase):
    def open_panel(self, extra=(), real=True, **kw):
        app = picks(1500, extra=extra, real=real, **kw)
        return app

    def test_collapsed_by_default_and_nothing_fetched_until_asked(self):
        app = self.open_panel()
        self.assertEqual(app.attr(".uv-card .collapse-btn", "aria-expanded"), "false")
        self.assertTrue(app.eval("document.querySelector('.uv-card .collapse-body').hidden"))
        self.assertEqual(app.attr(".uv-card .collapse-btn", "aria-controls"), app.attr(".uv-card .collapse-body", "id"))
        self.assertEqual([x for x in app.fetched() if "universe" in x], [])
        app.click(".uv-card .collapse-btn")
        self.assertEqual(app.attr(".uv-card .collapse-btn", "aria-expanded"), "true")
        self.assertFalse(app.eval("document.querySelector('.uv-card .collapse-body').hidden"))
        self.assertIn("開始驗證（一週）", app.text(".uv-card"))
        for needle in ("前 5 名", "本頁列出前 10 名", "不套用「偏多候選」", "相對強度"):       # the real rule, spelled out before the run (HANDOFF 4A.3)
            self.assertIn(needle, app.text(".uv-card"))
        self.assertEqual([x for x in app.fetched() if "universe" in x], [])           # opening is not enough: heavy job needs the button
        app.click(".uv-card .collapse-btn")
        self.assertTrue(app.eval("document.querySelector('.uv-card .collapse-body').hidden"))

    def test_run_shows_progress_then_verdict_kpis_stats_and_chart(self):
        comp = COMPUTING(89, 122, "前推式預測 29/60：2301 光寶科")
        ready = default_resp(True, r"^/api/backtest/universe")
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "seq": [comp, ready], "resp": ready}])
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(300)
        self.assertEqual([x for x in app.fetched() if "universe" in x], ["/api/backtest/universe?h=5&top=5"])
        self.assertIn("前推式預測 29/60：2301 光寶科", app.text(".uv-card"))
        self.assertIn("89 / 122", app.text(".uv-card"))
        self.assertEqual(app.attr(".uv-card .progress", "aria-valuenow"), "73")
        app.settle(3000)
        assert_clean(self, app)
        t = app.text(".uv-card")
        # real fixture backtest_universe_h5.json (fixed backend, 2026-10-30): no_edge, the top-5 basket is calmer than the universe and
        # loses to the volatility-only control.
        for needle in ("沒有統計上顯著的選股優勢", "無預測優勢", "排序能力 IC", "+0.006", "IR 0.02", "t 0.44", "p 0.66", "精選股上漲比例", "56.5%",
                       "p 0.008", "前段 − 後段報酬差", "−0.14%", "驗證期間", "2021-09-03 ~ 2026-10-30", "1,249 個交易日", "60 檔",
                       "模型精選：每日相對強度最高的前 5 名", "全體等權重（不扣成本）", "加權指數", "累積報酬", "夏普比率", "最大回撤", "年換手率", "26.2 倍",
                       "累積淨值曲線", "各檔個股的準確率", "存活者偏差"):
            self.assertIn(needle, t)
        # exposure comparison (P-05): the volatility / beta reminder is OUTSIDE the collapsed <details>, next to the numbers
        for needle in ("曝險對照", "精選股平均波動 σ", "0.54 倍", "精選股平均市場 β", "無預測對照", "沒有勝過這個不使用任何預測的波動度排序"):
            self.assertIn(needle, t)
        self.assertIn("曲線的差異可能只是曝險不同", app.text(".uv-card .chart-card"))           # P-05: reminder tight to the equity curve
        self.assertEqual(app.json("document.querySelectorAll('.uv-card .uv-block').length >= 2"), True)
        self.assertEqual(app.json("document.querySelector('.uv-card details .uv-block')"), None)
        self.assertEqual(app.json("document.querySelectorAll('.uv-card .verdict').length"), 1)
        # equity-curve chart is mounted, drawn and summarised
        self.assertGreaterEqual(app.json("TQ.charts.mountedCount()"), 2)
        label = app.attr(".uv-card canvas", "aria-label")
        self.assertIn("累積淨值曲線圖", label)
        self.assertGreater(app.json("document.querySelector('.uv-card canvas').getContext('2d').ops.length"), 50)
        legend = app.text(".uv-card .chart-legend")
        self.assertIn("模型精選：每日相對強度最高的前 5 名", legend)
        self.assertIn("無預測對照：每日預測波動度最高的前 5 名", legend)
        self.assertIn("加權指數", legend)
        # per-stock table is sortable and linked
        self.assertEqual(app.json("document.querySelector('.uv-card details table tbody a').getAttribute('href')").startswith("#/stock/"), True)

    def test_result_is_kept_per_horizon_and_reopen_does_not_refetch(self):
        app = self.open_panel()
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(1500)
        self.assertIn("累積淨值曲線", app.text(".uv-card"))
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .collapse-btn")
        self.assertIn("累積淨值曲線", app.text(".uv-card"))
        self.assertEqual(len([x for x in app.fetched() if "universe" in x]), 1)
        # another horizon starts from a clean panel (the validation depends on h)
        app.click(".page-head .seg-btn", 2)
        app.settle(1000)
        app.click(".uv-card .collapse-btn")
        self.assertIn("開始驗證（一個月）", app.text(".uv-card"))
        app.click(".uv-card .btn-primary")
        app.settle(500)
        self.assertEqual([x for x in app.fetched() if "universe" in x][-1], "/api/backtest/universe?h=20&top=5")

    def test_cancel_stops_polling(self):
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "resp": COMPUTING(1, 100, "x")}])
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(400)
        app.click(".uv-progress .btn")
        n = len([x for x in app.fetched() if "universe" in x])
        app.settle(15000)
        self.assertEqual(len([x for x in app.fetched() if "universe" in x]), n)
        self.assertIn("開始驗證", app.text(".uv-card"))
        self.assertEqual(app.errors(), [])

    def test_error_then_retry(self):
        err = {"status": 500, "body": {"status": "error", "error": {"code": "internal", "message": "宇宙驗證失敗"}}}
        ok = default_resp(True, r"^/api/backtest/universe")
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "seq": [err, ok], "resp": err}])
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(500)
        self.assertIn("宇宙驗證失敗", app.text(".uv-card"))
        app.click(".uv-card .error-box button")
        app.settle(1500)
        self.assertIn("排序能力 IC", app.text(".uv-card"))
        assert_clean(self, app)

    def test_schema_variants_nulls_and_missing_benchmark(self):
        app = self.open_panel(real=False)
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(1500)
        assert_clean(self, app)
        self.assertIn("橫斷面排序有微弱但不穩定的資訊", app.text(".uv-card"))
        nul = fixture("backtest_universe_nulls.json")
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "resp": OK(nul)}], real=False)
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(1500)
        assert_clean(self, app)
        t = app.text(".uv-card")
        self.assertNotIn("加權指數", t.split("累積淨值")[0].split("績效比較")[-1])      # no TAIEX column
        self.assertEqual(app.count(".uv-card details table"), 0)                         # no per-stock table
        junk = {"status": "ready", "report": {"verdict": 5, "ic": [], "picks": None, "top": "x", "curve": {"dates": "a"}, "per_stock": [None, 5], "notes": [None]}}
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "resp": OK(junk)}], real=False)
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(1500)
        assert_clean(self, app)

    def test_leaving_during_universe_poll_stops_it(self):
        app = self.open_panel(extra=[{"re": r"^/api/backtest/universe", "resp": COMPUTING(1, 100, "x")}])
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(400)
        app.goto("#/method", 100)
        n = len(app.fetched())
        app.settle(15000)
        self.assertEqual(len(app.fetched()), n)
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 0)


if __name__ == "__main__":
    unittest.main()
