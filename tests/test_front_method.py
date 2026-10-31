"""方法說明 view (#/method[/section]): content coverage, the HTML/CSS equation renderer, table of contents that does not break the hash
router, deep links, phone / desktop behaviour.  V8 + fake DOM."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, assert_clean, audit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SECTION_IDS = ["intro", "pipeline", "data", "targets", "garch", "hmm", "dlm", "naive", "dma", "calibration", "limit", "fhs", "recommender",
               "walkforward", "metrics", "costs", "limits", "disclaimer"]


def method(hash_="#/method", settle=300, viewport=1280, **kw):
    app = FrontApp(hash=hash_, viewport=viewport, **kw)
    app.settle(settle, step=min(100, settle))
    return app


class Base(unittest.TestCase):
    def has(self, text, needle):
        self.assertTrue(needle in text, f"missing {needle!r}")


def math_html(app, src):
    """HTML of the nodes mathNodes(src) builds (attributes and classes included)."""
    js = ("(function () { var d = document.createElement('div'); TQ.views.method._internals.mathNodes(" + json.dumps(src) + ").forEach(function (n) {"
          "d.appendChild(typeof n === 'string' ? document.createTextNode(n) : n); }); return JSON.parse(window.__serialize(d)).html; })()")
    html = app.eval(js)
    assert html.startswith("<div>") and html.endswith("</div>")
    return html[len("<div>"):-len("</div>")]


def strip_tags(html):
    return re.sub(r"<[^>]+>", "", html)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class MathRenderer(Base):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js", "charts.js", "core.js", "view_method.js"], fixtures=[])

    def test_letters_are_italic_digits_and_operators_are_not(self):
        html = math_html(self.app, "x = 2y + 3")
        self.assertIn("<i>x</i>", html)
        self.assertIn("<i>y</i>", html)
        self.assertNotIn("<i>2</i>", html)
        self.assertEqual(strip_tags(html), "x = 2y + 3")

    def test_scripts(self):
        self.assertEqual(math_html(self.app, "x_t"), "<i>x</i><sub><i>t</i></sub>")
        self.assertEqual(math_html(self.app, "x^2"), "<i>x</i><sup>2</sup>")
        self.assertEqual(math_html(self.app, "x_{t-1}"), "<i>x</i><sub><i>t</i>-1</sub>")
        # both scripts -> stacked
        html = math_html(self.app, "σ_t^2")
        self.assertIn('<span class="ss"><span class="sp">2</span><span class="sb"><i>t</i></span></span>', html)
        html = math_html(self.app, "σ^2_t")
        self.assertIn('<span class="ss"><span class="sp">2</span><span class="sb"><i>t</i></span></span>', html)
        # nested scripts and braces as grouping only
        self.assertEqual(strip_tags(math_html(self.app, "e^{-x^{2}}")), "e-x2")
        self.assertEqual(strip_tags(math_html(self.app, "{ab}")), "ab")

    def test_fractions_roots_accents_text(self):
        html = math_html(self.app, r"\frac{a}{b+1}")
        self.assertEqual(html, '<span class="frac"><span class="fn"><i>a</i></span><span class="fd"><i>b</i>+1</span></span>')
        self.assertEqual(strip_tags(math_html(self.app, r"\tfrac xn")), "xn")
        self.assertIn('<span class="sqrt">√<span class="sq-in"><i>v</i></span></span>', math_html(self.app, r"\sqrt{v}"))
        self.assertEqual(strip_tags(math_html(self.app, r"\hat\mu")), "μ̂")
        self.assertIn('<span class="wide-hat"><i>S</i><i>R</i></span>', math_html(self.app, r"\widehat{SR}"))
        self.assertEqual(strip_tags(math_html(self.app, r"\bar z")), "z̄")
        self.assertIn('<span class="tx">（標準化）</span>', math_html(self.app, r"\text{（標準化）}"))
        self.assertIn('<span class="tx">a_b {c}</span>', math_html(self.app, r"\text{a_b {c}}"))           # \text is verbatim, braces balanced
        self.assertIn('<b class="mb">1</b>', math_html(self.app, r"\mathbf{1}[x]"))

    def test_symbols_functions_greek_and_spaces(self):
        self.assertEqual(strip_tags(math_html(self.app, r"a \le b \ge c \ne d \approx e \cdot f \times g \to h")), "a ≤ b ≥ c ≠ d ≈ e · f × g → h")
        self.assertEqual(strip_tags(math_html(self.app, r"\ln x + \exp y + \max z + \Pr w")), "ln x + exp y + max z + Pr w")
        self.assertEqual(strip_tags(math_html(self.app, r"\alpha\beta\Phi\Lambda_0")), "αβΦΛ0")
        self.assertIn("<i>α</i>", math_html(self.app, r"\alpha"))                  # lower-case Greek is italic like Latin letters
        self.assertIn('<span class="sym">Φ</span>', math_html(self.app, r"\Phi"))
        self.assertEqual(math_html(self.app, r"a\,b"), "<i>a</i> <i>b</i>")
        self.assertEqual(math_html(self.app, r"a\quad b"), "<i>a</i>\u2003 <i>b</i>")          # like TeX source, the typed space stays
        self.assertEqual(math_html(self.app, r"a\ b"), "<i>a</i> <i>b</i>")
        self.assertEqual(strip_tags(math_html(self.app, r"\{x\}\%\_")), "{x}%_")
        self.assertIn('<span class="bigop">∑</span>', math_html(self.app, r"\sum_{k=1}^{h}"))
        self.assertEqual(strip_tags(math_html(self.app, r"\operatorname{expit}(a)")), "expit(a)")
        self.assertIn('<span class="dl">(</span>', math_html(self.app, r"\left(a\right)"))
        self.assertEqual(math_html(self.app, r"\big|a"), "|<i>a</i>")

    def test_unknown_commands_stay_visible_so_authoring_errors_are_caught(self):
        self.assertIn('<span class="bad">\\nonsense</span>', math_html(self.app, r"a \nonsense b"))

    def test_degenerate_input_never_throws(self):
        for src in ("", "{", "}", "_", "^", "x_", "x^", r"\frac", r"\frac{a}", r"\sqrt", r"\text", r"\text{abc", "\\", "{{{{", "a_{", "}}}}", r"\hat", r"\left", "💥"):
            self.app.eval("TQ.views.method._internals.mathNodes(" + json.dumps(src) + "); 0")
        self.assertEqual(self.app.eval("TQ.views.method._internals.mathNodes(null).length"), 0)
        self.assertEqual(self.app.eval("TQ.views.method._internals.mathNodes(undefined).length"), 0)

    def test_linear_reading_for_assistive_technology(self):
        lin = lambda s: self.app.eval("TQ.views.method._internals.linearMath(" + json.dumps(s) + ")")
        self.assertEqual(lin(r"\frac{a}{b+1}"), "(a)/(b+1)")
        self.assertEqual(lin(r"r_t = \ln\frac{P_t}{P_{t-1}}"), "r_t = ln(P_t)/(P_(t-1))")
        self.assertEqual(lin(r"\sigma_t^2 \le \alpha"), "σ_t^2 ≤ α")
        self.assertEqual(lin(r"\text{預測}_i = f\left(\text{資料}\right)"), "預測_i = f(資料)")
        self.assertEqual(lin(r"\mathrm{VaR}_{95}"), "VaR_(95)")
        self.assertEqual(lin(r"a\,b\;c\ d"), "a b c d")
        self.assertEqual(lin(r"\sqrt{v_k(h)}"), "√(v_k(h))")
        self.assertEqual(lin(r"\sum_{k=1}^{h} x_k"), "∑_(k=1)^h x_k")
        self.assertEqual(lin(r"\frac{\frac{a}{b}}{c}"), "((a)/(b))/(c)")
        self.assertEqual(lin(None), "")

    def test_inline_markup(self):
        def ins(s):
            return self.app.eval("(function () { var d = document.createElement('p'); TQ.views.method._internals.inline(" + json.dumps(s) + ").forEach(function (n) { d.appendChild(typeof n === 'string' ? document.createTextNode(n) : n); }); return JSON.parse(window.__serialize(d)).html; })()")
        self.assertEqual(ins("a **b** c"), "<p>a <strong>b</strong> c</p>")
        self.assertEqual(ins("**bold $n/h$ math**"), '<p><strong>bold <span class="m"><i>n</i>/<i>h</i></span> math</strong></p>')      # math inside bold
        self.assertEqual(ins("x $y_1$ z"), '<p>x <span class="m"><i>y</i><sub>1</sub></span> z</p>')
        self.assertEqual(ins("NT$5 and 10"), "<p>NT$5 and 10</p>")                     # a lone dollar sign is just text
        self.assertEqual(ins(""), "<p></p>")
        self.assertEqual(ins("**bold** and $a$ and **more**"), '<p><strong>bold</strong> and <span class="m"><i>a</i></span> and <strong>more</strong></p>')


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class PageContent(Base):
    @classmethod
    def setUpClass(cls):
        cls.app = method()
        cls.text = cls.app.text("#view")

    def test_clean_single_h1_and_title(self):
        assert_clean(self, self.app)
        self.assertEqual(audit(self.app)["h1"], 1)
        self.assertEqual(self.app.eval("document.title"), "方法說明 · 台股量化實驗室")
        self.assertEqual(self.app.count(".view-method"), 1)
        self.assertEqual(self.app.attr('#nav a[data-route="method"]', "aria-current"), "page")
        self.assertEqual(self.app.json("window.__fetchLog"), [], "a static page must not call the API")

    def test_no_authoring_leftovers_in_the_rendered_text(self):
        t = self.text
        for bad in ("\\", "$", "**", "\t", "\n", "\u0000"):
            self.assertNotIn(bad, t, repr(bad))
        self.assertEqual(self.app.count(".bad"), 0, "an equation uses a command the renderer does not know")

    def test_no_floating_point_artefacts_in_the_text(self):
        # 0.55 * 100 is 55.00000000000001 in IEEE arithmetic: every number on the page must come out of a formatter
        self.assertEqual(re.findall(r"\d\.\d{6,}", self.text), [])
        self.has(self.app.text("#m-recommender"), "上漲機率 ≥ 55%、扣成本後預期報酬 > 0")
        self.has(self.app.text("#m-recommender"), "上漲機率 ≤ 45% 且")
        self.has(self.app.text("#m-recommender"), "單檔上限 15%")
        self.has(self.app.text("#m-garch"), "偏度只採用模型隱含值的 60%")

    def test_all_sections_exist_in_order_with_stable_ids(self):
        ids = self.app.json("Array.prototype.map.call(document.querySelectorAll('.m-sec'), function (s) { return s.id; })")
        self.assertEqual(ids, ["m-" + i for i in SECTION_IDS])
        titles = self.app.texts("h2.card-title")
        self.assertEqual(len(titles), 18)
        self.assertTrue(titles[0].startswith("1．先讀這個"))
        self.assertTrue(titles[-1].startswith("18．免責聲明"))
        self.assertEqual(self.app.attr("#m-garch", "tabindex"), "-1")                 # focusable programmatically (after a TOC jump)

    def test_every_topic_of_the_brief_is_explained(self):
        t = self.text
        topics = ["還原權息", "累積對數報酬", "GJR-GARCH", "Student-t", "槓桿效應", "隱馬可夫", "濾波", "平滑", "動態線性模型", "Normal–Inverse-Gamma", "折現因子",
                  "動態模型平均", "遺忘因子", "線上貝氏", "漲跌停", "±10%", "濾波歷史模擬", "VaR", "CVaR", "成本調整", "James–Stein", "贏家詛咒", "分數化凱利",
                  "前推式", "截斷不變性", "指標詞彙表", "方向準確率", "二項檢定", "Brier", "ECE", "資訊係數", "Campbell–Thompson", "Diebold–Mariano", "Kupiec", "PIT",
                  "PSR", "DSR", "Stationary bootstrap", "台灣交易成本", "限制與誠實說明", "效率市場", "結構性改變", "存活者偏差", "過度配適", "多重檢定", "除息",
                  "模型風險", "非投資建議", "更", "隨機漫步"]
        for k in topics:
            if k != "更":
                self.has(t, k)

    def test_the_honesty_message_is_up_front(self):
        intro = self.app.text("#m-intro")
        for needle in ("非常接近不可預測", "樣本外", "沒有統計上顯著的預測優勢", "不是投資建議", "沒有偷看未來"):
            self.has(intro, needle)
        self.assertEqual(self.app.count("#m-intro .m-note-warn"), 1)

    def test_limits_section_lists_the_required_caveats(self):
        t = self.app.text("#m-limits")
        for needle in ("效率市場", "市場狀態會改變", "存活者偏差", "過度配適與多重檢定", "除息", "交易成本", "模型風險", "機率不是保證", "資料品質"):
            self.has(t, needle)
        self.assertGreaterEqual(self.app.count("#m-limits .m-olist li"), 9)

    def test_walk_forward_section_explains_why_look_ahead_is_impossible(self):
        t = self.app.text("#m-walkforward")
        for needle in ("擴張視窗", "已到期", "濾波機率", "截斷", "完全相同", "日期錯位", "有效樣本數", "Newey–West", "2021-08"):
            self.has(t, needle)
        self.assertEqual(self.app.count("#m-walkforward .m-timeline .tl-seg"), 6)

    def test_pipeline_flow_has_eight_steps(self):
        self.assertEqual(self.app.count(".m-flow-step"), 8)
        self.assertEqual(self.app.texts(".m-flow-title"), ["資料", "特徵", "四個模型", "動態模型平均", "機率校準", "漲跌停截斷", "輸出", "檢驗與推薦"])

    def test_feature_table_lists_all_fourteen_causal_features(self):
        rows = self.app.texts("#m-dlm table tbody tr")
        self.assertEqual(len(rows), 14)
        keys = self.app.json("TQ.views.method._internals.FEATURES.map(function (f) { return f[0]; })")
        for k in keys:
            self.assertTrue(any(k in r for r in rows), k)
        for needle in ("12-1 個月動能", "1 日反轉", "RSI（14）", "成交量異常", "大盤波動狀態", "特異動能（60 日）"):
            self.has(self.app.text("#m-dlm"), needle)

    def test_cost_table_numbers(self):
        from backend import config
        rows = self.app.json("Array.prototype.map.call(document.querySelectorAll('#m-costs table tbody tr'), function (r) { return r.textContent; })")
        self.assertEqual(len(rows), 5)
        t = self.app.text("#m-costs")
        for needle in ("0.1425%", "0.3%", "0.1%", "滑價", "每邊 5 bp", "下一個開盤價成交", "沒有計入的成本", "漲停鎖死", "市場衝擊"):
            self.has(t, needle)
        for disc, row in zip((0.28, 0.5, 0.6, 0.85, 1.0), rows):
            stock = config.roundtrip_cost("stock", disc, config.SLIPPAGE_BPS)
            etf = config.roundtrip_cost("etf", disc, config.SLIPPAGE_BPS)
            self.assertIn(f"{stock * 100:.3f}%", row, (disc, row))
            self.assertIn(f"{etf * 100:.3f}%", row, (disc, row))
        self.assertIn("0.571%", rows[2])                         # the default discount (6 折) row: stock round trip
        self.assertIn("0.371%", rows[2])
        self.assertEqual(self.app.count("#m-costs tr.row-total"), 1)       # the default discount row is highlighted

    def test_metric_glossary_has_formulas(self):
        self.assertEqual(self.app.count("#m-metrics .m-dl-item"), 13)
        self.assertGreaterEqual(self.app.count("#m-metrics .eq"), 9)
        t = self.app.text("#m-metrics")
        for needle in ("Wilson", "Bin", "BSS", "ECE", "Spearman", "OOS", "Newey–West", "Kupiec", "χ", "Kolmogorov–Smirnov", "pinball", "Bailey", "多重檢定"):
            self.has(t, needle)

    def test_equations_are_accessible_math(self):
        n = self.app.count(".eq")
        self.assertGreaterEqual(n, 35)
        bad = self.app.json("Array.prototype.filter.call(document.querySelectorAll('.eq'), function (e) { return e.getAttribute('role') !== 'math' || !e.getAttribute('aria-label') || /[\\\\$]/.test(e.getAttribute('aria-label')); }).length")
        self.assertEqual(bad, 0)
        self.assertEqual(self.app.count(".eq-in[aria-hidden=true]"), n)          # the visual rendering is hidden from AT: the aria-label is the reading
        # every equation has a plain-language gloss unless it is one of several lines of the same block
        self.assertGreaterEqual(self.app.count(".eq-note"), 25)
        self.has(self.app.attr(".eq", "aria-label"), "adjC")

    def test_numbers_quoted_in_the_text(self):
        t = self.text
        for needle in ("2,500", "1,000", "0.998", "0.99", "400", "0.25", "300", "0.995", "5,000", "250", "15%", "55%", "45%", "0.5%"):
            self.has(t, needle)

    def test_dark_theme_does_not_change_the_text(self):
        a = method(dark=True)
        self.assertEqual(a.text("#view"), self.text)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class TableOfContents(Base):
    def test_toc_links(self):
        app = method()
        self.assertEqual(app.count(".m-toc-link"), 18)
        links = app.json("Array.prototype.map.call(document.querySelectorAll('.m-toc-link'), function (a) { return [a.getAttribute('href'), a.getAttribute('data-section'), a.textContent]; })")
        self.assertEqual([l[0] for l in links], ["#/method/" + i for i in SECTION_IDS])
        self.assertEqual([l[1] for l in links], SECTION_IDS)
        self.assertTrue(links[0][2].startswith("1．先讀這個"))
        self.assertEqual(app.attr(".m-toc nav", "aria-label"), "本頁目錄")
        self.assertEqual(app.count(".m-toc-link[href^=#/method/]"), 18)
        self.assertEqual(app.count("a[href^=#m-]"), 0, "plain #anchors would break the hash router")
        self.assertEqual(app.count("a[href=#]"), 0)

    def test_click_scrolls_updates_the_url_and_does_not_remount_or_fire_hashchange(self):
        app = method()
        app.eval("window.__changes = 0; window.addEventListener('hashchange', function () { window.__changes++; }); window.__host = document.querySelector('.view-host'); window.__scrollLog = []; 0")
        r = app.json("(function () { var ev = new Event('click', { bubbles: true, cancelable: true }); document.querySelector('.m-toc-link[data-section=\"garch\"]').dispatchEvent(ev); return ev.defaultPrevented; })()")
        self.assertEqual(r, True, "the default navigation must be prevented")
        app.settle(300)
        self.assertEqual(app.json("window.__scrollLog.map(function (s) { return s.id; })"), ["m-garch"])
        self.assertEqual(app.eval("location.hash"), "#/method/garch")
        self.assertEqual(app.eval("window.__changes"), 0)
        self.assertEqual(app.eval("document.querySelector('.view-host') === window.__host"), True, "the page must not be re-mounted")
        self.assertEqual(app.eval("document.activeElement.id"), "m-garch")                  # keyboard / screen-reader users land in the section
        self.assertEqual(app.json("TQ.router.current().hash"), "#/method/garch")
        self.assertEqual(app.errors(), [])
        # jump again elsewhere, then use the nav link back to the top
        app.click(".m-toc-link[data-section=costs]")
        self.assertEqual(app.eval("location.hash"), "#/method/costs")
        app.goto("#/method", 200)
        self.assertEqual(app.eval("document.querySelector('.view-host') === window.__host"), False, "the nav link starts a fresh page")

    def test_smooth_scrolling_unless_reduced_motion(self):
        app = method()
        app.click(".m-toc-link[data-section=hmm]")
        self.assertEqual(app.json("window.__scrollLog[0].opts"), {"block": "start", "behavior": "smooth"})
        app = FrontApp(hash="#/method")
        app.eval("window.matchMedia = function (q) { return { matches: /reduced-motion/.test(q), addEventListener: function () {}, addListener: function () {} }; };")
        app.goto("#/picks", 100)
        app.goto("#/method", 200)
        app.click(".m-toc-link[data-section=hmm]")
        self.assertEqual(app.json("window.__scrollLog[0].opts"), {"block": "start", "behavior": "auto"})

    def test_back_to_top_links(self):
        app = method()
        self.assertEqual(app.count(".m-top-link"), 18)
        app.click(".m-top-link", 5)
        self.assertEqual(app.json("window.__scrollLog.pop().cls.indexOf('m-toc') >= 0"), True)
        self.assertEqual(app.eval("location.hash"), "#/method")
        self.assertEqual(app.errors(), [])

    def test_deep_link_scrolls_to_the_section_after_the_router_finished(self):
        app = method("#/method/walkforward", settle=40)
        self.assertEqual(app.json("window.__scrollLog || []"), [], "deferred: the router's own scroll-to-top must not win")
        app.settle(200)
        self.assertEqual(app.json("window.__scrollLog.map(function (s) { return s.id; })"), ["m-walkforward"])
        self.assertEqual(app.eval("location.hash"), "#/method/walkforward")
        self.assertEqual(app.json("window.__scrollLog[0].opts.behavior"), "auto")           # an arrival is not animated

    def test_unknown_section_is_ignored(self):
        app = method("#/method/does-not-exist", settle=500)
        self.assertEqual(app.json("window.__scrollLog || []"), [])
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.count(".m-sec"), 18)

    def test_leaving_before_the_deferred_scroll_cancels_it(self):
        app = method("#/method/hmm", settle=10)
        app.goto("#/picks", 5)
        app.settle(500)
        self.assertEqual(app.json("window.__scrollLog || []"), [])

    def test_toc_is_open_on_desktop_and_collapsed_on_phones(self):
        self.assertEqual(method(viewport=1280).json("document.querySelector('.m-toc').open"), True)
        self.assertEqual(method(viewport=390).json("document.querySelector('.m-toc').open"), False)
        self.assertEqual(method(viewport=390).count(".m-toc-link"), 18)                  # still in the DOM / reachable by opening the summary

    def test_summary_names_the_toc(self):
        self.assertEqual(method().text(".m-toc-sum"), "本頁目錄（18 節）")


if __name__ == "__main__":
    unittest.main()
