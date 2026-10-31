"""Router, header search box (combobox), theme toggle and boot sequence (index.html + app.js) in the fake browser."""
from __future__ import annotations

import json
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, default_rules

STUB_VIEWS = """
window.__log = [];
['stock', 'picks', 'backtest', 'method'].forEach(function (name) {
  TQ.views[name] = {
    title: function (p) { return 'T-' + name + (p && p.code ? '-' + p.code : ''); },
    mount: function (host, params, ctx) {
      var rec = { name: name, params: params, ctx: ctx, destroyed: 0, aborted: false };
      window.__log.push(rec);
      ctx.signal.addEventListener('abort', function () { rec.aborted = true; });
      host.appendChild(TQ.h('p', { class: 'stub' }, 'view ' + name + ' ' + (params.code || '')));
      return { destroy: function () { rec.destroyed++; window.__order.push('destroy-' + name); } };
    }
  };
});
window.__order = [];
"""

SCRIPTS = ["format.js", "charts.js", "core.js", "app.js"]


def stub_app(hash="", **kw):
    """Page with the real shell (index.html, core, app.js) but stub views; app.js is loaded last so the router boots with them."""
    app = FrontApp(scripts=["format.js", "charts.js", "core.js"], fixtures=[], hash=hash, **kw)
    app.eval(STUB_VIEWS)
    app.eval((__import__("front_helpers").FRONT / "app.js").read_text(encoding="utf-8"))
    app.settle(100)
    return app


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Parse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js", "charts.js", "core.js"], fixtures=[])

    def parse(self, h):
        return self.app.json(f"TQ.router.parse({json.dumps(h)})")

    def test_known_routes(self):
        self.assertEqual(self.parse("#/stock/2330"), {"name": "stock", "params": {"code": "2330"}, "canonical": "#/stock/2330", "redirect": False, "query": ""})
        self.assertEqual(self.parse("#/picks")["name"], "picks")
        self.assertEqual(self.parse("#/backtest/0050")["params"], {"code": "0050"})
        self.assertEqual(self.parse("#/method")["canonical"], "#/method")
        self.assertEqual(self.parse("#/method/garch")["params"], {"section": "garch"})
        self.assertEqual(self.parse("#/stock/00878?x=1")["canonical"], "#/stock/00878")
        self.assertEqual(self.parse("#/stock/00878?x=1")["query"], "x=1")      # the query is kept (raw) next to the canonical path

    def test_normalisation_and_defaults(self):
        r = self.parse("#/stock/00400a")
        self.assertEqual((r["params"]["code"], r["redirect"]), ("00400A", True))
        for h in ("", "#", "#/", "#/nonsense", "#/stock", "#/stock/", "#/stock/!!bad", "#/stock/%E0%A4%A", "#/stock/" + "9" * 40):
            r = self.parse(h)
            self.assertEqual((r["name"], r["params"], r["redirect"]), ("stock", {"code": "2330"}, True), h)
        r = self.parse("#/backtest")
        self.assertEqual((r["name"], r["params"]["code"], r["redirect"]), ("backtest", "2330", True))
        self.assertEqual(self.parse("#/stock/%5Etwii")["params"]["code"], "^TWII")

    def test_last_viewed_stock_is_the_default(self):
        app = FrontApp(scripts=["format.js", "charts.js", "core.js"], fixtures=[], storage={"tq.lastStock": '"0050"'})
        self.assertEqual(app.json("TQ.router.parse('')")["params"]["code"], "0050")
        app = FrontApp(scripts=["format.js", "charts.js", "core.js"], fixtures=[], storage={"tq.lastStock": '"<script>"'})
        self.assertEqual(app.json("TQ.router.parse('')")["params"]["code"], "2330")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Routing(unittest.TestCase):
    def test_boot_redirects_to_default_and_mounts_view(self):
        app = stub_app()
        self.assertEqual(app.eval("window.location.hash"), "#/stock/2330")
        self.assertEqual(app.text("#view"), "view stock 2330")
        self.assertEqual(app.attr(".view-host", "data-view"), "stock")
        self.assertEqual(app.eval("document.title"), "T-stock-2330 · 台股量化實驗室")
        self.assertIn("T-stock-2330", app.text("#status"))
        self.assertEqual(app.errors(), [])

    def test_nav_links_active_state_and_last_stock(self):
        app = stub_app("#/stock/0050")
        self.assertEqual(app.attr('#nav a[data-route="stock"]', "aria-current"), "page")
        self.assertIsNone(app.attr('#nav a[data-route="picks"]', "aria-current"))
        app.eval("TQ.router.setLastStock('6488')")
        self.assertEqual(app.attr('#nav a[data-route="stock"]', "href"), "#/stock/6488")
        self.assertEqual(app.attr('#nav a[data-route="backtest"]', "href"), "#/backtest/6488")
        self.assertEqual(app.eval("JSON.parse(window.__ls['tq.lastStock'])"), "6488")
        app.goto("#/picks", 100)
        self.assertEqual(app.attr('#nav a[data-route="picks"]', "aria-current"), "page")
        self.assertIsNone(app.attr('#nav a[data-route="stock"]', "aria-current"))
        self.assertEqual(app.eval("document.title"), "T-picks · 台股量化實驗室")

    def test_route_change_aborts_destroys_and_replaces(self):
        app = stub_app("#/stock/2330")
        app.goto("#/picks", 100)
        log = app.json("window.__log.map(function (r) { return { name: r.name, destroyed: r.destroyed, aborted: r.aborted }; })")
        self.assertEqual(log, [{"name": "stock", "destroyed": 1, "aborted": True}, {"name": "picks", "destroyed": 0, "aborted": False}])
        self.assertEqual(app.text("#view"), "view picks ")
        self.assertEqual(app.count(".view-host"), 1)
        app.goto("#/backtest/0050", 100)
        app.goto("#/method", 100)
        self.assertEqual(app.json("window.__order"), ["destroy-stock", "destroy-picks", "destroy-backtest"])
        self.assertEqual(app.errors(), [])

    def test_same_hash_does_not_remount_and_unknown_hash_redirects(self):
        app = stub_app("#/picks")
        app.goto("#/picks", 100)
        self.assertEqual(app.eval("window.__log.length"), 1)
        app.goto("#/does-not-exist", 100)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/2330")
        self.assertEqual(app.eval("window.__log.length"), 2)

    def test_view_that_throws_shows_error_box_and_router_keeps_working(self):
        app = stub_app("#/picks")
        app.eval("TQ.router.register('method', { title: function () { return 'M'; }, mount: function () { throw new Error('boom'); } }); 0")
        app.goto("#/method", 100)
        self.assertIn("畫面產生失敗", app.text("#view"))
        self.assertTrue(app.errors())
        app.goto("#/stock/2330", 100)
        self.assertEqual(app.text("#view"), "view stock 2330")

    def test_ctx_nav_set_status_and_title(self):
        app = stub_app("#/picks")
        app.eval("var c = window.__log[0].ctx; c.setStatus('進度 50%'); c.setTitle('自訂'); c.nav('#/method'); 0")
        app.settle(100)
        self.assertEqual(app.text("#status"), "已切換至「T-method」")      # nav() switched the view; its announcement replaced the old one
        app.eval("window.__log[0].ctx.setTitle('晚到的標題'); 0")          # a view that was torn down must not change the title any more
        self.assertEqual(app.eval("document.title"), "T-method · 台股量化實驗室")

    def test_focus_moves_to_main_on_navigation_but_not_on_first_load(self):
        app = stub_app("#/picks")
        self.assertNotEqual(app.eval("document.activeElement && document.activeElement.id"), "main")
        app.goto("#/method", 100)
        self.assertEqual(app.eval("document.activeElement.id"), "main")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ThemeToggleAndBoot(unittest.TestCase):
    def test_toggle_cycles_persists_and_labels(self):
        app = stub_app()
        self.assertEqual(app.text("#theme-label"), "自動")
        self.assertIn("目前：自動", app.attr("#theme-toggle", "aria-label"))
        app.click("#theme-toggle")
        self.assertEqual((app.text("#theme-label"), app.eval("document.documentElement.getAttribute('data-theme')")), ("淺色", "light"))
        app.click("#theme-toggle")
        self.assertEqual((app.text("#theme-label"), app.eval("document.documentElement.getAttribute('data-theme')")), ("深色", "dark"))
        self.assertEqual(app.eval("window.__ls['tq.theme']"), '"dark"')
        self.assertEqual(app.count("#theme-icon svg"), 1)
        self.assertEqual(app.text("#status"), "佈景主題：深色")
        app.click("#theme-toggle")
        self.assertIsNone(app.eval("document.documentElement.getAttribute('data-theme')"))

    def test_remembered_theme_applied_on_boot(self):
        app = stub_app(dark=True)
        self.assertEqual(app.eval("document.documentElement.getAttribute('data-theme')"), "dark")
        self.assertEqual(app.text("#theme-label"), "深色")

    def test_slash_focuses_search_but_not_while_typing(self):
        app = stub_app()
        app.eval("document.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true, cancelable: true, target: document.body })); 0")
        self.assertEqual(app.eval("document.activeElement === document.querySelector('#search-host input')"), True)
        app.eval("""(function () { var i = document.querySelector('#search-host input'); i.value = 'a';
                  var ev = new KeyboardEvent('keydown', { key: '/', bubbles: true, cancelable: true }); i.dispatchEvent(ev); window.__prevented = ev.defaultPrevented; })()""")
        self.assertEqual(app.eval("window.__prevented"), False)

    def test_unhandled_abort_rejections_are_swallowed(self):
        app = stub_app()
        app.eval("""(function () { var ev = new Event('unhandledrejection', { cancelable: true }); ev.reason = { name: 'AbortError' }; window.dispatchEvent(ev); window.__p = ev.defaultPrevented; })()""")
        self.assertTrue(app.eval("window.__p"))
        self.assertEqual(app.errors(), [])


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class SearchBox(unittest.TestCase):
    def app(self, extra_rules=(), **kw):
        a = FrontApp(scripts=["format.js", "charts.js", "core.js"], real=True, hash="", **kw)
        a.set_rules(list(extra_rules) + default_rules(True))
        a.eval(STUB_VIEWS)
        a.eval((__import__("front_helpers").FRONT / "app.js").read_text(encoding="utf-8"))
        a.settle(100)
        return a

    INPUT = "#search-host input"

    def options(self, app):
        return app.texts("#search-host .search-opt")

    def test_aria_combobox_pattern(self):
        app = self.app()
        self.assertEqual(app.attr(self.INPUT, "role"), "combobox")
        self.assertEqual(app.attr(self.INPUT, "aria-expanded"), "false")
        self.assertEqual(app.attr(self.INPUT, "aria-autocomplete"), "list")
        listbox_id = app.attr("#search-host ul", "id")
        self.assertEqual(app.attr(self.INPUT, "aria-controls"), listbox_id)
        self.assertEqual(app.attr("#search-host ul", "role"), "listbox")
        label_for = app.attr("#search-host label", "for")
        self.assertEqual(label_for, app.attr(self.INPUT, "id"))
        self.assertEqual(app.attr("#search-host .search", "role"), "search")

    def test_focus_shows_popular_list_then_typing_debounces_and_searches(self):
        app = self.app()
        app.eval("document.querySelector('#search-host input').focus()")
        app.settle(100)
        self.assertEqual(app.attr(self.INPUT, "aria-expanded"), "true")
        self.assertIn("/api/search?limit=20", app.fetched())          # popular list: not cut at 12 (12 universe names + 3 ETFs)
        opts = self.options(app)
        self.assertEqual(len(opts), 15)
        self.assertTrue(opts[0].startswith("2330台積電"))
        self.assertIn("ETF", opts[-1])
        n = len(app.fetched())
        app.type(self.INPUT, "台")
        app.settle(100)                         # still inside the 180 ms debounce
        app.type(self.INPUT, "台積")
        app.settle(100)
        self.assertEqual(len(app.fetched()), n)
        app.settle(200)
        self.assertEqual(app.fetched()[-1], "/api/search?q=%E5%8F%B0%E7%A9%8D&limit=12")
        self.assertEqual(len(app.fetched()), n + 1)
        opts = self.options(app)
        self.assertEqual(len(opts), 1)
        self.assertIn("2330", opts[0])
        self.assertIn("台積電", opts[0])
        self.assertIn("上市", opts[0])
        self.assertEqual(app.attr("#search-host .search-opt", "aria-selected"), "true")        # first hit is pre-selected for Enter
        self.assertEqual(app.attr(self.INPUT, "aria-activedescendant"), app.attr("#search-host .search-opt", "id"))

    def test_arrow_keys_wrap_and_enter_navigates(self):
        app = self.app([{"re": r"^/api/search\?q=%E5%8F%B0", "resp": {"status": 200, "body": {"status": "ready", "results": [
            {"code": "2330", "name": "台積電", "market": "TWSE", "kind": "stock"}, {"code": "6488", "name": "環球晶", "market": "TPEx", "kind": "stock"},
            {"code": "0050", "name": "元大台灣50", "market": "TWSE", "kind": "etf"}]}}}])
        app.type(self.INPUT, "台")
        app.settle(400)
        act = lambda: app.eval("document.querySelector('#search-host input').getAttribute('aria-activedescendant')")
        ids = app.json("Array.prototype.map.call(document.querySelectorAll('#search-host .search-opt'), function (o) { return o.id; })")
        self.assertEqual(len(ids), 3)
        self.assertEqual(act(), ids[0])
        app.press(self.INPUT, "ArrowDown")
        self.assertEqual(act(), ids[1])
        app.press(self.INPUT, "ArrowDown")
        app.press(self.INPUT, "ArrowDown")                     # wraps around to the first
        self.assertEqual(act(), ids[0])
        app.press(self.INPUT, "ArrowUp")                       # and back to the last
        self.assertEqual(act(), ids[2])
        app.press(self.INPUT, "ArrowUp")
        app.press(self.INPUT, "Enter")
        app.settle(100)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/6488")
        self.assertEqual(app.attr(self.INPUT, "aria-expanded"), "false")
        self.assertEqual(app.eval("document.querySelector('#search-host input').value"), "")

    def test_click_on_option_navigates(self):
        app = self.app()
        app.eval("document.querySelector('#search-host input').focus()")
        app.settle(100)
        app.click("#search-host .search-opt", 2)
        app.settle(100)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/2454")

    def test_escape_closes_then_clears(self):
        app = self.app()
        app.type(self.INPUT, "台積")
        app.settle(400)
        app.press(self.INPUT, "Escape")
        self.assertEqual(app.attr(self.INPUT, "aria-expanded"), "false")
        self.assertEqual(app.eval("document.querySelector('#search-host input').value"), "台積")
        app.press(self.INPUT, "Escape")
        self.assertEqual(app.eval("document.querySelector('#search-host input').value"), "")

    def test_enter_before_results_arrive_and_code_fallback(self):
        app = self.app([{"re": r"^/api/search\?q=6488", "resp": {"status": 200, "body": {"status": "ready", "results": []}}}])
        app.type(self.INPUT, "6488")
        app.press(self.INPUT, "Enter")                         # no debounce wait: searches now, finds nothing, navigates to the typed code anyway
        app.settle(400)
        self.assertEqual(app.eval("window.location.hash"), "#/stock/6488")

    def test_no_results_message_error_message_and_stale_responses(self):
        app = self.app([{"re": r"^/api/search\?q=zzz", "resp": {"status": 200, "body": {"status": "ready", "results": []}}},
                        {"re": r"^/api/search\?q=boom", "resp": {"status": 500, "body": {"status": "error", "error": {"code": "internal", "message": "搜尋壞了"}}}}])
        app.type(self.INPUT, "zzz")
        app.settle(400)
        self.assertEqual(self.options(app), [])
        self.assertIn("找不到符合「zzz」", app.text("#search-host .search-msg"))
        app.type(self.INPUT, "boom")
        app.settle(400)
        self.assertIn("搜尋壞了", app.text("#search-host .search-msg"))
        # keyboard Enter on an unknown, non-code query does nothing
        h0 = app.eval("window.location.hash")
        app.press(self.INPUT, "Enter")
        app.settle(300)
        self.assertEqual(app.eval("window.location.hash"), h0)

    def test_older_response_never_overwrites_newer(self):
        slow = {"status": 200, "body": {"status": "ready", "results": [{"code": "1111", "name": "舊結果", "market": "TWSE", "kind": "stock"}]}}
        fast = {"status": 200, "body": {"status": "ready", "results": [{"code": "2222", "name": "新結果", "market": "TWSE", "kind": "stock"}]}}
        app = self.app([{"re": r"^/api/search\?q=a&", "resp": slow}, {"re": r"^/api/search\?q=ab&", "resp": fast}], delay=300)
        app.type(self.INPUT, "a")
        app.settle(200)                                        # request for "a" is in flight (300 ms server delay)
        app.type(self.INPUT, "ab")
        app.settle(1000)
        texts = self.options(app)
        self.assertEqual(len(texts), 1)
        self.assertIn("新結果", texts[0])

    def test_clear_button_and_destroy(self):
        app = self.app()
        app.type(self.INPUT, "台")
        self.assertFalse(app.eval("document.querySelector('#search-host .search-clear').hidden"))
        app.click("#search-host .search-clear")
        app.settle(100)
        self.assertEqual(app.eval("document.querySelector('#search-host input').value"), "")
        self.assertTrue(app.eval("document.querySelector('#search-host .search-clear').hidden"))
        app.eval("window.__sb = TQ.searchBox.create({ onSelect: function (it) { window.__picked = it.code; } }); document.body.appendChild(window.__sb.el); window.__sb.input.value = '23'; window.__sb.input.dispatchEvent(new Event('input')); 0")
        app.settle(400)
        app.eval("window.__sb.input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })); 0")
        app.settle(300)
        self.assertEqual(app.eval("window.__picked"), "2330")
        self.assertEqual(app.eval("window.location.hash"), "#/stock/2330")   # custom onSelect: the router is NOT touched
        app.eval("window.__sb.destroy(); 0")
        self.assertEqual(app.count("body > .search"), 0)
        self.assertEqual(app.errors(), [])


class StaticIndexHtml(unittest.TestCase):
    """Pure python (no V8): index.html contract."""

    def test_scripts_stylesheets_and_ids(self):
        from html.parser import HTMLParser
        from front_helpers import FRONT

        html = (FRONT / "index.html").read_text(encoding="utf-8")
        scripts, links, ids, metas = [], [], set(), {}

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                d = dict(attrs)
                if tag == "script" and "src" in d:
                    scripts.append((d["src"], "defer" in d))
                if tag == "link" and d.get("rel") == "stylesheet":
                    links.append(d["href"])
                if "id" in d:
                    ids.add(d["id"])
                if tag == "meta" and "name" in d:
                    metas[d["name"]] = d.get("content")
                if tag == "html":
                    metas["lang"] = d.get("lang")

        P().feed(html)
        self.assertEqual([s for s, _ in scripts], ["format.js", "charts.js", "core.js", "view_stock.js", "view_picks.js", "view_backtest.js", "view_method.js", "app.js"])
        self.assertTrue(all(d for _, d in scripts), "all scripts must be deferred (document order is execution order)")
        self.assertEqual(links, ["style.css", "style_views.css"])
        for f in [s for s, _ in scripts] + links:
            self.assertTrue((FRONT / f).is_file(), f)
        for need in ("view", "nav", "status", "main", "search-host", "theme-toggle", "theme-icon", "theme-label"):
            self.assertIn(need, ids)
        self.assertEqual(metas["lang"], "zh-Hant")
        self.assertIn("width=device-width", metas["viewport"])
        self.assertIn("light dark", metas["color-scheme"])
        self.assertIn('href="#main"', html)               # skip link target exists as id=main
        self.assertIn('aria-live="polite"', html)

    def test_no_external_resources_anywhere(self):
        from front_helpers import FRONT
        import re

        for f in sorted(FRONT.glob("*")):
            if f.suffix not in (".html", ".js", ".css"):
                continue
            text = f.read_text(encoding="utf-8")
            urls = re.findall(r"https?://[^\s\"'<>)]+", text)
            urls = [u for u in urls if u not in ("http://www.w3.org/2000/svg",)]
            self.assertEqual(urls, [], f"{f.name}: external URL(s) found")
            self.assertNotRegex(text, r"@import\s+url\(\s*['\"]?https?:", f.name)

    def test_every_getElementById_in_js_exists_in_index_html(self):
        from front_helpers import FRONT
        import re

        html = (FRONT / "index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r'id="([^"]+)"', html))
        for f in sorted(FRONT.glob("*.js")):
            for m in re.findall(r"getElementById\(\s*'([^']+)'", f.read_text(encoding="utf-8")):
                self.assertIn(m, ids, f"{f.name} asks for #{m}")


if __name__ == "__main__":
    unittest.main()
