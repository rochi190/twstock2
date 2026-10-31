"""Hash router: the optional "?query" part (shareable view settings), ctx.query and ctx.replace().

Added by module D3 for #/backtest/:code?h=..&thr=.. -- the router used to canonicalise the hash and silently drop the query."""
from __future__ import annotations

import unittest

from front_helpers import FRONT, HAVE_RACER, SKIP_MSG, FrontApp
from test_front_shell_router import STUB_VIEWS, stub_app


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class QueryHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js", "charts.js", "core.js"], fixtures=[])

    def pq(self, s):
        return self.app.json(f"TQ.router.parseQuery({s!r})")

    def test_parse_query(self):
        self.assertEqual(self.pq("h=1&thr=0.55"), {"h": "1", "thr": "0.55"})
        self.assertEqual(self.pq("q=%E5%8F%B0%E7%A9%8D&x=a+b"), {"q": "台積", "x": "a b"})
        self.assertEqual(self.pq(""), {})
        self.assertEqual(self.pq("flag&k="), {"flag": "", "k": ""})
        self.assertEqual(self.pq("a=1&a=2"), {"a": "1"})                         # first one wins
        self.assertEqual(self.pq("%E0%A4%A=1&ok=1"), {"ok": "1"})                # malformed escapes are skipped, not thrown
        self.assertEqual(self.pq("__proto__=x&constructor=y&prototype=z&a=1"), {"a": "1"})
        self.assertEqual(self.app.json("Object.prototype.polluted"), None)
        self.assertEqual(self.app.json("TQ.router.parseQuery(null)"), {})

    def test_build_query(self):
        b = lambda js: self.app.json(f"TQ.router.buildQuery({js})")
        self.assertEqual(b("{ h: 1, thr: 0.55, exec: 'next_open' }"), "h=1&thr=0.55&exec=next_open")
        self.assertEqual(b("{ a: null, b: undefined, c: '', d: NaN, e: 0, f: false }"), "e=0&f=false")
        self.assertEqual(b("{ q: '台積 電&=' }"), "q=%E5%8F%B0%E7%A9%8D%20%E9%9B%BB%26%3D")
        self.assertEqual(b("null"), "")
        self.assertEqual(self.app.json("TQ.router.parseQuery(TQ.router.buildQuery({ q: '台積 電&=', n: 5 }))"), {"q": "台積 電&=", "n": "5"})

    def test_parse_keeps_canonical_free_of_the_query(self):
        p = lambda h: self.app.json(f"TQ.router.parse({h!r})")
        r = p("#/backtest/2330?h=5&thr=0.6")
        self.assertEqual((r["name"], r["params"], r["canonical"], r["query"], r["redirect"]), ("backtest", {"code": "2330"}, "#/backtest/2330", "h=5&thr=0.6", False))
        self.assertEqual(p("#/picks?x=1")["query"], "x=1")
        self.assertEqual(p("#/method/garch?x=1")["params"], {"section": "garch"})
        self.assertEqual(p("#/stock/2330")["query"], "")
        self.assertEqual(p("#/stock/00400a?z=1")["redirect"], True)
        self.assertEqual(p("#/nonsense?keep=1")["query"], "keep=1")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class QueryInTheRouter(unittest.TestCase):
    def test_query_survives_canonicalisation_and_reaches_the_view(self):
        app = stub_app("#/backtest/00400a?h=5&thr=0.6")
        self.assertEqual(app.eval("location.hash"), "#/backtest/00400A?h=5&thr=0.6")
        self.assertEqual(app.json("window.__log[0].params"), {"code": "00400A"})
        self.assertEqual(app.json("window.__log[0].ctx.query"), {"h": "5", "thr": "0.6"})
        self.assertEqual(app.json("TQ.router.current()"), {"name": "backtest", "params": {"code": "00400A"}, "hash": "#/backtest/00400A", "query": "h=5&thr=0.6"})

    def test_redirects_keep_the_query_too(self):
        app = stub_app("#/nonsense?keep=1")
        self.assertEqual(app.eval("location.hash"), "#/stock/2330?keep=1")
        self.assertEqual(app.json("window.__log[0].ctx.query"), {"keep": "1"})

    def test_no_query_means_empty_ctx_query(self):
        app = stub_app("#/picks")
        self.assertEqual(app.json("window.__log[0].ctx.query"), {})
        self.assertEqual(app.eval("location.hash"), "#/picks")

    def test_replace_refines_the_url_without_remounting_or_hashchange(self):
        app = stub_app("#/backtest/2330")
        app.eval("window.__changes = 0; window.addEventListener('hashchange', function () { window.__changes++; }); 0")
        self.assertEqual(app.eval("window.__log[0].ctx.replace('#/backtest/2330?h=5&thr=0.6')"), True)
        app.settle(300)
        self.assertEqual(app.eval("location.hash"), "#/backtest/2330?h=5&thr=0.6")
        self.assertEqual(app.eval("window.__changes"), 0)
        self.assertEqual(app.eval("window.__log.length"), 1, "the view must not be re-mounted")
        self.assertEqual(app.json("TQ.router.current().query"), "h=5&thr=0.6")
        # the router now knows the refined URL: navigating to exactly it is a no-op, to the bare path is a fresh mount
        app.eval("TQ.router.navigate('#/backtest/2330?h=5&thr=0.6')")
        app.settle(100)
        self.assertEqual(app.eval("window.__log.length"), 1)
        app.goto("#/backtest/2330", 100)
        self.assertEqual(app.eval("window.__log.length"), 2)
        self.assertEqual(app.json("window.__log[1].ctx.query"), {})
        self.assertEqual(app.json("window.__log.map(function (r) { return r.destroyed; })"), [1, 0])

    def test_replace_accepts_a_missing_hash_sign_and_refuses_other_routes(self):
        app = stub_app("#/method")
        self.assertEqual(app.eval("window.__log[0].ctx.replace('method/garch')"), True)
        self.assertEqual(app.eval("location.hash"), "#/method/garch")
        self.assertEqual(app.eval("window.__log[0].ctx.replace('#/picks')"), False)           # a view can only refine its own route
        self.assertEqual(app.eval("window.__log[0].ctx.replace('#/nonsense')"), False)
        self.assertEqual(app.eval("window.__log[0].ctx.replace('')"), False)
        self.assertEqual(app.eval("location.hash"), "#/method/garch")
        self.assertEqual(app.eval("window.__log.length"), 1)

    def test_replace_from_a_torn_down_view_is_ignored(self):
        app = stub_app("#/backtest/2330")
        app.goto("#/picks", 100)
        self.assertEqual(app.eval("window.__log[0].ctx.replace('#/backtest/2330?h=20')"), False)
        self.assertEqual(app.eval("location.hash"), "#/picks")

    def test_query_change_from_outside_remounts_with_the_new_settings(self):
        app = stub_app("#/backtest/2330?h=1")
        app.goto("#/backtest/2330?h=5", 100)
        self.assertEqual(app.eval("window.__log.length"), 2)
        self.assertEqual(app.json("window.__log[1].ctx.query"), {"h": "5"})
        self.assertEqual(app.eval("window.__log[0].destroyed"), 1)

    def test_other_routes_are_unaffected(self):
        app = stub_app("#/stock/2330")
        app.goto("#/stock/0050", 100)
        app.goto("#/picks", 100)
        self.assertEqual(app.eval("window.__log.length"), 3)
        self.assertEqual(app.errors(), [])

    def test_stub_helpers_come_from_the_shell_test_module(self):
        self.assertIn("window.__log", STUB_VIEWS)
        self.assertTrue((FRONT / "app.js").exists())


if __name__ == "__main__":
    unittest.main()
