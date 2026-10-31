"""The real backend (backend/server.py) serves the real frontend: every file index.html references is reachable with the right type, the
page works under the server's Content-Security-Policy, and the API routes the frontend calls exist.  Pure stdlib + the server module
(skipped when it cannot be imported); no network access, no engine call."""
from __future__ import annotations

import json
import re
import threading
import unittest
import urllib.error
import urllib.request

from front_helpers import FRONT

try:  # pragma: no cover - environment dependent
    from backend import server as backend_server
    HAVE_SERVER = True
except Exception:  # noqa: BLE001
    backend_server = None
    HAVE_SERVER = False


def get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path, timeout=15) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@unittest.skipUnless(HAVE_SERVER, "backend.server not importable")
class ServedByBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = backend_server.make_server("127.0.0.1", 0, quiet=True)
        cls.port = cls.srv.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        try:
            cls.srv.app.close()
        except Exception:  # noqa: BLE001
            pass
        cls.thread.join(5)

    def test_index_and_every_referenced_asset(self):
        status, headers, body = get(self.base, "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        html = body.decode("utf-8")
        refs = re.findall(r'(?:src|href)="([^"#:]+\.(?:js|css))"', html)
        self.assertEqual(refs, ["style.css", "style_views.css", "format.js", "charts.js", "core.js", "view_stock.js", "view_picks.js",
                                "view_backtest.js", "view_method.js", "app.js"])
        for ref in refs:
            st, h, data = get(self.base, "/" + ref)
            self.assertEqual(st, 200, ref)
            want = "javascript" if ref.endswith(".js") else "text/css"
            self.assertIn(want, h["Content-Type"], ref)
            self.assertEqual(data, (FRONT / ref).read_bytes(), ref)
            self.assertIn("no-cache", h.get("Cache-Control", ""), ref)

    def test_csp_allows_what_the_page_needs_and_nothing_external(self):
        _s, headers, body = get(self.base, "/")
        csp = headers["Content-Security-Policy"]
        directives = dict(d.strip().split(" ", 1) for d in csp.split(";") if d.strip())
        self.assertIn("'self'", directives["script-src"])
        self.assertIn("'unsafe-inline'", directives["script-src"])      # the tiny theme bootstrap in <head>
        self.assertIn("data:", directives["img-src"])                    # the inline SVG favicon
        self.assertIn("'self'", directives["connect-src"])               # fetch('/api/...')
        for name, value in directives.items():
            self.assertNotRegex(value, r"https?://", f"{name} must not allow external origins")
        # inline scripts in index.html: exactly the theme bootstrap, which only touches localStorage / data-theme
        inline = re.findall(r"<script>(.*?)</script>", body.decode("utf-8"), re.S)
        self.assertEqual(len(inline), 1)
        self.assertIn("data-theme", inline[0])

    def test_unknown_static_and_traversal_do_not_serve_files(self):
        self.assertEqual(get(self.base, "/nope.js")[0], 404)
        self.assertEqual(get(self.base, "/../DESIGN.md")[0], 404)
        self.assertEqual(get(self.base, "/%2e%2e/DESIGN.md")[0], 404)

    def test_api_routes_the_frontend_uses_exist(self):
        js = "".join((FRONT / f).read_text(encoding="utf-8") for f in ("core.js", "view_stock.js", "view_picks.js"))
        used = set(re.findall(r"'(/api/[a-z/]+)'", js))
        self.assertEqual(used, {"/api/search", "/api/status", "/api/history", "/api/forecast", "/api/recommendations", "/api/backtest", "/api/backtest/universe"})
        routes = set(self.srv.app.routes())
        self.assertTrue(used <= routes, used - routes)
        # error envelope the frontend parses (no engine / network involved: bad parameters are rejected up front)
        st, _h, body = get(self.base, "/api/history")
        self.assertEqual(st, 400)
        env = json.loads(body)
        self.assertEqual(env["status"], "error")
        self.assertEqual(env["error"]["code"], "bad_request")
        self.assertIn("symbol", env["error"]["message"])
        st, _h, body = get(self.base, "/api/nothing")
        self.assertEqual(st, 404)
        self.assertEqual(json.loads(body)["error"]["code"], "bad_request")


if __name__ == "__main__":
    unittest.main()
