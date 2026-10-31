"""Shared helpers for the frontend (module D) tests.

* ``make_ctx(files)`` -- a py_mini_racer (V8) context pre-loaded with ``front_prelude.js`` (fake DOM, clock,
  fetch, recording canvas) and the requested frontend scripts.  Needs the ``py_mini_racer`` package, which is only
  installed in the scratch venv; ``HAVE_RACER`` tells tests whether to skip.
* ``html_to_tree(html)`` -- parse index.html into a JSON tree that the fake DOM can instantiate.
* fixture loaders (schema-built ones in tests/fixtures/*.json, REAL API output in tests/fixtures/real/*.json),
* ``FrontApp`` -- the whole page (index.html skeleton + every script in document order + declarative fetch mocks) running
  inside V8, with helpers to settle the fake clock / promises, click, type, read text and serialise the DOM.
"""
from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "frontend"
TESTS = Path(__file__).resolve().parent
FIX = TESTS / "fixtures"
PRELUDE = TESTS / "front_prelude.js"

SKIP_MSG = ("py_mini_racer cannot be imported -- run these tests with a Python that has mini-racer installed "
            "(see HANDOFF.md, section 'Environment': python3 -m venv .venv && .venv/bin/pip install mini-racer pillow weasyprint pymupdf numpy scipy)")
try:  # pragma: no cover - environment dependent
    from py_mini_racer import MiniRacer
    HAVE_RACER = True
except Exception:  # noqa: BLE001
    MiniRacer = None
    HAVE_RACER = False



def script_files_from_html() -> list[str]:
    """The local <script src> files of index.html, in document order."""
    html = (FRONT / "index.html").read_text(encoding="utf-8")
    out: list[str] = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "script" and d.get("src"):
                out.append(d["src"])

    P().feed(html)
    return out


def fixture_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def fixture(name: str):
    return json.loads(fixture_text(name))


def make_ctx(files=("format.js", "charts.js"), dom: bool = False, timeout: int = 20000):
    """Fresh V8 context with the prelude and `files` (names relative to frontend/) loaded."""
    if not HAVE_RACER:
        raise RuntimeError(SKIP_MSG)
    ctx = MiniRacer()
    ctx.eval(PRELUDE.read_text(encoding="utf-8"))
    if dom:
        ctx.eval("window.__buildDom(" + json.dumps(html_to_tree((FRONT / "index.html").read_text(encoding="utf-8"))) + ")")
    for f in files:
        src = (FRONT / f).read_text(encoding="utf-8")
        ctx.eval(src)
    return ctx


def load_fixtures_into(ctx, names) -> None:
    """Expose fixtures as ``__fx['<name without .json>']`` inside the context (``real/x.json`` -> key ``real_x``)."""
    ctx.eval("window.__fx = window.__fx || {};")
    for n in names:
        fn = n if n.endswith(".json") else n + ".json"
        key = fn[:-5].replace("/", "_")
        ctx.eval("window.__fx[" + json.dumps(key) + "] = " + fixture_text(fn) + ";")


# --------------------------------------------------------------------------------------------- html -> tree
VOID = {"meta", "link", "br", "hr", "img", "input", "source", "area", "base", "col", "embed", "param", "track", "wbr"}


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "#root", "attrs": {}, "children": []}
        self.stack = [self.root]
        self.html_attrs: dict = {}
        self.in_head = False
        self.head: list = []
        self.body: list = []
        self.errors: list[str] = []

    def _container(self):
        return self.stack[-1]

    def handle_starttag(self, tag, attrs):
        a = {k: (v if v is not None else "") for k, v in attrs}
        if tag == "html":
            self.html_attrs = a
            return
        if tag == "head":
            self.in_head = True
            self.stack.append({"tag": "head", "attrs": a, "children": self.head})
            return
        if tag == "body":
            self.stack.append({"tag": "body", "attrs": a, "children": self.body})
            return
        node = {"tag": tag, "attrs": a, "children": []}
        self._container()["children"].append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        a = {k: (v if v is not None else "") for k, v in attrs}
        self._container()["children"].append({"tag": tag, "attrs": a, "children": []})

    def handle_endtag(self, tag):
        if tag in ("html",):
            return
        if tag in VOID:
            return
        if tag == "head":
            self.in_head = False
        if len(self.stack) > 1 and self.stack[-1]["tag"] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-1]['tag']})")

    def handle_data(self, data):
        if data.strip() == "" and len(data) > 0:
            return
        self._container()["children"].append({"t": data})

    def close(self):
        super().close()
        if len(self.stack) != 1:
            self.errors.append("unclosed tags: " + ", ".join(n["tag"] for n in self.stack[1:]))


def parse_html(html: str) -> _TreeBuilder:
    b = _TreeBuilder()
    b.root["children"] = []
    b.stack = [b.root]
    b.feed(html)
    b.close()
    return b


def html_to_tree(html: str) -> dict:
    b = parse_html(html)
    return {"htmlAttrs": b.html_attrs, "head": b.head, "body": b.body}


def out_dir() -> Path | None:
    """Directory for PNG / recording artefacts when TQ_FRONT_OUT is set."""
    p = os.environ.get("TQ_FRONT_OUT")
    if not p:
        return None
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------------- the whole page in V8
def _ok(body):
    return {"status": 200, "body": body}


def default_rules(real: bool = True) -> list[dict]:
    """Fetch-mock rules answering every endpoint from fixtures (see ``window.__routes`` in front_prelude.js).

    ``real=True`` uses tests/fixtures/real/* (captured from the running backend); ``False`` the schema-built set.
    Unknown symbols answer with the server's 404 ``unknown_symbol`` envelope.
    """
    p = "real_" if real else ""
    err = {"status": 404, "body": "real_error_unknown" if real else "error"}
    return [
        {"re": r"^/api/status", "resp": _ok(p + "status")},
        {"re": r"^/api/search\?limit=20$", "resp": _ok(p + "search_empty")},
        {"re": r"^/api/search\?q=", "resp": _ok(p + "search")},
        {"re": r"^/api/history\?symbol=2330", "resp": _ok(p + "history_2330")},
        {"re": r"^/api/history\?symbol=0050", "resp": _ok(p + "history_0050")},
        {"re": r"^/api/history\?symbol=6488", "resp": _ok(p + "history_6488" if real else "history_0050")},
        {"re": r"^/api/forecast\?symbol=2330", "resp": _ok(p + "forecast_2330")},
        {"re": r"^/api/forecast\?symbol=0050", "resp": _ok(p + "forecast_0050")},
        {"re": r"^/api/forecast\?symbol=6488", "resp": _ok(p + "forecast_6488" if real else "forecast_nulls")},
        {"re": r"^/api/forecast\?symbol=9999", "resp": _ok("forecast_sparse")},
        {"re": r"^/api/history\?symbol=9999", "resp": _ok("history_0050")},
        {"re": r"^/api/(forecast|history)\?symbol=", "resp": err},
        {"re": r"^/api/recommendations\?h=1&", "resp": _ok(p + "recommendations_h1")},
        {"re": r"^/api/recommendations\?h=20&", "resp": _ok(p + "recommendations_h20")},
        {"re": r"^/api/recommendations\?h=5&", "resp": _ok(p + "recommendations_h5")},
        {"re": r"^/api/backtest/universe", "resp": _ok(p + "backtest_universe_h5" if real else "backtest_universe")},
        {"re": r"^/api/backtest\?symbol=2330&h=1", "resp": _ok(p + "backtest_2330_h1" if real else "backtest_2330_h1")},
        {"re": r"^/api/backtest\?symbol=2330&h=5", "resp": _ok(p + "backtest_2330_h5" if real else "backtest_2330_h5")},
        {"re": r"^/api/backtest\?symbol=2330&h=20", "resp": _ok(p + "backtest_2330_h20")},
        {"re": r"^/api/backtest\?symbol=0050&h=20", "resp": _ok(p + "backtest_0050_h20" if real else "backtest_2330_h20")},
    ]


def default_resp(real: bool, regex: str):
    """The response of the default rule whose ``re`` is exactly ``regex`` (tests build sequences / variants from it; never index the list)."""
    for r in default_rules(real):
        if r["re"] == regex:
            return r["resp"]
    raise KeyError(regex)


FIXTURES_REAL = ["real/status", "real/search", "real/search_empty", "real/search_code", "real/history_2330", "real/history_0050",
                 "real/history_6488", "real/forecast_2330", "real/forecast_0050", "real/forecast_6488", "real/error_unknown",
                 "real/error_9999", "real/recommendations_h1", "real/recommendations_h5", "real/recommendations_h20",
                 "real/backtest_2330_h1", "real/backtest_2330_h5", "real/backtest_universe_h5", "real/backtest_2330_h20", "real/backtest_0050_h20",
                 "real/legacy_recommendations_h5", "real/legacy_backtest_universe_h5"]
FIXTURES_SCHEMA = ["status", "search", "search_empty", "history_2330", "history_0050", "forecast_2330", "forecast_0050",
                   "forecast_nulls", "forecast_sparse", "recommendations_h1", "recommendations_h5", "recommendations_h20", "recommendations_empty",
                   "backtest_2330_h1", "backtest_2330_h5", "backtest_2330_h20", "backtest_nulls", "backtest_universe", "backtest_universe_nulls",
                   "computing", "computing_nototal", "error", "error_insufficient"]


class FrontApp:
    """index.html + every <script src> (document order) + fixtures + mocks, running in one V8 context.

        app = FrontApp(hash="#/stock/2330")      # boots like a browser would
        app.settle(3000)                          # advance the fake clock, letting promises / polls / timers run
        app.text("#view")                         # rendered text
        app.click(".seg-btn"), app.type("#search-host input", "台積"), app.press(selector, "ArrowDown")
    """

    def __init__(self, hash: str = "", viewport: int = 1280, dark: bool = False, real: bool = True, rules=None,
                 fixtures=None, boot: bool = True, scripts=None, storage: dict | None = None, delay: int = 0,
                 prefers_dark: bool = False):
        if not HAVE_RACER:
            raise RuntimeError(SKIP_MSG)
        self.ctx = MiniRacer()
        self.ctx.eval(PRELUDE.read_text(encoding="utf-8"))
        names = fixtures if fixtures is not None else (FIXTURES_REAL if real else []) + FIXTURES_SCHEMA
        load_fixtures_into(self.ctx, names)
        self.ctx.eval("window.__buildDom(" + json.dumps(html_to_tree((FRONT / "index.html").read_text(encoding="utf-8"))) + ")")
        self.ctx.eval(f"window.__setViewport({int(viewport)}, 900); window.__locHash = {json.dumps(hash)}; window.__fetchDelay = {int(delay)};"
                      f"window.__prefersDark = {'true' if prefers_dark else 'false'};")
        if storage:
            self.ctx.eval("window.__ls = " + json.dumps(storage) + ";")
        if dark:
            self.ctx.eval("window.__ls['tq.theme'] = '\"dark\"';")
        self.set_rules(rules if rules is not None else default_rules(real))
        self.scripts = list(scripts) if scripts is not None else script_files_from_html()
        self.loaded: list[str] = []
        if boot:
            self.boot()

    # ---- lifecycle
    def boot(self):
        for f in self.scripts:
            self.ctx.eval((FRONT / f).read_text(encoding="utf-8"), )
            self.loaded.append(f)

    def set_rules(self, rules):
        self.ctx.eval("window.__mock = { rules: " + json.dumps(rules) + ", calls: {} };")

    def prepend_rule(self, rule):
        self.ctx.eval("window.__mock.rules.unshift(" + json.dumps(rule) + ");")

    # ---- JS access
    def eval(self, js):
        return self.ctx.eval(js)

    def json(self, js):
        """Evaluate an expression returning a JSON-able value."""
        out = self.ctx.eval("JSON.stringify(" + js + ")")
        return json.loads(out) if isinstance(out, str) else None

    def run(self, expr: str, ms: int = 3000):
        """Run a JS expression returning a Promise; settle the clock; -> {'ok': value} | {'err': {name, code, status, message}} | None."""
        self.ctx.eval("window.__r = undefined; (" + expr + ").then(function (v) { window.__r = { ok: v === undefined ? null : v }; }, "
                      "function (e) { window.__r = { err: { name: e && e.name, code: e && e.code, status: e && e.status, message: e && e.message } }; }); 0")
        self.settle(ms)
        return self.json("window.__r")

    def settle(self, ms: int = 2000, step: int = 100):
        """Advance the fake clock by ``ms`` in ``step`` slices (each eval flushes the promise microtask queue)."""
        n = max(1, ms // step)
        for _ in range(n):
            self.ctx.eval(f"window.__advance({step})")
        # let trailing microtasks finish
        self.ctx.eval("window.__advance(0)")

    # ---- DOM queries (selector strings are evaluated by the fake DOM's mini selector engine)
    def text(self, sel: str = "#view") -> str:
        return self.ctx.eval(f"(function(){{var e=document.querySelector({json.dumps(sel)}); return e? e.textContent : '';}})()")

    def count(self, sel: str) -> int:
        return int(self.ctx.eval(f"document.querySelectorAll({json.dumps(sel)}).length"))

    def attr(self, sel: str, name: str):
        return self.ctx.eval(f"(function(){{var e=document.querySelector({json.dumps(sel)}); return e? e.getAttribute({json.dumps(name)}) : null;}})()")

    def texts(self, sel: str) -> list[str]:
        return json.loads(self.ctx.eval(f"JSON.stringify(Array.prototype.map.call(document.querySelectorAll({json.dumps(sel)}), function(e){{return e.textContent;}}))"))

    def click(self, sel: str, index: int = 0):
        self.ctx.eval(f"document.querySelectorAll({json.dumps(sel)})[{index}].click()")

    def type(self, sel: str, value: str):
        self.ctx.eval(f"(function(){{var e=document.querySelector({json.dumps(sel)}); e.focus(); e.value={json.dumps(value)}; e.dispatchEvent(new Event('input',{{bubbles:true}}));}})()")

    def press(self, sel: str, key: str):
        self.ctx.eval(f"(function(){{var e=document.querySelector({json.dumps(sel)}); e.dispatchEvent(new KeyboardEvent('keydown',{{key:{json.dumps(key)},bubbles:true,cancelable:true}}));}})()")

    def goto(self, hash: str, settle_ms: int = 0):
        self.ctx.eval(f"window.location.hash = {json.dumps(hash)};")
        if settle_ms:
            self.settle(settle_ms)

    # ---- diagnostics
    def console(self, kind: str | None = None) -> list:
        items = json.loads(self.ctx.eval("JSON.stringify(window.__console)"))
        return [m for k, m in items if kind is None or k == kind]

    def errors(self) -> list[str]:
        return self.console("error")

    def fetched(self) -> list[str]:
        return json.loads(self.ctx.eval("JSON.stringify(window.__fetchLog)"))

    def classes_used(self) -> list[str]:
        return json.loads(self.ctx.eval("JSON.stringify(Object.keys(window.__classLog))"))

    def pending_timers(self) -> int:
        return int(self.ctx.eval("window.__pendingTimers()"))

    def serialize(self, sel: str = "body"):
        """-> {'html': body html with <img data-canvas> placeholders, 'canvases': [{id,W,H,ops,label}]}."""
        return json.loads(self.ctx.eval(f"window.__serialize(document.querySelector({json.dumps(sel)}))"))


BAD_TEXT = ("NaN", "undefined", "[object", "Infinity", "null")


def bad_text_hits(text: str) -> list[str]:
    """Substrings that must never appear in rendered text."""
    return [b for b in BAD_TEXT if b in text]


# --------------------------------------------------------------------------------------------- generic DOM audits
AUDIT_JS = r"""
(function () {
  var out = { dupIds: [], badRefs: [], unnamedButtons: [], unlabelledCanvas: [], unlabelledInputs: [], emptyLinks: [], imgsNoAlt: [] };
  var seen = {};
  var all = document.querySelectorAll('[id]');
  for (var i = 0; i < all.length; i++) { var id = all[i].id; if (seen[id]) out.dupIds.push(id); seen[id] = 1; }
  ['aria-describedby', 'aria-controls', 'aria-labelledby'].forEach(function (a) {
    var els = document.querySelectorAll('[' + a + ']');
    for (var i = 0; i < els.length; i++) {
      els[i].getAttribute(a).split(/\s+/).forEach(function (ref) { if (ref && !document.getElementById(ref)) out.badRefs.push(a + '=' + ref); });
    }
  });
  var bs = document.querySelectorAll('button');
  for (var j = 0; j < bs.length; j++) {
    var b = bs[j];
    if (!(b.textContent || '').trim() && !b.getAttribute('aria-label') && !b.getAttribute('title')) out.unnamedButtons.push(b.className);
  }
  var cs = document.querySelectorAll('canvas');
  for (var k = 0; k < cs.length; k++) {
    if (cs[k].getAttribute('aria-hidden') === 'true') continue;                    // decorative duplicate of real text
    if (cs[k].getAttribute('role') !== 'img' || !cs[k].getAttribute('aria-label')) out.unlabelledCanvas.push(cs[k].className);
  }
  var ins = document.querySelectorAll('input');
  for (var m = 0; m < ins.length; m++) {
    var inp = ins[m];
    var lab = inp.getAttribute('aria-label') || (inp.id && document.querySelector('label[for="' + inp.id + '"]'));
    if (!lab) out.unlabelledInputs.push(inp.className);
  }
  var as = document.querySelectorAll('a');
  for (var n = 0; n < as.length; n++) {
    if (!as[n].getAttribute('href')) out.emptyLinks.push(as[n].textContent);
    if (!(as[n].textContent || '').trim() && !as[n].getAttribute('aria-label')) out.emptyLinks.push('(unnamed)');
  }
  out.h1 = document.querySelectorAll('h1').length;
  return out;
})()
"""


def audit(app: "FrontApp") -> dict:
    """Accessibility / hygiene facts about the current DOM (duplicate ids, dangling aria references, unnamed controls)."""
    return app.json(AUDIT_JS)


def assert_clean(tc, app: "FrontApp", sel: str = "#view", allow_errors: bool = False) -> None:
    """Common post-conditions of every render: no NaN/undefined text, no console errors, sane accessibility."""
    text = app.text(sel)
    tc.assertEqual(bad_text_hits(text), [], "bad token in rendered text")
    if not allow_errors:
        tc.assertEqual(app.errors(), [], "console.error during render")
    a = audit(app)
    tc.assertEqual(a["dupIds"], [], "duplicate ids")
    tc.assertEqual(a["badRefs"], [], "dangling aria references")
    tc.assertEqual(a["unnamedButtons"], [], "buttons without an accessible name")
    tc.assertEqual(a["unlabelledCanvas"], [], "canvases without role=img + aria-label")
    tc.assertEqual(a["unlabelledInputs"], [], "inputs without a label")
    tc.assertEqual(a["emptyLinks"], [], "links without href / name")
