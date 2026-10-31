"""core.js: DOM builder, API client (get / poll / status), prefs, theme and the reusable UI components.

Runs inside V8 (py_mini_racer) with the fake DOM of front_prelude.js; skipped when py_mini_racer is missing
(run with the scratch venv python, see front_helpers.SKIP_MSG).
"""
from __future__ import annotations

import json
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, bad_text_hits

CORE = ["format.js", "charts.js", "core.js"]


def core_app(rules=None, **kw):
    kw.setdefault("fixtures", [])
    return FrontApp(scripts=CORE, rules=rules if rules is not None else [], **kw)


def R(status=200, body=None, **kw):
    d = {"status": status, "body": body if body is not None else {}}
    d.update(kw)
    return d


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class DomBuilder(unittest.TestCase):
    def setUp(self):
        self.app = core_app()

    def ev(self, js):
        return self.app.json(js)

    def test_props_children_and_events(self):
        r = self.ev("""(function () {
          var clicks = 0;
          var el = TQ.h('div', { class: ['a', null, ['b', false, 'c']], id: 'x', title: 'T', 'aria-hidden': true, 'aria-pressed': false,
                                 style: { color: 'red', marginTop: 4, '--k': '1px', opacity: 0.5, lineHeight: 1.5, zIndex: 3 }, dataset: { fooBar: 1, z: null },
                                 onClick: function () { clicks++; }, tabindex: 0 },
                       'hi', 7, [null, TQ.h('b', null, 'k'), false], undefined, NaN, Infinity, 'NaN', 'undefined', {a: 1});
          el.click();
          return { cls: el.className, id: el.id, title: el.getAttribute('title'), ah: el.getAttribute('aria-hidden'),
                   ap: el.getAttribute('aria-pressed'), color: el.style.color, mt: el.style.marginTop, custom: el.style['--k'],
                   op: el.style.opacity, lh: el.style.lineHeight, z: el.style.zIndex, d1: el.getAttribute('data-foo-bar'), d2: el.getAttribute('data-z'), ti: el.getAttribute('tabindex'),
                   text: el.textContent, clicks: clicks, nkids: el.childNodes.length };
        })()""")
        self.assertEqual(r["cls"], "a b c")
        self.assertEqual(r["id"], "x")
        self.assertEqual(r["title"], "T")
        self.assertEqual((r["ah"], r["ap"]), ("true", "false"))
        self.assertEqual((r["color"], r["mt"], r["custom"], r["op"]), ("red", "4px", "1px", "0.5"))
        self.assertEqual((r["lh"], r["z"]), ("1.5", "3"))                       # unitless properties never get "px"
        self.assertEqual((r["d1"], r["d2"]), ("1", None))
        self.assertEqual(r["ti"], "0")
        self.assertEqual(r["clicks"], 1)
        # NaN / Infinity / 'NaN' / 'undefined' are rendered as the em dash, objects and null are dropped
        self.assertEqual(r["text"], "hi7k———" + "—")
        self.assertEqual(bad_text_hits(r["text"]), [])

    def test_boolean_props_value_html_and_skips(self):
        r = self.ev("""(function () {
          var b = TQ.h('button', { disabled: true, type: 'button' }, 'x');
          var b2 = TQ.h('button', { disabled: false, hidden: false });
          var i = TQ.h('input', { value: 5, type: 'range', checked: true, placeholder: null, 'data-x': undefined });
          var s = TQ.h('span', { html: '<i>static</i>' });
          var t = TQ.h('p', { text: 'only text' }, 'ignored? no: children still append');
          var h = TQ.h('div', { hidden: true });
          return { d1: b.disabled, d2: b2.disabled, hid2: b2.hidden, val: i.value, chk: i.checked, ph: i.getAttribute('placeholder'),
                   type: i.getAttribute('type'), html: s.innerHTML, ptext: t.textContent, hid: h.hidden };
        })()""")
        self.assertTrue(r["d1"])
        self.assertFalse(r["d2"])
        self.assertFalse(r["hid2"])
        self.assertEqual(r["val"], "5")
        self.assertTrue(r["chk"])
        self.assertIsNone(r["ph"])
        self.assertEqual(r["html"], "<i>static</i>")
        self.assertIn("only text", r["ptext"])
        self.assertTrue(r["hid"])

    def test_props_may_be_omitted_and_svg_namespace(self):
        r = self.ev("""(function () {
          var p = TQ.h('p', 'hello', ' world');
          var q = TQ.h('p', null, 'x');
          var n = TQ.h('span', TQ.h('i', null, 'inner'));
          var s = TQ.h('svg', { viewBox: '0 0 1 1', class: 'ic' }, TQ.h('path', { d: 'M0 0' }));
          return { p: p.textContent, q: q.textContent, n: n.textContent, ns: s.namespaceURI, cls: s.getAttribute('class'),
                   vb: s.getAttribute('viewBox'), pns: s.firstChild.namespaceURI };
        })()""")
        self.assertEqual((r["p"], r["q"], r["n"]), ("hello world", "x", "inner"))
        self.assertEqual(r["ns"], "http://www.w3.org/2000/svg")
        self.assertEqual(r["pns"], "http://www.w3.org/2000/svg")
        self.assertEqual((r["cls"], r["vb"]), ("ic", "0 0 1 1"))

    def test_scope_dispose_and_nested(self):
        r = self.ev("""(function () {
          var log = [];
          var s = TQ.scope(), child = TQ.scope();
          s.add(function () { log.push('fn'); });
          s.add({ destroy: function () { log.push('obj'); } });
          s.add(function () { throw new Error('boom'); });
          child.add(function () { log.push('child'); });
          s.add(child);
          s.dispose(); s.dispose();
          return { log: log, errs: window.__console.filter(function (c) { return c[0] === 'error'; }).length };
        })()""")
        self.assertEqual(r["log"], ["fn", "obj", "child"])
        self.assertEqual(r["errs"], 1)

    def test_clear_uid_and_code_href(self):
        r = self.ev("""(function () {
          var d = TQ.h('div', null, 'a', TQ.h('b')); TQ.clear(d);
          return { n: d.childNodes.length, u1: TQ.uid('k'), u2: TQ.uid('k'), href: TQ.codeHref('00878'), hb: TQ.codeHref('^TWII', 'backtest') };
        })()""")
        self.assertEqual(r["n"], 0)
        self.assertNotEqual(r["u1"], r["u2"])
        self.assertEqual(r["href"], "#/stock/00878")
        self.assertEqual(r["hb"], "#/backtest/%5ETWII")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ApiClient(unittest.TestCase):
    def test_get_success_and_query_building(self):
        app = core_app([{"re": r"^/api/x\?a=1&b=%E5%8F%B0&e=0$", "resp": R(body={"status": "ready", "v": 3})}])
        r = app.run("TQ.api.get('/api/x', { a: 1, b: '台', c: null, d: '', e: 0, f: undefined, g: NaN })")
        self.assertEqual(r, {"ok": {"status": "ready", "v": 3}})
        self.assertEqual(app.fetched(), ["/api/x?a=1&b=%E5%8F%B0&e=0"])

    def test_error_envelope_becomes_api_error(self):
        env = {"status": "error", "error": {"code": "unknown_symbol", "message": "找不到股票代號「1234」"}}
        app = core_app([{"re": r"^/api/e", "resp": R(404, env)}])
        r = app.run("TQ.api.get('/api/e')")
        self.assertEqual(r["err"]["name"], "ApiError")
        self.assertEqual(r["err"]["code"], "unknown_symbol")
        self.assertEqual(r["err"]["status"], 404)
        self.assertEqual(r["err"]["message"], "找不到股票代號「1234」")
        self.assertTrue(app.json("(function(){ var e = new TQ.ApiError('m', 'c', 1); return [e instanceof Error, e instanceof TQ.ApiError]; })()") == [True, True])

    def test_error_envelope_with_http_200_and_bad_bodies(self):
        app = core_app([
            {"re": r"^/api/a", "resp": R(200, {"status": "error", "error": {"code": "internal", "message": "壞了"}})},
            {"re": r"^/api/b", "resp": R(500, invalidJson=True)},
            {"re": r"^/api/c", "resp": R(200, invalidJson=True)},
            {"re": r"^/api/d", "resp": R(503, {})},
        ])
        self.assertEqual(app.run("TQ.api.get('/api/a')")["err"]["code"], "internal")
        b = app.run("TQ.api.get('/api/b')")["err"]
        self.assertEqual((b["code"], b["status"]), ("http_500", 500))
        self.assertTrue(b["message"])
        self.assertEqual(app.run("TQ.api.get('/api/c')")["err"]["code"], "bad_response")
        self.assertEqual(app.run("TQ.api.get('/api/d')")["err"]["code"], "http_503")

    def test_network_failure(self):
        app = core_app([{"re": r"^/api/n", "resp": {"networkError": True}}])
        r = app.run("TQ.api.get('/api/n')")["err"]
        self.assertEqual((r["name"], r["code"], r["status"]), ("ApiError", "network", 0))
        self.assertIn("無法連線", r["message"])

    def test_abort_before_and_during(self):
        app = core_app([{"re": r"^/api/s", "resp": R(body={"status": "ready"})}], delay=500)
        r = app.run("(function(){ var c = new AbortController(); c.abort(); return TQ.api.get('/api/s', null, { signal: c.signal }); })()")
        self.assertEqual(r["err"]["name"], "AbortError")
        r = app.run("(function(){ var c = new AbortController(); var p = TQ.api.get('/api/s', null, { signal: c.signal }); setTimeout(function(){ c.abort(); }, 100); return p; })()")
        self.assertEqual(r["err"]["name"], "AbortError")

    def test_poll_follows_computing_until_ready(self):
        seq = [R(body={"status": "computing", "progress": {"done": 1, "total": 4, "message": "a"}}),
               R(body={"status": "computing", "progress": {"done": 3, "total": 4}}),
               R(body={"status": "computing"}),
               R(body={"status": "ready", "report": {"x": 1}})]
        app = core_app([{"re": r"^/api/j", "seq": seq, "resp": seq[0]}])
        app.eval("window.__prog = [];")
        r = app.run("TQ.api.poll('/api/j', { h: 1 }, { onProgress: function (p) { window.__prog.push(p); } })", ms=8000)
        self.assertEqual(r, {"ok": {"status": "ready", "report": {"x": 1}}})
        prog = app.json("window.__prog")
        self.assertEqual(prog, [{"done": 1, "total": 4, "message": "a"}, {"done": 3, "total": 4, "message": ""}, {"done": 0, "total": 0, "message": ""}])
        self.assertEqual(len(app.fetched()), 4)

    def test_poll_interval_is_respected(self):
        seq = [R(body={"status": "computing"}), R(body={"status": "ready"})]
        app = core_app([{"re": r"^/api/j", "seq": seq, "resp": seq[0]}])
        app.eval("window.__r = undefined; TQ.api.poll('/api/j', null, { intervalMs: 1500 }).then(function (v) { window.__r = 'done'; }); 0")
        app.settle(1000)
        self.assertEqual(len(app.fetched()), 1)
        app.settle(1000)
        self.assertEqual(len(app.fetched()), 2)
        app.settle(200)
        self.assertEqual(app.json("window.__r"), "done")

    def test_poll_error_and_abort(self):
        seq = [R(body={"status": "computing"}), R(500, {"status": "error", "error": {"code": "internal", "message": "計算失敗"}})]
        app = core_app([{"re": r"^/api/j", "seq": seq, "resp": seq[0]}])
        r = app.run("TQ.api.poll('/api/j')", ms=5000)
        self.assertEqual((r["err"]["code"], r["err"]["message"]), ("internal", "計算失敗"))
        # abort while sleeping between polls: rejects with AbortError and leaves no timers / requests behind
        app = core_app([{"re": r"^/api/j", "resp": R(body={"status": "computing", "progress": {"done": 0, "total": 5}})}])
        r = app.run("(function(){ var c = new AbortController(); var p = TQ.api.poll('/api/j', null, { signal: c.signal }); setTimeout(function(){ c.abort(); }, 2000); return p; })()", ms=6000)
        self.assertEqual(r["err"]["name"], "AbortError")
        n = len(app.fetched())
        app.settle(10000)
        self.assertEqual(len(app.fetched()), n)
        self.assertEqual(app.pending_timers(), 0)

    def test_poll_tolerates_transient_network_errors(self):
        seq = [{"networkError": True}, {"networkError": True}, R(body={"status": "ready", "v": 1})]
        app = core_app([{"re": r"^/api/j", "seq": seq, "resp": seq[0]}])
        self.assertEqual(app.run("TQ.api.poll('/api/j')", ms=20000), {"ok": {"status": "ready", "v": 1}})
        app = core_app([{"re": r"^/api/j", "resp": {"networkError": True}}])
        r = app.run("TQ.api.poll('/api/j')", ms=60000)
        self.assertEqual(r["err"]["code"], "network")
        self.assertEqual(len(app.fetched()), 4)

    def test_status_is_cached_and_never_rejects(self):
        app = core_app([{"re": r"^/api/status", "resp": R(body={"status": "ready", "market_open": False})}])
        a = app.run("TQ.api.status()")
        b = app.run("TQ.api.status()")
        self.assertEqual(a, b)
        self.assertEqual(len(app.fetched()), 1)
        app = core_app([{"re": r"^/api/status", "resp": {"networkError": True}}])
        self.assertEqual(app.run("TQ.api.status()"), {"ok": None})


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class PrefsAndTheme(unittest.TestCase):
    def test_prefs_roundtrip_corrupt_and_blocked_storage(self):
        app = core_app(storage={"tq.bad": "{not json"})
        r = app.json("""(function () {
          TQ.prefs.set('a', { n: [1, 2] });
          var out = { a: TQ.prefs.get('a'), none: TQ.prefs.get('zz', 'dflt'), bad: TQ.prefs.get('bad', 7), raw: window.__ls['tq.a'] };
          window.__lsThrow = true;
          out.setOk = TQ.prefs.set('b', 5);
          out.b = TQ.prefs.get('b');
          out.other = TQ.prefs.get('never', 'fallback');
          return out;
        })()""")
        self.assertEqual(r["a"], {"n": [1, 2]})
        self.assertEqual((r["none"], r["bad"]), ("dflt", 7))
        self.assertEqual(json.loads(r["raw"]), {"n": [1, 2]})
        self.assertFalse(r["setOk"])
        self.assertEqual((r["b"], r["other"]), (5, "fallback"))

    def test_theme_cycle_attribute_persist_and_event(self):
        app = core_app()
        r = app.json("""(function () {
          var el = document.documentElement, events = 0;
          window.addEventListener('tq:theme', function () { events++; });
          var out = { init: TQ.theme.init(), attr0: el.getAttribute('data-theme') };
          out.s1 = TQ.theme.cycle(); out.a1 = el.getAttribute('data-theme');
          out.s2 = TQ.theme.cycle(); out.a2 = el.getAttribute('data-theme'); out.stored = window.__ls['tq.theme'];
          out.s3 = TQ.theme.cycle(); out.a3 = el.getAttribute('data-theme');
          out.set = TQ.theme.set('bogus'); out.events = events; out.eff = TQ.theme.effective();
          return out;
        })()""")
        self.assertEqual((r["init"], r["attr0"]), ("auto", None))
        self.assertEqual((r["s1"], r["a1"]), ("light", "light"))
        self.assertEqual((r["s2"], r["a2"], r["stored"]), ("dark", "dark", '"dark"'))
        self.assertEqual((r["s3"], r["a3"]), ("auto", None))
        self.assertEqual(r["set"], "auto")
        self.assertEqual(r["events"], 4)
        self.assertEqual(r["eff"], "light")

    def test_theme_restores_choice_and_follows_system_in_auto(self):
        app = core_app(storage={"tq.theme": '"dark"'})
        self.assertEqual(app.json("[TQ.theme.init(), document.documentElement.getAttribute('data-theme'), TQ.theme.effective()]"), ["dark", "dark", "dark"])
        app = core_app(prefers_dark=True)
        self.assertEqual(app.json("[TQ.theme.init(), document.documentElement.getAttribute('data-theme'), TQ.theme.effective()]"), ["auto", None, "dark"])
        app = core_app(storage={"tq.theme": '"nonsense"'})
        self.assertEqual(app.json("TQ.theme.init()"), "auto")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class UiComponents(unittest.TestCase):
    def setUp(self):
        self.app = core_app()

    def ev(self, js):
        return self.app.json(js)

    def test_chip_card_kpi_kv_empty_banner(self):
        r = self.ev("""(function () {
          var ui = TQ.ui;
          var chip = ui.chip('買進', 'up', { icon: '▲', title: 't' });
          var card = ui.card('標題', TQ.h('p', null, 'body'), { sub: '副', tip: 'tip', actions: TQ.h('b', null, 'act'), id: 'cc', class: ['x', ['y']] });
          card.setBody('new body');
          var card0 = ui.card(null, 'only');
          var kpi = ui.kpi('標', null, 's', { dir: 'up', tip: 'k' });
          var kv = ui.kv([{ label: 'L1', value: 3, dir: 'down' }, null, { label: 'L2', value: undefined }]);
          var b1 = ui.banner('warn', 'w'), b2 = ui.banner('error', 'e', { title: 'T' }), b3 = ui.banner('bogus', 'x');
          var em = ui.emptyState('空');
          return { chip: chip.className, chipText: chip.textContent, card: card.className, id: card.id, body: card.bodyEl.textContent,
                   labelled: card.getAttribute('aria-labelledby') === card.querySelector('h2').id, h: card.querySelector('.card-title').tagName,
                   head0: card0.querySelector('.card-head'), kpiVal: ui.kpi('a', 5).querySelector('.kpi-value').textContent,
                   kpiNull: kpi.querySelector('.kpi-value').textContent, kpiCls: kpi.querySelector('.kpi-value').className,
                   kvTexts: kv.textContent, kvCls: kv.querySelectorAll('dd').map(function (d) { return d.className; }),
                   roles: [b1.getAttribute('role'), b2.getAttribute('role'), b3.className], bt: b2.textContent, em: em.textContent };
        })()""")
        self.assertEqual(r["chip"], "chip chip-up")
        self.assertEqual(r["chipText"], "▲買進")
        self.assertEqual(r["card"], "card x y")
        self.assertEqual(r["id"], "cc")
        self.assertEqual(r["body"], "new body")
        self.assertTrue(r["labelled"])
        self.assertEqual(r["h"], "H2")
        self.assertIsNone(r["head0"])
        self.assertEqual((r["kpiVal"], r["kpiNull"], r["kpiCls"]), ("5", "—", "kpi-value up"))
        self.assertIn("L1", r["kvTexts"])
        self.assertIn("—", r["kvTexts"])
        self.assertEqual(r["kvCls"], ["down", ""])
        self.assertEqual(r["roles"], ["status", "alert", "banner banner-info"])
        self.assertIn("T", r["bt"])
        self.assertEqual(r["em"], "空")

    def test_change_and_dir_class(self):
        r = self.ev("""(function () {
          var ui = TQ.ui;
          function c(x, dp) { var n = ui.change(x, dp); return [n.className, n.textContent]; }
          return { up: c(0.0123), down: c(-0.0145), flat: c(0), tiny: c(0.00001), nul: c(null), nan: c(NaN), str: c('0.02'),
                   dc: [ui.dirClass(1), ui.dirClass(-1), ui.dirClass(0), ui.dirClass(null), ui.dirClass('x')] };
        })()""")
        self.assertEqual(r["up"], ["chg up", "▲ +1.23%"])
        self.assertEqual(r["down"], ["chg down", "▼ −1.45%"])
        self.assertEqual(r["flat"], ["chg flat", "▬ 0.00%"])
        self.assertEqual(r["tiny"], ["chg flat", "▬ 0.00%"])  # rounds to zero: neither red nor green
        self.assertEqual(r["nul"], ["chg na", "—"])
        self.assertEqual(r["nan"], ["chg na", "—"])
        self.assertEqual(r["str"][1], "▲ +2.00%")
        self.assertEqual(r["dc"], ["up", "down", "flat", "na", "na"])

    def test_skeleton_error_box_and_retry(self):
        r = self.ev("""(function () {
          var ui = TQ.ui, n = 0;
          var sk = ui.skeleton(3);
          var e1 = ui.errorBox(new TQ.ApiError('找不到', 'unknown_symbol', 404), function () { n++; });
          var e2 = ui.errorBox(new Error('internal secret'), null);
          var e3 = ui.errorBox(null, null);
          e1.querySelector('button').click();
          return { sk: [sk.getAttribute('role'), sk.querySelectorAll('.sk-line').length, sk.textContent], e1: e1.textContent, role: e1.getAttribute('role'),
                   n: n, e2: e2.textContent, e2btn: e2.querySelector('button'), e3: e3.textContent };
        })()""")
        self.assertEqual(r["sk"], ["status", 3, "載入中…"])
        self.assertIn("找不到這檔股票", r["e1"])         # title chosen from the error code
        self.assertIn("找不到", r["e1"])
        self.assertIn("unknown_symbol", r["e1"])
        self.assertEqual((r["role"], r["n"]), ("alert", 1))
        self.assertNotIn("secret", r["e2"])          # generic JS errors are never shown verbatim
        self.assertIsNone(r["e2btn"])
        self.assertIn("未預期", r["e3"])

    def test_progress_bar(self):
        r = self.ev("""(function () {
          var p = TQ.ui.progressBar(0, 0, '準備');
          var o = { ind: p.classList.contains('progress-indeterminate'), now0: p.getAttribute('aria-valuenow'), t0: p.textContent };
          p.update(12, 60, '計算 2330');
          o.det = !p.classList.contains('progress-indeterminate'); o.now = p.getAttribute('aria-valuenow'); o.w = p.querySelector('.progress-fill').style.width;
          o.text = p.textContent; o.vt = p.getAttribute('aria-valuetext');
          p.update(99, 60, ''); o.over = p.querySelector('.progress-fill').style.width;
          p.update(NaN, undefined, null); o.back = p.classList.contains('progress-indeterminate');
          o.role = p.getAttribute('role');
          return o;
        })()""")
        self.assertTrue(r["ind"])
        self.assertIsNone(r["now0"])
        self.assertTrue(r["det"])
        self.assertEqual((r["now"], r["w"]), ("20", "20.0%"))
        self.assertIn("12 / 60", r["text"])
        self.assertIn("計算 2330", r["text"])
        self.assertEqual(r["over"], "100.0%")
        self.assertTrue(r["back"])
        self.assertEqual(r["role"], "progressbar")
        self.assertEqual(bad_text_hits(r["text"] + r["t0"]), [])

    def test_field_helper_links_label_control_and_hint(self):
        r = self.ev("""(function () {
          var sel = TQ.h('select', { id: 'f-h' }, TQ.h('option', { value: '1' }, '明日'));
          var f = TQ.ui.field('預測期間', sel, { id: 'f-h', hint: '提示', tip: '說明' });
          document.body.appendChild(f);
          var lab = f.querySelector('label'), hint = f.querySelector('.field-hint');
          return { cls: f.className, for_: lab.getAttribute('for'), desc: sel.getAttribute('aria-describedby') === hint.id, hint: hint.textContent, tip: f.querySelectorAll('.itip').length };
        })()""")
        self.assertEqual((r["cls"], r["for_"], r["desc"], r["hint"], r["tip"]), ("field", "f-h", True, "提示", 1))

    def test_info_tip_keyboard_and_pointer(self):
        r = self.ev("""(function () {
          var t = TQ.ui.infoTip('說明文字');
          document.body.appendChild(t);
          var btn = t.querySelector('button'), pop = t.querySelector('[role=tooltip]');
          var o = { desc: btn.getAttribute('aria-describedby') === pop.id, exp0: btn.getAttribute('aria-expanded'), open0: t.classList.contains('open') };
          btn.click(); o.open1 = t.classList.contains('open'); o.exp1 = btn.getAttribute('aria-expanded');
          btn.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); o.open2 = t.classList.contains('open');
          btn.focus(); o.focusOpen = t.classList.contains('open');
          btn.blur(); o.blurOpen = t.classList.contains('open');
          btn.click(); document.body.click(); o.docClick = t.classList.contains('open');
          var t2 = TQ.ui.infoTip('二');
          document.body.appendChild(t2);
          btn.click(); t2.querySelector('button').click(); o.exclusive = [t.classList.contains('open'), t2.classList.contains('open')];
          o.text = pop.textContent; o.label = btn.getAttribute('aria-label');
          return o;
        })()""")
        self.assertTrue(r["desc"])
        self.assertEqual((r["exp0"], r["open0"]), ("false", False))
        self.assertEqual((r["open1"], r["exp1"]), (True, "true"))
        self.assertFalse(r["open2"])
        self.assertTrue(r["focusOpen"])
        self.assertFalse(r["blurOpen"])
        self.assertFalse(r["docClick"])
        self.assertEqual(r["exclusive"], [False, True])
        self.assertEqual(r["text"], "說明文字")
        # the bubble is position:fixed and placed from the button's rectangle (fake rect: no room above -> below the button, left clamped to 8px)
        pos = self.ev("(function () { var t = TQ.ui.infoTip('x'); document.body.appendChild(t); t.querySelector('button').click(); var s = t.querySelector('.itip-pop').style; return [s.left, s.top]; })()")
        self.assertEqual(pos, ["8px", "308px"])          # fake rect: top 0, bottom 300 -> opens 8px below the button
        # scrolling closes an open tip
        self.assertFalse(self.ev("(function () { var t = TQ.ui.infoTip('x'); document.body.appendChild(t); t.querySelector('button').click(); window.dispatchEvent(new Event('scroll')); return t.classList.contains('open'); })()"))

    def test_segmented_click_keys_and_roving_tabindex(self):
        r = self.ev("""(function () {
          var got = [];
          var seg = TQ.ui.segmented([{ value: 1, label: '明日' }, { value: 5, label: '一週' }, { value: 20, label: '一個月' }], 5, function (v) { got.push(v); }, { label: '期間' });
          document.body.appendChild(seg);
          var b = seg.querySelectorAll('button');
          var st = function () { return b.map(function (x) { return [x.getAttribute('aria-checked'), x.getAttribute('tabindex')]; }); };
          var o = { role: seg.getAttribute('role'), label: seg.getAttribute('aria-label'), s0: st(), r0: b[0].getAttribute('role') };
          b[2].click(); o.s1 = st();
          b[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));   // wraps to the first
          o.s2 = st(); o.focused = document.activeElement === b[0];
          b[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
          b[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
          o.got = got; o.val = seg.getValue(); seg.setValue(1); o.val2 = seg.getValue(); o.gotAfterSet = got.length;
          var tabs = TQ.ui.segmented(['a', 'b'], 'a', null, { role: 'tablist' });
          o.tab = [tabs.getAttribute('role'), tabs.firstChild.getAttribute('role'), tabs.firstChild.getAttribute('aria-selected')];
          b[1].click(); b[1].click(); o.dup = got.length;
          return o;
        })()""")
        self.assertEqual(r["role"], "radiogroup")
        self.assertEqual(r["label"], "期間")
        self.assertEqual(r["r0"], "radio")
        self.assertEqual(r["s0"], [["false", "-1"], ["true", "0"], ["false", "-1"]])
        self.assertEqual(r["s1"], [["false", "-1"], ["false", "-1"], ["true", "0"]])
        self.assertEqual(r["s2"], [["true", "0"], ["false", "-1"], ["false", "-1"]])
        self.assertTrue(r["focused"])
        self.assertEqual(r["got"][:4], [20, 1, 20, 5])
        self.assertEqual(r["val"], 5)
        self.assertEqual(r["val2"], 1)
        self.assertEqual(r["gotAfterSet"], 4)                 # setValue() is silent
        self.assertEqual(r["dup"], 5)                         # re-clicking the active option does not fire again
        self.assertEqual(r["tab"], ["tablist", "tab", "true"])

    def test_table_sorting_missing_last_and_header_clicks(self):
        r = self.ev("""(function () {
          var rows = [{ n: '乙', v: 2 }, { n: '甲', v: null }, { n: '丙', v: 10 }, { n: '丁', v: NaN }, { n: '戊', v: -1 }];
          var t = TQ.ui.table([{ key: 'n', label: '名稱', sortable: true, rowHeader: true },
                               { key: 'v', label: '值', num: true, sortable: true, render: function (r) { return r.v === null ? '—' : String(r.v); } }], rows,
                              { caption: '表', sortKey: 'v', sortDir: 'asc' });
          document.body.appendChild(t);
          function col() { return t.querySelectorAll('tbody tr').map(function (tr) { return tr.firstChild.textContent; }).join(''); }
          function sorts() { return t.querySelectorAll('thead th').map(function (th) { return th.getAttribute('aria-sort'); }); }
          var o = { asc: col(), s0: sorts() };
          t.querySelectorAll('th button')[1].click(); o.desc = col(); o.s1 = sorts();            // same key toggles to desc
          t.querySelectorAll('th button')[0].click(); o.byName = col(); o.s2 = sorts();
          t.setRows([]); o.empty = t.querySelector('tbody').textContent;
          o.hasRegion = [t.getAttribute('role'), t.getAttribute('tabindex'), t.getAttribute('aria-label')];
          o.scope = t.querySelector('th').getAttribute('scope');
          return o;
        })()""")
        self.assertEqual(r["asc"], "戊乙丙甲丁")      # numbers ascending, missing values last (stable)
        self.assertEqual(r["s0"], ["none", "ascending"])
        self.assertEqual(r["desc"], "丙乙戊甲丁")     # missing STILL last when descending
        self.assertEqual(r["s1"], ["none", "descending"])
        self.assertEqual(r["s2"], ["ascending", "none"])
        self.assertEqual(r["empty"], "沒有資料")
        self.assertEqual(r["hasRegion"], ["region", "0", "表"])
        self.assertEqual(r["scope"], "col")

    def test_table_row_click_ignores_links_and_survives_bad_cells(self):
        r = self.ev("""(function () {
          var hits = [];
          var t = TQ.ui.table([{ key: 'a', label: 'A', render: function (r) { return TQ.h('a', { href: '#x' }, 'lnk'); } },
                               { key: 'b', label: 'B', render: function () { throw new Error('bad cell'); } },
                               { key: 'c', label: 'C' }], [{ a: 1, b: 2, c: null }, null], { onRowClick: function (r) { hits.push(r && r.a); } });
          document.body.appendChild(t);
          var trs = t.querySelectorAll('tbody tr');
          trs[0].querySelector('a').click();                 // bubbles to the tr, but originates in a link: ignored
          trs[0].click();
          return { hits: hits, text: t.querySelector('tbody').textContent, errs: window.__console.filter(function (c) { return c[0] === 'error'; }).length };
        })()""")
        self.assertEqual(r["hits"], [1])
        self.assertEqual(bad_text_hits(r["text"]), [])
        self.assertGreaterEqual(r["errs"], 1)

    def test_bars_and_stacked(self):
        r = self.ev("""(function () {
          var ui = TQ.ui;
          var hb = ui.hbar(1.4, { label: 'x' }), hn = ui.hbar(null), hz = ui.hbar(-3);
          var st = ui.stackedBar([{ key: 'a', label: 'A', value: 3, cls: 'w-garch' }, { key: 'b', label: 'B', value: 1, cls: 'w-hmm' },
                                  { key: 'c', value: 0 }, { key: 'd', value: null }, null], { label: '權重' });
          var segs = st.querySelectorAll('.stack-seg');
          return { w: hb.querySelector('.hbar-fill').style.width, now: hb.getAttribute('aria-valuenow'), n: hn.getAttribute('aria-valuenow'),
                   z: hz.querySelector('.hbar-fill').style.width, nseg: segs.length, w0: segs[0].style.width, w1: segs[1].style.width,
                   label: st.getAttribute('aria-label'), txt: st.textContent, empty: ui.stackedBar([]).getAttribute('aria-label') };
        })()""")
        self.assertEqual((r["w"], r["now"], r["n"], r["z"]), ("100.0%", "100", None, "0.0%"))
        self.assertEqual((r["nseg"], r["w0"], r["w1"]), (2, "75.00%", "25.00%"))
        self.assertEqual(r["label"], "權重：A 75%、B 25%")
        self.assertEqual(r["txt"], "75%25%")
        self.assertIn("無資料", r["empty"])

    def test_verdict_and_stats_table(self):
        r = self.ev("""(function () {
          var ui = TQ.ui;
          var v = ui.verdictBox({ level: 'no_edge', title: '沒有優勢', text: '內文', bullets: ['a', '', 5, 'b'] });
          var v0 = ui.verdictBox(null), v1 = ui.verdictBox({ level: 'zzz' });
          var st = ui.statsTable([{ label: '策略', stats: { total_return: 0.5, sharpe: 1.234, max_dd: -0.2, psr: null, dsr: null } },
                                  { label: '買進持有', stats: { total_return: 1.5, sharpe: null, max_dd: -0.4, n_trades: 1 } }, null, { label: 'x', stats: 5 }]);
          var none = ui.statsTable([{ label: 'z', stats: { psr: null } }]);
          return { v: v.textContent, bl: v.querySelectorAll('li').length, cls: v.className, v0: v0.textContent, v1: v1.className,
                   rows: st.querySelectorAll('tbody tr').map(function (tr) { return tr.textContent; }), heads: st.querySelectorAll('thead th').map(function (t) { return t.textContent; }),
                   none: none.textContent };
        })()""")
        self.assertIn("沒有優勢", r["v"])
        self.assertEqual(r["bl"], 2)
        self.assertIn("tone-neutral", r["cls"])
        self.assertIn("尚無結論", r["v0"])
        self.assertIn("tone-neutral", r["v1"])
        self.assertEqual(r["heads"], ["指標", "策略", "買進持有"])
        joined = "|".join(r["rows"])
        self.assertIn("+50.0%", joined)
        self.assertIn("1.23", joined)
        self.assertNotIn("機率夏普", joined)      # a row where every column is missing is dropped
        self.assertIn("交易次數", joined)
        self.assertEqual(bad_text_hits(joined), [])
        self.assertIn("沒有可比較", r["none"])
        # a single trade (buy & hold) has no meaningful "win rate"
        hit = self.ev("""(function () {
          var t = TQ.ui.statsTable([{ label: '策略', stats: { n_trades: 84, trade_hit_rate: 0.52, sharpe: 1 } }, { label: '持有', stats: { n_trades: 1, trade_hit_rate: 1, sharpe: 1 } }]);
          return t.querySelectorAll('tbody tr').filter(function (tr) { return tr.firstChild.textContent.indexOf('交易勝率') === 0; })[0].textContent; })()""")
        self.assertEqual(hit, "交易勝率52.0%—")

    def test_load_into_states(self):
        app = self.app
        app.eval("""window.__slot = TQ.h('div'); document.body.appendChild(window.__slot); window.__n = 0; window.__ctl = new AbortController();
          window.__h = TQ.ui.loadInto(window.__slot, { signal: window.__ctl.signal, load: function () { window.__n++;
                                                         return new Promise(function (res, rej) { setTimeout(function () { if (window.__n === 1) rej(new TQ.ApiError('壞掉', 'internal', 500)); else res({ v: window.__n }); }, 50); }); },
                                                         render: function (d) { return TQ.h('p', null, '值=' + d.v); } }); 0""")
        self.assertIn("載入中", app.json("window.__slot.textContent"))
        app.settle(100)
        self.assertIn("壞掉", app.text("body"))
        app.click(".error-box button")
        app.settle(100)
        self.assertEqual(app.json("window.__slot.textContent"), "值=2")
        # render exceptions become an error box, abort silences everything
        app.eval("""window.__slot2 = TQ.h('div'); TQ.ui.loadInto(window.__slot2, { load: function () { return Promise.resolve(1); }, render: function () { throw new Error('x'); } }); 0""")
        app.settle(100)
        self.assertIn("畫面產生失敗", app.json("window.__slot2.textContent"))
        app.eval("""window.__slot3 = TQ.h('div'); window.__c3 = new AbortController();
          TQ.ui.loadInto(window.__slot3, { signal: window.__c3.signal, load: function () { return new Promise(function (res) { setTimeout(function () { res(1); }, 100); }); }, render: function () { return TQ.h('p', null, 'late'); } });
          window.__c3.abort(); 0""")
        app.settle(300)
        self.assertNotIn("late", app.json("window.__slot3.textContent"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class JavaScriptFiles(unittest.TestCase):
    def test_every_frontend_script_parses_in_v8(self):
        """Syntax check of every .js file under frontend/ (new Function(src)); strict-mode friendly, no stray tokens."""
        from front_helpers import FRONT
        app = core_app()
        for f in sorted(FRONT.glob("*.js")):
            src = f.read_text(encoding="utf-8")
            ok = app.eval("(function () { try { new Function(" + json.dumps(src) + "); return 'ok'; } catch (e) { return String(e); } })()")
            self.assertEqual(ok, "ok", f"{f.name}: {ok}")

    def test_segmented_with_unknown_value_still_has_one_tab_stop(self):
        app = core_app()
        r = app.json("""(function () { var s = TQ.ui.segmented(['a', 'b', 'c'], 'zzz', null); return s.querySelectorAll('button').map(function (b) { return b.getAttribute('tabindex'); }); })()""")
        self.assertEqual(r, ["0", "-1", "-1"])


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ChartCard(unittest.TestCase):
    def test_chart_card_draws_summary_function_spec_and_destroy(self):
        app = core_app()
        app.eval("""window.__card = TQ.chartCard('圖', function (pal, W, H) {
              window.__palSeen = pal.accent; window.__w = W;
              return { dates: ['2026-01-01', '2026-01-02', '2026-01-05'], n: 3, panels: [{ title: 't', series: [{ key: 'a', label: 'A', color: pal.accent, values: [1, 2, 1.5] }] }] };
            }, { kind: 'timeSeries', sub: '副', height: 200, note: '註' });
          document.body.appendChild(window.__card); 0""")
        app.settle(200)
        r = app.json("""({ palSeen: window.__palSeen, w: window.__w, ops: window.__card.chart.canvas.getContext('2d').ops.length,
                          sum: window.__card.querySelector('.chart-summary').textContent, legend: window.__card.querySelector('.chart-legend').textContent,
                          note: window.__card.querySelector('.chart-note').textContent, title: window.__card.querySelector('h2').textContent,
                          mounted: TQ.charts.mountedCount(), tab: window.__card.chart.canvas.getAttribute('tabindex') })""")
        self.assertEqual(r["palSeen"], "#2457c5")
        self.assertGreater(r["w"], 100)                  # the spec function receives the real chart width
        self.assertGreater(r["ops"], 20)
        self.assertIn("時間序列", r["sum"])
        self.assertIn("A", r["legend"])
        self.assertEqual((r["note"], r["title"]), ("註", "圖"))
        self.assertEqual(r["mounted"], 1)
        self.assertEqual(r["tab"], "0")           # hover/keyboard capable chart is focusable
        app.eval("window.__card.setSpec({ dates: [], n: 0, panels: [] }); 0")
        app.settle(100)
        self.assertIn("沒有足夠", app.json("window.__card.querySelector('.chart-summary').textContent"))
        app.eval("window.__card.destroy(); 0")
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 0)
        self.assertEqual(app.errors(), [])

    def test_unknown_kind_and_dark_palette(self):
        app = core_app(dark=True)
        app.eval("TQ.theme.init(); window.__c = TQ.chartCard(null, {}, { kind: 'nope', bare: true, height: 120 }); document.body.appendChild(window.__c); 0")
        app.settle(100)
        self.assertIn("無法繪製", app.json("window.__c.querySelector('.chart-summary').textContent"))
        self.assertEqual(app.json("TQ.charts.currentMode()"), "dark")


if __name__ == "__main__":
    unittest.main()
