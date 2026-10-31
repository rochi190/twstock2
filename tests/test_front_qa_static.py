"""Static QA of the whole frontend (all four views): syntax, load order, no external resources, every TQ member that is read exists, every CSS class
the JS applies is styled, CSS tokens only, colour semantics, contrast of the colour pairs used by the backtest / method views."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

from front_helpers import FRONT, HAVE_RACER, SKIP_MSG, FrontApp, default_rules, fixture, script_files_from_html
from test_front_shell_classes import HOOK_ONLY as SHELL_HOOK_ONLY
from test_front_shell_classes import css_classes
from test_front_shell_static import Contrast, blocks, contrast, over, parse_color

JS_FILES = sorted(p.name for p in FRONT.glob("*.js"))
ORDER = ["format.js", "charts.js", "core.js", "view_stock.js", "view_picks.js", "view_backtest.js", "view_method.js", "app.js"]

# classes the backtest / method views apply only as structural hooks (tests, future styling): intentionally without a rule of their own
HOOK_ONLY = SHELL_HOOK_ONLY | {
    "view-backtest", "view-method", "slot-form", "slot-notice", "slot-status", "slot-result", "bt-form-card", "bt-idle-card", "bt-error-card",
    "bt-progress-card", "bt-head-card", "bt-verdict-card", "bt-quality-card", "bt-strategy-card", "bt-returns-card", "bt-bucket-card", "bt-params-card",
    "bt-f-h", "bt-f-thr", "bt-f-disc", "bt-f-years", "bt-copy", "bt-cur-name", "bt-cur-code", "kpi-neutral", "m-sec-intro", "m-sec-pipeline", "m-sec-data",
    "m-sec-targets", "m-sec-garch", "m-sec-hmm", "m-sec-dlm", "m-sec-naive", "m-sec-dma", "m-sec-calibration", "m-sec-limit", "m-sec-fhs", "m-sec-recommender",
    "m-sec-walkforward", "m-sec-metrics", "m-sec-costs", "m-sec-limits", "m-sec-disclaimer", "ss", "sp", "sb", "fn", "fd", "al-center", "al-right", "m",
    "notes-card", "verdict-no_edge", "verdict-weak_edge", "verdict-moderate_edge", "verdict-strong_edge", "bt-card", "bt-reset", "m-olist",
}


def read(name):
    return (FRONT / name).read_text(encoding="utf-8")


def strip_js_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"(^|[^:\\'\"`])//[^\n]*", r"\1", src)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Syntax(unittest.TestCase):
    def test_every_js_file_compiles_in_v8_as_a_plain_script(self):
        from py_mini_racer import MiniRacer
        for name in JS_FILES:
            ctx = MiniRacer()
            ctx.eval("new Function(" + json.dumps(read(name)) + ")")              # SyntaxError otherwise

    def test_no_syntax_newer_than_es2020(self):
        patterns = {r"\?\?=": "logical nullish assignment", r"\|\|=": "logical or assignment", r"&&=": "logical and assignment", r"\.at\(": "Array.prototype.at",
                    r"Object\.hasOwn\(": "Object.hasOwn", r"\.replaceAll\(": "replaceAll", r"structuredClone": "structuredClone", r"\bstatic\s*\{": "class static block",
                    r"#[a-zA-Z_]\w*\s*[=;(]": "private class member", r"\bawait\b\s+import": "top-level await", r"\.findLast\(": "findLast", r"\.toSorted\(": "toSorted"}
        for name in JS_FILES:
            src = strip_js_comments(read(name))
            src_no_strings = re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", src)          # skip text inside quotes (e.g. "#/stock")
            for pat, what in patterns.items():
                self.assertIsNone(re.search(pat, src_no_strings), f"{name}: {what} needs a newer engine than ES2020")

    def test_scripts_are_plain_iife_with_strict_mode_and_no_globals_leak(self):
        for name in JS_FILES:
            src = read(name)
            self.assertIn("'use strict'", src, name)
        app = FrontApp(fixtures=[], boot=False)
        before = set(app.json("Object.getOwnPropertyNames(window)"))
        app.boot()
        app.settle(300)
        after = set(app.json("Object.getOwnPropertyNames(window)"))
        leaked = sorted(k for k in after - before if not k.startswith("__"))
        self.assertEqual(leaked, ["TQ"], "frontend scripts must not leak globals other than window.TQ")


class IndexHtml(unittest.TestCase):
    def test_scripts_exist_load_in_contract_order_and_are_deferred(self):
        self.assertEqual(script_files_from_html(), ORDER)
        html = read("index.html")
        for f in ORDER:
            self.assertTrue((FRONT / f).exists(), f)
            self.assertRegex(html, rf'<script src="{re.escape(f)}" defer></script>')
        self.assertEqual(sorted(JS_FILES), sorted(ORDER), "a frontend script that index.html does not load (or the reverse)")
        for css in re.findall(r'<link rel="stylesheet" href="([^"]+)"', html):
            self.assertTrue((FRONT / css).exists(), css)
        self.assertEqual(re.findall(r'<link rel="stylesheet" href="([^"]+)"', html), ["style.css", "style_views.css"])

    def test_ids_the_shell_looks_up_exist(self):
        html = read("index.html")
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        for js in ("core.js", "app.js"):
            for ref in re.findall(r"getElementById\('([^']+)'\)", read(js)):
                self.assertIn(ref, ids, f"{js} looks up #{ref} which index.html does not define")
        for ref in ("view", "nav", "status", "main", "search-host", "theme-toggle", "theme-icon", "theme-label"):
            self.assertIn(ref, ids)

    def test_main_nav_routes_match_the_registered_views(self):
        html = read("index.html")
        routes = re.findall(r'data-route="(\w+)"', html)
        self.assertEqual(routes, ["stock", "picks", "backtest", "method"])
        for r in routes:
            self.assertRegex(read(f"view_{ 'stock' if r == 'stock' else r }.js"), rf"TQ\.views\.{r} = ")


class NoExternalResources(unittest.TestCase):
    ALLOWED = ("http://www.w3.org/2000/svg",)

    def test_no_url_anywhere_in_the_frontend(self):
        for path in sorted(FRONT.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"(?:https?:)?//[A-Za-z0-9][\w.-]*\.[A-Za-z]{2,}[^\s\"'<>)]*|https?://[^\s\"'<>)]+", text):
                url = m.group(0)
                if any(url.startswith(a) for a in self.ALLOWED):
                    continue
                if m.start() > 0 and text[m.start() - 1] == ":" and not url.startswith("http"):
                    continue
                self.fail(f"{path.name}: external reference {url!r}")
            self.assertNotRegex(text, r"@import\s+url\(\s*['\"]?https?:", path.name)
            self.assertNotRegex(text, r"""\bsrc=["']//""", path.name)
            self.assertNotRegex(text, r"""fetch\(\s*['"`]https?:""", path.name)
            self.assertNotRegex(text, r"\bnew\s+(?:WebSocket|EventSource)\b", path.name)
            self.assertNotRegex(text, r"""\b(?:importScripts|XMLHttpRequest)\b""", path.name)

    def test_no_cdn_font_or_script_hosts_named_in_text(self):
        for path in FRONT.glob("*"):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for host in ("googleapis", "gstatic", "cdnjs", "jsdelivr", "unpkg", "bootstrapcdn", "fontawesome", "cloudflare"):
                self.assertNotIn(host, text, f"{path.name} mentions {host}")

    def test_only_one_inline_script_and_it_is_the_theme_bootstrap(self):
        inline = re.findall(r"<script>(.*?)</script>", read("index.html"), re.S)
        self.assertEqual(len(inline), 1)
        self.assertIn("data-theme", inline[0])
        self.assertNotRegex(read("index.html"), r"\son[a-z]+\s*=\s*[\"']")           # no inline event handlers (CSP friendly)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class TqMembers(unittest.TestCase):
    """Every `TQ.a.b` / alias member (F.fmtX, ui.kpi, U.clamp ...) a script READS must exist at runtime."""

    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(hash="#/method")
        cls.app.settle(100)

    def reads(self):
        out = {}
        for name in JS_FILES:
            src = strip_js_comments(read(name))
            src = re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", src)
            refs = set()
            assigned = set(m.group(1) + "." + m.group(2) for m in re.finditer(r"\bTQ\.(\w+)\.(\w+)\s*=(?!=)", src))
            for m in re.finditer(r"\bTQ\.(\w+)\.(\w+)", src):
                key = m.group(1) + "." + m.group(2)
                if key in assigned:
                    continue
                refs.add(key)
            aliases = {}
            for m in re.finditer(r"\b(\w+)\s*=\s*TQ\.(fmt|ui|util|charts|api|router|prefs|theme|searchBox)\b(?![.\[(])", src):
                aliases[m.group(1)] = m.group(2)
            for alias, ns in aliases.items():
                for m in re.finditer(r"(?<![\w.])" + re.escape(alias) + r"\.(\w+)", src):
                    refs.add(ns + "." + m.group(1))
            top = set(m.group(1) for m in re.finditer(r"\bTQ\.(\w+)\b(?!\s*[.=(\[]?\s*=)", src))
            out[name] = (refs, top)
        return out

    def test_all_read_members_exist(self):
        missing = []
        for name, (refs, top) in self.reads().items():
            for ref in sorted(refs):
                ns, member = ref.split(".")
                ok = self.app.json(f"typeof TQ[{json.dumps(ns)}] !== 'undefined' && typeof TQ[{json.dumps(ns)}][{json.dumps(member)}] !== 'undefined'")
                if not ok:
                    missing.append(f"{name}: TQ.{ref}")
            for t in sorted(top):
                if not self.app.json(f"typeof TQ[{json.dumps(t)}] !== 'undefined'"):
                    missing.append(f"{name}: TQ.{t}")
        self.assertEqual(missing, [])

    def test_the_regex_actually_finds_things(self):
        r = self.reads()
        self.assertIn("ui.kpi", r["view_backtest.js"][0])
        self.assertIn("fmt.fmtPct", r["view_backtest.js"][0])
        self.assertIn("router.buildQuery", r["view_backtest.js"][0])
        self.assertIn("charts.mount", r["core.js"][0])
        self.assertGreater(sum(len(v[0]) for v in r.values()), 150)

    def test_every_view_has_the_documented_interface(self):
        for v in ("stock", "picks", "backtest", "method"):
            self.assertEqual(self.app.json(f"[typeof TQ.views.{v}.title, typeof TQ.views.{v}.mount]"), ["function", "function"], v)


class CssClasses(unittest.TestCase):
    CLASS_PROPS = re.compile(r"""class\s*:\s*(\[[^\]]*\]|'[^']*'|"[^"]*")""")

    def literal_classes(self):
        found = {}
        for name in JS_FILES:
            src = strip_js_comments(read(name))
            tokens = set()
            for m in self.CLASS_PROPS.finditer(src):
                body = m.group(1)
                # standalone literals only: "'a b'" or an array element "['a', cond ? 'x' : null, 'c']" -> a, c (not 'x': part of an expression)
                for lit in re.findall(r"(?:^|[\[,]\s*)(?:'([^']*)'|\"([^\"]*)\")(?=\s*(?:,|\]|$))", body):
                    for c in (lit[0] or lit[1]).split():
                        tokens.add(c)
            for m in re.finditer(r"(?:className\s*=|classList\.(?:add|remove|toggle|contains)\()\s*'([^']+)'", src):
                tokens.update(m.group(1).split())
            for m in re.finditer(r"\bel\('[a-z]+',\s*'([^']+)'\)", src):
                tokens.update(m.group(1).split())
            found[name] = {t for t in tokens if re.match(r"^[a-z][a-z0-9_-]*$", t) and not t.endswith("-")}
        return found

    def test_every_class_literal_in_the_js_is_styled_or_a_documented_hook(self):
        styled = css_classes() | HOOK_ONLY
        for name, tokens in self.literal_classes().items():
            missing = sorted(t for t in tokens if t not in styled)
            self.assertEqual(missing, [], f"{name}: classes without a CSS rule (add one or list as hook-only)")

    def test_the_extraction_finds_the_known_classes(self):
        found = self.literal_classes()
        self.assertIn("bt-run", found["view_backtest.js"])
        self.assertIn("eq-block", found["view_method.js"])
        self.assertIn("chart-wrap", found["charts.js"])
        self.assertIn("card-body", found["core.js"])

    def test_no_dead_rules_for_the_new_views(self):
        """Every .bt-* / .m-* / .tl-* / eq class in style_views.css is applied somewhere in the JS (a typo in either side shows up here)."""
        css = re.sub(r"/\*.*?\*/", "", read("style_views.css"), flags=re.S)
        used = set()
        for name in ("view_backtest.js", "view_method.js", "view_stock.js", "view_picks.js", "core.js"):
            used |= set(re.findall(r"[a-z][a-z0-9_-]*", read(name)))
        dynamic_ok = {"tl-known", "tl-future", "tl-blank", "tl-target", "m-note-info", "m-note-warn", "bt-warn", "dl", "ss", "sp", "sb", "fn", "fd", "m"}
        dead = []
        for cls in sorted(set(re.findall(r"\.([A-Za-z_][\w-]*)", css))):
            if cls not in used and cls not in dynamic_ok and cls not in {"seg", "seg-btn", "search", "search-field", "search-input", "search-pop", "field", "field-label", "field-hint",
                                                                       "card", "chip", "table-wrap", "tbl", "kv", "kpi", "kpi-row", "kpi-value", "kpi-label", "kpi-sub", "card-body",
                                                                       "card-title", "banner", "verdict", "verdict-title", "verdict-text", "chart-note", "btn", "empty", "progress",
                                                                       "progress-msg", "note-list", "disclaimer", "itip", "num", "row-total", "page-head", "slot", "table-more", "kv-row",
                                                                       "verdict-bullets", "search-list", "sub-title", "muted", "up", "down", "bad", "view"}:
                dead.append(cls)
        self.assertEqual(dead, [], "classes styled in style_views.css but never applied by any script")


class CssTokens(unittest.TestCase):
    CSS = read("style_views.css")

    def test_only_tokens_defined_in_style_css_are_used(self):
        light, _m, _d = blocks()
        used = set(re.findall(r"var\((--[\w-]+)", self.CSS)) - {"--tone"}
        self.assertEqual(sorted(u for u in used if u not in light), [])
        self.assertIn("--tone", set(re.findall(r"var\((--[\w-]+)", self.CSS)))        # set by .m-note-* here and .tone-* in style.css

    def test_no_hard_coded_colours_so_light_and_dark_always_follow_the_tokens(self):
        css = re.sub(r"/\*.*?\*/", "", self.CSS, flags=re.S)
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b", css), [])
        self.assertEqual(re.findall(r"\brgba?\(", css), [])
        self.assertEqual(re.findall(r"\bhsla?\(", css), [])
        self.assertNotRegex(css, r"\b(?:red|green|blue|orange|yellow)\b")

    def test_the_file_does_not_redefine_palette_tokens(self):
        css = re.sub(r"/\*.*?\*/", "", self.CSS, flags=re.S)
        defined = set(re.findall(r"(--[\w-]+)\s*:", css))
        self.assertEqual(defined, {"--tone"})

    def test_braces_balance_and_no_nan(self):
        self.assertEqual(self.CSS.count("{"), self.CSS.count("}"))
        self.assertNotIn("NaN", self.CSS)
        self.assertNotIn("undefined", self.CSS)

    def test_up_down_colours_are_not_used_for_evidence_strength_or_help_text(self):
        """Red = up, green = down (Taiwan): the verdict / backtest-strength / method styling must not borrow them (except the .bad authoring marker)."""
        css = re.sub(r"/\*.*?\*/", "", self.CSS, flags=re.S)
        for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            if re.search(r"var\(--(?:up|down)(?:-soft)?\)", body):
                self.assertTrue(".bad" in sel, f"red/green token used by {sel.strip()!r}")
        js = read("view_backtest.js")
        self.assertNotRegex(js, r"verdictBox\([^)]*dir")
        self.assertIn("var TONE_CHIP = { neutral: 'neutral', warn: 'warn', info: 'info', accent: 'accent' };", js)        # evidence strength maps to gray / amber / blue / violet chips


class ContrastOfNewPairs(Contrast):
    """Same WCAG machinery as test_front_shell_static; the pairs below are the ones view_backtest / view_method introduce."""

    def test_muted_and_text2_on_the_tinted_surfaces_of_the_new_views(self):
        for mode in ("light", "dark"):
            for fg, bg in (("--muted", "--gray-soft"), ("--text2", "--gray-soft"), ("--muted", "--accent-soft"), ("--text", "--gray-soft"), ("--text2", "--amber-soft"),
                           ("--accent", "--surface2"), ("--text2", "--surface2"), ("--muted", "--surface2"), ("--amber", "--surface"), ("--text", "--amber-soft")):
                self.check(mode, fg, bg, f"{fg} on {bg}")

    def test_equation_panel_colours(self):
        for mode in ("light", "dark"):
            self.check(mode, "--text", "--surface2", "equation text on the panel")
            self.check(mode, "--text2", "--surface2", "equation gloss on the panel")
            self.check(mode, "--accent", "--accent-soft", "the 白話 tag")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class RuntimeClassCoverage(unittest.TestCase):
    """The classes actually produced at runtime (including the dynamic ones: tone-*, verdict-*, chip-*, m-note-*, tl-*) all have CSS."""

    def test_runtime_classes(self):
        used = set()
        scenarios = [("#/backtest/2330?h=1", None), ("#/backtest/2330?h=5", {"re": r"^/api/backtest\?", "resp": {"status": 200, "body": "backtest_2330_h5"}}),
                     ("#/backtest/2330?h=1", {"re": r"^/api/backtest\?", "resp": {"status": 200, "body": "backtest_nulls"}}), ("#/backtest/2330", None), ("#/method", None)]
        levels = ["no_edge", "weak_edge", "moderate_edge", "strong_edge"]
        for level in levels:
            body = fixture("backtest_2330_h1.json")
            body["report"]["verdict"]["level"] = level
            scenarios.append(("#/backtest/2330?h=1", {"re": r"^/api/backtest\?", "resp": {"status": 200, "body": body}}))
        for hash_, rule in scenarios:
            app = FrontApp(hash=hash_, boot=False, real=False)
            app.set_rules(([rule] if rule else []) + default_rules(False))
            app.boot()
            app.settle(3500)
            app.click(".itip-btn")
            if app.count(".bt-run"):
                app.click(".bt-reset")
                app.click(".bt-f-h .seg-btn", 1)
            if app.count(".m-toc-link"):
                app.click(".m-toc-link", 3)
            used |= set(app.classes_used())
        styled = css_classes() | HOOK_ONLY
        missing = sorted(c for c in used if c not in styled)
        self.assertEqual(missing, [], "classes applied at runtime without a CSS rule")
        for expected in ("verdict-no_edge", "tone-neutral", "tone-warn", "tone-info", "tone-accent", "m-note-warn", "tl-known", "frac", "bt-run", "bt-kpis"):
            self.assertIn(expected, used)


if __name__ == "__main__":
    unittest.main()
