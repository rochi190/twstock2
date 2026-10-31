"""Layout of 歷史驗證 and 方法說明 with a real layout engine (WeasyPrint), at phone / tablet / desktop widths, light and dark.

Checks geometry only (WeasyPrint is not a browser): no horizontal page scroll, buttons are not collapsed / clipped, equations fit on desktop,
charts have the width of their card, every card has a visible title.  Set TQ_FRONT_OUT=/some/dir to keep full-page PNGs for a visual review."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import front_render as R
from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, default_rules, out_dir

NEEDS = HAVE_RACER and R.HAVE_RENDER
SKIP = SKIP_MSG + " (needs weasyprint, pymupdf and pillow as well)"
TALL = 60000


def boxes(doc, tag=None, cls=None):
    """All laid-out boxes (optionally filtered by element tag / class)."""
    found = []

    def walk(b):
        el = getattr(b, "element", None)
        if el is not None and (tag is None or getattr(b, "element_tag", None) == tag):
            classes = (el.get("class") or "").split()
            if cls is None or cls in classes:
                found.append(b)
        for ch in getattr(b, "children", []) or []:
            walk(ch)

    for page in doc.pages:
        walk(page._page_box)
    return found


def count_elements(doc, tag=None, cls=None):
    """Number of distinct DOM elements that produced a box (one element can yield several nested boxes)."""
    return len({id(b.element) for b in boxes(doc, tag, cls)})


def collapsed_buttons(doc, min_height=28.0):
    """.btn buttons whose box is shorter than a line of text plus padding (an element produces several nested boxes: the tallest one counts)."""
    tallest = {}
    for b in boxes(doc, "button"):
        key = id(b.element)
        if key not in tallest or b.border_height() > tallest[key].border_height():
            tallest[key] = b
    bad = []
    for b in tallest.values():
        classes = (b.element.get("class") or "").split()
        if ("btn" in classes or "collapse-btn" in classes) and b.border_height() < min_height:
            bad.append(f"<button class='{' '.join(classes)}'> is {b.border_height():.0f}px high")
    return bad


class Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="tq_layout_bt_")          # PNG / PDF scratch files live (and die) here unless TQ_FRONT_OUT is set
        cls.addClassCleanup(cls._tmp.cleanup)

    def render(self, name, hash_, width, storage=None, dark=False, rules=None, settle=6000, actions=None):
        app = FrontApp(hash=hash_, viewport=width, storage=storage, dark=dark, boot=False, real=True)
        app.set_rules((rules or []) + default_rules(True))
        app.boot()
        app.settle(settle)
        if actions:
            actions(app)
        self.assertEqual(app.errors(), [])
        d = out_dir()
        target = Path(d) if d is not None else Path(self._tmp.name)
        doc = R.layout(app, target, name, width=width, height=TALL, dark=dark)
        bad = R.overflow_report(doc, width)
        self.assertEqual(bad, [], f"{name}: content wider than the {width}px viewport")
        self.assertEqual(collapsed_buttons(doc), [], f"{name}: collapsed / clipped buttons")
        if d is not None:
            R.render(app, Path(d), name, width=width, height=TALL, dark=dark)
        return app, doc


@unittest.skipUnless(NEEDS, SKIP)
class BacktestLayout(Layout):
    def test_idle_form_phone_and_desktop(self):
        for width in (360, 1280):
            _app, doc = self.render(f"bt_idle_{width}", "#/backtest/2330", width, settle=800)
            self.assertEqual(count_elements(doc, "button", "bt-run"), 1)
            runs = sorted(boxes(doc, "button", "bt-run"), key=lambda b: -b.border_height())
            self.assertGreaterEqual(runs[0].border_height(), 36)
            if width == 360:
                self.assertGreaterEqual(runs[0].border_width(), 360 - 2 * 12 - 2 * 14 - 2, "the run button spans the card on phones")
            self.assertEqual(count_elements(doc, "div", "field"), 6)

    def test_result_at_360_768_1280(self):
        for width in (360, 768, 1280):
            app, doc = self.render(f"bt_result_{width}", "#/backtest/2330?h=1", width)
            self.assertEqual(app.count("canvas"), 4)
            imgs = boxes(doc, "img", "chart-canvas")
            self.assertEqual(count_elements(doc, "img", "chart-canvas"), 4)
            for im in imgs:
                self.assertGreater(im.width, 200)
                self.assertLessEqual(im.border_width(), width, "a chart wider than the viewport")
            titles = boxes(doc, "h2", "card-title")
            self.assertGreaterEqual(count_elements(doc, "h2", "card-title"), 12)
            for t in titles:
                self.assertGreater(t.height, 10)

    def test_result_dark_phone_and_nulls(self):
        self.render("bt_result_dark_390", "#/backtest/2330?h=1", 390, dark=True)
        nulls = [{"re": r"^/api/backtest\?", "resp": {"status": 200, "body": "backtest_nulls"}}]
        self.render("bt_nulls_390", "#/backtest/2330?h=1", 390, rules=nulls)

    def test_running_and_error_states(self):
        rules = [{"re": r"^/api/backtest\?", "resp": {"status": 200, "body": {"status": "computing", "progress": {"done": 400, "total": 1250, "message": "前推式預測 400/1250"}}}}]
        self.render("bt_running_390", "#/backtest/2330?h=1", 390, rules=rules, settle=1800)
        rules = [{"re": r"^/api/backtest\?", "resp": {"status": 404, "body": {"status": "error", "error": {"code": "unknown_symbol", "message": "找不到代號 9999。"}}}}]
        self.render("bt_error_390", "#/backtest/9999?h=1", 390, rules=rules, settle=1500)

    def test_weak_edge_with_trades(self):
        rules = [{"re": r"^/api/backtest\?", "resp": {"status": 200, "body": "backtest_2330_h5"}}]
        self.render("bt_trades_1280", "#/backtest/2330?h=5", 1280, rules=rules)
        self.render("bt_trades_390", "#/backtest/2330?h=5", 390, rules=rules, dark=True)


@unittest.skipUnless(NEEDS, SKIP)
class MethodLayout(Layout):
    def test_method_page_at_all_widths(self):
        for width, dark in ((360, False), (768, False), (1280, False), (390, True)):
            _app, doc = self.render(f"method_{width}{'_dark' if dark else ''}", "#/method", width, dark=dark, settle=400)
            eqs = boxes(doc, "figure", "eq-block")
            self.assertGreaterEqual(count_elements(doc, "figure", "eq-block"), 25)
            for e in eqs:
                self.assertLessEqual(e.border_width(), width - 16, "an equation block wider than the page")

    @staticmethod
    def wide_equations(doc):
        """(content width, panel width) of every display-equation line that is wider than its panel, i.e. would need horizontal scrolling."""
        wide = []
        for eq in boxes(doc, "div", "eq"):
            for line in getattr(eq, "children", []) or []:
                for piece in getattr(line, "children", []) or []:
                    w = getattr(piece, "width", 0) or 0
                    if w > eq.width + 1:
                        wide.append((round(w), round(eq.width)))
        return wide

    def test_equations_fit_without_scrolling_on_desktop(self):
        _app, doc = self.render("method_eqfit_1280", "#/method", 1280, settle=400)
        self.assertGreater(count_elements(doc, "div", "eq"), 35)
        self.assertEqual(self.wide_equations(doc), [], "equations wider than their panel at 1280px would force horizontal scrolling")

    def test_the_equation_width_check_is_sensitive(self):
        """Forcing no-wrap on a phone must make the same check find wide equations (guards against a check that can never fail)."""
        app = FrontApp(hash="#/method", viewport=390, boot=True)
        app.settle(400)
        target = Path(self._tmp.name)
        doc = R.layout(app, target, "eq_nowrap_390", width=390, height=TALL, extra_css=".eq-in { white-space: nowrap !important; }")
        self.assertGreater(len(self.wide_equations(doc)), 5)
        doc = R.layout(app, target, "eq_wrap_390", width=390, height=TALL)
        self.assertLessEqual(len(self.wide_equations(doc)), 3, "on phones nearly every equation wraps instead of scrolling")

    def test_toc_and_sections_exist_with_heights(self):
        _app, doc = self.render("method_struct_1280", "#/method", 1280, settle=400)
        self.assertEqual(count_elements(doc, "section", "m-sec"), 18)
        self.assertEqual(count_elements(doc, "a", "m-toc-link"), 18)
        self.assertEqual(count_elements(doc, "li", "m-flow-step"), 8)

    def test_deep_link_page_renders_the_same(self):
        self.render("method_deeplink_390", "#/method/garch", 390, settle=400)


if __name__ == "__main__":
    unittest.main()
