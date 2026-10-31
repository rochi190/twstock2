"""Every CSS class the JS applies at runtime must be styled (or be a documented hook-only class), and every selector in style.css
should be reachable.  Exercises the stock view, the picks view (incl. the universe panel), search, tips, errors and chart mounting in
ONE page so the fake DOM's class log (window.__classLog) covers them all."""
from __future__ import annotations

import re
import unittest

from front_helpers import FRONT, HAVE_RACER, SKIP_MSG, FrontApp, default_rules

# classes that are only structural hooks for tests / future styling: they intentionally have no rule of their own
HOOK_ONLY = {
    "view-host", "view-stock", "view-picks", "slot-head", "slot-warn", "slot-chart", "slot-fc", "slot-bt", "slot-notes", "slot-picks",
    "hz-row", "chart-summary", "bt-card", "notes-card", "buys-card", "all-card", "avoid-card", "pf-card", "uv-card", "progress-card",
    "chart-card", "hz-card", "hz-up", "hz-down", "hz-flat", "pick-flat", "row-3", "sh-chg-val", "stub", "view-root", "uv-content",
    "sl-name", "kpi-up", "tbl-empty", "market-strip", "uv-kpis", "al-left", "gauge-canvas", "strip-canvas", "th-sortable", "stack-seg",
    "verdict-no_edge", "verdict-weak_edge", "verdict-moderate_edge", "verdict-strong_edge", "verdict-none", "chip-flat",
    "w-key", "pick-up", "pick-down", "hbar-accent", "kpi-neutral", "bt-col", "sh-asof",
    # 歷史驗證 view (module D3): structural hooks without a rule of their own (its full class audit is in test_front_qa_static.py)
    "view-backtest", "slot-form", "bt-form-card", "bt-f-h", "bt-f-thr", "bt-f-disc", "bt-f-years", "bt-cur-name", "bt-reset",
    # 今日模型觀察 (picks) after the evidence / relative-strength rewrite: structural containers and column hooks; the visible parts have rules
    # (.ev-body, .lead-grid, .lead-item ...).  `bt-open` is the stock page's link button (styled through .btn / .btn-primary).
    "ev-host", "ev-run", "ev-open", "lead-card", "lead-leaders", "lead-laggards", "slot-lists", "bt-open",
}


def css_classes() -> set[str]:
    out = set()
    for f in ("style.css", "style_views.css"):
        text = (FRONT / f).read_text(encoding="utf-8")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        for sel in re.findall(r"([^{}@]+)\{", text):
            out.update(re.findall(r"\.([A-Za-z_][\w-]*)", sel))
    return out


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ClassCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FrontApp(hash="#/stock/2330", boot=False)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(6000)
        # info tips, search popup, table sorting, show-all
        app.click(".itip-btn")
        app.eval("document.querySelector('#search-host input').focus()")
        app.settle(300)
        app.click(".table-more button")
        app.click("#theme-toggle")
        # picks: default week (empty buys), then a month (cards, portfolio), universe panel with a result
        app.goto("#/picks", 1500)
        app.click(".page-head .seg-btn", 2)
        app.settle(1500)
        app.click(".all-card thead .th-btn", 2)
        app.click(".uv-card .collapse-btn")
        app.click(".uv-card .btn-primary")
        app.settle(1500)
        # error + computing states
        app.prepend_rule({"re": r"^/api/recommendations\?h=1&", "resp": {"status": 500, "body": {"status": "error", "error": {"code": "internal", "message": "x"}}}})
        app.click(".page-head .seg-btn", 0)
        app.settle(500)
        app.goto("#/stock/1234", 1500)
        app.goto("#/backtest/2330", 100)
        cls.used = set(app.classes_used())
        cls.app = app

    def test_every_class_used_by_js_is_styled(self):
        styled = css_classes() | HOOK_ONLY
        missing = sorted(c for c in self.used if c not in styled)
        self.assertEqual(missing, [], "classes applied by JS but not defined in style.css / style_views.css (add a rule or list them as hook-only)")


if __name__ == "__main__":
    unittest.main()
