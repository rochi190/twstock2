"""Layout checks with a real layout engine (WeasyPrint) -- the best we can do without a browser here.

The fake-DOM page is serialised, canvases become images (drawn by canvas_replay at the width their box really gets), and the laid-out
box tree is inspected: nothing may stick out of the viewport's right edge (no horizontal page scroll) at phone / tablet / desktop widths.
Set TQ_FRONT_OUT=/some/dir to also keep full-page PNGs (CJK text is drawn through a TrueType subset of Noto Sans TC, see front_render.py).
WeasyPrint is not a browser: it is used for geometry, not for pixel-exact styling.
"""
from __future__ import annotations

import unittest

import front_render as R
from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, default_rules, out_dir

NEEDS = HAVE_RACER and R.HAVE_RENDER
SKIP = SKIP_MSG + " (needs weasyprint, pymupdf and pillow as well)"

SCENARIOS = [
    ("stock_360", "#/stock/2330", 360, None, False),
    ("picks_360", "#/picks", 360, {"tq.picksH": "20"}, True),
    ("stock_1280", "#/stock/2330", 1280, None, False),
    ("picks_768", "#/picks", 768, {"tq.picksH": "20"}, True),
]


@unittest.skipUnless(NEEDS, SKIP)
class NoHorizontalScroll(unittest.TestCase):
    def run_scenario(self, name, hash_, width, storage, universe):
        app = FrontApp(hash=hash_, viewport=width, storage=storage, boot=False)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(6000)
        if universe:
            app.click(".uv-card .collapse-btn")
            app.click(".uv-card .btn-primary")
            app.settle(2500)
        self.assertEqual(app.errors(), [])
        d = out_dir()
        with __import__("tempfile").TemporaryDirectory(prefix="tq_layout_") as tmp:         # removed afterwards unless TQ_FRONT_OUT keeps the PNGs
            target = d if d is not None else tmp
            doc = R.layout(app, __import__("pathlib").Path(target), name, width=width)
            bad = R.overflow_report(doc, width)
            self.assertEqual(bad, [], f"{name}: content wider than the {width}px viewport")
            if d is not None:
                R.render(app, d, name, width=width)
        return app, doc

    def test_stock_phone(self):
        self.run_scenario(*SCENARIOS[0])

    def test_picks_phone_with_universe_panel_open(self):
        self.run_scenario(*SCENARIOS[1])

    def test_stock_desktop(self):
        self.run_scenario(*SCENARIOS[2])

    def test_picks_tablet_with_universe_panel_open(self):
        self.run_scenario(*SCENARIOS[3])


@unittest.skipUnless(NEEDS, SKIP)
class MediaFlattening(unittest.TestCase):
    def test_flatten_media_keeps_only_matching_width_blocks(self):
        css = "a{x:1} @media (min-width: 760px){ b{x:2} } @media (max-width: 560px){ c{x:3} } @media (prefers-color-scheme: dark){ d{x:4} } e{x:5}"
        phone = R.flatten_media(css, 390)
        self.assertIn("c{x:3}", phone)
        self.assertNotIn("b{x:2}", phone)
        self.assertNotIn("d{x:4}", phone)
        desk = R.flatten_media(css, 1280)
        self.assertIn("b{x:2}", desk)
        self.assertNotIn("c{x:3}", desk)
        self.assertIn("e{x:5}", desk)


if __name__ == "__main__":
    unittest.main()
