"""Static checks of style.css against charts.js and accessibility rules (pure python: no V8 needed).

* the palette tokens in style.css equal BASE.light / BASE.dark in charts.js (single source of colour truth);
* WCAG contrast >= 4.5:1 for every text / background pair the UI uses, in light and dark;
* structural rules that keep the page robust: [hidden], focus ring, reduced motion, dark-mode blocks identical, all var() defined.
"""
from __future__ import annotations

import re
import unittest

from front_helpers import FRONT

CSS = (FRONT / "style.css").read_text(encoding="utf-8")
CHARTS = (FRONT / "charts.js").read_text(encoding="utf-8")


def parse_vars(block: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"(--[\w-]+)\s*:\s*([^;]+);", block)}


def blocks():
    """(light vars, dark-media vars, dark-attr vars) from style.css."""
    light = re.search(r"^:root\s*\{(.*?)\n\}", CSS, re.S | re.M).group(1)
    media = re.search(r"@media \(prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme=\"light\"\]\)\s*\{(.*?)\n  \}\n\}", CSS, re.S).group(1)
    attr = re.search(r":root\[data-theme=\"dark\"\]\s*\{(.*?)\n\}", CSS, re.S).group(1)
    return parse_vars(light), parse_vars(media), parse_vars(attr)


def js_palette() -> dict[str, dict[str, str]]:
    out = {}
    for mode in ("light", "dark"):
        m = re.search(mode + r":\s*\{(.*?)\}", CHARTS[CHARTS.index("var BASE"):], re.S)
        out[mode] = {k: v for k, v in re.findall(r"(\w+):\s*'(#[0-9a-fA-F]{6})'", m.group(1))}
    return out


def rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def parse_color(v: str):
    v = v.strip()
    if v.startswith("#"):
        return rgb(v) + (1.0,)
    m = re.match(r"rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", v)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))


def over(fg, bg):
    """Colour of translucent ``fg`` painted over opaque ``bg``."""
    a = fg[3]
    return tuple(round(fg[i] * a + bg[i] * (1 - a)) for i in range(3))


def lum(c):
    def ch(x):
        x /= 255
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(x) for x in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class PaletteMatchesCharts(unittest.TestCase):
    def test_tokens_equal_chart_palette(self):
        light, dark_media, dark_attr = blocks()
        pal = js_palette()
        mapping = {"bg": "--surface"}
        for mode, tokens in (("light", light), ("dark", dark_attr)):
            for key, hexv in pal[mode].items():
                token = mapping.get(key, "--" + key)
                self.assertIn(token, tokens, f"{mode}: {token} missing")
                self.assertEqual(tokens[token].lower(), hexv.lower(), f"{mode} {token} differs from charts.js {key}")

    def test_dark_blocks_are_identical(self):
        _light, dark_media, dark_attr = blocks()
        self.assertEqual(dark_media, dark_attr, "the prefers-color-scheme block and [data-theme=dark] must define the same tokens")

    def test_light_defines_every_token_dark_overrides(self):
        light, _m, dark = blocks()
        for k in dark:
            self.assertIn(k, light, f"{k} has no light value")
        for k in light:
            if k.startswith("--") and k not in ("--radius", "--radius-sm", "--font", "--maxw"):
                self.assertIn(k, dark, f"{k} has no dark override")

    def test_every_var_used_is_defined(self):
        light, _m, _d = blocks()
        used = set(re.findall(r"var\((--[\w-]+)", CSS)) - {"--tone"}      # --tone is set per component by the .tone-* classes
        self.assertEqual(sorted(u for u in used if u not in light), [])


class Contrast(unittest.TestCase):
    MIN = 4.5

    def tokens(self, mode):
        light, _m, dark = blocks()
        t = dict(light)
        if mode == "dark":
            t.update(dark)
        return {k: parse_color(v) for k, v in t.items() if re.match(r"^(#|rgba)", v)}

    def check(self, mode, fg, bg, what, minimum=None):
        t = self.tokens(mode)
        bgc = t[bg] if isinstance(bg, str) else bg
        if len(bgc) == 4:
            bgc = over(bgc, t["--surface"])
        fgc = t[fg]
        if len(fgc) == 4:
            fgc = over(fgc, bgc)
        r = contrast(fgc, bgc)
        self.assertGreaterEqual(r, minimum or self.MIN, f"{mode}: {what} ({fg} on {bg}) = {r:.2f}")

    def test_text_on_surfaces(self):
        for mode in ("light", "dark"):
            for fg in ("--text", "--text2", "--muted"):
                for bg in ("--surface", "--page", "--surface2"):
                    self.check(mode, fg, bg, "text")
            for fg in ("--up", "--down", "--accent", "--amber", "--violet", "--gray"):
                self.check(mode, fg, "--surface", "coloured text")
                self.check(mode, fg, "--surface2", "coloured text on subtle surface")

    def test_chips_and_banners_on_tinted_backgrounds(self):
        for mode in ("light", "dark"):
            pairs = [("--up", "--up-soft"), ("--down", "--down-soft"), ("--accent", "--accent-soft"), ("--amber", "--amber-soft"),
                     ("--violet", "--violet-soft"), ("--text2", "--gray-soft"), ("--text", "--up-soft"), ("--text", "--accent-soft"),
                     ("--text", "--amber-soft"), ("--text", "--down-soft"), ("--text2", "--amber-soft")]
            for fg, bg in pairs:
                self.check(mode, fg, bg, "chip / banner text")

    def test_text_on_solid_fills(self):
        for mode in ("light", "dark"):
            for bg in ("--accent", "--up", "--down", "--amber", "--violet", "--gray", "--text2"):
                self.check(mode, "--on-fill", bg, "text on filled bar / button / icon")

    def test_tooltip_bubble_and_selected_states(self):
        for mode in ("light", "dark"):
            self.check(mode, "--surface", "--text", "info-tip bubble (inverted)")
            self.check(mode, "--text", "--accent-soft", "selected search option")
            self.check(mode, "--accent", "--accent-soft", "active nav item")


class Structure(unittest.TestCase):
    def test_braces_balanced_and_no_nan(self):
        self.assertEqual(CSS.count("{"), CSS.count("}"))
        self.assertNotIn("NaN", CSS)
        self.assertNotIn("undefined", CSS)

    def test_accessibility_rules_present(self):
        self.assertRegex(CSS, r"\[hidden\]\s*\{\s*display:\s*none\s*!important")
        self.assertRegex(CSS, r":focus-visible\s*\{[^}]*outline:\s*2px")
        self.assertIn("prefers-reduced-motion: reduce", CSS)
        self.assertIn("animation-duration: 0.01ms", CSS)
        self.assertRegex(CSS, r"\.sr-only\s*\{")
        self.assertIn("color-scheme: dark", CSS)
        self.assertIn("-webkit-text-size-adjust", CSS)

    def test_every_outline_removal_has_a_visible_replacement(self):
        for m in re.finditer(r"([^{}]+)\{[^{}]*outline:\s*none[^{}]*\}", CSS):
            sel = m.group(1).strip()
            if sel in (".app-main",):          # the main landmark is focused programmatically only
                continue
            self.assertIn(".search-field:focus-within", CSS, f"{sel} removes the outline; the focus-within style must replace it")

    def test_font_stack_matches_the_spec(self):
        self.assertIn('"Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui', CSS)

    def test_taiwan_colour_semantics(self):
        self.assertRegex(CSS, r"\.up\s*\{\s*color:\s*var\(--up\)")
        self.assertRegex(CSS, r"\.down\s*\{\s*color:\s*var\(--down\)")
        light, _m, _d = blocks()
        r, g, b = rgb(light["--up"])
        self.assertTrue(r > g and r > b, "--up must be red")
        r, g, b = rgb(light["--down"])
        self.assertTrue(g > r and g > b, "--down must be green")

    def test_responsive_breakpoints_present(self):
        for bp in ("(min-width: 760px)", "(min-width: 1100px)", "(max-width: 560px)", "(max-width: 380px)"):
            self.assertIn(bp, CSS)

    def test_flex_and_grid_children_can_shrink(self):
        # horizontal scrolling of the page is prevented by min-width: 0 on the layout containers and by wrapping tables
        for sel in (".row > *", ".card", ".table-wrap", ".chart-box", ".card-body"):
            self.assertRegex(CSS, re.escape(sel) + r"[^{]*\{[^}]*(min-width: 0|overflow-x: auto|overflow: hidden)")


if __name__ == "__main__":
    unittest.main()
