"""Layout renders without a browser: DOM (V8 fake) -> HTML -> WeasyPrint (PDF) -> PyMuPDF (PNG).

Used by tests/test_front_shell_layout.py and by hand (``python tests/front_render.py``).  Canvases are replaced by images
produced with tests/canvas_replay.py from the recorded Canvas2D operations.  Needs weasyprint, pymupdf and pillow (the scratch
venv has them); every function raises ``RuntimeError`` when they are missing so tests can skip.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

try:  # pragma: no cover - environment dependent
    import pymupdf  # type: ignore
    import weasyprint  # type: ignore
    from PIL import Image  # type: ignore
    import canvas_replay  # type: ignore
    HAVE_RENDER = True
except Exception:  # noqa: BLE001
    HAVE_RENDER = False

FONT_SOURCES = [
    str(Path.home() / ".local/share/fonts/NotoSansTC-Regular.otf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _collect_chars() -> str:
    chars = set(chr(c) for c in range(32, 127))
    for pat in ("frontend/*.js", "frontend/*.html", "frontend/*.css", "tests/fixtures/*.json", "tests/fixtures/real/*.json"):
        for f in ROOT.glob(pat):
            try:
                chars.update(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
    chars.update(" \u00a0\u3000")
    return "".join(sorted(c for c in chars if c.strip() or c in " \u00a0\u3000"))


def ensure_preview_font(cache_dir: Path):
    """MuPDF cannot rasterise the CFF-flavoured Noto Sans TC that WeasyPrint embeds, so previews would have no CJK ink.
    Build (once) a subset of it with TrueType outlines (glyf) for exactly the characters the app uses; returns its path
    or None when no source font / fontTools is available."""
    cache_dir = Path(cache_dir)
    chars = _collect_chars()
    # the subset depends on the characters the app uses: key the cache file by their hash so new UI text never renders as blanks
    out = cache_dir / ("NotoSansTC-preview-%s.ttf" % hashlib.md5(chars.encode("utf-8")).hexdigest()[:10])
    if out.exists():
        return out
    src = next((Path(p) for p in FONT_SOURCES if Path(p).exists() and p.endswith(".otf")), None)
    if src is None:
        return None
    try:
        from fontTools import subset
        from fontTools.pens.cu2quPen import Cu2QuPen
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        from fontTools.ttLib import TTFont, newTable
    except Exception:  # noqa: BLE001
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("NotoSansTC-preview*.ttf"):          # subsets for an older character set are obsolete
        try:
            old.unlink()
        except OSError:
            pass
    font = TTFont(str(src))
    opts = subset.Options()
    opts.layout_features = []
    opts.notdef_outline = True
    opts.name_IDs = ["*"]
    sub = subset.Subsetter(opts)
    sub.populate(text=chars)
    sub.subset(font)
    gs = font.getGlyphSet()
    glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(gs)
        try:
            gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
            glyphs[name] = pen.glyph()
        except Exception:  # noqa: BLE001
            glyphs[name] = TTGlyphPen(None).glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = font.getGlyphOrder()
    glyf.glyphs = glyphs
    for tag in ("CFF ", "VORG", "GSUB", "GPOS", "BASE", "DSIG", "vhea", "vmtx"):
        if tag in font:
            del font[tag]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, g in glyphs.items():
        adv, _lsb = hmtx.metrics[name]
        hmtx.metrics[name] = (adv, getattr(g, "xMin", 0))
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for attr in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                 "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    maxp.maxComponentDepth = 0
    maxp.maxPoints = maxp.maxContours = maxp.maxCompositePoints = maxp.maxCompositeContours = 0
    font["post"].formatType = 2.0
    font["post"].extraNames = []
    font["post"].mapping = {}
    font["post"].glyphOrder = font.getGlyphOrder()
    font["head"].glyphDataFormat = 0
    font["head"].indexToLocFormat = 0
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(str(out))
    return out


def flatten_media(css: str, width: int) -> str:
    """WeasyPrint does not evaluate (min|max)-width media queries: keep the rules of the blocks that match ``width`` (and drop
    prefers-* blocks: dark mode is driven by <html data-theme> in renders)."""
    out, i = [], 0
    while True:
        j = css.find("@media", i)
        if j < 0:
            out.append(css[i:])
            break
        out.append(css[i:j])
        k = css.index("{", j)
        cond = css[j + 6:k]
        depth, m = 1, k + 1
        while depth and m < len(css):
            depth += {"{": 1, "}": -1}.get(css[m], 0)
            m += 1
        inner = css[k + 1:m - 1]
        ok = "prefers-" not in cond and "print" not in cond
        for kind, val in re.findall(r"\(\s*(min|max)-width\s*:\s*(\d+)px\s*\)", cond):
            ok = ok and (width >= int(val) if kind == "min" else width <= int(val))
        if ok and re.search(r"\(\s*(min|max)-width", cond):
            out.append(inner)
        i = m
    return "".join(out)


LAYOUT_CSS = """
@page { size: %(W)dpx %(H)dpx; margin: 0; }
html, body { margin: 0; }
"""


def _css(width, height, dark, preview_font, out_dir, extra_css=''):
    from weasyprint.text.fonts import FontConfiguration
    fconf = FontConfiguration()
    font_css = ""
    pf = ensure_preview_font(Path(out_dir).parent / "fonts") if preview_font else None
    if pf is not None:
        font_css = ("@font-face { font-family: 'PreviewCJK'; src: url('file://%s'); }\n"
                    ":root { --font: 'PreviewCJK', sans-serif; }\nbody, button, input { font-family: 'PreviewCJK', sans-serif; }\n" % pf)
    css = [weasyprint.CSS(string=flatten_media((ROOT / "frontend" / "style.css").read_text(encoding="utf-8"), width), font_config=fconf),
           weasyprint.CSS(string=flatten_media((ROOT / "frontend" / "style_views.css").read_text(encoding="utf-8"), width), font_config=fconf),
           weasyprint.CSS(string=font_css + LAYOUT_CSS % {"W": width, "H": height} + extra_css, font_config=fconf)]
    return css, fconf


def _html(ser, name, dark, selector):
    body = ser["html"].replace('src="canvas_', f'src="{name}_canvas_')
    theme = ' data-theme="dark"' if dark else ""
    return (f'<!doctype html><html lang="zh-Hant"{theme}><head><meta charset="utf-8"></head>'
            f'{body if selector == "body" else "<body>" + body + "</body>"}</html>')


def _measure(doc):
    """{canvas id: content width of its .chart-box parent} from WeasyPrint's laid-out box tree."""
    out = {}

    def walk(box, parent):
        el = getattr(box, "element", None)
        if getattr(box, "element_tag", None) == "img" and el is not None and el.get("data-canvas") is not None and parent is not None:
            out[int(el.get("data-canvas"))] = float(parent.width)
        for ch in getattr(box, "children", []) or []:
            walk(ch, box)

    for page in doc.pages:
        walk(page._page_box, None)
    return out


def layout(app, out_dir: Path, name: str, width: int = 1280, height: int = 0, dark: bool = False, selector: str = "body",
           preview_font: bool = True, fit_canvases: bool = True, extra_css: str = ""):
    """Lay the current DOM out with WeasyPrint (canvases replaced by images, sized to their real boxes); returns the Document."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    height = height or (14000 if width >= 900 else 18000)
    css, fconf = _css(width, height, dark, preview_font, out_dir, extra_css)

    def write_canvases(ser):
        for cv in ser["canvases"]:
            rec = {"W": cv["W"], "H": cv["H"], "bg": "#151b24" if dark else "#ffffff", "ops": cv["ops"]}
            canvas_replay.replay(rec, 2).save(out_dir / f"{name}_canvas_{cv['id']}.png")

    ser = app.serialize(selector)
    doc = None
    for _ in range(3):
        write_canvases(ser)                    # images must exist BEFORE layout (WeasyPrint reads their size / data then)
        doc = weasyprint.HTML(string=_html(ser, name, dark, selector), base_url=str(out_dir) + "/").render(stylesheets=css, font_config=fconf)
        if not (fit_canvases and ser["canvases"]):
            break
        widths = _measure(doc)
        if all(abs(widths.get(c["id"], c["W"]) - c["W"]) < 1.5 for c in ser["canvases"]):
            break
        app.eval("window.__resizeCanvasBoxes(" + json.dumps(widths) + ")")
        app.settle(100)
        ser = app.serialize(selector)
    return doc


SCROLL_CLASSES = ("table-wrap", "seg", "search-list", "eq")       # eq = display equations of the method page (overflow-x: auto)


def overflow_report(doc, width: int, tol: float = 1.5) -> list[str]:
    """Boxes that stick out of the right edge of the viewport, ignoring the insides of intentionally scrolling containers
    (wide tables are wrapped in .table-wrap { overflow-x: auto }).  [] means: no horizontal page scroll."""
    bad: list[str] = []

    def walk(box, inside_scroller):
        el = getattr(box, "element", None)
        classes = (el.get("class") or "").split() if el is not None else []
        scroller = inside_scroller or any(c in classes for c in SCROLL_CLASSES)
        if not inside_scroller and hasattr(box, "margin_width") and getattr(box, "position_x", None) is not None:
            try:
                right = box.position_x + box.margin_width()
            except Exception:  # noqa: BLE001
                right = 0
            if right > width + tol and not any(c in classes for c in ("sr-only",)):
                bad.append(f"<{getattr(box, 'element_tag', '?')} class='{' '.join(classes)}'> right edge {right:.0f}px > {width}px")
        for ch in getattr(box, "children", []) or []:
            walk(ch, scroller and not any(c in classes for c in SCROLL_CLASSES) or scroller)

    for page in doc.pages:
        walk(page._page_box, False)
    return bad


def render(app, out_dir: Path, name: str, width: int = 1280, height: int = 0, dark: bool = False,
           selector: str = "body", crop: bool = True, scale: float = 1.0, preview_font: bool = True, fit_canvases: bool = True, extra_css: str = ''):
    """Render the current DOM of ``app`` (a front_helpers.FrontApp) to ``out_dir/name.png``; returns (path, size).

    With ``fit_canvases`` the page is laid out first, the real width of every chart box is measured and fed back to the fake DOM
    (which redraws every chart at that width), so the canvases in the picture have the size they would have in a browser.
    """
    if not HAVE_RENDER:
        raise RuntimeError("weasyprint / pymupdf / PIL not available")
    out_dir = Path(out_dir)
    doc = layout(app, out_dir, name, width, height, dark, selector, preview_font, fit_canvases, extra_css)
    pdf_path = out_dir / f"{name}.pdf"
    doc.write_pdf(str(pdf_path))
    pdf = pymupdf.open(str(pdf_path))
    zoom = (96.0 / 72.0) * scale
    pix = pdf[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    png_path = out_dir / f"{name}.png"
    pix.save(str(png_path))
    img = Image.open(png_path).convert("RGB")
    if crop:
        bg = img.getpixel((2, img.height - 2))
        y = img.height - 1
        px = img.load()
        while y > 0:
            if any(px[x, y] != bg for x in range(0, img.width, 3)):
                break
            y -= 1
        img = img.crop((0, 0, img.width, min(img.height, y + 24)))
        img.save(png_path)
    pdf.close()
    return png_path, img.size


def slices(png_path: Path, step: int = 1100, overlap: int = 0) -> list[Path]:
    """Cut a tall render into viewer-friendly pieces ``name_s1.png`` ... ; returns their paths."""
    img = Image.open(png_path)
    out = []
    y, i = 0, 1
    while y < img.height:
        piece = img.crop((0, y, img.width, min(img.height, y + step)))
        p = png_path.with_name(png_path.stem + f"_s{i}.png")
        piece.save(p)
        out.append(p)
        y += step - overlap
        i += 1
    return out
