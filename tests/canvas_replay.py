#!/usr/bin/env python3
"""Replay recorded Canvas2D operations into a PNG (PIL) so charts can be inspected without a browser.

Input JSON (written by the frontend tests)::

    {"W": 900, "H": 380, "bg": "#ffffff", "ops": [["set","fillStyle","#c62828"], ["fillRect",x,y,w,h], ...]}

Supported operations mirror ``CTX_METHODS`` in tests/front_prelude.js: save/restore, beginPath/moveTo/lineTo/
closePath/arc/rect, fill/stroke, fillRect/strokeRect/clearRect, fillText, setLineDash and the style properties
fillStyle, strokeStyle, lineWidth, lineCap, font, textAlign, textBaseline, globalAlpha.  Rendering is done at 2x
supersampling and downsampled for anti-aliasing.  Text uses Noto Sans TC when installed.

Usage:  python3 tests/canvas_replay.py recording.json out.png [--scale 2]
        (or ``replay(rec_dict, scale) -> PIL.Image`` from Python)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

# PIL lives in the conda python; the py_mini_racer venv does not have it.  Allow borrowing its site-packages.
_PIL_SITES = [os.environ.get("TQ_PIL_SITE", "")]  # site-packages of any Python that has Pillow, if the test venv lacks it
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:  # noqa: BLE001  pragma: no cover
    HAVE_PIL = False
    for _p in _PIL_SITES:
        if _p and Path(_p, "PIL").exists():
            sys.path.append(_p)
            try:
                from PIL import Image, ImageDraw, ImageFont  # type: ignore
                HAVE_PIL = True
                break
            except Exception:  # noqa: BLE001
                continue

FONT_CANDIDATES = [
    os.path.expanduser("~/.local/share/fonts/NotoSansTC-Regular.otf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache: dict = {}


def _font(size: float):
    key = round(size * 2) / 2
    if key not in _font_cache:
        for p in FONT_CANDIDATES:
            if Path(p).exists():
                try:
                    _font_cache[key] = ImageFont.truetype(p, max(6, int(round(key))))
                    break
                except Exception:  # noqa: BLE001
                    continue
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def parse_color(s, alpha: float = 1.0):
    """CSS colour -> (r, g, b, a) with a in 0..255."""
    s = (s or "#000000").strip()
    m = re.match(r"^#([0-9a-fA-F]{6})$", s)
    if m:
        n = int(m.group(1), 16)
        return ((n >> 16) & 255, (n >> 8) & 255, n & 255, int(round(255 * alpha)))
    m = re.match(r"^#([0-9a-fA-F]{3})$", s)
    if m:
        h = m.group(1)
        return (int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16), int(round(255 * alpha)))
    m = re.match(r"^rgba?\(([^)]*)\)$", s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        r, g, b = (int(float(parts[i])) for i in range(3))
        a = float(parts[3]) if len(parts) > 3 else 1.0
        return (r, g, b, int(round(255 * a * alpha)))
    return (0, 0, 0, int(round(255 * alpha)))


def _arc_points(cx, cy, r, a0, a1, ccw):
    if ccw:
        while a1 > a0:
            a1 -= 2 * math.pi
    else:
        while a1 < a0:
            a1 += 2 * math.pi
    sweep = a1 - a0
    if abs(sweep) > 2 * math.pi:
        sweep = math.copysign(2 * math.pi, sweep)
    n = max(8, int(abs(sweep) * max(r, 1) / 1.5) + 2)
    n = min(n, 720)
    return [(cx + r * math.cos(a0 + sweep * i / n), cy + r * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]


def replay(rec: dict, scale: int = 2):
    """Render the recording to a PIL RGBA image (bg-filled)."""
    if not HAVE_PIL:
        raise RuntimeError("PIL is not available")
    W, H = int(rec["W"]), int(rec["H"])
    S = scale
    bg = parse_color(rec.get("bg", "#ffffff"))
    img = Image.new("RGBA", (W * S, H * S), bg)
    st = dict(fillStyle="#000000", strokeStyle="#000000", lineWidth=1.0, lineCap="butt", lineJoin="miter",
              font="10px sans-serif", textAlign="start", textBaseline="alphabetic", globalAlpha=1.0)
    dash: list = []
    stack: list = []
    paths: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] | None = None

    def layer():
        return Image.new("RGBA", img.size, (0, 0, 0, 0))

    def commit(lyr):
        nonlocal img
        img = Image.alpha_composite(img, lyr)

    def polygon(points, color):
        lyr = layer()
        ImageDraw.Draw(lyr).polygon([(x * S, y * S) for x, y in points], fill=color)
        commit(lyr)

    def stroke_path(points, color, width, closed=False, round_ends=False):
        if len(points) < 2:
            return
        pts = list(points) + ([points[0]] if closed else [])
        lyr = layer()
        d = ImageDraw.Draw(lyr)
        w = max(1, int(round(width * S)))
        segs = [pts]
        if dash:
            segs = _dash(pts, [x for x in dash], )
        for seg in segs:
            if len(seg) >= 2:
                d.line([(x * S, y * S) for x, y in seg], fill=color, width=w, joint="curve")
        if round_ends and width >= 3 and not dash:
            r = w / 2
            for x, y in (pts[0], pts[-1]):
                d.ellipse([x * S - r, y * S - r, x * S + r, y * S + r], fill=color)
        commit(lyr)

    def _dash(pts, pattern):
        out = []
        on = True
        idx = 0
        remain = pattern[0] if pattern else 1e9
        seg_cur = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            pos = 0.0
            while L - pos > 1e-9:
                step = min(remain, L - pos)
                pos += step
                remain -= step
                px, py = x0 + dx * pos / L, y0 + dy * pos / L
                if on:
                    seg_cur.append((px, py))
                if remain <= 1e-9:
                    if on:
                        out.append(seg_cur)
                    on = not on
                    idx = (idx + 1) % len(pattern)
                    remain = pattern[idx]
                    seg_cur = [(px, py)] if on else []
        if on and len(seg_cur) >= 2:
            out.append(seg_cur)
        return out

    def font_size():
        m = re.search(r"(\d+(?:\.\d+)?)px", st["font"])
        return float(m.group(1)) if m else 10.0

    unknown: set = set()
    for op in rec["ops"]:
        name = op[0]
        if name == "set":
            k, v = op[1], op[2]
            st[k] = v
        elif name == "save":
            stack.append((dict(st), list(dash)))
        elif name == "restore":
            if stack:
                s, dash = stack.pop()
                st = s
        elif name == "beginPath":
            paths = []
            cur = None
        elif name == "moveTo":
            cur = [(op[1], op[2])]
            paths.append(cur)
        elif name == "lineTo":
            if cur is None:
                cur = [(op[1], op[2])]
                paths.append(cur)
            else:
                cur.append((op[1], op[2]))
        elif name == "arc":
            cx, cy, r, a0, a1 = op[1:6]
            ccw = bool(op[6]) if len(op) > 6 else False
            pts = _arc_points(cx, cy, r, a0, a1, ccw)
            if cur is None:
                cur = []
                paths.append(cur)
            cur.extend(pts)
        elif name == "rect":
            x, y, w, h = op[1:5]
            cur = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            paths.append(cur)
            cur = None
        elif name == "closePath":
            if cur and len(cur) > 1:
                cur.append(cur[0])
        elif name == "fill":
            color = parse_color(st["fillStyle"], float(st["globalAlpha"]))
            for p in paths:
                if len(p) >= 3:
                    polygon(p, color)
        elif name == "stroke":
            color = parse_color(st["strokeStyle"], float(st["globalAlpha"]))
            for p in paths:
                stroke_path(p, color, float(st["lineWidth"]), round_ends=(st["lineCap"] == "round"))
        elif name == "fillRect":
            x, y, w, h = op[1:5]
            polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], parse_color(st["fillStyle"], float(st["globalAlpha"])))
        elif name == "strokeRect":
            x, y, w, h = op[1:5]
            stroke_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], parse_color(st["strokeStyle"], float(st["globalAlpha"])), float(st["lineWidth"]), closed=True)
        elif name == "clearRect":
            x, y, w, h = op[1:5]
            polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], bg)
        elif name == "setLineDash":
            dash = [float(v) for v in (op[1] or [])]
            if len(dash) % 2 == 1:
                dash = dash * 2
        elif name == "fillText":
            text, x, y = str(op[1]), op[2], op[3]
            size = font_size()
            f = _font(size * S)
            ha = {"left": "l", "start": "l", "center": "m", "right": "r", "end": "r"}.get(st["textAlign"], "l")
            va = {"alphabetic": "s", "middle": "m", "top": "t", "bottom": "b", "hanging": "t", "ideographic": "b"}.get(st["textBaseline"], "s")
            lyr = layer()
            try:
                ImageDraw.Draw(lyr).text((x * S, y * S), text, font=f, fill=parse_color(st["fillStyle"], float(st["globalAlpha"])), anchor=ha + va)
            except Exception:  # noqa: BLE001
                ImageDraw.Draw(lyr).text((x * S, y * S), text, font=f, fill=parse_color(st["fillStyle"], float(st["globalAlpha"])))
            commit(lyr)
        elif name in ("setTransform", "translate", "scale"):
            if name != "setTransform":
                unknown.add(name)
        else:
            unknown.add(name)
    out = img.convert("RGB").resize((W, H), Image.LANCZOS)
    out.info["unsupported_ops"] = sorted(unknown)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    scale = 2
    if "--scale" in argv:
        scale = int(argv[argv.index("--scale") + 1])
    rec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    img = replay(rec, scale)
    img.save(argv[2])
    print("wrote", argv[2], img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
