/*
 * charts.js -- hand-written Canvas 2D charts (no libraries).
 *
 * Every drawing function has the signature
 *     draw*(ctx, W, H, pal, spec, hover)  ->  model
 * and only needs a Canvas2D-like `ctx` (save/restore/beginPath/moveTo/lineTo/closePath/arc/rect/fill/stroke/
 * fillRect/clearRect/fillText/measureText/setLineDash + style properties), W/H in CSS pixels.  The returned
 * `model` carries the geometry needed for hover (idxAtX, xOf), the tooltip content (tooltip(idx) -> plain
 * object), the legend, and an accessible text summary.  Nothing here touches the DOM except `mount()` at the
 * bottom, so the drawing code can be unit-tested against a recording context.
 *
 * Colour semantics: Taiwan convention, red = up, green = down; neutral series use accent / amber / violet / gray.
 * Charts follow the "thin marks, hairline recessive grid, legend for >= 2 series, text never wears series colour"
 * rules.  The palette values mirror the CSS custom properties in style.css (checked by a unit test).
 */
(function (root) {
  'use strict';
  var TQ = root.TQ = root.TQ || {};
  var F = TQ.fmt, U = TQ.util;
  var NA = F.NA;
  var PI = Math.PI;

  var FONT = '"Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,-apple-system,"Segoe UI",sans-serif';
  function font(size, weight) { return (weight || 400) + ' ' + size + 'px ' + FONT; }

  // ------------------------------------------------------------------------------------ palette
  var BASE = {
    light: {
      bg: '#ffffff', grid: '#e9edf3', axis: '#c3cbd8', text: '#141a26', text2: '#465167', muted: '#586376',
      up: '#c62828', down: '#17794a', accent: '#2457c5', amber: '#9a5800', violet: '#7447b8', gray: '#616c7e'
    },
    dark: {
      bg: '#151b24', grid: '#242d3a', axis: '#3a4556', text: '#e7ecf3', text2: '#b4bfce', muted: '#8d99ac',
      up: '#ff6b6b', down: '#35c27e', accent: '#6b9bff', amber: '#f0a93b', violet: '#b794f6', gray: '#8f9bb0'
    }
  };

  function hexToRgb(hex) {
    var m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
    if (!m) return [128, 128, 128];
    var n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(hex, a) {
    var c = hexToRgb(hex);
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
  }

  /** Full chart palette for 'light' | 'dark' (unknown -> light). */
  function palette(mode) {
    var b = BASE[mode === 'dark' ? 'dark' : 'light'];
    var p = {};
    Object.keys(b).forEach(function (k) { p[k] = b[k]; });
    p.mode = mode === 'dark' ? 'dark' : 'light';
    p.upSoft = rgba(b.up, 0.16);
    p.downSoft = rgba(b.down, 0.16);
    p.upMid = rgba(b.up, 0.5);
    p.downMid = rgba(b.down, 0.5);
    p.accentSoft = rgba(b.accent, 0.14);
    p.accentMid = rgba(b.accent, 0.28);
    p.accentFaint = rgba(b.accent, 0.07);
    p.future = rgba(b.accent, 0.045);
    p.pill = rgba(b.bg, 0.88);
    return p;
  }

  /** 'dark' | 'light' from the data-theme attribute, else from prefers-color-scheme. */
  function currentMode() {
    try {
      var el = root.document && root.document.documentElement;
      var t = el && el.getAttribute && el.getAttribute('data-theme');
      if (t === 'dark' || t === 'light') return t;
      if (root.matchMedia && root.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
    } catch (e) { /* ignore */ }
    return 'light';
  }

  // ------------------------------------------------------------------------------------ scales & ticks
  function finite(v) { return typeof v === 'number' && isFinite(v); }

  function linScale(d0, d1, r0, r1) {
    var span = d1 - d0;
    if (!finite(span) || span === 0) span = 1;
    var k = (r1 - r0) / span;
    var f = function (v) { return r0 + (v - d0) * k; };
    f.invert = function (p) { return d0 + (p - r0) / k; };
    f.d0 = d0; f.d1 = d1; f.r0 = r0; f.r1 = r1;
    return f;
  }

  function ceilNice(x) {
    if (!(x > 0)) return 1;
    var e = Math.floor(Math.log(x) / Math.LN10 + 1e-12);
    var base = Math.pow(10, e);
    var cands = [1, 2, 5, 10];
    for (var i = 0; i < cands.length; i++) {
      var v = cands[i] * base;
      if (v >= x * (1 - 1e-9)) return v;
    }
    return 10 * base;
  }

  /** 1-2-5 step close to `raw`, never below `minStep` (e.g. the TWSE tick size). */
  function niceStepFor(raw, minStep) {
    if (!(raw > 0) || !finite(raw)) raw = minStep > 0 ? minStep : 1;
    var e = Math.floor(Math.log(raw) / Math.LN10);
    var base = Math.pow(10, e);
    var f = raw / base;
    var nice = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10;
    var step = nice * base;
    if (minStep > 0 && step < minStep * (1 - 1e-9)) step = ceilNice(minStep);
    return step;
  }

  /**
   * Nice tick values strictly inside [min, max] (the domain is NOT extended) -> { step, values }.
   * Degenerate ranges give a single tick.
   */
  function tickSet(min, max, count, minStep) {
    if (!finite(min) || !finite(max) || max < min) return { step: minStep || 1, values: [] };
    if (max === min) return { step: minStep || 1, values: [min] };
    var step = niceStepFor((max - min) / Math.max(1, count || 5), minStep);
    var start = Math.ceil(min / step - 1e-9);
    var out = [];
    for (var k = start; k * step <= max + step * 1e-9 && out.length < 60; k++) {
      out.push(+(k * step).toFixed(10));
    }
    return { step: step, values: out };
  }

  function pctDecimals(step) {
    return Math.min(3, F.decimalsForStep(+(step * 100).toFixed(8)));
  }

  /** Expand [lo, hi] by a relative pad; makes a flat range non-degenerate. */
  function padDomain(lo, hi, pad, minSpan) {
    if (!finite(lo) || !finite(hi)) return [0, 1];
    var span = hi - lo;
    var mid = (hi + lo) / 2;
    if (span < (minSpan || 0)) { lo = mid - minSpan / 2; hi = mid + minSpan / 2; span = minSpan; }
    if (span <= 0) { var a = Math.abs(mid) * 0.05 || 1; return [mid - a, mid + a]; }
    return [lo - span * pad, hi + span * pad];
  }

  /**
   * Date tick positions for an index axis.  Returns [{i, label, year}] -- the first bar of each month (or of
   * every k-th month so that at most `maxLabels` fit).  Year starts are labelled with the year.
   */
  function dateTicks(dates, i0, i1, maxLabels) {
    var firsts = [];
    var lastKey = null;
    for (var i = i0; i <= i1 && i < dates.length; i++) {
      var t = F.parseDate(dates[i]);
      if (t === null) continue;
      var d = new Date(t);
      var key = d.getUTCFullYear() * 12 + d.getUTCMonth();
      if (key !== lastKey) {
        firsts.push({ i: i, y: d.getUTCFullYear(), m: d.getUTCMonth() + 1, key: key });
        lastKey = key;
      }
    }
    var strides = [1, 2, 3, 6, 12, 24, 60];
    var stride = 1;
    for (var s = 0; s < strides.length; s++) {
      stride = strides[s];
      if (Math.ceil(firsts.length / stride) <= Math.max(1, maxLabels)) break;
    }
    var out = [];
    firsts.forEach(function (f) {
      var ok = stride === 1 || (stride >= 12 ? (f.m === 1 && (f.y % (stride / 12) === 0 || stride === 12)) : ((f.m - 1) % stride === 0));
      if (!ok) return;
      // drop a partial first month that sits right at the left edge (label would hang off the axis)
      out.push({ i: f.i, label: f.m === 1 || stride >= 12 ? String(f.y) : f.m + '月', year: f.m === 1 });
    });
    return out;
  }

  // ------------------------------------------------------------------------------------ drawing helpers
  function setFont(ctx, size, weight) { ctx.font = font(size, weight); }

  function txt(ctx, s, x, y, color, size, align, baseline, weight) {
    if (s === null || s === undefined || s === '') return;
    if (!finite(x) || !finite(y)) return;
    setFont(ctx, size || 11, weight);
    ctx.fillStyle = color;
    ctx.textAlign = align || 'left';
    ctx.textBaseline = baseline || 'alphabetic';
    ctx.fillText(String(s), x, y);
  }

  function textWidth(ctx, s, size, weight) {
    setFont(ctx, size || 11, weight);
    var m = ctx.measureText(String(s));
    return m && finite(m.width) ? m.width : String(s).length * (size || 11) * 0.6;
  }

  function hline(ctx, x0, x1, y, color, w) {
    if (!finite(x0) || !finite(x1) || !finite(y)) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = w || 1;
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
  }

  function vline(ctx, x, y0, y1, color, w) {
    if (!finite(x) || !finite(y0) || !finite(y1)) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = w || 1;
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
  }

  function crisp(v) { return Math.round(v) + 0.5; }

  /** Rounded-rectangle path (top corners only when `topOnly`), r clamped to half the shorter side. */
  function roundRectPath(ctx, x, y, w, h, r, topOnly) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    if (r === 0) { ctx.rect(x, y, w, h); return; }
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arc(x + w - r, y + r, r, -PI / 2, 0, false);
    if (topOnly) {
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x, y + h);
    } else {
      ctx.lineTo(x + w, y + h - r);
      ctx.arc(x + w - r, y + h - r, r, 0, PI / 2, false);
      ctx.lineTo(x + r, y + h);
      ctx.arc(x + r, y + h - r, r, PI / 2, PI, false);
    }
    ctx.lineTo(x, y + r);
    ctx.arc(x + r, y + r, r, PI, PI * 1.5, false);
    ctx.closePath();
  }

  /** Polyline over (xs, ys) that breaks at non-finite points. */
  function polyline(ctx, xs, ys, color, width, dash) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    if (dash && ctx.setLineDash) ctx.setLineDash(dash);
    ctx.beginPath();
    var pen = false, any = false;
    for (var i = 0; i < xs.length; i++) {
      if (!finite(xs[i]) || !finite(ys[i])) { pen = false; continue; }
      if (!pen) { ctx.moveTo(xs[i], ys[i]); pen = true; } else { ctx.lineTo(xs[i], ys[i]); }
      any = true;
    }
    if (any) ctx.stroke();
    if (dash && ctx.setLineDash) ctx.setLineDash([]);
  }

  /** Filled area between (xs, ys) and a constant baseline y, per contiguous finite run. */
  function areaTo(ctx, xs, ys, baseY, color) {
    ctx.fillStyle = color;
    var run = [];
    function flush() {
      if (run.length < 2) { run = []; return; }
      ctx.beginPath();
      ctx.moveTo(run[0][0], baseY);
      for (var k = 0; k < run.length; k++) ctx.lineTo(run[k][0], run[k][1]);
      ctx.lineTo(run[run.length - 1][0], baseY);
      ctx.closePath();
      ctx.fill();
      run = [];
    }
    for (var i = 0; i < xs.length; i++) {
      if (finite(xs[i]) && finite(ys[i])) run.push([xs[i], ys[i]]); else flush();
    }
    flush();
  }

  /** Dot with a 2px surface ring (so it stays legible over lines). */
  function dot(ctx, x, y, r, color, ring) {
    if (!finite(x) || !finite(y)) return;
    ctx.beginPath();
    ctx.fillStyle = ring;
    ctx.arc(x, y, r + 2, 0, PI * 2, false);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, r, 0, PI * 2, false);
    ctx.fill();
  }

  function pill(ctx, pal, text, xRight, yCenter, size) {
    var w = textWidth(ctx, text, size) + 8, h = size + 6;
    ctx.fillStyle = pal.pill;
    roundRectPath(ctx, xRight - w, yCenter - h / 2, w, h, 4);
    ctx.fill();
    txt(ctx, text, xRight - 4, yCenter, pal.text2, size, 'right', 'middle');
  }

  function drawEmpty(ctx, W, H, pal, msg) {
    ctx.clearRect(0, 0, W, H);
    txt(ctx, msg || '沒有資料', W / 2, H / 2, pal.muted, 13, 'center', 'middle');
    return {
      W: W, H: H, empty: true, legend: [], a11y: msg || '沒有資料',
      idxAtX: function () { return -1; }, xOf: function () { return NaN; }, tooltip: function () { return null; }, count: 0
    };
  }

  function legendItem(label, color, kind) { return { label: label, color: color, kind: kind || 'line' }; }

  // ====================================================================================== price + fan chart
  function maOf(closes, win) {
    var out = new Array(closes.length), sum = 0, cnt = 0;
    for (var i = 0; i < closes.length; i++) {
      var c = closes[i];
      if (finite(c)) { sum += c; cnt++; } else { sum += 0; }
      if (i >= win) { var old = closes[i - win]; if (finite(old)) { sum -= old; cnt--; } }
      out[i] = (i >= win - 1 && cnt === win) ? sum / win : null;
    }
    return out;
  }

  function fillGaps(arr, anchor) {
    // linear interpolation of missing points; anchor is the value at day 0.  Returns null if too sparse.
    var n = arr.length, good = 0, i;
    for (i = 0; i < n; i++) if (finite(arr[i])) good++;
    if (good < Math.max(2, n * 0.5)) return null;
    var out = arr.slice();
    var prevI = -1, prevV = finite(anchor) ? anchor : null;
    for (i = 0; i < n; i++) {
      if (finite(out[i])) { prevI = i; prevV = out[i]; continue; }
      var j = i + 1;
      while (j < n && !finite(arr[j])) j++;
      if (prevV === null && j >= n) return null;
      if (j >= n) { out[i] = prevV; continue; }
      if (prevV === null) { out[i] = arr[j]; continue; }
      var t = (i - prevI) / (j - prevI);
      out[i] = prevV + (arr[j] - prevV) * t;
    }
    return out;
  }

  /** Pick the 5/25/50/75/95 rows of a fan payload; returns null when unusable. */
  function sanitizeFan(fan, lastClose) {
    if (!fan || !Array.isArray(fan.prices) || !Array.isArray(fan.levels)) return null;
    var targets = { lo90: 0.05, lo50: 0.25, med: 0.5, hi50: 0.75, hi90: 0.95 };
    var rows = {}, n = 0, any = false;
    Object.keys(targets).forEach(function (k) {
      var best = -1, bd = 0.03;
      for (var i = 0; i < fan.levels.length; i++) {
        var d = Math.abs(fan.levels[i] - targets[k]);
        if (d < bd && Array.isArray(fan.prices[i])) { bd = d; best = i; }
      }
      if (best < 0) { rows[k] = null; return; }
      var row = fan.prices[best].map(function (v) { return finite(v) ? v : null; });
      rows[k] = fillGaps(row, lastClose);
      if (rows[k]) { n = Math.max(n, rows[k].length); any = true; }
    });
    if (!any || !rows.med) {
      if (!rows.lo90 || !rows.hi90) return null;
      rows.med = rows.lo90.map(function (v, i) { return (v + rows.hi90[i]) / 2; });
    }
    n = rows.med.length;
    ['lo90', 'lo50', 'hi50', 'hi90'].forEach(function (k) { if (rows[k] && rows[k].length !== n) rows[k] = null; });
    // enforce lo <= med <= hi so polygons never self-intersect
    for (var i = 0; i < n; i++) {
      var m = rows.med[i];
      if (rows.lo90 && rows.lo90[i] > m) rows.lo90[i] = m;
      if (rows.hi90 && rows.hi90[i] < m) rows.hi90[i] = m;
      if (rows.lo50 && rows.lo50[i] > m) rows.lo50[i] = m;
      if (rows.hi50 && rows.hi50[i] < m) rows.hi50[i] = m;
      if (rows.lo90 && rows.lo50 && rows.lo90[i] > rows.lo50[i]) rows.lo90[i] = rows.lo50[i];
      if (rows.hi90 && rows.hi50 && rows.hi90[i] < rows.hi50[i]) rows.hi90[i] = rows.hi50[i];
    }
    return { n: n, rows: rows, days: Array.isArray(fan.days) && fan.days.length === n ? fan.days : null };
  }

  /**
   * Candles + volume + MA with the forecast fan continuing from the last close.
   * spec: { bars:[{d,o,h,l,c,v}], visible:120, fan:{days,levels,prices}, lastClose, asof, kind, ma:[20,60] }
   * hover: unified index u (0..nVis-1 = bars, nVis-1+k = forecast day k) or null.
   */
  function drawPriceFan(ctx, W, H, pal, spec, hover) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var kind = spec.kind;
    var allBars = (spec.bars || []).filter(function (b) { return b && typeof b === 'object'; });
    var nVis = Math.min(Math.max(1, spec.visible || 120), allBars.length);
    var i0 = allBars.length - nVis;
    var vis = allBars.slice(i0);
    var lastBarClose = nVis ? vis[nVis - 1].c : null;
    var lastClose = F.toNum(spec.lastClose);
    if (lastClose === null) lastClose = F.toNum(lastBarClose);
    var fan = sanitizeFan(spec.fan, lastClose);
    var nf = fan ? fan.n : 0;
    if (!nVis && !nf) return drawEmpty(ctx, W, H, pal, '尚無價格資料');
    if (!nVis && lastClose === null) return drawEmpty(ctx, W, H, pal, '尚無價格資料');

    // ---- data series
    var closesAll = allBars.map(function (b) { return F.toNum(b.c); });
    var maWins = spec.ma || [20, 60];
    var maSeries = maWins.map(function (w) { return maOf(closesAll, w).slice(i0); });
    var maColors = [pal.amber, pal.violet, pal.gray];

    // ---- y domain (price)
    var lo = Infinity, hi = -Infinity, i, v;
    for (i = 0; i < nVis; i++) {
      v = F.toNum(vis[i].l); if (v !== null && v < lo) lo = v;
      v = F.toNum(vis[i].h); if (v !== null && v > hi) hi = v;
    }
    if (lastClose !== null) { lo = Math.min(lo, lastClose); hi = Math.max(hi, lastClose); }
    if (fan) {
      for (i = 0; i < nf; i++) {
        var a = fan.rows.lo90 ? fan.rows.lo90[i] : fan.rows.med[i];
        var b = fan.rows.hi90 ? fan.rows.hi90[i] : fan.rows.med[i];
        if (a < lo) lo = a;
        if (b > hi) hi = b;
      }
    }
    if (!finite(lo) || !finite(hi)) return drawEmpty(ctx, W, H, pal, '價格資料不完整');
    var dom = padDomain(lo, hi, 0.05, Math.abs(hi) * 0.02);
    var minStep = F.tickSize(lastClose !== null ? lastClose : hi, kind);
    var ticks = tickSet(dom[0], dom[1], H < 260 ? 4 : 6, minStep);
    var dec = F.decimalsForStep(ticks.step);

    // ---- layout
    ctx.font = font(11);
    var labelW = 0;
    ticks.values.forEach(function (t) { labelW = Math.max(labelW, textWidth(ctx, F.fmtNum(t, dec), 11)); });
    var rightM = Math.ceil(labelW) + 12;
    var L = 6, R = W - rightM;
    var topPad = 10, axisH = 22, gap = 8;
    var hasVol = vis.some(function (b) { return F.toNum(b.v) !== null && b.v > 0; });
    var plotH = H - topPad - axisH;
    var volH = hasVol ? Math.max(30, Math.round(plotH * 0.19)) : 0;
    var priceH = plotH - volH - (hasVol ? gap : 0);
    var pTop = topPad, pBot = topPad + priceH;
    var vTop = pBot + gap, vBot = vTop + volH;
    var plotW = R - L;
    if (plotW < 40 || priceH < 40) return drawEmpty(ctx, W, H, pal, '圖表空間不足');

    var futureW = nf ? U.clamp(plotW * (plotW < 420 ? 0.36 : 0.28), 70, 360) : 0;
    if (futureW > plotW * 0.5) futureW = plotW * 0.5;
    var histW = plotW - futureW;
    var stepH = nVis ? histW / nVis : histW;
    var xa = nVis ? L + histW - stepH / 2 : L + 4;                  // last bar centre = forecast anchor
    var stepF = nf ? (R - 8 - xa) / nf : 0;
    var yP = linScale(dom[0], dom[1], pBot, pTop);
    function xOf(u) {
      if (u < nVis) return L + (u + 0.5) * stepH;
      return xa + (u - (nVis - 1)) * stepF;
    }
    var total = nVis + nf;

    // ---- future background + divider
    if (nf) {
      ctx.fillStyle = pal.future;
      ctx.fillRect(xa, pTop, R - xa, (hasVol ? vBot : pBot) - pTop);
    }

    // ---- grid + y labels
    ticks.values.forEach(function (t) {
      var y = crisp(yP(t));
      if (y < pTop - 1 || y > pBot + 1) return;
      hline(ctx, L, R, y, pal.grid, 1);
      txt(ctx, F.fmtNum(t, dec), R + 8, y, pal.muted, 11, 'left', 'middle');
    });
    hline(ctx, L, R, crisp(pBot), pal.axis, 1);
    if (hasVol) hline(ctx, L, R, crisp(vBot), pal.axis, 1);

    // ---- x labels (history months + forecast days)
    var dts = vis.map(function (b) { return b.d; });
    var xt = dateTicks(dts, 0, nVis - 1, Math.max(2, Math.floor((histW - 30) / 58)));
    var axisY = (hasVol ? vBot : pBot) + 14;
    xt.forEach(function (t) {
      var x = xOf(t.i);
      if (x < L + 14 || x > xa - 24) return;
      vline(ctx, crisp(x), (hasVol ? vBot : pBot), (hasVol ? vBot : pBot) + 4, pal.axis, 1);
      txt(ctx, t.label, x, axisY, pal.muted, 11, 'center');
    });
    if (nf) {
      var marks = stepF * 5 >= 38 ? [5, 10, 15, 20] : stepF * 10 >= 38 ? [10, 20] : [20];
      marks.forEach(function (k) {
        if (k > nf) return;
        var x = xa + k * stepF;
        vline(ctx, crisp(x), (hasVol ? vBot : pBot), (hasVol ? vBot : pBot) + 4, pal.axis, 1);
        txt(ctx, '+' + k + '日', x, axisY, pal.muted, 11, k === nf ? 'right' : 'center');
      });
    }

    // ---- volume
    if (hasVol) {
      var vmax = 0;
      for (i = 0; i < nVis; i++) { v = F.toNum(vis[i].v); if (v !== null && v > vmax) vmax = v; }
      var bw = Math.max(1, Math.min(14, stepH * 0.64));
      for (i = 0; i < nVis; i++) {
        var bar = vis[i];
        v = F.toNum(bar.v);
        if (v === null || vmax <= 0) continue;
        var up = F.toNum(bar.c) !== null && F.toNum(bar.o) !== null && bar.c >= bar.o;
        var hgt = Math.max(1, (v / vmax) * (volH - 2));
        ctx.fillStyle = up ? pal.upMid : pal.downMid;
        ctx.fillRect(xOf(i) - bw / 2, vBot - hgt, bw, hgt);
      }
      txt(ctx, F.fmtCompact(vmax / 1000) + '張', R + 8, vTop + 2, pal.muted, 10, 'left', 'top');
    }

    // ---- candles
    var cw = Math.max(1, Math.min(14, stepH * 0.64));
    for (i = 0; i < nVis; i++) {
      var k = vis[i];
      var o = F.toNum(k.o), h = F.toNum(k.h), l = F.toNum(k.l), c = F.toNum(k.c);
      if (o === null || h === null || l === null || c === null) continue;
      var x = xOf(i);
      var col = c > o ? pal.up : c < o ? pal.down : pal.muted;
      ctx.fillStyle = col;
      ctx.strokeStyle = col;
      var wx = stepH < 3 ? x : crisp(x);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(wx, yP(h));
      ctx.lineTo(wx, yP(l));
      ctx.stroke();
      var yb0 = yP(Math.max(o, c)), yb1 = yP(Math.min(o, c));
      ctx.fillRect(x - cw / 2, yb0, cw, Math.max(1, yb1 - yb0));
    }

    // ---- moving averages
    maSeries.forEach(function (series, idx) {
      var xs = [], ys = [];
      for (var j = 0; j < nVis; j++) { xs.push(xOf(j)); ys.push(finite(series[j]) ? yP(series[j]) : NaN); }
      polyline(ctx, xs, ys, maColors[idx % maColors.length], 1.5);
    });

    // ---- forecast fan
    var yLast = lastClose !== null ? yP(lastClose) : NaN;
    if (fan && finite(yLast)) {
      var fx = [], r = fan.rows;
      for (i = 1; i <= nf; i++) fx.push(xa + i * stepF);
      var band = function (loRow, hiRow, color) {
        if (!loRow || !hiRow) return;
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.moveTo(xa, yLast);
        for (var a = 0; a < nf; a++) ctx.lineTo(fx[a], yP(hiRow[a]));
        for (var b = nf - 1; b >= 0; b--) ctx.lineTo(fx[b], yP(loRow[b]));
        ctx.closePath();
        ctx.fill();
      };
      band(r.lo90, r.hi90, pal.accentSoft);
      band(r.lo50, r.hi50, pal.accentMid);
      var mx = [xa].concat(fx), my = [yLast].concat(r.med.map(function (p) { return yP(p); }));
      polyline(ctx, mx, my, pal.accent, 2);
    }

    // ---- last-close marker + divider
    if (finite(yLast) && nVis) {
      dot(ctx, xa, yLast, 3.5, pal.text, pal.bg);
    }
    if (nf) {
      vline(ctx, crisp(xa), pTop, hasVol ? vBot : pBot, pal.axis, 1);
      txt(ctx, '預測 →', xa + 6, pTop + 11, pal.muted, 11, 'left');
    }

    // ---- direct end labels (median / 95% / 5%): shortest variant that fits, skipped when they would collide
    if (fan && finite(yLast)) {
      var xr = R - 2, last = nf - 1, room = xr - (xa + 6);
      var pickText = function (variants) {
        for (var q = 0; q < variants.length; q++) if (textWidth(ctx, variants[q], 11) + 8 <= room) return variants[q];
        return variants[variants.length - 1];
      };
      var items = [];
      var medP = fan.rows.med[last];
      items.push({ y: yP(medP), text: pickText(['中位 ' + F.fmtPrice(medP, kind) + (lastClose ? ' (' + F.fmtPct(medP / lastClose - 1, 1, true) + ')' : ''),
        '中位 ' + F.fmtPrice(medP, kind), F.fmtPrice(medP, kind)]) });
      if (fan.rows.hi90) items.push({ y: yP(fan.rows.hi90[last]) - 9, text: pickText(['95% ' + F.fmtPrice(fan.rows.hi90[last], kind), F.fmtPrice(fan.rows.hi90[last], kind)]) });
      if (fan.rows.lo90) items.push({ y: yP(fan.rows.lo90[last]) + 9, text: pickText(['5% ' + F.fmtPrice(fan.rows.lo90[last], kind), F.fmtPrice(fan.rows.lo90[last], kind)]) });
      var placed = [];
      items.forEach(function (it) {
        var yy = U.clamp(it.y, pTop + 8, pBot - 8);
        for (var q = 0; q < placed.length; q++) if (Math.abs(placed[q] - yy) < 15) return;
        placed.push(yy);
        pill(ctx, pal, it.text, xr, yy, 11);
      });
    }

    // ---- hover
    var hoverU = (hover !== null && hover !== undefined && hover >= 0 && hover < total) ? hover : null;
    if (hoverU !== null) {
      var hx = xOf(hoverU);
      vline(ctx, crisp(hx), pTop, hasVol ? vBot : pBot, pal.text2, 1);
      if (hoverU < nVis) {
        var cc = F.toNum(vis[hoverU].c);
        if (cc !== null) dot(ctx, hx, yP(cc), 3.5, pal.text, pal.bg);
      } else if (fan) {
        dot(ctx, hx, yP(fan.rows.med[hoverU - nVis]), 3.5, pal.accent, pal.bg);
      }
    }

    // ---- tooltip content
    var futureDates = nf ? F.weekdaysAfter(spec.asof || (vis[nVis - 1] && vis[nVis - 1].d), nf) : [];
    function tooltip(u) {
      if (u === null || u === undefined || u < 0 || u >= total) return null;
      if (u < nVis) {
        var b = vis[u], gi = i0 + u;
        var prev = gi > 0 ? F.toNum(allBars[gi - 1].c) : null;
        var cl = F.toNum(b.c);
        var chg = prev !== null && cl !== null && prev !== 0 ? cl / prev - 1 : null;
        var rows = [
          { label: '收盤', value: F.fmtPrice(b.c, kind), strong: true },
          { label: '漲跌', value: F.fmtChange(chg, 2), dir: F.dirOf(chg) },
          { label: '開', value: F.fmtPrice(b.o, kind) }, { label: '高', value: F.fmtPrice(b.h, kind) },
          { label: '低', value: F.fmtPrice(b.l, kind) }, { label: '成交量', value: F.fmtLots(b.v) }
        ];
        maSeries.forEach(function (s, mi) {
          if (finite(s[u])) rows.push({ label: 'MA' + maWins[mi], value: F.fmtPrice(s[u], kind), color: maColors[mi % maColors.length], key: 'line' });
        });
        return { title: F.fmtDateW(b.d), rows: rows };
      }
      var kd = u - (nVis - 1), ix = kd - 1;
      var rr = fan.rows;
      function pr(p) { return F.fmtPrice(p, kind) + (lastClose ? ' (' + F.fmtPct(p / lastClose - 1, 1, true) + ')' : ''); }
      var out = [{ label: '中位數', value: pr(rr.med[ix]), strong: true, color: pal.accent, key: 'line' }];
      if (rr.lo50 && rr.hi50) out.push({ label: '50%區間', value: F.fmtPrice(rr.lo50[ix], kind) + ' ~ ' + F.fmtPrice(rr.hi50[ix], kind), color: pal.accentMid, key: 'box' });
      if (rr.lo90 && rr.hi90) out.push({ label: '90%區間', value: F.fmtPrice(rr.lo90[ix], kind) + ' ~ ' + F.fmtPrice(rr.hi90[ix], kind), color: pal.accentSoft, key: 'box' });
      return { title: '預測 第' + kd + '個交易日' + (futureDates[ix] ? ' (約 ' + F.fmtMD(futureDates[ix]) + ')' : ''), rows: out };
    }

    var legend = [legendItem('上漲(紅)', pal.up, 'box'), legendItem('下跌(綠)', pal.down, 'box')];
    maWins.forEach(function (w, idx) { legend.push(legendItem('MA' + w, maColors[idx % maColors.length], 'line')); });
    if (fan) {
      legend.push(legendItem('預測中位數', pal.accent, 'line'));
      if (fan.rows.lo50) legend.push(legendItem('50%區間', pal.accentMid, 'box'));
      if (fan.rows.lo90) legend.push(legendItem('90%區間', pal.accentSoft, 'box'));
    }
    var a11y = '價格走勢圖：近 ' + nVis + ' 個交易日K線' + (nf ? '，並延伸 ' + nf + ' 個交易日的預測區間' : '') +
      (lastClose !== null ? '，最新收盤價 ' + F.fmtPrice(lastClose, kind) : '') + '。';
    if (fan) a11y += ' 第' + nf + '日預測中位數 ' + F.fmtPrice(fan.rows.med[nf - 1], kind) + '。';

    return {
      W: W, H: H, count: total, nBars: nVis, nFuture: nf, L: L, R: R, top: pTop, bottom: hasVol ? vBot : pBot,
      xOf: xOf, yPrice: yP, anchorX: xa, legend: legend, a11y: a11y, tooltip: tooltip,
      idxAtX: function (px) {
        if (!finite(px) || px < L - 4 || px > R + 4) return -1;
        if (nf && px > xa + stepF / 2) return Math.min(total - 1, nVis - 1 + Math.max(1, Math.round((px - xa) / stepF)));
        return U.clamp(Math.floor((px - L) / stepH), 0, nVis - 1);
      },
      fan: fan, domain: dom, priceTicks: ticks
    };
  }

  // ====================================================================================== gauge
  /**
   * Half-circle probability gauge.  spec: { p, lo: 0.25, hi: 0.75 }.  The arc runs from 50% (12 o'clock) to p:
   * red to the right (P(up) > 50%), green to the left.  Values outside [lo, hi] are clamped at the ends.
   */
  function drawGauge(ctx, W, H, pal, spec) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var lo = spec.lo == null ? 0.25 : spec.lo, hi = spec.hi == null ? 0.75 : spec.hi;
    var p = F.toNum(spec.p);
    var sw = 9;
    var cx = W / 2, cy = H - 20;
    var rad = Math.min(W / 2 - sw - 6, cy - sw - 4);
    if (rad < 20) return drawEmpty(ctx, W, H, pal, NA);
    var ang = function (q) { return PI + U.clamp((q - lo) / (hi - lo), 0, 1) * PI; };
    // track
    ctx.beginPath();
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = sw;
    ctx.lineCap = 'round';
    ctx.arc(cx, cy, rad, PI, PI * 2, false);
    ctx.stroke();
    // 50% tick
    var a50 = ang(0.5);
    ctx.beginPath();
    ctx.strokeStyle = pal.axis;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'butt';
    ctx.moveTo(cx + (rad - sw) * Math.cos(a50), cy + (rad - sw) * Math.sin(a50));
    ctx.lineTo(cx + (rad + sw) * Math.cos(a50), cy + (rad + sw) * Math.sin(a50));
    ctx.stroke();
    txt(ctx, '50%', cx, cy - rad + sw + 12, pal.muted, 10, 'center');
    // end labels
    txt(ctx, F.fmtPct(lo, 0), cx - rad, cy + 14, pal.muted, 10, 'center');
    txt(ctx, F.fmtPct(hi, 0), cx + rad, cy + 14, pal.muted, 10, 'center');
    txt(ctx, '跌', cx - rad * 0.55, cy - 2, pal.muted, 11, 'center');
    txt(ctx, '漲', cx + rad * 0.55, cy - 2, pal.muted, 11, 'center');
    if (p === null) {
      txt(ctx, NA, cx, cy - rad * 0.35, pal.muted, 16, 'center', 'middle');
      return { W: W, H: H, empty: false, legend: [], a11y: '上漲機率：無資料', cx: cx, cy: cy, r: rad, tooltip: function () { return null; }, idxAtX: function () { return -1; }, xOf: function () { return NaN; }, count: 0 };
    }
    var a = ang(p);
    var color = p > 0.5 ? pal.up : p < 0.5 ? pal.down : pal.gray;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = sw;
    ctx.lineCap = 'round';
    if (p >= 0.5) ctx.arc(cx, cy, rad, a50, a, false); else ctx.arc(cx, cy, rad, a, a50, false);
    ctx.stroke();
    dot(ctx, cx + rad * Math.cos(a), cy + rad * Math.sin(a), 5, color, pal.bg);
    return {
      W: W, H: H, cx: cx, cy: cy, r: rad, angle: a, legend: [], count: 0, a11y: '上漲機率 ' + F.fmtPct(p, 1) + '，下跌機率 ' + F.fmtPct(1 - p, 1),
      tooltip: function () { return null; }, idxAtX: function () { return -1; }, xOf: function () { return NaN; }
    };
  }

  // ====================================================================================== interval strip
  /**
   * Horizontal 50% / 90% interval strip for a return forecast.
   * spec: { q05, q25, q50, q75, q95, mean }  (simple returns).  Zero is always inside the domain; the part of a
   * band right of zero is tinted red (gain) and the part left of zero green (loss), plus numeric labels in the card.
   */
  function drawIntervalStrip(ctx, W, H, pal, spec) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var q = ['q05', 'q25', 'q50', 'q75', 'q95'].map(function (k) { return F.toNum(spec[k]); });
    if (q[0] === null || q[4] === null) return drawEmpty(ctx, W, H, pal, NA);
    var lo = Math.min(q[0], 0), hi = Math.max(q[4], 0);
    var dom = padDomain(lo, hi, 0.08, 0.002);
    var L = 8, R = W - 8;
    if (R - L < 40) return drawEmpty(ctx, W, H, pal, NA);
    var axisY = H - 16;
    var cy = Math.max(10, (axisY - 4) / 2 + 2);
    var x = linScale(dom[0], dom[1], L, R);
    var t = tickSet(dom[0], dom[1], W < 260 ? 3 : 5, 0.0005);
    var dec = pctDecimals(t.step);
    hline(ctx, L, R, crisp(axisY), pal.axis, 1);
    t.values.forEach(function (v) {
      var xx = crisp(x(v));
      vline(ctx, xx, axisY, axisY + 4, pal.axis, 1);
      txt(ctx, F.fmtPct(v, dec, false), xx, axisY + 14, pal.muted, 10, 'center');
    });
    var x0 = x(0);
    var seg = function (a, b, h, cDown, cUp) {
      var xa = x(a), xb = x(b);
      var mid = U.clamp(x0, xa, xb);
      if (mid > xa) { roundRectPath(ctx, xa, cy - h / 2, mid - xa, h, 3); ctx.fillStyle = cDown; ctx.fill(); }
      if (xb > mid) { roundRectPath(ctx, mid, cy - h / 2, xb - mid, h, 3); ctx.fillStyle = cUp; ctx.fill(); }
    };
    seg(q[0], q[4], 6, pal.downMid, pal.upMid);
    if (q[1] !== null && q[3] !== null) seg(q[1], q[3], 14, pal.down, pal.up);
    vline(ctx, crisp(x0), cy - 14, axisY, pal.text2, 1);
    if (q[2] !== null) {
      ctx.beginPath();
      ctx.strokeStyle = pal.bg;
      ctx.lineWidth = 3;
      ctx.moveTo(crisp(x(q[2])), cy - 9);
      ctx.lineTo(crisp(x(q[2])), cy + 9);
      ctx.stroke();
      vline(ctx, crisp(x(q[2])), cy - 9, cy + 9, pal.text, 2);
    }
    var mean = F.toNum(spec.mean);
    if (mean !== null && mean >= dom[0] && mean <= dom[1]) {
      var mx = x(mean);
      ctx.beginPath();
      ctx.fillStyle = pal.bg;
      ctx.strokeStyle = pal.text;
      ctx.lineWidth = 1.5;
      ctx.moveTo(mx, cy - 5);
      ctx.lineTo(mx + 5, cy);
      ctx.lineTo(mx, cy + 5);
      ctx.lineTo(mx - 5, cy);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    return {
      W: W, H: H, legend: [], count: 0, x: x, domain: dom, tooltip: function () { return null; },
      idxAtX: function () { return -1; }, xOf: function () { return NaN; },
      a11y: '報酬區間：90%區間 ' + F.fmtPct(q[0], 1, true) + ' 至 ' + F.fmtPct(q[4], 1, true) +
        (q[1] !== null && q[3] !== null ? '，50%區間 ' + F.fmtPct(q[1], 1, true) + ' 至 ' + F.fmtPct(q[3], 1, true) : '') +
        (q[2] !== null ? '，中位數 ' + F.fmtPct(q[2], 1, true) : '')
    };
  }

  // ====================================================================================== generic time series panels
  function seriesExtent(series) {
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (s) {
      (s.values || []).forEach(function (v) { if (finite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } });
    });
    return lo <= hi ? [lo, hi] : null;
  }

  /**
   * Stacked panels sharing one index x-axis.
   * spec: { dates:[...], n, panels:[{ title, weight, domain?, includeZero?, pad?, minSpan?, ticks?, minStep?,
   *         fmt(v), refLines:[{v,label}], tickLabels:{0:'空手',1:'持股'},
   *         series:[{key,label,color,values,width,kind:'line'|'area'|'step',alpha,dash,endLabel,tipFmt(v)}] }] }
   */
  function drawTimeSeries(ctx, W, H, pal, spec, hover) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var dates = spec.dates || [];
    var panels = (spec.panels || []).filter(function (p) { return p && Array.isArray(p.series) && p.series.some(function (s) { return s && Array.isArray(s.values) && s.values.some(finite); }); });
    var n = spec.n || dates.length;
    if (!n) {
      panels.forEach(function (p) { p.series.forEach(function (s) { n = Math.max(n, (s.values || []).length); }); });
    }
    if (!panels.length || !n) return drawEmpty(ctx, W, H, pal, spec.emptyText || '沒有足夠的資料');

    ctx.font = font(11);
    var labelW = 0;
    panels.forEach(function (p) {
      var ext = p.domain || seriesExtent(p.series);
      var lo = ext[0], hi = ext[1];
      (p.refLines || []).forEach(function (r) { if (finite(r.v)) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); } });
      if (p.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      p._dom = p.domain ? p.domain.slice() : padDomain(lo, hi, p.pad == null ? 0.08 : p.pad, p.minSpan || 0);
      p._ticks = p.tickLabels ? { step: 1, values: Object.keys(p.tickLabels).map(Number) } : tickSet(p._dom[0], p._dom[1], p.ticks || 4, p.minStep);
      var fmt = p.fmt || function (v) { return F.fmtNum(v, F.decimalsForStep(p._ticks.step)); };
      p._fmt = fmt;
      p._labels = p._ticks.values.map(function (t) { return p.tickLabels ? p.tickLabels[t] : fmt(t, p._ticks.step); });
      p._labels.forEach(function (s) { labelW = Math.max(labelW, textWidth(ctx, s, 11)); });
    });
    var L = Math.ceil(labelW) + 12, R = W - 12;
    var top = 6, axisH = 22, gap = 10;
    var headerH = 16;
    var avail = H - top - axisH - gap * (panels.length - 1) - headerH * panels.length;
    var wsum = 0;
    panels.forEach(function (p) { wsum += p.weight || 1; });
    var y = top;
    panels.forEach(function (p) {
      var h = Math.max(24, avail * (p.weight || 1) / wsum);
      p._hdr = y;
      p._top = y + headerH;
      p._bot = y + headerH + h;
      y = p._bot + gap;
    });
    var plotW = R - L;
    if (plotW < 40) return drawEmpty(ctx, W, H, pal, '圖表空間不足');
    function xOf(i) { return n <= 1 ? L + plotW / 2 : L + (i / (n - 1)) * plotW; }

    var legend = [], seen = {};
    panels.forEach(function (p) {
      var yS = linScale(p._dom[0], p._dom[1], p._bot, p._top);
      p._y = yS;
      if (p.title) txt(ctx, p.title, L, p._hdr + 11, pal.text2, 11, 'left');
      // grid + ticks
      p._ticks.values.forEach(function (tv, ti) {
        var yy = crisp(yS(tv));
        if (yy < p._top - 1 || yy > p._bot + 1) return;
        hline(ctx, L, R, yy, pal.grid, 1);
        txt(ctx, p._labels[ti], L - 6, yy, pal.muted, 11, 'right', 'middle');
      });
      hline(ctx, L, R, crisp(p._bot), pal.axis, 1);
      (p.refLines || []).forEach(function (r) {
        if (!finite(r.v)) return;
        var yy = crisp(yS(r.v));
        ctx.setLineDash && ctx.setLineDash([4, 3]);
        hline(ctx, L, R, yy, r.color || pal.axis, 1);
        ctx.setLineDash && ctx.setLineDash([]);
        if (r.label) txt(ctx, r.label, R - 2, yy - 3, pal.muted, 10, 'right');
      });
      // series
      p.series.forEach(function (s) {
        if (!s || !Array.isArray(s.values)) return;
        var xs = [], ys = [];
        for (var i = 0; i < n; i++) { xs.push(xOf(i)); ys.push(finite(s.values[i]) ? yS(s.values[i]) : NaN); }
        if (s.kind === 'step') {
          // contiguous runs of positive values become single rectangles (no per-point striping)
          var bw = Math.max(1, plotW / Math.max(1, n - 1));
          ctx.fillStyle = s.color;
          var j = 0;
          while (j < n) {
            if (!finite(s.values[j]) || s.values[j] <= p._dom[0]) { j++; continue; }
            var j0 = j, vmax = s.values[j];
            while (j + 1 < n && finite(s.values[j + 1]) && s.values[j + 1] > p._dom[0]) { j++; vmax = Math.max(vmax, s.values[j]); }
            var xl = Math.max(L, xs[j0] - bw / 2), xr2 = Math.min(R, xs[j] + bw / 2);
            var yy2 = yS(Math.min(vmax, p._dom[1]));
            ctx.fillRect(xl, yy2, Math.max(1, xr2 - xl), p._bot - yy2);
            j++;
          }
        } else {
          if (s.kind === 'area') areaTo(ctx, xs, ys, finite(s.base) ? yS(s.base) : p._bot, s.fill || pal.accentSoft);
          polyline(ctx, xs, ys, s.color, s.width || 2, s.dash);
          if (s.endLabel) {
            var li = -1;
            for (var q = n - 1; q >= 0; q--) if (finite(s.values[q])) { li = q; break; }
            if (li >= 0) { s._endIdx = li; dot(ctx, xs[li], ys[li], 3.5, s.color, pal.bg); }
          }
        }
        if (s.label && !seen[s.label]) {
          seen[s.label] = 1;
          legend.push(legendItem(s.label, s.color, s.kind === 'step' || s.kind === 'area' ? 'box' : 'line'));
        }
      });
    });
    // end labels (after all series so they sit on top), de-collided per panel
    panels.forEach(function (p) {
      var labs = [];
      p.series.forEach(function (s) {
        if (s.endLabel && s._endIdx !== undefined) {
          labs.push({ y: p._y(s.values[s._endIdx]), text: (s.shortLabel || s.label) + ' ' + (s.endFmt ? s.endFmt(s.values[s._endIdx]) : p._fmt(s.values[s._endIdx], p._ticks.step)) });
        }
      });
      labs.sort(function (a, b) { return a.y - b.y; });
      var prevY = -1e9;
      labs.forEach(function (l) {
        var yy = Math.max(l.y - 11, prevY + 14);
        yy = U.clamp(yy, p._top + 6, p._bot - 6);
        pill(ctx, pal, l.text, R - 4, yy, 11);
        prevY = yy;
      });
    });

    // x ticks
    var xt = dateTicks(dates, 0, Math.min(n, dates.length) - 1, Math.max(2, Math.floor(plotW / 62)));
    var lastBot = panels[panels.length - 1]._bot;
    xt.forEach(function (t) {
      var xx = xOf(t.i);
      if (xx < L + 8 || xx > R - 8) return;
      vline(ctx, crisp(xx), lastBot, lastBot + 4, pal.axis, 1);
      txt(ctx, t.label, xx, lastBot + 16, pal.muted, 11, 'center');
    });

    // hover
    var hi = (hover !== null && hover !== undefined && hover >= 0 && hover < n) ? Math.round(hover) : null;
    if (hi !== null) {
      var hx = xOf(hi);
      vline(ctx, crisp(hx), panels[0]._top, lastBot, pal.text2, 1);
      panels.forEach(function (p) {
        p.series.forEach(function (s) {
          if (s.kind === 'step' || (s.kind === 'area' && s.noDot)) return;
          if (s.values && finite(s.values[hi])) dot(ctx, hx, p._y(s.values[hi]), 3.5, s.color, pal.bg);
        });
      });
    }

    function tooltip(i) {
      if (i === null || i === undefined || i < 0 || i >= n) return null;
      var rows = [];
      panels.forEach(function (p) {
        p.series.forEach(function (s) {
          if (!s.label) return;
          var v = s.values ? s.values[i] : null;
          var f = s.tipFmt || p.tipFmt || p._fmt;
          rows.push({ label: s.label, value: finite(v) ? f(v, p._ticks.step) : NA, color: s.color, key: s.kind === 'step' || s.kind === 'area' ? 'box' : 'line', strong: true });
        });
      });
      return { title: dates[i] ? F.fmtDateW(dates[i]) : '第 ' + (i + 1) + ' 點', rows: rows };
    }
    return {
      W: W, H: H, count: n, L: L, R: R, top: panels[0]._top, bottom: lastBot, xOf: xOf, legend: legend,
      a11y: spec.a11y || '時間序列圖', tooltip: tooltip, panels: panels.map(function (p) { return { top: p._top, bottom: p._bot, y: p._y, domain: p._dom }; }),
      idxAtX: function (px) {
        if (!finite(px) || px < L - 6 || px > R + 6) return -1;
        return n <= 1 ? 0 : U.clamp(Math.round((px - L) / plotW * (n - 1)), 0, n - 1);
      }
    };
  }

  // ====================================================================================== reliability diagram
  /**
   * spec: { bins:[{p_mean, freq, n}] }.  Dots are sized by sqrt(n) with 95% Wilson whiskers, the dashed
   * diagonal is perfect calibration, and a strip of bars underneath shows the bin counts.
   */
  function drawCalibration(ctx, W, H, pal, spec, hover) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var bins = (spec.bins || []).filter(function (b) { return b && F.isNum(b.p_mean) && F.isNum(b.freq); });
    if (!bins.length) return drawEmpty(ctx, W, H, pal, '沒有足夠的校準資料');
    var lo = 1, hi = 0;
    bins.forEach(function (b) {
      lo = Math.min(lo, b.p_mean, b.freq);
      hi = Math.max(hi, b.p_mean, b.freq);
    });
    lo = Math.max(0, Math.floor((lo - 0.03) * 20) / 20);
    hi = Math.min(1, Math.ceil((hi + 0.03) * 20) / 20);
    if (hi - lo < 0.2) { var mid = (lo + hi) / 2; lo = Math.max(0, mid - 0.1); hi = Math.min(1, lo + 0.2); lo = Math.max(0, hi - 0.2); }
    var t = tickSet(lo, hi, 5, 0.01);
    var dec = pctDecimals(t.step);
    ctx.font = font(11);
    var lw = 0;
    t.values.forEach(function (v) { lw = Math.max(lw, textWidth(ctx, F.fmtPct(v, dec), 11)); });
    var L = Math.ceil(lw) + 12, R = W - 12;
    var top = 20, axisH = 34, stripH = Math.max(26, Math.min(48, H * 0.14)), gap = 8;
    var pBot = H - axisH - stripH - gap;
    if (pBot - top < 40 || R - L < 60) return drawEmpty(ctx, W, H, pal, '圖表空間不足');
    var x = linScale(lo, hi, L, R), y = linScale(lo, hi, pBot, top);
    txt(ctx, '實際上漲頻率', 4, 11, pal.text2, 11, 'left');
    t.values.forEach(function (v) {
      hline(ctx, L, R, crisp(y(v)), pal.grid, 1);
      vline(ctx, crisp(x(v)), top, pBot, pal.grid, 1);
      txt(ctx, F.fmtPct(v, dec), L - 6, y(v), pal.muted, 11, 'right', 'middle');
    });
    hline(ctx, L, R, crisp(pBot), pal.axis, 1);
    vline(ctx, crisp(L), top, pBot, pal.axis, 1);
    // perfect-calibration diagonal
    ctx.setLineDash && ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.strokeStyle = pal.gray;
    ctx.lineWidth = 1.25;
    ctx.moveTo(x(lo), y(lo));
    ctx.lineTo(x(hi), y(hi));
    ctx.stroke();
    ctx.setLineDash && ctx.setLineDash([]);
    txt(ctx, '完美校準', x(hi) - 4, y(hi) + 14, pal.muted, 10, 'right');
    var sorted = bins.slice().sort(function (a, b) { return a.p_mean - b.p_mean; });
    var nmax = 0;
    sorted.forEach(function (b) { if (F.isNum(b.n)) nmax = Math.max(nmax, b.n); });
    // whiskers
    sorted.forEach(function (b) {
      var ci = F.isNum(b.n) && b.n > 0 ? U.wilson(b.freq * b.n, b.n) : null;
      b._ci = ci;
      if (ci) {
        var a = U.clamp(ci[0], lo, hi), c = U.clamp(ci[1], lo, hi), bx = crisp(x(b.p_mean));
        vline(ctx, bx, y(a), y(c), pal.accentMid, 1);
        if (ci[0] >= lo) hline(ctx, bx - 3, bx + 3, crisp(y(a)), pal.accentMid, 1);
        if (ci[1] <= hi) hline(ctx, bx - 3, bx + 3, crisp(y(c)), pal.accentMid, 1);
      }
    });
    polyline(ctx, sorted.map(function (b) { return x(b.p_mean); }), sorted.map(function (b) { return y(b.freq); }), pal.accent, 1.5);
    sorted.forEach(function (b, i) {
      var rad = nmax > 0 && F.isNum(b.n) ? 3.5 + 4.5 * Math.sqrt(b.n / nmax) : 4.5;
      dot(ctx, x(b.p_mean), y(b.freq), rad, pal.accent, pal.bg);
    });
    // n strip
    var sTop = pBot + gap;
    var sBot = sTop + stripH;
    hline(ctx, L, R, crisp(sBot), pal.axis, 1);
    txt(ctx, '各分箱樣本數', L, sTop + 2, pal.text2, 10, 'left', 'top');
    var bw = Math.max(3, Math.min(22, (R - L) / Math.max(10, sorted.length * 1.8)));
    sorted.forEach(function (b) {
      if (!F.isNum(b.n) || nmax <= 0) return;
      var hh = Math.max(1, (b.n / nmax) * (stripH - 14));
      roundRectPath(ctx, x(b.p_mean) - bw / 2, sBot - hh, bw, hh, 3, true);
      ctx.fillStyle = pal.accentMid;
      ctx.fill();
    });
    // x ticks + title
    t.values.forEach(function (v) {
      txt(ctx, F.fmtPct(v, dec), x(v), sBot + 13, pal.muted, 11, 'center');
    });
    txt(ctx, '預測上漲機率(分箱平均)', (L + R) / 2, H - 3, pal.text2, 11, 'center');
    // hover
    var hi2 = (hover !== null && hover !== undefined && hover >= 0 && hover < sorted.length) ? hover : null;
    if (hi2 !== null) {
      var hb = sorted[hi2];
      vline(ctx, crisp(x(hb.p_mean)), top, sBot, pal.text2, 1);
      dot(ctx, x(hb.p_mean), y(hb.freq), 5, pal.accent, pal.bg);
    }
    return {
      W: W, H: H, count: sorted.length, legend: [legendItem('各分箱(圓點大小=樣本數)', pal.accent, 'box'), legendItem('完美校準', pal.gray, 'line')],
      a11y: '校準圖：共 ' + sorted.length + ' 個機率分箱，橫軸為預測上漲機率，縱軸為實際上漲頻率。', xOf: function (i) { return x(sorted[i].p_mean); },
      tooltip: function (i) {
        if (i === null || i === undefined || i < 0 || i >= sorted.length) return null;
        var b = sorted[i], ci = b._ci;
        return { title: '分箱 ' + (i + 1), rows: [
          { label: '預測機率', value: F.fmtPct(b.p_mean, 1), strong: true }, { label: '實際頻率', value: F.fmtPct(b.freq, 1), strong: true },
          { label: '95%信賴區間', value: ci ? F.fmtCI(ci, 1) : NA }, { label: '樣本數', value: F.fmtInt(b.n) }
        ] };
      },
      idxAtX: function (px) {
        if (!finite(px) || px < L - 6 || px > R + 6) return -1;
        var best = -1, bd = 1e9;
        sorted.forEach(function (b, i) { var d = Math.abs(x(b.p_mean) - px); if (d < bd) { bd = d; best = i; } });
        return best;
      }
    };
  }

  // ====================================================================================== histogram (PIT)
  /**
   * spec: { values:[...] (fractions summing to ~1), expected, n?, edges:'0.0-1.0' }.  Draws the uniform
   * reference line and (when n is known) the 95% binomial band around it.
   */
  function drawHistogram(ctx, W, H, pal, spec, hover) {
    spec = spec || {};
    ctx.clearRect(0, 0, W, H);
    var vals = (spec.values || []).map(function (v) { return F.toNum(v); });
    var nb = vals.length;
    if (!nb || vals.every(function (v) { return v === null; })) return drawEmpty(ctx, W, H, pal, '沒有足夠的資料');
    var expected = F.toNum(spec.expected);
    if (expected === null) expected = 1 / nb;
    var mx = expected * 1.25;
    vals.forEach(function (v) { if (v !== null && v > mx) mx = v; });
    var band = null;
    if (F.isNum(spec.n) && spec.n > 0) {
      var sd = Math.sqrt(expected * (1 - expected) / spec.n);
      band = [Math.max(0, expected - 1.96 * sd), expected + 1.96 * sd];
      mx = Math.max(mx, band[1] * 1.05);
    }
    var dom = [0, mx * 1.08];
    var t = tickSet(0, dom[1], 4, 0.001);
    var dec = pctDecimals(t.step);
    ctx.font = font(11);
    var lw = 0;
    t.values.forEach(function (v) { lw = Math.max(lw, textWidth(ctx, F.fmtPct(v, dec), 11)); });
    var L = Math.ceil(lw) + 12, R = W - 10, top = 10, bot = H - 36;
    if (R - L < 40 || bot - top < 30) return drawEmpty(ctx, W, H, pal, '圖表空間不足');
    var y = linScale(dom[0], dom[1], bot, top);
    t.values.forEach(function (v) {
      hline(ctx, L, R, crisp(y(v)), pal.grid, 1);
      txt(ctx, F.fmtPct(v, dec), L - 6, y(v), pal.muted, 11, 'right', 'middle');
    });
    var slot = (R - L) / nb, gapPx = 2;
    var bw = Math.max(2, Math.min(44, slot - gapPx));
    if (band) {
      ctx.fillStyle = pal.accentFaint;
      ctx.fillRect(L, y(band[1]), R - L, y(band[0]) - y(band[1]));
    }
    var hi = (hover !== null && hover !== undefined && hover >= 0 && hover < nb) ? hover : null;
    vals.forEach(function (v, i) {
      if (v === null) return;
      var cx = L + (i + 0.5) * slot;
      var hh = Math.max(0, bot - y(v));
      roundRectPath(ctx, cx - bw / 2, bot - hh, bw, hh, 4, true);
      ctx.fillStyle = i === hi ? pal.accent : pal.accentMid;
      ctx.fill();
    });
    hline(ctx, L, R, crisp(bot), pal.axis, 1);
    ctx.setLineDash && ctx.setLineDash([5, 4]);
    hline(ctx, L, R, crisp(y(expected)), pal.text2, 1.25);
    ctx.setLineDash && ctx.setLineDash([]);
    txt(ctx, '均勻分布 ' + F.fmtPct(expected, dec > 0 ? dec : 0), R - 2, y(expected) - 4, pal.text2, 10, 'right');
    // x labels: bin edges 0 .. 1
    var every = nb <= 10 ? (slot * 2 >= 30 ? 1 : 5) : Math.ceil(nb / 5);
    for (var e = 0; e <= nb; e += every) {
      txt(ctx, F.fmtNum(e / nb, 1), L + e * slot, bot + 14, pal.muted, 11, 'center');
    }
    if (nb % every !== 0) txt(ctx, '1.0', R, bot + 14, pal.muted, 11, 'center');
    txt(ctx, spec.xTitle || '預測累積機率(PIT)分位', (L + R) / 2, H - 4, pal.text2, 11, 'center');
    return {
      W: W, H: H, count: nb, xOf: function (i) { return L + (i + 0.5) * slot; },
      legend: [legendItem('各區間占比', pal.accentMid, 'box'), legendItem('均勻分布', pal.text2, 'line')].concat(band ? [legendItem('95%隨機波動帶', pal.accentFaint, 'box')] : []),
      a11y: '預測機率積分轉換(PIT)直方圖，共 ' + nb + ' 個區間；若模型校準良好，各區間應接近 ' + F.fmtPct(expected, 0) + '。',
      tooltip: function (i) {
        if (i === null || i === undefined || i < 0 || i >= nb) return null;
        return { title: 'PIT 區間 [' + F.fmtNum(i / nb, 1) + ', ' + F.fmtNum((i + 1) / nb, 1) + ')', rows: [
          { label: '實際占比', value: F.fmtPct(vals[i], 1), strong: true }, { label: '期望占比', value: F.fmtPct(expected, 1) } ] };
      },
      idxAtX: function (px) { return (px < L || px > R) ? -1 : U.clamp(Math.floor((px - L) / slot), 0, nb - 1); }
    };
  }

  // ====================================================================================== mounting (DOM glue)
  var mounted = [];

  function el(tag, cls) {
    var e = root.document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  /**
   * Create a canvas inside `host`, size it for the device pixel ratio, draw, and wire hover / keyboard /
   * resize / theme changes.  opts: { height | heightFn(w), draw(ctx, W, H, pal, hover) -> model, label }.
   * Returns a controller { redraw, setDraw, destroy, canvas, model }.
   */
  function mount(host, opts) {
    var doc = root.document;
    var wrap = el('div', 'chart-wrap');
    var legendEl = el('div', 'chart-legend');
    var canvas = el('canvas', 'chart-canvas');
    var tip = el('div', 'chart-tip');
    tip.setAttribute('role', 'status');
    tip.hidden = true;
    canvas.setAttribute('role', 'img');
    if (opts.label) canvas.setAttribute('aria-label', opts.label);
    wrap.appendChild(legendEl);
    var box = el('div', 'chart-box');
    box.appendChild(canvas);
    box.appendChild(tip);
    wrap.appendChild(box);
    host.appendChild(wrap);

    var ctl = { canvas: canvas, model: null, hover: null, destroyed: false };
    var drawFn = opts.draw;
    var raf = null;

    function size() {
      var w = box.clientWidth || (box.getBoundingClientRect ? box.getBoundingClientRect().width : 0) || host.clientWidth || 0;
      var h = typeof opts.heightFn === 'function' ? opts.heightFn(w) : (opts.height || 240);
      return { w: Math.floor(w), h: Math.floor(h) };
    }

    function renderLegend(model) {
      while (legendEl.firstChild) legendEl.removeChild(legendEl.firstChild);
      var items = (model && model.legend) || [];
      legendEl.hidden = items.length < 1;
      items.forEach(function (it) {
        var sp = el('span', 'lg');
        var sw = el('i', 'lg-key lg-' + (it.kind || 'line'));
        sw.style.background = it.color;
        sw.setAttribute('aria-hidden', 'true');
        sp.appendChild(sw);
        sp.appendChild(doc.createTextNode(it.label));
        legendEl.appendChild(sp);
      });
    }

    function showTip(idx) {
      var m = ctl.model;
      var content = m && m.tooltip ? m.tooltip(idx) : null;
      if (!content) { tip.hidden = true; return; }
      while (tip.firstChild) tip.removeChild(tip.firstChild);
      var title = el('div', 'tip-title');
      title.textContent = content.title || '';
      tip.appendChild(title);
      (content.rows || []).forEach(function (r) {
        var row = el('div', 'tip-row');
        var key = el('i', 'lg-key lg-' + (r.key || 'none'));
        if (r.color) key.style.background = r.color;
        key.setAttribute('aria-hidden', 'true');
        row.appendChild(key);
        var lab = el('span', 'tip-label');
        lab.textContent = r.label;
        var val = el('span', 'tip-val' + (r.strong ? ' strong' : '') + (r.dir === 'up' ? ' up' : r.dir === 'down' ? ' down' : ''));
        val.textContent = r.value;
        row.appendChild(lab);
        row.appendChild(val);
        tip.appendChild(row);
      });
      tip.hidden = false;
      var bw = box.clientWidth || 0, tw = tip.offsetWidth || 150;
      var x = m.xOf ? m.xOf(idx) : 0;
      var left = x + 14;
      if (finite(x) && left + tw > bw - 4) left = x - tw - 14;
      tip.style.left = Math.max(4, finite(left) ? left : 4) + 'px';
      tip.style.top = '8px';
    }

    function draw() {
      raf = null;
      if (ctl.destroyed) return;
      var sz = size();
      if (sz.w < 20 || sz.h < 20) return;       // hidden or not laid out yet
      var dpr = Math.min(3, Math.max(1, root.devicePixelRatio || 1));
      var cw = Math.round(sz.w * dpr), ch = Math.round(sz.h * dpr);
      if (canvas.width !== cw) canvas.width = cw;
      if (canvas.height !== ch) canvas.height = ch;
      canvas.style.width = sz.w + 'px';
      canvas.style.height = sz.h + 'px';
      var ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var pal = palette(currentMode());
      var model = null;
      try {
        model = drawFn(ctx, sz.w, sz.h, pal, ctl.hover);
      } catch (e) {
        try { if (root.console) root.console.error('chart draw failed', e); } catch (e2) { /* ignore */ }
        model = drawEmpty(ctx, sz.w, sz.h, pal, '圖表繪製失敗');
      }
      ctl.model = model;
      renderLegend(model);
      if (model && model.a11y) canvas.setAttribute('aria-label', model.a11y);
      // only charts with something to inspect (hover / arrow keys) take part in the tab order; gauges and strips are static images
      if (model && model.count > 0) canvas.setAttribute('tabindex', '0'); else canvas.removeAttribute('tabindex');
      if (ctl.hover !== null && model && model.tooltip) showTip(ctl.hover); else tip.hidden = true;
    }

    function schedule() {
      if (raf !== null) return;
      if (root.requestAnimationFrame) raf = root.requestAnimationFrame(draw); else { raf = 1; draw(); }
    }

    function setHover(idx) {
      var v = (idx === null || idx === undefined || idx < 0) ? null : idx;
      if (v === ctl.hover) return;
      ctl.hover = v;
      draw();
    }

    function onMove(e) {
      if (!ctl.model || !ctl.model.idxAtX) return;
      var rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : { left: 0 };
      var px = (e.clientX !== undefined ? e.clientX : 0) - rect.left;
      var idx = ctl.model.idxAtX(px);
      setHover(idx >= 0 ? idx : null);
    }
    function onLeave() { setHover(null); }
    function onKey(e) {
      var m = ctl.model;
      if (!m || !m.count) return;
      var cur = ctl.hover === null ? -1 : ctl.hover;
      var key = e.key;
      if (key === 'ArrowRight') cur = Math.min(m.count - 1, cur + 1);
      else if (key === 'ArrowLeft') cur = cur < 0 ? m.count - 1 : Math.max(0, cur - 1);
      else if (key === 'Home') cur = 0;
      else if (key === 'End') cur = m.count - 1;
      else if (key === 'Escape') { setHover(null); return; }
      else return;
      if (e.preventDefault) e.preventDefault();
      setHover(cur);
    }
    function onTheme() { schedule(); }

    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerdown', onMove);
    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('mouseleave', onLeave);
    canvas.addEventListener('keydown', onKey);
    canvas.addEventListener('blur', onLeave);
    var ro = null;
    if (typeof root.ResizeObserver === 'function') {
      ro = new root.ResizeObserver(function () { schedule(); });
      ro.observe(box);
    } else if (root.addEventListener) {
      root.addEventListener('resize', onTheme);
    }
    if (root.addEventListener) root.addEventListener('tq:theme', onTheme);

    ctl.redraw = function () { draw(); };
    ctl.setDraw = function (fn, label) {
      drawFn = fn;
      ctl.hover = null;
      if (label) canvas.setAttribute('aria-label', label);
      draw();
    };
    ctl.destroy = function () {
      ctl.destroyed = true;
      if (ro) { try { ro.disconnect(); } catch (e) { /* ignore */ } }
      if (root.removeEventListener) { root.removeEventListener('resize', onTheme); root.removeEventListener('tq:theme', onTheme); }
      var ix = mounted.indexOf(ctl);
      if (ix >= 0) mounted.splice(ix, 1);
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
    };
    mounted.push(ctl);
    draw();
    return ctl;
  }

  function redrawAll() { mounted.slice().forEach(function (c) { try { c.redraw(); } catch (e) { /* ignore */ } }); }

  TQ.charts = {
    FONT: FONT, font: font, palette: palette, currentMode: currentMode, rgba: rgba,
    linScale: linScale, niceStepFor: niceStepFor, tickSet: tickSet, dateTicks: dateTicks, padDomain: padDomain,
    pctDecimals: pctDecimals, maOf: maOf, sanitizeFan: sanitizeFan,
    drawEmpty: drawEmpty, drawPriceFan: drawPriceFan, drawGauge: drawGauge, drawIntervalStrip: drawIntervalStrip,
    drawTimeSeries: drawTimeSeries, drawCalibration: drawCalibration, drawHistogram: drawHistogram,
    mount: mount, redrawAll: redrawAll, mountedCount: function () { return mounted.length; }
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
