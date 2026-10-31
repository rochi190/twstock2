/*
 * format.js -- number / date formatting and tiny numeric helpers (pure functions, no DOM).
 *
 * Contract: every formatter returns the em dash "—" for null / undefined / NaN / +-Infinity / non-numeric
 * input and NEVER throws.  Signs use the typographic minus "−" (U+2212).  Taiwan conventions:
 * TWSE tick sizes for prices, 億 (1e8) / 兆 (1e12) for large TWD amounts, 張 (1000 shares) for volume.
 *
 * Exposes  window.TQ.fmt  and  window.TQ.util.
 */
(function (root) {
  'use strict';
  var TQ = root.TQ = root.TQ || {};

  var NA = '—';       // em dash
  var MINUS = '−';    // typographic minus
  var WEEKDAYS = ['日', '一', '二', '三', '四', '五', '六'];

  /** Finite number or null.  Numeric strings are accepted, everything else (bool, object, '') is null. */
  function toNum(x) {
    if (typeof x === 'number') return isFinite(x) ? x : null;
    if (typeof x === 'string' && x.trim() !== '') {
      var v = Number(x);
      return isFinite(v) ? v : null;
    }
    return null;
  }

  function isNum(x) { return toNum(x) !== null; }

  /** Insert thousands separators into a string of digits. */
  function group(digits) {
    return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /** Fixed-point number with thousands separators; opts.sign: always show + / − . */
  function fmtNum(x, dp, sign) {
    var v = toNum(x);
    if (v === null) return NA;
    dp = dp == null ? 2 : Math.max(0, Math.min(10, dp | 0));
    if (Math.abs(v) >= 1e21) return String(v);
    var s = Math.abs(v).toFixed(dp);
    var isZero = Number(s) === 0;
    var parts = s.split('.');
    var out = group(parts[0]) + (parts.length > 1 ? '.' + parts[1] : '');
    if (isZero) return out;
    if (v < 0) return MINUS + out;
    return sign ? '+' + out : out;
  }

  function fmtInt(x, sign) { return fmtNum(x, 0, sign); }

  /** Fraction -> percent text.  fmtPct(0.0123) = "1.2%", fmtPct(0.0123, 1, true) = "+1.2%". */
  function fmtPct(x, dp, sign) {
    var v = toNum(x);
    if (v === null) return NA;
    dp = dp == null ? 1 : dp;
    var n = fmtNum(v * 100, dp, sign);
    return n === NA ? NA : n + '%';
  }

  /** Difference of two fractions in percentage points: fmtPP(0.012) = "+1.2pp". */
  function fmtPP(x, dp) {
    var v = toNum(x);
    if (v === null) return NA;
    return fmtNum(v * 100, dp == null ? 1 : dp, true) + 'pp';
  }

  /** Fraction -> basis points: fmtBps(0.0012) = "12bp". */
  function fmtBps(x, dp, sign) {
    var v = toNum(x);
    if (v === null) return NA;
    return fmtNum(v * 1e4, dp == null ? 0 : dp, sign) + 'bp';
  }

  /** Plain signed number: fmtSigned(0.5, 2) = "+0.50". */
  function fmtSigned(x, dp) { return fmtNum(x, dp == null ? 2 : dp, true); }

  /** NT$ amount with separators: fmtTWD(2385) = "NT$2,385". */
  function fmtTWD(x, dp) {
    var n = fmtNum(x, dp == null ? 0 : dp);
    return n === NA ? NA : (n.charAt(0) === MINUS ? MINUS + 'NT$' + n.slice(1) : 'NT$' + n);
  }

  /** TWSE/TPEx tick size for a price.  ETFs: 0.01 below NT$50, 0.05 from NT$50. */
  function tickSize(price, kind) {
    var p = toNum(price);
    if (p === null || p <= 0) return 0.01;
    if (kind === 'etf') return p < 50 ? 0.01 : 0.05;
    if (p < 10) return 0.01;
    if (p < 50) return 0.05;
    if (p < 100) return 0.1;
    if (p < 500) return 0.5;
    if (p < 1000) return 1;
    return 5;
  }

  /** Number of decimals needed to show multiples of `step` exactly (0.05 -> 2, 0.5 -> 1, 5 -> 0). */
  function decimalsForStep(step) {
    var s = toNum(step);
    if (s === null || s <= 0) return 2;
    for (var d = 0; d <= 6; d++) {
      var m = s * Math.pow(10, d);
      if (Math.abs(m - Math.round(m)) < 1e-6) return d;
    }
    return 6;
  }

  function priceDecimals(price, kind) { return decimalsForStep(tickSize(price, kind)); }

  /** Price with the decimals its tick size implies: 2385 -> "2,385", 60.35 (ETF) -> "60.35", 87.4 -> "87.4". */
  function fmtPrice(price, kind) {
    var p = toNum(price);
    if (p === null) return NA;
    return fmtNum(p, priceDecimals(p, kind));
  }

  /** Large TWD amounts in Taiwanese units: 6.5e10 -> "650億", 3.2e12 -> "3.20兆", 5e7 -> "5,000萬". */
  function fmtBigTWD(x) {
    var v = toNum(x);
    if (v === null) return NA;
    var a = Math.abs(v), sgn = v < 0 ? MINUS : '';
    if (a >= 1e12) return sgn + (a / 1e12).toFixed(2) + '兆';
    if (a >= 1e8) {
      var yi = a / 1e8;
      return sgn + (yi >= 100 ? fmtNum(yi, 0) : yi >= 10 ? yi.toFixed(1) : yi.toFixed(2)) + '億';
    }
    if (a >= 1e4) return sgn + fmtNum(a / 1e4, 0) + '萬';
    return sgn + fmtNum(a, 0);
  }

  /** Share volume -> 張 (1 lot = 1000 shares); >= 10,000 lots shown as 萬張. */
  function fmtLots(shares) {
    var v = toNum(shares);
    if (v === null) return NA;
    var lots = v / 1000;
    if (Math.abs(lots) >= 1e4) return (lots / 1e4).toFixed(2) + '萬張';
    return fmtNum(lots, 0) + '張';
  }

  /** Compact axis label for large numbers: 1234567 -> "123萬". */
  function fmtCompact(x) {
    var v = toNum(x);
    if (v === null) return NA;
    var a = Math.abs(v), sgn = v < 0 ? MINUS : '';
    if (a >= 1e8) return sgn + (a / 1e8).toFixed(a >= 1e10 ? 0 : 1) + '億';
    if (a >= 1e4) return sgn + (a / 1e4).toFixed(a >= 1e6 ? 0 : 1) + '萬';
    return sgn + fmtNum(a, a >= 100 ? 0 : 1);
  }

  /** p-value: "<0.001", "0.031", "0.31". */
  function fmtP(p) {
    var v = toNum(p);
    if (v === null) return NA;
    if (v < 0.001) return '<0.001';
    if (v < 0.1) return v.toFixed(3);
    return v.toFixed(2);
  }

  /** Confidence interval of fractions as percent: "[50.8%, 56.4%]". */
  function fmtCI(ci, dp) {
    if (!ci || ci.length < 2 || !isNum(ci[0]) || !isNum(ci[1])) return NA;
    return '[' + fmtPct(ci[0], dp == null ? 1 : dp) + ', ' + fmtPct(ci[1], dp == null ? 1 : dp) + ']';
  }

  /** Confidence interval of plain numbers: "[-0.31, 0.86]". */
  function fmtCIn(ci, dp) {
    if (!ci || ci.length < 2 || !isNum(ci[0]) || !isNum(ci[1])) return NA;
    return '[' + fmtNum(ci[0], dp == null ? 2 : dp) + ', ' + fmtNum(ci[1], dp == null ? 2 : dp) + ']';
  }

  // ------------------------------------------------------------------------------------ dates
  var RE_DATE = /^(\d{4})-(\d{2})-(\d{2})/;

  /** 'YYYY-MM-DD' (or ISO timestamp) -> UTC milliseconds, or null. */
  function parseDate(s) {
    if (typeof s !== 'string') return null;
    var m = RE_DATE.exec(s);
    if (!m) return null;
    var y = +m[1], mo = +m[2], d = +m[3];
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return null;
    var t = Date.UTC(y, mo - 1, d);
    var dt = new Date(t);
    if (dt.getUTCMonth() !== mo - 1) return null;   // e.g. 2026-02-31
    return t;
  }

  function isoOf(t) {
    var d = new Date(t);
    var m = d.getUTCMonth() + 1, day = d.getUTCDate();
    return d.getUTCFullYear() + '-' + (m < 10 ? '0' : '') + m + '-' + (day < 10 ? '0' : '') + day;
  }

  function fmtDate(s) {
    var t = parseDate(s);
    return t === null ? NA : isoOf(t);
  }

  /** "10/22" */
  function fmtMD(s) {
    var t = parseDate(s);
    if (t === null) return NA;
    var d = new Date(t);
    return (d.getUTCMonth() + 1) + '/' + d.getUTCDate();
  }

  /** "週四" */
  function fmtWeekday(s) {
    var t = parseDate(s);
    return t === null ? '' : '週' + WEEKDAYS[new Date(t).getUTCDay()];
  }

  /** "2026-10-22 (週四)" */
  function fmtDateW(s) {
    var t = parseDate(s);
    return t === null ? NA : isoOf(t) + ' (' + fmtWeekday(s) + ')';
  }

  /** "2026-10" for axis labels */
  function fmtYM(s) {
    var t = parseDate(s);
    return t === null ? NA : isoOf(t).slice(0, 7);
  }

  /** ISO timestamp -> "2026-10-22 18:31" (wall-clock of the string itself, no timezone shift). */
  function fmtDateTime(s) {
    if (typeof s !== 'string') return NA;
    var m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(s);
    if (m) return m[1] + ' ' + m[2];
    return fmtDate(s);
  }

  /** The next `n` weekdays after `startIso` (holidays are unknown on the client: this is an estimate). */
  function weekdaysAfter(startIso, n) {
    var t = parseDate(startIso), out = [];
    if (t === null) return out;
    var guard = 0;
    while (out.length < n && guard++ < 400) {
      t += 86400000;
      var wd = new Date(t).getUTCDay();
      if (wd !== 0 && wd !== 6) out.push(isoOf(t));
    }
    return out;
  }

  // ------------------------------------------------------------------------------------ semantics
  /** 'up' | 'down' | 'flat' | 'na' for the Taiwan colour convention (red = up, green = down). */
  function dirOf(x, eps) {
    var v = toNum(x);
    if (v === null) return 'na';
    eps = eps == null ? 0 : eps;
    return v > eps ? 'up' : v < -eps ? 'down' : 'flat';
  }

  function arrow(dir) { return dir === 'up' ? '▲' : dir === 'down' ? '▼' : dir === 'flat' ? '▬' : ''; }

  /** "▲ +1.23%" / "▼ −1.45%" -- direction is carried by the glyph and the sign, not only by colour. */
  function fmtChange(x, dp) {
    var v = toNum(x);
    if (v === null) return NA;
    var d = dirOf(Number((v * 100).toFixed(dp == null ? 2 : dp)));
    return arrow(d) + ' ' + fmtPct(v, dp == null ? 2 : dp, true);
  }

  // Audit P-10 / DESIGN 9(6): the server's `signal` is a statistical screen, not advice.  The labels say so: 偏多候選 / 中性觀望 /
  // 偏空／避開; the buy-advice wording 偏多·買進 exists only behind the evidence gate (see signalInfo(sig, unlocked)).
  var SIGNALS = {
    buy:   { label: '偏多候選', short: '偏多', dir: 'up',   icon: '▲', unlockedLabel: '偏多·買進', unlockedShort: '買進' },
    hold:  { label: '中性觀望', short: '觀望', dir: 'flat', icon: '◆' },
    avoid: { label: '偏空／避開', short: '避開', dir: 'down', icon: '▼' }
  };

  var hasOwn = Object.prototype.hasOwnProperty;
  /** table[key] for own keys only (never Object.prototype members such as "toString"). */
  function own(table, key) { return typeof key === 'string' && hasOwn.call(table, key) ? table[key] : undefined; }

  /** Label record of a signal.  `unlocked` (default false) is true only when the historical evidence supports buy wording
   *  (verdict >= moderate_edge and, where known, the basket beats the universe after costs): then 買進 is allowed for `buy`. */
  function signalInfo(sig, unlocked) {
    var info = own(SIGNALS, sig);
    if (!info) return { label: NA, short: NA, dir: 'na', icon: '' };
    var out = { label: info.label, short: info.short, dir: info.dir, icon: info.icon };
    if (unlocked === true && info.unlockedLabel) { out.label = info.unlockedLabel; out.short = info.unlockedShort; }
    return out;
  }

  /** True when an evidence verdict level allows buy-advice wording (moderate_edge or strong_edge). */
  function levelUnlocksBuy(level) { return level === 'moderate_edge' || level === 'strong_edge'; }

  var VERDICTS = {
    no_edge:       { label: '無預測優勢', icon: '⊘', tone: 'neutral' },
    weak_edge:     { label: '微弱優勢', icon: '△', tone: 'warn' },
    moderate_edge: { label: '中等優勢', icon: '◐', tone: 'info' },
    strong_edge:   { label: '顯著優勢', icon: '◆', tone: 'accent' }
  };

  function verdictInfo(level) {
    return own(VERDICTS, level) || { label: '未判定', icon: '?', tone: 'neutral' };
  }

  var BASELINES = {
    always_up: '永遠看漲',
    prev_sign: '沿用前一期方向',
    train_majority: '訓練期多數方向'
  };

  function baselineName(key) { return own(BASELINES, key) || (key ? String(key) : NA); }

  var MARKETS = { TWSE: '上市', TPEx: '上櫃', INDEX: '指數' };
  function marketName(m) { return own(MARKETS, m) || (m ? String(m) : ''); }
  function kindName(k) { return k === 'etf' ? 'ETF' : k === 'index' ? '指數' : '股票'; }

  var EXEC_MODES = { next_open: '次日開盤價成交', close: '收盤價成交(樂觀)' };
  function execName(e) { return own(EXEC_MODES, e) || (e ? String(e) : NA); }

  // ------------------------------------------------------------------------------------ numeric utils
  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

  /** Wilson score interval for k successes of n at 95%.  Returns [lo, hi] or null. */
  function wilson(k, n, z) {
    if (!(n > 0) || !isFinite(k)) return null;
    z = z || 1.959964;
    var p = k / n, z2 = z * z;
    var den = 1 + z2 / n;
    var centre = (p + z2 / (2 * n)) / den;
    var half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den;
    return [Math.max(0, centre - half), Math.min(1, centre + half)];
  }

  /** [min, max] of the finite numbers of an array (null when there are none). */
  function extent(arr) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < arr.length; i++) {
      var v = arr[i];
      if (typeof v === 'number' && isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    return lo <= hi ? [lo, hi] : null;
  }

  /** Safe nested lookup: get(obj, 'a.b.0.c') -> value or undefined; never throws. */
  function get(obj, path) {
    var cur = obj;
    var parts = typeof path === 'string' ? path.split('.') : path;
    for (var i = 0; i < parts.length; i++) {
      if (cur === null || cur === undefined) return undefined;
      cur = cur[parts[i]];
    }
    return cur;
  }

  TQ.fmt = {
    NA: NA, MINUS: MINUS,
    toNum: toNum, isNum: isNum, fmtNum: fmtNum, fmtInt: fmtInt, fmtPct: fmtPct, fmtPP: fmtPP, fmtBps: fmtBps,
    fmtSigned: fmtSigned, fmtTWD: fmtTWD, tickSize: tickSize, decimalsForStep: decimalsForStep,
    priceDecimals: priceDecimals, fmtPrice: fmtPrice, fmtBigTWD: fmtBigTWD, fmtLots: fmtLots, fmtCompact: fmtCompact,
    fmtP: fmtP, fmtCI: fmtCI, fmtCIn: fmtCIn,
    parseDate: parseDate, isoOf: isoOf, fmtDate: fmtDate, fmtMD: fmtMD, fmtWeekday: fmtWeekday, fmtDateW: fmtDateW,
    fmtYM: fmtYM, fmtDateTime: fmtDateTime, weekdaysAfter: weekdaysAfter,
    dirOf: dirOf, arrow: arrow, fmtChange: fmtChange, signalInfo: signalInfo, levelUnlocksBuy: levelUnlocksBuy, verdictInfo: verdictInfo,
    baselineName: baselineName, marketName: marketName, kindName: kindName, execName: execName
  };
  TQ.util = { clamp: clamp, wilson: wilson, extent: extent, get: get, toNum: toNum, isNum: isNum };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
