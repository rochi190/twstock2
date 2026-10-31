/*
 * core.js -- application kernel of the 台股量化實驗室 frontend (plain script, no modules, no network resources).
 *
 * Load order (index.html):  format.js, charts.js, core.js, view_stock.js, view_picks.js, view_backtest.js,
 *                           view_method.js, app.js.     Everything hangs off the single global  window.TQ .
 *
 * ======================================================================================== CONTRACT (stable)
 *
 *  TQ.h(tag, props, ...children) -> Element           DOM builder (never innerHTML unless props.html)
 *      props   : class (string | array of strings, falsy entries ignored) · id · text (textContent) ·
 *                html (innerHTML -- TRUSTED STATIC strings only) · style (object: camelCase, kebab-case or
 *                "--custom" keys; or a cssText string) · dataset (object -> data-* attributes) ·
 *                on<Event> handlers (onClick, onInput, onKeydown, ... -> addEventListener(lowercase)) ·
 *                boolean properties (disabled, checked, selected, hidden, open, readOnly, required, multiple) ·
 *                value (property) · everything else becomes an attribute (aria-*, role, tabindex, title, href,
 *                type, for, name, placeholder ...).  null / undefined / false props are skipped; for aria-* a
 *                boolean becomes "true"/"false".
 *      children: strings, numbers (non-finite -> "—"), Nodes, arrays (flattened); null/undefined/false/true skipped.
 *      SVG tags (svg, path, circle, line, rect, g, polyline, polygon, text) are created in the SVG namespace.
 *      `props` may be omitted:  TQ.h('p', 'hello').
 *  TQ.clear(el) / TQ.uid(prefix) / TQ.scope()          scope.add(fnOrObjWithDestroy) ; scope.dispose() (scopes nest: parent.add(child))
 *  TQ.codeHref(code, route='stock') -> "#/stock/2330"  ·  TQ.isNode(x) · TQ.logError(...) (console.error, never throws)
 *
 *  TQ.api   get(path, params, {signal}) -> Promise<json>
 *               rejects with TQ.ApiError (.code .status .message .detail) for HTTP errors, {"status":"error"} bodies
 *               and network failure (code "network"); rejects with an AbortError when the signal aborts.
 *           poll(path, params, {onProgress(progress), signal, intervalMs=1500}) -> Promise<ready payload>
 *               follows {"status":"computing","progress":{done,total,message}} until ready / error; rejects with
 *               AbortError when the signal aborts; tolerates up to 3 consecutive network failures.
 *           status() -> Promise<{market_open,last_trading_date,...} | null>   (cached 60 s, never rejects)
 *           isAbort(err) -> bool  (true for the AbortError of an aborted request / poll: ignore it, do not show it)
 *  TQ.ApiError
 *  TQ.prefs get(key, default) / set(key, value) -- JSON in localStorage, in-memory fallback when storage throws.
 *  TQ.theme get() -> 'auto'|'light'|'dark' · set(mode) · cycle() · effective() -> 'light'|'dark' · init()
 *           (sets/removes data-theme on <html>, persists, dispatches a "tq:theme" event on window).
 *
 *  TQ.ui (every function returns a Node; text is escaped; null/NaN -> "—")
 *      chip(text, kind, {icon,title})         kind: up down flat neutral info warn accent open closed muted danger
 *      card(title, body, {id,class,sub,tip,actions,level=2,bodyClass})   -> section.card ; .bodyEl ; .setBody(nodes)
 *      kpi(label, value, sub, {dir,tip,tone})  dir: up|down|flat -> colours the value (red up / green down)
 *      kv(rows)                                rows [{label, value, tip, dir}] -> <dl class="kv">
 *      skeleton(lines, {block})                loading placeholder (role=status)
 *      errorBox(err, onRetry)                  role=alert box with message + 重試 button
 *      progressBar(done, total, message)       .update(done,total,message)   (total<=0 -> indeterminate)
 *      infoTip(text, {label})                  the ⓘ tooltip: hover / focus / click, Escape closes, keyboard accessible
 *      table(columns, rows, opts)              see "table" below ; returns .table-wrap with .setRows(rows) .sortBy(key,dir)
 *      withTip(label, tipText)                 inline label + ⓘ
 *      segmented(options, value, onChange, {label, role})  radio-group (or tablist) ; .setValue(v) .getValue()
 *      field(label, control, {id, hint, tip})  labelled form control (div.field > label.field-label + control + p.field-hint)
 *      banner(kind, content, {title})          kind: info warn error ok neutral (grey note; unknown kinds fall back to info)
 *      disclaimerStrip(suffix)                 the short disclaimer strip (TQ.DISCLAIMER_SHORT + optional view-specific suffix), above the fold
 *      emptyState(text, {action})
 *      change(x, dp)                           "▲ +1.23%" span coloured red/green by sign (Taiwan convention)
 *      dirClass(x) -> 'up'|'down'|'flat'|'na'
 *      hbar(value01, {label, kind})            thin horizontal bar (probabilities, percentile meters)
 *      stackedBar(segments, {label})           segments [{key,label,value,cls}] -> one stacked bar
 *      verdictBox(verdict)                     backtest verdict {level,title,text,bullets} -> icon, level label, title, text, bullets (tone via .tone-*)
 *      statsTable(columns, opts)               columns [{label, stats, key}] -> strategy statistics comparison table
 *      loadInto(slot, {signal, load, render, skeleton})   skeleton -> content / errorBox(retry); returns {reload}
 *  table(columns, rows, opts):
 *      columns [{key,label,tip,num,align,sortable,value(row),render(row,i),cls,rowHeader,nowrap,defaultDir}]
 *      opts    {caption, sortKey, sortDir, rowClass(row), onRowClick(row), empty, label}
 *
 *  TQ.chartCard(title, spec, opts) -> section.card with a mounted TQ.charts chart (+ legend, tooltip, a11y summary)
 *      spec : the spec object of the draw function, OR a function (pal, W, H) -> spec (use it to take colours from
 *             the palette).  opts.kind: 'priceFan' | 'timeSeries' | 'calibration' | 'histogram' | 'gauge' | 'strip'
 *             (or opts.draw: function(ctx,W,H,pal,hover,spec) -> model) · opts.height | opts.heightFn(w) ·
 *             opts.sub · opts.tip · opts.actions · opts.note (small text under the chart) · opts.bare (no card).
 *      returns node with .chart (TQ.charts controller) .setSpec(spec) .destroy().
 *
 *  TQ.router  register(name, view) · start() · navigate(hash) · parse(hash) -> {name, params, canonical, redirect, query}
 *             current() · refreshNav() · setLastStock(code)   (call it once a stock code is known to be valid: it becomes the default
 *             of #/stock and #/backtest and is shown in the header navigation) · parseQuery("a=1&b=2") · buildQuery({a: 1})
 *             Routes:  #/stock/:code   #/picks   #/backtest/:code[?h=1&thr=0.55&...]   #/method[/section]   (default #/stock/<last or 2330>)
 *             The optional "?query" part of a hash is kept (never stripped by canonicalisation) and handed to the view as ctx.query;
 *             a query change from outside (pasted link, nav click) re-mounts the view, ctx.replace() changes it from inside without that.
 *  View interface:   TQ.views[name] = {
 *                        title(params) -> string,
 *                        mount(host, params, ctx) -> { destroy() }      // host: empty <div>, owned by the view
 *                    }
 *                    ctx = { signal (AbortSignal, aborted on route change), nav(hash), setStatus(text) [aria-live],
 *                            setTitle(text) [optional extra], route, query (decoded "?a=1" pairs), replace(hash) [same-route URL refinement] }
 *             A view must (a) abort/ignore async work when ctx.signal aborts, (b) destroy its charts in destroy()
 *             (TQ.scope() is made for that), (c) render exactly one <h1>.  Root element convention: <div class="view view-NAME">
 *             whose children are cards / .row containers (block layout; siblings are spaced by CSS, never add margins by hand).
 *  TQ.searchBox  create({onSelect(item), placeholder, id, label, blurOnSelect=true}) -> {el, input, focus(), setValue(), destroy()}
 *                mount(host, opts) -> same, appended to host.   Default onSelect navigates to #/stock/:code.
 *  TQ.views = {}   registry filled by the view files.        TQ.DISCLAIMER   default disclaimer text.   TQ.DISCLAIMER_SHORT  one-sentence strip text.
 */
(function (root) {
  'use strict';
  var TQ = root.TQ = root.TQ || {};
  var F = TQ.fmt, U = TQ.util;
  var NA = F.NA;

  TQ.APP_NAME = '台股量化實驗室';
  TQ.DISCLAIMER = '本工具為統計模型之研究用途，非投資建議。預測為機率分布而非保證，過去績效不代表未來，' +
    '投資一定有風險，請自行判斷並承擔盈虧。';
  // One sentence for the strip at the TOP of the pages that show probabilities / weights (audit P-09: a disclaimer only at the bottom
  // of a 4,000 px page is not "prominent").  Views append their own pointer ("請搭配…「歷史驗證」判讀").
  TQ.DISCLAIMER_SHORT = '以下為統計模型的研究輸出，非投資建議；機率不是保證，多數個股並無統計上顯著的預測優勢。';
  TQ.views = TQ.views || {};

  function doc() { return root.document; }
  function logError() {
    try { if (root.console && root.console.error) root.console.error.apply(root.console, arguments); } catch (e) { /* ignore */ }
  }
  TQ.logError = logError;

  var uidSeq = 0;
  function uid(prefix) { uidSeq += 1; return (prefix || 'tq') + '-' + uidSeq; }

  function isNode(x) { return !!x && typeof x === 'object' && typeof x.nodeType === 'number'; }
  function isProps(x) { return x !== null && typeof x === 'object' && !Array.isArray(x) && !isNode(x); }
  function arr(x) { return Array.isArray(x) ? x : []; }
  function num(x) { return F.toNum(x); }

  // ====================================================================================== DOM builder
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var SVG_TAGS = { svg: 1, path: 1, circle: 1, line: 1, rect: 1, g: 1, polyline: 1, polygon: 1, text: 1, ellipse: 1, title: 1 };
  var BOOL_PROPS = { disabled: 'disabled', checked: 'checked', selected: 'selected', multiple: 'multiple', required: 'required',
    readonly: 'readOnly', readOnly: 'readOnly', autofocus: 'autofocus', hidden: 'hidden', open: 'open' };

  function kebab(s) { return String(s).replace(/[A-Z]/g, function (c) { return '-' + c.toLowerCase(); }); }

  /** class value -> "a b c": strings and (nested) arrays; falsy entries dropped. */
  function classString(v) {
    if (Array.isArray(v)) return v.map(classString).filter(Boolean).join(' ');
    return v ? String(v) : '';
  }

  var UNITLESS = /^(opacity|flex|flexGrow|flexShrink|order|zIndex|lineHeight|fontWeight|zoom|columnCount|orphans|widows)$/;

  function setStyle(el, style) {
    if (typeof style === 'string') { el.setAttribute('style', style); return; }
    Object.keys(style).forEach(function (k) {
      var v = style[k];
      if (v === null || v === undefined || v === false) return;
      if (typeof v === 'number' && !isFinite(v)) return;
      if (k.indexOf('-') >= 0 && el.style.setProperty) el.style.setProperty(k, String(v));
      else el.style[k] = typeof v === 'number' && !UNITLESS.test(k) ? v + 'px' : String(v);     // plain numbers are pixels, except unitless properties
    });
  }

  function setProps(el, props, isSvg) {
    var keys = Object.keys(props);
    if (props.type !== undefined && keys.indexOf('type') > 0) keys = ['type'].concat(keys.filter(function (k) { return k !== 'type'; }));
    keys.forEach(function (k) {
      var v = props[k];
      if (k === 'class' || k === 'className') {
        var cls = classString(v);
        if (!cls) return;
        if (isSvg) el.setAttribute('class', cls); else el.className = cls;
      } else if (k === 'text') {
        if (v !== null && v !== undefined && v !== false) el.textContent = (typeof v === 'number' && !isFinite(v)) ? NA : String(v);
      } else if (k === 'html') {
        if (v !== null && v !== undefined) el.innerHTML = String(v);
      } else if (k === 'style') {
        if (v) setStyle(el, v);
      } else if (k === 'dataset') {
        if (v) Object.keys(v).forEach(function (d) { if (v[d] !== null && v[d] !== undefined && v[d] !== false) el.setAttribute('data-' + kebab(d), String(v[d])); });
      } else if (/^on[A-Z]/.test(k)) {
        if (typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (BOOL_PROPS[k] && !isSvg) {
        el[BOOL_PROPS[k]] = !!v;
      } else if (k === 'value') {
        if (v !== null && v !== undefined) el.value = String(v);
      } else if (v === null || v === undefined) {
        return;
      } else if (k.indexOf('aria-') === 0) {
        el.setAttribute(k, String(v));
      } else if (v === false) {
        return;
      } else if (v === true) {
        el.setAttribute(k, '');
      } else if (typeof v === 'number' && !isFinite(v)) {
        return;
      } else {
        el.setAttribute(k, String(v));
      }
    });
  }

  function appendKids(el, kids) {
    for (var i = 0; i < kids.length; i++) {
      var k = kids[i];
      if (k === null || k === undefined || k === false || k === true) continue;
      if (Array.isArray(k)) { appendKids(el, k); continue; }
      if (isNode(k)) { el.appendChild(k); continue; }
      if (typeof k === 'number') { el.appendChild(doc().createTextNode(isFinite(k) ? String(k) : NA)); continue; }
      if (typeof k === 'string') {
        el.appendChild(doc().createTextNode(k === 'NaN' || k === 'undefined' || k === 'null' ? NA : k));
      }
      // other objects are ignored (never "[object Object]")
    }
  }

  function h(tag, props) {
    var isSvg = !!SVG_TAGS[tag];
    var el = isSvg ? doc().createElementNS(SVG_NS, tag) : doc().createElement(tag);
    var start = 1;
    if (isProps(props)) { setProps(el, props, isSvg); start = 2; }
    else if (props === null || props === undefined) start = 2;
    var kids = [];
    for (var i = start; i < arguments.length; i++) kids.push(arguments[i]);
    appendKids(el, kids);
    return el;
  }

  function clear(el) {
    if (!el) return el;
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
  }

  /** Disposable scope: collect charts / listeners, dispose() them all (errors are logged, never thrown). */
  function scope() {
    var items = [];
    var self = {
      add: function (x) { if (x) items.push(x); return x; },
      dispose: function () {
        var list = items;
        items = [];
        list.forEach(function (x) {
          try { if (typeof x === 'function') x(); else if (typeof x.destroy === 'function') x.destroy(); } catch (e) { logError(e); }
        });
      }
    };
    self.destroy = self.dispose;      // scopes can be nested: parent.add(child)
    return self;
  }

  function codeHref(code, route) { return '#/' + (route || 'stock') + '/' + encodeURIComponent(String(code)); }

  TQ.h = h;
  TQ.clear = clear;
  TQ.uid = uid;
  TQ.scope = scope;
  TQ.codeHref = codeHref;
  TQ.isNode = isNode;

  // ====================================================================================== API
  function abortError() {
    var e;
    try {
      if (typeof root.DOMException === 'function') return new root.DOMException('The operation was aborted.', 'AbortError');
    } catch (x) { /* fall through */ }
    e = new Error('The operation was aborted.');
    e.name = 'AbortError';
    return e;
  }
  function isAbort(err) { return !!err && err.name === 'AbortError'; }

  function ApiError(message, code, status, detail) {
    this.name = 'ApiError';
    this.message = message || '發生未知錯誤';
    this.code = code || 'internal';
    this.status = status || 0;
    this.detail = detail;
    try { this.stack = (new Error(message)).stack; } catch (e) { /* ignore */ }
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;
  TQ.ApiError = ApiError;

  function buildUrl(path, params) {
    var qs = [];
    Object.keys(params || {}).forEach(function (k) {
      var v = params[k];
      if (v === null || v === undefined || v === '' || (typeof v === 'number' && !isFinite(v))) return;
      qs.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
    });
    return path + (qs.length ? (path.indexOf('?') >= 0 ? '&' : '?') + qs.join('&') : '');
  }

  var HTTP_MESSAGES = {
    404: '找不到所要求的資料。', 429: '請求過於頻繁，請稍候再試。', 500: '伺服器發生錯誤，請稍後再試。',
    502: '伺服器暫時無法回應，請稍後再試。', 503: '伺服器暫時無法回應，請稍後再試。'
  };

  function apiGet(path, params, opts) {
    opts = opts || {};
    var signal = opts.signal;
    if (signal && signal.aborted) return Promise.reject(abortError());
    if (typeof root.fetch !== 'function') return Promise.reject(new ApiError('此瀏覽器不支援必要功能 (fetch)。', 'network', 0));
    var init = { headers: { Accept: 'application/json' }, cache: 'no-store' };
    if (signal) init.signal = signal;
    return root.fetch(buildUrl(path, params), init).then(function (res) {
      var parsed;
      try { parsed = Promise.resolve(res.json()); } catch (e) { parsed = Promise.reject(e); }
      return parsed.then(function (js) { return { res: res, js: js }; }, function () { return { res: res, js: null }; });
    }, function (err) {
      if (isAbort(err) || (signal && signal.aborted)) throw (isAbort(err) ? err : abortError());
      throw new ApiError('無法連線到伺服器，請確認伺服器仍在執行並檢查網路連線。', 'network', 0);
    }).then(function (r) {
      var js = r.js, res = r.res;
      if (signal && signal.aborted) throw abortError();
      if (js && typeof js === 'object' && js.status === 'error') {
        var ei = js.error || {};
        throw new ApiError(ei.message || HTTP_MESSAGES[res.status] || '伺服器回報錯誤。', ei.code || ('http_' + res.status), res.status, ei.detail);
      }
      if (!res.ok) throw new ApiError(HTTP_MESSAGES[res.status] || ('伺服器回應錯誤 (HTTP ' + res.status + ')。'), 'http_' + res.status, res.status);
      if (js === null || typeof js !== 'object' || Array.isArray(js)) throw new ApiError('伺服器回應格式錯誤。', 'bad_response', res.status);       // every endpoint answers with a JSON object
      return js;
    });
  }

  function sleep(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal && signal.aborted) { reject(abortError()); return; }
      var t;
      function onAbort() { root.clearTimeout(t); reject(abortError()); }
      t = root.setTimeout(function () {
        if (signal && signal.removeEventListener) signal.removeEventListener('abort', onAbort);
        resolve();
      }, ms);
      if (signal && signal.addEventListener) signal.addEventListener('abort', onAbort);
    });
  }

  function normProgress(p) {
    p = isProps(p) ? p : {};
    var done = num(p.done), total = num(p.total);
    return { done: done === null ? 0 : done, total: total === null ? 0 : total, message: typeof p.message === 'string' ? p.message : '' };
  }

  function apiPoll(path, params, opts) {
    opts = opts || {};
    var signal = opts.signal;
    var interval = opts.intervalMs > 0 ? opts.intervalMs : 1500;
    var maxMs = opts.maxMs > 0 ? opts.maxMs : 45 * 60 * 1000;
    var elapsed = 0, netFails = 0;
    function step() {
      if (signal && signal.aborted) return Promise.reject(abortError());
      return apiGet(path, params, { signal: signal }).then(function (js) {
        netFails = 0;
        if (js.status !== 'computing') return js;
        if (typeof opts.onProgress === 'function') {
          try { opts.onProgress(normProgress(js.progress)); } catch (e) { logError(e); }
        }
        elapsed += interval;
        if (elapsed > maxMs) throw new ApiError('計算時間過長，已停止等待。請稍後重新整理再試。', 'timeout', 0);
        return sleep(interval, signal).then(step);
      }, function (err) {
        if (isAbort(err)) throw err;
        if (err && err.code === 'network' && netFails < 3) {
          netFails += 1;
          elapsed += interval;
          if (typeof opts.onProgress === 'function') {
            try { opts.onProgress({ done: 0, total: 0, message: '連線不穩，重新嘗試中（' + netFails + '/3）…' }); } catch (e) { logError(e); }
          }
          return sleep(interval * (netFails + 1), signal).then(step);
        }
        throw err;
      });
    }
    return step();
  }

  var statusCache = { at: 0, p: null };
  function apiStatus() {
    var now = Date.now();
    if (statusCache.p && now - statusCache.at < 60000) return statusCache.p;
    var p = apiGet('/api/status', null).then(function (js) { return js; }, function () { statusCache.p = null; return null; });
    statusCache = { at: now, p: p };
    return p;
  }

  TQ.api = { get: apiGet, poll: apiPoll, status: apiStatus, abortError: abortError, isAbort: isAbort };

  // ====================================================================================== prefs
  var PREF_PREFIX = 'tq.';
  var memPrefs = {};
  TQ.prefs = {
    get: function (key, def) {
      if (Object.prototype.hasOwnProperty.call(memPrefs, key)) return memPrefs[key];
      try {
        var raw = root.localStorage.getItem(PREF_PREFIX + key);
        if (raw !== null && raw !== undefined) return JSON.parse(raw);
      } catch (e) { /* storage unavailable or corrupt value */ }
      return def;
    },
    set: function (key, value) {
      memPrefs[key] = value;
      try { root.localStorage.setItem(PREF_PREFIX + key, JSON.stringify(value)); return true; } catch (e) { return false; }
    }
  };

  // ====================================================================================== theme
  var THEMES = ['auto', 'light', 'dark'];
  var THEME_LABELS = { auto: '自動', light: '淺色', dark: '深色' };
  var themeMode = 'auto';

  function fireTheme() {
    try { root.dispatchEvent(new root.Event('tq:theme')); } catch (e) { /* ignore */ }
  }
  function applyTheme(mode, silent) {
    themeMode = THEMES.indexOf(mode) >= 0 ? mode : 'auto';
    var el = doc() && doc().documentElement;
    if (el) {
      if (themeMode === 'auto') el.removeAttribute('data-theme'); else el.setAttribute('data-theme', themeMode);
    }
    if (!silent) fireTheme();
  }
  TQ.theme = {
    modes: THEMES, labels: THEME_LABELS,
    get: function () { return themeMode; },
    set: function (mode) { applyTheme(mode); TQ.prefs.set('theme', themeMode); return themeMode; },
    cycle: function () { return TQ.theme.set(THEMES[(THEMES.indexOf(themeMode) + 1) % THEMES.length]); },
    effective: function () { return TQ.charts && TQ.charts.currentMode ? TQ.charts.currentMode() : (themeMode === 'dark' ? 'dark' : 'light'); },
    init: function () {
      applyTheme(TQ.prefs.get('theme', 'auto'), true);
      try {
        var mq = root.matchMedia && root.matchMedia('(prefers-color-scheme: dark)');
        if (mq && !TQ.theme._mq) {
          TQ.theme._mq = mq;
          var onChange = function () { if (themeMode === 'auto') fireTheme(); };
          if (mq.addEventListener) mq.addEventListener('change', onChange); else if (mq.addListener) mq.addListener(onChange);
        }
      } catch (e) { /* ignore */ }
      return themeMode;
    }
  };

  // ====================================================================================== ui: small pieces
  var ui = {};
  TQ.ui = ui;

  function dirClass(x) {
    var v = num(x);
    return v === null ? 'na' : v > 0 ? 'up' : v < 0 ? 'down' : 'flat';
  }
  ui.dirClass = dirClass;

  ui.chip = function (text, kind, opts) {
    opts = opts || {};
    return h('span', { class: ['chip', 'chip-' + (kind || 'neutral')], title: opts.title },
      opts.icon ? h('span', { class: 'chip-icon', 'aria-hidden': 'true' }, opts.icon) : null, text);
  };

  ui.change = function (x, dp) {
    var v = num(x);
    if (v === null) return h('span', { class: 'chg na' }, NA);
    var rounded = Number((v * 100).toFixed(dp == null ? 2 : dp));
    return h('span', { class: ['chg', dirClass(rounded)] }, F.fmtChange(v, dp == null ? 2 : dp));
  };

  // ---- info tip --------------------------------------------------------------------------------------------
  var openTip = null;
  var tipDocBound = false;
  function closeTip(t) {
    if (!t) return;
    t.wrap.classList.remove('open');
    t.btn.setAttribute('aria-expanded', 'false');
    if (openTip === t) openTip = null;
  }
  /** The bubble is position:fixed (so scroll containers such as .table-wrap / .collapsible cannot clip it); place it over / under the button. */
  function placeTip(t) {
    var pop = t.pop;
    pop.style.left = '0px';
    pop.style.top = '0px';
    try {
      var b = t.btn.getBoundingClientRect(), p = pop.getBoundingClientRect();
      var pw = p.width || pop.offsetWidth || 250, ph = p.height || pop.offsetHeight || 40;
      var vw = root.innerWidth || (doc().documentElement && doc().documentElement.clientWidth) || 0;
      var left = b.left + b.width / 2 - pw / 2;
      if (vw) left = Math.max(8, Math.min(left, vw - pw - 8)); else left = Math.max(8, left);
      var top = b.top - ph - 8;
      if (top < 8) top = b.bottom + 8;                       // no room above (e.g. under the sticky header): open downwards
      pop.style.left = Math.round(left) + 'px';
      pop.style.top = Math.round(top) + 'px';
    } catch (e) { /* ignore */ }
  }
  function showTip(t) {
    if (openTip && openTip !== t) closeTip(openTip);
    t.wrap.classList.add('open');
    t.btn.setAttribute('aria-expanded', 'true');
    openTip = t;
    placeTip(t);
  }
  ui.infoTip = function (text, opts) {
    opts = opts || {};
    var id = uid('itip');
    var pop = h('span', { class: 'itip-pop', role: 'tooltip', id: id }, text);
    var btn = h('button', { type: 'button', class: 'itip-btn', 'aria-label': opts.label || '說明', 'aria-describedby': id, 'aria-expanded': 'false' }, 'ⓘ');
    var wrap = h('span', { class: 'itip' }, btn, pop);
    var t = { wrap: wrap, btn: btn, pop: pop };
    btn.addEventListener('click', function (e) {
      if (e.preventDefault) e.preventDefault();
      if (e.stopPropagation) e.stopPropagation();
      if (wrap.classList.contains('open')) closeTip(t); else showTip(t);
    });
    btn.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeTip(t); });
    wrap.addEventListener('mouseenter', function () { showTip(t); });
    wrap.addEventListener('mouseleave', function () { if (doc().activeElement !== btn) closeTip(t); });
    btn.addEventListener('focus', function () { showTip(t); });
    btn.addEventListener('blur', function () { closeTip(t); });
    if (!tipDocBound && doc() && doc().addEventListener) {
      tipDocBound = true;
      doc().addEventListener('click', function () { closeTip(openTip); });
      doc().addEventListener('keydown', function (e) { if (e.key === 'Escape') closeTip(openTip); });
      root.addEventListener('scroll', function () { closeTip(openTip); }, true);      // a fixed bubble would detach from its button
      root.addEventListener('resize', function () { closeTip(openTip); });
    }
    return wrap;
  };

  function withTip(label, tip) {
    return h('span', { class: 'with-tip' }, label, tip ? ui.infoTip(tip) : null);
  }
  ui.withTip = withTip;

  // ---- card ------------------------------------------------------------------------------------------------
  ui.card = function (title, body, opts) {
    opts = opts || {};
    var titleId = uid('card-t');
    var hasHead = title !== null && title !== undefined && title !== false || opts.actions;
    var head = null;
    if (hasHead) {
      head = h('header', { class: 'card-head' },
        h('div', { class: 'card-titles' },
          title ? h('h' + (opts.level || 2), { class: 'card-title', id: titleId }, title, opts.tip ? ui.infoTip(opts.tip) : null) : null,
          opts.sub ? h('p', { class: 'card-sub' }, opts.sub) : null),
        opts.actions ? h('div', { class: 'card-actions' }, opts.actions) : null);
    }
    var bodyEl = h('div', { class: ['card-body', opts.bodyClass] }, body);
    var card = h('section', { class: ['card', opts.class], id: opts.id, 'aria-labelledby': title ? titleId : null }, head, bodyEl);
    card.bodyEl = bodyEl;
    card.setBody = function (nodes) { clear(bodyEl); appendKids(bodyEl, [nodes]); };
    return card;
  };

  // ---- kpi / kv ----------------------------------------------------------------------------------------------
  ui.kpi = function (label, value, sub, opts) {
    opts = opts || {};
    var cls = opts.dir && opts.dir !== 'na' ? opts.dir : '';
    return h('div', { class: ['kpi', opts.tone ? 'kpi-' + opts.tone : null] },
      h('div', { class: 'kpi-label' }, label, opts.tip ? ui.infoTip(opts.tip) : null),
      h('div', { class: ['kpi-value', cls] }, value === null || value === undefined ? NA : value),
      sub !== null && sub !== undefined && sub !== '' ? h('div', { class: 'kpi-sub' }, sub) : null);
  };

  ui.kv = function (rows) {
    return h('dl', { class: 'kv' }, arr(rows).filter(Boolean).map(function (r) {
      return h('div', { class: 'kv-row' },
        h('dt', null, r.label, r.tip ? ui.infoTip(r.tip) : null),
        h('dd', { class: r.dir && r.dir !== 'na' ? r.dir : null }, r.value === null || r.value === undefined ? NA : r.value));
    }));
  };

  // ---- skeleton / error / progress / banner / empty -------------------------------------------------------------
  ui.skeleton = function (lines, opts) {
    opts = opts || {};
    var n = lines == null ? 3 : lines;
    var widths = ['100%', '86%', '94%', '70%', '88%'];
    var kids = [h('span', { class: 'sr-only' }, '載入中…')];
    if (opts.block) kids.push(h('div', { class: 'sk-block', style: { height: opts.block + 'px' }, 'aria-hidden': 'true' }));
    for (var i = 0; i < n; i++) kids.push(h('div', { class: 'sk-line', style: { width: widths[i % widths.length] }, 'aria-hidden': 'true' }));
    return h('div', { class: 'skeleton', role: 'status', 'aria-busy': 'true' }, kids);
  };

  var ERROR_TITLES = {
    network: '無法連線到伺服器', unknown_symbol: '找不到這檔股票', insufficient_history: '歷史資料不足',
    data_unavailable: '資料暫時無法取得', bad_request: '請求內容有誤', internal: '伺服器發生錯誤', timeout: '等待逾時',
    bad_response: '回應格式錯誤', render: '畫面產生失敗'
  };
  ui.errorBox = function (err, onRetry) {
    var code = err && err.code ? String(err.code) : 'render';
    var title = ERROR_TITLES[code] || (/^http_/.test(code) ? '伺服器回應錯誤' : '載入失敗');
    var msg = err && err.code && err.message ? err.message : '發生未預期的錯誤，請重新整理頁面或稍後再試。';
    return h('div', { class: 'error-box', role: 'alert' },
      h('span', { class: 'error-icon', 'aria-hidden': 'true' }, '!'),
      h('div', { class: 'error-main' },
        h('strong', { class: 'error-title' }, title),
        h('p', { class: 'error-msg' }, msg),
        h('small', { class: 'error-code' }, '錯誤代碼：' + code)),
      typeof onRetry === 'function' ? h('button', { type: 'button', class: 'btn', onClick: function () { onRetry(); } }, '重試') : null);
  };

  ui.progressBar = function (done, total, message) {
    var fill = h('div', { class: 'progress-fill' });
    var track = h('div', { class: 'progress-track' }, fill);
    var msg = h('div', { class: 'progress-msg' });
    var node = h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100' }, track, msg);
    node.update = function (d, t, m) {
      var dn = num(d), tn = num(t);
      var determinate = dn !== null && tn !== null && tn > 0;
      node.classList.toggle('progress-indeterminate', !determinate);
      var text = typeof m === 'string' ? m : '';
      if (determinate) {
        var pct = Math.max(0, Math.min(100, dn / tn * 100));
        fill.style.width = pct.toFixed(1) + '%';
        node.setAttribute('aria-valuenow', String(Math.round(pct)));
        node.setAttribute('aria-valuetext', (text ? text + '，' : '') + Math.round(pct) + '%');
        msg.textContent = (text || '計算中…') + '　' + F.fmtInt(Math.min(dn, tn)) + ' / ' + F.fmtInt(tn) + '（' + Math.round(pct) + '%）';
      } else {
        fill.style.width = '';
        node.removeAttribute('aria-valuenow');
        node.setAttribute('aria-valuetext', text || '計算中');
        msg.textContent = text || '計算中…';
      }
    };
    node.update(done, total, message);
    return node;
  };

  var BANNER_ICONS = { info: 'ⓘ', warn: '⚠', error: '!', ok: '✓', neutral: 'ⓘ' };
  ui.banner = function (kind, content, opts) {
    opts = opts || {};
    kind = BANNER_ICONS[kind] ? kind : 'info';
    return h('div', { class: ['banner', 'banner-' + kind], role: kind === 'error' ? 'alert' : 'status' },
      h('span', { class: 'banner-icon', 'aria-hidden': 'true' }, BANNER_ICONS[kind]),
      h('div', { class: 'banner-body' }, opts.title ? h('strong', { class: 'banner-title' }, opts.title) : null, content));
  };

  /** The short disclaimer strip (same markup as the stock page's): `suffix` is an optional view-specific pointer. */
  ui.disclaimerStrip = function (suffix) {
    return h('p', { class: 'disclaimer-strip' },
      h('span', { class: 'disclaimer-icon', 'aria-hidden': 'true' }, '⚠'),
      h('span', null, TQ.DISCLAIMER_SHORT + (suffix || '')));
  };

  ui.emptyState = function (text, opts) {
    opts = opts || {};
    return h('div', { class: 'empty' }, h('p', { class: 'empty-text' }, text), opts.action || null);
  };

  // ---- bars --------------------------------------------------------------------------------------------------
  ui.hbar = function (value, opts) {
    opts = opts || {};
    var v = num(value);
    var pct = v === null ? 0 : Math.max(0, Math.min(1, v)) * 100;
    var attrs = { class: ['hbar', opts.kind ? 'hbar-' + opts.kind : null], role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-label': opts.label };
    if (v !== null) { attrs['aria-valuenow'] = String(Math.round(pct)); attrs['aria-valuetext'] = F.fmtPct(v, 1); }
    return h('div', attrs, h('div', { class: 'hbar-fill', style: { width: pct.toFixed(1) + '%' } }));
  };

  ui.stackedBar = function (segments, opts) {
    opts = opts || {};
    var segs = arr(segments).filter(function (s) { return s && num(s.value) !== null && s.value > 0; });
    var total = 0;
    segs.forEach(function (s) { total += s.value; });
    var label = opts.label ? opts.label + '：' : '';
    var desc = segs.map(function (s) { return (s.label || s.key) + ' ' + F.fmtPct(s.value / (total || 1), 0); }).join('、');
    return h('div', { class: 'stack', role: 'img', 'aria-label': label + (desc || '無資料') },
      segs.map(function (s) {
        var share = total > 0 ? s.value / total : 0;
        return h('div', { class: ['stack-seg', s.cls], style: { width: (share * 100).toFixed(2) + '%' }, title: (s.label || s.key || '') + ' ' + F.fmtPct(share, 1) },
          share >= 0.12 ? h('span', { class: 'stack-txt', 'aria-hidden': 'true' }, F.fmtPct(share, 0)) : null);
      }));
  };

  // ---- segmented control -----------------------------------------------------------------------------------------
  ui.segmented = function (options, value, onChange, opts) {
    opts = opts || {};
    var role = opts.role === 'tablist' ? 'tablist' : 'radiogroup';
    var itemRole = role === 'tablist' ? 'tab' : 'radio';
    var checkedAttr = role === 'tablist' ? 'aria-selected' : 'aria-checked';
    var items = arr(options).map(function (o) { return typeof o === 'object' && o ? o : { value: o, label: String(o) }; });
    var current = value;
    var buttons = [];
    var node = h('div', { class: ['seg', opts.class], role: role, 'aria-label': opts.label });
    function paint() {
      var anyOn = items.some(function (it) { return it.value === current; });
      buttons.forEach(function (b, i) {
        var on = items[i].value === current;
        b.setAttribute(checkedAttr, on ? 'true' : 'false');
        b.setAttribute('tabindex', on || (!anyOn && i === 0) ? '0' : '-1');      // roving tabindex: exactly one tab stop, even for an unknown value
        b.classList.toggle('on', on);
      });
    }
    function choose(i, focus) {
      var v = items[i].value;
      if (focus && buttons[i].focus) buttons[i].focus();
      if (v === current) return;
      current = v;
      paint();
      if (typeof onChange === 'function') onChange(v);
    }
    items.forEach(function (it, i) {
      var b = h('button', { type: 'button', class: 'seg-btn', role: itemRole, title: it.title, dataset: { value: it.value },
        onClick: function () { choose(i, false); },
        onKeydown: function (e) {
          var k = e.key, n = items.length, to = -1;
          if (k === 'ArrowRight' || k === 'ArrowDown') to = (i + 1) % n;
          else if (k === 'ArrowLeft' || k === 'ArrowUp') to = (i - 1 + n) % n;
          else if (k === 'Home') to = 0;
          else if (k === 'End') to = n - 1;
          if (to < 0) return;
          if (e.preventDefault) e.preventDefault();
          choose(to, true);
        } }, it.label);
      buttons.push(b);
      node.appendChild(b);
    });
    node.setValue = function (v) { current = v; paint(); };
    node.getValue = function () { return current; };
    paint();
    return node;
  };

  /** Labelled form control: ui.field('預測期間', TQ.h('select', {id: 'x'}), {id: 'x', hint: '...', tip: '...'}) -> div.field (label[for] + control + hint). */
  ui.field = function (label, control, opts) {
    opts = opts || {};
    var hintId = opts.hint ? uid('hint') : null;
    if (hintId && control && control.setAttribute) control.setAttribute('aria-describedby', hintId);
    return h('div', { class: ['field', opts.class] },
      h('label', { class: 'field-label', for: opts.id || (control && control.id) || null }, label, opts.tip ? ui.infoTip(opts.tip) : null),
      control,
      opts.hint ? h('p', { class: 'field-hint', id: hintId }, opts.hint) : null);
  };

  // ---- table -----------------------------------------------------------------------------------------------------
  function isMissing(v) { return v === null || v === undefined || (typeof v === 'number' && !isFinite(v)) || v === ''; }
  function compareValues(a, b) {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    var sa = String(a), sb = String(b);
    try { return sa.localeCompare(sb, 'zh-Hant'); } catch (e) { return sa < sb ? -1 : sa > sb ? 1 : 0; }
  }

  ui.table = function (columns, rows, opts) {
    opts = opts || {};
    var cols = arr(columns);
    var state = { rows: arr(rows), key: opts.sortKey || null, dir: opts.sortDir === 'desc' ? 'desc' : 'asc' };
    var captionId = uid('tbl-cap');
    var thead = h('thead'), tbody = h('tbody');
    var caption = opts.caption ? h('caption', { class: opts.captionVisible ? 'tbl-caption' : 'sr-only', id: captionId }, opts.caption) : null;
    var tbl = h('table', { class: ['tbl', opts.dense ? 'tbl-dense' : null, opts.class] }, caption, thead, tbody);
    var wrap = h('div', { class: 'table-wrap', role: 'region', tabindex: '0', 'aria-label': opts.label || opts.caption || '資料表' }, tbl);

    function colAlign(c) { return c.align || (c.num ? 'right' : 'left'); }
    function sortVal(c, row) { return typeof c.value === 'function' ? c.value(row) : row ? row[c.key] : undefined; }

    function sortedRows() {
      var list = state.rows.slice();
      var col = null;
      cols.forEach(function (c) { if (c.key === state.key) col = c; });
      if (!col) return list;
      var sign = state.dir === 'desc' ? -1 : 1;
      var decorated = list.map(function (r, i) { return { r: r, i: i, v: sortVal(col, r) }; });
      decorated.sort(function (x, y) {
        var mx = isMissing(x.v), my = isMissing(y.v);
        if (mx || my) return mx && my ? x.i - y.i : (mx ? 1 : -1);   // missing values always last
        var c = compareValues(x.v, y.v);
        return c !== 0 ? sign * c : x.i - y.i;
      });
      return decorated.map(function (d) { return d.r; });
    }

    function renderHead() {
      clear(thead);
      var tr = h('tr');
      cols.forEach(function (c) {
        var al = colAlign(c);
        var content = c.th || c.label;
        var th;
        if (c.sortable) {
          var on = state.key === c.key;
          var ind = h('span', { class: 'sort-ind', 'aria-hidden': 'true' }, on ? (state.dir === 'desc' ? '▾' : '▴') : '↕');
          var btn = h('button', { type: 'button', class: 'th-btn', onClick: function () { sortBy(c.key, on ? (state.dir === 'asc' ? 'desc' : 'asc') : (c.defaultDir || (c.num ? 'desc' : 'asc'))); } }, content, ind);
          th = h('th', { scope: 'col', class: ['th-sortable', 'al-' + al, c.cls && typeof c.cls === 'string' ? c.cls : null], 'aria-sort': on ? (state.dir === 'desc' ? 'descending' : 'ascending') : 'none' },
            btn, c.tip ? ui.infoTip(c.tip) : null);
        } else {
          th = h('th', { scope: 'col', class: ['al-' + al, c.cls && typeof c.cls === 'string' ? c.cls : null] }, content, c.tip ? ui.infoTip(c.tip) : null);
        }
        tr.appendChild(th);
      });
      thead.appendChild(tr);
    }

    function cellContent(c, row, i) {
      try {
        if (typeof c.render === 'function') return c.render(row, i);
        var v = row ? row[c.key] : null;
        return isMissing(v) ? NA : v;
      } catch (e) { logError(e); return NA; }
    }

    function renderBody() {
      clear(tbody);
      var list = sortedRows();
      if (!list.length) {
        tbody.appendChild(h('tr', null, h('td', { class: 'tbl-empty', colspan: String(Math.max(1, cols.length)) }, opts.empty || '沒有資料')));
        return;
      }
      list.forEach(function (row, i) {
        var tr = h('tr', { class: [opts.rowClass ? opts.rowClass(row) : null, opts.onRowClick ? 'row-click' : null] });
        if (opts.onRowClick) {
          tr.addEventListener('click', function (e) {
            var t = e && e.target;
            if (t && t.closest && t.closest('a,button,input,select')) return;
            opts.onRowClick(row);
          });
        }
        cols.forEach(function (c) {
          var cls = ['al-' + colAlign(c), c.num ? 'num' : null, c.nowrap ? 'nowrap' : null, typeof c.cls === 'function' ? c.cls(row) : null];
          tr.appendChild(h(c.rowHeader ? 'th' : 'td', { class: cls, scope: c.rowHeader ? 'row' : null }, cellContent(c, row, i)));
        });
        tbody.appendChild(tr);
      });
    }

    function sortBy(key, dir) {
      state.key = key;
      state.dir = dir === 'desc' ? 'desc' : 'asc';
      renderHead();
      renderBody();
      if (typeof opts.onSort === 'function') opts.onSort(state.key, state.dir);
    }

    wrap.setRows = function (r) { state.rows = arr(r); renderBody(); };
    wrap.getRows = function () { return state.rows; };
    wrap.sortBy = sortBy;
    renderHead();
    renderBody();
    return wrap;
  };

  // ---- verdict / stats --------------------------------------------------------------------------------------------
  ui.verdictBox = function (verdict) {
    var v = isProps(verdict) ? verdict : {};
    var info = F.verdictInfo(v.level);
    var bullets = arr(v.bullets).filter(function (b) { return typeof b === 'string' && b; });
    return h('div', { class: ['verdict', 'verdict-' + (v.level || 'none'), 'tone-' + info.tone], role: 'group', 'aria-label': '驗證結論' },
      h('div', { class: 'verdict-head' },
        h('span', { class: 'verdict-icon', 'aria-hidden': 'true' }, info.icon),
        h('div', { class: 'verdict-titles' },
          h('span', { class: 'verdict-level' }, info.label),
          h('strong', { class: 'verdict-title' }, v.title || '尚無結論'))),
      v.text ? h('p', { class: 'verdict-text' }, v.text) : null,
      bullets.length ? h('ul', { class: 'verdict-bullets' }, bullets.map(function (b) { return h('li', null, b); })) : null);
  };

  var STAT_ROWS = [
    { key: 'total_return', label: '累積報酬', fmt: function (v) { return F.fmtPct(v, 1, true); } },
    { key: 'cagr', label: '年化報酬', fmt: function (v) { return F.fmtPct(v, 1, true); } },
    { key: 'vol', label: '年化波動', fmt: function (v) { return F.fmtPct(v, 1); } },
    { key: 'sharpe', label: '夏普比率', fmt: function (v) { return F.fmtNum(v, 2); }, tip: '每承擔一單位波動所獲得的超額報酬（年化）。越高越好；樣本短時數字很不穩定。' },
    { key: 'sortino', label: '索提諾比率', fmt: function (v) { return F.fmtNum(v, 2); }, tip: '類似夏普比率，但只把「下跌」的波動視為風險。' },
    { key: 'max_dd', label: '最大回撤', fmt: function (v) { return F.fmtPct(v, 1); }, tip: '期間內從高點到低點的最大跌幅。' },
    { key: 'calmar', label: '卡瑪比率', fmt: function (v) { return F.fmtNum(v, 2); }, tip: '年化報酬 ÷ 最大回撤的絕對值。' },
    { key: 'exposure', label: '持股比例', fmt: function (v) { return F.fmtPct(v, 0); }, tip: '期間內持有部位的平均比例（其餘為現金）。' },
    { key: 'turnover_per_year', label: '年換手率', fmt: function (v) { return F.fmtNum(v, 1) + ' 倍'; }, tip: '每年買賣金額相對於資金的倍數；越高，交易成本拖累越大。' },
    { key: 'n_trades', label: '交易次數', fmt: function (v) { return F.fmtInt(v); } },
    { key: 'trade_hit_rate', label: '交易勝率', fmt: function (v) { return F.fmtPct(v, 1); } },
    { key: 'psr', label: '機率夏普 (PSR)', fmt: function (v) { return F.fmtPct(v, 1); }, tip: 'Probabilistic Sharpe Ratio：考慮樣本長度、偏態與峰態後，真實夏普比率大於 0 的機率。' },
    { key: 'dsr', label: '通縮夏普 (DSR)', fmt: function (v) { return F.fmtPct(v, 1); }, tip: 'Deflated Sharpe Ratio：再扣除「試過多組參數而挑到好結果」的運氣成分。' }
  ];

  /** columns: [{label, stats}] -> comparison table; rows where every column is missing are dropped. */
  ui.statsTable = function (columns, opts) {
    opts = opts || {};
    var cs = arr(columns).filter(function (c) { return c && isProps(c.stats); });
    var rows = STAT_ROWS.filter(function (r) { return cs.some(function (c) { return !isMissing(c.stats[r.key]); }); });
    if (!cs.length || !rows.length) return ui.emptyState('沒有可比較的績效統計。');
    var tcols = [{ key: 'label', label: '指標', rowHeader: true, render: function (r) { return h('span', null, r.label, r.tip ? ui.infoTip(r.tip) : null); } }];
    cs.forEach(function (c, i) {
      tcols.push({ key: 'c' + i, label: c.label, num: true, render: function (r) {
        var v = c.stats[r.key];
        // a "win rate" over one single trade (buy-and-hold, an index) is not a statistic: show it as missing
        if (r.key === 'trade_hit_rate' && !(num(c.stats.n_trades) > 1)) return NA;
        return isMissing(v) ? NA : r.fmt(v);
      } });
    });
    return ui.table(tcols, rows, { caption: opts.caption || '績效統計比較', dense: true });
  };

  // ---- async helper ----------------------------------------------------------------------------------------------
  ui.loadInto = function (slot, opts) {
    var seq = 0;
    function run() {
      var my = ++seq;
      clear(slot);
      slot.appendChild(opts.skeleton && isNode(opts.skeleton) ? opts.skeleton : ui.skeleton(opts.skeleton == null ? 4 : opts.skeleton));
      var p;
      try { p = Promise.resolve(opts.load(opts.signal)); } catch (e) { p = Promise.reject(e); }
      p.then(function (data) {
        if (my !== seq || (opts.signal && opts.signal.aborted)) return;
        var out;
        try { out = opts.render(data); } catch (e) {
          logError(e);
          clear(slot);
          slot.appendChild(ui.errorBox(null, run));
          return;
        }
        clear(slot);
        appendKids(slot, [out]);
      }, function (err) {
        if (my !== seq || isAbort(err) || (opts.signal && opts.signal.aborted)) return;
        clear(slot);
        slot.appendChild(ui.errorBox(err, run));
      });
    }
    run();
    return { reload: run };
  };

  // ====================================================================================== chart card
  var DRAW = { priceFan: 'drawPriceFan', timeSeries: 'drawTimeSeries', calibration: 'drawCalibration', histogram: 'drawHistogram',
    gauge: 'drawGauge', strip: 'drawIntervalStrip' };

  TQ.chartCard = function (title, spec, opts) {
    opts = opts || {};
    var reserve = opts.height || (typeof opts.heightFn === 'function' ? opts.heightFn(360) : 240);
    var holder = h('div', { class: 'chart-host', style: { minHeight: reserve + 'px' } });
    var summary = h('p', { class: 'sr-only chart-summary' });
    var note = opts.note ? h('p', { class: 'chart-note' }, opts.note) : null;
    var inner = [holder, summary, note];
    var node = opts.bare ? h('div', { class: ['chart-bare', opts.class] }, inner)
      : ui.card(title, inner, { sub: opts.sub, tip: opts.tip, actions: opts.actions, class: ['chart-card', opts.class], id: opts.id, level: opts.level });
    var curSpec = spec;
    var drawName = DRAW[opts.kind];
    function drawFn(ctx, W, H, pal, hover) {
      var s = typeof curSpec === 'function' ? curSpec(pal, W, H) : curSpec;
      var model;
      if (typeof opts.draw === 'function') model = opts.draw(ctx, W, H, pal, hover, s);
      else if (drawName && TQ.charts && typeof TQ.charts[drawName] === 'function') model = TQ.charts[drawName](ctx, W, H, pal, s, hover);
      else model = TQ.charts.drawEmpty(ctx, W, H, pal, '無法繪製圖表');
      summary.textContent = model && model.a11y ? model.a11y : '';
      return model;
    }
    var ctl = TQ.charts.mount(holder, { draw: drawFn, height: opts.height, heightFn: opts.heightFn, label: opts.label || (typeof title === 'string' ? title : '圖表') });
    node.chart = ctl;
    node.setSpec = function (s) { curSpec = s; ctl.redraw(); };
    node.destroy = function () { ctl.destroy(); };
    return node;
  };

  // ====================================================================================== router
  var routes = {};
  var cur = { name: null, params: {}, inst: null, ctl: null, host: null, hash: null, query: null };
  var rootEls = { view: null, nav: null, status: null, main: null };
  var routerStarted = false;
  var firstRender = true;
  var CODE_RE = /^[0-9A-Za-z^][0-9A-Za-z.^-]{0,11}$/;

  function lastStock() {
    var c = TQ.prefs.get('lastStock', '2330');
    return typeof c === 'string' && CODE_RE.test(c) ? c : '2330';
  }

  /** "a=1&b=%E5%8F%B0" -> { a: '1', b: '台' } (first occurrence wins, malformed pairs and prototype keys are skipped). */
  function parseQuery(qs) {
    var out = {};
    String(qs || '').split('&').forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf('=');
      var k = i < 0 ? kv : kv.slice(0, i), v = i < 0 ? '' : kv.slice(i + 1);
      try { k = decodeURIComponent(k.replace(/\+/g, ' ')); v = decodeURIComponent(v.replace(/\+/g, ' ')); } catch (e) { return; }
      if (!k || k === '__proto__' || k === 'constructor' || k === 'prototype' || Object.prototype.hasOwnProperty.call(out, k)) return;
      out[k] = v;
    });
    return out;
  }
  /** { a: 1, b: null, c: '台' } -> "a=1&c=%E5%8F%B0" (null / undefined / non-finite values are dropped). */
  function buildQuery(obj) {
    return Object.keys(obj || {}).filter(function (k) {
      var v = obj[k];
      return v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && !isFinite(v));
    }).map(function (k) { return encodeURIComponent(k) + '=' + encodeURIComponent(String(obj[k])); }).join('&');
  }

  /** Parse "#/name/segment?query": `canonical` never contains the query part, `query` is the raw (still encoded) query string. */
  function parseHash(hash) {
    var s = String(hash || '').replace(/^#/, '');
    var q = s.indexOf('?');
    var query = '';
    if (q >= 0) { query = s.slice(q + 1); s = s.slice(0, q); }
    var segs = s.split('/').filter(function (x) { return x !== ''; });
    var name = segs[0] || '';
    var res = { name: name, params: {}, canonical: '', redirect: false, query: query };
    function code(seg) {
      if (seg === undefined) return null;
      var c;
      try { c = decodeURIComponent(seg); } catch (e) { return null; }
      c = c.toUpperCase();
      return CODE_RE.test(c) ? c : null;
    }
    if (name === 'stock' || name === 'backtest') {
      var c = code(segs[1]);
      if (!c) { c = lastStock(); res.redirect = true; }
      res.params.code = c;
      res.canonical = '#/' + name + '/' + encodeURIComponent(c);
      if (segs[1] !== undefined && !res.redirect && decodeURIComponent(segs[1]) !== c) res.redirect = true;
    } else if (name === 'picks' || name === 'method') {
      if (name === 'method' && segs[1]) res.params.section = segs[1];
      res.canonical = '#/' + name + (name === 'method' && segs[1] ? '/' + segs[1] : '');
    } else {
      res.name = 'stock';
      res.params.code = lastStock();
      res.canonical = '#/stock/' + encodeURIComponent(res.params.code);
      res.redirect = true;
    }
    return res;
  }

  function setStatusText(text) {
    if (!rootEls.status) return;
    rootEls.status.textContent = text || '';
  }

  function refreshNav() {
    if (!rootEls.nav) return;
    var code = lastStock();
    arr(rootEls.nav.querySelectorAll ? rootEls.nav.querySelectorAll('a') : []).forEach(function (a) {
      var r = a.getAttribute('data-route');
      if (r === 'stock') a.setAttribute('href', '#/stock/' + encodeURIComponent(code));
      else if (r === 'backtest') a.setAttribute('href', '#/backtest/' + encodeURIComponent(code));
      if (r && r === cur.name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
  }

  function teardown() {
    if (cur.ctl) { try { cur.ctl.abort(); } catch (e) { /* ignore */ } }
    if (cur.inst && typeof cur.inst.destroy === 'function') {
      try { cur.inst.destroy(); } catch (e) { logError(e); }
    }
    if (cur.host && cur.host.parentNode) cur.host.parentNode.removeChild(cur.host);
    cur.inst = null; cur.ctl = null; cur.host = null;
  }

  function navigate(hash) {
    var target = String(hash || '');
    if (target.charAt(0) !== '#') target = '#' + target;
    if (root.location.hash === target) { render(); return; }
    root.location.hash = target;
  }

  function render() {
    var r = parseHash(root.location.hash);
    var pathNow = String(root.location.hash || '').split('?')[0];
    if (r.redirect || pathNow !== r.canonical) {
      var fixed = r.canonical + (r.query ? '?' + r.query : '');       // canonicalise the path, keep the query (shareable view settings)
      try { root.history.replaceState(null, '', fixed); } catch (e) { root.location.hash = fixed; }
    }
    if (cur.hash === r.canonical && cur.query === r.query && cur.inst) return;
    teardown();
    var view = routes[r.name] || TQ.views[r.name];
    var host = h('div', { class: 'view-host', dataset: { view: r.name } });
    var viewRoot = rootEls.view;
    if (viewRoot) { clear(viewRoot); viewRoot.appendChild(host); }
    var AC = root.AbortController;
    var ctl = AC ? new AC() : { signal: { aborted: false, addEventListener: function () {}, removeEventListener: function () {} }, abort: function () { this.signal.aborted = true; } };
    cur = { name: r.name, params: r.params, inst: null, ctl: ctl, host: host, hash: r.canonical, query: r.query };
    var ctx = {
      signal: ctl.signal, route: r.name,
      nav: navigate,
      query: parseQuery(r.query),
      setStatus: setStatusText,
      setTitle: function (t) { if (ctl.signal.aborted) return; try { doc().title = (t ? t + ' · ' : '') + TQ.APP_NAME; } catch (e) { /* ignore */ } },
      /** Refine the URL of the CURRENT view (same route name, e.g. add "?h=5&thr=0.55" or "#/method/garch") without a hashchange and
       *  without re-mounting it.  Returns false when the view was replaced meanwhile, the hash belongs to another route, or the browser refuses. */
      replace: function (hash) {
        if (ctl.signal.aborted || cur.ctl !== ctl) return false;
        var t = String(hash || '');
        if (t.charAt(0) !== '#') t = '#' + t;
        var p = parseHash(t);
        if (p.name !== r.name || p.redirect) return false;
        try { root.history.replaceState(null, '', p.canonical + (p.query ? '?' + p.query : '')); } catch (e) { return false; }
        cur.hash = p.canonical; cur.query = p.query; cur.params = p.params;
        return true;
      }
    };
    refreshNav();
    try {
      if (!view) throw new Error('unknown view ' + r.name);
      var title = typeof view.title === 'function' ? view.title(r.params) : '';
      ctx.setTitle(title);
      cur.inst = view.mount(host, r.params, ctx) || {};
    } catch (e) {
      logError(e);
      clear(host);
      host.appendChild(ui.errorBox(null, function () { cur.hash = null; render(); }));
    }
    setStatusText('已切換至「' + (view && typeof view.title === 'function' ? view.title(r.params) : r.name) + '」');
    if (!firstRender) {
      try { root.scrollTo(0, 0); } catch (e) { /* ignore */ }
      if (rootEls.main && rootEls.main.focus) { try { rootEls.main.focus({ preventScroll: true }); } catch (e) { rootEls.main.focus(); } }
    }
    firstRender = false;
  }

  TQ.router = {
    register: function (name, view) { routes[name] = view; },
    start: function (els) {
      var d = doc();
      els = els || {};
      rootEls.view = els.view || d.getElementById('view');
      rootEls.nav = els.nav || d.getElementById('nav');
      rootEls.status = els.status || d.getElementById('status');
      rootEls.main = els.main || d.getElementById('main');
      Object.keys(TQ.views).forEach(function (k) { if (!routes[k]) routes[k] = TQ.views[k]; });
      if (!routerStarted) {
        routerStarted = true;
        root.addEventListener('hashchange', render);
      }
      firstRender = true;
      render();
    },
    navigate: navigate, parse: parseHash, parseQuery: parseQuery, buildQuery: buildQuery,
    current: function () { return { name: cur.name, params: cur.params, hash: cur.hash, query: cur.query }; },
    refreshNav: refreshNav, setLastStock: function (code) { if (typeof code === 'string' && CODE_RE.test(code)) { TQ.prefs.set('lastStock', code); refreshNav(); } },
    _reset: function () { teardown(); cur = { name: null, params: {}, inst: null, ctl: null, host: null, hash: null, query: null }; routerStarted = false; routes = {}; }
  };

  // ====================================================================================== search box
  var CODE_LIKE = /^\d{4,6}[A-Za-z]?$/;
  var POPULAR_NOTE = '熱門與常用標的';

  function searchIcon() {
    return h('svg', { class: 'search-icon', viewBox: '0 0 20 20', width: '16', height: '16', 'aria-hidden': 'true', focusable: 'false' },
      h('circle', { cx: '8.5', cy: '8.5', r: '5.5', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.8' }),
      h('path', { d: 'M12.8 12.8 17 17', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.8', 'stroke-linecap': 'round' }));
  }

  function createSearch(opts) {
    opts = opts || {};
    var inputId = opts.id || uid('sb-input');
    var listId = uid('sb-list');
    var state = { results: [], active: -1, open: false, loading: false, error: null, q: '', loadedQ: null };
    var timer = null, ctrl = null, seq = 0, blurTimer = null, destroyed = false;

    var input = h('input', { id: inputId, class: 'search-input', type: 'text', role: 'combobox', autocomplete: 'off', spellcheck: 'false',
      autocapitalize: 'off', inputmode: 'search', enterkeyhint: 'search', placeholder: opts.placeholder || '搜尋代號或名稱，例如 2330、台積電',
      'aria-expanded': 'false', 'aria-controls': listId, 'aria-autocomplete': 'list', 'aria-haspopup': 'listbox' });
    var label = h('label', { class: 'sr-only', for: inputId }, opts.label || '搜尋股票或 ETF（代號或名稱）');
    var clearBtn = h('button', { type: 'button', class: 'search-clear', 'aria-label': '清除搜尋', hidden: true, tabindex: '-1' }, '×');
    var list = h('ul', { class: 'search-list', role: 'listbox', id: listId, 'aria-label': '搜尋結果' });
    var msg = h('div', { class: 'search-msg', role: 'status', 'aria-live': 'polite' });
    var pop = h('div', { class: 'search-pop', hidden: true }, list, msg);
    var el = h('div', { class: 'search', role: 'search' }, label, h('div', { class: 'search-field' }, searchIcon(), input, clearBtn), pop);

    function setOpen(v) {
      state.open = v;
      pop.hidden = !v;
      input.setAttribute('aria-expanded', v ? 'true' : 'false');
      if (!v) input.removeAttribute('aria-activedescendant');
    }

    function optId(i) { return listId + '-o' + i; }

    function paint() {
      clear(list);
      state.results.forEach(function (it, i) {
        var on = i === state.active;
        var li = h('li', { class: ['search-opt', on ? 'active' : null], role: 'option', id: optId(i), 'aria-selected': on ? 'true' : 'false',
          onMousedown: function (e) { if (e.preventDefault) e.preventDefault(); },
          onClick: function () { choose(it); },
          onMousemove: function () { if (state.active !== i) { state.active = i; paintActive(); } } },
          h('span', { class: 'search-code' }, it.code),
          h('span', { class: 'search-name' }, it.name || NA),
          h('span', { class: 'search-tags' },
            it.market ? ui.chip(F.marketName(it.market), 'muted') : null,
            it.kind === 'etf' ? ui.chip('ETF', 'info') : null));
        list.appendChild(li);
      });
      var text = '';
      if (state.loading && !state.results.length) text = '搜尋中…';
      else if (state.error) text = '搜尋失敗：' + state.error + '（可直接輸入完整代號後按 Enter）';
      else if (!state.results.length && state.loadedQ !== null && state.loadedQ !== '') {
        text = '找不到符合「' + state.loadedQ + '」的股票或 ETF。' + (CODE_LIKE.test(state.loadedQ) ? ' 按 Enter 直接前往代號 ' + state.loadedQ.toUpperCase() + '。' : '');
      } else if (state.results.length && state.loadedQ === '') text = POPULAR_NOTE;
      else if (state.results.length) text = '共 ' + state.results.length + ' 筆結果，使用 ↑ ↓ 選擇、Enter 開啟';
      msg.textContent = text;
      msg.hidden = !text;
      paintActive();
    }

    function paintActive() {
      var opts2 = list.childNodes;
      for (var i = 0; i < opts2.length; i++) {
        var on = i === state.active;
        opts2[i].setAttribute('aria-selected', on ? 'true' : 'false');
        opts2[i].classList.toggle('active', on);
      }
      if (state.active >= 0 && state.results.length) {
        input.setAttribute('aria-activedescendant', optId(state.active));
        var node = opts2[state.active];
        if (node && node.scrollIntoView) { try { node.scrollIntoView({ block: 'nearest' }); } catch (e) { /* ignore */ } }
      } else input.removeAttribute('aria-activedescendant');
    }

    function normalize(rs) {
      return arr(rs).filter(function (r) { return r && typeof r.code === 'string' && r.code; }).map(function (r) {
        return { code: r.code, name: typeof r.name === 'string' ? r.name : '', market: r.market || '', kind: r.kind || 'stock' };
      });
    }

    function fetchNow(q) {
      if (timer) { root.clearTimeout(timer); timer = null; }
      if (ctrl) { try { ctrl.abort(); } catch (e) { /* ignore */ } }
      var AC = root.AbortController;
      ctrl = AC ? new AC() : null;
      var my = ++seq;
      state.loading = true;
      state.error = null;
      paint();
      // typed query: a short list is enough; empty query: the server's popular list is 12 universe names + 3 ETFs, so do not cut it at 12
      return apiGet('/api/search', { q: q, limit: q ? 12 : 20 }, { signal: ctrl ? ctrl.signal : undefined }).then(function (js) {
        if (my !== seq || destroyed) return null;
        state.loading = false;
        state.results = normalize(js.results);
        state.loadedQ = q;
        state.active = q && state.results.length ? 0 : -1;
        paint();
        return state.results;
      }, function (err) {
        if (isAbort(err) || my !== seq || destroyed) return null;
        state.loading = false;
        state.results = [];
        state.loadedQ = q;
        state.active = -1;
        state.error = err && err.message ? err.message : '無法取得搜尋結果';
        paint();
        return null;
      });
    }

    function schedule() {
      if (timer) root.clearTimeout(timer);
      var q = state.q;
      timer = root.setTimeout(function () { timer = null; fetchNow(q); }, 180);
    }

    function choose(item) {
      if (!item) return;
      setOpen(false);
      input.value = '';
      state.q = '';
      clearBtn.hidden = true;
      if (typeof opts.onSelect === 'function') opts.onSelect(item);
      else TQ.router.navigate(codeHref(item.code));
      if (opts.blurOnSelect !== false && input.blur) input.blur();
    }

    function submit() {
      var q = state.q.trim();
      if (timer || state.loadedQ !== q) {
        // results are stale or still pending: search now, then take the first hit (or the typed code)
        fetchNow(q).then(function (rs) {
          if (destroyed) return;
          if (rs && rs.length) choose(rs[0]);
          else if (CODE_LIKE.test(q)) choose({ code: q.toUpperCase(), name: '', market: '', kind: 'stock' });
        });
        return;
      }
      if (state.active >= 0 && state.results[state.active]) choose(state.results[state.active]);
      else if (state.results.length && q) choose(state.results[0]);
      else if (CODE_LIKE.test(q)) choose({ code: q.toUpperCase(), name: '', market: '', kind: 'stock' });
    }

    input.addEventListener('input', function () {
      state.q = input.value;
      clearBtn.hidden = !input.value;
      setOpen(true);
      if (input.value.trim() === '' && state.loadedQ === '' && state.results.length) { state.active = -1; paint(); return; }
      schedule();
    });
    input.addEventListener('focus', function () {
      if (blurTimer) { root.clearTimeout(blurTimer); blurTimer = null; }
      setOpen(true);
      if (state.loadedQ === null) fetchNow(state.q.trim()); else paint();
    });
    input.addEventListener('blur', function () {
      blurTimer = root.setTimeout(function () { blurTimer = null; if (!destroyed) setOpen(false); }, 150);
    });
    input.addEventListener('keydown', function (e) {
      var k = e.key;
      if (k === 'ArrowDown' || k === 'ArrowUp') {
        if (e.preventDefault) e.preventDefault();
        if (!state.open) setOpen(true);
        var n = state.results.length;
        if (!n) return;
        state.active = k === 'ArrowDown' ? (state.active + 1) % n : (state.active <= 0 ? n - 1 : state.active - 1);
        paintActive();
      } else if (k === 'Enter') {
        if (e.preventDefault) e.preventDefault();
        submit();
      } else if (k === 'Escape') {
        if (state.open) { setOpen(false); if (e.stopPropagation) e.stopPropagation(); }
        else if (input.value) { input.value = ''; state.q = ''; clearBtn.hidden = true; }
      } else if (k === 'Tab') {
        setOpen(false);
      }
    });
    clearBtn.addEventListener('click', function () {
      input.value = '';
      state.q = '';
      clearBtn.hidden = true;
      if (input.focus) input.focus();
      setOpen(true);
      fetchNow('');
    });

    var api = {
      el: el, input: input,
      focus: function () { input.focus(); },
      setValue: function (v) { input.value = v || ''; state.q = input.value; clearBtn.hidden = !input.value; },
      destroy: function () {
        destroyed = true;
        if (timer) root.clearTimeout(timer);
        if (blurTimer) root.clearTimeout(blurTimer);
        if (ctrl) { try { ctrl.abort(); } catch (e) { /* ignore */ } }
        if (el.parentNode) el.parentNode.removeChild(el);
      }
    };
    return api;
  }

  TQ.searchBox = {
    create: createSearch,
    mount: function (host, opts) {
      var sb = createSearch(opts);
      host.appendChild(sb.el);
      return sb;
    }
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
