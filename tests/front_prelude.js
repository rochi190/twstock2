/*
 * front_prelude.js -- minimal browser fakes for testing the frontend inside bare V8 (py_mini_racer).
 *
 * Provides: window/self, console capture, a manual clock (setTimeout & co., advanced with __advance(ms)),
 * a fake DOM (documents, elements, text nodes, events, selectors), a recording Canvas2D context,
 * localStorage, location/history with hashchange, matchMedia, AbortController, URLSearchParams, Event,
 * and a routable fetch().  Nothing here is shipped to users.
 */
var window = this;
window.window = window;
window.self = window;
window.globalThis = window;

// ------------------------------------------------------------------------------------------ console
window.__console = [];
window.console = {
  log: function () { window.__console.push(['log', Array.prototype.slice.call(arguments).map(String).join(' ')]); },
  info: function () { window.__console.push(['info', Array.prototype.slice.call(arguments).map(String).join(' ')]); },
  warn: function () { window.__console.push(['warn', Array.prototype.slice.call(arguments).map(String).join(' ')]); },
  error: function () { window.__console.push(['error', Array.prototype.slice.call(arguments).map(function (a) { return (a && a.stack) ? a.stack : String(a); }).join(' ')]); }
};

// ------------------------------------------------------------------------------------------ manual clock
window.__now = 0;
window.__timers = [];
window.__timerSeq = 1;
window.setTimeout = function (fn, ms) {
  var id = window.__timerSeq++;
  window.__timers.push({ id: id, at: window.__now + Math.max(0, +ms || 0), fn: fn, args: Array.prototype.slice.call(arguments, 2), every: 0 });
  return id;
};
window.setInterval = function (fn, ms) {
  var id = window.__timerSeq++;
  window.__timers.push({ id: id, at: window.__now + Math.max(1, +ms || 1), fn: fn, args: [], every: Math.max(1, +ms || 1) });
  return id;
};
window.clearTimeout = window.clearInterval = function (id) {
  window.__timers = window.__timers.filter(function (t) { return t.id !== id; });
};
window.requestAnimationFrame = function (fn) { return window.setTimeout(function () { fn(window.__now); }, 16); };
window.cancelAnimationFrame = window.clearTimeout;
window.performance = { now: function () { return window.__now; } };
/** Advance the fake clock by ms, running due timers in order (timers may schedule more timers). */
window.__advance = function (ms) {
  var end = window.__now + ms, guard = 0;
  while (guard++ < 100000) {
    var next = null;
    for (var i = 0; i < window.__timers.length; i++) {
      var t = window.__timers[i];
      if (t.at <= end && (next === null || t.at < next.at || (t.at === next.at && t.id < next.id))) next = t;
    }
    if (!next) break;
    window.__now = Math.max(window.__now, next.at);
    if (next.every) next.at += next.every; else window.__timers = window.__timers.filter(function (t) { return t !== next; });
    next.fn.apply(null, next.args);
  }
  window.__now = end;
};
window.__pendingTimers = function () { return window.__timers.length; };

// ------------------------------------------------------------------------------------------ events
function FEvent(type, init) {
  init = init || {};
  this.type = type;
  this.bubbles = !!init.bubbles;
  this.cancelable = !!init.cancelable;
  this.defaultPrevented = false;
  this.target = null;
  this.currentTarget = null;
  this._stop = false;
  for (var k in init) if (Object.prototype.hasOwnProperty.call(init, k)) this[k] = init[k];
}
FEvent.prototype.preventDefault = function () { this.defaultPrevented = true; };
FEvent.prototype.stopPropagation = function () { this._stop = true; };
FEvent.prototype.stopImmediatePropagation = function () { this._stop = true; };
window.Event = FEvent;
window.CustomEvent = FEvent;
window.KeyboardEvent = FEvent;
window.MouseEvent = FEvent;

function addL(self, type, fn, opts) {
  if (typeof fn !== 'function') return;
  (self._l[type] = self._l[type] || []).push(fn);
}
function rmL(self, type, fn) {
  var a = self._l[type];
  if (!a) return;
  var i = a.indexOf(fn);
  if (i >= 0) a.splice(i, 1);
}
function fire(self, ev) {
  ev.currentTarget = self;
  var a = (self._l[ev.type] || []).slice();
  for (var i = 0; i < a.length; i++) a[i].call(self, ev);
  var h = self['on' + ev.type];
  if (typeof h === 'function') h.call(self, ev);
}

// ------------------------------------------------------------------------------------------ recording canvas context
var CTX_METHODS = ['save', 'restore', 'beginPath', 'moveTo', 'lineTo', 'closePath', 'arc', 'rect', 'fill', 'stroke',
  'fillRect', 'strokeRect', 'clearRect', 'fillText', 'strokeText', 'setLineDash', 'setTransform', 'clip',
  'quadraticCurveTo', 'bezierCurveTo', 'translate', 'scale'];
var CTX_PROPS = ['fillStyle', 'strokeStyle', 'lineWidth', 'lineCap', 'lineJoin', 'font', 'textAlign', 'textBaseline', 'globalAlpha'];
window.__ctxAllowed = CTX_METHODS.concat(['measureText', 'createLinearGradient']);

function measureWidth(str, font) {
  var m = /(\d+(?:\.\d+)?)px/.exec(font || '');
  var size = m ? +m[1] : 10;
  var w = 0;
  for (var i = 0; i < str.length; i++) {
    var c = str.charCodeAt(i);
    if (c >= 0x2E80) w += size;                       // CJK / full-width
    else if (c === 32) w += size * 0.3;
    else if (c >= 48 && c <= 57) w += size * 0.58;    // digits
    else if (c === 46 || c === 44 || c === 58) w += size * 0.3;
    else w += size * 0.56;
  }
  return w;
}
window.__measureWidth = measureWidth;

function RecCtx(canvas) {
  var self = this;
  this.canvas = canvas;
  this.ops = [];
  this.unknown = [];
  this._p = { fillStyle: '#000000', strokeStyle: '#000000', lineWidth: 1, lineCap: 'butt', lineJoin: 'miter', font: '10px sans-serif', textAlign: 'start', textBaseline: 'alphabetic', globalAlpha: 1 };
  this._stack = [];
  CTX_PROPS.forEach(function (name) {
    Object.defineProperty(self, name, {
      get: function () { return self._p[name]; },
      set: function (v) { self._p[name] = v; self.ops.push(['set', name, v]); }
    });
  });
  CTX_METHODS.forEach(function (name) {
    self[name] = function () {
      var args = Array.prototype.slice.call(arguments);
      if (name === 'save') self._stack.push(JSON.stringify(self._p));
      if (name === 'restore') { var s = self._stack.pop(); if (s) self._p = JSON.parse(s); }
      self.ops.push([name].concat(args));
    };
  });
}
RecCtx.prototype.__reset = function () {
  this.ops = [];
  this._stack = [];
  this._p = { fillStyle: '#000000', strokeStyle: '#000000', lineWidth: 1, lineCap: 'butt', lineJoin: 'miter', font: '10px sans-serif', textAlign: 'start', textBaseline: 'alphabetic', globalAlpha: 1 };
};
RecCtx.prototype.measureText = function (s) { return { width: measureWidth(String(s), this._p.font) }; };
RecCtx.prototype.createLinearGradient = function () { return { addColorStop: function () {} }; };

// ------------------------------------------------------------------------------------------ fake DOM
var __classLog = {};
window.__classLog = __classLog;
function logClasses(str) {
  String(str || '').split(/\s+/).forEach(function (c) { if (c) __classLog[c] = (__classLog[c] || 0) + 1; });
}

function FNode(type) {
  this.nodeType = type;
  this.parentNode = null;
  this.childNodes = [];
  this._l = {};
}
FNode.prototype.addEventListener = function (t, f, o) { addL(this, t, f, o); };
FNode.prototype.removeEventListener = function (t, f) { rmL(this, t, f); };
FNode.prototype.dispatchEvent = function (ev) {
  ev.target = ev.target || this;
  var n = this;
  while (n) {
    fire(n, ev);
    if (ev._stop || !ev.bubbles) break;
    n = n.parentNode || (n === window.document.documentElement ? window.document : null);
  }
  return !ev.defaultPrevented;
};
Object.defineProperty(FNode.prototype, 'firstChild', { get: function () { return this.childNodes[0] || null; } });
Object.defineProperty(FNode.prototype, 'lastChild', { get: function () { return this.childNodes[this.childNodes.length - 1] || null; } });
Object.defineProperty(FNode.prototype, 'nextSibling', { get: function () { if (!this.parentNode) return null; var a = this.parentNode.childNodes; return a[a.indexOf(this) + 1] || null; } });
Object.defineProperty(FNode.prototype, 'previousSibling', { get: function () { if (!this.parentNode) return null; var a = this.parentNode.childNodes; return a[a.indexOf(this) - 1] || null; } });
FNode.prototype.appendChild = function (c) {
  if (c === null || c === undefined || typeof c !== 'object') throw new TypeError('appendChild: not a Node');
  if (c.parentNode) c.parentNode.removeChild(c);
  c.parentNode = this;
  this.childNodes.push(c);
  this._html = undefined;
  return c;
};
FNode.prototype.insertBefore = function (c, ref) {
  if (!ref) return this.appendChild(c);
  if (c.parentNode) c.parentNode.removeChild(c);
  var i = this.childNodes.indexOf(ref);
  if (i < 0) throw new Error('insertBefore: ref not a child');
  c.parentNode = this;
  this.childNodes.splice(i, 0, c);
  return c;
};
FNode.prototype.removeChild = function (c) {
  var i = this.childNodes.indexOf(c);
  if (i < 0) throw new Error('removeChild: not a child');
  this.childNodes.splice(i, 1);
  c.parentNode = null;
  return c;
};
FNode.prototype.remove = function () { if (this.parentNode) this.parentNode.removeChild(this); };
FNode.prototype.contains = function (n) { while (n) { if (n === this) return true; n = n.parentNode; } return false; };
FNode.prototype.replaceChildren = function () {
  while (this.childNodes.length) this.removeChild(this.childNodes[0]);
  for (var i = 0; i < arguments.length; i++) {
    var a = arguments[i];
    this.appendChild(typeof a === 'string' ? new FText(a) : a);
  }
};

function FText(s) { FNode.call(this, 3); this.data = String(s); }
FText.prototype = Object.create(FNode.prototype);
Object.defineProperty(FText.prototype, 'textContent', { get: function () { return this.data; }, set: function (v) { this.data = String(v); } });
Object.defineProperty(FText.prototype, 'nodeValue', { get: function () { return this.data; } });

function Style() { this._s = {}; }
Style.prototype.setProperty = function (k, v) { this[k] = v; };
Style.prototype.removeProperty = function (k) { delete this[k]; };
Style.prototype.getPropertyValue = function (k) { return this[k] || ''; };

function FElement(tag) {
  FNode.call(this, 1);
  this.localName = String(tag).toLowerCase();
  this.tagName = String(tag).toUpperCase();
  this._attrs = {};
  this._cls = '';
  this.style = new Style();
  this.hidden = false;
  this.disabled = false;
  this.value = '';
  this.checked = false;
  this._w = undefined;
  this._h = undefined;
  this._cw = 300;
  this._ch = 150;
  this._ctx = null;
  var self = this;
  this.classList = {
    add: function () { for (var i = 0; i < arguments.length; i++) { if (!self.classList.contains(arguments[i])) self.className = (self._cls + ' ' + arguments[i]).trim(); } },
    remove: function () { var rm = Array.prototype.slice.call(arguments); self.className = self._cls.split(/\s+/).filter(function (c) { return c && rm.indexOf(c) < 0; }).join(' '); },
    contains: function (c) { return self._cls.split(/\s+/).indexOf(c) >= 0; },
    toggle: function (c, force) {
      var has = self.classList.contains(c);
      var want = force === undefined ? !has : !!force;
      if (want && !has) self.classList.add(c);
      if (!want && has) self.classList.remove(c);
      return want;
    }
  };
}
FElement.prototype = Object.create(FNode.prototype);
Object.defineProperty(FElement.prototype, 'className', {
  get: function () { return this._cls; },
  set: function (v) { this._cls = String(v); logClasses(v); }
});
Object.defineProperty(FElement.prototype, 'id', {
  get: function () { return this._attrs.id || ''; },
  set: function (v) { this._attrs.id = String(v); }
});
Object.defineProperty(FElement.prototype, 'children', { get: function () { return this.childNodes.filter(function (n) { return n.nodeType === 1; }); } });
Object.defineProperty(FElement.prototype, 'textContent', {
  get: function () { if (this._html !== undefined && !this.childNodes.length) return this._html.replace(/<[^>]*>/g, ''); return this.childNodes.map(function (n) { return n.textContent; }).join(''); },
  set: function (v) { this.replaceChildren(); this._html = undefined; if (v !== '' && v !== null && v !== undefined) this.appendChild(new FText(v)); }
});
Object.defineProperty(FElement.prototype, 'innerHTML', {
  get: function () { return this._html !== undefined ? this._html : this.childNodes.map(function (n) { return n.nodeType === 3 ? n.data : n.outerHTML; }).join(''); },
  set: function (v) { this.replaceChildren(); this._html = String(v); }
});
Object.defineProperty(FElement.prototype, 'outerHTML', {
  get: function () { return '<' + this.localName + '>' + this.innerHTML + '</' + this.localName + '>'; }
});
Object.defineProperty(FElement.prototype, 'clientWidth', {
  get: function () { if (this._w !== undefined) return this._w; return this.parentNode && this.parentNode.clientWidth !== undefined ? this.parentNode.clientWidth : 800; },
  set: function (v) { this._w = v; }
});
Object.defineProperty(FElement.prototype, 'clientHeight', { get: function () { return this._h !== undefined ? this._h : 300; }, set: function (v) { this._h = v; } });
Object.defineProperty(FElement.prototype, 'offsetWidth', { get: function () { return this.clientWidth; } });
Object.defineProperty(FElement.prototype, 'offsetHeight', { get: function () { return this.clientHeight; } });
Object.defineProperty(FElement.prototype, 'dataset', { get: function () { throw new Error('dataset is not supported by the fake DOM: use get/setAttribute("data-...")'); } });
FElement.prototype.setAttribute = function (k, v) {
  v = String(v);
  if (k === 'class') { this.className = v; return; }
  if (k === 'hidden') { this.hidden = true; }
  this._attrs[k] = v;
};
FElement.prototype.getAttribute = function (k) {
  if (k === 'class') return this._cls || null;
  if (k === 'hidden') return this.hidden ? '' : null;
  return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null;
};
FElement.prototype.hasAttribute = function (k) { return this.getAttribute(k) !== null; };
FElement.prototype.removeAttribute = function (k) { if (k === 'class') this._cls = ''; else if (k === 'hidden') this.hidden = false; else delete this._attrs[k]; };
FElement.prototype.focus = function () { if (window.document.activeElement !== this) { window.document.activeElement = this; this.dispatchEvent(new FEvent('focus')); this.dispatchEvent(new FEvent('focusin', { bubbles: true })); } };
FElement.prototype.blur = function () { if (window.document.activeElement === this) { window.document.activeElement = window.document.body; this.dispatchEvent(new FEvent('blur')); } };
FElement.prototype.click = function () { this.dispatchEvent(new FEvent('click', { bubbles: true })); };
FElement.prototype.scrollIntoView = function (opts) {
  window.__scrolled = this;
  (window.__scrollLog = window.__scrollLog || []).push({ id: this.id || '', cls: this._cls, opts: opts === undefined ? null : opts });
};
FElement.prototype.select = function () {};
FElement.prototype.setSelectionRange = function () {};
FElement.prototype.getBoundingClientRect = function () {
  var w = this.clientWidth, h = this.clientHeight;
  return { left: 0, top: 0, right: w, bottom: h, width: w, height: h, x: 0, y: 0 };
};
FElement.prototype.getContext = function (kind) {
  if (this.localName !== 'canvas' || kind !== '2d') return null;
  if (!this._ctx) this._ctx = new RecCtx(this);
  return this._ctx;
};
FElement.prototype.__setSize = function (w, h) { this._w = w; this._h = h; };
// canvas.width / canvas.height: assigning either RESETS the 2D context (as in browsers)
['width', 'height'].forEach(function (k) {
  Object.defineProperty(FElement.prototype, k, {
    get: function () { return k === 'width' ? this._cw : this._ch; },
    set: function (v) {
      if (k === 'width') this._cw = +v; else this._ch = +v;
      if (this._ctx) this._ctx.__reset();
    }
  });
});

// ---- selectors: compound (tag#id.class[attr=val]) with descendant (space) and child (>) combinators
function parseCompound(s) {
  var c = { tag: null, id: null, classes: [], attrs: [] };
  var re = /^([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w:-]+)(?:([~|^$*]?=)"?([^\]"]*)"?)?\]/;
  var rest = s, m;
  while (rest.length && (m = re.exec(rest))) {
    if (m[1]) c.tag = m[1].toLowerCase();
    else if (m[2]) c.id = m[2];
    else if (m[3]) c.classes.push(m[3]);
    else if (m[4]) c.attrs.push({ k: m[4], op: m[5] || null, v: m[6] });
    rest = rest.slice(m[0].length);
  }
  return c;
}
function matchCompound(el, c) {
  if (el.nodeType !== 1) return false;
  if (c.tag && el.localName !== c.tag) return false;
  if (c.id && el.id !== c.id) return false;
  for (var i = 0; i < c.classes.length; i++) if (!el.classList.contains(c.classes[i])) return false;
  for (var j = 0; j < c.attrs.length; j++) {
    var a = c.attrs[j], v = el.getAttribute(a.k);
    if (v === null) return false;
    if (a.op === '=' && v !== a.v) return false;
    if (a.op === '^=' && v.indexOf(a.v) !== 0) return false;
    if (a.op === '*=' && v.indexOf(a.v) < 0) return false;
  }
  return true;
}
function parseSelector(sel) {
  return sel.split(',').map(function (part) {
    var toks = part.trim().replace(/\s*>\s*/g, ' > ').split(/\s+/), out = [], comb = ' ';
    toks.forEach(function (t) { if (t === '>') comb = '>'; else { out.push({ comb: comb, c: parseCompound(t) }); comb = ' '; } });
    return out;
  });
}
function matchesSel(el, chain, idx) {
  var part = chain[idx];
  if (!matchCompound(el, part.c)) return false;
  if (idx === 0) return true;
  if (part.comb === '>') return !!el.parentNode && matchesSel(el.parentNode, chain, idx - 1);
  var p = el.parentNode;
  while (p) { if (matchesSel(p, chain, idx - 1)) return true; p = p.parentNode; }
  return false;
}
function matchesAny(el, sel) { return parseSelector(sel).some(function (chain) { return matchesSel(el, chain, chain.length - 1); }); }
function walk(node, fn) {
  for (var i = 0; i < node.childNodes.length; i++) {
    var ch = node.childNodes[i];
    if (ch.nodeType === 1) { if (fn(ch) === false) return false; if (walk(ch, fn) === false) return false; }
  }
  return true;
}
FElement.prototype.matches = function (sel) { return matchesAny(this, sel); };
FElement.prototype.closest = function (sel) { var n = this; while (n && n.nodeType === 1) { if (matchesAny(n, sel)) return n; n = n.parentNode; } return null; };
FElement.prototype.querySelectorAll = function (sel) { var out = []; var self = this; walk(this, function (e) { if (matchesAny(e, sel)) out.push(e); }); return out; };
FElement.prototype.querySelector = function (sel) { return this.querySelectorAll(sel)[0] || null; };

function FDocument() {
  FNode.call(this, 9);
  this.documentElement = new FElement('html');
  this.head = new FElement('head');
  this.body = new FElement('body');
  this.documentElement.appendChild(this.head);
  this.documentElement.appendChild(this.body);
  this.documentElement.parentNode = null;
  this.activeElement = this.body;
  this.readyState = 'complete';
  this.hidden = false;
  this.title = '';
}
FDocument.prototype = Object.create(FNode.prototype);
FDocument.prototype.createElement = function (t) { return new FElement(t); };
FDocument.prototype.createElementNS = function (ns, t) { var e = new FElement(t); e.namespaceURI = ns; return e; };
FDocument.prototype.createTextNode = function (s) { return new FText(s); };
FDocument.prototype.getElementById = function (id) {
  var found = null;
  walk(this.documentElement, function (e) { if (e.id === id) { found = e; return false; } });
  return found;
};
FDocument.prototype.querySelectorAll = function (sel) { return this.documentElement.querySelectorAll(sel); };
FDocument.prototype.querySelector = function (sel) { return this.documentElement.querySelector(sel); };
window.document = new FDocument();

/** Build the static page skeleton from a JSON tree produced by tests/front_helpers.html_to_tree(). */
window.__buildDom = function (tree) {
  function build(n) {
    if (n.t !== undefined) return new FText(n.t);
    var e = new FElement(n.tag);
    Object.keys(n.attrs || {}).forEach(function (k) { e.setAttribute(k, n.attrs[k]); });
    (n.children || []).forEach(function (c) { e.appendChild(build(c)); });
    return e;
  }
  var body = window.document.body;
  body.replaceChildren();
  (tree.body || []).forEach(function (n) { body.appendChild(build(n)); });
  var head = window.document.head;
  head.replaceChildren();
  (tree.head || []).forEach(function (n) { head.appendChild(build(n)); });
  var attrs = tree.htmlAttrs || {};
  Object.keys(attrs).forEach(function (k) { window.document.documentElement.setAttribute(k, attrs[k]); });
};

window.__textOf = function (node) { return node ? node.textContent : ''; };
window.__all = function (node, sel) { return node.querySelectorAll(sel); };

// ------------------------------------------------------------------------------------------ location / history
window.__locHash = '';
window.location = {
  get hash() { return window.__locHash; },
  set hash(v) {
    v = String(v);
    if (v.charAt(0) !== '#') v = '#' + v;
    if (v === window.__locHash) return;
    var old = window.__locHash;
    window.__locHash = v;
    window.setTimeout(function () { fireWin(new FEvent('hashchange', { oldURL: old, newURL: v })); }, 0);
  },
  get href() { return 'http://localhost:8000/' + window.__locHash; },
  get pathname() { return '/'; },
  get search() { return ''; },
  origin: 'http://localhost:8000'
};
window.history = {
  replaceState: function (s, t, url) { if (typeof url === 'string') { var i = url.indexOf('#'); window.__locHash = i >= 0 ? url.slice(i) : ''; } },
  pushState: function (s, t, url) { if (typeof url === 'string') { var i = url.indexOf('#'); window.__locHash = i >= 0 ? url.slice(i) : ''; } }
};

// window-level listeners
window._l = {};
window.addEventListener = function (t, f, o) { addL(window, t, f, o); };
window.removeEventListener = function (t, f) { rmL(window, t, f); };
function fireWin(ev) { ev.target = ev.target || window; fire(window, ev); }
window.dispatchEvent = function (ev) { fireWin(ev); return !ev.defaultPrevented; };

// ------------------------------------------------------------------------------------------ storage etc.
window.__ls = {};
window.__lsThrow = false;
window.localStorage = {
  getItem: function (k) { if (window.__lsThrow) throw new Error('SecurityError'); return Object.prototype.hasOwnProperty.call(window.__ls, k) ? window.__ls[k] : null; },
  setItem: function (k, v) { if (window.__lsThrow) throw new Error('QuotaExceededError'); window.__ls[k] = String(v); },
  removeItem: function (k) { if (window.__lsThrow) throw new Error('SecurityError'); delete window.__ls[k]; }
};
window.sessionStorage = window.localStorage;
window.__prefersDark = false;
window.matchMedia = function (q) {
  return { matches: /dark/.test(q) ? window.__prefersDark : false, media: q, addEventListener: function () {}, removeEventListener: function () {}, addListener: function () {}, removeListener: function () {} };
};
window.devicePixelRatio = 2;
window.innerWidth = 1280;
window.innerHeight = 800;
window.navigator = { userAgent: 'fake', language: 'zh-TW' };
window.getComputedStyle = function () { return { getPropertyValue: function () { return ''; } }; };
window.scrollTo = function () {};

// ------------------------------------------------------------------------------------------ AbortController
function FSignal() { this.aborted = false; this.reason = undefined; this._l = {}; this.onabort = null; }
FSignal.prototype.addEventListener = function (t, f) { addL(this, t, f); };
FSignal.prototype.removeEventListener = function (t, f) { rmL(this, t, f); };
FSignal.prototype.throwIfAborted = function () { if (this.aborted) throw this.reason; };
function AbortError() { var e = new Error('The operation was aborted.'); e.name = 'AbortError'; return e; }
function FAbortController() { this.signal = new FSignal(); }
FAbortController.prototype.abort = function (reason) {
  if (this.signal.aborted) return;
  this.signal.aborted = true;
  this.signal.reason = reason || AbortError();
  fire(this.signal, new FEvent('abort'));
};
window.AbortController = FAbortController;

// ------------------------------------------------------------------------------------------ URLSearchParams
function FUSP(init) {
  this._p = [];
  var self = this;
  if (typeof init === 'string') {
    init.replace(/^\?/, '').split('&').forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf('=');
      var k = i < 0 ? kv : kv.slice(0, i), v = i < 0 ? '' : kv.slice(i + 1);
      self._p.push([decodeURIComponent(k.replace(/\+/g, ' ')), decodeURIComponent(v.replace(/\+/g, ' '))]);
    });
  } else if (init && typeof init === 'object') {
    Object.keys(init).forEach(function (k) { self._p.push([k, String(init[k])]); });
  }
}
FUSP.prototype.get = function (k) { for (var i = 0; i < this._p.length; i++) if (this._p[i][0] === k) return this._p[i][1]; return null; };
FUSP.prototype.has = function (k) { return this.get(k) !== null; };
FUSP.prototype.set = function (k, v) { for (var i = 0; i < this._p.length; i++) if (this._p[i][0] === k) { this._p[i][1] = String(v); return; } this._p.push([k, String(v)]); };
FUSP.prototype.append = function (k, v) { this._p.push([k, String(v)]); };
FUSP.prototype.delete = function (k) { this._p = this._p.filter(function (p) { return p[0] !== k; }); };
FUSP.prototype.forEach = function (f) { this._p.forEach(function (p) { f(p[1], p[0]); }); };
FUSP.prototype.toString = function () { return this._p.map(function (p) { return encodeURIComponent(p[0]) + '=' + encodeURIComponent(p[1]); }).join('&'); };
window.URLSearchParams = FUSP;

// ------------------------------------------------------------------------------------------ fetch
window.__fetchLog = [];
window.__fetchDelay = 0;
window.fetch = function (url, opts) {
  opts = opts || {};
  window.__fetchLog.push(String(url));
  return new Promise(function (resolve, reject) {
    var signal = opts.signal;
    if (signal && signal.aborted) { reject(AbortError()); return; }
    var done = false;
    function finish() {
      if (done) return;
      done = true;
      var r;
      try { r = window.__routes(String(url), opts); } catch (e) { reject(e); return; }
      if (r && r.networkError) { reject(new TypeError('Failed to fetch')); return; }
      var status = (r && r.status) || 200;
      var body = r ? r.body : null;
      resolve({
        ok: status >= 200 && status < 300, status: status, statusText: String(status),
        json: function () {
          if (r && r.invalidJson) return Promise.reject(new SyntaxError('Unexpected token <'));
          try { return Promise.resolve(JSON.parse(JSON.stringify(body))); } catch (e) { return Promise.reject(e); }
        },
        text: function () { return Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)); }
      });
    }
    if (signal) signal.addEventListener('abort', function () { if (!done) { done = true; reject(AbortError()); } });
    if (window.__fetchDelay > 0) window.setTimeout(finish, window.__fetchDelay); else finish();
  });
};

// ------------------------------------------------------------------------------------------ test helpers
/** Render a chart draw function against a fresh recording canvas; returns {model, ops} (stored in __rec[name]). */
window.__rec = {};
window.__renderChart = function (name, fnName, W, H, mode, spec, hover) {
  var cv = window.document.createElement('canvas');
  cv.width = W; cv.height = H;
  var ctx = cv.getContext('2d');
  var pal = window.TQ.charts.palette(mode);
  var model = window.TQ.charts[fnName](ctx, W, H, pal, spec, hover === undefined ? null : hover);
  window.__rec[name] = { W: W, H: H, bg: pal.bg, mode: mode, ops: ctx.ops };
  return model;
};
window.__recJson = function (name) { return JSON.stringify(window.__rec[name]); };

// ------------------------------------------------------------------------------------------ ResizeObserver / viewport
/** Like browsers: the callback fires once (asynchronously) right after observe(). */
window.__observers = [];
function FResizeObserver(cb) { this._cb = cb; this._els = []; window.__observers.push(this); }
FResizeObserver.prototype.observe = function (el) {
  var self = this;
  this._els.push(el);
  window.setTimeout(function () { if (self._els.indexOf(el) >= 0) self._cb([{ target: el, contentRect: el.getBoundingClientRect() }], self); }, 0);
};
FResizeObserver.prototype.unobserve = function (el) { var i = this._els.indexOf(el); if (i >= 0) this._els.splice(i, 1); };
FResizeObserver.prototype.disconnect = function () { this._els = []; };
window.ResizeObserver = FResizeObserver;
/** Fire every live observer (what a browser does after a layout change). */
window.__fireResize = function () {
  window.__observers.slice().forEach(function (o) { o._els.slice().forEach(function (el) { o._cb([{ target: el, contentRect: el.getBoundingClientRect() }], o); }); });
};
/** Give the .chart-box parent of canvas #id an explicit width (measured by a real layout engine), then fire resize. */
window.__resizeCanvasBoxes = function (widths) {
  Object.keys(widths).forEach(function (id) {
    var cv = window.__canvasEls[id];
    if (cv && cv.parentNode) cv.parentNode._w = widths[id];
  });
  window.__fireResize();
};
window.__canvasEls = [];
/** Width of the layout viewport: every element without an explicit size inherits it (charts size themselves from it). */
window.__setViewport = function (w, h) {
  window.innerWidth = w;
  window.innerHeight = h || 800;
  window.document.documentElement.__setSize(w, h || 800);
};

// ------------------------------------------------------------------------------------------ serializer (for layout renders)
window.__serialize = function (node) {
  var canvases = [];
  window.__canvasEls = [];
  var VOID = { img: 1, br: 1, hr: 1, input: 1, meta: 1, link: 1 };
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function escA(s) { return esc(s).replace(/"/g, '&quot;'); }
  function kebab(k) { return k.indexOf('--') === 0 ? k : k.replace(/[A-Z]/g, function (c) { return '-' + c.toLowerCase(); }); }
  function styleStr(st) {
    var out = [];
    Object.keys(st).forEach(function (k) {
      var v = st[k];
      if (k === '_s' || typeof v === 'function' || v === '' || v === undefined) return;
      out.push(kebab(k) + ':' + v);
    });
    return out.join(';');
  }
  function ser(n) {
    if (n.nodeType === 3) return esc(n.data);
    var tag = n.localName;
    if (tag === 'script' || tag === 'noscript') return '';
    if (tag === 'canvas') {
      var id = canvases.length;
      window.__canvasEls[id] = n;
      var W = parseFloat(n.style.width) || n.width, H = parseFloat(n.style.height) || n.height;
      var ctx = n._ctx;
      canvases.push({ id: id, W: W, H: H, ops: ctx ? ctx.ops : [], label: n._attrs['aria-label'] || '' });
      return '<img class="chart-canvas" data-canvas="' + id + '" src="canvas_' + id + '.png" style="width:' + W + 'px;height:' + H + 'px" alt="">';
    }
    var attrs = '';
    Object.keys(n._attrs).forEach(function (k) { attrs += ' ' + k + '="' + escA(n._attrs[k]) + '"'; });
    if (n._cls) attrs += ' class="' + escA(n._cls) + '"';
    var st = styleStr(n.style);
    if (st) attrs += ' style="' + escA(st) + '"';
    ['hidden', 'disabled', 'checked', 'selected', 'open'].forEach(function (b) { if (n[b] === true) attrs += ' ' + b; });
    if (tag === 'input' && n.value) attrs += ' value="' + escA(n.value) + '"';
    var out = '<' + tag + attrs + '>';
    if (VOID[tag]) return out;
    if (n._html !== undefined && !n.childNodes.length) out += n._html;
    else out += n.childNodes.map(ser).join('');
    return out + '</' + tag + '>';
  }
  var html = ser(node);
  return JSON.stringify({ html: html, canvases: canvases });
};

// ------------------------------------------------------------------------------------------ declarative fetch mocks
/**
 * window.__mock = { rules: [ { re: 'regex source', resp: {status, body}|{networkError:true}, seq: [resp,...] } ] }.
 * The FIRST rule whose regex matches the request URL (path + query) answers; `seq` answers successive calls and repeats its last
 * element.  A response body may be a JSON object or the name of a fixture (string) looked up in window.__fx.
 */
window.__mock = { rules: [], calls: {} };
window.__routes = function (url) {
  var rules = window.__mock.rules;
  for (var i = 0; i < rules.length; i++) {
    var r = rules[i];
    if (!new RegExp(r.re).test(url)) continue;
    var resp = r.resp;
    if (r.seq) {
      var n = window.__mock.calls[r.re] || 0;
      window.__mock.calls[r.re] = n + 1;
      resp = r.seq[Math.min(n, r.seq.length - 1)];
    }
    if (resp && typeof resp.body === 'string') resp = { status: resp.status, body: window.__fx[resp.body] };
    return resp;
  }
  return { status: 404, body: { status: 'error', error: { code: 'bad_request', message: '找不到此 API 路徑：' + url } } };
};

/** Sanity report of a recorded chart (see __renderChart): non-finite numbers, junk text, text anchored outside the canvas. */
window.__opsReport = function (name) {
  var rec = window.__rec[name], W = rec.W, H = rec.H;
  var bad = [], badText = [], oob = [], texts = 0, shapes = 0;
  rec.ops.forEach(function (op, i) {
    var k = op[0];
    if (k === 'set') {
      var v = op[2];
      if (typeof v === 'number' && !isFinite(v)) bad.push(i + ':' + op[1]);
      if (typeof v === 'string' && /NaN|undefined|Infinity/.test(v)) bad.push(i + ':' + op[1] + '=' + v);
      return;
    }
    for (var j = 1; j < op.length; j++) {
      var a = op[j];
      if (typeof a === 'number' && !isFinite(a)) bad.push(i + ':' + k + '[' + j + ']');
      if (Array.isArray(a)) a.forEach(function (z) { if (typeof z === 'number' && !isFinite(z)) bad.push(i + ':' + k + '[arr]'); });
    }
    if (k === 'fillText') {
      texts++;
      var t = String(op[1]);
      if (/NaN|undefined|Infinity|null|\[object/.test(t)) badText.push(t);
      var x = op[2], y = op[3];
      if (x < -2 || x > W + 2 || y < -2 || y > H + 2) oob.push(t + '@' + Math.round(x) + ',' + Math.round(y));
    }
    if (k === 'fill' || k === 'stroke' || k === 'fillRect') shapes++;
  });
  return { n: rec.ops.length, bad: bad, badText: badText, oob: oob, texts: texts, shapes: shapes };
};
