/*
 * app.js -- boot sequence: theme -> header (search box, theme toggle, "/" shortcut) -> router.
 * All behaviour lives in core.js and the view files; this file only wires the static page skeleton (index.html) to them.
 */
(function (root) {
  'use strict';
  var TQ = root.TQ;
  var h = TQ.h;

  var ICONS = {
    auto: [['circle', { cx: '12', cy: '12', r: '8', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' }], ['path', { d: 'M12 4a8 8 0 0 0 0 16z', fill: 'currentColor' }]],
    light: [['circle', { cx: '12', cy: '12', r: '4', fill: 'none', stroke: 'currentColor', 'stroke-width': '2' }],
      ['path', { d: 'M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6L17 7M7 17l-1.4 1.4', fill: 'none', stroke: 'currentColor', 'stroke-width': '2', 'stroke-linecap': 'round' }]],
    dark: [['path', { d: 'M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z', fill: 'none', stroke: 'currentColor', 'stroke-width': '2', 'stroke-linejoin': 'round' }]]
  };

  function themeIcon(mode) {
    var kids = (ICONS[mode] || ICONS.auto).map(function (p) { return h(p[0], p[1]); });
    return h('svg', { viewBox: '0 0 24 24', width: '18', height: '18', focusable: 'false', 'aria-hidden': 'true' }, kids);
  }

  function wireThemeToggle(doc) {
    var btn = doc.getElementById('theme-toggle');
    var icon = doc.getElementById('theme-icon');
    var label = doc.getElementById('theme-label');
    if (!btn) return;
    function paint() {
      var mode = TQ.theme.get();
      var name = TQ.theme.labels[mode];
      btn.setAttribute('aria-label', '切換佈景主題，目前：' + name);
      btn.setAttribute('title', '佈景主題：' + name + '（點擊切換 自動 / 淺色 / 深色）');
      if (label) label.textContent = name;
      if (icon) { TQ.clear(icon); icon.appendChild(themeIcon(mode)); }
    }
    btn.addEventListener('click', function () {
      TQ.theme.cycle();
      paint();
      var st = doc.getElementById('status');
      if (st) st.textContent = '佈景主題：' + TQ.theme.labels[TQ.theme.get()];
    });
    paint();
  }

  function boot() {
    var doc = root.document;
    TQ.theme.init();
    wireThemeToggle(doc);
    var host = doc.getElementById('search-host');
    var search = host ? TQ.searchBox.mount(host, {}) : null;
    doc.addEventListener('keydown', function (e) {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey || !search) return;
      var t = e.target, tag = t && t.tagName ? String(t.tagName).toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || (t && t.isContentEditable)) return;
      if (e.preventDefault) e.preventDefault();
      search.focus();
    });
    root.addEventListener('unhandledrejection', function (e) {
      var r = e && e.reason;
      if (r && r.name === 'AbortError') { if (e.preventDefault) e.preventDefault(); return; }
      TQ.logError('unhandled rejection', r && (r.stack || r.message) || r);
    });
    TQ.router.start();
  }

  var d = root.document;
  if (d.readyState === 'loading') d.addEventListener('DOMContentLoaded', boot); else boot();
  TQ.boot = boot;
})(typeof window !== 'undefined' ? window : this);
