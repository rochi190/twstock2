/*
 * view_stock.js -- 個股預測 (#/stock/:code).
 *
 * Data flow (all requests carry ctx.signal, so a route change aborts them):
 *   /api/history?symbol=&limit=250  -> candles (arrives first, draws the chart immediately)
 *   /api/forecast?symbol=           -> header numbers, fan chart, horizon cards, regime, volatility, risk, weights, drivers
 *   /api/status                     -> market open / closed chip (best effort, cached)
 *   /api/backtest?symbol=&h=1|5|20  -> 歷史驗證摘要 card, polled lazily AFTER the forecast rendered (one horizon after the other); never
 *                                      blocks the page.  Its verdicts decide how the signal chips of the three horizon cards read (audit P-08):
 *                                      a buy / avoid signal whose horizon has no statistical edge is shown muted as 偏多（未達顯著） / 偏空（未達顯著）.
 * Every block degrades on its own: a failed request only replaces that block by an error box with a retry button.
 */
(function (root) {
  'use strict';
  var TQ = root.TQ;
  var F = TQ.fmt, ui = TQ.ui, h = TQ.h;
  var NA = F.NA;

  function arr(x) { return Array.isArray(x) ? x : []; }
  function num(x) { return F.toNum(x); }
  function obj(x) { return x !== null && typeof x === 'object' && !Array.isArray(x) ? x : {}; }

  var HORIZONS = [
    { h: 1, key: '1', title: '明日', meaning: '模型估計明日收盤高於今日的機率', short: '明日' },
    { h: 5, key: '5', title: '未來5日', meaning: '模型估計 5 個交易日後收盤高於今日的機率', short: '5日' },
    { h: 20, key: '20', title: '未來20日', meaning: '模型估計 20 個交易日後收盤高於今日的機率', short: '20日' }
  ];
  var MODEL_ORDER = ['garch', 'hmm', 'dlm', 'naive'];
  var MODEL_FALLBACK = { garch: 'GJR-GARCH-t', hmm: '隱馬可夫regime', dlm: '動態線性模型(貝氏)', naive: '隨機漫步基準' };
  var TONE_CHIP = { neutral: 'neutral', warn: 'warn', info: 'info', accent: 'accent' };

  /** value of a {"0.05": x, ...} quantile map at `level` (keys are strings; tolerant of "0.5" / "0.50"). */
  function qv(map, level) {
    if (!map || typeof map !== 'object') return null;
    var best = null, bd = 0.005;
    Object.keys(map).forEach(function (k) {
      var d = Math.abs(Number(k) - level);
      if (d < bd) { bd = d; best = k; }
    });
    return best === null ? null : num(map[best]);
  }

  function hzOf(fc, def) {
    var hs = obj(fc && fc.horizons);
    return obj(hs[def.key] || hs[def.h]);
  }

  function hideFromAT(host) {
    var cv = host.querySelector ? host.querySelector('canvas') : null;
    if (cv) cv.setAttribute('aria-hidden', 'true');
  }

  function signed(v, dp) { return F.fmtPct(v, dp == null ? 2 : dp, true); }

  /** text colour class for a probability: follows the SIGNAL when there is one (so "觀望" is never painted red). */
  function probDir(p, signal) {
    if (signal === 'buy') return 'up';
    if (signal === 'avoid') return 'down';
    if (signal === 'hold') return 'flat';
    var v = num(p);
    if (v === null) return 'na';
    return v >= 0.52 ? 'up' : v <= 0.48 ? 'down' : 'flat';
  }

  function rangeText(lo, hi, f) {
    return lo === null || hi === null ? NA : f(lo) + ' ~ ' + f(hi);
  }

  // ====================================================================================== header
  function headerCard(S) {
    var fc = S.fc, hist = S.hist, st = S.status;
    var bars = hist ? arr(hist.bars) : [];
    var last = bars.length ? obj(bars[bars.length - 1]) : null;
    var prev = bars.length > 1 ? obj(bars[bars.length - 2]) : null;
    var name = (fc && fc.name) || (hist && hist.name) || '';
    var market = (fc && fc.market) || (hist && hist.market) || '';
    var kind = (fc && fc.kind) || (hist && hist.kind) || 'stock';
    var lastClose = fc && num(fc.last_close) !== null ? num(fc.last_close) : (last ? num(last.c) : null);
    var chg = fc && num(fc.chg_1d) !== null ? num(fc.chg_1d)
      : (last && prev && num(last.c) !== null && num(prev.c) ? last.c / prev.c - 1 : null);
    var asof = (fc && fc.asof) || (hist && hist.asof) || (last && last.d) || null;
    var loading = !fc && !hist && !S.err;

    var chips = [];
    if (market) chips.push(ui.chip(F.marketName(market), 'muted'));
    if (kind === 'etf') chips.push(ui.chip('ETF', 'info'));
    var meta = [];
    meta.push(h('span', { class: 'sh-asof' }, asof ? '資料截至 ' + F.fmtDateW(asof) + ' 收盤' : (S.err || S.histErr || S.fcErr ? '資料無法取得' : '正在取得資料…')));
    if (st && typeof st.market_open === 'boolean') {
      meta.push(st.market_open ? ui.chip('開盤中', 'open', { title: '台股交易時間 09:00–13:30；盤中不納入未完成的當日 K 棒' }) : ui.chip('已收盤', 'closed'));
    }
    if (S.stale) meta.push(ui.chip('資料可能過期', 'warn', { icon: '⚠' }));
    if (st && st.last_trading_date && asof && st.last_trading_date > asof && !st.market_open) {
      meta.push(ui.chip('尚未更新至 ' + F.fmtMD(st.last_trading_date), 'warn', { icon: '⚠', title: '最新交易日為 ' + st.last_trading_date + '，目前資料較舊' }));
    }
    var priceDir = ui.dirClass(chg !== null ? Number((chg * 100).toFixed(2)) : null);
    return h('section', { class: 'card stock-head', 'aria-label': '股票基本資料' },
      h('div', { class: 'sh-main' },
        h('div', { class: 'sh-title' },
          h('h1', { class: 'sh-name' }, name || (loading ? '載入中…' : S.code)),
          h('span', { class: 'sh-code' }, S.code),
          chips),
        h('div', { class: 'sh-meta' }, meta)),
      h('div', { class: 'sh-price', 'aria-label': '最新收盤價與一日漲跌' },
        h('div', { class: 'sh-last' }, lastClose === null ? NA : F.fmtPrice(lastClose, kind)),
        h('div', { class: ['sh-chg', priceDir] },
          h('span', { class: 'sh-chg-val' }, F.fmtChange(chg, 2)),
          h('span', { class: 'sh-chg-label' }, '1日'))));
  }

  function warningsBlock(S) {
    var fc = S.fc, st = S.status;
    var out = [];
    out.push(ui.disclaimerStrip('請搭配下方「歷史驗證摘要」判讀。'));
    if (S.stale || (fc && fc.stale)) {
      out.push(ui.banner('warn', '網路暫時無法更新，目前顯示的是本機快取資料，價格與預測可能已落後。'));
    }
    if (fc && (fc.intraday_bar_dropped === true || fc.asof_complete === false)) {
      out.push(ui.banner('info', '目前為盤中（尚未收盤）：未完成的當日 K 棒不納入模型，預測以前一個完整交易日（' + F.fmtDate(fc.asof) + '）的收盤為基準。'));
    } else if (st && st.market_open === true) {
      out.push(ui.banner('info', '目前為盤中，所有預測以前一個交易日的收盤資料計算，不含今日盤中走勢。'));
    }
    if (fc && fc.short_history === true) {
      out.push(ui.banner('warn', '此標的僅有 ' + F.fmtInt(fc.n_bars) + ' 個交易日的歷史資料，模型參數的不確定性較高，請謹慎看待。'));
    }
    return out;
  }

  // ====================================================================================== price chart
  function priceChartCard(S, sc) {
    var ranges = [{ value: 60, label: '60日' }, { value: 120, label: '120日' }, { value: 250, label: '250日' }];
    var seg = ui.segmented(ranges, S.range, function (v) { S.range = v; TQ.prefs.set('chartBars', v); card.chart.redraw(); }, { label: '價格圖顯示區間' });
    var card = TQ.chartCard('價格走勢與預測區間', function () {
      var fc = S.fc;
      return {
        bars: S.hist ? arr(S.hist.bars) : [], visible: S.range, fan: fc ? fc.fan : null,
        lastClose: fc ? fc.last_close : null, asof: (fc && fc.asof) || (S.hist && S.hist.asof),
        kind: (fc && fc.kind) || (S.hist && S.hist.kind), ma: [20, 60]
      };
    }, {
      kind: 'priceFan', sub: 'K 線（紅漲綠跌）、成交量、MA20 / MA60，右側為模型對未來 20 個交易日的價格預測區間',
      actions: seg, heightFn: function (w) { return w < 480 ? 300 : w < 900 ? 360 : 420; },
      note: '預測扇形圖顯示 50% 與 90% 的區間（中位數為藍線）：未來價格有一半的機率落在深色帶內、九成落在淺色帶內。區間寬代表不確定性大，並不代表方向。'
    });
    sc.add(card);
    return card;
  }

  // ====================================================================================== horizon cards
  var SIGNAL_RULE = '訊號規則：校準後上漲機率 ≥ 55% 且扣成本後預期報酬 > 0 → 偏多候選；上漲機率 ≤ 45% 且扣成本後預期報酬 < 0 → 偏空／避開；其餘中性觀望。這是統計篩選，不是買賣建議。';

  /** What the historical validation of ONE horizon says about its signal: undefined = still loading, null = unavailable,
   *  {level, label, unlocked} once loaded.  `unlocked` (buy wording allowed) = verdict >= moderate_edge AND the strategy did not lose to
   *  buy-and-hold in CAGR (the stock-level analogue of DESIGN 9(4) `beats_ew_after_costs`). */
  function btState(rep) {
    var v = obj(rep.verdict), level = typeof v.level === 'string' ? v.level : null;
    if (level === null) return null;
    var st = obj(obj(rep.strategy).stats), bm = obj(obj(rep.strategy).benchmark);
    var beats = num(st.cagr) !== null && num(bm.cagr) !== null ? st.cagr >= bm.cagr : false;
    return { level: level, label: F.verdictInfo(level).label, unlocked: F.levelUnlocksBuy(level) && beats };
  }

  /** Signal chip whose wording follows the evidence of its horizon (audit P-08): hold = plain; buy / avoid = muted until the validation
   *  has loaded and whenever it found no edge ("偏多（未達顯著）"), normal colour otherwise. */
  function signalChip(sig, bt) {
    var info = F.signalInfo(sig);
    if (info.dir === 'na') return null;
    if (sig === 'hold') return ui.chip(info.label, info.dir, { icon: info.icon, title: SIGNAL_RULE });
    var word = sig === 'buy' ? '偏多' : '偏空';
    if (bt === undefined) return ui.chip(info.label, 'muted', { icon: info.icon, title: '驗證載入中：尚未確認這個期間的訊號在過去是否有統計上顯著的預測力。 ' + SIGNAL_RULE });
    if (bt === null) return ui.chip(info.label, 'muted', { icon: info.icon, title: '這個期間的歷史驗證無法取得，訊號沒有經過驗證。 ' + SIGNAL_RULE });
    if (bt.level === 'no_edge') {
      return ui.chip(word + '（未達顯著）', 'muted', { icon: info.icon, title: '同一模型在過去的樣本外驗證中沒有統計上顯著的預測優勢，這個' + word + '訊號請視為「不確定」。 ' + SIGNAL_RULE });
    }
    return ui.chip(F.signalInfo(sig, bt.unlocked).label, info.dir, { icon: info.icon, title: '歷史驗證結論：' + bt.label + '。 ' + SIGNAL_RULE });
  }

  function confidenceChip(c) {
    if (c === null || c === undefined || c === '') return null;
    return ui.chip('模型一致度 ' + c, c === '高' ? 'accent' : c === '中' ? 'info' : 'neutral',
      { title: '依機率偏離 50% 的程度、預期波動與各模型方向是否一致綜合而定；這只是模型內部的一致程度，不是經過驗證的把握，請看下方「歷史驗證摘要」。' });
  }

  function horizonCard(fc, def, sc, S) {
    var hz = hzOf(fc, def);
    var kind = fc.kind;
    var lastClose = num(fc.last_close);
    var p = num(hz.p_up);
    var sig = hz.signal;
    var dir = probDir(p, sig);
    var title = hz.days_label || def.title;

    // --- gauge
    var gaugeHost = h('div', { class: 'gauge-canvas' });
    var gauge = TQ.charts.mount(gaugeHost, {
      height: 112, label: title + '上漲機率',
      draw: function (ctx, W, H, pal) { return TQ.charts.drawGauge(ctx, W, H, pal, { p: p, lo: 0.3, hi: 0.7 }); }
    });
    sc.add(gauge);
    hideFromAT(gaugeHost);                       // the number is real text next to it: do not announce it twice
    var gaugeVal = h('div', { class: ['gauge-val', dir], title: p === null ? null : '模型輸出 ' + F.fmtPct(p, 1) + '。歷史驗證的校準誤差通常有數個百分點，所以不宜讀到小數點。' },
        h('span', { class: 'sr-only' }, '上漲機率 '), p === null ? NA : F.fmtPct(p, 0));
    var gaugeBox = h('div', { class: 'gauge-box' }, gaugeHost, gaugeVal, h('div', { class: 'gauge-cap' }, '上漲機率'));

    // --- numbers
    var exp = num(hz.exp_return), net = num(hz.net_exp_return), sg = num(hz.sigma);
    var kv = ui.kv([
      { label: '預期報酬', value: exp === null ? NA : h('span', { class: 'nowrap' }, F.fmtChange(exp, 2)), dir: ui.dirClass(exp === null ? null : Number((exp * 100).toFixed(2))),
        tip: '模型對這段期間「平均報酬」的估計（簡單報酬）。數字通常遠小於波動，不代表會賺到這個數字。' },
      { label: '扣成本後', value: net === null ? NA : h('span', { class: 'nowrap' }, F.fmtChange(net, 2)), dir: ui.dirClass(net === null ? null : Number((net * 100).toFixed(2))),
        tip: '預期報酬扣掉一次買進加賣出的手續費、證交稅與滑價後的估計（股票約 0.57%、ETF 約 0.37%）。' },
      { label: '預期波動 σ', value: sg === null ? NA : F.fmtPct(sg, 2), tip: '這段期間報酬的標準差（模型估計），可視為「典型起伏幅度」。' }
    ]);

    // --- intervals
    var q05 = qv(hz.quantiles, 0.05), q25 = qv(hz.quantiles, 0.25), q50 = qv(hz.quantiles, 0.5), q75 = qv(hz.quantiles, 0.75), q95 = qv(hz.quantiles, 0.95);
    var pq = hz.price_quantiles;
    function pr(level, q) {
      var v = qv(pq, level);
      if (v === null && q !== null && lastClose !== null) v = lastClose * (1 + q);
      return v;
    }
    var p05 = pr(0.05, q05), p25 = pr(0.25, q25), p50 = pr(0.5, q50), p75 = pr(0.75, q75), p95 = pr(0.95, q95);
    var stripHost = h('div', { class: 'strip-canvas' });
    var strip = TQ.charts.mount(stripHost, {
      height: 62, label: title + '報酬區間',
      draw: function (ctx, W, H, pal) { return TQ.charts.drawIntervalStrip(ctx, W, H, pal, { q05: q05, q25: q25, q50: q50, q75: q75, q95: q95, mean: exp }); }
    });
    sc.add(strip);
    hideFromAT(stripHost);                       // the table below carries the same numbers
    function priceFmt(v) { return F.fmtPrice(v, kind); }
    var ivRows = [
      { name: '90% 區間', price: rangeText(p05, p95, priceFmt), ret: rangeText(q05, q95, function (v) { return signed(v, 1); }) },
      { name: '50% 區間', price: rangeText(p25, p75, priceFmt), ret: rangeText(q25, q75, function (v) { return signed(v, 1); }) },
      { name: '中位數', price: p50 === null ? NA : priceFmt(p50), ret: q50 === null ? NA : signed(q50, 2) }
    ];
    var ivTable = ui.table([
      { key: 'name', label: '區間', rowHeader: true },
      { key: 'price', label: '價格 (TWD)', num: true },
      { key: 'ret', label: '報酬', num: true }
    ], ivRows, { caption: title + '的價格與報酬區間', dense: true });

    // --- components
    var comps = obj(hz.components);
    var names = obj(obj(fc.models).names);
    var compKeys = MODEL_ORDER.filter(function (k) { return comps[k] !== undefined; }).concat(Object.keys(comps).filter(function (k) { return MODEL_ORDER.indexOf(k) < 0; }));
    var compRows = compKeys.map(function (k) { return { name: names[k] || MODEL_FALLBACK[k] || k, p: num(obj(comps[k]).p_up), mu: num(obj(comps[k]).mu) }; });
    if (p !== null) compRows.push({ name: '綜合（已校準）', p: p, mu: exp, total: true });
    var compTable = compRows.length ? ui.table([
      { key: 'name', label: '模型', rowHeader: true },
      { key: 'p', label: '上漲機率', num: true, render: function (r) { return r.p === null ? NA : F.fmtPct(r.p, 1); } },
      { key: 'mu', label: '預期報酬', num: true, render: function (r) { return r.mu === null ? NA : signed(r.mu, 2); } }
    ], compRows, { caption: title + '各模型的看法', dense: true, rowClass: function (r) { return r.total ? 'row-total' : null; } }) : null;

    // signal + confidence chips: re-painted when this horizon's validation arrives (S.chips[h]())
    var chipsHost = h('div', { class: 'hz-chips' });
    var cardEl = null;
    function fillChips() {
      var bt = S && S.bt ? S.bt[def.h] : undefined;
      TQ.clear(chipsHost);
      var a = signalChip(sig, bt), b = confidenceChip(hz.confidence);
      if (a) chipsHost.appendChild(a);
      if (b) chipsHost.appendChild(b);
      // a buy / avoid signal whose validation found no edge must not keep painting the probability and the card red / green
      var unsupported = (sig === 'buy' || sig === 'avoid') && bt && bt.level === 'no_edge';
      var eff = unsupported ? 'flat' : dir;
      gaugeVal.className = 'gauge-val ' + eff;
      if (cardEl) { ['hz-up', 'hz-down', 'hz-flat'].forEach(function (c) { cardEl.classList.remove(c); }); cardEl.classList.add('hz-' + (eff === 'na' ? 'flat' : eff)); }
    }
    fillChips();
    if (S) { S.chips = S.chips || {}; S.chips[def.h] = fillChips; }

    var when = def.h === 1 && fc.next_trading_day ? '目標：' + F.fmtDateW(fc.next_trading_day) + ' 收盤' : '目標：約 ' + def.h + ' 個交易日後的收盤';
    cardEl = ui.card(title, [
      chipsHost,
      h('div', { class: 'hz-top' },
        h('div', { class: 'hz-gauge' }, gaugeBox),
        h('div', { class: 'hz-stats' }, kv)),
      h('p', { class: 'hz-meaning' }, def.meaning,
        h('span', { class: 'hz-meaning-sub' }, '（約 50% 代表漲跌機率相當；這是統計估計，不是保證。）')),
      h('div', { class: 'hz-block' },
        h('h3', { class: 'sub-title' }, '報酬區間', ui.infoTip('條形為 90% 區間（細）與 50% 區間（粗），白線為中位數，菱形為平均。左側綠色為虧損、右側紅色為獲利。')),
        h('div', { class: 'strip-box' }, stripHost),
        ivTable),
      compTable ? h('div', { class: 'hz-block' }, h('h3', { class: 'sub-title' }, '各模型看法'), compTable) : null
    ], { class: ['hz-card', 'hz-' + (dir === 'na' ? 'flat' : dir)], sub: when, level: 2 });
    fillChips();                                   // a validation that arrived while the card was being built
    return cardEl;
  }

  // ====================================================================================== regime / vol / risk
  function regimeCard(fc) {
    var rg = obj(fc.regime);
    var labels = arr(rg.labels), probs = arr(rg.probs);
    var curIdx = -1, best = -1;
    labels.forEach(function (l, i) {
      var p = num(probs[i]);
      if (p !== null && p > best) { best = p; curIdx = i; }
      if (rg.current && l === rg.current) { curIdx = i; best = 2; }
    });
    var body;
    // DESIGN 6.1 (additive `regime.confident`): when the largest filtered probability is < 0.6 the state is not "the" state:
    // no row is marked and the card says so (the filtered probabilities of a ~0.75-0.8 persistent chain move fast).
    var unsure = rg.confident === false;
    if (!labels.length) {
      body = ui.emptyState('沒有市場狀態資料。');
    } else {
      body = h('ul', { class: 'regime-list' }, labels.map(function (l, i) {
        var p = num(probs[i]);
        var cur = i === curIdx && !unsure;
        return h('li', { class: ['regime-row', cur ? 'current' : null] },
          h('span', { class: 'rg-label' }, l, cur ? ui.chip('目前最可能', 'accent') : null),
          ui.hbar(p, { label: l + '的機率', kind: cur ? 'accent' : 'gray' }),
          h('span', { class: 'rg-pct' }, p === null ? NA : F.fmtPct(p, 1)));
      }));
    }
    var trans = arr(rg.transition);
    var transTable = null;
    if (labels.length && trans.length === labels.length && trans.every(function (r) { return Array.isArray(r) && r.length === labels.length; })) {
      var cols = [{ key: 'from', label: '從 ＼ 到', rowHeader: true }].concat(labels.map(function (l, j) {
        return { key: 'c' + j, label: l, num: true, render: function (r) { return r.v[j] === null ? NA : F.fmtPct(r.v[j], 1); } };
      }));
      transTable = h('details', { class: 'disclosure' },
        h('summary', null, '狀態轉移機率（下一個交易日）'),
        ui.table(cols, labels.map(function (l, i) { return { from: l, v: trans[i].map(num) }; }), { caption: '狀態轉移機率矩陣', dense: true }));
    }
    return ui.card('市場狀態', [unsure && labels.length ? h('p', { class: 'rg-unsure' }, ui.chip('判斷不明確', 'warn', { icon: '⚠' }), ' 各狀態的機率相近（最高不到 60%），模型無法確定現在處於哪一個狀態，下列數字只是濾波機率。') : null, body, transTable], {
      sub: '隱馬可夫模型（HMM）根據報酬與波動判斷目前處於哪一種市場狀態；機率為「已知到今日為止」的濾波結果。',
      tip: '狀態由模型從資料自動分群，名稱只是依各群的平均報酬與波動命名；狀態會切換，機率會隨新資料更新。'
    });
  }

  function volCard(fc) {
    var v = obj(fc.vol);
    var ann = num(v.sigma_annual), day = num(v.sigma_daily), pct = num(v.percentile);
    var levelChip = pct === null ? null : pct >= 0.8 ? ui.chip('波動偏高', 'warn') : pct <= 0.2 ? ui.chip('波動偏低', 'info') : ui.chip('波動中段', 'neutral');
    return ui.card('波動度', [
      h('div', { class: 'vol-main' },
        ui.kpi('年化波動（預期）', ann === null ? NA : F.fmtPct(ann, 1), '約 ' + (day === null ? NA : F.fmtPct(day, 2)) + ' / 日', { tip: '模型（GJR-GARCH）對明日波動的預測，以 245 個交易日年化。' }),
        h('div', { class: 'vol-meter' },
          h('div', { class: 'vol-meter-head' }, h('span', null, '歷史百分位'), levelChip),
          ui.hbar(pct, { label: '目前波動度的歷史百分位', kind: pct !== null && pct >= 0.8 ? 'warn' : 'accent' }),
          h('div', { class: 'vol-meter-foot' },
            h('span', null, '低'),
            h('strong', null, pct === null ? NA : '高於歷史 ' + F.fmtPct(pct, 0) + ' 的時間'),
            h('span', null, '高')))),
      v.regime_note ? h('p', { class: 'vol-note' }, v.regime_note) : null
    ], { sub: '目前的預期波動相對於過去的高低。波動大代表價格起伏大，與漲跌方向無關。' });
  }

  function riskCard(fc) {
    var r = obj(fc.risk);
    var rows = [
      { name: 'VaR 95%', d1: num(r.var95_1d), d20: num(r.var95_20d), tip: '在險價值：有 95% 的機率，虧損不會比這個數字更糟（單一情境下的「門檻」，不是最大虧損）。' },
      { name: 'CVaR 95%', d1: num(r.cvar95_1d), d20: num(r.cvar95_20d), tip: '條件在險價值：萬一落入最糟的 5% 情境，平均會虧多少。' }
    ];
    var t = ui.table([
      { key: 'name', label: '指標', rowHeader: true, render: function (x) { return h('span', null, x.name, ui.infoTip(x.tip)); } },
      { key: 'd1', label: '1 日', num: true, render: function (x) { return x.d1 === null ? NA : signed(x.d1, 2); } },
      { key: 'd20', label: '20 日', num: true, render: function (x) { return x.d20 === null ? NA : signed(x.d20, 1); } }
    ], rows, { caption: '風險指標：在險價值與條件在險價值', dense: true });
    var extra = ui.kv([
      { label: '20 日內最大回撤（中位數）', value: num(r.median_max_drawdown_20d) === null ? NA : signed(r.median_max_drawdown_20d, 1), tip: '模擬未來 20 個交易日，途中從高點回落的幅度，取中位數。' },
      { label: '20 日虧損超過 10% 的機率', value: num(r.prob_loss_gt_10pct_20d) === null ? NA : F.fmtPct(r.prob_loss_gt_10pct_20d, 1), tip: '以蒙地卡羅模擬（含波動叢聚與市場狀態）估計。' }
    ]);
    return ui.card('風險', [t, extra], { sub: '以模擬 5,000 條價格路徑估計的下檔風險（負值代表虧損）。' });
  }

  function weightsCard(fc) {
    var models = obj(fc.models);
    var names = obj(models.names);
    var weights = obj(models.weights);
    var keysSeen = [];
    HORIZONS.forEach(function (d) { Object.keys(obj(weights[d.key])).forEach(function (k) { if (keysSeen.indexOf(k) < 0) keysSeen.push(k); }); });
    var order = MODEL_ORDER.filter(function (k) { return keysSeen.indexOf(k) >= 0; }).concat(keysSeen.filter(function (k) { return MODEL_ORDER.indexOf(k) < 0; }));
    var rows = HORIZONS.map(function (d) {
      var w = obj(weights[d.key]);
      var segs = order.map(function (k) { return { key: k, label: names[k] || MODEL_FALLBACK[k] || k, value: num(w[k]), cls: 'w-' + (MODEL_ORDER.indexOf(k) >= 0 ? k : 'other') }; })
        .filter(function (s) { return s.value !== null; });
      return h('div', { class: 'w-row' }, h('span', { class: 'w-label' }, d.short), segs.length ? ui.stackedBar(segs, { label: d.title + '模型權重' }) : h('span', { class: 'muted' }, NA));
    });
    var legend = h('ul', { class: 'w-legend' }, order.map(function (k) {
      return h('li', null, h('i', { class: ['w-key', 'w-' + (MODEL_ORDER.indexOf(k) >= 0 ? k : 'other')], 'aria-hidden': 'true' }), names[k] || MODEL_FALLBACK[k] || k);
    }));
    if (!order.length) return ui.card('模型權重', ui.emptyState('沒有模型權重資料。'));
    return ui.card('模型權重', [rows, legend], {
      sub: '最終預測是四個模型的加權平均；權重依各模型近期的預測表現動態調整（動態模型平均）。',
      tip: '權重分散代表目前沒有哪個模型明顯勝出；「隨機漫步基準」權重高，代表模型認為資料幾乎沒有可預測的訊號。'
    });
  }

  // ====================================================================================== drivers
  function guessUnit(key) {
    key = String(key || '');
    if (/rsi/.test(key)) return 'level';
    if (/ratio|vol_z/.test(key)) return 'ratio';
    if (/mom|rev|gap|ret|idio|cross|vol$/.test(key)) return 'pct';
    return 'ratio';
  }
  function driverValue(d) {
    var v = num(d.value);
    if (v === null) return NA;
    var unit = d.unit || guessUnit(d.key);
    return unit === 'pct' ? F.fmtPct(v, 1, true) : unit === 'level' ? F.fmtNum(v, 1) : F.fmtNum(v, 2);
  }

  function driversCard(fc) {
    var all = arr(fc.drivers).filter(function (d) { return d && typeof d === 'object'; });
    if (!all.length) return ui.card('預測驅動因子', ui.emptyState('沒有因子資料。'));
    var maxAbs = 0;
    all.forEach(function (d) { var c = num(d.contribution); if (c !== null && Math.abs(c) > maxAbs) maxAbs = Math.abs(c); });
    var showAll = false;
    var toggle;
    function rowsToShow() { return showAll ? all : all.slice(0, 8); }
    var table = ui.table([
      { key: 'name', label: '因子', rowHeader: true },
      { key: 'value', label: '目前數值', num: true, render: function (d) { return driverValue(d); },
        tip: '因子的原始數值，例如 12-1 個月動能為過去 12 個月（不含最近 1 個月）的報酬。' },
      { key: 'z', label: 'z 分數', num: true, render: function (d) { return num(d.z) === null ? NA : F.fmtSigned(d.z, 2); },
        tip: '該因子相對於自身歷史的標準化數值；絕對值越大代表越極端（約 ±2 以上算罕見）。' },
      { key: 'contribution', label: '對預期報酬的貢獻', num: true,
        tip: '因子經動態線性模型換算，對「未來 5 日預期報酬」的加減貢獻（bp = 萬分之一）。正（紅）推升預期、負（綠）拖低預期。',
        render: function (d) {
          var c = num(d.contribution);
          if (c === null) return NA;
          var dir = ui.dirClass(Number((c * 1e4).toFixed(1)));
          var w = maxAbs > 0 ? Math.min(100, Math.abs(c) / maxAbs * 100) : 0;
          return h('span', { class: 'drv-cell' },
            h('span', { class: ['chg', dir] }, F.arrow(dir) + ' ' + F.fmtBps(c, 1, true)),
            h('span', { class: ['drv-bar', dir], 'aria-hidden': 'true' }, h('i', { style: { width: w.toFixed(0) + '%' } })));
        } }
    ], rowsToShow(), { caption: '預測驅動因子', dense: true });
    var body = [table];
    if (all.length > 8) {
      toggle = h('button', { type: 'button', class: 'btn btn-ghost', 'aria-expanded': 'false', onClick: function () {
        showAll = !showAll;
        table.setRows(rowsToShow());
        toggle.textContent = showAll ? '只顯示前 8 項' : '顯示全部 ' + all.length + ' 項';
        toggle.setAttribute('aria-expanded', showAll ? 'true' : 'false');
      } }, '顯示全部 ' + all.length + ' 項');
      body.push(h('div', { class: 'table-more' }, toggle));
    }
    return ui.card('預測驅動因子', body, {
      sub: '依對預期報酬貢獻的大小排序。因子之間會互相抵銷，單一因子的方向不等於最終預測。'
    });
  }

  // ====================================================================================== backtest summary
  function btColumn(rep, def, code) {
    var d = obj(rep.direction), pr = obj(rep.probabilistic), v = obj(rep.verdict), per = obj(rep.period);
    var info = F.verdictInfo(v.level);
    var n = num(d.n);
    var best = d.best_baseline;
    var bestVal = best && d.baselines ? num(obj(d.baselines)[best]) : null;
    var kvs = ui.kv([
      { label: '方向準確率', value: h('span', null, F.fmtPct(d.accuracy, 1), h('span', { class: 'kv-sub' }, ' ' + F.fmtCI(d.ci95, 1))),
        tip: '模型在樣本外預測「漲或跌」猜對的比例；括號為 95% 信賴區間（Wilson）。與隨機（50%）相比，更要與下一列的簡單基準相比。' },
      { label: '最佳基準：' + F.baselineName(best), value: bestVal === null ? NA : F.fmtPct(bestVal, 1),
        tip: '不用任何模型、只用簡單規則（如「一直猜漲」）也能得到的準確率。模型必須明顯超過它才算有價值。' },
      { label: '對最佳基準的 p 值', value: F.fmtP(d.p_value_vs_best_baseline),
        tip: '二項檢定：若模型其實沒有優勢，僅靠運氣得到這麼高準確率的機率。通常小於 0.05 才視為顯著；越大代表越可能只是運氣。' },
      { label: 'Brier 技能分數', value: num(pr.brier_skill) === null ? NA : F.fmtPct(pr.brier_skill, 2, true),
        tip: '機率預測品質相對「永遠猜歷史上漲比例」的改善幅度。大於 0 才代表機率預測有加分；接近 0 或為負代表沒有優勢。' },
      { label: '樣本外預測數', value: n === null ? NA : F.fmtInt(n) + ' 次（' + F.fmtDate(per.start) + ' ~ ' + F.fmtDate(per.end) + '）' }
    ]);
    return h('div', { class: 'bt-col' },
      h('div', { class: 'bt-col-head' },
        h('h3', { class: 'sub-title' }, def.title),
        ui.chip(info.label, TONE_CHIP[info.tone] || 'neutral', { icon: info.icon })),
      h('p', { class: ['bt-verdict', 'tone-' + info.tone] }, v.title || '尚無結論'),
      kvs,
      v.text ? h('details', { class: 'disclosure' }, h('summary', null, '結論全文'), h('p', { class: 'bt-text' }, v.text)) : null,
      h('div', { class: 'bt-foot' }, h('a', { class: 'btn btn-link', href: TQ.codeHref(code, 'backtest') + '?h=' + def.h }, '查看 ' + def.title + ' 完整驗證 →')));
  }

  /** Where the primary 「開啟完整歷史驗證」 button goes: the longest horizon that carries a buy / avoid signal (that is the card the
   *  reader is looking at), else the plain page (default horizon). */
  function primaryHref(S) {
    var best = null;
    HORIZONS.forEach(function (d) {
      var sg = hzOf(S.fc, d).signal;
      if (sg === 'buy' || sg === 'avoid') best = d.h;
    });
    return TQ.codeHref(S.code, 'backtest') + (best === null ? '' : '?h=' + best);
  }

  function backtestCard(S, ctx, sc) {
    var defs = HORIZONS.slice();
    var cells = defs.map(function () { return h('div', { class: 'bt-cell' }); });
    var card = ui.card('歷史驗證摘要', [
      h('p', { class: 'bt-intro' }, '同一個模型在過去約 5 年「只用當時已知資料」逐日預測（前推式、樣本外）的真實表現，已與簡單基準比較。上方三個期間的訊號標籤會依這裡的結論調整。'),
      h('div', { class: 'bt-grid' }, cells),
      h('p', { class: 'bt-caveat' }, '過去的樣本外表現不保證未來；若結論為「無預測優勢」，請把上方機率視為接近「不確定」，而非可操作的訊號。'),
      h('div', { class: 'bt-foot' }, h('a', { class: 'btn btn-primary bt-open', href: primaryHref(S) }, '開啟完整歷史驗證 →'))
    ], { sub: '載入時間視是否已有快取而定，首次可能需要一分鐘左右；三個期間依序計算，進行中不影響上方內容。', class: 'bt-card' });
    var started = false;
    function settle(def, state) {
      S.bt[def.h] = state;
      if (S.chips && typeof S.chips[def.h] === 'function') S.chips[def.h]();
    }
    function loadOne(i) {
      var cell = cells[i], def = defs[i];
      TQ.clear(cell);
      var bar = ui.progressBar(0, 0, '準備中…');
      cell.appendChild(h('h3', { class: 'sub-title' }, def.title));
      cell.appendChild(bar);
      return TQ.api.poll('/api/backtest', { symbol: S.code, h: def.h }, {
        signal: ctx.signal,
        onProgress: function (p) { bar.update(p.done, p.total, p.message || '計算中…'); }
      }).then(function (js) {
        if (ctx.signal.aborted) return;
        var rep = obj(js.report || js);
        TQ.clear(cell);
        cell.appendChild(btColumn(rep, def, S.code));
        settle(def, btState(rep));
      }, function (err) {
        if (TQ.api.isAbort(err) || ctx.signal.aborted) return;
        TQ.clear(cell);
        cell.appendChild(h('h3', { class: 'sub-title' }, def.title));
        cell.appendChild(ui.errorBox(err, function () { loadOne(i); }));
        settle(def, null);
      });
    }
    card.start = function () {
      if (started) return;
      started = true;
      cells.forEach(function (cell, i) {
        TQ.clear(cell);
        cell.appendChild(h('h3', { class: 'sub-title' }, defs[i].title));
        cell.appendChild(i === 0 ? ui.progressBar(0, 0, '準備中…') : h('p', { class: 'muted' }, '排隊中：待「' + defs[i - 1].title + '」驗證完成後接著計算（共用同一份前推式預測）。'));
      });
      var chain = loadOne(0);
      for (var k = 1; k < defs.length; k++) {
        (function (idx) { chain = chain.then(function () { if (!ctx.signal.aborted) return loadOne(idx); }); })(k);
      }
    };
    return card;
  }

  // ====================================================================================== notes
  function notesCard(fc) {
    var notes = fc ? arr(fc.notes).filter(function (n) { return typeof n === 'string' && n; }) : [];
    var disclaimer = (fc && typeof fc.disclaimer === 'string' && fc.disclaimer) || TQ.DISCLAIMER;
    return ui.card('注意事項與免責聲明', [
      notes.length ? h('ul', { class: 'note-list' }, notes.map(function (n) { return h('li', null, n); })) : null,
      h('p', { class: 'disclaimer' },
        h('span', { class: 'disclaimer-icon', 'aria-hidden': 'true' }, '⚠'),
        h('span', null, disclaimer))
    ], { class: 'notes-card' });
  }

  // ====================================================================================== view
  function title(params) {
    return '個股預測 ' + (params && params.code ? params.code : '');
  }

  function mount(host, params, ctx) {
    var code = String(params && params.code || '2330');
    var S = { code: code, fc: null, hist: null, status: null, err: null, stale: false, range: 120, bt: {}, chips: {} };
    var range = Number(TQ.prefs.get('chartBars', 120));
    S.range = range === 60 || range === 120 || range === 250 ? range : 120;
    var destroyed = false;
    var scChart = TQ.scope(), scFc = TQ.scope();
    var chartNode = null, btNode = null, waitMsg = null;

    var slotHead = h('div', { class: 'slot slot-head' });
    var slotWarn = h('div', { class: 'slot slot-warn' });
    var slotChart = h('div', { class: 'slot slot-chart' });
    var slotFc = h('div', { class: 'slot slot-fc' });
    var slotBt = h('div', { class: 'slot slot-bt' });
    var slotNotes = h('div', { class: 'slot slot-notes' });
    var view = h('div', { class: 'view view-stock' }, slotHead, slotWarn, slotChart, slotFc, slotBt, slotNotes);
    host.appendChild(view);

    function alive() { return !destroyed && !ctx.signal.aborted; }
    function remember() { if (S.fc || S.hist) TQ.router.setLastStock(code); }

    function renderHead() { TQ.clear(slotHead); slotHead.appendChild(headerCard(S)); }
    function renderWarn() { TQ.clear(slotWarn); warningsBlock(S).forEach(function (n) { slotWarn.appendChild(n); }); }

    function renderChart() {
      scChart.dispose();
      TQ.clear(slotChart);
      chartNode = null;
      if (S.histErr && !S.fc) {
        slotChart.appendChild(ui.card('價格走勢與預測區間', ui.errorBox(S.histErr, loadHistory)));
        return;
      }
      if (!S.hist && !S.fc) {
        slotChart.appendChild(ui.card('價格走勢與預測區間', ui.skeleton(0, { block: 280 })));
        return;
      }
      if (S.histErr) slotChart.appendChild(ui.banner('warn', '價格歷史載入失敗，圖表僅顯示預測區間。', { title: S.histErr.message || '' }));
      chartNode = priceChartCard(S, scChart);
      slotChart.appendChild(chartNode);
    }

    function renderFc() {
      scFc.dispose();
      TQ.clear(slotFc);
      if (S.fcErr) {
        slotFc.appendChild(ui.card('預測', ui.errorBox(S.fcErr, loadForecast)));
        return;
      }
      if (!S.fc) {
        waitMsg = h('p', { class: 'muted fc-wait', role: 'status' }, S.fcMsg || '正在計算預測，首次載入約需數秒…');
        slotFc.appendChild(waitMsg);
        slotFc.appendChild(h('div', { class: 'row' }, [0, 1, 2].map(function () { return h('div', { class: 'card' }, h('div', { class: 'card-body' }, ui.skeleton(6))); })));
        return;
      }
      var fc = S.fc;
      slotFc.appendChild(h('div', { class: 'row row-3 hz-row' }, HORIZONS.map(function (d) {
        try { return horizonCard(fc, d, scFc, S); } catch (e) { TQ.logError(e); return ui.card(d.title, ui.errorBox(null)); }
      })));
      var cards = [regimeCard, volCard, riskCard, weightsCard, driversCard];
      var built = cards.map(function (fn) {
        try { return fn(fc); } catch (e) { TQ.logError(e); return ui.card('資料顯示失敗', ui.errorBox(null)); }
      });
      slotFc.appendChild(h('div', { class: 'row row-2' }, built[0], built[1]));
      slotFc.appendChild(h('div', { class: 'row row-2' }, built[2], built[3]));
      slotFc.appendChild(built[4]);
    }

    function renderNotes() { TQ.clear(slotNotes); slotNotes.appendChild(notesCard(S.fc)); }

    function renderAll() {
      if (!alive()) return;
      renderHead(); renderWarn(); renderChart(); renderFc(); renderNotes();
    }

    function fatal(err) {
      scChart.dispose(); scFc.dispose();
      TQ.clear(view);
      view.appendChild(ui.card('個股預測', [
        ui.errorBox(err),
        h('p', { class: 'muted' }, '請使用上方搜尋框輸入股票代號或名稱（例如 2330、台積電、0050）。'),
        h('p', null, h('a', { class: 'btn', href: TQ.codeHref('2330') }, '前往台積電 2330'), ' ', h('a', { class: 'btn btn-ghost', href: TQ.codeHref('0050') }, '前往元大台灣50 0050'))
      ], { level: 1 }));
      ctx.setTitle('找不到 ' + code);
    }

    function loadForecast() {
      S.fc = null; S.fcErr = null; S.fcMsg = '';
      renderFc();
      TQ.api.poll('/api/forecast', { symbol: code }, { signal: ctx.signal, onProgress: function (p) {
        if (!alive() || !p.message || S.fc) return;
        S.fcMsg = p.message;
        if (waitMsg) waitMsg.textContent = p.message;
      } }).then(function (fc) {
        if (!alive()) return;
        S.fc = fc;
        S.stale = S.stale || fc.stale === true;
        if (fc.name) ctx.setTitle(fc.name + ' ' + code + ' 個股預測');
        remember();
        ctx.setStatus((fc.name || code) + ' 的預測已更新');
        renderHead(); renderWarn();
        if (chartNode) chartNode.chart.redraw(); else renderChart();
        renderFc(); renderNotes();
        startBacktest();
      }, function (err) {
        if (!alive() || TQ.api.isAbort(err)) return;
        if (err && err.code === 'unknown_symbol') { fatal(err); return; }
        S.fcErr = err;
        renderFc();
      });
    }

    function loadHistory() {
      S.histErr = null;
      TQ.api.get('/api/history', { symbol: code, limit: 250 }, { signal: ctx.signal }).then(function (hist) {
        if (!alive()) return;
        S.hist = hist;
        S.stale = S.stale || hist.stale === true;
        remember();
        renderHead(); renderWarn(); renderChart();
      }, function (err) {
        if (!alive() || TQ.api.isAbort(err)) return;
        if (err && err.code === 'unknown_symbol') { fatal(err); return; }
        S.histErr = err;
        renderChart();
      });
    }

    function startBacktest() {
      if (btNode) return;
      btNode = backtestCard(S, ctx, scFc);
      TQ.clear(slotBt);
      slotBt.appendChild(btNode);
      root.setTimeout(function () { if (alive() && btNode) btNode.start(); }, 250);
    }

    renderAll();
    loadHistory();
    loadForecast();
    TQ.api.status().then(function (st) {
      if (!alive() || !st) return;
      S.status = st;
      renderHead(); renderWarn();
    });

    return {
      destroy: function () {
        destroyed = true;
        scChart.dispose();
        scFc.dispose();
      }
    };
  }

  TQ.views.stock = { title: title, mount: mount, _internals: { qv: qv, probDir: probDir, driverValue: driverValue, guessUnit: guessUnit } };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
