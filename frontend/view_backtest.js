/*
 * view_backtest.js -- 歷史驗證 (#/backtest/:code?h=1&thr=0.55&exec=next_open&disc=0.6&years=5).
 *
 * The page lets the user choose stock / horizon / probability threshold / execution mode / commission discount / test window, runs
 * the walk-forward validation on the server (GET /api/backtest, a job: polled with a progress bar and a cancel button, 10-60 s the
 * first time, cached afterwards) and renders the report of DESIGN.md 6.4: verdict, KPI cards with plain-language tooltips, equity
 * curve (+ position band + drawdown), reliability diagram, rolling accuracy, PIT histogram, coverage and confidence tables, costs.
 *
 * URL = settings:  every run writes its settings into the query part of the hash (ctx.replace, no re-mount), and a link that carries
 * a query is restored AND run on load, so a result can be shared.  A bare #/backtest/2330 shows the controls only (idle): the first
 * run of a stock costs real computation, so it is explicit.  Invalid query values are normalised and reported in a note.
 *
 * Colour: the verdict uses neutral / amber / accent tones (never red / green, which mean up / down in Taiwan).  Everything tolerates
 * null fields (the schema says undefined statistics are null).
 */
(function (root) {
  'use strict';
  var TQ = root.TQ;
  var F = TQ.fmt, ui = TQ.ui, h = TQ.h;
  var NA = F.NA;

  function arr(x) { return Array.isArray(x) ? x : []; }
  function num(x) { return F.toNum(x); }
  function obj(x) { return x !== null && typeof x === 'object' && !Array.isArray(x) ? x : {}; }
  var hasOwn = Object.prototype.hasOwnProperty;
  function own(o, k) { return o && typeof k === 'string' && hasOwn.call(o, k) ? o[k] : undefined; }
  function signedPct(v, dp) { return F.fmtPct(v, dp == null ? 1 : dp, true); }
  function nowMs() { return root.performance && typeof root.performance.now === 'function' ? root.performance.now() : Date.now(); }

  // ====================================================================================== settings
  /** Taiwan cost constants -- MIRROR backend/config.py (tests/test_front_backtest_static.py compares them). */
  var COST = { commission: 0.001425, taxStock: 0.003, taxEtf: 0.001, slippageBps: 5 };
  var DEFAULTS = { h: 1, thr: 0.55, exec: 'next_open', disc: 0.6, years: 5 };
  var THR_MIN = 0.5, THR_MAX = 0.7;
  var HORIZONS = [
    { value: 1, label: '明日', long: '下一個交易日' },
    { value: 5, label: '一週', long: '未來 5 個交易日' },
    { value: 20, label: '一個月', long: '未來 20 個交易日' }
  ];
  var EXECS = [
    { value: 'next_open', label: '下一個開盤價成交', short: '下一個開盤價成交' },
    { value: 'close', label: '收盤成交(樂觀)', short: '收盤成交(樂觀)' }
  ];
  var YEARS = [2, 3, 5, 8];
  var DISCOUNTS = [0.28, 0.35, 0.5, 0.6, 0.7, 0.85, 1];
  var TONE_CHIP = { neutral: 'neutral', warn: 'warn', info: 'info', accent: 'accent' };

  function hzDef(v) { for (var i = 0; i < HORIZONS.length; i++) if (HORIZONS[i].value === Number(v)) return HORIZONS[i]; return HORIZONS[0]; }
  function execLabel(v) { for (var i = 0; i < EXECS.length; i++) if (EXECS[i].value === v) return EXECS[i].short; return v ? String(v) : NA; }
  function round(x, dp) { var k = Math.pow(10, dp); return Math.round(x * k) / k; }

  /** 0.6 -> "6 折", 0.28 -> "2.8 折", 1 -> "不打折（牌告價）". */
  function discLabel(d) {
    var v = num(d);
    if (v === null) return NA;
    if (v >= 0.9995) return '不打折（牌告價）';
    return String(round(v * 10, 2)) + ' 折';
  }

  function nearest(list, x) {
    var best = list[0], bd = Infinity;
    list.forEach(function (v) { var d = Math.abs(v - x); if (d < bd) { bd = d; best = v; } });
    return best;
  }

  /** Round-trip cost components for a given discount / slippage / kind (same arithmetic as config.roundtrip_cost). */
  function costBreakdown(discount, slippageBps, kind) {
    var d = num(discount), s = num(slippageBps);
    if (d === null || s === null) return null;
    var tax = kind === 'etf' ? COST.taxEtf : COST.taxStock;
    var commission = 2 * COST.commission * d, slippage = 2 * s / 1e4;
    return { commission: commission, tax: tax, slippage: slippage, total: commission + tax + slippage };
  }

  /** Normalise the decoded "?h=..&thr=.." pairs -> { p: settings, notes: [why a value was changed], given: any known key present }. */
  function readQuery(q) {
    q = obj(q);
    var p = { h: DEFAULTS.h, thr: DEFAULTS.thr, exec: DEFAULTS.exec, disc: DEFAULTS.disc, years: DEFAULTS.years };
    var notes = [], given = false;
    ['h', 'thr', 'exec', 'disc', 'years'].forEach(function (k) { if (hasOwn.call(q, k)) given = true; });
    var v;
    if (hasOwn.call(q, 'h')) {
      v = Number(q.h);
      if (v === 1 || v === 5 || v === 20) p.h = v;
      else notes.push('預測期間 h=' + q.h + ' 無效（只能是 1、5、20），已改用 ' + DEFAULTS.h);
    }
    if (hasOwn.call(q, 'thr')) {
      v = q.thr === '' ? NaN : Number(q.thr);
      if (isFinite(v)) {
        p.thr = round(Math.max(THR_MIN, Math.min(THR_MAX, v)), 2);
        if (p.thr !== round(v, 2)) notes.push('機率門檻 thr=' + q.thr + ' 超出可調範圍 ' + F.fmtPct(THR_MIN, 0) + '–' + F.fmtPct(THR_MAX, 0) + '，已改為 ' + F.fmtPct(p.thr, 0));
      } else notes.push('機率門檻 thr=' + q.thr + ' 無法解讀，已改用 ' + F.fmtPct(DEFAULTS.thr, 0));
    }
    if (hasOwn.call(q, 'exec')) {
      if (q.exec === 'next_open' || q.exec === 'close') p.exec = q.exec;
      else notes.push('成交方式 exec=' + q.exec + ' 無效，已改用 ' + execLabel(DEFAULTS.exec));
    }
    if (hasOwn.call(q, 'disc')) {
      v = q.disc === '' ? NaN : Number(q.disc);
      if (isFinite(v) && v > 0 && v <= 1) p.disc = round(v, 3);
      else notes.push('手續費折扣 disc=' + q.disc + ' 無效（必須大於 0 且不超過 1），已改用 ' + discLabel(DEFAULTS.disc));
    }
    if (hasOwn.call(q, 'years')) {
      v = q.years === '' ? NaN : Number(q.years);
      if (isFinite(v) && v > 0) {
        p.years = nearest(YEARS, v);
        if (p.years !== v) notes.push('驗證年數 years=' + q.years + ' 不在可選範圍（' + YEARS.join('、') + ' 年），已改為最接近的 ' + p.years + ' 年');
      } else notes.push('驗證年數 years=' + q.years + ' 無效，已改用 ' + DEFAULTS.years + ' 年');
    }
    return { p: p, notes: notes, given: given };
  }

  function queryOf(p) {
    return TQ.router.buildQuery({ h: p.h, thr: round(p.thr, 2), exec: p.exec, disc: round(p.disc, 3), years: p.years });
  }
  function hashFor(code, p) { return '#/backtest/' + encodeURIComponent(code) + (p ? '?' + queryOf(p) : ''); }
  function sameSettings(a, b) {
    return !!a && !!b && a.h === b.h && Math.abs(a.thr - b.thr) < 1e-9 && a.exec === b.exec && Math.abs(a.disc - b.disc) < 1e-9 && a.years === b.years;
  }
  function copyOf(p) { return { h: p.h, thr: p.thr, exec: p.exec, disc: p.disc, years: p.years }; }

  // ====================================================================================== small renderers
  function line(content) { return h('span', { class: 'kpi-line' }, content); }
  function chipLine(chip) { return chip ? h('span', { class: 'kpi-line' }, chip) : null; }

  function sigChip(p, what) {
    var v = num(p);
    if (v === null) return null;
    return v < 0.05 ? ui.chip('單項檢定顯著', 'info', { title: (what || '') + ' p=' + F.fmtP(v) + ' < 0.05（未做多重檢定校正；最終結論見上方「驗證結論」）' })
      : ui.chip('不顯著', 'neutral', { title: (what || '') + ' p=' + F.fmtP(v) + ' ≥ 0.05：結果可能只是運氣' });
  }

  /** IC significance chip (audit S-02): the p-value is two-sided, so "significant" would also be shown for a significantly NEGATIVE
   *  ranking.  Only a positive IC can carry the chip; zero or negative gets an explicit "no positive information" chip. */
  function icChip(ic, p) {
    var v = num(ic);
    if (v === null) return null;
    if (v <= 0) return ui.chip('無正向排序資訊', 'neutral', { title: 'IC ≤ 0：預期報酬較高的預測，實際報酬並沒有比較高；不論 p 值多小都不代表模型有排序能力。' });
    return sigChip(p, 'IC');
  }

  // ---- result header ----------------------------------------------------------------------------------------------
  function resultHeader(js, rep, p, code, actions) {
    var per = obj(rep.period), info = F.verdictInfo(obj(rep.verdict).level);
    var name = rep.name || '';
    var nF = num(per.n_forecasts), nS = num(per.n_scored), nT = num(per.n_tie_excluded);
    var pending = nF !== null && nS !== null && nT !== null ? nF - nS - nT : null;
    var meta = [
      { label: '樣本外期間', value: per.start ? F.fmtDate(per.start) + ' ~ ' + F.fmtDate(per.end) : NA,
        tip: '「前推式」驗證：從這段期間的第一天起，每天只用「當天以前」的資料重新預測，再與之後實際發生的結果比對。模型從沒看過這段期間的未來。' },
      { label: '預測次數', value: nF === null ? NA : F.fmtInt(nF) + ' 次',
        tip: '期間內每個交易日各做一次預測。最後 ' + p.h + ' 天的結果尚未發生，不列入評分；實際漲跌幅為 0 的日子沒有方向，也不列入。' },
      { label: '實際評分', value: nS === null ? NA : F.fmtInt(nS) + ' 次' + (nT !== null && nT > 0 ? '（持平排除 ' + F.fmtInt(nT) + '）' : '') + (pending !== null && pending > 0 ? '（未到期 ' + F.fmtInt(pending) + '）' : '') },
      { label: '資料截至', value: js.asof ? F.fmtDate(js.asof) : NA }
    ];
    return ui.card('驗證結果', [
      h('div', { class: 'bt-res-head' },
        h('p', { class: 'bt-res-title' },
          h('strong', null, name || code), ' ', h('span', { class: 'muted' }, code), ' ',
          ui.chip(hzDef(p.h).label + '預測', 'muted'),
          ui.chip(info.label, TONE_CHIP[info.tone] || 'neutral', { icon: info.icon, title: '驗證結論等級（顏色不代表漲跌）' }),
          js.stale === true ? ui.chip('資料可能過期', 'warn', { icon: '⚠' }) : null),
        actions),
      h('dl', { class: 'bt-meta' }, meta.map(function (m) {
        return h('div', { class: 'bt-meta-item' }, h('dt', null, m.label, m.tip ? ui.infoTip(m.tip) : null), h('dd', null, m.value));
      })),
      js.computed_at ? h('p', { class: 'bt-computed muted' }, '計算於 ' + F.fmtDateTime(js.computed_at) + '；相同設定的結果會被快取，再次執行幾乎立即完成。') : null
    ], { class: 'bt-head-card', sub: '同一個模型、同一套程式碼，在過去這段期間逐日「只用當時已知資料」預測的真實紀錄。' });
  }

  // ---- verdict -------------------------------------------------------------------------------------------------------
  var LEVEL_HELP = [
    ['no_edge', '無預測優勢', '預測沒有比簡單基準（例如永遠猜漲）好到統計上說得通的程度；機率應視為「不確定」。這是最常見、也是最誠實的結論。'],
    ['weak_edge', '微弱優勢', '至少兩項檢定同時顯示優勢，且經多重檢定校正後仍顯著，但證據有限；多半不足以覆蓋交易成本。'],
    ['moderate_edge', '中等優勢', '多項檢定各自達 1% 顯著水準，優勢相當一致；仍需持續觀察，因為市場會改變。'],
    ['strong_edge', '顯著優勢', '三項以上檢定達 0.1% 顯著、樣本充足（至少 500 次），且扣成本後的策略顯著勝過買進持有。極為罕見。']
  ];

  function verdictCard(rep) {
    return ui.card('驗證結論', [
      ui.verdictBox(rep.verdict),
      h('details', { class: 'disclosure' },
        h('summary', null, '結論等級怎麼判定？'),
        h('p', { class: 'bt-text' }, '依四項「模型是否比免費的簡單基準更好」的單尾檢定（方向準確率、Brier 機率預測、排序相關 IC、報酬預測誤差）綜合判定，規則刻意保守：只靠單一檢定顯著不算數，並且做了多重檢定校正。只重現整體上漲比例的模型一定落在「無預測優勢」。'),
        h('ul', { class: 'note-list' }, LEVEL_HELP.map(function (l) { return h('li', null, h('strong', null, l[1] + '：'), l[2]); })))
    ], { class: 'bt-verdict-card', sub: '先看結論，再看下面的數字。顏色只表示證據強度（灰＝無、琥珀＝弱、藍＝中、紫＝強），與漲跌無關。' });
  }

  // ---- direction / probability / IC -------------------------------------------------------------------------------------
  function qualityCard(rep, p) {
    var d = obj(rep.direction), pr = obj(rep.probabilistic), rt = obj(rep.returns);
    var best = d.best_baseline;
    var base = obj(d.baselines);
    var bestVal = num(own(base, best));
    var n = num(d.n);
    var kpis = h('div', { class: 'kpi-row bt-kpis' },
      ui.kpi('方向準確率', num(d.accuracy) === null ? NA : F.fmtPct(d.accuracy, 1),
        [line('最佳基準「' + F.baselineName(best) + '」' + (bestVal === null ? NA : F.fmtPct(bestVal, 1))),
          line('95% 信賴區間 ' + F.fmtCI(d.ci95, 1)),
          line('對最佳基準 p = ' + F.fmtP(d.p_value_vs_best_baseline) + '　對丟銅板 p = ' + F.fmtP(d.p_value_vs_coin)),
          chipLine(sigChip(d.p_value_vs_best_baseline, '方向準確率對最佳基準'))],
        { tip: '模型在樣本外預測「漲或跌」猜對的比例（上漲機率 ≥ 50% 視為預測上漲）。重點不是高於 50%，而是要明顯高於「不用任何模型」的最佳簡單基準（例如永遠猜漲）。95% 信賴區間（Wilson 區間）是真實準確率可能落在的範圍；p 值是「若模型其實沒有優勢，僅靠運氣就達到這麼高」的機率，通常要小於 0.05 才算顯著。' + (p.h > 1 ? '預測期間大於 1 天時，相鄰預測高度重疊，檢定已改用有效樣本數（約 n ÷ ' + p.h + '）。' : '') }),
      ui.kpi('Brier 技能分數', num(pr.brier_skill) === null ? NA : signedPct(pr.brier_skill, 2),
        [line('Brier ' + F.fmtNum(pr.brier, 4) + ' vs 基準 ' + F.fmtNum(pr.brier_baseline, 4)),
          num(pr.brier_skill_oracle) === null ? null : line('對事後基準 ' + signedPct(pr.brier_skill_oracle, 2)),
          line('校準誤差 ECE ' + F.fmtPct(pr.ece, 1) + (num(pr.ece_noise_floor) === null ? '' : '（雜訊底限約 ' + F.fmtPct(pr.ece_noise_floor, 0) + '）'))],
        { tip: 'Brier 分數＝「預測上漲機率」與實際結果（漲＝1、跌＝0）的平均平方誤差，越小越好。技能分數＝1 − 模型 Brier ÷ 基準 Brier。基準是「因果氣候基準」：每個預測日只用當時已知的歷史上漲比例（下方的「對事後基準」則用整段驗證期的上漲比例，是當時不可能知道的最嚴格標準）。大於 0 才代表機率預測有加分；接近 0 或為負代表機率沒有提供額外資訊。ECE 是預測機率與實際頻率的平均落差，越小代表機率越誠實；即使是完全校準的預測，分箱的雜訊也會讓 ECE 不為 0（「雜訊底限」，預測期間越長越高），低於底限的 ECE 不能解讀為不準。' + (pr.p_up_event === 'up_given_move' ? '平盤（收盤價不變）不是漲也不是跌，不計入機率評分。' : '') }),
      ui.kpi('預測排序相關 IC', num(rt.ic) === null ? NA : F.fmtSigned(rt.ic, 3),
        [line('雙尾 p = ' + F.fmtP(rt.ic_pvalue)), chipLine(icChip(rt.ic, rt.ic_pvalue))],
        { tip: '資訊係數（Spearman 等級相關）：模型預測的「預期報酬高低」與實際報酬高低的排序有多一致。+1 完全一致、0 毫無關係、−1 完全相反。個股實務上 0.02–0.05 已算不錯，但一定要搭配 p 值：樣本夠大時，微小的相關也會顯著，樣本小時，偶然的相關可能看起來很高。' }));

    var accuracy = num(d.accuracy);
    var rows = [{ name: '模型（上漲機率 ≥ 50% 即預測上漲）', acc: accuracy, model: true, best: false }];
    ['always_up', 'prev_sign', 'train_majority'].concat(Object.keys(base).filter(function (k) { return ['always_up', 'prev_sign', 'train_majority'].indexOf(k) < 0; })).forEach(function (k) {
      if (hasOwn.call(base, k)) rows.push({ name: F.baselineName(k), acc: num(base[k]), model: false, best: k === best });
    });
    var baseTable = ui.table([
      { key: 'name', label: '預測方式', rowHeader: true, render: function (r) { return h('span', null, r.name, r.best ? ui.chip('最佳基準', 'muted') : null); } },
      { key: 'acc', label: '準確率', num: true, render: function (r) { return r.acc === null ? NA : F.fmtPct(r.acc, 1); } },
      { key: 'gap', label: '模型 − 此基準', num: true, render: function (r) {
        return r.model || r.acc === null || accuracy === null ? NA : F.fmtPP(accuracy - r.acc, 1); },
        tip: '百分點（pp）差距。正值代表模型比該基準準；要看是否「穩定且顯著」地為正，而不是單次的數字。' }
    ], rows, { caption: '方向準確率與簡單基準比較', dense: true, rowClass: function (r) { return r.model ? 'row-total' : null; } });

    var hc = obj(d.high_conf);
    var hcN = num(hc.n);
    var hcText = hcN === null || hcN === 0
      ? '高信心預測（上漲機率 ≥ ' + F.fmtPct(hc.threshold == null ? 0.6 : hc.threshold, 0) + ' 或 ≤ ' + F.fmtPct(1 - (hc.threshold == null ? 0.6 : hc.threshold), 0) + '）：驗證期間內一次也沒有出現。模型很少對方向有強烈把握，這在效率市場中是正常的。'
      : '高信心預測（上漲機率 ≥ ' + F.fmtPct(hc.threshold, 0) + ' 或 ≤ ' + F.fmtPct(1 - hc.threshold, 0) + '）共 ' + F.fmtInt(hcN) + ' 次，準確率 ' + F.fmtPct(hc.accuracy, 1) + '，對基準 p = ' + F.fmtP(hc.p_value) + '。';

    return ui.card('預測品質', [
      kpis,
      h('div', { class: 'bt-block' },
        h('h3', { class: 'sub-title' }, '方向準確率 vs 簡單基準', ui.infoTip('「基準」是完全不用模型、也不看未來的簡單做法：永遠猜漲、沿用前一期漲跌方向、猜訓練期較常出現的方向。模型要有價值，就必須在樣本外穩定勝過它們。')),
        baseTable,
        h('p', { class: 'bt-note' }, hcText),
        pr.p_up_event === 'up_given_move' ? h('p', { class: 'bt-note' },
          '機率評分的事件是「收盤上漲，且排除收盤價不變（平盤）的日子」，所以' + (num(pr.n) === null ? '' : '評分樣本 ' + F.fmtInt(pr.n) + ' 次，') + '平盤不計入。上漲機率是「在收盤價有變動的前提下」的機率。') : null,
        n !== null && n < 250 ? h('p', { class: 'bt-note bt-warn' }, '評分樣本只有 ' + F.fmtInt(n) + ' 次，統計檢定的力量很低，數字僅供參考。') : null)
    ], { class: 'bt-quality-card', sub: '機率有沒有道理？排序有沒有資訊？所有數字都是「樣本外」——模型預測當下完全沒看過答案。' });
  }

  // ---- strategy vs buy and hold ------------------------------------------------------------------------------------------
  /** True when the strategy never held a position in the test window (no trades, or zero exposure). */
  function neverTraded(stats) {
    return num(stats.n_trades) === 0 || num(stats.exposure) === 0;
  }

  function strategyCard(rep, p) {
    var st = obj(obj(rep.strategy).stats), bm = obj(obj(rep.strategy).benchmark);
    var strat = obj(rep.strategy);
    var idle = neverTraded(st);
    var kpis = h('div', { class: 'kpi-row bt-kpis' },
      ui.kpi('策略夏普比率', num(st.sharpe) === null ? NA : F.fmtNum(st.sharpe, 2),
        idle ? [line('買進持有 ' + F.fmtNum(bm.sharpe, 2)), line('策略全程空手，沒有夏普比率')] :
          [line('買進持有 ' + F.fmtNum(bm.sharpe, 2)),
            line('兩者差的 95% 信賴區間 ' + F.fmtCIn(strat.sharpe_diff_ci95, 2)),
            line('PSR ' + F.fmtPct(st.psr, 0) + '　DSR ' + F.fmtPct(st.dsr, 0))],
        { tip: '夏普比率＝年化平均報酬 ÷ 年化波動（無風險利率以 0 計），衡量每承擔一單位風險賺到多少。策略已扣交易成本；買進持有不扣成本（對策略較嚴格）。「差的 95% 信賴區間」以隨機重抽樣（stationary bootstrap）估計：若區間包含 0，就無法確定策略真的比買進持有好。PSR＝考慮樣本長度、偏態、峰態後真實夏普大於 0 的機率；DSR 再扣除「試了很多組參數才挑到」的運氣成分。' }),
      ui.kpi('最大回撤', num(st.max_dd) === null ? NA : F.fmtPct(st.max_dd, 1),
        idle ? [line('買進持有 ' + F.fmtPct(bm.max_dd, 1)), line('空手期間沒有回撤')] :
          [line('買進持有 ' + F.fmtPct(bm.max_dd, 1)), line('卡瑪比率 ' + F.fmtNum(st.calmar, 2) + ' vs ' + F.fmtNum(bm.calmar, 2))],
        { tip: '期間內資產從最高點一路下跌到最低點的最大跌幅，越小代表過程越平順。空手的日子不會有回撤，所以策略的回撤通常比買進持有小——這是少持有的代價（也少賺），不一定代表策略聰明。' }),
      ui.kpi('交易次數與周轉', num(st.n_trades) === null ? NA : F.fmtInt(st.n_trades) + ' 次',
        idle ? [line('沒有進出場，不產生交易成本')] :
          [line('年換手率 ' + F.fmtNum(st.turnover_per_year, 1) + ' 倍'), line('交易勝率 ' + (num(st.n_trades) > 1 ? F.fmtPct(st.trade_hit_rate, 1) : NA))],
        { tip: '交易次數＝完整的買進與賣出各算一次進出。年換手率＝每年買賣金額 ÷ 資金，越高，交易成本拖累越大（來回一次約 0.4%–0.6%）。交易勝率＝獲利的交易占全部交易的比例；只有一兩筆交易時沒有統計意義。' }),
      ui.kpi('平均持倉比例', num(st.exposure) === null ? NA : F.fmtPct(st.exposure, 0),
        [line('其餘時間空手（現金）')],
        { tip: '期間內平均有多少比例的資金在持有股票。比例低代表策略大部分時間選擇不進場；買進持有永遠是 100%。報酬要和持倉比例一起看，才知道是「選時間」還是單純「少持有」。' }));

    var lead = [
      h('p', { class: 'bt-note' }, '策略規則：' + (strat.rule || NA) + '。每天用當天收盤後的預測決定部位；成交方式：' + execLabel(p.exec) + (p.exec === 'close' ? '（假設收盤集合競價就能以收盤價成交，較樂觀）' : '（訊號隔天開盤才成交，較貼近實務）') + '。'),
      idle ? ui.banner('warn', '在「' + hzDef(p.h).label + '預測、機率門檻 ' + F.fmtPct(p.thr, 0) + '」這組設定下，策略從未進場：校準後的上漲機率沒有同時達到門檻，且扣成本後的預期報酬為正，所以全程空手，報酬為 0，也沒有夏普比率可算。這本身就是資訊：模型很少對方向有足夠把握。可以試著調低門檻或改看其他預測期間，但門檻越低，越接近「隨便買」。', { title: '策略全程空手' }) : null
    ];

    var cols = [{ label: '策略（已扣成本）', stats: st }, { label: '買進持有（不扣成本）', stats: bm }];
    var ciKv = ui.kv([
      { label: '策略夏普 95% 信賴區間', value: F.fmtCIn(st.sharpe_ci95, 2), tip: '以 stationary bootstrap（保留報酬的時間相依性）重抽樣得到的夏普比率區間。區間很寬代表夏普比率非常不穩定。' },
      { label: '買進持有夏普 95% 信賴區間', value: F.fmtCIn(bm.sharpe_ci95, 2) },
      { label: '夏普差（策略 − 買進持有）95% 信賴區間', value: F.fmtCIn(strat.sharpe_diff_ci95, 2), tip: '若區間跨過 0，就不能說策略的風險調整報酬優於買進持有。' }
    ]);
    return ui.card('假想策略 vs 買進持有', [
      lead,
      kpis,
      h('div', { class: 'bt-block' },
        h('h3', { class: 'sub-title' }, '績效統計比較'),
        ui.statsTable(cols, { caption: '策略與買進持有的績效統計' }),
        ciKv)
    ], { class: 'bt-strategy-card', sub: '把預測變成買賣規則，計入手續費、證交稅與滑價後，看看比單純買進持有如何。' });
  }

  // ---- charts ------------------------------------------------------------------------------------------------------------
  function pctAxis(v, step) { return F.fmtPct(v, step !== undefined && step < 0.01 ? 1 : 0); }

  function equityCard(rep, p, sc) {
    var c = obj(obj(rep.strategy).curve);
    var dates = arr(c.dates);
    var st = obj(obj(rep.strategy).stats), bm = obj(obj(rep.strategy).benchmark);
    function last(a) { for (var i = a.length - 1; i >= 0; i--) if (num(a[i]) !== null) return num(a[i]); return null; }
    var card = TQ.chartCard('資產曲線、持倉與回撤', function (pal) {
      return {
        dates: dates, n: dates.length, emptyText: '沒有資產曲線資料',
        a11y: '資產曲線圖：策略（已扣成本）期末淨值 ' + F.fmtNum(last(arr(c.strategy)), 2) + '，買進持有期末淨值 ' + F.fmtNum(last(arr(c.buyhold)), 2) + '，起點皆為 1；下方為持股狀態與回撤。',
        panels: [
          { title: '資產淨值（起點 = 1）', weight: 3, ticks: 4, fmt: function (v) { return F.fmtNum(v, 2); }, refLines: [{ v: 1 }],
            series: [
              { key: 'strategy', label: '策略（已扣成本）', shortLabel: '策略', color: pal.accent, values: arr(c.strategy), width: 2.2, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); }, tipFmt: function (v) { return F.fmtNum(v, 3); } },
              { key: 'buyhold', label: '買進持有（不扣成本）', shortLabel: '買進持有', color: pal.gray, values: arr(c.buyhold), width: 1.8, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); }, tipFmt: function (v) { return F.fmtNum(v, 3); } }] },
          { title: '策略部位（持股 / 空手）', weight: 1, domain: [0, 1], tickLabels: { 0: '空手', 1: '持股' },
            series: [{ key: 'position', label: '策略持股', color: pal.accentMid, values: arr(c.position), kind: 'step', tipFmt: function (v) { return v >= 0.995 ? '持股' : v <= 0.005 ? '空手' : F.fmtPct(v, 0) + ' 持股'; } }] },
          { title: '回撤（距前高的跌幅）', weight: 1.4, includeZero: true, ticks: 3, fmt: pctAxis,
            series: [
              { key: 'dds', label: '策略回撤', color: pal.accent, values: arr(c.drawdown_strategy), kind: 'area', base: 0, fill: pal.accentSoft, width: 1.5, tipFmt: function (v) { return F.fmtPct(v, 1); } },
              { key: 'ddb', label: '買進持有回撤', color: pal.gray, values: arr(c.drawdown_buyhold), width: 1.4, tipFmt: function (v) { return F.fmtPct(v, 1); } }] }
        ]
      };
    }, {
      kind: 'timeSeries', heightFn: function (w) { return w < 480 ? 440 : w < 900 ? 480 : 520; },
      sub: '滑鼠移到圖上（手機：手指橫向滑動；鍵盤：聚焦後按 ← →）可看各日數值。',
      tip: '藍線是策略（已扣手續費、證交稅、滑價），灰線是從頭到尾持有（不扣成本）。中間的色塊是策略持股的時段；下方是兩者各自從歷史高點算起的跌幅。',
      note: neverTraded(st) ? '策略全程空手，所以藍線貼在 1.00；灰線是同期買進持有。' :
        '買進持有不扣交易成本，並以還原權息價（含股利）計算。期末淨值：策略 ' + F.fmtNum(last(arr(c.strategy)), 2) + '、買進持有 ' + F.fmtNum(last(arr(c.buyhold)), 2) + '；總報酬：策略 ' + signedPct(st.total_return, 1) + '、買進持有 ' + signedPct(bm.total_return, 1) + '。'
    });
    sc.add(card);
    return card;
  }

  function calibrationCard(rep, sc) {
    var pr = obj(rep.probabilistic);
    var card = TQ.chartCard('校準圖：預測機率 vs 實際頻率', { bins: arr(pr.calibration) }, {
      kind: 'calibration', height: 340,
      sub: '校準誤差 ECE ' + F.fmtPct(pr.ece, 1) + '。點越貼近虛線，代表「模型說 60% 就真的約 60% 上漲」。',
      tip: '把所有預測依機率分成幾組（每組樣本數相近），比較「該組平均預測機率」與「該組實際上漲的比例」。圓點大小代表樣本數，細線是 95% 信賴區間；底部長條是各組樣本數。若所有點都擠在 50% 附近，代表模型幾乎沒有表達出方向的把握。',
      note: '預測機率若集中在很窄的區間（例如 45%–55%），圖上就只會看到幾個靠得很近的點：這是模型「沒有把握」的樣子，不是圖壞了。'
    });
    sc.add(card);
    return card;
  }

  function rollingCard(rep, sc) {
    var r = obj(rep.rolling);
    var dates = arr(r.dates), win = num(r.window);
    var card = TQ.chartCard('滾動方向準確率 vs 基準', function (pal) {
      return {
        dates: dates, n: dates.length, emptyText: '資料不足，無法計算滾動準確率',
        a11y: '滾動' + (win === null ? '' : win + '日') + '方向準確率與基準（同期實際上漲比例）的時間序列圖。',
        panels: [{ title: '近 ' + (win === null ? NA : win) + ' 個預測的方向準確率', ticks: 4, fmt: pctAxis, refLines: [{ v: 0.5, label: '50%' }],
          series: [
            { key: 'acc', label: '模型準確率', shortLabel: '模型', color: pal.accent, values: arr(r.accuracy), width: 2, endLabel: true, endFmt: function (v) { return F.fmtPct(v, 1); }, tipFmt: function (v) { return F.fmtPct(v, 1); } },
            { key: 'base', label: '基準（永遠猜漲）', shortLabel: '基準', color: pal.gray, values: arr(r.base_rate), width: 1.6, dash: [5, 4], endLabel: true, endFmt: function (v) { return F.fmtPct(v, 1); }, tipFmt: function (v) { return F.fmtPct(v, 1); } }] }]
      };
    }, {
      kind: 'timeSeries', height: 300,
      sub: '模型準確率若長期貼著或低於虛線，就代表沒有勝過「永遠猜漲」。',
      tip: '每個點是「最近 ' + (win === null ? '數百' : win) + ' 次預測」的準確率。虛線是同一個視窗內實際上漲的比例（也就是永遠猜漲的準確率）。短暫高於虛線很常見（運氣）；有意義的優勢必須長期、穩定地高於虛線。'
    });
    sc.add(card);
    return card;
  }

  function pitCard(rep, sc) {
    var rt = obj(rep.returns), d = obj(rep.direction);
    var hz = num(rep.horizon) || 1;
    var nEff = num(d.n) === null ? undefined : Math.max(1, Math.round(d.n / hz));
    var card = TQ.chartCard('PIT 直方圖：預測分布是否校準', { values: arr(rt.pit_hist), expected: arr(rt.pit_hist).length ? 1 / arr(rt.pit_hist).length : 0.1, n: nEff }, {
      kind: 'histogram', height: 280,
      sub: 'KS 檢定 p = ' + F.fmtP(rt.pit_ks_pvalue) + '（小於 0.05 代表預測分布與現實有系統性落差）。',
      tip: 'PIT（機率積分轉換）：把實際發生的報酬放進「當時預測的分布」，看它落在第幾百分位。如果模型的預測分布完全正確，這些百分位應該均勻分布，每個區間各占 10%。左右兩端偏高＝實際波動比模型預期大（低估風險）；中間偏高＝高估風險；整體偏左或偏右＝系統性高估或低估報酬。灰色帶是純屬隨機波動時的 95% 範圍。'
    });
    sc.add(card);
    return card;
  }

  // ---- coverage / buckets --------------------------------------------------------------------------------------------------
  function coverageOf(cov, level) {
    cov = obj(cov);
    var best = null, bd = 0.005;
    Object.keys(cov).forEach(function (k) { var d = Math.abs(Number(k) - level); if (d < bd) { bd = d; best = k; } });
    return best === null ? null : num(cov[best]);
  }

  function coverageJudge(nominal, actual) {
    if (actual === null) return NA;
    var d = actual - nominal;
    if (Math.abs(d) <= 0.03) return '接近預期';
    return d < 0 ? '偏低：實際風險比預期大' : '偏高：區間偏寬';
  }

  function returnsCard(rep) {
    var rt = obj(rep.returns);
    var rows = [0.5, 0.8, 0.9].map(function (lv) {
      var a = coverageOf(rt.coverage, lv);
      return { nominal: lv, actual: a, diff: a === null ? null : a - lv, judge: coverageJudge(lv, a) };
    });
    var t = ui.table([
      { key: 'nominal', label: '預測區間', rowHeader: true, render: function (r) { return F.fmtPct(r.nominal, 0) + ' 區間'; } },
      { key: 'actual', label: '實際涵蓋', num: true, render: function (r) { return r.actual === null ? NA : F.fmtPct(r.actual, 1); },
        tip: '實際報酬落在預測區間內的比例。理想值等於名目涵蓋率。' },
      { key: 'diff', label: '差距', num: true, render: function (r) { return r.diff === null ? NA : F.fmtPP(r.diff, 1); } },
      { key: 'judge', label: '判讀' }
    ], rows, { caption: '預測區間的名目涵蓋率與實際涵蓋率', dense: true });
    var kv = ui.kv([
      { label: 'Kupiec 涵蓋率檢定 p 值', value: F.fmtP(rt.kupiec_pvalue), tip: '檢定「90% 區間的實際涵蓋率是否等於 90%」。p 值小（< 0.05）代表區間的寬度與實際不符。' },
      { label: 'PIT 均勻性 KS 檢定 p 值', value: F.fmtP(rt.pit_ks_pvalue) },
      { label: '報酬預測 OOS R²', value: num(rt.oos_r2) === null ? NA : signedPct(rt.oos_r2, 2),
        tip: '樣本外 R²（Campbell–Thompson）：模型的預期報酬比「直接預測歷史平均」少了多少平方誤差。大於 0 才代表有加分；個股日報酬的 R² 通常貼近 0 甚至為負。' },
      { label: 'RMSE：模型 / 基準', value: F.fmtPct(rt.rmse_model, 3) + ' / ' + F.fmtPct(rt.rmse_naive, 3), tip: '報酬預測的均方根誤差（越小越好）。基準＝零預測與歷史平均預測中誤差較小者。' },
      { label: 'Diebold–Mariano 統計量（p）', value: F.fmtNum(rt.dm_stat, 2) + '（' + F.fmtP(rt.dm_pvalue) + '）', tip: '檢定模型的平方誤差是否顯著小於基準（已用 Newey–West 修正重疊預測的相關性）。統計量為負代表模型誤差較大。' },
      { label: '分位數損失 (pinball)', value: F.fmtNum(rt.pinball_loss, 5), tip: '預測區間（分位數）的整體損失，越小越好；用來比較不同模型的分布預測品質。' }
    ]);
    return ui.card('報酬預測與區間涵蓋', [
      h('h3', { class: 'sub-title' }, '預測區間有沒有說實話？'), t,
      h('div', { class: 'bt-block' }, kv)
    ], { class: 'bt-returns-card', sub: '預測的不只是方向，還有「可能的範圍」。名目 90% 的區間，實際上應該約 90% 的時間包住結果。' });
  }

  function bucketCard(rep) {
    var rows = arr(obj(rep.direction).by_confidence).filter(function (b) { return b && typeof b === 'object'; });
    function range(b) {
      var lo = num(b.lo), hi = num(b.hi);
      return lo === null || hi === null ? (typeof b.bucket === 'string' ? b.bucket : NA) : F.fmtPct(lo, 0) + ' – ' + F.fmtPct(hi, 0);
    }
    var t = ui.table([
      { key: 'bucket', label: '預測上漲機率', rowHeader: true, value: function (b) { return num(b.lo); }, render: range, sortable: true, defaultDir: 'asc' },
      { key: 'n', label: '預測次數', num: true, sortable: true, render: function (b) { return F.fmtInt(b.n); } },
      { key: 'mean_p_up', label: '平均預測機率', num: true, render: function (b) { return num(b.mean_p_up) === null ? NA : F.fmtPct(b.mean_p_up, 1); } },
      { key: 'actual_up_rate', label: '實際上漲比例', num: true, render: function (b) { return num(b.actual_up_rate) === null ? NA : F.fmtPct(b.actual_up_rate, 1); },
        tip: '該組預測中，實際上漲的比例。若模型有用，機率越高的組別，實際上漲比例應越高。' },
      { key: 'accuracy', label: '方向準確率', num: true, render: function (b) { return num(b.accuracy) === null ? NA : F.fmtPct(b.accuracy, 1); },
        tip: '該組預測中，方向（以 50% 為界）猜對的比例。' }
    ], rows, { caption: '依預測機率分組的準確率', dense: true, sortKey: 'bucket', sortDir: 'asc', empty: '沒有足夠的資料可分組' });
    return ui.card('信心分組：機率越高，真的越準嗎？', [t,
      h('p', { class: 'bt-note' }, '一個有用的模型，「預測上漲機率」越高的組別，實際上漲的比例應該隨之遞增；如果各組幾乎一樣，代表機率高低沒有傳達資訊。')
    ], { class: 'bt-bucket-card', sub: '把預測依機率高低分組（每組至少要有足夠樣本才可信）。' });
  }

  // ---- parameters, costs, notes --------------------------------------------------------------------------------------------
  function paramsCard(rep) {
    var pa = obj(rep.params), per = obj(rep.period);
    var kind = rep.kind;
    var cb = costBreakdown(pa.commission_discount, pa.slippage_bps, kind);
    var rt = num(pa.roundtrip_cost);
    var costOk = cb && (rt === null || Math.abs(cb.total - rt) < 6e-5);
    var kv = ui.kv([
      { label: '標的類型', value: F.kindName(kind), tip: 'ETF 賣出時的證交稅是 0.1%，股票是 0.3%。' },
      { label: '預測期間', value: hzDef(rep.horizon).label + '（' + hzDef(rep.horizon).long + '）' },
      { label: '買進機率門檻', value: num(pa.threshold) === null ? NA : F.fmtPct(pa.threshold, 0), tip: '校準後的上漲機率達到門檻、且扣成本後預期報酬為正，策略才持有。' },
      { label: '成交方式', value: execLabel(pa.exec) },
      { label: '手續費折扣', value: discLabel(pa.commission_discount) },
      { label: '滑價（每邊）', value: num(pa.slippage_bps) === null ? NA : F.fmtNum(pa.slippage_bps, 1) + ' bp', tip: '1 bp＝0.01%。模擬成交價比理想價差一點點的成本。' },
      { label: '來回交易成本', value: rt === null ? NA : F.fmtPct(rt, 2), tip: '買進再賣出一次的總成本：手續費（買賣各一次）＋證交稅（賣出）＋滑價（買賣各一次）。' },
      { label: '初始訓練長度', value: num(pa.min_train) === null ? NA : F.fmtInt(pa.min_train) + ' 個交易日', tip: '第一個樣本外預測之前，模型至少已看過這麼多天的歷史。' },
      { label: '參數重新估計頻率', value: num(pa.refit_every) === null ? NA : '每 ' + F.fmtInt(pa.refit_every) + ' 個交易日', tip: '模型參數（GARCH、HMM）每隔這麼多天、只用當時以前的資料重新估計一次。' },
      { label: '選擇偏誤修正的試驗次數', value: num(pa.n_trials) === null ? NA : F.fmtInt(pa.n_trials), tip: '計算 DSR（通縮夏普）時假設嘗試過幾組設定。這裡只回測一組，所以為 1。' },
      { label: '評分預測數', value: num(per.n_scored) === null ? NA : F.fmtInt(per.n_scored) }
    ]);
    var costTable = costOk ? ui.table([
      { key: 'item', label: '成本項目', rowHeader: true },
      { key: 'rate', label: '比例（來回一次）', num: true, render: function (r) { return F.fmtPct(r.rate, 3); } },
      { key: 'note', label: '計算方式' }
    ], [
      { item: '手續費（買進＋賣出）', rate: cb.commission, note: '牌告 ' + F.fmtPct(COST.commission, 4) + ' × ' + discLabel(pa.commission_discount) + ' × 2 次' },
      { item: '證券交易稅（賣出）', rate: cb.tax, note: kind === 'etf' ? 'ETF 賣出 0.1%' : '股票賣出 0.3%' },
      { item: '滑價（買進＋賣出）', rate: cb.slippage, note: F.fmtNum(pa.slippage_bps, 1) + ' bp × 2 邊' },
      { item: '合計', rate: cb.total, note: '以上三項相加（買進再賣出一次）' }
    ], { caption: '來回交易成本拆解', dense: true, rowClass: function (r) { return r.item === '合計' ? 'row-total' : null; } }) : null;
    return ui.card('參數與交易成本', [kv, costTable ? h('div', { class: 'bt-block' }, h('h3', { class: 'sub-title' }, '成本拆解'), costTable) : null],
      { class: 'bt-params-card', sub: '這份報告使用的全部設定。台灣的交易成本相當可觀：預測的優勢必須先贏過它。' });
  }

  function notesCard(rep) {
    var notes = arr(rep.notes).filter(function (n) { return typeof n === 'string' && n; });
    return ui.card('假設、限制與注意事項', [
      notes.length ? h('ul', { class: 'note-list' }, notes.map(function (n) { return h('li', null, h('span', null, n)); })) : null,
      h('ul', { class: 'note-list' },
        h('li', null, h('strong', null, '過去不等於未來：'), '樣本外只代表「如果當時就用這個模型」的結果；市場結構、成交量與法規都會改變。'),
        h('li', null, h('strong', null, '挑選偏誤：'), '你是在看過股價之後才挑這檔股票。若挑的是近年表現最好的，買進持有的對照會特別亮眼，驗證結果的解讀也要打折。請同時參考「今日模型觀察」頁的整體驗證。')),
      h('p', { class: 'disclaimer' }, h('span', { class: 'disclaimer-icon', 'aria-hidden': 'true' }, '⚠'), h('span', null, TQ.DISCLAIMER))
    ], { class: 'notes-card' });
  }

  function shortHistoryBanner(rep, p) {
    var per = obj(rep.period), nF = num(per.n_forecasts);
    var wanted = p.years * 245;
    if (nF !== null && nF < wanted * 0.9) {
      return ui.banner('info', '你選的是 ' + p.years + ' 年（約 ' + F.fmtInt(wanted) + ' 個交易日），但這檔標的的歷史資料只夠做 ' + F.fmtInt(nF) + ' 次樣本外預測（約 ' + F.fmtNum(nF / 245, 1) + ' 年），所以驗證期間自動縮短。', { title: '驗證期間比預期短' });
    }
    return null;
  }

  // ====================================================================================== the view
  function title(params) { return '歷史驗證 ' + (params && params.code ? params.code : ''); }

  function mount(host, params, ctx) {
    var code = String(params && params.code || '2330');
    var rq = readQuery(ctx.query);
    var S = { code: code, name: '', kind: '', p: rq.p, shown: null, ran: false, running: false, hasResult: false };      // shown = settings of the displayed result
    var destroyed = false;
    var sc = TQ.scope();            // search box
    var rsc = TQ.scope();           // charts of the current result
    var run = { ctl: null, bar: null, t0: 0, onParent: null };
    sc.add(rsc);

    // ---- layout ------------------------------------------------------------------------------------------------------
    var stockNameEl = h('strong', { class: 'bt-cur-name' }, code);
    var stockCodeEl = h('span', { class: 'muted bt-cur-code' });
    var curLink = h('a', { class: 'btn btn-ghost bt-stock-link', href: TQ.codeHref(code) }, '查看個股預測 →');
    var head = h('div', { class: 'page-head' },
      h('div', { class: 'page-head-text' },
        h('h1', { class: 'page-title' }, '歷史驗證'),
        h('p', { class: 'page-sub' }, '用過去的資料誠實檢驗：這個模型如果當時就拿來預測，到底準不準？是否贏過簡單基準與買進持有？沒有優勢也是合理的答案。')),
      h('div', { class: 'page-head-ctl bt-cur' }, h('span', { class: 'bt-cur-label' }, '目前標的'), h('span', { class: 'bt-cur-id' }, stockNameEl, ' ', stockCodeEl), curLink));
    var slotNotice = h('div', { class: 'slot slot-notice' });
    var slotStatus = h('div', { class: 'slot slot-status' });
    var slotResult = h('div', { class: 'slot slot-result' });
    var formHost = h('div', { class: 'slot slot-form' });
    var view = h('div', { class: 'view view-backtest' }, head, formHost, slotNotice, slotStatus, slotResult);
    host.appendChild(view);

    function alive() { return !destroyed && !ctx.signal.aborted; }

    function paintStock() {
      stockNameEl.textContent = S.name || S.code;
      stockCodeEl.textContent = S.name ? S.code : '';
      curLink.setAttribute('href', TQ.codeHref(S.code));
    }

    // ---- controls ----------------------------------------------------------------------------------------------------
    var sb = sc.add(TQ.searchBox.create({
      id: 'bt-stock', placeholder: '輸入代號或名稱，例如 2330、台積電、0050', label: '搜尋要驗證的股票或 ETF（代號或名稱）',
      onSelect: function (item) {
        if (!item || !item.code) return;
        ctx.nav(hashFor(item.code, S.ran ? S.p : null));
      }
    }));

    var hzSeg = ui.segmented(HORIZONS.map(function (x) { return { value: x.value, label: x.label, title: x.long }; }), S.p.h, function (v) { S.p.h = v; changed(); }, { label: '預測期間' });
    var execSeg = ui.segmented(EXECS.map(function (x) { return { value: x.value, label: x.label }; }), S.p.exec, function (v) { S.p.exec = v; changed(); }, { label: '成交方式' });
    var yearsSeg = ui.segmented(YEARS.map(function (y) { return { value: y, label: y + ' 年' }; }), S.p.years, function (v) { S.p.years = v; changed(); }, { label: '樣本外驗證期間' });

    var thrScale = h('div', { class: 'bt-scale', 'aria-hidden': 'true' }, h('span', null, F.fmtPct(THR_MIN, 0)), h('span', null, '60%'), h('span', null, F.fmtPct(THR_MAX, 0)));
    var thrOut = h('output', { class: 'bt-thr-out', for: 'bt-thr' }, F.fmtPct(S.p.thr, 0));
    var thrInput = h('input', { type: 'range', min: String(THR_MIN), max: String(THR_MAX), step: '0.01', value: S.p.thr.toFixed(2), id: 'bt-thr', class: 'bt-range' });
    function onThr() {
      var v = Number(thrInput.value);
      if (!isFinite(v)) return;
      S.p.thr = round(Math.max(THR_MIN, Math.min(THR_MAX, v)), 2);
      thrOut.textContent = F.fmtPct(S.p.thr, 0);
      thrInput.setAttribute('aria-valuetext', F.fmtPct(S.p.thr, 0));
      changed();
    }
    thrInput.addEventListener('input', onThr);
    thrInput.addEventListener('change', onThr);
    thrInput.setAttribute('aria-valuetext', F.fmtPct(S.p.thr, 0));

    var discSel = h('select', { id: 'bt-disc', class: 'bt-select' });
    function fillDiscOptions() {
      TQ.clear(discSel);
      var list = DISCOUNTS.slice();
      if (list.indexOf(S.p.disc) < 0) { list.push(S.p.disc); list.sort(function (a, b) { return a - b; }); }
      list.forEach(function (d) {
        discSel.appendChild(h('option', { value: String(d), selected: d === S.p.disc }, discLabel(d) + (d === DEFAULTS.disc ? '（預設）' : '')));
      });
      discSel.value = String(S.p.disc);
    }
    fillDiscOptions();
    var costHint = h('span', { class: 'bt-cost-hint' });
    function paintCost() {
      var cb = costBreakdown(S.p.disc, COST.slippageBps, S.kind || 'stock');
      costHint.textContent = cb ? '預估來回交易成本約 ' + F.fmtPct(cb.total, 2) + '（手續費 ' + F.fmtPct(cb.commission, 3) + ' ＋ 證交稅 ' + F.fmtPct(cb.tax, 1) + ' ＋ 滑價 ' + F.fmtPct(cb.slippage, 2) + '，' + (S.kind === 'etf' ? 'ETF' : '股票') + '）' : '';
    }
    discSel.addEventListener('change', function () {
      var v = Number(discSel.value);
      if (!(v > 0 && v <= 1)) return;
      S.p.disc = v;
      paintCost();
      changed();
    });

    var dirtyChip = h('span', { class: 'bt-dirty', role: 'status', hidden: true }, ui.chip('設定已變更，尚未套用', 'warn', { icon: '⚠' }));
    var runBtn = h('button', { type: 'button', class: 'btn btn-primary bt-run', onClick: function () { startRun(); } }, '執行驗證');
    var resetBtn = h('button', { type: 'button', class: 'btn btn-ghost bt-reset', onClick: resetDefaults }, '還原預設值');

    /** Field for controls that are not a single labelable element (segmented radio groups): a span label instead of label[for]. */
    function plainField(label, control, opts) {
      var hintId = TQ.uid('hint');
      control.setAttribute('aria-describedby', hintId);
      return h('div', { class: ['field', opts.class] }, h('span', { class: 'field-label' }, label), control, h('p', { class: 'field-hint', id: hintId }, opts.hint));
    }
    var stockField = ui.field('股票', sb.el, { id: 'bt-stock', class: 'bt-f-stock', hint: '輸入代號或名稱後從清單選取；也可直接輸入完整代號後按 Enter。選擇後會以目前的設定重新驗證。' });
    var stockHint = stockField.querySelector('.field-hint');
    if (stockHint) { sb.el.removeAttribute('aria-describedby'); sb.input.setAttribute('aria-describedby', stockHint.id); }
    var thrField = ui.field(h('span', null, '買進機率門檻 ', thrOut), h('div', { class: 'bt-range-wrap' }, thrInput, thrScale), { id: 'bt-thr', class: 'bt-f-thr',
      hint: '策略只在「校準後上漲機率 ≥ 門檻」且扣成本後預期報酬為正時持有，否則空手。門檻越高，越少進場。',
      tip: '這是策略的進場條件，不影響預測品質（準確率、校準等指標與門檻無關）。' });
    var thrHint = thrField.querySelector('.field-hint');
    if (thrHint) { thrInput.parentNode.removeAttribute('aria-describedby'); thrInput.setAttribute('aria-describedby', thrHint.id); }
    var form = h('form', { class: 'bt-form', 'aria-label': '驗證設定', novalidate: true, onSubmit: function (e) { if (e && e.preventDefault) e.preventDefault(); startRun(); } },
      h('div', { class: 'bt-fields' },
        stockField,
        plainField('預測期間', hzSeg, { class: 'bt-f-h', hint: '驗證「明日」「一週」或「一個月」後的漲跌預測。' }),
        thrField,
        plainField('成交方式', execSeg, { class: 'bt-f-exec', hint: '「下一個開盤價成交」較貼近實務：收盤後才得到訊號，隔天開盤才能下單。「收盤成交」假設能在收盤價買賣，結果偏樂觀。' }),
        ui.field('手續費折扣', discSel, { id: 'bt-disc', class: 'bt-f-disc', hint: costHint }),
        plainField('樣本外驗證期間', yearsSeg, { class: 'bt-f-years', hint: '越長越穩健，但計算較久；上市時間不足時會自動縮短。' })),
      h('div', { class: 'bt-actions' }, runBtn, resetBtn, dirtyChip));
    paintCost();
    formHost.appendChild(ui.card('驗證設定', form, { class: 'bt-form-card', sub: '每次執行都會把設定寫進網址，複製網址即可分享同一份驗證。' }));

    function syncControls() {
      hzSeg.setValue(S.p.h);
      execSeg.setValue(S.p.exec);
      yearsSeg.setValue(S.p.years);
      thrInput.value = S.p.thr.toFixed(2);
      thrInput.setAttribute('aria-valuetext', F.fmtPct(S.p.thr, 0));
      thrOut.textContent = F.fmtPct(S.p.thr, 0);
      fillDiscOptions();
      paintCost();
    }

    function resetDefaults() {
      S.p = copyOf(DEFAULTS);
      syncControls();
      changed();
    }

    function changed() {
      dirtyChip.hidden = !(S.hasResult && !S.running && S.shown && !sameSettings(S.p, S.shown));
    }

    // ---- notices ------------------------------------------------------------------------------------------------------
    function paintNotice(extra) {
      TQ.clear(slotNotice);
      if (rq.notes.length) {
        slotNotice.appendChild(ui.banner('info', h('ul', { class: 'note-list bt-note-list' }, rq.notes.map(function (n) { return h('li', null, n); })), { title: '網址中的部分設定已調整' }));
      }
      if (extra) slotNotice.appendChild(extra);
    }
    paintNotice(null);

    // ---- idle state -------------------------------------------------------------------------------------------------------
    function showIdle(message) {
      TQ.clear(slotStatus);
      if (S.hasResult) return;
      slotStatus.appendChild(ui.card(message ? '已取消' : '尚未執行驗證', [
        ui.emptyState(message || '調整上方設定後，按「執行驗證」。', {
          action: h('p', { class: 'empty-sub' }, '驗證會用同一個模型，從過去的每一天往前推：每天只用當天以前的資料重新預測，再與之後真正發生的結果比對。第一次計算某檔股票通常需要 10–60 秒，完成後會被快取。')
        }),
        h('ul', { class: 'note-list bt-idle-list' },
          h('li', null, '你會看到：結論、方向準確率與簡單基準的比較、機率是否誠實、預測區間有沒有說實話、扣掉交易成本後的假想策略 vs 買進持有。'),
          h('li', null, '結論可能是「沒有統計上顯著的預測優勢」——對大多數股票而言，這才是常態。'))
      ], { class: 'bt-idle-card' }));
    }
    function showResultOrIdle() { if (!S.hasResult) showIdle(); }

    // ---- running ------------------------------------------------------------------------------------------------------------
    function undim() {
      slotResult.classList.remove('bt-stale');
      slotResult.removeAttribute('aria-busy');
    }

    function stopRun() {
      undim();
      if (run.onParent) { try { ctx.signal.removeEventListener('abort', run.onParent); } catch (e) { /* ignore */ } run.onParent = null; }
      if (run.ctl) { try { run.ctl.abort(); } catch (e) { /* ignore */ } run.ctl = null; }
      S.running = false;
      runBtn.disabled = false;
    }

    function elapsedText() {
      var s = Math.max(0, Math.round((nowMs() - run.t0) / 1000));
      return s >= 2 ? '（已等待 ' + s + ' 秒）' : '';
    }

    function startRun() {
      if (!alive()) return;
      stopRun();
      var p = copyOf(S.p);
      S.ran = true;
      S.running = true;
      if (S.hasResult) { slotResult.classList.add('bt-stale'); slotResult.setAttribute('aria-busy', 'true'); }      // the old result stays visible but is dimmed while the new one is computed
      if (typeof ctx.replace === 'function') ctx.replace(hashFor(S.code, p));
      var AC = root.AbortController;
      var ctl = AC ? new AC() : null;
      run.ctl = ctl;
      run.t0 = nowMs();
      function onParent() { if (ctl) ctl.abort(); }
      run.onParent = onParent;
      ctx.signal.addEventListener('abort', onParent);
      dirtyChip.hidden = true;
      var bar = ui.progressBar(0, 0, '送出計算請求…');
      run.bar = bar;
      TQ.clear(slotStatus);
      slotStatus.appendChild(ui.card('正在計算歷史驗證', [
        bar,
        h('p', { class: 'muted' }, '前推式驗證需要逐日重新估計模型，首次計算 ' + S.code + ' 約需 10–60 秒（取決於標的的歷史長度與主機速度）；同樣的設定之後會直接讀取快取。'),
        h('div', { class: 'bt-cancel' }, h('button', { type: 'button', class: 'btn', onClick: cancelRun }, '取消'))
      ], { class: 'bt-progress-card' }));
      ctx.setStatus('開始計算 ' + S.code + ' 的歷史驗證…');
      var my = ctl;
      TQ.api.poll('/api/backtest', { symbol: S.code, h: p.h, thr: p.thr, exec: p.exec, disc: p.disc, years: p.years }, {
        signal: ctl ? ctl.signal : ctx.signal,
        onProgress: function (pr) { if (run.ctl === my) bar.update(pr.done, pr.total, (pr.message || '計算中…') + elapsedText()); }
      }).then(function (js) {
        if (!alive() || run.ctl !== my) return;
        finishRun();
        showReport(js, p);
      }, function (err) {
        if (!alive() || run.ctl !== my || TQ.api.isAbort(err)) return;
        finishRun();
        restoreUrl();
        changed();
        TQ.clear(slotStatus);
        var retry = err && (err.code === 'unknown_symbol' || err.code === 'bad_request') ? null : startRun;
        slotStatus.appendChild(ui.card('歷史驗證', [
          ui.errorBox(err, retry),
          err && err.code === 'unknown_symbol' ? h('p', { class: 'muted' }, '請在上方「股票」欄位搜尋正確的代號或名稱（例如 2330、台積電、0050）。') : null,
          err && err.code === 'insufficient_history' ? h('p', { class: 'muted' }, '歷史驗證需要足夠長的歷史資料（至少約兩年）才能訓練並做樣本外檢驗；新上市的標的暫時無法驗證。') : null
        ], { class: 'bt-error-card', level: 2 }));
        ctx.setStatus('歷史驗證失敗');
      });
    }

    /** After a cancelled / failed run the URL goes back to the settings of the result that is still on screen. */
    function restoreUrl() {
      if (S.hasResult && S.shown && typeof ctx.replace === 'function') ctx.replace(hashFor(S.code, S.shown));
    }

    function finishRun() {
      undim();
      if (run.onParent) { try { ctx.signal.removeEventListener('abort', run.onParent); } catch (e) { /* ignore */ } run.onParent = null; }
      run.ctl = null;
      S.running = false;
      runBtn.disabled = false;
    }

    function cancelRun() {
      stopRun();
      restoreUrl();
      TQ.clear(slotStatus);
      if (!S.hasResult) showIdle('已取消計算。伺服器端的計算可能仍會完成並被快取，稍後再按「執行驗證」通常會很快。');
      else slotStatus.appendChild(ui.banner('info', '已取消。目前顯示的是上一次的結果。'));
      changed();
      ctx.setStatus('已取消歷史驗證');
    }

    // ---- results ------------------------------------------------------------------------------------------------------------
    function showReport(js, p) {
      js = obj(js);
      var rep = obj(js.report);
      if (!rep.direction && !rep.strategy && !rep.verdict) {
        restoreUrl();
        changed();
        TQ.clear(slotStatus);
        slotStatus.appendChild(ui.card('歷史驗證', ui.errorBox(new TQ.ApiError('伺服器回應中沒有驗證結果。', 'bad_response', 200), startRun)));
        return;
      }
      rsc.dispose();
      TQ.clear(slotStatus);
      TQ.clear(slotResult);
      S.hasResult = true;
      S.shown = p;
      S.name = rep.name || S.name;
      S.kind = rep.kind || S.kind;
      paintStock();
      paintCost();
      TQ.router.setLastStock(S.code);
      ctx.setTitle((S.name ? S.name + ' ' : '') + S.code + ' 歷史驗證');

      var copyBtn = null;
      if (root.navigator && root.navigator.clipboard && typeof root.navigator.clipboard.writeText === 'function') {
        copyBtn = h('button', { type: 'button', class: 'btn btn-ghost bt-copy', onClick: function () {
          var done = function (msg) { copyBtn.textContent = msg; root.setTimeout(function () { if (!destroyed) copyBtn.textContent = '複製連結'; }, 1800); };
          try { root.navigator.clipboard.writeText(String(root.location.href)).then(function () { done('已複製 ✓'); }, function () { done('無法複製，請手動複製網址'); }); } catch (e) { done('無法複製，請手動複製網址'); }
        } }, '複製連結');
      }
      var blocks = [];
      function safe(fn, label) {
        try { var n = fn(); if (n) blocks.push(n); } catch (e) {
          TQ.logError(e);
          blocks.push(ui.card(label, ui.errorBox(null)));
        }
      }
      safe(function () { return resultHeader(js, rep, p, S.code, copyBtn); }, '驗證結果');
      safe(function () { return shortHistoryBanner(rep, p); }, '提示');
      if (js.stale === true) blocks.push(ui.banner('warn', '網路暫時無法更新價格資料，這份驗證使用本機快取的資料，可能不包含最近幾天。'));
      safe(function () { return verdictCard(rep); }, '驗證結論');
      safe(function () { return qualityCard(rep, p); }, '預測品質');
      safe(function () { return strategyCard(rep, p); }, '假想策略 vs 買進持有');
      safe(function () { return equityCard(rep, p, rsc); }, '資產曲線');
      safe(function () { return h('div', { class: 'row row-2' }, calibrationCard(rep, rsc), rollingCard(rep, rsc)); }, '校準與滾動準確率');
      safe(function () { return h('div', { class: 'row row-2' }, pitCard(rep, rsc), returnsCard(rep)); }, 'PIT 與區間涵蓋');
      safe(function () { return bucketCard(rep); }, '信心分組');
      safe(function () { return h('div', { class: 'row row-2' }, paramsCard(rep), notesCard(rep)); }, '參數與注意事項');
      blocks.forEach(function (b) { slotResult.appendChild(b); });
      var lv = F.verdictInfo(obj(rep.verdict).level).label;
      ctx.setStatus((S.name || S.code) + ' 歷史驗證完成：' + lv + (obj(rep.verdict).title ? '，' + rep.verdict.title : ''));
      changed();
    }

    // ---- boot --------------------------------------------------------------------------------------------------------------
    paintStock();
    if (rq.given) startRun(); else showIdle();

    return {
      destroy: function () {
        destroyed = true;
        stopRun();
        rsc.dispose();
        sc.dispose();
      }
    };
  }

  TQ.views.backtest = {
    title: title, mount: mount,
    _internals: { readQuery: readQuery, queryOf: queryOf, hashFor: hashFor, costBreakdown: costBreakdown, discLabel: discLabel, coverageOf: coverageOf,
      coverageJudge: coverageJudge, COST: COST, DEFAULTS: DEFAULTS, DISCOUNTS: DISCOUNTS, YEARS: YEARS, neverTraded: neverTraded }
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
