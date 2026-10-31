/*
 * view_picks.js -- 今日模型觀察 (#/picks).
 *
 *   horizon control (明日 / 一週 / 一個月 -> h = 1 / 5 / 20, remembered)  ->  TQ.api.poll('/api/recommendations')
 *   with a progress bar while the server computes.  Honesty first (audit P-09 / P-10, DESIGN 9):
 *     disclaimer strip above the fold  ->  evidence banner built from `evidence` (amber when the universe validation found no edge or the
 *     basket does not beat the universe after costs; grey note + "執行驗證" button when there is none)  ->  market strip  ->
 *     相對強度排名 (leaders 「相對偏多 Top N」 / laggards 「相對偏空 Top N」, ranked by `rank_signal` = mu/sigma, every row with sigma and a
 *     「低波動」 tag, a mark when the edge does not cover costs)  ->  the STRICT lists 偏多候選 (cards) and 偏空／避開  ->  模型配置範例
 *     (100 % cash + the reason when the server assigns no weights)  ->  full sortable table  ->  collapsible 推薦整體歷史驗證 panel
 *     (polls '/api/backtest/universe' only when the user asks; its result also refreshes the evidence banner)  ->  notes.
 *   The words 「建議買進」 / 「偏多·買進」 are never used unless the evidence gate (verdict >= moderate_edge and the basket not losing to
 *   the equal-weight universe after costs) is open: F.signalInfo(sig, unlocked).
 * An empty candidate list is a legitimate (and common) answer and is explained, not hidden.
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
    { value: 1, label: '明日', long: '下一個交易日', short: '明日' },
    { value: 5, label: '一週', long: '未來 5 個交易日', short: '5 日' },
    { value: 20, label: '一個月', long: '未來 20 個交易日', short: '20 日' }
  ];
  function hzDef(v) { for (var i = 0; i < HORIZONS.length; i++) if (HORIZONS[i].value === v) return HORIZONS[i]; return HORIZONS[1]; }

  var CONF_KIND = { '高': 'accent', '中': 'info', '低': 'neutral' };
  var SIG_ORDER = { buy: 0, hold: 1, avoid: 2 };
  var PICKS_SUFFIX = '「模型配置權重」與「模型配置範例」只是資金配置的試算，請搭配「歷史驗證」判讀。';
  var SUPPRESSED = {
    B0: '各檔預測的差異與雜訊無法區分（收縮後所有檔的分數相同），所以不配置任何權重。',
    no_positive_edge: '沒有任何偏多候選同時具有正的淨預期報酬與正的收縮後優勢，所以不配置任何權重。'
  };

  function signedPct(v, dp) { return F.fmtPct(v, dp == null ? 2 : dp, true); }
  function sigDir(sig) { return F.signalInfo(sig).dir; }

  /** list of record objects that have a usable stock code (anything else is dropped, never rendered as "undefined"). */
  function recs(list) {
    return arr(list).filter(function (r) { return r && typeof r === 'object' && typeof r.code === 'string' && r.code; });
  }

  function nameLink(rec) {
    return h('a', { class: 'stock-link', href: TQ.codeHref(rec.code) },
      h('span', { class: 'sl-name' }, rec.name || rec.code),
      h('span', { class: 'sl-code' }, rec.code));
  }

  // ====================================================================================== evidence (DESIGN 9(4), audit P-10)
  /** Normalise the server's `evidence` block.  -> {source:'none'} or the usable fields plus `tone` ('warn' | 'info'), `beats`
   *  (true / false / null) and `unlocked` (buy wording allowed: verdict >= moderate_edge and beats_ew_after_costs is not false). */
  function evidenceInfo(ev, hv) {
    ev = obj(ev);
    var level = typeof ev.verdict_level === 'string' ? ev.verdict_level : null;
    var usable = ev.source === 'universe_validation' && level !== null && F.verdictInfo(level).icon !== '?';
    var hzv = num(ev.horizon);
    if (!usable || (hzv !== null && hv != null && hzv !== hv)) return { source: 'none' };
    var net = num(ev.net_cagr), ew = num(ev.ew_cagr);
    var beats = typeof ev.beats_ew_after_costs === 'boolean' ? ev.beats_ew_after_costs : (net !== null && ew !== null ? net > ew : null);
    return {
      source: 'universe_validation', level: level, ic: num(ev.ic), p: num(ev.ic_pvalue), hit: num(ev.hit_rate), base: num(ev.base_rate),
      computedAt: typeof ev.computed_at === 'string' ? ev.computed_at : null,
      net: net, ew: ew, sigma: num(ev.sigma_cagr), beats: beats,
      tone: (level === 'no_edge' || level === 'weak_edge' || beats === false) ? 'warn' : 'info',
      unlocked: F.levelUnlocksBuy(level) && beats !== false
    };
  }

  /** The same block the server builds (universe_bt.evidence_from_report), from a universe report the user just ran. */
  function evidenceFromReport(rep, hv, at) {
    rep = obj(rep);
    var lvl = obj(rep.verdict).level;
    if (typeof lvl !== 'string' || F.verdictInfo(lvl).icon === '?') return { source: 'none' };
    var top = obj(rep.top), ew = obj(rep.benchmark_equal_weight), sg = obj(rep.benchmark_top_sigma), ic = obj(rep.ic), pk = obj(rep.picks);
    var net = num(top.cagr), eqw = num(ew.cagr);
    return {
      horizon: hv, source: 'universe_validation', verdict_level: lvl, ic: num(ic.mean), ic_pvalue: num(ic.pvalue), hit_rate: num(pk.hit_rate),
      base_rate: num(pk.base_rate), computed_at: at || null, net_cagr: net, ew_cagr: eqw, sigma_cagr: num(sg.cagr),
      beats_ew_after_costs: net !== null && eqw !== null ? net > eqw : null
    };
  }

  function icText(info) {
    return 'IC ' + (info.ic === null ? NA : F.fmtSigned(info.ic, 3)) + '，p=' + F.fmtP(info.p);
  }

  function evidenceLines(info, hz) {
    var lines = [];
    var lbl = F.verdictInfo(info.level).label;
    var after = info.net !== null && info.ew !== null;
    if (info.level === 'no_edge') {
      lines.push('模型的選股排序沒有統計上顯著的預測力（' + icText(info) + '）。以下只是相對位置，非買賣建議。');
    } else if (info.level === 'weak_edge') {
      lines.push('模型的選股排序只有「' + lbl + '」（' + icText(info) + '），證據有限，多半不足以覆蓋交易成本。以下只是相對位置，非買賣建議。');
    } else if (info.beats === false) {
      lines.push('選股排序的統計檢定達「' + lbl + '」（' + icText(info) + '），但扣除交易成本後，前 5 名組合沒有勝過全體等權重：不構成買進建議，以下只是相對位置。');
    } else if (info.beats === true) {
      lines.push('選股排序的統計檢定達「' + lbl + '」（' + icText(info) + '），且扣除成本後前 5 名組合勝過全體等權重。這只代表過去有可用的排序資訊，不保證未來；仍有存活者偏差，非投資建議。');
    } else {
      lines.push('選股排序的統計檢定達「' + lbl + '」（' + icText(info) + '），但尚未確認扣成本後是否勝過全體等權重；非投資建議。');
    }
    var extra = [];
    if (info.hit !== null && info.base !== null) extra.push('精選股上漲比例 ' + F.fmtPct(info.hit, 1) + '（同日全體 ' + F.fmtPct(info.base, 1) + '；這一項沒有控制波動度，低波動股本來就較常上漲，不算獨立證據）');
    if (after) {
      extra.push('扣成本後前 5 名組合年化 ' + F.fmtPct(info.net, 1) + '，全體等權重（不扣成本）年化 ' + F.fmtPct(info.ew, 1) +
        (info.sigma !== null ? '，無預測對照（每天挑預測波動度最高者）年化 ' + F.fmtPct(info.sigma, 1) : ''));
    }
    if (extra.length) lines.push(extra.join('；') + '。');
    if (info.computedAt) lines.push('驗證計算於 ' + F.fmtDateTime(info.computedAt) + '（' + hz.label + '）。');
    return lines;
  }

  /** The banner above everything else.  `acts` = { run(), open() }. */
  function evidenceBanner(info, hz, acts) {
    var btn;
    if (info.source === 'none') {
      btn = h('button', { type: 'button', class: 'btn btn-primary ev-run', onClick: function () { acts.run(); } }, '執行驗證（' + hz.label + '）');
      return ui.banner('neutral', h('div', { class: 'ev-body' },
        h('p', null, '尚未取得「' + hz.label + '」的歷史驗證：下列相對強度排名與偏多候選只是模型的相對看法，尚未經過歷史驗證，請勿當成買賣建議。'),
        h('p', { class: 'ev-sub' }, '驗證會把同一套排序規則套到過去每一天（首次計算需要數分鐘，完成後會快取）。'),
        h('div', { class: 'ev-actions' }, btn)), { title: '歷史驗證：尚無資料' });
    }
    btn = h('button', { type: 'button', class: 'btn btn-ghost ev-open', onClick: function () { acts.open(); } }, '查看完整驗證');
    var lines = evidenceLines(info, hz);
    return ui.banner(info.tone, h('div', { class: 'ev-body' },
      lines.map(function (t, i) { return h('p', { class: i ? 'ev-sub' : null }, t); }),
      h('div', { class: 'ev-actions' }, btn)), { title: '歷史驗證' });
  }

  // ====================================================================================== small pieces
  function confChip(c) {
    return c ? ui.chip('模型一致度 ' + c, CONF_KIND[c] || 'neutral', { title: '依機率偏離 50% 的程度、預期波動與各模型方向是否一致綜合而定。這只是模型內部的一致程度，不是經過歷史驗證的把握。' }) : null;
  }

  function sigChip(sig, unlocked) {
    var info = F.signalInfo(sig, unlocked);
    return info.dir === 'na' ? null : ui.chip(info.label, info.dir, { icon: info.icon });
  }

  var LEAN = {
    bullish: { label: '相對偏多', dir: 'up', icon: '▲', title: '相對強度排名在全體前 20%，且上漲機率高於 50%。' },
    bearish: { label: '相對偏空', dir: 'down', icon: '▼', title: '相對強度排名在全體後 20%，且上漲機率低於 50%。' },
    neutral: { label: '中性', dir: 'flat', icon: '◆', title: '不在前 20% 或後 20%，或上漲機率與排名方向不一致。' }
  };
  function leanChip(lean) {
    var l = typeof lean === 'string' && Object.prototype.hasOwnProperty.call(LEAN, lean) ? LEAN[lean] : null;
    return l ? ui.chip(l.label, l.dir, { icon: l.icon, title: l.title }) : null;
  }

  /** true when sigma is below the universe median (server flag `low_vol`; falls back to comparing with `universe_sigma_median`). */
  function isLowVol(rec, data) {
    if (typeof rec.low_vol === 'boolean') return rec.low_vol;
    var s = num(rec.sigma), m = num(obj(data).universe_sigma_median);
    return s !== null && m !== null ? s < m : false;
  }

  function lowVolTag(rec, data) {
    if (!isLowVol(rec, data)) return null;
    var m = num(obj(data).universe_sigma_median);
    return ui.chip('低波動', 'muted', { title: '預測波動度低於本次掃描全體的中位數' + (m === null ? '' : '（' + F.fmtPct(m, 2) + '）') + '：相對強度偏向低波動個股。' });
  }

  function costMark(rec) {
    return rec.edge_covers_cost === false ? ui.chip('未覆蓋成本', 'warn', { icon: '⚠', title: '預期報酬未覆蓋交易成本' }) : null;
  }

  function metric(label, value, dir, tip) {
    return h('div', { class: 'pick-metric' },
      h('dt', null, label, tip ? ui.infoTip(tip) : null),
      h('dd', { class: dir && dir !== 'na' ? dir : null }, value));
  }

  // ====================================================================================== leaders / laggards
  function leadItem(rec, data) {
    var p = num(rec.p_up), net = num(rec.net_exp_return), rs = num(rec.rank_signal), sg = num(rec.sigma);
    return h('li', { class: 'lead-item' },
      h('div', { class: 'lead-main' },
        h('span', { class: 'rank-badge', title: '依相對強度排名（全體第 ' + (rec.rank === undefined ? NA : rec.rank) + ' 名）' }, h('span', { class: 'sr-only' }, '排名 '), rec.rank === undefined ? NA : rec.rank),
        nameLink(rec), leanChip(rec.lean), lowVolTag(rec, data), costMark(rec)),
      h('div', { class: 'lead-nums' },
        h('span', { class: 'lead-num' }, h('small', null, '相對強度 '), rs === null ? NA : F.fmtSigned(rs, 3)),
        h('span', { class: 'lead-num' }, h('small', null, '上漲機率 '), p === null ? NA : F.fmtPct(p, 1)),
        h('span', { class: 'lead-num' }, h('small', null, '淨預期 '), net === null ? NA : h('span', { class: ['chg', ui.dirClass(Number((net * 100).toFixed(2)))] }, signedPct(net, 2))),
        h('span', { class: 'lead-num' }, h('small', null, '波動 σ '), sg === null ? NA : F.fmtPct(sg, 2))));
  }

  function leadNote(data) {
    var corr = num(data.rank_sigma_corr), med = num(data.universe_sigma_median);
    var s = '相對強度 = 預期對數報酬 ÷ 預期對數波動度（不含交易成本）。';
    if (corr !== null) {
      s += '今日與預測波動度的 Spearman 相關為 ' + F.fmtSigned(corr, 2) +
        (Math.abs(corr) >= 0.3 ? '：排序偏向' + (corr < 0 ? '低' : '高') + '波動的個股，這是相對位置，不是選股能力的證明。' : '。');
    }
    s += '標示「低波動」者的預測波動度低於本次掃描的中位數' + (med === null ? '' : '（' + F.fmtPct(med, 2) + '）') + '；標示「未覆蓋成本」者的預期報酬扣掉交易成本後不為正。';
    return s;
  }

  function leadersCard(data, hz) {
    var leaders = recs(data.leaders), laggards = recs(data.laggards);
    var allAbove = data.laggards_all_above_half === true;
    var lag = laggards.length, ld = leaders.length;
    function col(cls, title, sub, list, emptyText) {
      return h('div', { class: ['lead-col', cls] },
        h('h3', { class: 'lead-title' }, title),
        h('p', { class: 'lead-sub' }, sub),
        list.length ? h('ul', { class: 'lead-list' }, list.map(function (r) { return leadItem(r, data); })) : ui.emptyState(emptyText));
    }
    var body = [
      h('div', { class: 'lead-grid' },
        col('lead-leaders', '相對偏多 Top ' + ld, '相對強度由高到低 · 上漲機率皆高於 50%', leaders, '今日沒有上漲機率高於 50% 的標的。'),
        col('lead-laggards', allAbove ? '相對最弱（上漲機率仍高於 50%）' : '相對偏空 Top ' + lag,
          allAbove ? '相對強度最低的 ' + lag + ' 檔 · 整體漂移為正，這不代表看空'
            : '相對強度由低到高 · 底部 ' + lag + ' 檔（不要求上漲機率低於 50%）',
          laggards, '今日沒有可列出的標的。')),
      h('p', { class: 'lead-note' }, leadNote(data))
    ];
    return ui.card('今日相對強度排名', body, {
      sub: hz.long + ' · 依「相對強度（預期報酬/風險）」排序：這是模型對各檔的相對看法，不是買賣建議',
      tip: '相對強度 = 預期對數報酬 ÷ 預期對數波動度。排名不使用交易成本調整後的分數（那個分數會偏向高波動個股），也沒有經過歷史驗證時，請只把它當成相對位置。',
      class: 'lead-card'
    });
  }

  // ====================================================================================== strict candidate cards
  function pickCard(rec, hz, data, unlocked) {
    var p = num(rec.p_up), net = num(rec.net_exp_return), w = num(rec.weight), rs = num(rec.rank_signal), sg = num(rec.sigma);
    var dir = sigDir(rec.signal);
    return h('article', { class: ['pick-card', 'pick-' + (dir === 'na' ? 'flat' : dir)] },
      h('header', { class: 'pick-head' },
        h('span', { class: 'rank-badge', title: '依相對強度排名（全體第 ' + (rec.rank === undefined ? NA : rec.rank) + ' 名）' }, h('span', { class: 'sr-only' }, '排名 '), rec.rank === undefined ? NA : rec.rank),
        h('div', { class: 'pick-id' }, nameLink(rec)),
        sigChip(rec.signal, unlocked)),
      h('div', { class: 'pick-price' },
        h('span', { class: 'pick-last' }, F.fmtPrice(rec.last_close, rec.kind)),
        ui.change(rec.chg_1d, 2),
        h('span', { class: 'pick-price-label' }, '1日')),
      h('dl', { class: 'pick-metrics' },
        metric('上漲機率', p === null ? NA : F.fmtPct(p, 1), dir, '模型估計' + hz.long + '收盤高於今日收盤的機率（已校準）。這是模型輸出，不是經過驗證的勝率。'),
        metric('淨預期報酬', net === null ? NA : F.fmtChange(net, 2), ui.dirClass(net === null ? null : Number((net * 100).toFixed(2))), '預期報酬扣除一次買賣的手續費、證交稅與滑價後的估計。'),
        metric('模型配置權重', w === null || w <= 0 ? NA : F.fmtPct(w, 1), 'flat', '經收縮分數的 1/4 凱利比例，單一標的上限 15%，其餘為現金。這是模型的資金配置試算，不是投資建議。')),
      h('p', { class: 'pick-rs' }, '相對強度 ' + (rs === null ? NA : F.fmtSigned(rs, 3)) + ' · 預測波動 σ ' + (sg === null ? NA : F.fmtPct(sg, 2))),
      h('div', { class: 'pick-foot' }, confChip(rec.confidence), lowVolTag(rec, data), costMark(rec), rec.liquid === false ? ui.chip('流動性偏低', 'warn') : null),
      rec.reason ? h('p', { class: 'pick-reason' }, rec.reason) : null);
  }

  function driftNote(data, nTotal) {
    var n = num(data.n_scored);
    if (nTotal === null || n === null || n <= 0) return null;
    if (nTotal < Math.max(5, 0.25 * n)) return null;
    return ui.banner('warn', '符合條件的有 ' + F.fmtInt(nTotal) + ' / ' + F.fmtInt(n) + ' 檔（' + F.fmtPct(nTotal / n, 0) + '）：比例這麼高，通常是整體市場報酬為正（正漂移）造成的，而不是模型挑出了特別的個股；請不要把它們當成各自獨立的買進理由。');
  }

  function buysCard(data, hz, unlocked) {
    var buys = recs(data.buys);
    var total = num(data.n_buy_total);
    var shown = buys.length;
    var body;
    if (!buys.length) {
      body = ui.emptyState('目前沒有同時符合偏多候選條件的標的。', {
        action: h('p', { class: 'empty-sub' }, '偏多候選條件：校準後上漲機率 ≥ 55%、扣除交易成本後預期報酬 > 0、且近 20 日平均成交值夠高。沒有訊號本身就是模型的答案——多數時候市場並沒有可靠的可預測性，空手或觀望是合理選項。')
      });
    } else {
      body = [driftNote(data, total), h('div', { class: 'pick-grid' }, buys.map(function (r) { return pickCard(r, hz, data, unlocked); }))];
    }
    var count = !shown ? '沒有符合條件的標的' : (total !== null && total > shown ? '顯示 ' + shown + ' 檔（共 ' + F.fmtInt(total) + ' 檔符合）' : '共 ' + shown + ' 檔') +
      '，依相對強度由高到低；通過上漲機率與成本門檻，不等於買進建議；點擊名稱可查看個股預測';
    return ui.card('偏多候選（嚴格條件）', body, {
      sub: hz.long + ' · ' + count,
      tip: '嚴格條件：上漲機率 ≥ 55%、扣成本後預期報酬 > 0、流動性足夠。排名使用相對強度（預期報酬 ÷ 波動度）；成本調整後的分數只用來把關與計算配置權重。',
      class: 'buys-card'
    });
  }

  function portfolioCard(data) {
    var pf = obj(data.portfolio);
    var cash = num(pf.cash_weight);
    var n = num(pf.n);
    var inv = cash === null ? null : Math.max(0, 1 - cash);
    var allCash = cash !== null && cash >= 0.9995;
    var reason = typeof pf.suppressed_reason === 'string' && Object.prototype.hasOwnProperty.call(SUPPRESSED, pf.suppressed_reason) ? SUPPRESSED[pf.suppressed_reason] : null;
    var body = [
      allCash ? h('p', { class: 'pf-cash' }, '模型配置結果：100% 現金，不配置任何個股。') : null,
      allCash && (reason || pf.note) ? h('p', { class: 'pf-note' }, reason ? h('strong', null, '為什麼沒有權重：') : null, reason, reason && pf.note ? ' ' : null, pf.note || null) : null,
      h('div', { class: 'kpi-row' },
        ui.kpi('持股檔數', n === null ? NA : F.fmtInt(n) + ' 檔', null, { tip: '依模型配置權重實際配置的標的數。' }),
        ui.kpi('股票部位', inv === null ? NA : F.fmtPct(inv, 0), '合計配置權重'),
        ui.kpi('現金比例', cash === null ? NA : F.fmtPct(cash, 0), allCash ? '全數持有現金' : null, { tip: '權重不足 100% 的部分保留現金；沒有足夠把握時，現金就是預設部位。' })),
      cash === null || allCash ? null : ui.stackedBar([
        { key: 'stock', label: '股票', value: inv, cls: 'w-garch' }, { key: 'cash', label: '現金', value: cash, cls: 'w-naive' }
      ], { label: '模型資金配置' }),
      !allCash && pf.note ? h('p', { class: 'pf-note' }, pf.note) : null
    ];
    return ui.card('模型配置範例（研究用）', body, { sub: '權重 = 0.25 倍凱利、單檔上限 15%；假設各檔彼此獨立，未考慮相關性；只是試算，不是投資建議。', class: 'pf-card' });
  }

  function avoidsCard(data, hz, unlocked) {
    var av = recs(data.avoids);
    var body = av.length ? h('ul', { class: 'avoid-list' }, av.map(function (r) {
      var p = num(r.p_up), net = num(r.net_exp_return), sg = num(r.sigma);
      return h('li', { class: 'avoid-item' },
        h('div', { class: 'avoid-main' }, nameLink(r), sigChip(r.signal, unlocked), lowVolTag(r, data)),
        h('div', { class: 'avoid-nums' },
          h('span', { class: 'avoid-num' }, h('small', null, '上漲機率 '), p === null ? NA : F.fmtPct(p, 1)),
          h('span', { class: 'avoid-num' }, h('small', null, '1日 '), ui.change(r.chg_1d, 2)),
          h('span', { class: 'avoid-num' }, h('small', null, '淨預期 '), net === null ? NA : signedPct(net, 2)),
          h('span', { class: 'avoid-num' }, h('small', null, '波動 σ '), sg === null ? NA : F.fmtPct(sg, 2))),
        r.reason ? h('p', { class: 'avoid-reason' }, r.reason) : null);
    })) : ui.emptyState('目前沒有明確偏空、需要避開的標的。');
    return ui.card('偏空／避開（嚴格條件）', body, { sub: hz.long + ' · 上漲機率 ≤ 45% 且扣成本後預期報酬為負；僅提醒勿追買，本工具不提供放空建議', class: 'avoid-card' });
  }

  // ====================================================================================== full table
  /** The shrunk cost-adjusted score is identical for every name when the shrinkage factor B is 0 (all scores collapse to the grand mean):
   *  a column of 60 equal numbers says nothing, so it is only shown when the values differ (audit P-09 / HANDOFF 4A.2). */
  function scoreVaries(all) {
    var first = null, seen = 0;
    for (var i = 0; i < all.length; i++) {
      var v = num(all[i].score_shrunk);
      if (v === null) continue;
      if (seen === 0) first = v; else if (Math.abs(v - first) > 1e-9) return true;
      seen++;
    }
    return false;
  }

  function allCard(data, hz, unlocked) {
    var all = recs(data.all);
    var filter = 'all';
    var cols = [
      { key: 'rank', label: '排名', num: true, sortable: true, defaultDir: 'asc', tip: '依相對強度（預期報酬 ÷ 波動度）由高到低排序；不是成本調整後的分數。' },
      { key: 'name', label: '名稱', sortable: true, rowHeader: true, value: function (r) { return r.code; }, render: function (r) {
        return h('span', { class: 'name-cell' }, nameLink(r), r.liquid === false ? ui.chip('流動性偏低', 'warn') : null);
      } },
      { key: 'last_close', label: '收盤', num: true, sortable: true, render: function (r) { return F.fmtPrice(r.last_close, r.kind); } },
      { key: 'chg_1d', label: '1日', num: true, sortable: true, render: function (r) { return ui.change(r.chg_1d, 2); } },
      { key: 'p_up', label: '上漲機率', num: true, sortable: true, render: function (r) { return num(r.p_up) === null ? NA : F.fmtPct(r.p_up, 1); },
        tip: '校準後的上漲機率。50% 附近代表模型沒有明確看法；這是模型輸出，不是經過驗證的勝率。' },
      { key: 'net_exp_return', label: '淨預期報酬', num: true, sortable: true, render: function (r) {
        var v = num(r.net_exp_return);
        return v === null ? NA : h('span', { class: ['chg', ui.dirClass(Number((v * 100).toFixed(2)))] }, signedPct(v, 2));
      }, tip: '預期報酬扣除一次買賣的交易成本。' },
      { key: 'rank_signal', label: '相對強度', num: true, sortable: true, render: function (r) { return num(r.rank_signal) === null ? NA : F.fmtSigned(r.rank_signal, 3); },
        tip: '預期對數報酬 ÷ 預期對數波動度（相對強度，排名依據）。它偏向低波動個股，不是選股能力的證明。' },
      { key: 'sigma', label: '波動 σ', num: true, sortable: true, render: function (r) { return num(r.sigma) === null ? NA : F.fmtPct(r.sigma, 2); },
        tip: '模型預測這段期間報酬的標準差。' },
      scoreVaries(all) ? { key: 'score_shrunk', label: '成本後分數', num: true, sortable: true, render: function (r) { return num(r.score_shrunk) === null ? NA : F.fmtNum(r.score_shrunk, 2); },
        tip: '（預期報酬 − 交易成本）÷ 波動度，經經驗貝氏收縮後的分數。只用來把關偏多候選／偏空與計算配置權重，不用於排名（它偏向高波動個股）。' } : null,
      { key: 'weight', label: '模型配置權重', num: true, sortable: true, render: function (r) { var w = num(r.weight); return w === null || w <= 0 ? NA : F.fmtPct(w, 1); } },
      { key: 'confidence', label: '模型一致度', sortable: true, value: function (r) { return { '高': 0, '中': 1, '低': 2 }[r.confidence]; }, render: function (r) { return r.confidence || NA; } },
      { key: 'signal', label: '訊號', sortable: true, value: function (r) { return SIG_ORDER[r.signal]; }, render: function (r) { return sigChip(r.signal, unlocked) || NA; } }
    ].filter(Boolean);
    function filtered() {
      return filter === 'all' ? all : all.filter(function (r) { return r.signal === filter; });
    }
    var tbl = ui.table(cols, filtered(), {
      caption: '全部評分名單（' + hz.long + '）', sortKey: 'rank', sortDir: 'asc', empty: '沒有符合條件的標的', dense: true
    });
    var count = h('span', { class: 'muted filter-count' });
    function paintCount() { count.textContent = '顯示 ' + filtered().length + ' / ' + all.length + ' 檔'; }
    var seg = ui.segmented([
      { value: 'all', label: '全部' }, { value: 'buy', label: F.signalInfo('buy', unlocked).label }, { value: 'hold', label: F.signalInfo('hold').label },
      { value: 'avoid', label: F.signalInfo('avoid').label }
    ], 'all', function (v) { filter = v; tbl.setRows(filtered()); paintCount(); }, { label: '訊號篩選' });
    paintCount();
    return ui.card('全部評分名單', [h('div', { class: 'table-tools' }, seg, count), tbl], {
      sub: '共 ' + all.length + ' 檔；點擊欄位標題可排序。名單不等於推薦：只有符合條件的才會出現在「偏多候選」。', class: 'all-card'
    });
  }

  // ====================================================================================== market strip / banners
  function marketStrip(data) {
    var m = obj(data.market);
    var px = num(m.last_close);
    return h('div', { class: 'kpi-row market-strip', role: 'group', 'aria-label': '大盤概況' },
      ui.kpi('加權指數', px === null ? NA : F.fmtNum(px, 0), ui.change(m.chg_1d, 2), { tip: 'TAIEX（^TWII）收盤與一日漲跌。' }),
      ui.kpi('大盤明日上漲機率', num(m.p_up_1) === null ? NA : F.fmtPct(m.p_up_1, 1), null, { tip: '同一模型對加權指數的預測，約 50% 代表沒有明確方向。' }),
      ui.kpi('大盤未來5日上漲機率', num(m.p_up_5) === null ? NA : F.fmtPct(m.p_up_5, 1)),
      ui.kpi('大盤狀態', m.regime || NA, null, { tip: '隱馬可夫模型判斷的市場狀態。' }));
  }

  function bannersFor(data, st) {
    var out = [];
    var asof = data.asof;
    if (data.asof_complete === false) {
      out.push(ui.banner('warn', '盤中資料：最後一根 K 線尚未收盤，下列名單僅供參考。'));
    } else {
      out.push(ui.banner('info', [
        '資料截至 ', h('strong', null, asof ? F.fmtDateW(asof) : NA), ' 收盤；下列預測以該日收盤為起點，非即時報價，也不含之後的盤中走勢。',
        data.n_scored !== undefined ? ' 本次掃描 ' + F.fmtInt(data.n_scored) + ' / ' + F.fmtInt(data.universe_size) + ' 檔流動性較佳的大型股' + (data.computed_at ? '（計算於 ' + F.fmtDateTime(data.computed_at) + '）' : '') + '。' : null
      ]));
    }
    if (st && st.market_open === true) out.push(ui.banner('warn', '目前為盤中：尚未收盤，本頁仍以上一個完整交易日的資料計算。'));
    else if (st && st.market_open === false) out.push(ui.banner('info', '目前市場已收盤（休市中）：名單適用於下一個交易日的參考。'));
    if (st && st.last_trading_date && asof && st.last_trading_date > asof && !st.market_open) {
      out.push(ui.banner('warn', '資料尚未更新到最新交易日 ' + F.fmtDate(st.last_trading_date) + '，名單可能落後一個交易日。'));
    }
    if (data.stale === true) out.push(ui.banner('warn', '網路暫時無法更新，部分行情來自本機快取，資料可能落後。'));
    if (arr(data.skipped).length) out.push(ui.banner('info', '有 ' + arr(data.skipped).length + ' 檔因資料不足或取得失敗而略過。'));
    return out;
  }

  function notesCard(data) {
    var serverNotes = arr(data.notes).filter(function (n) { return typeof n === 'string' && n; });
    return ui.card('如何解讀這份名單', [
      h('ul', { class: 'note-list' },
        h('li', null, h('strong', null, '相對強度：'), '排名依「預期對數報酬 ÷ 預期對數波動度」，只代表模型對各檔的相對看法。預測均值被強烈收縮，所以這個排序偏向低波動個股；它不是選股能力的證明，請同時看每檔的波動 σ 與「低波動」標示。'),
        h('li', null, h('strong', null, '偏多候選與偏空／避開：'), '只是通過上漲機率與成本門檻的名單。短天期幾乎都沒有候選；一個月期間常有一大批，那通常只是整體市場正漂移，不是挑股訊號。'),
        h('li', null, h('strong', null, '收縮（shrinkage）：'), '各檔分數已向整體平均收縮（經驗貝氏 / James-Stein），只用來把關與計算配置權重。若各檔預測的差異不比雜訊大，不配置權重，建議組合就是 100% 現金。'),
        h('li', null, h('strong', null, '贏家詛咒：'), '從 60 檔裡挑「最高」的，其分數有一部分只是運氣；實際表現平均會比模型分數差。'),
        h('li', null, h('strong', null, '存活者偏差：'), '股票池是今日流動性最佳的股票，已下市或衰退的公司不在其中，歷史驗證因此會偏樂觀。'),
        h('li', null, h('strong', null, '機率不是保證：'), '55% 的上漲機率代表「十次裡約五到六次上漲」，單次仍可能虧損；請自行評估風險並自負盈虧。')),
      serverNotes.length ? h('details', { class: 'disclosure' }, h('summary', null, '計算方式與假設（' + serverNotes.length + ' 則）'),
        h('ul', { class: 'note-list' }, serverNotes.map(function (n) { return h('li', null, n); }))) : null,
      h('p', { class: 'disclaimer' }, h('span', { class: 'disclaimer-icon', 'aria-hidden': 'true' }, '⚠'), h('span', null, TQ.DISCLAIMER))
    ], { class: 'notes-card' });
  }

  // ====================================================================================== universe validation
  function ratioKpis(rep) {
    var ic = obj(rep.ic), pk = obj(rep.picks), sp = obj(rep.spread), per = obj(rep.period);
    return h('div', { class: 'kpi-row uv-kpis' },
      ui.kpi('排序能力 IC（平均）', num(ic.mean) === null ? NA : F.fmtSigned(ic.mean, 3),
        'IR ' + F.fmtNum(ic.ir, 2) + ' · t ' + F.fmtNum(ic.t, 2) + ' · p ' + F.fmtP(ic.pvalue),
        { tip: '每天計算「相對強度」與「實際未來報酬」的 Spearman 排名相關。0 代表排序沒有資訊；實務上 0.02–0.05 已算不錯，但需要統計顯著才可信。IR = 平均 ÷ 標準差；t 值與 p 值已考慮重疊期間（單尾，只檢定「是否為正」）。' }),
      ui.kpi('精選股上漲比例', num(pk.hit_rate) === null ? NA : F.fmtPct(pk.hit_rate, 1),
        '同日全體 ' + F.fmtPct(pk.base_rate, 1) + ' · p ' + F.fmtP(pk.p_value) + ' · n ' + F.fmtInt(pk.n),
        { tip: '被選中的股票在持有期間上漲的比例，對比同一天所有股票的上漲比例。這一項沒有控制波動度：低波動股票上漲的機率本來就較高，所以精選股偏向低波動時它會偏向顯著，不算獨立的選股證據。' }),
      ui.kpi('前段 − 後段報酬差', num(sp.mean) === null ? NA : signedPct(sp.mean, 2), 't ' + F.fmtNum(sp.t, 2),
        { tip: '每天相對強度最高的一群與最低的一群，平均「對數報酬」的差距（採對數報酬，避免波動度造成的 Jensen 項讓高波動排序白賺報酬差）。' }),
      ui.kpi('驗證期間', per.start ? F.fmtDate(per.start) + ' ~ ' + F.fmtDate(per.end) : NA,
        F.fmtInt(per.n_days) + ' 個交易日 · ' + F.fmtInt(per.n_stocks) + ' 檔'));
  }

  /** Does the picked basket merely hold calmer / lower-beta names?  (`basket` of DESIGN 6.5.) */
  function basketBlock(rep) {
    var b = obj(rep.basket);
    if (!Object.keys(b).length) return null;
    var sb = obj(b.sigma_basket);
    var kpis = h('div', { class: 'kpi-row uv-kpis' },
      ui.kpi('精選股平均波動 σ', num(b.mean_sigma) === null ? NA : F.fmtPct(b.mean_sigma, 2),
        '全體 ' + (num(b.universe_mean_sigma) === null ? NA : F.fmtPct(b.universe_mean_sigma, 2)) + '（' + (num(b.sigma_ratio) === null ? NA : F.fmtNum(b.sigma_ratio, 2)) + ' 倍）',
        { tip: '被選中時，模型預測的波動度的平均。遠低於全體，代表精選股其實是「比較不波動的股票」。' }),
      ui.kpi('精選股平均市場 β', num(b.mean_beta) === null ? NA : F.fmtNum(b.mean_beta, 2),
        '全體 ' + (num(b.universe_mean_beta) === null ? NA : F.fmtNum(b.universe_mean_beta, 2)) + (b.beta_market === 'equal_weight' ? '（相對全體等權重）' : '（相對加權指數）'),
        { tip: '滾動 250 日、只用當時已知資料估計的市場 β。β 低代表漲跌較不跟大盤，牛市中少賺、熊市中少賠。' }),
      ui.kpi('名單日對日重疊率', num(b.topk_overlap) === null ? NA : F.fmtPct(b.topk_overlap, 0), '相對強度與 σ 的相關 ' + (num(b.corr_rank_sigma) === null ? NA : F.fmtSigned(b.corr_rank_sigma, 2)),
        { tip: '前一天與今天的前 5 名有多少相同。重疊率高代表名單很少換，結果更像「固定持有幾檔低波動股票」。' }),
      ui.kpi('淨預期報酬為正的比例', num(b.share_net_positive) === null ? NA : F.fmtPct(b.share_net_positive, 0), '覆蓋交易成本的股票日',
        { tip: '全部「股票 × 日期」中，預期報酬扣掉交易成本後仍為正的比例。' }));
    var notes = [];
    if (num(b.sigma_ratio) !== null && b.sigma_ratio <= 0.8) {
      notes.push('精選股的波動度只有全體的 ' + F.fmtNum(b.sigma_ratio, 2) + ' 倍：排序偏向低波動股票，命中率與報酬的差異不一定是選股能力。');
    }
    if (num(sb.mean_sigma) !== null) notes.push('對照：每天只挑「預測波動度最高」的前 5 名，平均 σ ' + F.fmtPct(sb.mean_sigma, 2) + (num(sb.mean_beta) !== null ? '、β ' + F.fmtNum(sb.mean_beta, 2) : '') + '。');
    return h('div', { class: 'uv-block' },
      h('h3', { class: 'sub-title' }, '曝險對照：精選股是不是只是低波動、低 β 的股票？'),
      kpis,
      notes.length ? h('p', { class: 'bt-note' }, notes.join('')) : null);
  }

  function equityCard(rep, sc) {
    var curve = obj(rep.curve);
    var dates = arr(curve.dates);
    var topK = rep.top_k || 5;
    var labels = obj(rep.labels);
    var topName = labels.top || ('相對強度前 ' + topK + ' 名');
    var card = TQ.chartCard('累積淨值曲線', function (pal) {
      var series = [
        { key: 'top', label: topName, shortLabel: '相對強度前' + topK, color: pal.accent, values: arr(curve.top), width: 2.2, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); } },
        { key: 'ew', label: labels.benchmark_equal_weight || '全體等權重', shortLabel: '等權重', color: pal.gray, values: arr(curve.equal_weight), width: 1.6, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); } }
      ];
      if (arr(curve.top_sigma).length) series.push({ key: 'ts', label: labels.benchmark_top_sigma || '無預測對照：預測波動度最高的前 ' + topK + ' 名', shortLabel: '波動最高', color: pal.violet, values: arr(curve.top_sigma), width: 1.4, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); } });
      if (arr(curve.taiex).length) series.push({ key: 'tx', label: '加權指數', shortLabel: '大盤', color: pal.amber, values: arr(curve.taiex), width: 1.6, endLabel: true, endFmt: function (v) { return F.fmtNum(v, 2); } });
      return {
        dates: dates, n: dates.length, emptyText: '沒有淨值曲線資料',
        a11y: '累積淨值曲線圖：相對強度前 ' + topK + ' 名、等權重全體' + (arr(curve.top_sigma).length ? '、無預測的波動度對照' : '') + (arr(curve.taiex).length ? '與加權指數' : '') + '，起點皆為 1。',
        panels: [{ title: '累積淨值（起點 = 1；模型精選組合已扣成本）', series: series, fmt: function (v) { return F.fmtNum(v, 2); }, refLines: [{ v: 1, label: '起點' }], ticks: 5 }]
      };
    }, { kind: 'timeSeries', sub: '滑鼠移到圖上可查看各日數值；縱軸為倍數（2 = 資產翻倍）' + equityReminder(rep), heightFn: function (w) { return w < 480 ? 260 : 340; } });
    sc.add(card);
    return card;
  }

  /** The exposure reminder printed right under the equity curve (audit P-05): curve differences are mostly volatility / beta differences. */
  function equityReminder(rep) {
    var b = obj(rep.basket), r = num(b.sigma_ratio), beta = num(b.mean_beta), ub = num(b.universe_mean_beta);
    if (r === null && beta === null) return '';
    return '。提醒：精選股的波動度' + (r === null ? '' : '為全體的 ' + F.fmtNum(r, 2) + ' 倍') + (beta === null || ub === null ? '' : '、市場 β ' + F.fmtNum(beta, 2) + '（全體 ' + F.fmtNum(ub, 2) + '）') +
      '，曲線的差異可能只是曝險不同，不一定是選股能力';
  }

  function icCard(rep, sc) {
    var ser = obj(obj(rep.ic).series);
    var dates = arr(ser.dates), vals = arr(ser.values);
    if (!dates.length || !vals.length) return null;
    var card = TQ.chartCard('每日橫斷面 IC（滾動平均）', function (pal) {
      return {
        dates: dates, n: dates.length, a11y: '橫斷面 IC 的時間序列圖',
        panels: [{ title: 'IC（高於 0 代表相對強度越高、之後報酬越高）', includeZero: true, refLines: [{ v: 0, label: '0' }],
          series: [{ key: 'ic', label: 'IC', color: pal.violet, values: vals, width: 1.8 }], fmt: function (v) { return F.fmtNum(v, 2); } }]
      };
    }, { kind: 'timeSeries', sub: '長期在 0 附近上下擺動，代表排序能力不穩定或不存在', height: 220 });
    sc.add(card);
    return card;
  }

  function perStockTable(rep) {
    var rows = recs(rep.per_stock);
    if (!rows.length) return null;
    var t = ui.table([
      { key: 'code', label: '代號', sortable: true, rowHeader: true, render: function (r) { return h('a', { class: 'stock-link', href: TQ.codeHref(r.code) }, h('span', { class: 'sl-code' }, r.code)); } },
      { key: 'name', label: '名稱', sortable: true },
      { key: 'accuracy', label: '方向準確率', num: true, sortable: true, render: function (r) { return F.fmtPct(r.accuracy, 1); } },
      { key: 'brier_skill', label: 'Brier 技能', num: true, sortable: true, render: function (r) { return num(r.brier_skill) === null ? NA : F.fmtSigned(r.brier_skill, 4); },
        tip: '相對「因果氣候基準」（每個預測日只用當時已知的上漲比例）：大於 0 代表機率預測優於永遠猜歷史上漲比例。' },
      { key: 'brier_skill_oracle', label: 'Brier 技能（事後基準）', num: true, sortable: true, render: function (r) { return num(r.brier_skill_oracle) === null ? NA : F.fmtSigned(r.brier_skill_oracle, 4); },
        tip: '相對含事後資訊的基準（整段驗證期的上漲比例）：最嚴格的標準，但當時並不可知。' },
      { key: 'ic', label: 'IC', num: true, sortable: true, render: function (r) { return num(r.ic) === null ? NA : F.fmtSigned(r.ic, 3); } },
      { key: 'n', label: '樣本數', num: true, sortable: true, render: function (r) { return F.fmtInt(r.n); } }
    ], rows, { caption: '各檔個股的樣本外表現', sortKey: 'code', dense: true });
    return h('details', { class: 'disclosure' }, h('summary', null, '各檔個股的準確率（' + rows.length + ' 檔）'), t);
  }

  function statCols(rep) {
    var labels = obj(rep.labels), k = rep.top_k || 5;
    return [
      { label: labels.top || ('相對強度前 ' + k + ' 名（模型精選）'), stats: rep.top },
      { label: labels.benchmark_equal_weight || '等權重全體', stats: rep.benchmark_equal_weight },
      rep.benchmark_top_sigma ? { label: labels.benchmark_top_sigma || ('無預測對照：預測波動度最高的前 ' + k + ' 名'), stats: rep.benchmark_top_sigma } : null,
      rep.benchmark_taiex ? { label: labels.benchmark_taiex || '加權指數', stats: rep.benchmark_taiex } : null
    ].filter(Boolean);
  }

  function renderUniverse(rep, sc) {
    rep = obj(rep);
    var notes = arr(rep.notes).filter(function (n) { return typeof n === 'string' && n; });
    return h('div', { class: 'uv-result' },
      ui.verdictBox(rep.verdict),
      ratioKpis(rep),
      basketBlock(rep),
      h('div', { class: 'uv-block' },
        h('h3', { class: 'sub-title' }, '績效比較（模型精選已扣交易成本；基準不扣成本）'),
        ui.statsTable(statCols(rep), { caption: statCols(rep).map(function (c) { return c.label; }).join('、') + '的績效統計' })),
      equityCard(rep, sc),
      icCard(rep, sc),
      perStockTable(rep),
      notes.length ? h('details', { class: 'disclosure' }, h('summary', null, '假設與限制（' + notes.length + ' 則）'),
        h('ul', { class: 'note-list' }, notes.map(function (n) { return h('li', null, n); }))) : null);
  }

  /** The collapsible panel.  State (result per horizon) lives in `store`, owned by the view.  hooks.onResult(report, payload) is called
   *  when a run finishes (the view refreshes its evidence banner).  panel.open() expands it, panel.runNow() expands it and starts the
   *  run (or shows the stored result), panel.stop() aborts a poll in flight. */
  function universePanel(hv, store, ctx, sc, hooks) {
    var hz = hzDef(hv);
    var bodyId = TQ.uid('uv-body');
    var content = h('div', { class: 'uv-content' });
    var open = false;
    var running = null;     // AbortController of the poll in flight
    var psc = sc.add(TQ.scope());   // charts of the current result (disposed on re-run / view teardown)
    var inner = h('div', { class: 'uv-inner' }, content);
    var body = h('div', { class: 'card-body collapse-body', id: bodyId, hidden: true }, inner);
    var chev = h('span', { class: 'chev', 'aria-hidden': 'true' }, '▾');
    var btn = h('button', { type: 'button', class: 'collapse-btn', 'aria-expanded': 'false', 'aria-controls': bodyId, onClick: function () { setOpen(!open); } },
      h('span', { class: 'collapse-title' }, '推薦整體歷史驗證'),
      h('span', { class: 'collapse-sub' }, '把「相對強度前 5 名」的規則套到過去每一天，看看到底有沒有用'),
      chev);
    var panel = h('section', { class: 'card collapsible uv-card' }, h('h2', { class: 'card-title collapse-h' }, btn), body);

    function setOpen(v) {
      open = v;
      btn.setAttribute('aria-expanded', v ? 'true' : 'false');
      body.hidden = !v;
      panel.classList.toggle('open', v);
      if (v && !content.firstChild) showIdle();
      if (v && store[hv] && !content.querySelector('.uv-result')) showResult(store[hv]);
    }

    function showIdle() {
      TQ.clear(content);
      content.appendChild(h('p', { class: 'uv-intro' },
        '每天依本頁相同的規則（預測期間：' + hz.long + '）挑出「相對強度」最高的前 5 名等權重持有，扣除交易成本，與「全體等權重」、「只挑預測波動度最高者」的無預測對照及加權指數比較，並檢定排序能力（IC）與命中率是否顯著。'));
      content.appendChild(h('p', { class: 'uv-scope' }, '驗證的是「相對強度排序」本身：全體 60 檔都參與排名，只取前 5 名檢驗（本頁列出前 10 名）；不套用「偏多候選」的上漲機率與成本門檻，所以它不是在驗證本頁嚴格條件的偏多候選名單。'));
      content.appendChild(h('p', { class: 'muted' }, '首次計算需要對全部股票逐一做前推式預測，可能需要數分鐘；完成後會快取。'));
      content.appendChild(h('button', { type: 'button', class: 'btn btn-primary', onClick: run }, '開始驗證（' + hz.label + '）'));
    }

    function showResult(rep) {
      psc.dispose();
      TQ.clear(content);
      content.appendChild(renderUniverse(rep, psc));
      content.appendChild(h('p', { class: 'uv-rerun' }, h('button', { type: 'button', class: 'btn btn-ghost', onClick: function () { delete store[hv]; run(); } }, '重新計算')));
    }

    function run() {
      TQ.clear(content);
      var bar = ui.progressBar(0, 0, '準備中…');
      var AC = root.AbortController;
      var ctl = AC ? new AC() : null;
      running = ctl;
      function onParentAbort() { if (ctl) ctl.abort(); }
      ctx.signal.addEventListener('abort', onParentAbort);
      content.appendChild(h('div', { class: 'uv-progress', role: 'status' }, bar,
        h('button', { type: 'button', class: 'btn btn-ghost', onClick: function () { if (ctl) ctl.abort(); showIdle(); } }, '取消')));
      TQ.api.poll('/api/backtest/universe', { h: hv, top: 5 }, {
        signal: ctl ? ctl.signal : ctx.signal,
        onProgress: function (p) { bar.update(p.done, p.total, p.message || '計算中…'); }
      }).then(function (js) {
        ctx.signal.removeEventListener('abort', onParentAbort);
        if (ctx.signal.aborted || (ctl && ctl.signal.aborted)) return;
        store[hv] = obj(js.report || js);
        showResult(store[hv]);
        if (hooks && typeof hooks.onResult === 'function') hooks.onResult(store[hv], js);
      }, function (err) {
        ctx.signal.removeEventListener('abort', onParentAbort);
        if (TQ.api.isAbort(err) || ctx.signal.aborted || (ctl && ctl.signal.aborted)) return;
        TQ.clear(content);
        content.appendChild(ui.errorBox(err, run));
      });
    }

    panel.stop = function () { if (running) { try { running.abort(); } catch (e) { /* ignore */ } } };
    panel.open = function () { setOpen(true); try { if (panel.scrollIntoView) panel.scrollIntoView({ block: 'start' }); } catch (e) { /* ignore */ } };
    panel.runNow = function () {
      setOpen(true);
      if (!store[hv] && !running) run();
      try { if (panel.scrollIntoView) panel.scrollIntoView({ block: 'start' }); } catch (e) { /* ignore */ }
    };
    return panel;
  }

  // ====================================================================================== view
  function title() { return '今日模型觀察'; }

  function mount(host, params, ctx) {
    var stored = Number(TQ.prefs.get('picksH', 5));
    var hv = hzDef(stored).value;
    var destroyed = false;
    var cache = {};          // horizon -> recommendations payload (kept while the view lives)
    var uvStore = {};        // horizon -> universe-validation report
    var uvEv = {};           // horizon -> evidence block built from a report the user ran in this view (newer than the payload's)
    var sc = TQ.scope();
    var ctl = null;
    var panelNode = null;
    var cur = null;          // { data, st, hv } of the page on screen

    var seg = ui.segmented(HORIZONS.map(function (x) { return { value: x.value, label: x.label, title: x.long }; }), hv, function (v) {
      hv = v;
      TQ.prefs.set('picksH', v);
      load(false);
    }, { label: '預測期間' });
    var head = h('div', { class: 'page-head' },
      h('div', { class: 'page-head-text' },
        h('h1', { class: 'page-title' }, '今日模型觀察'),
        h('p', { class: 'page-sub' }, '模型對流動性較佳大型股的相對看法，不是買賣建議。每天掃描一次，依「相對強度（預期報酬/風險）」排名，並列出通過嚴格條件的偏多候選與偏空／避開名單。')),
      h('div', { class: 'page-head-ctl' }, h('span', { class: 'ctl-label', 'aria-hidden': 'true' }, '預測期間'), seg));
    var slot = h('div', { class: 'slot slot-picks' });
    var view = h('div', { class: 'view view-picks' }, head, slot);
    host.appendChild(view);

    function alive() { return !destroyed && !ctx.signal.aborted; }

    function evidenceNow(data, hvv) { return uvEv[hvv] ? evidenceInfo(uvEv[hvv], hvv) : evidenceInfo(obj(data).evidence, hvv); }

    function render(data, st) {
      sc.dispose();
      if (panelNode) { panelNode.stop(); panelNode = null; }
      var hz = hzDef(hv);
      cur = { data: data, st: st, hv: hv, unlocked: false };
      TQ.clear(slot);
      slot.appendChild(ui.disclaimerStrip(PICKS_SUFFIX));
      var evHost = h('div', { class: 'ev-host' });
      var listsHost = h('div', { class: 'slot slot-lists' });
      cur.evHost = evHost;
      cur.listsHost = listsHost;
      slot.appendChild(evHost);
      bannersFor(data, st).forEach(function (b) { slot.appendChild(b); });
      slot.appendChild(marketStrip(data));
      slot.appendChild(listsHost);
      panelNode = universePanel(hv, uvStore, ctx, sc, {
        onResult: function (rep, js) {
          uvEv[hz.value] = evidenceFromReport(rep, hz.value, js && js.computed_at);
          if (alive() && cur && cur.hv === hz.value) {
            var was = cur.unlocked;
            paintEvidence();
            if (cur.unlocked !== was) paintLists();          // the buy wording follows the gate: repaint only when it flipped (keeps the table's sort / filter)
          }
        }
      });
      slot.appendChild(panelNode);
      slot.appendChild(notesCard(data));
      paintEvidence();
      paintLists();
      ctx.setStatus('今日模型觀察（' + hz.label + '）已更新，共 ' + arr(data.all).length + ' 檔，偏多候選 ' + arr(data.buys).length + ' 檔');
    }

    function paintEvidence() {
      if (!cur) return;
      var hz = hzDef(cur.hv);
      var info = evidenceNow(cur.data, cur.hv);
      cur.unlocked = info.source !== 'none' && info.unlocked === true;
      TQ.clear(cur.evHost);
      cur.evHost.appendChild(evidenceBanner(info, hz, {
        run: function () { if (panelNode) panelNode.runNow(); },
        open: function () { if (panelNode) panelNode.open(); }
      }));
    }

    function paintLists() {
      if (!cur) return;
      var data = cur.data, hz = hzDef(cur.hv), unlocked = cur.unlocked;
      TQ.clear(cur.listsHost);
      if (Array.isArray(data.leaders) || Array.isArray(data.laggards)) cur.listsHost.appendChild(leadersCard(data, hz));
      cur.listsHost.appendChild(buysCard(data, hz, unlocked));
      cur.listsHost.appendChild(h('div', { class: 'row row-2' }, portfolioCard(data), avoidsCard(data, hz, unlocked)));
      cur.listsHost.appendChild(allCard(data, hz, unlocked));
    }

    function load(force) {
      if (ctl) { try { ctl.abort(); } catch (e) { /* ignore */ } }
      var my = hv;
      if (!force && cache[my]) {
        TQ.api.status().then(function (st) { if (alive() && hv === my) render(cache[my], st); });
        return;
      }
      var AC = root.AbortController;
      var mine = AC ? new AC() : null;
      ctl = mine;
      function onParentAbort() { if (mine) mine.abort(); }
      ctx.signal.addEventListener('abort', onParentAbort);
      sc.dispose();
      if (panelNode) { panelNode.stop(); panelNode = null; }
      cur = null;
      TQ.clear(slot);
      var bar = ui.progressBar(0, 0, '連線中…');
      slot.appendChild(ui.card('正在計算今日模型觀察', [
        bar,
        h('p', { class: 'muted' }, '首次計算需要預測全部 60 檔，通常約一分鐘；同一天內再次開啟會直接使用快取。')
      ], { class: 'progress-card' }));
      TQ.api.poll('/api/recommendations', { h: my, top: 10 }, {
        signal: mine ? mine.signal : ctx.signal,
        onProgress: function (p) { if (hv === my) bar.update(p.done, p.total, p.message || '計算中…'); }
      }).then(function (data) {
        ctx.signal.removeEventListener('abort', onParentAbort);
        if (!alive() || hv !== my || (mine && mine.signal.aborted)) return;
        cache[my] = data;
        return TQ.api.status().then(function (st) { if (alive() && hv === my) render(data, st); });
      }, function (err) {
        ctx.signal.removeEventListener('abort', onParentAbort);
        if (!alive() || TQ.api.isAbort(err) || hv !== my) return;
        TQ.clear(slot);
        slot.appendChild(ui.card('今日模型觀察', ui.errorBox(err, function () { load(true); })));
      });
    }

    load(false);
    return {
      destroy: function () {
        destroyed = true;
        if (ctl) { try { ctl.abort(); } catch (e) { /* ignore */ } }
        if (panelNode) panelNode.stop();
        sc.dispose();
      }
    };
  }

  TQ.views.picks = {
    title: title, mount: mount,
    _internals: { evidenceInfo: evidenceInfo, evidenceFromReport: evidenceFromReport, isLowVol: isLowVol, leadNote: leadNote, driftNote: driftNote }
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
