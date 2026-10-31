/*
 * view_method.js -- 方法說明 (#/method[/section]): a long-form, phone-friendly explanation of every technique the app uses.
 *
 * Content is data (SECTIONS below); a tiny renderer turns it into DOM with TQ.h (no innerHTML, no libraries):
 *   inline text  "…$\sigma_t^2$…**bold**…"   -> $…$ = inline math, **…** = strong
 *   block types  p · eq · ul · ol · h3 · note (callout) · dl (glossary) · flow (pipeline) · timeline · features · costs
 * Equations are written in a small LaTeX subset (mathNodes below): letters italic, _{} ^{} scripts (stacked when both are present),
 * \frac{}{} \sqrt{} \hat \bar \text{} \mathbf{} and a few symbols; everything becomes <span>/<i>/<sub>/<sup> with CSS fractions.
 * The table of contents uses real hrefs (#/method/<id>, for copy-link / no-JS) but its click handler scrolls to the section and
 * refines the URL with ctx.replace -- it never triggers a hashchange, so the hash router never re-mounts the page.
 * Numbers that exist in backend/config.py / quant code are stated here as constants; tests/test_front_method_static.py compares them.
 */
(function (root) {
  'use strict';
  var TQ = root.TQ;
  var F = TQ.fmt, ui = TQ.ui, h = TQ.h;
  var NA = F.NA;
  var R = String.raw;

  // ====================================================================================== constants (mirror backend/*.py)
  var CONST = {
    commission: 0.001425, taxStock: 0.003, taxEtf: 0.001, slippageBps: 5, discount: 0.6, dailyLimit: 0.10, tradingDays: 245,
    minTrain: 500, refitEvery: 21, testBars: 1250, buyP: 0.55, avoidP: 0.45, kelly: 0.25, maxWeight: 0.15, minAvgValue: 1e8, universe: 60,
    garchWindow: 2500, hmmWindow: 1000, hmmStates: 3, dmaAlpha: 0.99, dmaWindow: 400, dmaEta: 0.25, calWindow: 300, calLambda: 0.995,
    calSigmaA: 0.12, calSigmaB: 0.35, dlmDelta: 0.998, dlmBetaV: 0.998, dlmTau: 0.025, naiveNu: 6, mcPaths: 5000, mcPool: 1500, shrinkBars: 250,
    skewDamping: 0.6, nFeatures: 14
  };

  /** 0.55 -> "55" (no floating-point artefacts such as 55.00000000000001). */
  function pc(x) { return String(Math.round(x * 1e6) / 1e4); }

  // ====================================================================================== math renderer (LaTeX subset)
  var SYMBOLS = {
    sum: '∑', prod: '∏', cdot: '·', times: '×', le: '≤', ge: '≥', ne: '≠', approx: '≈', sim: '∼', to: '→', infty: '∞', in: '∈',
    propto: '∝', pm: '±', mid: '∣', partial: '∂', top: '⊤', leftarrow: '←', rightarrow: '→', Rightarrow: '⇒', ldots: '…', dots: '…', cdots: '⋯',
    lbrace: '{', rbrace: '}', 'lbrace ': '{', '{': '{', '}': '}', '_': '_', '%': '%', '&': '&', '|': '‖'
  };
  var FUNCS = { ln: 'ln', log: 'log', exp: 'exp', max: 'max', min: 'min', Pr: 'Pr', Var: 'Var', Cov: 'Cov', sign: 'sign', logit: 'logit', sup: 'sup', arg: 'arg' };
  // non-collapsing spaces (U+2009 thin, U+2005 four-per-em, U+2003 em, U+00A0): ordinary spaces would be collapsed by HTML
  var SPACES = { ',': '\u2009', ';': '\u2005', ':': '\u2005', ' ': '\u00a0', quad: '\u2003', qquad: '\u2003\u2003', '!': '' };
  var GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι',
    kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ',
    omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω' };
  var ACCENTS = { hat: '̂', bar: '̄', tilde: '̃' };
  var LETTER = /[A-Za-zΑ-Ωα-ωℓ]/;

  /** Parse the LaTeX subset into an array of DOM nodes / strings.  Unknown commands are kept visibly (as "\name") so tests catch them. */
  function mathNodes(src) {
    src = String(src == null ? '' : src);
    var i = 0, n = src.length;

    function cmdName() {
      var j = i + 1, c = src.charAt(j);
      if (c && !/[A-Za-z]/.test(c)) return { name: c, end: j + 1 };
      var k = j;
      while (k < n && /[A-Za-z]/.test(src.charAt(k))) k++;
      return { name: src.slice(j, k), end: k };
    }
    function skipSpaces() { while (i < n && src.charAt(i) === ' ') i++; }

    function rawGroup() {            // {…} read verbatim (for \text{…}); nested braces are balanced
      skipSpaces();
      if (src.charAt(i) !== '{') { return src.charAt(i++); }
      var depth = 0, start = i + 1;
      for (; i < n; i++) {
        var c = src.charAt(i);
        if (c === '{') depth++;
        else if (c === '}') { depth--; if (depth === 0) { var out = src.slice(start, i); i++; return out; } }
      }
      return src.slice(start);
    }

    function arg() {                 // one argument: {…} | \command | single character
      skipSpaces();
      var c = src.charAt(i);
      if (c === '{') { i++; var nodes = seq('}'); if (src.charAt(i) === '}') i++; return nodes; }
      if (c === '\\') { return command(); }
      if (!c) return [];
      i++;
      return [atom(c)];
    }

    function atom(c) { return LETTER.test(c) ? h('i', null, c) : c; }

    function scripts(first) {        // called with i just after '_' or '^'
      var subN = null, supN = null;
      if (first === '_') subN = arg(); else supN = arg();
      if (first === '_' && src.charAt(i) === '^') { i++; supN = arg(); }
      else if (first === '^' && src.charAt(i) === '_') { i++; subN = arg(); }
      if (subN && supN) return h('span', { class: 'ss' }, h('span', { class: 'sp' }, supN), h('span', { class: 'sb' }, subN));
      return subN ? h('sub', null, subN) : h('sup', null, supN);
    }

    function command() {             // i at '\'; returns an array of nodes
      var c = cmdName();
      i = c.end;
      var name = c.name;
      if (name === 'frac' || name === 'tfrac' || name === 'dfrac') {
        var a = arg(), b = arg();
        return [h('span', { class: 'frac' }, h('span', { class: 'fn' }, a), h('span', { class: 'fd' }, b))];
      }
      if (name === 'sqrt') { var r = arg(); return [h('span', { class: 'sqrt' }, '√', h('span', { class: 'sq-in' }, r))]; }
      if (name === 'text' || name === 'mathrm') return [h('span', { class: 'tx' }, rawGroup())];
      if (name === 'mathbf') return [h('b', { class: 'mb' }, arg())];
      if (name === 'mathcal') return [h('span', { class: 'cal' }, arg())];
      if (name === 'operatorname') return [h('span', { class: 'fnm' }, rawGroup())];
      if (name === 'widehat') return [h('span', { class: 'wide-hat' }, arg())];
      if (Object.prototype.hasOwnProperty.call(GREEK, name)) return [name.charAt(0) === name.charAt(0).toLowerCase() ? h('i', null, GREEK[name]) : h('span', { class: 'sym' }, GREEK[name])];
      if (ACCENTS[name]) { var base = arg(); return [h('span', { class: 'acc' }, base, ACCENTS[name])]; }
      if (name === 'left' || name === 'right') { var d = src.charAt(i); i++; return d === '.' ? [] : [h('span', { class: 'dl' }, d)]; }
      if (name === 'big' || name === 'Big' || name === 'bigg') return [];
      if (Object.prototype.hasOwnProperty.call(SPACES, name)) return [SPACES[name]];
      if (Object.prototype.hasOwnProperty.call(FUNCS, name)) return [h('span', { class: 'fnm' }, FUNCS[name])];
      if (Object.prototype.hasOwnProperty.call(SYMBOLS, name)) return [h('span', { class: name === 'sum' || name === 'prod' ? 'bigop' : 'sym' }, SYMBOLS[name])];
      return [h('span', { class: 'bad' }, '\\' + name)];
    }

    function seq(stop) {
      var out = [], buf = '';
      function flush() { if (buf) { out.push(buf); buf = ''; } }
      while (i < n) {
        var c = src.charAt(i);
        if (stop && c === stop) break;
        if (c === '{') { flush(); i++; out = out.concat(seq('}')); if (src.charAt(i) === '}') i++; continue; }
        if (c === '_' || c === '^') { flush(); i++; out.push(scripts(c)); continue; }
        if (c === '\\') { flush(); out = out.concat(command()); continue; }
        if (LETTER.test(c)) { flush(); out.push(h('i', null, c)); i++; continue; }
        buf += c;
        i++;
      }
      flush();
      return out;
    }
    return seq(null);
  }

  /** Linear plain-text reading of parsed math nodes: fractions "(a)/(b)", scripts "x_t" / "x_(t-1)" / "x^2", roots "√(…)". */
  function nodeText(n) {
    if (typeof n === 'string') return n;
    if (!n) return '';
    if (n.nodeType === 3) return n.data;
    var kids = Array.prototype.map.call(n.childNodes || [], nodeText);
    var cls = (n.getAttribute && n.getAttribute('class')) || '';
    var tag = String(n.localName || '').toLowerCase();
    function wrap(x) { return x.length > 1 ? '(' + x + ')' : x; }
    if (cls === 'frac') return '(' + kids[0] + ')/(' + kids[1] + ')';
    if (cls === 'sqrt') return '√(' + kids.slice(1).join('') + ')';
    if (cls === 'ss') return '_' + wrap(kids[1]) + '^' + wrap(kids[0]);
    if (tag === 'sub') return '_' + wrap(kids.join(''));
    if (tag === 'sup') return '^' + wrap(kids.join(''));
    return kids.join('');
  }

  /** Plain-text reading of a LaTeX-subset string (aria-label of equations). */
  function linearMath(src) {
    return mathNodes(src).map(nodeText).join('').replace(/[\u2003\u2005\u2009\u00a0]+/g, ' ').replace(/\s+/g, ' ').trim();
  }

  // ====================================================================================== inline markup:  $math$  **bold**
  function inline(str) {
    var out = [], s = String(str == null ? '' : str);
    var re = /\$([^$]+)\$|\*\*([^*]+)\*\*/g, last = 0, m;
    while ((m = re.exec(s)) !== null) {
      if (m.index > last) out.push(s.slice(last, m.index));
      if (m[1] !== undefined) out.push(h('span', { class: 'm' }, mathNodes(m[1])));
      else out.push(h('strong', null, inline(m[2])));
      last = re.lastIndex;
    }
    if (last < s.length) out.push(s.slice(last));
    return out;
  }

  // ====================================================================================== content
  /** Table rows for the cost section, computed from the same constants as backend/config.roundtrip_cost. */
  function costRows() {
    return [0.28, 0.5, 0.6, 0.85, 1].map(function (d) {
      var comm = CONST.commission * d, slip = CONST.slippageBps / 1e4;
      var buy = comm + slip, sellStock = comm + slip + CONST.taxStock, sellEtf = comm + slip + CONST.taxEtf;
      return { disc: d, comm: comm, buy: buy, sellStock: sellStock, rtStock: buy + sellStock, rtEtf: buy + sellEtf };
    });
  }
  function discText(d) { return d >= 0.9995 ? '不打折' : String(Math.round(d * 1000) / 100) + ' 折'; }

  var FEATURES = [
    ['rev_1', '1 日反轉', '最近 1 日報酬 ÷ 當日波動', '漲多回檔、跌深反彈的短期反轉傾向'],
    ['rev_5', '5 日反轉', '最近 5 日報酬 ÷ (σ√5)', '一週尺度的反轉'],
    ['mom_20', '1 個月動能', '最近 20 日報酬 ÷ (σ√20)', '近期趨勢是否延續'],
    ['mom_60', '3 個月動能', '最近 60 日報酬 ÷ (σ√60)', '中期趨勢'],
    ['mom_12_1', '12-1 個月動能', '第 −252 日到第 −21 日的報酬 ÷ (σ√231)', '學術上最常被研究的動能因子（刻意略過最近一個月，避開短期反轉）'],
    ['ma_gap', '均線乖離（20 日）', 'ln(收盤 ÷ 20 日均價) ÷ 波動', '離短期均線多遠；過遠常被視為過熱或過冷'],
    ['ma_cross', '均線多空（20/60 日）', 'ln(20 日均價 ÷ 60 日均價) ÷ 波動', '短均線在長均線之上或之下（趨勢方向）'],
    ['rsi', 'RSI（14）', 'Wilder 相對強弱指標', '超買超賣的震盪指標'],
    ['rvol_ratio', '區間波動比', '0.5·ln(短期 ÷ 長期的高低價區間變異)', '近期波動相對長期是放大還是縮小'],
    ['vol_z', '成交量異常', 'ln 成交量相對其指數移動平均的 z 分數', '量能是否突然放大或萎縮'],
    ['mkt_ret', '大盤當日報酬', '加權指數 1 日報酬 ÷ 大盤波動', '大盤剛剛發生的事'],
    ['mkt_mom20', '大盤 1 個月動能', '加權指數 20 日報酬 ÷ 大盤波動', '大盤的中期趨勢'],
    ['mkt_vol', '大盤波動狀態', '0.5·ln(短期 ÷ 長期的大盤變異)', '市場整體是平靜還是緊張'],
    ['idio_mom', '特異動能（60 日）', '市場模型殘差的 60 日累積 ÷ (殘差波動·√60)', '扣除大盤影響後，這檔股票自己的動能']
  ];

  var SECTIONS = [
    // ------------------------------------------------------------------------------------------------ 0
    { id: 'intro', title: '先讀這個：這個工具到底在做什麼', blocks: [
      { t: 'p', s: '這個工具把每檔股票或 ETF 的歷史價格丟進幾個統計模型，估計「未來 1 日、5 日、20 日報酬的**機率分布**」，再用過去的資料誠實檢驗這些估計有沒有用。本頁說明每一步怎麼算、用了哪些假設，以及為什麼你不該盡信它。' },
      { t: 'note', kind: 'warn', title: '最重要的一句話', s: '股票報酬非常接近不可預測。這裡的每個預測都只是機率；任何「有預測能力」的說法，都必須附上**樣本外**證據、與簡單基準的比較、以及 p 值；結論允許（也經常會）是「沒有統計上顯著的預測優勢」。' },
      { t: 'h3', s: '四個設計原則' },
      { t: 'ol', items: [
        '**預測分布，而不是一個價格。** 輸出是上漲機率、預期報酬、50% 與 90% 區間，不是「明天會漲到多少」。',
        '**沒有偷看未來。** 第 $i$ 天的預測只使用第 $0,\\dots,i$ 天的資料，包含參數估計、特徵標準化、模型權重與機率校準；這一點有自動化測試守住（見「前推式驗證」一節）。',
        '**成本與偏誤要算進去、要講出來。** 回測計入手續費、證交稅與滑價，預設隔天開盤成交；股票池的存活者偏誤、挑選偏誤與多重檢定都會明講。',
        '**這不是投資建議。** 它是統計研究與教育用途的工具；過去的表現不代表未來，投資一定有風險。'
      ] },
      { t: 'p', s: '閱讀建議：先看下面的「整體流程」，再依興趣讀個別模型；看不懂公式也沒關係，每條公式後面都有白話說明。想直接知道「有沒有用」，請跳到「前推式驗證」與「限制與誠實說明」。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 1
    { id: 'pipeline', title: '整體流程', blocks: [
      { t: 'p', s: '從原始價格到你在畫面上看到的數字，依序經過下列八個步驟。前六步每天（或每次重新整理）都會對每檔股票執行一次；最後兩步是「檢驗這一切到底準不準」。' },
      { t: 'flow', steps: [
        ['資料', '日 K 線、除權息、大盤指數；還原權息價、清理異常值'],
        ['特徵', '14 個只用「當天以前」資料算出的因子（動能、反轉、量能、大盤…）'],
        ['四個模型', 'GJR-GARCH-t、隱馬可夫、動態線性模型、隨機漫步，各自給出一個預測分布'],
        ['動態模型平均', '依各模型最近的預測表現加權，合成一個分布'],
        ['機率校準', '用已經發生的結果修正「過度自信」或系統性偏差'],
        ['漲跌停截斷', '明日報酬限制在 ±10% 內'],
        ['輸出', '上漲機率、預期報酬、區間、扇形圖、VaR / CVaR、驅動因子'],
        ['檢驗與模型觀察', '前推式回測、與基準比較、每日模型觀察與整體驗證']
      ] },
      { t: 'p', s: '四個模型數學上各不相同：GARCH 只看「波動」；HMM 看「市場處於哪種狀態」；動態線性模型看「因子與未來報酬的關係」；隨機漫步則是「什麼都不知道」的基準。把它們混合，並讓權重隨近期表現調整，是為了避免把全部賭注押在單一模型上。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 2
    { id: 'data', title: '資料與還原權息', blocks: [
      { t: 'p', s: '價格來自 Yahoo Finance 的日 K 線（開高低收、成交量、股利），股票名稱與上市／上櫃別來自證交所與櫃買中心的公開資料。這些資料可能延遲或出錯，工具只能「盡量清理」。' },
      { t: 'h3', s: '為什麼要還原權息' },
      { t: 'p', s: '除息日當天，股價會「自然」減少約等於股利的金額，但持有人並沒有虧損，股利進了帳戶。若直接用原始收盤價，除息會被誤認成下跌。因此建模一律使用**還原權息收盤價**（Yahoo 的 adjusted close）：每逢除息日 $d$，把 $d$ 之前的所有價格乘上一個小於 1 的係數。' },
      { t: 'eq', s: R`\mathrm{adjC}_t = C_t \cdot \prod_{d>t}\left(1 - \frac{D_d}{C_{d-1}}\right)`, note: '$C_t$ 是當日實際收盤價，$D_d$ 是除息日 $d$ 的每股現金股利，$C_{d-1}$ 是除息前一日的收盤價。股票分割以同樣的方式調整。' },
      { t: 'p', s: '之後所有的報酬都用還原價計算，也就是「股利再投入、不計稅」的**總報酬**近似：' },
      { t: 'eq', s: R`r_t = \ln\frac{\mathrm{adjC}_t}{\mathrm{adjC}_{t-1}}`, note: '圖表上顯示的 K 線與價格仍是實際成交價（未還原）。' },
      { t: 'h3', s: '資料清理' },
      { t: 'ul', items: [
        '台股單日漲跌幅限制為 ±10%（2015 年以前為 ±7%），所以單日 $|r_t|>25\\%$ 一定是資料錯誤（例如未還原的分割），模型使用的報酬把它設為 0；介於 12% 與 25% 之間的值截為 ±12%。',
        '沒有報價的缺漏列會被丟棄，大盤指數的缺值列同樣略過。',
        '盤中（台北時間 14:00 之前）最後一根 K 棒尚未完成，會被排除；所有預測都以「上一個完整交易日的收盤」為起點。',
        '加權指數（^TWII）作為市場因子，以「日期不晚於該股當日的最後一筆大盤資料」對齊，只會取用過去的值。'
      ] },
      { t: 'note', kind: 'info', title: '還原價不等於你能買到的價格', s: '還原價假設股利立刻再投入且不課稅、可以零成本買到任意數量。真實的成交價是未還原價，股利還要扣稅。回測裡的報酬因此略為樂觀。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 3
    { id: 'targets', title: '預測目標：累積對數報酬與方向', blocks: [
      { t: 'p', s: '對每個預測起點 $i$（第 $i$ 天收盤後）與期間 $h\\in\\{1,5,20\\}$（明日、一週、一個月），預測的對象是**累積對數報酬**：' },
      { t: 'eq', s: R`R_{i,h} = \ln\frac{\mathrm{adjC}_{i+h}}{\mathrm{adjC}_i} = \sum_{k=1}^{h} r_{i+k}`, note: '對數報酬可以逐日相加，這讓多日報酬的變異數與分布容易推導。畫面上顯示的報酬是簡單報酬 $e^{R}-1$。' },
      { t: 'p', s: '所有模型都輸出 $R_{i,h}$ 的**預測分布**（累積分布函數 $F$），因此其他數字都能由它導出：' },
      { t: 'eq', s: R`p_{\mathrm{up}} = \Pr(R_{i,h} > 0) = 1 - F(0), \qquad q_\tau = F^{-1}(\tau)`, note: '上漲機率就是分布在 0 右邊的面積；區間就是分位數 $q_\\tau$（例如 5% 與 95% 分位數構成 90% 區間）。' },
      { t: 'p', s: '評估「方向」時，實際報酬剛好為 0 的日子沒有方向，會被排除並另行計數。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 4
    { id: 'garch', title: '模型一：GJR-GARCH(1,1) 與 Student-t', blocks: [
      { t: 'p', s: '報酬的「方向」很難預測，但「波動大小」卻有很強的規律：大波動後面常接著大波動（波動叢聚），而且下跌時的波動通常比上漲時更大（槓桿效應）。GJR-GARCH 正是為此設計的。' },
      { t: 'eq', s: [R`r_t = m + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim t_\nu\ \text{（標準化為變異數 1）}`,
        R`\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbf{1}[\varepsilon_{t-1}<0])\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2`], note: '今天的變異數 $\\sigma_t^2$ = 長期水準 + 昨天衝擊的影響 + 昨天變異數的慣性。當昨天是負報酬（$\\mathbf{1}[\\cdot]=1$），衝擊的係數由 $\\alpha$ 變成 $\\alpha+\\gamma$，即壞消息讓波動升得更高。' },
      { t: 'ul', items: [
        '條件：$\\alpha,\\beta,\\gamma\\ge 0$，且持續性 $\\varphi=\\alpha+\\beta+\\gamma/2<1$（否則波動會發散）。',
        '變異數目標化：$\\omega=(1-\\varphi)\\,v$，$v$ 為估計窗口內報酬平方的平均，使長期波動回到歷史水準，少估一個難估的參數。',
        '誤差分布採 Student-t（自由度 $\\nu$ 也由資料估計）：真實報酬的「肥尾」比常態分布嚴重得多，用常態會嚴重低估大跌的機率。',
        '以最大概似法（L-BFGS-B）估計，使用最近 ' + F.fmtInt(CONST.garchWindow) + ' 個交易日（約 10 年）；每 ' + CONST.refitEvery + ' 個交易日重新估計一次。'
      ] },
      { t: 'h3', s: '多日預測' },
      { t: 'p', s: '$h$ 日累積報酬的變異數有封閉解，不需要模擬：' },
      { t: 'eq', s: [R`\mathbf{E}[\sigma_{t+k+1}^2] = \omega + \varphi\,\mathbf{E}[\sigma_{t+k}^2]`,
        R`\mathrm{Var}\!\left(\sum_{k=1}^{h} r_{t+k}\right) = \sum_{k=1}^{h} \mathbf{E}[\sigma_{t+k}^2]`], note: '波動預測會從現在的水準慢慢回歸長期平均，所以「一個月」的不確定性並不是「明日」的剛好 $\\sqrt{20}$ 倍：目前波動高於長期水準時會小於 $\\sqrt{20}$ 倍，低於長期水準時則會大於 $\\sqrt{20}$ 倍。' },
      { t: 'p', s: '多日報酬的分布還會變得負偏（槓桿效應造成）。引擎用「兩段式 Student-t」（Fernández–Steel）去配合變異數、峰度與偏度；偏度只採用模型隱含值的 ' + pc(CONST.skewDamping) + '%（阻尼係數由模型自己的蒙地卡羅檢驗校準，沒有用市場資料調整）。' },
      { t: 'note', kind: 'info', title: '它能預測什麼、不能預測什麼', s: 'GARCH 主要貢獻「區間寬度」，也就是不確定性的大小；它對漲跌方向幾乎沒有資訊（上漲機率只來自微小的漂移與偏態）。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 5
    { id: 'hmm', title: '模型二：高斯隱馬可夫模型（市場狀態）', blocks: [
      { t: 'p', s: '市場有時平靜盤整、有時穩定上漲、有時劇烈動盪。隱馬可夫模型（HMM）假設有 ' + CONST.hmmStates + ' 個看不見的「狀態」，每個狀態各有自己的平均報酬與波動，並且以固定的機率在狀態之間切換。' },
      { t: 'eq', s: [R`y_t = (r_t,\ \ln \text{區間波動}_t), \qquad y_t \mid x_t = k \ \sim\ \mathcal{N}\!(\mu_k,\ \mathrm{diag}(\sigma_k^2))`,
        R`\Pr(x_{t+1} = l \mid x_t = k) = A_{kl}`], note: '觀察值 $y_t$ 是「日報酬」與「對數高低價區間波動（Parkinson）」兩個數字；$x_t$ 是隱藏狀態，$A$ 是轉移機率矩陣。' },
      { t: 'h3', s: '濾波，而不是平滑' },
      { t: 'eq', s: R`\pi_t(k) = \Pr(x_t = k \mid y_1,\dots,y_t) \ \propto\ [\sum_{j}\pi_{t-1}(j)\,A_{jk}]\mathcal{N}(y_t;\mu_k,\sigma_k^2)`, note: '$\\pi_t(k)$ 是「看完今天為止」，目前處於狀態 $k$ 的機率。' },
      { t: 'p', s: '**平滑**機率（用整段歷史包含「之後的資料」回頭判斷某天是什麼狀態）會把未來的資訊洩漏到過去，讓回測看起來神準。這裡只使用**濾波**機率，每一天的狀態判斷都只依據當天以前的資料。' },
      { t: 'h3', s: '多日預測' },
      { t: 'eq', s: R`F_h(x) = \sum_{k} \pi_t(k)\ \Phi\!\left(\frac{x - m_k(h)}{\sqrt{v_k(h)}}\right), \qquad m_k(h) = \sum_{s=1}^{h}(A^{s}\mu)_k`, note: '預測是各狀態常態分布的混合；$m_k(h)$ 與 $v_k(h)$ 由轉移矩陣的冪次精確算出，已包含「期間內狀態可能切換」造成的額外變異。' },
      { t: 'ul', items: [
        '估計：EM 演算法（Baum–Welch）加共軛先驗（MAP），使用最近 ' + F.fmtInt(CONST.hmmWindow) + ' 個交易日；各狀態的平均報酬被拉向一個保守的漂移值，避免「某個狀態剛好是大漲期」被當成會持續。',
        '初始化是確定性的（不用隨機數），所以同樣的資料永遠得到同樣的結果。',
        '狀態依報酬變異數由小到大排序（0 = 最平靜）。畫面上的中文名稱（如「低波動盤整」「多頭穩定」「高波動空頭」）由各狀態擬合出的平均報酬與波動自動命名，只是方便閱讀的標籤。'
      ] }
    ] },
    // ------------------------------------------------------------------------------------------------ 6
    { id: 'dlm', title: '模型三：動態線性模型（貝氏卡爾曼迴歸）', blocks: [
      { t: 'p', s: '這個模型想回答：「動能、反轉、量能、大盤狀態這些因子，和未來報酬有沒有（會隨時間變化的）關係？」做法是一個係數會緩慢漂移的線性迴歸，用卡爾曼濾波逐日更新（West & Harrison 的折現因子動態線性模型）。' },
      { t: 'eq', s: R`y_j = x_j^{\top}\theta_t + v, \qquad v \sim \mathcal{N}(0, V)`, note: '$y_j$ 是「經波動標準化、扣除漂移」的 $h$ 日報酬；$x_j$ 是起點 $j$ 當天已知的 ' + CONST.nFeatures + ' 個因子（加上截距）；$\\theta_t$ 是會隨時間漂移的係數。' },
      { t: 'h3', s: '折現因子與貝氏更新' },
      { t: 'eq', s: [R`P \leftarrow \delta\,P + (1-\delta)\,\Lambda_0, \qquad \delta = ` + CONST.dlmDelta,
        R`e = y - x^{\top}m, \quad Q = h + x^{\top}Cx, \quad A = \frac{Cx}{Q}`,
        R`m \leftarrow m + A\,e, \qquad C \leftarrow C - Q\,AA^{\top}`], note: '$m$ 是係數的後驗平均，$C$ 是其共變異（尺度）。折現因子 $\\delta$ 讓舊資訊逐漸失效（半衰期約 1.4 年），使係數能跟著市場改變；折現是往「先驗精度」$\\Lambda_0$ 而不是往 0，讓係數不會因遺忘而變得雜訊過大。' },
      { t: 'ul', items: [
        '**嶺式先驗**：$\\theta\\sim\\mathcal{N}(0,\\tau^2 I)$，$\\tau=' + CONST.dlmTau + '$。這是一個明確的信念：股票日報酬幾乎無法預測（先驗的 $R^2$ 不到 1%），資料必須提出夠強的證據才會讓係數離開 0。',
        '**噪音變異數也在學習**：採 Normal–Inverse-Gamma 共軛形式，$S=d/n$ 估計 $V$，並以折現 $\\beta_v=' + CONST.dlmBetaV + '$ 緩慢遺忘。',
        '**只用已經「到期」的結果**：起點 $j$ 的目標要到第 $j+h$ 天才知道，所以第 $t$ 天的更新使用的是配對 $(x_{t-h},\\,y_{t-h})$，從不使用還沒發生的結果。',
        '**重疊目標的回火**：相鄰兩個 $h$ 日報酬共用 $h-1$ 天，資訊量只有獨立樣本的約 $1/h$。更新採「冪次概似」（指數 $1/h$）避免把同一份資訊重複計算。'
      ] },
      { t: 'p', s: '預測的平均值是 $x_t^{\\top}m$，預測分布是以 GARCH 的波動為尺度的 Student-t（再加上係數不確定性造成的變異）。個股頁的「預測驅動因子」表就是把 $m_j\\,x_{t,j}$ 逐項列出，顯示每個因子對預期報酬的加減貢獻。' },
      { t: 'h3', s: '用到的 ' + CONST.nFeatures + ' 個因子' },
      { t: 'p', s: '全部因子都以「波動單位」表示（可跨股票、跨時期比較），並用**擴張視窗**（只含過去資料）的平均與標準差做 z 分數標準化、截在 ±4。每個因子在第 $i$ 天的值只用到第 $0,\\dots,i$ 天的資料；這件事有截斷測試驗證。' },
      { t: 'features' }
    ] },
    // ------------------------------------------------------------------------------------------------ 7
    { id: 'naive', title: '模型四：隨機漫步基準', blocks: [
      { t: 'p', s: '這是「什麼都不知道」的模型：報酬的平均只是一個被大幅收縮的長期漂移，波動用緩慢更新的指數加權移動平均，分布為 $\\nu=' + CONST.naiveNu + '$ 的 Student-t。它的角色是**錨**：當其他模型都沒有可靠的訊號，最終預測就會靠近它，也就是接近「不確定」。' },
      { t: 'eq', s: R`\hat\mu_i = \frac{N_0\,\mu_0 + \sum_{s\le i} r_s}{N_0 + i}, \qquad N_0 = \frac{v_i}{\tau_0^{2}}`, note: '貝氏收縮：先驗平均 $\\mu_0$ 為每日 0.02%（約年化 5%），先驗標準差 $\\tau_0$ 為每日 0.04%（約年化 10%）；歷史越短或越不穩定，漂移越貼近先驗。兩個數字是事先固定的，沒有用回測調整。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 8
    { id: 'dma', title: '動態模型平均：把四個模型合成一個', blocks: [
      { t: 'p', s: '四個模型各給一個預測分布 $F_k$。動態模型平均（Raftery–Kárný–Ettler）依各模型「最近預測得有多好」決定權重：近期對實際結果給出較高機率密度的模型，權重較大。' },
      { t: 'eq', s: [R`\ln w_k(t) = \frac{\eta}{h}\sum_{s=t-W+1}^{t}\alpha^{\,t-s}\,\ln p_k(y_s) + \text{常數}`,
        R`F(x) = \sum_{k} w_k\,F_k(x)`], note: '$\\ln p_k(y_s)$ 是模型 $k$ 對「已到期結果」的對數預測密度；$\\alpha=' + CONST.dmaAlpha + '$ 是遺忘因子，$W=' + CONST.dmaWindow + '$ 為視窗長度，$\\eta=' + CONST.dmaEta + '$ 為學習率。第二式是「線性意見池」：把各模型的累積分布按權重平均。' },
      { t: 'ul', items: [
        '只用已到期的結果，且同樣以 $1/h$ 回火（重疊目標）。',
        '學習率 $\\eta$ 刻意很小：在開發期間（2021-08 以前的資料）比較過 {1, 0.5, 0.25, 0}，結果是**等權重（$\\eta=0$）得分最高**；為了讓持續更準的模型仍能拿到較高權重，取一個很小的 $\\eta$，所以權重只會緩慢移動。',
        '混合分布的分位數沒有封閉解，用二分法在精確的混合累積分布上求解（精度達機器誤差）。'
      ] },
      { t: 'p', s: '個股頁的「模型權重」圖就是這組權重；「隨機漫步基準」的權重偏高，表示目前各模型都沒有拿出像樣的證據。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 9
    { id: 'calibration', title: '線上貝氏機率校準', blocks: [
      { t: 'p', s: '模型合成後的上漲機率 $\\tilde p$ 可能系統性地偏高、偏低或過度自信。校準步驟用「已經到期」的預測與實際結果，學習一個把 $\\tilde p$ 映射成更誠實機率的函數（Platt 縮放的貝氏版本）。' },
      { t: 'eq', s: R`\Pr(\text{上漲}\mid \tilde p) = \operatorname{expit}\!(a + b\,\operatorname{logit}(\tilde p)), \quad a\sim\mathcal{N}(0,\,` + CONST.calSigmaA + R`^2),\ \ b\sim\mathcal{N}(1,\,` + CONST.calSigmaB + R`^2)`, note: '先驗把 $(a,b)$ 拉向「恆等映射」$a=0,b=1$（不修正）。$a$ 吸收一個小而持續的偏差，$b<1$ 表示原本過度自信、要往 50% 收縮。' },
      { t: 'ul', items: [
        '以數次阻尼牛頓法求後驗眾數（對數後驗為嚴格凹函數，穩定收斂）。',
        '只使用最近 ' + CONST.calWindow + ' 個起點中已到期的結果，並以 ' + CONST.calLambda + ' 的遺忘因子和 $1/h$ 回火。',
        '為什麼先驗這麼緊：原始機率大約落在 35%–65%，若校準器可以隨意移動，它帶來的雜訊會比它修正的偏差還多。',
        '個股頁同時保留校準前（raw）與校準後的機率；所有績效檢驗使用校準後的機率。'
      ] }
    ] },
    // ------------------------------------------------------------------------------------------------ 10
    { id: 'limit', title: '漲跌停（±10%）截斷', blocks: [
      { t: 'p', s: '台股每日漲跌幅限制為 ±10%。因此明日報酬的預測分布不可能有超出這個範圍的機率，引擎把它截斷並重新正規化：' },
      { t: 'eq', s: R`F_{\text{trunc}}(x) = \frac{F(x) - F(\ln 0.9)}{F(\ln 1.1) - F(\ln 0.9)}, \qquad \ln 0.9 \le x \le \ln 1.1`, note: '影響通常很小（大跌 10% 本來就很罕見），但分位數、上漲機率與 PIT 都因此嚴格遵守現實的限制。截斷只作用於「明日」的分布；$h=5,20$ 的累積報酬可超過 10%。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 11
    { id: 'fhs', title: '扇形圖、VaR 與 CVaR：濾波歷史模擬', blocks: [
      { t: 'p', s: '個股頁的扇形圖畫出未來 20 個交易日的價格區間（50% 與 90%）。它由 ' + F.fmtInt(CONST.mcPaths) + ' 條模擬路徑的分位數構成，採用「濾波歷史模擬」（Barone-Adesi 等）：路徑遵循擬合出的 GJR-GARCH 波動遞迴，但每天的衝擊不是從常態分布抽，而是從歷史上的**標準化殘差**重抽樣。' },
      { t: 'eq', s: [R`r_{t+s} = \hat\mu + \sigma_{t+s}\,z^{*}_{s}, \qquad z^{*}_{s} \ \text{取自}\ \{\,z_u = \varepsilon_u/\sigma_u\,\}`,
        R`\mathrm{VaR}_{95} = q_{0.05}(R), \qquad \mathrm{CVaR}_{95} = \mathbf{E}[\,R \mid R \le \mathrm{VaR}_{95}]`], note: '$z_u$ 是歷史上實際的標準化衝擊（含真實的肥尾與偏態）。VaR 是「有 95% 的機率不會比這更糟」的門檻；CVaR 是「真的落入最差 5% 時，平均虧多少」。' },
      { t: 'ul', items: [
        '**狀態混合**：每條路徑帶著一個隱藏狀態，依 HMM 的轉移矩陣演化（起點為今天的濾波機率）；在高波動狀態下，衝擊從歷史上高波動時期的殘差中抽取。',
        '每天的 ±10% 漲跌幅限制都被強制執行（被截斷的報酬也會回饋到波動遞迴）。',
        '**與精確分布對齊**：為避免扇形圖與三張期間卡片互相矛盾，模擬分位數在 1、5、20 日處被平移到與精確集成分位數一致，其間線性內插。',
        '另外由模擬得到的還有：20 日內最大回撤的中位數、20 日虧損超過 10% 的機率。'
      ] }
    ] },
    // ------------------------------------------------------------------------------------------------ 12
    { id: 'recommender', title: '每日模型觀察：相對強度排名、成本把關與配置', blocks: [
      { t: 'p', s: '「今日模型觀察」對固定股票池中的每檔股票回答兩個**不同**的問題，而且刻意用兩個不同的數字：**誰相對較強**（排名），以及**扣掉交易成本後還站不站得住**（把關）。這一頁是模型的相對看法，不是買賣建議。' },
      { t: 'h3', s: '排名：相對強度' },
      { t: 'eq', s: R`\mathrm{rs}_h = \frac{\mu_{\log,h}}{\sigma_{\log,h}}`, note: '$\\mu_{\\log,h}$ 與 $\\sigma_{\\log,h}$ 是 $h$ 日對數報酬的預期值與標準差（來自集成後的預測分布；任一缺漏時改用 $\\Phi^{-1}(p_{\\text{上漲}})$）。它不含交易成本，也沒有波動度的凸性項。排名、「相對偏多 Top N」與「相對偏空 Top N」都依它排序；同分時依上漲機率、再依代號。' },
      { t: 'note', kind: 'warn', title: '它偏向低波動的股票', s: '預測均值被強烈收縮到共同的漂移，所以分母（波動度）決定了大半的排序：在實際資料上，每日相對強度與預測波動度的 Spearman 相關約為 −0.4 ～ −0.6，名單日對日變動也很小。這代表它是一個「低波動偏好」的相對排序，**不是選股能力的證明**。因此每一列都會顯示波動 σ，並對預測波動低於全體中位數者標上「低波動」；「相對偏空」若所有名單的上漲機率仍高於 50%，標題會直接寫成「相對最弱（上漲機率仍高於 50%）」：整體市場漂移為正，不代表看空。' },
      { t: 'h3', s: '把關：成本調整分數' },
      { t: 'eq', s: R`\mathrm{net}_h = \mathbf{E}[\text{簡單報酬}_h] - c_{\text{來回}}, \qquad \mathrm{score}_h = \frac{\mathrm{net}_h}{\sigma_h}`, note: '$c_{\\text{來回}}$ 是買進再賣出一次的總成本（見「台灣交易成本」一節），$\\sigma_h$ 是該期間報酬的預期標準差。這是一個類似夏普比率的成本調整分數；它**只**用來決定偏多候選／偏空的旗標、「預期報酬未覆蓋交易成本」的標示與配置權重，**不用來排名**（短天期時所有人的 net 都是負的，除以 σ 後反而是最波動的排第一）。' },
      { t: 'h3', s: '贏家詛咒與 James–Stein 收縮' },
      { t: 'p', s: '從 ' + CONST.universe + ' 檔裡挑出「分數最高」的，這個高分有一部分只是估計誤差（運氣）。把每檔的分數視為「真實優勢＋雜訊」，並用經驗貝氏（正部 James–Stein 估計量）向全體平均收縮：' },
      { t: 'eq', s: R`z_i = \mathrm{score}_i\sqrt{T/h}, \qquad \hat\theta_i = \bar z + \left[\,1 - \frac{k-3}{\sum_j (z_j-\bar z)^2}\,\right]_{+}(z_i - \bar z)`, note: '$k$ 是股票池檔數，$T=' + CONST.shrinkBars + '$ 是一個預測平均值背後約略的有效日資料量；重疊的 $h$ 日目標只有 $T/h$ 份獨立資訊。若各檔分數的差異不比雜訊大，括號內變成 0，所有人都被收縮到同一個值：沒有人被「挑出來」，名單可能是空的。' },
      { t: 'h3', s: '偏多候選、偏空／避開與配置' },
      { t: 'ul', items: [
        '**偏多候選**：校準後上漲機率 ≥ ' + pc(CONST.buyP) + '%、扣成本後預期報酬 > 0，且近 20 日平均成交值 ≥ 新台幣 ' + (CONST.minAvgValue / 1e8) + ' 億元（流動性）。這只是通過篩選，不是買進建議；一個月期間常有一大批候選，那通常只是整體市場正漂移。',
        '**偏空／避開**：上漲機率 ≤ ' + pc(CONST.avoidP) + '% 且扣成本後預期報酬 < 0。本工具只做多，「避開」不是放空建議。',
        '**模型一致度（高／中／低）**：由機率偏離 50% 的程度、各模型方向是否一致，以及目前波動是否極高（歷史 90 百分位以上）綜合決定；它是模型內部的一致程度，不是經過驗證的把握。',
        '**「買進」字樣的條件**：只有當「推薦整體歷史驗證」達「中等優勢」以上、且扣成本後的精選組合沒有輸給全體等權重時，頁面才會使用「買進」；否則一律寫「偏多候選」「偏空」「中性觀望」。'
      ] },
      { t: 'eq', s: R`w_i = f\cdot\frac{\max(\hat\theta_i\,\sigma_i,\ 0)}{\sigma_i^{2}}, \qquad f = ` + CONST.kelly + R`, \quad w_i \le ` + pc(CONST.maxWeight) + R`\%, \quad \sum_i w_i \le 1`, note: '分數化凱利（Fractional Kelly）：完整凱利比例在估計有誤差時風險過高，所以只取 ' + CONST.kelly + ' 倍，單檔上限 ' + pc(CONST.maxWeight) + '%，其餘為現金。權重假設各檔彼此獨立，沒有考慮相關性，所以只是粗略的配置參考（頁面稱為「模型配置範例」）。若各檔預測的差異無法與雜訊區分，或沒有偏多候選同時具有正的收縮後優勢，**不配置任何權重，配置結果就是 100% 現金**。' },
      { t: 'note', kind: 'warn', title: '股票池的限制', s: '股票池是「今天」流動性最好的 ' + CONST.universe + ' 檔大型股（不含 ETF）。這已經是用事後資訊挑出來的名單：曾經很大、後來下市或衰退的公司不在其中，因此對這個池子的歷史驗證會偏樂觀（存活者偏差）。' },
      { t: 'h3', s: '頁面上方的「歷史驗證」橫幅' },
      { t: 'p', s: '「推薦整體歷史驗證」把同樣的選股規則套到過去的每一天：每天依相對強度挑前 5 名，各持有 $h$ 天（以重疊分批的方式，每批占 $1/h$ 資金），扣成本後與全體等權重、加權指數，以及一個**完全不使用預測**的對照組（每天只挑預測波動度最高的前 5 名）比較，並計算每日橫斷面 IC 與命中率；統計檢定採用 Newey–West 修正，且判定門檻比單一股票更嚴格。頁面最上方的橫幅就是這份驗證的摘要：沒有統計上顯著的預測力、或扣成本後打不贏全體等權重時是琥珀色，尚未計算時是灰色並提供「執行驗證」按鈕。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 13
    { id: 'walkforward', title: '前推式驗證，以及為什麼不可能偷看未來', blocks: [
      { t: 'p', s: '「回測很漂亮、實盤很難看」的最常見原因是**偷看未來**：不小心讓過去的某個決定用了未來才知道的資訊。這個工具用「前推式」（walk-forward）的設計，把偷看的路堵死：' },
      { t: 'timeline' },
      { t: 'eq', s: R`\text{預測}_i = f(\text{資料}_{0},\dots,\text{資料}_{i})`, note: '第 $i$ 天的預測是「第 0 到第 $i$ 天資料」的函數，不多不少。' },
      { t: 'ul', items: [
        '**擴張視窗**：起點 $i$ 的所有估計只用 $0,\\dots,i$ 的資料；第一個樣本外預測前至少有 ' + CONST.minTrain + ' 個交易日的訓練資料。',
        '**絕對時間表的重新估計**：模型參數在固定的列數（每 ' + CONST.refitEvery + ' 列）重新估計；起點 $i$ 使用的是 $\\le i$ 的最近一次估計。時間表只取決於資料的列號，不取決於資料有多長，因此資料變長不會改寫過去的預測。',
        '**只用已到期的結果**：DLM 更新、模型權重、機率校準，在第 $t$ 天都只使用「第 $t$ 天以前已經知道答案」的預測（起點 $j\\le t-h$）。',
        '**因子標準化用擴張統計量**：沒有全樣本平均、沒有置中視窗、沒有往前看的初始值。',
        '**HMM 只用濾波機率**，不用平滑機率（見前述）。'
      ] },
      { t: 'h3', s: '自動化測試：截斷不變性' },
      { t: 'p', s: '最直接的檢查是：把資料在第 $i$ 天截斷，重跑一次，第 $i$ 天的預測必須**完全相同**。這個性質由測試程式在隨機選出的許多天上驗證，所有期間、所有預測量都必須吻合：' },
      { t: 'eq', s: R`\text{walk_forward}(\text{資料}[0..i])\big|_{\text{最後一列}} \;=\; \text{walk_forward}(\text{資料})\big|_{\text{第 } i \text{ 列}}`, note: '同理，個股頁的即時預測與歷史驗證的最後一列走的是同一條程式路徑。若模型偷看了未來，截斷後的預測就會改變，測試會立刻失敗。' },
      { t: 'p', s: '此外，評估端獨立地由價格重算「實際報酬」，並與預測表中的欄位比對，防止日期錯位。' },
      { t: 'h3', s: '其他重要的細節' },
      { t: 'ul', items: [
        '預設的樣本外期間約 5 年（' + F.fmtInt(CONST.testBars) + ' 個交易日），可選 2、3、5、8 年；歷史不足時會自動縮短並提示。',
        '$h>1$ 時，相鄰預測彼此重疊（共用 $h-1$ 天報酬）。二項檢定、Spearman、Kupiec 與 KS 檢定都使用**有效樣本數 $n/h$**；Diebold–Mariano 檢定使用 Newey–West 變異數（落後 $h-1$ 期）。',
        '**超參數沒有偷看評估期**：模型的超參數（先驗、窗口、折現因子）事先由理論與慣例決定；僅有的兩個以數字輔助決定的值（模型平均的學習率、GARCH 偏態阻尼）只用到 2021-08 之前的資料，或根本沒用市場資料。但設計者看過歷史這件事本身，仍可能帶來難以量化的隱性影響。',
        '驗證結論只依據樣本外數字；沒有任何指標是用訓練期算的。'
      ] }
    ] },
    // ------------------------------------------------------------------------------------------------ 14
    { id: 'metrics', title: '指標詞彙表', blocks: [
      { t: 'p', s: '歷史驗證頁會出現下列指標。每個指標回答一個不同的問題；「樣本外」代表預測當下完全沒看過答案。' },
      { t: 'dl', items: [
        ['方向準確率與基準', '模型預測「漲或跌」猜對的比例（上漲機率 ≥ 50% 視為預測上漲）。基準是完全不用模型的簡單做法：永遠猜漲、沿用前一期的漲跌方向、猜訓練期較常出現的方向。**模型必須勝過「最佳基準」才有意義**，只高於 50% 並不夠，因為股市長期上漲，永遠猜漲本身就有約 52%–55% 的準確率。'],
        ['二項檢定與信賴區間', '若模型其實沒有優勢，準確率應該和基準一樣。單尾二項檢定計算「僅靠運氣得到這麼高準確率」的機率：', { eq: R`p\text{-value} = \Pr(X \ge k),\quad X\sim\mathrm{Bin}(n,\,p_0)`, note: '$k$ 為猜對次數，$p_0$ 為最佳基準的準確率。' }, '95% 信賴區間採 Wilson 區間（小樣本比常見的常態近似更可靠）：', { eq: R`\frac{\hat p + z^2/2n \ \pm\ z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1 + z^2/n}` }],
        ['Brier 分數與技能分數', '衡量「機率」本身有多準：', { eq: R`B = \frac1n\sum_i (p_i - y_i)^2, \qquad \mathrm{BSS} = 1 - \frac{B}{B_{\text{ref}}}`, note: '$y_i$ 為 1（上漲）或 0（下跌）。$B_{\\text{ref}}$ 是永遠預測整體上漲比例的 Brier 分數。技能分數 > 0 才代表機率預測比基準好；接近 0 代表沒有加分。' }],
        ['校準誤差 ECE 與校準圖', '把預測依機率分成幾組（每組樣本數相近），比較「該組平均預測機率」與「該組實際上漲頻率」：', { eq: R`\mathrm{ECE} = \sum_b \frac{n_b}{N}\,|\bar p_b - f_b|`, note: '$\\bar p_b$ 為第 $b$ 組的平均預測機率，$f_b$ 為實際頻率。ECE 越小，機率越「誠實」。' }],
        ['資訊係數 IC', '模型預期報酬的**排序**與實際報酬的**排序**之間的 Spearman 相關係數。+1 完全一致、0 無關、−1 完全相反。個股的 IC 即使真的有效，也常只有 0.02–0.05，一定要看 p 值。'],
        ['樣本外 $R^2$（Campbell–Thompson）', '預期報酬的預測誤差比基準（零預測或歷史平均預測，取誤差較小者）少了多少：', { eq: R`R^2_{\mathrm{OOS}} = 1 - \frac{\sum (R - f)^2}{\sum (R - f_{\mathrm{bench}})^2}`, note: '日報酬的 $R^2_{\\mathrm{OOS}}$ 通常貼近 0，甚至為負；只要大於 0 一點點就已經很難得。' }],
        ['Diebold–Mariano 檢定', '檢定兩個預測的損失是否有顯著差異。令 $d_t=L_{\\text{基準},t}-L_{\\text{模型},t}$：', { eq: R`\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}/m}}`, note: '長期變異數 $\\widehat{\\mathrm{LRV}}$ 用 Newey–West 估計（落後 $h-1$ 期，Bartlett 權重），處理重疊預測造成的自我相關。統計量為正代表模型的損失較小。' }],
        ['區間涵蓋率與 Kupiec 檢定', '名目 90% 的預測區間，實際報酬應該約 90% 的時間落在其中。Kupiec 的「比例失敗」概似比檢定判斷偏離是否顯著：', { eq: R`\mathrm{LR} = -2\ln\![(1-p)^{n-x}p^{x}] + 2\ln\![(1-\tfrac xn)^{n-x}(\tfrac xn)^{x}] \ \sim\ \chi^2_1`, note: '$p$ 為名目失敗率（例如 10%），$x$ 為實際落在區間外的次數。' }],
        ['PIT 直方圖與 KS 檢定', '把每個實際報酬 $R_i$ 代入「當時預測的累積分布」得到 $u_i=F_i(R_i)$。若預測分布完全正確，$u_i$ 應服從 0 到 1 的均勻分布。兩端偏高代表低估風險，中間偏高代表高估風險；Kolmogorov–Smirnov 檢定量化偏離均勻分布的程度。'],
        ['分位數損失（pinball）', '評估分位數預測的標準損失，越小越好：', { eq: R`\rho_\tau(u) = u(\tau - \mathbf{1}[u<0]), \qquad u = R - q_\tau` }],
        ['夏普比率、PSR 與 DSR', '夏普比率是年化平均報酬除以年化波動（無風險利率以 0 計）。機率夏普比率（PSR，Bailey 與 López de Prado）考慮樣本長度、偏態 $\\gamma_3$ 與峰態 $\\gamma_4$，估計真實夏普大於基準值的機率：', { eq: R`\mathrm{PSR} = \Phi\!(\frac{(\widehat{SR}-SR^{*})\sqrt{n-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^{2}}})` }, '通縮夏普（DSR）再把「試過 $N$ 組設定才挑出最好那組」的運氣成分扣掉（提高門檻 $SR^{*}$）。本工具單次回測 $N=1$，所以 DSR 與 PSR 相同。'],
        ['Stationary bootstrap', '用隨機長度（幾何分布）的連續區塊對報酬序列重抽樣，保留時間相依性，估計夏普比率及「策略與買進持有夏普差」的 95% 信賴區間。若區間跨過 0，就不能說策略比買進持有好。'],
        ['多重檢定與結論等級', '同時檢定四項指標（方向、Brier、IC、報酬誤差），總有機會碰巧有一項顯著。結論採保守規則：至少兩項檢定各自 < 5%，且校正後最小 p 值 < 1%，才算「微弱優勢」；達「顯著優勢」還需三項檢定 < 0.1%、樣本 ≥ 500，且扣成本後的策略要顯著勝過買進持有。只會重現整體上漲比例的模型，必然落在「無預測優勢」。']
      ] }
    ] },
    // ------------------------------------------------------------------------------------------------ 15
    { id: 'costs', title: '台灣交易成本', blocks: [
      { t: 'p', s: '回測與推薦的分數都計入下列成本；單邊成本由「手續費＋滑價（＋賣出時的證交稅）」組成：' },
      { t: 'ul', items: [
        '**手續費**：牌告 ' + F.fmtPct(CONST.commission, 4) + '，買賣各收一次；多數券商提供折扣（本工具預設 ' + discText(CONST.discount) + '，可在歷史驗證頁調整）。',
        '**證券交易稅**：只在賣出時收取，股票 ' + F.fmtPct(CONST.taxStock, 1) + '，ETF ' + F.fmtPct(CONST.taxEtf, 1) + '。',
        '**滑價**：實際成交價與理想價之間的差距，以每邊 ' + CONST.slippageBps + ' bp（' + F.fmtPct(CONST.slippageBps / 1e4, 2) + '）估計。',
        '**成交時點**：預設「下一個開盤價成交」（收盤後得到訊號，隔天開盤才能下單）；「收盤成交」假設能在收盤集合競價成交，較為樂觀。'
      ] },
      { t: 'costs' },
      { t: 'note', kind: 'info', title: '成本的重要性', s: '一次來回約 0.4%–0.7%。若模型每次交易的平均優勢小於這個數字，再準的預測也會被成本吃掉；頻繁交易（年換手率高）的策略尤其吃虧。' },
      { t: 'p', s: '**沒有計入的成本**：漲停鎖死或跌停無法成交、單筆手續費下限（新台幣 20 元）、市場衝擊（大單推動價格）、融資融券利息、現金利息、股利所得稅，以及期末持倉的賣出成本。這些都會讓真實結果比回測更差。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 16
    { id: 'limits', title: '限制與誠實說明', blocks: [
      { t: 'p', s: '以下是理解這個工具時最重要的幾件事。它們不是客套話，而是結論可靠度的上限。' },
      { t: 'ol', items: [
        '**效率市場假說。** 公開資訊很快就會反映在價格上；用歷史價格與成交量預測未來報酬，本來就非常困難。即使方法再精緻，多數個股、多數期間的預測優勢預期都很小，甚至不存在。',
        '**市場狀態會改變（結構性改變）。** 過去有效的關係可能在制度、利率、參與者結構或流動性改變後失效。模型用折現與遺忘因子適應變化，但無法預見從未發生過的事件（疫情、戰爭、政策突變）。',
        '**存活者偏差與股票池偏誤。** 推薦股票池是「今天」流動性最好的大型股，歷史上下市或衰退的公司不在其中；你在看過近年表現後才挑一檔股票來驗證，也有選擇偏誤。兩者都使驗證結果偏樂觀。',
        '**過度配適與多重檢定。** 看夠多的股票、指標與參數，總會找到「看起來有效」的組合。這就是為什麼結論採保守規則、要求多項檢定一致，並提供 DSR 這類修正；一個單獨顯著的結果，大多只是運氣。',
        '**除息造成的價格下跌與還原價假設。** 回測使用還原權息價（股利再投入、不計稅、零成本），實際持有人收到的是現金股利並要繳稅，價格在除息日會真的下跌。',
        '**微小的優勢對上交易成本。** 就算統計上有微弱優勢，每筆交易的期望超額報酬往往低於 0.5% 的來回成本，扣完之後可能什麼都不剩。',
        '**模型風險。** 任何模型都是對現實的簡化：分布形狀、狀態數、因子選擇都是假設。肥尾事件發生時，模型估計的機率與區間可能明顯偏離。',
        '**資料品質。** 價格與股利來自第三方公開資料，可能有延遲、缺漏或錯誤；清理規則能處理明顯的異常，但無法保證完全正確。',
        '**樣本有限。** 一個月（20 日）的預測，5 年只有約 60 組互不重疊的樣本；統計檢定的力量有限，「沒有顯著」不等於「確定沒有」。',
        '**機率不是保證。** 55% 的上漲機率代表「十次裡約五到六次上漲」，單次結果仍然很可能是下跌。'
      ] },
      { t: 'note', kind: 'info', title: '如果結論是「無預測優勢」', s: '這不是工具壞了，而是這個領域的常態。此時上方的機率應視為「接近不確定」；有價值的是區間與風險（VaR、波動狀態）：它們即使沒有方向資訊，仍能幫助你衡量部位大小與可承受的損失。' }
    ] },
    // ------------------------------------------------------------------------------------------------ 17
    { id: 'disclaimer', title: '免責聲明', blocks: [
      { t: 'note', kind: 'warn', title: '非投資建議', s: TQ.DISCLAIMER + ' 本工具提供的所有數字、名單、權重與結論，僅為統計模型的研究輸出，不構成任何買賣建議、要約或招攬。作者不對使用本工具造成的任何損失負責。投資前請自行研究，並在必要時諮詢合格的專業人士。' }
    ] }
  ];

  // ====================================================================================== block renderers
  function eqBlock(b) {
    var lines = Array.isArray(b.s) ? b.s : [b.s];
    return h('figure', { class: 'eq-block' },
      lines.map(function (src) {
        return h('div', { class: 'eq', role: 'math', 'aria-label': linearMath(src) }, h('span', { class: 'eq-in', 'aria-hidden': 'true' }, mathNodes(src)));
      }),
      b.note ? h('figcaption', { class: 'eq-note' }, h('span', { class: 'eq-tag' }, '白話'), inline(b.note)) : null);
  }

  function noteBlock(b) {
    return h('aside', { class: ['m-note', 'm-note-' + (b.kind || 'info')] },
      h('strong', { class: 'm-note-title' }, b.title), h('p', { class: 'm-note-body' }, inline(b.s)));
  }

  function flowBlock(b) {
    return h('ol', { class: 'm-flow' }, b.steps.map(function (st, i) {
      return h('li', { class: 'm-flow-step' },
        h('span', { class: 'm-flow-num', 'aria-hidden': 'true' }, String(i + 1)),
        h('div', { class: 'm-flow-body' }, h('strong', { class: 'm-flow-title' }, st[0]), h('span', { class: 'm-flow-text' }, st[1])));
    }));
  }

  function timelineBlock() {
    var rows = [
      { label: '第 $i$ 天的預測', seg: [['已知資料：第 0 到第 i 天（擴張視窗）', 62, 'known'], ['未知', 38, 'future']] },
      { label: '第 $i+1$ 天的預測', seg: [['已知資料：第 0 到第 i+1 天', 66, 'known'], ['未知', 34, 'future']] },
      { label: '評分時機', seg: [['', 62, 'blank'], ['目標在第 i+h 天到期，之後才能評分', 38, 'target']] }
    ];
    return h('figure', { class: 'm-timeline' },
      rows.map(function (r) {
        return h('div', { class: 'tl-row' },
          h('div', { class: 'tl-label' }, inline(r.label)),
          h('div', { class: 'tl-bar', 'aria-hidden': 'true' }, r.seg.map(function (s) {
            return h('span', { class: ['tl-seg', 'tl-' + s[2]], style: { width: s[1] + '%' } }, s[0]);
          })));
      }),
      h('figcaption', { class: 'm-caption' }, '時間由左向右。每個預測只使用「已知」的部分（藍色）；右邊的「未知」永遠不會被模型看到，直到它變成過去並且結果已經發生。'));
  }

  function featuresBlock() {
    return ui.table([
      { key: 'name', label: '因子', rowHeader: true, render: function (r) { return h('span', null, r[1], h('small', { class: 'm-key' }, ' ' + r[0])); } },
      { key: 'def', label: '計算方式', render: function (r) { return r[2]; } },
      { key: 'why', label: '想捕捉的現象', render: function (r) { return r[3]; } }
    ], FEATURES, { caption: '動態線性模型使用的 ' + CONST.nFeatures + ' 個因子', dense: true, class: 'm-table' });
  }

  function costsBlock() {
    var rows = costRows();
    return ui.table([
      { key: 'disc', label: '手續費折扣', rowHeader: true, render: function (r) { return discText(r.disc); } },
      { key: 'comm', label: '手續費（單邊）', num: true, render: function (r) { return F.fmtPct(r.comm, 4); } },
      { key: 'buy', label: '買進單邊', num: true, render: function (r) { return F.fmtPct(r.buy, 4); }, tip: '手續費＋滑價' },
      { key: 'sellStock', label: '賣出單邊（股票）', num: true, render: function (r) { return F.fmtPct(r.sellStock, 4); }, tip: '手續費＋滑價＋證交稅 0.3%' },
      { key: 'rtStock', label: '來回（股票）', num: true, render: function (r) { return F.fmtPct(r.rtStock, 3); }, tip: '買進單邊＋賣出單邊' },
      { key: 'rtEtf', label: '來回（ETF）', num: true, render: function (r) { return F.fmtPct(r.rtEtf, 3); }, tip: '證交稅 0.1%' }
    ], rows, { caption: '依手續費折扣試算的交易成本（滑價以每邊 ' + CONST.slippageBps + ' bp 計）', dense: true, class: 'm-table',
      rowClass: function (r) { return Math.abs(r.disc - CONST.discount) < 1e-9 ? 'row-total' : null; } });
  }

  function dlBlock(b) {
    return h('dl', { class: 'm-dl' }, b.items.map(function (it) {
      var parts = it.slice(1).map(function (x) {
        if (typeof x === 'string') return h('p', { class: 'm-dd-p' }, inline(x));
        if (x && x.eq) return eqBlock({ s: x.eq, note: x.note });
        return null;
      });
      return h('div', { class: 'm-dl-item' }, h('dt', null, inline(it[0])), h('dd', null, parts));
    }));
  }

  function renderBlock(b) {
    switch (b.t) {
      case 'p': return h('p', { class: 'm-p' }, inline(b.s));
      case 'h3': return h('h3', { class: 'm-h3' }, inline(b.s));
      case 'ul': return h('ul', { class: 'm-list' }, b.items.map(function (x) { return h('li', null, inline(x)); }));
      case 'ol': return h('ol', { class: 'm-list m-olist' }, b.items.map(function (x) { return h('li', null, inline(x)); }));
      case 'eq': return eqBlock(b);
      case 'note': return noteBlock(b);
      case 'flow': return flowBlock(b);
      case 'timeline': return timelineBlock();
      case 'features': return featuresBlock();
      case 'costs': return costsBlock();
      case 'dl': return dlBlock(b);
      default: return null;
    }
  }

  // ====================================================================================== view
  function title() { return '方法說明'; }

  function mount(host, params, ctx) {
    var destroyed = false, timer = null;
    var sections = [];
    var tocLinks = [];

    function reducedMotion() {
      try { return !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (e) { return false; }
    }

    function goTo(id, opts) {
      opts = opts || {};
      var sec = null;
      sections.forEach(function (s) { if (s.id === id) sec = s; });
      if (!sec) return false;
      try { sec.node.scrollIntoView({ block: 'start', behavior: reducedMotion() || opts.instant ? 'auto' : 'smooth' }); } catch (e) { try { sec.node.scrollIntoView(); } catch (e2) { /* ignore */ } }
      if (opts.focus !== false && sec.node.focus) { try { sec.node.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
      if (opts.replace !== false && typeof ctx.replace === 'function') ctx.replace('#/method/' + id);
      return true;
    }

    var tocItems = SECTIONS.map(function (s, i) {
      var a = h('a', { class: 'm-toc-link', href: '#/method/' + s.id, dataset: { section: s.id },
        onClick: function (e) { if (e && e.preventDefault) e.preventDefault(); goTo(s.id); } },
      h('span', { class: 'm-toc-num' }, String(i + 1) + '．'), s.title);
      tocLinks.push(a);
      return h('li', null, a);
    });
    var wide = (root.innerWidth || 0) >= 760;
    var toc = h('details', { class: 'card m-toc', open: wide },
      h('summary', { class: 'm-toc-sum' }, '本頁目錄（' + SECTIONS.length + ' 節）'),
      h('nav', { 'aria-label': '本頁目錄' }, h('ol', { class: 'm-toc-list' }, tocItems)));

    var cards = SECTIONS.map(function (s, i) {
      var body = s.blocks.map(renderBlock).filter(Boolean);
      body.push(h('p', { class: 'm-top' }, h('a', { class: 'm-top-link', href: '#/method', dataset: { top: '1' }, onClick: function (e) {
        if (e && e.preventDefault) e.preventDefault();
        try { toc.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' }); } catch (x) { /* ignore */ }
        if (typeof ctx.replace === 'function') ctx.replace('#/method');
      } }, '▲ 回到目錄')));
      var card = ui.card((i + 1) + '．' + s.title, body, { id: 'm-' + s.id, class: ['m-sec', 'm-sec-' + s.id] });
      card.setAttribute('tabindex', '-1');
      sections.push({ id: s.id, node: card });
      return card;
    });

    var view = h('div', { class: 'view view-method' },
      h('div', { class: 'page-head' }, h('div', { class: 'page-head-text' },
        h('h1', { class: 'page-title' }, '方法說明'),
        h('p', { class: 'page-sub' }, '每一個數字是怎麼算出來的、做了哪些假設、有哪些限制。看不懂公式沒關係，每條公式後面都有白話說明。'))),
      h('div', { class: 'm-article' }, toc, cards));
    host.appendChild(view);

    var want = params && typeof params.section === 'string' ? params.section : null;
    if (want) timer = root.setTimeout(function () { timer = null; if (!destroyed && !ctx.signal.aborted) goTo(want, { replace: false, instant: true }); }, 60);

    return { destroy: function () { destroyed = true; if (timer) { root.clearTimeout(timer); timer = null; } } };
  }

  TQ.views.method = {
    title: title, mount: mount,
    _internals: { mathNodes: mathNodes, linearMath: linearMath, inline: inline, SECTIONS: SECTIONS, CONST: CONST, FEATURES: FEATURES, costRows: costRows }
  };
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
