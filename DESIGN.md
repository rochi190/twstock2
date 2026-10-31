# 台股量化實驗室 (TW Quant Lab) — Design & Contracts

Single source of truth for every module.  Four builders work in parallel; **the interfaces below are
binding**.  Read `backend/config.py` and `backend/types.py` first (they are shared and READ-ONLY for
builders — if you need a constant, define it in your own module and mention it in your final report).

## 0. Product

A local web app (Python stdlib HTTP server + numpy/scipy, vanilla-JS frontend, no build step, no CDN) for
Taiwan-listed stocks/ETFs (TWSE 上市 and TPEx 上櫃):

1. **個股預測** – pick any stock/ETF → predicted probability of rising / falling, expected return and
   uncertainty bands for next day, 1 week, 1 month, a fan chart, regime & volatility state.
2. **今日推薦** – each trading day, scan a liquid universe, rank by the relative-strength signal `rank_signal = mu_log / sigma_log`
   (§9.1; the cost-adjusted score only gates the strict flags and the weights), list the leaders / laggards, the strict
   buy candidates (偏多候選) and names to avoid, with suggested weights only when the cross-section carries a view.
3. **歷史驗證** – walk-forward, strictly out-of-sample backtest of the very same model on past data:
   direction accuracy vs baselines, calibration, return-forecast skill, interval coverage, and a
   cost-aware strategy simulation vs buy-and-hold.  Universe-level validation of the daily recommender.
4. **方法說明** – plain-Chinese + math description of every technique and its limits.

### Honesty principles (non-negotiable)
* Equity returns are close to unpredictable.  The app must NEVER imply guaranteed profit.  Every forecast
  is a probability; every claim of skill is shown with out-of-sample evidence, baselines, p-values, and a
  plain-language verdict that is allowed to say "沒有統計上顯著的預測優勢".
* **No look-ahead anywhere.**  Row i of any forecast uses bars `0..i` only, including parameter fitting,
  calibration, feature scaling, ensemble weights.  This is unit-tested by truncation invariance.
* Backtests include Taiwan trading costs (`config.roundtrip_cost`), next-open execution by default, and
  disclose survivorship/selection bias of the universe (today's liquid names).
* Taiwan conventions: **紅漲綠跌** (red = up, green = down) everywhere in the UI; ±10 % daily limit; 1 lot =
  1000 shares (odd lots exist); TAIEX (`^TWII`) is the market factor; ETF sell tax 0.1 % vs stock 0.3 %.

## 1. Layout & ownership

```
backend/config.py, backend/types.py      shared contracts (done, read-only)
backend/quant/        [A QUANT]    garch.py hmm.py dlm.py ensemble.py calibration.py mc.py features.py engine.py
backend/evaluation/   [C EVAL]     metrics.py strategy.py report.py recommend.py universe_bt.py
backend/data.py       [B SERVER]   Yahoo/TWSE fetch + cache + universe + calendar
backend/jobs.py       [B SERVER]   background jobs + progress, process pool
backend/server.py     [B SERVER]   HTTP API + static files   (python -m backend.server)
frontend/             [D FRONT]    index.html style.css app.js charts.js (+ more js files if wanted)
tests/                each builder adds own files: test_quant_*.py / test_eval_*.py / test_data_*.py / test_server_*.py / test_front_*.py
README.md run.sh requirements.txt      [B SERVER]
```
Never edit another builder's files.  Tests use stdlib `unittest` (`python -m unittest discover -s tests`).
Python 3.11, numpy 2.x, scipy 1.16, pandas available but **core code must only need numpy + scipy**.
No numba/statsmodels/arch/hmmlearn/sklearn at runtime (implement GARCH/HMM/etc. yourself).

## 2. Data (module B) — `backend/data.py`

```python
get_price_data(symbol: str, refresh: bool = False, allow_stale: bool = True) -> PriceData
get_market_data(refresh=False) -> PriceData               # ^TWII (rows with null OHLC dropped)
get_universe(n: int = config.UNIVERSE_SIZE, refresh=False) -> list[dict]   # [{code,name,market,kind}] liquid names
search_instruments(q: str, limit: int = 20) -> list[dict]  # by code prefix or Chinese-name substring
instrument_info(symbol) -> dict                            # {code,name,market,kind} or raise KeyError
last_trading_date_expected(now=None) -> date               # Taipei calendar aware (weekends, TWSE holidays if fetchable)
next_trading_day(d: date) -> date
market_is_open(now=None) -> bool                           # Taipei 09:00-13:30 on trading days
```
* **Yahoo chart API**: `https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2={now}&interval=1d&events=div%2Csplits`
  with `User-Agent: Mozilla/5.0`.  **NEVER use `range=max`** — Yahoo silently downgrades it to monthly bars.
  Symbols: `2330.TW` (TWSE), `6488.TWO` (TPEx), index `%5ETWII`.  Try `.TW` then `.TWO` and remember which
  worked.  Verified: 2330 → 6669 daily bars from 2000-01-04; `^TWII` has ~141 null rows (drop them);
  stocks have a handful of null rows (drop).  Volume of `^TWII` is 0.
* `timestamp + 8h` (gmtoffset 28800) → Taipei date.  Use `adjclose` for modelling; keep raw OHLC for display.
* **Incomplete today bar**: if the last bar's date is today (Taipei) and the session is not finished
  (before 14:00 Taipei), set `last_bar_complete=False`; downstream modelling must drop that bar
  (`PriceData.slice(n-1)`) — provide helper `completed(data) -> PriceData`.
* Names/universe: TWSE OpenAPI `https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL` (code, Chinese
  name, TradeValue, ...; 1383 rows, ~1096 four-digit codes; ETFs start with `00`, e.g. `0050`, `00878`,
  `006208`, some end in a letter like `00400A`) and `.../v1/opendata/t187ap03_L` (listed company info:
  公司代號, 公司簡稱, 產業別).  TPEx OpenAPI `https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes`
  exists but the big response failed with IncompleteRead once → read robustly with retries, tolerate failure
  (then OTC names are simply absent from search but any OTC code typed by the user must still work via `.TWO`).
  Holidays: `https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&queryYear=2026` (optional).
* `kind`: `'etf'` if code starts with `00`, else `'stock'`.
* **Universe** (`get_universe`): Taiwan-50-style large caps (hard-code a fallback list of ~60 well-known
  liquid codes with Chinese names inside data.py, e.g. 2330 2317 2454 2308 2382 2412 2881 2882 2891 2886 1301
  1303 2002 3711 2603 2609 3034 2379 …, verified against STOCK_DAY_ALL) topped up with the highest `TradeValue`
  four-digit stocks from today's STOCK_DAY_ALL to reach `n`; ETFs excluded from the recommender universe
  (kept searchable).  Cache for 7 days in `data_cache/universe.json` so the list is stable.
* **Cache**: `data_cache/prices/{symbol}.json` (or .npz). Refresh when the last cached bar is older than
  `last_trading_date_expected()` (and a new bar can exist), at most once per ~10 min per symbol.  On network
  failure serve the cache with `stale=True`.  Atomic writes (tmp + rename).  Thread-safe (per-symbol lock).
  Retries with backoff on HTTP 429/5xx; limit concurrency (≤6 parallel Yahoo requests).

## 3. Quant core (module A) — `backend/quant/`

Goal: a calibrated predictive **distribution** of the cumulative log return `R = ln(adjC[i+h]/adjC[i])` for
h ∈ `config.HORIZONS`, from a small set of mathematically distinct models combined adaptively.

### Required public API — `backend/quant/engine.py`
```python
walk_forward(data: PriceData, market: PriceData | None, horizons=config.HORIZONS,
             min_train=config.MIN_TRAIN, refit_every=config.REFIT_EVERY,
             start_idx: int | None = None, progress=None, cfg: dict | None = None) -> types.WalkForward
forecast_latest(data: PriceData, market: PriceData | None, horizons=config.HORIZONS,
                cfg: dict | None = None) -> dict          # the "forecast core" JSON object, §6.1
```
* `start_idx`: first forecast origin row (default `max(min_train, 0)`); callers pass `n - TEST_BARS_DEFAULT`.
  The engine must still use ALL rows `< start_idx` for the initial fit (expanding window) — never the future.
  (No-trade placeholder bars are dropped before modelling, §3.9: rows are then counted in real bars and a placeholder row repeats
  the previous real forecast.)
* `progress(done:int,total:int)` optional callback.
* `market` is aligned to `data` by date (left-join on the stock's dates, forward-fill the market's last known
  value only from the PAST).  `None` → market features disabled.
* **Single code path**: `forecast_latest(data)` must equal the last row of `walk_forward` on the same data
  (same parameters schedule), plus the extras (fan chart, regime, risk, drivers).  Refit schedule is absolute:
  refit at rows `first_fit_idx + k*refit_every`; `first_fit_idx` is determined by `start_idx`/`min_train` only
  (not by `len(data)`), so truncating the data never changes earlier forecasts.
* `forecast_latest` needs ≥ `config.MIN_BARS_FORECAST` REAL bars (no-trade placeholder bars, §3.9, are not counted) else raise
  `ValueError("insufficient_history")`; `walk_forward` raises the same error when `start_idx` / `min_train` leave no reportable origin.
* Performance targets (4-core laptop): `forecast_latest` ≤ 5 s on 25 years of data; `walk_forward` over 1250
  test rows × 3 horizons ≤ 40 s (≤ 15 s preferred).  Hint: GJR-GARCH variance recursion
  `σ²_t = ω + (α + γ·1[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}` is linear in σ² given the data →
  `scipy.signal.lfilter([1],[1,-β], u)` evaluates the whole filter vectorised; warm-start refits from the
  previous parameters; cap fitting windows (e.g. last 2500–3000 bars) if needed.

### Modelling menu (make good, defensible choices; document them in module docstrings)
Components, each giving for every horizon a **closed-form or tabulated predictive CDF** (so quantiles, P(R>0),
PIT and log-score are cheap and exact):
1. **GJR-GARCH(1,1)-t** (Student-t innovations, MLE by scipy.optimize with sane bounds, stationarity
   `α+β+γ/2<1`, variance targeting or free ω).  Gives σ̂_{i+k|i} term structure; h-step variance
   `Σ_k σ²_{i+k|i}`; fat-tail aggregation across h (moment-matched Student-t / Cornish–Fisher; verify against
   Monte-Carlo in a unit test).  Optional small drift.
2. **Gaussian HMM** (K=3 regimes on `[daily log return, log range-based vol]`, EM / Baum–Welch in numpy with
   scaling; several random inits; warm start on refits; states sorted by variance).  Use **filtered** (never
   smoothed) state probabilities → h-step mixture via transition-matrix powers → regime-conditional drift &
   vol.  Provide readable Chinese `regime_labels` (e.g. 低波動盤整 / 多頭穩定 / 高波動空頭 — derive from fitted
   state mean & vol, don't hard-code the order).  Audit M-06: the volatility tier (低/中/高波動) comes from the variance rank;
   a trend word (多頭/空頭) is attached only when `|mu_k / sigma_k| > hmm.TREND_THRESHOLD` (= 0.10, the 95th percentile of the calm
   regime's ratio on development refits), because the engine shrinks every regime mean to the drift and smaller ratios are noise
   (with the old 0.04 the calm state's word flipped between 低波動盤整 and 低波動多頭 in 9.9 % of consecutive refits, now 1.5 %).
   The usual labels are therefore 低波動盤整 / 中波動震盪 / 高波動震盪.  The fitted persistence is only ~0.75-0.8, so the filtered
   regime probabilities move quickly: the live note (`vol.regime_note`) says "判斷不明確" instead of naming a state when
   `max(probs) < engine.REGIME_CONFIDENT` (0.6); `regime.current` is still the arg-max label.
3. **Dynamic Linear Model / Bayesian regression** (West & Harrison discount-factor DLM, i.e. Kalman filter on
   time-varying regression coefficients with Normal–Inverse-Gamma variance learning; ridge-like prior) of
   the **vol-standardised** h-step return on features computed from data ≤ i: multi-horizon momentum
   (incl. 12-1 month), short-term reversal, MA gap / RSI-type z-scores, range-vol ratio, volume z-score,
   market return/vol (if `market` given), idiosyncratic residual momentum.  Careful: the target for origin i
   is only observed at i+h, so the update at time i uses the pair `(x_{i-h}, R_{i-h→i})` ("matured" outcomes
   only).  Predictive is Student-t with the regression mean and GARCH-based scale.
4. **Naive baseline** (random walk with shrunk long-run drift and slowly varying EWMA vol) — it anchors the
   ensemble so the model can fall back to "no information".
5. **Combiner: Dynamic Model Averaging** (Raftery–Kárný–Ettler forgetting factor α≈0.98–0.995 on the
   log predictive likelihood of matured outcomes) → linear opinion pool of component predictive CDFs.
   (Optionally compare with exponentially-weighted Hedge; keep whichever is better and tested.)
6. **Calibration**: online Bayesian logistic (Platt) recalibration of `P(R>0)` using only matured outcomes,
   regularised toward identity (slope 1) so it is stable with little data.  Store raw and calibrated `p_up`.
   After combining, also report PIT-based diagnostics via `HorizonForecasts.pit`.
   * **Target = the direction of moving closes (audit C-03 / M-01).**  The pooled predictive is continuous (no atom at 0) but a
     realised close on the tick grid is unchanged with probability 6.8 % on average at h = 1 (25 % for 2412-type low-priced
     names; 2.7 % at h = 5, 1.2 % at h = 20).  The recalibrator therefore learns `1[R>0]` only on the rows where `R != 0`
     (unchanged closes carry no target, `engine._Ctx.up` is NaN there), so the published `p_up` is `P(R>0 | R != 0)` =
     `1 - CDF(0)` of the continuous predictive, exactly what DESIGN §4 scores (ties excluded) and what the causal base rate
     estimates.  A flat close is neither up nor down: the downside probability of a name with many flat closes is NOT `1 - p_up`.
     (Before the fix the calibrator was trained on `1[R>0]` of ALL rows, flat = "not up", but the continuous ensemble it
     recalibrates has no flat mass and the Platt prior removed only ~38 % of that gap, so the shipped `p_up` matched neither
     convention: +2.0 pp too high against the all-rows frequency of `R>0` (+7 pp for 2412 / 3045) and 1.4 pp too low against
     `P(R>0 | R != 0)`; the evaluation module scored the second one, which is where the apparent Brier deficit at h = 1 came from.)
   * **`HorizonForecasts.pit` is the randomised PIT of the rounded terminal close (audit M-05)**: with `F` the predictive CDF and
     `[lo, hi] = R + ln(1 ∓ tick/(2 c))` the tick cell of the terminal close `c` (Taiwan tick table: stocks < 10: 0.01, < 50:
     0.05, < 100: 0.1, < 500: 0.5, < 1000: 1, else 5; ETFs < 50: 0.01, else 0.05; none for an index),
     `pit = F(lo) + U (F(hi) - F(lo))`, exactly U(0,1) for a calibrated forecaster.  `U` is a pure function of (symbol, origin date,
     horizon), so a row never depends on later rows.  The plain `F(R)` piled up at `F(0)` whenever a close was unchanged and made
     the PIT-KS test reject a perfect forecaster for 16 of 63 stocks.
7. **Limit handling**: 1-day cumulative simple return must lie in [−10 %, +10 %]: truncate/clip the 1-day
   predictive (renormalise) — small effect, but quantiles must respect it.
8. **Live-only extras (forecast_latest)**: filtered-historical-simulation Monte-Carlo (≥ 4000 paths, GARCH
   standardised residuals + regime mixing, daily limit respected) for the **fan chart** (`config.FAN_LEVELS`
   over days 1..`FAN_DAYS`), VaR/CVaR, median 20-day max drawdown; feature `drivers` with value, z-score and
   signed contribution to the DLM mean; regime probabilities; current vol and its percentile in history.
9. **No-trade placeholder bars are not observations (audit C-09)**: Yahoo fills trading halts, exchange holidays (the Lunar New
   Year runs of 5-7 bars until ~2005 in every stock) and pre-listing periods with flat zero-volume bars (open = high = low = close,
   volume 0, adjusted close unchanged to within 1e-4 in log terms; an index series, which has no volume, is exempt).  Modelled as real
   sessions they are zero returns: the volatility filters decay although nothing was observed (2330 with 60 flat bars appended:
   one-day sigma 1.66 % -> 0.21 %, 90 % interval (-2.6 %, +2.8 %) -> (-0.2 %, +0.3 %)).  `features.placeholder_mask` finds them (a prefix
   function) and the engine removes them before any model sees the data.  Consequences: the absolute refit grid, `min_train` and
   `start_idx` count REAL bars (`start_idx` is a row of the data as given, the first reported origin is the first real bar at or after
   it; `WalkForward.first_idx` is a row of the data as given); a placeholder row of `walk_forward` repeats the forecast of the last real bar
   (nothing new was observed, so there is no NaN after `first_idx`); `realized` and `pit` follow the rows of the data as given (what an
   independent evaluation recomputes); `forecast_latest` ignores trailing placeholders (a halt in progress) but still reports the last
   row's `asof` / `last_close` / `chg_1d`.  Counts: `WalkForward.config["placeholder_bars_dropped"]`, `forecast["engine"]["placeholder_bars_dropped"]`.
   Not done: the first traded bar after a halt carries the accumulated multi-session move and is not given a wider predictive.
10. **Calendar time (audit M-04)**: the volatility models count every bar as one unit of time, but the bar that follows a weekend carries
   more variance than the average bar (squared standardised one-day error 1.19 times that of other bars on 178k development forecasts,
   stock-bootstrap interval 1.15-1.23, date-clustered bootstrap 1.13-1.26, replicated on the untouched test rows; French & Roll 1986 for US stocks).  The one-day predictive
   (h = 1 only) is therefore wider in VARIANCE by the factor `1 + cfg["weekend_var_uplift"]` (0.16) at every Friday origin, for every
   component alike (GARCH, DLM, naive, HMM regimes); the means are untouched.  For h >= 5 every window holds exactly one weekend bar, a
   constant that merges into the level, and the development rows show no gain, so h = 5 and h = 20 are identical to the uncorrected model.
   The multiplier is a function of the origin's own date (`features.next_bar_after_weekend`), so truncation invariance holds
   (`WalkForward.vol1d` and `vol.sigma_daily` stay the GARCH volatility STATE, before the uplift).  **Not
   modelled:** the extra variance of the bar after a weekday holiday (about +0.7 of a normal day per skipped weekday, ~4 % of the rows):
   it needs the exchange calendar of the FUTURE bars, the TWSE schedule endpoint serves only the current year (`queryYear` is ignored)
   and inferring it from the realised dates would break the no-look-ahead rule (typhoon closures are not known in advance).

Hard invariants (unit-tested): quantiles monotone in level; `0<p_up<1`; all finite in `forecast_latest`;
no NaN after `first_idx`; `HorizonForecasts` arrays all length n; `p_up` ≈ `1-CDF(0)` of the same predictive.

### Required tests (`tests/test_quant_*.py`)
* **Truncation / no-look-ahead**: for several random i, `walk_forward(data.slice(i+1))` last-row outputs ==
  `walk_forward(data)` row i (all horizons, `np.allclose`), using the same `start_idx`/`min_train`.
* Synthetic recovery: GARCH params, HMM regimes (sticky 2-state sim), DLM tracking of a drifting coefficient,
  ensemble weight shifts to the better component.
* i.i.d. noise: mean `p_up` ≈ 0.5±0.03, 90 % interval coverage within ±3 %, PIT roughly uniform, no systematic
  "edge" (accuracy ≈ base rate).  A planted weak predictable signal must produce measurably better log-score.
* Real-data smoke (skip gracefully if offline): 2330, 0050 — speed targets and invariants.

## 4. Evaluation (module C) — `backend/evaluation/`

Pure numpy/scipy; consumes `PriceData` + `WalkForward` (§ types.py).  Compute realized returns independently
from `PriceData` and assert they match `HorizonForecasts.realized` (guards alignment bugs).

### `metrics.py` (each function documented, hand-checked in tests)
Direction (`up = R>0`; rows with `R==0` excluded and counted, since a direction is only defined when the price moves):
accuracy + Wilson CI, base rate, baselines
(always-up, previous-day sign (for h=1) / trailing-h-return sign, majority-class from training),
exact binomial p-value vs best baseline rate, accuracy by confidence bucket, accuracy of "high-confidence"
calls; probabilistic block (Brier, log-loss, skill, reliability table (equal-count bins), ECE): scores `p_up` as the EVENT THE
FORECASTER DECLARES in `WalkForward.config["p_up_event"]` (audit C-02): `"up_given_move"` = `P(R>0 | R≠0)` (what the quant engine
forecasts, §3.6: flat closes are neither up nor down and are left out of the scoring, like the direction block); undeclared / `"up"`
= `P(R>0)` (the types.py contract: scored on EVERY evaluated forecast with `y = 1[R>0]`, a flat outcome counts as "not up").  The
reference is the CAUSAL expanding-window base rate of the same event known at each origin (§9(a)); the oracle (test-sample base
rate) references are reported separately in `*_oracle` keys; ECE is shown next to the level an exactly calibrated forecaster
reaches by binning noise (`ece_noise_floor`); return forecast:
Spearman IC (+p), Campbell–Thompson OOS R² vs zero/historical-mean forecast, Diebold–Mariano with
Newey–West HAC (lag h−1) vs naive; interval coverage (50/80/90 % from the QLEVELS grid) + Kupiec/Christoffersen
tests, PIT histogram (10 bins) + Kolmogorov–Smirnov p, pinball loss; rolling-window accuracy vs base rate;
strategy stats: CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, turnover, #trades, exposure, hit-rate of trades,
**Probabilistic Sharpe Ratio** and **Deflated Sharpe Ratio** (Bailey–López de Prado, `n_trials` param),
stationary-bootstrap CI for Sharpe and for Sharpe difference vs buy-and-hold.

### `strategy.py`
Position rule from calibrated forecasts: long when `p_up ≥ thr` AND cost-adjusted expected return > 0 (else
cash); optional fractional-Kelly sizing `w = clip(KELLY_FRACTION·μ/σ²…, 0, 1)` mode.  Execution modes
(`exec`): `'next_open'` (DEFAULT, realistic): signal at close t, trade at open t+1, overnight return of day t+1
earned with position `s_{t-1}`, intraday open→close with `s_t`:  `r_{t+1} = s_{t-1}·co_{t+1} + s_t·oc_{t+1}`;
and `'close'` (optimistic: trade at the close auction): `r_{t+1} = s_t·cc_{t+1}`.  Cost = `|s_t − s_{t-1}|`·
per-side cost (half of `roundtrip_cost` minus the tax part on buys: buys pay commission+slippage, sells pay
commission+slippage+tax) charged at the trade.  Use adjusted OHLC for returns.  Benchmark = buy-and-hold
close-to-close over the same dates.  Must be unit-tested on tiny hand-computed examples, including a test that
proves the signal at t cannot earn return before t+1.

### `report.py`
```python
build_backtest_report(data: PriceData, wf: WalkForward, horizon: int, *, threshold=config.BUY_P_UP,
                      exec_mode='next_open', commission_discount=config.COMMISSION_DISCOUNT,
                      slippage_bps=config.SLIPPAGE_BPS, sizing='threshold', n_trials=1) -> dict   # §6.4 schema
validate_backtest_report(report: dict) -> list[str]     # list of schema problems ([] if OK) — used by tests & server
```
`verdict` (honest, rule-based, Chinese): level ∈ `no_edge | weak_edge | moderate_edge | strong_edge` decided
from significance (binomial p vs best baseline, Brier skill, IC p, DM p, strategy PSR vs benchmark after costs)
— conservative; a model that merely matches the base rate must be `no_edge`.  Text lists the concrete numbers.

### `recommend.py`
```python
annotate_forecast(fc: dict, kind: str, *, buy_p=config.BUY_P_UP, avoid_p=config.AVOID_P_UP,
                  commission_discount=..., slippage_bps=...) -> dict   # adds per-horizon signal/score/confidence
build_recommendations(forecasts: list[dict], horizon: int, *, top_k=10, ...) -> dict    # §6.3 schema
```
* `score_h = (E[simple return_h] − roundtrip_cost(kind)) / σ_h` (a cost-adjusted Sharpe-like score; **only** a gate for
  `signal`, `edge_covers_cost` and the Kelly weights — the RANKING uses `rank_signal = mu_log/sigma_log`, §9.1).
* Cross-sectional **empirical-Bayes (James–Stein) shrinkage** of the z-scores across the universe (winner's-curse
  control), **Ledoit–Wolf**-style shrinkage covariance optional for weights; weights = fractional Kelly of the
  shrunk edge, long-only, capped at `MAX_WEIGHT`, summing ≤ 1 (rest is cash).
* `signal`: `buy` if calibrated `p_up ≥ BUY_P_UP` and net expected return > 0 and liquid; `avoid` if
  `p_up ≤ AVOID_P_UP` and net expected return < 0; else `hold`.  `confidence` 高/中/低 from |p_up−0.5|, σ and
  model agreement (components same sign).  `reason`: one Chinese sentence from the drivers/regime.

### `universe_bt.py`
```python
evaluate_universe(panel: dict[str, tuple[PriceData, WalkForward]], horizon: int, *, top_k=5,
                  commission_discount=..., slippage_bps=..., exec_mode='next_open') -> dict   # §6.5 schema
```
Aligns stocks by date; each day ranks by the same `rank_signal` as `recommend.py` (§9.1; using only that day's
forecasts), holds the top-k equal-weight for `horizon` days (overlapping-tranche or rebalance every h days —
pick one, document), net of costs, vs equal-weight universe and vs TAIEX if provided.  Report per-day
cross-sectional Spearman IC (mean, IR, t-stat), top-minus-bottom spread (of mean h-bar LOG returns), hit rate of picks, equity
curves, a per-stock accuracy table, the picks' mean sigma / market beta vs the universe's, a model-free top-k-by-sigma
benchmark and an explicit survivorship-bias note (§6.5).

## 5. Server (module B) — `backend/server.py`, `backend/jobs.py`

`python -m backend.server` (env `HOST` default 127.0.0.1, `PORT` default 8000) starts a
`ThreadingHTTPServer`: JSON API under `/api/*`, static files from `frontend/` (`/` → `index.html`,
correct MIME incl. `application/javascript; charset=utf-8`, `Cache-Control: no-cache` for html/js/css).
JSON encoder: numpy → python, **NaN/±inf → null**, dates → `YYYY-MM-DD`.  All responses UTF-8.
Errors: HTTP 4xx/5xx with `{"status":"error","error":{"code":"...","message":"繁中訊息"}}`; codes:
`bad_request`, `unknown_symbol`, `insufficient_history`, `data_unavailable`, `internal`.
Heavy work (backtest, recommendations, universe validation) runs as **jobs** (thread/process pool, ≤ CPU-1
workers, results cached in memory + disk under `data_cache/`, keyed by (params, last bar date)) and endpoints
return `{"status":"computing","progress":{"done":int,"total":int,"message":str}}` until
`{"status":"ready", ...}`; the frontend polls every ~1.5 s.  Identical concurrent requests share one job.
Cache forecasts per (symbol, last bar date) and walk-forwards per (symbol, last bar date, params) as pickles so the
single-stock backtest and the universe validation reuse them.  `Ctrl-C` shuts down cleanly.  Orchestration:
data → `quant.engine` → `evaluation.*` exactly through the signatures in §3/§4.

## 6. API & JSON schemas (binding for server, evaluation, quant, frontend)

All returns are **simple fractions** (0.012 = +1.2 %) unless a name ends in `_log`.  Dates `YYYY-MM-DD`.

### 6.0 Small endpoints
* `GET /api/status` → `{"status":"ready","server_time":ISO,"taipei_time":ISO,"market_open":bool,
  "last_trading_date":"YYYY-MM-DD","version":"1.0"}`
* `GET /api/search?q=台積` → `{"status":"ready","results":[{"code":"2330","name":"台積電","market":"TWSE","kind":"stock"}]}`
  (empty `q` → popular list = first 12 of the universe + 0050/0056/00878).
* `GET /api/history?symbol=2330&limit=750` → `{"status":"ready","symbol","name","market","kind","asof":"2026-10-22",
  "last_bar_complete":true,"stale":false,"bars":[{"d":"2026-10-22","o":2435,"h":2435,"l":2385,"c":2385,"v":27360404}],
  "dividends":[{"d":"2026-09-17","amt":6.0}]}`   (raw unadjusted OHLC, ascending, last `limit` bars)

### 6.1 `GET /api/forecast?symbol=2330`  (synchronous, ≤ ~8 s; cached per symbol+last bar date)
```json
{"status":"ready","symbol":"2330","name":"台積電","market":"TWSE","kind":"stock",
 "asof":"2026-10-22","asof_complete":true,"next_trading_day":"2026-10-23","last_close":2385.0,"chg_1d":-0.0145,
 "n_bars":6669,"short_history":false,
 "horizons":{
  "1": {"h":1,"days_label":"明日","p_up":0.52,"p_up_raw":0.55,"exp_return":0.0004,"median_return":0.0002,"sigma":0.016,
        "quantiles":{"0.05":-0.025,"0.25":-0.009,"0.5":0.0002,"0.75":0.0098,"0.95":0.026},
        "price_quantiles":{"0.05":2325.4,"0.25":..., "0.5":..., "0.75":..., "0.95":...},
        "signal":"hold","score":0.02,"net_exp_return":-0.0053,"confidence":"低","components":{"garch":{"p_up":0.5,"mu":0.0001},"hmm":{...},"dlm":{...},"naive":{...}}},
  "5": {...same keys...}, "20": {...}},
 "fan":{"days":[1,2,...,20],"levels":[0.05,0.25,0.5,0.75,0.95],"prices":[[...20 values per level...],...5 rows...],
        "returns":[[...],...]},
 "regime":{"labels":["低波動盤整","多頭穩定","高波動空頭"],"probs":[0.6,0.3,0.1],"current":"低波動盤整","confident":true,
           "transition":[[...3x3...]]},
 "vol":{"sigma_daily":0.016,"sigma_annual":0.25,"percentile":0.7,"regime_note":"..."},
 "risk":{"var95_1d":-0.025,"cvar95_1d":-0.034,"var95_20d":-0.11,"cvar95_20d":-0.15,"median_max_drawdown_20d":-0.05,"prob_loss_gt_10pct_20d":0.04},
 "models":{"weights":{"1":{"garch":0.3,"hmm":0.2,"dlm":0.3,"naive":0.2},"5":{...},"20":{...}},"names":{"garch":"GJR-GARCH-t","hmm":"隱馬可夫regime","dlm":"動態線性模型(貝氏)","naive":"隨機漫步基準"}},
 "drivers":[{"name":"12-1個月動能","key":"mom_12_1","value":0.31,"z":1.2,"contribution":0.0003}],
 "notes":["資料截至收盤…","預測以還原權息價計算…"],
 "disclaimer":"本工具為統計模型之研究用途，非投資建議…"}
```
`signal` ∈ `buy|hold|avoid`, `confidence` ∈ `高|中|低` are added by `recommend.annotate_forecast`.
`quant.engine.forecast_latest` produces everything except `name, market, kind, signal, score, net_exp_return,
confidence, short_history, next_trading_day, asof_complete, disclaimer, notes` (server fills those).
Semantics (audit, additive): `p_up` is `P(R>0 | the price moves)` — a flat close (an unchanged price: 6.8 % of next-day outcomes on average,
25 % for low-priced names) is neither up nor down, so the downside probability is NOT `1 − p_up` (§3.6); `mu_log` / `sigma_log` of every
horizon are the log-return moments of the pooled predictive (`sigma_log` is wider at Friday origins for h = 1, §3.10); `vol.regime_note`
says "判斷不明確" instead of naming a regime when the largest regime probability is < 0.6 (§3.2) and `regime.confident` (additive) is
then `false` (the UI should not present `regime.current` as "the" state); the extra block `engine`
`{version, refit_every, last_refit_row, state_window, has_market, placeholder_bars_dropped, p_up_event, mc}` is informational (§3.9;
`p_up_event = "up_given_move"` names the event of `p_up`, also in `WalkForward.config["p_up_event"]`).

### 6.2 Job envelope (all job endpoints)
`{"status":"computing","progress":{"done":12,"total":60,"message":"計算 2330 …"}}` |
`{"status":"ready", ...payload...}` | `{"status":"error","error":{...}}`.

### 6.3 `GET /api/recommendations?h=5&top=10`
```json
{"status":"ready","asof":"2026-10-22","asof_complete":true,"horizon":5,"computed_at":"ISO","universe_size":60,
 "n_scored":58,"market":{"p_up_1":0.51,"p_up_5":0.53,"regime":"多頭穩定","last_close":46041.4,"chg_1d":-0.011},
 "buys":[REC,...],"avoids":[REC,...],"all":[REC,...],
 "leaders":[REC,...],"laggards":[REC,...],"laggards_all_above_half":true,"n_buy_total":26,
 "rank_key":"rank_signal","rank_sigma_corr":-0.57,"universe_sigma_median":0.095,
 "portfolio":{"cash_weight":0.3,"n":5,"note":"…","suppressed_reason":null},
 "evidence":{"source":"none"},
 "notes":["…排名依相對強度…","…"]}
REC = {"rank":1,"code":"2330","name":"台積電","kind":"stock","last_close":2385.0,"chg_1d":-0.0145,
  "p_up":0.58,"exp_return":0.012,"net_exp_return":0.0063,"sigma":0.03,"score":0.21,"score_shrunk":0.12,
  "signal":"buy","confidence":"中","weight":0.08,"reason":"…",
  "p_up_by_h":{"1":0.53,"5":0.58,"20":0.6},"regime":"多頭穩定","avg_value_20d":6.5e10,"liquid":true,
  "rank_signal":0.17,"score_rank_pct":1.0,"lean":"bullish","edge_covers_cost":true,"low_vol":true,"edge_vs_baseline":0.026}
```
Ordering (§9.1, binding): `rank`, the order of `all`, `leaders`, `laggards`, `lean` and `score_rank_pct` come from
`rank_signal = mu_log/sigma_log` (descending; ties by `p_up`, then code) and never from `score` / `score_shrunk`.  `rank` is ONE
global rank (position in `all`, 1 = best) in every list, so a name has the same rank in `buys`, `avoids`, `leaders`, `laggards`
and `all`.  `buys` = the strict `signal=buy` names in that order (≤ top_k; `n_buy_total` counts all of them), `avoids` = the
`signal=avoid` names, weakest first (≤ top_k), `leaders` = top top_k with `p_up>0.5` (signal-independent), `laggards` = bottom
min(top_k, n_scored//2) (no `p_up<0.5` requirement; `laggards_all_above_half`), `all` = every scored name.
`score_rank_pct = (n−rank)/(n−1)` (0.5 when n = 1); `lean` = bullish if `score_rank_pct ≥ 0.8` and `p_up>0.5`, bearish if `≤ 0.2` and
`p_up<0.5`, else neutral (bearish is often empty: on real data `p_up<0.5` holds for no name on ~20 % of days at h = 5 / 20); `edge_covers_cost` = `net_exp_return>0`; `low_vol` = `sigma` below the median of the scored names (`rank_signal` is a
low-volatility-tilted ordering, see §9.1; `rank_sigma_corr` = today's Spearman(rank_signal, sigma));
`edge_vs_baseline` = `p_up − p_up of the naive component` (null when missing).  `weight` is the fractional Kelly weight of the
shrunk edge of `buy` names only (`score`/`score_shrunk` gate `signal`, `edge_covers_cost` and the weights): when James–Stein
retention `B == 0`, or no buy name has a positive shrunk net edge, NO weights are assigned (`portfolio.cash_weight = 1`,
`portfolio.n = 0`, `portfolio.suppressed_reason = "B0" | "no_positive_edge"`, note "個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移").
`evidence` is passed through from the server (`build_recommendations(..., evidence=None)` → `{"source":"none"}`);
`universe_bt.evidence_from_report(report, h, computed_at)` builds the §9(4) block from a universe report of the same horizon and
adds the optional after-cost keys `net_cagr` (validated top-k basket, net of costs), `ew_cagr` (equal-weight universe),
`sigma_cagr` (model-free top-k by sigma) and `beats_ew_after_costs`: the `notes` quote them, and a `moderate_edge` / `strong_edge`
whose basket does not beat the universe after costs (real h = 1: IC t = 3.6, basket −79 %) is called "not a buy recommendation".
The `notes` also say WHY the strict buy list is empty (every net return ≤ 0: costs exceed the expected return) or long (≥ max(5, 25 %)
of the scored names: the signature of a positive market drift, not of stock selection).
### 6.4 `GET /api/backtest?symbol=2330&h=1&thr=0.55&exec=next_open&disc=0.6&slip=5&years=5&sizing=threshold`
Envelope as §6.2; payload key `report`:
```json
{"symbol":"2330","name":"台積電","kind":"stock","horizon":1,
 "params":{"threshold":0.55,"exec":"next_open","commission_discount":0.6,"slippage_bps":5,"sizing":"threshold","n_trials":1,"roundtrip_cost":0.0057,"min_train":500,"refit_every":21},
 "period":{"start":"2021-10-25","end":"2026-10-21","n_forecasts":1240,"n_scored":1236,"n_tie_excluded":4},
 "direction":{"accuracy":0.536,"ci95":[0.508,0.564],"n":1236,"base_rate_up":0.55,
   "baselines":{"always_up":0.55,"prev_sign":0.49,"train_majority":0.55},"best_baseline":"always_up",
   "p_value_vs_best_baseline":0.31,"p_value_vs_coin":0.01,
   "by_confidence":[{"bucket":"[0.40,0.45)","lo":0.4,"hi":0.45,"n":20,"accuracy":0.55,"mean_p_up":0.43,"actual_up_rate":0.45}],
   "high_conf":{"threshold":0.6,"n":57,"accuracy":0.6,"p_value":0.1}},
 "probabilistic":{"brier":0.2501,"brier_baseline":0.2499,"brier_skill":-0.0008,"log_loss":0.69,"log_loss_baseline":0.69,
   "ece":0.02,"calibration":[{"p_mean":0.45,"freq":0.47,"n":120}],
   "log_loss_skill":-0.0006,"brier_baseline_oracle":0.2497,"brier_skill_oracle":-0.0016,"log_loss_baseline_oracle":0.6931,
   "log_loss_skill_oracle":-0.0004,"ece_noise_floor":0.05},
 "returns":{"ic":0.03,"ic_pvalue":0.3,"oos_r2":-0.002,"rmse_model":0.0161,"rmse_naive":0.0160,"dm_stat":0.4,"dm_pvalue":0.69,
   "coverage":{"0.50":0.51,"0.80":0.79,"0.90":0.89},"kupiec_pvalue":0.6,"pit_hist":[0.1,0.1,...10 bins],"pit_ks_pvalue":0.4,"pinball_loss":0.0051},
 "strategy":{"rule":"機率≥55%且扣成本後期望報酬>0 時持有，否則空手","stats":STATS,"benchmark":STATS,
   "sharpe_diff_ci95":[-0.4,0.6],
   "curve":{"dates":[...],"strategy":[1.0,...],"buyhold":[1.0,...],"position":[0,1,...],"drawdown_strategy":[...],"drawdown_buyhold":[...]}},
 "rolling":{"window":250,"dates":[...],"accuracy":[...],"base_rate":[...]},
 "verdict":{"level":"no_edge","title":"沒有統計上顯著的預測優勢","text":"…含具體數字…","bullets":["…"]},
 "notes":["已計入手續費…","未計入 …"]}
STATS = {"total_return":..,"cagr":..,"vol":..,"sharpe":..,"sortino":..,"max_dd":..,"calmar":..,"exposure":..,
         "turnover_per_year":..,"n_trades":..,"trade_hit_rate":..,"psr":..,"dsr":..,"sharpe_ci95":[..,..]}
```
Curves are downsampled to ≤ 600 points by the evaluation module (keep first/last).  `years` → test window
`min(years*245, n - min_train)` bars.
`probabilistic` (additive keys after `calibration` are optional for validators): `brier_baseline` / `brier_skill` /
`log_loss_baseline` / `log_loss_skill` are against the CAUSAL expanding-window base rate of the scored event known at each origin
(§9(a)); the `*_oracle` keys use the verification-sample up-rate (hindsight, the hardest reference, labelled as such in the UI);
`p_up_event` (`"up_given_move"` | `"up"`, from `WalkForward.config["p_up_event"]`) names the scored event and `n` the number of
rows scored: for `"up_given_move"` (the quant engine) flat closes are left out (`n == direction.n`), for `"up"` the block is scored on
every evaluated forecast with `y = 1[R>0]` (flat counts as not up, `n == period.n_forecasts`), while `direction.n` always counts
only non-flat days (`period.n_tie_excluded`); `ece_noise_floor` = about the 95th percentile of the ECE that an exactly calibrated
forecaster reaches by binning noise alone for this n, h and bin count (worst case = a constant forecast; overlapping h-day
outcomes make it grow with h: about 5.4 / 10 / 20 % at n = 1250 and h = 1 / 5 / 20), the ECE warning fires only above
`max(5 %, floor)`.  The Brier test of the verdict requires the model to beat BOTH references (the larger one-sided p-value counts).
`verdict.level` is capped at `weak_edge` when the probability legs contradict an edge claim (direction or Brier one-sided
p > 0.95, or both > 0.5: IC/MSE significance alone must not outvote evidence that the model is worse than the baseline); every
`no_edge` text (and the notes) carries the detectable-effect disclosure ("沒有偵測到不等於沒有"; minimum IC detectable with ~80 %
power, `metrics.MIN_DETECTABLE_IC`).  The two-sided IC p-value is shown as uncalibrated (history-derived forecasts have a null IC
centred below 0, about -0.04 / -0.09 / -0.18 at h = 1 / 5 / 20); the verdict only uses the one-sided positive p-value.

### 6.5 `GET /api/backtest/universe?h=5&top=5&exec=next_open&disc=0.6&slip=5` → envelope with payload key `report`:
```json
{"horizon":5,"top_k":5,"period":{"start":"…","end":"…","n_days":1100,"n_stocks":58},
 "ic":{"mean":0.012,"ir":0.2,"t":2.1,"pvalue":0.03,"series":{"dates":[…],"values":[…]}},
 "picks":{"hit_rate":0.53,"base_rate":0.52,"n":5400,"p_value":0.12,"mean_return":0.003},
 "top":STATS,"benchmark_equal_weight":STATS,"benchmark_taiex":STATS|null,"spread":{"mean":0.0004,"t":1.1},
 "benchmark_top_sigma":STATS|null,
 "basket":{"k":5,"mean_sigma":0.025,"universe_mean_sigma":0.046,"sigma_ratio":0.54,"mean_beta":0.42,"universe_mean_beta":0.84,
   "beta_market":"taiex"|"equal_weight","topk_overlap":0.84,"corr_rank_sigma":-0.56,"share_net_positive":0.12,
   "sigma_basket":{"mean_sigma":0.088,"mean_beta":1.42}},
 "exposure":{"mean_sigma_top":…,"mean_sigma_universe":…,"beta_top":…,"beta_universe":…,"benchmark_top_sigma":STATS},
 "rank_key":"rank_signal","only_buy_signals":false,"labels":{"top":"…","benchmark_equal_weight":"…","benchmark_top_sigma":"…","benchmark_taiex":"…"},
 "curve":{"dates":[…],"top":[…],"equal_weight":[…],"taiex":[…]|null,"top_sigma":[…]|null},
 "per_stock":[{"code":"2330","name":"台積電","accuracy":0.54,"brier_skill":-0.001,"brier_skill_oracle":-0.003,"ic":0.02,"n":1200}],
 "verdict":{"level":"weak_edge","title":"…","text":"…","bullets":[…]},
 "notes":["存活者偏差：…","…"]}
```
Additive keys of §9.1 (all optional for validators): the picks are ranked by `rank_signal` (`rank_key`; `only_buy_signals` tells
whether the strict buy gate was applied — default false, i.e. every name competes); `basket` = pick-weighted mean forecast
sigma and mean CAUSAL market beta (trailing 250-day OLS slope on TAIEX, equal-weight universe when TAIEX is missing →
`beta_market`) of the top-k at selection time vs the universe's, the day-to-day top-k overlap, the mean daily
Spearman(rank_signal, sigma), the share of name-days whose net expected return covers the round-trip cost and the same
sigma/beta of the sigma benchmark; `exposure` repeats the exposure numbers in a flat shape; `benchmark_top_sigma` = the
model-free "top-k by forecast sigma" basket through the same tranche / cost / execution simulator (`curve.top_sigma` its equity
curve); `labels` name every block; `spread` is the mean top-minus-bottom h-bar LOG return (no Jensen term for volatility sorts);
`per_stock[].brier_skill` is against the causal base rate of the stock's declared event (§6.4 `p_up_event`: flat closes left out for
`"up_given_move"`, counted as not up for `"up"`), `brier_skill_oracle` against the stock's test-sample up-rate on the same rows.  A verdict bullet warns when the IC / spread is significantly NEGATIVE (all tests are one-sided "better").  The picks hit-rate test
compares with the same-day universe up-rate and does NOT control for volatility (`P(up) = Φ(mean / sigma)` is higher for calm names, a
model-free lowest-sigma basket gets +1.9 / +3.0 / +4.4 pp on the real h = 1 / 5 / 20 panel against +2.65 / +2.83 / +3.42 pp for the
`rank_signal` basket): when `basket.sigma_ratio ≤ 0.8` the verdict text says that leg is not independent evidence and a reminder
bullet is appended (the level rule is unchanged).  `universe_bt.evidence_from_report(report, h, computed_at)` builds the picks
payload's `evidence` block (§9.4) from this report.
Costs/time: computing it needs a walk-forward for every universe stock (process pool, cached); the first
run can take minutes — progress must be reported per stock.

## 7. Frontend (module D) — `frontend/`
Vanilla ES2020 modules or plain scripts, **no external network resources** (no CDN, no web fonts — use
`font-family: "Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,sans-serif`).  Hand-written canvas
(or SVG) charts in `charts.js`: candlesticks + volume + MA, fan chart continuing the price chart, gauge/bar
charts, line charts with hover crosshair + tooltip, reliability diagram, histogram.  HiDPI-aware, resizes,
light/dark (CSS variables; follow `prefers-color-scheme` + manual toggle remembered in localStorage).
**Colour semantics: 紅 = 漲/偏多, 綠 = 跌/偏空** (Taiwan).  Colour is never the only cue (also ▲/▼ and text).
All UI text Traditional Chinese.  Responsive (phone → desktop), keyboard accessible, no horizontal scroll.

Views (hash routes `#/stock/2330`, `#/picks`, `#/backtest/2330`, `#/method`):
1. **個股預測**: search box with autocomplete (`/api/search`, debounced, ↑↓ Enter, shows code+name+market);
   header (name, code, last close, 1-day change, asof, market-open/closed chip, stale warning); price chart
   (≈120 bars candles + volume, MA20/60) with the **fan chart** extended 20 trading days; three horizon cards
   (明日 / 未來5日 / 未來20日): large 漲/跌機率 with gauge, expected return, 90 %/50 % intervals, signal chip
   (偏多·買進 / 中性·觀望 / 偏空·避開), confidence, tiny component table; regime probability bars; volatility
   meter (annualised, percentile); risk (VaR/CVaR); model-weights stacked bars; drivers table; **歷史驗證摘要**
   card that lazily loads `/api/backtest` (h=1 & 5) and shows accuracy vs baseline, Brier skill, verdict
   chip — with a link to the full backtest; prominent but non-obnoxious disclaimer.
2. **今日推薦**: horizon toggle (明日/一週/一個月), asof/closed-market banner, progress bar while
   computing, "買進候選" table/cards (rank, name, price, 1-day chg, P(漲), 淨預期報酬, 建議權重, 信心,
   reason), "建議避開" list, full sortable table, suggested portfolio summary (cash weight), click → stock page;
   collapsible **推薦整體歷史驗證** panel loading `/api/backtest/universe` on demand (button, progress).
3. **歷史驗證**: controls (stock via search, horizon, probability threshold slider, execution mode,
   commission discount, years); on run: verdict banner (use `verdict.level` colour/icon, text), KPI cards
   (方向準確率 vs 最佳基準 with CI + p-value, Brier skill, IC, 策略夏普 vs 買進持有, 最大回撤), equity-curve
   chart (strategy vs buy&hold, position band, drawdown subplot), reliability diagram (calibration), rolling
   accuracy vs base rate, PIT histogram, coverage table, confidence-bucket table, params + costs, plain-language
   tooltips (ⓘ) explaining each metric.
4. **方法說明**: static content explaining the pipeline with equations (unicode/HTML sub/sup, no external
   libs): data & adjustment, GJR-GARCH-t, HMM filtering, DLM discount-factor Kalman regression, dynamic
   model averaging, calibration, filtered historical simulation, empirical-Bayes shrinkage, fractional Kelly,
   walk-forward testing, PSR/DSR, limits ("為何多數預測沒有優勢"), disclaimer.
Robustness: loading/skeleton states, retry on network error, handles `status:"computing"` polling with cancel on
route change, handles `null` numbers (show "—"), never throws on partial payloads.

### Frontend testing without a browser
No node/browser works here.  Use the V8 engine from the Python package **mini-racer** (installed in the
venv (`python3 -m venv .venv && .venv/bin/pip install mini-racer pillow weasyprint pymupdf numpy scipy`),
`py_mini_racer.MiniRacer`) to (a) syntax-check every JS file (`new Function(src)`), (b) run pure logic
(formatters, schema adapters, route parsing) and the chart code against a minimal fake `document`/canvas-2D
recorder.  Keep rendering logic separated from DOM so it is testable.  Ship a fixtures set built from REAL
API output (or the schemas above) and test that every view renders without exceptions on: full payload, payload
with nulls, `computing`, `error`.

## 8. Definition of done (every builder)
* All public functions/endpoints above exist with exactly these names/shapes.
* Tests you wrote pass (`python -m unittest discover -s tests -p "test_<yours>*.py"`); report counts.
* No look-ahead (A, C), no network access in unit tests except clearly marked, skippable smoke tests.
* Final report (≤ 40 lines): files created, public API as implemented, deviations from this spec (with reason),
  test results, measured timings, known limitations / things the integrator must check.

## 9. Addendum (post-integration, 2026-10-23) — recommender semantics, evidence, audit rules

Real-data findings: the models match the base rate everywhere (no demonstrable edge); at h=1/5 the strict
`buy`/`avoid` lists are empty; James–Stein retention B can be 0.00 so every `score_shrunk` is identical and
`rank` degenerates to tie-breaks; at h=20 almost every name is `buy` merely because all drifts are positive.
Binding changes to §6.3 (all additive; keep every existing key):
1. **(revised 2026-10-23, supersedes the earlier "raw cost-adjusted score" rule; reproduced on real data: Spearman(score, sigma)
   = 0.98 / 0.94 / 0.72 at h=1/5/20 while Spearman(score, p_up) = 0.15 / 0.03 / 0.04 — the cost-adjusted score ranks
   VOLATILITY, because at h=1 every `net = E[r]-cost` is negative so net/sigma is largest for the most volatile names, and at
   h=5/20 `E[e^R-1] ≈ mu_log + sigma²/2` adds a convexity term that grows with sigma.)**  Separate the RELATIVE ranking from the
   TRADABILITY test:
   * New per-REC key `rank_signal` = `mu_log / sigma_log` of the horizon forecast (log-return information ratio; fall back to
     `Φ⁻¹(p_up)` if either is missing).  It contains no trading cost and no convexity term.  **HONESTY CAVEAT (audit P-06, verified on the
     1250-day real panel):** because forecast means are heavily shrunk toward a common drift, `rank_signal` is a LOW-VOLATILITY-TILTED
     relative ordering (mean daily Spearman(rank_signal, sigma) = -0.36 / -0.56 / -0.62 at h=1/5/20; top-10 overlap day-to-day
     71 / 90 / 95 %), i.e. the mirror image of the old volatility sort — it removes the high-vol artefact but does NOT prove stock-picking
     skill.  Therefore: every leader/laggard row must show `sigma` (UI adds a 「低波動」 tag when sigma < universe median), the wording is
     「相對強度（預期報酬/風險）」 and never 「風險調整報酬」 or advice, and the universe report must publish the basket's mean sigma/beta and
     the day-to-day top-k overlap next to the universe's.  Do NOT switch the key to a baseline-relative one such as
     `(mu - mu_naive)/sigma` until it has been evaluated on a stock-wise holdout (it looked promising in-sample, IC t = 4.1 / 2.7 / 2.5,
     but it was one of 5 keys tried after looking at the data, on a survivorship-biased universe).  `rank`, the order of `all`,
     `leaders`, `laggards`, `lean` and `score_rank_pct` are ALL based on `rank_signal` (descending; ties by `p_up`, then code) —
     never on `score`.  Add a regression test: with all `net < 0`, the ordering must NOT equal the sigma ordering
     (|Spearman(rank_signal, sigma)| < 0.9 on a synthetic universe where p_up is independent of sigma).
   * `score` / `score_shrunk` (cost-adjusted, James–Stein) are kept ONLY for the strict `signal` gate (`buy`/`avoid`),
     `edge_covers_cost` and the Kelly weights.  A name with `net_exp_return <= 0` has zero tradable edge and contributes
     nothing to weights; a negative `-cost/sigma` must never take part in any cross-sectional ordering.
   * `universe_bt.py` MUST rank by the same `rank_signal` (the validated "top-k" portfolio must be the same thing the picks
     page shows).  In addition the universe report must state, for the top-k basket vs the universe: mean sigma, mean market
     beta, and the return of two model-free benchmarks (a) top-k by sigma, (b) equal-weight of the universe — so that any
     apparent outperformance that merely reflects high-volatility/high-beta exposure is visible and labelled.
2. Each REC gains `lean`: `"bullish" | "bearish" | "neutral"` — relative, from the raw cross-sectional rank:
   top quintile of `rank_signal` AND `p_up > 0.5` → bullish; bottom quintile AND `p_up < 0.5` → bearish; else neutral. Also
   `rank_signal`, `score_rank_pct` (0..1, 1 = best by `rank_signal`) and `edge_covers_cost` (bool: `net_exp_return > 0`).
3. The response gains `leaders` (top `top_k` by `rank_signal` with `p_up>0.5`, signal-independent) and `laggards` (bottom
   `top_k` by `rank_signal`, WITHOUT requiring `p_up<0.5` — on real data p_up>0.5 for 60/60 names at h=5/20 and the p_up<0.5 set is
   empty on ~20 % of days), both lists of REC; add `laggards_all_above_half` (bool) so the UI can label them 「相對最弱（上漲機率仍高於 50%）」.
   `lean=bearish` keeps the stricter rule (bottom quintile AND `p_up<0.5`), so it is often empty — that is expected.  `buys`/`avoids` keep
   their strict meaning.
4. The response gains `evidence`: `{"horizon":h,"source":"universe_validation"|"none","verdict_level":"no_edge|…"|null,
   "ic":..,"ic_pvalue":..,"hit_rate":..,"base_rate":..,"computed_at":..}`; the SERVER fills it from the cached
   universe validation for the same horizon when it exists (never triggers the heavy job), else `source:"none"`.  Additive optional
   keys (use `universe_bt.evidence_from_report`): `net_cagr`, `ew_cagr`, `sigma_cagr`, `beats_ew_after_costs`; the UI must not
   unlock buy-advice wording when `beats_ew_after_costs` is `false` even if `verdict_level` is `moderate_edge` or better.
5. When `B == 0` or no name has positive shrunk net edge: no weights are assigned (100 % cash suggestion) and a note says
   "個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移".
6. UI wording: `buy` → 「偏多候選」, `avoid` → 「偏空／避開」, never 「建議買進」 unless evidence level ≥ `moderate_edge` (and
   `evidence.beats_ew_after_costs` is not `false`, item 4).
   The picks page always shows leaders/laggards, plus a banner built from `evidence` (amber when `no_edge` or when
   `beats_ew_after_costs` is `false`).

Audit rules for statistics (binding on evaluation): (a) baselines must be CAUSAL — climatology/base rate for Brier
skill, log-loss skill and `train_majority` must be the expanding-window frequency known at each forecast origin
(matured outcomes only); report the oracle test-sample baseline separately and label it as such (headline numbers causal,
`*_oracle` keys hindsight).  Refinement (audit S-01, verified by simulation): the causal reference is a NOISY constant, so any
forecaster that merely knows the base rate "beats" it (one-sided Brier Diebold–Mariano rejections at 5 %: about 13–15 / 25–27 /
20–26 % at h = 1 / 5 / 20 for a zero-information forecaster) while the oracle reference is size-correct but hindsight; therefore the
verdict's Brier leg must beat BOTH (it uses the larger p-value), never the causal reference alone; (b) inference on
overlapping h-day outcomes must account for overlap (effective sample size ≈ n/h or HAC / block bootstrap) — p-values
for accuracy, IC, coverage and PIT-uniformity must not treat overlapping observations as independent; (c) cross-sectional
pooling (picks hit-rate) must account for same-day correlation (cluster by date).
