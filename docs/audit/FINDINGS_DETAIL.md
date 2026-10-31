# Findings in detail (description, repro, evidence, proposed fix, verifier reasoning)

## C-01 [high] Brier/log-loss skill and the Brier DM test use a hindsight test-sample climatology (violates DESIGN 9(a)); 'negative Brier skill everywhere' is largely this reference

- owner: evaluation  
- location: backend/evaluation/report.py:173-174 (clim_const -> metrics.probabilistic_summary), :222 (DM test); backend/evaluation/universe_bt.py:259-261 (per_stock Brier skill with b = up.mean())  
- reviewer: review:causal

**Description.** report.analyse sets p_ref = the full-sample up-rate of the scored rows (clim_const) and uses it for brier_baseline, brier_skill, log_loss_baseline and again as the reference of the Diebold-Mariano Brier test that feeds the verdict. universe_bt.per_stock_table does the same (b = up.mean()). DESIGN 9(a) (binding) requires the expanding-window frequency known at each origin (matured outcomes only) with the oracle baseline reported separately. The causal reference already exists (metrics.expanding_up_rate, used only for the train_majority accuracy baseline). Over the trending 2021-08..2026-10 window the hindsight rate (e.g. 0.571 realised up-rate at h=20 vs 0.5465 mean forecast) is not knowable in real time, so the model's skill is pushed negative. The direction of the bias is conservative (model looks worse) and the note text says 含事後資訊, but the displayed number and the verdict test are not the ones the spec demands. My replication of the report's brier_skill is exact (2330 h=20: report -0.027456, mine -0.027456).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 ties_bias.py   # per-horizon mean BSS under current / causal / ties-kept variants (writes ties_bias.pkl) ; python3 verdict_ref_effect.py   # verdict + DM-p effect over 62 stocks x 3 horizons. Observed numbers above; expected per DESIGN 9(a): Brier/log-loss/DM reference = metrics.expanding_up_rate(R,h)[scored rows].

**Evidence.** 62 cached walk-forwards (60 universe + 0050 + 6488). Mean Brier skill, current (oracle reference) vs causal expanding reference (ties dropped as today): h=1 -0.00312 vs -0.00118 (share<0 0.87 vs 0.69); h=5 -0.00604 vs -0.00041 (0.98 vs 0.66); h=20 -0.02352 vs -0.00009 (0.97 vs 0.61). Examples h=20: 2330 -0.0275 vs -0.0163, 0050 -0.0133 vs -0.0083, 2412 -0.0381 vs -0.0206. One-sided Brier-DM p median 0.877->0.759 (h=1), 0.862->0.726 (h=5), 0.824->0.677 (h=20); share p<0.05 (model better) 0 -> 1.6%/6.5%/6.5%; share p>0.95 (model worse) 22.6%->21.0% (h=1), 9.7%->12.9% (h=5), 9.7%->17.7% (h=20). Verdict changes in 1 of 186 reports (2890 h=20 no_edge -> weak_edge, causal DM p=0.026). So the causal reference makes the AVERAGE skill zero but widens the tails; the verdict text 'Brier 技能分數 -2.4% (正值才代表優於只猜整體上漲比例)' overstates underperformance.

**Proposed fix.** In report.analyse pass p_ref = M.expanding_up_rate(R, h)[Sx] to probabilistic_summary and to the Brier dm_test; expose the oracle values only as extra keys (brier_baseline_oracle, brier_skill_oracle) with an explicit label; same change in universe_bt.per_stock_table (use the causal rate per row, not b = up.mean()) and in the verdict/notes wording.

**Verifier:** real=True severity=medium  
Real, but medium rather than high. DESIGN 9(a) is binding and says Brier skill, log-loss skill and train_majority must use the expanding-window frequency known at each origin, with the oracle baseline reported separately and labelled. report.analyse (lines 171-174) builds clim_const as the constant full-sample up-rate and passes it to probabilistic_summary, whose docstring says p_ref is the causal climatology. The same constant is the reference for the Brier Diebold-Mariano test at line 222, which feeds ev["p"]["brier"] and the verdict. universe_bt.per_stock_table (lines 259-261) uses b = up.mean() as well. The causal metrics.expanding_up_rate exists and is used only for the train_majority baseline. I re-ran this independently on 25 of the 63 cached walk-forwards (not the reviewer's 62), computing the causal Brier skill with M.expanding_up_rate(R,h)[scored rows]. Mean Brier skill, current vs causal: h=1 -0.00154 vs +0.00014; h=5 -0.00428 vs -0.00088; h=20 -0.01717 vs -0.00226. The share of stocks with skill below 0 drops from 0.80/0.96/0.92 to 0.60/0.68/0.68. This matches the reviewer's direction and magnitude, so "negative skill everywhere" is largely the hindsight reference. Severity is medium because the oracle choice is not hidden: make_notes line 425 says 含事後資訊 and 屬較嚴格的比較, and the bias makes the model look worse, not better. The reviewer found the verdict changes in only 1 of 186 reports. The real problem is that the displayed skill and the DM test use the spec-forbidden reference. No causal figure is shown, and the verdict text at report.py:385 ('正值才代表優於只猜整體上漲比例') overstates underperformance. The project's headline finding, 'slightly negative Brier skill everywhere', is partly an artifact. A causal reference gives roughly zero average skill, which still means no demonstrable edge.

**Verifier refined fix.** 1) backend/evaluation/report.py analyse: after `clim = M.expanding_up_rate(R, h)`, set `clim_causal = clim[Sx]` and call `M.probabilistic_summary(p, up.astype(float), clim_causal)`. In the Brier DM test use `M.dm_test((clim_causal - up) ** 2, (p - up) ** 2, h)`. The existing `clim_const` stays only as the oracle reference. Add extra keys to out["probabilistic"]: brier_baseline_oracle = M.brier_score(clim_const, up), brier_skill_oracle = 1 - brier/brier_baseline_oracle, log_loss_baseline_oracle = M.log_loss(clim_const, up). Add the three keys to the schema at report.py ~498 as "num?". Optionally also report the oracle DM p-value in ev. 2) Update the make_notes line ~425 and the verdict text at lines ~339 and ~384-386. The new wording should say the baseline is the base rate known at each forecast date (matured outcomes only), and that the oracle skill is shown separately. 3) universe_bt.per_stock_table: compute the causal reference per row, `ref = M.expanding_up_rate(pn.R[:, j], pn.h)[sc]`, then `brier_skill = 1 - brier(p, up) / brier(ref, up)`, guarded by `brier(ref, up) > 0`. This needs the h-specific R column, which is already in the panel. Keep the oracle value as an optional extra key if wanted, and add it to the universe_bt schema at ~504. 4) Frontend (view_backtest.js:194-195, view_stock.js:416, view_picks.js:274): the label can stay, but the baseline caption should say the baseline is causal, and the oracle value can be shown as a secondary line. Side effects: evaluation tests that assert brier_baseline == up.mean() Brier, or that check per_stock brier_skill against b*(1-b), must be updated. Verdict thresholds may shift slightly (about 1 in 186 verdicts changes, 2890 h=20 from no_edge to weak_edge). Any frontend test that checks the note text or the schema key set must accept the new keys. The server contract in DESIGN section 6.4 should be extended with the oracle keys."


## C-02 [medium] Evaluation drops R==0 rows (outcome-dependent selection) so it scores P(up | not flat), while the calibrator/model target P(R>0) with ties counted as 'not up'

- owner: evaluation  
- location: backend/evaluation/report.py:155-156 (tie = R[E]==0; Sx = E[~tie]); backend/quant/engine.py:155 (up = Rc>0); backend/evaluation/universe_bt.py:211-216 and 255-261 (R != 0 filters); backend/evaluation/metrics.py:172 (prev_sign pred for tr==0)  
- reviewer: review:causal

**Description.** analyse() removes every row with realised R == 0 (Sx = E[~tie]) before accuracy, base rate, Brier, ECE and reliability. At h=1 unchanged closes are frequent (tick size vs volatility), so the evaluation base rate is inflated by 1/(1-tie rate) relative to the event the engine is trained on: engine._Ctx.up = (Rc > 0) counts a tie as 'not up' (calibrator target) and the continuous predictive puts no mass at 0. The three places use three different definitions of the 'up' event, so p_up cannot be calibrated under any single reading. The prev_sign baseline is additionally forced to 'down' on flat previous days (23% of 2412 rows; effect on its accuracy 0.4177 vs 0.4097, negligible).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 ties_bias.py ; python3 prev_sign_ties.py

**Evidence.** h=1, 62 stocks: tie rate mean 6.8% (median 6.0%), max 25.1% (2412), 19.9% (3045), 12.8% (2883), 12.1% (2887); h=5 mean 2.7%, h=20 1.2%. Mean p_up on scored rows 0.4983 vs realised up-rate with ties dropped 0.5127 (bias -1.44pp, 77% of stocks under-forecast) but vs up-rate with ties=not up 0.4776 (bias +2.06pp). Brier skill (ties kept, causal reference) mean +0.0011 (50% of stocks negative) vs the reported -0.0031 (87% negative); h=20 +0.0029 vs -0.0235. The same discreteness ruins PIT uniformity tests (see C-10).

**Proposed fix.** Score the event y = 1[R>0] on ALL evaluated rows (ties count as not-up, consistent with the calibrator) or move to a 3-way outcome {up, flat, down}; report the tie rate; if ties must be excluded, exclude them from the calibrator's training and make the model predict P(R>0 | R != 0). Make prev_sign abstain on flat previous days.

**Verifier:** real=True severity=low  
The mismatch is real and I reproduced it. report.py:155-156 drops every R==0 row (Sx = E[~tie]) and scores p_up on the rest. The engine's calibrator target is up = (Rc > 0) (engine.py:155), so a flat day counts as not-up. The model therefore forecasts P(R>0), and the evaluation scores it against P(R>0 | R != 0). That event is larger by a factor 1/(1-tie rate).

The size is small. At h=1 the tie rate averages 6.8% (max 25% for 2412, 20% for 3045), at h=5 2.7%, at h=20 1.2%. The base-rate bias on reliability and ECE is about -1.4pp at h=1 against the dropped-ties event and +2.1pp against the ties-as-not-up event. The exclusion is documented in DESIGN.md section 4 ("rows with R==0 excluded and counted"). The UI reports n_tie_excluded (frontend/view_backtest.js:137). Nothing is hidden. The omission is that the dropped event differs from the model's target, and the docs do not say so.

It barely moves the verdict. The finding's headline contrast (+0.0011 vs -0.0031 Brier skill) is mostly the oracle-versus-causal climatology reference, which is a separate issue (DESIGN section 9(a)). Isolating ties, with the oracle reference in both cases, gives -0.0037 (ties kept) against -0.0031 (ties dropped) at h=1, and -0.0054 against -0.0060 at h=5. That is a few thousandths in the same direction, so "no edge" holds either way. Direction accuracy among non-flat days is well defined, and the prev_sign flat-day effect is negligible (0.4177 vs 0.4097, from the reviewer's own measurement). Only the calibration and reliability reading is slightly biased, and it is worst for high-tie stocks. That makes it a low-severity consistency defect, not a medium one.

**Verifier refined fix.** Minimal change in backend/evaluation/report.py analyse(): score the probabilistic block on all evaluated rows with y = 1[R>0], so ties count as not-up, matching the calibrator and the continuous predictive. Concretely, for the Brier, log-loss, reliability and ECE block, the rolling accuracy and base-rate block, and the Brier DM test, replace Sx with E and use up = R[E] > 0; keep p = p_all[E]. Keep n_tie_excluded as a reported tie rate only. Alternatively keep the exclusion for the direction block (accuracy vs baselines is well defined among non-flat days) and use E only for probabilistic_summary, ev["brier"], and the rolling and reliability output. Mirror the same change in universe_bt.py lines 211-216 and 255-261 for the Brier skill; the picks_stats hit-rate filter can stay. Optional: make prev_sign abstain (valid=False) when tr == 0 in metrics.trailing_sign_prediction, line 172, e.g. valid[h:] = ok & (tr != 0). Side effects: tests that assert n_scored, n_tie_excluded, Brier or ECE values on tied data in tests/test_evaluation.py and the universe tests will change. The DESIGN.md section 4 sentence (R==0 excluded) must be amended. The frontend view_backtest.js nS/nT text should say that ties are scored as not-up for calibration. Keep this separate from the causal-reference change (9(a)), which dominates the Brier-skill shift.


## C-03 [medium] h=1 p_up is biased upward by about 0.27 x tie-rate (mean +2.1pp, +7pp for 2412/3045): continuous predictive has no atom at 0 and the calibrator removes only ~38%

- owner: quant  
- location: backend/quant/engine.py:328-331 (p_raw from the continuous pool; Cal.recalibrate), backend/quant/calibration.py:38-89  
- reviewer: review:causal

**Description.** The ensemble predictive is a continuous mixture, so P(R>0) + P(R<0) = 1, but realised daily returns have an atom at exactly 0 (price unchanged). With y = 1[R>0] (the calibrator's own target) the raw ensemble over-forecasts by about 0.44 x tie-rate; the Bayesian logistic recalibrator (sigma_a = 0.12, 1/h tempering) removes only ~38% of it. The user-visible 上漲機率 is therefore overstated for low-priced / large-tick names (telecoms, banks).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 ties_bias.py (columns mean_p_all vs up_all); bias_raw_vs_cal snippet: see slope numbers above (p_up_raw and p_up arrays of cache_causal/wf/*.pkl.gz).

**Evidence.** 62 stocks, h=1, target P(R>0) with ties = not up: mean bias p_up_raw -> p_up: +3.33pp -> +2.06pp; slope on tie rate 0.438 -> 0.270 (theory 0.5, R2 0.44); 11 of 62 stocks above +3pp; 2412: mean p_up 0.474 vs realised up-frequency 0.403 (tie rate 25.1%, +7.1pp); 3045: 0.4815 vs 0.4171 (+6.4pp). No comparable tie effect at h=5/20 (slope not significant).

**Proposed fix.** Model the flat mass explicitly (e.g. tick-aware discretised predictive: P(R>0) = 1 - F(+tick/2), P(R<0) = F(-tick/2)) or condition the target on R != 0 everywhere; at minimum allow the calibrator intercept enough prior width at h=1 and show P(flat) in the UI so that 漲/跌 are not presented as complementary.

**Verifier:** real=True severity=low  
The numbers reproduce, but the cause is a target-definition mismatch rather than a causal or look-ahead defect, and the size is small. The ensemble predictive is continuous, so p_up_raw is effectively P(up | price moves). The calibrator is trained on y = 1[Rc>0] (engine.py:155), which counts a flat close (ties) as "down". At h=1 it therefore pulls p_up toward P(R>0 including ties), and that pull is weakest where ties are common. The evaluation module drops ties: report.py:155 excludes R==0, and metrics.expanding_up_rate uses non-zero outcomes only. So the calibrator and the evaluation disagree about the target. The Platt prior is too tight (sigma_a=0.12) to fully absorb the offset. Measured against the strict-up target, the displayed p_up is high by +2.05pp on average, +7.1pp for 2412 and +6.4pp for 3045. Measured against the target the evaluation uses (up given the price moved), the same p_up is slightly low (-1.4pp on average, -6.3pp for 2412), and the raw ensemble is almost unbiased (-0.16pp). In other words, the calibrator adds the bias relative to the evaluation target. Practical impact is limited. The model has no demonstrable edge and p_up sits near 0.5. The mean error is about 2pp, and only 11 of 64 stocks exceed 3pp, mostly low-priced, large-tick names. At h=5 and h=20 the tie rate is low and there is no real tie bias, so only h=1 is affected. The effect is a small, definition-dependent inconsistency, and also a presentation issue, because the gauge label shows 下跌機率 = 1 - p (charts.js:703). I rate it low, not medium. The proposed tick-aware discretised predictive is overkill.

**Verifier refined fix.** Make the calibrator target consistent with the evaluation's "ties excluded" convention instead of modelling the atom.

File and change: backend/quant/engine.py:155. Change `self.up = np.where(np.isfinite(self.Rc), (self.Rc > 0.0).astype(float), np.nan)` so that up is NaN wherever the raw log return is exactly 0 (`self.Rraw[a] == 0.0`, or `Rc == 0`).

Why this works: calibration.recalibrate already skips NaN targets via its `ok` mask. p_up then means P(up | price moved), which matches both the continuous ensemble and the evaluation. Measured with the raw ensemble, the remaining bias is -0.16pp, versus -1.4pp now.

Optional UI addition: show the empirical flat-close frequency for large-tick names, and avoid presenting 下跌 as exactly 1 - 上漲 (frontend/charts.js:703 a11y label).

Side effects:
- Cached walk-forwards are invalidated, so the cache key or version must change.
- tests/test_quant*.py and server fixtures that pin exact calibrated p_up values may shift slightly.
- Universe and backtest numbers change slightly: p_up moves up by about 1 to 2pp on average at h=1, and up to about 6pp for 2412 and 3045.
- The tie-excluded evaluation code needs no change.


## C-04 [medium] A refresh deletes a valid cached bar when Yahoo serves a null row for the last completed day (history regresses by one bar, stale=False)

- owner: data  
- location: backend/data.py:700-709 (_merge), :675-697 (_consistent_overlap, tolerance 2 at :687), :738-743 (full path in _refresh); backend/jobs.py:1098 (asof_ref = most common date) and 1110-1111 (skip text)  
- reviewer: review:causal

**Description.** Yahoo intermittently returns an all-null OHLC row for the most recent completed session. parse_yahoo_chart drops null rows; _merge then replaces everything in the cache from the first date of the incremental window by the new data, and _consistent_overlap explicitly tolerates up to 2 'vanished' old bars (in_window - len(common) > 2). The full re-download path (record older than 30 days) replaces the whole record. Result: a previously final, valid last bar is removed. Downstream: /api/forecast reports asof one day too old with stale=False, asof_complete=True and next_trading_day equal to a date already in the past; the recommendations job takes the most common last date as asof_ref and silently skips every name with a different date -- including names that are CORRECT and newer (message says 'data only until X, not synchronised') -- so if half of the universe regressed the up-to-date half is dropped. The bar heals only after Yahoo serves the row again and the 10-minute refresh throttle elapses.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 regress_repro.py   # deterministic, offline ; python3 regress_forecast_payload.py   # what /api/forecast returns afterwards ; python3 t36_regress.py   # live Yahoo (temp cache dir); raw reply check: see the printed rows for 0050.TW in the session log (all-null 2026-10-22 row)

**Evidence.** LIVE today (2026-10-23 11:30 CST): cache copy of 0050 had 4355 bars ending 2026-10-22 (fetched 2026-10-22T22:01:54); one ordinary get_price_data('0050') call refreshed it to 4354 bars ending 2026-10-21, stale=False, fetched_at 2026-10-23T11:30:28. Raw Yahoo reply for 0050.TW at that moment: 2026-10-22 [None x5] and 2026-10-23 [None x5] (2330.TW, 2409.TW, 2014.TW had valid 10-22 rows). 48 of the 113 cached series (fetched 2026-10-23 00:30-00:48) end on 2026-10-21 while all 65 fetched on 10-22 evening end on 10-22. Offline deterministic repro (Yahoo reply rebuilt from the cache with the 10-22 row nulled): incremental path 4355 -> 4354, full path 4355 -> 4354, healthy reply restores 4355; forecast payload shows asof 2026-10-21, next_trading_day 2026-10-22, stale False, n_bars 4354.

**Proposed fix.** Make refresh monotone: never drop an old FINAL bar that is missing from the new reply (union semantics keyed by date, new wins only where it has non-null OHLC); treat all-null rows as 'not yet published'; reject a refreshed record whose last date is earlier than the cached last date unless that cached bar was provisional; in _run_recommendations pick asof_ref as the MAX date supported by a quorum and treat older names (not newer ones) as laggards.

**Verifier:** real=True severity=medium  
Confirmed by reading the code and re-running the offline repro. _consistent_overlap tolerates up to 2 vanished old bars (data.py:687). _merge (700-709) replaces everything from the new window's first date onward. The full path (738-743) replaces the whole record. Nothing checks that the refreshed last date is not older than the cached last date when the cached last bar was final. So a Yahoo reply with an all-null row for the last completed day, which parse_yahoo_chart drops, deletes a valid final bar and returns stale=False. The null-row behaviour is real: DESIGN.md section 2 says stocks have a handful of null rows, and the cache shows 48 of 113 series fetched 2026-10-23T00:xx ending on 10-21 while the 65 fetched on 10-22 evening end on 10-22. In jobs.py:1098 the asof_ref is the most common last date, so if most names regressed the correct, newer names get skipped with 'data only until X, not synchronised'. Severity is medium rather than high for three reasons. The effect is a one-day-old but honestly dated forecast, not look-ahead and not corrupted values. It self-heals on the next refresh once Yahoo serves the row (verified: the healthy reply restores 4355 bars). The trigger is an intermittent upstream glitch. Today's live Yahoo reply for 0050.TW, 2330.TW and 2317.TW is healthy, with 10-22 rows present. I therefore could not re-trigger the live glitch, but the deterministic repro covers the logic.

**Verifier refined fix.** Minimal fix in backend/data.py, so the data layer is monotone. (1) In _refresh, after computing the candidate record (incremental merge or full download) and before _save_rec, add a guard. If rec is not None and _bar_final_at_fetch(rec) and new["d"][-1] < rec["d"][-1], re-attach the cached tail: for each old bar with date > new["d"][-1], append o/h/l/c/vol unchanged. For adj, append old["adj"][tail] * (new_adj_at_last_common / old_adj_at_last_common), where the last common date is new["d"][-1] if it exists in old, otherwise skip the rescale. In the incremental path _merge already keeps old adj, so the simple form is enough: in _merge, after concatenating, if old["d"][-1] > new["d"][-1] and old's last bar is final, append old[...][searchsorted(old['d'], new['d'][-1], 'right'):]. Keep the stored fetched_at as the new time so the 10-minute throttle still applies. (2) In _consistent_overlap, leave the tolerance of 2 unchanged, because the guard in (1) now covers the last-bar loss. Do not add a hard 'reject and re-download' rule: a full re-download would hit the same null row. (3) Optional hardening in backend/jobs.py:1098. Choose asof_ref as the maximum date held by at least about 30% of the loaded names, or by at least a few names, instead of Counter.most_common, so that regressed laggards are skipped and not the correct newer names. Side effects: tests in tests/test_data.py that assert the replace-from-first-date semantics of _merge must be checked, since a test that deliberately truncates the tail would change. A genuine delisting or suspension is unaffected because the old bars remain true history. A bar that was provisional (intraday, not final at fetch) must still be replaced, which is why the guard is conditioned on _bar_final_at_fetch(rec). No change to the API schema or to the quant, evaluation or frontend modules.


## C-05 [medium] Cache keys do not cover every model input / parameter: price_signature ignores open/high/low/volume; code fingerprint ignores config.py, types.py, data.py; horizons/min_train/refit_every are not in the key

- owner: server  
- location: backend/data.py:876-882 (price_signature); backend/jobs.py:194-221 (code_fingerprint), :870-873 (_forecast_key), :979-982 (_wf_key), :1025-1026 (_result_key)  
- reviewer: review:causal

**Description.** DESIGN/README claim stale or differently-parameterised results can never be served. data.price_signature hashes only symbol|n|last date|close|adjclose, yet features use high/low (Parkinson vol feeding the HMM and rvol_ratio) and volume (vol_z). jobs.code_fingerprint hashes (name,size,mtime) of backend/quant/*.py (and evaluation/*.py) only, so edits to config.py (REFIT_EVERY, MIN_TRAIN, HORIZONS, QLEVELS, DAILY_LIMIT, TAX_*, COMMISSION_*) or types.py/data.py leave forecast/wf/result keys unchanged; _wf_key does not contain horizons, min_train or refit_every (they come from config defaults at call time).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 cachekey_test.py

**Evidence.** Revising only open/high/low/volume of the last 30 bars of 2330: price_signature, forecast key and wf key identical, but forecast changes: p_up h1 0.51643->0.51857, h5 0.54721->0.54226, h20 0.58397->0.57816; sigma h5 0.0370->0.0473, h20 0.0762->0.0881 (+16%); regime probs [0.401 0.521 0.078] -> [0 0 1]. Walk-forward p_up(h=1) moves by up to 0.0086 after revising only volume/high of the last 200 bars with the key unchanged. Copying backend/ and editing config.py (REFIT_EVERY 21->10, TAX_STOCK 0.003->0.006, DAILY_LIMIT 0.10->0.07): code_fingerprint('quant') 8c8fcb132b and ('quant','evaluation') 1c5b615dee identical for both trees. Yahoo volume is also revised after the first fetch (see C-13), so this path is reachable in normal operation.

**Proposed fix.** Hash all six price arrays plus dates in price_signature; add config.py, types.py and data.py (or a hash of the relevant config constants) to code_fingerprint; put horizons, min_train, refit_every and cfg into _wf_key / _forecast_key.

**Verifier:** real=True severity=low  
I re-ran the repro and it reproduces. I also read the code. data.price_signature hashes only symbol, n, last date, close and adjclose, but its docstring says it "changes on any revision". features.py uses high and low (parkinson_logvol) and volume (vol_z), so those inputs affect the output without changing the key. jobs.code_fingerprint hashes only backend/quant/*.py and backend/evaluation/*.py. Editing config.py (REFIT_EVERY, TAX_*, DAILY_LIMIT) therefore leaves the fingerprint unchanged. The _forecast_key, _wf_key and _result_key keys also contain none of horizons, min_train or refit_every, because those come from config at call time. The README claims keys cover "data hash, code fingerprint". The gap is real and the stale results are on the same last bar and the same close. I downgrade medium to low for three reasons. (1) The revisions in the evidence are synthetic and large (volume x5, high +3%, low -3% over 30 bars). The realistic case, a small Yahoo volume revision, moves p_up by well under 0.01. (2) The config trigger is a developer-time edit. The results are model outputs with no demonstrated edge anyway, and a VERSION bump or clearing the cache works around it. (3) Nothing wrong is shown to a user in normal operation beyond tiny numeric differences. It is still a documentation or contract mismatch with a cheap fix.

**Verifier refined fix.** 1) backend/data.py price_signature: also h.update the float64 bytes of data.open, data.high, data.low and data.volume (and ideally "|".join(dates), or at least the first date). Keep the [:16] truncation. 2) backend/jobs.py code_fingerprint: also hash (name, size, mtime) of backend/config.py, backend/types.py and backend/data.py. For example, take the *.py files of the "" package, restricted to those three names. Or hash repr of the relevant config constants: HORIZONS, MIN_TRAIN, REFIT_EVERY, QLEVELS, DAILY_LIMIT, TAX_*, COMMISSION_*. Hashing the constants is more precise, because it does not invalidate on comment-only edits. 3) _wf_key and _forecast_key need no change if the config hash is in the fingerprint. Optionally add config.HORIZONS, config.MIN_TRAIN and config.REFIT_EVERY to the _hash arguments of _wf_key for explicitness. Side effects: every existing disk cache (forecast, wf, results, backtest, universe) is invalidated once and gets recomputed, and the 60-stock validation is slow to refill. Tests that pin the price_signature or code_fingerprint format or value might need adjusting (a quick grep is advised). jobs.py:1212-1269 calls price_signature, so the backtest and universe result keys change too. That is intended.


## C-06 [medium] The universe 'recommender' score ranks volatility, not predicted edge, at h=1/5; the validated top-5 portfolio is a high-sigma basket and a model-free sigma ranking matches or beats it

- owner: evaluation  
- location: backend/evaluation/universe_bt.py:105-112 (score), :123-135 (picks), :234-246 (spread_stats); backend/evaluation/recommend.py (score_h)  
- reviewer: review:causal

**Description.** universe_bt (and recommend.py) rank by score = (E[simple R] - roundtrip_cost)/sigma. At h=1 the net expected return is negative for every name (cost 0.57% >> E[R] ~ 0.04%), so score ~ -cost/sigma and the ranking is almost exactly a sigma ranking (high-vol names on top). The universe report's top-5 CAGR (43-58%) vs equal weight (31%) and TAIEX (21%) and the top-minus-bottom 'spread' test (t=+3.57 at h=1) mostly measure a high-volatility tilt in a hindsight bull-market universe, not forecasting skill. The verdict machinery contains the lone 'spread' significance (requires >= 2 agreeing tests, so the verdict stays no_edge), but a model-free baseline (rank by sigma) is not reported and the spread test's null is not 'no skill' because the ranking rule itself is sigma-driven.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 t30_score_vs_sigma.py ; python3 t33_pit_universe_bt.py (point-in-time comparison; about 5 min)

**Evidence.** today's 60 names, next_open, costs 0.6 discount + 5bp: cross-sectional Spearman(score, sigma) = +0.991 (h=1), +0.951 (h=5), +0.629 (h=20); share of names with negative net E[R]: 1.00 / 0.88 / 0.07. Top-5 by validated score CAGR 43.0% / 58.1% / 42.0% (Sharpe 1.03/1.26/1.09) vs top-5 by sigma with NO forecast 40.4% / 71.2% / 63.5% (Sharpe 0.99/1.45/1.37); daily-return correlation of the two baskets 0.961 / 0.950 / 0.889. EW benchmark Sharpe 1.56 (CAGR 31.1%). On the point-in-time universe (C-07) the same top-5 earns -16.7% / 11.2% / 25.7% CAGR and the h=1 IC is significantly NEGATIVE (-0.021, t=-2.76) and unflagged because every test is one-sided positive.

**Proposed fix.** Report a model-free sigma-rank (and equal-weight) baseline next to 'top', test the spread against the sigma-matched baseline (or rank on E[R]/sigma without subtracting cost), add a two-sided check for significantly negative IC/spread, and state that at h=1 no name has positive net expectation.

**Verifier:** real=True severity=medium  
I reproduced the finding by re-running t30_score_vs_sigma.py against the 60-stock cache. The code still ranks by the cost-adjusted score. universe_bt.py:105-112 sets score = (E[simple R] - cost)/sigma, and picks(), bottoms() and spread_stats() all use it. recommend.py:356 sorts by shrunk score, which is a monotone map of the same score. The measured numbers match the report. Cross-sectional Spearman(score, sigma) is 0.991 / 0.951 / 0.629 at h=1/5/20. The share of names with negative net E[R] is 1.00 / 0.88 / 0.07. Top-5 by score earns CAGR 43.0 / 58.1 / 42.0% against 40.4 / 71.2 / 63.5% for top-5 by sigma with no forecast. The daily-return correlation between the two baskets is 0.96 / 0.95 / 0.89. The universe report has no sigma-ranked baseline, no mean-sigma or beta line, and no two-sided check for significantly negative IC or spread. So the 'top' versus equal-weight outperformance can be read as skill when it is mostly a high-volatility tilt. The defect is real, but it only misleads readers; it does not corrupt any statistic. The verdict stays no_edge because it needs at least 2 agreeing tests. The addendum in DESIGN.md section 9, item 1 (revised 2026-10-23), already records this exact diagnosis and requires the fix: rank by rank_signal = mu_log/sigma_log, and report top-k-by-sigma and equal-weight baselines plus the top-k's mean sigma and beta. `grep rank_signal` over backend, frontend and tests finds nothing, and the files were last modified on 2026-10-22, so the required fix has not been implemented. Severity stays medium. It is an interpretation and ranking-semantics flaw that misleads, but it does not change the verdict, and it is not a look-ahead or data bug.

**Verifier refined fix.** This is already specified in DESIGN.md section 9, item 1. The minimal change has four parts.

1. backend/evaluation/universe_bt.py, _Panel.__init__: add a rank-signal panel RS = mu / sg, which is the log-return information ratio because mu and sigma are log-scale. Fall back to norm.ppf(clip(p_up)) where mu or sigma is missing. Keep SC and net for the only_buy gate (net > 0 and PU >= buy_p). Change picks(), bottoms() and ic_series() to rank and score on RS instead of SC. Keep avail and eligibility unchanged. Do the same in recommend.py: sort by (-rank_signal, -p_up, code), and set rank, leaders, laggards, lean and score_rank_pct from rank_signal. Keep score and score_shrunk only for the signal gate, edge_covers_cost and the Kelly weights.

2. run_universe_validation: add a 'benchmark_top_sigma' stats block (top-k by SG with the same tranche, cost and next_open simulation) next to benchmark_equal_weight. Add mean_sigma and mean_beta of the top-k basket versus the universe to the report. Add a curve.top_sigma series. Update the _SCHEMA at lines 500-520 with optional keys, so existing keys stay compatible.

3. Optionally add a two-sided flag for significantly negative IC or spread, using the existing p_two and t, and report it in the notes without changing the verdict rule.

4. Optionally add a note that no name has positive net expectation at h=1.

Side effects: tests in tests/test_evaluation* and tests/test_server* that assert picks, spread or IC values computed from the score will need updated expectations. Any test that checks recommend ordering by shrunk score will also need updating. The frontend universe and picks views should accept the new optional keys. DESIGN.md requires a regression test that |Spearman(rank_signal, sigma)| < 0.9 on a synthetic universe where p_up is independent of sigma.


## C-07 [medium] Hindsight universe membership (deliberate, disclosed) inflates absolute universe results; the magnitude is not quantified in the product

- owner: evaluation  
- location: backend/data.py:128-142,1308-1327 (FALLBACK_UNIVERSE + today's traded-value top-up); backend/evaluation/universe_bt.py:358-365 (generic note)  
- reviewer: review:causal

**Description.** The validation universe is today's 60 liquid names (curated large caps topped up by today's traded value). This is a deliberate design choice and the report notes warn about survivorship, but the size of the effect is large and not stated. Only 18 of today's 60 were among the 60 most liquid four-digit stocks in the week before the test window; 28 were not even in the top 100 (e.g. 2360 rank 313, 2207 270, 5876 257, 5880 217, 5876, 4904 189, 2890 183).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 t11_pit_universe.py ; CAUSAL_CACHE=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_causal_net TWQ_OFFLINE=0 python3 t12_survivorship.py ; python3 t33_pit_universe_bt.py

**Evidence.** TWSE MI_INDEX 2021-08-20..26 mean traded value rank (reproduced identically on re-run): overlap of today's 60 with PIT top-60 / 80 / 100 / 150 = 18 / 27 / 32 / 44. Equal-weight daily-rebalanced total return 2021-08-26..2026-10-22: today's 60 +302.7% (CAGR 31.4%) vs PIT top-60 +131.8% (CAGR 17.9%) vs TAIEX +169.8% (CAGR 21.5%). Universe backtest top-5 CAGR (h=1/5/20): today's 60 43.0/58.1/42.0% vs PIT-60 -16.7/11.2/25.7%; EW 31.1% vs 17.5%; verdict no_edge in all six cells.

**Proposed fix.** Keep the design but put the measured size of the effect in the universe notes/UI banner (e.g. 'hindsight universe inflates CAGR by roughly 10-45 pp'), and offer a point-in-time membership (top-N by traded value on each rebalance date) for the validation.

**Verifier:** real=False severity=low  
The numbers reproduce from the reviewer's saved outputs: 18/27/32/44 of today's 60 fall inside the point-in-time top 60/80/100/150 by 2021-08-20..26 traded value. But this is not a defect. The hindsight universe is a deliberate design choice that DESIGN.md names twice (line 28, "disclose survivorship/selection bias of the universe", and the section 4 requirement for "an explicit survivorship-bias note"). It is disclosed explicitly in universe_bt.py `make_universe_notes`, which says the list uses hindsight and results are "很可能被高估" (likely overestimated). The same warning appears in `make_universe_report` and in the frontend: view_picks.js:208 and view_method.js:367 and 432. The disclosure is neither missing nor misleading, so the only open point is that it gives no magnitude. A magnitude would not be a reliable product constant: it comes from one start date and one 5-day traded-value snapshot, with gaps of roughly 10-45 pp between cells. The point-in-time top-60 by 5-day traded value was itself dominated by a shipping and meme-stock boom in Aug 2021, so it is not a clean counterfactual. A hard-coded "inflates CAGR by 10-45 pp" banner would be an unfounded claim. Impact is also small because the product's verdict is relative to the equal-weight benchmark on the same universe (no_edge in every cell), so it never presents the inflated absolute CAGR as evidence of an edge. Building point-in-time membership needs historical constituent or turnover data the app does not have, so it is a feature request, not a bug.

**Verifier refined fix.** No change required. If anything is added, put a qualitative sentence in make_universe_notes (backend/evaluation/universe_bt.py), for example "in a test, hindsight membership raised the equal-weight CAGR by about 14 pp over a point-in-time liquid set in this window". It should be worded as a single-window example, not a general bound. Side effects: notes text only, and any tests that assert note strings would need an update.


## C-08 [medium] Verdict aggregator returns 'moderate_edge' for 3 of 62 h=1 single-stock reports although direction accuracy is below always-up and Brier skill is negative (tick-size bid-ask-bounce predictability)

- owner: evaluation  
- location: backend/evaluation/report.py:240-270 (assess_evidence: n_agree/n_confirm rules, no veto for adverse tests)  
- reviewer: review:causal

**Description.** assess_evidence counts a test as support if p<5%/1% but ignores contradictory tests: any 2 of the 4 tests with p<1% (plus Bonferroni p_fw<1%) give moderate_edge even when the direction and Brier tests say the model is significantly WORSE than baseline (p = 1.0 / 0.96). The two supportive tests (Spearman IC of mu, DM on squared error) pick up a genuine but untradeable effect: for stocks whose tick is about half a daily sigma the closing price bounces between bid and ask (lag-1 Spearman autocorrelation of returns -0.13..-0.16), the DLM reversal features forecast it, and IC(mu) equals IC(-r_t). The text then says 有一定的預測優勢, contradicting the lead's 'no demonstrable edge' reading of the product.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 moderate_edge_cases.py ; python3 telecom_mechanism.py

**Evidence.** 2412 h=1: IC 0.1166 (p=3.6e-5), DM-mse p=0.0061, direction acc 0.4626 vs always-up 0.5374 (p=1.0), Brier skill -0.0148 (DM p=0.959), oos_r2 +0.0048 -> moderate_edge. 3045: IC 0.1248 (p=1e-5), mse p=0.00075, acc 0.4935 vs 0.5205, BSS -0.0026 -> moderate_edge. 4904: IC 0.0885 (p=0.0017), mse p=0.0024, acc 0.5241 vs 0.5353, BSS -0.0017 -> moderate_edge. 3 of 62 stocks at h=1 (4.8%) vs 0 of 124 at h=5/20 (expected false-positive rate per the module docstring is about 0.06%). Mechanism: 2412 tick/price 0.40%, daily sigma 0.72%, tick/sigma 0.56, tie rate 25.1%, Spearman AC1 -0.164, IC(mu) 0.117 vs IC(-r_t) 0.164; 3045 tick/sigma 0.49, AC1 -0.132, IC(mu) 0.125. Across the 60 names corr(tick/sigma, AC1) = -0.40; mean IC(mu) of the 5 largest tick/sigma names +0.0475 vs -0.0019 for the other 55.

**Proposed fix.** Add a veto: any test significantly adverse (one-sided p > 0.95 for direction or Brier) caps the level at no_edge/weak_edge with an explicit 'conflicting evidence' bullet; flag microstructure-dominated names (tick/sigma > 0.3 or AC1 < -0.1) and show the net-of-cost tradability before using the word 優勢.

**Verifier:** real=True severity=low  
I reproduced the report numbers and confirmed the code path. assess_evidence (report.py:240-270) counts only supportive tests. It has no veto for tests that point the wrong way. With two tests at p<1% (ic, mse) and p_fw<1%, it returns moderate_edge even when the direction and brier tests are adverse. The verdict text then says "統計證據相當一致" and the title says 有一定的預測優勢. It does this while the same text shows direction accuracy of 46.3% against an always-up baseline of 53.7%, and a Brier skill of -1.5%. That is a real internal contradiction, and it conflicts with DESIGN §0's honesty principles and the "conservative" level description. It is not a documented, deliberate design choice. The module docstring only describes the agree and confirm counting rules, and it does not mention that adverse tests are allowed.

I rate it low rather than medium for three reasons. (1) The ic and mse effects are statistically genuine, not false positives: IC p is about 1e-5 and the DM test on squared error is significant. Tick-size bid-ask bounce is a plausible mechanism, so "predictable" is technically true. (2) The moderate_edge text already adds "但扣除交易成本後，策略尚未能穩定勝過買進持有" when the strategy is not tradable, and the adverse numbers are printed in the head and bullets. (3) It affects only the single-stock h=1 verdict, about 3 of 62 names (4.8%), and the universe verdict and the recommender are unaffected.

I did not independently verify the microstructure mechanism numbers (tick/sigma, AC1). They are not needed to confirm the verdict logic flaw. The proposed fix is also incomplete: with a veto at p>0.95, 4904 would still get moderate_edge, because its direction p is 0.787 and its brier p is 0.686. A rule on the sign of the point estimate is needed to catch that case.

**Verifier refined fix.** In backend/evaluation/report.py, assess_evidence, after the `level = "weak_edge"` / moderate block at lines 264-270 and before `res["level"] = level`:

1. Compute adverse = [k for k in ("direction", "brier") if ps.get(k, 0) > 0.95] (both are one-sided "better than baseline" p-values). Also treat the case where both direction and brier have p > 0.5 as soft-adverse, which catches 4904.
2. If any hard-adverse test exists, cap the level at "weak_edge". If the model is clearly worse than the baseline on direction or Brier, cap at "no_edge" with reason "conflicting". Always set res["conflict"] = adverse and res["reason"] = "conflicting_evidence". In 2412 and 3045 the level can only reach weak_edge because 2 of 4 tests contradict. The simplest and most honest choice is to cap it at weak_edge.
3. In make_verdict (report.py ~365-372), for moderate_edge and weak_edge, if ass.get("conflict") is set, drop the phrase "統計證據相當一致" and append a sentence such as "但 方向/Brier 檢驗顯示模型不優於簡單基準（p=…），訊號可能來自價格最小跳動單位造成的買賣價反彈，未必能交易。" Optionally add a tick-size warning bullet when the daily sigma is small relative to the tick (tick/price / sigma > 0.3).

Side effects: tests in tests/test_eval_report.py that assert moderate_edge or strong_edge on planted-edge synthetic worlds should be unaffected, because a real edge has favourable direction and brier p-values. This should be re-run. Tests that build an evidence dict by hand with "direction" or "brier" p near 1 and expect moderate_edge would break. The universe_bt.py verdict reuses assess_evidence with THRESH_UNIVERSE, so the veto would apply there too, which is acceptable. The frontend text at view_backtest.js:166 may need softer wording but no schema change. Add the keys "conflict" and "reason" only additively, so validate_backtest_report is unaffected.


## C-09 [medium] Flat zero-volume placeholder bars (halts / Yahoo gaps) are modelled as real zero-return days: volatility collapses and p_up becomes extreme; resumption outcomes fall outside the 90% interval

- owner: quant  
- location: backend/quant/features.py:60-63,201-215 (clean_returns / rc), :186-198 (only Parkinson masks placeholders); backend/quant/garch.py/hmm.py consume rc  
- reviewer: review:causal

**Description.** Yahoo fills trading halts and missing data with bars of constant price, high==low and volume 0. clean_returns/GARCH/EWMA/HMM treat them as observed zero returns (only the Parkinson range and vol_z mask them). After k such bars the live forecast's volatility shrinks instead of widening, and the first real bar after a halt carries the accumulated multi-session move. Real examples exist in the evaluation windows (2614 21 flat bars, 3481 21, 2327 14, 2603 7, 2409 7, 2607 6, 6415 5, 0050 5 during the 2025-06 split halt).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 t31_halt.py ; python3 halt_events.py ; python3 coverage_pooled.py ; python3 look_flat.py

**Evidence.** 2330 live forecast with k artificial flat bars appended: sigma_1d 1.66% (k=0) -> 1.51% (5) -> 1.04% (20) -> 0.21% (60) -> 0.13% (120); 90% 1-day interval (-2.55%,+2.81%) -> (-1.58%,+1.74%) at 20 -> (-0.23%,+0.33%) at 60; p_up h20 0.584 -> 0.594 (20) -> 0.753 (60). Real data: of the 5 halt-resumption origins (>= 3 flat bars, next bar trades) in the 62 test windows, 4 realised outside the 90% interval: 2603 2022-09-16 -17.8% vs sigma 2.63% (z=-6.8), 2327 2022-10-28 +8.6% (z=+3.4), 2327 2025-08-22 +4.65% (z=+3.1), 6415 2022-07-12 -10.4% (z=-2.7); 0050 2025-06-17 inside. Only 56/115/325 rows (0.07%/0.15%/0.43%) of h=1/5/20 test rows touch a flat bar, so this does not explain the pooled 90%-coverage shortfall (88.7%/89.5%/89.0%).

**Proposed fix.** Detect placeholder bars (volume==0 and high==low, or close unchanged with zero volume) and treat them as missing: skip them in return/variance recursions (carry the state), count elapsed sessions for the first real bar (inflate its variance by the number of skipped sessions), and flag the forecast as 'post-halt, low reliability'.

**Verifier:** real=True severity=low  
I reproduced the mechanism. Appending k flat, zero-volume bars to 2330 gives the same numbers as the finding. sigma_1d goes 1.66% -> 1.51% (k=5) -> 1.04% (k=20) -> 0.21% (k=60). The 1-day 90% interval narrows from (-2.55%,+2.81%) to (-0.23%,+0.33%) at k=60. p_up for h=20 rises from 0.584 to 0.753 at k=60. On the real walk-forward windows, halt_events.py finds 5 resumption origins, and 4 of them fall outside the 90% interval (mean |z| 3.35). The code confirms the cause. clean_returns only zeroes |r|>0.25. The module docstring (features.py lines 14-16) says zero-volume placeholder bars are handled, but nothing masks them in the return, EWMA, GARCH or HMM paths. Only the Parkinson range ignores them, via H<=L. So the defect is real, and it contradicts the module's own documented intent. I rate it low, not medium, for these reasons. (1) It is rare: 0.07%, 0.15% and 0.43% of h=1/5/20 test rows touch a flat bar, and the finding itself says this does not explain the pooled coverage shortfall. (2) It only bites during or just after a halt, and the live damage needs a flat tail. I scanned all 112 cached symbols and none has a flat bar in its last 60 bars, so no current live forecast is affected. (3) The failure is localized and cannot silently bias the pooled evaluation or the "no edge" conclusions. It only affects a user who queries a stock that is halted, or just back from a halt, and the interval is then too narrow. That is a correctness blemish, not a systemic issue.

**Verifier refined fix.** Minimal change: drop placeholder bars at the data layer instead of editing every model. In backend/data.py, inside completed() or the PriceData builder (around line 799), remove rows where volume<=0 and high==low and close==previous close. Skip this for kind=="index" (^TWII has volume 0, though its H!=L). The first real bar after the halt then carries the multi-session move, which clean_returns already clips at +-0.12. Optionally add a "post_halt" flag and widen the first-step variance by the number of dropped sessions. A narrower alternative is to apply the filter only in the engine's forecast_latest and walk-forward input. Side effects: (1) row indices shift, so any evaluation or test that indexes bars by position or hard-codes n will move. (2) Horizon steps then span calendar days that include the halt, which is acceptable. (3) Synthetic test fixtures with constant price and zero volume (data and quant tests) may need adjusting. (4) Cached walk-forward results (wf/*.pkl.gz) must be invalidated, and only the ~8 affected symbols change. Do not change the 0050 2025-06 split halt, where the adjusted price is already continuous. Keeping those 5 bars would still be harmless.


## C-10 [low] PIT histogram and KS uniformity test are invalid for lattice outcomes (ties and tick discreteness): 65% of high tick/sigma stocks 'fail' calibration at h=1

- owner: evaluation  
- location: backend/quant/engine.py:340 (pit = pool.cdf(raw_r)); backend/evaluation/metrics.py:510-530 (pit_histogram, pit_ks_pvalue)  
- reviewer: review:causal

**Description.** hf.pit evaluates a continuous predictive CDF at outcomes that sit on the price tick lattice (and 25% exact zeros for 2412), so the PIT is spiky, not uniform, even for a perfect model. The report publishes pit_hist and pit_ks_pvalue (n_eff-scaled) without randomisation.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 pit_discreteness.py

**Evidence.** h=1, 61 non-ETF stocks: KS p<0.01 in 26% (p<0.001 in 15%); stocks with tick/sigma > 0.15 (n=17): 65% reject vs 11% of the other 44; Spearman(tick/sigma, -log10 KS p) = 0.54. 2412 PIT bins [0.096 0.092 0.144 0.014 0.240 0.010 0.023 0.181 0.086 0.112], KS p = 1.3e-13; 3045 KS p = 3.8e-8; 2330 (few ties) p = 0.34.

**Proposed fix.** Use a randomised PIT for discrete outcomes (draw uniformly between CDF(R - tick/2) and CDF(R + tick/2), or between the left and right limits at ties), or suppress the KS p-value when tick/sigma exceeds ~0.15.


## C-11 [low] features.build_features is not a prefix function: global count criteria (has_market, vol_z availability) make early rows depend on the future; truncation invariance fails for 2317 (p_up h1 shifts by up to 0.024) and early rows of 6 more series

- owner: quant  
- location: backend/quant/features.py:274 (vol_z >= 30 switch), :290-291 (has_market), :292-300 (zero-return market for NaN rows)  
- reviewer: review:causal

**Description.** has_market = np.isfinite(lm).sum() > 60 and the vol_z switch np.isfinite(lv).sum() >= 30 look at the WHOLE series. For rows whose past has fewer than 61 market-aligned bars / 30 positive volumes, the full-series build and a truncated build give different features. For 2317 (history from 1993, TAIEX from 1997-07) the full build fabricates market features from a zero-return market for rows 59..1171 (mkt_vol X down to -1.27, idio_mom +-2.5) while the truncated build has none; the DLM state then differs for ever after, decaying with discount 0.998. No outcome information leaks (only an availability switch), and no default test window is affected, but it breaks the documented invariant walk_forward(data.slice(i+1)).last_row == walk_forward(data).row(i) and the single-code-path guarantee on early rows.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 t16_feature_trunc_all.py ; python3 feature_nonprefix.py ; python3 feature_nonprefix_detail.py ; python3 engine_trunc_early.py

**Evidence.** build_features(d.slice(k)) vs build_features(d)[:k] over 112 series x 15 cuts x 8 arrays (13,248 comparisons): 34 mismatches, in 7 series: 2317 (all cuts up to k=1100; rows 59..), 0056 (rows 251-259 at k=260; 250 flat bars at start), 3661 (rows 103+ at k=126/150), 4938, 4961 (rows 180+ at k=200), 5285, 8046; first row where both criteria hold: 2317 1231, 0056 279, 3661 150, 4938 386, 4961 210, 5285 153, 8046 130 (only 2317 exceeds MIN_TRAIN=500). Engine level: walk_forward(2317[:1200], start_idx=600) vs full: 33 of 53 arrays differ on all 600 rows; max |dp_up| h1 0.0239, h5 0.0127, h20 0.0017; sigma up to 1.5e-3. The same test on 3661 (k=200), 0056 (k=300), 4938 (k=450) is clean at engine level.

**Proposed fix.** Decide availability per row from prefix counts (cumulative finite count at row i >= 30 / > 60), not from the global count, and set market features to NaN (not zero-return market) for rows before the market series exists.


## C-12 [low] HMM regime probabilities (and the displayed regime label) are numerically unstable: a 1e-6 relative, tie-preserving price perturbation moves them by up to 0.72

- owner: quant  
- location: backend/quant/hmm.py:329-376 (fit_hmm: 3 inits, EM), backend/quant/engine.py:234-240 (anchor cold fits)  
- reviewer: review:causal

**Description.** The HMM is cold-fitted at absolute anchor rows with three competing deterministic initialisations; tiny input changes select a different local optimum. Forecast p_up/sigma are robust (the HMM is one pool member) but the displayed regime probabilities/current regime are not.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 t24b_noise_stability_ties_preserved.py

**Evidence.** Value-based multiplicative noise 1e-6 on all of open/high/low/close/adjclose (equal prices stay equal, 3 seeds): max |d regime prob| 2603 0.722, 0050 0.153, 3661 0.137, 2330 0.059, 2317 0.009, 6488 0.003; p_up changes <= 0.0020 (h1), 0.0014 (h5), 0.0025 (h20); sigma relative change <= 2.8%. Plain (tie-breaking) 1e-6 noise moves h1 p_up by up to 0.044 (2412) because ties flip (see C-02).

**Proposed fix.** Regularise the init competition (e.g. continuity prior to the previous anchor, deterministic tolerance band before switching optimum) and present regime output with its uncertainty; do not show 'current regime' when the top two probabilities are close or the fit changed optimum.


## C-13 [low] Yahoo series contain session gaps, placeholder bars, stale copies and un-adjusted/mis-timed jumps; none is detected or flagged (516 adjusted daily moves beyond +-11% over 113 series, 47 since 2015-06)

- owner: data  
- location: backend/data.py:431-510 (parse_yahoo_chart drops null rows silently), :1308-1327; backend/evaluation/report.py:103-117 (DATA_ALERT_MOVE = 0.25)  
- reviewer: review:causal

**Description.** The parser silently drops null rows (no gap detection against ^TWII or the TWSE calendar). Scan of all 113 cached series (60 universe stocks, 52 other stocks/ETFs, ^TWII): no non-increasing dates, no non-positive prices, no high<low, OHLC repairs 4 bars; but (a) official sessions missing from the stock series: 4,012 over history, 151 in the last 1250 rows, all of them 2025-08-01 for the 60 universe names (TWSE STOCK_DAY shows 2330 traded 45,352,225 shares at 1,140 that day; Yahoo has a null row, ^TWII has the bar), so e.g. 2360 +14.9% and 1301 -11.3% on 2025-08-04 are 2-session moves treated as one bar (clean_returns clips +14.9% to +12%); (b) phantom bars on days TWSE was closed: 6,480 (none in the last 1250 rows); (c) stale-copy duplicate bars (identical OHLC and non-zero volume): 117, on 2000-07-19 (42 series), 2001-04-26 (46), 2006-05-30 (7) plus 5 isolated; (d) dividend steps landing on placeholder bars; (e) Yahoo volume is systematically below TWSE official.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 scan_data.py ; python3 scan_big.py ; python3 classify_big.py ; python3 calendar_check.py ; python3 look_div.py ; python3 look_flat.py ; python3 yahoo_vs_official.py   # network, TWSE STOCK_DAY

**Evidence.** 516 moves beyond +-11% (|simple adj return|): by cause 219 in the +-7% limit era (2000-2015) not attributable, 96 missing official session(s) between bars, 82 after a flat placeholder bar, 42 on a flat placeholder bar, 42 ex-dividend steps, 12 emerging-board 6669 (2018), 10 new listings, 7 unexplained in the +-10% era, 6 bars dated on closed days. Since 2015-06: 47 moves: 13 missing sessions, 12 emerging board, 9 after placeholder, 7 unexplained (4961 2015-08-27 +18.6%, 2015-08-28 +12.1%, 2016-02-25 +12.2%, 2016-08-10 +12.0%, 2018-04-30 +23.5%; 6531 2015-10-14 +21.3%; 5608 2021-10-04 -11.3% = TWSE 權 ex-right day plus limit-down), 3 on placeholder bars, 2 dividend steps, 1 closed-day bar. In the evaluation windows (last 1250 rows) 7 events: 1301 and 2360 2025-08-04 (missing session), 2409 2022-09-29 +12.2% (ex-div 2.0 booked on a zero-volume bar, price drop prints only after the 7-bar halt), 2607 2025-09-25 +16.7% (ex-div 10 on a flat bar) and 2025-10-07 -15.7%, 2603 2022-09-19 -16.3% (4 placeholder bars before), 5608 2021-10-04. Large historical artefacts: 0050 2014-01-02 -75.06% (unadjusted split), 2371 2011-03-24 -59.98% and 2011-04-11 +144.4% (zero-volume bars), 8215 2010-11-12 +151%, 2317 1994-05-24 +117.5% / 1995-08-16 -27.2% (pre-TAIEX), 2351 2001-11-05 +40.2%, 2357 2010-06-24 -28.4%, 8261 2015-09-11 -25.7%, 2612 2016-09-28 +29.9% / 09-29 -22.6% / 10-07 -16.3%, 2357 2008-08-29 +19.0% and 2009-08-18 +28.5% (ex-dividend steps of 24.6% / 33.5%). Yahoo volume vs TWSE official (588 days, 2330/2317/2454/2412): Yahoo lower on 100% of days, median -3.0%, mean -4.9%, p5 -17.2%; OHLC identical (0 mismatches > 0.2%). report.data_alerts only flags |move| > 25%.

**Proposed fix.** Cross-check each series against the ^TWII/TWSE session calendar and log/flag missing sessions (fill from TWSE STOCK_DAY or mark the multi-session bar), flag placeholder runs, lower the data-alert threshold to the +-11% limit regime (after 2015-06) and exclude flagged rows from realised-return statistics; do not book dividend steps on placeholder bars.


## C-14 [low] Live Monte-Carlo fan/VaR/CVaR are seeded by (last date, len(data)): numbers change when the history length changes by one bar

- owner: quant  
- location: backend/quant/engine.py:507-508  
- reviewer: review:causal

**Description.** forecast_latest seeds the filtered-historical simulation with date*7919 + n, so any change of n (e.g. a historical session filled in, C-13) redraws all paths. Horizon cards are aligned to the exact ensemble quantiles, but the fan between anchor days and the risk block (VaR/CVaR/drawdown/P(loss>10%)) are not reproducible. Speculative as a user impact (pure MC noise, 5000 paths).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/causal && python3 mc_seed_dependence.py

**Evidence.** Dropping the first 1-3 bars of history (n -> n-1..n-3) for 2330/0050/2603: fan non-anchor days change by up to 0.0058/0.0073/0.0090 (simple return), CVaR95(20d) by up to 0.0050/0.0076/0.0037, P(loss>10%,20d) by up to 0.0010/0.0026/0.0096, median max drawdown by up to 0.0019 (part of this is the real effect of removing a bar).

**Proposed fix.** Seed from (symbol, last date) only and use common random numbers; raise mc_paths or report MC standard errors for the risk block.


## C-15 [low] WalkForward.regime_labels are derived from the LAST fitted HMM (hindsight labels for past rows); currently unused (speculative)

- owner: quant  
- location: backend/quant/engine.py:415,425  
- reviewer: review:causal

**Description.** engine.walk_forward sets regime_labels = Hm.regime_labels(last_hp) from the final block's HMM while regime_prob rows come from earlier fits whose states may have different drift/vol; a consumer mapping past regime_prob rows to these labels would show future-informed labels. grep shows no consumer today, so this is a latent hazard, not an observed defect.

**Repro.** cd /home/gary/stock && grep -rn regime_labels backend/ frontend/*.js

**Evidence.** grep -rn regime_labels backend/: only types.py:97 (field), engine.py:425 (set) and engine.py:522 (live labels from the current HMM, correct).

**Proposed fix.** Store per-block labels (or per-row label ids) in WalkForward, or document that regime_labels apply to the final fit only.


## M-01 [high] Unchanged closes (R==0) are scored three different ways: counted as failures by the calibrator, dropped by the evaluation, absent from the predictive; UI shows P(down)=1-p_up

- owner: quant  
- location: backend/quant/engine.py:155; backend/quant/calibration.py (recalibrate); backend/evaluation/report.py:155-156,173-174; backend/evaluation/metrics.py:137; frontend/charts.js:703  
- reviewer: review:math

**Description.** engine._Ctx.up = 1[Rc>0], so calibration.recalibrate learns P(R>0) with unchanged closes as failures. report.analyse (report.py:155-156) and metrics.expanding_up_rate (the causal base, metrics.py:137) DROP R==0 rows. The pooled predictive is continuous (no mass at 0) and frontend/charts.js:703 prints downside probability as 1-p_up. On Taiwan tick grids ties are structural: 6.8% of h=1 outcomes on average (59 of 62 stocks above 3%; 2412 25.1%, 3045 19.9%, 2883 12.8%, 2887 12.1%), 2.7% at h=5, 1.2% at h=20. Consequences: the shipped calibrated p_up matches neither convention; an apparent Brier deficit vs the base rate at h=1; a PIT point mass at F(0) (M-05); cards where P(up)<50% while the median and mean return on the same card are >0; P(down) overstated by the flat mass. DEFECT (inconsistent definitions), not a deliberate conservative choice.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m6_tie_world.py 0 1 2 3   (controlled tick-rounded world; log m6_tie_world.log). On real data: python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m26_tie_conventions.py (both conventions), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m19_tie_dose_response.py (dose-response), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b11_ties.py (tie rates, level shift), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b13_grid.py all (variant grid; saved output logs/b13_grid_base.log).

**Evidence.** 63 stocks, test window 2021-08..2026-10, h=1. (a) mean(calibrated p_up - realised up-rate) = -1.43pp with ties dropped, +2.04pp with ties counted as not-up; 2412: p=0.474 vs 0.537 (dropped) / 0.403 (as failures). (b) delta Brier vs causal base (x1e-4, NW t): +2.69 (t=+1.59) under the evaluation convention but -3.02 (t=-0.57) under the calibrator's own convention (h=5: +0.91 vs -3.35; h=20: -0.39 vs -7.33), so the h=1 'worse than base rate' is a convention artefact. (c) dose-response across stocks: delta Brier slope +107.8e-4 per unit tie share (r=+0.50, p=3.6e-5); level gap (p_ship - base) slope -0.2 (r=-0.68, p=6e-10); high-tie half +4.54e-4 vs low-tie half +1.02e-4; with calibration on non-tie targets the correlation vanishes (r=-0.22). (d) 23 of 64 cached live h=1 forecasts have p_up<0.5 with median_return>0. (e) Controlled test: latent GJR-GARCH-t (the model's home world) with closes rounded to the Taiwan tick grid, 15% ties, 4 seeds: shipped PIT-KS p = 9e-5 / 3e-8 / 0.048 / 5e-12 vs 0.89 / 0.67 / 0.19 / 0.08 after tick randomisation; mean calibrated p 0.481 vs realised 0.5125 (ties dropped) / 0.4345 (ties as failures); calibrated Brier worse than causal base by +4.5..+15.6e-4 while the raw ensemble is better (-5.8..-0.2e-4).

**Proposed fix.** Make the predictive tick-aware (Taiwan tick table by price and kind; ETF 0.01 below 50, 0.05 from 50): P(up)=1-F(ln(1+tick/2c)), P(down)=F(ln(1-tick/2c)), P(flat)=remainder; publish all three and show 'flat' in the UI instead of 1-p_up. Calibrate and score ONE event: either y=1[R>0] on all rows with a causal base that also counts ties (then the shipped p_up already beats the base: -3.0/-3.4/-7.3e-4 at h=1/5/20, t=-0.6/-0.8/-1.7) or p_cond=P(up)/(P(up)+P(down)) on non-ties. Interim prototype (calibrate on non-tie targets only): test delta Brier vs causal base +2.78e-4 (t 1.60) -> -0.18e-4 (t -0.19) at h=1 (dev +4.36 -> -0.53), h=5 +0.97 -> +0.48, h=20 -0.75 -> -0.87 (unchanged: ties are 1.2% there). Randomised PIT (M-05).

**Verifier:** real=True severity=medium  
The inconsistency is real and I reproduced the key numbers independently, but the practical impact is smaller than "high" and the reviewer's framing is partly off.

Code check:
- engine.py:155 sets `up = 1[Rc>0]`, so unchanged closes are failures for `Cal.recalibrate` (engine.py:330).
- report.py:155-156 drops `R==0` rows before scoring. `metrics.expanding_up_rate` (metrics.py:137) drops them from the causal base.
- DESIGN.md §4 documents the evaluation convention ("R==0 excluded and counted"), so that part is by design. The mismatch between the calibrator target and the scoring event is not documented.
- The predictive is continuous. frontend/charts.js:703 prints `1 - p` as downside probability.

My independent re-run used the cached 1250-bar walk-forwards, 62 stocks, and the same test window. It matched the reviewer:
- Tie share: h=1 6.79% on average, with 59 of 62 stocks above 3%; 2412 at 25.1%, 3045 at 19.9%. At h=5 it is 2.72% and at h=20 1.23%.
- Calibrated p_up minus realised up-rate: -1.44pp with ties dropped, +2.06pp with ties counted as not-up.
- Delta Brier vs the causal base, drop convention: +2.78e-4 (NW t 1.62) at h=1, +0.92e-4 at h=5, -0.68e-4 at h=20.
- Delta Brier vs the causal base, ties-as-failures convention: -2.92e-4 (t -0.54), -3.36e-4 (t -0.82), -7.61e-4 (t -1.76).
- Dose-response: across stocks at h=1, corr(tie share, delta Brier) = +0.49 and corr(tie share, level gap) = -0.62.
- 22 of 64 cached live h=1 forecasts have p_up<0.5 with median_return>0 (reviewer said 23).

Why not high:
1. The shipped p_up is a calibrated estimate of P(R>0) and is labelled that way. By the ties-as-failures convention it beats the base rate, so the calibrator is arguably right. The faulty piece is the evaluation, which scores P(R>0) against the conditional event P(up | non-tie). That is also why p_up<0.5 can sit next to a median slightly above 0: the flat mass is real.
2. No verdict changes. Both conventions give insignificant delta Brier (|t| < 1.8), so the verdict stays no_edge. The h=1 "worse than base" is not significant either way (t=+1.6), so it is a convention artefact but not a decision-changing one.
3. The UI harm is narrow. `1 - p` appears only in the a11y text of the probability gauge. The gauge draws p_up only, and no other frontend code computes 1 - p_up.
4. The visible harm is concentrated in high-tie names. For 2412 the backtest shows p=0.474 vs a 0.537 realised up-rate, and per-stock reliability, ECE and delta Brier for such names are biased by the convention.

It is still a genuine definitional defect, so I keep it at medium rather than dismissing it.

**Verifier refined fix.** Smallest consistent fix, keeping DESIGN §4 (ties excluded from scoring):
- backend/quant/engine.py:155, change `self.up` to `np.where(np.isfinite(self.Rc) & (self.Rc != 0.0), (self.Rc > 0.0).astype(float), np.nan)`. `recalibrate` already masks non-finite `up` via `ok = isfinite(p_raw) & isfinite(up)`, so ties drop out of the Platt fit with no change to calibration.py. Then p_up = P(up | non-tie) matches the evaluation, the causal base and 1 - CDF(0) of the continuous predictive.
- Update the calibration.py docstring ("1[R>0]" becomes "1[R>0] among non-zero R") and the DESIGN §3.6 wording for the calibrated p_up.
- Reword charts.js:703 so it no longer reports `1 - p` as P(down). Either label it "conditional on a non-zero move", or drop the downside figure.

Alternative (more principled, larger): make the predictive tick-aware and publish p_up/p_flat/p_down, then score y=1[R>0] on all rows with a base that also counts ties. This touches the types, the server schema, the evaluation and tests.

Side effects of the minimal fix:
- `_fp_quant` changes, so all `wf/*.pkl.gz`, `forecast/*.json` and cached results recompute.
- Quant tests with synthetic data have no exact ties and should be unaffected. The evaluation and frontend tests use fixed fixtures and are unaffected.
- p_up rises about 1-2pp on average at h=1, and by more for high-tie names (2412 +6pp).
- The "p_up<0.5 but median>0" cards become rarer, and the Brier convention gap disappears.
- The reviewer's prototype reports h=1 test delta Brier +2.78e-4 (t 1.60) falling to -0.18e-4. I did not re-run that prototype.


## M-02 [high] Brier/log-loss skill and the Brier Diebold-Mariano test use the verification window's hindsight base rate, contradicting DESIGN 9(a); reported skill is -0.30/-0.60/-2.30% but vs a causal base it is -0.11/-0.04/+0.02%

- owner: evaluation  
- location: backend/evaluation/report.py:171-174,222; backend/evaluation/metrics.py:318 (probabilistic_summary)  
- reviewer: review:math

**Description.** report.analyse sets clim_const = up.mean() of the scored rows (report.py:171-174) and uses it for brier_baseline, log_loss_baseline, brier_skill and the DM test feeding the verdict (report.py:222). DESIGN 9(a) (binding) requires the expanding-window frequency known at each origin; the code comment calls the oracle a 'harder reference'. Direction is conservative (it cannot manufacture an edge) but the headline KPI is misleading: at h=20 essentially the whole negative skill is the hindsight reference. DEFECT against the binding spec.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m8_app_bss.py (app code path, 4 stocks x 3 h); python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b9_brier.py (63-stock decomposition: reliability/resolution/uncertainty, noise cost vs signal gain, paired NW test).

**Evidence.** 63 stocks, test window, ties dropped as the app does. Reported (hindsight) Brier skill -0.303% / -0.597% / -2.297% at h=1/5/20; against the causal expanding base on identical rows -0.107% / -0.035% / +0.021%; oracle-vs-causal baseline gap alone 0.195% / 0.562% / 2.318%. Model vs causal base: delta Brier +2.7e-4 (NW t +1.58, 43/63 stocks worse) / +0.9e-4 (t +0.43, 41/63) / -0.5e-4 (t -0.10, 39/63). Verified on the app's own code path: report.analyse brier_baseline equals the in-sample constant to 1e-12 for 2330, 0050, 6488, 2412 at all horizons; skill sign flips per stock (6488 h=20 app -1.38% vs causal +2.27%; 6488 h=5 -0.79% vs +0.09%; 2330 h=20 -2.75% vs -1.63%). Answer to 'is the model genuinely worse?': at h=5/20 no; at h=1 modestly (+2.7e-4) and that is largely M-01 + M-03.

**Proposed fix.** Use the causal expanding (tie-consistent with M-01) base rate as the primary brier_baseline / log_loss_baseline / DM reference and compute skill and the DM p-value against it; keep the in-sample constant only as a separately labelled 'oracle' field. Update verdict evidence accordingly.

**Verifier:** real=True severity=medium  
Confirmed. report.analyse (report.py:171-174) sets clim_const = up.mean() of the scored rows. That hindsight constant is the p_ref for probabilistic_summary, so brier_baseline, log_loss_baseline and brier_skill all use it. It is also the reference in the Brier DM test (report.py:222), which feeds ev["p"]["brier"] and the verdict. DESIGN 9 audit rule (a) is binding. It requires the Brier/log-loss baseline to be the causal expanding-window base rate known at each origin, with the oracle baseline reported separately and labelled as such. The code does not do this. It already computes the causal series `clim = M.expanding_up_rate(R, h)` a few lines earlier and uses it only for train_majority. The docstring of metrics.probabilistic_summary also says "vs the causal climatology p_ref", which contradicts what the caller passes.

I re-ran the reviewer's app-path script (m8_app_bss.py) on 2330, 0050, 6488 and 2412 at h=1/5/20. In all 12 cases app brier_baseline equals the in-sample oracle value b(1-b) to 1e-12. Skill against the causal base differs materially and sometimes flips sign. 6488 h=20 is -1.38% in the app vs +2.27% causal. 6488 h=5 is -0.79% vs +0.09%. 2330 h=20 is -2.75% vs -1.63%. 2412 h=20 is -3.81% vs -2.06%. The oracle-minus-causal baseline gap is 0.03% to 3.6%, largest at h=20. The reviewer's 63-stock numbers were not rerun by me, but the mechanism and the per-stock pattern match.

I rate it medium rather than high because the bias is conservative. A hindsight reference is the harder reference, so it cannot manufacture an edge or a false 'tradable' verdict, and the verdict is not biased toward a false positive. The harm is that the displayed Brier skill KPI (view_backtest, view_stock, picks) overstates how much worse than naive the model is. The verdict can also be pushed slightly toward no_edge. That misleads the user about model quality and breaks a binding spec, so it needs fixing, but it does not produce a dangerous false claim.

**Verifier refined fix.** 1) backend/evaluation/report.py, analyse(): replace the clim_const block with the causal reference `p_ref = clim[Sx]`, i.e. `M.expanding_up_rate(R, h)` already computed above. It is tie-consistent because expanding_up_rate drops R==0 from both numerator and denominator. Call `M.probabilistic_summary(p, up.astype(float), p_ref)`, and in the evidence block use `M.dm_test((p_ref - up)**2, (p - up)**2, h)`. Keep ev["effect"]["brier"] = the new causal brier_skill.
2) Report the oracle separately and labelled. Add `out["probabilistic"]["oracle"] = {"brier_baseline": b(1-b), "brier_skill": 1 - brier/(b(1-b)), "log_loss_baseline": log_loss(const,up)}` with b = up.mean(). Do this in report.py, not in metrics.probabilistic_summary, because tests/test_eval_metrics.py:278 asserts the exact key set of that function. validate_backtest_report and tests/schema_check.py must allow the extra key. If strict key checking is used there, update both.
3) backend/evaluation/universe_bt.py:261: use the causal reference for the per-stock brier_skill, e.g. 1 - mean((p-up)^2) / mean((clim[Sx]-up)^2). Do this consistently, or leave it labelled as oracle. The pooled universe KPI should be checked for the same hindsight reference.
4) Tests to update: tests/test_eval_report.py indep() (brier_clim = mean of (expanding_up_rate - up)^2 over scored rows) and the assertions at lines 121-122; tests/test_eval_universe.py:324; optionally add an assertion for the oracle field. test_eval_report.py:499 (`<=0.0005`) still passes because a model equal to the causal climatology gets skill exactly 0.
5) Frontend: the 'vs 基準' label in view_backtest.js:195 should say causal base rate. Optionally show the oracle as a secondary line. The fixtures in make_front_fixtures.py need no change unless the oracle field is rendered.
Side effects: displayed Brier skill moves toward 0 or positive, so some verdict evidence p-values (the brier DM) become less negative. Verdict levels cannot become more optimistic than the causal truth, and the no_edge results for the real data should be unchanged.


## M-03 [medium] Where the noise in p_up enters: the recalibrator (rolling intercept) and the DLM component; slope-prior, DLM-shrinkage, vol-only-HMM and pooling fixes do NOT help, tie-consistent target + tight intercept prior removes the penalty but creates no skill (DESIGN CHOICE, sub-optimal)

- owner: quant  
- location: backend/quant/calibration.py:38-89; backend/quant/engine.py:330-333 (cal_sigma_a=0.12, cal_window=300); backend/quant/dlm.py (tau 0.025)  
- reviewer: review:math

**Description.** calibration.recalibrate re-estimates a logit intercept every day on a 300-origin window (lambda 0.995, prior sd 0.12, tempered 1/h): effectively a noisy rolling base-rate tracker that cannot tell the raw probability is noise (with pure-noise p_raw the posterior slope stays 0.868/0.973/0.994 at h=1/5/20 because of prior b~N(1,0.35^2) and 1/h tempering). The DLM component (14 features, prior sd 0.025) is the noisiest input: sd(p)=0.021 at h=1, 3-8x the others. Neither the data nor the prior supports within-stock timing information in p_up. Brier decomposition at h=1 (model vs causal base, per stock then averaged): noise cost E[d^2]=+4.0e-4 (level 2.1e-4 + within-stock variance 2.0e-4) minus signal gain 2E[d(y-b)]=+1.4e-4.

**Repro.** python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b10_noise_sources.py ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b14_components_dev.py ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a7_calib.py (slope under pure noise) ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m4b_slope_hac.py ; variant grid /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b13_grid.py all (output logs/b13_grid_base.log); vol-only HMM: /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/p4_cfg_run_shipped.py hmm_vol then WFDIR=<that wfcache_hmm_vol> b13_grid.py; pooled regression /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b16_pooled_dlm.py. All with TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math.

**Evidence.** Within-stock sd of p: raw 0.0068 -> calibrated 0.0142 (h=1), 0.0068 -> 0.0105 (h=5), 0.0095 -> 0.0110 (h=20); logit shift sd over time 0.043/0.023/0.015. Raw ensemble delta Brier vs causal base (x1e-4, NW t): dev -1.82 (-1.74) / -1.90 / -9.01, test -0.36 (-0.65) / +0.49 / +0.29 at h=1/5/20; shipped (calibrated): dev +4.36 (1.21) / -1.32 / -8.77, test +2.78 (1.60) / +0.97 / -0.75. Components, h=1 test: garch +0.02, hmm -0.66, dlm +2.81 (t 1.69), naive +0.07. HAC slope of (y-base) on (p-base), within-stock: raw +0.29 (se 0.49), -0.45 (1.10), -0.62 (1.53); shipped -0.12 (0.24), -1.25 (0.87), -1.51 (1.53): no evidence time variation of p carries information. REJECTED (dev and test, delta Brier vs causal base): slope prior centred at 0.7/0.4/0 -> test h=20 +6.2/+16.9/+38.3e-4 (dev also worse at h>=5); DLM tau 0.010 -> raw h=1 test -0.13 vs -0.36; vol-only HMM (hmm_mean_kappa=1e9) -> raw h=20 test +3.73 vs +0.29; pooled cross-stock ridge/OLS fitted on dev -> test OOS R2 +0.0006/-0.0012/-0.0035. WORKS (chosen on dev; sigma_a dev ranking monotone: 0.06/0.03/0.015/0.005 = -1.66/-1.81/-1.83/-1.84e-4): non-tie target + sigma_a=0.03 -> test -0.47 (t -0.77) / +0.44 (0.33) / -0.09 (-0.01); empirical-Bayes 'base + 0.5*(p_raw - mean p_raw)' -> test -0.41 (-1.24) / -0.57 (-0.69) / -1.88 (-0.70), same sign on dev (-0.44/-0.12/-1.22), not significant. Honest conclusion: matches the causal base rate; no structural change tested yields demonstrable skill.

**Proposed fix.** (1) Calibrate on the event that is evaluated (M-01). (2) Shrink the intercept hard (sigma_a ~0.03) or anchor to a long-memory base rate: p = b_t + s*(p_raw - mean p_raw), s~0.5 (hierarchical shrink to a slowly varying mean). (3) Do NOT centre the slope prior below 1 and do not expect stronger DLM/HMM shrinkage to help (tested). (4) Report model-vs-causal-base skill as the headline (M-02).

**Verifier:** real=False severity=low  
M-03 describes a real mechanism, but it is not a defect. It is a documented, conservative design choice (the calibration.py docstring explains the tight prior and the 1/h tempering). The finding itself concludes that no tested fix gives demonstrable skill. Mechanism: with pure-noise p_raw the posterior slope stays at about 0.87/0.97/0.99 for h=1/5/20, as the reviewer says. The recalibrator's rolling intercept also adds noise to p_up. I checked its size with my own simulation: pure-noise p_raw around a constant true base rate of 0.52, 20 replications, Newton recalibrate with window 300. The Brier penalty versus the true base rate is about +2.6e-4 at h=1 with sigma_a=0.12. That compares with +0.9e-4 for the raw p. At h=5 the penalty is +0.46e-4 versus +0.35e-4, and at h=20 it is +1.41e-4 versus +1.43e-4. A penalty of 1 to 3e-4 on a Brier of 0.25 is economically negligible. It is also not statistically significant on real data: the reviewer's own test numbers are +2.78e-4 with a Newey-West t of 1.60 at h=1, and under 1 in absolute terms at h=5/20. No proposed fix produces skill. The reviewer's own 'WORKS' variants are all non-significant (t about -0.8 to -1.2), and the dev/test signs are inconsistent for the raw p_up. Slope-prior, DLM-shrinkage, vol-only-HMM and pooling alternatives were all rejected. The calibrated p_up is still a valid, causal (matured-only), identity-regularised probability whose deviation from the base rate is about 1 percentage point. DESIGN.md section 9 already states 'no demonstrable edge' as the headline and the UI already shows an amber no_edge banner. The remaining useful parts (calibrate on the tie-consistent event, report skill versus the causal base) are already tracked as M-01 and M-02. As a standalone item this is an informational observation, not an actionable bug.

**Verifier refined fix.** No code change required. If anything is done, make it a minor optional tweak. Lowering cal_sigma_a from 0.12 to about 0.03 in backend/quant/engine.py (default cfg, line 92) trims the h=1 noise cost only marginally. It would not produce skill and would touch tests that pin calibration behaviour (tests/test_quant*). The substantive items belong to M-01 (calibrate on the tie-consistent event) and M-02 (report skill versus the causal base).


## M-04 [medium] Calendar time is ignored: bars after weekends/closures carry more variance than GARCH/EWMA/HMM assume (Monday +14%, post-holiday z^2 2.6, 21.7% interval miss), explaining 68% of the h=1 coverage excess

- owner: quant  
- location: backend/quant/garch.py (variance_filter, hstep_*); backend/quant/features.py (ewma_variance, v97, sigma_ref); backend/quant/engine.py:_horizon_pool  
- reviewer: review:math

**Description.** All volatility models treat each bar as one unit of time. On development rows (before 2021-08, 16 full-history stocks, 70,031 bars) E[z^2] = 0.957 + 0.155*(weekend-only gap) + 0.727*(skipped weekdays); refitted on the untouched test rows it is 0.980 + 0.167 + 0.735, so the effect replicates out of sample. Effect is mainly h=1 (and h=5); at h=20 it is diluted and not monotone. DEFECT (missing structure), partly mitigated by symmetry of other errors.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m2_cov_decomp.py (calendar/shock decomposition), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b7_gap_dev.py (dev-fitted coefficients, test replication), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/p1_eval.py gap_mean1 (prototype before/after), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m1_0050_6488.py and m3_6488.py (the two Kupiec cases), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m28_dev_scale_stopgap.py, /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m15_oracle_adjust.py, /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b1_sim_world.py 6 (nominal coverage in simulated worlds), /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m9_short_history.py. Prototype source: /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/proto/gap/backend/quant/engine.py (26-line diff vs shipped).

**Evidence.** Pooled h=1 test rows (78,372; 63 stocks), 90% interval miss rate (nominal 10%): regular bars 10.56% (77.8% of rows), Monday 12.42% (18.1%, var(z)=1.13), post-holiday 21.67% (4.1%, var(z)=2.57); contribution to the pooled 1.35pp excess: Monday 0.44pp, post-holiday 0.48pp, regular 0.43pp. Remaining excess: 20 market-wide shock days of 1,249 carry ~0.6pp (regular-day miss 10.56% -> 9.95% without them; daily cross-sectional miss-share variance is 7.2x the independent-binomial value, so pooled/universe p-values overstate evidence; per-stock Kupiec at h=1 rejects 25/63 at 5% while h=5 rejects 3/63, h=20 5/63). Prototype (proto/gap engine; bar multiplier (b0 + 0.155 wk + 0.727 skip)/b0, horizon-averaged; coefficients dev-only), untouched test window: h=1 cov50/80/90/98 0.491/0.787/0.886/0.966 -> 0.501/0.796/0.893/0.968, pinball -0.152e-4 (-0.4%, NW t -2.78), Winkler90 -0.92e-3 (t -2.95), dev t -4.2; h=5 cov90 0.894 -> 0.904, pinball -0.14e-4 (t -1.14); h=20 test pinball -0.07e-4 (t -0.39) but dev +0.37e-4 (t +1.87), i.e. no help; Kupiec rejections at h=1 only 25 -> 24 of 63. Caveat: the prototype uses the realised future trading calendar (scheduled holidays are known in advance, typhoon closures are not). 6488: 2026 volatility explosion (realised rms 5-7% vs predicted 4-6%) gives a 29.9% miss rate in 2026 and regular-day Kupiec p=0.0003; 0050: all-days p=0.013 but regular days p=0.31, excluding 20 shock days p=0.85. Oracle re-centring leaves cov90 unchanged (0.8865 -> 0.8866); width multiplier needed +5.5%/+2.2%/+3.7% (h=1/5/20); a global multiplier chosen on dev (1.020/0.969/0.942) moves test cov90 0.8865->0.8918 (h=1) but 0.8944->0.8854 (h=5) and 0.8908->0.8737 (h=20). A causal EWMA of matured z^2 (dev-selected half-life 60, prior 50, cap 16) improves h=1 pinball (-0.115e-4, t -3.2) but not coverage (0.8861->0.8871) and does nothing at h=5/20.

**Proposed fix.** Introduce a trading-time clock: per-bar variance multiplier m = b0 + b_w*1[weekend gap] + b_s*skipped weekdays (fit on development rows, shrink to the pooled value, normalise by b0), applied to GARCH/EWMA/HMM scales and averaged over the forecast window using only the published exchange calendar (data.next_trading_day / holiday JSON), not realised dates. Do not use a static width multiplier (fails out of sample at h=5/20). Treat market-wide shock days with a jump/tail component (see M-07) rather than inflating widths.

**Verifier:** real=True severity=low  
The core finding replicates, but it is smaller than described, so I rate it low rather than medium.

What is real:
- The engine has no calendar awareness. Every bar is one time unit, and `grep` finds no use of calendar gaps in `backend/quant`.
- I re-ran `m2_cov_decomp.py` and got the same numbers: pooled h=1 test miss 11.35%, regular 10.56%, Monday 12.42%, post-holiday 21.67%, contributions 0.44/0.48/0.43pp.
- I re-ran `b7_gap_dev.py` and got the same regression: dev E[z^2] = 0.957 + 0.155*weekend + 0.727*skip, test 0.980 + 0.167 + 0.735. The effect holds out of sample.
- I re-ran `p1_eval.py gap_mean1` and got the same prototype result: h=1 test cov90 0.886 -> 0.893, pinball -0.4% (NW t -2.78), and no help at h=20.
- My own regression from scratch on the walk-forward outputs (84,625 h=1 rows, all `*_base_1250.pkl` files) gave 1.00 + 0.135*weekend + 0.659*skip.

Why it is not medium:
- The 68% share is overstated. It counts the whole miss rate of Monday and post-holiday bars. The excess relative to regular bars is (12.42-10.56)*0.181 + (21.67-10.56)*0.041 = 0.34 + 0.45 = 0.79pp, about 59% of 1.35pp. The prototype actually recovers about 0.65pp of coverage.
- The post-holiday effect is dominated by a few market shocks. Mean z^2 for skip=2 is 5.5 (743 rows, which looks like Apr 2025), and 2020 closures reach 40. After removing the 20 worst days, the group means are Monday 1.02, post-holiday 1.54 and regular 0.94. The medians are 0.346, 0.530 and 0.324.
- The Monday effect is weak. The date-clustered SE on the weekend coefficient is 0.084 against 0.135, so t is about 1.6. The skipped-weekday coefficient has SE 0.256 against 0.659, so t is about 2.6.
- The per-year Monday mean z^2 is not stable, ranging from 0.72 to 1.35.
- The h=5 gain is not significant, and the h=20 dev result is negative.
- The prototype uses the realised future calendar, which is a mild look-ahead for typhoon closures.

I found no sign of a deliberate design choice or any documentation of this. It is a genuine but modest missing-structure miscalibration, concentrated in about 22% of h=1 rows.

**Verifier refined fix.** Treat this as optional, since the gain is only about 0.4% pinball and 0.7pp of coverage at h=1.

If it is done, the minimal change is in backend/quant/engine.py (_horizon_pool or the per-horizon scale step, not garch.py):
- For each origin i and horizon h, compute the horizon-average multiplier M_h(i) = mean over the next h bars of (1 + a*weekend_gap + c*skipped_weekdays).
- Use a ≈ 0.16 and c ≈ 0.75, shrunk toward the pooled values. Normalise by b0 ≈ 0.96-1.0, and use b0 itself rather than 1 if the baseline is not already centred.
- Scale sigma and the quantile deviations from the median by sqrt(M_h(i)).
- Get the future calendar from known weekdays plus the TWSE holiday list already fetched in data.py. Fall back to weekend-only when the list is unavailable. Do not use realised dates, because typhoon closures are not known in advance and this would break the truncation invariance and no-look-ahead rule.
- Apply it only at h=1, and at h=5 if desired. Skip h=20, where it does not help.
- Fit the coefficients on development rows with robust or winsorised z^2, because the OLS fit is dominated by a few shock days. The Monday term (t ≈ 1.6 after clustering) could be dropped, keeping only the skipped-weekday term.

Side effects:
- The walk-forward numbers change, so the golden-value assertions in tests/test_quant_*.py and the evaluation tests that depend on the engine may need updating.
- The quant module would need a calendar or holiday input. PriceData has none, so this adds an interface parameter or a dependency on data.py.
- The truncation-invariance test needs the future calendar to come from a static schedule, not from `data.dates`.
- The live forecast fan chart and the frontend are unaffected apart from slightly wider Monday and post-holiday intervals.


## M-05 [medium] Engine PIT is F(R_obs) with no randomisation for discrete outcomes: the app's PIT-KS test false-alarms (16 of 63 stocks p<0.01; correlation of KS distance with tie share 0.75)

- owner: quant  
- location: backend/quant/engine.py:340; backend/evaluation/metrics.py:510-530  
- reviewer: review:math

**Description.** engine._combine stores pit = pool.cdf(raw realised R). With unchanged closes (6.8% of h=1 rows, up to 25%) and tick-rounded returns, PIT has a point mass at F(0) and is not uniform even for a perfectly calibrated forecaster; metrics.pit_ks_pvalue and the histogram then flag miscalibration that is a rounding artefact. The proper PIT for a rounded price is U[F(R-half tick), F(R+half tick)]. 0.09% of rows (27 at exactly 0, 40 at exactly 1) are also limit-move point masses of the truncated h=1 predictive. DEFECT in the diagnostic; the engine itself is fine.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m6_tie_world.py 0 1 2 3 ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a8_pit_ties.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m14_pit_masses.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/b15_multi_h_shape.py (PIT deciles).

**Evidence.** Real data h=1: app PIT-KS p<0.01 for 16/63 stocks (expected ~0.6); tick-randomised PIT (grid-interpolated CDF, a8) 8/62; corr(tie share, KS D) = 0.75 (engine) vs -0.20 (randomised); PIT decile 5 mass 0.14 (dev) / 0.122 (test) vs 0.10 expected; 2412 KS p=1e-13 (25% ties). Dropping tie rows is NOT a fix (35/63 reject). Controlled test with the exact pool CDF (tick-rounded home world, 4 seeds): shipped KS p 9e-5/3e-8/0.048/5e-12 vs randomised 0.89/0.67/0.19/0.08.

**Proposed fix.** Emit a randomised PIT in the engine: pit = F(ln((c'-tick/2)/c)) + U*(F(ln((c'+tick/2)/c)) - F(ln((c'-tick/2)/c))) with a seeded RNG (c = origin close, c' = realised next close; kind-specific tick table); keep the raw F(R) as a separate field. For h>1 the same rule applies with the tick of the terminal price.

**Verifier:** real=True severity=medium  
Confirmed by code reading and by re-running the controlled test. engine._combine (engine.py:340) sets pit = pool_o.cdf(raw realised R) with no randomisation. Realised returns come from tick-rounded closes, so unchanged closes (R == 0) put a point mass at F(0) and the PIT is not uniform even for a perfect forecaster. metrics.pit_ks_pvalue (metrics.py:519-530) and pit_histogram (metrics.py:510) then consume that PIT without correction. report.py:195-196 passes both into the backtest report, and frontend/view_backtest.js:354-356 and :393 show them to the user. The UI text says p < 0.05 means the predictive distribution has a systematic gap relative to reality, so the false alarm is user-facing and misleading.

I re-ran m6_tie_world.py for seeds 0 and 2. The world is the model's own home world (GJR-GARCH-t with closes rounded to the Taiwan tick grid), so the forecaster is correctly specified.
- Seed 0 (15.7% ties): shipped KS p = 8.7e-5, tick-randomised p = 0.887.
- Seed 2 (8.3% ties): shipped KS p = 0.048, tick-randomised p = 0.19.
- These match the reported values for the same seeds (9e-5 and 0.048 shipped, 0.89 and 0.19 randomised).
- 90% coverage is fine there (0.907 and 0.900), so the PIT-KS rejection comes from the rounding artefact and not from model error.

Scope and severity: this is a diagnostic defect only. Forecasts, quantiles and p_up are untouched, so it is not high. It does mislead the user on calibration, which is the app's central question, so I keep it at medium. I did not re-run the real-data counts (16/63, correlation 0.75). I did not look for a faulty premise there, and the mechanism is a standard one.

Related point, not part of M-05: the fix does not change the real 83-90% interval-coverage shortfall. That shortfall is measured with raw R against quantiles and is a separate issue.

**Verifier refined fix.** Preferred fix, in backend/quant/engine.py _combine, line 340, when building pit:
1. Keep the existing value as a new field, e.g. res['pit_raw'] = pool_o.cdf(raw_r). Add the new field to the HorizonForecasts construction at engine.py:409, or drop it if nothing needs it.
2. Compute the terminal price c1 = c0*exp(R). Use ctx.close or ctx.Rraw to get the origin close c0 and the terminal close c1 at origin+h, and a tick function tick(c) with the Taiwan table: <10: 0.01, <50: 0.05, <100: 0.1, <500: 0.5, <1000: 1, else 5.
3. Compute lo = clip(ln((c1 - tick(c1)/2)/c0), LOG_LIMIT) and hi = clip(ln((c1 + tick(c1)/2)/c0), LOG_LIMIT). For h = 1, clipping to LOG_LIMIT gives the truncated-CDF values F(lo) and F(hi), which handles the 0.09% limit-move point masses at 0 and 1.
4. Set pit = F(lo) + U*(F(hi) - F(lo)) with U from np.random.default_rng(fixed seed). Seed it per (code, h) or use a fixed constant so output stays deterministic across runs.
5. Use the tick of the terminal price for h > 1. The randomisation barely matters there, because the tick is small relative to the h-day spread.

Side effects:
- Engine and quant tests that assert exact PIT values or PIT equal to cdf(R) will need updating. Search the tests for 'pit' in tests/test_quant*.py and tests/test_evaluation*.py.
- If the engine does not hold a kind-specific tick table, either add one in config.py or pass in the tick from the data layer. Flag this for ETFs and stocks that use different ticks, since ETFs have different tick rules, e.g. 0.01 below 50.
- The PIT is consumed in evaluation/report.py:195-196 and the walk-forward and backtest paths that store pit. Check those paths for fixtures that rely on determinism.
- pit_hist and pit_ks_pvalue need no code change, because randomised PIT is U(0,1) under calibration.
- The coverage and Brier numbers are unaffected.
- Cheaper alternative if the engine should stay untouched: add a footnote to the UI text saying the PIT test is biased at high tie shares. This only mitigates the problem and does not fix it. Dropping tie rows is not a fix, since 35 of 63 stocks would still reject.


## M-06 [medium] HMM 'regimes' are not persistent and their labels flicker: mean dwell 4.6-9 days, argmax regime changes on 31.7% of days, calm-state label flips between 盤整 and 多頭 on noise

- owner: quant  
- location: backend/quant/hmm.py:357 (prior diag 0.95, a0=10), :411-442 (regime_labels, trend_threshold 0.04)  
- reviewer: review:math

**Description.** The fitted transition matrices have diagonals 0.80/0.74/0.79 on average (min 0.38) despite the 0.95-diagonal prior (a0=10 pseudo-transitions against ~1000 observations), so the three states classify the day's noisy Parkinson log-range rather than persistent regimes. regime_labels() decides the trend word from mu/sigma > +-0.04 while regime means are shrunk to the drift (kappa=500), so the calm state sits on the threshold and its label flips between 低波動盤整 and 低波動多頭. The UI regime note ('目前最可能處於...機率 52%') therefore changes every ~3 days. DEFECT in the displayed regime story (the h-step mixture itself is exact, M-0x checks).

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m10_regime_jumps.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a9_hmm_hstep.py (label stability section); transition matrices: python3 -c reading the 'regime.transition' field of /home/gary/stock/data_cache/forecast/*.json.

**Evidence.** 64 live forecasts: A diagonal mean (calm, mid, high) 0.802/0.739/0.786, min 0.378/0.424/0.531, expected dwell 9.0/4.6/8.8 days; 2330 live probs [0.40, 0.52, 0.08]. 62 stocks x 1,249 test rows: max |dP| day-to-day mean 0.29, argmax-regime flips on 31.7% of days (refit-boundary days 31.2%, others 31.7% -> not caused by refits). 2330: label tuple changes at 14 of 37 consecutive refits (4 distinct tuples); 6488 1 of 7.

**Proposed fix.** Make states persistent: strong sticky prior (diag >=0.98, a0 in the hundreds) and/or fit on EWMA-smoothed log-range; label by volatility tier only (drop the drift word because drifts are shrunk to ~0) or add hysteresis (switch label only when probability >0.6 for 3 consecutive days); report regime probabilities smoothed over time.

**Verifier:** real=True severity=low  
The facts reproduce, but this is a display and labelling problem, not a forecast error. The h-step mixture is exact and the HMM is the honest MLE/MAP fit, so I downgrade medium to low. Part of the proposed fix does not work as described.

(1) Low persistence is real. The 64 live forecast files have mean transition diagonals of 0.802/0.739/0.786 and a minimum of 0.378. That gives an expected dwell of about 5, 3.8 and 4.7 days when computed from the mean diagonal. The reviewer's 9.0/4.6/8.8 is the mean of per-stock dwell times, so the figures differ by method only. Both are far from the 0.95 prior, so the likelihood overrides the weak prior (a0=10 against about 1000 transitions).

(2) Argmax flicker is real. Across 72 cached walk-forwards, the argmax regime changes on 31.3% of days (84,625 transitions). About half of those flips are between states 0 and 1 only (16.5% of days). This matches the reviewer's 31.7%.

(3) Label instability is real. On a fresh 2330 refit sequence (36 consecutive refits, cold fits without anchor warm-starts) the label tuple changes 9 times and there are 5 distinct tuples. The reviewer found 14/37 using the live engine's warm starts and anchors. The calm state's mu/sigma ratio ranges from -0.008 to 0.049, which straddles the 0.04 threshold in hmm.py:424. Mean returns are shrunk to the drift (kappa=500), so the "多頭" or "盤整" word is essentially noise. It is also misleading: the repo's own findings show no edge, yet the UI can say "低波動多頭".

(4) The proposed persistence fix only partly works. I patched a copy of hmm.py with a stickier prior (outside backend/). With diag 0.98 and a0=300, the fitted diagonal rises to about 0.94 and the argmax flip rate falls from 28.8% to 17.4% (250-day filter). The log-likelihood drops by about 23 nats. With diag 0.99 and a0=500, flips are still 14.6%. So the flicker is mostly intrinsic to a noisy log-range emission, and a sticky prior cannot remove it. A sticky prior does not fix the label problem. Calm-label changes across refits went from 1 of 36 (base) to 3 of 36 and 7 of 36 with the stickier priors, and tuple changes were 9, 5 and 11 of 36. The label flip comes from the ratio sitting near the threshold, not from the transition matrix.

The drift word is the minimal robust fix. A stickier prior is optional, it trades fit for readability, and it changes the mixture forecast and therefore the metrics.

**Verifier refined fix.** Minimal, effective change: stop naming the drift direction from the shrunk state means. In backend/quant/hmm.py regime_labels(), replace the ratio test (mu/sd against 0.04) with either (a) volatility tier names only ('低波動', '中波動', '高波動', plus '極低波動'/'極高波動' for K=4), or (b) keep the trend word only when |mu_k/sigma_k| exceeds a much larger threshold (about 0.10, well above the shrunk range of 0.00 to 0.06). Optionally add a tolerance band around the threshold so the label cannot flip when the ratio moves by a small amount between refits.

Do not rely on a stickier prior (diag 0.95 to 0.98, a0 10 to 300) to fix the labels. It lowers the flip rate from about 29% to about 17%, costs about 23 nats of log-likelihood, changes the h-step mixture and so the walk-forward metrics, and does not stabilise the label word.

If the persistence is wanted for display only, smooth the shown probabilities with a short EWMA (about 3 days) in engine.py:521-532 and leave the forecast pi untouched. Alternatively, show the regime note with the probability vector and drop 'current' when max(prob) < 0.6.

Side effects: tests/test_quant_hmm.py:209-211 asserts the exact label strings, including 低波動多頭, 多頭穩定, 高波動反彈, 空頭修正 and 低波動緩跌. tests/test_front_stock.py:69 expects '低波動多頭' and '多頭穩定'. Fixtures and the picks market.regime strings use 多頭穩定, though those come from fixtures. frontend/view_method.js:286 describes the naming, so its text would need to change. Option (a) breaks the label tests. Option (b) with a 0.10 threshold mostly keeps the strings, because the test params presumably have large mu."


## M-07 [low] Tail thickness: Student-t floor nu>=4.5 binds in 32% of GARCH refits and the fitted process has infinite 4th moment in 70% of final fits; 98% intervals cover 96.6% and the 99% quantile is exceeded 2.05% of days (DESIGN CHOICE with measurable cost)

- owner: quant  
- location: backend/quant/garch.py:54 (NU_MIN), :231-277 (hstep_moments); backend/quant/ensemble.py (SkewT.from_moments nu_min)  
- reviewer: review:math

**Description.** garch.NU_MIN=4.5 keeps E[z^4] finite but binds often; an unconstrained fit gives nu=3.7-4.1 with a tiny likelihood gain (+0.3..+6 nats on 2500 obs), so the likelihood does not demand it, yet empirical standardised-residual tails are 10-60% heavier than the fitted model even in sample. hstep_moments' finite-h kurtosis is exact for the model but psi>=1 (infinite 4th moment) in 44/63 fits, so the aggregated kurtosis is an h-dependent finite-horizon number and 24% of the h=20 predictives clip to nu=4.5. Effect on the 90% interval is small (+5.5% width at h=1); it matters for 1%/99% quantiles (WalkForward grid only; live API shows 5-95%).

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a11_nu_floor.py ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m25_tail_universe.py ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m30_psi.py ; /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/c1_universe_runner.py 2 then c1_summary.py (refit statistics).

**Evidence.** c1 sweep: nu<4.6 in 2083/6550 refits (31.8%; 2609 and 2353 94/94); final-window fits 22/63 pinned. Empirical vs model-implied P(|z|>2/3/4.5): 0.0545/0.0134/0.0025 vs 0.0492/0.0115/0.0021 (ratio 1.11/1.16/1.22); pinned stocks 1.09/1.31/1.62. Walk-forward test coverage cov98 0.9657 (h=1), 0.9725, 0.9690; below-1% 1.37%, above-99% 2.05% at h=1. Model excess kurtosis fed to nu=4+6/k: median 6.31/3.69/4.82 at h=1/5/20.

**Proposed fix.** If extreme quantiles matter: blend an EVT/Hill tail beyond the 5/95% level or allow nu down to ~3.2 with an aggregator that does not rely on a finite 4th moment (simulation or stable-law scaling); surface the 'nu at floor' flag (see M-09).


## M-08 [low] Warm-started HMM refits (single start, 60 iterations) can be materially under-converged and the converged=False flag is never surfaced

- owner: quant  
- location: backend/quant/hmm.py:329-376; backend/quant/engine.py:234-240 (hmm_fit warm start)  
- reviewer: review:math

**Description.** For refits between anchors fit_hmm warm-starts from the last anchor with max_iter=60. 149/6550 (2.3%) of refits end converged=False. Re-running cold with 500 iterations shows some are materially sub-optimal; the flag is dropped.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a12_hmm_nonconv.py

**Evidence.** 3661 at rows 2604/2625/2646/2667: log-likelihood gap +8.2/+11.6/+11.6/+8.3 nats vs a cold 500-iteration fit, max |dA| 0.09-0.14, h=20 HMM component sd differs 12.9%/16.5%/16.0%/13.7%; 1101 gaps +2.3..+4.8 nats (sd 1-3%); 2883 +0.0..+3.6; 2618, 3017 <=+2.9. Stocks with the most flags: 3661 (8), 1101 (7), 2883/3017/2618 (6 each).

**Proposed fix.** Continue warm-started EM until converged (cap ~200 iterations, it is cheap: median 13 iterations) or fall back to the cold fit when the warm start hits the cap; expose n_nonconverged in the engine block.


## M-09 [low] Fit diagnostics and silent fallbacks are not surfaced; degenerate inputs produce confident-looking outputs (constant price: p_up 0.09/0.12/0.25 with sigma 3e-7, 'avoid')

- owner: quant  
- location: backend/quant/engine.py:164-194 (_fit_garch/_fit_hmm fallbacks), :546-560 (engine block); backend/quant/features.py:60  
- reviewer: review:math

**Description.** forecast_latest's engine block carries only has_market and MC settings. GARCH/HMM converged flags, nu-at-floor, near-unit-root (psi/phi>0.998 in 215/6550 refits), fallback usage and the number of returns altered by features.clean_returns are computed but never exposed. On degenerate histories the fits succeed silently (GARCH on all-zero returns reports converged=True, loglik 42026, omega 6e-16, nu 4.55).

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/c3_fallbacks.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/c2_nasty.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m31_degenerate_to_recommend.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/c1_universe_runner.py 2 && python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/c1_summary.py

**Evidence.** Sweep of 139 real runs + 36 synthetic cases: 0 exceptions except intended insufficient_history (<250 bars) / bad_price_data; GARCH non-converged 0/6550 and fallback 0/6550; HMM fallback 0/6550. Synthetic: constant price -> p_up [0.0911, 0.1202, 0.2462], sigma_ann 0.0, annotate_forecast gives signal=avoid with score -20,446; 250 zero-return tail bars -> sigma5 0.0021 vs 0.038 normal (score -2.5); linear ramp -> p_up 0.9999 clipped; alternating +-3% -> p_up 0.1147 at h=1. Volume NaN/inf/0/negative, split-like jumps, high==low, low>high, open=0 all give finite, monotone outputs (volume cases identical to baseline within 1e-4).

**Proposed fix.** Add engine.diagnostics = {garch_converged, nu, nu_at_floor, persistence, hmm_converged, n_iter, n_returns_cleaned, zero_return_share_250} to forecast_latest and WalkForward.config; mark forecasts 'degenerate' (and suppress p_up/score) when sigma_1d < ~0.1% or the zero-return share of the last 60 bars exceeds ~50%.


## M-10 [low] engine._simple_moments trims 0.125% of each tail: the reported sigma (std of the simple return) is biased low by 1-3%, by 49% in extreme cases

- owner: quant  
- location: backend/quant/engine.py:429-434  
- reviewer: review:math

**Description.** E[exp(R)-1] and std are computed from a 400-point midpoint grid of the quantile function. The grid drops the outer 0.125% tails, which carry a non-trivial share of variance for fat tails; sigma feeds the score denominator in recommend.annotate_forecast. VaR95 differs from the 5% ensemble quantile by up to 3.3e-4 (alignment rank mapping) - negligible.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a10_live_extras.py

**Evidence.** vs numerical integration of the exact skew-t density: sd ratio 0.990 (nu=6, sd_log 1.6%), 0.982 (nu=4.6, 1.6%), 0.988 (nu=5.5, 4%), 0.993 (nu=8, 10%), 0.969 (nu=4.6, 10%), 0.510 (nu=4.6, sd_log 20%, gamma 0.85); mean error <=1.5e-4. Cached live forecasts: |var95_1d - q05_h1| <= 3.3e-4, CVaR <= VaR in 64/64.

**Proposed fix.** Integrate E[e^R] and E[e^{2R}] with an exact or tail-extended quadrature (extend the u-grid to 1e-6..1-1e-6 with tail-aware weights) or use the SkewT moment generating approximation.


## M-11 [low] Latent: Pool.ppf safeguarded Newton fails (error up to 0.76 in CDF, 49 non-monotone rows) when active components differ in scale by more than ~60x; production ratios are <=4.9 so it is never triggered

- owner: quant  
- location: backend/quant/ensemble.py:216-259  
- reviewer: review:math

**Description.** The root is correctly bracketed by the component quantiles, but Newton steps accepted inside the bracket make negligible progress in thin-density regions and 60 iterations are not enough; there is no 'step must halve |f| else bisect' rule. SPECULATIVE for production (needs an unrealistic mixture), reported because the docstring promises |F(x)-u|<tol.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m11_pool_fuzz.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m11b_pool_debug.py ; python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m12_pool_production.py

**Evidence.** Fuzz 6 x 20,000 mixtures x 13 levels with component sd log-uniform in [1e-4, 0.5] and sparse Dirichlet weights: max |cdf(ppf(u))-u| = 0.758, 49 rows with decreasing quantiles; failures only when max/min component sd >= ~60 (0 failures among 1,706 rows with ratio <30, 8 of 4,879 below 100). Production: 8 stocks x 3 horizons x 1,250 rows = 30,000 pools, max |cdf(ppf(u))-u| = 1.1e-11, 0 monotonicity violations, max component sd ratio 4.9.

**Proposed fix.** Switch to a bracketed hybrid (Brent/ITP or Newton with a fallback to bisection whenever |f| does not at least halve) and raise max_iter to ~100; add a unit test with a 3-orders-of-magnitude scale mixture.


## M-12 [low] HMM M-step is not an exact coordinate maximiser of any penalised likelihood (variance update omits kappa*(mu-m0)^2 and the prior degrees of freedom), so EM is not monotone in the raw or the derived penalised objective

- owner: quant  
- location: backend/quant/hmm.py:214-231  
- reviewer: review:math

**Description.** _m_step updates mu with the Normal-Inverse-chi2 shrinkage (fixed kappa) but updates the variance as (ss + nu0*s0)/(w + nu0), ignoring kappa*(mu_hat-m0)^2 and +3 dof. Effect is immaterial (about 0.2% of the variance) but 'MAP-EM' monotonicity does not hold exactly.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/a3_hmm.py

**Evidence.** Real windows (2330, 0050, 6488, 2603, 3 refit rows each): raw log-likelihood decreases in 29-49 of 59 iterations in 5/11 windows (worst single drop 0.029 nats; expected for MAP-EM), penalised objective (derived NIX prior) decreases in 17-33/59 iterations in 4/11 windows by at most 0.0010 nats on log-likelihoods of ~2000-2800. Filtered probabilities and moments are unaffected at the 1e-15 level.

**Proposed fix.** Use var = (ss + kappa*(mu-m0)^2 + nu0*s0^2)/(w + nu0 + 3) for the NIX prior (or drop the mean prior from the objective claim); update the docstring from 'MAP EM' accordingly.


## M-13 [low] SPECULATIVE (7 highly correlated ETFs): ETF p_up is below the causal base rate and the realised up-rate at every horizon

- owner: quant  
- location: backend/quant/features.py:168-183 (shrunk_drift)  
- reviewer: review:math

**Description.** For 0050, 0056, 006208, 00878, 00919, 00929 (and 00940) the calibrated p_up is 1-6pp below the causal base rate and below the realised up-rate in a strong bull window; even the RAW ensemble is below the base. Likely the drift shrinkage prior (N(0.0002/day, 0.0004^2)) plus drift under-prediction in 2021-2026 (all assets: realised mean 0.0009/day vs predicted 0.0004). The ETFs track the market so the effective sample is ~1-2 independent observations; flagged for awareness, not as a proven defect.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m20_etf_calib.py

**Evidence.** Test window, ties dropped, delta Brier vs causal base (x1e-4): calibrated positive in 21/21 ETF x horizon combos (+3.2..+148), raw positive in 18/18 (+1.6..+59.3); e.g. 0056 h=20 mean p 0.594 vs base 0.606 vs realised 0.648; 00919 h=20 0.647 vs 0.665 vs 0.715. Stocks (63): +2.8/+1.0/-0.75e-4.

**Proposed fix.** Anchor p_up to a slowly varying causal base rate (M-03 'base + s*(p_raw - mean)') so asset-class level differences are learned; consider a kind-specific drift prior (ETF total-return adjclose includes dividends).


## M-14 [low] features.clean_returns silently clips returns beyond +-12% and zeroes beyond +-25%; the user is not told (young listings: 6805 has 10 clipped bars in 1,435)

- owner: quant  
- location: backend/quant/features.py:60-63  
- reviewer: review:math

**Description.** The model path is altered for 196 clipped and 21 zeroed bars over 68 symbols while the evaluation uses raw returns; for young OTC/TWSE listings most of the large moves fall in the model window (new listings, missing-bar gaps, adjustment artefacts). Documented as hygiene but not exposed.

**Repro.** TWQ_CACHE_DIR=/tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/cache_math python3 /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/math/m13_cleaning.py

**Evidence.** Counts over 68 symbols: 21 zeroed (>25%), 196 clipped (12-25%), last 2,500 bars 1 zeroed and 28 clipped; 165 bars above 10.5% (beyond the +-10% limit) overall; 6805: 10/1,435 bars (e.g. +23.2% on 2021-03-31); 6669: 9 in the last 2,500; 7769 -40.5% on 2025-04-07 (zeroed).

**Proposed fix.** Return the count/dates of cleaned bars in engine.diagnostics (M-09) and show a data-quality note when more than ~3 bars in the last 250 are altered.


## S-01 [medium] DESIGN 9(a) is statistically unsafe if implemented literally: the hindsight Brier baseline biases skill negative (confirms/quantifies C-01, M-02), the causal expanding-window baseline biases it positive and makes the Brier DM leg anti-conservative

- owner: evaluation  
- location: backend/evaluation/report.py:173,222; backend/evaluation/metrics.py:318-332 (probabilistic_summary), 129-144 (expanding_up_rate)  
- reviewer: review:stats

**Description.** [DEFECT vs binding rule 9(a) for the oracle part; DESIGN HAZARD for the proposed fix]. report.py:173 uses the verification-window up-rate (up.mean()) as the Brier/log-loss reference and again in the Brier DM leg (report.py:222): hindsight. train_majority (metrics.expanding_up_rate) IS causal (900 poison-the-future checks, 0 violations). The oracle reference is the best possible constant forecast, so every real forecaster scores <= 0 against it by construction; the artefact is about (mean(p)-u)^2/B0 plus noise and grows with h because outcomes overlap. Conversely a noisy expanding-window p-hat (n/h matured outcomes) is a worse constant than the truth, so any forecaster that merely knows the base rate 'beats' it: the literal 9(a) fix therefore turns a negative bias into a positive one and makes the Brier DM leg reject far above its level. Neither reference answers 'is there direction information beyond the base rate'.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_brier_trap_minimal.py ; python3 s3_brier_calibrated.py ; python3 s2_hyp_verdict_causal.py ; python3 a_baseline_real.py ; python3 s3_causal_baselines.py

**Evidence.** Real 62 symbols: mean Brier skill vs oracle -0.31/-0.60/-2.35% (h=1/5/20; median -0.26/-0.49/-1.65%) vs causal -0.12/-0.04/-0.01%; the causal reference removes 62/93/100% of the deficit; mean-bias term (pbar-u)^2/B0 = 0.19/0.40/1.90% of B0 (62/66/81% of the deficit); log-loss skill -0.23/-0.44/-1.70% vs -0.09/-0.03/+0.00%. Zero-information forecaster (true constant P(up), iid returns, n=1250, 1500 runs): skill vs oracle -0.08/-0.31/-1.21%, vs causal +0.10/+0.35/+1.47%; one-sided DM(Brier) rejects at 5%: 0/0/0% vs oracle but 15.1/26.8/25.9% vs causal (harness cells: 15.5-18.5/24.8-28.5/23-28%). Real quant engine on random walks (250 paths): DM vs causal rejects 0.8/13.6/23.2%; over 19,950 null runs the verdict leaves no_edge 0.025% (oracle) -> 0.11% (causal Brier). The engine's own nested 'naive' component is a size-correct reference (skill -0.10/-0.05/-0.03%, DM rejects 0/1.7/4.2% on null engine paths).

**Proposed fix.** Do not replace the oracle by the raw expanding-window rate. Report BOTH references (oracle = conservative bound, causal = optimistic bound, each labelled) and call skill only if significant against both, or use a model-internal nested causal reference (comp_p_up['naive']) or a long-history baseline; calibrate the Brier-difference null by simulation (block permutation) instead of the asymptotic DM; update DESIGN 9(a) accordingly.

**Verifier:** real=True severity=low  
The numbers reproduce, and the mechanism is as described. report.py:173 passes a hindsight constant (up.mean() over the verification window) to probabilistic_summary, and report.py:222 uses the same constant in the Brier DM leg. That contradicts binding rule 9(a), and the metrics.py docstring ("causal climatology p_ref") is wrong about what it receives. The oracle is the best constant forecast in-sample, so skill against it is biased negative by about (pbar-u)^2/B0. My re-run of s3_brier_trap_minimal.py (400 runs instead of 1500) gave, for a forecaster that knows only the true constant P(up): skill vs oracle -0.08/-0.26/-1.08% and one-sided DM rejection 0/0/0%, versus skill vs causal +0.10/+0.39/+1.51% and rejection 16.2/24.5/24.0% at h=1/5/20. The reviewer's figures (15.1/26.8/25.9%) are consistent with this.

I rate it low rather than medium for three reasons.
- The current code is the conservative choice. The oracle reference makes false "edge" calls essentially impossible. The reviewer's own null-run figures show the verdict leaving no_edge only 0.025% of the time with the oracle.
- The real-data effect is a mislabeled magnitude, not a wrong verdict. The headline "slightly negative Brier skill everywhere" is largely this artefact. The causal skill is about -0.12/-0.04/-0.01%, which is consistent with no edge, and the verdict is unchanged.
- The "anti-conservative DM" part is a hazard of the proposed literal fix, not a defect in the present code. It is also somewhat overstated. A forecaster that knows the true p really does beat a noisy expanding-window p-hat, so DM is rejecting a false equal-accuracy null. The problem is the interpretation as "direction edge", not the test's size. The related claim that a real engine on null data rejects 13-23% against the causal reference at h=5/20 is plausible but I did not re-run it.

The valid, actionable points are two. (1) The Brier and log-loss reference is hindsight, and it is neither labelled as such nor accompanied by the causal figure. (2) DESIGN 9(a) should not be applied by simply swapping in the raw expanding-window rate for the DM leg.

**Verifier refined fix.** Make the change additive and keep the verdict conservative.

1. In backend/evaluation/report.py around lines 171-176, build two references on Sx. `clim_causal = clim[Sx]` is the expanding_up_rate already computed at line 163. `clim_oracle = np.full(len(Sx), up.mean())`.
   - Call `M.probabilistic_summary(p, y, clim_causal)` for the headline `brier_baseline`, `brier_skill` and `log_loss_baseline`.
   - Add the keys `brier_baseline_oracle`, `brier_skill_oracle`, `log_loss_baseline_oracle` and `log_loss_skill_oracle`, computed with the same function against `clim_oracle`. These are additive, so no existing key is removed.
   - Also add a `log_loss_skill` field if one does not exist yet.
2. Leave the Brier DM leg (report.py:222) against the ORACLE reference, as it is now. It is size-correct, rejecting 0%. Add a second DM against `clim_causal` for display only. Set `ev["p"]["brier"] = max(p_oracle, p_causal)`, so the leg only counts when the model beats both. In practice that equals the oracle p-value. Do NOT replace the oracle with the raw expanding-window rate in the verdict leg, because the causal reference gives false positives of about 15-26% (h=5/20) for any forecaster that knows the base rate.
3. Fix the comment at report.py:171 and the docstring of `probabilistic_summary` (metrics.py:~318) to say which reference is which.
4. Update DESIGN 9(a) to say that the Brier and log-loss skill headline is causal, the oracle is reported as a labelled conservative bound, and the evidence leg requires significance against both.
5. Optionally add a size-correct nested reference, `comp_p_up['naive']`.

Side effects:
- The `probabilistic` schema at report.py:~498 needs the new optional keys.
- frontend/view_backtest.js:195 should show both baselines with labels (「因果氣候基準」 and 「事後樣本基準（保守上界）」).
- Evaluation tests that assert `brier_baseline == up.mean()` or the sign of `brier_skill` will change.
- The report text at report.py:384 should mention both references.
- The universe/picks evidence dict only reads `ev["p"]["brier"]` and `effect["brier"]`. effect["brier"] would then be the causal skill, so check any threshold on that effect size used by the verdict logic.


## S-02 [medium] Spearman IC (and its p-value) is mis-centred for every history-derived forecast: null IC is -0.03/-0.09/-0.19 at h=1/5/20, so displayed two-sided p-values are mis-sized (12-37% at nominal 5%)

- owner: evaluation  
- location: backend/evaluation/metrics.py:343-367 (spearman_ic); backend/evaluation/report.py:181,226-228,337-340  
- reviewer: review:stats

**Description.** [DEFECT]. Any forecast that is a smoothed function of past returns (naive drift, HMM regime drift, DLM mean - i.e. every real forecaster here) has a finite-sample Kendall/Stambaugh-type bias against realised returns of the same window. metrics.spearman_ic centres the null at 0 with sd 1/sqrt(n/h), so the displayed 雙尾 p (verdict head and bullets, 'ic_pvalue') is unreliable and mis-signed; the one-sided positive p used by the verdict is therefore conservative (bias shifts it by about -1.0/-1.4/-1.5 report-sigma), which destroys power of the IC leg at h>=5 (see S-04). The bias does not shrink with history length. The report docstring's '~1.7x too liberal' was measured for INDEPENDENT persistent signals only; for history-derived forecasts the two-sided size is 2.4-7x nominal. Verdict decisions are not wrongly triggered (one-sided positive rejections 0-2.7% on null).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_ic_bias_expanding_mean.py ; python3 s3_ic_vs_null.py ; python3 analyze_single.py eng_garch eng_normal ; python3 s2_bank_analysis_long.py ; python3 s3_signflip_null.py 6 20 1

**Evidence.** Self-contained: expanding-mean forecaster on iid returns (true IC 0): IC -0.044/-0.095/-0.182, two-sided p<5% in 32.0/27.9/24.9% (one-sided+ 0.1/0/0%). Real quant engine on random walks: IC -0.028/-0.090/-0.189 (500-bar history, 250 paths; two-sided<5% 15.6/22.4/24.0%) and -0.031/-0.086/-0.177 (3750-bar history, 60 paths; 16.7/26.7/25.0%); null sd 0.027/0.051/0.095 vs the report's 0.028/0.063/0.126. Sign-flip surrogates of 6 real stocks (real magnitudes, 20 each): null IC -0.009/-0.059/-0.160 (sd 0.030/0.057/0.101), implied two-sided size ~7/13/19%. Synthetic history-based forecasters: ewma_plain two-sided 22-37%, slow_drift 11-16%, ewma_mom 11-18%. Real 62 symbols: IC mean +0.002/-0.039/-0.091 and 10/6/4 'significant' two-sided (7+3-, 0+6-, 0+4-); e.g. 6488 h=5 IC -0.136 shown with p=0.03 although inside the null (18th percentile). Caution: against the synthetic random-walk null the real ICs look high (mean percentile 0.72-0.76, 19/14/16 of 62 above the null 95th pct) but against the sign-flip real-magnitude null they sit at the centre at h=1/5 (percentile 0.52/0.48; 0.77 at h=20, 6 stocks) - no evidence of skill, only of a wrong centre.

**Proposed fix.** Stop presenting the textbook two-sided IC p as a test of skill: centre/scale the IC null by simulation of the same forecaster on a no-predictability surrogate (sign-flip or GARCH bootstrap; table by h) or replace the IC and MSE legs by a nested-model OOS test (Clark-West: size 2.5-3.5% on the null engine bank vs 0.5-1.5% for the production DM). At minimum label the p as 'uncalibrated' and fix the docstring.

**Verifier:** real=True severity=low  
The statistical claim reproduces. I wrote an independent script (scipy.spearmanr plus the same t formula) for an expanding-mean forecaster on iid returns, where the true IC is 0. It gave mean IC -0.045/-0.095/-0.182 at h=1/5/20. The null sd is 0.024/0.053/0.102 against the report's nominal 0.028/0.063/0.126. Two-sided p<5% in 30/29/24% of paths, so the displayed two-sided IC p is mis-centred and mis-sized for history-derived forecasts. The reviewer's own script gave the same numbers (32/28/25%). The report.py docstring only mentions "~1.7x too liberal", which was measured for independent persistent signals. The impact is smaller than 'medium' for these reasons. (1) The verdict uses only the one-sided positive p (report.py:226-227, `p_one_sided_positive`). On the null this is conservative: 0-0.1% of paths fall below 5%. So the verdict cannot be wrongly triggered, and a conservative bias fits the project's stated design stance. (2) The loss of power at h>=5 is a separate issue (S-04), not a new harm here. (3) The harm is in display. The verdict text and bullets (report.py:340, 389) and the frontend KPI (view_backtest.js:199) show the 'two-sided p'. The frontend `sigChip` (view_backtest.js:126-131) raises the info chip '單項檢定顯著' whenever this two-sided p is below 0.05, whatever the sign of the IC. A negative IC such as 6488 h=5 at -0.136 (p=0.03) therefore carries a 'significant' chip, although it lies well inside the null (about the 18th percentile) and is not evidence of anything. That is misleading, but only mildly, since the verdict panel is authoritative and the chip tooltip says 'see the verdict above'. The defect is real, the label is misleading, and the verdict is unaffected. Severity is low.

**Verifier refined fix.** No change to `metrics.spearman_ic` or to the verdict logic; the verdict is unaffected and the one-sided leg is conservative. Make three minimal changes. (1) In frontend/view_backtest.js line 199, call `sigChip` for the IC KPI only when ic>0. For example, pass a p of null/1 when rt.ic<=0, or use the one-sided p (ic>0 ? p/2 : 1-p/2), so a negative IC never shows '單項檢定顯著'. (2) In report.py, change the docstring sentence (lines 33-35) to say that for history-derived forecasts (drift, HMM, DLM) the IC null is biased negative, about -0.04/-0.09/-0.18 at h=1/5/20, and the two-sided p is uncalibrated and about 2.4-7x nominal; change the wording of 'IC 雙尾 p' (lines 340, 389) to 'IC (p 值僅供參考，未校準)' or drop the two-sided p. (3) Optional and larger: replace the IC null with a simulated sign-flip or surrogate calibration table by h, or use Clark-West. That is a separate change that would alter p-values in tests/test_eval_report.py. Side effects of (1) and (2): the frontend tests (view_backtest) that assert chip text for negative IC, and any tests asserting the verdict/bullet strings containing '雙尾' (tests/test_eval_report.py), need updating.


## S-03 [medium] ECE warning ('校準偏差偏大 ... 預測機率不宜照字面解讀') fires for perfectly calibrated forecasters: ECE is dominated by binning noise that scales with overlap

- owner: evaluation  
- location: backend/evaluation/report.py:403-404; backend/evaluation/metrics.py:292-332 (reliability_table, expected_calibration_error)  
- reviewer: review:stats

**Description.** [DEFECT, false alarm]. ECE with 10 equal-count bins has an expected value of about 3.6% for n~1250 even for an exactly calibrated forecaster and the noise grows with the horizon because h-bar outcomes overlap; the fixed 5% trigger (report.py:403) therefore flags nearly every h=5 and h=20 report, including exactly calibrated models. The real symbols' ECE is statistically indistinguishable from the calibrated-forecaster noise floor at h=5/20, so the bullet conveys no information (and states something false about a calibrated model). The KPI/title also shows the raw ECE with no noise band.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_ic_vs_null.py ; python3 s3_text_small_h1.py ; python3 s2_single_summaries.py

**Evidence.** Exactly calibrated forecasters (cal_oracle, 3 DGPs x 400 runs): ECE mean 3.57/6.16/11.00%, p95 5.10/9.43/17.89% (h=1/5/20); ECE>5% in 5.9/69.9/86.1% pooled (garch 4.2/34/58; normal 7.5/89.5/100; t4 6.0/86/100). Real engine on random walks: 14/86/99% (GARCH) and 7/86/100% (normal). Real 62 symbols: mean ECE 4.25/6.38/11.58%, ECE>5% for 19/79/100% of symbols (every h=20 report carries the warning); only 11/1/1 of 62 exceed the null p95. Repro text: exactly calibrated h=20 forecaster gets ECE 15.7% and the warning bullet.

**Proposed fix.** Warn only when ECE exceeds an h- and n-dependent noise floor (simulated p95 for calibrated forecasts: 5.1/9.4/17.9% at n~1250, roughly 3.6% * sqrt(0.45h+0.55)) or replace by a calibration intercept/slope test with block-bootstrap CI; show the noise band next to the KPI and in the reliability chart.

**Verifier:** real=True severity=low  
Reproduced independently. report.py:403 warns whenever ECE exceeds a fixed 5%. ECE uses equal-count bins (about 10, up to n//30), so a perfectly calibrated forecaster has an expected ECE of about 0.4*sqrt(B/n), roughly 3.6% at n=1250. Overlapping h-bar outcomes inflate this noise further. In my own simulation (n=1250, exactly calibrated p, outcomes built from overlapping h-day windows, 300 runs each) the mean ECE was 3.6/4.6/7.4% at h=1/5/20. The 5% trigger fired in 4%/35%/87% of runs, so it is a false alarm for calibrated models at long horizons. The finding's own numbers are larger because its DGPs differ, but the direction is the same: the warning carries no information at h=20 and the text 'does not read literally' is not backed by evidence. Severity is low rather than medium. It is an advisory bullet, not a headline verdict, and the app's real models have no demonstrated edge, so telling users not to take the probabilities literally is practically harmless. The defect is a missing statistical noise floor and the ECE shown with no noise band. It is not a wrong headline conclusion. The reviewer's other statistics (real-symbol ECE, null p95) were not rechecked.

**Verifier refined fix.** In report.py near line 403, replace the fixed `> 0.05` test with an n- and h-aware noise floor. For example: `B = len(pr['calibration']); n = ev['n_scored']; floor = 1.4 * 0.40 * math.sqrt(B * (0.45*h + 0.55) / n); if pr['ece'] > max(0.05, floor): ...`. At n~1250, B=10 this gives roughly 5.0/7.9/15% at h=1/5/20, close to the simulated null p95 (the finding's own p95 figures were 5.1/9.4/17.9%). Optionally append the floor to the ECE sentence at report.py:386 (for example 'noise floor about X%') so the raw ECE is not shown alone. The `ece` value in the JSON contract and the other modules are untouched. Report-text tests for the ECE bullet may need a fixture update if they exercise it, but I found none that do. A larger change (a calibration intercept/slope test with block-bootstrap CI) is not needed for a minimal fix.


## S-04 [medium] Verdict has very little power and never discloses it: no_edge is also what moderate real edges produce (h=20: IC 0.33, +7.5-10pp accuracy, is called no_edge in 50-75% of runs); engine+evaluation together only see lag-1 autocorrelation >= 0.15 at h=1

- owner: evaluation  
- location: backend/evaluation/report.py:240-270 (assess_evidence), 326-373 (make_verdict), 7-33 (docstring); README.md:177  
- reviewer: review:stats

**Description.** [DESIGN CHOICE (conservative) with an undisclosed consequence = messaging defect]. Bonferroni over 4 tests at 1%, >=2 tests below 5%, direction test against the hindsight max-of-3 baseline with n_eff=n/h, oracle Brier baseline and the S-02 bias together leave the verdict almost blind beyond h=1, while the title (「沒有統計上顯著的預測優勢」), README (「沒有可靠的預測優勢」) and report notes never say what effect size could have been detected. Separate (a) what the verdict rule loses given perfect forecasts from (b) what the engine can learn. (a) Oracle-latent planted edges (forecaster is told the latent signal; 200-300 runs per cell; Gaussian / GARCH-t5): the IC leg alone detects 99-100% at h=1 IC 0.125 and 88-96% at h=20 IC 0.33, the verdict far less. (b) Real engine learning from prices only: lag-1 autocorrelation of returns phi=0.05/0.10/0.15 (population IC_1 0.05/0.10/0.15, ideal single-test t = 1.8/3.5/5.3): verdict flags 0/150, 4.7% (GARCH-t5 5.3%), 48% at h=1 and 0 of 450 at h=5/20 (population IC <= 0.07 there, t <= 1.1: undetectable by any method); latent slow drift sd 0.0008/0.0015 per day (oracle IC_h 0.05/0.12/0.20 and 0.10/0.21/0.36, t up to ~3): 0% and 1.3-6.0% (h=1), 0-0.7% (h=5), 0% (h=20). The report docstring's 'IC~0.2 found at 100%' is true at h=1 and 5 only.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_power_table.py ; python3 s3_analyze_engpow.py ; python3 mc_single.py s3pow_h20_0.33 200 normal,garch_t5 edge 20 0.0003 ic=0.33 ; (engine runs: python3 s3_engine_power.py ar1_n_0.10 150 ar1 0.10 normal)

**Evidence.** Verdict != no_edge (Gaussian/GARCH-t5), accuracy lift over base in brackets: h=1 IC 0.05 (+0.8/+1.3pp) 9/19%; IC 0.08 (+1.8/+2.6pp) 35/62%; IC 0.125 (+3.2/+4.6pp) 74/94%; IC 0.196 (+5.4/+7.7pp) 98.5/100%. h=5 IC 0.10 (+1.8/+2.2pp) 7/18%; IC 0.15 28/46%; IC 0.33 (+9.4/+11.3pp) 95.5/98%. h=20 IC 0.10 (+0.8/+1.7pp) 1.0/1.7%; IC 0.20 (+3.5/+5.3pp) 6/13%; IC 0.33 (+7.5/+10.2pp) 25/50.5% (direction leg only 6.5/9%). Approx. minimum detectable IC at 80% power: ~0.11-0.14 (h=1), ~0.25 (h=5), >0.4 (h=20). Real symbols: p_vs_coin<5% in 10/29/29% but p_vs_best<5% in 3/0/0% (see S-14).

**Proposed fix.** Add a detectability sentence/block to every report (minimum detectable IC and accuracy lift at 80% power for the actual n and h, from a precomputed table) and word the level as 'not detected' rather than 'no edge'; soften README 'no reliable edge' to 'no edge was detectable at this sample size'; consider replacing 2-of-4 plus Bonferroni at h>=5 by one pre-specified primary test per horizon (and fix S-02/S-07 first).

**Verifier:** real=True severity=low  
The power numbers reproduce, but the defect is only a wording and disclosure gap, so I rate it low, not medium. I re-ran the real pipeline (RP.build_backtest_report) in my own scratch copy with 150 oracle-latent runs per cell, Gaussian noise. At h=20, IC about 0.31 and accuracy lift +7.2pp, the verdict left no_edge in 24.7% of runs (finding: 25%). At h=1, IC about 0.117 and lift +3.3pp, it left no_edge in 73% (finding: 74%). The IC leg alone fired in 89% (h=20) and 99% (h=1), and the direction leg in only 5% (h=20), as the finding says. I also re-printed the stored power table with s3_power_table.py, and it matches the finding's numbers.

Why the severity is low:
(1) The low power is mostly statistical inevitability. With n of roughly 1250 and h=20, n_eff is about 60. A true IC of 0.33 gives t of about 2.5 for the single best test, so no valid test could do much better. The 4-test Bonferroni at 1% and the 2-of-4 agreement rule cost some power, but only on top of that limit. An IC of 0.33 at h=20 is also far larger than any plausible equity edge.
(2) The wording is literally accurate and does not claim the edge is absent. The title says 「沒有統計上顯著的預測優勢」, and the body says 「不能說這個模型有預測優勢」. DESIGN.md 198-200 documents the rule as conservative by design, and the module docstring states the size (about 0.06% false positives) and that IC about 0.2 is found at 100%.
(3) What is missing is disclosure. Nothing tells the reader which effect sizes could have been detected. The docstring's 'IC~0.2 found at 100%' holds only at h=1 and h=5; at h=20 the rate is 6-13%. The README line 「沒有可靠的預測優勢」 and the level key 'no_edge' can be read as proof of absence, which they are not. This is a real but mild messaging defect. It is not a bug in the statistics or a safety issue.
(4) The proposed 'replace 2-of-4 plus Bonferroni with one primary test' is a methodology change. It would trade false-positive control for power, and it depends on S-02 and S-07. I would not recommend it as part of this finding.

**Verifier refined fix.** Make a disclosure-only change and leave the verdict thresholds alone.

1. backend/evaluation/report.py, make_verdict, no_edge branch (the else/not_significant and not_consistent tails): add one sentence after the existing tail. Suggested wording: 「這只代表在目前樣本量下沒有偵測到優勢，並不證明沒有優勢；以 n_eff≈{ev['n_eff']} 估計，約有八成把握偵測到的最小排序相關(IC)約為 {x}。」 Take x from a small constant table keyed by horizon, for example {1: 0.12, 5: 0.25, 20: 0.40}, or compute it as about 2.8/sqrt(n_eff) for the single-test case and then widen it for the Bonferroni and 2-of-4 rules. The table is conservative and fits the simulation. Use the same sentence in make_notes.

2. Correct the docstring at lines 29-31: 'IC~0.2 found at 100%' holds only at h=1 and h=5, and at h=20 an IC of 0.33 is found in only about 25-50% of runs.

3. README.md line 177: change 「沒有可靠的預測優勢」 to something like 「在目前樣本量下沒有偵測到可靠的預測優勢（此驗證檢定力有限，尤其是 5 日與 20 日預測）」.

Side effects: tests that assert exact no_edge text or notes could break. Check tests/test_eval_report.py and the server and frontend tests that compare title/text/notes. Do not change LEVEL_TITLES['no_edge'], because DESIGN.md line 320 and the frontend schema use that string verbatim. Do not change the level key. Do not touch THRESH_STOCK or the 2-of-4 rule here, because that is a methodology decision tied to S-02 and S-07.


## S-05 [medium] Universe verdict cannot tell volatility / beta / Jensen exposure from skill and the score ranking hides a learnable edge; DESIGN 9.1 mandatory benchmark rows (mean sigma, mean beta, top-k-by-sigma) are absent (adds quantification to C-06)

- owner: evaluation  
- location: backend/evaluation/universe_bt.py:233-246 (spread_stats), 401-490 (evaluate_universe), 355-381 (notes); DESIGN.md 9.1  
- reviewer: review:stats

**Description.** [DEFECT vs binding rule 9.1 + statistical confound]. Three mechanical sources of 'significance' for a basket that merely ranks volatility: (i) market premium x beta (top basket EW-beta 1.80 on real data); (ii) the spread leg compares ARITHMETIC mean h-bar returns (spread_stats uses expm1(R)), so a vol sort earns a Jensen term sigma_h^2/2 even at equal LOG drift; (iii) the IC leg of a vol ranking collects the same premium. Real data: spread t=3.36/3.07/2.04 (h=1/5/20) falls to 1.39/0.24 after beta-neutralising (h=1/5) and 3.36->2.40, 3.07->2.02, 2.04->1.29 when measured in log returns; a pure 'sigma only' ranking reproduces it (t 3.06/3.58/3.32) with HIGHER CAGR/Sharpe than the model's basket at h=5/20 (69.7/61.2% vs 55.4/39.7%) while random ranking gives none; top-5 by score has sigma-rank 0.94-0.97, Spearman(score,sigma)=0.99/0.95/0.64. The real verdict stayed no_edge only because the IC and picks legs did not agree (luck). MC (30 stocks, common GARCH-t factor, beta rising with vol, zero idiosyncratic alpha, constant-forecast ranking = pure vol sort): the verdict calls it an edge in a share of null panels that grows with the market premium. Same ranking also destroys power for a real learnable edge. universe_bt.evaluate_universe returns none of: mean sigma / mean beta of the basket, top-k-by-sigma benchmark (DESIGN 9.1).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s2_analyze_univ.py 'size_*' ; PREM=0.0011 python3 s3_mc_univ_prem.py prem0.0011 60 30 capm const_true 1,5,20 seed0=9000 ; python3 s3_spread_beta.py ; python3 d_volrank_real.py ; python3 s3_univ_engine_power.py phi010 30 12 0.10 7000 && python3 s3_analyze_uengpow.py

**Evidence.** MC zero-skill vol ranking, verdict != no_edge: premium 0.04%/day (~10% p.a.) 7.3/6.7/6.0% (150 panels/cell; fresh seeds 10.0/1.7/10.0%), IC one-sided p<5% 32/23/34%, spread p<5% 55/49/61%; 0.08%/day (~20% p.a.) 21.7/38.3/26.7% (moderate+ 16.7/31.7/18.3%), IC 70-73%, spread 82-87%; 0.11%/day (~27% p.a., real EW CAGR 30.8%) 43.3/63.3/58.3% (moderate+ 35/57/43%), IC 88-92%, spread 90-95%; alpha vs EW -0.7..-2.6%/yr throughout. Equal LOG drift, no premium (Jensen only): spread p<5% 20.7/18.0/24.0% (h=1/5/20). Worlds without priced vol/beta and mu/sigma ranking: verdict 0.07-0.30%. End-to-end with the REAL engine and a planted idiosyncratic AR(1) in 12 stocks (30 panels): phi=0.15 h=1 verdict != no_edge 6.7% (score ranking, IC +0.007) vs 96.7% (mu/sigma ranking, IC +0.052, t 5.8); phi=0.10: 0% vs 36.7% (IC +0.025). Oracle-latent per-stock IC 0.03, h=1: 31% (score) vs 100% (mu/sigma).

**Proposed fix.** Implement 9.1 (rank_signal ranking) and add report fields: basket mean sigma and beta, top-k-by-sigma and random-k benchmarks; replace the spread leg by a beta-neutral log-return spread (regress on the same-window EW return) or drop it from the verdict family; compute the IC leg on beta/vol-residualised returns (or within sigma-buckets); add alpha vs EW (regression intercept, HAC t) to the verdict text.

**Verifier:** real=True severity=medium  
Confirmed from code and re-run. (1) DESIGN 9.1 is binding: universe_bt MUST rank by rank_signal = mu_log/sigma_log and the universe report must state basket mean sigma, mean beta, a top-k-by-sigma benchmark and the equal-weight benchmark. universe_bt.py still ranks by SC = (E[simple R]-cost)/sigma. grep finds no rank_signal anywhere under backend/, and the evaluate_universe report has no sigma, beta or sigma-benchmark fields. The module docstring and the 選股方式 note still describe the old cost-adjusted score. So the report omits the exposure disclosure the design requires. (2) The vol-ranking confound is real and measurable. On the real panel (58 stocks, 0050 excluded), Spearman(score, sigma) is 0.991/0.951/0.635 at h=1/5/20. The model's spread t is 3.36/3.07/2.04. A pure sigma-only ranking gives t 3.06/3.58/3.32, and its CAGR is higher than the model's at h=5 and h=20 (69.7% vs 55.4%, 61.2% vs 39.7%). A random ranking gives no spread. After beta-neutralising against the same-window EW return, the model spread t falls to 1.39 at h=1 and 0.24 at h=5. (3) The existing Monte Carlo output reproduces the false-positive claim. Setup: 30 stocks, common GARCH-t factor, beta rising with vol, constant forecast so the ranking is a pure vol sort, zero idiosyncratic alpha. The share of panels with a verdict other than no_edge under the current ranking: 43.3/63.3/58.3% at 0.0011/day premium and 21.7/38.3/26.7% at 0.0008/day. These match the finding. Under the DESIGN-9 mu/sigma ranking in the same panels it is 0/60 in every cell. alpha vs EW is negative throughout (about -1 to -2.6%/yr), while basket beta is about 1.5 against 1.0 for the DESIGN-9 basket. So most of the problem is the ranking choice, and the 9.1 fix removes it in this null. (4) Severity stays medium. On the real data the verdict is still no_edge, because the IC and picks legs do not agree, so no wrong claim has been shown. But in a bull market the verdict could report a vol/beta premium as a stock-selection edge, and a binding disclosure rule is unmet. The proposed beta-neutral spread and residualised IC go beyond what the design requires. They are optional hardening, not part of the defect. The planted-AR(1) power claims (96.7% vs 6.7%) were not re-run, but the ranking-choice mechanism is the same.

**Verifier refined fix.** Minimal change, all in backend/evaluation/universe_bt.py, to implement DESIGN 9.1:

1. In _Panel.__init__, build a ranking matrix RK = mu/sigma (log), falling back to ndtri(clip(p_up, 1e-6, 1-1e-6)) where mu or sigma is missing. It is filled on the same `ok` rows as SC. Keep SC and net only for the only_buy gate (net > 0 and p_up >= buy_p). Keep avail based on SC or RK, but make sure the two share one mask.
2. Use RK instead of SC in picks() (the `sc = np.where(elig, RK, -inf)` line, with ties broken by PU and then by code order), in bottoms(), and in ic_series() (the spearman_ic(pn.SC[t, m], ...) call). Update the module docstring and the 選股方式 note, which still say "(預期報酬 − 來回成本) ÷ 波動".
3. In spread_stats, use log returns pn.R, or the arithmetic return minus the same-window EW return, so the Jensen term sigma^2/2 does not enter the spread. A beta-neutral spread is optional hardening only.
4. Add additive report keys, with matching entries in UNIVERSE_SPEC and validate_universe_report. Make them optional or "num?" so the existing frontend and fixtures still pass.
   - basket: {"mean_sigma", "universe_mean_sigma", "mean_beta", "universe_mean_beta"}. Beta is each held stock's OLS slope of its daily close-to-close return on the EW universe return, estimated from data up to t only (causal rolling or expanding window). Report the weight-averaged beta of W.
   - benchmark_top_sigma: STATS from the same tranche machinery with RK replaced by SG (top-k by sigma), plus optionally a random-k reference. Compute it with a second pn.picks() / tranche_weights() / simulate_weights() pass. Cost is small compared with the walk-forward.
   - benchmark_equal_weight already exists.
5. Add one bullet to make_universe_verdict and the notes, labelling the exposure when basket beta or sigma is well above the universe's, for example "精選股平均波動/β 高於全體".
6. Optionally report alpha vs EW (regression intercept with a HAC t) as extra evidence. Do not add it to assess_evidence unless thresholds are re-calibrated.

Side effects:
- tests/test_evaluation* cases that assert the picks, IC or ordering come from SC, or that compare exact numbers, will need updating. This includes the planted-signal power tests, which key off score ranking.
- The frontend only reads the existing keys, so new keys are additive. The frontend fixtures and the schema test (validate_universe_report) must be extended.
- server.py's evidence block is unaffected.
- THRESH_UNIVERSE calibration should be re-checked. In the reviewer's MC the DESIGN-9 ranking gives 0% non-no_edge in the vol/beta nulls.
- Real-data verdicts at h=1/5/20 will change, because the ranking changes.


## S-06 [low] Date-clustered HAC tests of the universe verdict are about 2x oversized for very persistent rankings (rank autocorrelation 0.77-0.98 like the real score)

- owner: evaluation  
- location: backend/evaluation/universe_bt.py:164-197 (hac_lag, _t_stats), 210-246  
- reviewer: review:stats

**Description.** [DEFECT, limited impact]. IC, picks and spread tests cluster by date (DESIGN 9(c) satisfied) but the real recommender score is almost static (cross-sectional rank autocorrelation 0.98/0.90/0.86/0.77 at 1/20/60/250 days for h=1; 0.99/0.93/0.89/0.80 for h=5; 1.00/0.97/0.93/0.81 for h=20), so dependence is by stock, not by date. Placebo on the REAL 61-stock panel: random AR(1) scores with rho=0.999 (rank autocorrelation 0.98/0.94/0.78 at 20/60/250 days) vs iid random scores through the unchanged evaluate_universe.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_placebo_universe.py 100 ar0.999 1,5,20 && python3 s3_placebo_universe.py 200 iid 1,5,20 && python3 s3_analyze_placebo.py

**Evidence.** iid placebo (200 draws/h): sd(t_ic) 1.00-1.01, IC(+)<5% 4.5-5.0%, picks 6.0-7.5%, spread 6.0-7.5%, verdict != no_edge 1/600 (0.17%). rho=0.999 (300 draws/h): sd(t_ic) 1.51/1.40/1.42, IC(+)<5% 11.3/8.7/10.3%, picks 12.7/8.0/11.0%, spread 7.7/8.7/9.7%, verdict != no_edge 5/900 (0.56%, all weak_edge via picks/IC + spread) vs the 0.3% design level. rho=0.99: sd(t) 1.1-1.2, near nominal, 0/300.

**Proposed fix.** Add stock-level (two-way) clustering or calibrate p-values by placebo: draw random rankings with matched rank autocorrelation (and matched sigma tilt, e.g. permutation within sigma buckets) and take the p-value from their distribution.


## S-07 [medium] 'At least two tests agree' double-counts the IC and MSE legs (one mu series): on real returns the single-stock false-edge rate is 0.22-0.49% per report (20-45x the synthetic 0.011%); all 16 placebo false edges and all 3 real moderate_edge verdicts rest on IC+MSE only

- owner: evaluation  
- location: backend/evaluation/report.py:226-232 (evidence legs), 240-270 (assess_evidence)  
- reviewer: review:stats

**Description.** [DEFECT in the verdict logic; root cause of the C-08 cases]. The four legs are not independent: IC and the DM(MSE) leg are both functions of (mu, R), direction and Brier of (p_up, R). assess_evidence counts any two legs below 5% as 'agreement' and any two below 1% as 'moderate', so IC+MSE alone can reach moderate_edge although accuracy is below the always-up baseline. In addition the MSE leg is not a timing-skill test: a forecaster that only knows a positive drift beats the zero benchmark (placebo MSE-DM rejects 10-13% at h=5/20). Synthetic iid/GARCH data hide this (zero drift, no persistence structure); real returns expose it.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_placebo_single.py 12 0 0.9 && python3 s3_placebo_single.py 12 0 0.99 && python3 s3_analyze_placebo_single.py && python3 s3_leg_dependence.py

**Evidence.** Placebo on 62 REAL return series x 12 draws x h=1/5/20 (2,232 evaluations per setting; independent AR(1) fake forecaster scaled to the real engine's sd(mu)/mean(sigma)=0.016): verdict != no_edge 5 (0.22%, rho=0.9) and 11 (0.49%, rho=0.99), vs 5/45,150 (0.011%) on synthetic data and the docstring's ~0.06%; expected false edges among the 186 real reports 0.4-0.9. All 16 resting on IC+MSE (direction and Brier legs both >=5%). Joint rejection: P(IC<1% and MSE<1%) = 0.00/0.27/0.00% (rho=0.9) and 0.40/0.94/0.13% (rho=0.99) at h=1/5/20 vs 0.005-0.06% if independent; P(both<5%) 2.6/2.0/0.4% vs 0.24/0.35/0.20%; Spearman(p_ic,p_mse)=0.69/0.47/0.28. Real reports: 2412 (acc 46.3% vs always-up 53.7%; legs dir 1.00, brier 0.96, ic 0.000, mse 0.006), 3045 (49.4% vs 52.0%; 0.96/0.69/0.000/0.0007), 4904 (52.4% vs 53.5%; 0.79/0.69/0.0009/0.0024). A rule 'at least one probability leg (direction or Brier) < 5% as well' keeps synthetic false edges at 5/45,150, removes 16/16 placebo false edges and the 3 real verdicts, and costs little power (oracle h=1 IC .125: 93.7->91.7% GARCH, 73.7->57.0% Gaussian; h=20 IC .33: 50.5->44.0%, 25->21%).

**Proposed fix.** Count evidence families, not tests: require at least one probability-based leg (direction or Brier) AND one mean-based leg (IC or MSE) for any edge level (or merge IC/MSE into one Clark-West leg); keep Bonferroni over families; add the placebo FPR (real-return) to the docstring.

**Verifier:** real=True severity=low  
The mechanism is real but the severity and the proposed fix are overstated.

What reproduces:
- I re-ran the placebo from scratch with fresh seeds (draws 100 and up, 80 draws x 62 real series x h=1/5/20 = 14,880 evaluations per rho).
- The false-edge rate (verdict != no_edge) is 0.25% pooled at rho=0.9 and 0.50% at rho=0.99. By h that is 0.16/0.38/0.20% at rho=0.9 and 0.30/0.65/0.54% at rho=0.99.
- Only 0.06% (rho=0.9) and 0.13% (rho=0.99) reach moderate_edge or higher.
- In 49 of the 50 verdict-bearing placebo cases with a valid leg count (rho=0.9: 7 of 8 at h=1... see evidence), the direction and Brier legs are both at or above 5%, so the edge rests on IC and MSE alone. Across both rho settings only 2 of 111 false edges had a probability leg below 5%. This confirms that IC and MSE are positively dependent (both are functions of mu and R), so the "two tests agree" rule does not buy independence.
- The MSE-DM leg rejects 10-13% at h=5/20 for a forecaster that knows only a positive drift, so it is partly a drift test rather than a timing test.
- The docstring's "about 0.06%" comes from synthetic zero-drift worlds only.

Why it is not medium:
- The design target is a family-wise level of 1% (THRESH_STOCK fw=0.01). A measured real-return false-edge rate of 0.25-0.5% per report, at worst 0.65% at one h, is still inside that nominal level, so the rule is not broken.
- The placebo is deliberately adversarial: it adds a constant +0.0004*h drift and uses very persistent AR(1) signals.
- The claim that the 3 real moderate_edge verdicts are false edges is not supported. I re-ran them. 2412, 3045 and 4904 at h=1 have IC p-values of 1.8e-5, 5e-6 and 8.7e-4 (IC 0.09-0.12). These are far deeper than any placebo IC tail would produce.
- Across the real panel, IC p<1% occurs in 7 of 62 stocks at h=1, against about 0.6 expected under the null. That points to genuine small mean-forecast skill or a separate IC-bias issue (a different finding). It does not show a dependence artefact.
- Accuracy below always-up does not contradict a mean-ranking edge, because the direction leg tests a different quantity.
- Requiring a probability leg would cost real power (the reviewer's own figures: 73.7 to 57.0% for Gaussian at h=1 with IC 0.125), and p_up is itself a function of mu, so it would not buy true independence either.

Conclusion: the dependence and the understated docstring are real but low severity. I do not recommend changing the verdict rule.

**Verifier refined fix.** Minimal change, no logic change: edit only the docstring in backend/evaluation/report.py (lines ~25-31). State that the ~0.06% figure is measured on synthetic zero-drift worlds. Add that on 62 real return series, with independent persistent AR(1) fake forecasters (rho 0.9-0.99) and a constant +0.0004*h drift, the single-stock false-edge rate is about 0.25-0.5% per report (moderate or higher about 0.06-0.13%). That is still below the 1% family-wise target. Add that the IC and MSE legs share mu and are positively dependent, and that the MSE leg is partly a positive-drift test, so "two tests agree" is weaker than independent evidence. No tests are affected, because only the docstring changes.

Optional, not recommended: if the stricter rule is wanted anyway, in assess_evidence require, for n_confirm>=2 (moderate_edge), at least one of ps['direction'] or ps['brier'] to be below thr['agree']. Side effects:
- The rule cuts the real power the reviewer measured.
- It will downgrade the 3 real h=1 moderate_edge verdicts, which have strongly significant IC.
- It will break tests in tests/test_eval_report.py that plant IC-only edges with a no-skill probability model.
- It would also need a UI wording and DESIGN.md update.


## S-08 [low] DM test kernel: Bartlett weights (1-k/h) on an MA(h-1) loss differential and no data-driven bandwidth give size ~10% at h=5/20 under exact H0; production DM is also under-sized for nested models

- owner: evaluation  
- location: backend/evaluation/metrics.py:389-421 (newey_west_lrv, dm_test)  
- reviewer: review:stats

**Description.** [DEFECT, limited impact]. metrics.newey_west_lrv uses Bartlett weights 1-k/(lag+1) with lag=h-1; for overlapping h-step loss differentials whose autocovariances are those of an MA(h-1) the textbook choice is the rectangular window to lag h-1 (consistent) with the Harvey-Leybourne-Newbold factor; Bartlett under-estimates the long-run variance when forecasts are persistent. HLN itself is immaterial at n=1250 (<=1.6%). At h=1 the lag is 0, so persistent forecast differences are not covered at all. Separately the engine nests its naive component, for which DM is known to be under-sized (Clark-McCracken); the MSE leg is weak because of it.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_dm_size.py 1500 ; python3 s3_dm_size.py 1500 4 ; python3 s3_cw_bank.py

**Evidence.** H0 world (two independent zero-skill AR(1) forecasters, equal expected loss, R = overlapping h-sums, n=1250, 1500 sims; Gaussian / Student-t4 returns): two-sided rejection at nominal 5% - production: h=5 rho_f=0.9 10.9/11.3%, rho_f=0.99 21.3/18.8%; h=20 rho_f=0.9 9.8/10.5%, rho_f=0.99 13.8/15.2%; h=1 rho_f=0.9 8.4/8.3%, rho_f=0.99 37%; rectangular+HLN: 5.3/5.5%, 13.1/11.6%, 4.9/6.4%, 6.7/7.3%. Null engine bank (200 paths) one-sided MSE leg: production DM 0.5/1.5/1.5% vs Clark-West 2.5/3.5/3.0% (h=1/5/20).

**Proposed fix.** Use the rectangular (or Andrews/Newey-West automatic) bandwidth with the HLN factor and at least ~6 lags even at h=1; for model-vs-nested-benchmark comparisons use Clark-West.


## S-09 [low] best_baseline is chosen by realised test-sample accuracy (hindsight max over 3 rules) and is not labelled as such; it halves the power of the direction test

- owner: evaluation  
- location: backend/evaluation/metrics.py:246-250; backend/evaluation/report.py:334-336,377-379  
- reviewer: review:stats

**Description.** [DESIGN CHOICE, conservative, unlabelled]. metrics.direction_summary picks max over {always_up, prev_sign, train_majority} on the verification sample (metrics.py:246) and tests against that realised accuracy as if it were a known constant. The three rules are causal, but taking their maximum after seeing the data is a hindsight choice (winner's curse on noisy baselines such as prev_sign, ~50% accuracy). The Brier reference is labelled 「含事後資訊」 in the notes, this choice is not.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s2_baseline_choice.py

**Evidence.** On no-skill forecasters prev_sign is the 'best' baseline in 16-42% of runs; max-of-3 exceeds the mean of the three by 0.95-1.13/1.6-1.8/3.0-3.5pp of accuracy at h=1/5/20. Planted IC 0.05 at h=1 (GARCH/Gaussian): direction test rejects 22.7/18.3% against always_up only but 11.3/10.0% against the 'best' baseline; IC 0.125: 86.7/61.3% vs 80.0/47.0%.

**Proposed fix.** Pre-specify one comparator (train_majority/always_up) for the test and show the others descriptively, or correct for the max (Bonferroni over 3); add a note 「最佳基準由驗證期間事後選出」.


## S-10 [low] Survivorship/selection bias is stated but not quantified, and the TAIEX comparison is price-only against total-return stocks ('略不利' understates ~4-5pp/yr)

- owner: evaluation  
- location: backend/evaluation/universe_bt.py:355-381 (make_universe_notes), 277-289 (index_returns), 440-448  
- reviewer: review:stats

**Description.** [MESSAGING GAP]. universe_bt notes/bullets state survivorship bias qualitatively (note #1, last bullet; UI shows bullets un-collapsed) but give no magnitude; the benchmark curve compares total-return stock baskets (adjclose) with the ^TWII price index.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 d_survivorship.py ; python3 s2_survivorship2.py   # second script needs network; its saved output is s2_survivorship2.out

**Evidence.** Window 2021-08-27..2026-10-22 (1,249 days), 61 names: EW daily-rebalanced CAGR 30.8% total return / 25.8% price-only vs TAIEX price index 21.4%; random other names listed before 2021-07 (today-rank 61-800, n=80): 20.2% total / 16.3% price-only, i.e. selection about +10.6pp/yr; mean per-stock CAGR 26.2% vs 14.4% (median 21.8 vs 12.3); by today's liquidity rank 1-30 47.4%, 31-60 33.5%, 61-150 17.1%, 151-300 17.0%, 301-500 15.0%, 501-800 10.8%. Dividends contribute 4.95pp of the EW CAGR (universe mean cash yield 3.86%/yr); daily-rebalanced EW is 5.35pp BELOW buy-and-hold EW (36.1%).

**Proposed fix.** Add the quantified selection effect to note #1 (about +10pp/yr for today's top-60 vs other listed names, gradient by liquidity rank) and show EW price-only and/or TAIEX total return (or add the universe's mean dividend yield) so the index line is comparable.


## S-11 [low] DSR n_trials is hard-wired to 1 (DSR == PSR always) while the UI lets users vary horizon/threshold/exec/years/sizing; V in DSR is the series' own SE, not cross-trial variance

- owner: server  
- location: backend/jobs.py:1221-1223; frontend/view_backtest.js:443; backend/evaluation/metrics.py:639-650  
- reviewer: review:stats

**Description.** [DESIGN LIMITATION]. jobs.py passes n_trials=1; the UI tip says 「這裡只回測一組，所以為 1」, which is inaccurate for a user who tried many settings, so the deflation the report advertises never happens. metrics.deflated_sharpe_ratio uses sharpe_standard_error(r)^2 as the cross-trial variance V (disclosed in the docstring, a proxy).

**Repro.** grep -n n_trials /home/gary/stock/backend/jobs.py ; cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 c_formulas.py

**Evidence.** DSR equals PSR in every real report; formula check: DSR identical to the Bailey-Lopez de Prado expression to 1.8e-13 for explicit V and monotone in N.

**Proposed fix.** Accept an n_trials parameter (or count distinct parameter sets per symbol/session) and reword the UI tip; supply cross-trial variance when several runs exist.


## S-12 [low] Small-sample verdict text says 'forecast periods overlap' at h=1

- owner: evaluation  
- location: backend/evaluation/report.py:347-349  
- reviewer: review:stats

**Description.** [TEXT DEFECT]. make_verdict's small_sample tail always prints 「有效樣本只有約 n 筆獨立觀察（共 n 筆，預測期間互相重疊）」 even for h=1, where 1-day targets do not overlap and n_eff = n.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_text_small_h1.py

**Evidence.** h=1, 59 scored forecasts: 「有效樣本只有約 59 筆獨立觀察（共 59 筆，預測期間互相重疊），樣本太少…」. Also: constant-price runs print 「回測期間沒有可評估的樣本外預測」 although 599 forecasts exist (all ties).

**Proposed fix.** Add the overlap clause only if h>1; for all-tie samples say that every forecast fell on an unchanged price.


## S-13 [low] Christoffersen tests exist but are never used; interval calibration is tested only at 90% and without an independence check

- owner: evaluation  
- location: backend/evaluation/metrics.py:465-507; backend/evaluation/report.py:183  
- reviewer: review:stats

**Description.** [SPEC GAP]. DESIGN 4 asks for Kupiec/Christoffersen tests on 50/80/90% coverage; metrics.christoffersen_independence/christoffersen_cc are implemented (and match an independent implementation) but report.py/universe_bt.py never call them; coverage_summary applies Kupiec only at the 90% headline. Violation clustering (volatility regimes) is therefore invisible; real h=1 Kupiec rejects for 24 of 62 symbols (39% vs 5% nominal) with no diagnosis.

**Repro.** grep -n christoffersen /home/gary/stock/backend/evaluation/report.py /home/gary/stock/backend/evaluation/universe_bt.py ; cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 c_formulas.py

**Evidence.** grep christoffersen backend/evaluation/report.py backend/evaluation/universe_bt.py -> no hit; coverage_summary(headline=0.90). Real 90% coverage min/median/max 0.845/0.889/0.924 (h=1), 0.865/0.894/0.942 (h=5), 0.828/0.891/0.980 (h=20); Kupiec p<5%: 24/3/5 of 62.

**Proposed fix.** Report Kupiec at 50/80/90 and the Christoffersen conditional-coverage p for h=1 (skip or block-adjust for h>1).


## S-14 [low] p_value_vs_coin and the universe per_stock table show direction accuracy without the base rate (bull-market base rates look like skill)

- owner: evaluation  
- location: backend/evaluation/report.py:379; backend/evaluation/universe_bt.py:249-265; frontend/view_picks.js:268-285  
- reviewer: review:stats

**Description.** [MESSAGING]. p_value_vs_coin tests against 50% while real up-rates are 51-57%, so any forecaster that leans up 'beats' a coin; the universe per_stock rows (code, name, accuracy, brier_skill, ic, n; UI table view_picks.js:268-285) carry no base rate or always-up accuracy.

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_real_direction_summary.py

**Evidence.** Real 62 symbols: p_vs_coin<0.05 for 10/29/29% of symbols (h=1/5/20) while p_vs_best<0.05 for 3/0/0%; accuracy is BELOW always-up in 68/85/98% of symbols (mean -0.86/-0.06/-0.16pp). Universe h=5 per_stock accuracy median 54.0% (46.4-59.5%), 84% of stocks above 50%, while median accuracy minus base rate is -0.06pp and median per-stock Brier skill -0.5%.

**Proposed fix.** Add base_rate and always_up_accuracy columns to per_stock and the UI table; label p_vs_coin as 'vs 50%, not a skill test'.


## S-15 [low] End-to-end sensitivity: the real engine captures 36-72% of a planted lag-1 autocorrelation at h=1 and only ~15% of a planted slowly-varying drift at h=5/20 (design property from shrinkage, not a bug)

- owner: quant  
- location: backend/quant (dlm.py / ensemble.py shrinkage); measured via backend/evaluation  
- reviewer: review:stats

**Description.** [DESIGN PROPERTY, informational for the quant owner]. The ensemble anchors to the naive/shrunk-drift component, so structure that an oracle could use is only partly learned; together with S-02 and S-04 this bounds what any backtest of this engine can show. IC gain over the engine's own null (null IC -0.03/-0.09/-0.19 at h=1/5/20).

**Repro.** cd /tmp/claude-1000/-home-gary-stock/464acd23-01ad-4607-92f8-afb2bdc019fb/scratchpad/audit/stats && python3 s3_analyze_engpow.py

**Evidence.** AR(1) phi=0.05/0.10/0.15 (population IC_1 0.05/0.10/0.15): engine IC_1 -0.01/+0.02/+0.08 (Gaussian) and +0.04 (GARCH, phi=0.10) -> gain +0.02/+0.05-0.07/+0.11 (36/48-68/72% of population IC). Slow latent drift sd 0.0015 (oracle IC .10/.21/.36): engine IC -0.00/-0.05/-0.14 (Gaussian) and +0.01/-0.03/-0.13 (GARCH): gain ~+0.03/+0.04/+0.05 at h=1/5/20 (10-20% of the oracle at h=5/20). Slow drift sd 0.0008: engine IC equals the null.

**Proposed fix.** No change required; document the engine's learning limits next to the 'no edge' conclusions, or add a latent-drift component if long-horizon structure is wanted.


## P-01 [high] Yahoo flat zero-volume placeholder bar for a day the exchange was closed is ingested as a real trading bar (live today: 2026-10-23, 63/63 stocks)

- owner: data  
- location: backend/data.py parse_yahoo_chart (zero-volume rows kept), _refresh/_merge, last_trading_date_expected (calendar ignores unscheduled closures)  
- reviewer: review:product

**Description.** Observed 2026-10-24 21:03-22:07 CST. Yahoo returns a bar dated 2026-10-23 for every stock/ETF (o=h=l=c=previous close, volume 0; ^TWII has a null row). TWSE has no 10-23 session: STOCK_DAY_ALL Date is '1151022' and STOCK_DAY for 2330 ends at 115/10/22. data.parse_yahoo_chart keeps zero-volume rows, so the placeholder becomes a genuine bar with last_bar_complete=True. Consequences on the real server: /api/history asof=2026-10-23; the forecast note says '資料截至 2026-10-23 收盤'; the stock header shows '▬ 0.00% 1日' and a flat doji with no volume; the picks page has chg_1d == 0.0 for 60/60 names while the market strip shows TAIEX -1.81% (index asof is 10-22, so stock and index pages disagree on the date); the 'stale/尚未更新' chip disappears. The model is fed a fake zero-return day: for 2330, p_up changes -0.11/+0.03/+0.12 pp (h=1/5/20), vol percentile 0.316->0.260, annual vol 22.4%->21.7%, driver '1日反轉' -1.45%->0.0%, '大盤當日報酬' -1.8%->0.0%, vol_z value None. The bar is permanent (same mechanism as 2016-09-27/28 typhoon closures): the 60-name universe cache holds 7,118 zero-volume bars (6,994 flat), 33 of them inside the 1250-bar walk-forward test window. README only admits that closures cannot be predicted; it does not mention that Yahoo fabricates bars for them. Probability impact is small, but the as-of date and 1-day change are wrong for every stock.

**Repro.** Deterministic/offline: python3 $MINE/p3/phantom_unit.py  (payload whose last row is flat+zero volume is parsed as a normal 6th bar). Live: TWQ_CACHE_DIR=$(mktemp -d) TWQ_HOST=127.0.0.1 PORT=18999 python3 -m backend.server & then curl -s 'http://127.0.0.1:18999/api/history?symbol=2330&limit=3' -> last bar {d:2026-10-23,o:2385,h:2385,l:2385,c:2385,v:0}, asof 2026-10-23; curl 'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL' | head -c 200 -> Date '1151022'. Forecast delta: python3 $MINE/p3/phantom_effect.py (needs the refreshed cache copy $SC/cache_product3).

**Evidence.** fresh cache: 63/63 stock/ETF price files end with the 2026-10-23 flat zero-volume bar, index ends 10-22. Screenshot of the live stock page: $MINE/render3/live_stock_2330_1280_s1.png. Effect numbers: $MINE/p3/phantom_effect.json.

**Proposed fix.** In parse_yahoo_chart/_refresh drop, for non-index symbols, a bar with volume==0 and o==h==l==c==previous close; stricter: treat a date as non-trading when the index has no/null bar and >=90% of the cached universe shows such a bar, and use TWSE STOCK_DAY_ALL 'Date' (ROC, e.g. 1151022) as the authoritative last official session (ignore Yahoo bars after it). Add the detected closure date to the in-memory holiday set so last_trading_date_expected / market_is_open / next_trading_day / the '資料尚未更新' banner stay right. Regression test = phantom_unit.py payload. Historic phantom bars (7,118 in the universe) share the root cause (causal reviewer's area).

**Verifier:** real=True severity=medium  
Reproduced independently. (1) TWSE confirms no 2026-10-23 session: STOCK_DAY for 2330 ends at 115/10/22, STOCK_DAY_ALL Date is 1151022 (queried 22:35 Sat), and the holiday schedule does not list 10-23, so it is an unscheduled closure. (2) The refreshed cache copy (scratchpad/cache_product3) has 2330, 0050 and 6488 ending with a 2026-10-23 bar where o=h=l=c=previous close and volume 0. 8 of 67 files carry the 10-23 bar, all zero-volume. The others are probably older, not re-fetched files. (3) Offline repro phantom_unit.py shows parse_yahoo_chart keeps such a row as a normal bar with r=0. Reading the code confirms nothing drops it: parse_yahoo_chart only drops null/non-positive OHLC, volume null is coerced to 0, and last_trading_date_expected uses only the TWSE holiday table, so the fake bar counts as a fresh session and last_bar_complete is True. The only mitigation is in quant/features.py, which treats H<=L bars as missing range-vol and makes vol_z NaN. The return is still a genuine 0.0 day. The cache also holds 8,889 zero-volume bars in 67 files, 7,409 of them flat, so historic closures behave the same way. The README says closures are unpredictable, not that Yahoo fabricates bars for them, so this is not a documented design choice. Severity is downgraded from high to medium. It is user-visible and wrong: the as-of date, the 1-day change of 0.00% and the missing stale chip are wrong for every stock, and the stock and index pages disagree. But the probability impact reported is at most 0.12 pp. It is the same class of artefact the cleaning layer already tolerates historically, and the visible date error resolves itself once the next real session (Tue 10-27, since 10-26 is a make-up holiday) is fetched. It is not a wrong-decision-grade error.

**Verifier refined fix.** Do both parts together, in backend/data.py. A naive bar filter alone would make every symbol look stale (last bar 10-22 < last_trading_date_expected 10-23), so it would trigger a refresh per symbol every 10 minutes and show the stale chip with the wrong cause.

1. parse_yahoo_chart: after the OHLC/adj validity mask, add a closure-placeholder mask and drop those rows: `ph = (v == 0) & (o == h) & (h == l) & (l == c) & (c == c_prev_valid)`, where c_prev_valid is the previous kept bar's close (np.r_[nan, c[:-1]]). Apply it only when meta.instrumentType != 'INDEX' (the tests use volume=[0]*400 for ^TWII). Do not apply it to the first row. Be careful about unadjusted prices: compare close to the previous close, not adjclose. A genuinely no-trade day for an illiquid TPEx stock is also dropped, which is harmless.
2. Calendar: make the system know about the closure. In _refresh/_merge, or in a small helper run after a batch of refreshes, record a date D into an in-memory plus persisted extra-closure set when the non-index parse dropped a placeholder bar for D (use a quorum, e.g. >=80% of the symbols fetched that day, or the index having no or a null bar for D). Have is_trading_day consult that set (add it next to `holidays(d.year)` as `hol | _EXTRA_CLOSED`). This also fixes last_trading_date_expected, market_is_open, next_trading_day and the stale chip. Alternative with less machinery: in last_trading_date_expected, if the cached ^TWII record's last date < the expected date and a fetch after 14:00 has completed, fall back to prev_trading_day.
3. Cache: bump CACHE_VERSION (116) or force a full refresh so the already-cached phantom bars (including the historic 8,889 zero-volume bars, 7,409 of them flat) are re-parsed. Otherwise incremental refresh keeps them until FULL_REFRESH_DAYS (30). Note that dropping the historic ones changes the walk-forward windows slightly, so previously stored backtests and validation caches would be recomputed or invalidated. price_signature changes automatically.

Side effects and tests: tests/test_data_parse.py and tests/test_data_cache.py use fixtures with flat zero-volume rows. At least test_avg_traded_value (2331.TW with volume=[0]*260 and close=[100.0]*260, if the helper makes o=h=l=c) would be emptied by the filter and raise NotFound, so adjust that fixture, for example by giving the bars a nonzero high-low range. Add the phantom_unit.py payload as a regression test (the last row flat+zero-volume must not be a bar). Add a test that a calendar closure is learned. Check the quant tests that rely on zero-volume placeholders (quant/features.py cleaning docs) and the server tests that assert asof, to be sure none depend on the phantom bar.


## P-02 [high] At h=20 with James-Stein B=0.00 the picks page still presents a 62%-invested, 7-name 'suggested portfolio' and weighted 偏多·買進 cards; contradicts its own note and DESIGN s9(5)

- owner: evaluation  
- location: backend/evaluation/recommend.py build_recommendations (weights block, pnote)  
- reviewer: review:product

**Description.** Real h=20 payload (asof 2026-10-22): B=0.00, score_shrunk is the same value (0.0864) for all 60 names, yet portfolio = {n:7, cash_weight:0.381} with weights 玉山金 15.0%, 中信金 11.6%, 富邦金 9.5%, 聯詠 9.1%, 南亞 5.6%, 奇鋐 5.5%, 健策 5.6%. The same payload's notes say '目前各檔預測差異不比雜訊更大…建議權重趨近於零或平均'. With B=0 the Kelly edge is common_score x sigma_i, so weights are a function only of sigma and the Ledoit-Wolf covariance, not of any view. Placebo (shuffle (p_up, exp_return, median, mu_log) across names, rebuild 300 times with the server's covariance inputs): B==0 in 100% of shuffles, #buy always 26, invested share 84-100% (median 100%), median 9 names held. The weights are positive only because the cross-sectional mean score at h=20 is positive (convexity E[e^R-1] ~ mu+sigma^2/2 and the market drift). At h=5 (mean shrunk score negative) the same logic gives 0 names. DESIGN s9(5) requires 100% cash when B==0; not implemented.

**Repro.** python3 $MINE/p3/b_placebo2.py  (reads cache_product2 offline; prints ACTUAL vs PLACEBO). Payload: python3 -c "import json;j=json.load(open('$MINE/real2/recs_h20.json'));print(j['portfolio'],len({round(r['score_shrunk'],12) for r in j['all']}),[(r['code'],round(r['weight'],4)) for r in j['all'] if r['weight']>0])"

**Evidence.** $MINE/real2/recs_h20.json portfolio/weights/notes; placebo output (ran in this session): h=20 ACTUAL B=0.00 #buy=26 invested=62% in 7 names; PLACEBO x300 #buy 26 [26-26], B==0 100%, invested median 100% [84-100%].

**Proposed fix.** If B==0 (or n_scored<4) or evidence.verdict_level in {none,no_edge} or no name has edge_covers_cost: weights=0, cash_weight=1, portfolio.note='個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移', hide the 建議組合 card or collapse it behind an explicit 'research only' toggle; add portfolio.suppressed_reason.

**Verifier:** real=True severity=high  
Reproduced from the code and the real h=20 payload. DESIGN s9(5) says that when B==0 or no name has positive shrunk net edge, no weights are assigned and the portfolio is 100% cash, with the note "個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移". build_recommendations does not do this. The weights block runs kelly_weights on buys_rows whenever any exist. It uses edge = shrunk*sigma, and with B=0 the shrunk score is one common value, so the weights depend only on sigma and the covariance. The code only appends a note when B<=0 and n_scored>=4, and that note says weights "趨近於零或平均". It never zeroes them, and "平均" is not zero. The real payload (asof 2026-10-22, h=20) has B=0.00 and a single distinct score_shrunk across all 60 names. It still returns n=7, cash_weight=0.381, and weights up to 15%, while its own note says there is no distinguishable edge. view_picks.js renders this as a 建議組合 card with no suppression. The spec requirement is explicit, the payload contradicts itself, and the real-data results show no edge, so this is a real spec violation and misleading output. I rate it high, not critical, because an amber evidence banner exists and the output is labelled as not investment advice. I did not re-run the placebo, but the code path makes its result plausible.

**Verifier refined fix.** In backend/evaluation/recommend.py build_recommendations, compute a guard before the weights block: `no_edge = (B <= 0.0 and n_scored >= 4) or not any(r["net"] * r["shrunk"] > 0 and r["shrunk"] > 0 for r in buys_rows)`. Use the shrunk net edge `r["shrunk"] * r["sg"]` (> 0) as the "positive shrunk net edge" test. Also require the shrunk score to differ across names, because at B=0 every name shares one positive value. A simple rule is `B <= 0.0` (and n_scored >= 4) OR no buys_rows row with shrunk*sg > 0. When `no_edge` is true: skip kelly_weights, leave weights = {}, so n_hold=0, total_w=0 and cash_weight=1.0. Set pnote to '個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移，因此不配置權重，建議全數持有現金。'. Add portfolio.suppressed_reason ('B0' or 'no_positive_edge'). Reword the B<=0 note to say the weights are set to 0, not 趨近於零或平均. Do not use evidence.verdict_level inside build_recommendations. It is filled by the server after the call, so the evaluation module cannot see it. If the verdict should gate the portfolio, do it in the server. In frontend/view_picks.js portfolioCard, when pf.suppressed_reason is set, show the note and skip the weight table, and drop the hard-coded '0.25 倍凱利…' sub-line. Side effects: evaluation tests that assert nonzero weights or cash_weight<1 with a uniform or B=0 score set will need new expectations. Check the test that builds identical forecasts and the portfolio tests. Server and frontend tests that fixture portfolio.n>0 are unaffected unless B=0 there. The `weight` field on every rec becomes 0 in the B=0 case. Check that the UI sorts and filters by weight only when it is positive.


## P-03 [high] `rank` and the James-Stein 'retention B' carry no stock information: B==0 on 100% of days at h=5/20, and B=0.69 at h=1 is the cost/sigma artefact; table shows 60 identical 分數 values

- owner: evaluation  
- location: backend/evaluation/recommend.py shrink_scores/james_stein (SHRINK_BARS=250), build_recommendations order; frontend/view_picks.js allCard '分數' column, header sub text  
- reviewer: review:product

**Description.** Measured over the 1250 walk-forward days x 60 stocks (b_js_degenerate.py): B==0 on 100.0% of days at h=5 and at h=20 (median SS/(k-3) = 0.27 and 0.02; B>0 needs >1), B>0 on 99.9% of days at h=1 (median 0.87, SS/(k-3) median 7.74). Latest real day: SS=182.4 vs k-3=57 at h=1 (B=0.69), 11.2 at h=5 (B=0), 1.3 at h=20 (B=0). At h=1 R2(score ~ -cost/sigma)=0.983 and var(-cost/sigma)=104% of var(score) while var(E[r]/sigma) is 1.8% of it: the 'differences between names' that B certifies are 98% the round-trip cost divided by sigma. The note printed on the page ('保留比例 B=0.69 … B 越小代表各檔預測差異多半是雜訊') therefore tells a retail user that 69% of the cross-sectional spread is signal. At h>=5 the second-stage shrinkage treats an already Bayesian-shrunk forecast mean as a raw sample mean with noise variance h/250 (double shrinkage), so B is always 0 and the order falls back to the tie-break by raw score = net/sigma (Spearman(score,sigma)=0.99/0.95/0.63 mean over 1250 days; 0.98/0.94/0.72 latest day, i.e. C-06). UI consequence (real render): 排名 1..60 next to a 分數 column with one identical value (-0.04 at h=5, +0.09 at h=20); rank 1 at h=1 is 南亞 (sigma 4.29%, -6.08% yesterday, P(漲) 51.2%, net -0.45%). The page header literally says it ranks by '扣除交易成本後的風險調整報酬'.

**Repro.** python3 $MINE/p3/b_js_degenerate.py  (≈2 min, offline); python3 $MINE/p3/b_buy26.py (latest-day tables: distinct score_shrunk, top5 by raw score); decomposition one-liner in the report was run on $MINE/real2/recs_h*.json: sc=score, R2=corr(score,-0.0057/sigma)**2.

**Evidence.** $MINE/render/picks_h5_1280_s2.png, picks_h1_1280_s2.png; $MINE/logs/b_rank_structure.out; this session: b_js_degenerate.py and the decomposition (R2 0.983/0.924/0.523 for h=1/5/20).

**Proposed fix.** After switching rank to rank_signal (s9), do not print B as evidence unless B is computed on the rank_signal cross-section against a noise level estimated empirically (e.g. dispersion under date-shuffled placebo), or drop James-Stein from ordering and keep it only for the weight gate. Hide the 分數 column when its cross-sectional SD is <1e-9 and show a percentile bar ('相對位置 0-100%') instead of integer ranks 1..60 when ties exist; remove the sentence that the ordering is by 風險調整報酬.

**Verifier:** real=True severity=medium  
I reproduced the symptoms on the saved latest-day recs (recs_h1/5/20.json) and confirmed in code that nothing in backend/evaluation/recommend.py or frontend/view_picks.js has changed since the finding. The code has no rank_signal at all. The §9 addendum describes the fix but it is not implemented.

What reproduces:
- **Note text:** at h=1 the page note says 保留比例 B=0.69, which reads as 69% of the cross-sectional spread being signal. Yet score is almost entirely -cost/sigma, with R2=0.983 and Spearman(score, sigma)=0.976 against Spearman(score, p_up)=0.149.
- **B is a cost/volatility artefact, not stock information:** shrink_scores treats score = net/sigma as a raw sample mean with noise variance h/250, so any cross-sectional spread of net/sigma counts as 'signal'. At h=1 the cost term (identical for every stock, divided by each stock's sigma) supplies that spread.
- **h=5 and h=20:** B=0.00 on the real latest day, and score_shrunk has exactly 1 distinct value across 60 names. Rank is then decided only by the raw-score tie-break, which is the volatility ordering (Spearman with sigma 0.94 at h=5 and 0.716 at h=20, R2 0.924 and 0.523).
- **UI text:** the header says 依經收縮的風險調整分數由高到低 and the rank column tip repeats it. The 分數 column then shows one identical value for all 60 rows at h=5 and h=20. That text is false whenever B is 0 and misleading at h=1.

Calling this medium rather than high is a judgment call, for two reasons:
- Much of the harm duplicates C-06, the rank-by-volatility problem. §9 already mandates the rank_signal switch, and §9 item 5 already requires a B==0 note.
- This is a no-edge research tool. It already prints notes saying the ranking is not validated and that there is no signal. The strict buy/avoid lists are empty at h=1/5 and the weights are cash.

Genuinely new here is the misleading B note when B>0, the identical 分數 column, and the double shrinkage at h>=5.

**Verifier refined fix.** Implement it together with the §9 rank_signal switch (C-06), in this order:

1. backend/evaluation/recommend.py, build_recommendations:
   - Sort `order` by (-rank_signal, -p, code), with rank_signal = mu_log/sigma_log (fallback Φ⁻¹(p_up)). Add the rank_signal, score_rank_pct, lean and edge_covers_cost keys.
   - Keep score and score_shrunk only for the buy/avoid gate and the Kelly weights.

2. Stop using B as an informational note:
   - Delete the f-string note '排序已做經驗貝氏…保留比例 B=…'. Use B only as the weight gate. When B==0 or no name has positive shrunk net edge, emit zero weights and the §9 item 5 note '個股間的預測差異無法與雜訊區分；偏多主要反映整體市場正漂移'.
   - For the weight gate, compute the James-Stein input on the exp_return/sigma cross-section, or on rank_signal, not on net/sigma. The -cost/sigma term is a per-stock constant that is not an estimate. Stop applying the h/T noise scale to forecasts that are already Bayesian-shrunk. Change shrink_scores to take a noise level, and estimate it from a date-shuffled placebo or set it explicitly.
   - If B is not recomputed this way, drop the shrink from the weight path at h>=5 and keep only 'edge_covers_cost' plus the buy gate.

3. frontend/view_picks.js:
   - Lines 59, 91, 135: replace '依經收縮的風險調整分數' with a rank_signal description ('依模型上漲訊號強度，非風險調整報酬').
   - Line 147: show 分數 as score_rank_pct, or hide the column when its cross-sectional SD is <1e-9. Update the tip text at lines 92, 148, 206.

Side effects: the tests in tests/test_evaluation* that assert on the shrink note text or on the order of `all` (and any server and frontend tests that read score_shrunk or the notes) need updating. The new §9 regression test (|Spearman(rank_signal, sigma)| < 0.9) must be added. universe_bt.py must use the same rank_signal. The Kelly weights change only if the shrink inputs change, so check the weight tests.


## P-04 [medium] Inconsistent rank numbering and counts on the picks page (card rank vs table rank; 10 cards vs 26 buy chips vs 7 held)

- owner: evaluation  
- location: backend/evaluation/recommend.py rec()/rank enumerate; frontend/view_picks.js buysCard sub text ('共 N 檔' = len(buys))  
- reviewer: review:product

**Description.** build_recommendations numbers `rank` per list: buys[] = 1..k within buys, all[] = global. Real h=20: 台積電 is badge '4' in the 買進候選 cards but '8' in 全部評分名單; 健策 3 vs 4. The cards header says '共 10 檔' (top_k truncation) while the table filter '買進' shows 26 names and the portfolio card says 7 held; weights are not monotone in rank (ranks 1,2,4 get ~5.6%, ranks 8-21 buys get nothing, ranks 23-26 get 9-15%). A retail reader sees three different counts (7/10/26) and two different rank numbers for the same stock.

**Repro.** python3 $MINE/p3/b_buy26.py | grep 'rank-in-buys'  (prints [('1303',1,1),('3017',2,2),('3653',3,4),('2330',4,8)])

**Evidence.** $MINE/real2/recs_h20.json: buys[3] = {code:'2330',rank:4} while all[] has 2330 at rank 8; screenshot $MINE/render/picks_h20_1280_s1.png and picks_h20_1280_s3.png.

**Proposed fix.** One global `rank` (and `score_rank_pct`) for every list; separate `buy_rank` if needed; header text 'N 檔符合條件，顯示前 k 檔'; card badge = global rank or percentile; only show a weight column when portfolio is shown (see P-02).

**Verifier:** real=True severity=low  
I confirmed this from the code. I did not re-run the real-data repro. build_recommendations numbers `rank` separately per list. buys uses enumerate(buys_rows) and avoids uses enumerate(avoid_rows), both counting from 1 after filtering by signal and truncating to top_k. `all` uses enumerate(order), a global ranking. A stock in more than one list therefore carries two different `rank` values. The card badge (frontend/view_picks.js pickCard) shows rec.rank with the title "依經收縮的風險調整分數排名". The table's 排名 column uses the same key from `all`. A name can show 4 in the cards and 8 in the table, with no explanation in the UI. The counts disagree for the same reason. The cards header says '共 ' + buys.length + ' 檔', where buys is cut at top_k=10. The table's 買進 filter counts every row with signal==='buy', which can be 26. The portfolio card's n is the number of names with weight > 0, which is 7. The table subtitle says "只有符合條件的才會出現在「買進候選」", but names with a buy chip can be absent from the cards, so that line is misleading. I rate it low rather than medium because every number shown is correct and nothing is fabricated. The problem is two inconsistent rank labels and unexplained counts, and it does not affect the model or the evaluation. The weight pattern in the finding (not monotone in rank) is a separate matter, covered by P-02 and the Kelly and covariance weighting. It is not part of this finding. The §9 addendum redefines `rank` as based on rank_signal, which recommend.py does not implement yet. That change does not fix this bug, because per-list enumeration would remain unless it is changed explicitly.

**Verifier refined fix.** 1) backend/evaluation/recommend.py, build_recommendations: use one global rank for every list. Compute `grank = {id(r): i+1 for i, r in enumerate(order)}` and make rec(r) emit rank=grank[id(r)] for buys, avoids and all. If a within-list position is wanted, add a separate key such as `buy_rank = i+1`, not `rank`. Also add `n_buy_total = sum(r["signal"]=="buy" for r in rows)` to the response. 2) frontend/view_picks.js buysCard: change the sub text to 'N 檔符合條件，顯示前 k 檔' when n_buy_total > buys.length, and fall back to the current text otherwise. Change the allCard sub text so it no longer claims that only qualifying names appear in 買進候選. For example: "標示為「買進」的名單中，僅前 10 檔列入買進候選與建議組合" (only the top 10 buy-signal names enter the candidate cards and the suggested portfolio). 3) Side effects: change the tests/test_eval_recommend.py:385 assertion, because buys[].rank will no longer be 1..k and will instead be strictly increasing global ranks. Check tests/test_front_picks.py around line 110 and any server test or fixture that checks buys[].rank. The DESIGN.md REC example (rank:1) stays valid. The §9 addendum's `score_rank_pct` and `rank_signal` ordering should be applied on top of this global rank when implemented. Keep the weight column out of the cards when the portfolio is empty (see P-02)."


## P-05 [high] '推薦整體歷史驗證' panel: the +932% equity curve is volatility/beta exposure, the validated basket is not what the page recommends, and the intro text says it is

- owner: evaluation  
- location: backend/evaluation/universe_bt.py evaluate_universe (only_buy_signals default), backend/jobs.py _run_universe_backtest; frontend/view_picks.js universePanel showIdle (line ~335), statCols/ratioKpis  
- reviewer: review:product

**Description.** Real h=5 report (verdict no_edge, IC t=0.92, hit 53.5% = base 53.5%): visible KPI table shows '推薦組合' +932.5% total, +58.1%/yr, Sharpe 1.26, PSR 99.8%, spread +1.10% (t=3.17) vs equal-weight +298.1% and TAIEX +168.8%, and the equity chart ends at 10.32x vs 3.98x vs 2.69x. Annual vol of the basket is 44.0% vs 18.5% (beta 1.49). Same engine, swapping only the ranking key (b_sigma_bench2.py, offline): most-volatile-5 basket returns +464.8% / +1448.4% / +1124.8% at h=1/5/20 versus the model's score basket +519.8% / +932.5% / +497.7%, i.e. at h=5/20 a model-free volatility sort beats the 'recommended' basket; least-volatile-5: -26.5% / +35.1% / +72.0%. The basket's forecast sigma is 3.96% / 8.46% / 15.0% (universe ~2.4% / 5.7% / 11.4%). Also the panel intro says '每天依本頁相同的規則…挑出分數最高的前 5 名' but evaluate_universe is called with only_buy_signals=False (server notes inside a collapsed <details> say '不論是否達買進門檻'): at h=1/h=5 the page shows 0 buy candidates while the validated basket trades every day, and at h=1 every name in it has negative expected net return (net>0 on 0.0% of name-days). The page also shows top 10 while validation uses 5. DESIGN s9(1) benchmarks (mean sigma, mean beta, top-k-by-sigma, equal-weight) are not implemented, so the misleading view is currently unlabelled.

**Repro.** python3 $MINE/b_sigma_bench2.py  (offline, cache_product2, ≈4 min under load); UI: python3 $MINE/a_render_all.py uv (V8 harness) then view $MINE/render/picks_uv_h5_1280_s4.png

**Evidence.** $MINE/render/picks_uv_h5_1280_s4.png, picks_uv_h5_1280_s5.png; this session's b_sigma_bench2.out ($MINE/p3/out/b_sigma_bench2.out); real2/ubt_h5.json notes[1].

**Proposed fix.** Implement DESIGN s9(1) benchmarks and show them in the table (forecast sigma and beta of the basket, top-k-by-sigma return, equal-weight) with the verdict colour; validate the same object the page shows (only_buy_signals=True and the rank_signal leaders) and say which; replace '本頁相同的規則' by the true rule; put the volatility/beta sentence next to the equity curve, not in <details>; cap the headline return display or show Sharpe/vol first.

**Verifier:** real=True severity=medium  
The core claims hold, but the page is less misleading than the finding says, so I downgraded it from high to medium. DESIGN s9(1) is a binding addendum and is not implemented in the validation panel.

What is confirmed:
- `rank_signal`, `leaders` and `laggards` do not exist anywhere in backend/ or frontend/ (grep returns nothing).
- `universe_bt.py` still ranks by the cost-adjusted score `(E[R]-cost)/sigma` (`_Panel.__init__` line ~111, `picks()` line ~123). DESIGN s9(1) says it MUST rank by `rank_signal`.
- The report has no mean sigma, no mean beta and no top-k-by-sigma benchmark. Its only benchmarks are the equal-weight universe and TAIEX.
- The saved real h=5 report shows the basket at +932.5% total, 44.0% annual vol and Sharpe 1.26. Equal-weight is +298.1% with 18.5% vol and Sharpe 1.56.
- The saved offline benchmark (`b_sigma_bench2.json`) shows a model-free most-volatile-5 basket returning +464.8% / +1448.4% / +1124.8% at h=1/5/20. The current score basket returns +519.8% / +932.5% / +497.7%. At h=5 and h=20 the volatility sort therefore beats the "recommended" basket. Its beta is about 1.45 against 0.82 for equal-weight. I read these numbers from the saved output and did not rerun the 4-minute job.
- `jobs._run_universe_backtest` calls `evaluate_universe` without `only_buy_signals`, so the default `False` applies. The note says "不論是否達買進門檻", and that note sits inside a collapsed `<details>`.
- The intro in `view_picks.js` (line 335) says "每天依本頁相同的規則…挑出分數最高的前 5 名". The validation trades every day regardless of the buy gate, while the page shows 0 buys at h=1 and h=5. The page lists top 10 while validation uses `top_k=5`.
- In the rendered screenshot the equity curve shows 10.32x against 3.98x and 2.69x with no volatility or beta caveat beside it.

Why I downgraded:
- The panel opens with a prominent verdict box titled "無預測優勢 / 沒有統計上顯著的選股優勢".
- The KPI table shows annual vol (44.0% vs 18.5%), max drawdown (-46% vs -26%), a lower Sharpe than equal-weight (1.26 vs 1.56) and PSR vs equal-weight of 25.4%.
- A careful reader is therefore not told the strategy beats the market. The remaining harm is that the headline return and chart are unlabelled volatility/beta exposure, and that the intro wording is inaccurate.

The finding's suggestion to cap the headline return display is optional polish. The binding requirement is s9(1).

**Verifier refined fix.** Minimal change set (evaluation plus frontend text):

1. backend/evaluation/universe_bt.py, `_Panel.__init__`: also store `RS = mu_log/sigma_log`, falling back to `Φ⁻¹(p_up)`, as a (T,S) matrix. Make `picks()` rank by `RS` instead of `SC`, with ties broken by `PU`, then by code. This is the DESIGN s9(1) requirement. Keep `SC` and `net` only for the `only_buy` gate. IC, spread and `picks_stats` must use the same ranking matrix.

2. `evaluate_universe`: add an additive report key `exposure`, for example `{"mean_sigma_top","mean_sigma_universe","beta_top","beta_universe","benchmark_top_sigma":STATS}`.
   - Mean sigma is the average `SG` over held names, taken from the tranche weights `W`, versus the universe average.
   - Beta is the OLS slope of the basket's daily return `ret` on the TAIEX return `tx`, or on the equal-weight return `ew` when TAIEX is missing.
   - The top-k-by-sigma benchmark is `pn.picks` with the key `SG`, run through `S.simulate_weights` and `stats_block`.
   - Add the key to `UNIVERSE_SPEC` as optional so old cached reports still validate. Bump the cache key or version in jobs.py (`parts`) so stale reports are recomputed.
   - Update the notes text for the new ranking. Replace "(預期報酬 − 來回成本) ÷ 波動" with the true rule and state whether the buy gate was applied.

3. frontend/view_picks.js:
   - Fix the intro at line 335 so it states the real rule (rank by `rank_signal`, all names, top 5, no buy gate).
   - In `renderUniverse`, add a "風險暴露" row or caption directly under the KPI table or equity chart: basket vol, beta and forecast sigma versus universe, and the top-k-by-sigma return. Do not leave this only in `<details>`.
   - Add a column for the top-by-sigma benchmark and a curve in the chart.

Side effects:
- Tests in tests/test_evaluation*.py that assert score-based `picks` or universe IC values will need updating. The DESIGN s9 regression test is the new one: |Spearman(rank_signal, sigma)| < 0.9 on a synthetic universe where p_up is independent of sigma.
- frontend fixtures and the schema test need the new optional key.
- Cached `universe_backtest` results change after the ranking switch, so re-run them.
- Real-data results will change a lot. From the offline benchmark, the `rank_signal` basket at h=1 returned -77.7% with IC t=3.6 (an IC significantly positive but a negative return after costs). The verdict wording must handle that case.


## P-06 [medium] DESIGN s9 rank_signal = mu_log/sigma_log is the mirror-image volatility sort; leaders would be low-vol defensives, laggards are often empty, and at h=20 leaders historically trailed laggards

- owner: evaluation  
- location: DESIGN.md s9(1)-(3); future backend/evaluation/recommend.py + universe_bt.py  
- reviewer: review:product

**Description.** Real 1250-day panel (b_rank_structure.py / b_alt_key.py / b_sigma_bench2.py): mean daily Spearman(rank_signal, sigma) = -0.362 / -0.556 / -0.618 (h=1/5/20; p10-p90 -0.60..-0.07, -0.70..-0.42, -0.71..-0.52), top-10 list overlap day-to-day 71% / 90% / 95%. Cross-sectional information of the key: IC +0.0231 (t=+3.64) at h=1 but the top-5 net-of-cost basket loses -77.7% (Sharpe -1.35); at h=5 IC +0.0054 (t=0.41), basket +54.8% (Sharpe 0.70, beta 0.38) vs equal-weight +298.1%; at h=20 IC -0.0284 (t=-1.13), basket +64.0%, top-minus-bottom spread -2.64% (t=-2.23). So the proposed 'leaders' (today h=20: 台積電, 中華電, 玉山金, 台灣大, 元大金, mean sigma 5.8% vs universe 11.4%) are the low-vol end and the validated top-k would show a drag/neutral result, not a view. The s9 definition of laggards/lean=bearish requires p_up<0.5, but p_up>0.5 for 60/60 names today at h=5 and h=20; historically #names with p_up<0.5 has median 2 (h=5) / 1 (h=20) and the set is empty on 21.2% / 19.5% of days (h=1: median 31, never empty). Exploratory alternative (sigma-neutral, baseline-relative; NOT a recommendation, one of 5 keys tried after looking, in-sample, survivorship-biased universe, so multiple-comparison inflated): (mu - mu_naive)/sigma has IC +0.0253 (t 4.08) / +0.0413 (t 2.71) / +0.0753 (t 2.52) with corr to sigma +0.14/+0.26/+0.30 and top-bottom spread +0.23% (t 3.80) / +0.76% (t 2.44) / +2.98% (t 2.70); p_up - p_naive: IC +0.0199 (t 2.95) / +0.0376 (t 2.42) / +0.0466 (t 1.76). argon_advice.md s3 claims mu/sigma 'eliminates the artefact'; the data shows it replaces a high-vol tilt by a low-vol tilt.

**Repro.** python3 $MINE/b_rank_structure.py; python3 $MINE/b_alt_key.py; python3 $MINE/p3/b_laggards_empty.py  (all offline on $SC/cache_product2)

**Evidence.** $MINE/logs/b_rank_structure.out, $MINE/p3/out/b_alt_key.out (re-run this session, identical to earlier), b_laggards_empty.py output.

**Proposed fix.** Keep the relative list but (a) define it on a baseline-relative key ((mu-mu_naive)/sigma or p_up-p_naive) and test that key's IC/spread with the same HAC/cluster rules before adopting, (b) always display sigma (and a '低波動' tag) beside leaders so the tilt is visible, (c) define laggards as the lowest k by rank_signal without requiring p_up<0.5 and label them '相對最弱（機率仍高於 50%）' when all p_up>0.5, (d) require the universe report to publish the basket's mean sigma/beta so the leaders list can be judged.

**Verifier:** real=True severity=low  
The numbers reproduce exactly. I re-ran all three scripts on a private copy of the cache (cache_verifyP06). They give mean Spearman(rank_signal, sigma) of -0.362 / -0.556 / -0.618 at h=1/5/20, and top-10 day-to-day overlap of 71 / 90 / 95%. The h=20 top-5 basket has IC -0.0284 and a top-minus-bottom spread of -2.64% (t -2.23). Names with p_up<0.5 have median 31 / 2 / 1 and the set is empty on 0% / 21.2% / 19.5% of days. Under the s9 definition the lean=bearish set is empty on the same share of days (21.2% at h=5, 19.5% at h=20, 0% at h=1), because its p_up<0.5 condition is the same filter.

The defect is real as a specification issue. None of the spec's text is implemented yet: grep finds no `rank_signal` in backend, frontend or tests, and `laggards` appears only in an unrelated comment in jobs.py.

Two parts of the finding hold up:
- DESIGN s9(1) claims the new key fixes the volatility artefact. It only flips it: the old score sorted by volatility (rho +0.95 to +0.99) and mu/sigma sorts the opposite way (rho -0.36 to -0.62).
- s9(2) and (3) require p_up<0.5 for laggards and lean=bearish. Since p_up>0.5 for almost every name at h=5/20, the "always shown" laggards list is empty about one day in five, and was empty for 60/60 names at the latest date.

Severity is low rather than medium for three reasons:
- s9 already partly mitigates it. It requires the universe report to publish top-k mean sigma and beta and the sigma-sorted and equal-weight benchmarks, and it requires an evidence banner (amber when no_edge). Because universe_bt ranks by the same key, the negative h=20 IC would be shown to the user.
- Preferring a higher mu/sigma, and so a low-vol tilt, is an intrinsic property of a Sharpe-style key, not a bug. The negative h=20 spread reflects the models having no edge, not a wrong key.
- Empty laggards is an honest outcome, not a wrong claim. It is a product and UX gap.

Two parts of the proposed fix are weaker than the finding suggests:
- Fix (a), switching to a baseline-relative key, rests on in-sample comparison of 5 keys on a survivorship-biased universe, and the reviewer flags this itself. It should not be adopted without out-of-sample testing. The t-stats of 2.4 to 4 are inflated by the multiple comparisons.
- The "leaders trailed laggards" claim holds only at h=20 (t -2.23). At h=1 the spread is +0.10% (t 1.98) and at h=5 it is -0.28% (t -1.05).

**Verifier refined fix.** This is a documentation and spec change, to be made before recommend.rank_signal and universe_bt are implemented. It touches DESIGN.md s9(2)-(3), the future recommend.py and the frontend picks page. It has no side effects on the existing tests, because nothing implements the key yet.

1. DESIGN s9(3): define `laggards` as the bottom `top_k` by `rank_signal` with no p_up<0.5 requirement. Add a flag, for example `laggards_all_above_half: true`, or per-REC `p_up`, so the UI labels them 「相對最弱（機率仍高於 50%）」 when all of them have p_up>0.5. Keep `lean=bearish` as is, since it is already neutral when p_up>=0.5 and so remains conservative. Optionally add a note that bearish is often empty.
2. DESIGN s9(1): replace the claim that rank_signal removes the volatility artefact with an accurate statement. rank_signal is a low-volatility-tilted relative ordering (mean Spearman with sigma -0.36 / -0.56 / -0.62). Require every leader row to show sigma, with a 「低波動」 tag when sigma is below the universe median.
3. The universe report already must publish basket mean sigma and beta and the sigma-sorted and equal-weight benchmarks. Add the universe mean sigma next to the basket's, and the day-to-day top-k overlap.
4. Do not switch the key to (mu-mu_naive)/sigma or p_up-p_naive yet. Test it out-of-sample, with a pre-registered key, the HAC and cluster rules from the s9 audit rules, and a point-in-time universe. Until then, keep the baseline-relative key only as an exploratory candidate.


## P-07 [high] The strict `buy` flag is a persistent long bias at h=20 (23 names/day, avoid never fires) and cannot fire at h=1; flagged names under-performed the universe by 1.04 pp/20d with no information after sigma-matching

- owner: evaluation  
- location: backend/evaluation/recommend.py classify_signal / config.BUY_P_UP, AVOID_P_UP; server evidence block (not yet implemented)  
- reviewer: review:product

**Description.** Over 1250 days x 60 stocks (b_signal_freq.py): h=1 net>0 on 0.0% of name-days (round-trip cost 0.57% vs E[r_1]~0.05%), buy list empty on 98.3% of days (max 1), avoid empty on 70.4%, both empty on 68.9%; h=5 buy empty on 81.8% of days (max 4), avoid empty on 99.8%, both empty 81.5%; h=20 buy list NEVER empty (median 23, p10-p90 15-30, max 35), 38.1% of all name-days flagged buy, P(buy today|buy yesterday)=97.7% (mean run 41 days), avoid fired 0 times in 75,000 name-days. Today's 26/60 buys: p_up mean 0.547 (sd 0.016, range 0.514-0.584), net>0 for 59/60, the universe's own historical 20-day up-rate is 57.1% (> BUY_P_UP 55%), the model's no-information baseline component already has p mean 0.531 (9 names >=0.55; 8 of the 26 buys qualify on baseline drift alone), component means garch 0.546 / hmm 0.551 / dlm 0.547; Spearman(p_up, p_naive)=0.77; p_up is also lower for high-sigma names (rho -0.57). Realised outcomes (b_buy_value.py/b_buy_value2.py, 1230 days with >=3 buys): hit-rate buys 56.7% vs all names 56.4% (+0.25 pp, CI -1.70..+2.14); mean 20d simple return buys +1.39% vs all +2.43% = -1.04 pp (block-bootstrap CI -1.62..-0.46 for 20-day blocks, -1.76..-0.35 for 120-day blocks; non-overlapping samples t=-3.28/-2.35/-2.53); buys have forecast sigma 6.4% vs 9.1%; matching each buy to the non-buy with the closest sigma gives +0.08 pp (CI -0.43..+0.57). So the h=20 'buy' chip is low-vol/drift, not selection; the horizon the UI marks 偏多·買進 is the one with a net negative history relative to simply holding the universe.

**Repro.** python3 $MINE/p3/b_signal_freq.py; python3 $MINE/p3/b_buy_value2.py; python3 $MINE/p3/b_buy26.py   (offline, $SC/cache_product2; each 1-3 min)

**Evidence.** $MINE/p3/b_signal_freq.json; this session's outputs of b_buy_value.py, b_buy_value2.py, b_buy26.py.

**Proposed fix.** Gate 'buy' on edge relative to the baseline (p_up - p_naive >= margin AND edge_covers_cost AND evidence level >= weak_edge) instead of an absolute 55% that sits below the universe base rate; at h=1 drop the buy/avoid chips and say '成本高於可預期的一日報酬，無法交易'; surface the historical emptiness ('歷史上 98%/82% 的日子沒有任何標的達標') and the buy-flag validation numbers in the evidence block.

**Verifier:** real=True severity=medium  
I reran b_signal_freq.py and b_buy_value2.py against the cache_product2 data and the numbers match the finding. At h=1 the buy list is empty on 98.3% of days and avoid is empty on 70.4%. At h=5 buy is empty on 81.8% of days. At h=20 buy is never empty (median 23 names/day, 38.1% of name-days, 97.7% persistence) and avoid fires 0 times. The buys-minus-universe 20-day contrast is -1.04 pp, with block-bootstrap CI -1.62..-0.46 (20-day blocks) and -1.76..-0.35 (120-day blocks). The non-overlapping t-stats are -3.28/-2.35/-2.53. Matching each buy to the non-buy with the closest forecast sigma gives +0.08 pp (CI -0.43..+0.57), so the h=20 buy flag is a low-vol/positive-drift flag, not selection. The code confirms the mechanism: classify_signal in backend/evaluation/recommend.py fires buy on an absolute p_up >= 0.55 and net > 0. Because the universe's own 20-day up-rate is about 57%, that threshold sits below the base rate. At h=1, net = E[r] - 0.57% is never positive, so a buy cannot fire. The UI still labels the flag 偏多·買進 (frontend/format.js SIGNALS.buy; view_stock.js chip), and the picks page card is titled 買進候選. DESIGN §9 items 4, 5 and 6 (evidence banner, 偏多候選 wording, lean/leaders/laggards) are not implemented anywhere in backend or frontend (grep for evidence, lean, rank_signal and leaders finds nothing in recommend.py, server.py or view_picks.js). I rate it medium rather than high for four reasons. (1) The h=20 drift-driven buy list is already documented in §9, along with the planned wording and banner fix, so the main part is a known to-do item. (2) The headline -1.04 pp underperformance is statistically real but, per the reviewer's own sigma-matched test, is a volatility composition effect. The universe is also today's liquid top-60 (survivorship-biased toward high-vol winners), and an absolute buy flag is not defined as a promise to beat the universe. (3) In absolute terms the buys' hit rate is 56.7% against 56.4% for all names, and their mean 20-day return is +1.39%, so the flag is not harmful in absolute terms, only uninformative and mislabeled. (4) The empty-list behaviour at h=1 and h=5 is already handled with an honest empty state in view_picks.js. The real defect is wording and gating: a no-information flag is presented as 買進 at the horizon where it fires constantly.

**Verifier refined fix.** Do not change classify_signal's thresholds. The tests, strategy.py and the universe_bt only_buy picks all depend on it, and the p_naive-margin gate needs p_naive in every horizon dict. Take the minimal presentation and evidence fix instead. (1) Implement DESIGN §9 item 4 and item 6. In backend/server.py, fill recommendations.evidence from the cached universe validation (source: universe_validation|none). In frontend/format.js SIGNALS.buy, change label to 偏多候選 and short to 偏多 whenever the evidence verdict_level is not moderate_edge or strong_edge (default when evidence is missing). Keep 偏多·買進 only at moderate_edge or above. The same applies to the chip in view_stock.js:153 and the 買進候選 card title in view_picks.js:90. (2) Add to the chip title and the picks empty-state a sentence with the measured numbers: the h=20 flag fires on about 38% of name-days with 97.7% persistence, so it mostly reflects market drift and low volatility, not selection. At h=1 say 成本 0.57% 高於一日預期報酬，買進訊號實際上不會觸發. At h=5 say 歷史上約 82% 的日子沒有標的達標. (3) Optionally add an additive per-REC flag edge_vs_baseline = p_up - p_naive (when the naive component's p is available) and require it >= 0.02 only for a stricter high-confidence tier, leaving signal unchanged. Side effects: frontend tests that assert the 偏多·買進 label, and the server tests for the recommendations schema, need updating. The backend evaluation, quant and data tests are unaffected as long as classify_signal is left alone.


## P-08 [high] Stock page labels 20-day (and 5-day) horizons '偏多·買進 / 信心 中-高' with P(漲) to 0.1 pp, but shows no validation for those horizons and the evidence contradicts the label

- owner: frontend  
- location: frontend/view_stock.js backtestCard/def list (lines ~402-450), horizon card chips; format.js SIGNALS  
- reviewer: review:product

**Description.** view_stock.js loads the 歷史驗證摘要 for h=1 and h=5 only (header comment line 8, request at line 446), while the 20-day card is the one that carries the red '偏多·買進' chip (and 0050's 5-day card too). Real evidence for 2330 h=20 (bt_2330_h20.json): verdict no_edge, accuracy 63.7% = always-up 63.7%, Brier skill -2.75%, ECE 10.2%, calibration bins predicted 0.538-0.579 vs realised 0.49-0.82, the higher bucket is LESS accurate ([0.55,0.60) 63.0% n=1015 vs [0.50,0.55) 67.6% n=204), and the exact 'p>=55% & net>0' rule returns +186% (Sharpe 0.92) vs buy-and-hold +337% (Sharpe 1.11). For 0050 the live card says P(漲) 60.9% '信心 高' (h=20) while the [0.60,0.65) bucket realised 65.2% (n=431) vs base rate 64.0% (p=0.61), ECE 6.3%, 90% interval coverage 83.3%. The 信心 label is |p-0.5|-based, not validated skill. Probabilities are shown at 0.1 pp (51.5%, 58.4%) while 2330 h=1 ECE is 3.1% (predicted p spans only 0.471-0.524) and the h=5 ECE is 5.0%. The 'open full validation' button links to #/backtest/2330 (h defaults to 1), not the horizon of the buy chip.

**Repro.** python3 $MINE/p3/render_live.py (V8 harness, venv python) or python3 $MINE/a_render_all.py stock; numbers: python3 - <<'EOF' (load real2/bt_2330_h20.json -> report.probabilistic.ece, direction.by_confidence) EOF

**Evidence.** $MINE/render/stock_2330_1280_s1.png (20日 card), stock_0050_1280_s1.png; real2/bt_2330_h20.json, real2/bt_0050_h20.json.

**Proposed fix.** Load the summary for all three horizons (or at least the horizon of any non-hold chip) and put the verdict chip inside each horizon card; when verdict is no_edge render the signal chip as outlined/neutral '偏多（未達顯著）' and drop 信心 高/中 (use '模型一致度'); round probabilities to integers (or 'about 55% ±3'); deep-link the button with ?h=<horizon>.

**Verifier:** real=True severity=medium  
The core of the finding holds. The stock page shows a red '偏多·買進' chip and '信心 高/中' on the 20-day card (and on 0050's 5-day card), and no validation is shown for h=20. The 歷史驗證摘要 card is built from defs = [HORIZONS[0], HORIZONS[1]] and only loads h=1 and h=5 (view_stock.js lines 8 and 431). The cached forecasts confirm the live labels: 0050 h=20 p_up is 0.609, which gives buy with 信心 高 (|p-0.5| >= 0.10), and 2330 h=20 p_up is 0.584, which gives buy. The real evidence for 2330 h=20 contradicts the chip. bt_2330_h20.json has verdict no_edge, accuracy 63.7% equal to always_up 63.7%, Brier skill -2.7% and ECE 10.4%. At h=1 and h=5 the 2330 verdict is also no_edge, with ECE 3.1% and 5.0%. The confidence label comes from confidence_label() in backend/evaluation/recommend.py. It uses only |p-0.5|, model agreement and the volatility percentile, so it is not validated skill. That conflicts with DESIGN section 0, which says every claim of skill is shown with out-of-sample evidence.

Four things lower the severity from high to medium:
1. The h=1 and h=5 validation card is itself specified in DESIGN section 7, as is the chip wording '偏多·買進'. The omission of h=20 is a gap in that design, not a code deviation.
2. The page is not silent about weak evidence. The chip tooltip states the 55% and net>0 rule. The backend note at the bottom says '多數個股的預測並沒有統計上顯著的優勢…請搭配「歷史驗證」'. The backtest card caveat says to treat the probabilities as near 'uncertain' when the verdict is no_edge.
3. The reviewer's claim about the button is partly wrong. btColumn already deep-links with '?h=' + def.h, and view_backtest.js reads ?h=1, 5 or 20. Only the bottom primary button '開啟完整歷史驗證' lacks ?h, and no h=20 link exists anywhere on the stock page.
4. The 0.1 pp display of P(漲) is a precision nit. The page's own caption says 約 50% means odds are even, so it is not misleading by itself.

Net: a real honesty and UX gap, concentrated on the h=20 and 0050 h=5 cards. It does not break any contract and the page already has generic disclaimers.

**Verifier refined fix.** Minimal change, all in frontend/view_stock.js. (1) In backtestCard, set defs = HORIZONS (add h=20). In card.start, chain loadOne(0) -> loadOne(1) -> loadOne(2). Show a '排隊中' placeholder for cells 1 and 2. Update the header comment from h=1|5 to h=1|5|20. (2) Cache each verdict level in S.bt[h] once its column loads, and re-render the matching horizon card chip row. For any horizon with S.bt[h].verdict.level === 'no_edge', show the signal chip as muted with text '偏多（未達顯著）' or '偏空（未達顯著）' (keep the ▲/▼ icon). Replace confidenceChip's '信心 高/中' with '模型一致度 高/中'. Before validation has loaded, put the same muted wording on the chip with a '驗證載入中' title. Do not change the backend signal or confidence fields. (3) Make the primary button href TQ.codeHref(S.code,'backtest') + '?h=20', or add a third per-column link; the per-column links already carry ?h. (4) Optional: show P(漲) with fmtPct(p, 0) in the gauge-val, or keep one decimal and add a '±' note. Side effects: the frontend test (tests/test_frontend*.py) that asserts two bt-cells and requests only h=1/5 must be updated to 3. The summary card adds a third /api/backtest h=20 job, which is about one more walk-forward run per first visit and is cached afterwards. DESIGN.md section 7 wording 'h=1 & 5' becomes 'h=1, 5 & 20'. No backend or evaluation tests are affected. Alternative with no extra compute: add only the generic muted '未經驗證' wording to every non-hold chip.


## P-09 [medium] No disclaimer above the fold on the picks page (the page with buy cards, weights and a 'suggested portfolio'); stock page has one at the top

- owner: frontend  
- location: frontend/view_picks.js bannersFor/notesCard  
- reviewer: review:product

**Description.** view_stock.js:112 renders a disclaimer-strip right under the header; view_picks.js has the disclaimer only inside notesCard at the bottom (line ~212) plus the global footer. Rendered heights: picks h=5 4,262 px at 1280 (first disclaimer at y≈4,110) and 6,369 px at 390; h=20 5,168 px / 9,046 px. The first screen (3 banners, 4 market tiles, then 偏多·買進 cards with 建議權重) contains no 非投資建議 text. DESIGN s7 asks for a prominent but non-obnoxious disclaimer.

**Repro.** python3 $MINE/a_render_all.py picks ; view $MINE/render/picks_h20_1280_s1.png

**Evidence.** $MINE/render/picks_h20_1280_s1.png and picks_h20_390_s1.png (no disclaimer visible); stock_2330_1280_s1.png (strip present).

**Proposed fix.** Add the same disclaimer-strip (one line, amber) directly under the page header and merge it into the evidence banner (see P-10).

**Verifier:** real=True severity=medium  
Confirmed by reading the code. view_stock.js warningsBlock (lines 112-114) pushes a disclaimer-strip as the first element under the stock header. view_picks.js render() (lines 406-418) emits bannersFor banners, then marketStrip, buysCard, portfolioCard/avoidsCard, allCard, universePanel and finally notesCard. notesCard holds the only in-page disclaimer (line 212), at the very bottom. bannersFor contains only data-freshness and market-state banners and no disclaimer wording. The only other 非投資建議 text on the page is the global footer in index.html (bottom of page) and a hover-only tooltip on 建議權重. DESIGN section 7 asks for a prominent but non-obnoxious disclaimer on the stock view. The picks page is the one showing buy cards, weights and a 建議組合, and the project's own results show no demonstrable edge, so the omission matters more there. It is a UX and compliance gap, not a functional bug, and the footer and bottom disclaimer exist, so medium severity is right and I would not raise it. I did not re-render screenshots. The code-level claim is unambiguous.

**Verifier refined fix.** In frontend/view_picks.js render() (around line 411), insert the strip before the bannersFor banners: slot.appendChild(h('p', { class: 'disclaimer-strip' }, h('span', { class: 'disclaimer-icon', 'aria-hidden': 'true' }, '⚠'), h('span', null, '以下為統計模型的研究輸出，非投資建議；機率不是保證，多數個股並無統計上顯著的優勢，建議權重與建議組合僅為模型的資金配置試算，請搭配「歷史驗證」判讀。'))). Better to put it in bannersFor so it is also present in the asof_complete===false branch. Keep notesCard's bottom disclaimer. The existing .disclaimer-strip CSS already works, so no style change is needed. Side effects: tests in tests/ that assert the picks banner count or order (frontend tests, venv python) may need updating, and any layout or height assertions shift by one line. No backend impact. Optional: reuse the same text constant as view_stock.js, for example by adding TQ.DISCLAIMER_SHORT in core.js, so the wording stays consistent.


## P-10 [medium] Wording/layout: current labels imply advice (今日推薦, 偏多·買進, 建議組合/建議權重/建議避開, 信心高, P(漲), 分數); proposed honest-but-useful picks page

- owner: frontend  
- location: frontend/view_picks.js, format.js, core.js; server evidence block (jobs.py)  
- reviewer: review:product

**Description.** Current strings: nav/title '今日推薦', chip '偏多·買進' (format.js SIGNALS), cards '買進候選', '建議組合', '建議權重', '建議避開', '信心 高/中/低', 'P(漲)', constant '分數' column. DESIGN s9(6) (偏多候選 / 偏空／避開, never 建議買進 below moderate_edge, amber evidence banner, leaders/laggards) is not implemented. Real-data facts the design must respect: strict buy/avoid lists are empty on 98%/82% of days at h=1/5 and the buy list is never empty at h=20 (P-07); B is degenerate (P-03); rank_signal is a low-vol sort (P-06); universe validation verdict is no_edge for h=1/5/20 (IC t=-0.47/0.92/1.41; Brier skill negative for 52/59/58 of 60 stocks).

**Repro.** Render current pages: python3 $MINE/a_render_all.py picks; compare with proposal below.

**Evidence.** All numbers in P-03/P-06/P-07; real3/real2 payloads.

**Proposed fix.** LAYOUT (top to bottom). 0) Title '今日模型觀察' (not 推薦); sub: '模型對 60 檔大型股的相對看法，不是買賣建議'. 1) Evidence banner from `evidence`, always visible, amber when no_edge, grey when source=none (with a button '執行驗證（約 2 分鐘）'): '歷史驗證（{start}~{end}，{n} 檔，{h} 日）：模型的選股排序沒有統計上顯著的預測力（IC {ic:+.3f}，p={p}）。以下只是相對位置，非買賣建議、不保證獲利。' (the disclaimer lives here). Add to evidence: sigma/beta of the validated basket, top-k-by-sigma return, buy-flag validation (hit rate and excess 20d return vs all and vs sigma-matched), share of days with an empty list. 2) One-line market tone computed from counts: 'N/60 檔上漲機率 > 50%：反映整體市場長期上漲趨勢，不代表個股有優勢' when N>=55. 3) '相對偏多 / 相對偏弱' = leaders/laggards (5 each) by the validated key, neutral blue/gray chips (no red/green buy card), columns: 名稱, 上漲機率 (integer %), 預期波動 σ (+ '低波動' tag), 相對位置 percentile bar; subtitle '在這 60 檔裡誰相對較被模型看好'. 4) '扣成本後能否交易': KPI '扣成本後預期報酬為正：x/60 檔（明日 0/60）' + the cost explanation (一次買賣先要賺 0.57%) + history ('歷史上 98% 的日子沒有任何標的達標'); list names only if edge_covers_cost AND evidence>=weak_edge, and call them '偏多候選'. 5) '模型配置範例（研究用）' card collapsed by default and rendered only if B>0 AND evidence>=weak_edge; otherwise one line '目前不提供配置：個股間差異無法與雜訊區分 → 100% 現金'. 6) Full table sorted by percentile: 相對位置, 名稱, 收盤, 1日, 上漲機率, σ, 淨預期報酬, 覆蓋成本 ✔/✖, 傾向 chip; drop 分數 and integer ranks. WORDING: 買進→偏多候選 (買進候選 only when evidence>=moderate_edge); 避開→偏空／避開 (註：僅提醒勿追買，不提供放空); 建議權重/組合→模型配置權重/範例（研究用）; P(漲)→上漲機率; 信心→模型一致度; 分數→相對位置; probabilities as integer % ('約 55%，誤差約 ±3 個百分點'). Colours: keep 紅漲綠跌 for price changes only; use neutral/blue for model stance when verdict=no_edge.

**Verifier:** real=True severity=medium  
Reproduced by reading the code. The advice-implying strings are present as reported. frontend/index.html has the nav label 今日推薦. view_picks.js has the title 今日推薦, the cards 買進候選, 建議組合 and 建議避開, the column 建議權重, the chips 信心 高/中/低, the metric P(漲) and the 分數 column (score_shrunk). format.js SIGNALS has 偏多·買進 and 偏空·避開. view_stock.js also has 信心. I could not find leaders, laggards, evidence, rank_signal, lean or score_rank_pct anywhere under frontend/*.js. In backend/*.py, only evaluation/report.py has an unrelated "evidence" key, and there is none in jobs.py or server.py. So DESIGN section 9 items 1-4 and 6 are only partly implemented, and the section 9(6) wording rule and amber banner are not implemented. Section 9(6) is explicitly binding: 買進 becomes 偏多候選, 避開 becomes 偏空／避開, and 建議買進 is never used below moderate_edge. The no_edge evidence is established by the task's real-data results: the models match the base rate and Brier skill is negative. Under that evidence, 建議/推薦/信心 wording is misleading. It is a real finding, but a product-honesty issue, not a computational bug. Mitigations exist: the footer and the DISCLAIMER state 非投資建議, the empty-state text says 沒有訊號本身就是模型的答案, and the 建議避開 card says no short selling. That keeps it at medium. The proposed fix is a full page redesign that depends on backend work from P-03, P-06 and P-07, which is out of scope here. The rewording and banner can be done independently. A note on the finding's repro: the cited script a_render_all.py was not needed, because the strings are static in the source.

**Verifier refined fix.** Do this in two independent steps, and skip the full redesign.

Step 1, wording only (frontend, no backend dependency). Keep internal keys and the REC schema unchanged.
- index.html nav and view_picks.js title() and h1: 今日推薦 becomes 今日模型觀察. The page-sub becomes '模型對流動性較佳大型股的相對看法，不是買賣建議'.
- format.js SIGNALS: buy label becomes 偏多候選 and short becomes 偏多; avoid label becomes 偏空／避開 and short becomes 避開. Use 買進/建議買進 only if evidence level >= moderate_edge, which in practice means never for now.
- view_picks.js: 買進候選 becomes 偏多候選; 建議組合 becomes 模型配置範例（研究用）; 建議權重 becomes 模型配置權重 (card and table); 建議避開 becomes 偏空／避開 (sub-line: 僅提醒勿追買，不提供放空); P(漲) becomes 上漲機率 (also in the avoid card); 信心 becomes 模型一致度 (also in view_stock.js:158); 分數 becomes 風險調整分數 (a full change to 相對位置 needs rank_signal from P-06).
- Update the setStatus text at about line 419 and the strings asserted in test_front_picks.py, test_front_shell_format.py and test_front_stock.py.

Step 2, evidence banner (needs the section 9(4) server work).
- Add an `evidence` block to the recommendations response (jobs.py or server.py), filled from the cached universe validation for the same horizon, or source:'none'.
- In view_picks.js bannersFor(), add a banner. It is amber when verdict_level is 'no_edge' with the text '歷史驗證：模型的選股排序沒有統計上顯著的預測力（IC x，p=y）。以下只是相對位置，非買賣建議。'. It is grey with an '執行驗證' button when source is 'none'.
- Add leaders/laggards cards only once P-06 provides rank_signal.

Side effects: the frontend tests above and fixtures need their strings updated. The server tests need a new evidence key, and it is additive. The backtest and method pages mention '今日推薦' in view_backtest.js:466 ('請同時參考「今日推薦」頁的整體驗證'), so rename it there too.


## P-11 [low] Static UI text contradicts or duplicates server behaviour (independence vs Ledoit-Wolf, hard-coded 60/55%/45%/0.25/15%, computed_at time zone)

- owner: frontend  
- location: frontend/view_picks.js:112,207,209,440; view_stock.js:153; backend/evaluation/recommend.py:396  
- reviewer: review:product

**Description.** (a) At h=20 the 建議組合 card subtitle (view_picks.js:112) and the 方法說明 page say '假設各檔彼此獨立，未考慮相關性' while the server note on the same page says '已用 Ledoit-Wolf 收縮共變異數考慮各檔相關性' (jobs.py passes `returns`; recommend.py note). (b) '從 60 檔…', '首次計算需要預測全部 60 檔，通常約一分鐘', '≥ 55%', '≤ 45%', '0.25 倍凱利', '單檔上限 15%' are hard-coded in view_picks.js/view_stock.js while config.py and TWQ_UNIVERSE_SIZE (4-500, README) can change them. (c) recommend.build_recommendations stamps computed_at with datetime.now().astimezone() (server local zone) whereas backtest/universe payloads use Taipei time, and format.fmtDateTime prints the digits verbatim: TZ=UTC gives 2026-10-24T14:12:32+00:00 and the banner '計算於 2026-10-24 14:12' for a Taipei user.

**Repro.** for tz in Asia/Taipei UTC America/New_York; do TZ=$tz python3 -c "from backend.evaluation import recommend as R;print(R.build_recommendations([],5)['computed_at'])"; done  (from /home/gary/stock); grep -n '60 檔\|55%\|15%' frontend/view_picks.js

**Evidence.** real2/recs_h20.json notes[3]; view_picks.js lines 112, 207, 440; TZ test output.

**Proposed fix.** Drive thresholds, universe size and the covariance sentence from the payload (`portfolio.note`, add `params:{buy_p,avoid_p,kelly,max_weight,universe_size}`), use data.taipei_now() in recommend.py.


## P-12 [low] backend.server.main prints the 'ready' banner before installing SIGINT/SIGTERM handlers: a SIGINT sent right after the banner is lost 50% of the time under load when SIGINT is inherited as ignored (`./run.sh &`)

- owner: server  
- location: backend/server.py main() lines ~512-531  
- reviewer: review:product

**Description.** main() prints '台股量化實驗室已啟動…' then installs handlers (including the SIG_IGN -> default_int_handler repair). A supervisor or test that signals as soon as it reads the banner hits the window. With 30 trials at load average 7.9, 15/30 SIGINTs were lost (server kept running; needed SIGTERM). This is the cause of test_sigint_works_even_when_started_in_the_background failing in 3/3 loaded full-suite runs in clean venvs ('timeout' != 0); default (non-ignored) SIGINT before serve_forever prints a traceback instead of a clean exit.

**Repro.** cd /home/gary/stock && python3 $MINE/p3/sigint_race.py 30 3   (higher CPU load -> higher loss rate)

**Evidence.** this session: 'SIGINT lost (server still running after 3.0s): 15/30 | load1 7.94'; $MINE/logs earlier d_fresh/unittest_full.log.

**Proposed fix.** Install SIGINT/SIGTERM handlers (and the SIG_IGN repair) before make_server/print; print the banner last.


## P-13 [low] Unit test suite is not hermetic and is load-flaky: real network calls + writes data_cache/names.json, 300/994 tests silently skipped without mini-racer, Linux-only tests, wall-clock assertions

- owner: server  
- location: tests/server_fakes.py:295-312, tests/test_server_jobs.py, tests/test_server_process.py, tests/test_data_http.py:283, tests/test_quant_engine.py:614, README.md:173  
- reviewer: review:product

**Description.** (a) README says unit tests do not use the network, but test_server_jobs.Params.test_backtest_params (9 attempts) and RealProcessPool.test_a_dying_worker_is_survived (30 attempts) contact openapi.twse.com.tw / www.tpex.org.tw (logging proxy: CONNECT openapi.twse.com.tw:443 x6, www.tpex.org.tw:443 x3 in one module run) and, when the network works, write data_cache/names.json (240 KB) into the project's REAL cache dir (both clean-copy full runs ended with data_cache/names.json although none existed before). (b) In clean venvs on a loaded 4-core box the full suite (994 tests, 524-660 s) failed 2-3 tests each time but never the same set: test_raw_utf8_query_like_curl_sends_it (tests/server_fakes.py raw_exchange stops reading as soon as headers+'Content-Length' arrive, so the body is empty whenever the server's two send() calls land in separate segments; failed 6/8 on py3.11 and 5/8 on py3.9 at load 9.4), test_sigint_works_even_when_started_in_the_background (P-12), test_forecast_latest_speed_invariants (5.16 s and 5.00 s vs <5.0 s), test_twenty_four_symbols_in_parallel_never_exceed_six_requests (server-side peak 7 > 6 although the client semaphore is 6). (c) Without py_mini_racer 300 tests are skipped (skip text embeds an absolute scratch path), the README test section does not say so. (d) test_server_process uses /proc, pgrep and preexec_fn without skip markers (Linux only).

**Repro.** cd $MINE/d3/repo_t39 && HTTPS_PROXY=http://127.0.0.1:18471 NO_PROXY=127.0.0.1,localhost ../py39/bin/python ../which_test_hits_network.py test_server_jobs  (start python3 $MINE/d3/logproxy.py 18471 $MINE/d3/proxy_hits.log first); flaky helper: for i in 1..8 run python -m unittest discover -s tests -p test_server_api.py -k test_raw_utf8_query_like_curl_sends_it under load

**Evidence.** $MINE/d3/full_repo_full311.log, full_repo_full312.log, t39_*.log, proxy_hits.log.

**Proposed fix.** Point every test at a temp cache dir and fake _http_get/names; make raw_exchange read until len(body)>=Content-Length; replace wall-clock asserts by generous bounds or mark them slow; measure concurrency client-side; add skipUnless for POSIX-only tests; document the mini-racer requirement (and `pip install` line) in README; fix README line 173.


## P-14 [medium] Implausible or degenerate upstream data is accepted and forecast as if real (sigma=0 with p_up 7.6%, +900% last bar, 2-week-old data not flagged, malformed answers mapped to unknown_symbol or HTTP 500)

- owner: data  
- location: backend/data.py parse_yahoo_chart/_refresh/get_price_data; backend/jobs.py forecast_payload/assemble_forecast  
- reviewer: review:product

**Description.** Fault injection through TWQ_YAHOO_CHART with the mock at $MINE/c/mock_yahoo.py (circuit breaker kept closed by interleaving good fetches). Results: (1) perfectly flat series (mock 9013; real analogue: a halted/suspended stock): /api/forecast returns 200 with p_up=[0.076,0.094,0.12] and sigma=[0,0,0] and no warning (picks drops such names via plausible(), the single-stock page does not). (2) +900% last bar (9012: 140.96->1409.55): accepted, forecast 200, no note although the market has a ±10% limit. (3) data ending 2026-10-08 (9022): asof 10-08, stale=False, '明日' target date in the past. (4) timestamps in ms (9016): HTTP 500 internal ('year 58776 is out of range') instead of data_unavailable. (5) response without a quote block or with empty timestamps (9017/9004) is turned into unknown_symbol 404 and remembered for UNKNOWN_TTL=600 s, so a transient malformed answer makes a valid, never-cached code look non-existent for 10 minutes. Handled well: HTML 200, 429 (Retry-After 1), truncated JSON, gzip header lie, HTTP 500 -> 503 data_unavailable + Retry-After 5 (6 s of retries); nulls/zeros/negatives/duplicate and unordered rows cleaned (1500->1493 bars); 100-bar listing -> 422 insufficient_history; 300 bars -> forecast with short_history note; 1.8e11 prices fine. Items (2)-(5) are hypothetical upstream faults (low likelihood); (1) is plausible for suspended stocks.

**Repro.** python3 $MINE/c/mock_yahoo.py 18453 & cd /home/gary/stock && TWQ_CACHE_DIR=$(mktemp -d) TWQ_HOST=127.0.0.1 PORT=18452 TWQ_YAHOO_CHART='http://127.0.0.1:18453/v8/finance/chart/{sym}' python3 -m backend.server &  then python3 $MINE/c/c_mock2.py 18452 9013 9012 9016 9017 9022

**Evidence.** outputs of c_mock2.py in this session (9013 forecast p_up/sigma, 9012 lastbar, 9016 500, 9022 asof 2026-10-08); server_mock.log traceback.

**Proposed fix.** Reject/flag series whose last-k bars are constant or whose |1-day return| exceeds ~11% without a split/dividend event (return data_unavailable or add a visible note); return insufficient_data for sigma==0; wrap malformed-payload parse errors (ValueError/OverflowError) as DataUnavailable; do not negative-cache 'unknown' when the answer was malformed rather than an explicit Not Found; mark data older than last_trading_date_expected by >1 session as stale.

**Verifier:** real=True severity=low  
I reproduced every item with a fresh mock (own ports and private cache copy). The code gaps are real, but they all need an upstream fault that real Yahoo or TWSE data essentially does not produce, and the user-visible damage is small. Real, hypothetical-trigger gaps, none a deliberate design choice: (1) A flat series of 1300 bars gives 200 with p_up=[0.076,0.094,0.12] and sigma=[0,0,0], with no note. This is purely synthetic. The realistic case, a halt with the last 40 bars frozen and volume 0 (mock 9025), gives a sane forecast: sigma about 0.0097 and p_up about 0.45. recommend.plausible() already drops the degenerate case in the picks pipeline, so only the single-stock page lacks the guard. (2) A +900% last bar is accepted and chg_1d=8.99 is shown. The jump is visible on the chart and in the change field, and a real ±10% limit market cannot produce it. A one-day jump above 10% can only be a split or adjustment glitch, and _consistent_overlap already guards incremental refreshes. (3) A series ending 2026-10-08 returns stale=False. The note states 資料截至 2026-10-08, and view_stock.js (lines 91-92) and view_picks.js (lines 194-195) already show a '尚未更新至…' warning when status.last_trading_date > asof and the market is closed. So this is surfaced in the UI. The only wrong text is the '明日' target date, which lies in the past. In picks, laggards are skipped as '未與其他標的同步'. 'stale' deliberately means a network-failure fallback to cache, not an old last bar. (4) Millisecond timestamps give HTTP 500 internal instead of data_unavailable, confirmed. It is cosmetic robustness, since the response is still a structured error and no real feed does this. (5) A missing quote block or empty timestamps becomes NotFoundError('no usable rows'), then UnknownSymbol 404, then negative-cached for UNKNOWN_TTL (600 s). This is confirmed (9017 and 9004 both return unknown_symbol). It is the only item with a plausible transient trigger, and the effect is a wrongly 'not found' answer for at most 10 minutes. Overall it is a robustness polish, not a correctness or honesty failure of the forecasts, so 'medium' is too high.

**Verifier refined fix.** Worth doing cheaply, all in backend/data.py plus one guard in backend/jobs.py. (a) get_price_data/_refresh: distinguish an explicit Yahoo 'Not Found' from 'parsed but no usable rows'. In parse_yahoo_chart, if timestamps are empty or the quote block is missing, raise DataUnavailable('malformed Yahoo response') instead of returning empty arrays. In _refresh, remove the `if len(bars['d'])==0: raise NotFoundError` branch, or keep it only when the payload was an explicit Not Found. A transient malformed answer then becomes 503 data_unavailable and is not negative-cached. This is the same protection as the existing HTML/truncated-JSON handling. (b) parse_yahoo_chart: validate the timestamp scale before the date arithmetic, for example `if n and (ts.max() > 4e9 or ts.min() < 0): raise DataUnavailable('timestamps out of range')`. Alternatively wrap the datetime64 conversion in _to_price_data and _bars_to_rec with except (ValueError, OverflowError) and raise DataUnavailable. This turns the 500 into 503 + Retry-After. (c) jobs.py forecast_payload: after computing the forecast, if any horizon sigma <= 0 or not finite (or the last 20 completed log returns are all 0 with a nonzero history), raise ApiError('insufficient_history' or 'data_unavailable', '價格序列無變動（停牌/資料異常），無法預測'). This reuses the idea behind recommend.plausible(p, er, sg). Optionally append a note when abs(chg_1d) > 0.11. Do NOT change the meaning of 'stale', and skip the '2-week-old data' flag: the frontend already shows the 尚未更新 chip. Optionally make the note text say 該日之後的「明日」已過 when next_trading_day < today. Side effects: tests/test_data_*.py cases that assert empty-timestamp payloads map to UnknownSymbol (expect NotFoundError/UnknownSymbol for 'no usable rows') would need updating; test_server_* error-code expectations for a flat series are not currently covered. Nothing in quant, evaluation or frontend depends on these paths.


## P-15 [low] Name-table parsing crashes on non-dict rows (search and recommendation jobs -> 500); a short but valid table collapses the universe

- owner: data  
- location: backend/data.py parse_name_tables, _fetch_name_tables, build_universe  
- reviewer: review:product

**Description.** Injecting TWSE/TPEx answers by patching data._http_get: a JSON list of strings, nulls or lists makes parse_name_tables raise AttributeError ('str' object has no attribute 'get'), which _fetch_name_tables does not catch, so /api/search, get_universe and recommendation jobs fail with 500/internal. A valid-JSON table with a single row (good_but_weird case) makes build_universe drop every curated code that is absent from the live table, giving a 1-name universe until the names TTL (24 h, 30 min if a source is missing) expires; recommendations then fail with '可用於驗證的股票少於 4 檔'. HTML, dict, empty list and null-field cases degrade correctly. Hypothetical (needs a malformed upstream), hence low.

**Repro.** python3 $MINE/p3/c_names_garbage.py

**Evidence.** c_names_garbage.py output: list_of_strings/nulls/lists -> EXC AttributeError; good_but_weird -> universe ['2330'] 1.

**Proposed fix.** isinstance(r, dict) guard per row; catch AttributeError/TypeError in _fetch_name_tables; sanity floor (e.g. >=500 STOCK_DAY_ALL rows) before replacing the cached table/universe.


## P-16 [low] Runtime disk cache grows without bound per distinct request (results/ 47 KB per parameter set, wf/ per years value); only pruned at startup by age

- owner: server  
- location: backend/jobs.py DiskStore/save_result/prune_cache, JobManager  
- reviewer: review:product

**Description.** 150 distinct thr values for one stock: results/ 13 -> 163 files, +7.07 MB (47.1 KB per distinct request) in 37 s, server RSS 166 -> 199 MB (finished-job registry holds up to 128). thr has 6 significant decimals and disc/slip/years/exec/sizing multiply the key space, wf/ keeps one pickle per (symbol, test_bars) so fractional `years` creates a new walk-forward file each time (compute cost limits abuse speed). prune_cache runs only in the warm-up thread (skipped by --no-warm) and only for files older than 14 days. No request-rate or per-client limits; any web page can fire GET requests at http://localhost:8000 (Host check does not stop simple GETs) to start jobs (max_queue=64). Local-only tool, so low.

**Repro.** python3 $MINE/c/c_diskgrowth.py <port> <server_pid> <cache_dir>  (against a server started with TWQ_CACHE_DIR=<cache_dir>)

**Evidence.** c_diskgrowth.py output this session: '150 distinct thr values -> ready=150 in 37.1s ; results dir (13, 645861) -> (163, 7717684) ; +7.07 MB (47.1 KB per distinct request) ; server rss 199 MB'.

**Proposed fix.** Cap results/ and wf/ by count/size (LRU delete on save), quantise thr/disc/slip/years to the UI grid, prune periodically not only at startup.


## P-17 [low] HTTP hardening observations (all local-only): GET with a body is parsed as a pipelined request, HEAD executes jobs, unbounded connection threads, non-JSON error bodies

- owner: server  
- location: backend/server.py Handler/_handle/make_server  
- reviewer: review:product

**Description.** c_http.py: a GET with Content-Length whose body is another request line yields two 200 responses on one connection (request-desync pattern if ever fronted by a proxy); HEAD /api/* runs the full handler (starts jobs); 400 half-open connections create 406 threads (+8 MB RSS) until the 120 s idle timeout, /api/status still answers in 7 ms; 300 parallel history requests all 200 (p50 1.1 s, max 1.75 s). TRACE/CONNECT/PROPFIND -> 501, 414 and 431 answer with stdlib HTML not the documented JSON envelope. Host guard blocks evil.com, localhost.evil.com, 127.0.0.1.evil.com, 127.0.0.1@evil.com, 0.0.0.0, 127.1, 2130706433, 'localhost.' but accepts empty/absent Host, duplicate Host (first wins) and absolute-form URIs (not browser-generatable). Range is ignored (200 full). Path traversal (28 variants incl. %2e%2e, %5c, double-encoding, NUL, backslash, dotfiles) all 404. No defect demonstrated beyond robustness; speculative for a localhost tool.

**Repro.** python3 $MINE/c/c_http.py <port> | sed -n '/smuggling/,+2p'; python3 $MINE/c/c_load.py <port> <pid>

**Evidence.** c_http.py/c_http2.py/c_load.py outputs this session.

**Proposed fix.** Set close_connection when Content-Length>0 on GET/HEAD (or read and discard the body), answer HEAD for /api/* without starting jobs, cap concurrent connections (BoundedSemaphore in process_request), send JSON for 4xx/5xx generated by http.server (override send_error).


## P-18 [low] UX polish list found while viewing real renders (first viewport, mobile table, fan-chart dates, retry on permanent errors, search, progress text)

- owner: frontend  
- location: frontend/view_stock.js, charts.js:601, format.js:233, core.js search, view_picks.js; backend/jobs.py:966  
- reviewer: review:product

**Description.** (a) Stock page: the three probability cards start at y≈870 px (1280 wide) and y≈1,040 px (390 wide), below a ~600 px chart, so the first viewport shows no probability at 1366x768 or on a phone. (b) 00400A (141 bars): permanent insufficient_history error shows a 重試 button (picks 503 is retryable; this is not). (c) Picks full table at 390 px: columns after P(漲) are cut off with no scroll hint (render/picks_h5_390_s3.png). (d) /api/search?q=大盤 and ?q=TWII return [] although the API accepts ^TWII/大盤 and the index page renders. (e) Progress card shows two denominators: '下載行情 37/60  37 / 122（30%）'. (f) Fan-chart tooltips use weekday-only dates (format.js weekdaysAfter; charts.js:601): for asof 2026-10-23 day 1 reads '約 10/26' (a TWSE holiday) while the card says 2026-10-27. (g) Between 13:30 and 14:00 the forecast note says '目前為盤中（尚未收盤）' although status says market closed (bar finality is 14:00). (h) Ctrl-C/SIGTERM while a job runs prints a 30-line ERROR traceback ('RuntimeError: compute pool is closed') although all workers exit within 1 s. (i) For stocks at h=1 the threshold slider is inert: 0 trades for every thr in 0.5-0.95 even with disc=0 slip=0 (2330), because net>0 never holds; the page explains the empty strategy but keeps the control. (j) Jargon: 'P(漲)', 'bp' in the drivers table.

**Repro.** python3 $MINE/a_render_all.py stock picks; python3 $MINE/c/c_clock3.py (injected clock server); curl 'http://127.0.0.1:<port>/api/search?q=%E5%A4%A7%E7%9B%A4'

**Evidence.** renders in $MINE/render and $MINE/render3; c_clock3.py output; shutdown log $MINE/d3/run_sh_cold.log.

**Proposed fix.** Put a compact 3-horizon summary strip above the chart; hide 重試 for insufficient_history/unknown_symbol; make the table sticky-first-column with a scroll shadow; include ^TWII in search; add fan.dates (trading calendar) to the forecast payload; distinguish 13:30-14:00 in the note; swallow shutdown RuntimeError in JobManager._run.


## P-19 [low] README/requirements accuracy: Python floor is 3.9 not 3.10; test claims; DESIGN s6 lacks the additive keys the server sends

- owner: server  
- location: README.md:21,173; DESIGN.md s6  
- reviewer: review:product

**Description.** What is true: all backend modules import on Python 3.9.25 (numpy 2.0.2, scipy 1.13.1) and 10 test modules pass there (the only failure is the flaky helper in P-13); vermin reports 3.9 as the minimum (str.removeprefix at server.py:423; 3.7 for syntax alone, tests 3.8), so README's 'Python 3.10 以上' is conservative, not wrong. numpy>=1.26/scipy>=1.11 holds (py3.10 + numpy 1.26.4 + scipy 1.11.4). The same data gives the same forecast under py3.9/3.10/3.11/3.12 x numpy 1.26.4-2.5.2 x scipy 1.11.4-1.18.1 (max |diff| 1.1e-7 over p_up/E[r]/sigma/q05/q95, 2330 and 2317, h=1/5/20). README accurate: ~77 MB data_cache after a cold run (claims ~80 MB), precompute --n 8 works (77 s under load), CPU cost of cold recommendations ≈150 CPU-s and of the 60-stock universe validation ≈356 CPU-s, i.e. ≈50 s / ≈120 s on 3 idle workers as README says (wall times I measured, 181 s and 573 s, were inflated by load 5-10 from parallel agents). README inaccuracies: 'unit tests never use the network' (P-13). DESIGN s6 does not document: status.{jobs,modules,uptime_s}, search.q, history.n_total, forecast.{data_through,engine,intraday_bar_dropped,stale,horizons[].mu_log/sigma_log,drivers[].contribution_by_h/unit}, recommendations.{skipped,stale}, backtest.{asof,computed_at,stale}, universe.{computed_at,skipped}. Frontend reads error.detail (5 reads) which the server never sends (by design, harmless). Windows/macOS: run.sh is bash only (README says use python -m backend.server), spawn pool and os.nice/malloc_trim paths are guarded, tests are Linux-only (P-13).

**Repro.** python3 -m venv v && v/bin/pip install vermin && v/bin/vermin --no-tips --eval-annotations backend ; python3 $MINE/p3/env_consistency.py <repo_root> (per interpreter)

**Evidence.** this session: vermin output, 3.9 import + test runs, p3/env_consistency.py outputs ($MINE/p3/out/env_*.json), cpu_cold.log.

**Proposed fix.** State 'Python 3.9+ (tested 3.9-3.12)'; fix the network claim; add the additive keys to DESIGN s6 (or mark them as extensions).


## P-20 [low] Synthetic test fixtures drift from real payloads (history dividends order, recommendations empty-state shape)

- owner: server  
- location: tests/fixtures/history_*.json, recommendations_h1.json, recommendations_h5.json, recommendations_empty.json  
- reviewer: review:product

**Description.** tests/fixtures/history_2330.json and history_0050.json list dividends newest-first ([2026-09-16, 2026-06-17]) while the real API returns oldest-first (12 entries from 2023-12-14). tests/fixtures/recommendations_h1.json has 0 buys/10 avoids and recommendations_h5.json 8 buys/10 avoids with 2 short notes and a generic portfolio.note, whereas the real h1/h5 payloads have 0 buys/0 avoids over 60 scored names, 7-8 notes and the note '目前建議全數持有現金：沒有同時符合買進條件、且經收縮後仍具正期望報酬的標的。'; recommendations_empty.json models n_scored=0, not the common real state '60 scored, nothing actionable'. Only tests/fixtures/real/* match reality (those are fine and the V8 renders of real payloads work). The frontend ignores history.dividends entirely, so no user-visible effect today. schema_check.py passes on all 21 real payloads (0 problems).

**Repro.** python3 - <<'EOF'
import json;j=json.load(open('/home/gary/stock/tests/fixtures/history_2330.json'));print([d['d'] for d in j['dividends']])
EOF  vs  json.load(open('$MINE/real2/history_2330.json'))['dividends'][:3]

**Evidence.** python3 comparison output this session; $MINE/p3/e_diff.py.

**Proposed fix.** Regenerate the synthetic fixtures from real payload shapes (ascending dividends, 60 scored/0 actionable) or delete them in favour of tests/fixtures/real/*.

