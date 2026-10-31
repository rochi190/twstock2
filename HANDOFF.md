# HANDOFF — 台股量化實驗室 (TW Quant Lab)

交接時間：2026-10-31。給「接手繼續做完」的 agent / 開發者。先讀這份，再讀 `DESIGN.md`（規格與契約，§9 為最新、最重要），再看 `docs/audit/`（審查與修正的完整紀錄）。

> 專案資料夾**沒有 git 歷史**（這個 repo 是第一次把它放上來）。所有「之前怎麼了」的資訊都在本文件與 `docs/audit/`。

---

## 0. 使用者要什麼（原文意圖）

使用者（使用繁體中文溝通，回覆請用繁體中文）最初的需求：

> 幫我寫一個台股投資的網頁app，用前沿數學，讓我可以選指定股票，可以看到他預計的漲跌，以及會有當日推薦的股票購買建議(因為它會跌或漲)，可以用過去的紀錄檢驗模型正確性

也就是：(1) 選一檔台股看預測漲跌；(2) 每日推薦（偏多／偏空）；(3) 用歷史紀錄驗證模型是否正確；(4) 數學要「前沿」。

使用者偏好與限制（請沿用）：
- 誠實第一：報酬近乎不可預測，UI 必須如實呈現基準線、p 值、校準，**允許結論是「沒有統計優勢」**。不要為了好看而調參數、不要用測試期結果挑模型。
- 凡是對外、不可逆的動作（推送、刪除、發佈）先確認。
- 使用者的額度經常用完；工作被切斷是常態。**每個階段結束前先讓測試全綠、寫下狀態**，不要留半成品。
- 不要把使用者的個人資訊（信箱等）寫進任何公開檔案或 commit。

---

## 1. 現況總覽（一眼看懂）

| 模組 | 路徑 | 狀態 | 測試（交接時實測） |
|---|---|---|---|
| 量化核心 | `backend/quant/` | ✅ 審查修正完成（有若干「刻意未做」項目，見 §6） | 177 通過 |
| 評估/回測/推薦 | `backend/evaluation/` | ✅ 審查修正完成，**有一項需要人決定（§5 D1）** | 347 通過 |
| 資料層 | `backend/data.py` | 🟡 修正已做、有測試，但修正者在回報前被切斷 → **需驗收（§4 B）** | 160 通過 |
| 伺服器/工作佇列 | `backend/server.py`, `backend/jobs.py` | 🟡 同上（evidence 填入在 `jobs.py`） | 172 通過 |
| 前端 | `frontend/` | 🟢 §4A 完成（2026-10-31）：測試與 fixtures 已對齊新契約；四個頁面在 1280/390 px 渲染後肉眼檢查過。**仍未在真實瀏覽器驗證（§7.1）** | 353 全過、1 略過 |

後端合計 856 個測試全過；前端 353 個全過（第一次接手時有 16 個失敗，原因全是前端測試停在舊契約，已於 2026-10-31 修完，見 §4 A）。

**app 現在可以跑**（啟動方式見 §3）。⚠️ 2026-10-31 接手時 Yahoo 對沙箱回 429，**沒有用真實資料跑過端到端驗證（§4D 的伺服器實測尚未做）**。

### 誠實結論（務必保留在產品文字裡）
- 真實資料（2330、0050、6488 與 60 檔流動性最高的股票，1250 個交易日、2021-08 ~ 2026-10，walk-forward）：三個預測期間（1/5/20 日）的模型**都沒有可證明的預測優勢**；命中率≈基準率。（注意：驗證的檢定力有限，見 §6「統計（檢定力很低）」——這是「未能證明有優勢」，不是「證明沒有優勢」。）
- 原本看到的「Brier skill 全面偏負」大半是**事後（oracle）基準線的假象**；改用因果（expanding window）基準後 Brier skill ≈ 0（h=1/5/20 的 62 檔平均 +0.004 / −0.021 / +0.004 %）。
- 原本推薦頁的排名排的是**波動度**（成本調整分數 Spearman(score,σ)=0.98/0.94/0.72），整體驗證的「前 5 名 CAGR 43~58%」其實是高波動、高 beta（約 1.5）曝險，且存在存活者偏差；模型無關的「波動度前 5 名」籃子表現相同或更好。
- 改用 `rank_signal = mu_log/sigma_log` 後，排名變成**低波動偏向**（Spearman(rank_signal,σ) −0.19/−0.42/−0.57，今日截面），這**也不是選股能力**，UI 必須如實標示（§2）。
- 區間涵蓋率：90% 區間約 83~90%（略偏窄；週一/連假後的變異數是已知原因之一）。
- 從未在真實瀏覽器中驗證過前端（見 §7）。

---

## 2. 最重要的設計變更（相對於 v1.0；都已寫入 `DESIGN.md`）

1. **推薦排序**（DESIGN §9.1，最重要）：舊的成本調整分數 `score=(E[r]−cost)/σ` 排的是波動度（h=1 時 60 檔 `net` 全為負，σ 越大分數越接近 0 → 排越前面；h=5/20 還含 σ²/2 的凸性項）。改為：
   - `rank_signal = mu_log / sigma_log`（缺值時退回 `Φ⁻¹(p_up)`）；`rank`、`all` 的順序、`leaders`、`laggards`、`lean`、`score_rank_pct` 全用它，**絕不再用 `score`**。
   - `score` / `score_shrunk`（James–Stein）只用於嚴格 `buy/avoid` 門檻、`edge_covers_cost`、Kelly 權重。
   - **誠實標示**：`rank_signal` 是低波動偏向的排序；每個領先/落後者必須顯示 σ，σ 低於全體中位數加「低波動」標籤（`low_vol`）；用詞為「相對強度（預期報酬/風險）」，不是「風險調整報酬」，也不是買賣建議。
   - `laggards` 不要求 `p_up<0.5`（真實資料 h=5/20 全部 60 檔 `p_up>0.5`），改以 `laggards_all_above_half` 讓 UI 標「相對最弱（上漲機率仍高於 50%）」。`lean=bearish` 維持嚴格規則（底部 20% 且 `p_up<0.5`），常為空是預期的。
   - **不要**直接改用「相對基準的預測差 `(mu−mu_naive)/σ`」：它在樣本內看起來有效（IC t≈4.1/2.7/2.5），但那是看過資料後從 5 個候選鍵中挑的，需先在「不同股票的 holdout」驗證（§8）。
2. **整體驗證 `evaluate_universe`**：用同一個 `rank_signal` 選股；報告新增 `exposure`（籃子 σ/beta vs 全體）、`benchmark_top_sigma`（依 σ 選前 k 的無模型對照）、等權基準、前 k 名日間重疊率；日期分群 HAC 檢定。
3. **統計基準線必須因果**（DESIGN §9 審查規則 a/b/c）：Brier/log-loss skill 與 `train_majority` 用 expanding-window 基準；oracle（測試期事後）基準另列、標明；重疊 h 日樣本以有效樣本數 n/h 或 HAC 處理；橫斷面合併以日期分群。
4. **平盤（R=0）定義一致**：模型發布的 `p_up = P(R>0 | R≠0)`（`WalkForward.config['p_up_event']='up_given_move'`），評估排除平盤；校準器的目標也排除平盤（原本 h=1 的 p_up 被系統性高估 +2pp，2412/3045 達 +7pp）。⚠️ 因此 UI 不可再用 `1 − p_up` 當「下跌機率」（`frontend/charts.js:703` 仍在這樣做 → 待辦）。
5. **PIT 隨機化**（離散 tick/平盤）、**無成交占位 K 棒**（成交量 0 且 H=L）在建模前剔除、**週五的 h=1 變異數 ×1.16**（週末效應）、HMM 狀態標籤門檻 0.10。
6. **判定（verdict）**：方向/Brier 檢定與基準相反時封頂為 `weak_edge`（原因 `conflicting_evidence`）；`no_edge` 文字揭露檢定力不足；ECE 警告需超過雜訊下限 `ece_noise_floor`。
7. **推薦組合**：`B==0` 或沒有正的收縮後淨優勢 → 不配權重，`portfolio={cash_weight:1,n:0,suppressed_reason:'B0'|'no_positive_edge'}` 與說明文字（100% 現金）。
8. **用詞**：`buy`→「偏多候選」、`avoid`→「偏空／避開」；**除非 evidence 等級 ≥ moderate_edge 且 `beats_ew_after_costs!==false`，否則不得出現「建議買進」/「偏多·買進」**。
9. **快取鍵**：含 OHLCV 與程式碼指紋（`jobs.code_fingerprint` 雜湊 `backend/quant/*.py` 的名稱/大小/mtime，`ENGINE_VERSION='1.1'`、`CACHE_VERSION=3`）。

完整契約差異：`docs/audit/FIX_REPORT_quant.md`、`docs/audit/FIX_REPORT_evaluation.md` 的 `contract_changes` 段。

---

## 3. 環境與執行

- Python 3.11；執行期只需 `numpy`、`scipy`（其餘純標準庫；前端無建置、無 CDN）。
- 啟動：`python3 -m backend.server`（或 `./run.sh`）。預設埠 8000——**開發機上 8000 常被別的專案佔用，請用別的埠**：
  ```
  TWQ_HOST=127.0.0.1 PORT=8765 python3 -m backend.server      # 開 http://localhost:8765/
  ```
  - 用 `TWQ_HOST` 而不是 `HOST`：conda 會匯出 `HOST=x86_64-conda-linux-gnu`，會讓伺服器啟動失敗。
  - 其他環境變數 `TWQ_*`：見 `README.md`「環境變數」。`TWQ_CACHE_DIR` 可換快取目錄，`TWQ_OFFLINE=1` 只用快取。
- 首次執行會下載資料並建立 `data_cache/`（約 95 MB，**不在 repo**，可隨時刪除重建）。需要網路（Yahoo chart API、TWSE OpenAPI）。冷啟動時間（4 核）：單檔預測約 5 s；推薦（60 檔）約 70 s；單檔回測 5~6 s；整體驗證約 2 分鐘。
- 測試（`tests/` 不是 package，用 discover；frontend 測試需要有 `mini-racer` 的 Python）：
  ```
  python3 -m unittest discover -s tests -p "test_quant_*.py"     # 177
  python3 -m unittest discover -s tests -p "test_eval_*.py"      # 347
  python3 -m unittest discover -s tests -p "test_data_*.py"      # 160
  python3 -m unittest discover -s tests -p "test_server_*.py"    # 172
  # 前端（V8 假 DOM）：
  python3 -m venv .venv && .venv/bin/pip install mini-racer pillow weasyprint pymupdf numpy scipy
  .venv/bin/python -m unittest discover -s tests -p "test_front_*.py"   # 345（交接時 16 失敗）
  ```
  quant 測試約 3 分鐘；其他各 20~110 秒。
- **沒有 node、沒有能用的瀏覽器**：無頭 Chromium/playwright 在含 canvas 的頁面會卡住（已放棄）。前端用 mini-racer(V8) 跑假 DOM；渲染檢查用 weasyprint→PDF→pymupdf→PNG（`tests/front_render.py`，CJK 字型需要 Noto Sans TC 或 `fonts-noto-cjk`），canvas 用 `tests/canvas_replay.py`（Pillow）重放。
- 資料來源陷阱：Yahoo chart API 必須用 `period1=0&period2=now&interval=1d`（用 `range=max` 會被降成月線！）；上市 `.TW`、上櫃 `.TWO`；TWSE OpenAPI `STOCK_DAY_ALL`/`t187ap03_L` 取名稱與成交值；TPEx openapi 大回應曾 `IncompleteRead`；今天尚未收盤的 K 棒在台北時間 14:00 前會被丟棄；Yahoo 對休市日會塞「成交量 0、平盤」的占位 K 棒（已處理，見 §2.5）。
- 開發守則：**不要** `pkill -f <字串>`（字串出現在自己的指令列會把 shell 殺掉）；起 server 做測試請用私有快取目錄與自己的埠，並確實關掉。

---

## 4. 待辦（依優先順序；每項都有驗收條件）

### A. 前端收尾 — ✅ 已完成（2026-10-31）
**結果**：16 個失敗全修；前端 353 測試全綠。做法：舊行為的測試依新契約刻意改寫並寫明原因（用詞「偏多候選」、用 `rank_signal` 排名、儀表板整數百分比、`模型一致度`、h=20 驗證卡、`signalInfo(sig, unlocked)`），沒有放寬任何斷言；新增測試涵蓋證據橫幅、領先/落後者的 σ 與低波動標籤、分數欄隱藏、IC 晶片、平盤說明、驗證面板範圍說明。補做的程式變更：`view_picks.js`（B=0 時全體相同的「成本後分數」欄隱藏；驗證面板寫明前 5 名／列出前 10 名／不套用偏多候選門檻；權益曲線下方貼上波動與 β 提醒）、`view_backtest.js`（IC≤0 不顯示顯著晶片 S-02；`p_up_event=up_given_move` 時說明平盤不計入）、`charts.js`（儀表板無障礙文字不再用 `1−p_up` 當下跌機率）、`style.css`（`uv-scope`）。下面的清單保留作為原始需求紀錄。

**仍開放**：D1（整體驗證 h=1 判定在淨虧損時是否封頂）——政策決定，未動；前端 `verdictBox` 仍會在 `beats_ew_after_costs===false` 時顯示「中等優勢」。

（以下為原始待辦，已完成。）
推薦頁（`frontend/view_picks.js`）已重寫（有 leaders/laggards/rank_signal/低波動標籤/evidence 橫幅/`suppressed_reason`/用詞閘門），但 fixtures 與測試還停在舊契約。16 個失敗測試：
- `test_front_method`：`PageContent.test_pipeline_flow_has_eight_steps`
- `test_front_picks`：`HorizonSwitching.test_one_month_has_candidates_cards_with_all_fields`、`…test_switch_while_computing_aborts_the_old_poll`、`RealOneWeek.test_banner_market_strip_and_empty_buys`、`RealOneWeek.test_clean`、`RealOneWeek.test_full_table_has_every_scored_name_and_is_sortable`、`StatesAndVariants.test_empty_universe_and_missing_blocks`、`StatesAndVariants.test_schema_payload_with_buys_and_avoids`、`UniversePanel.test_run_shows_progress_then_verdict_kpis_stats_and_chart`
- `test_front_qa_static`：`CssClasses.test_every_class_literal_in_the_js_is_styled_or_a_documented_hook`；`test_front_shell_classes`：`ClassCoverage.test_every_class_used_by_js_is_styled`（JS 用了新 class 但 CSS 沒樣式）
- `test_front_shell_format`：`Format.test_semantics`
- `test_front_stock`：`FullPayload.test_backtest_summary_loaded_lazily_after_forecast_and_sequentially`、`…test_intervals_returns_components_confidence`、`…test_three_horizon_cards_with_probabilities_and_meaning`、`PartialPayloads.test_sparse_payload`

要做的事（來源：`docs/audit/FIX_REPORT_evaluation.md` 的 `open_issues`、`docs/audit/FINDINGS.md`）：
1. 更新 fixtures（`tests/fixtures/*.json`、`tests/fixtures/make_*.py`；已部分更新，含新增的 h=20 檔）與測試，使其符合新契約；**不得為了讓測試過而放寬測試**，舊行為的測試要刻意改並寫明原因。補 CSS 讓新 class 都有樣式。
2. 推薦頁：表頭與提示改為「相對強度（預期報酬/風險）」（`view_picks.js` 約 59/91/135/147/206 行與 401 行的「收縮」說明）；`分數` 欄在 B=0 時 60 檔相同 → 隱藏或改顯示 `score_rank_pct`；領先/落後者每列顯示 σ 與「低波動」標籤；`laggards_all_above_half` 時標「相對最弱（上漲機率仍高於 50%）」；顯示 `lean`、`edge_covers_cost`、`edge_vs_baseline`；「共 N 檔」用 `n_buy_total`；`portfolioCard` 在有 `suppressed_reason` 時只顯示說明、不畫權重表；evidence 橫幅（`no_edge` 或 `beats_ew_after_costs===false` 時為琥珀色）；**免責聲明放在首屏**（P-09）。
3. 整體驗證面板：說明真實規則（`rank_signal`、全體、前 k 名、**不套買進門檻**、驗證 5 檔而頁面顯示 10 檔）；`basket`/`exposure`/`benchmark_top_sigma`/`curve.top_sigma`/labels 放在 `<details>` **外面**；波動/beta 的提醒要緊貼權益曲線（P-05）。
4. 個股頁（`view_stock.js`）：h=20 也要有驗證卡（P-08）；`no_edge` 的 horizon 的訊號 chip 改灰色「偏多（未達顯著）／偏空（未達顯著）」；「信心 高/中」改「模型一致度 高/中」；`SIGNALS.buy`（`format.js:265`）標籤為「偏多候選」，僅在 evidence ≥ moderate_edge 且 `beats_ew_after_costs!==false` 時才解鎖「偏多·買進」；**`charts.js:703` 不可再顯示 `1−p_up` 當下跌機率**（`p_up` 現在是 P(R>0|R≠0)）；`view_method.js:286` 的 regime 命名與 PIT/p_up 說明要更新。
5. 回測頁（`view_backtest.js`）：基準說明改為「因果基準率」，另列 `brier_skill_oracle`/`brier_baseline_oracle`；ECE 旁顯示 `ece_noise_floor`；IC chip 只有 `ic>0` 才顯示（S-02）；樣本數文字依 `probabilistic.p_up_event`（`up_given_move` 時平盤不計）。
6. 驗收：前端 345 測試全綠；用 weasyprint 把每個頁面（`#/stock/2330`、`#/picks`、`#/backtest/2330`、`#/method`）在 1280 與 390 px 渲染成 PNG **實際看過**；頁面任何地方在 evidence 為 `no_edge` 時都不出現「建議買進」。

### B. 驗收伺服器與資料層（P0）— 測試層面已驗收（2026-10-31），**實機驗收未做**
`data.py`/`jobs.py`/`server.py` 的修正者在回報前被切斷，但程式與測試都在，且測試全過。請逐項確認（對照 `docs/audit/FINDINGS.md` 標「verify」的列）：
- **C-04**：刷新時 Yahoo 對最後一個已完成日給全 null 列，不得刪掉快取中有效的最後一根 K 棒（`tests/test_data_cache.py`、`tests/test_server_audit.py`）。
- **C-05**：快取鍵含 open/high/low/volume 與程式碼指紋。
- **P-01**：休市日占位 K 棒不入庫、`asof`/`next_trading_day` 正確（`tests/test_data_closures.py`、`tests/test_data_parse.py`；2026-10-23 颱風日 63/63 檔都有占位棒）。
- **P-14**：退化/異常上游資料（σ=0 時 p_up=7.6%、+900% 單日、兩週前的資料）回 422/503，不當真預測。
- **evidence 填入**（DESIGN §9.4）：`jobs._run_recommendations` 查同一 horizon 的已快取整體驗證結果（**不啟動重工作**）並用 `universe_bt.evidence_from_report`；沒有則 `{source:"none"}`。請用真實 payload 確認。
- **`tests/schema_check.py::check_recommendations`（約 634~649 行）仍是舊契約**（要求 `all` 依 `score_shrunk` 遞減）。換成：`all` 依 `rank_signal` 遞減、`rank` 為 1..n、`buys/avoids/leaders/laggards` 的 rank 等於在 `all` 中的位置、`n_buy_total>=len(buys)`、`portfolio.suppressed_reason ∈ {null,'B0','no_positive_edge'}` 且設定時權重全為 0。用真實 h=1/5/20 payload 跑過。
- 資料層尾端占位棒（量化修正者回報）：快取資料尾端有 2026-10-23 的占位 K 棒（60/63 檔），`data.completed()` 保留它；引擎已忽略（預測起點為最後一根真實棒 2026-10-22），但 `forecast['asof']`/`last_close`/`chg_1d(=0.0)` 與伺服器備註「預測起點為該日收盤價」仍取自占位列。確認 P-01 的修正已涵蓋；否則在資料層修掉。
- 其他：`/api/search` 空 `q` 要回約定的熱門清單（12 檔 + 0050/0056/00878），小 `limit` 也一樣；`README.md` 與 `requirements.txt` 對齊現實（Python 下限、`TWQ_HOST`、誠實限制章節）。

**2026-10-31 驗收紀錄**：C-04（`test_data_cache.py`）、C-05（快取鍵含 OHLCV 與程式碼指紋）、P-01（`test_data_closures.py`、`test_data_parse.py`、`test_server_audit.py`）、P-14（退化／異常上游資料回 422/503）、evidence 填入（`test_server_audit.py`：同 horizon 的已快取驗證才算 evidence，其他為 `source:none`）、`/api/search` 空 `q` 熱門清單，每項都有對應的測試且全綠。`tests/schema_check.py::check_recommendations` 其實**已經是新契約**（`all` 依 `rank_signal` 遞減、全域 rank、`n_buy_total`、`suppressed_reason`），並對真實 h=1/5/20 fixtures 全數通過——本節原本寫「仍是舊契約」已過期。`README.md`（測試基線、`TWQ_HOST`、前端測試安裝、未在瀏覽器驗證的限制）已更新。**未做**：用真實資料起 server 打各端點（沙箱連 Yahoo 回 429）；請在能連網的機器上做 §4D。

### C. 型別註解（小）— ✅ 已完成（`backend/types.py` 的 `p_up`、`pit` 註解已更新）
`backend/types.py`（原為建構者唯讀，現可更新）的註解過期：`HorizonForecasts.p_up` 實為 `P(R>0|R≠0)`，`pit` 實為對「四捨五入收盤價」的隨機化 PIT。

### D. 最終驗證（P0，收尾必做）
全部 5 組測試全綠 → 用私有快取起 server → 打 `/api/status`、`/api/forecast?symbol=2330`、`/api/recommendations?h=1|5|20`、`/api/backtest?symbol=2330&h=1`、`/api/backtest/universe?h=5` → 以 `tests/schema_check.py` 驗證每個回應 → 確認推薦頁排名不再等於波動度排序（`Spearman(rank, σ)` 不得 >0.9）→ 更新 `README.md` 的測試數字與限制 → 清掉過期的 `data_cache/{wf,forecast,results}` 孤兒檔。

---

## 5. 需要人（或使用者）決定的事

- **D1 — 整體驗證 h=1 的判定在新排名下變成 `moderate_edge`，但同一個前 5 名籃子扣成本後虧 78.8%**（CAGR −26.2%、毛報酬 +25.5%、年換手 186 倍、0.0% 的標的日預期報酬能覆蓋 0.57% 成本）。IC +0.023（t=3.62，是真的：純 −σ 的鍵 IC t=0.8），來源是 3 檔高平盤率股票的「單日反轉」（見 D2），不可交易。目前程式的做法：判定等級邏輯不變，文字與 bullet 說明這一點，`evidence_from_report` 給 `beats_ew_after_costs=false`，UI 保持琥珀色、不解鎖買進用詞；但前端 `verdictBox` 仍顯示「中等優勢」（tone info）。**建議**：當淨籃子虧損時把整體驗證判定封頂在 `weak_edge`（`make_universe_verdict`/`assess_evidence` 小改），或至少讓前端在 `beats_ew_after_costs===false` 時不顯示「中等優勢」。這是政策變更，原先不在發現清單內，故未做。
- **D2 — 單檔 h=1 的 `moderate_edge`（2412、3045、4904）**：它們的 IC（+0.115/+0.123/+0.089）是「前一日報酬的單日反轉」（IC(−r_t) 0.164/0.133/0.141），tick/σ≈0.5、7~25% 平盤，屬買賣價反彈的微結構。因果修正後它們的機率檢定不再相反，反向否決不再觸發；文字說明「僅均值檢定顯著」並附 tick 提示。是否要更嚴格（例如至少一個機率檢定 <5% 才能 moderate）？目前不套用。
- **D3 — James–Stein 保留比例 B 的輸入**：仍用成本調整分數；h=1 的 B>0 只是 `cost/σ` 的假象，h≥5 對已收縮過的預測又以 h/T 噪音縮一次（B 恆為 0）。目前 B 只是權重閘門並如實說明（真實資料三個 horizon 權重都被抑制）。若要改，需要用安慰劑（日期打亂）校準噪音水準。
- **D4 — 週一/連假後變異數**：只實作了「週五的 h=1 ×1.16」；連假後（約 4% 列、E[z²]≈2.6×）需要未來交易日曆，而 TWSE 行事曆端點只服務當年、由已實現日期推導會洩漏颱風停市資訊。是否值得引入日曆輸入（會動到 API 與資料層）？

---

## 6. 審查與修正的做法、結果（摘要；細節在 `docs/audit/`）

流程：4 位對抗式審查者（因果/數學/統計/產品整合）→ 每個非 low 的發現交給獨立驗證者重新重現（32 項：29 成立、2 駁回 `C-07`/`M-03`）→ 修正者依確認清單修（量化、評估、伺服器/資料、前端）。

- 因果性：**無法證偽**。148 次「毒化未來」測試（真實資料、任意 cut 點、未來列換成瘋狂價格/NaN/巨額股利）在 ≤cut 的列全部逐位元相同；`forecast_latest(data[:i+1])` 與 `walk_forward` 第 i 列相等（差 <3e-17）；策略/整體驗證的時序以獨立暴力法比對一致。
- 數學：各元件（GJR-GARCH-t、HMM、DLM、Pool CDF/ppf、DMA、Platt 校準）與暴力法/Monte-Carlo 對齊到 1e-11~1e-16。
- 統計（偽陽性）：無技巧的合成世界（iid、t4、GARCH-t5、漂移切換、真實引擎跑隨機漫步，250/500/750 根窗）單檔判定離開 `no_edge` 僅 5/45,150 次（0.011%）；零技巧面板 320 個 → 0 個 edge 判定。
- 統計（**檢定力很低 — S-04，請務必在產品與回報中講清楚**）：`no_edge` 不等於「證明沒有優勢」。在 h=20 植入 IC=0.33 的真實訊號，仍有 50~75% 的模擬被判成 `no_edge`；引擎+評估合起來在 h=1 只看得到 ≥0.15 的 lag-1 自相關。也就是說：目前的驗證能排除「大到明顯」的優勢，抓不到小而真實的優勢。UI 的 `no_edge` 文字已加揭露，但文案要保持「未能證明有優勢」而非「沒有優勢」。
- 刻意未做／僅部分（量化報告 `not_fixed`）：停牌復牌首日不加寬預測（4/5 的真實事件仍在 90% 區間外）；連假項與 h≥5 的日曆時間；HMM 狀態持續性（轉移對角約 0.75~0.8，arg-max 狀態每日翻轉約 31%，只修了標籤）。
- 其他 `not_fixed`（評估報告）：`trailing_sign_prediction` 平盤強制 'down'（對最佳基準無影響）；IC 虛無的模擬校準/Clark–West；DM 檢定核函數（`S-08`）；`S-05` 的超額報酬 HAC t。
- 量化的 open issues：扇形圖第 15~19 天寬度在 9/63 檔不嚴格單調（0.1~0.3%，既有、僅外觀；`schema_check.check_forecast` 會報）；h=5/20 的 `p_up` 低於測試期上漲頻率（−0.9pp/−2.2pp，因漂移收縮；2330 於 h=20 −7.6pp）；Friday 的 h=1 區間比其他日寬 ~7.7%，時間序列會出現週期性。
- **低嚴重度 backlog（未驗證）**：C-10~C-15、M-07~M-14、P-11~P-13、P-15~P-20、S-06、S-08~S-15（清單與標題見 `docs/audit/FINDINGS.md`）。其中值得優先做的：P-13（單元測試不夠密閉、會打網路並寫 `data_cache/names.json`）、P-16（磁碟快取無上限）、P-17（HTTP 強化）、P-15（名稱表非 dict 列會 500）、S-10（存活者偏差量化；整體驗證用的是「今天」的流動性前 60 名，點時間宇宙的前 5 名 CAGR 只有 −17~26%）。

---

## 7. 已知限制（寫在產品與回報裡，不要淡化）

1. **從未在真實瀏覽器驗證**前端；只有 V8 假 DOM + weasyprint 渲染（CJK 墨跡需字型）。上線前請人用 Chrome/Firefox 實測一輪，尤其是 Canvas 圖表、行動版、深色模式。
2. **存活者/選擇偏差**：整體驗證的宇宙是「今天」成交值前 60 名，絕對報酬被高估；不要拿來宣稱績效。
3. 盤中行為只用假時鐘測過（未收盤 K 棒丟棄、`asof` 標示）。
4. 整體驗證與單檔回測的「績效」都不是投資建議；模型沒有可證明的優勢。
5. 預測用的是還原權息收盤價的累積對數報酬；漲跌停 ±10% 以截斷處理；交易成本：手續費 0.1425%×折扣（預設 0.6）、賣出證交稅 0.3%（ETF 0.1%）、滑價 5bp。

---

## 8. 可選的後續研究（請先問使用者，並維持嚴格切分）

- **相對基準鍵** `(mu−mu_naive)/σ`（或 `p_up−p_naive`）：樣本內 IC +0.025/+0.041/+0.075（t 4.1/2.7/2.5），但是事後從 5 個候選挑的。若要採用：以**不同股票的 holdout**（例如流動性第 61~120 名）或「前瞻」新資料評估，hyper-parameter 只能用開發期資料決定。
- 跨股票階層式/合併面板收縮（取代現有的 James–Stein 用法）；三大法人籌碼特徵。
- 使用者有 RTX 2080 SUPER（8 GB），目前不需要（模型很小、深度序列模型容易過擬合）；若做深度序列模型實驗，必須用同一套 walk-forward 與同樣的檢定，預設不納入。

---

## 9. 檔案地圖

```
DESIGN.md            規格 + 契約（§6 JSON、§9 最新語意）— 改契約先改它
README.md            使用說明 / API / 數學摘要 / 限制
HANDOFF.md           本文件
run.sh, requirements.txt
backend/
  config.py          全域常數（HORIZONS、成本、門檻、埠）
  types.py           PriceData / HorizonForecasts / WalkForward
  data.py            資料層：Yahoo/TWSE 抓取、快取（CACHE_VERSION=3）、占位棒/休市處理
  jobs.py            工作佇列/行程池、快取鍵（code_fingerprint）、推薦與驗證工作、evidence 填入
  server.py          stdlib ThreadingHTTPServer、API 路由
  precompute.py      預先暖機
  quant/             GJR-GARCH-t、HMM、DLM、ensemble(DMA/linear pool)、calibration、mc、features、engine
  evaluation/        metrics、strategy、report(單檔回測)、recommend(推薦)、universe_bt(整體驗證)
frontend/            index.html, style*.css, core.js(路由/共用), format.js, charts.js(手寫 Canvas),
                     view_stock.js, view_picks.js, view_backtest.js, view_method.js, app.js
tests/               test_{quant,eval,data,server,front}_*.py、fixtures/、front_*.py/canvas_replay.py 輔助
docs/audit/          FINDINGS.md(64 項總表+狀態)、FINDINGS_DETAIL.md、REVIEW_SUMMARIES.md、
                     FIX_REPORT_quant.md、FIX_REPORT_evaluation.md
docs/external/       argon_advice.md（使用者提供的外部模型審查意見；其核心主張——推薦分數排的是波動度——
                     已獨立重現並採納；其建議的 μ_log/σ_log 鍵已實作，但見 §2.1 的低波動偏向提醒）
```

給接手者的最後提醒：先跑 §3 的全部測試確認基線（後端 856 全綠、前端 16 失敗），做 §4 A→B→D，過程中隨時保持「測試全綠或失敗清單明確」；每完成一個階段就更新本文件的 §1 狀態表。
