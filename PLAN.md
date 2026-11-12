# PLAN — 模型輸入強化計畫（籌碼／營收／美股隔夜／新聞事件）

撰寫日：2026-11-12。對象：接手執行的 agent（可用較便宜的模型）。
先讀 `HANDOFF.md`（現況與誠實原則），再讀本文件 §0，然後**依序逐階段執行**：每個階段結束先讓測試全綠、更新 §8 狀態表、本機 commit，才進下一個階段。接手時先看 §8 狀態表與 `git log`，從第一個未完成的階段繼續。

無人看管連續執行時的範圍：**P0 → P2 → P1 → P5 → P4a → P3a → P3b**，做完就停。以下一律不做、留待使用者回來決定：建立 Windows 排程（P4a 只寫好程式並手動跑一次 `--once`）、P3c、P4b 的產品化部分、P4c、P6、任何 push。長時間下載用背景指令執行並等待完成；遇到流量限制就等一小時再續跑，不要改用其他資料來源。

本計畫與 `HANDOFF.md §4` 的待辦（前端收尾、資料層驗收）**互相獨立**；P0–P3、P5 不碰前端與 `backend/quant/`，可以先做。

---

## 0. 給執行者的硬性規則（違反任何一條，結果就不可信）

1. **研究與產品分離**：新程式只放在 `backend/altdata/`（抓資料）、`research/`（實驗）、`tests/test_alt_*.py`、`tests/test_research_*.py`。在 §6 的閘門通過、且使用者同意之前，**不得修改** `backend/quant/*`、`backend/evaluation/*`、`backend/data.py`（改了會讓快取指紋失效，並動到 856 個既有測試）。
2. **時間點正確（point-in-time）**：每一筆外部資料都要有 `available_at`（台北時間，這筆資料「公開可得」的時刻）。決策日 `t` 的特徵只能用 `available_at <= t 日 23:59` 的資料，並且**在 t+1 開盤才交易**。各資料的可得規則見 §3 表格，照表實作，不得自行放寬。
3. **目標報酬從進場價起算**：`y_h[t,i] = ln(adjOpen[t+1+h] / adjOpen[t+1])`（開盤到開盤）。**不可**用 `close[t]` 當起點（那會把買不到的隔夜跳空算成獲利）。
4. **樣本切分固定**：DEV = 2013-01-01 ~ 2022-12-31；OOS = 2023-01-01 之後。特徵挑選、任何參數只能看 DEV。OOS **只能看一次**（§6），每次看都要寫進 `docs/research/LOOKS.md`（日期、git commit、看了什麼、結果）。
5. **特徵清單已預先登記**（§5）。不得新增、改定義、改視窗來讓數字變好。想到新點子 → 寫進 `docs/research/IDEAS.md`，不要跑。
6. **結果是「沒有優勢」也是合格的交付**。不要調參數救結果；如實寫進 `docs/research/RESULTS.md`。
7. **執行期相依只有 numpy、scipy 與標準庫**。不得加入 pandas、requests、feedparser、lightgbm、torch。XML 用 `xml.etree.ElementTree`，HTTP 沿用 `backend.data._http_get`。
8. **測試必須密閉**：新測試不得連網，用 `tests/fixtures/alt/` 內的小型樣本 JSON／XML。
9. **禮貌抓取**：TWSE／TPEx 每次請求間隔 ≥ 3 秒；FinMind 無 token 上限約 300 次／小時（下載器要能中斷續跑、遇到 402／429 就停下並回報）。新聞只存標題、連結、來源、時間，不存全文。
10. **機密**：`FINMIND_TOKEN`、任何 API key 只從環境變數讀取，不寫進檔案、不進 commit。
11. **每個階段結束**：該階段測試全綠 → 更新本文件 §8 狀態表 → 本機 commit（**不要 push**，push 要先問使用者）。額度隨時可能中斷，不要留半成品。
12. 本機是 Windows + PowerShell：用 `python`（不是 `python3`），環境變數寫法 `$env:TWQ_CACHE_DIR='...'`，不能用 `&&`。

---

## 1. 診斷：為什麼目前無法穩定賺錢

數字皆來自 `HANDOFF.md §1`、`§5`、`§6` 與 `docs/audit/`，已獨立重現過。

| # | 原因 | 證據 |
|---|---|---|
| 1 | **輸入只有該股自己的價量 + 大盤**（`backend/quant/features.py` 的 14 個技術特徵）。這是全市場最公開、最早被套利掉的資訊。 | 62 檔平均 Brier skill ≈ 0（+0.004／−0.021／+0.004 %）；命中率 ≈ 基準率。 |
| 2 | **交易成本遠大於預測到的報酬**。來回成本約 0.57%，而 h=1 的預期報酬約 +0.05%。 | h=1：0.0% 的標的日預期報酬能覆蓋成本；前 5 名籃子毛報酬 +25.5%、扣成本後 −78.8%（年換手 186 倍）。 |
| 3 | **這是「風險模型」不是「報酬模型」**。GARCH／HMM 擅長預測波動（σ），對方向（μ）幾乎沒有資訊；μ 又被強力收縮到共同漂移。 | 排名鍵 `mu/sigma` 實際上是低波動排序（Spearman −0.19／−0.42／−0.57）。 |
| 4 | **每檔股票各自建模，沒有橫斷面合併**。單檔約 1250 筆樣本、訊噪比極低。 | 單檔檢定在 h=1 只看得到 ≥ 0.15 的自相關；植入 IC=0.33 仍有 50–75% 被判成 `no_edge`。 |
| 5 | **宇宙是 60 檔最大、最有效率的股票**，而且是「今天」的前 60 名（存活者偏差）。 | 點時間宇宙前 5 名 CAGR 只有 −17~26%。 |
| 6 | **看不到事件**：美股隔夜、營收公布、重大訊息、新聞都不在輸入裡。事件日的跳空完全在模型之外。 | 2026-11-12 黃仁勳請辭：輝達一度 −12%、台積電 ADR 與台指期夜盤同步下挫——模型對此一無所知。 |

**期望值要先講清楚**：即使加入新資料後真的找到優勢，合理的樣子是「橫斷面 rank IC 約 0.02–0.04、多頭前 20% 每年贏等權全體幾個百分點、常有連續數季落後」。**「穩定賺錢」不是可達成的目標**；可達成的目標是「有統計證據的小優勢 + 誠實的風險提示」。

---

## 2. 可以加入的資訊（依「值得做的程度」排序）

端點全部在 2026-11-12 實測可用（回應欄位見 §9）。

| 優先 | 資訊 | 為什麼可能有用 | 能否回測 | 階段 |
|---|---|---|---|---|
| ★★★ | **月營收**（每月 10 日前公布，台灣獨有的月頻基本面） | 營收驚喜後的股價漂移在文獻中有紀錄；公布時點明確、散戶反應慢 | ✅ 2010 起 | P2 |
| ★★★ | **三大法人買賣超**（外資／投信／自營） | 投信連續買超在中型股有延續性；每日公布、台灣特有 | ✅ 2012-05 起 | P1 |
| ★★ | **融資融券、外資持股比** | 融資增加＝散戶槓桿（偏負面訊號）；融券回補 | ✅ 2012 起 | P1 |
| ★★ | **美股隔夜**（費半 ^SOX、那斯達克、TSM ADR、NVDA、VIX、美元台幣） | 幾乎確定能預測「隔日開盤跳空」；**新聞的影響已經反映在這些價格裡** | ✅ 1994 起 | P3 |
| ★★ | **事件／新聞存檔與風險旗標** | 不預測方向，只標示「今天有模型沒看到的大事，區間可能低估」 | ⚠️ 只能從現在開始累積 | P4 |
| ★ | 本益比／股價淨值比／殖利率 | 價值因子，慢、弱，當控制變數 | ✅ 2012 起 | P1 |
| ★ | 新聞情緒（LLM 判讀標題） | 理論上有用，但沒有歷史可驗證，要先累積半年存檔 | ❌（目前） | P4c，延後 |

**橫斷面合併**（把 150 檔放在一起學「誰相對強」）是比任何單一資料都重要的結構改變，放在 P0 的評估框架裡。

---

## 3. 資料可得時間（實作 `available_at` 的唯一依據）

`t` = 台股交易日。決策在 t 日晚上（21:30 後）或 t+1 盤前做，t+1 開盤交易。

| 資料 | 公布時間（台北，約） | 規則 |
|---|---|---|
| 日 K | 13:30 收盤 | t 日的 K 棒可用於 t 日決策 |
| 三大法人 | 當日 15:00–17:00 | t 日數值可用於 t 日決策 |
| 融資融券 | 當日約 21:00 | t 日數值可用於 t 日決策（實盤須 21:30 後跑） |
| 本益比等 | 當日收盤後 | t 日數值可用於 t 日決策 |
| 外資持股比 | 可能隔日才更新 | **保守**：t 日決策只能用 t−1 日（含）以前的數值 |
| 月營收（M 月） | M+1 月 10 日前（遇假日順延） | **保守**：自「M+1 月 10 日之後的第 2 個交易日」起可用。不得使用 FinMind 的 `create_time`（舊資料為空字串） |
| 美股日 K（美東日期 D） | 台北隔日 04:00／05:00 收盤 | 用於台北交易日 S 開盤的交易 ⇔ `D < S`（嚴格小於）。另外：`now_utc < D 日 21:30 UTC` 時該根 K 棒視為未完成，丟棄 |
| 重大訊息 | 隨時 | 用回應中的「發言日期 + 發言時間」；晚於 t 日 23:59 者歸 t+1 |
| 新聞 | 隨時 | **只信自己抓取時的 `fetched_at`**，不信來源的發布時間 |

---

## 4. 新聞可以抓，但要先知道三件事

1. **抓得到**：不需要爬蟲框架，RSS／公開 JSON 就夠（§9 有實測）。2026-11-12 實測時，Google News RSS、Yahoo 股市 RSS、鉅亨網都第一則就是黃仁勳請辭。
2. **新聞很難直接變成獲利**：以這次為例，消息出來時台指期夜盤先跌 600 點、之後「黑翻紅」。台股隔天是**開盤就跳空**，日線模型無法在跳空前進場；而且方向並不單純。
3. **RSS 沒有歷史**，無法回測；FinMind 新聞資料集有歷史（2021 可查）但一次只能查「一檔一天」，免費額度下無法大量回補。所以新聞情緒模型**現在無法驗證**，只能從今天開始存檔。

**因此本計畫對新聞的處理分三層**：

- **(a) 用價格代替新聞（P3，可回測）**：NVDA／費半／TSM ADR 的隔夜報酬本身就是「市場對新聞的總結」。用每檔股票對這些標的的滾動 beta，算出「預期開盤跳空」與盤前更新的漲跌機率。黃仁勳事件不需要讀懂新聞，NVDA −12% 就是訊號。
- **(b) 事件存檔 + 風險旗標（P4a/b，立即有用）**：存標題，UI 顯示「重大事件日：模型未納入此資訊」，並列出相關標題連結。不預測方向。
- **(c) LLM 情緒（P4c，延後）**：存檔滿 120 個交易日後，才用事件研究法檢驗是否有用；在那之前**不得餵進預測**。

---

## 5. 預先登記的特徵清單（定義固定，不得更動）

記號：`vol20[t]` = 過去 20 日平均成交股數（Yahoo 的 volume，單位：股）。融資融券餘額單位是「張」，要 ×1000。所有特徵在每個交易日對宇宙做**橫斷面排名轉成 [-0.5, 0.5]**（缺值給 0）後才使用。

**籌碼（P1，資料：FinMind）**

| 代號 | 定義 | 預期方向 |
|---|---|---|
| C1 `fi_net_5` | 近 5 日外資（`Foreign_Investor`）買賣超股數合計 ÷ 近 5 日成交股數合計 | + |
| C2 `fi_net_20` | 同上，20 日 | + |
| C3 `it_net_5` | 近 5 日投信（`Investment_Trust`）買賣超 ÷ 近 5 日成交股數 | + |
| C4 `it_net_20` | 同上，20 日 | + |
| C5 `it_streak` | 投信連續買超天數（正）或連續賣超天數（負），截斷於 ±10 | + |
| C6 `dealer_net_5` | 自營商自行買賣（`Dealer_self`，不含避險）5 日買賣超 ÷ 成交股數 | ? |
| C7 `margin_chg_20` | （融資餘額[t] − 融資餘額[t−20]）×1000 ÷ (20 × vol20) | − |
| C8 `short_chg_20` | （融券餘額[t] − 融券餘額[t−20]）×1000 ÷ (20 × vol20) | ? |
| C9 `fi_hold_chg_20` | 外資持股比率[t−1] − [t−21]（百分點；注意多落後 1 日） | + |

**月營收（P2，資料：FinMind `TaiwanStockMonthRevenue`）**——用「當下可得的最新月份」

| 代號 | 定義 | 預期方向 |
|---|---|---|
| R1 `rev_yoy` | 最新月營收 ÷ 去年同月 − 1，截斷於 [−0.9, 3] | + |
| R2 `rev_yoy_accel` | R1 − 前 3 個月 YoY 的平均 | + |
| R3 `rev_3m_yoy` | 近 3 個月營收合計 ÷ 去年同期 3 個月合計 − 1 | + |
| R4 `rev_high12` | 最新月營收是否為近 12 個月最高（1／0） | + |

**評價（P1，資料：FinMind `TaiwanStockPER`）**：V1 `ey` = 1/PER（PER ≤ 0 或缺值 → NaN）；V2 `bp` = 1/PBR；V3 `dy` = 殖利率。

**價格控制組（P0，直接取自既有 `features.build_features` 的 `raw`／`sigma_ref`）**：P1 `rev_5`、P2 `mom_12_1`、P3 `idio_mom`、P4 `neg_sigma`（= −`sigma_ref`，低波動）。

共 20 個候選。**DEV 篩選門檻**（多重檢定校正）：平均 rank IC 的 HAC |t| ≥ 3.0，且在 DEV 前半（2013–2017）與後半（2018–2022）符號相同，且符號與「預期方向」一致（標 ? 者不限）。

---

## 6. 評估協定與閘門

**宇宙**：候選池 = 目前成交值前 300 名的普通股（排除 ETF；以 `data.build_universe(items, size=300)` 取得）。每個交易日 t 的宇宙 = 候選池中「截至 t 日已有 ≥ 500 根 K 棒」者，依過去 60 日成交值（close×volume）中位數取前 150 名。
已知限制（要寫進結果）：候選池用了今天的資訊、已下市股票不在內 → 絕對報酬偏高；相對指標（IC、多頭減全體）受影響較小。

**指標**（主要 h=5，次要 h=20）：

- `IC_t` = 當日宇宙內 Spearman(特徵, `y_h`)；報告平均值與 HAC t（重用 `backend.evaluation.universe_bt._t_stats`）。
- 多頭籃子：每日取前 20%（30 檔）等權，持有 h 日的重疊批次（做法同 `universe_bt._Panel.tranche_weights`），以 `strategy.simulate_weights(..., exec_mode="next_open")` 與 `strategy.side_costs` 計成本。基準：等權全體。報告年化超額報酬、超額報酬 HAC t、最大回撤、年換手、籃子 σ 與 beta 對全體的比值。
- 殘差 IC：每日先把特徵對 [`neg_sigma`, `mom_12_1`, `rev_5`] 做橫斷面 OLS 取殘差，再算 IC（確認不是低波動或動能的替身）。

**組合訊號**：通過 DEV 篩選的特徵，各自橫斷面排名後乘上預期方向符號、**等權平均**（不擬合權重）。規格寫進 `research/composite_spec.json` 並 commit，之後不得更改。

**閘門（全部在 OOS、只看一次、h=5）**：

| 代號 | 條件 |
|---|---|
| G1 | 組合訊號平均 rank IC > 0 且 HAC t ≥ 2.5 |
| G2 | 多頭前 20% 扣成本後年化超額報酬 > 0 且 HAC t ≥ 1.65 |
| G3 | 殘差 IC 的 HAC t ≥ 2.0 |
| G4 | 2023、2024、2025、2026 四個年度中至少 3 個年度的平均 IC > 0 |
| G5 | 安慰劑與植入訊號測試通過（P0 的單元測試） |

- **全部通過** → 寫 `RESULTS.md`，停下來問使用者是否進入 P6 整合。
- **任一未過** → 寫 `RESULTS.md`（如實寫出每個條件的數字），P6 不做。不得改規格重跑 OOS；之後只能等新資料累積（至少 250 個新交易日）再評估。

---

## 7. 階段

每階段格式：**要做的事 → 檔案 → 驗收**。建議模型：P0、P3、P5 用 Sonnet（時間對齊與統計容易出錯）；P1a、P2a、P4a 的抓取與解析可用 Haiku。

### P0 — 評估框架與價格基準（不需要任何新資料）

1. 建立 `research/__init__.py`、`research/panel.py`：
   - `build_panel(codes, start="2012-01-01") -> Panel`：用 `backend.data.get_price_data` 取資料，對齊到聯集交易日曆。欄位（皆為 `(T, S)`）：`adj_open`、`adj_close`、`volume`、`value`（close×volume）、`has_bar`；另有 `cal`（日期）、`codes`。
   - `universe_mask(panel, n=150, lookback=60, min_bars=500) -> (T, S) bool`（定義見 §6，只用 ≤ t 的資料）。
   - `forward_open_return(panel, h) -> (T, S)`：`ln(adjO[t+1+h]/adjO[t+1])`，任一端缺 K 棒為 NaN。
   - `untradable_open(panel) -> (T, S) bool`：t+1 開盤價相對 t 收盤 ≥ +9.5%（漲停鎖死買不到），多頭籃子要排除。
   - 以 `np.savez_compressed` 存到 `data_cache/research/panel_v1.npz`（`data_cache/` 已在 `.gitignore`）。
2. `research/harness.py`：
   - `xs_rank(F, mask)`：橫斷面排名到 [-0.5, 0.5]。
   - `ic_report(F, y, mask, h, period)`、`quintile_report(F, panel, mask, h, period)`、`residual_ic(F, controls, y, mask, h, period)`。
   - `period` 只接受 `"dev"`、`"dev_a"`、`"dev_b"`、`"oos"`；`"oos"` 必須同時傳入 `spec_hash`，並自動在 `docs/research/LOOKS.md` 附加一行，否則 raise。
3. `research/price_baseline.py`：算 P1–P4 四個價格特徵在 DEV 的 IC／殘差 IC／多頭報告，輸出 `research/out/price_baseline_dev.json` 與 `docs/research/RESULTS.md` 的第一節。
4. `tests/test_research_harness.py`（合成資料、不連網）：
   - 安慰劑：每日把特徵在股票間隨機打亂 → `|mean IC| < 3 × SE`。
   - 植入訊號：`F = y + 雜訊`（目標 IC≈0.05）→ HAC t > 4。
   - 無偷看：把 t0 之後的價格換成亂數，`universe_mask`、特徵在 ≤ t0 的列逐位元相同；`forward_open_return` 在 ≤ t0−h−2 的列相同。
   - `"oos"` 沒帶 `spec_hash` 會 raise。
5. **驗收**：新測試全綠；既有後端 856 測試不受影響（只需跑 `test_eval_*` 確認）；`RESULTS.md` 有價格基準表。

### P1 — 籌碼與評價

- **P1a 抓取**：`backend/altdata/__init__.py`、`backend/altdata/finmind.py`
  - `fetch_dataset(dataset, stock_id, start_date) -> list[dict]`；URL：`https://api.finmindtrade.com/api/v4/data?dataset=..&data_id=..&start_date=..`（有 `FINMIND_TOKEN` 就加 `&token=`）。`msg != "success"` 或 HTTP 402／429 → raise `RateLimited`。
  - `backfill(datasets, codes)`：逐檔存 `data_cache/alt/finmind/{dataset}/{code}.json`（含 `fetched_at`），已存在就跳過，每次請求間隔 ≥ 12 秒（無 token）；可中斷續跑；CLI：`python -m backend.altdata.finmind --datasets inst,margin,hold,per --universe 300`。
  - 資料集：`TaiwanStockInstitutionalInvestorsBuySell`（長格式，`name` ∈ `Foreign_Investor`、`Foreign_Dealer_Self`、`Investment_Trust`、`Dealer_self`、`Dealer_Hedging`，欄位 `buy`／`sell`）、`TaiwanStockMarginPurchaseShortSale`（`MarginPurchaseTodayBalance`、`ShortSaleTodayBalance`）、`TaiwanStockShareholding`（`ForeignInvestmentSharesRatio`）、`TaiwanStockPER`（`PER`、`PBR`、`dividend_yield`）。
  - 300 檔 × 4 資料集 = 1200 次請求，無 token 約 4 小時。**先跑 `inst` 一個資料集確認流程，再跑其他。**
- **P1b 特徵**：`research/feat_chips.py`：把 JSON 對齊到 `panel.cal`（日期對不到的列為 NaN，**不得往未來補值**；往過去 ffill 最多 5 日），依 §5 定義算 C1–C9、V1–V3。
- **P1c 評估**：`research/run_chips.py` → `research/out/chips_dev.json`、`RESULTS.md` 第二節（每個特徵：DEV IC、t、前後半符號、殘差 IC、是否通過篩選）。
- **測試** `tests/test_alt_finmind.py`、`tests/test_research_chips.py`：用 `tests/fixtures/alt/` 的小樣本驗證解析、單位（張→股）、C9 的額外落後 1 日、毒化未來後特徵不變。
- **驗收**：測試全綠；`RESULTS.md` 有 12 個特徵的 DEV 表；**沒有碰 OOS**。

### P2 — 月營收

- 抓取：沿用 `finmind.backfill`，資料集 `TaiwanStockMonthRevenue`（`revenue_year`、`revenue_month`、`revenue`；`date` 是營收月份的下個月 1 日）。
- `research/feat_revenue.py`：實作 §3 的可得日規則（函式 `revenue_available_day(year, month, cal)`，要有單元測試：10 日是週六／連假的情況）；算 R1–R4；在可得日之前沿用上一個月的值。
- `research/run_revenue.py` → `RESULTS.md` 第三節。額外輸出「可得日後第 1–20 日」的事件時間累積超額報酬表（依 R1 分五組），只用 DEV。
- **驗收**：同 P1。

### P3 — 美股隔夜與盤前更新

- **P3a 抓取**：`backend/altdata/usmarket.py`：用 `backend.data._fetch_bars(ysym, 0)`（已回傳交易所當地日期）抓 `^SOX`、`^IXIC`、`^GSPC`、`NVDA`、`TSM`、`^VIX`、`TWD=X`，存 `data_cache/alt/us/{name}.json`。實作 §3 的「未完成 K 棒丟棄」與 `D < S` 對齊函式 `align_us_to_tw(us_dates, us_ret, tw_cal) -> (T,)`：回傳「可用於 S 日開盤」的最新一根美股報酬；若兩個台股交易日之間有多根美股 K 棒，取累積報酬。
- **P3b 研究**（`research/run_overnight.py`，DEV 篩選、OOS 只看一次並記錄）：
  1. 跳空模型：`gap[S,i] = ln(O[S]/C[S−1])` 對 `r_SOX`、`r_TSM`、`r_NVDA` 的滾動迴歸（EWMA 半衰期 120 日，係數只用 < S 的資料）。報告樣本外 R² 與跳空方向命中率。
  2. 盤前更新的收盤漲跌機率：`mu' = mu + 預測跳空`、`sigma'² = sigma² × (1 − R²_滾動)`、`p_up' = Φ(mu'/sigma')`，其中 `mu`、`sigma`、`p_up` 取既有引擎 h=1 的 walk-forward 輸出。比較 `p_up'` 與 `p_up` 的 Brier 分數與命中率（配對、HAC）。
  3. 可交易性檢驗：`ln(C[S]/O[S])`（盤中報酬）與預測跳空的 IC；|美股 z| ≥ 2 的衝擊日之後 5 日的開盤到開盤超額報酬。
- **預期**：1、2 會明顯成立（這是「預測準確度」的真實提升）；3 多半不成立（跳空在開盤就反映完）。**兩者都要如實寫**，UI 文案必須區分「盤前預測更準」與「可交易的優勢」。
- **P3c 產品化**（P3b 的第 2 項 OOS 成立才做；動到 API 與前端，需先完成 `HANDOFF.md §4A`）：新增 `GET /api/premarket?symbol=`，個股頁加「盤前更新」卡片，標示資料時間與「跳空無法以昨日收盤價買到」。先把契約寫進 `DESIGN.md` 新的 §10。

### P4 — 新聞／事件

- **P4a 存檔器**：`backend/altdata/news.py`，CLI `python -m backend.altdata.news --once`
  - 來源（§9 皆實測）：TWSE 重大訊息 `t187ap04_L`、TPEx 重大訊息 `mopsfin_t187ap04_O`、Yahoo 股市個股 RSS（僅宇宙前 60 檔）、鉅亨網台股新聞、中央社財經 RSS、Google News RSS（關鍵字清單放 `backend/altdata/news_queries.json`：輝達、台積電、聯準會、關稅、費城半導體…）。
  - 每則存一行 JSON 到 `data_cache/alt/news/YYYY-MM-DD.jsonl`：`{id(sha1(來源+連結)), source, title, link, published_at, fetched_at, codes[]}`；以 `id` 去重。
  - `codes` 對應：重大訊息直接有公司代號；Yahoo 個股 RSS 用查詢的代號；其餘用公司名稱表（`data._names_table`）做字串比對，名稱 ≤ 2 個字的不比對（避免誤判）。
  - 排程由使用者決定（需使用者同意後才建立）：`schtasks /Create /SC MINUTE /MO 30 /TN twq-news /TR "<python 路徑> -m backend.altdata.news --once" /ST 07:00`。
- **P4b 風險旗標**：`backend/altdata/events.py` → `event_flags(code, asof) -> {market: [...], stock: [...]}`
  - 市場層：隔夜 `^SOX`／`NVDA`／`TSM` 報酬的 |z| ≥ 2.5（z 用 EWMA λ=0.94 的波動），或 VIX 單日 +20% 以上。
  - 個股層：最近一個交易日內有重大訊息；或（存檔滿 20 日後）當日新聞則數 ≥ 20 日中位數的 3 倍。
  - 產品化（同樣需先完成 `HANDOFF.md §4A`）：`/api/forecast` 回應加 `events` 區塊；個股頁與推薦頁顯示琥珀色橫幅「重大事件日：模型未納入此資訊，區間可能低估」＋相關標題連結。**旗標不改變任何預測數字。**
- **P4c LLM 判讀（延後，條件：存檔 ≥ 120 個交易日）**：用 `claude-haiku-5-5` 把標題轉成 `{codes, event_type, direction(-2..2), is_new_info}`；以標題雜湊快取；只送通過關鍵字預篩的標題並設每日上限。**實作前先載入 `claude-api` skill，不要憑記憶寫 API 參數。** 先做事件研究（事件後 1／5／20 日開盤到開盤超額報酬），有證據才談納入模型。
- **測試** `tests/test_alt_news.py`：RSS／JSON 解析、去重、代號對應、民國日期轉換（`1151112` → 2026-11-12）、`event_flags` 的門檻，全部用 fixtures。
- **驗收**：`--once` 連跑兩次第二次新增 0 則；測試全綠。

### P5 — 組合訊號與 OOS 一次性驗證

1. 依 §5 門檻從 P0–P2 的 DEV 結果選特徵 → 寫 `research/composite_spec.json`（特徵、符號、h、宇宙參數）→ commit → 記下 commit hash。
2. 在 DEV 上跑組合訊號確認程式正確（不是為了調整）。
3. **跑一次 OOS**（`period="oos"`、帶 `spec_hash`）→ 依 §6 逐條判定 G1–G5 → 寫 `RESULTS.md` 最終節與 `LOOKS.md`。
4. 停下來向使用者回報。若沒有任何特徵通過 DEV 篩選，直接寫「無可組合的訊號」，**不要跑 OOS**。

### P6 — 整合進產品（僅在 G1–G5 全過且使用者同意後）

此時才另寫細部設計：正式的每日抓取改用官方端點（TWSE `T86`、`MI_MARGN`、`t187ap05_L`；TPEx openapi，§9）、推薦頁新增獨立的 `xs_signal` 排序（與既有 `rank_signal` 並列、不取代）、`DESIGN.md` 契約、evidence 等級沿用既有規則（未達 `moderate_edge` 不得出現「建議買進」）。

---

## 8. 狀態表（每階段結束時更新）

| 階段 | 狀態 | 完成日 | 備註 |
|---|---|---|---|
| P0 評估框架 | 完成 | 2026-11-13 | 30 個測試全綠；價格基準 DEV 已寫入 RESULTS.md 第一節（`rev_5` 為 h=5 反轉，IC −0.038、t −6.8）。`ic_report`/`residual_ic` 多一個 `cal` 參數（日期切分需要） |
| P1 籌碼與評價 | 未開始 | | |
| P2 月營收 | 未開始 | | |
| P3 美股隔夜 | P3a 完成；P3b DEV 完成、OOS 待跑 | 2026-11-13 | 抓取器＋對齊函式測試全綠；DEV：跳空 OOS R² 23%、盤前更新 Brier skill +5%（30/30 檔），但盤中報酬 IC≈0、衝擊日後無延續 → 「預測更準」成立、「可交易優勢」不成立。`research/overnight_spec.json` 已凍結，OOS 只能跑一次 |
| P4a/b 新聞存檔與旗標 | P4a 完成；P4b 不在本輪範圍 | 2026-11-13 | `python -m backend.altdata.news --once` 已跑，2927 則；**未建立排程**（待使用者同意） |
| P4c LLM 判讀 | 延後 | | 條件：存檔 ≥ 120 交易日 |
| P5 OOS 驗證 | 未開始 | | 只能跑一次 |
| P6 整合 | 待閘門 | | 需使用者同意 |

### 換機器／續跑指引（2026-11-13 寫入，P2/P1 資料下載進行到一半時中斷）

`data_cache/` 不在 git 內（可整個資料夾複製到新機器，或照下列步驟重建）。已完成並提交：P0、P3a、P3b(DEV)、P4a 程式與測試，以及 P2/P1 的特徵＋評估程式（`research/feat_*.py`、`run_*.py`，測試全綠）。**尚未完成**：FinMind 資料下載（下載時月營收約 174/300 檔），因此 `run_revenue`／`run_chips` 還沒跑出完整結果、P5 還沒做。續跑順序：

1. 價格快取（新機器才需要）：以 `backend.data.build_universe(backend.data._names_table(network=True,block=True)["items"], size=300)` 寫出 `data_cache/research/pool300.json`，再對每檔呼叫 `backend.data.get_price_data(code)`（約 10 分鐘）；然後 `python -m research.price_baseline` 會重建 `panel_v1.npz`。
2. 美股：`python -m backend.altdata.usmarket`。
3. FinMind（可中斷續跑，已存在的檔案會跳過；遇 402/429 等一小時再跑同一指令）：依序 `python -m backend.altdata.finmind --datasets rev --universe 300`、`inst`、`per`、`margin`、`hold`（無 token 每個資料集約 1 小時）。
4. `python -m research.run_revenue`、`python -m research.run_chips`（寫 RESULTS.md 第二、三節）。
5. P5：依 §5 門檻選特徵 → 寫 `research/composite_spec.json` 並 commit → 跑一次 OOS（`spec_hash`）→ 判定 G1–G5。P3b 的 OOS 也只能跑一次：`python -m research.run_overnight --oos --spec-hash <sha256 of research/overnight_spec.json>`。
6. 完成後更新上面狀態表並 commit；**不要 push、不要建立排程**，先問使用者。

建議順序：**P4a（先讓存檔開始累積）→ P0 → P2 → P1 → P5 → P3 → P4b**。

---

## 9. 附錄：端點實測（2026-11-12，User-Agent `Mozilla/5.0`，皆 HTTP 200）

**FinMind**（`https://api.finmindtrade.com/api/v4/data?dataset=…&data_id=…&start_date=…`，回應 `{msg, status, data:[…]}`；單次請求可取整段歷史）

| dataset | 起始 | 列範例 |
|---|---|---|
| `TaiwanStockInstitutionalInvestorsBuySell` | 2012-05-02 | `{"date":"2012-05-02","stock_id":"2330","buy":38841300,"name":"Foreign_Investor","sell":32897643}`（上櫃 6488 也有） |
| `TaiwanStockMarginPurchaseShortSale` | 2012-01-02 | `MarginPurchaseTodayBalance`、`MarginPurchaseYesterdayBalance`、`ShortSale…`（單位：張） |
| `TaiwanStockShareholding` | 2012-01-02 | `ForeignInvestmentSharesRatio`（%） |
| `TaiwanStockPER` | 2012-01-02 | `{"dividend_yield":3.99,"PER":13.56,"PBR":3.25}` |
| `TaiwanStockMonthRevenue` | 2010-01 | `{"date":"2026-11-01","revenue":586890319000,"revenue_month":10,"revenue_year":2026,"create_time":"2026-11-10"}`；舊資料 `create_time` 為 `""` |
| `TaiwanStockNews` | 2021 可查 | `{"date":"2026-11-10 01:02:00","stock_id":"2330","link":…,"source":…,"title":…}`；一次一檔一天 |
| `TaiwanStockDividendResult`、`TaiwanStockSecuritiesLending` | — | 可用（本計畫未使用） |

**TWSE／TPEx 官方**（P6 每日更新用；一次回傳全市場）

| 用途 | URL | 注意 |
|---|---|---|
| 三大法人 | `https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALLBUT0999&response=json` | 2013 年 12 欄、2026 年 19 欄 → **用欄位名稱解析，不可用位置**；數字含千分位逗號 |
| 融資融券 | `https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=YYYYMMDD&selectType=ALL&response=json` | `tables[1]` 為個股；「買進／賣出／今日餘額」欄名在融資與融券各出現一次 → 這張表只能用位置 |
| 本益比等 | `https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date=YYYYMMDD&selectType=ALL&response=json` | |
| 外資持股 | `https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date=YYYYMMDD&selectType=ALLBUT0999&response=json` | |
| 重大訊息（上市／上櫃） | `https://openapi.twse.com.tw/v1/opendata/t187ap04_L`、`https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O` | 欄位 `發言日期`（民國 `1151111`）、`發言時間`（`10405` = 01:04:05，要左補零）、`公司代號`／`SecuritiesCompanyCode`、`主旨`（上市的鍵是 `"主旨 "`，**尾端有一個空白**） |
| 月營收最新月（上市／上櫃） | `https://openapi.twse.com.tw/v1/opendata/t187ap05_L`、`https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O` | 只有最新一個月 |
| 上櫃三大法人最新日 | `https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading` | 鍵名是英文且空白不一致 |
| 處置股、除權息預告 | `https://openapi.twse.com.tw/v1/announcement/punish`、`https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL` | 本計畫未使用 |

**美股／匯率**：既有的 Yahoo chart 端點，代號 `^SOX`（1994 起，8188 根）、`^IXIC`、`^GSPC`、`NVDA`、`TSM`、`^VIX`、`TWD=X`；URL 中 `^` 編碼為 `%5E`、`=` 為 `%3D`。

**新聞**

| 來源 | URL | 格式 |
|---|---|---|
| Google News 搜尋 | `https://news.google.com/rss/search?q=<關鍵字>+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant` | RSS，最多 100 則 |
| Yahoo 股市個股 | `https://tw.stock.yahoo.com/rss?s=2330` | RSS，20 則，有 `pubDate` |
| 鉅亨網台股 | `https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30` | JSON：`items.data[]`，欄位 `newsId`、`title`、`publishAt`（epoch 秒）、`stock`、`keyword` |
| 中央社財經 | `https://feeds.feedburner.com/rsscna/finance` | RSS，20 則 |
| GDELT | — | 實測回空結果，**不使用** |
