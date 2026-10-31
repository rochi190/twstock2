# 台股量化實驗室 (TW Quant Lab)

> **開發交接（2026-10-31）**：目前進度、待辦與需要決定的事項見 [HANDOFF.md](HANDOFF.md)；規格與契約見 [DESIGN.md](DESIGN.md)（§9 最新）；審查與修正紀錄見 [docs/audit/](docs/audit/)。前端收尾尚未完成，推薦頁在收尾前請勿當真。

一個在本機執行的台股（上市＋上櫃，含 ETF）量化預測網頁應用程式：後端是 Python 標準函式庫 HTTP 伺服器加上 numpy / scipy，
前端是沒有建置步驟、沒有外部資源（無 CDN、無網路字型）的原生 JavaScript。

它做四件事：

1. **個股預測**：輸入任一上市／上櫃股票或 ETF，得到「明日、未來 5 日、未來 20 日」上漲／下跌的**機率**、預期報酬、
   50%／90% 區間、扇形圖、波動與市場狀態（regime）、風險（VaR／CVaR）。
2. **今日推薦**：每個交易日掃描 60 檔流動性佳的股票，依「相對強度」（預期對數報酬 ÷ 預期對數波動度，不含交易成本）排序，
   同時列出相對最強／最弱的名單、偏多候選與偏空／避開名單（偏多候選只代表模型的相對看法，不是買進建議）；
   建議權重（經驗貝氏收縮、分數凱利、單檔上限）只在各檔差異大於雜訊且有正的扣成本期望報酬時才給，否則答案就是「空手」。
3. **歷史驗證**：同一個模型在過去資料上做嚴格樣本外（walk-forward）回測：方向準確率對基準、校準、報酬預測能力、區間涵蓋率、
   含台灣交易成本的策略模擬，以及整個推薦名單的橫斷面驗證。結論允許（也經常）是「沒有統計上顯著的預測優勢」。
4. **方法說明**：每個技術與其限制的白話＋數學說明（網頁內「方法說明」頁）。

> **重要**：本工具為統計模型的研究／教學用途，**不是投資建議**。股價報酬近乎無法預測；所有輸出都是機率，不保證獲利。
> 詳見文末〈限制〉與〈免責聲明〉。

## 快速開始

需求：Python 3.10 以上（建議 3.11）、`numpy`、`scipy`（執行期只需要這兩個），以及可以連到 Yahoo Finance 與 TWSE／TPEx
開放資料的網路。

```bash
pip install -r requirements.txt     # numpy, scipy
./run.sh                            # 然後開啟 http://localhost:8000/
```

`run.sh` 只會在專案資料夾內建立檔案（快取在 `data_cache/`），並執行 `python3 -m backend.server`。常用方式：

```bash
PORT=8080 ./run.sh                  # 換連接埠（--port 0 = 自動挑選一個空的）
./run.sh --port 8080 --quiet        # 其餘參數直接傳給 backend.server
HOST=0.0.0.0 ./run.sh               # 對區域網路開放（預設只綁 127.0.0.1；沒有任何登入機制，請勿暴露在公網）
./run.sh --precompute --universe-bt # 先預先計算快取（見下），再啟動伺服器
```

`Ctrl-C`（或 `SIGTERM`）會乾淨地關閉伺服器與運算行程。

### 預先暖機（可選）：`python -m backend.precompute`

第一次使用時，「今日推薦」要下載 60 檔歷史行情並各算一次預測（約 70 秒），「推薦整體歷史驗證」要為 60 檔各做一次
5 年的前推式回測（約 2 分鐘，4 核心）。這些結果都會存進 `data_cache/`，所以可以先用 CLI 暖機，之後網頁幾乎是即時的：

```bash
python -m backend.precompute                     # 下載宇宙行情＋計算今日預測與推薦（預設 --prices --forecasts）
python -m backend.precompute --universe-bt       # 另外做 60 檔前推式回測與推薦整體歷史驗證（--validate 為舊別名）
python -m backend.precompute --h 1,5,20          # 針對多個預測天數
python -m backend.precompute --symbols 2330,0050,6488 --forecasts --backtest   # 指定股票的預測與預設回測
python -m backend.precompute --refresh           # 強制重新下載行情（忽略快取新鮮度）
python -m backend.precompute --n 20 --workers 2  # 縮小宇宙／限制行程數
```

進度會印在 stderr；`Ctrl-C` 可中斷，已完成的部分仍留在快取中。

### 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 綁定位址與連接埠。`HOST` 只接受 IP 位址或 `localhost`（conda 會把 `HOST` 設成 `x86_64-conda-linux-gnu` 之類的名稱，這種值會被忽略並提示）；也可用 `TWQ_HOST` |
| `TWQ_CACHE_DIR` | `<專案>/data_cache` | 快取資料夾 |
| `TWQ_UNIVERSE_SIZE` | `60` | 推薦／整體驗證的股票檔數（4–500）；開發時設成 `8` 可快速驗證流程 |
| `TWQ_OFFLINE` | 未設定 | 設為 `1`：所有網路請求立刻失敗，用來模擬斷網（快取仍可用，資料會標示 `stale`） |
| `TWQ_YAHOO_CHART` | Yahoo v8 chart 網址 | 取代 Yahoo 端點的網址樣板（含 `{sym}`），例如指向一個連不上的位址來模擬 Yahoo 故障 |
| `TWQ_HTTP_TIMEOUT` | `45` | 單次網路請求逾時秒數 |
| `TWQ_SKIP_SLOW` | 未設定 | 設為 `1` 時測試略過需要啟動真實運算行程的整合測試 |

## 架構

```
backend/config.py, types.py   共用常數與資料容器（唯讀契約）
backend/data.py               行情與名稱：Yahoo 日K、TWSE／TPEx 開放資料、交易日曆、宇宙、磁碟快取
backend/quant/                量化核心：GJR-GARCH-t、HMM、貝氏 DLM、動態模型平均、校準、蒙地卡羅、engine
backend/evaluation/           評估：指標、策略模擬、回測報告、推薦器、宇宙驗證
backend/jobs.py               運算編排：預測／前推式快取、行程池、背景工作與進度、JSON 轉換
backend/server.py             HTTP API 與靜態檔案（python -m backend.server）
backend/precompute.py         暖機 CLI
frontend/                     index.html、style*.css、app.js、view_*.js、charts.js …（無建置步驟）
tests/                        unittest（python -m unittest discover -s tests）
data_cache/                   執行時產生：prices/ forecast/ wf/ results/ names.json universe.json holidays_YYYY.json
```

資料流：`data`（下載、清理、快取）→ `quant.engine`（`forecast_latest`／`walk_forward`）→ `evaluation.*`
（`annotate_forecast`、`build_recommendations`、`build_backtest_report`、`evaluate_universe`）→ `server`（JSON）。

要點：

* **資料**：Yahoo chart API（`period1=0&period2=now&interval=1d`，**不用** `range=max`，因為它會被悄悄降級成月線）。
  先試 `.TW` 再試 `.TWO` 並記住結果；時間戳 +8 小時轉成台北日期；建模用還原權息價（`adjclose`），顯示用未還原的原始 OHLC。
  快取為 `data_cache/prices/<代號>.json`，原子寫入（暫存檔＋rename）、每檔一把鎖、同時最多 6 個 Yahoo 請求、429／5xx 指數退避重試；
  只補抓最近約 25 天並先驗證重疊區間沒有被除權息或分割改寫，否則整段重抓；網路失敗就回傳舊快取並標示 `stale: true`；
  連續失敗 3 次後 30 秒內不再嘗試（避免斷網時每檔都等數秒）。
* **未收盤的 K 棒**：台北時間 14:00 前今天的 K 棒視為未完成（`last_bar_complete=false`），模型一律丟棄它（`data.completed`）。
* **背景工作**：回測、推薦、整體驗證以「工作」執行：回傳 `{"status":"computing","progress":{done,total,message}}`，前端每 ~1.5 秒輪詢，
  完成後回傳 `{"status":"ready",…}`。相同參數的同時請求共用同一個工作；已完成的結果在記憶體保留 30 分鐘（最多 128 筆，最舊的先淘汰；執行中的工作永不淘汰，排隊過多時會拒絕新工作），
  並以（參數、最後一根 K 棒日期、資料雜湊、程式碼指紋）為鍵存在磁碟上。
* **行程池**：CPU 密集的前推式回測與整批預測跑在 `spawn` 行程池（最多 CPU 數 − 1 個，BLAS 單執行緒），單檔即時預測則在請求執行緒內計算
  （工作行程以較低優先權執行，所以計算期間網頁仍然順暢）。每檔股票完成就更新一次進度。
* **快取**：預測以（代號、最後一根 K 棒、價格雜湊、大盤雜湊、量化程式碼指紋）為鍵；前推式結果存成 pickle，**單檔回測與整體驗證共用同一份**
  （預設 5 年 = 1250 個樣本外交易日），所以先看過 2330 的回測，之後做整體驗證就不必再算 2330。

## API

所有回應皆為 UTF-8 JSON；數字中的 NaN／±Infinity 一律輸出為 `null`；日期為 `YYYY-MM-DD`；報酬為小數（0.012 = +1.2%）。
詳細欄位定義見 `DESIGN.md` 第 6 節。

| 路徑 | 參數（預設／範圍） | 說明 |
|---|---|---|
| `GET /api/status` | — | 伺服器時間、台北時間、`market_open`、`last_trading_date`、版本、工作統計 |
| `GET /api/search` | `q`（空 = 熱門清單）、`limit`（20／1–50） | 代號前綴或中文名稱子字串（臺／台視為相同、不分大小寫與全形） |
| `GET /api/history` | `symbol`、`limit`（750／1–20000） | 未還原 OHLCV（升冪）、股利、`asof`、`last_bar_complete`、`stale` |
| `GET /api/forecast` | `symbol` | 同步計算（冷快取約 3–4 秒、熱快取數毫秒）：機率、區間、扇形圖、regime、波動、風險、模型權重、驅動因子 |
| `GET /api/recommendations` | `h`（5；1／5／20）、`top`（10／1–50） | 工作：買進候選、建議避開、全部排名、建議投資組合 |
| `GET /api/backtest` | `symbol`、`h`（1）、`thr`（0.55／0.5–0.95）、`exec`（`next_open`／`close`）、`disc`（0.6／0–1）、`slip`（5／0–100 bp）、`years`（5／1–25）、`sizing`（`threshold`／`kelly`） | 工作：單檔前推式回測報告（`report`） |
| `GET /api/backtest/universe` | `h`（5）、`top`（5／1–20）、`exec`、`disc`、`slip` | 工作：每天買進推薦名單前 k 檔的整體驗證（`report`） |

錯誤一律是 `{"status":"error","error":{"code":…,"message":"繁中訊息"}}`：

| `code` | HTTP | 情況 |
|---|---|---|
| `bad_request` | 400（找不到 API 路徑 404、非 GET 405、不允許的 Host 403） | 參數缺漏或超出範圍 |
| `unknown_symbol` | 404 | 代號格式正確但查無此股 |
| `insufficient_history` | 422 | 歷史資料少於預測（250 日）或回測（620 日）所需 |
| `data_unavailable` | 503（含 `Retry-After`） | 網路／資料來源故障且沒有可用快取 |
| `internal` | 500（佇列過滿時 503） | 其他錯誤（細節只寫在伺服器日誌） |

靜態檔案來自 `frontend/`（`/` → `index.html`）：正確的 MIME（JS 為 `application/javascript; charset=utf-8`）、HTML／JS／CSS 為
`Cache-Control: no-cache` 並附 `ETag`／`Last-Modified`（條件式請求回 304）；拒絕路徑穿越、以 `.` 開頭的檔案與指向資料夾外的符號連結；
回應附 CSP 與 `nosniff`；綁定本機位址時還會拒絕非本機的 `Host` 標頭（防 DNS rebinding）。

## 數學原理（兩段摘要）

**預測。** 對每個天數 h，模型輸出累積對數報酬 R = ln(adjC[i+h] / adjC[i]) 的**完整預測分布**，由四個數學上不同的元件組成：
GJR-GARCH(1,1) 加標準化 Student-t 新息（最大概似估計，h 步變異數與峰度精確推算）、3 狀態高斯隱馬可夫模型（只使用「濾波」而非「平滑」的
狀態機率，再以轉移矩陣的冪外推）、對波動標準化報酬做貝氏迴歸的折扣因子動態線性模型（Kalman 濾波，Normal-Inverse-Gamma 變異數學習，
只用已經「成熟」的結果更新），以及錨定「沒有資訊」的隨機漫步基準。四者以動態模型平均（對成熟結果的對數預測概似做指數折扣）做線性意見池，
再用線上貝氏羅吉斯（Platt）迴歸校準 P(R>0)，先驗收斂到恆等映射，所以資料少時很穩。一日預測依 ±10% 漲跌停截斷；扇形圖與 VaR／CVaR
用濾波歷史模擬（GARCH 標準化殘差＋regime 混合）。任何一列預測都只用到該列（含）以前的資料——包含參數估計、校準與特徵標準化——
並由「截斷不變性」單元測試保證。

**驗證與推薦。** 回測是擴張視窗的前推式（walk-forward）：每 21 根 K 棒重估參數，永遠只用過去預測未來；報告列出方向準確率與 Wilson 信賴區間
對「一直猜漲／前期方向／訓練期多數」基準的精確二項檢定、Brier 技能分數、ECE、Spearman IC、Campbell-Thompson 樣本外 R²、
Newey-West 修正的 Diebold-Mariano 檢定、50／80／90% 區間涵蓋率（Kupiec 檢定）、PIT 均勻性（KS 檢定），並模擬含手續費、證交稅（股票 0.3%／ETF 0.1%）、
滑價、隔日開盤成交的策略，報告機率夏普比（PSR／DSR）與 bootstrap 信賴區間；結論用保守的規則產生，只有證據夠強才會說「有優勢」。
推薦器的分數是 (預期報酬 − 來回成本) ÷ 波動度，跨股票做 James-Stein 經驗貝氏收縮（抑制贏家詛咒），建議權重是收縮後優勢的 0.25 倍凱利，
單檔上限 15%、其餘為現金。

## 效能與資源（4 核心、Python 3.11、台灣網路，實測）

| 動作 | 冷快取 | 熱快取 |
|---|---|---|
| 伺服器啟動後第一個 `/api/status`／`/api/search` | ~1 秒（行事曆）／~3 秒（名稱表，啟動時會在背景預先載入） | 幾毫秒 |
| `/api/history`（2330，6600 根） | ~1.9 秒（下載） | ~6 毫秒 |
| `/api/forecast`（新股票） | 3–4 秒（下載＋模型；首檔另含大盤） | ~3 毫秒 |
| `/api/recommendations`（60 檔） | ~70 秒（下載 ~18 秒＋預測 ~50 秒，3 個工作行程） | 0.1 秒（換 h 不必重算） |
| `/api/backtest`（單檔 5 年） | 5–7 秒（前推式 1250 日） | h／參數不同但同一股票：~0.1 秒 |
| `/api/backtest/universe`（60 檔） | ~2 分鐘（60 次前推式，CPU 約 75–80%） | 數毫秒 |

伺服器常駐記憶體約 150–250 MB（每個工作行程約 135 MB）；整體驗證瞬間需要同時載入 60 檔的前推式結果，尖峰可達約 0.4 GB，結束後會歸還給系統。
磁碟快取約 80 MB（`data_cache/`，超過 14 天的預測／前推式／報告檔案會在啟動時清理）。

## 測試

```bash
python3 -m unittest discover -s tests                           # 全部
python3 -m unittest discover -s tests -p "test_data_*.py"       # 資料層（假網路、假時鐘）
python3 -m unittest discover -s tests -p "test_server_*.py"     # 伺服器：真實的行程內 HTTP 伺服器＋假資料／假引擎
TWQ_SKIP_SLOW=1 python3 -m unittest discover -s tests -p "test_server_*.py"   # 略過啟動真實行程池與引擎的整合測試
```

`tests/schema_check.py` 提供 `check_forecast`、`check_recommendations`、`check_history`、`check_search`、`check_backtest`、
`check_universe_report` 等函式，回傳「不符合 DESIGN.md 第 6 節格式的問題清單」（空清單 = 合格），可重複用於手動端到端驗證。
單元測試不連網；唯一連網的是你手動執行的端到端檢查。

## 限制（請務必閱讀）

* **在目前樣本量下沒有偵測到可靠的預測優勢（此驗證的檢定力有限，尤其是 5 日與 20 日預測）。** 在本專案自己的前推式驗證裡，
  多數股票與整個推薦名單都被判定為「沒有統計上顯著的預測優勢」；但「沒有偵測到」不等於「沒有」：約 1,250 筆樣本只有約八成把握偵測到
  排序相關(IC)至少約 0.12（明日）／0.25（5 日）／0.40（20 日）的優勢，更小的效果與雜訊分不開。
  輸出的機率通常落在 50%±5%；少數看似漂亮的數字在多重檢定校正與成本之後多半消失。請把它當成「風險與不確定性的儀表板」，而不是選股神器。
* **存活者偏差。** 推薦宇宙是「今天」流動性最佳的股票（以人工挑選的大型股清單為底，再用今日成交值補足）；已下市、合併或今日已不熱門的公司沒有納入，
  歷史績效因此偏樂觀，報告內已明確揭露。
* **資料來源是非官方的 Yahoo Finance。** 可能延遲、被限流、改版或修正歷史資料；還原權息價以 Yahoo 為準。名稱與成交值來自 TWSE／TPEx 開放資料
  （TPEx 偶爾失敗，上櫃名稱暫時搜尋不到，但輸入代號仍可使用 `.TWO`）。無分時資料，只有日 K。
* **交易日曆。** 以 TWSE 假日表為準；抓不到時只排除週末，颱風假等臨時休市無法預知。年底時下一年度假日表尚未公布，元旦等假日在公布前會被當成交易日。
* **成本模型是簡化的。** 計入手續費（可調折扣）、證交稅、固定滑價；未計入最低手續費、漲跌停鎖死無法成交、零股、市場衝擊、融資融券與股利所得稅。
  回測假設「訊號出現後隔日開盤成交」（可選收盤成交，較樂觀）。
* **調參。** 超參數事先依理論固定；僅有兩個數字用到 2021-08 以前的 2330／0050 資料選定（見 `backend/quant/engine.py` 文件字串），
  但仍不能排除細微的選擇偏誤。
* **只適合本機單人使用。** 沒有身分驗證，預設只綁定 127.0.0.1；前端與 API 未針對公開網路部署強化。尚未在 Windows 上測試（`run.sh` 是 bash，
  可改用 `python -m backend.server`）。

## 免責聲明

本工具為統計模型之研究與教學用途，**非投資建議**。所有預測皆為機率分布而非保證，過去績效不代表未來；模型可能錯誤、資料可能延遲或有誤。
投資一定有風險，請自行判斷並承擔盈虧。作者不對任何因使用本工具而產生的損失負責。
