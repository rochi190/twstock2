# 外部獨立審查與進度評估表（供 Claude 續接與討論）

## 一、目前執行狀態快照（2026-10-23 暫停後盤點）

1. **主專案狀態**：`backend/`、`frontend/`、`tests/` 程式碼皆未被修改（停在唯讀 Audit 階段），994 個單元測試（quant 141, eval 251, data 119, server 138, front 345）全數通過。
2. **Audit Workflow (`wf_d30c4e9f-9d7`) 4 個子代理實際進度**：
   - `review:causal`（`agent-ae68c410be24af4fe`）：**已完成**（寫入 `journal.jsonl` Entry 17）。148 次 poison-future 測試 100% bit-identical 無偷看未來；回報 C-01 ~ C-15。
   - `review:math`（`agent-a0ca4831febf08e78`）：**已完成**（寫入 `journal.jsonl` Entry 19）。各數學元件與獨立暴力法/MC 對齊至 `1e-11..1e-16`；回報 M-01 ~ M-14。
   - `review:stats`（`agent-aa48e36d6f14bdebb`）：雖於 12:41 被 `TaskStop`，但其啟動的背景腳本 `run_chain1.sh` **已於 13:07 在背景全部跑完**（`/tmp/.../scratchpad/audit/stats/chain1.done` 與 `mc_single_*.csv` 皆已產出，不必重跑 MC！）。`analyze_single.py main_d0 main_d4` 顯示在無預測能力下 `verdict != no_edge` 的偽陽性率僅 `0.00% ~ 0.25%`。
   - `review:product`（`agent-ae4d1a7a072128cbe`）：被暫停前已完成真實 API 抓取（位於 `scratchpad/audit/product/real/*.json`）與 `decomp.py` 分析。⚠️ **注意：遺留兩個背景測試 server 尚未關閉（PID `1606527` port 18431、PID `1607557` port 18432），續跑時請先清理。**

---

## 二、各模組正確性與邏輯總體評估表

| 模組 / 面向 | 評估結果 | 正確與優秀之處 | 發現的邏輯缺陷 / 待修正項 | 嚴重度 |
| :--- | :---: | :--- | :--- | :---: |
| **1. 因果性與無偷看未來**<br/>(`quant/`, `evaluation/`) | ✅ **極佳** | 絕對重估網格 `k*refit_every`、DLM/DMA/Platt 只用已到期結果 `(t-h)`、`forecast_latest == walk_forward[-1]` 誤差 `< 3e-17`、回測 `next_open` 訊號不提前獲利。 | `features.py:274,291` (`C-11`)：`vol_z` 與 `has_market` 使用全序列 `np.isfinite().sum()` 判斷而非 prefix 累積計數，導致極早期截斷時（如 2317）有微小差異。 | Low |
| **2. 推薦排序公式**<br/>(`recommend.py`, `universe_bt.py`, `DESIGN.md §9`) | ❌ **有重大數學扭曲** | James–Stein 收縮與 Ledoit–Wolf 共變異數、分數凱利權重上限設計合理。 | **核心盲點（見下方第三節詳述）**：`score = (exp_return - cost) / sigma` 在 $h=1,5$ 因分子為負，除以 $\sigma$ 後變成**波動度越大排名越高**；在 $h=5,20$ 則因 $E[e^R-1] \approx \mu_{\text{log}} + \frac{1}{2}\sigma^2$ 受 $+\frac{1}{2}\sigma^2$ 支配。只照 `DESIGN.md §9` 改用 raw `score` 排 `leaders` 仍會排出一堆高波動股。 | **High** |
| **3. 統計基準與檢定**<br/>(`report.py`, `metrics.py`, `universe_bt.py`) | ⚠️ **架構佳但基準線需修正** | Bonferroni + 雙指標共識 + HAC 標準誤 + $n/h$ 有效樣本數，MC 偽陽性率 $\le 0.25\%$。 | 1. **`C-01 / M-02`**：`report.py:173,222` 與 `universe_bt.py:259` 的 Brier skill / DM 檢定使用測試期**事後平均** `up.mean()`，違反 `DESIGN.md §9(a)`，人為造成 Brier skill 全面偏負（實測因果基準下為 $-0.11\% \sim +0.02\%$，事後基準下為 $-0.30\% \sim -2.30\%$）。<br/>2. **`C-08`**：`report.py:240` `assess_evidence` 缺少反向否決（veto）：2412、3045、4904 在 $h=1$ 方向準確率低於 `always_up` 且 Brier skill 為負，卻因 tick 微結構反彈使 IC/MSE 顯著而被判為 `moderate_edge`。 | **High** |
| **4. 平盤 ($R=0$) 與離散價格處理**<br/>(`engine.py`, `calibration.py`, `report.py`) | ⚠️ **定義不一致** | 漲跌停 $\pm 10\%$ 截斷與單調分位數維持良好。 | 1. **`C-02 / C-03 / M-01`**：校準器將 $R=0$ 視為失敗（`Rc > 0`），但 `report.py:155` 將 $R=0$ 剔除，導致高平盤率股票（如 2412/3045）$h=1$ 的 $p_{\text{up}}$ 系統性偏高 $+2\sim 7\text{pp}$。<br/>2. **`M-05 / C-10`**：`engine.py:340` 的 `pit = pool.cdf(raw_r)` 未對離散 tick/平盤做隨機化（randomised PIT），導致 63 檔中有 16 檔在 $h=1$ 的 PIT KS 檢定假警報（$p < 0.01$）。 | Medium |
| **5. 資料層與快取鍵**<br/>(`data.py`, `jobs.py`) | ⚠️ **有邊界漏洞** | Yahoo `period1=0`、台北時區轉換、未收盤 K 棒過濾 (`completed`)、原子寫入與離線降級完整。 | 1. **`C-04`**：`data.py:700 (_merge)` 當 Yahoo 盤後回傳最後一日為 null row 時，refresh 會把快取中原本有效的最後一根 K 棒刪掉（歷史倒退一天且 `stale=False`）。<br/>2. **`C-05`**：`price_signature` 未包含 `open/high/low/volume`；`code_fingerprint` 未包含 `config.py/types.py/data.py`。 | Medium |
| **6. 區間涵蓋率與 HMM 狀態**<br/>(`garch.py`, `hmm.py`) | ℹ️ **模型特性（可選優化）** | 模擬環境下 90% 區間涵蓋率精確為 `89.9%~90.9%`。 | 1. **`M-04`**：實盤 $h=1$ 的 90% 區間未涵蓋率為 `11.35%`（理想 10%），其中 68% 來自週一（12.4%）與連假後（21.7%）的跨日變異數（交易日曆時間 vs 自然日時間）。<br/>2. **`M-06 / C-12`**：HMM 狀態平均停留僅 4.6~9 天，且標籤在「盤整」與「多頭」間容易隨微小擾動翻轉。 | Low / Medium |

---

## 三、重點討論：為什麼 `DESIGN.md §9` 的推薦排序需要進一步修正？

在 `DESIGN.md §9` 第 1、3 點中，原本規劃：
> 「`rank` 與 `all`、`leaders`、`laggards` 改用未收縮的原始 `score = (E[simple return] - cost) / sigma` 排序。」

但根據 `review:causal (C-06)` 與 `review:product (decomp.py)` 在真實 60 檔快取資料上的實測，**直接用原始 `score` 排序會產生嚴重的反直覺結果**：

### 1. 實測數據證據（來自 `recs_h1.json`, `recs_h5.json`, `recs_h20.json`）
- **在 $h=1$ 時**：全市場預期報酬 $E[r_1] \approx +0.05\%$，扣除來回交易成本 $c \approx 0.57\%$ 後，**所有股票的分子 `net = E[r_1] - c` 都是負數（約 $-0.45\% \sim -0.57\%$）**。
  - 當分子為負數時，除以 $\sigma$ 會讓 **$\sigma$ 越大（波動越劇烈）的股票，負分越接近 0，排名反而衝到最前面**（`Spearman(score, sigma) = 0.99`）！
  - 實測 $h=1$ 依 raw `score` 排名前五名：**1303 南亞 ($\sigma=4.29\%$)、1326 台化 ($\sigma=3.81\%$)、3653 健策 ($\sigma=3.83\%, P(\text{漲})=0.492 < 50\%$！)、6505 台塑化、3037 欣興**。
  - 實測 $h=1$ 倒數五名：**2002 中鋼 ($\sigma=0.92\%$)、2412 中華電 ($\sigma=1.08\%$)、2912 統一超 ($\sigma=1.10\%$)、1216 統一 ($\sigma=1.16\%$)、3045 台灣大 ($\sigma=1.34\%$)** —— 全部只是因為低波動就被排在最後！
- **在 $h=5$ 與 $h=20$ 時**：模型預測的是對數報酬 $R \sim (\mu_{\text{log}}, \sigma_{\text{log}}^2)$，轉成算術簡單報酬時：
  $$E[e^R - 1] \approx \mu_{\text{log}} + \frac{1}{2}\sigma_{\text{log}}^2 \implies \text{score} \approx \frac{\mu_{\text{log}} - c}{\sigma} + \frac{1}{2}\sigma$$
  - 因為 $\mu_{\text{log}}$ 被強烈收縮，`er - median` 與 $\frac{1}{2}\sigma^2$ 的相關係數高達 **0.994**（`Spearman(er, sigma) = 0.89 ~ 0.91`，但 `Spearman(p_up, sigma) = -0.23 ~ -0.58`）！
  - 結果在 $h=5$ 與 $h=20$，依 raw `score` 排名前五名**依然是 1303 南亞 ($\sigma_{20}=24.34\%$)、3017 奇鋐、3037 欣興、3653 健策、3443 創意**；而真正 $P(\text{漲})$ 最高、訊號最穩的 **0050 ($P(\text{漲})=60.9\%, \sigma_{20}=6.93\%$)** 與 **2330 台積電 ($P(\text{漲})=58.4\%, \sigma_{20}=7.62\%$)** 反而被高波動股擠下去。

### 2. 建議與 Claude 討論的修正方案
將「**相對多空排序訊號**」與「**扣成本後是否可交易 / 凱利部位權重**」解耦：
1. **相對方向排序（用於 `rank`、`all`、`leaders`、`laggards`、`lean`）**：
   - 改用**不減去固定交易成本、且不含 $+\frac{1}{2}\sigma^2$ 凸性膨脹的無偏風險調整訊號**：例如對數報酬資訊比 $\text{ir\_log} = \mu_{\text{log}} / \sigma_{\text{log}}$（或中位數報酬訊號比 $\text{median\_return} / \sigma$，或標準化勝率 $z(p_{\text{up}})$）。
   - 實測改用 $\mu_{\text{log}} / \sigma_{\text{log}}$ 後：
     - $h=1$ Top 3 變為：`2330 台積電`、`0050`、`2884 玉山金`
     - $h=5$ Top 3 變為：`0050`、`2330 台積電`、`2884 玉山金`
     - $h=20$ Top 3 變為：`0050`、`2330 台積電`、`1234 黑松 / 2412 中華電`
     排名與 $P(\text{漲})$ 完全一致，徹底消除「負報酬除以 $\sigma$」與「$+\frac{1}{2}\sigma^2$ 波動膨脹」的假象。
2. **交易成本門檻與部位權重（用於 `edge_covers_cost`、`signal == "buy"`、`weight`）**：
   - 維持使用扣除交易成本後的 `net_exp_return = er - cost`（或以中位數/幾何報酬扣成本）與收縮後的 `score_shrunk` 來決定是否滿足嚴格 `buy` 門檻及凱利權重；當 `net_exp_return <= 0` 時，直接將該股的交易優勢視為 0，不讓負的 `-cost / sigma` 參與跨股排序。