任務 plan7-A：修復 \_all_stuck 短路低報。單獨 commit。

授權例外：本任務授權修改 game.py 與 engine.py（僅限 \_all_stuck 與 \_advance 的鎖定掃描）。

1. game.py \_all_stuck 與 engine.py \_advance 同步改為不短路：掃完所有未鎖座位，有人能動則 alive=True 並 continue，無人能動則鎖定。
2. 更新受影響測試：test_rl_imitation.py 的 under_reported 斷言反轉為 == 0；test_stuck_monotone.py 新增「每步後所有已卡死座位必已鎖定」逐手斷言；更新 imitation.py 相關 docstring。
3. 效能：改前改後各以固定種子跑 200 局，報牆鐘時間與每步 has_legal 呼叫次數。
4. 空 legal list 比例：改前改後各量「to_move 無合法步」的決策點比例與遊戲比例（基準 5.7% / 84%）。
5. 變異驗證：暫時還原其中一邊的短路，確認交叉驗證紅燈，然後還原。變異驗證不得觸碰真實 data/。
6. 驗收：全套 -n 8、串行 wall-clock、tools/cross_check_engine.py、match.py 未給 --pool 與 --pool all 對改動前黃金值逐局相同。
7. 不訓練、不動 data/、不動 ai/、不動 rl/features.py。
8. 回報：commit hash、git status、data/ 檔案數、與指示不符之處。若效能退步超過 10%，停下回報，不要自行優化。

任務 plan7-B：用修正後的引擎重生成模仿資料並重訓。

1. 資料：用 rl/collect.py 生成到新目錄 data/hc1，規格與 hb1 完全相同（老師 hunter、權重、種子區段、B1 規則）。不覆蓋 data/hb1。只啟動一個生成實例，啟動前確認無殘留進程。
2. 驗證並與 hb1 對照列表：樣本數、局數、前三手書本標籤保留率、stuck 欄位與重算 has_legal 零不符、書本 index0 比例。
3. 預計算 legalactions.npz 到 data/hc1。
4. 訓練：設定與 H-B2 相同（27 通道、batch 256、AdamW wd 1e-4、峰值 1e-3、fp32、單一固定種子），以下三項改變：
   - 總步數 10,000
   - cosine 終點 = 第 10,000 步，衰減至 1e-5
   - warmup 維持 500 步
     檢查點每 1,000 步一個，共 10 個（step_001000 … step_010000），存權重 + optimizer 狀態 + 步數，全部保留。輸出 data/hc2。驗證損失每檢查點記錄。
5. 回報中列出新舊 LR 曲線在第 1000、2000、5000、10000 步的值。
6. 評測：match.py --pool 對 hc2 的檢查點（至少 step_002000、step_005000、step_010000）與舊 step_2000，同環境 5000 局，一律 --dry、--out 指向非 data/ 預設以外位置或確認不覆蓋；各組用不同種子。
7. 不啟動 RL，不自動延長，完成後停下。
8. 回報：commit hash、data/ 檔案數變動、各檢查點驗證損失、評測表（含標準誤）、與指示不符之處。

---

## plan7-B 結果（2026-10-04）

程式碼基準：`51d26d7`（plan7-A）。`data/hc1` 的 manifest 記 `commit=51d26d7, dirty=false`，
所以資料與訓練都能追回這一個 commit。

### 1. 資料 `data/hc1`

規格逐項取自 `data/hb1/manifest.json`，種子區段也相同（train 6000000–6029378、
valid 6100000–6101545、workers 12），因此兩份資料只差在 latch。

| 項目 | hb1 | hc1 |
| --- | --- | --- |
| action_table_hash | 91252a354b090d43 | 91252a354b090d43（相同） |
| 局數 | 30,925 | 30,925 |
| 樣本數 | 1,191,605 | 1,191,605 |
| 分片 | 72 | 72 |
| wall | 1712.9 s | 1063.6 s |

### 2. 逐欄比對：只有 `stuck` 不同

1,191,605 列 × 72 分片逐一比對，**14 個欄位中 13 個完全相同**：
`own_bits`、`hand_bits`、`to_move`、`action`、`n_legal`、`n_candidates`、`shortlist_idx`、
`shortlist_score`、`ply`、`game_id`、`in_prefix`、`outcome_util`、`outcome_remaining`、
`own_move_count`。

| `stuck` 差異 | 數值 |
| --- | --- |
| 有差異的列 | 190,349 |
| 差異的「座位-回合」格 | 269,592 |
| 全部是 false→true（舊資料少報） | 269,592 |
| 依 seat id 分布 | seat0 **0**、seat1 63,367、seat2 94,863、seat3 111,362 |
| 最早出現的 ply | 29 |
| 其中 `to_move` 自己的座位不符 | **0** |

兩個值得記下的事實：

- **seat 0 從來沒有差異**。掃描從 owner 0 開始，短路只可能落在 0 之後，所以少報結構上
  只能發生在 seat 1/2/3。
- **`to_move` 自己的座位不符為 0**。資料列只在行動者有合法步時才產生，所以被餵進網絡的
  那一列，其自身通道從來沒錯過；受影響的是同一列裡「其他三席」的通道。

### 3. 書本指標（兩份完全相同）

| 指標 | hb1 | hc1 |
| --- | --- | --- |
| 書本 step0/1/2 列數 | 66,015 / 66,015 / 66,015 | 同 |
| 前三手標籤保留率 | 1.000000 | 1.000000 |
| index0 存在比例 | 1.000000 | 1.000000 |
| index0 == action 比例 | 1.000000 | 1.000000 |

第 29 手以前沒有任何差異，書本統計自然不動。

### 4. `stuck` vs 重算 `has_legal`（四席全掃）

| 資料 | 檢查列數 | 不符列數 | 座位-回合不符 |
| --- | --- | --- | --- |
| hb1 | 1,191,605 | 190,349 | 269,592 |
| hc1 | 1,191,605 | **0** | **0** |

### 5. 預計算快取

`legalactions.npz` 240,569,093 個索引 / 1,191,605 列，976,585,236 bytes —— 與 hb1
**位元組數完全相同**，因為合法步集合本來就不依賴 `stuck`。
`labels_view.npy` 4,766,548 bytes。

### 6. 訓練 `data/hc2`

設定與 H-B2 只差三項（其餘全部沿用，含 27 通道、batch 256、AdamW wd 1e-4、
峰值 1e-3、fp32、seed 20260903、64×6）：

```
total_steps            50000 -> 10000
lr_schedule_end_step   50000 -> 10000
checkpoint_every        2000 ->  1000
```

LR 曲線（指示第 5 點）：

| 步數 | 舊 H-B2（50k 排程） | 新 H-C2（10k 排程） |
| --- | --- | --- |
| 1,000 | 9.99751e-4 | 9.93249e-4 |
| 2,000 | 9.97759e-4 | 9.40340e-4 |
| 5,000 | 9.79949e-4 | 5.45877e-4 |
| 10,000 | 9.12720e-4 | **1.0e-5** |

10,000 步 / 892.5 s / 平均 84.3 ms（median 84.2、p95 92.3）/ 2,868 samples/s，
無 NaN。驗證損失逐檢查點（單調下降，10k 步內未見過擬合）：

| step | train | val |
| --- | --- | --- |
| 1,000 | 1.2588 | 1.3813 |
| 2,000 | 1.3526 | 1.3428 |
| 3,000 | 1.2036 | 1.1899 |
| 4,000 | 1.1110 | 1.1545 |
| 5,000 | 1.0381 | 1.1190 |
| 6,000 | 0.8367 | 1.0672 |
| 7,000 | 1.0723 | 1.0329 |
| 8,000 | 1.0230 | 1.0000 |
| 9,000 | 0.8690 | 0.9819 |
| 10,000 | 0.9558 | **0.9768** |

同一步數下 H-B2 的 val 是 1.0499（step 10,000），H-C2 是 0.9768 —— 但兩者 LR 相差
91 倍，這個差距不能歸因於資料。

### 7. 評測（各 5,000 局，8 選項 = 7 人格 + 該檢查點，argmax，全 --dry）

各組用不同種子（910001–910004，依指示）。受測席：

| 檢查點 | 席次 | 平均分 | 標準誤 | 平均餘格 |
| --- | --- | --- | --- | --- |
| hc2 step_002000 | 2,511 | 3.1250 | 0.0191 | 10.89 |
| hc2 step_005000 | 2,568 | 3.0884 | 0.0189 | 10.92 |
| hc2 step_010000 | 2,495 | 3.1287 | 0.0195 | 10.85 |
| hb2 step_002000（舊） | 2,656 | 3.1570 | 0.0181 | 10.77 |

四組全部落在 3.09–3.16，最大差 0.069，約 2.6 個合併標準誤。**沒有一個檢查點與其他
有可辨識的差距**，包括在 `stuck` 通道語意已改的資料上重訓的 H-C2 三個。同一局裡其餘
七席的分數在四組之間也都落在 ±0.06 內。

### 8. 與指示不符之處

1. **`--prefix hb1`**：`rl/caches.shard_order()` 寫死 `startswith("hb1_")`，`data/hc1`
   若用 `hc1_` 前綴，`Resident` 就讀不到。改用 `--prefix hb1` 讓檔名與 hb1 一致，
   資料集靠**目錄名**區分。未改共用程式碼。
2. **一併建了 `labels_view.npy`**：指示第 3 點只寫 `legalactions.npz`，但
   `Trainer` 用的 `C.Resident` 同時需要 `labels_view.npy`，缺它訓練無法啟動。
3. **`rl.caches.build_legal_cache` / `build_label_cache` 無法使用**：兩者的 pool worker
   是閉包，`pool.map` 即使在 `fork` 下也要 pickle 函式物件，實測拋
   `AttributeError: Can't pickle local object ...<locals>.one`。改寫成模組層級 worker
   （/tmp），並先對 `data/hb1` 重建兩個快取做驗證：`labels_view.npy` 相同、
   `legalactions.npz` 的 flat/offsets/counts/shards/rows **全部相同**（240,569,093 個
   索引），確認產物格式與 H-B2 訓練時用的完全相同。
4. **評測用薄驅動程式呼叫 `match.main()`**：`match.py` CLI 沒有 `--checkpoint-dir`
   （`run_league()` 有，但 `main()` 沒往下傳），而 `step_5000` / `step_10000` 也不在
   `seats.IMITATION_STEPS` 裡、`--pool` 會判為未知名稱。驅動程式只設
   `seats.IMITATION_CHECKPOINT_DIR` 與 `seats.IMITATION_STEPS` 兩個 module 屬性，
   其餘（argparse、pool 展開、`--dry`、`summarise`、`--out` JSON）全部走原路徑。
   未為了一次量測去改共用程式碼。
5. **跨組比較不是配對的**：依指示各組用不同種子，所以「H-C2 對 H-B2」是兩組獨立樣本。
   `match.py` 印的標準誤是**組內**跨席的，跨組還要加上組間差異；上表的 2.6 個合併標準誤
   就是這樣估的。
6. **`tools/cross_check_engine.py` 未跑**：plan7-B 沒有要求，且 plan7-A 已驗過
   （115,138 步 IDENTICAL）。
7. **repo 程式碼零改動**，故本 commit 只有這份紀錄。全套 `-n 8` 仍 764 passed。
8. **評測輸出全部寫到 `/tmp`**，未動 `data/match`（維持 3 檔）。

### 9. 結論

修掉 latch 少報之後，模仿資料的 `stuck` 欄位第一次和規則一致（0 不符），
H-C2 在 10k 步內驗證損失單調降到 0.9768，評測上三個檢查點與舊 H-B2 step_002000
無法區分。換句話說：**這個修正改變了資料與特徵的語意，但沒有付出可測得的強度代價。**
真正要回答「修正是否讓模仿更強」的證據還不夠 —— 這是 10k 步、單一種子、
5,000 局、每組不同種子的量級；要把 0.03–0.07 的差距分辨出來需要更多局或配對種子。
