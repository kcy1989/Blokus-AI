# data/ — 訓練資料與檢查點

`data/` 幾乎全部被 `.gitignore` 排除（`data/*`），因為它可重建、且大（2026-10-19
實測約 6.6G）。這個目錄按**學習者血統**分組，與 `eval/` 同構。

**被席位載入的路徑一律以 `ai/registry.json` 的 `checkpoint` 為準**；`source` 與
`seats.RL_SEATS` 記的是「在哪裡訓練」。下表「席位」欄指的是該目錄是哪個席位的訓練來源；
「發布檔」欄是 `ai/registry.json` 的 `checkpoint` 的**手動同步副本**，若與 `registry.json`
不一致，一律以 `registry.json` 為準。

## `data/imitation/` — H 階段（資料集與老師）

| 目錄 | 大小 | 席位 | 可否刪 |
| --- | --- | --- | --- |
| `h0` | 4.6M | — | 保留（H0 證據） |
| `h1` | 5.9M | — | 保留（H1 證據；`rl.paired3` 讀 manifest） |
| `hb1` | 1001M | — | 保留（H-B1 資料集；`rl/collect.py`、`rl/caches.py` 引用） |
| `hb2` | 136M | — | 保留（H-B2 檢查點，仍可按步載入） |
| `hc1` | 1002M | — | 保留（H-C1 資料集；`test_rl_collect` 讀 manifest） |
| `hc2` | 55M | `IMITATION_CHECKPOINT_DIR`（非席位；`rl/` 與 `--adhoc` 讀） | 保留（其中三個 `.pt` 入版控） |
| `hcb` | 84M | — | 保留 |
| `hci` | 77M | — | 保留 |
| `hco` | 80M | — | 保留 |
| `hco_trial_dirty` | 112K | — | 可刪（暫存 trial） |
| `imit_b1000` | 54M | 0k 權重（`--adhoc` 重放讀） | 保留（`step_001000.pt` 入版控） |
| `imit_i1000` | 54M | 0k 權重（`--adhoc` 重放讀） | 保留（`step_001000.pt` 入版控） |
| `imit_o1000` | 54M | 0k 權重（`--adhoc` 重放讀） | 保留（`step_001000.pt` 入版控） |

## `data/rl/` — 學習者訓練輸出

| 目錄 | 血統 | 大小 | 席位 | 發布檔 | 可否刪 |
| --- | --- | --- | --- | --- | --- |
| `h/20k` | h | 224M | `rl_h1000_20k` | `ai/checkpoints/rl_h1000_20k/step_000040.pt` | 保留（席位來源） |
| `h/30k` | h | 5.5M | `rl_h1000_30k` | `ai/checkpoints/rl_h1000_30k/step_000060.pt` | 保留（席位來源；v2 run 的 round 20 副本） |
| `h/50k_v2` | h | 7.3M | — | — | 可刪（從未開跑，只有前置檔） |
| `h/50k_v3` | h | 337M | — | — | 可刪（最終未入池；round 20 已複製到 `h/30k`） |
| `h/50k_v4` | h | 229M | — | — | 可刪（最終未入池；r35 僅作後續起點） |
| `h/100k` | h | 581M | `rl_h1000_100k` | `ai/checkpoints/rl_h1000_100k/step_000200.pt` | 保留（席位來源） |
| `h/t05` | h | 337M | — | — | 可刪（T=0.5 run 未入池） |
| `i/20k` | i | 224M | `rl_i1000_20k` | `ai/checkpoints/rl_i1000_20k/step_000040.pt` | 保留（席位來源） |
| `i/20k_par` | i | 19M | — | — | 可刪（掃描變體） |
| `i/20k_seq` | i | 19M | — | — | 可刪（掃描變體） |
| `i/20k_v2` | i | 332M | — | — | 可刪（v2 run 未入池） |
| `i/37.5k` | i | 5.5M | `rl_i1000_37.5k` | `ai/checkpoints/rl_i1000_37.5k/step_000075.pt` | 保留（席位來源；v3 run 的 round 35 副本） |
| `i/50k_v3` | i | 337M | — | — | 可刪（最終未入池；round 35 已複製到 `i/37.5k`） |
| `o/20k` | o | 224M | `rl_o1000_20k` | `ai/checkpoints/rl_o1000_20k/step_000040.pt` | 保留（席位來源） |
| `o/20k_par` | o | 19M | — | — | 可刪（掃描變體） |
| `o/20k_seq` | o | 19M | — | — | 可刪（掃描變體） |
| `o/50k_v2` | o | 332M | — | — | 可刪（v2 run 未入池） |
| `o/50k_v3` | o | 332M | `rl_o1000_50k_v3` | `ai/checkpoints/rl_o1000_50k_v3/step_000060.pt` | 保留（席位來源） |
| `b/20k` | b | 224M | `rl_b1000_20k` | `ai/checkpoints/rl_b1000_20k/step_000040.pt` | 保留（席位來源） |
| `b/20k_par` | b | 19M | — | — | 可刪（掃描變體） |
| `b/20k_seq` | b | 19M | — | — | 可刪（掃描變體） |
| `shared/four-at-once` | 共同 | 109M | `rl_{h,o,b,i}1000_50k` | `ai/checkpoints/rl_{h,o,b,i}1000_50k/step_000100.pt` | 保留（席位來源） |
| `shared/four-at-once-rb80` | 共同 | 152M | — | — | 可刪（balanced-schedule 變體，未產出席位） |

「可刪」是**候選**，不是本輪已刪。刪除前要確認對應的 `eval/` 證據不需要重放；席位來源
（保留欄）在席位仍載入時不得刪。

## 不動的檔與其他殘檔

以下不屬於任何血統，本輪不動：

| 路徑 | 用途 |
| --- | --- |
| `humanlog/` | runtime；`gamelog.LOG_DIR` 指向它 |
| `match/` | runtime；`match.DEFAULT_OUT_DIR` 指向它 |
| `manifest.json` | 原始資料集 manifest（用途未查明） |
| `records_pre_hc2.json`、`records_pre_rl.json` | 舊 `records.json` 快照（用途未查明） |
| `imit_train_*.npz`、`imit_valid_*.npz` | H 階段 imitation 分片，留在根層（見下） |

三個訓練日誌已於 2026-10-19 搬進各自的 run 目錄，檔名不變，搬動前後 md5 逐檔一致：

| 舊路徑 | 新路徑 | md5 |
| --- | --- | --- |
| `rl_h50k_v3_train.log` | `rl/h/50k_v3/rl_h50k_v3_train.log` | `6801ce62…` |
| `rl_h50k_v4_train.log` | `rl/h/50k_v4/rl_h50k_v4_train.log` | `c4372a5e…` |
| `rl_i50k_v3_train.log` | `rl/i/50k_v3/rl_i50k_v3_train.log` | `b5cd83ea…` |

**刪除候選（待裁決，本輪未動）**：`plan9_step1_backup/`（5.5M）。

### 24 個 `imit_*.npz` 為什麼留在根層

`data/imit_train_*.npz` 與 `data/imit_valid_*.npz` 共 24 檔（12 個 train 分片、種子段
1,000,000；12 個 valid 分片、種子段 2,000,000），合計 6,011,986 bytes。它們**不屬於
h / i / o / b 任何單一血統**：是 H 階段 `rl/collect.py` 產出的 imitation 分片，跨血統
共用同一套命名，所以沒有可以歸進去的血統目錄。三個留在根層的理由：

1. **有已提交證據在盤點它們**：`docs/reports/g_raw.json` 的 `disk.files` 以**裸檔名**
   記下每一個檔與大小（`imit_train_1000000_1000166.npz`: 440284 …，合計 6,011,986
   bytes）。搬動會讓那份盤點描述的目錄內容對不上。
2. **程式的產出預設就在根層**：`rl/collect.py` 的 `--out` 預設是 `DATA_DIR =
   os.path.join(HERE, "data")`，重跑仍會生成在 `data/` 根層；把現有檔搬走會立刻與
   程式預設不一致。
3. **計畫文件以這個路徑當範例**：`docs/plans/plan3.md:227` 寫 `data/imit_train_…npz`。

因此這 24 檔**不可隨任一血統一起刪除**（例如「刪掉 h 的 v2 分支」不涵蓋它們）。要動
就整批另案處理，並同步 `rl/collect.py` 的 `--out` 預設與 `g_raw.json` 的說明。
