# `rl_i1000_50k_v2` 訓練與驗收證據（plan10）

**這份檔案寫在任何一局開跑之前。**判準與停止規則先註冊，跑後不改。

---

## 1. 參數（使用者裁示實值）

| 參數 | 值 |
|---|---|
| 學習者 | 風格 **i**（intruder），單一學習者，其餘對手全部凍結 |
| 起點 | `rl_i1000_20k` |
| 目標局數 | 累計 **50k** |
| 訓練後名字 | `rl_i1000_50k_v2` |
| 判準類型 | 優於起點（主判準見 §4） |

### 派生值

| 項 | 值 |
|---|---|
| 起點局數 | 20,000 |
| **新增局數** | 50,000 − 20,000 = **30,000** |
| **輪數** | 30,000 ÷ 500 = **60 輪** |
| 續訓輪號 | `resume=True`，自 **round 40** 續 **41 → 100** |
| 每輪 | 500 局；學習者座位均勻；每局從池內**有放回抽 3 個** |

### 對手池（11 席，訓練期間全部凍結）

```
fox, chess, wolf, optimizer, builder, intruder, hunter,
rl_h1000_20k, rl_i1000_50k, rl_o1000_20k, rl_b1000_50k
```

11 席全數經 `ai.registry` 解析成功，`enabled`/`selectable` 皆為 `true`。
7 個人格恰為 `rl.rollout.PERSONALITY_POOL` 全集。

**抽樣規則**：自訂池以 `rng.choice` ×3 **有放回**（plan10 §1/§4）；
預設七人格池維持 `rng.sample` **不放回**，逐位不變（`tests/test_rl_rollout.py::test_the_default_pool_draws_exactly_what_it_drew_before`）。

---

## 2. 池的例外聲明（使用者裁示）

### 2-1 `rl_i1000_50k`：違反 plan10 §1 第 3 項，經裁示保留

`rl_i1000_50k` 是學習者**從同一個 `rl_i1000_20k` 訓練出來的 50k**
（registry `source: data/multi/i/step_000100.pt`），正是 §1-3 舉的
「訓練 h20k→50k 時池內有 h50k」那個形況，屬**明文禁止**。

**經使用者明確裁示保留，理由：它是實際遊戲池的對手。**

> **報告須標明「v2 可能適應 i50k」。** 訓練是單向的、凍結對手不會反過來
> 適應，v2 學到這批對手的弱點時無法從指標上區分「變強」與「適應了池內
> 特定對手」。

### 2-2 `rl_b1000_50k` 與 `rl_i1000_50k`：有量測、無正向驗證

> **「有預先註冊量測，四筆全部判為『無證據』，未取得正向驗證；
> 經使用者裁示納入。」**

出處 `eval/shared/four-at-once/big_eval/README.md`，「## 判準（預先註冊）」之下：

| 席位 | argmax（400 局，兩份排程） | softmax（400 局，兩份排程） |
|---|---|---|
| `rl_b1000_50k` | `−0.0384`（0.59 se）、`+0.0343`（0.53 se） | `+0.0991`（1.28 se）、`+0.1706`（2.20 se） |
| `rl_i1000_50k` | `+0.0340`（0.50 se）、`+0.1295`（1.82 se） | `+0.0811`（1.08 se）、`+0.1073`（1.53 se） |

四筆皆未過 2 se，判為「無證據」—— 即與各自 20k **無顯著差異**，
**沒有任何一筆構成「已驗證」的正向結論**。

### 2-3 自我對戰

**池不含 `rl_i1000_20k`，故無自我對戰。**
但池含 `rl_i1000_50k`，是**同源下游對戰**（學習者 i@20k 對上 i@50k），
非嚴格自我對戰，但屬 §1-1 的鄰近形況，解讀時須一併保守。

### 2-4 其餘已知風險（plan10 §8）

| # | 風險 | 與本任務的關係 |
|---|---|---|
| 2 | **對凍結池過擬合** | 11 席中 4 席是靜態網路；單向訓練，凍結對手不反向適應 → 評測池見 §5-2 |
| 3 | **池與部署分布不一致** | 訓練池 ≠ 實際遊戲池 → 這就是主判準用 argmax 的原因之一 |
| 4 | **argmax 不受訓練目標控制** | 訓練優化 softmax 期望報酬，argmax 是副產品，可能漂移 |
| 5 | **多重比較** | 只跑**一條**候選 |
| 6 | **事後選擇** | 判準、種子段、池皆已於本檔寫死 |
| 7 | **種子守衛** | `own` 豁免比對完整三元組；`_claimed_train_block` 已加 anchor，見 §3 |
| 8 | **entropy 基準錯配** | 見 §6 |
| 10 | **凍結席與學習者不得共用權重物件** | 見 §7 |

---

## 3. 種子段（`reject_reserved_seeds` 已實跑驗過 FREE）

| 用途 | `seed_base` | 實際取用段 | 個數 | 三元組 |
|---|---|---|---|---|
| **訓練** rounds 41–100 | **7,150,000** | **7,170,000 – 7,199,999** | 30,000 | `("stage RL i50k v2 train", 7170000, 7199999)` |
| **評測** 1000 局 | 7,540,000 | **7,540,000 – 7,540,999** | 1,000 | `("stage RL i50k v2 eval", 7540000, 7540999)` |

兩段**互不重疊**，與全部 18 個已註冊列零重疊。
訓練段夾在 `stage RL validation`（…7,100,999）與 `stage RL students`
（7,200,000…）之間 —— 該 gap 從未被保留、也從未被使用。

**`own` 與 `_claimed_train_block` 一致**：`seed_base` 是 round 1 *本來*
會有的位置，不是被取用的種子，所以 containment 找不到列。`TRAIN_ANCHOR_BLOCKS`
把它對到註冊列，`own` 因而精確等於三元組 → 通過。
`stage RL students` 的行為**未變**（anchor 表優先於 containment，但
7,200,000 不在表內），由
`tests/test_rl_train.py::test_the_i50k_anchor_claims_rounds_41_to_100_and_leaves_the_students_alone`
釘住。

---

## 4. 主判準（預先寫死，跑後不改）

**評測對象**：訓練後的 `rl_i1000_50k_v2` vs 起點 `rl_i1000_20k`。

| 項 | 值 |
|---|---|
| 模式 | **argmax**（單一模式） |
| 局數 | **1000** |
| 種子段 | `stage RL i50k v2 eval` = 7,540,000–7,540,999（新註冊，FREE） |
| 配對 | 同 specs、同模式，current 對起點；`paired_diff = mean(reward_current − reward_start)` |

### 通過條件

```
paired_diff > 0   且   paired_diff ≥ 1.3 × paired_se
```

| 項 | 值 |
|---|---|
| 單側假陽性率 | **約 9.7%**（P(Z > 1.3)，標準常態） |
| 1000 局下的門檻 | SE ≈ 0.0444（由 400 局實測平均 0.0702 按 SE ∝ 1/√n 換算），1.3×SE ≈ **+0.058** |
| 使用者裁示 | **接受約 10% 假陽性** |

### 判準選擇依據

**實際遊戲時 AI 使用 argmax**（`ui.py` 座席畫面、`match.py --mode` 預設）。
訓練 rollout 用 softmax（`rl/rollout.LEARNER_MODE`），所以 argmax 是訓練目標的
副產品而非其直接優化對象（§8 風險 4）。

### 結論措辭（預先寫死）

- **通過僅屬「中等證據」。報告不得寫「已證明更強」。**
- 未通過：**不入池、不開第二條**，結果照記。
- 只比起點 `rl_i1000_20k`，**不與其他版本比**（含 `rl_i1000_50k`）。
- **softmax 只記錄，不入判準，不得作為入池理由。**
- 候選數固定為**一條**；重做是新決定，由使用者下，並註明是在看過結果後做的。

### 過線之後

**過線只是取得入池資格。** 與既有 `rl_i1000_50k` **並存或取代，
由使用者屆時裁示**，本檔不預設。

---

## 5. 評測設定

### 5-1 迴圈內評測（曲線，**不入判準**）

| 項 | 值 |
|---|---|
| 頻率 | `--eval-every 5` → rounds 45, 50, …, 100（12 次） |
| 局數 / 種子 | 100 局，`stage RL validation` = 7,100,000–7,100,999 |
| 模式 | argmax |
| **錨點** | **`--anchor ai/checkpoints/rl_i1000_20k/step_000040.pt`**（**不是**預設的 `data/hc2/step_001000.pt` 0k） |
| 輸出 | `data/rl_i1000_v2/eval.jsonl`，欄名 `current` / `hc_1000`（**歷史欄名**，內容是上面那個錨點） |

> **這條曲線只是曲線，不入 §4 判準，不得作為入池理由。**
> 欄名 `hc_1000` 是 0k 退役前留下來的歷史欄名，改名會把一份基線切成兩段；
> **本 run 該欄的實際內容是 `rl_i1000_20k`。**

> **已知顯示瑕疵**：`--dry-run` 的 `anchor` 行印的是模組常數
> `ANCHOR_CHECKPOINT`，**不是**實際傳入的 `--anchor`（`_dry_run` 收不到
> `anchor_path`）。`init` 行同理。實際接線已實測確認
> `main()` → `train(anchor_path=args.anchor)` → `evaluate(..., anchor_path)`。

### 5-2 驗收評測（**主判準**）

| 項 | 值 |
|---|---|
| 模式 | **argmax**（主判準）+ **softmax**（僅記錄） |
| 局數 | 各 **1000**，同 specs |
| 種子段 | `stage RL i50k v2 eval` = 7,540,000–7,540,999 |
| 配對 | `rl_i1000_20k`（`ai/checkpoints/rl_i1000_20k/step_000040.pt`） |

**評測對手池 = 預設七人格**（`fox, chess, wolf, optimizer, builder, intruder, hunter`）：

> ⚠ **它是訓練池的子集，不是真正的池外對手，且不含任何網路席位。**
> 訓練池 11 席 = 這 7 個人格 + 4 個網路席位；評測用前 7 席。
> 所以 §8 風險 2（對凍結池過擬合）**在本評測下無法被偵測** ——
> 學習者對這 7 席的適應不會被這個測試發現。
> 另外評測池含 `fox`，而訓練池**也**含 `fox`（11 席含全部 7 人格），
> 因此連「訓練沒見過的子集」都不是。

---

## 6. Guardrail 規則（plan10 §6，跑後不改）

`rl_train` 的 guardrail 是**中止**不是告警：
順序為 **update → save → check → `exit 1`**，觸發那輪**會被存下來**
（`rl/rl_train.py` 模組 docstring 與迴圈；`main()` 對 `stopped_by` 回傳 1）。

| guardrail | 條件 | 本 run 的值 |
|---|---|---|
| `nan_or_inf` | 統計欄位非有限值 | — |
| `kl_anchor` | `kl > 0.05` | `kl_coef = 0` → `kl_anchor=None` → **永不觸發** |
| **`entropy`** | **`ent < 0.5 × entropy_reference`** | **`0.5 × 1.394941998852624 = 0.697470999426312`** |
| `first_ratio_dev` | `> 1e-6` | — |
| `disk` | free < 5.0 GB | — |

### 觸發處理（預先寫死）

- **任何非 `disk` 的觸發（`nan_or_inf` / `entropy` / `first_ratio_dev`）：
  訓練視為未完成 —— 不評測、不入池、不續跑、不調閾值、不開第二條。**
  結果照記為「**第 k 輪被 `<name>` 中止**」，附 entropy 軌跡與
  `rounds.jsonl` 摘錄，**停下等裁示**。
- **`disk` 例外**：清出空間後 `resume=True` 續跑，並在本檔註明。
- 不設「達到目標局數前自動停止」以外的停止條件；**中途 entropy 下滑也不得手動介入。**

### `entropy_reference` 的來源環境（§6 明文要求）

> `entropy_reference = 1.394941998852624`，取自 `rl_i1000_20k` 檔內，
> 是 **0k→20k 那段在「七個人格環境」下第 1 輪量到的 entropy**。
> **本次訓練環境是 11 席池（含 4 個網路席位），與來源環境不同。**
> 因此觸發線 **0.69747 可能與新環境不相稱**（plan10 §8 風險 8）。

**相關事實（不作結論）**：i 的 `entropy_reference`（1.3949）是四個 20k 裡
**最低**的（h 是 1.5294），距離觸發線的絕對餘裕比其他風格小。
**這不構成「會觸發」的預測**，只是記錄。

### 已知瑕疵：checkpoint 內的 `name` 欄

`rl_train.NAME` 是模組常數 `'rl_1000_20k'`，`save_checkpoint` 寫進每個檔。
**這是純資訊欄，席位名由 `ai/registry.json` 決定**；`build_brain` 走
`load_brain_at(..., key=...)`，載入不受影響。已裁示接受，不改碼。

---

## 7. 凍結席 sha256 檢查（plan10 §3 步驟 3、§8 風險 10）

四個非人格席**檔案權限皆 `0o664`（owner 可寫）**，所以必須真的檢查，
不能靠唯讀權限保證。

| 席位 | 路徑（registry `checkpoint`） | sha256 |
|---|---|---|
| `rl_h1000_20k` | `ai/checkpoints/rl_h1000_20k/step_000040.pt` | `e1dc6230a70d205a89287468c47db6126717f0e9d42369bc1beca31404801636` |
| `rl_i1000_50k` | `ai/checkpoints/rl_i1000_50k/step_000100.pt` | `ef6ab92bbc11c3ec0c2f02945257a2ba435bfe9b295cb54bf5cb69ea847484a6` |
| `rl_o1000_20k` | `ai/checkpoints/rl_o1000_20k/step_000040.pt` | `272b2edf35e9157ed137a33787ceb5fc736c62f785c304cb6abfa2a273c1ad90` |
| `rl_b1000_50k` | `ai/checkpoints/rl_b1000_50k/step_000100.pt` | `ab6ec0a120667bd98a8f0352c7e35b5c6ac1ec17126a433050fdb93be92d09e3` |

**起點**：`ai/checkpoints/rl_i1000_20k/step_000040.pt` =
`3db862d34fbe48943621e373d29adc456016c12675daf86133abf0460518c9d0`

### 檢查方案

```
階段 0（開訓前）  對四檔各算 sha256，比對上表逐字相同；不同則停
                 另跑 python -m ai --check
階段 1（開訓後）  再算一次，四個 sha256 與階段 0 逐位相等
                 任一不同 → 凍結席被寫 → 訓練作廢，停下等裁示
                 另跑 python -m ai --check
```

**唯讀的依據（已查碼）**：`rl/rl_train.py` 全檔唯一寫入是 `save_checkpoint`，
其路徑一律來自 `checkpoint_paths(out_dir)` = `data/rl_i1000_v2/`；
`rl/rollout.py` 這層只有 `load_policy` / `torch.load`，無 `torch.save`。
`Game.setup_seats(..., checkpoint_dir=None)` → `seats.build_brain` 讀 registry
`checkpoint`，只讀。

---

## 8. 輸出與命令

### 輸出

| | 值 |
|---|---|
| 訓練目錄 | `data/rl_i1000_v2/`（已由 `.gitignore:80` 的 `data/*` 忽略） |
| 佈置 | `cp data/rl_i1000/step_000040.pt data/rl_i1000_v2/latest.pt`，**sha256 逐位相等後才啟動** |
| 產出 | `step_000041.pt` … `step_000100.pt` + `latest.pt` / `current.pt` / `rounds.jsonl` / `eval.jsonl` |
| 不碰 | `data/rl_i1000/`、`data/multi/`、`data/multi_rb80/`、`data/rl_h50k_v2/`、`ai/checkpoints/` 內既有檔 |

### 起點檢查點

| 項 | 值 |
|---|---|
| 路徑 | `ai/checkpoints/rl_i1000_20k/step_000040.pt`（來源 `data/rl_i1000/step_000040.pt`） |
| sha256 | `3db862d34fbe48943621e373d29adc456016c12675daf86133abf0460518c9d0`（實測 = registry） |
| `round` / `step` | 40 / 40 |
| **`entropy_reference`** | **1.394941998852624** |
| **優化器** | ✅ AdamW，`state` 43 參數，動量全部非零；`ppo_cfg` 與 `rl_train` 預設逐欄相同 |
| `rng_state` | 真實 tuple（非 `None`）；實測 `rng` 在迴圈中只被存檔、從未使用 → 無行為影響 |
| 由 `rl_train` 產生 | ✅ `train_cfg.seed_base = 7200000`（學生跑，`stage RL students`） |
| `resume=True` 續 41–100 | ✅ 已端到端驗證（`find_resume_point`→40、optimizer 載入無缺漏、dry-run 40→100） |

### 訓練命令

```bash
python -m rl.rl_train \
  --resume \
  --out data/rl_i1000_v2 \
  --games 30000 --per-round 500 \
  --seed-base 7150000 \
  --n-procs 8 \
  --eval-every 5 \
  --anchor ai/checkpoints/rl_i1000_20k/step_000040.pt \
  --opponent-pool "fox,chess,wolf,optimizer,builder,intruder,hunter,rl_h1000_20k,rl_i1000_50k,rl_o1000_20k,rl_b1000_50k"
```

### 時間預估（實測）

**開訓前小批實測**（拋棄種子 9,900,000 / 9,910,000，未落盤）：

| 項 | 條件 | 實測 |
|---|---|---|
| 訓練 rollout | 500 局、11 席池、softmax、`n_procs=8` | **58.2 s（8.59 局/s）** |
| update | 500 局 batch、cuda | **32.49 s**（取自 `data/rl_i1000/rounds.jsonl` 40 輪平均） |
| 評測 | 200 局、11 席池、argmax、`n_procs=4` | **≈ 4.7 局/s**（current 42.1 s、ref 43.0 s） |

| | 計算 | 時間 |
|---|---|---|
| 訓練 rollout | 60 × 58.2 s | 58 分 |
| 訓練 update | 60 × 32.49 s | 32 分 |
| 迴圈內評測 | 12 × 100 局 × 2 側 = 2,400 局 ÷ 4.7 | 9 分 |
| **訓練合計** | | **≈ 99 分** |
| 驗收評測 argmax | 1000 × 2 側 ÷ 4.7 ≈ 426 s | ≈ 7 分 |
| 驗收評測 softmax | 同上 | ≈ 7 分 |
| **評測合計** | | **≈ 14 分** |

> 評測前會**再做一次小批實測**再排程（plan10 §3 步驟 4）。
> 上表的評測數字是估算，不是排程依據。

---

## 9. 結果（跑完才補；判準見 §4，跑後未改）

### 9-1 訓練

60 輪（round 41 → 100），30,000 局，平均 **85.4 s/輪**（共 85.4 分）。

| 項 | 值 |
|---|---|
| guardrail 觸發 | **0 次**（60 輪全 `OK`，無 `STOPPED after round`） |
| 種子實際範圍 | **7,170,000 – 7,199,999**，與註冊列逐位相同 |
| entropy | 首輪 1.7435 → 末輪 **1.6423**；區間 min 1.6423（round 100）/ max 1.8535 |
| 觸發線 | 0.697471；**最低值餘裕 2.35×** |
| 末輪 | `reward +2.430  rank 2.43  clip 0.287  EV +0.577` |

**凍結席 sha256 前後檢查**：四檔與 §7 表**逐位相同，全部 UNCHANGED**
（`rl_h1000_20k` / `rl_i1000_50k` / `rl_o1000_20k` / `rl_b1000_50k`）。
`python -m ai --check` 訓練前後皆 exit 0。
`git status` 僅剩 `plan10.md`（使用者改寫，未納入本任務）。

**迴圈內評測**（12 次，rounds 45…100）：錨點列 `hc_1000` 的
`avg_points` 恆為 **2.9119**（12 列相同），確認 `--anchor`
指向的是凍結的 `rl_i1000_20k` 而非 0k。**此為曲線，不入判準。**

### 9-2 評測前速度實測

200 局 × 2 側、argmax、預設七人格、拋棄種子 9,900,000（未碰註冊段）：

| `n_procs` | 速度 | 1000 局 × 2 側推算 |
|---|---|---|
| 4 | 5.68 局/s | 352 s |
| 8 | 10.77 局/s | 186 s |
| **16（採用）** | **14.25 局/s** | **140 s** |

實際執行 argmax **127.5 s**、softmax **128.9 s**，與推算相符。
（§8 的 14 分估計用的是 `n_procs=4`，已由本實測取代。）

### 9-3 驗收評測結果

設定：1000 局、種子 **7,540,000–7,540,999**（`stage RL i50k v2 eval`）、
預設七人格池、`n_procs=16`、配對 `ai/checkpoints/rl_i1000_20k/step_000040.pt`
（`sha256 3db862d34fbe…`，與 §7 相同）。

證據檔：`step_000100_argmax.json`、`step_000100_softmax.json`。

#### 主判準：argmax（依 §4）

| 項 | 值 |
|---|---|
| `paired_diff` | **−0.029730** |
| `paired_se` | **0.041346** |
| `paired_diff / paired_se` | **−0.719** |
| 門檻 | `> 0` 且 `≥ 1.3 × paired_se`（= 0.0537） |
| 平均分 | current 2.9632、起點 2.9929 |

**`paired_diff < 0`，不滿足 `> 0`。判準未通過。**

#### softmax（僅記錄，**不入判準**）

| 項 | 值 |
|---|---|
| `paired_diff` | +0.091540 |
| `paired_se` | 0.046037 |
| `paired_diff / paired_se` | +1.988 |
| 平均分 | current 2.7384、起點 2.6468 |

**依 §4：softmax 只記錄，不得作為入池理由。**

#### 依 §4 的結論

- **argmax 主判準未通過。**
- **不入池、不開第二條**（§4「未過線不入池、不開第二條」）。
- `rl_i1000_50k` 與 `rl_i1000_20k` **全部不動**。
- `paired_diff` 為負但僅 0.719 se，**不構成「已證明更差」**—— §4 的門檻
  只定義通過條件，未定義反向結論；此處僅陳述比值，不下方向性結論。

