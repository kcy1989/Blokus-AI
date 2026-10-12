# `rl_o1000_50k_v2` 訓練與驗收證據（plan10）

**這份檔案寫在任何一局開跑之前。**判準與停止規則先註冊，跑後不改。

---

## 1. 參數（使用者裁示實值）

| 參數 | 值 |
|---|---|
| 學習者 | 風格 **o**（optimizer），單一學習者，其餘對手全部凍結 |
| 起點 | `rl_o1000_20k` |
| 目標局數 | 累計 **50k** |
| 訓練後名字 | `rl_o1000_50k_v2` |
| 判準 | argmax 優於起點（見 §4） |

### 派生值

| 項 | 值 |
|---|---|
| **新增局數** | 50,000 − 20,000 = **30,000** |
| **輪數** | 30,000 ÷ 500 = **60 輪** |
| 續訓輪號 | `resume=True`，自 **round 40** → **41…100** |
| 每輪 | 500 局、座位均勻、池內**有放回抽 3 個** |

### 對手池（11 席，訓練期間全部凍結）

```
rl_b1000_50k, hunter, rl_o1000_20k, rl_i1000_50k, wolf, chess,
fox, rl_h1000_20k, optimizer, builder, intruder
```

11 席全數經 `ai.registry` 解析成功、`enabled=true`、去重後 11。
**7 個人格恰為 `rl.rollout.PERSONALITY_POOL` 全集**（無缺漏）。
自訂池以 `rng.choice` ×3 **有放回**（`resolve_opponent_pool` 驗證，
`eabe366` 起已支援；預設七人格池維持不放回、逐位不變）。

---

## 2. 池的例外聲明（使用者裁示）

### 2-1 ❌ `rl_o1000_20k`：含自我對戰，違反 plan10 §1，經裁示保留

```
池內的 rl_o1000_20k ＝ 學習者起點 ＝ ai/checkpoints/rl_o1000_20k/step_000040.pt
```

依 §1 第 2 項標明：**含自我對戰**（plan10 §8 風險 1）。

**經使用者明確裁示保留，理由：它是實際遊戲池的對手。**

**三個具體後果（事實，非推測）：**

1. **round 41 的學習者與池內的 `rl_o1000_20k` 是同一份權重**
   （同一檔、sha256 `272b2edf…`）。該輪部分對局是**策略完全相同的鏡像對打**
   —— 兩個相同策略在同一局的期望報酬相同，`r = (5−排名) − 餘格/100`
   對兩邊對稱，**這部分對局的梯度資訊接近於零**。
2. **round 42 起兩者分離**（學習者更新、池席凍結），自我對戰仍在但不再鏡像。
3. **報告須標明「v2 可能適應自己的舊版」**：訓練指標混入了「打自己」，
   對外實力解讀須保守，且 v2 對池內那份凍結的起點可能有針對性。

**權重物件未共用**（§8 風險 10）：學習者由 `load_checkpoint` 載入、
池席由 `seats.build_brain` → `load_policy` 各自載入，兩份獨立物件；
訓練前後以 sha256 檢查（§7）。

### 2-2 `rl_i1000_50k`、`rl_b1000_50k`

> **「有預先註冊量測，四筆全部判為『無證據』，未取得正向驗證；
> 經使用者裁示納入。」**

出處 `eval/rl-multiple-train/big_eval/README.md`，「## 判準（預先註冊）」之下：

| 席位 | argmax（400 局，兩份排程） | softmax（400 局，兩份排程） |
|---|---|---|
| `rl_i1000_50k` | `+0.0340`（0.50 se）、`+0.1295`（1.82 se） | `+0.0811`（1.08 se）、`+0.1073`（1.53 se） |
| `rl_b1000_50k` | `−0.0384`（0.59 se）、`+0.0343`（0.53 se） | `+0.0991`（1.28 se）、`+0.1706`（2.20 se） |

四筆皆未過 2 se，判為「無證據」—— 與各自 20k **無顯著差異**，
**沒有任何一筆構成「已驗證」的正向結論**。

### 2-3 §1 第 3 項（下游版本）

**不違反。** `rl_o1000_50k`（o 的 50k）**不在池內**。
池內的 `rl_i1000_50k` / `rl_b1000_50k` 是 **i / b 的**下游版本，非 o 譜系。

### 2-4 其他相關風險（plan10 §8）

| # | 風險 | 本任務 |
|---|---|---|
| **1** | **自我對戰** | **適用**，見 2-1 |
| **2** | **對凍結池過擬合** | 11 席中 4 席靜態網路；**評測池是訓練池的子集且不含網路席位 → 過擬合偵測受限**，見 §5-2 |
| 4 | argmax 不受訓練目標控制 | 主判準正是 argmax |
| 5 | 多重比較 | 只跑**一條** |
| 6 | 事後選擇 | 判準、種子段、池已於本檔寫死 |
| 7 | 種子守衛 | 見 §3 |
| 8 | entropy 基準錯配 | 見 §6 |
| 10 | 權重物件不得共用 | 見 2-1、§7 |

---

## 3. 種子段（`reject_reserved_seeds` 已實跑驗過 FREE）

| 用途 | `seed_base` | 實際取用段 | 個數 | 三元組 |
|---|---|---|---|---|
| **訓練** rounds 41–100 | **7,530,000** | **7,550,000 – 7,579,999** | 30,000 | `("stage RL o50k v2 train", 7550000, 7579999)` |
| **評測** 1000 局 | 7,590,000 | **7,590,000 – 7,590,999** | 1,000 | `("stage RL o50k v2 eval", 7590000, 7590999)` |

算式核對：`seed_base + (41−1)×500 = 7,550,000`、
`seed_base + (100−1)×500 + 499 = 7,579,999` ✅

兩段**互不重疊**，與全部 20 個已註冊列零重疊；兩段都在既有列**之上**
（最高既有列 `stage RL i50k v2 eval` 止於 7,540,999），不動 7,1xx,xxx 那個 gap。

**`own` 與 `_claimed_train_block` 一致**：`seed_base` 是算術 anchor、不在任何
註冊列內，由 `TRAIN_ANCHOR_BLOCKS` 對到註冊列 → `own` 精確等於三元組 → 通過。
`stage RL students` 與既有 h / i 兩個 anchor 的行為**未變**
（`tests/test_rl_train.py::test_the_anchor_table_is_exactly_the_three_registered_runs`
把整張表釘成字面值，並斷言沒有任何 anchor 落在註冊列內）。

---

## 4. 主判準（預先寫死，跑後不改）

**評測對象**：訓練後的 `rl_o1000_50k_v2` vs 起點 `rl_o1000_20k`。

| 項 | 值 |
|---|---|
| 模式 | **argmax**（單一模式） |
| 局數 | **1000** |
| 種子段 | `stage RL o50k v2 eval` = 7,590,000–7,590,999（新註冊，FREE） |
| 配對 | 同 specs、同模式，current 對起點 |

### 通過條件

```
paired_diff > 0   且   paired_diff ≥ 1.0 × paired_se
```

| 項 | 值 |
|---|---|
| 單側假陽性率 | **約 15.9%**（P(Z > 1)，標準常態） |
| 1000 局下的門檻 | SE 估 **0.041–0.044**（400 局實測平均 0.0702 按 SE ∝ 1/√n 換算得 0.0444；本專案 1000 局 argmax 實測過 0.0413）→ 門檻約 **+0.041 ~ +0.044** |

### ⚠ 本判準覆寫 plan10 §5

> plan10 §5 類型 B「優於」原文寫死 `paired_diff ≥ 2 × paired_se`
> （單側假陽性約 2.3%）。
> **本次改為 `≥ 1.0 × paired_se`，係使用者於看過 §3 步驟 1 的唯讀回報後
> 明確裁示接受約 15.9% 的單側假陽性率。**
> 這不是 agent 自行放寬，也不是評測後調整 —— 本檔在任何一局開跑之前寫下，
> 之後不改。

### 判準選擇依據

**實際遊戲時 AI 使用 argmax**（`ui.py` 座席畫面、`match.py --mode` 預設）。
訓練 rollout 用 softmax（`rl/rollout.LEARNER_MODE`），所以 argmax 是訓練目標的
副產品而非其直接優化對象（§8 風險 4）。

### 結論措辭（預先寫死）

- **通過僅屬「弱證據」。報告不得寫「已證明更強」。**
- 未通過：**不入池、不開第二條**，結果照記。
- **只比起點 `rl_o1000_20k`**，不與其他版本比（含 `rl_o1000_50k`）。
- **softmax 只記錄，不入判準，不得作為入池理由。**
- 候選數固定為**一條**；重做是新決定，由使用者下，並註明是在看過結果後做的。

### 過線之後

**過線只是取得入池資格。** 與既有 `rl_o1000_50k` **並存或取代，
由使用者屆時裁示**，本檔不預設。

---

## 5. 評測設定

### 5-1 迴圈內評測（曲線，**不入判準**）

| 項 | 值 |
|---|---|
| 頻率 | `--eval-every 5` → rounds 45, 50, …, 100（12 次） |
| 局數 / 種子 | 100 局，`stage RL validation` = 7,100,000–7,100,999 |
| 模式 | argmax |
| **錨點** | **`--anchor ai/checkpoints/rl_o1000_20k/step_000040.pt`**（**不是**預設的 0k） |
| 輸出 | `data/rl_o50k_v2/eval.jsonl`，欄名 `current` / `hc_1000`（**歷史欄名**） |

> **這條曲線只是曲線，不入 §4 判準，不得作為入池理由。**
> 欄名 `hc_1000` 是 0k 退役前留下的歷史欄名，改名會把一份基線切成兩段；
> **本 run 該欄的實際內容是 `rl_o1000_20k`。**

### 5-2 驗收評測（**主判準**）

| 項 | 值 |
|---|---|
| 模式 | **argmax**（主判準）+ **softmax**（僅記錄） |
| 局數 | 各 **1000**，同 specs |
| 種子段 | `stage RL o50k v2 eval` = 7,590,000–7,590,999 |
| 配對 | `ai/checkpoints/rl_o1000_20k/step_000040.pt`（`sha256 272b2edf35e9…`） |
| `n_procs` | 評測前小批實測後決定（plan10 §3 步驟 4） |

**評測對手池 = 預設七人格**（`hunter, optimizer, builder, intruder, fox,
chess, wolf`）：

> ⚠ **它是訓練池的子集，且不含任何網路席位。**
> 訓練池 11 席 = 這 7 個人格 + 4 個網路席位；評測用前 7 席。
> 因此：
> - **§8 風險 2（對凍結池過擬合）在本評測下無法被偵測** ——
>   學習者對那 4 個網路席位（含它自己的凍結起點）的適應不會被這個測試發現；
> - 連「訓練沒見過的子集」都不是，因為訓練池**也**含全部 7 個人格。

---

## 6. Guardrail 規則（plan10 §6，跑後不改）

`rl_train` 的 guardrail 是**中止**不是告警：
順序為 **update → save → check → `exit 1`**，觸發那輪**會被存下來**
（`rl/rl_train.py` 模組 docstring 與迴圈；`main()` 對 `stopped_by` 回傳 1）。

| guardrail | 條件 | 本 run 的值 |
|---|---|---|
| `nan_or_inf` | 統計欄位非有限值 | — |
| `kl_anchor` | `kl > 0.05` | `kl_coef = 0` → `kl_anchor=None` → **永不觸發** |
| **`entropy`** | **`ent < 0.5 × entropy_reference`** | **`0.5 × 1.3728407901876114 = 0.6864203950938057`** |
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

> `entropy_reference = 1.3728407901876114`，取自 `rl_o1000_20k` 檔內，
> 是 **0k→20k 那段在「七個人格環境」下第 1 輪量到的 entropy**。
> **本次訓練環境是 11 席池（含 4 個網路席位，其中一個是學習者自己的凍結
> 起點），與來源環境不同。**
> 因此觸發線 **0.68642 可能與新環境不相稱**（plan10 §8 風險 8）。

**相關事實（不作結論）**：o 的 `entropy_reference`（1.3728）與 i 的（1.3949）
接近，都是四個 20k 裡偏低的一端（h 是 1.5294）。
**這不構成「會觸發」的預測**，只是記錄。

### 已知瑕疵：checkpoint 內的 `name` 欄

`rl_train.NAME` 是模組常數 `'rl_1000_20k'`，`save_checkpoint` 寫進每個檔。
**這是純資訊欄，席位名由 `ai/registry.json` 決定**；`build_brain` 走
`load_brain_at(..., key=...)`，載入不受影響。已裁示接受，不改碼。

### 已知的 dry-run 顯示問題（不改碼，僅提醒）

1. **`seeds` 行**顯示 `seed_base .. seed_base + games - 1`，
   即 `7,530,000..7,559,999`，**這不是實際取用的種子**。
   實際是 rounds 41..100 → **`7,550,000..7,579,999`**（見 §3）。
   resume 的 run 這行一律錯位，是既有顯示瑕疵。
2. **`anchor` 行**印的是模組常數 `ANCHOR_CHECKPOINT`（`data/hc2/step_001000.pt`），
   **不是**實際傳入的 `--anchor`（`_dry_run` 收不到 `anchor_path`）。
   實際接線已實測確認 `main()` → `train(anchor_path=args.anchor)` →
   `evaluate(..., anchor_path)`。

---

## 7. 凍結席 sha256 檢查（plan10 §3 步驟 3、§8 風險 10）

四個非人格席**檔案權限皆 `0o664`（owner 可寫）**，所以必須真的檢查，
不能靠唯讀權限保證。

| 席位 | 路徑（registry `checkpoint`） | sha256 |
|---|---|---|
| `rl_h1000_20k` | `ai/checkpoints/rl_h1000_20k/step_000040.pt` | `e1dc6230a70d205a89287468c47db6126717f0e9d42369bc1beca31404801636` |
| `rl_i1000_50k` | `ai/checkpoints/rl_i1000_50k/step_000100.pt` | `ef6ab92bbc11c3ec0c2f02945257a2ba435bfe9b295cb54bf5cb69ea847484a6` |
| `rl_b1000_50k` | `ai/checkpoints/rl_b1000_50k/step_000100.pt` | `ab6ec0a120667bd98a8f0352c7e35b5c6ac1ec17126a433050fdb93be92d09e3` |
| **`rl_o1000_20k`** | `ai/checkpoints/rl_o1000_20k/step_000040.pt` | `272b2edf35e9157ed137a33787ceb5fc736c62f785c304cb6abfa2a273c1ad90` |

**起點 = 池席 = 同一個檔**（§2-1），所以第四列同時是「起點」與「凍結席」。

### 檢查方案

```
階段 0（開訓前）  對四檔各算 sha256，比對上表逐字相同；不同則停
                 另跑 python -m ai --check
                 佈置 data/rl_o50k_v2/latest.pt，驗 sha256 = 272b2edf…
階段 1（開訓後）  再算一次，四個 sha256 與階段 0 逐位相等
                 任一不同 → 凍結席被寫 → 訓練作廢，停下等裁示
                 另跑 python -m ai --check
```

**唯讀的依據（已查碼）**：`rl/rl_train.py` 全檔唯一寫入是 `save_checkpoint`，
路徑一律來自 `checkpoint_paths(out_dir)` = `data/rl_o50k_v2/`；
`rl/rollout.py` 只有 `load_policy` / `torch.load`，無 `torch.save`。

---

## 8. 輸出與命令

### 輸出

| | 值 |
|---|---|
| 訓練目錄 | `data/rl_o50k_v2/`（新；`.gitignore:80` 的 `data/*` 已覆蓋） |
| 佈置 | `cp data/rl_o1000/step_000040.pt data/rl_o50k_v2/latest.pt`，**sha256 逐位相等後才啟動** |
| 產出 | `step_000041.pt` … `step_000100.pt` + `latest.pt` / `current.pt` / `rounds.jsonl` / `eval.jsonl` |
| 證據目錄 | `eval/rl-single-train-v2o/`（本檔所在） |
| **不碰** | `data/rl_o1000/`、`data/multi/`、`data/multi_rb80/`、`data/rl_i1000_v2/`、`data/rl_h50k_v2/`、`ai/checkpoints/` 內既有檔 |

### 起點檢查點

| 項 | 值 |
|---|---|
| 路徑 | `ai/checkpoints/rl_o1000_20k/step_000040.pt`（來源 `data/rl_o1000/step_000040.pt`） |
| sha256 | `272b2edf35e9157ed137a33787ceb5fc736c62f785c304cb6abfa2a273c1ad90`（實測 = registry） |
| `round` / `step` | 40 / 40 |
| **`entropy_reference`** | **1.3728407901876114** |
| **優化器** | ✅ AdamW，`state` 43 參數，**動量全部非零**；`ppo_cfg` 與 `rl_train` 預設逐欄相同 |
| `rng_state` | 真實 tuple；實測迴圈中只被存檔、從未使用 |
| 由 `rl_train` 產生 | ✅ `train_cfg.seed_base = 7200000`（學生跑，`stage RL students`） |
| `resume=True` 續 41–100 | ✅ 已端到端驗證（`find_resume_point`→40、optimizer 載入無缺漏、dry-run 40→100、本池 11 席解析正確） |

### 訓練命令

```bash
python -m rl.rl_train \
  --resume \
  --out data/rl_o50k_v2 \
  --games 30000 --per-round 500 \
  --seed-base 7530000 \
  --n-procs 8 \
  --eval-every 5 \
  --anchor ai/checkpoints/rl_o1000_20k/step_000040.pt \
  --opponent-pool "rl_b1000_50k,hunter,rl_o1000_20k,rl_i1000_50k,wolf,chess,fox,rl_h1000_20k,optimizer,builder,intruder"
```

### 時間預估（實測）

**開訓前小批實測**（拋棄種子 9,900,000，未落盤）：

| 項 | 條件 | 實測 |
|---|---|---|
| **rollout** | 500 局、本池 11 席、softmax、`n_procs=8` | **57.3 s（8.73 局/s）** |
| **update** | 取自 `data/rl_o1000/rounds.jsonl` 40 輪平均 | **33.29 s** |
| 每輪合計 | | **90.6 s** |

| | 計算 | 時間 |
|---|---|---|
| rollout | 60 × 57.3 s | 57 分 |
| update | 60 × 33.29 s | 33 分 |
| 迴圈內評測 | `eval_every=5` → 12 次 × 200 局，`n_procs=4` | ≈ 8 分 |
| **訓練合計** | | **≈ 98 分** |
| 驗收評測 | 1000 × 2 側 × 2 模式；上一 run（i）`n_procs=16` 實測 127.5 / 128.9 s | ≈ 4.3 分 |

> 評測前會**再做一次小批實測**再排程（plan10 §3 步驟 4）。

### 入池（僅在 §4 通過後；§7）

**資料**：`ai/checkpoints/rl_o1000_50k_v2/step_000100.pt`
（sha256 與來源相等後才進版控；`data/` 原件保持 gitignored）。

**改碼**：`ai/registry.json` 末尾一列、`seats.RL_SEATS` 一列、`config.I` 一個說明鍵、
`tests/test_registry_json.py`（白名單 +1、列序 15→16、三元組 15→16）、
`tests/test_match_pool.py`（**`GOLDEN_POOL` 15→16、`GOLDEN` 三個種子 20260928/7/99 全數重採**，
附舊→新對照與命令）。

**§7.6 不覆蓋既有席位** ✅：`rl_o1000_50k_v2` 的 key、路徑、列全部與
`rl_o1000_50k` 不同名。**取代或並存由使用者屆時裁示。**

---

## 9. 結果（跑完才補；判準見 §4，跑後未改）

（待訓練與評測完成後填入。）
