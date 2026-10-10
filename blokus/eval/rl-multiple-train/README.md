# eval/rl-multiple-train — plan10 任務二的設計與證據

本目錄收 plan10 **任務二**（四學習者同場訓練）的設計記錄，以及每個里程碑的
日誌快照。快照寫進 `step_<n>/`，**不覆寫**；重跑一律寫新目錄。

訓練權重在 `data/multi/{h,o,b,i}/`（gitignored）。本目錄只放證據。

---

## 1. 一「步」是什麼

| 項目 | 規格 |
| --- | --- |
| 同場局 | **240 局** = 4! = 24 種座位排列 × 10 輪，**每輪一個種子、24 種排列共用** |
| 固定池局 | **260 局**，有放回抽 3 個對手；每一局由**四個學習者各玩一次，坐同一個座位** |
| 學習者軌跡 | 同場 240×4 = 960，固定池 260×4 = 1040，合計 **2000** |
| 每個學習者 | 240 + 260 = **500 局**（與舊的單打 500/輪相同） |
| 更新 | 每個學習者各自 `ppo_update`，自己的 AdamW，互相不共用參數 |
| 獎勵 | `r = (5 − 排名) − 餘格/100`（沿用 `rl.policy.episode_reward`） |

「每輪一個種子、24 種排列共用」是**設計**，不是省略：四席全是神經網路時
`Game.setup_seats` 只抽顏色與先手，不再抽人格權重，所以一個種子 = 一套設定，
24 局只差「誰坐在哪裡」——這正是「4! 排列」存在的理由。
plan10 第 5 節風險 9 已把代價寫下：**每步種子數只有 10 組**。

## 2. 兩條曲線，以及哪一條才是進步的證據

`rounds.jsonl` 每步記三個數（`curves` 欄）：

| 欄位 | 意思 |
| --- | --- |
| `table` | 同場局 960 條軌跡的平均獎勵 |
| `fixed` | 固定池局 1040 條軌跡的平均獎勵 |
| `merged` | `(table + fixed) / 2`，plan10「兩種局各算平均獎勵，相加後除以 2」 |

**`merged` 只是回報數字，不是梯度權重。** PPO 的批次就是該學習者 500 條軌跡的
直線串接，沒有任何一條被加權。plan10 的表格「每條同場軌跡權重 1/1920、固定池
1/2080」描述的是這個**統計量**的隱含權重（0.5/960、0.5/1040），比值 13:12，
不是 loss 的權重——plan10 沒有做第二個決定，這裡也不替它做。

**能跨時間比較的只有 `eval.jsonl`。** plan10 風險 1 說動態同伴曲線對手隨訓練
改變、上升不代表變強；固定池曲線的對手雖凍結，訓練種子卻每步不同。
所以每 `eval_every` 步另跑一次 `evaluate()`：

- 固定種子集（`EVAL_SEED_BASE` = 7,400,000 起，一次定案，整場訓練不變）
- 固定池（`FIXED_POOL` 的 11 席）
- `argmax`
- 每個學習者的**當前權重**與**它自己的凍結 20k** 各跑一次，同一批種子

回報 `paired_diff` 與 `paired_se`（成對差的標準誤）。兩列共享每一次抽籤，
剩下的噪聲就是兩個策略本身的差異。**這是判斷進步的那條曲線。**

## 3. 固定池與四個學習者

```python
FIXED_POOL = ("builder", "chess", "fox", "hunter", "intruder", "optimizer",
              "rl_b1000_20k", "rl_h1000_20k", "rl_i1000_20k",
              "rl_o1000_20k", "wolf")           # 7 人格 + 4 凍結 20k，依 key 排序
LEARNER_KEYS = ("rl_h1000_20k", "rl_o1000_20k", "rl_b1000_20k", "rl_i1000_20k")
```

- 池是**寫死**的，不從 `seats.automated_options()` 推導：池是一個決定，從名册推導
  等於讓未來的名册變動悄悄走進訓練。
- **訓練期間池不擴張**（plan10 §6b 第 3 點）：固定池曲線要能跨里程比較。
- **有放回**：同桌可能有兩個相同的席位，學習者也可能抽中自己的凍結 20k 當對手。
  plan10 第 5 節風險 2 已聲明：固定池曲線是「對一組固定對手的表現」，不是純對外實力。
- 座位：第 `i` 局坐 `i % 4`，四個學習者各坐 65 次。

## 4. 權重隔離（plan10 風險 7）

| 事實 | 出處 |
| --- | --- |
| `ImitationBrain` 每次都 `build_model` 再 `load_state_dict`，是**新模組** | `rl/imitation.py` |
| `_BLOB_CACHE` 快取的是**原始 dict**，不是模組 | `rl/imitation.py::_load_blob` |
| 學習者的權重由 `load_policy(current.pt)` 獨立載入 | `rl/multi_train.py` |
| 每步重新算四個凍結 20k 的 sha256，不同就 `RuntimeError` | `rl/multi_train.py::train` |
| 兩個席位吃同一個檔案會拿到兩個模組、改一個不影響另一個 | `tests/test_rl_multi.py` |

`seat_for` 的舊後備（`rl_h1000_0k`）在 plan9 任務四離開名册，所以四個學習者
一律用**自己的 20k 註冊 key** 當替身，`Game.setup_seats` 先建出凍結席，再被換成
當前權重。若沒換，學習者會拿自己的凍結權重訓練而一切看起來正常——
`test_the_learner_seat_plays_the_handed_network_not_the_registry_weights` 釘住這點。

## 5. 種子

三個新區塊，已註冊進 `rl/paired3.RESERVED_RANGES`：

| 區塊 | 範圍 | 用法 |
| --- | --- | --- |
| `stage RL multiple table` | 7,300,000–7,300,999 | 每步 10 個同場種子 |
| `stage RL multiple fixed` | 7,310,000–7,335,999 | 每步 260 個固定池種子 |
| `stage RL multiple validation` | 7,400,000–7,400,999 | 評測固定種子集 |

種子是 `(step, index)` 的**純函式**，不帶游標：`make_table_specs(41)` 不必重播
40 步。區塊覆蓋 100 步 = 150k，plan10 全程。

## 6. 存檔與重現

- `data/multi/<learner>/current.pt` — 每步兩次：更新前（給 rollout 讀）與更新後
  （給下次 rollout 與 `evaluate()` 讀）。
- `data/multi/<learner>/latest.pt` — 每步。`--resume` 讀它，四個學習者的步數必須
  一致，否則拒絕起跑。
- `data/multi/<learner>/step_%06d.pt` — **每 20 步**（plan10「存檔每 10k 局」）。
  編號是**全域步數**，所以 `step_000100.pt` 就是 50k。
- `eval/rl-multiple-train/step_<n>/` — 里程碑當下三份日誌的快照
  （`rounds.jsonl` / `eval.jsonl` / `milestones.jsonl`），不覆寫。
- `milestones.jsonl` 每個存檔點記四個檔的 **sha256**、當時的命令列，以及評測的
  重放配方。

### 重跑

```bash
# 一整步（240 同場 + 260 固定池 x4 學習者）
.venv-rl/bin/python -m rl.multi_train --dry            # 只印計畫
.venv-rl/bin/python -m rl.multi_train                  # 41..100
.venv-rl/bin/python -m rl.multi_train --resume         # 從 latest.pt 續跑

# 只重算評測曲線（固定種子集、固定池、argmax、與凍結 20k 成對）
.venv-rl/bin/python -c "from rl import multi_train as mt; print(mt.evaluate(mt.MultiConfig(), 'data/multi'))"
```

---

## 7. 前置回報七項的結論（plan10 任務二）

| # | 項目 | 結論 |
| --- | --- | --- |
| 1 | 現有程式碼如何改為四學習者同場 | `play_episode` 拆成 `play_game(keys, seat_nets, seed)`；一局回傳 `{seat: Episode}`。`play_table_batch` 一局四份、`play_paired_batch` 一份規格玩四次 |
| 2 | 座位排列與種子生成 | 24 排列寫死為 `itertools.permutations`；每輪一個種子、24 排列共用（見 §1） |
| 3 | 固定池抽樣與成對重放 | 有放回抽 3、座位 `i%4`、同種子四次；`play_paired_batch` 一個行程池跑完四個學習者 |
| 4 | 兩條曲線的格式 | `curves.{table,fixed,merged}` + `per_learner.*.curves`；可比較的那條在 `eval.jsonl` |
| 5 | 存檔命名 | `data/multi/<learner>/step_<全域步數>.pt`，每 20 步 |
| 6 | 凍結席與學習者權重是否隔離 | **隔離**（見 §4），另加每步 sha256 檢查 |
| 7 | `hc_2000` / `hc_10000` 的權重位置 | `data/hc2/step_002000.pt` / `step_010000.pt`，**在版控內**（`.gitignore` 反白），plan9 任務四起**不是任何席位的載入路徑**。不得刪除 |

---

## 8. 已知限制（不要把「測試全綠」讀成別的）

1. **同場局的 24 局共享一個種子**，所以它們不是 24 個獨立樣本。統計時以**局**為
   單位（plan10 風險 6），而且同一輪內的局互相耦合。
2. **同場局不是零和**：平手共享較好名次，同桌獎勵總和不固定（plan10 風險 3）。
3. **固定池含學習者自己的凍結 20k**，固定池曲線不是純對外實力（plan10 風險 2）。
4. **同一條同場軌跡的權重是固定池軌跡的約 1.08 倍**（1/1920 對 1/2080），
   且同場局環境非平穩，梯度雜訊占比偏高（plan10 風險 8）。
5. **不設自動停止判準**（plan10 §4）。看曲線再決定要不要繼續，會讓選擇點偏向
   有利結果；分析時應先固定指標與種子集。
