# 重構基準（refactor/main）

這份文件是 `refactor/main` 分支的基準事實，隨每一步更新。**動碼前先讀。**
任何一步的失敗集合若不等於「已知失敗」清單，停下並解釋。

## 分支與標記

| 項目 | 值 |
|---|---|
| 基準標記 | `pre-refactor-baseline`（`fce95b1`） |
| 工作分支 | `refactor/main` |
| 驗證環境 | `.venv-test`（CPU-only torch） |
| 全量指令 | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-test/bin/python -m pytest tests/ -q -n 8` |

> 三個線程變數不是可選的（見 `AGENTS.md` 第六節：少了它們 `-n 8` 會在
> `multiprocessing` fork 上死結）。

## 全量基準（截至 S6）

- **收集 1147** → 預期 **1144 passed / 2 skipped / 1 failed**。
- **唯一已知失敗**：`tests/test_rl_multi.py::test_the_frozen_weights_are_untouched_by_a_training_step`。
- **耗時波動診斷**：全量 wall time 在 311–375s 間跳動。`load average` 5.05/28 核屬未飽和，
  故 375s **不能只歸因於負載**。若再超時，**另查 CPU 頻率（降頻/熱節流）與記憶體壓力**，
  不只貼 `load average`。

### 已知失敗是**環境相依**的

該測試把模型移到 CUDA（`rl/policy.py` 的 `net.to(device)`，預設 `update_device`）。
在 CPU-only 的 `.venv-test` 下必然以 `AssertionError: Torch not compiled with CUDA
enabled` 失敗。**在有 GPU 的環境它可能轉綠——轉綠不算基準漂移**，不需停下解釋；
只有「環境相依以外的原因」造成的失敗集合變動才要停下。

（前一項已知失敗 `tests/test_seat_rename.py::test_records_json_names_only_the_new_key`
已於 S1 消除：它原本讀本機 runtime `records.json`，改為讀 tmp 檔並驗證寫入端正規化。）

### 已知負載敏感時序測試（flake）

`tests/test_trace.py::test_trace_costs_nothing_when_not_requested` 比較「不開 trace」與
「開 trace」兩次整局計時（`off <= on * 1.15 + 0.05`）。它**未登記於 `AGENTS.md` 原 5 處
計時測試**，在全量滿載（8 worker）下偶發變紅；隔離序列與單檔 `-n 8` 皆穩定綠。

**判定規則**：全量出現此項失敗時，允許「隔離重跑一次 ＋ 全量重跑一次」，**兩者皆綠
才算通過**，並在回報中註明；隔離跑也紅則視為真失敗。**不放寬斷言係數**（不授權改
測試）。若再出現，評估是否排除出快速子集或補登 `AGENTS.md`。

## 快速子集（每步使用，階段 3）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-test/bin/python -m pytest \
  tests/test_golden.py tests/test_rng_pins.py tests/test_setup_seats_rng.py \
  tests/test_wall_budget_branch.py \
  tests/test_match_pool.py tests/test_match_paired_rng.py tests/test_seats.py \
  tests/test_registry_json.py \
  -q -n 8
```

- 保護對象：`test_golden`（對局序列）、`test_rng_pins`（`make_profile` 值、
  `personality_keys` 順序）、`test_setup_seats_rng`（`setup_seats` 的 rng 消耗順序）、
  `test_match_pool`／`test_match_paired_rng`／`test_seats`／`test_registry_json`
  （池順序、抽籤、席位）。
- **測試分工**：`test_setup_seats_rng` 的 stub 消耗 0 次 rng，所以它只釘「席位編排的
  消耗順序與次數」；人格腦**內部**的 7 次抽取由 `test_rng_pins`（值）與 `test_golden`
  （對局）負責，三者互不重疊。
- **判定方式：CPU 時間為準**。wall time 受機器負載影響達 36%（實測 39–53s），55s wall
  線已失去意義。改以 `time` 的 `user+sys` 為硬線＝**已錄基準 × 1.25**；已錄基準：
  **8 檔 241s、7 檔 237s**（CPU 時間對其他行程負載不敏感，波動約 6%）。另保留 **90s wall**
  作粗檢，只擋暴增。新增進子集的測試須同步量 CPU 時間。
- `test_ui_smoke.py`（約 32s）**不**入每步子集；僅在該步改動到 `ui.py` 時額外跑。

## 黃金基準

`tests/golden/`：7 個 case、533 plies，逐手記錄（`turn_order`／`colors`／`owner_key`／
`[owner,name,oi,x,y]`／`remaining`／`state`）。產生時的 HEAD 與時間在 `golden.json` 的
`meta`。`replay` 期間關閉 `USE_WALL_BUDGET`（否則受機器負載影響）。**不讀寫
`records.json` 或 `data/`。**

## 突變驗證結果（階段 1）

四項突變（`make_profile` 多抽一次、`personality_keys` 換序、`draw_turn_order` 少抽一次、
`pool_order` 改序）**全部被既有測試抓到**。

| 突變 | 新測試（golden／pins）抓到？ |
|---|---|
| `make_profile` 抽樣 | golden ✅、pins ✅ |
| `personality_keys` 順序 | golden ✅、pins ✅ |
| `draw_turn_order` | golden ✅、pins ✗ |
| `pool_order` | **golden ✗、pins ✗**（只由既有池測試擋下） |

**已知覆蓋分層**：`pool_order` 的改動由 `test_match_pool`／`test_seats`／
`test_registry_json` 保護；golden 的職責是釘對局序列，不是釘池順序。**不補測試。**

**觀察**：rng 消耗順序類的改動會同時讓約 19 條測試變紅；階段 3 若出現這個量級的紅，
先查 rng 順序。

## 盲區（已知，尚未處理）

- **`USE_WALL_BUDGET` 觸發分支**：已由 `tests/test_wall_budget_branch.py`（S6）覆蓋。
  以 `WALL_BUDGET = 0` 使 `elapsed < 0` 恆假而**確定性**觸發跳過分支，不依賴真實時鐘；
  該測試 docstring 明寫「只證明分支可達，不證明真實負載行為」。
- **`test_setup_seats_rng` 對「插入一次小額抽取」在現有 seed 序列上未偵測到**：它用
  `getstate()` 相等釘住消耗，對**順序**改變敏感，但對插入一次 `randrange`，在**現有
  seed 序列上未偵測到**（盲區取決於序列，換一組 seed 可能偵測到）。原因已由實測確認：
  `randrange(4)` 走拒絕取樣（`_randbelow`），每次消耗的 MT 字數**可變（1–2）**；插入的
  那次讓最後一次少消耗 1 格、總字數相抵 → `getstate()` 相等。**不是 `getstate` 不敏感**
  （比較方式就是直接 `getstate()` 相等），而是「抽取次數」≠「消耗字數」。次數類保護由
  `test_rng_pins`（值）與 `test_golden`（對局）負責；若要補強（**另案**，不在 S6），以
  計數代理包住 rng、直接斷言抽取次數。
- **黃金基準無 rl／imitation 席位**：網路席位的 rng 消耗不在其內；由
  `tests/test_setup_seats_rng.py` 的 stub 補上。
- **`Board._update_regions` 不優化**（裁決）。日後若要優化，先量「每手 `choose_move`
  內 `place`/`unplace` 呼叫次數」，且不寫成測試斷言。
- **`test_board.py` 的 `test_full_hand_fills_each_quadrant` 覆蓋略窄**：`:120` 的斷言
  由「該 owner 的棋子集合非空」改為「`owner_bits[o]` 非零」（S3d，因 `all_owner_cells`
  已刪）。它因此不再同時驗證「棋子集合內容」與「位元遮罩」一致；該一致性改由
  `test_board` 其他測試與 `test_golden` 逐手比對保護。

## 階段進度

| 步驟 | 內容 | 狀態 |
|---|---|---|
| S1 | `test_seat_rename` 改讀 tmp `records.json` | 完成 |
| S2 | `setup_seats` 的 stub brain rng 消耗釘子測試 | 完成 |
| S3 | 死碼移除（`config.piece`／`Hand.all`／`Board.owner`／`Board.all_owner_cells`） | 完成 |
| S4 | `Game.setup_seats` 加 `brain_factory`（預設 `None` 哨兵） | 完成 |
| S6 | `USE_WALL_BUDGET` 觸發分支覆蓋（docstring 註明限制） | 完成 |
| S7 | 魔術數字 `4` → 具名常數（只改確定為座位數者；S7a–S7i 全部完成） | 完成 |
| S9 | UI 接縫 1（`Layout` 抽出） | 待授權 |

S5（`resolve_seats` 拆分）與 S8（分層倒置）**不執行**（裁決；S8 只記錄）。

## S7g 量測基線（S7f 之後、S7g 之前）

S7g 會改 `ai/formulas.py` 的 `board_feats`（`:620/:629/:632/:640/:646/:647`）與
`:416`。以下為**改動前**基線，S7g 後以**同條件**重跑比對。

- **機器負載（`uptime`）**：`load average: 9.95, 8.59, 6.13`（up 22:09）。
  這解釋全量 wall 312 → 333 → 365s 的逐次上升：`-n 8` 的八個 worker 在滿載
  機器上互相競爭，wall 因此不穩；**判準仍以快速子集 CPU 時間為準**（該值不受
  並行負載影響）。
- **`bench_engine.py` f4b（end-to-end `choose_move`）**：
  `--sections f4b --f4b-games 100 --f4b-samples 200`，wall 35.59s、
  CPU 35.18+0.40s。`_measurement_decision_ms`（每決策 ms）：

  | 人格 | mean | median | p95 | max |
  | --- | --- | --- | --- | --- |
  | chess | 9.2951 | 9.4892 | 12.2423 | 15.4079 |
  | fox | 9.0381 | 9.4443 | 11.2852 | 23.3211 |
  | wolf | 9.9103 | 10.1105 | 13.3725 | 18.6103 |

  （`optimizer` 不做 lookahead，bench 不記其決策時間。）此數字是 wall-clock、
  受機器負載影響，只作旁證。
- **`board_feats` 直接微基準**（S7g 真正改到的函式）：961 個 midgame grid、
  每輪 3000 次、取三輪最佳 → **0.8818 ms/call**。CPU-bound，較不受負載影響，
  是 S7g 前後對比的**主要**數字。
- **工具注意**：`bench_engine.py` 在 `--sections` 非 `all`（缺任一 section）時
  會**提前 return**，只把 JSON 印到 stdout、**不寫** `--json` 指定的檔。要留存
  數字請導向 stdout 再解析。

## S7g 結果（`ai/formulas.py`）

判準：**單一行程交錯 A/B** —— 兩版本（`git worktree` 檢出的對照版與工作樹）各
載入一份 `ai.formulas`，按 A、B、A、B 輪流跑 6 輪，各取最小值再比。961 個
midgame grid、每輪 3000 次呼叫。

- **S7g-1（只換字面值，`b4a15b5`）**：A(min 0.9209 / median 0.9501) vs
  B(min 0.9116 / median 0.9384) → **ratio min 0.9899、ratio median 0.9877**，
  快約 1%，無退步。已提交。
- **S7g-2（`:646` 改 `sum(nc[n])`）**：**否決，未提交**。兩次交錯測量
  ratio min **1.0386 / 1.0498**、ratio median **1.0600 / 1.0701**，一致退步
  **4–7%**，超過 3% 門檻 → 還原為原展開式，並在 `:646` 上方加一行註解說明
  （`4cad870`）。
- **`bench_engine.py` f4b 的單次 vs 交錯**：單次前後對比看似 +6%（chess/fox），
  但**交錯** A/B（pre-S7g vs S7g-1，各 3 輪）最小值比值為 chess 1.013、
  fox 1.001、wolf 1.015 —— 全在 1.5% 內。差異來自機器負載（單次對比時
  `load average` 10.11 vs 較低時），**單次前後對比在這台機器上不可靠**。

**方法論**：快速子集 CPU 時間無法交錯，只作粗略安全線，**不作回歸證據**；
真實回歸判定一律用單一行程交錯 A/B（≥5 輪取最小）。S7 至今沒有任何回歸疑慮
被交錯測量證實。

## S7d 結果（`engine.py`，含 `config.py` 的 assert）

- **`config.py`（`6e4a958`）**：新增 `assert len(CLOCKWISE_OWNERS) == SEATS`。
  這把 `SEATS` 綁死在四角幾何上：把 `SEATS` 改 5 會在 **import 就** `AssertionError`。
- **`engine.py`（`07f331e`）**：12 處座位數改用 `SEATS` —— `:190` turn_order 檢查、
  `:193`/`:194`/`:197` State 建構、`:213` `result`、`:332`/`:334` `_advance`、
  `:409`/`:418` `deserialize`、`:557` `normalize`、`:650`/`:651` `hand_vectors`；
  並在 `:17` 的 config import 加 `SEATS`。
- **保留（+ 一行註解）**：`:70` `empty_of` 的四項 OR（每手熱路徑）；`rotate_state`
  `:505-508` 與 `normalize` 的 `moved = list(range(4))`（`ec6de10` 補註解）——
  同屬 `normalize`/`rotate_state` 家族、在每步 observation 路徑上，改成 `SEATS`
  收益為零。保留處註解皆寫明「4 由四角 cycle 決定，`config.py` 有 assert」。
  訊息 `:191`（"permutation of 0..3"）／`:410`（"four players"）與 docstring 未動。
- **反證**：用 runtime patch `engine.SEATS = 5`（**不可**用 `config.SEATS=5`，會被
  新 assert 擋在 import）→ `initial_state`/`result`/`serialize`/`deserialize` 長度
  變 5；`initial_state()` 配預設 4-cycle 時 `ValueError`；`_advance`/`hand_vectors`
  撞到保留的 `OWNER_CORNER`/`ROT_OWNER` 四角而 `KeyError`。**僅驗證常數被接上，
  不代表支援 5 席。**
- **效能（`bench_engine.py` f1 交錯 A/B，`--games 60 --func-games 60`，各 5 輪取
  最小 median；A＝worktree `6e4a958`、B＝工作樹）**：10 個函式最大比值
  **1.0075**（`has_any_legal`，+0.75%），最小 0.9782（`legal_moves`）；無退步。

  | function | A | B | ratio |
  | --- | --- | --- | --- |
  | apply_move | 0.0496 | 0.0499 | 1.0053 |
  | deserialize | 0.0292 | 0.0293 | 1.0042 |
  | hand_vectors | 0.1757 | 0.1752 | 0.9974 |
  | has_any_legal | 0.0081 | 0.0081 | 1.0075 |
  | legal_move_mask | 0.3886 | 0.3828 | 0.9850 |
  | legal_moves | 0.4940 | 0.4798 | 0.9782 |
  | normalize | 0.1472 | 0.1459 | 0.9907 |
  | pass_turn | 0.0397 | 0.0398 | 1.0042 |
  | serialize | 0.0292 | 0.0291 | 0.9935 |
  | to_plane | 0.2413 | 0.2377 | 0.9852 |
  | [game median] | 0.0459 | 0.0452 | 0.9859 |



