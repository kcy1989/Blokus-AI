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

## 全量基準（截至 S2）

- **收集 1144** → 預期 **1141 passed / 2 skipped / 1 failed**。
- **唯一已知失敗**：`tests/test_rl_multi.py::test_the_frozen_weights_are_untouched_by_a_training_step`。

### 已知失敗是**環境相依**的

該測試把模型移到 CUDA（`rl/policy.py` 的 `net.to(device)`，預設 `update_device`）。
在 CPU-only 的 `.venv-test` 下必然以 `AssertionError: Torch not compiled with CUDA
enabled` 失敗。**在有 GPU 的環境它可能轉綠——轉綠不算基準漂移**，不需停下解釋；
只有「非這兩類（環境相依）原因造成的失敗集合變動」才要停下。

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
- **預算 60 秒**（硬線）；餘裕有限，新增進子集的測試須同步量耗時。**超過 55s 即回報**
  取捨（機器波動約 5 秒）。
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

- **`USE_WALL_BUDGET` 觸發分支無覆蓋**：`ai/chooser.py` 的
  `elapsed >= WALL_BUDGET → 跳過 lookahead` 沒有任何測試；黃金基準刻意關閉該預算。
  排程：S6（`WALL_BUDGET=0` 逼出分支；docstring 須註明「只證明分支可達，不證明真實
  負載行為」）。
- **黃金基準無 rl／imitation 席位**：網路席位的 rng 消耗不在其內；由
  `tests/test_setup_seats_rng.py` 的 stub 補上。
- **`Board._update_regions` 不優化**（裁決）。日後若要優化，先量「每手 `choose_move`
  內 `place`/`unplace` 呼叫次數」，且不寫成測試斷言。

## 階段進度

| 步驟 | 內容 | 狀態 |
|---|---|---|
| S1 | `test_seat_rename` 改讀 tmp `records.json` | 完成 |
| S2 | `setup_seats` 的 stub brain rng 消耗釘子測試 | 完成 |
| S3 | 死碼移除（`config.piece`／`Hand.all`／`Board.owner`／`Board.all_owner_cells`） | 完成 |
| S4 | `Game.setup_seats` 加 `brain_factory`（預設 `None` 哨兵） | 待授權 |
| S6 | `USE_WALL_BUDGET` 觸發分支覆蓋（docstring 註明限制） | 待授權 |
| S7 | 魔術數字 `4` → 具名常數（只改確定為座位數者） | 待授權 |
| S9 | UI 接縫 1（`Layout` 抽出） | 待授權 |

S5（`resolve_seats` 拆分）與 S8（分層倒置）**不執行**（裁決；S8 只記錄）。
