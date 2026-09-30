# Blokus AI 訓練基礎建設：Copilot 工作清單

## 0. 背景與總則

本專案是 20x20、四人 Blokus，棋盤用 400 位元整數表示（bit index = `y*B + x`，左上為 bit 0）。
目標是為日後的 MCTS 加神經網絡訓練打好基礎。現有六個 AI 人格（wolf、chess、fox、intruder、optimizer、builder）將作為「老師」和對手池，**其棋力和選步行為不可改變**。

### 硬性規則（每個任務都必須遵守）

1. **先寫測試，再改程式。** 每個任務的驗收標準都必須有對應測試。
2. **行為保持不變的重構，必須用「同種子逐手比對」驗證。** 舊實作與新實作用相同亂數種子跑至少 1000 局，每一手的選步（`name, oi, x, y`）必須完全一致。任何差異都視為 bug，不可用「差不多」放行。
3. **不要修改 `ui.py`。**
4. **不要改變枚舉順序。** `choose_move` 的同分排序依賴 Python 穩定排序，順序為：棋子（`Hand.names` 字母序）→ 方向 `oi` 由小到大 → 落點 `base` 由小到大。
5. **不要猜測。** 若需要某個數字（例如 action 總數），用程式計算，不要寫死。
6. 每個任務完成後，跑完整測試套件，並在提交訊息說明改了什麼、驗證了什麼。
7. 不確定的地方，停下來提問，不要自行假設規則。

### 已確認的規則（新程式碼必須照做）

| 項目 | 規則 |
|---|---|
| 首步 | `placed[owner] == 0` 時，棋子必須覆蓋自己的起始角（`OWNER_CORNER[owner]`） |
| 後續步 | 必須與自己的棋子角接觸，且不可邊接觸（`Reach(need, avoid)`） |
| Pass | 自動判定，不是玩家的行動 |
| 終局 | 四位玩家都沒有合法步 |
| 計分 | 剩餘格數，越少越好。**沒有**完成加分 |
| 輪次 | 順時針循環，起點隨機。`CLOCKWISE_OWNERS = (1, 2, 0, 3)` |
| 棋子 | 21 種，總格數 89。方向總數應為 91（1+2+6+19+63），需用程式驗證 |

---

## 階段 A：修正 bug 與凍結資料格式（最優先） — ✅ 已完成

> **狀態：** 已完成並提交（commit `4387934`）。165 項測試通過。
> 實測結果與本文假設不符之處見下方 A1 的「實測補充」。

### A1. 驗證並修正 `_adjoining_bases` 的首步分支 — ✅ 已完成

**檔案：** `ai/formulas.py`

**疑似 bug：**

```python
if cbit:
    hit = 0
    for o in od["offs"]:
        hit |= od["m"] & (cbit >> o)      # 疑似錯誤
    return hit & od["valid"]
```

`cbit >> o` 的第 b 位表示「base 為 b 時，第 o 個偏移格恰好落在角上」，這是 **base 的集合**。`od["m"]` 是棋子在 base 0 的形狀遮罩，兩者意義不同。只有左上角（owner 1）會碰巧正確，其餘三個角（owner 0、2、3）多半得 0。

**步驟：**

1. **先寫測試，確認 bug 存在。** 不要假設它一定錯：

   ```python
   from board import ALL
   import ai.formulas as F
   from config import B, OWNER_CORNER

   def brute(od, cbit):
       out = 0
       for b in od["bases"]:
           if (od["m"] << b) & cbit:
               out |= 1 << b
       return out

   def test_first_move_bases_all_corners():
       for owner, (cx, cy) in OWNER_CORNER.items():
           cbit = 1 << (cx + cy * B)
           for name, oris in F.ODIRS.items():
               for oi, od in oris.items():
                   assert F._legal_bases(od, ALL, None, cbit) == brute(od, cbit), (owner, name, oi)
   ```

2. 若測試通過，代表我的判斷有誤，**停下來回報，不要修改**。
3. 若測試失敗，把 `hit |= od["m"] & (cbit >> o)` 改成 `hit |= cbit >> o`（最後仍然 `& od["valid"]`），重跑測試直到通過。
4. 再加一個測試：在有其他玩家棋子佔用部分格的局面上，`_legal_bases`（帶 `empt`）與暴力法一致。

**驗收：** 四個角、所有 91 個方向的首步合法落點與暴力法完全一致。

**注意：** 這是 A 之後所有「改用 `_legal_bases`」任務的前提。在此修好之前，**不要**做 C 階段的任何任務，否則 owner 0、2、3 的首步可能被誤判為無合法步並自動 pass。

#### 實測補充（A1 執行後）

| 項目 | 實測結果 |
|---|---|
| bug 是否存在 | **是**。owner 1（左上）91/91 正確（巧合），owner 0/2/3 各 **58/91 方向錯誤** |
| 錯誤結果 | 合法落點 **0** 個，正確應為 58 個 |
| 同種子逐手比對 1000 局 | **1000/1000 完全一致**，棋力與選步行為零改變 |
| 本文「會自動 pass」的預測 | **不成立**。`choose_move` 用直接檢查 `(empt & shifted)` / `shifted & cbit`，`Game.has_legal` 用 `board.has_legal_move`，兩者都不經過此分支 |
| 實際受影響的呼叫點 | 只有 `_opponent_pool` 的後備掃描與 `placement_counts`；抽樣 1200 個首步位置，對手預判池總數新舊完全相同（42876） |
| C2 的必要性 | **仍然成立且是硬前提**。用舊分支把 `has_legal` 改寫到 `_legal_bases`，owner 0/2/3 首步會回傳 `False`，四人中有三人自動跳過 |

修正後新增 6 項測試於 `tests/test_geometry_reference.py`，其中
`test_first_move_branch_is_not_vacuous` 專門防止「兩邊都回傳空遮罩所以看起來通過」。

### A2. 凍結動作表 — ✅ 已完成

**新增檔案：** `tools/freeze_actions.py`，輸出 `action_table.json`。

```python
import json, hashlib
from pieces import MASTER

names = sorted(MASTER)                       # 棋子位元編號 0..20
table = [(n, oi) for n in names
         for oi in range(len(MASTER[n]["orientations"]))]

assert len(names) == 21
assert len(table) == 91
assert sum(MASTER[n]["size"] for n in names) == 89

blob = json.dumps({"pieces": names, "actions": table})
digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
json.dump({"pieces": names, "actions": table, "hash": digest},
          open("action_table.json", "w"))
print(digest)
```

額外要求：

- 另外輸出 `total_placements = sum(od["valid"].bit_count() for each ODIRS entry)`，寫入 JSON 並在終端顯示。
- 新增測試 `test_action_table_frozen`：重新計算雜湊，與 `action_table.json` 裏的比對，不同就失敗。
- 測試方向已正規化：每個方向的 `min x == 0` 且 `min y == 0`。

**驗收：** 測試通過；回報 `hash`、方向總數、`total_placements`。

#### 實測結果（A2）

```
pieces           21
orientations     91      （1+2+6+19+63，與本文預測一致）
cells            89
total_placements 30433
hash             91252a354b090d43
```

實作補充：

- `total_placements` 用**兩種獨立算法**計算（由 `MASTER` 幾何直接算 vs 由 `ODIRS["valid"]`
  算），測試釘住兩者必須一致，避免定義漂移。
- `freeze_actions.py` 只 import `hashlib/json/os/sys` 與 `pieces/config`，**不碰 pygame
  與 AI**，且可從任何目錄執行。
- JSON 採「一行一個 action」的排版，方便日後 diff；雜湊取自 `digest_of` 而非檔案
  位元組，所以排版可自由修改而不影響已發布資料。

---

## 階段 B：量測（在動手優化之前） — ✅ 已完成

> **狀態：** 已完成。**未修改任何遊戲邏輯**（B 階段規定只量測）。
> 新增的只有量測工具 `tools/bench_profiles.py`，它從外部包裝 `choose_move`，
> 引擎本身完全沒有被改動。

### B1. 跑 profile 並回報 — ✅ 已完成

```bash
python3 -m cProfile -s cumtime match.py --games 3 --dry | head -60
```

**每局耗時：**

| 量測方式 | 樣本 | 每局 |
|---|---|---|
| 無 profiler（真實速度） | 20 局 | **0.84 秒** |
| cProfile（有測試開銷） | 8 局 | 1.11 秒 |

**累計時間前 15 名**（cProfile，8 局 = 8.97 秒，已濾掉 import 雜訊）：

| # | 函式 | 呼叫次數 | cumtime | 佔比 |
|---|---|---|---|---|
| 1 | `chooser.py:126 choose_move` | 616 | 8.451 | **94%** |
| 2 | `builder.py:84 rescore` | 93 | 2.602 | 29% |
| 3 | `builder.py:103 _score` | 21,845 | 2.446 | 27% |
| 4 | `formulas.py:521 fill_components` | 21,845 | 1.544 | 17% |
| 5 | `formulas.py:676 _score_move` | 522,742 | 1.431 | 16% |
| 6 | `board.py:29 dilate` | **1,414,902** | 1.316 | 15% |
| 7 | `formulas.py:603 board_feats` | 616 | 0.574 | 6% |
| 8 | `formulas.py:494 fillable_closure` | 21,845 | 0.559 | 6% |
| 9 | `chooser.py:19 _opponent_pool` | 672 | 0.394 | 4% |
| 10 | `intruder.py:73 rescore` | 89 | 0.392 | 4% |
| 11 | `formulas.py:545 pack_lost` | 21,845 | 0.128 | 1% |
| 12 | `formulas.py:448 place_state` | 66,600 | 0.323 | 4% |
| 13 | `formulas.py:644 _bfs_count` | 20,856 | 0.167 | 2% |
| 14 | `game.py:149 _all_stuck` | 616 | 0.267 | 3% |
| 15 | `board.py:246 has_legal_move` | 726 | 0.256 | 3% |

**各人格每步平均耗時**（`tools/bench_profiles.py`，20 局、種子固定、可重現）：

| 人格 | 類別 | 出手數 | ms/手 |
|---|---|---|---|
| **builder** | 規則式 | 233 | **22.14** ← 明顯離群 |
| wolf | 權重式 | 267 | 9.80 |
| chess | 權重式 | 300 | 9.62 |
| fox | 權重式 | 265 | 9.72 |
| intruder | 規則式 | 230 | 6.77 |
| optimizer | 規則式 | 228 | 5.19 |

全體平均 10.49 ms/手。三個權重式人格幾乎一樣（9.6～9.8 ms），因為它們共用同一條
候選枚舉與對手預判路徑，差別只在權重。**builder 是次慢者的 2.2 倍**，而它 27% 的
時間花在 `fill_components` / `fillable_closure` → `dilate` 這條鏈上。

**推論（供 C 階段排序用）：** 優先順序應該是
(1) `builder` 的閉包／裝箱鏈、(2) `choose_move` 的全 base 迴圈（自身耗時 2.750 秒，
佔 31%，是最大的單一 self-time)、(3) 對手預判的 `_score_move`（52 萬次呼叫）。
`board_feats` 只有 6%，優先級較低。


推測的熱點（需驗證，不是結論）：`Game._all_stuck` → `has_legal` → `Board.any_legal`；`choose_move` 的全 base 迴圈；`board_feats`；`Board._update_regions`；對手前瞻的 `_score_move`。

#### 實測對照（推翻兩項推測）

| 推測 | 實測 | 結論 |
|---|---|---|
| `choose_move` 全 base 迴圈 | 自身耗時 2.750s，佔 31% | ✅ **最大熱點**，預期正確 |
| 對手前瞻 `_score_move` | 52 萬次呼叫，1.431s，佔 16% | ✅ 正確 |
| `board_feats` | 0.574s，佔 6% | ✅ 正確但**優先級比預期低** |
| `_all_stuck` → `has_legal` → `any_legal` | 合計 0.267s，佔 3% | ⚠️ 存在但**遠比預期小** |
| `Board._update_regions` | 每次 0.013ms，每局 64 次 = **0.08%** | ❌ **不是熱點**（見 B2） |
| （文件未預測） | `builder` 佔全部時間 27% | ⚠️ **新的最大單一人格負擔** |

`board.dilate` 以 **141 萬次**呼叫吃掉 1.316 秒，是全站呼叫次數最多、self-time 第三名
的函式，幾乎全部來自 `fill_components` 與 `fillable_closure` 的逐輪泛洪。

### B2. 查依賴 — ✅ 已完成

```bash
grep -rn "corner_regions\|borders\|\.grid\|all_owner_cells\|connected_region" --include=*.py .
```

回報每個使用點屬於 UI、AI 還是 `board.py` 內部。**已知 `choose_move` 直接讀 `board.grid`、`board.corner_regions`、`board.borders`，所以 `_update_regions` 目前不可移除。**

#### 實測結果

| 資料 | 撰寫 | 讀取 | 判定 |
|---|---|---|---|
| `board.grid` | `board.py`（`place` / `unplace` / `can_place` / `connected_region` / `all_owner_cells`） | `ai/chooser.py:139`（`board_feats`）、`ui.py:1149`（繪製）、多個測試 | **共享**：UI 與 AI 都直接依賴 |
| `board.corner_regions` | `board.py:218`（`_update_regions`） | `ai/chooser.py:140` → 傳給 `_score_move` / `_corner_delta` | **AI 依賴** |
| `board.borders` | `board.py:219`（`_update_regions`） | `ai/chooser.py:141` → `_corner_delta` | **AI 依賴** |
| `connected_region()` | `board.py` | 只在 `board.py:214`（`_update_regions`）內部呼叫 | **board.py 內部** |
| `all_owner_cells()` | `board.py:253` 定義 | **引擎內完全沒有呼叫者**；只有 `tests/test_board.py:120` 斷言它有回傳 | **死碼**（見下） |

**結論：`_update_regions` 目前確實不可移除** —— 本文原有的判斷正確。`corner_regions`
與 `borders` 只有 `choose_move` 這條路徑在讀。

**但它不是效能問題**：`_update_regions` 每次 0.013 ms，每局只被 `place` / `unplace`
呼叫約 64 次，合計約 0.9 ms／局，佔一局約 0.08%。**保留它的成本可以忽略**，所以
C 階段沒有任何理由去動它；反過來說，若日後要讓新引擎 `E1` 擺脫 `board.py`，
`corner_regions` / `borders` 就必須重新實作（或放棄角位加成）。

**額外發現：`Board.all_owner_cells` 是死碼。** 引擎裡沒有任何呼叫者，唯一引用是
一個測試斷言。可以安全刪除（連同該斷言），但這不屬階段 B 的範圍，留給 C 階段決定。

---

## 階段 C：行為不變的效能優化 — ✅ 已完成

> **狀態：** C1～C4 全部完成。**棋力與選步行為完全不變**，已用同種子逐手比對
> 與候選清單逐項比對證明。整體速度提升約 **30%**。
> 驗證工具：`tools/verify_same_moves.py`（內含階段 C 之前的 `frozen_choose_move`
> 作為永久對照組，以後不再改它）。

每個任務都要用「同種子逐手比對」驗證。建立共用測試工具：

```python
def play_recorded(seed, impl) -> list[(seed_game_idx, turn, owner, move_or_None)]
```

用同一個 seed 跑舊實作與新實作，逐手比較，不一致時印出第一個差異點的局面。

### C1. 永久 pass 標記 — ✅ 已完成

**理由：** 玩家一旦沒有合法步，之後永遠沒有。他的角接觸點只由自己的棋子決定（沒下子就不變），而空格只會減少。

**步驟：**

1. **先寫驗證測試（不改 `Game`）：** 跑 1000 局隨機對局，對每個被判定為無合法步的玩家，在之後每一步都重新檢查，斷言他不曾重新有合法步。若此測試失敗，**停止並回報，不要實作 C1**。
2. 在 `Game` 加入 `self.stuck = [False]*4`。`_all_stuck` 只檢查未標記的玩家，發現無合法步時標記。
3. `reset` 和 `start` 都要重置。

**驗收：** 同種子 1000 局逐手比對一致；`_all_stuck` 的呼叫成本下降（附 profile 前後對照）。

#### 實測結果（C1）

**前置閘門通過：** `tests/test_stuck_monotone.py` 在實作之前先跑，1000 局兩次獨立執行
（929.65s 與 928.28s）**全數通過**，沒有任何反例，因此才進行實作。

**同種子逐手比對：** 1000 局 / 76,617 手，**完全一致**（`IDENTICAL`）。

**profile 前後（各 8 局）：**

| 項目 | 前 | 後 | 變化 |
|---|---|---|---|
| `any_legal` 呼叫次數 | 2,963 | 1,117 | **−62%** |
| `any_legal` 自身耗時 | 0.234s | 0.071s | **−70%** |
| `_all_stuck` 累計耗時 | 0.258s | 0.014s | **−95%** |

實作補充：

- `has_legal` **不讀** `stuck`，永遠重算。UI 需要它決定是否顯示「必須跳過」提示，
  測試也需要它當作 `stuck` 的對照組。
- `stuck` 的不變式是**單向**的：`stuck[o] == True` 必須蘊含 `has_legal(o) == False`。
  逆方向不成立 —— `_all_stuck` 遇到第一個還有合法步的玩家就提早返回，後面的座位
  可能已經無步但尚未被標記。

### C2. `has_legal` 改用 `ODIRS` 與 `_legal_bases` — ✅ 已完成

**前提：A1 已完成。**

**檔案：** `game.py`、`board.py`（`any_legal` 保留不刪，作為參照實作）。

```python
def has_legal(self, owner):
    empt = self.board.empty_bits
    reach = self.reach(owner)
    mc = self.must_cover(owner)
    cbit = 1 << (mc[0] + mc[1] * B) if mc is not None else 0
    for name in self.hands[owner].names:
        for od in ODIRS[name].values():
            if _legal_bases(od, empt, reach, cbit):
                return True
    return False
```

注意 `ODIRS` 位於 `ai.formulas`。避免 `game.py` 與 `ai` 之間的循環 import（`game.py` 已 import `ai`，應可行；若出問題，把 `ODIRS`/`_legal_bases` 的幾何部分搬到不依賴人格的模組）。

**驗收：**

- 隨機對局 1 萬局，每一步對每位玩家比較 `has_legal_new(owner) == board.has_legal_move(...)`（舊實作），必須全部相同。
- 同種子逐手比對一致。

#### 實測結果（C2）

`game.py` 照上例改寫，`from ai.formulas import ODIRS, _legal_bases`。**沒有循環
import**：`game.py` 本來就 import `ai`，`ai.formulas` 不依賴 `game`，所以成立。
`Board.has_legal_move` / `any_legal` 依本文要求**保留**，作為參照實作。

| 檢查 | 結果 |
|---|---|
| **隨機對局 1 萬局、逐步對每位玩家比對（本文驗收標準）** | **3,066,404 次比對，0 差異** |
| 400 局、逐步對每位玩家比對 | 122,744 次比對，**0 差異** |
| 常駐測試（12 局，預設） | 通過（`tests/test_optimised_paths.py`） |
| 同種子 1000 局逐手比對 | 76,617 手**完全一致**（與 C1 同一份基準） |

1 萬局那一次在 2×N 核機器上跑了約 2.5 小時——每一步都要對四位玩家各跑一次舊的
暴力 `has_legal_move`，而已無合法步的玩家沒有短路，整個盤面都要掃完。

profile（8 局）：`has_legal` 累計 0.086s → **0.013s**；`any_legal` 已跌出前 15 名。

### C3. `choose_move` 的候選枚舉改用 `_legal_bases` — ✅ 已完成

**前提：A1、C2 已完成。**

將 `for base in od["bases"]: ... if (empt & shifted) != shifted: continue ...` 改為：

```python
mask = _legal_bases(od, empt, reach, cbit)
while mask:
    low = mask & -mask
    mask ^= low
    base = low.bit_length() - 1
    shifted = m << base
    ...
```

**必須保持：** 候選列表的元素、分數與順序與舊實作完全相同（由低位到高位的取位順序與 `od["bases"]` 一致）。`_opponent_pool` 的後備掃描同理可用。

**驗收：** 對 1000 局、六種人格，逐手比對選步完全一致；候選列表（`cands`）在每個決策點逐項相等。

#### 實測結果（C3）

| 檢查 | 結果 |
|---|---|
| **候選清單逐項比對（1000 局，本文驗收標準）** | **76,617 個決策點、12,889,875 列候選，0 差異** |
| 候選清單逐項比對（30 局，快速版） | 2,345 個決策點、387,056 列候選，0 差異 |
| 常駐測試（12 局，預設） | 通過 |
| 枚舉順序測試 | 候選的 `(name, oi, base)` rank 嚴格遞增且不重複 |
| 同種子 1000 局逐手比對（六種人格都涵蓋） | 76,617 手**完全一致** |

76,617 個決策點正好等於 1000 局的總出手數，代表**每一手**的候選清單都被逐項比對過，
包含元素、順序與分數。

實作補充：

- 枚舉獨立抽成 `chooser._candidates(...)`，讓驗證工具能直接比對候選清單，
  而不是只能比對最後一步。
- **順序確認：** 從遮罩最低位取值 = base 由小到大，而 `od["bases"]` 也是由小到大
  （`for y ... for x ...`），兩者一致，所以 `cands.sort`（Python 穩定排序）的
  並列順序沒有改變。這是硬性規則 4 的核心，測試
  `test_candidate_order_is_still_piece_then_oi_then_base` 專門釘住它。
- profile（8 局）：`choose_move` 自身耗時 **2.750s → 0.524s**。

### C4. 預算改為計數式 — ✅ 已完成

- `WALL_BUDGET`（0.9 秒）會令同局面、同種子在不同機器負載下選步不同。
- 新增模組級開關 `F.USE_WALL_BUDGET = True`（預設保持舊行為）。訓練與評測時設為 `False`。
- `SIM_EVAL_BUDGET` 已是計數式，保留。

**驗收：** 開關關閉時，同種子重跑兩次結果逐手相同，且與機器負載無關。

#### 實測結果（C4）

`chooser.choose_move` 的判斷式改為：

```python
if other_brains and opps and brain.uses_lookahead \
        and (not F.USE_WALL_BUDGET
             or time.perf_counter() - t0 < F.WALL_BUDGET):
```

| 檢查 | 結果 |
|---|---|
| 開關關閉，同種子跑兩次 | **完全相同** |
| 開關關閉，3 個執行緒燒 CPU 時重跑 | **與未負載時完全相同** |
| 預設值 | `USE_WALL_BUDGET is True`（保持舊行為） |

`SIM_EVAL_BUDGET` 是純計數，因此開關關掉後，lookahead 仍完全受它約束，引擎就變成
與機器無關、與負載無關的純函數。這是訓練與評測可重現的前提。

### 階段 C 總效果

| 量測 | 階段 B | 階段 C 之後 | 變化 |
|---|---|---|---|
| cProfile 8 局總時間 | 8.965s | 6.661s | **−26%** |
| 無 profiler（20 局，seed 100） | 0.837 s/局 | **0.548 s/局** | **−35%** |
| 平均每手 | 10.49 ms | **7.10 ms** | **−32%** |
| builder | 22.14 ms | 18.13 ms | −18% |
| intruder | 6.77 ms | 3.51 ms | −48% |
| optimizer | 5.19 ms | 1.94 ms | −63% |

**行為不變的證據：** 同一組種子（`--games 20 --seed 100`）在階段 C 前後，剩餘格總數
**同為 1508**，各人格出手數**完全相同**（233 / 300 / 267 / 265 / 230 / 228）。

#### 如何重跑階段 C 的驗證

```bash
# C1 閘門（1000 局，約 15 分鐘）
BLOKUS_STUCK_GAMES=1000 python3 -m pytest tests/test_stuck_monotone.py

# C1/C2/C3 的同種子逐手比對（1000 局，約 25 分鐘）
python3 tools/verify_same_moves.py --impl frozen  --games 1000 --record /tmp/base.json
python3 tools/verify_same_moves.py --impl current --games 1000 --baseline /tmp/base.json

# C2 的逐步比對（本文驗收標準，1 萬局，約 2.5 小時）
python3 tools/verify_same_moves.py --check-has-legal 10000

# C3 的候選清單比對（1000 局，約 1.5 小時）
python3 tools/verify_same_moves.py --check-candidates 1000

# 常駐測試（12 局，約 1 分鐘）
python3 -m pytest tests/test_optimised_paths.py
```

**長跑項目的預設值刻意調小**，否則每次 `pytest tests/` 都要多花好幾小時：

| 環境變數 | 預設 | 用途 |
|---|---|---|
| `BLOKUS_STUCK_GAMES` | 120 | C1 閘門的局數 |
| `BLOKUS_EQUIV_GAMES` | 12 | 常駐等價性測試的局數 |

---

## 階段 D：決策記錄與可重現性 — ✅ 已完成

> **狀態：** D1、D2 全部完成。D1 的 `trace` 是**純觀測**，不影響任何選步；
> D2 的評測賽在相同參數下分數**逐位重現**。

### D1. `choose_move` 加可選的 `trace` 參數 — ✅ 已完成

**不改變回傳值。** 若 `trace` 不是 None，往其中寫入一個 dict：

```python
{
  "owner": owner,
  "brain_key": brain.key,
  "profile": dataclasses.asdict(brain.profile),
  "n_candidates": len(cands_before_restrict),
  "shortlist": [(score, name, oi, base), ...],
  "picked": (name, oi, base),
  "was_mistake": bool,          # 是否走了 _weighted_pick 分支
  "lookahead_used": bool,
}
```

**驗收：** `trace=None` 時行為與效能完全不變（同種子逐手比對）；`trace` 有值時，`picked` 與實際回傳一致。

### D2. 固定座位輪換的評測賽 — ✅ 已完成

**新增檔案：** `tools/benchmark.py`，不改動現有 `match.py`。

- 不隨機抽人格。指定四個人格，並輪換全部座位排列（4! = 24 種，或指定子集），每種排列跑 N 局。
- 每局用固定種子，可重現。
- 記錄：每個人格的平均剩餘格數、平均名次、勝率、每步平均耗時。
- **並列名次要平分：** 兩人並列第二各得 2.5，不要沿用 `Game.standings` 按座位編號決勝的做法（那會令座位編號成為偏差來源）。`standings` 本身不要改，因為 UI 依賴它。

**驗收：** 相同參數跑兩次，輸出逐位相同。

#### 實測補充（D1 與 D2）

**D1 的關鍵設計：`trace=None` 時熱路徑完全不受影響。** 所有記錄動作都包在
`if trace is not None:` 之內，預設不傳就不執行任何額外分支。`n_candidates` 在
`cands.sort()` 之後、`brain.restrict()` 之前記錄，符合本文定義。

| 檢查 | 結果 |
|---|---|
| `trace` 的 8 個欄位 | 齊全且型別正確 |
| `picked` 與實際回傳值 | **每一步都相等** |
| `picked` 一定在 `shortlist` 內 | 通過 |
| `profile` | 等於 `dataclasses.asdict(brain.profile)` |
| 規則式人格的 `lookahead_used` | 恆為 `False`（符合 `uses_lookahead = False`） |
| **`trace=None` 的同種子比對，1000 局** | **76,617 手完全一致（`IDENTICAL`）** |
| 不開 trace 的耗時 | 未因這個功能變慢（測試有粗略上限把關） |

兩個容易誤判的地方，已在 `tests/test_trace.py` 釘住：

- **兩次呼叫同一局面會得到不同答案**，因為決策會消耗腦的 RNG（出錯抽籤）。所以
  「`trace=None` 與不傳 `trace` 相同」必須用**兩個相同種子的獨立局面**比較，
  不能在同一個 `Game` 上呼叫兩次。
- **`picked` 等於短名單首位不代表沒有出錯**：`_weighted_pick` 是加權抽籤，
  完全可能抽回首位。所以 `was_mistake == False` 無法由 `picked` 推出；反過來
  `picked` **不等於**首位，則一定走過加權抽籤分支。

**D2 的可重現性有一個必須說清楚的例外：`ms/move` 是掛鐘量測，本來就不會重現。**
本文要求「輸出逐位相同」與「記錄每步平均耗時」兩者本質衝突，因此做法是：

- **分數部分**（games / avg_place / avg_cells / wins）**逐位重現**，已驗證。
- `ms/move` 移到獨立區塊並標註為「量測，非結果」。
- 另有 `--no-timing` 開關，輸出**完全逐位相同**（含 JSON 檔，已用 `diff` 驗證）。

**並列平分的實作：** 不用 `Game.standings`（它按座位編號決勝，那正是本工具要消除的
偏差），改用 `average_ranks`：並列者平分他們所佔的順位區間。例如 `[3,8,8,12]`
→ `[1, 2.5, 2.5, 4]`。測試直接釘住這張對照表，並驗證「交換座位不影響任何人的名次」
與「四人平均名次恆為 2.5」。

24 種座位排列各跑 2 局的實測結果（`--personalities wolf chess fox optimizer`）：

```
key          games  avg_place   avg_cells   wins   ms/move
optimizer       48      1.177        6.02     38      1.97
wolf            48      2.719       23.31      2      6.44
chess           48      2.812       23.83      5      6.40
fox             48      3.292       28.48      0      6.37
```

這也回答了 B 階段留下來的問題：優化者確實最強，而且不是座位造成的偏差（它贏了 48 局
裡的 38 局）。

---

## 階段 E：新引擎與旋轉 — ✅ 已完成

> **狀態：** E1、E2、E3 全部完成。新引擎 `engine.py` 是**第二套獨立實作**，
> 與現有 `Board`/`Game` 並存，不替換它們。1 萬局交叉驗證**零差異**。
>
> **相依變更：** E3 需要 numpy 平面，與 AGENTS.md「無外部相依」衝突，經確認後
> 放行 `numpy`（僅限 E3 那一段），並已更新 AGENTS.md。

### E1. 建立獨立的無畫面引擎 `engine.py`

**與現有 `Board`/`Game` 並存，不替換。** 新引擎不得 import `pygame`、`ai.chooser`、`ai.registry`、`records`、`ui`。

局面只含：

- `own_bits`：4 個 400 位元整數。
- `hand_bits`：4 個 21 位元整數（位元編號按 `action_table.json` 的 `pieces` 順序）。
- `to_move`、`turn_order`。
- `stuck`：4 個布林。
- `placed_count`（或由 `own_bits == 0` 判斷是否首步）。

介面：

```python
state = initial_state(turn_order)
legal_moves(state)      # 回傳 (piece_idx, oi, base) 的集合或遮罩
apply_move(state, mv)   # 回傳新局面（不可共享可變物件）
is_over(state)
result(state)           # 四人剩餘格數
serialize(state) / deserialize(blob)
```

規則檢查**不可為可選**：`apply_move` 對非法步一律拋例外。

**驗收（交叉驗證）：** 讓現有 `Game` 帶老師 AI 跑 1 萬局。每一步在 `engine.py` 重播同一手，比較：

- 當前玩家的合法步集合（以 `(piece, oi, base)` 比較）。
- 四個 `owner_bits`。
- 各玩家 stuck 狀態。
- 終局剩餘格數。

任何一步不同就是 bug，資料不可使用。另加：序列化再還原後局面完全相同；複製後修改不影響原局面。

### E2. 旋轉對稱表

**前提：**座位順序 `CLOCKWISE_OWNERS = (1, 2, 0, 3)`（TL、TR、BR、BL），順時針 90° 旋轉的座標變換是 `(x, y) → (B-1-y, x)`。

```python
def rot_orient(name, oi):
    cells = MASTER[name]["orientations"][oi]
    h = max(y for _, y in cells) + 1
    r = frozenset((h - 1 - y, x) for x, y in cells)
    return MASTER[name]["orientations"].index(r)

def rot_base(name, oi, base):
    cells = MASTER[name]["orientations"][oi]
    h = max(y for _, y in cells) + 1
    x0, y0 = base % B, base // B
    return (B - y0 - h) + x0 * B
```

以上是推導出來的草稿，未經測試，以測試結果為準。

**測試：** 取 1000 個隨機局面（可從 E1 的自我對弈取樣），把所有玩家的位元遮罩逐格旋轉。對每個 `(棋子, oi)`，檢查：

```
rotate(legal_bases(state)) == legal_bases(rotate(state))
```

並檢查旋轉四次回到原狀、座位順序保持順時針。

**不要實作鏡像。**鏡像會把順時針變逆時針，等於換成另一個遊戲。

**驗收：** 上述測試全部通過。

### E3. 「以當前玩家為視角」的局面正規化

提供函數，把局面旋轉，使當前玩家的起始角落在 TL，並按「我、下家、對家、上家」重排玩家。輸出 numpy 平面（`float32`，形狀 `(C, 20, 20)`），至少含：

- 四位玩家的佔格。
- 我方可用角接觸格（`need & ~avoid & empty`）。
- 我方被禁止格（`avoid`）。
- 四位玩家的剩餘棋子（21 維向量，可廣播成平面或另外輸出）。

位元轉 numpy 的方式（未測試，請驗證）：

```python
import numpy as np
def bits_to_plane(mask):
    b = mask.to_bytes(50, "little")     # 400 bits = 50 bytes
    return np.unpackbits(np.frombuffer(b, np.uint8), bitorder="little").reshape(20, 20)
```

**驗收：** 隨機抽 100 個格子與位元，逐項對照；旋轉前後 `sum` 不變。

---

### 階段 E 實測結果

#### E1：`engine.py`

**刻意不共用程式碼。** 如果新引擎 import `board.dilate`，交叉驗證就只是自己跟自己
比。所以 `dilate` / `dilate_diag` / 合法性判斷全部重新推導，只在測試裡拿舊的
`board.py` 對照。

四個最容易寫錯的地方，都照著寫下來了，並且各自釘住測試：

| 陷阱 | 寫法 | 測試 |
|---|---|---|
| `cbit >> o` 已經是 base 集合 | 開局分支**不可**再 AND `od["m"]` | 4 個角全測 |
| `dilate` 的欄位遮罩要遮「來源」 | `(mask & ~COL0) >> 1` | 邊界 3 鄰居 |
| `dilate_diag` 不含四方向鄰居 | 兩者是不同的集合 | 與 `dilate` 不相交 |
| 物件不共用 | `State` 是 frozen dataclass，全 tuple/ int | 複製後互不影響 |

`apply_move` 對非法步**一律拋 `ValueError`**，不提供「檢查」以外的選項，也沒有
`unplace` 可回退。序列化用 JSON（大整數走 hex），`deserialize(serialize(s)) == s`。

> **實測更正一則本文的假設：** `X5`（十字形）的 3x3 外框有中心空洞，
> **永遠開不了局**（四個角都蓋不到）。這不是 bug，是規則正確運作，已用測試釘住。

1 萬局交叉驗證（`tools/cross_check_engine.py`）逐步比對：合法步集合、四個
`owner_bits`、四個 `stuck`、`to_move`，終局再比剩餘格數與結束判定。

```
games          : 10000
steps compared : 766601
legal sets     : 766601 checks, 129060099 legal moves total,
                 2000 brute-force audits
elapsed        : 6605s
IDENTICAL: engine.py and Game agree on every step of every game
```

**1 萬局、766,601 個決策點、1.29 億個合法步，零差異。** 其中 2,000 個取樣局面另外
拿 `Board.can_place`（逐 base 逐格掃描）重算一次，確認用來對照的快速路徑本身也沒錯 ——
否則「兩邊都對著同一個錯的東西」是可能的。

（順帶一提：1,000 局時的 76,617 手與 D1 的驗證數字完全一致，代表兩套引擎在同種子下
走的是同一局棋。）

#### E2：旋轉對稱表

本文給的 `rot_orient` / `rot_base` 草稿**經實測是對的**，已逐項驗證：

- `rot_orient` 對全部 91 個方向四次旋轉回到原點（0 失敗）。
- `rot_base` 與「整片棋盤做一次 `(x,y) → (B-1-y, x)`」比對 3,109 個落點，0 失敗。
- 座位置換由程式**推導**出來，不是手寫：`{0:3, 1:2, 2:0, 3:1}`，把順時針循環
  `(1,2,0,3)` 送到 `(2,0,3,1)`，即每個座位往前一步。

**寫 E3 時抓到 `rotate_state` 兩個真的 bug**（都已修）：

1. `hand_bits` 沒有跟著置換。`own_bits` 與 `hand_bits` 都是用 owner 編號的，
   只搬石頭不搬手牌，等於把某人的石頭算給一人、手牌算給另一人。
2. `turn_order` 沒有跟著置換，而 `to_move` 換了。`_advance` 靠
   `turn_order[turn_pos] == to_move` 前進，不置換等於**轉向錯誤**。
   （owner 編號是綁在角上的，轉棋盤本來就要換編號；不變的是「誰接著誰」的相對順序。）

驗收：1,000 個隨機局面，檢查 `rotate(legal(state)) == legal(rotate(state))`，
覆蓋四位玩家手上的**每一種**棋子，> 100,000 個落點全對。

#### E3：當前玩家視角

`normalize` 把局面旋轉到當前玩家坐在 TL，再把四人重編為
**我、下家、對家、上家**。`seat_of[i]` 回報原本的 owner 編號。

七個通道（順序固定，換順序等於讓模型學錯）：

```
me, next, opposite, previous, my_need, my_avoid, empty
```

外加 `hand_vectors()` 給 `(4, 21) float32`。**不**把手牌廣播成 400 格，
因為那是每局的特徵而不是每格的特徵。

本文給的 `bits_to_plane` 轉換**經驗證正確**（500 個隨機遮罩、100 格逐格對照），
`row = y`、`column = x`，不需轉置。

> **驗收中值得記錄的一點：** `need` 與 `avoid` **可以重疊**。一格可能斜角貼著我的一顆
> 子、正邊貼著另一顆。這正是本文規定 `need & ~avoid & empty` 的原因 ——
> 邊優先於角。已用 `(0,0)` 與 `(1,2)` 兩子讓 `(1,1)` 同時屬於兩者來釘住。

**最重要的性質：** 旋轉 1、2、3 次後的同一局面，`to_plane` 輸出**逐位元組相同**。
做不到這件事，模型就得學會同一局有四種寫法。

---

## 階段 F：先不要做

- 不要寫 PyTorch 訓練迴圈（引擎介面與 value 目標尚未穩定）。
- 不要重寫成 Rust。
- 不要移除 `Board._update_regions`（`choose_move` 仍依賴）。
- 不要把 AI 人格當成 MCTS 的 rollout 策略（每步評估數千候選，太慢）。
- 不要修改 `ui.py`。

---

## 執行順序摘要

| 順序 | 任務 | 依賴 | 狀態 |
|---|---|---|---|
| 1 | A1 修 `_adjoining_bases` | 無 | ✅ 完成 |
| 2 | A2 凍結動作表 | 無 | ✅ 完成 |
| 3 | B1、B2 量測與查依賴 | 無 | ✅ 完成 |
| 4 | C1 永久 pass（先做驗證測試） | 無 | ✅ 完成 |
| 5 | C2 `has_legal` 改寫 | A1 | ✅ 完成 |
| 6 | C3 枚舉改寫 | A1、C2 | ✅ 完成 |
| 7 | C4 預算計數式 | 無 | ✅ 完成 |
| 8 | D1 trace、D2 基準賽 | C3 | ✅ 完成 |
| 9 | E1 新引擎 | A1、A2 | ✅ 完成 |
| 10 | E2 旋轉表、E3 輸入編碼 | E1 | ✅ 完成 |

每完成一項，回報：改了哪些檔案、新增哪些測試、驗證結果，以及任何與本文件假設不符的發現。

---

