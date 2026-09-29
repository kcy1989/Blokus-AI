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

## 階段 A：修正 bug 與凍結資料格式（最優先）

### A1. 驗證並修正 `_adjoining_bases` 的首步分支

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

### A2. 凍結動作表

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

---

## 階段 B：量測（在動手優化之前）

### B1. 跑 profile 並回報

```bash
python3 -m cProfile -s cumtime match.py --games 3 --dry | head -60
```

回報：

- 每局耗時。
- 累計時間前 15 名的函數。
- 各人格每步平均耗時（如果現有程式能取得，否則新增計時記錄，不改行為）。

**這一步只量測，不做任何修改。** 之後的優化優先次序由結果決定。

推測的熱點（需驗證，不是結論）：`Game._all_stuck` → `has_legal` → `Board.any_legal`；`choose_move` 的全 base 迴圈；`board_feats`；`Board._update_regions`；對手前瞻的 `_score_move`。

### B2. 查依賴

```bash
grep -rn "corner_regions\|borders\|\.grid\|all_owner_cells\|connected_region" --include=*.py .
```

回報每個使用點屬於 UI、AI 還是 `board.py` 內部。**已知 `choose_move` 直接讀 `board.grid`、`board.corner_regions`、`board.borders`，所以 `_update_regions` 目前不可移除。**

---

## 階段 C：行為不變的效能優化

每個任務都要用「同種子逐手比對」驗證。建立共用測試工具：

```python
def play_recorded(seed, impl) -> list[(seed_game_idx, turn, owner, move_or_None)]
```

用同一個 seed 跑舊實作與新實作，逐手比較，不一致時印出第一個差異點的局面。

### C1. 永久 pass 標記

**理由：** 玩家一旦沒有合法步，之後永遠沒有。他的角接觸點只由自己的棋子決定（沒下子就不變），而空格只會減少。

**步驟：**

1. **先寫驗證測試（不改 `Game`）：** 跑 1000 局隨機對局，對每個被判定為無合法步的玩家，在之後每一步都重新檢查，斷言他不曾重新有合法步。若此測試失敗，**停止並回報，不要實作 C1**。
2. 在 `Game` 加入 `self.stuck = [False]*4`。`_all_stuck` 只檢查未標記的玩家，發現無合法步時標記。
3. `reset` 和 `start` 都要重置。

**驗收：** 同種子 1000 局逐手比對一致；`_all_stuck` 的呼叫成本下降（附 profile 前後對照）。

### C2. `has_legal` 改用 `ODIRS` 與 `_legal_bases`

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

### C3. `choose_move` 的候選枚舉改用 `_legal_bases`

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

### C4. 預算改為計數式

- `WALL_BUDGET`（0.9 秒）會令同局面、同種子在不同機器負載下選步不同。
- 新增模組級開關，例如 `F.USE_WALL_BUDGET = True`（預設保持舊行為）。訓練與評測時設為 `False`。
- `SIM_EVAL_BUDGET` 已是計數式，保留。

**驗收：** 開關關閉時，同種子重跑兩次結果逐手相同，且與機器負載無關。

---

## 階段 D：決策記錄與可重現性

### D1. `choose_move` 加可選的 `trace` 參數

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

### D2. 固定座位輪換的評測賽

**新增檔案：** `tools/benchmark.py`，不改動現有 `match.py`。

- 不隨機抽人格。指定四個人格，並輪換全部座位排列（4! = 24 種，或指定子集），每種排列跑 N 局。
- 每局用固定種子，可重現。
- 記錄：每個人格的平均剩餘格數、平均名次、勝率、每步平均耗時。
- **並列名次要平分：** 兩人並列第二各得 2.5，不要沿用 `Game.standings` 按座位編號決勝的做法（那會令座位編號成為偏差來源）。`standings` 本身不要改，因為 UI 依賴它。

**驗收：** 相同參數跑兩次，輸出逐位相同。

---

## 階段 E：新引擎與旋轉

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

## 階段 F：先不要做

- 不要寫 PyTorch 訓練迴圈（引擎介面與 value 目標尚未穩定）。
- 不要重寫成 Rust。
- 不要移除 `Board._update_regions`（`choose_move` 仍依賴）。
- 不要把 AI 人格當成 MCTS 的 rollout 策略（每步評估數千候選，太慢）。
- 不要修改 `ui.py`。

---

## 執行順序摘要

| 順序 | 任務 | 依賴 |
|---|---|---|
| 1 | A1 修 `_adjoining_bases` | 無 |
| 2 | A2 凍結動作表 | 無 |
| 3 | B1、B2 量測與查依賴 | 無 |
| 4 | C1 永久 pass（先做驗證測試） | 無 |
| 5 | C2 `has_legal` 改寫 | A1 |
| 6 | C3 枚舉改寫 | A1、C2 |
| 7 | C4 預算計數式 | 無 |
| 8 | D1 trace、D2 基準賽 | C3 |
| 9 | E1 新引擎 | A1、A2 |
| 10 | E2 旋轉表、E3 輸入編碼 | E1 |

每完成一項，回報：改了哪些檔案、新增哪些測試、驗證結果，以及任何與本文件假設不符的發現。

---

