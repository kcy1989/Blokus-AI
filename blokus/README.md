# 單人 BLOKUS

20x20 棋盤的 Blokus 對戰程式：一位真人玩家對上三個 AI，AI 每局換一種
「人格」。附一個全 AI 聯賽工具，可以把七種人格的強弱跑成排行榜。

- 語言：Python 3
- 介面：pygame
- 依賴：`pip install -r requirements.txt`

---

## 目錄

1. [快速開始](#快速開始)
2. [遊戲規則](#遊戲規則)
3. [計分方式](#計分方式)
4. [七種 AI 人格](#七種-ai-人格)
5. [介面版面](#介面版面)
6. [程式架構](#程式架構)
7. [AI 套件詳解](#ai-套件詳解)
8. [位元運算的關鍵設計](#位元運算的關鍵設計)
9. [排行榜資料](#排行榜資料)
10. [測試](#測試)
11. [新增一種 AI](#新增一種-ai)
12. [歷史證據與已知事項](#歷史證據與已知事項)
13. [已知限制與取捨](#已知限制與取捨)

---

## 快速開始

```bash
pip install -r requirements.txt

python3 main.py              # 開始遊戲（pygame 視窗）

python3 match.py --games 100 # 全 AI 聯賽 100 局，結果寫進排行榜
python3 match.py --games 20 --dry   # 只跑不寫檔
python3 match.py --games 100 --reset  # 先清空排行榜再跑
.venv-rl/bin/python -m pytest tests/ -q -n 8   # 跑測試（約 160–175 秒，601 項）
```

> 測試要用 `.venv-rl`（torch 與 `pytest-xdist` 裝在那裡），不是系統的 `python3`。
> 詳細的測試規則見 `AGENTS.md` 第六節。

### 遊戲內操作

| 按鍵 | 功能 |
|---|---|
| 左鍵點棋盤 | 鎖定位置 |
| Enter / 確認 | 放下棋塊 |
| 右鍵 / X | 旋轉 90° |
| C | 取消選擇 |
| `[` `]` | 調整手牌縮放（放不下時會自動縮小） |
| `-` `+` | 調整棋盤縮放 |
| F | 視窗符合螢幕 |
| R | 重開一局 |

> 視窗固定 16:9，無法用手動拖曳改變比例；`-` / `+` 是唯一的縮放方式。

---

## 遊戲規則

Blokus 是經典的版圖佔領遊戲，四人共用 20x20 的棋盤，每人一套共 21 種、
共 89 格的棋塊。

1. **輪轉**：順時針繞棋盤一圈，每輪每人放一塊。起點隨機，所以玩家這一局可能
   第一也可能第四。
2. **開局**：第一塊必須覆蓋自己指定的那一角位（藍色右下、綠色左上、紅色右上、
   黃色左下）。
3. **角對角規則**：第一塊之後，每一塊都必須
   - **至少有一格**與自己既有的棋**角對角相鄰**（斜對角，共一個交點），而且
   - **不可以有任何一格**與自己既有的棋**共邊**。

   這兩條合併成一個 `Reach(need, avoid)` 結構：`need` 是角對角可碰的空格、
   `avoid` 是共邊會犯規的空格。合成一個物件是為了讓呼叫端不可能只執行規則
   的一半。
4. **過關**：沒辦法放的人自動跳過，四人都不能放時結束。

### 跨越：Blokus 的勝負關鍵

如果一個 2x2 方塊裡，**一條對角線全是自己的棋、另一條對角線有對手的棋**，
這一局就算贏了。這個判定不需要歷史，只看當前棋盤
（`ai.formulas.has_crossed`）。整個 AI 設計都是圍繞它。

---

## 計分方式

**分數 = 放不上去而留在手上的格數，愈少愈好。**

這個定義是整個程式最重要的設計決定，因為它讓「AI 想什麼」變成可寫在程式裡
的數量關係：

| 目標 | 量 | 單位 |
|---|---|---|
| 少留格 | 剩餘格數 | 格 |
| 保留延伸空間 | 落子後的可放空格數 | 格 |
| 不被封死 | 可用交點數 | 個 |
| 圍出放得下大棋的地 | 裝不下手牌的格數（`pack_lost`） | 格 |

每一個 AI 指標的單位都刻意對齊到「格」，所以權重之間的大小關係可以直接比較，
不需要換算係數。

單局結束後四人依剩餘格排名，第一名 4 分、第二名 3 分、第三名 2 分、第四名
1 分，摺進各自的歷史平均。並列時共用名次，後面的名次順延。

---

## 七種 AI 人格

人格分兩類：

- **權重式**（狼、棋手、狐狸）：六個權重線性加總，評分路徑完全相同。
- **規則式**（入侵者、優化者、築城者、獵手）：目標函式是一條具體的規則，不做
  權重加總，而且不抽籤、不做對手預判。

| key | 名稱 | 類別 | 風格 |
|---|---|---|---|
| `wolf` | 狼 | 權重 | 激進 — 搶占四角、擺放大塊 |
| `chess` | 棋手 | 權重 | 平衡 — 穩健佈局、權重均衡 |
| `fox` | 狐狸 | 權重 | 防守 — 壓制接合、把控邊界 |
| `intruder` | 入侵者 | 規則 | 跨越 — 快速伸長、深入敵方領地 |
| `optimizer` | 優化者 | 規則 | 效率 — 大塊先放、留住可放空間 |
| `builder` | 築城者 | 規則 | 築城 — 圍住一整塊放得下大棋的地 |
| `hunter` | 獵手 | 規則 | 攻勢 — 開局定式，其後搶對手的可放空間 |
| `rl_h1000_0k` | 模仿 1000 步 | 網絡 | H-C2 模仿資料上訓 1,000 步（舊名 `hc_1000`，仍為永久別名） |
| `hc_2000` | 模仿 2000 步 | 網絡 | 同上，2,000 步 |
| `hc_10000` | 模仿 10000 步 | 網絡 | 同上，10,000 步 |
| `rl_h1000_20k` | 強化學習 rl_h1000_20k | 網絡 | 從 `rl_h1000_0k` 出發的 PPO，40 輪共 20,000 局 |

### 訓練策略名稱的兩段數字單位不同

`rl_h1000_20k` 裡的兩個數字**不是同一種東西的兩次计数**，讀錯單位會把訓練量低估
或高估兩個數量級：

| 片段 | 單位 | 意義 |
| --- | --- | --- |
| `h` | — | 0k 起點的老師是 `hunter` 人格（`data/hc1/manifest.json` 記 `teacher = hunter`） |
| `1000` | **模仿步數** | 0k 版 `rl_h1000_0k` 的模仿訓練步數，**不是** 1,000 局 |
| `20k` | **RL 局數** | PPO 單打 20,000 **局**（40 輪 × 500 局），**不是** 20 個模仿步 |

所以 `rl_h1000_0k` 的 `1000` 與 `rl_h1000_20k` 的 `20k` 單位不同：前者數模仿步、後者
數對局。兩個改名都用**別名**而不是第二列（`seats.ALIASES`）：舊名 `rl_1000_20k`
與 `hc_1000` 仍解析到同一座位，舊指令與舊的 `records.json` 紀錄都讀得到；別名**不**
進 `automated_options()`，所以池裡只列正式名，池大小不因別名增加（現為 14，
見「GOLDEN 重採（plan9a 階段 6）」），排行榜也仍是同一列。

權重檔本身沒有改名：`data/rl1/step_000040.pt` 的檔名編碼的是**第幾輪**，不是策略
名稱。

### `records.json` 的 `commit` 欄位是「訓練出這些權重的程式碼版本」

不是「登記這個席位時的版本」，也不是「把它改名的版本」。這三件事的答案不同，而把第二
或第三個填進來比留 `null` 更糟：登記用的 commit 會在任何人改名時就過期，到那時它已經
不再描述旁邊那個權重檔。

**`rl_h1000_20k` 的 `commit` 目前是 `null`，原因是它的訓練沒有記錄這件事。**
`rl/collect.py` 有 `code_hash()`，會同時記錄 commit **與工作樹是否乾淨**——它的註解
寫得很直白：「工作樹有未提交的變更，所以單靠 commit 不能識別這份資料」。但
`rl/rl_train.py` 兩者都沒記（checkpoint 的 blob、`rounds.jsonl`、`eval.jsonl` 都沒有
commit 欄位）。

從 git 歷史可以把訓練時段（2026-10-05 07:34–08:28，由 `data/rl1/` 各檔的 mtime 決定）
夾到唯一的 commit `82bfadf`：該時段內沒有任何 commit，前一個 commit 是 `82bfadf`
（10-05 01:12），下一個是 `1f91b04`（10-05 08:56）。**但這是推論，不是紀錄**——沒有
任何東西能證明當時工作樹是乾淨的，而 reflog 只記 HEAD 移動、不記工作樹內容。按本專案
自己的標準（`code_hash()` 的那句註解），單獨一個 commit 不足以識別資料，所以這裡填
`null`。

**修法在訓練端**：`rl/rl_train.py` 應比照 `rl/collect.py` 的 `code_hash()`，把
commit 與 dirty 記進 checkpoint。那是訓練程式碼，屬於之後動訓練時要一起處理的事。

### 權重式

評分 = 六項加總：

```
分數 = w_large × 棋塊大小
     − 0.25 × (5 − 棋塊大小)        # 固定的小棋懲罰
     + w_center × 中心親近度
     + w_open  × 鄰接空格數
     + w_block × 貼著對手棋的格數
     + w_defend × 貼著自己棋旁的對手
     + w_corner × 3.0 × 角位連通區塊的擴張量
```

| 權重 | wolf | chess | fox | 意思 |
|---|---|---|---|---|
| `w_corner` | **4.0** | 1.0 | 0.5 | 搶角位的食慾 |
| `w_center` | 1.0 | 1.0 | 1.0 | 往盤心擠 |
| `w_block` | 2.0 | 1.0 | **3.0** | 壓住對手 |
| `w_defend` | 0.3 | 1.0 | **3.0** | 保護接合處 |
| `w_open` | 0.5 | 1.0 | **2.0** | 留白的價值 |
| `w_large` | **3.0** | 1.0 | 0.5 | 偏好大棋 |
| `mistake_rate` | 0.15 | 0.05 | 0.25 | 抽籤出錯率 |

每局的權重還會乘上 0.7～1.3 的隨機擾動（`ai.base.make_profile`），否則排行
榜會變成一組固定對局的重播。

### 規則式

**入侵者**（`ai/heuristics/intruder.py`）五個階段，**都是軟限制**——指定棋塊一格都下不了
就退回下一階段：

1. 手上還有跨越棋（V5／W5／Z5）就只用跨越棋
2. 否則用長手臂棋（L5／N5／I5）
3. 都沒有了，用能佔到「關鍵格」的棋（佔下去之後存在一個落點可以完成跨越）
4. 這一手本身完成跨越 → 大幅加分，並要求跨過去之後還有地方可放
5. 一般局面：5 格優先，戰略點放寬到 4 格、3 格

**優化者**（`ai/heuristics/optimizer.py`）目標函式只有一個量：落子後自己還有幾個可放空格。
緊急規則是安全閥——某塊棋只剩唯一一個落點時先救它。開局不適用（那時落點少是
角位規則壓出來的，不是棋盤變擠）。

**築城者**（`ai/heuristics/builder.py`）主項是 `pack_lost`：把剩餘手牌用 FFD 裝進落子後
的活區塊，裝不進去的格數以負權重計。大小偏好與可放空間總數只是次要項，都刻意
小到讓不過主項，所以它不設階段也不設緊急規則。

**獵手**（`ai/heuristics/hunter.py`）是優化者的後代，兩件事疊在一起：

1. **開局定式**：前三手走固定的標準開局 —— Z5 覆蓋自己的角與該對角線往內第二
   格，然後 V5、W5 各往前兩格。每一步在該步的候選裡**均勻抽一個**（共 2 個候選，
   等於擲硬幣）。候選集為空就**放棄整本定式**、退回評分，而且一局之內不再重試。
   放棄這條路徑在真實對局中不可達：定式的六格都沿自己的對角線，而離中心最近的
   角就是自己的角，三步之遙 —— 掃描 174 個合法對手開局，沒有一個清空候選集。
2. **攻勢評分**：第 4 手起把目標函式從「我的可放空格」換成
   **「我的可放空格 − 強敵的可放空格」**。

```
分數 = A_我(後) − A_j*(後)
```

- `A_我(後)` 是落子後**自己**可放的空格數 —— 與優化者完全相同的表達式。
- `A_j*(後)` 是三位對手裡「可放空格最多」那位（記為 `j*`）可放的空格數。
- **`j*` 在落子前一次決定**，同一個決策的所有候選都對同一位 `j*` 計分。並列時取
  行棋順序上最接近自己的下一位。

這項差分的代價極小：自己那一半的計算是原本的呼叫，而對手那一半可以用一個位元
運算直接求出 —— `A_j*(後) = (j* 原本的可放格 & ~placed).bit_count()`。因為
`own_reach` 只讀 `board.owner_bits[owner]`，自己的棋子不會進入對手的角接觸集合，
只會蓋掉對手原本可放的格。實測 2,010 組（對手 × 候選）全部相符。

獵手是**階段 H 的模仿學習老師**。成對實驗（`rl/paired3.py`，7,500 對）中，
A 組 5,000 對的平均名次差為 **−0.0817**、95% 區間 [−0.1086, −0.0543]，也就是
獵手在名次上顯著優於只改定式的版本。但**對強對手（築城者、入侵者、優化者）時
獵手反而多留約 1 格**（+0.962，區間 [+0.664, +1.2628]），而且**表現高度依賴
座位**（席位 1 是 +0.4286、席位 2 是 −0.4212，兩個都顯著）。細節見
`reports/h1_report.md`。

獵手與優化者的差別只在評分式與開局：**權重檔完全相同**（兩者都用同一份 chess 權重檔），
`mistake_rate` 也都是 0。定式那三次抽籤是**開局的一部分**，不是人格的出錯率 ——
它抽的是「兩個都合法、都屬於定式的候選要走哪一個」，所以不會產生低質量的著法。

### 為什麼規則式人格不抽籤、不做對手預判

- **不抽籤**：戰略獎勵一被隨機性蓋掉就沒有意義，「前 3 手只用跨越棋」這種性質
  也才檢查得起來。
- **不做對手預判**：預判用的是**權重**評分，量級跟規則式的目標函式完全不同。
  那一項會直接蓋掉規則排序，等於白算。

規則式人格仍然帶一份 chess 的權重檔，只用在第一階段的粗篩與被別人預判時的評分。

---

## 介面版面

視窗固定 **16:9**：`Layout` 先定 `H`，再由 `W = H * 16 / 9` 反推寬度，所以比例
不會在任何縮放等級走樣。手動拖曳視窗會被彈回，只有 `-` / `+` 能改變大小。

```
┌──────────┬─────────────────┬──────────────┐
│  玩家資訊  │                 │  5格          │
│  欄       │                 │ ┌──────────┐ │
│          │      棋盤        │ │ 5格方塊  │ │
│  出牌     │      20x20      │ └──────────┘ │
│  順序     │                 │ ┌──────────┐ │
│          │                 │ │ 4格方塊  │ │
│  回合     │                 │ └──────────┘ │
│  提示     │  ┌───┬───┬───┐  │   ...        │
│          │  │確認│旋轉│取消│  │ ┌──────────┐ │
│          │  └───┴───┴───┘  │ │ 1格方塊  │ │
└──────────┴─────────────────┴──────────────┘
```

- **三欄共用同一個 `MARGIN`**：玩家欄與棋盤之間、棋盤與選方塊區之間，間距都是
  同一個數值，所以整個視窗讀起來是一個網格。
- **棋盤大小由高度反推**（`board_px = H - 3*MARGIN - BTN_H`），16:9 之下棋盤
  永遠是能放的最大值。
- **選方塊區在右欄**，一整列高度。五張卡（1～5格）由左至右排滿一行再折行，每張
  卡 = 上方標籤條 + 框住該組所有方塊的外框。1～4 格的方塊少，會並排在同一行。
  欄很窄所以組會自動折行，方塊在卡內置中。
- **卡片位置固定**：用掉一塊棋只是留下空格，其他方塊不會移動，所以不會看錯位。
- **按鈕在棋盤正下方並左右置中**，選好落點後不必移動滑鼠太遠。
- **方塊大小由右欄的空間反推**：`Layout` 會找出整副手牌放得下的最大方塊尺寸，
  所以右欄幾乎被填滿，不會在下方留一大片空白（旁邊的棋盤是滿高的，兩邊留白
  不對稱會讓整個視窗看起來歪）。`hand_zoom` 是這個最大值的比例，`[` `]` 調低
  可以換成更小、更容易掃視的縮圖。**寧可縮小，不可裁切。**
- 玩家資訊欄依**出牌順序**排列（`game.turn_order`），第一列就是這一局先手。

---

## 程式架構

```
main.py      進入點：pygame 初始化、事件迴圈、60 FPS
  └── ui.py          介面：棋盤繪製、手牌互動、排行榜顯示、訊息提示
         └── game.py        遊戲狀態機 SETUP → PLAYING → GAME_OVER
              ├── board.py    棋盤：格網、放置合法性、連通區塊
              ├── pieces.py   棋塊定義、旋轉生成、手牌管理
              ├── config.py   常數、角位、顏色、介面文字、棋塊載入
              └── ai/         AI 套件（見下）
match.py     全 AI 聯賽工具（不影響玩家對局流程）
  └── game.py, records.py
records.py   排行榜：每個人格的平均分與平均餘格
engine.py    純 Python 的規則引擎：400-bit 狀態、旋轉、合法步、正規化視圖
rl/          模仿學習的實驗骨架（見「rl/ 套件」）
tools/       量測與驗證腳本（benchmark、跨引擎檢查）
bench_engine.py  F 階段的跨層量測
tests/       30 個檔案，601 項測試
```

資料流是單向的：`ui.py` 畫出 `game.py` 的狀態，玩家操作呼叫 `game.act()`，
`game.py` 需要 AI 決策時才呼叫 `ai.choose_move()`。AI 不會反向呼叫 UI。

`game.py`（人類可讀的實作）與 `engine.py`（給學習者與測試用的規則引擎）**各有一份
規則**，兩者必須逐步一致 —— 由 `tests/test_engine_cross.py` 在整局中逐步比對
`State` 與 `Game` 的 `stuck` 旗標與剩餘格數。`rl/` 底下的一切只吃 `engine.py`，
不吃 `game.py`。

### 檔案職責

| 檔案 | 職責 |
|---|---|
| `config.py` | 盤面大小、角位、顏色、**所有介面文字**、棋塊 JSON 載入 |
| `board.py` | `Board` 類別：格網、`empty_bits`／`owner_bits`、`Reach`、連通區塊、合法性判定 |
| `pieces.py` | 21 種棋塊、旋轉去重、手牌容器 |
| `game.py` | 回合狀態、開局角位規則、順時針輪轉、勝負判定 |
| `engine.py` | 與 `board.py` 平行的另一份規則實作，用不可變的 `State` 描述整局 |
| `records.py` | 排行榜持久化（`records.json`） |
| `ai/` | AI 套件，見下一節 |
| `rl/` | 模仿學習實驗骨架，見下 |

介面文字全部集中在 `config.py` 的 `I` 字典，所以新增語系或改文案不必碰邏輯。

---

## rl/ 套件

`rl/` 是模仿學習的實驗骨架，全部模組只吃 `engine.py`，不碰 `game.py` 與 `ui.py`。
每個檔案的頂部 docstring 都寫明「這層只回答什麼問題」。

| 檔案 | 回答的問題 |
|---|---|
| `rl/actions.py` | 一個合法落子如何變成單一整數，反之亦然（動作編碼表） |
| `rl/env.py` | 呼叫端要怎麼跟引擎對話（`reset`／`step`／`legal_indices`） |
| `rl/features.py` | 一個局面對**走子者**長什麼樣（14 個通道 + 12 個純量 + 4 組手牌） |
| `rl/opening.py` | 開局定式是否比優化者更好的老師（兩階段實驗） |
| `rl/v3.py` | 「我的可放格減強敵的可放格」是否更強 |
| `rl/paired.py` | 定式 vs 原版（兩臂）的成對實驗與自助法區間 |
| `rl/paired3.py` | 原版 / 定式 / 獵手（三臂）的成對實驗 |
| `rl/collect.py` | 一份「老師選步」的資料集要怎麼生成與重現 |
| `rl/smoke_gpu.py` | GPU 到底能不能用、有多快（階段 G 的量測） |

**特徵會把局面旋轉到走子者坐在左上角並重新編號**，所以兩局互為旋轉的局面會產生
**逐位元相同**的特徵 —— 這是旋轉式資料增強的前提。代價是**特徵裡沒有真實座位**：
同一位玩家坐在不同角落，會得到完全相同的輸入。

---

## AI 套件詳解

原本全部集中在單檔 `ai.py`（1099 行），現在拆成一個套件：

```
ai/
├── __init__.py     把所有舊名字重新匯出（相容層）
├── formulas.py     共用的計算公式：位元幾何、量化指標、權重表
├── base.py         Brain 介面、Profile 權重檔型別
├── chooser.py      選棋流水線，不認識任何人格
├── registry.py     key → Brain 類別／權重檔的對照表；下層載入 registry.json
├── registry.json   名冊：誰是選手、順序、別名、池（資料層，見「名冊」一節）
│
└── heuristics/     Python 算法型 AI：一種人格一個模組
    ├── wolf.py         ┐
    ├── chess.py        ├ 權重式：只有權重不同，評分路徑完全相同
    ├── fox.py          ┘
    │
    ├── intruder.py     ┐
    ├── optimizer.py    ├ 規則式：各自的目標函式
    ├── builder.py      │
    └── hunter.py       ┘ 規則式 + 前三手開局定式（繼承優化者）
```

**`ai/` 不 import `rl/`、`engine`、`numpy`、`torch` 或 `pygame`。** 獵手的定式因此
在 `ai/heuristics/hunter.py` 內重新實作，而不是從 `rl/opening.py` 匯入 —— 那會把依賴方向
反過來並把 numpy 拉進 `ai/`。代價是同一本定式在樹上有兩份實作，由
`tests/test_hunter.py` 交叉比對（96 次局面 × 書本步）把關。

### 分層原則

由下往上：

- **`formulas.py`** 只回答「把一個落子換算成數字是多少」。它不認識任何人格。
- **`base.py`** 定義 `Brain` 介面。候選枚舉、對手預判、抽籤**不在**這裡。
- **人格模組** 只負責三件事：

  ```python
  context(board, hand_names, owner, must_cover, reach)  # 算一次的前置量
  restrict(cands, ctx)                                  # 階段（軟）過濾
  rescore(cands, ctx)                                   # 重新排分
  ```

- **`chooser.py`** 跑完整條流水線，但只認 `Brain` 的三個 method，不認人格。
  所以拆檔不改變任何一手的結果。
- **`registry.py`** 是對照表，也是新增人格時要改的地方；它同時載入
  `registry.json`，見下一節。

### `ai/registry.json`：名冊（資料層）

`ai/registry.py` 一個檔案兩層，回答兩個不同的問題：

| 層 | 檔案 | 回答什麼 |
|---|---|---|
| 上層 | `ai/registry.py` 的 `WEIGHTED_SPECS` / `RULE_BRAIN_CLASSES` | 這一手棋**怎麼評分**（JSON 裝不下 `Brain`） |
| 下層 | `ai/registry.json` | **誰是選手**：key、順序、別名、label、desc_key、pools、enabled、selectable |

`seats.automated_options()` 與 `seats.ALIASES` 讀下層，所以「換對手池」是改資料不是
改程式碼。**健康檢查的指令是 `python -m ai --check`**（不是 `python -m ai.registry
--check`）：`ai/__init__.py` 匯出 `ai.registry`，跑子模組會把同一個模組執行兩次，
runpy 會先噴一條 `RuntimeWarning`；掛在套件上就沒有這個問題。指令會印出四個池的
內容並檢查權重檔。

**欄位與權責**

| 欄位 | 誰說了算 |
|---|---|
| `key`、`aliases`、`pools`、`enabled`、`selectable`、`label`、`desc_key` | **JSON** |
| `checkpoint`（**從哪裡載入**，階段 4 起） | **JSON**；`seats.rl_checkpoint` / `seats.imitation_checkpoint` 只是它的讀者 |
| `source`（**在哪裡訓練**） | JSON 記錄，權威仍在 `seats.RL_SEATS` / `seats.IMITATION_CHECKPOINT_DIR` |
| `module`、`kind` | 上層／`seats`，JSON 同步一份 |
| `sha256` | **JSON**，由 `check_files()` 對實際檔案比對 |

路徑權威已在**階段 4** 移交 JSON（見下面的「階段 4 完成狀態」）。
`tests/test_registry_json.py` 逐欄比對，任何一邊單獨改動都會失敗而不是靜默分歧。

**驗證分兩層，時機不同：**

| 檢查 | 什麼時候跑 |
|---|---|
| 結構：key 唯一、別名不衝突、pools 名稱合法、`label`／`desc_key` 在 `config.I`、`enabled` 與 `selectable` 一致 | **import**（失敗就是模組載入失敗） |
| 檔案**存在** + **sha256** | **測試** 與 **`python -m ai --check`** |

`enabled` 與 `selectable` 文案上是兩個問題（進常規池 vs. 出現在 UI 與隨機抽籤），
但它們今天切的是同一組十四個 key 的兩種視圖，所以**兩個必須一起翻**；不一致會在
import 就被拒絕，錯誤訊息指名那個 key。

**`anchors()` 目前只有測試呼叫**。它回傳 `("rl_h1000_0k",)`，但 `rl/rl_train.py`
還自己寫死 `hc_1000`：那是 `data/rl1/eval.jsonl` 的列名，改了會把同一條基線切成
兩欄。RL 端要不要改接這個函式，是後續階段的決定。

**池的順序 = 依 key 排序**（plan9a 階段 3）。排序規則只有一處 ——
`ai.registry.pool_order()`，見 `## 測試` 底下的「池的順序」。註冊表陣列順序
因此不再是任何行為的一部分：重排 `ai/registry.json` 的列不會動到任何一場對局。

### 階段 4 完成狀態：權重搬進 `ai/checkpoints/`

**只搬了兩個。** 依裁決 B2-(甲)，`rl_o1000_0k` / `rl_b1000_0k` / `rl_i1000_0k`
還沒有註冊（它們要到階段 6 才進名冊），所以「`ai/checkpoints/` 內無未登記權重」
這條驗證要求**先搬已註冊的兩個**；三個學生等階段 6 註冊時一起搬。

| key | 來源（`source`，仍在 `data/`） | 載入路徑（`checkpoint`，已入版控） | sha256（來源 == 目的地） |
|---|---|---|---|
| `rl_h1000_20k` | `data/rl1/step_000040.pt` | `ai/checkpoints/rl_h1000_20k/step_000040.pt` | `e1dc6230…4801636` |
| `rl_h1000_0k` | `data/hc2/step_001000.pt` | `ai/checkpoints/rl_h1000_0k/step_001000.pt` | `fc0052d3…cd11b74` |

複製前後逐檔 `sha256` 比對通過才繼續；`hc_2000` / `hc_10000` 依裁決 B **不搬**，
`checkpoint` 仍是 `data/hc2/step_002000.pt` / `step_010000.pt`。

**`.gitignore` 是過渡狀態，不是終點。** 已撤銷 `!data/hc2/step_001000.pt` 與
`!data/rl1/step_000040.pt`（並 `git rm --cached`，檔留在磁碟上），所以那兩份
`data/` 副本不再入版控 —— 乾淨 clone 靠 `ai/checkpoints/`。
`!data/hc2/step_002000.pt` 與 `!data/hc2/step_010000.pt` **保留**：這兩個檔是
`hc_2000` / `hc_10000` 僅有的權重，撤了就會讓乾淨 clone 連 `--pool hc_2000` 都開
不了。**階段 6 把它們設成 `enabled: false` 之後要回來處理這兩行**，那時它們既不
在常規池也不該佔版控。

**行為零變更的證據**：`HEAD` 與階段 3 的 `63eb992` 各跑一次
`run_league(2, seed=20260928)`，輸出逐字相同（md5
`8166309cdcddb098bdc4c5b8233d6fcc`，`diff` 無輸出）—— 改的是檔案從哪裡來，
不是檔案是什麼。同一次比對也記錄了真正被開啟的權重檔：兩棵樹只差
`data/hc2/step_001000.pt` → `ai/checkpoints/rl_h1000_0k/step_001000.pt`，其餘
逐位元相同。

### 選棋流水線

```
枚舉合法候選（每種棋塊 × 每個方向 × 每個落點）
   ↓
用權重粗評分排序                          ← 這段是所有 AI 共用的初篩
   ↓
brain.context / restrict / rescore        ← 人格接手
   ↓
對手預判（扣掉對手能得到的最好分）          ← 只有權重式，且預算內
   ↓
短名單（前 34 名 + 最多 6 個搶角候補）
   ↓
依 mistake_rate 抽籤（softmax）
   ↓
回傳 (棋名, 方向, x, y)
```

**階段過濾為什麼是「軟」的**：如果做成硬過濾，某個階段篩不到東西時候選清單會
被清空，`choose_move` 回傳 `None`，呼叫端就會把玩家誤判成無棋可下而自動跳過。
軟過濾在篩不到時退回原本的候選清單，所以 `choose_move` 只會在真的沒棋可下時
回傳 `None`。

### 相容性

`ai/__init__.py` 把所有舊名字重新匯出，所以 `import ai` 之後
`ai.choose_move`、`ai.ODIRS`、`ai.IntruderBrain`、`ai._opponent_pool` 之類的舊
寫法都照舊可用。`game.py`、`match.py`、`ui.py` 和全部測試都不需要修改。

---

## 位元運算的關鍵設計

20x20 = 400 格，用**一顆 400-bit 的整數**表示。整顆盤面就是一個 bitmask，所以
「這一手合法嗎」「落子後長什麼樣」都是幾個大整數的位移與求交，不用枚舉格子。

```python
placed = od["m"] << base        # 這一手占掉哪些格
legal  = need & empt & ~avoid   # 落子後自己還能下的位置
```

### 方向資料預計算

每一種「棋塊 × 旋轉方向」在載入時就算一次（`formulas._build_od`），記住：

| 欄位 | 內容 |
|---|---|
| `m` | 這一種方向占掉的格子位元 |
| `offs` | 每格相對左上角的位移 |
| `bases` | 所有不會出界的落點 |
| `valid` | 上述落點的位元 |
| `csum` | 每個落點的中心親近度總和 |
| `corner` | 每個落點占掉哪些角位 |
| `touch` | 會碰到的 2x2 方塊（跨越判定用） |
| `pair_shifts` | 棋塊內部角對角相鄰的兩格所形成的跨越位移 |
| `neutral` | 前 20 個「中性」落點（角位與盤心加權），對手預判先掃這個 |

21 種棋塊、89 格、每種最多幾十種方向，總共只算一次。

### 交點語言

角對角規則用「交點」講比用「格」講清楚得多：兩格共邊 ⇔ 共兩個交點，角對角
相接 ⇔ 共一個交點。20x20 的格子因此投影到 21x21 的交點網格（441 bits），
跨越的判定就是「某個 2x2 的兩個對角，一個全滿、一個有對手」。

### 跨越的兩種來源

`formulas` 裡 `cross_anchors` 與 `pair_shifts` 分別處理：

1. **一格既有棋 + 一格新棋**角對角組成跨越那條對角線
2. **兩格都屬於同一手**（棋塊內部就角對角相鄰）——這是為什麼每種方向要預先算
   `pair_shifts`

### 成本上限

AI 要在幾百毫秒內對 21 種棋塊、數千個候選做決策，所以每一層都有上限：

| 上限 | 值 | 作用 |
|---|---|---|
| `WALL_BUDGET` | 0.9 秒 | 單手思考時間，超過就跳過對手預判 |
| `SIM_EVAL_BUDGET` | 3000 | 模擬落子次數 |
| `OPP_POOL_K` | 40 | 每個對手評幾個應手 |
| `INTRUDER_SCAN` | 1200 | 入侵者昂貴評估的候選上限 |
| `OPTIMIZER_SCAN` | 1200 | 優化者同上 |
| `BUILDER_SCAN` | 800 | 築城者同上（真的會擋到東西） |
| `REGION_CAP` | 120 | 角位連通區塊大於此就不算加成 |

---

## 排行榜資料

`records.py` 把每局結果存成 `records.json`：

```json
{
  "optimizer": {
    "games": 326,
    "total_points": 1091.0,
    "total_remaining": 3176.0
  }
}
```

只存累積值，均值在讀取時算，所以資料格式不受顯示精度影響。載入與存檔都包在
`try` 裡：檔案損壞時當作沒有紀錄，不會讓遊戲開不起來。

**排行榜以人格為 key，不是以座位為 key**：三個 AI 席位每局拿到的性格會換，
用座位當 key 會把狼和狐狸的紀錄混在一起。`Game.setup_match` 檢查四個人格必須
互不相同。

**別名與正規名（plan9a 階段 9）**：寫入端一律先正規化 —— `records.py` 的
`record` 與 `set_meta` 把 key 經 `seats.canonical_key` 換成正式名，所以用
`hc_1000` 打出來的結果只會長出 `rl_h1000_0k` 列，不會另開一列；正規化在排名
**之前**，同名兩席的名次也用正式名算。未知 key（`human`、`player`、手打的
名字）原樣通過，不篩不猜。

**讀路徑完全不動**：改名前就存在的 `hc_1000` 舊列原樣保留，與 `rl_h1000_0k`
新列**並存、數字各自獨立**（`test_a_records_row_under_the_old_key_still_reads`）。
兩列在排行榜上會顯示同一個中文名（`seat_label` 解析別名，見
`test_a_leaderboard_row_under_the_old_key_is_named_in_chinese`），所以要分辨
得看 `records.json` 的 key。**程式永遠不會自動合併**：想合併就手動編輯
`records.json`，把舊列的 `games`／`total_points`／`total_remaining` 三欄加進
正式名列，再刪掉舊列。

**literal 批次是唯一保留原拼法的地方，而且只保留到批次檔**：批次檔的 `pool`
欄記當初怎麼拼（逐字重現要用），排行榜寫入端照樣正規化 —— 同一條指令
`--pool hc_1000 --pool-order literal` 寫出的批次檔是 `hc_1000`，排行榜長的是
`rl_h1000_0k`（`test_a_literal_run_keeps_the_batch_spelling_and_normalises_the_leaderboard`）。

`records.json` 和 `settings.json` 都由程式在執行時自動生成，已經加進
`.gitignore`，不會上傳到 GitHub。每台機器、每個玩家各自一份。

---

## 測試

```bash
.venv-rl/bin/python -m pytest tests/ -q -n 8    # 601 項，約 160–175 秒
```

**不要用 `-n 12` 或更高。** `test_rl_env.py` 有模組層級的快取，每個 worker 各自
建一份，worker 越多總 CPU 工作量越多：實測 `-n 8` 159–175 秒，`-n 12` **217.66 秒**。
完整理由見 `AGENTS.md` 第六節。

**量時間或除錯時用 `-n 0` 或不加 `-n`** —— 有五處測試斷言執行時間，在並行下量出
來的數字沒有意義。

### 池的順序：一律依 key 排序

**排序規則只有一處**：`ai.registry.pool_order()` —— 對 key 做純字串 `sorted`，
不看 locale、不做大小寫折疊、也不對 `hc_1000` / `hc_10000` 裡的數字做自然排序。
`ai.registry.automated_options()` 與 `ai.registry.pool()` 都走它，而
`seats.automated_options()` → `seat_options` / `seat_menu_options` /
`resolve_random_ai` → `match._ORDER` / `POOL_PRESETS` / `expand_pool` /
`league_options` 全是這兩個函式的下游，所以整個專案只有一份順序規則。

**為什麼是 key 而不是註冊順序**：`rng.choice` 抽的是 index，池的順序就是每個同種子
對局所依賴的 index→key 對應。註冊順序是 `ai/registry.json` 的屬性，改名册就會改；
key 排序是**集合**的函式，只有集合變了才變。階段 3 之前兩者恰好相同，所以重排
JSON 陣列等於重排每一場已提交的對局。

**`--pool-order`**（`match.py`）：

- `sorted`（預設）：上面那條規則。日常對局與所有評測都用這個。
- `literal`：照 `--pool` 原樣 —— **不排序、不去重、不看 `enabled`、連拼法都保留**。
  **只用來逐字重現階段 3 之前提交的歷史證據**，日常評測不該使用它。
  必須搭配明確的 `--pool` 名單，且不得混入 preset 名稱（preset 的順序就是註冊表的
  順序，正是 literal 假定不存在的東西）；這種組合會以用法錯誤（exit 2）擋下。
  literal 下 `--subject` 也不會被正規化，好讓重現出來的 `games_detail` 與舊批次
  逐字相同，不必先做任何字串對照。這只管**批次檔**；排行榜寫入端在 plan9a 階段 9
  之後一律正規化，兩者互不影響（見「排行榜資料」）。

已提交的 0.5b 批次在 literal 上的重現方式（`eval/plan9/README.md` 有完整說明）：

```bash
.venv-rl/bin/python match.py \
    --pool wolf,chess,fox,intruder,optimizer,builder,hunter \
    --pool-order literal \
    --games 2000 --seed 20261005 --paired-rng --subject hc_1000 \
    --mode argmax --gzip --out <輸出.json>
```

### GOLDEN 重採（plan9a 階段 3）

`tests/test_match_pool.py` 的 `GOLDEN` 釘住三個種子各兩局的
`(option, colour, remaining)`。池改成 key 排序後 `rng.choice` 抽到不同的 index，
三個種子**全部變動**。這是**純排序變更**：十一個 key 一個都沒增刪，
`tests/test_registry_json.py::test_the_json_rows_are_unchanged_and_the_pool_is_them_sorted`
把 JSON 陣列的原樣釘住，就是這句話的證據。

| | commit |
| --- | --- |
| 舊值（註冊表順序） | `1789b46` |
| 新值（key 排序） | 本提交（訊息 `Sort every pool by key, ...`） |

舊值：

```
20260928  [('rl_h1000_20k','red',14), ('hc_2000','yellow',9), ('optimizer','blue',0),
           ('wolf','green',12), ('chess','yellow',35), ('rl_h1000_20k','blue',0),
           ('hc_2000','green',12), ('hc_2000','red',7)]
7         [('builder','green',4), ('fox','red',25), ('hunter','yellow',12),
           ('rl_h1000_20k','blue',15), ('rl_h1000_20k','green',18),
           ('builder','yellow',4), ('wolf','blue',32), ('rl_h1000_0k','red',16)]
99        [('hunter','yellow',12), ('hunter','red',32), ('intruder','blue',15),
           ('hc_10000','green',16), ('fox','yellow',38), ('builder','blue',0),
           ('rl_h1000_0k','red',13), ('wolf','green',13)]
```

新值：

```
20260928  [('wolf','red',23), ('rl_h1000_0k','yellow',12), ('hc_2000','blue',17),
           ('builder','green',13), ('wolf','yellow',43), ('chess','blue',16),
           ('hunter','red',31), ('hunter','green',9)]
7         [('hunter','green',0), ('fox','red',34), ('intruder','yellow',8),
           ('wolf','blue',27), ('hc_10000','yellow',3), ('fox','red',38),
           ('chess','blue',36), ('fox','green',26)]
99        [('intruder','yellow',13), ('intruder','red',21), ('hc_10000','blue',11),
           ('rl_h1000_20k','green',0), ('chess','yellow',39), ('fox','red',15),
           ('hc_2000','green',18), ('rl_h1000_20k','blue',4)]
```

同時 `GOLDEN_POOL` 改成測試內**寫死的明確清單**，不再從註冊表推導 ——
GOLDEN 是證據，證據要靠「名册改了它就紅」來失敗，而不是跟著名册一起動。
兩者唯一允許相遇的地方是
`test_the_pinned_pool_is_still_the_registry_s_pool`。

### GOLDEN 重採（plan9a 階段 6）

註冊三個 1000 步學生（`rl_o1000_0k` / `rl_b1000_0k` / `rl_i1000_0k`）之後，
`all` 池從 **11 列變 14 列**，`rng.choice` 抽的是 index，所以排序位置之後的
每一個 key 都讀到不同的 index，三個種子全部改變。這是**組成變更**，不是排序
或改名 —— 與階段 3 不同，這次是真的多出三個 key。

| 池 | 階段 6 前 | 階段 6 後 |
|---|---|---|
| `all` | 11 | 14 |
| `imitation_only` | 3 | 6 |
| `no_imitation` | 7 | 7 |
| `rl_only` | 1 | 1 |

`hc_2000` / `hc_10000` **維持在池內**（使用者裁決），所以是 14/6 而不是
已定決策寫的 12/4 —— 退出常規池與 `.gitignore` 兩行留在
[plan9a 的「待清理項目」](plan9a-ai.md)。

| | commit |
| --- | --- |
| 舊值（十一 key 池） | `63eb992` |
| 新值（十四 key 池） | 本提交（訊息 `Register the three 1000-step students, ...`） |

舊值：

```
20260928  [('wolf','red',23), ('rl_h1000_0k','yellow',12), ('hc_2000','blue',17),
           ('builder','green',13), ('wolf','yellow',43), ('chess','blue',16),
           ('hunter','red',31), ('hunter','green',9)]
7         [('hunter','green',0), ('fox','red',34), ('intruder','yellow',8),
           ('wolf','blue',27), ('hc_10000','yellow',3), ('fox','red',38),
           ('chess','blue',36), ('fox','green',26)]
99        [('intruder','yellow',13), ('intruder','red',21), ('hc_10000','blue',11),
           ('rl_h1000_20k','green',0), ('chess','yellow',39), ('fox','red',15),
           ('hc_2000','green',18), ('rl_h1000_20k','blue',4)]
```

新值：

```
20260928  [('rl_h1000_20k','green',0), ('rl_b1000_0k','red',18),
           ('hc_2000','yellow',12), ('rl_i1000_0k','blue',9), ('hunter','green',14),
           ('rl_h1000_0k','blue',8), ('hc_10000','yellow',10),
           ('rl_i1000_0k','red',12)]
7         [('hunter','green',25), ('fox','red',26), ('intruder','yellow',5),
           ('rl_h1000_20k','blue',5), ('rl_h1000_20k','blue',11),
           ('wolf','green',25), ('optimizer','red',21), ('hc_2000','yellow',4)]
99        [('intruder','yellow',19), ('intruder','red',9), ('hc_10000','blue',20),
           ('rl_h1000_0k','green',16), ('builder','green',30), ('intruder','red',10),
           ('hunter','blue',12), ('fox','yellow',21)]
```

**「組成是唯一變動來源」的證據**（不只留在 `/tmp`）：把池釘住不動，
改動前後各跑一次 `run_league(2, seed=20260928, records=None)`，輸出逐位元相同 ——
同一支探測腳本、同一個種子、只換工作樹：

| 釘住的池 | 改動前 md5 | 改動後 md5 | `diff` |
|---|---|---|---|
| `["hunter","optimizer","builder","intruder"]` | `92a99b6ade2794ef130e341f57cf3534` | `92a99b6ade2794ef130e341f57cf3534` | 無輸出 |
| `no_imitation`（七人格） | `97403a9ff49612e2ff79f0ce5467aecb` | `97403a9ff49612e2ff79f0ce5467aecb` | 無輸出 |

釘住池不變 = 非組成路徑逐位元未動；`GOLDEN` 的變化全部來自池本身。
`tests/test_match_paired_rng.py` 的 `LEGACY_GOLDEN_MD5`
（`3fa9fba063a5f1cd3868098454b05b4c`）同樣釘住四個人格，**未改動**。

### 依主題

| 檔案 | 覆蓋 |
|---|---|
| `test_ai.py` | 合法性、風格比例、單手速度、對手候選池的退場路徑 |
| `test_intruder.py` | 五條規則逐條驗證（跨越偵測、關鍵格、階段順序） |
| `test_optimizer.py` | 緊急規則、大棋優先 |
| `test_builder.py` | 閉包、活區塊、裝箱的形狀 |
| `test_hunter.py` | 獵手：與 v3 全盤等價、定式交叉比對、放棄規則、種子可重現性 |
| `test_simulation.py` | 整局跑完、逐手驗證規則、人格覆蓋率、獵手前三手必須來自定式 |
| `test_board.py` | 角對角規則、連通區塊、邊界 |
| `test_pieces.py` | 旋轉生成 |
| `test_rotation.py` | 棋盤的旋轉對稱性，以及遊戲規則是否與旋轉相容 |
| `test_engine_cross.py` | `engine.py` 與 `Game` 在整局中逐步一致 |
| `test_geometry_reference.py` | 開局角位規則對照暴力搜尋 |
| `test_stuck_monotone.py` | 沒有合法步的玩家永遠不會恢復 |
| `test_optimised_paths.py` | 最佳化路徑與樸素實作一致；關掉掛鐘預算後結果與機器負載無關 |
| `test_trace.py` | 選項傳入的 trace 不影響引擎 |
| `test_benchmark.py` | benchmark 可重現，排名與專案自己的排名一致 |
| `test_bench_engine.py` | F' 階段的量測腳本 |
| `test_match.py` | 聯賽流程 |
| `test_records.py` | 排名與並列處理 |
| `test_plane.py` | 局面轉成走子者視角的數字 |
| `test_ui_smoke.py` | 介面狀態機煙霧測試 |
| `test_rl_isolation.py` | 匯入圖是契約的一部分（乾淨子進程檢查） |
| `test_rl_actions.py` | 動作編碼與引擎的合法步清單一致 |
| `test_rl_env.py` | 環境只回傳合法步，並在終局給出空集合 |
| `test_rl_features.py` | 特徵與引擎的視圖逐位元相同 |
| `test_rl_normalize_probe.py` | 對 `engine.normalize` 的一個懷疑，留下紀錄 |
| `test_rl_opening.py` | 開局定式對自己的宣稱 |
| `test_rl_paired.py` | 兩臂成對實驗對自己的宣稱 |
| `test_rl_paired3.py` | 三臂成對實驗對自己的宣稱 |
| `test_rl_v3.py` | 差分評分：公式、並列規則、耗時比 |
| `test_rl_collect.py` | 資料集可重現、可重建 |

測試都是從**真實對局位置**出發（用 `Game.act` 走出來），不是把棋塊硬放上去：
角對角規則意味著 owner 0 必須先開在自己的角上，否則 `reach` 對所有候選都是錯的。

---

## 新增一種 AI

1. 選一類：

   **權重式**（只需要不同的取捨傾向）——建 `ai/heuristics/<name>.py`：

   ```python
   from .base import WeightedBrain

   KEY = "your_key"
   # (w_corner, w_center, w_block, w_defend, w_open, w_large, mistake_rate)
   PROFILE_SPEC = (2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.1)


   class YourBrain(WeightedBrain):
       """一句話描述這個風格。"""

       key = KEY
   ```

   **規則式**（有自己的目標函式）——建 `ai/heuristics/<name>.py`，繼承 `Brain`，實作
   `context` / `restrict` / `rescore`，並設 `uses_lookahead = False`、
   `self.mistake_rate = 0.0`。

   **帶開局定式的規則式**（像獵手）——繼承既有的規則式類別，把定式寫成
   `restrict` 的候選過濾加上 `rescore` 只回傳抽中的一個候選。用這種寫法而不是在
   選棋前直接回傳，是因為 `ui.py` 與 `match.py` 會直接呼叫 `ai.choose_move()`，
   那兩個檔案不該為了某個人格而改。**注意：`choose_move` 只在短名單有兩個以上
   候選時才從呼叫方的 rng 取一個數，所以「只回傳一個候選」正是讓定式步不消費
   遊戲隨機流的原因。**

2. 在 `ai/registry.py` 登記（`WEIGHTED_SPECS` 或 `RULE_BRAIN_CLASSES`）。
3. 在 `ai/registry.json` 的 `entries` **最後**加一列（`key`／`kind`／`family`／
   `module`／`pools`／`label`／`desc_key` 等）。**漏加或加在中間**都會被
   `tests/test_registry_json.py` 擋下來：名冊與類別表必須是同一份清單、同一個順序，
   因為順序就是抽籤順序。
4. 在 `ai/__init__.py` 匯出。
5. 在 `config.py` 的 `I` 字典加上中文名與描述，並加進 `PERSONALITY_ORDER`。
6. 寫測試。

改動人格順序會影響 `personality_keys()` 的抽籤順序，進而改變既有種子下的對局
結果。要保持可重現性就把新的人格加在最後。

**加入一個人格會連帶改動三處既有斷言**，因為池的大小與輪空數都變了：
`test_match.py` 的 `len(pool) == 6` → `7`、輪空 `2` → `3`，以及
`test_builder.py` 的 `PERSONALITY_ORDER[-1] == "builder"`（若新人格加在最後就會失效）。
這些是「預期必改」，不是回歸。

**但 RL 那邊不是「改測試」這麼簡單**：`rl/collect.py` 的 `controller_weights()`
由人格清單推導，加一個人格會讓非老師的權重從 `0.5/6` 變成 `0.5/7`
（`rl/collect.py:135-138`）。那改變的是**RL 資料收集的行為**，不只是測試變紅 ——
先決定那張權重表跟不跟，再動手。

## 如何新增或更換對手（網路席位）

**沒有發布命令**（使用者裁決 2026-10-08）：複製權重、算雜湊、寫草稿列，全部手動。
四步：

1. **複製權重並比對雜湊**。目錄名就是 `key`，檔名必須是 `step_%06d.pt`
   （`imitation_step` 從檔名讀步數，不是 step 的表）：

   ```bash
   mkdir -p ai/checkpoints/<key>
   cp <來源檔> ai/checkpoints/<key>/step_<NNNNNN>.pt
   sha256sum <來源檔> ai/checkpoints/<key>/step_<NNNNNN>.pt   # 兩行必須相同
   ```

2. **先補 `config.py` 的 `I` 文字，再手寫 `ai/registry.json` 的草稿列**。

   **先做 `config.I` 這一步**：label 與 desc 各加一個**新的** key（名字隨
   慣例，例如 `imit_o_fmt` / `imit_o_desc_fmt`）。選單要求每個 selectable
   席位的名字**互不相同**——沿用別人的 label 會被
   `test_every_selectable_option_has_a_name_and_a_description_of_its_own` 擋下
   （虛構人格實測抓到過）；desc 只寫可驗證的事實（步數、老師），不要寫比較。

   然後才是草稿列，起手 `enabled: false` / `selectable: false`
   （兩者必須一致，`_validate` 會擋），其餘欄位：

   | 欄位 | 填什麼 |
   |---|---|
   | `key` / `aliases` | 與目錄同名；舊名放 `aliases`，**不要**再加一列 |
   | `kind` / `family` | `network` + `imitation` 或 `rl` |
   | `checkpoint` / `source` | `ai/checkpoints/...`（發布後）／來源路徑 |
   | `sha256` | 第 1 步的**完整**值 |
   | `pools` | 先照家族填，確認後再加 `all` |
   | `label` / `desc_key` | **指向剛剛在 `config.I` 補的那兩個 key** |
   | `note` | 英文維護註記；不進 `--list`、不是畫面上的文字 |

3. **驗證，過了才改成 `true`**：

   ```bash
   python -m ai --list     # 逐列：key/kind/family/en-sel/pools/sha256 前 12 字元/file/aliases
   python -m ai --check    # 檔案存在 + 全量 sha256，再列出四個池
   ```

   兩個入口都掛在**套件**上：`python -m ai.registry --list` 會把同一個模組執行
   兩次並噴 runpy 的 `RuntimeWarning`（`--check` 同理，見 AGENTS「健康檢查的指令」）。
   列序是 `pool_order`（key 升冪），也就是抽籤真正讀的那一個順序。

4. **親自對戰體驗**，再決定要不要讓它進常規池（`pools` 的 `all`）。

## 新增一個 AI 之後仍需改的測試

plan9a 階段 8 把「池有多少個」全部改為與 `ai/registry.json` 比對，所以
**多數測試一行都不用動**。兩種家族各實測過一次，**情境不同、清單不同**：

### 實測一：imitation 家族的第 15 個 AI（`network` + `family: imitation`）

2026-10-08：加一列 `rl_fake_0k`（含 checkpoint、`enabled: true`）跑全套，
**紅 10 項**；其中 1 項是同一次改動裡一併轉換掉的組成性斷言，現存 **9 項**：

| 測試 | 為什麼會紅 | 性質 |
|---|---|---|
| `test_match_pool::test_pool_all_is_identical_to_no_pool_at_all`（3 個種子） | `all` 池變了，同種子的座位全變 | **GOLDEN 重採**：舊值/新值寫回上方「GOLDEN 重採（plan9a 階段 6）」同一格式 |
| `test_match_pool::test_the_pinned_pool_is_still_the_registry_s_pool` | `GOLDEN_POOL` 是字面清單 | **證據**：名册改了就必須紅，這正是它存在的理由 |
| `test_match::test_only_imitation_options_can_be_asked_for` | 字面列出六個 imitation key | 證據：哪些 key 屬於 imitation 家族 |
| `test_registry_json::test_the_json_rows_are_unchanged_and_the_pool_is_them_sorted` | 「前 11 + 後 3」的逐列 pin | 證據：名册原樣，重排與新增都看得見 |
| `test_seat_alias_hc::test_the_rename_left_every_other_seat_where_it_was` | 字面十四個 key 的集合 | 證據：改名不增不減 |
| `test_seats::test_every_option_is_one_of_the_three_kinds_of_contestant` | 「四個席位在 step 1000」的字面步數清單 | 證據：哪些席位在第 1000 步 |
| `test_ui_smoke::test_every_selectable_option_has_a_name_and_a_description_of_its_own` | 新席位沿用了別人的 label，撞上「名字互不相同」 | **不用改測試**：加一對自己的 `config.I` 文字就過 |
| `test_match_pool::test_the_imitation_preset_on_its_own` | `appearances` 要求「池內每個 key 都坐過」 | **本輪已轉換**：改為 containment（只保證不出池），15 鍵下單獨 `1 passed` |

### 實測二：heuristic 人格的第 8 個人格（`kind: heuristic` + `family: personality`）

2026-10-08：照「新增一種 AI」的完整流程加一個虛構人格 `fake`——
`ai/heuristics/fake.py`、`registry.py` 的 `WEIGHTED_SPECS`、`ai/__init__.py`
匯出、`config.I` 兩個新 key、`PERSONALITY_ORDER`、`registry.json` 一列。
兩處**位置**有講究，插錯會立刻紅（都是測試在做事，不是測試的錯）：

- `registry.json` 的列要放在**類別表順序**（`wolf, chess, fox, **fake**, intruder, ...`），
  不是「加在最後」——我第一次插在 `wolf` 之後，
  `test_the_json_lists_the_personalities_in_the_class_tables_order` 就紅了；
- `config.PERSONALITY_ORDER` 同理，`test_builder.py` 釘住
  `PERSONALITY_ORDER == ai.personality_keys()`。

位置正確之後跑全套：**14 failed / 1002 passed / 639.72s**：

| 測試 | 為什麼會紅 | 性質 |
|---|---|---|
| `test_match_pool::test_pool_all_is_identical_to_no_pool_at_all`（3 個種子） | `all` 池多了一个人格，同種子座位全變 | **GOLDEN 重採** |
| `test_match_pool::test_the_pinned_pool_is_still_the_registry_s_pool` | `GOLDEN_POOL` 是字面 14 列 | **證據**：名册改了就紅 |
| `test_registry_json::test_the_json_rows_are_unchanged_and_the_pool_is_them_sorted` | 逐列 pin（`key` 欄；裁決 10 之後另含 `kind`／`family`） | **證據** |
| `test_seat_alias_hc::test_the_rename_left_every_other_seat_where_it_was` | 字面十四 key 的集合 | **證據** |
| `test_match::test_setup_match_seats_four_distinct_personalities` | `len(PERSONALITY_ORDER) == 7` | **證據**：人格數 |
| `test_hunter::test_the_coverage_type_conditions_hold_with_seven_keys` | `len(pool) == 7` 與函式名 | **證據**：人格數 |
| `test_rl_rollout::test_the_personality_pool_is_the_seven_named_in_the_plan` | 字面七個人格 + 明文要求「先決定要不要讓新人格進 RL 對手池」 | **設計上的決定點**，不是壞掉的測試 |
| `test_rl_collect`（4 項） | `rl/collect.py` 的 `controller_weights()` **由人格清單推導**，非老師的權重從 `0.5/6` 變 `0.5/7`，而測試釘住 `0.5/6` 與「7 個 key」 | **真連帶**：加一個人格會改掉資料收集的權重表，要先決定那張表跟不跟 |
| `test_match_pool::test_the_same_run_without_dry_does_write_the_leaderboard` | 固定種子下 8 選項的抽籤不再抽中 `wolf`，`appearances["wolf"]` 讀不到 | **種子相關的組成**：要嘛改取值方式，要嘛換種子 |

### 兩次實測的共同結論

**會紅的是證據（GOLDEN、逐列 pin、字面集合）、名字（`config.I`）、
以及「人格清單的連帶表」（`rl/collect` 的權重）**，不是池大小——後者已全部
改由註冊表驅動。純粹新增一個 `network` 席位不會碰 `rl/collect`；新增一個人格
會，這是兩份清單的差別，不是測試寫錯。

---

## 歷史證據與已知事項

以下每條都是**抄錄既有文件的原文並標出處**，不另加評價；原文以
`eval/plan9/README.md`、`plan9a-ai.md` 與本檔既有章節為準。這四項（0.063、
i 過弱、o 未播種、`imitation_only` 不一致）就是 plan9a「README 必須記錄」的
第 1、2、3、5 項；寫入端的別名行為見「排行榜資料」，第 6 項（key 排序與
GOLDEN 重採）見本檔既有三節，索引在本節末。

### 0.063 間距門檻已廢止，過擬合改為只看 val 曲線

> **【註記,2026-10-06】上表第三列的 0.063 間距門檻已廢止**……**原文一字未刪**,
> 第一、二列照舊有效。

取代它的規則（同檔原文，表列三格合併抄錄）：

> **過擬合**:val loss 自其**最低點**起上升 **> 0.05**,且**連續兩個檢查點**都高於
>「最低點 + 0.05」

同節另記:「**0.05 是判斷值,沒有證據支持這個數字**」。

出處:`eval/plan9/README.md` §「事後修訂:0.063 間距門檻廢止,過擬合改為只看
val 曲線(2026-10-06)」、§「取代它的規則:只看 val loss」。

### `rl_i1000_0k` 的過弱門檻觸發，處置是停下回報

> **按「任一條觸發即停下回報」:停下,回報,不調整步數、不換資料、不補跑。**

門檻與實測（同檔原文表列，補上表頭分隔列）:

> | 規則 | 門檻 | 實測 | 結果 |
> | --- | --- | --- | --- |
> | **過弱** | argmax 平均分 **≥ 2.93** | **2.8250**(SE 0.0710) | **觸發** |

2.93 的依據（同檔 §「過弱:2.93 的依據」原文）:

> `hc_1000` 的 argmax 平均分是 3.125……0.20 的餘裕**約等於 2 個差值標準誤**

後續處置是使用者裁決:「i 過弱觸發由使用者人工對弈判定取代」(`plan9a-ai.md`
§「已定決策(使用者)」)。

出處:`eval/plan9/README.md` §「`rl_i1000_0k`:intruder 的 0k,過弱門檻觸發
(2026-10-06)」。

### `rl_o1000_0k` 的訓練曲線不可重放

> **沒有 `torch.manual_seed`。**……**所以 o 的初始權重無法還原,它的曲線無法重放。**
> b 與 i 的訓練**有**播種(`torch.manual_seed(20260903)`,與 `rl/train.py` 自己的
> `main()` 一致)

同節記有補播種後重跑與記錄值的差異:step 100 的 train loss **2.938606**(cuda)
對記錄值 **2.926840**。另外 `eval/plan9/` 只有 b、i 兩份逐局成對檔、**沒有 o 的**,
所以 o 那 200 局的統計也無法從 repo 重算(下表另列)。

出處:`eval/plan9/README.md` §「訓練程序與 o 的一處差異:torch 從未被播種」。

### 池的組成：`hc_*` 先留在常規池

plan9a「README 必須記錄」原第 4 項「舊三模仿者移出常規池(使用者決定)」**已被
2026-10-08 使用者裁決取代**:`hc_2000` / `hc_10000` **先裁留在常規池**(`all` 14、
`imitation_only` 6,見「GOLDEN 重採（plan9a 階段 6）」),移出改列 plan9a
「待清理項目」(a)。`hc_1000` → `rl_h1000_0k` 的改名與永久別名不變(見
「訓練策略名稱的兩段數字單位不同」)。

### `imitation_only` 組內規格不一致

池內六席的步數與來源不同，逐欄讀自 `ai/registry.json` 的 `checkpoint`／`source`：

| key | 步數 | 權重來源 |
| --- | --- | --- |
| `rl_h1000_0k` | 1000 | `data/hc2`（H-C2，老師 hunter） |
| `rl_o1000_0k` / `rl_b1000_0k` / `rl_i1000_0k` | 1000 | 各自 `data/imit_*`（optimizer / builder / intruder） |
| `hc_2000` | 2000 | `data/hc2` |
| `hc_10000` | 10000 | `data/hc2` |

即四個 1000 步席位來自三份不同訓練，與 2000／10000 步混編在同一個 preset 裡。

### 抽籤順序與 GOLDEN 重採（plan9a 必須記錄第 6 項的索引）

原因、舊值與 commit 都已寫在本檔三處：「池的順序：一律依 key 排序」、
「GOLDEN 重採（plan9a 階段 3）」、「GOLDEN 重採（plan9a 階段 6）」。
這裡只留索引，不重抄。

### `data/` 與 `ai/` 的分工

> `data/` 是工廠(gitignore、不動),`ai/` 是展示櫃(只放 active 對手,入版控)。

出處：`plan9a-ai.md` §「目標」第 4 點。實際規則在 `.gitignore`：`data/*` 整體
忽略，只反白 `hc_2000` / `hc_10000` 兩個權重（現況見「階段 4 完成狀態」）。

### `hc_2000` / `hc_10000` 在乾淨 clone 的現況（plan9a 階段 9 的已知取捨）

兩者權重在 `data/hc2/`，靠 `.gitignore` 的 `!` 反白**入版控**，所以乾淨 clone
**目前可跑** `--pool hc_2000` 與 `--pool hc_10000`。plan9a 待清理項目 (a) 若
執行（撤兩行反白 + `git rm --cached`），兩者會退出版控，屆時依賴它們的證據只剩
本檔與測試中的數字 —— **可讀、不可重跑**。

### 逐批可重現性核對（2026-10-08 實測）

plan9a 階段 9 要求逐批核對「是否有歷史結果因改名或搬遷而無法重現」。下表每列
都是當日實跑的輸出，不是預期。後兩欄分開回答兩件事：

- **可執行** —— 能用現行程式跑同一份權重（權重在、名稱能表達）。**只表示程式與
  權重齊備，不代表已實際執行過** —— 有沒有真的跑、跑出來比沒比，看右欄。
- **數字逐位重現** —— 重跑出的數字與原報告**逐位相同**，或有等價的逐位比對證據。
  本次**沒有**實際跑出 `reports/hc2` 的四組再比對數字，凡未做這一步的一律寫
  「未驗證」：權重相同、名稱能改寫，都不等於數字一致。

| 證據 | 實測 | 可執行 | 數字逐位重現 |
| --- | --- | --- | --- |
| `eval/plan9/` 六批 0.5b（各 2000 局，種子 `20261005`） | 逐檔讀取：`pool_order` 欄**不存在**（階段 3 之前產出）；`subject` `hc_1000` → `rl_h1000_0k`、`hunter` → `ai.heuristics.hunter`、`rl_h1000_20k` → 自身；兩個網路權重 `exists=True sha_ok=True`（`ai/checkpoints/…`） | **是** —— 走 `--pool-order literal`，權重 sha256 對上 | **前 40 局逐位一致**（`test_literal_replays_a_batch_committed_before_the_sort` 比對 `games_detail[:40]` 與提交批次逐字相同）；**其餘 1960 局未重跑比對**，以批次檔為準 |
| `rl_b1000_0k-pair.json` / `rl_i1000_0k-pair.json` | md5 `394d9d1c55e82569f4016ad201ac6672` / `ff777cb422d1f7e8ba8cfdd173bd9eb5`，與 `eval/plan9/README.md` 記錄逐字一致；subject 皆為已發布 key | **是**（權重已發布；成對機制在套件內；**僅指可重算，未實際執行**） | **未驗證** —— 只驗了逐局檔 md5 與記錄一致（檔案未變），本次**未重算統計數字** |
| `rl_o1000_0k` 的 200 局統計 | `eval/plan9/` **沒有** o 的逐局檔；訓練曲線未播種（上一節） | **是**（`rl_o1000_0k` 權重已發布，現行程式可載入；**僅指可另跑新批次，未實際執行**）；訓練曲線另不可重放 | **無法比對** —— 原逐局檔不在 repo，無從對照 |
| `reports/hc2_raw.json` 四組 5000 局（種子 910001–910004） | 四個 option 名 `step_2000`（兩組）、`step_5000`、`step_10000` 現在 `expand_pool(..., "literal")` **全部** `ValueError: unknown --pool item`（舊名已廢止，由 `test_match_pool` 釘住） | **原指令一律不可原樣跑**；逐組如下 | **四組本次都未跑出比對**；逐組如下 |
| ├ hc2 `step_002000`（910001） | 權重 `data/hc2/step_002000.pt` **已入版控**（`git ls-files --error-unmatch` 通過） | **是**（option 改寫成 `hc_2000` + literal 順序，同一檔案） | **未驗證** —— 未實際跑出並比對數字 |
| ├ hc2 `step_005000`（910002） | `data/hc2/step_005000.pt` **未入版控**；名冊沒有任何 key 指向 5000 步 | **否**：乾淨 clone 無權重，且沒有席位能表達它 | **無法重現**（不可執行） |
| ├ hc2 `step_010000`（910003） | `data/hc2/step_010000.pt` **已入版控** | **是**（option 改寫成 `hc_10000` + literal 順序） | **未驗證** —— 未實際跑出並比對數字 |
| └ hb2 `step_002000`（910004，舊 H-B2） | `data/hb2/step_002000.pt` **未入版控**；H-B2 席位已被 H-C2 取代，名冊無此 key | **否**：乾淨 clone 無權重，且沒有席位能表達它 | **無法重現**（不可執行） |
| `data/rl1/eval.jsonl` 的 `hc_1000` 欄 | 欄名按裁決**維持**（改名會把同一條基線切成兩欄，見 AGENTS）；對應權重 `rl_h1000_0k` 與 `rl_h1000_20k` 都已發布 | —（讀檔項目：可讀、可續寫，不改名） | 不適用 |
| `records.json` 舊列 | `test_a_records_row_under_the_old_key_still_reads` | —（讀檔項目：可讀、不重寫、不合併） | 不適用 |

本表收錄受改名或搬遷影響的證據；其他報告檔（`reports/h0`–`h1`、`g`、`f_prime`、
`hc2` 的結論文字）未逐批列入。

---

## 已知限制與取捨

- **`fillable_closure` 只是上界**，當排序 proxy 夠用，不拿來預測分數：
  (a) 沒有要求泛洪途中每一步都重新滿足角對角接觸；(b) 沒算對手接下來會佔掉多少；
  (c) 含斜對角步的棋塊（Z5 等）能跨過 4 鄰接不連通的地方，閉包會少算一點。
- **`_score_move` 在 `chooser.choose_move` 裡被重寫了一次**。候選枚舉是熱路徑，
  那裡為了省掉函式呼叫把算式內嵌了。兩處必須同步。
- **規則式人格看不到對手的意圖**。預判只用權重評分，所以規則式人格的
  `uses_lookahead` 是 `False`。
- **角位加成依賴連通區塊**。區塊大於 `REGION_CAP`（120）時直接不算加成，這是
  為了避免一開局就為了一小塊地做昂貴的 BFS。
- **權重有隨機擾動**。同一個人格每局權重差 ±30%，這是為了讓排行榜有意義，但
  也代表單局結果有隨機成分，要比較人格請跑足夠多局。
- **開局定式有兩份實作**（`ai/heuristics/hunter.py` 與 `rl/opening.py`），因為 `ai/` 不得
  import `rl/`。兩份由 `tests/test_hunter.py` 交叉比對，但它們仍可能各自演化 ——
  改其中一份而忘記另一份，測試會抓到，只是不會在你改的當下就抓到。
- **定式的種子決定前兩手走哪一個候選，而那個數在一般遊戲裡沒有對應的特徵**。
  特徵裡沒有種子，也沒有真實座位，所以模仿網絡在理論上無法學會「這一局該走
  哪一個候選」，只能學會「這兩格都是合法定式步」。
- **`rl/` 的實驗只吃 `engine.py`，不吃 `game.py`**。兩份規則必須逐步一致，靠
  `tests/test_engine_cross.py` 維持，不是靠結構保證。
- **`rows(order=...)` 的 `order` 從未生效，CLI 排行榜一律按平均分排序**。
  `match.py` 收尾印排行榜時把 pool 當 `order` 傳進 `records.rows(order=pool)`
  （literal 批次就是舊拼法的 pool），但 `rows()` 在 return 前**無條件**
  `out.sort(key=(-平均分, 餘格, key))`，連 key 都在排序鍵裡，中間「依 `order`
  組 key 清單」那一段永遠被蓋掉 —— 實測 `rows(order=[...])` 與 `rows()` 輸出
  逐項相同，且自 first release（`452000f`）即如此（`ui.py` 乾脆不傳 `order`）。
  所以 **literal 批次非 `--dry` 時，舊拼法 pool 不影響任何顯示**：列序永遠是
  平均分排序，沒有「正式名列排到尾端」這回事；`records.py` docstring 說的
  「`order` optionally pins the sequence (used to keep the player first)」目前
  是死意圖。要修（真的釘列序）或要刪（連同 `match.py` 的呼叫）列入 plan9a
  「待清理項目」(f)。
