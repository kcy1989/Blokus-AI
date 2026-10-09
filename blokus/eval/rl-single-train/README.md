# eval/rl-single-train — plan9 步驟 4 的成對評測證據

本目錄收 plan9 **步驟 4** 的證據:32 個批次檔,以及 5 個開局前 10 手選塊分布檔。
所有檔案都是證據 —— 內容不改、不覆寫,重跑一律寫新檔名。

---

## 1. 批次檔(32 個)

檔名 `<前綴>-<mode>-s-<subject>.json.gz`;`four20k-` 前綴是四生互比那 8 批(原名
`x-`,新舊對照見 [`../RENAMES.md`](../RENAMES.md))。

32 批全同的參數:

| 項目 | 值 |
| --- | --- |
| 種子 | `20261005` |
| 局數 | 每批 **3,000 局**,4 席/局 |
| 隨機流 | `paired-v1-subject`(每局 setup 流 + 每席位流,成對) |
| 池順序 | `sorted` |
| device | cuda |
| 選步方式 | argmax 與 softmax 各一輪,分開列表 |
| 被測對象 | 批次檔的 `subject` 欄:各生的 20k、各生的 0k、各生的人格老師 |

檔內 `pool_raw` 記的是**該批實際輸入的字串**,`pool` 記展開後依 key 排序的名單。

---

## 2. 池口徑:12 席自動池(維持,2026-10-09 裁決)

**12 席的組成**:

```
rl_h1000_0k, rl_o1000_0k, rl_b1000_0k, rl_i1000_0k      ← 4 個 0k 模仿
hunter, optimizer, builder, intruder, fox, chess, wolf  ← 7 個人格
rl_<x>1000_20k                                          ← 被測學生自己的 20k
```

第 12 席**隨被測對象換人**:h 批是 `rl_h1000_20k`、o 批是 `rl_o1000_20k`,以此類推。
h 的 20k 已在名冊上,o / b / i 的 20k 未入名冊,靠 `--adhoc KEY=PATH` 綁檔使用,
批次檔的 `adhoc` 欄記著路徑與 md5(檔在 `data/`,不在版控)。所以 `pool_raw` 共有
**四種**,各 6 批 = 24 批;另有 **8 批是 15 席**(`four20k-*`:`7 人格 + 4×0k +
4×20k`,四生互比用)。

### 三個 20k 的 provenance(2026-10-09)

| 欄位 | 值 |
| --- | --- |
| 訓練檔 | `data/rl_o1000/step_000040.pt` 等;md5 `b0f7c122…` / `0a8c0341…` / `430c28cb…` |
| 入庫檔 | `ai/checkpoints/rl_<x>1000_20k/step_000040.pt`;sha256 `272b2edf…` / `63aeb600…` / `3db862d3…` |
| `history` | 「模仿 1000 步 + 單打 20k」 |
| `train_seed` | **`7200000`(int,種子基底)**;範圍 **`7200000–7219999`** |
| `teacher` | `optimizer` / `builder` / `intruder` |
| `pool_contains_teacher` | `true` |
| `commit` | **`null`** —— `MANIFEST.json` 記 `88c8a18f…`,但未驗證工作樹乾淨,按 `code_hash()` 的標準不採用(使用者裁決 2026-10-09) |
| 寫進 `records.json` 了嗎 | **沒有**(裁決選 b:`Records._load` 丟棄 `games = 0` 的 meta-only 列,寫了下次載入整條消失;見 AGENTS 待辦) |

**雜湊核對**:檔案 md5 ≡ 32 個驗收批次 `adhoc` payload 的 md5,每個 key 各 **14 筆
逐筆相同**。`MANIFEST.json` **沒有雜湊欄**(只有 name / init / anchor / seed /
commit / command)—— 第三方不存在,記為事實,不補、不改訓練端。

> **在 `records.json` 補上這些欄位之前,三個 20k 的 provenance 以本檔與
> `data/rl_*1000/MANIFEST.json` 為準。**

**與計畫「7 人格池」的差異**:

| | 計畫的組 A | 0.5b 實際批次 | **步驟 4 批次(本目錄)** |
| --- | --- | --- | --- |
| 池 | 7 個規則人格,池內不含任何 RL 模型 | `pool_raw = no_imitation`,7 個人格 | **12 席**(4×0k + 7 人格 + 被測 20k) |
| 局數 | 2,000 | 2,000 | **3,000** |
| 種子 | 20261005 | 20261005 | 20261005 |

也就是說:

- **步驟 4 是 12 席池 3000 局,0.5b 組 A 是 7 人格池 2000 局,兩者是不同的評測。**
- **h 的 0.5b 數字不得與 o / b / i 的步驟 4 數字直接並列。**
- 計畫要求「步驟 4 的評測池與步驟 0.5b 的組 A 完全相同」——**未達成**。
- 差別不只是多五席:單席權重從 1/7 變 1/12,對手裡多了神經網路,而且**含被測者
  自己的 20k**(它每局必在場,其餘三席有機會再抽到同一個 key)。

**裁決(2026-10-09):維持已用的 12 席自動池,不重做 32 批。** 本目錄的數字只能在
12 席池內部比,不能當成 0.5b 的續篇讀。

---

## 3. 組 B 取消(2026-10-09)

組 B(`hc_2000`、`hc_10000`、`random_ai`)**取消,不評測**。

理由:組 B 的三個席位與 h 版同源(兩個是 hunter 蒸餾),對 o / b / i 而言是異源,
計畫自己就註明「組 B 對 h 與對其他三個學生的難度性質不同,不得直接比較絕對值」。
在組 A 的 12 席池已足以回答主標準的情況下,再跑一組口徑本來就不能橫向比較的池,
只會多出一份容易被誤讀的數字。**步驟 4 因此沒有組 B 的證據,這一點不補。**

---

## 4. 主標準改為只看組 A(2026-10-09)

- **主標準**:各學生對**其 0k 模仿版**、**argmax** 的成對差距大於 **2 個成對標準誤**
  (在 12 席池、seed 20261005、3000 局之下)。
- **softmax 照列表出,不作判定** —— 沿用先前裁決(理由:0.5b 中 h 版 softmax 對
  `hc_1000` 的 t = −0.44,同樣未達 2 個成對標準誤)。
- 「學生 vs 其人格老師本尊」只報告,不作通過判定(計畫原定)。
- 組 B 的主標準子句因組 B 取消而**不再適用**。

---

## 5. 限制:泛化未檢驗

**這批證據只回答「在同一個池、同一個種子、同一套選步下,誰比較強」,它不回答任何
跨池、跨種子的問題。**

- 池是**單一固定池**(12 席),沒有第二個池做交叉驗證 → **跨池泛化未檢驗**。
- 種子只有 `20261005` 一個 → 換一批種子是否同向,**未檢驗**。
- 每個比較只在 3,000 局上成立,沒有獨立重跑 → 逐批重現性靠 `paired-v1-subject`
  的可重算性,不靠第二份樣本。
- 池內含被測者自己,所以「平均餘格/平均分」包含同桌互打的情形;成對差距的口徑是
  **只取被測席位那一列**,不是全桌平均。

寫明這四條,是因為**沒有任何一項測試涵蓋它們**:不要把「測試全綠」讀成「泛化成立」。

---

## 6. 開局前 10 手選塊分布(描述,不檢定)

| 檔案 | 池 | subject |
| --- | --- | --- |
| `openings-12-h.json` | 12 席 | `rl_h1000_20k` |
| `openings-12-o.json` | 12 席 | `rl_o1000_20k`(`--adhoc`) |
| `openings-12-b.json` | 12 席 | `rl_b1000_20k`(`--adhoc`) |
| `openings-12-i.json` | 12 席 | `rl_i1000_20k`(`--adhoc`) |
| `openings-four20k.json` | 15 席 | 無(`paired-v1`,四生互比) |

由根目錄的 `openings.py` 產生:與批次檔同一套 `paired_streams` / `subject_draw` /
`setup_seats` 排局,錄下每席**前 10 手選塊**後即停局(不計 pass,`play_match` 的
`on_move` 本來就只在落子時呼叫)。seed `20261005`、3,000 局、argmax。

檔內 `openings[key].by_ply["1"…"10"]` 是每個 ply 的選塊計數,`pieces` 是該選項十手
合計。`games_with_moves` 記錄有多少局真的錄滿 10 手。

> **只作描述,不檢定,不得據此下結論。** 計畫原文:「這只是描述性指標,不做檢定,
> 不得據此下結論。」這裡的每一個數字都沒有對照組、沒有標準誤、沒有假設檢定。

---

## 7. 重跑方式

```bash
# 批次檔(match.py,成對 + 指定席位 + gzip)
.venv-rl/bin/python match.py --games 3000 --seed 20261005 --paired-rng \
    --subject rl_h1000_20k --mode argmax --gzip --dry \
    --pool rl_h1000_0k,rl_o1000_0k,rl_b1000_0k,rl_i1000_0k,hunter,optimizer,builder,intruder,fox,chess,wolf,rl_h1000_20k \
    --out <新檔名>.json.gz

# 開局分布(openings.py,同排局,第 10 手停局)
.venv-rl/bin/python openings.py --seed 20261005 --games 3000 --mode argmax \
    --stop-after 10 --subject rl_h1000_20k \
    --pool rl_h1000_0k,rl_o1000_0k,rl_b1000_0k,rl_i1000_0k,hunter,optimizer,builder,intruder,fox,chess,wolf,rl_h1000_20k \
    --out <新檔名>.json
```

兩個腳本都**拒絕覆寫已存在的檔**;`--pool-order literal` 只用於逐字重現階段 3 之前的
歷史批次(本目錄的批次全是 `sorted`)。

### 註冊前 / 註冊後:兩套重放命令(2026-10-09)

跑這 32 批時 `rl_{o,b,i}1000_20k` 未入名冊,所以**當時**的命令帶 `--adhoc`;
發布之後 `seats.register_adhoc` 會**拒絕**已是名冊席位的 key
(`"%r is already a seat option"`),舊命令會 `ValueError`。同一條命令的兩套寫法:

```bash
# (甲) 註冊前 —— 批次檔當初就是這樣跑的
.venv-rl/bin/python match.py --games 3000 --seed 20261005 --paired-rng \
    --subject rl_o1000_20k --mode argmax --gzip --dry \
    --pool rl_h1000_0k,rl_o1000_0k,rl_b1000_0k,rl_i1000_0k,hunter,optimizer,builder,intruder,fox,chess,wolf,rl_o1000_20k \
    --adhoc rl_o1000_20k=data/rl_o1000/step_000040.pt \
    --out <新檔名>.json.gz

# (乙) 註冊後(現在) —— 去掉 --adhoc,其餘逐字相同
.venv-rl/bin/python match.py --games 3000 --seed 20261005 --paired-rng \
    --subject rl_o1000_20k --mode argmax --gzip --dry \
    --pool rl_h1000_0k,rl_o1000_0k,rl_b1000_0k,rl_i1000_0k,hunter,optimizer,builder,intruder,fox,chess,wolf,rl_o1000_20k \
    --out <新檔名>.json.gz
```

`openings.py` 同理:當時的命令帶 `--adhoc KEY=data/rl_*1000/step_000040.pt`,
註冊後去掉即可。

**兩套等價的實測**(2026-10-09,20 局、seed `20261005`、argmax、同池同 subject,
只差有沒有 `--adhoc`):

| 欄位 | 甲(有 `--adhoc`) | 乙(無) | 相同 |
| --- | --- | --- | --- |
| `pool_raw` / `pool` / `pool_order` / `subject` / `rng_mode` | — | — | ✔ |
| `games_detail`(20 局逐局) | — | — | ✔ |
| `summary` / `appearances` | — | — | ✔ |
| `adhoc` 欄 | `rl_o1000_20k = b0f7c122…` | `{}` | 甲多記一條,內容與名冊檔同 md5 |

也就是說:**註冊改的是「權重從哪個路徑讀」,兩邊讀到的位元相同,對局逐局不變。**
批次檔裡的 `adhoc` 欄是歷史紀錄,重放時不要再抄它。
