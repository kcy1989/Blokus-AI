# plan9 評測輸出

`plan9.md` 階段 A 的評測結果放在這裡。**這裡放的是證據，不是計畫書**——評測一旦跑完就
必須能從 repository 重算，否則它只是某個終端機上的一次輸出。

## 為什麼這個目錄要單獨加一行 .gitignore

`.gitignore` 第 10 行的 `plan*` 會匹配**任何以 `plan` 開頭的路徑元件**，所以
`eval/plan9/` 原本整個被忽略，而且是靜默的：檔案照樣寫得進去，`git status` 只是永遠
不會提到它們。`!eval/plan9/` 一行把它救回來，`plan*.md` 仍維持忽略。

這裡**不需要** `data/` 那段的三行寫法。那裡必須先取消排除父目錄，是因為 git 不會
descend 進被排除的父目錄；`eval/` 本身沒有被排除，所以指名那一個目錄就夠了。

## 目錄內容

| 檔案 | 入庫 | 內容 |
| --- | --- | --- |
| `README.md` | 是 | 本檔 |
| `<批次>-summary.json` | 是 | 每個選項的席次、平均分、平均餘格、標準誤 |
| `<批次>-games.json.gz` | 是 | 逐局明細，**gzip 壓縮** |

未壓縮的逐局明細約 245 bytes/局（實測自
`data/match/seed20260928-g500-poolno_imitation_hc_1000.json`：500 局 122,411 bytes）。
`gzip -9` 把它壓到 **6.7%**，即約 16 bytes/局——N=2000 的一批從 0.47 MB 降到 0.03 MB。
壓縮後進版控，這樣成對統計的原始資料仍在 repo 裡，不必靠「當初那台機器還在」才能重算。

## 產出方式

逐局明細由 `match.py --out` 寫出，之後另行 gzip：

```bash
.venv-rl/bin/python match.py --pool no_imitation --games 2000 \
    --seed 20261005 --paired-rng --subject <KEY> \
    --mode argmax --out eval/plan9/<批次>-games.json
gzip -9 eval/plan9/<批次>-games.json      # -> <批次>-games.json.gz
```

`--subject` 尚未實作（設計待確認）。`--out` 一定要帶：`match.py` 的預設輸出目錄是
`data/match/`，而 `data/` 受測試的檔案守衛看管。

## 每個批次必須記錄的東西

摘要檔與逐局檔都要帶上以下欄位，缺一就不算完整證據：

- **`rng_mode`**：`match.py` 自己會寫（`paired-v1` 或 `legacy-single-stream`）。
  成對結果與排行榜結果的抽籤方式不同，兩者不可混合平均。
- **`--seed`**、**`--pool`** 原始字串與展開結果、**`--mode`**（argmax / softmax）。
- **母體定義**：哪些局被納入統計、被排除的局有幾局、排除理由。
  （固定 subject 席位的模式下母體是全部有效局；在場率不是 100% 的舊設計才需要這一段。）
- **有效局數**與**排定局數**。
- 對照物名稱、以及對照物是否也出現在對手席。

## 待確認

- `--subject` 的旗標介面與抽籤細節（見 plan9 步驟 0.5a′ 續報）。
- 批次檔名的最終格式。