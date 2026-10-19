# i — intruder 血統

學習者風格 i（intruder）。兩條單打 50k 嘗試都沒有把最終權重送進池子；入池的是
起點 `rl_i1000_20k`、四學習者同桌的 `rl_i1000_50k`，以及 v3 run 的中間檢查點
`rl_i1000_37.5k`。

## 分支圖

```
rl_i1000_20k  (shared/0k-to-20k)
├── v2 run → rl_i1000_50k_v2     未入池（argmax −0.029730，−0.719 se）
└── v3 run (data/rl/i/50k_v3)
    ├── round 35 → rl_i1000_37.5k   入池（+0.065620，1.597 se）
    └── round 60 → rl_i1000_50k_v3  未入池（argmax −0.036140）
```

`rl_i1000_50k` 來自四學習者同桌，證據在
[`shared/four-at-once/`](../shared/four-at-once/README.md)。

## 入池

| 席位 | 來源 | 狀態 |
| --- | --- | --- |
| `rl_i1000_20k` | `shared/0k-to-20k/` | 已入池 |
| `rl_i1000_37.5k` | `v3/` §15（run 的 round 35） | 已入池，待校準 |
| `rl_i1000_50k` | `shared/four-at-once/` | 已入池 |

`rl_i1000_50k_v2`、`rl_i1000_50k_v3` 的最終權重都不是席位。

## 目錄

| 目錄 | 大小 | 內容 | 可否刪 |
| --- | --- | --- | --- |
| `v2/` | 28K | `rl_i1000_50k_v2` 訓練與失敗的驗收 | 否 |
| `v3/` | 48K | `rl_i1000_50k_v3`；含 round 35 入池記錄 | 否（37.5k 席位證據） |
