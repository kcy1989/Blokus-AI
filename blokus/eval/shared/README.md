# shared — 四個學習者共同的 run

這三個目錄不屬於單一血統：它們同時訓練或評測 h / i / o / b，因此放在 `shared/`。
b 沒有專屬 eval 目錄，它的證據也有一部分在這裡。

## 分支圖

```
四學習者的 0k 權重
└── 0k-to-20k/           plan9 步驟 4：成對評測 0k → 20k
    └── 四個 20k 席位     rl_{h,i,o,b}1000_20k

四個 20k 席位
├── four-at-once/        plan10 任務二：四學習者同桌
│   ├── milestone 1      rl_h1000_50k / rl_o1000_50k / rl_i1000_50k / rl_b1000_50k
│   └── big_eval/        同桌之後的評測
└── four-at-once-rb80/   同一設計的 balanced-schedule 變體
```

## 產出的入池席位

| 席位 | 來源 |
| --- | --- |
| `rl_h1000_20k` / `rl_i1000_20k` / `rl_o1000_20k` / `rl_b1000_20k` | `0k-to-20k/` |
| `rl_h1000_50k` / `rl_o1000_50k` / `rl_i1000_50k` | `four-at-once/` |
| `rl_b1000_50k` | `four-at-once/`（無正面證據，使用者裁示入池） |

## 目錄

| 目錄 | 大小 | 內容 | 可否刪 |
| --- | --- | --- | --- |
| `0k-to-20k/` | 1.6M | 0k→20k 的成對評測、開局分布 | 否（四個 20k 席位證據） |
| `four-at-once/` | 848K | 四學習者同桌的設計、三個里程碑、`big_eval/` | 否（三個 50k 席位證據） |
| `four-at-once-rb80/` | 368K | balanced-schedule 變體 | 否 |
