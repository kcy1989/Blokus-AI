# eval/ — 學習者血統的評測證據

這個目錄按**學習者血統**分組，與 `data/rl/` 同構。目的只有一個：看到一個目錄就知道
它屬於哪條分支、現在最強的是哪個權重、以及它有沒有入池。

`eval/` 全部都是證據，**不刪**。舊路徑的對照見 [RENAMES.md](RENAMES.md)；2026-10-19
的目錄層搬家已追加在該檔第二段。

## 分組

```
eval/
├── imitation/                  H 階段的資料集與老師（不動）
├── shared/                    四個學習者共同的 run
│   ├── 0k-to-20k/             plan9 步驟 4：四個 0k → 20k
│   ├── four-at-once/          plan10 任務二：四學習者同桌
│   └── four-at-once-rb80/     同上，balanced-schedule 變體
├── h/                         hunter
│   ├── v2/                    50k v2（最終未入池；r20 成為 30k）
│   ├── v4/                    50k v4（最終未入池；r35 是 100k/t05 的起點）
│   ├── t05/                   溫度 T=0.5 單變量（未入池）
│   └── 100k/                  rl_h1000_100k（裁示入池）
├── i/                         intruder
│   ├── v2/                    50k v2（未入池）
│   └── v3/                    50k v3（最終未入池；r35 是 rl_i1000_37.5k）
└── o/                         optimizer
    ├── v2/                    50k v2（未入池）
    └── v3/                    50k v3（+0.2286，入池）
```

**b（builder）沒有專屬 eval 目錄**：它的證據在 `shared/four-at-once/` 內，因此不建立
空的 `eval/b/`。

## 目前入池的網路席位（`rl_only`）

| 血統 | 席位 | 證據 |
| --- | --- | --- |
| h | `rl_h1000_20k` | `shared/0k-to-20k/` |
| h | `rl_h1000_30k` | `h/v2/` §17–§18（run 的 round 20） |
| h | `rl_h1000_50k` | `shared/four-at-once/` |
| h | `rl_h1000_100k` | `h/100k/`（裁示入池） |
| i | `rl_i1000_20k` | `shared/0k-to-20k/` |
| i | `rl_i1000_37.5k` | `i/v3/` §15（run 的 round 35） |
| i | `rl_i1000_50k` | `shared/four-at-once/` |
| o | `rl_o1000_20k` | `shared/0k-to-20k/` |
| o | `rl_o1000_50k` | `shared/four-at-once/` |
| o | `rl_o1000_50k_v3` | `o/v3/` §14/§15 |
| b | `rl_b1000_20k` | `shared/0k-to-20k/` |
| b | `rl_b1000_50k` | `shared/four-at-once/`（無正面證據，使用者裁示入池） |

權重的載入路徑一律以 `ai/registry.json` 的 `checkpoint` 為準；`seats.py` 的
`RL_SEATS` 只是它的讀者。

## 各目錄大小與處置

| 目錄 | 大小 | 處置 |
| --- | --- | --- |
| `imitation/` | 344K | 保留（H 階段證據） |
| `shared/0k-to-20k/` | 1.6M | 保留 |
| `shared/four-at-once/` | 848K | 保留 |
| `shared/four-at-once-rb80/` | 368K | 保留 |
| `h/v2/` | 60K | 保留 |
| `h/v4/` | 48K | 保留 |
| `h/t05/` | 32K | 保留 |
| `h/100k/` | 52K | 保留 |
| `i/v2/` | 28K | 保留 |
| `i/v3/` | 48K | 保留 |
| `o/v2/` | 32K | 保留 |
| `o/v3/` | 36K | 保留 |

大小是 2026-10-19 的 `du -sh`。這些是 KB 級的證據檔，沒有刪除候選；真正佔空間的是
`data/`。

## 血統 README

- [h/README.md](h/README.md)
- [i/README.md](i/README.md)
- [o/README.md](o/README.md)
- [shared/README.md](shared/README.md)
