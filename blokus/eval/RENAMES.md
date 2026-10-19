# eval/ 改名對照（RENAMES）

- 改名日期：2026-10-09。只搬路徑、不改內容：46 個資料檔（`.json`、`.json.gz`）搬移
  前後 md5 逐檔一致，`x-` → `four20k-` 的八檔以內容 md5 對照，無一例外。
- 舊 commit：`8e6839d`（`eval/plan9/step4/` 的 32 個批次入庫）、
  `117d89b`（`--adhoc` 席位載入，`o` / `b` / `i` / 四生池批次所依賴的機制）。
- 批次檔 summary 內的 games_file 為凍結純檔名，故家族一批次檔不改名。

| 舊路徑 | 新路徑 |
| --- | --- |
| `eval/plan9/README.md` | `eval/imitation/README.md` |
| `eval/plan9/argmax-hc_1000-games.json.gz` | `eval/imitation/argmax-hc_1000-games.json.gz` |
| `eval/plan9/argmax-hc_1000-summary.json` | `eval/imitation/argmax-hc_1000-summary.json` |
| `eval/plan9/argmax-hunter-games.json.gz` | `eval/imitation/argmax-hunter-games.json.gz` |
| `eval/plan9/argmax-hunter-summary.json` | `eval/imitation/argmax-hunter-summary.json` |
| `eval/plan9/argmax-rl_h1000_20k-games.json.gz` | `eval/imitation/argmax-rl_h1000_20k-games.json.gz` |
| `eval/plan9/argmax-rl_h1000_20k-summary.json` | `eval/imitation/argmax-rl_h1000_20k-summary.json` |
| `eval/plan9/rl_b1000_0k-pair.json` | `eval/imitation/rl_b1000_0k-pair.json` |
| `eval/plan9/rl_i1000_0k-pair.json` | `eval/imitation/rl_i1000_0k-pair.json` |
| `eval/plan9/softmax-hc_1000-games.json.gz` | `eval/imitation/softmax-hc_1000-games.json.gz` |
| `eval/plan9/softmax-hc_1000-summary.json` | `eval/imitation/softmax-hc_1000-summary.json` |
| `eval/plan9/softmax-hunter-games.json.gz` | `eval/imitation/softmax-hunter-games.json.gz` |
| `eval/plan9/softmax-hunter-summary.json` | `eval/imitation/softmax-hunter-summary.json` |
| `eval/plan9/softmax-rl_h1000_20k-games.json.gz` | `eval/imitation/softmax-rl_h1000_20k-games.json.gz` |
| `eval/plan9/softmax-rl_h1000_20k-summary.json` | `eval/imitation/softmax-rl_h1000_20k-summary.json` |
| `eval/plan9/step4/b-argmax-s-builder.json.gz` | `eval/rl-single-train/b-argmax-s-builder.json.gz` |
| `eval/plan9/step4/b-argmax-s-rl_b1000_0k.json.gz` | `eval/rl-single-train/b-argmax-s-rl_b1000_0k.json.gz` |
| `eval/plan9/step4/b-argmax-s-rl_b1000_20k.json.gz` | `eval/rl-single-train/b-argmax-s-rl_b1000_20k.json.gz` |
| `eval/plan9/step4/b-softmax-s-builder.json.gz` | `eval/rl-single-train/b-softmax-s-builder.json.gz` |
| `eval/plan9/step4/b-softmax-s-rl_b1000_0k.json.gz` | `eval/rl-single-train/b-softmax-s-rl_b1000_0k.json.gz` |
| `eval/plan9/step4/b-softmax-s-rl_b1000_20k.json.gz` | `eval/rl-single-train/b-softmax-s-rl_b1000_20k.json.gz` |
| `eval/plan9/step4/h-argmax-s-hunter.json.gz` | `eval/rl-single-train/h-argmax-s-hunter.json.gz` |
| `eval/plan9/step4/h-argmax-s-rl_h1000_0k.json.gz` | `eval/rl-single-train/h-argmax-s-rl_h1000_0k.json.gz` |
| `eval/plan9/step4/h-argmax-s-rl_h1000_20k.json.gz` | `eval/rl-single-train/h-argmax-s-rl_h1000_20k.json.gz` |
| `eval/plan9/step4/h-softmax-s-hunter.json.gz` | `eval/rl-single-train/h-softmax-s-hunter.json.gz` |
| `eval/plan9/step4/h-softmax-s-rl_h1000_0k.json.gz` | `eval/rl-single-train/h-softmax-s-rl_h1000_0k.json.gz` |
| `eval/plan9/step4/h-softmax-s-rl_h1000_20k.json.gz` | `eval/rl-single-train/h-softmax-s-rl_h1000_20k.json.gz` |
| `eval/plan9/step4/i-argmax-s-intruder.json.gz` | `eval/rl-single-train/i-argmax-s-intruder.json.gz` |
| `eval/plan9/step4/i-argmax-s-rl_i1000_0k.json.gz` | `eval/rl-single-train/i-argmax-s-rl_i1000_0k.json.gz` |
| `eval/plan9/step4/i-argmax-s-rl_i1000_20k.json.gz` | `eval/rl-single-train/i-argmax-s-rl_i1000_20k.json.gz` |
| `eval/plan9/step4/i-softmax-s-intruder.json.gz` | `eval/rl-single-train/i-softmax-s-intruder.json.gz` |
| `eval/plan9/step4/i-softmax-s-rl_i1000_0k.json.gz` | `eval/rl-single-train/i-softmax-s-rl_i1000_0k.json.gz` |
| `eval/plan9/step4/i-softmax-s-rl_i1000_20k.json.gz` | `eval/rl-single-train/i-softmax-s-rl_i1000_20k.json.gz` |
| `eval/plan9/step4/o-argmax-s-optimizer.json.gz` | `eval/rl-single-train/o-argmax-s-optimizer.json.gz` |
| `eval/plan9/step4/o-argmax-s-rl_o1000_0k.json.gz` | `eval/rl-single-train/o-argmax-s-rl_o1000_0k.json.gz` |
| `eval/plan9/step4/o-argmax-s-rl_o1000_20k.json.gz` | `eval/rl-single-train/o-argmax-s-rl_o1000_20k.json.gz` |
| `eval/plan9/step4/o-softmax-s-optimizer.json.gz` | `eval/rl-single-train/o-softmax-s-optimizer.json.gz` |
| `eval/plan9/step4/o-softmax-s-rl_o1000_0k.json.gz` | `eval/rl-single-train/o-softmax-s-rl_o1000_0k.json.gz` |
| `eval/plan9/step4/o-softmax-s-rl_o1000_20k.json.gz` | `eval/rl-single-train/o-softmax-s-rl_o1000_20k.json.gz` |
| `eval/plan9/step4/x-argmax-s-rl_b1000_20k.json.gz` | `eval/rl-single-train/four20k-argmax-s-rl_b1000_20k.json.gz` |
| `eval/plan9/step4/x-argmax-s-rl_h1000_20k.json.gz` | `eval/rl-single-train/four20k-argmax-s-rl_h1000_20k.json.gz` |
| `eval/plan9/step4/x-argmax-s-rl_i1000_20k.json.gz` | `eval/rl-single-train/four20k-argmax-s-rl_i1000_20k.json.gz` |
| `eval/plan9/step4/x-argmax-s-rl_o1000_20k.json.gz` | `eval/rl-single-train/four20k-argmax-s-rl_o1000_20k.json.gz` |
| `eval/plan9/step4/x-softmax-s-rl_b1000_20k.json.gz` | `eval/rl-single-train/four20k-softmax-s-rl_b1000_20k.json.gz` |
| `eval/plan9/step4/x-softmax-s-rl_h1000_20k.json.gz` | `eval/rl-single-train/four20k-softmax-s-rl_h1000_20k.json.gz` |
| `eval/plan9/step4/x-softmax-s-rl_i1000_20k.json.gz` | `eval/rl-single-train/four20k-softmax-s-rl_i1000_20k.json.gz` |
| `eval/plan9/step4/x-softmax-s-rl_o1000_20k.json.gz` | `eval/rl-single-train/four20k-softmax-s-rl_o1000_20k.json.gz` |

## 2026-10-19 血統分組（目錄層）

只搬目錄、不改檔案內容。上面 2026-10-09 的逐檔對照維持原文，新的位置以本表為準。
`eval/imitation/` 與本檔不動；b 沒有專屬 eval 目錄（它的證據在共同 run 內），因此
`eval/b/` 不建立。

| 舊路徑 | 新路徑 | 血統 |
| --- | --- | --- |
| `eval/rl-single-train/` | `eval/shared/0k-to-20k/` | 共同（plan9 步驟 4） |
| `eval/rl-multiple-train/` | `eval/shared/four-at-once/` | 共同（plan10 任務二） |
| `eval/rl-multiple-train-rb80/` | `eval/shared/four-at-once-rb80/` | 共同（balanced-schedule 變體） |
| `eval/rl-single-train-v2h/` | `eval/h/v2/` | h |
| `eval/rl-single-train-v4h/` | `eval/h/v4/` | h |
| `eval/rl-single-train-t05/` | `eval/h/t05/` | h |
| `eval/rl-single-train-h100k/` | `eval/h/100k/` | h |
| `eval/rl-single-train-v2i/` | `eval/i/v2/` | i |
| `eval/rl-single-train-v3i/` | `eval/i/v3/` | i |
| `eval/rl-single-train-v2o/` | `eval/o/v2/` | o |
| `eval/rl-single-train-v3o/` | `eval/o/v3/` | o |
