# AI 註冊表重整計畫書(v2)

## 目標
1. 所有可選對手(Python 算法、模仿、RL)集中由 `ai/` 管理,以 `ai/registry.json` 為唯一真實來源。
2. 更換常規對手池 = 改註冊表,不改程式碼、不改測試。
3. `hc_1000` 統一改名為 `rl_h1000_0k`,舊名永久作別名。
4. RL 持續訓練:`data/` 是工廠(gitignore、不動),`ai/` 是展示櫃(只放 active 對手,入版控)。
5. UI 對手選單選項過多時自動出現垂直捲動條。

## 已定決策(使用者)
- 權重以「複製發布」進 `ai/checkpoints/`,不搬、不改 `data/`。
- `data/*` 維持 gitignore;不再為 `data/` 內檔案開 `!` 例外(撤銷先前 17MB 反白指令)。僅 `ai/checkpoints/` 入版控。
- 座位抽籤依 key 排序,與註冊順序脫鉤。
- 新池組成:
  - 常規池(enabled):7 人格 + rl_h1000_20k + rl_h1000_0k + rl_o1000_0k + rl_b1000_0k + rl_i1000_0k
  - rl_only = (rl_h1000_20k,)
  - imitation_only = (rl_h1000_0k, rl_o1000_0k, rl_b1000_0k, rl_i1000_0k),共 4 個;與 rl_only 交集必須為空,加測試
- hc_2000、hc_10000 移出常規池,保留為歷史。
- 修正文案(rl_desc_fmt、random_ai_desc)並走 config.I。
- 重採 GOLDEN,舊值與 commit 寫入 README。
- i 過弱觸發由使用者人工對弈判定取代;o 未播種不可重放。

## 總原則
- 每階段獨立提交;階段結束須全套測試通過、工作樹乾淨、提交回報,再進下一階段。
- 發現非預期(非組成性)失敗:停下回報,不自行調整測試。
- 歷史證據(records.json、0.5b 的 6 個 gz、舊 README 數據)不改寫。
- 複製權重後以 sha256 逐檔證明與來源一致。
- 不確定處停下問使用者,不自行命名、創建或決定。
- 使用者可見文字一律走 config.I。
- 推送由使用者手動。

## 階段 0:盤點(只讀,不改檔)
- 列出所有 AI 的來源:座位表、checkpoint 路徑、模組位置、現有 preset、別名機制。
- 列出 `hc_1000`、`hc_2000`、`hc_10000` 在 repo 中所有出現處(代碼、測試、README、records、證據檔),分類為「可改」與「歷史不可改」。
- 列出所有依賴 hc_2000、hc_10000 的證據與測試。
- 確認 test_seat_rename.py 的別名機制能否直接重用。
- 現行座位抽籤如何決定順序(是否依池順序),說明改為依 key 排序會影響哪些種子。
- 報告並等待使用者裁決:hc_2000、hc_10000 是否發布到 ai/(保留重跑能力)還是僅保留可讀的歷史。

## 階段 1:別名改名 hc_1000 → rl_h1000_0k
- 新 key 為正式名;`hc_1000` 為別名,解析到同一座位。
- 別名不進池、不重複計分;歷史檔案不改寫。
- `--subject hc_1000` 與 `--subject rl_h1000_0k` 皆有效;實際重跑一個 0.5b 證據確認。
- 測試:別名解析、別名不進池、records 舊 key 可讀。
- RL 錨點(rl/rollout.py、rl/rl_train.py)仍可解析。

## 階段 2:註冊表與載入器(不搬檔、不改池組成)
- 新增 `ai/registry.json` 與 `ai/registry.py`。
- 此階段註冊表只忠實描述現況(含現有池組成),行為必須零變化。
- 欄位:
  - key、aliases
  - kind:network / heuristic
  - family:personality / imitation / rl
  - checkpoint(network)或 module(heuristic)
  - source(原訓練路徑,純記錄)、sha256(network)
  - pools:所屬池名稱清單
  - enabled(進常規池)、selectable(可出現在 UI 與隨機抽籤)
  - label、desc_key(指向 config.I)、note
- 載入器提供:`automated_options()`、`pool(name)`、`resolve(key)`、`anchors()`。
- 驗證(啟動與測試皆跑):
  - key 唯一;別名不與任何 key 衝突
  - pools 名稱合法;desc_key 存在於 config.I
  - network 的 checkpoint 存在、sha256 一致
- seats.py 與 match.py 改由載入器取值;全套測試證明零行為變化。

## 階段 3:抽籤改依 key 排序
- 座位抽籤的候選順序改為依 key 排序,不依註冊順序。
- 此階段單獨提交,以便辨識 GOLDEN 變動的原因。
- 預期:凡依賴池順序的同種子對局都會變動。
- 把失敗逐一分類;全屬「順序性」才可重採 GOLDEN;有非順序性失敗則停下回報。
- ~~新增不變式測試:增刪一個不相關的 AI,不改變其他 AI 在同種子下的座位對應~~
  **使用者裁定不做、若有則刪除** —— 依 key 排序之後,增刪一個 key 會改變集合,
  因此也改變排序位置,這條不變式本來就不成立。改為驗證
  「同一池內容、不同註冊順序,sorted 抽籤結果相同」。
- 新增 `--pool-order {sorted,literal}`:`literal` 只用來逐字重現階段 3 之前的
  批次,不排序、不去重、不看 `enabled`、`--subject` 不正規化,且不得與 preset
  併用。重現 0.5b 的完整指令見 `eval/plan9/README.md`。

## 階段 4:複製發布權重到 ai/checkpoints/
- 結構:
  - ai/registry.json、ai/registry.py
  - ai/checkpoints/<key>/…(入版控)
  - ai/heuristics/(Python 算法型 AI,見階段 5)
- 發布對象:rl_h1000_20k、rl_h1000_0k、rl_o1000_0k、rl_b1000_0k、rl_i1000_0k。
- 複製不搬移,`data/` 不動;複製後 sha256 逐檔比對並列表回報。
- registry 的 checkpoint 指向 ai/ 內路徑;source 記錄原路徑。
- .gitignore:`data/*` 維持忽略,僅 `ai/checkpoints/` 入版控;以 `git ls-files` 確認。
- 「每個網路座位的權重都在版控」測試改指 ai/ 內路徑。
- 驗證新增:ai/checkpoints/ 內無未登記權重;無登記缺檔。
- 說明:sha256 是複製的來源證明,不要求與 data/ 同步;data/ 之後被覆蓋不算錯。
- 新增發布命令:`python -m ai.registry --publish <key> --from <path>`
  - 複製、算雜湊、寫入 enabled:false 的草稿項,由使用者手動改為 true。

## 階段 5:Python 算法型 AI 搬入 ai/heuristics/
- 只改 import 路徑,不改邏輯。
- 以全套測試與固定種子對局輸出不變為證。
- 若搬遷牽涉面過大,先回報,讓使用者決定是否暫緩,只在註冊表登記 module 路徑。

## 階段 6:新池組成與文案
- 依「已定決策」更新註冊表:hc_2000、hc_10000 設 enabled:false、selectable:false(是否發布依階段 0 裁決)。
- rl_only 與 imitation_only 依註冊表 pools 定義;加交集為空的測試。
- config.I:0k 學生用「模仿學習 1000 步」說明,不套 PPO 文案;random_ai_desc 去掉不實的「模仿」措辭。
- 受影響測試先分類:全屬組成性才重採 GOLDEN;舊值寫入 README。
- `--subject` 允許 hc_* 與 rl_h1000_0k,即使不在常規池。
- 注意 `imitation_only` 現在含 rl_h1000_0k(113 萬列),與 o/b/i(約 7.9 萬列)規格不同,README 註明。

## 階段 7:UI 下拉捲動
- 對手選單選項超出可視高度時自動垂直捲動(滑鼠滾輪與鍵盤上下鍵皆可)。
- 以 20 個以上選項實測;若有測試框架,加選項數多時的測試。
- 文字走 config.I。獨立提交。

## 階段 8:換 AI 的工作流與工具
- `python -m ai --check`:執行全部驗證,列出各池內容。(原寫 `python -m ai.registry --check`;`ai/__init__` 匯出 `ai.registry`,跑子模組會重複執行同一個模組並噴 runpy 的 `RuntimeWarning`,所以入口掛在套件上。)
- `python -m ai.registry --list`:顯示全部 AI、所屬池、enabled/selectable、sha256 前綴。
- README 新增「如何新增或更換對手」:
  1. `--publish` 複製權重並產生草稿
  2. 編輯 registry.json,設 enabled:true 與 pools
  3. `--check`
  4. 親自對戰體驗
- 池組成相關測試改為由註冊表驅動,保留不變式(如 imitation_only ∩ rl_only = ∅);新增 AI 不應需要改測試,若仍需要,記錄原因。

## 階段 9:records 與評測相容
- records.json 新舊 key 皆可讀,新記錄用新 key。
- 評測工具與 `--subject` 經 resolve() 解析別名。
- 已知取捨:hc_2000、hc_10000 若不發布到 ai/,其依賴的歷史證據在乾淨 clone 上只可讀、不可重跑;README 明確寫出。
- 回報:是否有歷史結果因改名或搬遷而無法重現。

## README 必須記錄
- 0.063 規則廢止與新 val-only 規則(已完成者確認即可)。
- i 過弱觸發與處理:門檻以 hunter 系校準,對較弱老師不適用,改由使用者人工對弈判定。
- o 未播種,不可重放。
- 舊三模仿者移出常規池(使用者決定);hc_1000 改名 rl_h1000_0k,保留別名。
- imitation_only 組內規格不一致。
- 抽籤改依 key 排序及其原因;GOLDEN 重採原因與舊值、commit。
- data/ 與 ai/ 的分工:data/ 為訓練工廠(gitignore),ai/ 為發布對手(入版控)。

## 每階段回報格式
- 提交 hash 與一句內容
- 全套測試:通過/失敗數、耗時
- records.json 是否變動
- 風險與需使用者裁決事項

## 使用者人工驗收(不屬 agent)
- 親自對戰 o/b/i,以事前寫下的「明顯失誤」定義(例如放棄大角落、明顯自封)計次判斷,再決定是否進入各自的 20k 局 RL。
