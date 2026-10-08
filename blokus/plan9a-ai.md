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
- 發布對象:**2 個** —— `rl_h1000_20k`、`rl_h1000_0k`。
  (原列的 `rl_o1000_0k`、`rl_b1000_0k`、`rl_i1000_0k` **留到階段 6**:它們還沒註冊,
  先搬進 `ai/checkpoints/` 會撞上同階段的「無未登記權重」驗證 —— 兩者互斥,
  依裁決 B2-(甲) 先讓已註冊的兩個到位。)
- 複製不搬移,`data/` 不動;複製後 sha256 逐檔比對並列表回報。
- registry 的 checkpoint 指向 ai/ 內路徑;source 記錄原路徑。
- .gitignore:`data/*` 維持忽略,僅 `ai/checkpoints/` 入版控;以 `git ls-files` 確認。
  **實際狀態是過渡的**:已撤 `step_001000.pt` 與 `step_000040.pt` 的反白,
  `step_002000.pt`／`step_010000.pt` 的反白**保留**(裁決 B1 選 1,它們沒發布),
  階段 6 兩者設 `enabled:false` 時再回來處理。
- 「每個網路座位的權重都在版控」測試改指 ai/ 內路徑。
- 驗證新增:ai/checkpoints/ 內無未登記權重;無登記缺檔。
- 說明:sha256 是複製的來源證明,不要求與 data/ 同步;data/ 之後被覆蓋不算錯。
  (注意 `hc_2000`／`hc_10000` 仍在 `data/`,它們的 sha256 **仍在測試裡** ——
  這是刻意的,見 AGENTS「權重檔案放哪」。)
- **發布不設命令**(使用者裁決 2026-10-08,見「待清理項目」(b) 的結案):
  複製權重、算雜湊、手寫 `enabled: false` 的草稿列,全部手動完成 ——
  README「如何新增或更換對手」有完整步驟。原本這裡列的是一條發布命令,
  從未實作,現在連字樣一併拿掉。

## 階段 5:Python 算法型 AI 搬入 ai/heuristics/
- 只改 import 路徑,不改邏輯。
- 以全套測試與固定種子對局輸出不變為證。
- 若搬遷牽涉面過大,先回報,讓使用者決定是否暫緩,只在註冊表登記 module 路徑。

**已完成**(提交 hash 見提交訊息):
- `git mv` 七個:`ai/{wolf,chess,fox,intruder,optimizer,builder,hunter}.py` →
  `ai/heuristics/`,另加 `ai/heuristics/__init__.py`(僅 docstring)。
  `formulas`/`base`/`chooser`/`registry`/`registry.json`/`checkpoints` **不搬**。
- 改動面:七檔內的 `.base` → `..base`、`. import formulas` → `.. import formulas`;
  `ai/__init__.py` 與 `ai/registry.py` 的匯入;`rl/v3.py` 一行;
  `ai/registry.json` 七個 `module` 欄位 → `ai.heuristics.<name>`(測試比對
  `cls.__module__`);`tests/{test_hunter,test_simulation,test_rl_v3,test_registry_json,
  test_rl_features}.py` 的路徑字串;README/AGENTS 路徑與目錄樹。
  歷史檔 `plan.md`、`plan6.md`、`reports/` 未動。
- **零行為變化證據**:`run_league(2, seed=20260928, records=None)` 於 `8249f1a`
  (搬移前)與搬移後各跑一次,md5 兩邊皆 `8166309cdcddb098bdc4c5b8233d6fcc`,
  `diff` 無輸出。
- 全套測試 `1003 passed in 524.10s`,`pytest exit=0`。
- `GOLDEN_POOL` / `GOLDEN` / `LEGACY_GOLDEN_MD5` 所在兩檔未被改動,值不變。
- `records.json`:本階段開始時 `14d0809e950fc3bbfc73216ca5900255`;
  全套測試前後皆 `5ff6867c5bc98c8d3d2cf37711bb074c`(中間 22:22:11 有一筆
  測試之外的寫入,見回報,測試本身不寫它)。

## 階段 6:新池組成與文案
- 依「已定決策」更新註冊表:hc_2000、hc_10000 設 enabled:false、selectable:false(是否發布依階段 0 裁決)。
- rl_only 與 imitation_only 依註冊表 pools 定義;加交集為空的測試。
- config.I:0k 學生用「模仿學習 1000 步」說明,不套 PPO 文案;random_ai_desc 去掉不實的「模仿」措辭。
- 受影響測試先分類:全屬組成性才重採 GOLDEN;舊值寫入 README。
- `--subject` 允許 hc_* 與 rl_h1000_0k,即使不在常規池。

**已完成**(提交 hash 見提交訊息;實際順序:先存 `stage6-wip` → 階段 7 →
`git cherry-pick -n` 恢復本階段,因為十七列選單沒有捲動就放不進 720p):
- 池組成變更是唯一的行為變動來源:`all` 14、`imitation_only` 6、
  `no_imitation` 7、`rl_only` 1。**`hc_2000` / `hc_10000` 維持現狀**
  (registry 兩列、`.gitignore` 兩行、`data/` 權重、`enabled` 全不動),
  所以是 14/6 而不是已定決策的 12/4 —— 12/4 與其三項後果記在
  「待清理項目」(a)。`rl_only ∩ imitation_only == []` 由既有 preset 測試守住。
- 三份權重複製進 `ai/checkpoints/`,複製前後 sha256 與預期值三方一致。
- **β**:`imitation_step` 從 `checkpoint` 檔名 `step_%06d.pt` 解析,失敗 raise
  不回預設;哪些 key 屬 imitation 由 registry 的 `family` 決定;
  `IMITATION_KEYS` 只保留 step→key 方向(H-C2 鏈)。
  新測試:每個 imitation key 的 `imitation_step` == 檔名步數;非 step 檔名 raise。
- **config.I**:四個 1000 步席位的 label 各帶老師名(可區分);desc 只寫步數與
  老師;`random_ai_desc` 去掉池大小數字;`rl_desc_fmt` 刪掉無法引用證據的
  「比啟動它的 rl_h1000_0k 強」。`ui.seat_label` / `seat_desc` 只接線
  label/desc_key,其他行為不動。
- 三條結構性測試依裁決改寫,新舊差異寫在回報與測試 docstring 內。
- **GOLDEN 重採**(11→14,組成性):舊值 `63eb992`、新值本提交,連同釘住池
  探測的 md5(`92a99b6a…` / `97403a9f…`,diff 無輸出)都寫進 README
  「GOLDEN 重採（plan9a 階段 6）」。
- 全套測試 `1014 passed in 469.37s`,`pytest exit=0`;
  `records.json` 前後皆 `5ff6867c5bc98c8d3d2cf37711bb074c`;
  `LEGACY_GOLDEN_MD5` 未變;`python -m ai --check` 四池 = 14/6/7/1。
- 注意 `imitation_only` 現在含 rl_h1000_0k(113 萬列),與 o/b/i(約 7.9 萬列)規格不同,README 註明。

## 階段 7:UI 下拉捲動
- 對手選單選項超出可視高度時自動垂直捲動(滑鼠滾輪與鍵盤上下鍵皆可)。
- 以 20 個以上選項實測;若有測試框架,加選項數多時的測試。
- 文字走 config.I。獨立提交。

**已完成**(提交 hash 見提交訊息):
- `ui.menu_rects` 改回「可見視窗」:每頁列數與 `max_scroll` 由 `L.H` 推導,
  取消鍵先預算、永遠在畫面內且不隨捲動移動;超出的列根本不回傳,
  所以畫圖與命中測試共用同一份答案。捲動狀態 `self.menu_scroll` 在開/關
  picker 時歸 0,並在 `menu_rects` 內 clamp。
- 輸入:滾輪(`MOUSEWHEEL`,向下 → `menu_scroll` 增加,已用測試釘住方向)、
  `K_UP`/`K_DOWN` 逐列、`K_PAGEUP`/`K_PAGEDOWN` 整頁、`K_HOME`/`K_END`。
  捲軸只畫位置、不在 `menu_rects` 裡,因此不可點、也不會被當成選項。
- **未新增任何可見文字**,所以 `config.I` 沒動。
- 測試:人造 24 列清單(不用真實池)驗證捲到最後一列並點中、取消鍵在每個
  捲動位置都可點、列聯集完整、每窗內列互不重疊/全在列區內/無重複 key、
  以及 `H=600` / `720` / `900` 三種高度。原 picker 測試與
  `test_a_colour_another_seat_took_is_not_on_offer` 的「全部列同時存在」
  改為「捲動位置的聯集」,新舊差異寫在測試 docstring 與回報裡。

## 階段 8:換 AI 的工作流與工具
- `python -m ai --check`:執行全部驗證,列出各池內容。(原寫 `python -m ai.registry --check`;`ai/__init__` 匯出 `ai.registry`,跑子模組會重複執行同一個模組並噴 runpy 的 `RuntimeWarning`,所以入口掛在套件上。)
- `python -m ai --list`:顯示全部 AI、所屬池、enabled/selectable、sha256 前綴。
  (原文寫 `python -m ai.registry --list`,與 `--check` 同一個 runpy 問題,改掛套件。)
- README 新增「如何新增或更換對手」:
  1. 手動複製權重到 `ai/checkpoints/`、`sha256sum` 比對、手寫 `enabled: false` 的草稿列
  2. 編輯 registry.json,設 enabled:true 與 pools
  3. `--check` 與 `--list`
  4. 親自對戰體驗
- 池組成相關測試改為由註冊表驅動,保留不變式(如 imitation_only ∩ rl_only = ∅);新增 AI 不應需要改測試,若仍需要,記錄原因。

**已完成**(提交 hash 見提交訊息):
- `python -m ai --list` 實作:`pool_order` 列序(key 升冪)、欄位
  key/kind/family/en-sel/pools/sha256 前 12 字元/file/aliases,**不印 `note`**;
  sha256 現算現讀(`_digest_prefix`),只在 CLI 跑,不掛 import。usage 同時列
  `--check` 與 `--list`;`python -m ai.registry --list` 不做(runpy 警告)。
- **不設發布命令**,README 新增「如何新增或更換對手」的**手動**四步
  (cp + sha256sum + 手寫 `enabled: false` 草稿列 → 改 true → `--check`/`--list`
  → 親自對戰);階段 4 與本階段原本的指令字樣一併改掉,待清理 (b) 結案。
- **測試二分法**:四池大小、各類席位計數、選單列數、子進程印的池總數等
  改為與註冊表比對;`GOLDEN_POOL`、`GOLDEN`、`LEGACY_GOLDEN_MD5`、
  名册逐列 pin、六個 imitation key 字面、七個人格字面 **維持字面不變**。
  結構不變式改為顯式斷言:四池互斥(含 `imitation_only ∩ rl_only == ∅`)、
  聯集 == `all`、key 升冪、`match` 與 registry 兩邊同一批成員。
  逐組「舊擋什麼 / 新擋什麼 / 少了什麼」見 README 與回報。
- 虛構第 15 個 AI(**imitation**)與虛構第 8 個人格(**heuristic**)各實測一次全套,
  兩份紅清單都寫進 README「新增一個 AI 之後仍需改的測試」,並註明各自情境
  (純 network 席位不碰 `rl/collect`;新增人格會連帶改 `controller_weights()`
  推出的權重表,那是真連帶,不是測試寫錯)。
- 收尾修正:`seats.kind_of` 對 imitation 家族讀的是 registry 的 `family`
  (階段 6 β),所以「kind 計數 == family 計數」對那一列是同源自證、不是互證。
  兩處 docstring(`test_seats` / `test_match`)已改寫,並列明真正獨立的四條
  檢查與「逐列 pin 只釘 `key`、不含 `family`」。
  (**已被取代,2026-10-08 使用者裁決 10**:逐列 pin 已加入 `kind` / `family`
  欄位,堵住「`family` 與 `pools` 同時改錯」的漏洞;兩處 docstring 與
  README 的同一句一併改掉,此句留作當時的記錄。)

## 階段 9:records 與評測相容(**已完成**,2026-10-08)

- **程式碼只改 `records.py` 一處**:`record` 與 `set_meta` 在寫入前把 key 經
  `seats.canonical_key` 正規化(新增頂部 `from seats import canonical_key`);
  讀路徑(`_load` / `meta` / `rows` / `entries`)逐字不動。未知 key 原樣通過,
  由 `test_canonical_key_leaves_what_it_does_not_recognise_alone` 與
  `test_a_key_that_is_not_registered_is_recorded_verbatim` 釘住(裁決 1)。
  依賴方向(裁決 5):`seats` / `ai` 不 import `records`、`records` 不拉
  pygame/torch,由 `test_importing_seats_and_ai_never_pulls_in_records` 在乾淨
  子進程中斷言。
- **新記錄用新 key**:別名拼法寫入只長正式名列
  (`test_a_write_through_the_old_spelling_lands_under_the_new_key`);
  舊列不合併、原樣可讀(既有 `test_a_records_row_under_the_old_key_still_reads`,
  裁決 3);UI 對舊列顯示中文 label
  (`test_a_leaderboard_row_under_the_old_key_is_named_in_chinese`)。
- **literal 批次(裁決 2)**:批次檔 `pool` 保留原拼法,排行榜寫入端照樣正規化,
  由 `test_a_literal_run_keeps_the_batch_spelling_and_normalises_the_leaderboard`
  同時釘兩半。
- **`--subject` 經 resolve()**:非 literal 走 `validate_subject(canonicalise=True)`
  (階段 3 既有,本階段無改動);literal 保留原拼法,兩者都在 README
  「池的順序」與「排行榜資料」寫明分工。
- **已知取捨以實測改寫(裁決 9)**:原條目「hc_2000、hc_10000 若不發布到 ai/...
  只可讀、不可重跑」的條件句與現況不符 —— 兩份權重靠 `.gitignore` 的 `!` 反白
  **入版控**,乾淨 clone 目前可跑;待清理 (a) 執行後才變成可讀不可重跑。README
  「`hc_2000` / `hc_10000` 在乾淨 clone 的現況」照實測寫,不照條件句寫。
- **逐批可重現性(裁決 8,貼實測)**:六批 0.5b 可重現(literal;subject 解析到
  已發布權重,sha256 對上);b/i pair 兩檔 md5 與 README 記錄逐字一致;o 無逐局
  檔、曲線未播種 → 不可重算/不可重放;`reports/hc2` 四組 5000 局的舊 option 名
  現在全部 `ValueError`,其中 `step_005000` 與 hb2 `step_002000` 權重未入版控 →
  **不可重跑**(README「逐批可重現性核對」逐列表格,含原因);
  `data/rl1/eval.jsonl` 欄名維持;records 舊列可讀。
- **README 新增頂級章節「歷史證據與已知事項」**:四項原文抄錄+出處(裁決 6)、
  必須記錄第 4 項改寫(裁決 4)、`imitation_only` 不一致、data/ai 分工、三節
  索引、`hc_*` 取捨、逐批核對表。**不寫 5000 局互對與「觀測」小節**(確認 1:
  來源缺指令、日期、檔案);**池組成那條自訂 11 選項(7 人格 + 四個 1000 步)
  只留給日後**(確認 2);AGENTS.md 不動(確認 3)。
- 驗證:釘住池 probe(重建後先驗吻合才繼續,確認 4)前後 md5 皆
  `92a99b6ade2794ef130e341f57cf3534` / `97403a9ff49612e2ff79f0ce5467aecb`,
  `diff` rc=0;`python -m ai --check` rc=0、`--list` 14 列;`git diff` 顯示
  `ai/registry.json` 零改動、`GOLDEN`/`GOLDEN_POOL` 無改動、
  `LEGACY_GOLDEN_MD5` 仍是 `3fa9fba063a5f1cd3868098454b05b4c`;
  全套結果與 `records.json` 前後 md5 見提交訊息。

## README 必須記錄
- 0.063 規則廢止與新 val-only 規則(已完成者確認即可)。
- i 過弱觸發與處理:門檻以 hunter 系校準,對較弱老師不適用,改由使用者人工對弈判定。
- o 未播種,不可重放。
- ~~舊三模仿者移出常規池(使用者決定);hc_1000 改名 rl_h1000_0k,保留別名。~~
  **已被取代(2026-10-08 使用者裁決)**:hc_* 先裁留在常規池(`all` 14、
  `imitation_only` 6),移出改列本檔「待清理項目」(a);改名與別名部分不變。
  README「歷史證據與已知事項」的「池的組成」段照這句寫。
- imitation_only 組內規格不一致。
- 抽籤改依 key 排序及其原因;GOLDEN 重採原因與舊值、commit。
- data/ 與 ai/ 的分工:data/ 為訓練工廠(gitignore),ai/ 為發布對手(入版控)。

(以上七項已寫進 README,見階段 9 區塊:第 1、2、3、5 項在「歷史證據與已知事項」
逐項原文抄錄,第 4 項按裁決改寫,第 6 項是三節索引,第 7 項同節另段。)

## 每階段回報格式
- 提交 hash 與一句內容
- 全套測試:通過/失敗數、耗時
- records.json 是否變動
- 風險與需使用者裁決事項

## 使用者人工驗收(不屬 agent)
- 親自對戰 o/b/i,以事前寫下的「明顯失誤」定義(例如放棄大角落、明顯自封)計次判斷,再決定是否進入各自的 20k 局 RL。

## 待清理項目(使用者裁決 2026-10-07,不屬任何階段)
階段 6 刻意不碰這些,日後另排「清理 codebase 計劃」再處理:

- **(a) `hc_2000` / `hc_10000` 退出常規池**:兩列設 `enabled: false`、
  `selectable: false`(兩者必須一致,`_validate` 會擋不一致);撤 `.gitignore` 的
  `!data/hc2/step_002000.pt` 與 `!data/hc2/step_010000.pt`;
  `git rm --cached data/hc2/step_002000.pt data/hc2/step_010000.pt`(檔留磁碟)。
  在此之前兩者 `enabled: true`,所以常規池是 14 而不是已定決策寫的 12,
  `imitation_only` 是 6 而不是 4 —— 這三件事是同一件,做完才回到 12/4。
- ~~**(b) `python -m ai.registry --publish` 是否補做**~~
  **已結案(使用者裁決 2026-10-08)**:**不補做**。改為手動流程
  (`cp` + `sha256sum` + 手寫 `enabled: false` 草稿列),README 的四步工作流與
  plan9a 階段 4 / 階段 8 與 README 的工作流已同步改掉,沒有任何地方再留下
  一條看似可用的發布指令。
  日後若要重開這條,唯一的新理由是「草稿列的欄位太多,手寫會漏」。
- **(c) 重訓覆寫 `data/` 造成 sha256 測試紅的摩擦**:
  `test_every_registered_checkpoint_hashes_as_recorded` 對 `hc_2000` /
  `hc_10000` 仍算 `data/` 內的檔,重訓覆寫會讓它紅。這是刻意的(證據數字不能
  悄悄改變),但和 (a) 綁在一起:兩者退出版控與常規池之後,這條要不要改成只驗
  `ai/checkpoints/`,要一併決定。
- **(d) `tests/test_trace.py::test_trace_costs_nothing_when_not_requested`
  是計時測試**:斷言 `off <= on * 1.15 + 0.05`,拿牆鐘比較「沒開 trace 比開了
  還慢就算輸」。`-n 8` 下它和另外七個 worker 搶 CPU,會偶發變紅 ——
  **已在 `-n 8` 下紅過兩次**(2026-10-07 階段 6 首次全跑;2026-10-08 階段 6
  amend 前的全跑 `1 failed, 1013 passed`),兩次之後單獨跑或重跑都綠
  (`1 passed` / `1014 passed`)。
  AGENTS 第六節列的五處計時斷言(`test_ai` / `test_builder` / `test_intruder` /
  `test_optimizer` 的 `dt < 1.5` 與 `test_rl_paired3` 的 `elapsed < 3.0`)有
  10.5 倍餘裕,這條**不在**那個名單裡。日後要嘛放寬門檻、要嘛改成不看牆鐘的
  斷言;本輪不處理,只記下來。
- **(e) `tests/test_rl_train.py::test_a_guardrail_stops_after_saving_and_says_why`
  曾把全跑卡到 900s 逾時(2026-10-08,僅記錄,不修)**:
  - **現象**:全套跑到 98% 停住,`pytest-xdist` controller 被逾時機制殺掉後,
    留下兩個行程 ——
    `239019 [pytest-xdist running] tests/test_rl_train.py::test_a_guardrail_stops_after_saving_and_says_why`
    (`ppid=1`、`State=S`、`wchan=do_wait`、70.7% CPU)與其子行程
    `242378`(`ppid=239019`、cmdline 同名、0.9% CPU)。子行程 cmdline 與父
    相同是 fork 繼承來的,**不代表它在跑同一個測試**;`ppid=1` 是 controller
    被殺之後才被過繼給 init,不是測試自己 spawn 的野行程。
  - **子行程從哪來**:該測試的 `small_cfg(n_procs=2)` 會走
    `rl/rl_train.py:455/547` → `rl/rollout.py:663` 的
    `mp.get_context("fork").Pool(...)`,所以「同時有兩個 python 子行程」是
    設計如此(`rl/caches.py:388/414` 另有一個 fork pool,但那是資料前處理,
    這個測試不碰)。
  - **復現(2026-10-08,單獨 `-n 0`)**:`1 passed in 15.53s`,期間恰有兩個子
    行程(`336324` / `336325`,`ppid`=controller,`R` 且 100% CPU 約 3–6 秒),
    結束後**無殘留行程**。
  - **當時機器**:`load average 1.41, 5.87, 10.10`(1/5/15 分鐘)—— 全跑的
    8 個 worker 加上孤兒行程把負載推到 10。
  - **判斷**:偏「機器高載 + 逾時殺 controller 造成孤兒」,而不是「測試異常
    結束時未回收子行程」—— 單獨跑會把兩個 pool 子行程收乾淨。
    無法排除的一種可能是高載下 fork pool 的子行程卡在 `S`(0.9% CPU、
    非自旋)而父行程在 `do_wait` 等它;事後快照分不出「卡死」與「等不到
    CPU」,要分辨得在卡住當下取子行程的 stack。僅記錄,不修。
- **(f) `records.rows(order=...)` 的 `order` 是死參數(2026-10-08 實測,僅記錄,
  不修)**:`match.py:711` 收尾印排行榜時傳 `order=pool`(literal 批次就是舊拼法
  的 pool),但 `records.py` 在 return 前**無條件** `out.sort(key=(-平均分,
  餘格, key))`,連 key 都在排序鍵裡,依 `order` 組 key 清單那段永遠被蓋掉 ——
  實測 `rows(order=[...])` 與 `rows()` 輸出逐項相同,且自 first release
  (`452000f`)即如此;`ui.py:1694` 乾脆不傳 `order`。
  後果只在顯示面:列序一律平均分,literal 舊拼法 pool 不影響任何輸出(README
  「已知限制與取捨」有同一條,含實測數據)。日後二選一:讓 `order` 真的釘列序
  (docstring 的原意「keep the player first」),或把參數與 `match.py` 的呼叫
  一併刪掉。兩者都動程式碼,doc-only 輪次不碰。
