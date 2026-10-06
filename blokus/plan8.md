任務 plan8-A:RL 基礎設施只讀設計確認。不改任何檔案,不 commit。

背景與已定決策(不得自行更改):

- 目標產物:rl_1000_5k。起點與 KL 錨點 = hc_1000,共 5,000 局,每 200 局更新一次(25 輪)。
- 每局 1 席學習者 + 從 7 個人格 AI 中不放回抽 3 個,學習者座位每局隨機。
- 人格池 = hunter、optimizer、builder、intruder、fox、chess、wolf。hc*\* 與 rl*\* 不進訓練對手池。
- 資源:不得全佔機器。對局進程約 6–8 個,nice 19。
- RL 階段不做自動評測,評測由使用者以 match.py 自行進行。
- 命名:rl*<底>*<局數>。更新週期等超參數寫進 manifest,不進名稱。

禁止:改檔、commit、寫 data/、啟動訓練。量測只允許寫 /tmp,量測腳本事後刪除。
先確認無殘留進程。

請逐項回答,每項附檔案路徑與行號作證據,查不到就寫「未查到」,不要推測。

1. 人格名單
   a. 確認上述 7 個名稱就是引擎裡現有的 7 個規則式 AI(即「原 7 個風格 AI」),
   列出它們在程式碼中的 key。若名單有出入,列出差異。
   b. 這 7 個 AI 是否有隨機性?各自的決策是否只依賴盤面與自己的 RNG?

2. 網絡與推論路徑(rl/ 與網絡定義)
   a. hc_1000 檢查點的位置、格式、載入方式。網絡是否有 value 頭?
   若沒有,列出要加什麼、不改 policy 頭參數的前提下如何加。
   b. 輸出空間:logits 的維度、與 action table 的對應、合法步遮罩如何套用。
   imitation 訓練時的 shortlist(shortlist_idx、n_candidates)在推論時
   是否參與?RL 的 policy 應在全部合法步上做 softmax,還是在 shortlist 上?
   說明兩者的差別與現有程式碼支持哪一種。
   c. 取得 log_prob、entropy 所需的最小新增介面(只列簽名,不實作)。
   d. 單步推論一次與批次推論的實測延遲(fp32,GPU 與 CPU 各一),
   僅用 /tmp 腳本。

3. 環境(rl/env.py、game.py、engine.py)
   a. 現有 BlokusEnv 是否支援「由外部提供某一席的動作」?若否,
   最小改動點在哪。
   b. 學習者座位如何插入現有座位機制(seats.py)?座位隨機化已有的機制是什麼?
   c. 一局的決策點數:學習者席平均每局幾次決策?(以 200 局人格對局實測,
   含「to_move 無合法步」被跳過的比例。)
   d. 終局資訊:可得的 outcome 欄位(outcome_util、outcome_remaining 等)
   與 match.py 的名次分定義。列出可作獎勵的候選,
   不要替我選。指出各自的優缺點(稀疏性、與評測指標的一致性)。
   e. 同種子重現性:相同種子與相同學習者策略,軌跡是否位元級重現?

4. 對局速度(關鍵,決定牆鐘)
   a. 純人格對局(無網絡):單進程每局秒數(已知約 0.58)。
   b. 含學習者 CPU 推論 與 含學習者 GPU 推論:單進程每局秒數,各 50 局。
   c. 多進程各自載入網絡到 GPU,與「集中推論伺服器批次化」兩種架構,
   各自的優缺點與預估瓶頸。僅分析,不實作。
   d. 在 nice 19、6 與 8 進程下,5,000 局的牆鐘預估(用實測外推,
   標明外推假設)。若預估超過 3 小時,明確標出。

5. 種子與保留區段
   a. RESERVED_RANGES 現況,RL 種子區段應放在哪裡才不與
   模仿資料(6000000–6101545)、評測種子(910001–910004 等)衝突。
   b. 5,000 局訓練種子建議區段與 manifest 應記錄的欄位。

6. 註冊 rl*1000_5k 進常規池需動的位置
   a. 列出所有需要變動的地點(seats.py、池定義、is_imitation_key、
   match.py、測試、文件)。
   b. 池從 10 變 11 會改變抽樣分布:列出哪些黃金值/測試要重建,
   並確認這與 plan6 的先例流程一致。
   c. 「key 與目錄綁定」測試要怎麼擴充,避免再出現靜默載入。
   rl* 前綴是否會被現有 startswith 判斷誤分類?

7. 守護缺口(上次發現)
   確認目前沒有「全套測試前後 data/ 檔案清單與 size 不變」的測試,
   並提出該測試的最小設計(只提案,不寫)。

8. 風險清單
   a. 對手只有規則式 AI 的分布過擬合風險,程式面有無可低成本緩解的設計
   (例如記錄對各人格的分項得分)。
   b. hc_1000 作為 KL 錨點的穩定性風險。
   c. 其他你在讀碼中發現的、我沒問到的問題。

9. 超參數:只給建議與理由,不下定論
   PPO epoch 數、minibatch、clip、學習率、KL 係數、GAE 的 gamma/lambda、
   value 頭熱身步數。每項附依據(來自 4 的決策量、網絡規模等)。
   學習率須明確低於模仿階段的 1e-3 並說明量級理由。

回報格式:

- 以上 9 項逐項作答,每項結尾標「已證實 / 推測 / 未查到」。
- 實測數字附樣本數。
- 與指示不符之處。
- 建議 plan8-B 的切分(哪些模組、哪些測試),不寫實作。
- 確認 git status 乾淨、data/ 檔案數不變。

任務 plan8-B0:RL 前置(三個獨立 commit,依序 a、b、c)。
不訓練、不動 data/ 內容、不動 ai/ 的行為、不動 rl/features.py 與 rl/actions.py。
開始前確認無殘留進程。

已定決策(供背景,本任務不得更動):

- 目標產物 rl_1000_20k:起點與 KL 錨點 hc_1000,20,000 局,每 500 局更新一次(40 次)。
- 訓練種子區段 7,000,000–7,019,999;驗證保留 7,100,000–7,100,999。
- 獎勵 (5-名次) - 餘格/100,名次用 records.rank_rows(本任務不實作獎勵)。

== plan8-B0a:device 修復與文件校正 ==
授權例外:允許修改 rl/imitation.py 的 logits_from_state(僅限遮罩搬到 device)。

1. rl/imitation.py:157 的 numpy 遮罩轉成 tensor 後 .to(logits 所在 device)。CPU 路徑必須位元級不變。
2. 新增測試:
   a. CPU 回歸:修復前後同一批位置的 logits 與選步位元級相同。先在修復前存參考值到 /tmp。
   b. CUDA 測試(無 CUDA 則 skip):同一批位置,CPU 與 CUDA 的遮罩位置完全一致,
   合法槽 logits 差 < 1e-3,argmax 相同。
   c. match.py 以 --device cuda --pool hc_1000 --games 2 --dry --out /tmp/... 能跑完(手動驗證,結果貼在回報)。
3. 文件校正。先 grep 核對每一處,與下列清單不符的以實際為準並回報:
   - seats.py:41, 100, 126, 132 的 "eleven" 應為 ten
   - match.py:62 的 "eleven" 應為 ten
   - tests/test_match.py:93、tests/test_match_pool.py:435 的 "eleven" 應為 ten
   - rl/imitation.py:1 的 "H-B2" 應為 "H-C2"
   - ai/base.py:15 註解「3 加 3」應為 3 加 4
     不得改任何有行為的程式碼,不得改 tests/test_match_pool.py:289-290 描述變更本身的 "eleven options to ten"。
4. 驗收:全套 -n 8、串行 wall-clock、match.py 未給 --pool 與 --pool all 對改動前黃金值逐局相同。
5. 回報:commit hash、git status、data/ 檔案數、與指示不符之處。

== plan8-B0b:種子保留區段守護 ==
授權例外:允許修改 rl/paired3.py 的 RESERVED_RANGES(僅新增項目)。
若有其他模組消費 RESERVED_RANGES 而新增會改變其行為,停下回報,不要自行處理。

1. 先查證實際用過的區段,不要照本指示的數字抄,以檔案證據為準:
   - H1 的 OWN_RANGES(rl/paired3.py:98-104)與 check_seed_ranges 的關係
   - data/hb1 與 data/hc1 manifest 的 train/valid 種子首尾
   - 評測種子 910001–910004 與 920K 各自實際涵蓋的種子範圍
     (先讀 match.py 如何由 --seed 展開成每局種子,再算上界)
2. 把查到的區段補進 RESERVED_RANGES,並新增 RL 區段:
   train 7,000,000–7,019,999、valid 7,100,000–7,100,999。
3. 新增測試:
   a. 全部登錄區段兩兩不重疊(列出任何重疊並回報,不要自行改歷史區段)。
   b. 7M 區段與每個既有區段不重疊。
   c. 一個公開函式(簽名由你設計,回報)接受 (lo, hi),與保留區重疊時拋錯;測試其拒絕 6,000,000–6,000,010。
4. 驗收:全套 -n 8、串行 wall-clock。
5. 回報:新增的區段清單與每一項的證據來源、commit hash、與指示不符之處。

== plan8-B0c:data/ 守護測試 ==

1. 範圍:data/h\*(資料集與檢查點目錄)與 data/records.json。排除 data/humanlog/ 與 data/match/。
   先確認 records.json 的實際路徑。
2. 快照 = {相對路徑: (size, mtime_ns)},遞迴涵蓋範圍內所有檔案。
   寫成 snapshot(root) 純函式,root 可注入。
3. 接線:全套測試前後各取一次快照並比對,不一致則整個 session 失敗並列出差異。
   重要:在 pytest -n 8 下,快照與比對只能在 controller 行程執行,
   不得在每個 worker 各做一次(用 hasattr(config, "workerinput") 區分)。
   確認串行與 -n 8 兩種模式都有效。
4. 單元測試(用 tmp_path,不碰真實 data/):snapshot 能偵測新增、刪除、就地改寫(size 不變但 mtime 變)、改名。
5. 變異驗證:暫時讓一個測試在真實 data/ 之外的臨時 root 寫檔以外,
   另以注入 root 的方式驗證守護確實紅燈;不得在真實 data/ 寫入任何東西。
   然後還原。
6. 已知限制,寫進測試 docstring:使用者在測試期間做不加 --dry 的對局,
   會改動 records.json 而讓守護誤報。失敗訊息要提示這一點。
7. 驗收:全套 -n 8、串行 wall-clock;守護在兩種模式下皆通過;data/ 檔案數不變。
8. 回報:commit hash、git status、與指示不符之處。

任務 plan8-B0d:重置 records.json(資料操作,非程式碼變更,不 commit)。

1. 確認無殘留進程,且你與使用者目前沒有進行中的對局或評測。
2. 記錄現況:records.json 的 MD5、筆數、總席數。
3. 備份現況到 data/records_pre_rl.json,驗證備份 MD5 與原檔相同。
   不得覆蓋既有的 data/records_pre_hc2.json(須確認其 MD5 仍為
   5eef3eea9041c6c73f90692e70462e56)。
4. 用專案既有的重置機制(沿用上次 hc2 重置的做法,先查明是哪個函式或指令),
   不要手寫 JSON。重置只執行一次。
5. 驗證:records.json 為空結構,可被 records 模組正常載入;
   隨後跑一次 match.py --games 2 --dry,確認 --dry 不會寫入 records.json(MD5 不變)。
6. 回報:新舊 MD5、備份路徑與 MD5、data/ 檔案數變動(預期 +1)、與指示不符之處。

任務 plan8-B1:新增 rl/policy.py(純推論與數值介面,不含訓練迴圈、不含對局迴圈)。
單獨 commit。不改 ai/、rl/features.py、rl/actions.py、rl/train.py、game.py、engine.py。
開始前確認無殘留進程。量測腳本寫 /tmp,事後刪除。

已定決策(不得更動):

- 起點 hc_1000(data/hc2/step_001000.pt),RL 的 policy 在「全部合法步」上做 softmax,
  使用與模仿訓練相同的 legal_mask_view 與 MASK_FILL。
- 獎勵 = (5 - 名次) - 餘格/100,名次用 records.rank_rows(競賽排名,平手共享名次)。
  本任務實作獎勵計算函式,不實作對局。
- value 頭存在(Conv2d(channels,4,1))但從未訓練。

== 第一步:釐清 value 頭語意(只讀,先做,結果決定後續) ==

1. value 頭輸出 (B,4,20,20)。查明:
   a. 4 個通道在網絡看到的輸入中對應什麼?輸入是 view 順序(行動者視角旋轉後)
   還是 owner 順序?讀 features_27、engine.normalize、LUT_VIEW,給出檔案行號證據。
   b. 空間維度 20×20 如何縮成標量?模仿階段沒有任何 value 損失可參考,
   列出可行的 readout 方案(例如對空間取平均、取行動者角落格、全域池化後線性),
   各自的優缺點。不替我選,但標出哪一個「不改動 trunk 與 policy 頭參數」即可實現。
   c. 因為 value 頭沒訓練,任何 readout 都是隨機初始化的。
   結論:是否建議新增一個獨立小 value 頭(新參數,不動既有頭)而不是重用 Conv2d(channels,4,1)?
   說明理由並給出新頭的最小設計(參數量、輸入、輸出標量)。
2. 回報這一步結果並停下確認設計,除非 1c 的結論明確指向「新增獨立頭」,
   此時可繼續實作,但在回報中標明。

== 實作(1c 之後) == 3. rl/policy.py 提供:
a. load_policy(path, device) -> (net, meta)
strict 載入,回傳 checkpoint 的 step 與 settings;記錄檔案 MD5。
b. forward_batched(net, states, device) -> (policy_logits (N,36400) 已遮罩, value (N,))
一次 forward 同時給 policy 與 value;遮罩與 imitation.logits_from_state 同一函式,遮罩搬到 device。
c. log_prob_entropy(masked_logits, action_index) -> (log_prob (N,), entropy (N,))
非法槽機率恰為 0 且 entropy 計算不得出現 NaN(0\*log0 要處理)。
d. kl_to_anchor(masked_logits, anchor_masked_logits) -> (N,)
同遮罩下的 KL(current || anchor),只在合法槽上求和。
e. episode_reward(remaining_by_owner, owner) -> float
= (5 - rank) - remaining/100,rank 用 records.rank_rows(不得另寫排名邏輯)。
remaining 的單位用 engine.result 的原始格數,先確認 rank_rows 吃什麼輸入。
f. 若採新增獨立 value 頭:一個明確的 ValueHead 類別與其初始化,
並保證 policy 部分的參數與 hc_1000 位元級相同(載入後比對 state_dict 的 policy/trunk 部分)。4. 測試(tests/test_rl_policy.py):
a. 遮罩:隨機 40 個位置,非法槽 softmax 機率恰為 0,合法機率和為 1(容差 1e-6)。
b. log_prob_entropy 與手算(float64 參考實作)在 1e-5 內一致;含只有 1 個合法步的位置(entropy=0,不得 NaN)。
c. kl_to_anchor:自己對自己 = 0;與 float64 參考一致;非負。
d. episode_reward:構造平手案例(兩席餘格同為 0)驗證與 records.rank_rows 同名次;
滿分案例(第1名、餘格0)= 4.0;第1名餘格5 = 3.95;最差案例的值。
e. 載入後 trunk 與 policy 頭參數與檔案內 state_dict 位元級相同。
f. CPU 與 CUDA(無 CUDA 則 skip):argmax 相同、log_prob 相對差 < 2e-3。
g. 同一批位置,batch=1 逐個算與 batch=N 一次算,log_prob 的最大差,量化並記錄
(這決定 PPO ratio 的雜訊底,不要求為 0,只要求回報)。5. 驗收:全套 -n 8、串行 wall-clock;data/ 檔案數不變。6. 回報:value 語意結論(1a、1b、1c 各附行號)、介面簽名、第 4g 的雜訊底數字、
commit hash、git status、與指示不符之處。

任務 plan8-B2:新增 rl/rollout.py(對局生成器)。單獨 commit。
不訓練、不改 ai/、game.py、engine.py、rl/features.py、rl/actions.py、rl/policy.py 的既有行為。
開始前確認無殘留進程。量測寫 /tmp,事後刪除。

已定決策:

- 學習者 = hc_1000 權重(後續輪次為當前權重),其餘三席為 7 個具名人格 AI 不放回抽樣。
- 整個 rollout 包在 \_Budget(False) 內,確保 wolf/chess/fox 位元級可重現。
- 每進程 torch.set_num_threads(1),必須結構性保證(不靠呼叫者記得)。
- rollout 與更新必須同一裝置;rollout 記錄的 log_prob 即用於 PPO 舊策略。
- 學習者席位隨機(0-3),每局一個學習者。
- 訓練種子 7,000,000 起,經 reject_reserved_seeds 驗證。

== 第一步:只讀查證(先回報再實作,若結論與假設不同則停下) ==

1. 一局 4 席的流程:現有 match.py 的對局迴圈在哪?人格 AI 如何被呼叫?
   是否可重用,還是需要新迴圈?給出行號。
2. 學習者決策點的定義:到學習者回合、有合法步。學習者無合法步(pass)如何處理?
   是否計入決策?單局預期約 17 個決策,實測 20 局確認。
3. \_Budget(False) 的用法與作用域;seats.py 中 7 個人格的建構方式。
4. 每局每席的 RNG 如何由種子導出,使同一個 (seed, learner_seat, opponents) 完全可重現。

== 實作 == 5. 單局函式 play_episode(spec, net, device) -> Episode
spec = (seed, learner_seat, opponent_names 三個);
Episode 記錄每個學習者決策:state 的壓縮表示(足以重建 features 與 mask)、
action_index、log_prob(rollout 時算)、value(rollout 時算)、
以及終局:名次、餘格、reward(用 policy.episode_reward)、對手名稱、學習者席位。
動作取樣:從遮罩後 softmax 取樣(不是 argmax),溫度 1.0。6. 批次函式 play_batch(specs, weights_path, n_procs, device) -> list[Episode]
每進程各載入網絡;結構性地在進程初始化時 set_num_threads(1);
結果依 spec 順序回傳(與完成順序無關)。7. 對手抽樣函式 make_specs(start_seed, n, rng_seed)
7 人格不放回抽樣 3 個;學習者席位均勻。記錄於 manifest 欄位。
種子區段經 reject_reserved_seeds 驗證。8. 測試(tests/test_rl_rollout.py,保持快速):
a. 同一 spec 跑兩次,Episode 位元級相同(含 log_prob)。
b. 同一 spec 在 n_procs=1 與 n_procs=4 下結果相同(顯式證明與調度無關)。
c. 對手席位的動作與 match.py 以相同種子與席位跑出的完全一致(重用證明)。
若做不到一致,回報原因,不要放寬。
d. 終局 reward 與 records.rank_rows 的名次一致;四席名次分和 = 10。
e. 每局決策數落在合理範圍(回報分佈,不設死界限)。
f. 取樣有隨機性:不同 seed 下學習者的動作不同;但同 seed 完全相同。
g. make_specs:不放回、席位均勻(大樣本卡方粗檢)、種子在 7M 區段內。
h. 重載記錄的 log_prob:用 policy.log_prob_entropy 在同裝置重算,與 rollout 記錄的差 < 1e-4。9. 吞吐量實測:200 局、n_procs=8,回報局/秒與 Episode 記憶體大小。
用此推算 500 局一輪的實際牆鐘,以及 Episode 序列化後的大小(決定是否需落盤)。10. 驗收:全套 -n 8、串行 wall-clock;data/ 檔案數不變;records.json MD5 不變。11. 回報:第一步的 4 項答案(附行號)、介面簽名、吞吐量數字、commit hash、
git status、與指示不符之處。

任務 plan8-B3:新增 rl/ppo.py(單次 PPO 更新的純函式與損失,不含訓練主迴圈、不含存檔)。
單獨 commit。不改 ai/、game.py、engine.py、rl/features.py、rl/actions.py、rl/policy.py、rl/rollout.py 的既有行為。
開始前確認無殘留進程。量測寫 /tmp,事後刪除。

已定決策:

- 起點與 KL 錨點 hc_1000。獎勵 (5-名次)-餘格/100。
- 超參數候選(非定論,本任務只實作可配置,不調參):
  epochs 3、minibatch 512、clip 0.1、actor lr 3e-5、value lr 1e-3、gamma 0.99、lambda 0.95、
  KL 係數可配置(預設 0.0,由後續任務決定)、advantage 按批次標準化。
- 更新在 GPU(若可用);更新開始時用「更新裝置」重算舊 log_prob 與舊 value 作為 PPO 的 old,
  rollout 記錄的 log_prob 只用於診斷。
- 每個 Episode 的學習者決策序列即時間步序列(pass 不產生時間步)。
- 終局獎勵只在最後一個學習者決策給,中間獎勵為 0。

== 第一步:只讀查證與量測(先回報,若與假設不同則停下) ==

1. Episode 的 rewards 欄位是 owner 索引的 4 元組;學習者的獎勵如何取出?確認與 learner_seat 的對應。
2. 量 rollout 記錄的 log_prob 與 GPU 重算 log_prob 的差(用 20 局 hc_1000 rollout):
   報 max/mean 絕對差。這是「取樣策略與更新策略的失配」基準。
3. 量 GPU 上一次 minibatch 512 的前向+反向耗時,以及重算 8,800 個 old log_prob 的耗時。
   據此推算一輪更新(3 epoch)牆鐘。

== 實作 == 4. compute_gae(rewards_per_episode, values_per_episode, gamma, lam) -> (advantages, returns)
逐 Episode 獨立計算,終局後 value 視為 0。5. batch_from_episodes(episodes) -> 攤平的訓練資料(states 以 rl.caches.states_from_rows 重建,
action_index、n_legal 等)。6. ppo_update(net, anchor_net, batch, cfg, device) -> stats

- 開頭在 no_grad 下重算 old_log_prob 與 old_value(同裝置)。
- 損失 = -min(r*A, clip(r,1±ε)*A) + c_v*(v - return)^2 - c_e*entropy + c_kl\*KL(current||anchor)。
- Actor 與 value 頭用不同學習率(參數分組);value 頭的 trunk 梯度為零已由零初始化保證,
  但本任務不得假設:僅 policy 損失更新 trunk,value 損失用 detach 的 trunk 特徵,
  明確寫成結構性,並測試 value 損失不改變 trunk 參數。
- advantage 在整批上標準化(不是 minibatch 內)。
- 梯度裁剪(max norm 可配置,預設 0.5)。
- 回傳 stats:policy loss、value loss、entropy、近似 KL(新舊)、clip fraction、
  對 anchor 的 KL、explained variance、各 epoch 的值。
- 必須能被 CPU 與 GPU 呼叫;minibatch 順序由傳入的 rng 決定(可重現)。

7. 測試(tests/test_ppo.py):
   a. compute_gae:手算小例(3 步)與實作一致;gamma=lam=1 時 advantage = return - value。
   b. 零更新恆等:lr=0 時 ppo_update 後所有參數位元級不變;第一個 epoch 第一個 minibatch 的 ratio 恰為 1(同裝置重算)。
   c. value 損失不改變 trunk 與 policy 頭參數(只動 value 頭)。
   d. 有利方向:構造一批全為正 advantage 的合成資料,一次更新後這些動作的 log_prob 上升。
   e. clip 生效:ratio 超出 1±ε 且方向有利的樣本,其 policy 梯度為零。
   f. KL 錨點:c_kl>0 時對 anchor 的 KL 小於 c_kl=0 的對照組(同資料同種子)。
   g. 可重現:相同 rng 與資料,兩次更新後參數位元級相同(CPU)。
   h. 小規模端到端:play_batch 取 20 局 → ppo_update 一次 → 參數有變、無 NaN、explained variance 有限。
8. 驗收:全套 -n 8、串行 wall-clock;data/ 檔案數不變;records.json MD5 不變。
9. 回報:第一步 3 項數字、介面簽名、一輪更新牆鐘推算、commit hash、git status、與指示不符之處。

任務 plan8-B4:新增 rl/rl_train.py(PPO 主迴圈)與 CLI。單獨 commit。
本任務「先 smoke、再正式」:實作與測試完成後只跑 smoke,不啟動 20,000 局正式訓練。
不改 ai/、game.py、engine.py、features、actions、policy、rollout、ppo 的既有行為。
開始前確認無殘留進程。量測寫 /tmp,事後刪除。

已定決策:

- 起點與 KL 錨點:data/hc2/step_001000.pt (hc_1000,MD5 95dc7c46...)。
- 總 20,000 局,每輪 500 局,共 40 輪;訓練種子 7,000,000 起連續;命名 rl_1000_20k。
- rollout:CPU、8 進程、softmax 取樣、溫度 1.0;更新:CUDA。
- 每輪流程:make_specs → play_batch(當前權重) → batch_from_episodes → ppo_update → 存檔。
- 每輪用「當前權重」做 rollout:權重如何分發給 8 進程由你設計(寫暫存檢查點檔再載入最簡單),
  並量測每輪重載成本。
- 檢查點輸出目錄 data/rl1/(新目錄,不得寫入 data/h\* 既有目錄)。

== 實作要求 ==

1. 檢查點格式:與 hc_1000 同格式可被 rl.policy.load_policy 載入,
   另含 value 頭參數、優化器狀態、輪次、rng 狀態、cfg、種子游標;
   寫檔先寫暫存再原子 rename。
2. 可續跑:--resume 從最後一個完整輪次繼續,且續跑後的軌跡與不中斷的位元級相同
   (CUDA 更新若無法位元級,退到 CPU 驗證此性質並如實回報 CUDA 的差異量級)。
3. 護欄(每輪檢查,觸發則存檔後乾淨停止並印原因):
   - kl_anchor > 0.05
   - entropy < 起始熵的 50%
   - 任何 NaN/Inf
   - first_ratio_dev > 1e-6
   - 磁碟剩餘 < 5 GB
     閾值可配置。
4. 每輪日誌(JSONL):輪次、種子區段、rollout 秒、更新秒、重載秒、
   平均獎勵、平均名次、學習者名次分佈、依對手名稱分項的平均獎勵、
   stats 全部欄位。stdout 一行摘要。
5. 輪內評測(便宜版):每 5 輪,用當前權重對固定 7 人格各自的固定種子(7,100,000 起),
   argmax 取行動,跑 100 局,報告 avg_points 與餘格;與同一批種子的 hc_1000 基線並排。
   評測局不進入訓練資料、不寫 records.json。
6. 防 nice:所有訓練行程以 nice 19 啟動;結束或中斷時不留殘留進程(含 worker)。
7. CLI:--games、--per-round、--resume、--out、--dry-run(只印計畫與預估,不執行)、
   --max-rounds(smoke 用)。
8. 測試(tests/test_rl_train.py,快速):
   a. 2 輪 × 10 局的微型訓練可跑完;檢查點可被 load_policy 載入且與 hc_1000 的 policy 參數不同。
   b. resume:跑 2 輪 vs 跑 1 輪後 resume 再 1 輪,CPU 上參數位元級相同。
   c. 護欄:構造使 kl 閾值極小的設定,確認訓練存檔後停止並回報原因。
   d. 原子寫:模擬寫到一半中斷,不留殘缺的「最新」檔。
   e. 種子游標:連續兩輪的種子區段不重疊,且全在 7M 區段內。
   f. 評測不寫 records.json、不寫 data/h\*。
9. smoke(實跑):--max-rounds 2 --per-round 100,回報兩輪日誌、重載成本、評測輸出。
10. 驗收:全套 -n 8、串行 wall-clock;data/ 既有檔案不變(data/rl1 為新增,說明守護測試如何處理它);
    records.json MD5 不變。
11. 回報:CLI 用法、smoke 日誌、重載成本、每輪牆鐘、commit hash、git status、與指示不符之處。
