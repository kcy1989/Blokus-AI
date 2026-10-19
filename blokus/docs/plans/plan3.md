0. 性質與限制

本階段建立 RL 的基礎設施,共五項:環境(G0)、動作編碼(G1)、特徵(G2)、對局環境包裝(G3)、模仿資料收集(G4)。另有一項只記錄不修正的查證(G5)。

不寫神經網絡,不寫訓練迴圈,不寫搜尋。
不得修改的檔案

engine.py、board.py、game.py、pieces.py、config.py、ai/ 下所有檔案、ui.py、action_table.json、bench_engine.py、tools/benchmark.py、plan.md、plan2.md,以及所有既有測試檔。
允許新增

    新套件目錄 rl/(含 rl/__init__.py)。
    tests/test_rl_*.py
    reports/g_report.md、reports/g_raw.json
    資料輸出目錄 data/(不要 commit 資料檔本身,請在 .gitignore 加入 data/ 與 .venv-rl/)。

允許修改

僅 AGENTS.md 的相依性一行,以及 .gitignore。
通用原則

    既有的 322 項測試必須全部繼續通過。
    遇到與預期不符的情況,停下來如實回報,不要自行修正既有檔案。
    所有隨機性使用固定種子。
    所有「新實作對照舊實作」的測試,以既有實作為準(oracle)。新實作和 oracle 不得共用程式碼。
    模組隔離:
        rl/ 下的模組不得 import pygame、ui。
        engine.py 不得 import torch。
        torch 只允許出現在 rl/ 下。本階段只有 G0 的煙霧測試腳本會用到 torch。
    這份指令中的每個數字(例如 91、36400、58)都是預期值。實測不符時,以回報為準,不要改測試去配合。

G0. 環境:安裝 PyTorch 並驗證 GPU

**背景:**上一階段回報 torch 未安裝。GPU 是 RTX 5060 Ti,16 GB,compute capability 12.0(Blackwell)。我的理解是這張卡需要 CUDA 12.8 以上的 PyTorch 預編譯版本,但這未經驗證,所以本項要求實測。
步驟

    建立虛擬環境:python3 -m venv --system-site-packages .venv-rl,啟用後確認 import pygame, numpy 可用。若不可用,在該環境內用 pip 安裝。
    安裝 PyTorch:pip install torch --index-url https://download.pytorch.org/whl/cu128
    若安裝失敗、torch.cuda.is_available() 為 False,或出現「sm_120 is not compatible」之類錯誤:停下來,貼出完整錯誤訊息,不要嘗試繞過或換版本。
    寫 rl/smoke_gpu.py,驗證並回報:
        torch 版本、torch.version.cuda、torch.cuda.is_available()、get_device_name(0)、get_device_capability(0)、get_device_properties(0).total_memory。
        numpy 2.5.3 與 torch 互通:torch.from_numpy(np.zeros((2,2), np.float32)) 與 .numpy() 往返。
        fp16 煙霧測試:torch.autocast("cuda", dtype=torch.float16) 下,對輸入 (256, 14, 20, 20)、64 通道、3×3 的卷積做一次前向與一次反向傳播,確認無錯誤、梯度非 NaN。
        bf16 同樣測一次,回報是否可用。
    吞吐粗測(標註為「量測非結果」,模型用完即丟,不存入 repo):建立一個 6 個殘差塊、64 通道、輸入 14 通道的 ResNet,每塊兩層 3×3 卷積加 BatchNorm 加 ReLU。policy 頭用 1×1 卷積輸出 91 通道(對應 (91, 20, 20)),value 頭輸出 4 個值。在 batch = 64、256、1024 下,用 fp16 autocast 與 torch.inference_mode(),回報每秒推論的局面數(預熱 10 次後取 50 次平均)。另外回報 batch = 256 時一次「前向加反向加優化器步」的耗時。
    更新 AGENTS.md 的相依性一行為:「pygame、numpy(只允許出現在 engine.py 的 E3 段落與 rl/ 下)、torch(只允許出現在 rl/ 下)」。
    新增測試 tests/test_rl_isolation.py:在乾淨子進程中 import engine 之後,sys.modules 裡不得有 torch。仿照既有的 numpy 隔離測試。再加一項:import rl.actions 之後,sys.modules 裡不得有 pygame 與 torch。
    在新虛擬環境中重跑全部既有測試,確認仍是 322 項全過。

G1. 動作編碼(rl/actions.py)
定義

Copy
import numpy as np
from engine import (ORIENTS, PIECE_ORDER, N_PIECES, rotate_move,
                    unpack_move_mask, legal_move_mask)
from config import B, N, CLOCKWISE_OWNERS

ORIENT_OFFSET = [0]
for p in range(N_PIECES):
    ORIENT_OFFSET.append(ORIENT_OFFSET[-1] + len(ORIENTS[p]))
N_ORIENT = ORIENT_OFFSET[-1]      # 必須是 91
N_ACTIONS = N_ORIENT * N          # 必須是 36400

def move_to_index(move):          # move = (piece_idx, oi, base)
    piece_idx, oi, base = move
    return (ORIENT_OFFSET[piece_idx] + oi) * N + base

需要實作

    index_to_move(idx):move_to_index 的反函數。
    legal_indices(state, owner=None):回傳排序後的 np.int32 一維陣列,是真實座標下的動作索引。實作時直接從 legal_move_mask 的位元解碼,不要經過 Python 的 set。
    LUT_ROT:長度 4 的列表,LUT_ROT[k] 是長度 36400 的 np.int32 陣列。LUT_ROT[k][i] 是動作 i 被順時針旋轉 k 次後的索引。若 i 的 base 不在該方向的 valid 集合內(即 i 不是真實動作),填 −1。必須用 engine.rotate_move 反覆套用來建立,不得另寫旋轉公式。
    real_to_view(idx_array, p) 與 view_to_real(idx_array, p):
        p = CLOCKWISE_OWNERS.index(state.to_move)。
        真實座標轉行動者視角,是旋轉 k = (4 - p) % 4 次(與 engine.normalize 一致)。反向是旋轉 p 次,即 LUT_ROT[p]。
        輸入若含 −1(非真實動作),拋出例外。
    legal_mask_view(state):回傳長度 36400 的 bool 陣列,在行動者視角座標下,合法動作處為 True。

測試(tests/test_rl_actions.py,全部必須通過,任何失敗都要回報)

    ORIENT_OFFSET[-1] == 91。對 action_table.json 的 actions,逐項檢查 table["actions"][j] == [PIECE_ORDER[p], oi],其中 j 依 ORIENT_OFFSET 展開。
    index_to_move(move_to_index(m)) == m 對全部 30,433 個落點成立,且 30,433 個索引互不相同、全部在 [0, 36400) 內。
    legal_indices(state) 等於 sorted(move_to_index(m) for m in unpack_move_mask(legal_move_mask(state)))。用 100 局隨機對局的所有局面(種子 0 到 99,沿用 bench_engine.random_playout 的抽法,但如果 mask 為 0 的局面直接跳過)。
    LUT_ROT[0] 在所有真實動作上是恆等。LUT_ROT[k] 在 30,433 個真實動作上都不是 −1,且是 30,433 個真實動作的一個排列。LUT_ROT[1] 連套四次回到恆等。
    旋轉等變性:取 300 個隨機局面(覆蓋四位行動者,各 75 個),對每個局面 s、每個 k 屬於 {1,2,3}:令 s_k 為 engine.rotate_state 連套 k 次的結果,則 sorted(LUT_ROT[k][legal_indices(s)]) == sorted(legal_indices(s_k))。 (這裡用 rotate_state 而不是 normalize,理由見 G5。)
    view_to_real(real_to_view(x, p), p) == x,對每個 p 與全部真實動作成立。
    legal_mask_view(state) 的 True 個數等於 legal_move_mask(state).bit_count(),且 np.flatnonzero(legal_mask_view(state)) 經 view_to_real 後等於 legal_indices(state)。
    空盤四個玩家各自的 legal_indices 數量都是 58。

G2. 特徵(rl/features.py)
設計目標

    比 engine.to_plane 快(目前平均 0.26 ms,其中 normalize 佔 65%)。
    在原有 7 個通道之上擴充。
    前 7 個通道必須與 engine.to_plane 逐位元相同,這是正確性的 oracle。

通道定義

視角為行動者視角,座位順序為「我、下家、對家、上家」,即 engine.normalize 回傳的 seat_of 順序。
通道序號	名稱	內容
0 到 3	me, next, opposite, previous	四位玩家的石頭
4	my_need	與 engine.to_plane 相同
5	my_avoid	與 engine.to_plane 相同
6	empty	空格
7	next_need	下家的 need & ~avoid & empty
8	opposite_need	對家的同上
9	previous_need	上家的同上
10 到 13	start_me, start_next, start_opposite, start_previous	若該玩家尚未開局,其起始角格為 1,否則全 0

對手的 need/avoid 用既有的 engine.need_avoid(view_state, seat) 取得(在 normalize 後的 view 上,seat 為 0 到 3)。尚未開局者的 need_avoid 回傳空集合,這是預期行為,由 start_* 通道補上資訊。
標量與手牌

scalars:float32,長度 12,順序固定為:

    stuck_me, stuck_next, stuck_opposite, stuck_previous(0 或 1,取自 state.stuck,順序依座位)
    remaining_me, remaining_next, remaining_opposite, remaining_previous(各除以 89)
    opened_me, opened_next, opened_opposite, opened_previous(該玩家石頭是否非空)

hands:float32,形狀 (4, 21),與 engine.hand_vectors 相同。
實作要求

    只做一次 normalize,或者完全不用 normalize:先在真實座標算出所有位元遮罩,轉成平面後,用 np.rot90 旋轉 k 次,再依座位順序重排通道。
        np.rot90(a, k=-1) 是順時針 90 度,對應 (x, y) -> (B-1-y, x)。這個方向的判斷不要假設,以測試為準。
        若直接用 normalize 也可以,但此時必須只呼叫一次,並且同時產出平面、標量、手牌。
        兩種做法選更快的一種。報告中說明選了哪一種及兩種的耗時對照。
    提供 featurize(state) -> (planes, scalars, hands),形狀分別為 (14, 20, 20)、(12,)、(4, 21),dtype 皆為 float32。
    提供批次版本 featurize_batch(states) -> (planes, scalars, hands),第一維為 batch。
    常數 PLANE_CHANNELS_V2 記錄 14 個通道名,順序固定。
    不得修改 engine.py 既有的 to_plane、hand_vectors。

測試(tests/test_rl_features.py)

    featurize(s)[0][:7] 與 engine.to_plane(s) 逐位元相同。featurize(s)[2] 與 engine.hand_vectors(s) 逐位元相同。用 300 個隨機局面(種子 0 到 99,覆蓋四位行動者與開局、中盤、殘局),全部通過。
    旋轉不變性:對同一局面,rotate_state 連套 1、2、3 次後,featurize 的三項輸出與原局面逐位元相同。
    構造一個局面,使下家尚未開局、對家已開局:驗證 start_next = 1 的位置在視角座標中正確(下家的起始角格,在行動者視角下的位置)、start_opposite 全 0、opened_* 與 stuck_* 的座位順序正確。
    構造一個「兩個對手的 need 與 avoid 重疊」的局面,驗證 *_need 通道確實排除了 avoid。
    featurize_batch 與逐個 featurize 堆疊的結果相同。
    速度:報告 featurize 在開局、中盤、殘局的平均耗時(毫秒),以及與「to_plane 加 hand_vectors」的對比。這是量測,不是測試斷言。

G3. 對局環境包裝(rl/env.py)
目的

隱藏兩個陷阱:

    state.to_move 指向的玩家可能已無合法步(F′ 實測:5.7% 的決策點、84% 的局)。
    終局名次的並列處理。

介面

Copy
class BlokusEnv:
    def __init__(self): ...
    def reset(self): ...                  # 回到空盤,turn_order = CLOCKWISE_OWNERS
    @property
    def state(self): ...                  # 目前的 engine.State
    @property
    def done(self): ...
    def legal_indices(self): ...          # 真實座標,排序的 np.int32;done 時為空陣列
    def step(self, real_index): ...       # 套用動作,然後自動 pass,直到某位玩家有合法步或遊戲結束

行為規定

    reset 後 turn_order == CLOCKWISE_OWNERS,並用 assert 確認。
    step(idx) 流程:
        驗證 idx 在 legal_indices() 內,否則拋 ValueError。
        用 index_to_move 還原,呼叫 engine.apply_move。
        自動 pass 迴圈:當 not is_over(state) 且 legal_move_mask(state) == 0,呼叫 engine.pass_turn,並累計 passes。連續 pass 超過 100 次就拋例外。
    step 回傳 dict:{"done": bool, "passes_after": int}。
    **環境對外不暴露「沒有合法步的 to_move」。**對外保證:done 為 False 時,legal_indices() 一定非空。
    提供 utilities(state):回傳長度 4、依真實 owner id 索引的 float 列表。
        名次效用:每位玩家的餘格由小到大排。名次用「平均名次」處理並列(例如餘格 [5, 5, 9, 12] 的名次是 [1.5, 1.5, 3, 4])。
        效用 = (4 - 平均名次) / 3,即第 1 名 1.0,第 4 名 0.0。
        四人效用總和恆為 2.0。
    提供 aux_targets(state):回傳長度 4、依行動者視角座位順序(我、下家、對家、上家)的餘格除以 89。
    提供 utilities_in_view(state):把效用重排成行動者視角座位順序。座位順序由 engine.normalize(state) 的 seat_of 取得。

測試(tests/test_rl_env.py)

    1,000 局隨機對局(種子 0 到 999,用 BlokusEnv 驅動):不出錯、全部 done、每步的 legal_indices() 非空。
    對每局終局:utilities 總和為 2.0(容許 1e-9);並與 tools.benchmark.average_ranks 在同一餘格輸入下算出的名次一致(這裡可以 import tools.benchmark 作為 oracle)。
    並列測試:構造餘格 [5, 5, 9, 12],效用應為 [(4-1.5)/3, (4-1.5)/3, 2/3, 0]。三方並列和四方並列也各測一個。
    使用 BlokusEnv 隨機對局的終局餘格,和 engine 直接隨機對局(bench_engine.random_playout,同種子)的終局餘格,逐局相同。
    step 對非法動作拋 ValueError。對 done 的環境呼叫 step 也要拋例外。
    座標還原測試(這個最容易出錯,且出錯時不會報錯):1,000 個局面,隨機選一個合法動作 a_real。令 v = real_to_view(a_real, p);驗證 view_to_real(v, p) == a_real,且 v 在 legal_mask_view(state) 內。

G4. 模仿學習資料收集(rl/collect.py)
目的與關鍵風險

用 Optimizer 人格產生「局面 → 老師選擇」的資料。

**關鍵風險:Optimizer 完全確定,四個 Optimizer 互打,每一局都一模一樣。**資料沒有多樣性。所以必須加入下面的多樣性機制。
每局的設定(由局種子 seed 決定,用 random.Random(seed))

    隨機開局前綴:從 {0, 2, 4, 6, 8, 10, 12} 中抽一個 R,前 R 個真實手(不含 pass)由「隨機合法步」代替。
    座位控制者:每個座位獨立抽一種人格,機率為 optimizer 50%、wolf 12.5%、chess 12.5%、fox 12.5%、intruder 6.25%、builder 6.25%。若四座都沒抽到 optimizer,重抽整局設定,確保每局至少有一個 optimizer 座位。
    標籤的產生:在每個真實決策點(legal_move_mask != 0),若該座位的控制者是 optimizer,則呼叫 optimizer 的 choose_move 取得標籤。即使在隨機前綴期間(實際下的是隨機步),也照樣取標籤。這樣前綴期間的局面也有標籤。
    實際執行的步:前綴期間下隨機步,之後按該座位控制者的選擇。
    pass 的決策點不得產生樣本。

呼叫 choose_move 的要求

    必須關閉掛鐘預算:對 ai.formulas.USE_WALL_BUDGET 暫時設為 False,並在 finally 還原。這個做法 bench_engine.py 已用過,照抄其寫法。
    呼叫 choose_move 需要 Game/Board 物件,請仿照 bench_engine.py 的 F4 段與 tests/test_engine_cross.py 的 test_engine_tracks_a_live_game_exactly,用 Game 與 engine.State 同步推進。每一步斷言 engine 的 stuck 與 Game 的 stuck 相同。
    只讀、不修改 ai/ 下任何東西。
    取得 picked(Board 格式 (name, oi, base))後,用 (PIECE_IDX[name], oi, base) 轉成 engine 格式,再用 move_to_index 轉成真實動作索引。轉換後必須在 legal_indices(state) 內,否則拋例外中止。
    同時從 trace 取得短名單,轉成 (score, 真實動作索引) 列表,以及 n_candidates。

每筆樣本記錄的欄位
欄位	dtype/形狀	說明
own_bits	uint8, (4, 50)	四位玩家的石頭,以 int.to_bytes(50, "little") 存,依真實 owner id 索引
hand_bits	uint32, (4,)	手牌,依真實 owner id
stuck	bool, (4,)	依真實 owner id
to_move	int8	真實 owner id
action	int32	老師選擇,真實座標動作索引
n_legal	int16	合法步數
n_candidates	int16	trace 的 n_candidates
shortlist_idx	int32, (40,)	短名單的動作索引,不足補 −1
shortlist_score	float32, (40,)	對應分數,不足補 NaN
ply	int16	此局第幾個真實手(從 0 起)
game_id	int32	局種子
in_prefix	bool	該局面是否位於隨機前綴期間
outcome_util	float32, (4,)	此局終局效用,依真實 owner id(對局結束後回填)
outcome_remaining	float32, (4,)	此局終局餘格除以 89,依真實 owner id
儲存

    每個 worker 寫一個 .npz 分片(np.savez_compressed)到 data/,每分片最多 20,000 個樣本。
    檔名含種子範圍,例如 data/imit_train_1000000_1000099.npz。
    寫一個 data/manifest.json,記錄每個分片的檔名、種子範圍、樣本數、各欄位 dtype 與 shape、產生時的程式碼 commit hash、action_table.json 的 hash。
    提供讀回函式 rl.collect.load_shard(path),回傳 dict of arrays。

種子範圍與資料切分

    訓練局種子:1_000_000 起。驗證局種子:2_000_000 起。兩者不得重疊。
    這兩個範圍不得與 bench_engine.py(種子 0 到 199)和 D2 基準賽使用的種子重疊。請在程式裡檢查並 assert,D2 基準賽使用的種子範圍請從 tools/benchmark.py 讀出並在報告中寫明。
    按局切分,同一局的樣本不得同時出現在訓練與驗證。

並行

    用 multiprocessing,12 個進程(--workers 參數,預設 12),每個進程處理一段連續種子。
    進程間不得共用 RNG。
    同一組種子重跑,輸出必須逐位元相同(分片內樣本順序按 game_id、ply)。

本階段的資料量

    先跑試產:訓練 2,000 局、驗證 200 局。
    回報:樣本總數、耗時、每局平均耗時、磁碟用量。
    之後的大規模收集等我看過試產報告再決定,本階段不要自行擴大。

試產後必須回報的統計

    訓練與驗證的樣本數、局數。
    各座位控制者的人格分佈(實際統計,應接近上述機率)。
    in_prefix 比例。
    n_legal 的平均與最大。
    shortlist 長度的平均與最大。
    老師動作有多少比例同時是短名單的第一名(預期應接近 100%,因為 Optimizer 不犯錯)。若不是,這是異常,要列出。
    局面多樣性:試產資料中重複局面(完全相同的 own_bits、hand_bits、to_move)的比例。這是檢驗隨機前綴有沒有發揮作用的核心指標。
    每個真實 ply 區間(每 8 手一段)的樣本數。
    終局效用的分佈:optimizer 座位的平均名次效用(用來看資料中 optimizer 座位的實際強度)。

測試(tests/test_rl_collect.py,要快,整檔小於 60 秒)

    同一個種子連跑兩次,產出的樣本逐欄位相同。
    每筆樣本的 action 都在重建出的局面的 legal_indices 內(用 20 局抽查,重建方式:由 own_bits、hand_bits、stuck、to_move 與 CLOCKWISE_OWNERS 構造 State)。
    pass 決策點沒有出現在樣本中:樣本局面的 legal_move_mask != 0 全成立。
    驗證集的 game_id 與訓練集沒有交集。
    標籤取自 optimizer 座位:隨機抽 20 局,驗證樣本的 to_move 在該局的 optimizer 座位集合內。
    開啟與關閉 USE_WALL_BUDGET 的還原:收集函式執行完後,ai.formulas.USE_WALL_BUDGET 回到原值,即使途中拋出例外。

G5. 查證:normalize 與開局規則的潛在陷阱(只記錄,不修)

我的疑點(可能錯,需要測試證實或推翻):

    engine.normalize 回傳的 view 把「行動者」放在索引 0,並設 to_move = 0。
    但 engine 的開局規則用 OWNER_CORNER[owner],而 owner 0 的角格是 (19, 19),即右下角。
    所以對 view 呼叫 legal_move_mask,若行動者尚未開局,可能把「owner 0 的右下角」當作起始角格,算出錯誤的合法步。
    已開局的玩家不受影響。to_plane 本身沒有踩到,因為開局時 need_avoid 回傳空集合。

測試腳本(寫在 tests/test_rl_normalize_probe.py):

    收集 200 個「行動者尚未開局」的局面(隨機對局的前幾手,涵蓋四位行動者)。
    對每個局面 s,計算 view, seat_of = engine.normalize(s),然後比較:
        A = real_to_view(legal_indices(s), p) 的集合
        B = legal_indices(view) 的集合(對 view 直接呼叫引擎)
    回報:A 與 B 相等的局面數、不相等的局面數,按「行動者 owner」分組。若不相等,各舉一個例子,印出 serialize(s) 與兩邊的合法步數。
    再對 200 個「行動者已開局」的局面做同樣比較。
    這個測試無論結果如何都要回報,但不要把「不相等」當成測試失敗(測試用 pytest.mark 記錄結果即可,或只輸出報告,不斷言)。不要修改 engine.py。

**無論結果如何,rl/ 的所有程式一律遵守下面這條規則:合法步只在真實座標的 State 上計算,再用 LUT_ROT 轉到視角座標。**不得對 normalize 回傳的 view 呼叫 legal_move_mask 或 apply_move。
G6. 報告與提交
報告格式(reports/g_report.md)

對話中最後要貼回給我的內容就是這份報告的全文。

Copy
# 階段 G 報告
## G0 環境與 GPU(含完整版本資訊、吞吐表,吞吐標註「量測非結果」)
## G1 動作編碼(各測試結果與實際數字)
## G2 特徵(通道表、測試結果、耗時對照表:開局/中盤/殘局)
## G3 環境包裝(測試結果)
## G4 試產資料(第 G4 節要求的九項統計,逐項附實際數字)
## G5 normalize 查證(A/B 比較結果,按行動者 owner 分組)
## 異常與未預期發現
## 測試結果(總數、是否全過,新增測試數)
## 提交資訊(commit hash、新增與修改的檔案清單、git status 與 git diff --stat 輸出)

報告規則

    所有表格給實際數字,不要用「約」或「大致」。
    若某項無法完成,寫明原因,不要跳過、不要編造數字。
    所有異常原樣列出,不要修改既有檔案去消除它。

驗收清單(自我檢查)

    git diff --stat 只顯示 AGENTS.md 與 .gitignore,沒有其他既有檔案被修改。
    既有 322 項測試全部通過,報告列出最終總數。
    import engine 不拉進 torch;import rl.actions 不拉進 pygame、torch。
    featurize 前 7 通道與 engine.to_plane 逐位元相同。
    G4 同種子重跑結果逐位元相同。
    G5 的結果已如實回報,即使與預期不同。
    已 commit。

