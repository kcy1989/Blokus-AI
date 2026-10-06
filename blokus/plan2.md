0. 任務性質與硬性限制

這是純量測與查證階段。目的是取得後續架構決定所需的事實數字。

禁止事項(違反任何一條就視為失敗):

    不修改 engine.py、board.py、pieces.py、config.py、ai/ 下任何檔案、ui.py、action_table.json。
    不做任何最佳化,即使發現某處很慢。只回報,不修。
    不安裝 torch 或其他套件。只檢查已安裝的。
    不寫任何訓練程式碼。
    發現異常(例如引擎與預期不符)時,停下來如實回報,不要自行修正。

只允許新增:

    bench_engine.py(主量測腳本)
    tests/test_bench_engine.py(少量測試,見第 7 節)
    reports/f_prime_report.md(最終報告)
    reports/f_prime_raw.json(原始數字)

所有隨機性都用固定種子。耗時數字不可能逐位重現,所以報告分兩區:「結果區」(合法步數、次數等,必須逐位重現)與**「量測區」**(耗時,標註「量測非結果」)。腳本要提供 --no-timing 選項,只輸出結果區。
1. 共用工具:隨機對局函數

在 bench_engine.py 內寫一個隨機對局函數,語義如下:

Copy
import random
from engine import (initial_state, is_over, legal_move_mask,
                    unpack_move_mask, apply_move, pass_turn)

def random_playout(seed, hook=None, max_passes=1000):
    """用 random.Random(seed) 隨機選合法步,直到 is_over。
    - 合法步排序後再抽:moves = sorted(unpack_move_mask(mask)),
      rng.randrange(len(moves)),確保跨環境可重現。
    - 若 legal_move_mask(s) == 0,呼叫 pass_turn(s),passes += 1。
    - passes 超過 max_passes 就 raise,防止無窮迴圈。
    - hook(state, mask, n_real_moves, n_passes) 在每次「做決定之前」被呼叫
      (包含 mask == 0 的情況),供量測用。
    - 回傳 (final_state, n_real_moves, n_passes)。
    """

「階段」定義,依已下的真實手數 n_real_moves 分三段:
階段	範圍
開局	0 ≤ n < 16
中盤	16 ≤ n < 48
殘局	n ≥ 48
2. F1:engine.py 速度量測

**資料:**種子 0 到 199 的 200 局隨機對局。用 hook 收集每個決策前的 State。

A. 整局耗時(200 局全用):

    每局總耗時(time.perf_counter),回報平均、中位數、p95、最大值(秒)。
    每局真實手數、pass 次數,回報平均與最大。
    估算「每秒完成的真實手數」。

B. 單函數耗時(用種子 0 到 99 的 100 局收集到的全部局面):

對每個局面,各函數呼叫一次並單獨計時(先對前 50 個局面預熱,不計入)。對以下每個函數,按開局/中盤/殘局/全部四欄回報 n、平均、中位數、p95、最大(單位毫秒):

    legal_move_mask(state)
    legal_moves(state)(含解包成 frozenset)
    has_any_legal(state, state.to_move)
    apply_move(state, m),其中 m 為該局面隨機對局實際選的步。這需要 hook 同時記錄實際選的步,請在 hook 之外另存 (state, move) 對。
    pass_turn(state),只對 mask == 0 的局面量(樣本少就如實回報 n)。
    to_plane(state)
    hand_vectors(state)
    normalize(state)
    serialize(state) 與 deserialize(blob)

另外回報:to_plane 耗時中,normalize 佔多少比例(用第 6、8 項的平均值相除即可)。

**C. 環境資訊記錄:**Python 版本、platform.processor()、os.cpu_count(),並註明量測時為單進程。
3. F1b:合法步數分佈

**資料:**同上,種子 0 到 199 的 200 局,每個「真實決策點」(mask != 0)一筆。

用 int.bit_count() 數 legal_move_mask 的 1 的個數,回報:

    全體與按階段(開局/中盤/殘局):平均、中位數、p95、最大、最小。
    按真實手數分段(每 8 手一段:0–7、8–15、……):平均與最大。
    有合法步的棋子種類數(mask 中不同 piece_idx 的個數):平均與最大。
    mask == 0 的決策點占全部決策點的比例(這些需要 pass)。

這是結果區(必須逐位重現)。
4. F2:動作表與引擎的一致性

逐項檢查並回報 PASS/FAIL 及實際數字,不要因 FAIL 而修改任何檔案:

    sum(len(ORIENTS[p]) for p in range(N_PIECES)) 是否等於 91。
    sum(od["valid"].bit_count() for p in ORIENTS for od in p) 是否等於 30,433。
    讀取 action_table.json,先印出它的頂層鍵與各鍵的型別和長度(讓我知道格式),再檢查其中的落點總數是否也等於 30,433。
    空盤四個玩家各自的開局合法步數:

    Copy
    from dataclasses import replace
    s0 = initial_state()
    for owner in range(4):
        s = replace(s0, to_move=owner)
        n = legal_move_mask(s, owner).bit_count()

    回報四個數字。預期:因為棋盤四角對稱,
    四個數字應該相等。若不等,這是旋轉對稱的反例,必須如實回報。
    四個數字與「該玩家角格被蓋到的落點」是否一致:對每個 owner,用暴力法(遍歷 ORIENTS 所有方向與 valid 內的所有 base,檢查 od["m"] << base 是否包含該角格位元)獨立算一次,與第 4 項比較。
    回報 config.py 內這四個常數的值:B、N、CLOCKWISE_OWNERS、OWNER_CORNER。

5. F3:_advance 延遲鎖存問題的實測

背景:engine._advance 在換手後按 owner id 0 到 3 的順序掃描,遇到第一個仍能走的玩家就 break,所以 to_move 指向的玩家可能已經無合法步,但 stuck 旗標尚未設為 True。

**資料:**種子 0 到 199 的 200 局。在 hook 裡對每個決策前的局面檢查:

    情況 X: legal_move_mask(state) == 0 且 state.stuck[state.to_move] == False。回報:
        出現次數,占全部決策點(含 pass 決策點)的比例;
        有多少局至少出現一次;
        隨機挑一個例子,印出 serialize(state) 與它是第幾手。
    情況 Y(不應發生): state.stuck[o] == True 但 has_any_legal(state, o) 為 True,對任一 o。若出現任何一次,立刻回報並附局面。
    情況 Z: 連續 pass 的最長連續次數(跨所有局)。
    驗證所有 200 局都以 is_over(state) == True 結束,且最後所有玩家 has_any_legal 皆為 False。
    額外比對:200 局的每一局,終局 result(state) 與「對同一串步驟用舊的 Game 重播」的餘格是否一致。若重播接口複雜,可改為回報「現有 tests/test_engine_cross.py 是否已涵蓋此項」並引用其行號,不要重寫。

6. F4:choose_move 與人格 AI 的查證

只讀 ai/ 下程式碼,不修改。

A. trace 欄位清單。列出 choose_move 的 trace 參數在被填入時的全部欄位名,每欄給一行說明:型別、內容、是否包含「所有候選」或只含短名單。再回答:

    短名單長度 K 是固定值還是依局面變化?各人格的 K 值是多少?
    trace 中是否有每個候選步的分數?是全部合法步的分數,還是只有短名單的?
    picked(最終被選的步)是什麼格式(Board 的表示法還是 engine 的 (piece_idx, oi, base))?若不同,寫出把它轉成 engine 格式的對應規則,並用程式驗證:200 局 optimizer 對局中,轉換後的步在 legal_move_mask 內的比例必須是 100%。

**B. 隨機性測量。**對四個人格(optimizer、wolf、chess、fox)各做:

    取種子 0 到 99 局的隨機對局中共 500 個中盤局面(固定抽樣,種子 12345)。
    對每個局面,用兩個不同 RNG 種子各呼叫一次 choose_move(每次要從相同的局面複本開始,因為呼叫會消耗 RNG)。
    回報「兩次選擇不同」的比例。

這個比例就是該人格 top-1 模仿的上限的反面:若為 0% 則是確定性;若為 30% 則 top-1 命中率天花板約 70% 以下。

**C. 終局評分規則。**找出 Game 中計算名次與餘格的函數(Game.standings 與 average_ranks 及其呼叫者),回答:

    評分是否僅用剩餘格數?有沒有「全部用完加分」或「最後一塊為單格加分」的邏輯?請貼出相關程式碼行號與片段。
    並列的處理規則是什麼?

7. F5:硬體與環境(唯讀檢查)

回報:

    CPU 型號、邏輯核心數、實體核心數(lscpu 或 /proc/cpuinfo);
    記憶體總量;
    python --version;
    numpy 版本;
    torch 是否可以 import?若可以:版本、torch.cuda.is_available()、GPU 名稱與顯存;若不可以:只回報「未安裝」,不要安裝;
    若有 nvidia-smi,貼出其輸出的前幾行(驅動版本、CUDA 版本、GPU 名稱、顯存)。

8. 測試(tests/test_bench_engine.py)

只需以下幾項,要快(整檔 < 30 秒):

    random_playout(seed=1) 連跑兩次,回傳的終局 serialize 與手數完全相同(可重現)。
    random_playout 的終局滿足 is_over。
    random_playout 不修改傳入的任何 State(State 是 frozen,此項驗證 hook 收集到的狀態在對局後仍與收集當下的 serialize 相同)。
    --no-timing 模式下,同一命令連續跑兩次,輸出完全相同(逐位元組)。
    合法步數統計函數對已知小例子(空盤 owner 0 的開局)給出與 legal_move_mask(...).bit_count() 相同的數字。

既有的 316 項測試必須全部仍然通過,回報最終測試總數。
9. 報告格式(reports/f_prime_report.md)

對話中最後要貼回給我的內容就是這份報告的全文,結構如下:

Copy
# 階段 F′ 報告
## 環境(第 7 節)
## 結果區(可逐位重現)
  ### 合法步數分佈(第 3 節)
  ### 動作表一致性(第 4 節)
  ### _advance 延遲鎖存(第 5 節)
  ### trace 欄位與人格隨機性(第 6 節)
  ### 終局評分規則(第 6 節 C)
## 量測區(量測非結果)
  ### 整局耗時(第 2 節 A)
  ### 單函數耗時表(第 2 節 B,開局/中盤/殘局/全部四欄)
## 異常與未預期發現
## 測試結果(總數、是否全過)
## 提交資訊(commit hash、變更檔案清單)

所有表格請給實際數字,不要用「約」或「大致」。若某項無法完成,寫明原因,不要跳過或編造數字。
10. 驗收標準(自我檢查清單)

    僅新增了第 0 節允許的四個檔案,沒有改任何既有檔案(請用 git status 與 git diff --stat 證明)。
    報告中每個章節都有實際數字。
    --no-timing 兩次輸出逐位元組相同。
    316 項既有測試全部通過。
    異常項(例如四個開局合法步數不相等、情況 Y 出現、落點總數不是 30,433)已明確列出,未被修改或隱藏。
    已 commit。

