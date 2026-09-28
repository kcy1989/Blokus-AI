"""Constants, corners, colors, i18n, piece data loading."""
import json
import os

B = 20
N = B * B
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PIECE_JSON = os.path.join(PROJECT_DIR, "piece_data.json")

CORNERS = ((0, 0), (B - 1, 0), (0, B - 1), (B - 1, B - 1))
CORNERS_IDX = [x + y * B for x, y in CORNERS]

OWNER_CORNER = {0: (B - 1, B - 1), 1: (0, 0), 2: (B - 1, 0), 3: (0, B - 1)}
CORNERS_BY_OWNER = {ci: (x, y) for (ci, (x, y)) in enumerate(CORNERS)}
OWNER_BY_CORNER = {pos: owner for owner, pos in OWNER_CORNER.items()}

# 輪轉必須順時針繞棋盤一圈：左上 → 右上 → 右下 → 左下。CORNERS 的順序是
# TL, TR, BL, BR，BR 排在 BL 後面才是順時針，所以這裡換一下。
CLOCKWISE_OWNERS = tuple(OWNER_BY_CORNER[c] for c in
                          (CORNERS[0], CORNERS[1], CORNERS[3], CORNERS[2]))
assert sorted(CLOCKWISE_OWNERS) == [0, 1, 2, 3]
PLAYER_OWNER = 0

COLORS = {
    "blue": (64, 120, 255),
    "green": (70, 200, 120),
    "red": (220, 70, 90),
    "yellow": (235, 195, 70),
}

I = {
    "title": "單人 BLOKUS",
    "choose_color": "請選擇你的顏色",
    "player": "玩家",
    "ai": "AI",
    "corner_tl": "左上",
    "corner_tr": "右上",
    "corner_bl": "左下",
    "corner_br": "右下",
    "rotate": "旋轉 90°",
    "confirm": "確認",
    "cancel": "取消",
    "redraw": "重抽 AI",
    "start_game": "開始",
    "new_game": "新遊戲",
    "game_over": "遊戲結束",
    "turn_note": ["每位玩家每輪各放 1 件。第一塊之後，每塊都必須以「角對角」碰到自己的棋，",
                  "且不可與自己的棋共邊；四位玩家都無合法放置時結束。"],
    "thinking": "正在思考…",
    "auto_pass": "無合法放置，自動過",
    "win_player": "玩家獲勝！",
    "win_ai": "{0} 獲勝",
    "draw": "平手",
    "left": "剩餘",
    "cells": "格",
    "left_fmt": "剩餘 {0} 塊 / {1} 格",
    "left_won_fmt": "排第 {0} 名，餘 {1} 格",
    "less_is_better": "餘格愈少愈佳",
    "this_game": "本局成績",
    "leaderboard": "排行榜",
    "avg_fmt": "平均 {0:.1f} 分 / 餘 {1:.1f} 格",
    "games_fmt": "{0} 局",
    "rank_fmt": "第 {0} 名",
    "points_fmt": "{0} 分",
    "new_game_hint": "Enter 或點擊「{0}」開始新的一局",

    "scale": "視窗縮放",
    "scale_hint": "棋盤 {0}%",
    "hand_zoom_hint": "手牌 {0}%",
    "keys_hint": ["[-／+] 棋盤　[／] 手牌", "F 符合螢幕　R 重開"],
    "hand_hint": ["左鍵點棋盤＝鎖定位置", "Enter／確認＝放置", "右鍵／X 旋轉，C 取消"],
    "locked": "已鎖定",
    "open_rule": "第一塊必須放在角位",
    "open_rule_hint": "第一塊必須覆蓋你的 {0} 角位",
    "open_rule_toast": "第一塊必須覆蓋你的 {0} 角位",
    "touch_rule_hint": "每塊須角對角接自己的棋，且不可共邊",
    "touch_rule_toast": "必須角對角接上自己的棋（不可共邊）",
    "group_fmt": "{0}格",
    "rotate_badge": "⟳{0}",
    "turn": "目前回合",
    "turn_order_fmt": "下棋順序（順時針）：{0}",
    "turn_first": "（先手）",
    "wolf": "狼",
    "wolf_desc": "激進 — 搶占四角、擺放大塊",
    "chess": "棋手",
    "chess_desc": "平衡 — 穩健佈局、權重均衡",
    "fox": "狐狸",
    "fox_desc": "防守 — 壓制接合、把控邊界",
    "intruder": "入侵者",
    "intruder_desc": "跨越 — 快速伸長、深入敵方領地",
    "optimizer": "優化者",
    "optimizer_desc": "效率 — 大塊先放、留住可放空間",
    "builder": "築城者",
    "builder_desc": "築城 — 圍住一整塊放得下大棋的地",
}

PERSONALITY_ORDER = ("wolf", "chess", "fox", "intruder", "optimizer", "builder")

PIECES = []


def load_pieces():
    global PIECES
    with open(PIECE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for name in sorted(data["pieces"]):
        p = data["pieces"][name]
        cells = tuple((int(x), int(y)) for x, y in p["cells"])
        out.append({"name": name, "size": p["size"], "cells": cells})
    PIECES = out
    return PIECES


def piece(name):
    for p in PIECES:
        if p["name"] == name:
            return p
    raise KeyError(name)


load_pieces()

HAND_CELLS_TOTAL = sum(p["size"] for p in PIECES)
