"""Game state machine: SETUP -> PLAYING -> GAME_OVER."""
from board import Board
from config import CLOCKWISE_OWNERS, OWNER_CORNER, PIECES, PLAYER_OWNER
from ai import draw_personalities, make_brain
from pieces import Hand, MASTER

HAND_NAMES = [p["name"] for p in PIECES]
COLOR_NAMES = ["blue", "green", "red", "yellow"]
# 一局有四個席位，所以一場全 AI 比賽剛好抽四個人格出來互比。
MATCH_SEATS = 4


class Game:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.board = Board()
        self.state = "SETUP_COLOR"
        self.turn_count = 0
        self.turn_pos = 0
        self.turn_order = [0, 1, 2, 3]
        self.brains = {}
        self.owner_key = {0: None, 1: None, 2: None, 3: None}
        self.colors = {0: None, 1: None, 2: None, 3: None}
        self.hands = [Hand(HAND_NAMES) for _ in range(4)]
        self.placed = [0, 0, 0, 0]

    def setup_match(self, keys=None):
        """一局全 AI 的比賽：四個席位都給人格，含 1 號那個「玩家」席位。

        每局抽 4 個人格，所以每局會有幾個人格輪空——這就是「所有人格互比」要的
        輪廓。顏色只是為了讓 `Game` 的欄位保持一致，headless 比賽不看。
        """
        if keys is None:
            keys = draw_personalities(self.rng, MATCH_SEATS)
        if len(set(keys)) != MATCH_SEATS or len(keys) != MATCH_SEATS:
            raise ValueError("a match needs %d distinct personalities, got %r"
                             % (MATCH_SEATS, list(keys)))
        self.owner_key = {o: keys[o] for o in range(4)}
        self.colors = {o: COLOR_NAMES[o] for o in range(4)}
        self.brains = {o: make_brain(keys[o], self.rng) for o in range(4)}
        self.turn_order = self._draw_turn_order()
        self.turn_pos = 0
        self.state = "SETUP_INFO"

    def set_player_color(self, color):
        self.colors[0] = color
        rest = [c for c in COLOR_NAMES if c != color]
        self.rng.shuffle(rest)
        self.colors[1], self.colors[2], self.colors[3] = rest[0], rest[1], rest[2]
        keys = draw_personalities(self.rng, 3)
        self.owner_key[1], self.owner_key[2], self.owner_key[3] = keys[0], keys[1], keys[2]
        self.brains = {
            0: make_brain("chess", self.rng),
            1: make_brain(keys[0], self.rng),
            2: make_brain(keys[1], self.rng),
            3: make_brain(keys[2], self.rng),
        }
        self.turn_order = self._draw_turn_order()
        self.turn_pos = 0
        self.state = "SETUP_INFO"

    def _draw_turn_order(self):
        """順時針的輪轉，**起點**隨機：玩家這一局可能第一，也可能第四。

        順時針的循環只有 CLOCKWISE_OWNERS 一個；隨機的只有從哪一個角開始，
        也就是「誰搶到下棋權」。這裡不能用 shuffle——那會讓相鄰兩手有時是對角
        的兩個角，違反順時針。
        """
        start = (CLOCKWISE_OWNERS.index(PLAYER_OWNER)
                 + self.rng.randrange(4)) % 4
        return [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]

    def redraw_personalities(self):
        """重抽三位 AI 的人格；顏色與下棋順序不動。

        顏色是玩家剛選完的，重抽掉會讓那個選擇看起來白按。下棋順序也保留，
        這樣玩家可以在同一個順序下重抽想要的對手組合。玩家的 seat 也不換——
        它本來就固定是 chess。
        """
        keys = draw_personalities(self.rng, 3)
        self.owner_key[1], self.owner_key[2], self.owner_key[3] = keys[0], keys[1], keys[2]
        for o, k in zip((1, 2, 3), keys):
            self.brains[o] = make_brain(k, self.rng)

    def start(self):
        self.board = Board()
        self.hands = [Hand(HAND_NAMES) for _ in range(4)]
        self.placed = [0, 0, 0, 0]
        self.turn_count = 0
        self.turn_pos = 0
        self.state = "PLAYING"

    def current_owner(self):
        return self.turn_order[self.turn_pos % 4]

    def must_cover(self, owner):
        """A player's opening piece has to sit on their own corner square."""
        return OWNER_CORNER[owner] if self.placed[owner] == 0 else None

    def reach(self, owner):
        """Corner-contact rule: after the opening piece every stone must touch
        one of this owner's own stones at a corner and must not share an edge
        with one. None on the opening move, where `must_cover` takes over - the
        two never apply at the same time."""
        return None if self.placed[owner] == 0 else self.board.reach(owner)

    def has_legal(self, owner):
        cells_iter = [MASTER[n]["orientations"] for n in self.hands[owner].names]
        return self.board.has_legal_move(owner, cells_iter, self.must_cover(owner),
                                         self.reach(owner))

    def can_act(self, name, oi, x, y, owner=None):
        owner = self.current_owner() if owner is None else owner
        cells = MASTER[name]["orientations"][oi]
        return self.board.can_place(x, y, cells, owner, self.must_cover(owner),
                                    self.reach(owner))

    def act(self, name, oi, x, y):
        owner = self.current_owner()
        if not self.can_act(name, oi, x, y, owner):
            raise ValueError("illegal move %s oi=%d at (%d,%d) for owner %d"
                             % (name, oi, x, y, owner))
        self.hands[owner].names.remove(name)
        self.board.place(x, y, MASTER[name]["orientations"][oi], owner)
        self.placed[owner] += 1
        self._advance()

    def act_pass(self):
        self._advance()

    def auto_pass(self, owner):
        if owner is not None and not self.has_legal(owner):
            self.act_pass()

    def _advance(self):
        self.turn_count += 1
        self.turn_pos = (self.turn_pos + 1) % 4
        if self._all_stuck():
            self.state = "GAME_OVER"

    def _all_stuck(self):
        """The game ends only when *every* player, the human included, has no
        legal move left."""
        for o in range(4):
            if self.has_legal(o):
                return False
        return True

    def remaining_cells(self, owner):
        return sum(MASTER[n]["size"] for n in self.hands[owner].names)

    def standings(self):
        """(owner, remaining) for all four, best first.

        Remaining squares *are* the score: the fewer a player leaves on the
        board, the better they did. Ties are ordered by owner so the result is
        deterministic.
        """
        rows = [(o, self.remaining_cells(o)) for o in range(4)]
        rows.sort(key=lambda t: (t[1], t[0]))
        return rows

    def winner(self):
        best = self.standings()[0][1]
        top = [o for o, rem in self.standings() if rem == best]
        if 0 in top:
            return "player" if len(top) == 1 else "draw"
        return str(top[0])
