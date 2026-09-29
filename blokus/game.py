"""Game state machine: SETUP -> PLAYING -> GAME_OVER."""
from board import Board
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER, PIECES, PLAYER_OWNER
from ai import draw_personalities, make_brain
from ai.formulas import ODIRS, _legal_bases
from pieces import Hand, MASTER

HAND_NAMES = [p["name"] for p in PIECES]
COLOR_NAMES = ["blue", "green", "red", "yellow"]
# A game has four seats, so one all-AI match draws exactly four personalities.
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
        self.stuck = [False] * 4

    def setup_match(self, keys=None):
        """A game played entirely by AIs: all four seats get a personality,
        including seat 1, which is normally the human.

        Four personalities are drawn per game, so several sit out any given game
        - that is the shape "everyone competes" needs. Colours are only there to
        keep the `Game` fields consistent; a headless match ignores them.
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
        """Clockwise rotation with a random **starting point**: the player may go
        first this game, or fourth.

        There is only one clockwise cycle (CLOCKWISE_OWNERS); only the corner it
        begins from is random, which is to say who wins the right to move first.
        This cannot be a shuffle - that would sometimes make two neighbouring
        turns be opposite corners, which breaks the clockwise rule.
        """
        start = (CLOCKWISE_OWNERS.index(PLAYER_OWNER)
                 + self.rng.randrange(4)) % 4
        return [CLOCKWISE_OWNERS[(start + i) % 4] for i in range(4)]

    def redraw_personalities(self):
        """Redraw the three AI personalities; colours and turn order stay.

        The player has just picked a colour, so re-rolling it would make that
        choice look wasted. The turn order is kept too, so the player can reroll
        opponents under the same order. The player's seat does not change either
        - it is fixed to chess anyway.
        """
        keys = draw_personalities(self.rng, 3)
        self.owner_key[1], self.owner_key[2], self.owner_key[3] = keys[0], keys[1], keys[2]
        for o, k in zip((1, 2, 3), keys):
            self.brains[o] = make_brain(k, self.rng)

    def start(self):
        self.board = Board()
        self.hands = [Hand(HAND_NAMES) for _ in range(4)]
        self.placed = [0, 0, 0, 0]
        self.stuck = [False] * 4
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
        """Is there any legal placement for `owner` right now?

        Bitmask version: `_legal_bases` answers "which placements are legal"
        for a piece orientation with a handful of big-integer operations, so
        there is no per-cell scan. `Board.has_legal_move` is kept as the
        obviously-correct reference that this is checked against.

        Always recomputed, even for a player already known to be stuck: the UI
        asks this to decide whether to show a pass hint, and the tests use it
        as the reference the cached `stuck` flag is checked against.
        """
        empt = self.board.empty_bits
        reach = self.reach(owner)
        must_cover = self.must_cover(owner)
        cbit = 1 << (must_cover[0] + must_cover[1] * B) if must_cover else 0
        for name in self.hands[owner].names:
            for od in ODIRS[name].values():
                if _legal_bases(od, empt, reach, cbit):
                    return True
        return False

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
        legal move left.

        A player who has no legal move never will: their contact rule depends
        only on their own stones, and empty squares never come back. So each
        seat is tested once and the answer is remembered in `self.stuck`,
        which turns the per-move sweep over all four seats into a sweep over
        the ones still in play. `tests/test_stuck_monotone.py` is the evidence
        that the permanence assumption actually holds.
        """
        for o in range(4):
            if self.stuck[o]:
                continue
            if self.has_legal(o):
                return False
            self.stuck[o] = True
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
