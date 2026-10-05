"""Game state machine: SETUP -> PLAYING -> GAME_OVER."""
from board import Board
from config import B, CLOCKWISE_OWNERS, OWNER_CORNER, PIECES, PLAYER_OWNER
from ai import draw_personalities, make_brain
from ai.formulas import ODIRS, _legal_bases
from pieces import Hand, MASTER

import seats as seats_mod

HAND_NAMES = [p["name"] for p in PIECES]
COLOR_NAMES = ["blue", "green", "red", "yellow"]
# A game has four seats, so one all-AI match draws exactly four personalities.
MATCH_SEATS = 4


class Game:
    def __init__(self, rng):
        self.rng = rng
        # Set by `setup_seats`. `None` means "not configured", which is the
        # state `setup_match` and `set_player_color` leave it in, so a caller
        # that only ever used those sees no change.
        self.seat_kinds = None
        # A `gamelog.GameLog` when at least one seat is "human (record)", else
        # `None`. `act` consults it, so a game with no recording seat writes
        # nothing at all - not an empty file, not a header.
        self.log = None
        self.reset()

    def reset(self):
        self.board = Board()
        self.state = "SETUP_SEATS"
        self.turn_count = 0
        self.turn_pos = 0
        self.turn_order = [0, 1, 2, 3]
        self.brains = {}
        self.owner_key = {0: None, 1: None, 2: None, 3: None}
        self.colors = {0: None, 1: None, 2: None, 3: None}
        self.hands = [Hand(HAND_NAMES) for _ in range(4)]
        self.placed = [0, 0, 0, 0]
        self.stuck = [False] * 4

    # -- who is playing ------------------------------------------------------

    def setup_seats(self, option_keys, colours=None, rng=None,
                    checkpoint_dir=None, device=None,
                    mode="argmax", include_humans=True):
        """Configure all four seats by hand: one option and one colour each.

        `option_keys` is four option keys from `seats.seat_options`, in seat
        order, and may repeat freely. A seat left on `seats.RANDOM_AI_KEY` is
        dealt one of the eleven automated options here, one draw per seat, so
        by the time this returns every seat names a specific contestant.
        `colours` is four entries, each a colour name or `None`/`"random"` to be
        dealt one of whatever is left; specified colours must be distinct.

        Turn order is drawn here rather than passed in, and is independent of
        both: any owner may open the game.
        """
        option_keys = list(option_keys)
        if len(option_keys) != MATCH_SEATS:
            raise ValueError("a game has %d seats, got %d options"
                             % (MATCH_SEATS, len(option_keys)))
        rng = self.rng if rng is None else rng
        # Resolve the deferred choice before anything reads a seat, so the rest
        # of this method only ever sees concrete options.
        option_keys = seats_mod.resolve_random_ai(option_keys, rng)
        kinds = {o: seats_mod.kind_of(k, include_humans)
                 for o, k in enumerate(option_keys)}
        if colours is None:
            colours = [None] * MATCH_SEATS
        resolved = seats_mod.assign_colours(colours, rng)
        self.owner_key = {o: option_keys[o] for o in range(MATCH_SEATS)}
        self.colors = {o: resolved[o] for o in range(MATCH_SEATS)}
        self.seat_kinds = kinds
        self.brains = {
            o: seats_mod.build_brain(option_keys[o], rng, kinds[o],
                                     checkpoint_dir=checkpoint_dir,
                                     device=device, mode=mode)
            for o in range(MATCH_SEATS)
        }
        self.attach_brains()
        self.turn_order = self._draw_turn_order()
        self.turn_pos = 0
        self._open_log()
        self.state = "SETUP_INFO"
        return self

    def attach_brains(self):
        """Give every brain that wants the live game a reference to it.

        An imitation player needs the whole position - the stuck latches and
        the turn order live on the `Game`, not on the board it is handed - so it
        cannot featurise from `choose_move`'s arguments alone.
        """
        for brain in self.brains.values():
            attach = getattr(brain, "attach", None)
            if attach is not None:
                attach(self)
        return self

    def brain_map(self):
        """`{owner: brain}` for the seats that have one.

        Human seats have no brain, so this is not the same as `self.brains`.
        Anything that hands the whole dictionary to `ai.choose_move` as the
        opponent model wants this one.
        """
        return {o: b for o, b in self.brains.items() if b is not None}

    def seats_configured(self):
        """Has `setup_seats` run?

        `setup_match` and `set_player_color` leave `owner_key[0]` as `None`
        because seat 0 is the person there and is not named by a personality.
        A screen that asks "is this seat set up" has to accept that, so it asks
        this rather than testing one dictionary entry.
        """
        return self.seat_kinds is not None

    def is_human(self, owner):
        """Does a person play this seat?"""
        kinds = self.seat_kinds
        if kinds is None:
            return owner == PLAYER_OWNER
        return seats_mod.is_human_kind(kinds[owner])

    def humans(self):
        """The owners a person plays, in seat order."""
        return [o for o in range(MATCH_SEATS) if self.is_human(o)]

    def recording_seats(self):
        """The owners whose moves are written to the log."""
        kinds = self.seat_kinds or {}
        return [o for o in range(MATCH_SEATS)
                if kinds.get(o) == seats_mod.KIND_HUMAN_LOG]

    def _open_log(self):
        """Start a log if this game has a "human (record)" seat, else none."""
        seats = self.recording_seats()
        if not seats:
            self.log = None
            return None
        from gamelog import GameLog
        self.log = GameLog(self, seats)
        return self.log

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
        """Clockwise rotation with a random **starting point**: any seat may go
        first this game.

        There is only one clockwise cycle (CLOCKWISE_OWNERS); only the corner it
        begins from is random, which is to say who wins the right to move first.
        This cannot be a shuffle - that would sometimes make two neighbouring
        turns be opposite corners, which breaks the clockwise rule.

        The draw lives in `seats.draw_turn_order` so the UI, the league and the
        tests all get the same one.
        """
        return seats_mod.draw_turn_order(self.rng)

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
        if self.log is not None:
            self.log.record_move(owner, name, oi, x, y)
        self.hands[owner].names.remove(name)
        self.board.place(x, y, MASTER[name]["orientations"][oi], owner)
        self.placed[owner] += 1
        self._advance()

    def act_pass(self):
        if self.log is not None:
            self.log.record_pass(self.current_owner())
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

        The scan deliberately does *not* stop at the first seat that can still
        move. An earlier version returned there, which left every later seat
        untested: a seat that was already stuck kept reporting `False` until
        the sweep finally reached it. That is harmless for the end of the game
        - all four still latch on the same move - but `stuck` is fed straight
        to the network as four feature channels, and a channel that says "I can
        still move" while the seat cannot is a lie the model is trained on.
        The extra cost is one `has_legal` per unlatched seat: over 200 real
        games it went from 1.04 to 3.45 calls per turn, for no measurable wall
        clock (99.5s -> 98.7s, inside the run-to-run noise).
        """
        alive = False
        for o in range(4):
            if self.stuck[o]:
                continue
            if self.has_legal(o):
                alive = True
                continue
            self.stuck[o] = True
        return not alive

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

    def winner_owner(self):
        """The winning owner id, or `None` for a draw.

        `winner()` below answers the same question in the vocabulary the old
        one-seat player UI used, where owner 0 was always the human. With four
        hand-chosen seats that shortcut is wrong - there may be no human, or four
        - so this is the form the new screens and the league use.
        """
        best = self.standings()[0][1]
        top = [o for o, rem in self.standings() if rem == best]
        return None if len(top) != 1 else top[0]

    def winner(self):
        best = self.standings()[0][1]
        top = [o for o, rem in self.standings() if rem == best]
        if 0 in top:
            return "player" if len(top) == 1 else "draw"
        return str(top[0])
