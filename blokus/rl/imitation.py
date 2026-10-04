"""An H-B2 checkpoint as a seat that plays: read-only inference.

This is the counterpart of `rl/train.py`. Training wrote 25 checkpoints; this
turns one of them into something `ai.choose_move` can drive, which is why it is
a `Brain` and not a parallel code path: the shared pipeline enumerates the
legal candidates, and `restrict` throws all of them away except the one the
network named. Everything after that - shortlist, mistake draw, placement - runs
unchanged, so an imitation seat and a personality seat are the same kind of
thing to the rest of the repository.

Three things make this the *same* function that was trained:

  * 27 channels, 14 planes then the 13 scalars broadcast, produced by the very
    `rl.caches.features_27` the trainer called.
  * The illegal slots are filled with -1e9, the same `MASK_FILL` the loss used,
    applied to the same `rl.actions.legal_mask_view` mask.
  * The output is read as 91 orientation planes of 20x20, which is the action
    index laid out as `(orientation, base)`.

Read-only by construction. The optimiser state in the checkpoint is ignored, the
module is put in `eval()` and every forward runs under `torch.no_grad()`, and
nothing about the position is carried between calls - which is what lets one
checkpoint drive two or four colours in the same game. The only mutable field is
the sampling stream, and each seat gets its own instance, so two seats on the
same step do not draw from each other.

Nothing here imports torch at module scope, so a game with no imitation seat
never loads it.
"""
import os
import random

import engine
from ai.base import Profile
from ai.formulas import B
from config import PROJECT_DIR
from rl.actions import (index_to_move, legal_mask_view, move_to_index,
                        seat_of_mover, view_to_real)
from rl.caches import features_27

# The value the loss filled illegal slots with. Same constant as `rl.train`, and
# duplicated rather than imported so that a module which must not pull torch at
# import time does not import the trainer to get a float.
MASK_FILL = -1e9

# The policy head emits 91 orientation planes; 21 pieces have that many
# orientations between them. See `rl.actions.ORIENT_OFFSET`.
N_ORIENT = 91

DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "data", "hb2")
DEFAULT_MODE = "argmax"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 20_260_903

_BLOB_CACHE = {}


def checkpoint_path(step, checkpoint_dir=None):
    """Where the checkpoint for `step` is expected."""
    d = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return os.path.join(d, "step_%06d.pt" % int(step))


def _load_blob(path):
    """A whole checkpoint, cached per path.

    The optimiser state is dropped on the way out: inference needs weights and
    nothing else, and the run that produced them is over. Caching means two
    seats on the same step read the file once while still getting independent
    modules, so neither can observe the other's sampling stream.
    """
    import torch
    key = os.path.abspath(path)
    if key not in _BLOB_CACHE:
        blob = torch.load(key, map_location="cpu", weights_only=False)
        blob.pop("optimizer", None)
        _BLOB_CACHE[key] = blob
    return _BLOB_CACHE[key]


def hand_bits_from_names(names):
    """Piece names -> the engine's 21-bit hand for one seat."""
    v = 0
    for n in names:
        v |= 1 << engine.PIECE_ORDER.index(n)
    return v


def state_from_game(game):
    """A live `Game` as an `engine.State`.

    The bridge both inference and the move log need. `turn_order` is carried
    through but nothing in the engine's rules reads it: the opening corner comes
    from `OWNER_CORNER[owner]` and the corner-contact rule from the owner's own
    stones, so a position is fully described by the four owner masks, the four
    hands, who is to move, and who is stuck.
    """
    hands = tuple(hand_bits_from_names(game.hands[o].names) for o in range(4))
    return engine.State(
        own_bits=tuple(int(b) for b in game.board.owner_bits),
        hand_bits=hands,
        turn_order=tuple(game.turn_order),
        to_move=game.current_owner(),
        stuck=tuple(bool(x) for x in game.stuck))


def stuck_from_state(state):
    """The four stuck latches, recomputed.

    A seat is stuck exactly when it has no legal placement. Deriving it here
    rather than reading `Game.stuck` keeps this function usable from a bare
    `State`, which is what the log round-trip test has.

    Until plan7-A this was a strictly stronger statement than the latched one:
    `Game._all_stuck` stopped scanning at the first seat that could still move,
    so seats after it could be stuck while `Game.stuck` still said False. The
    latch no longer short-circuits, so the two now agree on every seat of every
    position, and this function is an independent check of that rather than a
    second, different definition.
    """
    return tuple(engine.legal_move_mask(state, o) == 0 for o in range(4))


def logits_from_state(net, state, device=None):
    """The policy logits for one position: `(36400,)`, illegal slots at -1e9."""
    import torch
    x = features_27([state])
    t = torch.from_numpy(x)
    if device is not None:
        t = t.to(device)
    with torch.no_grad():
        policy, _value = net(t)
    flat = policy.reshape(-1).float()
    mask = legal_mask_view(state)
    if not mask.any():
        return flat
    return flat.masked_fill(torch.from_numpy(~mask), MASK_FILL)


def pick_action(logits, mode=DEFAULT_MODE, rng=None,
                temperature=DEFAULT_TEMPERATURE):
    """One action index out of masked logits.

    `argmax` is deterministic and is the default, so a checkpoint plays the same
    game every time. `softmax` samples from the masked distribution, which is how
    a single checkpoint gets more than one style out of itself.
    """
    import torch
    if mode == "argmax":
        return int(torch.argmax(logits).item())
    if mode != "softmax":
        raise ValueError("unknown selection mode %r; use argmax or softmax"
                         % (mode,))
    if rng is None:
        rng = random.Random()
    t = max(1e-6, float(temperature))
    probs = torch.softmax(logits / t, dim=-1)
    total = float(probs.sum())
    if not (total > 0.0):
        return int(torch.argmax(logits).item())
    target = rng.random() * total
    acc = 0.0
    nz = torch.nonzero(probs, as_tuple=False).reshape(-1)
    for i in nz.tolist():
        acc += float(probs[i])
        if acc >= target:
            return int(i)
    return int(nz[-1].item())


def action_to_move(action_index, state):
    """A view-frame action index -> `(piece_name, oi, base)` in real coordinates.

    The network's head is in the mover's frame, so the index has to come back
    across `LUT_ROT` before it names a square on the real board.
    """
    piece_idx, oi, base = index_to_move(int(action_index))
    real = int(view_to_real(move_to_index((piece_idx, oi, base)),
                            seat_of_mover(state)))
    _pi, oi, base = index_to_move(real)
    return engine.PIECE_ORDER[piece_idx], int(oi), int(base)


class ImitationBrain:
    """A checkpoint, shaped like a `Brain` so `choose_move` can drive it.

    `context` captures the position once; `restrict` then returns the single
    candidate the network chose. Returning one candidate is what makes the rest
    of the pipeline a no-op - the shortlist is that one move - and it means a
    move the network could not have chosen never reaches the board, because the
    legal mask excluded it and the enumeration agrees with the mask.
    """

    def __init__(self, step, checkpoint_dir=None, device=None,
                 mode=DEFAULT_MODE, temperature=DEFAULT_TEMPERATURE,
                 seed=DEFAULT_SEED, state_dict=None, profile=None):
        import torch
        from rl.smoke_gpu import build_model

        self.key = "step_%d" % int(step)
        self.step = int(step)
        self.mode = mode
        self.temperature = temperature
        self.rng = random.Random(seed)
        # `choose_move` reads `.profile` to score candidates before `restrict`
        # runs, and `restrict` discards those scores. Chess's profile is the
        # neutral one the rule-based personalities already borrow for exactly
        # this reason; nothing downstream depends on its numbers.
        self.profile = profile or _chess_profile()
        self.mistake_rate = 0.0
        # A network has no opponent model, so the lookahead stage - which is a
        # weighted-score quantity an order of magnitude away from what it was
        # trained on - is switched off rather than fed nonsense.
        self.uses_lookahead = False
        self.device = torch.device(device) if device is not None else None

        path = checkpoint_path(step, checkpoint_dir)
        blob = _load_blob(path)
        sd = state_dict if state_dict is not None else blob["state_dict"]
        channels, blocks, in_ch = _shape_settings(
            {"state_dict": sd, "settings": blob.get("settings")})
        self.net = build_model(channels=channels, blocks=blocks, in_ch=in_ch)
        self.net.load_state_dict(sd)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if self.device is not None:
            self.net.to(self.device)

    # -- the Brain interface -------------------------------------------------

    def context(self, board, names, owner, must_cover=None, reach=None):
        """Read the position once and decide the move.

        The board alone is not the position: the features also want the four
        stuck latches and the turn order, which live on the `Game`. So this
        reads the bound game and checks it is the game being asked about - a
        stale binding would silently featurise a different position than the
        one whose candidates are on the table.
        """
        g = getattr(self, "game", None)
        if g is None:
            raise ValueError("checkpoint %s has no game bound; call "
                             "attach(game) before choosing" % self.key)
        if g.current_owner() != owner or g.board is not board:
            raise ValueError("checkpoint %s is bound to a game where %r moves, "
                             "but was asked about owner %d"
                             % (self.key, g.current_owner(), owner))
        state = state_from_game(g)
        if engine.legal_move_mask(state, owner) == 0:
            return {"state": state, "move": None}
        logits = logits_from_state(self.net, state, self.device)
        action = pick_action(logits, self.mode, self.rng, self.temperature)
        return {"state": state, "move": action_to_move(action, state)}

    def restrict(self, cands, ctx):
        move = ctx["move"]
        if move is None:
            return []
        name, oi, base = move
        for c in cands:
            if c[1] == name and c[2] == oi and c[3] == base:
                return [c]
        # The network named a move the enumeration did not offer. Returning
        # nothing would be silently "no legal move", so say so instead.
        raise ValueError("checkpoint %s chose %r, which the candidate "
                         "enumeration did not offer" % (self.key, move))

    def rescore(self, cands, ctx):
        return cands

    # -- inference -----------------------------------------------------------

    def attach(self, game):
        """Bind the live game so `context` can read the whole position.

        Held per brain rather than passed through, so that two seats on the same
        checkpoint in one game each see their own board.
        """
        self.game = game
        return self

    def choose(self, game=None):
        """The move this checkpoint wants to play in `game` right now.

        `(name, oi, x, y)`, or `None` when nothing is legal. Does not place it -
        `choose_move` does that - so this is also usable directly by a test.
        """
        g = game if game is not None else getattr(self, "game", None)
        if g is None:
            raise ValueError("no game to choose in; call attach(game) first")
        owner = g.current_owner()
        state = state_from_game(g)
        if engine.legal_move_mask(state, owner) == 0:
            return None
        logits = logits_from_state(self.net, state, self.device)
        action = pick_action(logits, self.mode, self.rng, self.temperature)
        name, oi, base = action_to_move(action, state)
        return (name, oi, base % B, base // B)


def _chess_profile():
    """A neutral, fixed profile for the candidate scoring `choose_move` does
    before `restrict` runs.

    Chess's, because that is what the rule-based personalities already borrow
    for exactly this reason. `restrict` discards those scores, so the numbers
    are never used; the field exists because the pipeline reads it.
    """
    from ai.registry import make_profile
    return make_profile("chess", random.Random(0))


def _shape_settings(blob):
    """`(channels, blocks, in_ch)` for a checkpoint.

    Taken from the settings the run froze into the file, so inference cannot
    disagree with training about the architecture. Falls back to counting the
    residual blocks in the keys, which is enough to recognise a file written
    without them.
    """
    sd = blob["state_dict"]
    s = blob.get("settings") or {}
    for k, v in sd.items():
        if k.endswith("trunk.0.weight"):
            channels, in_ch = int(v.shape[0]), int(v.shape[1])
            break
    else:
        raise ValueError("no trunk input convolution in the checkpoint")
    blocks = s.get("blocks")
    if not blocks:
        idx = set()
        for k in sd:
            parts = k.split(".")
            if len(parts) > 2 and parts[-1] in ("weight", "bias") \
                    and parts[-2].isdigit() and parts[-3].isdigit():
                idx.add(int(parts[-3]))
        blocks = len(idx)
        if not blocks:
            raise ValueError("cannot tell how many residual blocks this "
                             "checkpoint has")
    if s and (s.get("in_ch") not in (None, in_ch)
              or s.get("channels") not in (None, channels)):
        raise ValueError("the checkpoint's own settings disagree with its "
                         "shapes: %r vs channels=%d in_ch=%d"
                         % (s, channels, in_ch))
    return channels, blocks, in_ch


def load_brain(step, checkpoint_dir=None, device=None, mode=DEFAULT_MODE,
               temperature=DEFAULT_TEMPERATURE, seed=DEFAULT_SEED):
    """One seat's imitation player for the checkpoint saved at `step`."""
    return ImitationBrain(step, checkpoint_dir=checkpoint_dir, device=device,
                          mode=mode, temperature=temperature, seed=seed)