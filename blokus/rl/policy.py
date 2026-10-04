"""This layer answers one question: what numbers does an RL run need from a
network, and how are they put together so that everyone computes them the same
way?

Pure inference and pure numerics. No training loop, no game loop, no rollout -
this module is imported by things that already have both, and putting either
here would make it impossible to test the arithmetic without playing a game
first.

Five pieces:

  * `load_policy` reads a checkpoint strictly and hands back the architecture
    the file declares plus a fingerprint of the file it came from.
  * `ValueHead` is the critic. It is **new parameters**, not the checkpoint's
    existing `value` head - its docstring says why, at length.
  * `forward_batched` runs one trunk pass and returns both heads, with the
    policy logits already masked.
  * `log_prob_entropy` and `kl_to_anchor` are the quantities PPO needs, each
    computed in the one place that computes it.
  * `episode_reward` turns a finished game into one seat's number, ranking it
    through `records.rank_rows` rather than a second ranking rule.

Nothing here imports torch at module scope, so `rl` stays importable in an
interpreter with no GPU stack - the rule `rl.imitation` already follows and
`tests/test_rl_isolation.py` checks.
"""
import hashlib
import os

# The value the imitation loss filled illegal slots with, imported rather than
# written out. A second copy of this constant is a number that can drift away
# from the one the weights were trained under, and the drift would be silent:
# the policy still runs, just slightly wrong.
from rl.imitation import MASK_FILL, apply_legal_mask, mask_from_states

N_ACTIONS = 91 * 20 * 20

# How a slot is recognised as illegal. Half of MASK_FILL, which is the same
# test `tests/test_rl_imitation.py` uses, rather than `== MASK_FILL`: a caller
# that has already replaced the fill with `finfo.min` - which several reduction
# helpers do - must still be understood.
ILLEGAL_AT_OR_BELOW = MASK_FILL / 2.0


def file_md5(path):
    """The MD5 of a file, read in chunks.

    Recorded in the metadata because `hc_1000` is a *starting point*, not just
    a step number. The file is gitignored and regenerable, and two files called
    `step_001000.pt` from different runs are not interchangeable.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ValueHead(channels):
    """The critic: trunk output `(N, channels, 20, 20)` -> `(N,)`.

    A factory rather than a plain class because `torch.nn` is imported inside
    it, which is the same trick `rl.smoke_gpu.build_model` uses and for the
    same reason. `build_model` defines its `TwinHead` in the body; so does this.

    **Not** the checkpoint's own `value` head, which is `Conv2d(channels, 4, 1)`
    and produces `(N, 4, 20, 20)`. Three reasons, in order of weight:

    1. That head has never been trained. `rl/train.py` sets `value_aux: False`
       and its `masked_loss` never touches the value output, so the 260 weights
       are exactly what PyTorch's default initialisation left behind. Measured
       on hc_1000, its output runs -4.18 to +3.77 with a per-element standard
       deviation of 0.39, and its four channels' spatial means correlate at
       0.32 to 0.77 with one another - they are four random projections of a
       single representation, not four learned quantities. A PPO advantage is a
       difference of two such values, so it would begin life dominated by that
       random projection rather than by anything about the position.

    2. Reading four numbers off a 20x20 map means reducing the spatial axes
       anyway, and what survives the reduction is four random projections.
       This pools first and reads all 64 trunk channels, so the critic sees the
       whole representation instead of four lossy slices of it.

    3. 65 parameters against 260. A single `Linear` cannot grow a second
       representation, which is the right limit for 40 updates when the
       representation it would be growing is already 40,000 steps of imitation
       behind it.

    The final linear layer starts at **exactly** zero - weight and bias both.
    Two consequences, both wanted. PPO's first advantage batch is `reward - 0`
    rather than being offset by an arbitrary initial prediction, and with the
    weight at zero the critic's gradient does not reach the trunk:
    `dL/dpool = W^T dL/dout` is zero while `W` is. That is not an accident to
    be fixed later - it is the property that keeps the first value loss from
    pulling on the representation the policy head shares, which is the reason
    this head is separate from the checkpoint's in the first place.

    It is not stuck at zero. The weight's own gradient is `dL/dout *
    pool(trunk)`, so any loss whose gradient with respect to the output is
    non-zero moves it on the first optimiser step - measured at 4.58e0 for a
    plain `v.sum()` and 1.24e0 for an advantage-shaped `-v * adv`, with the
    trunk still at exactly 0. A squared error happens to leave it at zero,
    because its gradient is `2v` and `v` is zero; that is a property of that
    loss being minimised at the current output, not of the head.
    """
    from torch import nn

    class _ValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.lin = nn.Linear(channels, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

        def forward(self, trunk):
            return self.lin(self.pool(trunk).flatten(1)).squeeze(-1)

        def extra_repr(self):
            return "channels=%d, params=%d" % (
                channels, self.lin.weight.numel() + self.lin.bias.numel())

    return _ValueHead()


def build_checkpoint_model(channels, blocks, in_ch):
    """The architecture `rl.smoke_gpu.build_model` builds, by that function.

    Named separately so `load_policy` has one line that says where the shape
    comes from, and so a test can assert the two are the same object.
    """
    from rl.smoke_gpu import build_model
    return build_model(channels=channels, blocks=blocks, in_ch=in_ch)


def load_policy(path, device=None):
    """`(net, meta)` for the checkpoint at `path`, loaded strictly.

    `meta` carries the step, the architecture the file froze in, and the file's
    MD5. The shape comes from `_shape_settings`, which reads it out of the
    checkpoint rather than out of this module, so a run cannot build a
    differently-shaped network from the one the weights were written for.

    The critic is attached as `net.rl_value` after the strict load, so
    `forward_batched` reaches it without a second argument. Two consequences
    worth knowing:

      * `net.parameters()` then includes the critic, which is what a trainer
        wants - the critic is part of the model being optimised.
      * `net.state_dict()` therefore has `rl_value.*` keys that the
        checkpoint's own `state_dict` does not, so re-running
        `net.load_state_dict(blob["state_dict"])` on an already-attached net
        would fail on the extra keys. Load first, attach second; that is the
        order here.
    """
    import torch
    from rl.imitation import _shape_settings

    blob = torch.load(path, map_location="cpu", weights_only=False)
    channels, blocks, in_ch = _shape_settings(blob)
    net = build_checkpoint_model(channels=channels, blocks=blocks, in_ch=in_ch)
    net.load_state_dict(blob["state_dict"])        # strict: raises on any gap
    net.eval()
    if device is not None:
        net.to(device)
    head = ValueHead(channels)
    if device is not None:
        head.to(device)
    net.rl_value = head
    meta = {
        "path": os.path.abspath(path),
        "md5": file_md5(path),
        "step": blob.get("step"),
        "settings": dict(blob.get("settings") or {}),
        "channels": channels,
        "blocks": blocks,
        "in_ch": in_ch,
        "value_params": sum(p.numel() for p in head.parameters()),
        "device": None if device is None else str(device),
    }
    return net, meta


def forward_batched(net, states, device=None):
    """One trunk pass; `(masked_logits (N, 36400), value (N,))`.

    `states` is a list of `engine.State`. The mask comes from
    `rl.imitation.mask_from_states` and is applied through
    `rl.imitation.apply_legal_mask` - the same rotation, the same `MASK_FILL`
    and the same function `logits_from_state` uses, so there is no second
    masking rule for the two paths to disagree about.

    Gradients are deliberately left enabled. An inference path wants
    `no_grad` and a training path does not; a numerical interface that forced
    either would be wrong for one of them, and the caller is the only one that
    knows which it is.

    The checkpoint's dead `net.value` head is never called. The trunk output is
    taken once and fed to `net.policy` and `net.rl_value` directly, so the
    untrained parameters in the file stay untouched.
    """
    import torch
    from rl.caches import features_27

    states = list(states)
    if not states:
        raise ValueError("forward_batched needs at least one state")
    mask = mask_from_states(states)
    x = features_27(states)
    t = torch.from_numpy(x)
    if device is not None:
        t = t.to(device)
    trunk = net.trunk(t)
    logits = net.policy(trunk).reshape(len(states), N_ACTIONS).float()
    value = net.rl_value(trunk).float()
    return apply_legal_mask(logits, mask), value


def _live_logits(masked_logits, what):
    """The illegal slots lifted to `finfo.min`, and a per-row liveness flag.

    `MASK_FILL` is finite, so a row in which *every* slot is illegal is
    indistinguishable from a normal row by arithmetic alone - and it would
    quietly come back as a uniform 36,400-way distribution. That is not a
    distribution any position has: `BlokusEnv` guarantees `legal_indices()` is
    non-empty whenever `done` is False, so a dead row means the caller lost
    track of whose turn it is. Refusing it is the useful answer.

    Lifting to `finfo.min` rather than leaving -1e9 is what stops the illegal
    slots from contributing to `logsumexp` at all.
    """
    import torch
    illegal = masked_logits <= ILLEGAL_AT_OR_BELOW
    dead = illegal.all(dim=-1)
    if bool(dead.any()):
        rows = torch.nonzero(dead).reshape(-1).tolist()
        raise ValueError("%s: row(s) %s have no legal slot, so there is no "
                         "distribution over the action space"
                         % (what, rows[:8]))
    live = masked_logits.masked_fill(illegal, torch.finfo(masked_logits.dtype).min)
    return live


def log_prob_entropy(masked_logits, action_index):
    """`(log_prob (N,), entropy (N,))` under the masked distribution.

    Two properties this is careful about, both of which would otherwise show up
    as NaN rather than as a wrong number:

    * an illegal slot's probability is **exactly** zero, not merely small.
      `log_softmax` subtracts the row maximum before exponentiating and the
      illegal logits are lifted to `finfo.min`, so they underflow to 0.0
      rather than to a denormal that would then be multiplied by a large
      negative number.
    * entropy never evaluates `0 * log 0`. `log_probs` is finite on every
      legal slot, so each term is either a real negative or `0 * finite`,
      which is zero. `nansum` is deliberately not used: it would hide exactly
      the defect these two properties exist to rule out.
    """
    import torch

    if masked_logits.dim() != 2:
        raise ValueError("expected (N, %d) logits, got %s"
                         % (N_ACTIONS, tuple(masked_logits.shape)))
    # Carried onto the logits' device here rather than left to the caller. A
    # CUDA run has its action indices built on the host from a mask, and
    # `gather` refuses a CPU index against a CUDA tensor - the same class of
    # mistake as the mask that `rl.imitation.logits_from_state` used to make,
    # and found the same way: only by running on a device that is not CPU.
    idx = action_index.reshape(-1).to(masked_logits.device).long()
    if idx.shape[0] != masked_logits.shape[0]:
        raise ValueError("%d actions for %d rows"
                         % (idx.shape[0], masked_logits.shape[0]))
    live = _live_logits(masked_logits, "log_prob_entropy")
    log_probs = torch.log_softmax(live, dim=-1)
    log_prob = log_probs.gather(1, idx.reshape(-1, 1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_prob, entropy


def kl_to_anchor(masked_logits, anchor_masked_logits):
    """`KL(current || anchor)` per row, under the current row's mask.

    Both arguments are expected to be masked already and with the same mask:
    the anchor is `hc_1000` scored on the same position, so it carries the same
    legal set. The sum runs over all 36,400 slots rather than over a gathered
    list of legal ones because the illegal slots contribute exactly zero - both
    distributions give them probability 0, so `p * (log p - log q)` is `0 * 0`.

    Exactly zero against itself: the same tensor through both paths makes
    `log_p - log_q` identically zero, bit for bit.

    KL is a non-negative quantity in exact arithmetic. In float32 the sum of
    36,400 terms can land a few ulp below zero, so a caller asserting
    `kl >= 0` is asserting something the arithmetic does not promise;
    `tests/test_rl_policy.py` bounds it at 1e-6 and says why.
    """
    import torch

    if masked_logits.shape != anchor_masked_logits.shape:
        raise ValueError("shapes differ: %s vs %s"
                         % (tuple(masked_logits.shape),
                            tuple(anchor_masked_logits.shape)))
    live = _live_logits(masked_logits, "kl_to_anchor")
    # The anchor is lifted by the *same* legality pattern, not its own: if the
    # two masks ever disagreed, comparing them would be comparing two different
    # action spaces rather than two policies over one.
    anchor = anchor_masked_logits.masked_fill(
        masked_logits <= ILLEGAL_AT_OR_BELOW,
        torch.finfo(masked_logits.dtype).min)
    log_p = torch.log_softmax(live, dim=-1)
    log_q = torch.log_softmax(anchor, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def episode_reward(remaining_by_owner, owner):
    """One seat's reward for a finished game: `(5 - rank) - remaining / 100`.

    `remaining_by_owner` is `engine.result(state)` - the four seats' unplaced
    squares in raw cells, indexed by owner id. `plan8.md` fixes the units at raw
    cells precisely so this stays comparable with `records.rank_rows`, which
    also takes raw cells.

    The **ranking is not written out here**: `records.rank_rows` decides it,
    competition style, so a tie shares a place and the following places skip.
    That is the same rule the leaderboard folds and the one `Game.standings`
    feeds, and re-deriving it would be a third definition free to disagree with
    the other two about who won.

    `rank_rows` sorts by `(remaining, key)`, so the keys matter for ties. They
    are the owner ids here, which is what `Game.standings` passes, so two seats
    level on cells are ordered by seat the same way the game's own standings
    order them.

    `records` is imported inside the function, following `rl.imitation`'s
    precedent for reaching the game layer: `rl` does not depend on it at import
    time.
    """
    from records import rank_rows

    remaining = tuple(int(x) for x in remaining_by_owner)
    if len(remaining) != 4:
        raise ValueError("a game has four seats, got %d" % len(remaining))
    owner = int(owner)
    if not 0 <= owner < 4:
        raise ValueError("owner %r is not a seat" % (owner,))
    for key, rank, _points, rem in rank_rows([(o, remaining[o])
                                              for o in range(4)]):
        if key == owner:
            return (5 - rank) - rem / 100.0
    raise ValueError("owner %d is not in the standings" % owner)


def mask_fill_value():
    """The constant illegal logits are filled with, for a caller that wants to
    assert on it without importing `rl.imitation` itself."""
    return MASK_FILL