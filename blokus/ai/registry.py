"""Personality registry: one file, two layers, two different questions.

**Upper layer - "how does it score".** `WEIGHTED_SPECS` and
`RULE_BRAIN_CLASSES` map a personality key to the Python that evaluates a
position, and `personality_keys()` reads that mapping out in a fixed order.
Adding a personality takes three steps: define `KEY` / `PROFILE_SPEC`
(weighted type) or a `Brain` subclass (rule-based type) in your own module,
then register it here. Nothing about a seat's *file*, or about whether it is
*in a pool*, lives in this layer - only which class decides a move.

**Lower layer - "who is a contestant".** `ai/registry.json` is the single
source of truth for the roster: which keys exist, under which aliases, with
which `label` / `desc_key`, and which of the four pools list them under
`enabled` / `selectable`. `seats.automated_options()` and `seats.ALIASES`
read it, so changing the opponent pool is a data edit rather than a code
change. The array's order is *registration* order and nothing else: every pool
leaves this module through `pool_order`, which sorts by key, so reordering the
rows reorders the file and not a single game.

As of plan9a stage 4 the JSON owns `checkpoint` - the path a key is *loaded
from* - while `source` records where it was trained and `seats.IMITATION_STEPS`
/ `seats.RL_SEATS` still decide which keys are network seats in the first
place. `tests/test_registry_json.py` fails the moment the two disagree.
`module` and `kind` are mirrored from the code tables for the same reason.

Import-time validation is structural only: unique keys, aliases that collide
with nothing, legal pool names, `label` / `desc_key` present in `config.I`,
and `enabled` agreeing with `selectable`. Touching the filesystem at import
would make every `import seats` pay for a stat, so file existence and sha256
are left to `check_files()`, which `python -m ai --check` and the test suite
call.
"""
from .heuristics import (builder, chess, fox, hunter, intruder, optimizer,
                         wolf)
from .base import (BUILDER_KEY, HUNTER_KEY, INTRUDER_KEY, OPTIMIZER_KEY,
                   RULE_KEYS)
from .base import make_profile as _make_profile

# (key, Brain class, profile spec) for the weighted personalities. The order
# decides the draw order of `personality_keys()`, and also the UI order.
WEIGHTED_SPECS = (
    (wolf.KEY, wolf.WolfBrain, wolf.PROFILE_SPEC),
    (chess.KEY, chess.ChessBrain, chess.PROFILE_SPEC),
    (fox.KEY, fox.FoxBrain, fox.PROFILE_SPEC),
)

# Rule-based personalities = the ones whose objective function is not a
# weighted sum. They still carry a profile spec, used only for the coarse
# screen in stage 1 and for scoring when they are predicted.
RULE_BRAIN_CLASSES = {
    INTRUDER_KEY: intruder.IntruderBrain,
    OPTIMIZER_KEY: optimizer.OptimizerBrain,
    BUILDER_KEY: builder.BuilderBrain,
    HUNTER_KEY: hunter.HunterBrain,
}

PROFILE_SPECS = {key: spec for key, _cls, spec in WEIGHTED_SPECS}

WEIGHTED_BRAIN_CLASSES = {key: cls for key, cls, _spec in WEIGHTED_SPECS}


def personality_keys():
    """The keys of all personalities, in a fixed order."""
    return tuple(key for key, _cls, _spec in WEIGHTED_SPECS) + RULE_KEYS


def make_profile(key, rng):
    """The profile spec for `key`, plus a random perturbation. Only the
    weighted personalities have a profile spec."""
    return _make_profile(PROFILE_SPECS[key], rng)


def draw_personalities(rng, n=3):
    """Draw n **mutually distinct** personalities."""
    return rng.sample(list(personality_keys()), n)


def make_brain(key, rng, game_seed=None):
    """Build a personality.

    Rule-based personalities are handed chess's profile spec (not their own),
    used only for the coarse screen in stage 1 and for scoring when they are
    predicted; their own objective functions live entirely in their respective
    Brain classes.
    `game_seed` is passed only to a class that advertises `accepts_game_seed`,
    and `rng` only alongside it. A driver that knows the game's seed passes the
    seed; a caller with none - a normal game, where `Game` exposes only an `rng` -
    lets Hunter derive one from that rng's state without consuming from it. Every
    other personality ignores both, so adding the arguments changed nothing about
    how they are built.
    """
    cls = RULE_BRAIN_CLASSES.get(key)
    if cls is not None:
        profile = make_profile(chess.KEY, rng)
        if getattr(cls, "accepts_game_seed", False):
            return cls(key, profile, game_seed=game_seed, seed_rng=rng)
        return cls(key, profile)
    return WEIGHTED_BRAIN_CLASSES[key](key, make_profile(key, rng))


# ==========================================================================
# Lower layer: `ai/registry.json`, the data half of this module.
#
# The stdlib imports sit down here rather than at the top so that the layer
# above stays exactly the code that answers "which Brain scores a move".
# ==========================================================================
import hashlib
import json
import os
import sys

from config import I

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "registry.json")

# The four pools `match.POOL_PRESETS` defines, named here rather than imported.
# `match` imports `seats`, and `seats` imports this module, so importing `match`
# back would close the loop; `tests/test_registry_json.py` is what keeps this
# constant and `match.POOL_PRESETS` telling the same story.
POOL_NAMES = ("all", "no_imitation", "imitation_only", "rl_only")

KINDS = ("network", "heuristic")
FAMILIES = ("personality", "imitation", "rl")

# The fields that only a network entry may carry. They are recorded, not yet
# obeyed: which file a key means is still `seats.RL_SEATS` /
# `seats.IMITATION_STEPS`, and that is deliberate while `data/` is still the
# training factory. plan9a stage 4 repoints them at `ai/checkpoints/`.
NETWORK_FIELDS = ("checkpoint", "source", "sha256")

_ENTRY_FIELDS = frozenset(("key", "aliases", "kind", "family", "module",
                           "checkpoint", "source", "sha256", "pools",
                           "enabled", "selectable", "label", "desc_key",
                           "note"))

# The seat PPO measures itself against: one, and it is the 0k checkpoint,
# because that is where training starts and what stays frozen as the KL anchor.
# Named as a *seat* rather than a file on purpose - `anchors()` answers with the
# registered name, and the file still comes from `seats`. `hc_1000` is an alias
# for the same seat and is never an anchor in its own right.
RL_KL_ANCHOR = "rl_h1000_0k"


def _validate(entries):
    """Every structural complaint about `entries`, collected before raising.

    Collected rather than raised on the first one because this runs at import:
    an editor who mistypes two fields should see both mistakes at once, not
    play whack-a-mole against a module that refuses to import.
    """
    problems = []
    bad = problems.append
    keys = []
    alias_owner = {}
    for i, e in enumerate(entries):
        where = "entries[%d]" % i
        if not isinstance(e, dict):
            bad("%s: not an object" % where)
            continue
        unknown = set(e) - _ENTRY_FIELDS
        if unknown:
            bad("%s: unknown field(s) %s" % (where, sorted(unknown)))
        key = e.get("key")
        if not isinstance(key, str) or not key:
            bad("%s: `key` must be a non-empty string" % where)
            continue
        where = key
        keys.append(key)

        kind, family = e.get("kind"), e.get("family")
        if kind not in KINDS:
            bad("%s: kind %r is not one of %s" % (where, kind, KINDS))
        if family not in FAMILIES:
            bad("%s: family %r is not one of %s" % (where, family, FAMILIES))
        # A personality is the only thing scored by Python in this repository;
        # everything else is a file. Getting this pair wrong is how a checkpoint
        # ends up being handed to `make_brain`.
        if kind is not None and family is not None:
            if (kind == "heuristic") != (family == "personality"):
                bad("%s: kind %r and family %r do not go together"
                    % (where, kind, family))
        if kind == "heuristic":
            if not isinstance(e.get("module"), str) or not e.get("module"):
                bad("%s: a heuristic entry needs `module`" % where)
        elif kind == "network":
            for f in NETWORK_FIELDS:
                if not isinstance(e.get(f), str) or not e.get(f):
                    bad("%s: a network entry needs `%s`" % (where, f))

        aliases = e.get("aliases", [])
        if not isinstance(aliases, list) or \
                not all(isinstance(a, str) and a for a in aliases):
            bad("%s: `aliases` must be a list of non-empty strings" % where)
            aliases = []
        if len(set(aliases)) != len(aliases):
            bad("%s: duplicate alias in %r" % (where, aliases))
        for a in aliases:
            if a in alias_owner:
                bad("alias %r is claimed by both %s and %s"
                    % (a, alias_owner[a], key))
            alias_owner[a] = key

        pools = e.get("pools")
        if not isinstance(pools, list):
            bad("%s: `pools` must be a list" % where)
        else:
            if len(set(pools)) != len(pools):
                bad("%s: duplicate pool in %r" % (where, pools))
            for p in pools:
                if p not in POOL_NAMES:
                    bad("%s: pool %r is not one of %s" % (where, p, POOL_NAMES))

        for f in ("enabled", "selectable"):
            if not isinstance(e.get(f), bool):
                bad("%s: `%s` must be true or false" % (where, f))
        # The two flags document two different questions - "is it in the
        # regular pool" and "can the seat menu and the random draw offer it" -
        # but today they gate two views of the same seventeen keys, so flipping
        # one without the other would give the league and the seat menu two
        # different answers with no failure to show for it. They are flipped
        # together, and a disagreement is refused here rather than discovered
        # when a pool comes out one key short.
        if isinstance(e.get("enabled"), bool) and \
                isinstance(e.get("selectable"), bool) and \
                e["enabled"] != e["selectable"]:
            bad("%s: enabled=%s and selectable=%s disagree; the two are "
                "flipped together" % (where, e["enabled"], e["selectable"]))
        for f in ("label", "desc_key"):
            v = e.get(f)
            if not isinstance(v, str) or not v:
                bad("%s: `%s` must be a non-empty string" % (where, f))
            elif v not in I:
                bad("%s: %s %r is not a key in config.I" % (where, f, v))
        if "note" in e and not isinstance(e["note"], str):
            bad("%s: `note` must be a string" % where)

    unique = set(keys)
    if len(unique) != len(keys):
        bad("duplicate key(s) %s" % sorted({k for k in keys if keys.count(k) > 1}))
    for a in alias_owner:
        if a in unique:
            bad("alias %r collides with a registered key" % a)

    if problems:
        raise ValueError("ai/registry.json is not valid:\n  - "
                         + "\n  - ".join(problems))


def _validate_roster(keys):
    """Checks that only make sense against a whole roster, not one entry.

    Split out so that `_validate` can be exercised entry by entry: the anchor
    is a property of the roster, and demanding it of a single entry would make
    every isolated validation report an unrelated complaint.
    """
    if keys and RL_KL_ANCHOR not in set(keys):
        raise ValueError("ai/registry.json is not valid:\n  - "
                         "RL_KL_ANCHOR %r is not a registered key"
                         % RL_KL_ANCHOR)


def _load():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise ValueError("ai/registry.json must be an object with an "
                         "`entries` list")
    _validate(raw["entries"])
    _validate_roster([e["key"] for e in raw["entries"]
                      if isinstance(e, dict) and isinstance(e.get("key"), str)])
    return tuple(raw["entries"])


ENTRIES = _load()
_BY_KEY = {e["key"]: e for e in ENTRIES}
_KEYS = tuple(e["key"] for e in ENTRIES)
_ALIASES = {}
for _e in ENTRIES:
    for _a in _e.get("aliases", ()):
        _ALIASES[_a] = _e["key"]


def keys():
    """Every registered key, in registration order - the JSON array's order.

    Registration order is a property of the file, not of behaviour: since stage 3
    no draw reads it. `rng.choice` draws an index into a pool, and every pool is
    ordered by `pool_order` below, so moving a row in the JSON now changes
    nothing anyone can observe. What still reads this order is the cross-check in
    `tests/test_registry_json.py`, which compares it against `WEIGHTED_SPECS` and
    `IMITATION_STEPS` - those tables are ordered too, and they have to agree.
    """
    return _KEYS


def pool_order(keys):
    """The one rule for the order of any pool: by key, ascending.

    Plain `sorted` on the key string - no locale, no case folding, no natural
    sort of the digits in `hc_1000` / `hc_10000`. It is written once, here,
    because every pool in the project funnels through `automated_options()` or
    `pool()`, and a second rule anywhere else would make a pool's order depend on
    which function produced it.

    Why key order and not registration order: `rng.choice` draws an *index*, so
    a pool's order is the index-to-key mapping every same-seed game depends on.
    Registration order belongs to `ai/registry.json`, which is edited when the
    roster changes - so on registration order, adding an unrelated AI silently
    reseated every committed batch. Key order is a function of the *set*, so it
    changes only when the set does.
    """
    return tuple(sorted(keys))


def resolve(key):
    """The registered name for `key`, following the JSON's alias table.

    One hop, deliberately - see `seats.canonical_key`, which is the same lookup
    read from the seat side.
    """
    return _ALIASES.get(key, key)


def aliases():
    """The alias table as a fresh dict: `{old spelling: registered name}`."""
    return dict(_ALIASES)


def entry(key):
    """The JSON object for `key`, resolving an alias first."""
    key = resolve(key)
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ValueError("no registered seat %r; known: %s"
                         % (key, ", ".join(_KEYS)))


def automated_options():
    """The selectable keys, in `pool_order` - by key, ascending.

    What the seat menu shows, what `RANDOM_AI_KEY` draws and what a league draws
    its four seats from - one list, because those three have always been the same
    seventeen things, and because one order means a pool produced by any of the
    three is the same pool.
    """
    return pool_order(e["key"] for e in ENTRIES if e["selectable"])


def pool(name):
    """The keys of one named pool, in `pool_order` - by key, ascending.

    `enabled` gates every pool at once rather than only "the regular one",
    because at this stage the regular pool and the four presets are all the
    same seventeen keys. `_validate` refuses an entry whose `enabled` and
    `selectable` disagree, so this view and `automated_options()` cannot drift
    apart behind a half-flipped flag.
    """
    if name not in POOL_NAMES:
        raise ValueError("no such pool: %r; known: %s"
                         % (name, ", ".join(POOL_NAMES)))
    return pool_order(e["key"] for e in ENTRIES
                      if e["enabled"] and name in e["pools"])


def pools():
    """Every named pool, as `{name: keys}` in `pool_order`."""
    return {name: pool(name) for name in POOL_NAMES}


def anchors():
    """The seats an RL run measures itself against, by registered name.

    One seat, and `rl_h1000_0k` - the 0k checkpoint PPO starts from and keeps
    frozen as its KL anchor. It answers with the *registered* name and never
    with `hc_1000`: that spelling is an alias for the same seat, and an anchor
    recorded two ways is how an eval log grows a second column for one model.

    Nothing reads it yet. `rl/rl_train.py` still spells the anchor `hc_1000`
    itself, because that spelling is what `data/rl1/eval.jsonl` rows are keyed
    by and changing it would split one baseline into two columns. Wiring the RL
    side to this function is a later-stage decision, not done here.
    """
    return (RL_KL_ANCHOR,)


def label_key(key):
    """The `config.I` key naming `key`."""
    return entry(key)["label"]


def desc_key(key):
    """The `config.I` key describing `key`."""
    return entry(key)["desc_key"]


def checkpoint(key):
    """The file a network seat loads from - the authority since stage 4.

    `source` still records where it was trained, and `seats.IMITATION_STEPS` /
    `seats.RL_SEATS` still decide *which keys are network seats at all*; but
    which path a key means is read from here. A published weight's identity is
    the file, and the file belongs with the code that reads it - that is the
    whole reason `ai/checkpoints/` exists.
    """
    e = entry(key)
    if e["kind"] != "network":
        raise ValueError("%r is a %s seat; it has no checkpoint"
                         % (key, e["kind"]))
    return e["checkpoint"]


def check_files(verify_sha256=True):
    """Every complaint about the files the JSON names. Empty means all good.

    Deliberately not called at import: it stats and hashes, and a game that
    never plays a checkpoint should not pay for that. `python -m ai --check`
    is its only caller outside the unit tests of this function itself - the
    suite deliberately does **not** assert the real checkpoints' digests until
    stage 4 publishes them into `ai/checkpoints/`.
    """
    problems = []
    for e in ENTRIES:
        if e["kind"] != "network":
            continue
        path = e["checkpoint"]
        if not os.path.isfile(path):
            problems.append("%s: checkpoint %s does not exist"
                            % (e["key"], path))
            continue
        if not verify_sha256:
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != e["sha256"]:
            problems.append("%s: %s sha256 %s, registry says %s"
                            % (e["key"], path, got, e["sha256"]))
    return problems


def _digest_prefix(path, length=12):
    """The first `length` hex characters of a file's sha256, or "-".

    Display only - `check_files` is what *verifies* a digest and only when it
    is asked to. `--list` hashes because a prefix that is not read off the
    file would be a claim about a file nobody opened; five published
    checkpoints are about 28 MB, which is a fraction of a second.
    """
    if not os.path.isfile(path):
        return "-"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def list_rows():
    """One tuple per registered seat, in `pool_order` - by key, ascending.

    Key order because that is the only order the project has: `pool_order` is
    what every draw reads, so a listing sorted any other way would describe a
    pool that does not exist.
    """
    rows = []
    for key in pool_order(keys()):
        e = entry(key)
        rows.append((
            key, e["kind"], e["family"],
            "%s/%s" % ("y" if e["enabled"] else "n",
                       "y" if e["selectable"] else "n"),
            ",".join(e["pools"]),
            _digest_prefix(e["checkpoint"]) if e["kind"] == "network" else "-",
            e.get("checkpoint") or e.get("module") or "-",
            ",".join(e.get("aliases") or []) or "-",
        ))
    return rows


_LIST_HEADER = ("key", "kind", "family", "en/sel", "pools", "sha256",
                "file", "aliases")
# One space wider than each column so two padded cells never touch: `personality`
# is exactly eleven characters and would otherwise run into `en/sel`.
_LIST_WIDTHS = (15, 11, 12, 7, 27, 13, 47, 0)


def _print_row(cells):
    """Pad every column except the last, which runs to the end of the line."""
    for i, cell in enumerate(cells):
        width = _LIST_WIDTHS[i]
        if width:
            print("%-*s" % (width, cell), end=" " if width else "")
        else:
            print(cell)


def main(argv):
    """`python -m ai --check` (verify) and `python -m ai --list` (roster)."""
    if argv == ["--check"]:
        problems = check_files()
        for p in problems:
            print("FAIL %s" % p)
        for name in POOL_NAMES:
            print("%-15s %s" % (name, ", ".join(pool(name))))
        return 1 if problems else 0
    if argv == ["--list"]:
        _print_row(_LIST_HEADER)
        for row in list_rows():
            _print_row(row)
        return 0
    print("usage: python -m ai --check | python -m ai --list")
    return 2


if __name__ == "__main__":
    # `python -m ai.registry` executes this module a second time, beside the
    # copy `ai/__init__` already imported, and runpy warns about the duplicate
    # before a line here runs. The package entry point does not have that
    # problem, so this branch only exists to say where the real command is.
    print("use `python -m ai --check` or `python -m ai --list` instead",
          file=sys.stderr)
    raise SystemExit(2)

