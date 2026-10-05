"""All-AI league: each game draws four seats from the eleven automated options,
and the results feed the leaderboard.

This is a tool for answering "which option is actually strongest"; it does not
affect the flow of a normal player game.

The eleven are the seven personalities, the three H-C2 imitation steps, and one
PPO-trained policy. The two human seats are excluded: a league of people cannot
be replayed from a seed, and a seat with no brain would have nothing to choose
with.

    python3 match.py --games 100            # record into records.json
    python3 match.py --games 20 --dry       # run only, write nothing
    python3 match.py --games 100 --reset    # clear the leaderboard first
    python3 match.py --games 5 --pool imitation_only   # only the checkpoints
    python3 match.py --games 5 --pool no_imitation,hunter
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys

import ai
import seats as seats_mod
from config import PROJECT_DIR
from game import Game
from records import POINTS_FOR_RANK, Records, rank_rows

# Where a batch evaluation is written. Under `data/`, which `.gitignore`
# excludes, and deliberately **not** `records.json`: that file is the
# cumulative leaderboard of the human player's own games, and a benchmark run
# would flood it with thousands of machine-vs-machine results.
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "data", "match")

# Bumped whenever the paired streams are derived differently, and written into
# every batch file produced in that mode. A paired result is not comparable with
# an unpaired one and neither is comparable with a paired result from another
# version, so the label has to travel with the numbers rather than live in
# somebody's memory.
PAIRED_RNG_VERSION = "paired-v1"
PAIRED_RNG_SUBJECT_VERSION = "paired-v1-subject"

LEGACY_RNG_MODE = "legacy-single-stream"


def rng_mode_of(paired_rng, subject=None):
    """The label that travels with a batch, from how its games were laid out.

    A subject batch is a third thing rather than a variant of the second: its
    seat draw is four draws in a different pattern, so it plays different games
    from the same seed. Labelling it `paired-v1` would let it be averaged with a
    subjectless paired batch by anyone who checked the one field they were told
    to check.
    """
    if not paired_rng:
        return LEGACY_RNG_MODE
    return PAIRED_RNG_SUBJECT_VERSION if subject else PAIRED_RNG_VERSION


def paired_stream_seed(*parts):
    """A seed for one stream, derived from labels rather than from position.

    The whole point of the split is that a stream's contents cannot depend on
    how much of it an earlier game happened to consume, so nothing here may be
    "the next draw" - every stream has to be nameable. Hence blake2b over the
    labels `(seed, game index, "seat", owner)` rather than arithmetic on the
    game index: arithmetic invites an off-by-one that silently reuses a stream,
    and `hash()` is salted per process, which would make a paired run
    unreproducible from the seed alone.

    The separator matters. `repr` of each part plus a byte that cannot appear
    inside a repr keeps `("a", "bc")` and `("ab", "c")` distinct.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def paired_streams(seed, game_index):
    """`(setup_rng, [seat_rng, seat_rng, seat_rng, seat_rng])` for one game.

    Five streams: one that decides who is playing and who goes first, and one per
    seat for that seat's own decisions. All five are functions of `(seed,
    game_index)` alone, so game 300 is drawn exactly the same way whether it is
    reached by playing 299 games before it or on its own.
    """
    setup = random.Random(paired_stream_seed(seed, game_index, "setup"))
    seats = [random.Random(paired_stream_seed(seed, game_index, "seat", o))
             for o in range(4)]
    return setup, seats

# The three shorthands `--pool` accepts. `all` is the whole automated pool, and
# is spelled out as the ordered tuple rather than a filter so that
# `expand_pool("all")` is the *same list object contents* as the default - the
# same length in the same order, which is what makes `rng.choice` consume the
# random stream identically and keeps a default run bit-for-bit identical to one
# that was given `--pool all`.
#
# Which entries are the imitation seats is asked of `seats.kind_of` rather than
# tested against a string prefix. A prefix test is exactly what goes stale the
# moment the key naming changes, and it fails *silently*: a prefix that matches
# nothing turns `imitation_only` into an empty pool and `no_imitation` into the
# whole pool. That is not hypothetical - the pool moved from `step_*` to `hc_*`
# when H-B2's checkpoints were replaced by H-C2's.
def _is_imitation(key):
    return seats_mod.kind_of(key) == seats_mod.KIND_IMITATION


def _is_rl(key):
    return seats_mod.kind_of(key) == seats_mod.KIND_RL


_ORDER = seats_mod.automated_options()
POOL_PRESETS = {
    "all": _ORDER,
    # "no network" rather than "not an imitation checkpoint": the trained policy
    # is a network, so leaving it in here would make the name wrong.
    "no_imitation": tuple(k for k in _ORDER
                          if not _is_imitation(k) and not _is_rl(k)),
    # Only the checkpoints that were trained *by imitation*, which is what the
    # name has always meant here.
    "imitation_only": tuple(k for k in _ORDER if _is_imitation(k)),
    # The trained policy alone - the question "how does it actually do" asked
    # without also measuring three imitation checkpoints it is descended from.
    "rl_only": tuple(k for k in _ORDER if _is_rl(k)),
}


def league_options():
    """The eleven options a league seat can be drawn from."""
    return seats_mod.seat_options(include_humans=False)


def expand_pool(text):
    """`--pool` text -> the deduplicated, ordered list of options to draw from.

    A comma-separated list of items, each either a preset name or one option
    name. Whitespace around items is allowed, and so is the whole string being
    quoted, because that is what a shell hands over when the commas are inside
    quotes.

    **Every distinct option in the result carries the same weight, 1/n.** Items
    are a union, not a multiset: naming `hunter` as well as `no_imitation` does
    not give `hunter` two chances, because `hunter` is already in that preset.
    Repeating an item, or listing a preset and a name that overlaps it, changes
    nothing.

    The order is `seats.automated_options()` order, always - not the order the
    items were given in, and not the traversal order of a set. Two `--pool`
    arguments with the same contents therefore expand to the same list and play
    the same games from the same seed.

    An unknown name is an error naming what is legal, never a silent omission:
    a typo that dropped an option would quietly change what is being measured.
    """
    if text is None:
        return list(_ORDER)
    body = text.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
        body = body[1:-1].strip()
    items = [p.strip() for p in body.split(",")]
    items = [p for p in items if p]
    if not items:
        raise ValueError(
            "--pool was given nothing to expand; expected a comma-separated "
            "list of %s or one of %s"
            % (", ".join(_ORDER), ", ".join(sorted(POOL_PRESETS))))
    chosen = set()
    for item in items:
        if item in POOL_PRESETS:
            chosen.update(POOL_PRESETS[item])
        elif item in _ORDER:
            chosen.add(item)
        else:
            raise ValueError(
                "unknown --pool item %r. Presets: %s. Options: %s"
                % (item, ", ".join(sorted(POOL_PRESETS)), ", ".join(_ORDER)))
    # Ordered by the fixed option order, so the input order cannot matter.
    pool = [k for k in _ORDER if k in chosen]
    if not pool:
        raise ValueError("--pool %r expanded to an empty pool" % (text,))
    return pool


def play_match(game, rng, on_move=None, seat_rngs=None):
    """Play one game to the end.

    Returns one row per **seat**: `(option, colour, remaining, rank, points)`.
    The colour is part of the result, not decoration: the same option can be
    drawn twice in one game on two different corners, and only the pair
    identifies which seat a row is about. Rank and points are per seat too, so
    two seats on one option each carry their own - which is the whole reason
    this is a list and not something keyed by option.

    `on_move(game, owner, move)` can be hooked in to check something after each
    turn (verifying the corner-contact rule move by move, say), the same way
    `tests/test_simulation.py` does it.

    `seat_rngs` gives each seat the generator its decisions are drawn from,
    instead of every seat sharing `rng`. The reason is the same as in
    `Game.setup_seats`: `choose_move` draws a number of times that depends on
    the position and on the model's own mistake rate, so on one shared stream
    two models sitting in the same seat leave the rest of the league reading
    from different places. With a stream per seat the consumption stops being
    anyone else's problem. Left `None`, every seat reads from `rng`, which is
    the original behaviour and still the default.
    """
    game.start()
    while game.state == "PLAYING":
        owner = game.current_owner()
        move = ai.choose_move(game.board, game.hands[owner].names, owner,
                              game.brains[owner],
                              rng if seat_rngs is None else seat_rngs[owner],
                              other_brains=game.brain_map(),
                              must_cover=game.must_cover(owner),
                              reach=game.reach(owner),
                              other_must_cover={o: game.must_cover(o)
                                                for o in range(4)},
                              other_reach={o: game.reach(o) for o in range(4)})
        if move:
            if on_move is not None:
                on_move(game, owner, move)
            game.act(*move)
        else:
            game.act_pass()
    # Ranked by the seat option, exactly as the leaderboard folds it, so a row
    # here and that entry cannot disagree.
    #
    # `rank_rows` returns the seats best-first, so it has to be walked alongside
    # the same ordering rather than zipped with seat order. The two sorts use the
    # same key and are both stable over seat order, so even when two seats tie
    # on (remaining, option) they stay paired - and a tie gives them the same
    # rank and the same points anyway.
    standings = [(game.owner_key[o], game.remaining_cells(o)) for o in range(4)]
    order = sorted(range(4), key=lambda o: (standings[o][1], standings[o][0]))
    out = [None] * 4
    for o, (_key, rank, points, _rem) in zip(order, rank_rows(standings)):
        out[o] = (game.owner_key[o], game.colors[o], standings[o][1],
                  rank, points)
    return out


def summarise(rows):
    """Per-option statistics over seat appearances.

    Every row is one seat, so an option that held two seats in a game is
    counted twice - once for each seat's own rank, points and remaining squares.
    That is the same accounting `records.py` folds into the leaderboard, and the
    point of a pool benchmark is to compare options on equal terms: averages are
    per appearance, so they stay on the 0-4 and 0-89 scales however many seats
    an option happened to hold.

    `points_stderr` is the sample standard deviation of the points divided by
    the square root of the number of appearances, so two options can be told
    apart from their averages alone. With a single appearance there is no spread
    to measure and it is reported as 0.0.
    """
    acc = {}
    for key, _colour, remaining, _rank, points in rows:
        rec = acc.setdefault(key, {"appearances": 0, "points": [],
                                   "remaining": []})
        rec["appearances"] += 1
        rec["points"].append(points)
        rec["remaining"].append(remaining)
    out = []
    for key in sorted(acc):
        rec = acc[key]
        pts, rem = rec["points"], rec["remaining"]
        n = len(pts)
        mean_p = sum(pts) / n
        mean_r = sum(rem) / n
        if n > 1:
            var = sum((p - mean_p) ** 2 for p in pts) / (n - 1)
            stderr = math.sqrt(var) / math.sqrt(n)
        else:
            stderr = 0.0
        out.append({
            "option": key,
            "appearances": n,
            "avg_points": mean_p,
            "avg_remaining": mean_r,
            "points_stderr": stderr,
        })
    out.sort(key=lambda r: (-r["avg_points"], r["avg_remaining"], r["option"]))
    return out


def default_out_path(seed, games, pool_raw):
    """Where a batch evaluation is written when `--out` is not given."""
    safe = "".join(c if c.isalnum() else "_" for c in (pool_raw or "all"))
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    return os.path.join(DEFAULT_OUT_DIR,
                        "seed%d-g%d-pool%s.json" % (seed, games, safe[:40]))


def subject_draw(rng, options, subject):
    """`(seat_index, keys)` for one game with exactly one designated seat.

    The seat is drawn first, then the other three, and the draw count is four in
    both this and the subjectless path - the same number, a different pattern, so
    a subject batch and a subjectless batch play different games from the same
    seed and are labelled differently for exactly that reason.

    **The seat is an index, and it stays an index.** Nothing downstream may count
    occurrences of the subject *key* to decide which seat was designated, because
    the subject is allowed to appear in the pool: `--subject hunter` against a
    pool containing `hunter` seats the same contestant twice in about 37% of
    games, and `keys.index(subject)` would then name the opponent rather than the
    seat that was meant. The duplicate is intended - it is the training condition
    being reproduced - but the statistics must be able to tell the two apart.
    """
    seat = rng.randrange(4)
    rest = [rng.choice(options) for _ in range(3)]
    return seat, rest[:seat] + [subject] + rest[seat:]


def subject_in_pool(subject, options):
    """True when the designated seat's key can also be drawn as an opponent.

    Not an error - for a personality subject that is the training condition, and
    silently dropping the subject from the pool would quietly measure a weaker
    model than the one trained. It is reported rather than prevented, because the
    reader of the result has to know that the same contestant can be seated twice
    in one game before they can read the numbers.
    """
    return subject is not None and subject in options


def validate_subject(subject, options, paired_rng):
    """Raise unless `subject` is usable, naming what is legal.

    Two ways it can be wrong, and both are silent failures otherwise: an unknown
    name would build a seat that does not exist, and a subject without the split
    would produce numbers labelled as paired that are not paired. The subject
    being in the pool is **not** checked - see `subject_in_pool`.
    """
    if subject is None:
        return
    legal = sorted(seats_mod.automated_options())
    if subject not in legal:
        raise ValueError("unknown --subject %r. Options: %s"
                         % (subject, ", ".join(legal)))
    if not paired_rng:
        raise ValueError(
            "--subject needs --paired-rng: a designated seat is only worth "
            "anything if the same game index lays out the same board twice, and "
            "without the split it does not")


def batch_text(payload):
    """The exact bytes a batch file holds, plain or compressed.

    Serialised once and handed to whichever writer there is, because "the
    compressed batch contains the same JSON" is only true if both go through the
    same `json.dumps` and nobody is tempted to re-encode on the way in.
    """
    return json.dumps(payload, indent=1, ensure_ascii=False) + "\n"


def write_batch(path, payload, compress=False):
    """Write a batch file and return the path actually written.

    `compress` writes `<path>.gz` **instead of** `<path>` rather than both. Two
    copies of an evaluation is two things that can disagree, and the one that is
    silently stale is the one somebody reads.

    The gzip header carries `mtime=0` and no stored filename, so the same payload
    always produces the same bytes. That is what makes a committed batch
    comparable by hash: without it every run stamps the current time into the
    header and `git diff` reports a change for a re-run that produced identical
    numbers.

    Nothing else is normalised. The level is 9, and the compressed bytes are
    byte-for-byte the decompressed ones - this is a transport choice, not a
    different format.
    """
    text = batch_text(payload)
    if not compress:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path
    final = path if path.endswith(".gz") else path + ".gz"
    with open(final, "wb") as raw:
        # `filename=""` suppresses the FNAME field. Left to itself GzipFile takes
        # the name from the file object, which would put this batch's basename
        # in the header and make two identical batches differ whenever they were
        # written under different names.
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                           fileobj=raw, mtime=0) as fh:
            fh.write(text.encode("utf-8"))
    return final


def run_league(games=100, seed=0, records=None, on_game=None, on_move=None,
               options=None, mode="argmax", checkpoint_dir=None, device=None,
               paired_rng=False, subject=None):
    """Run `games` league matches, recording each into the leaderboard.

    Every seat is drawn independently from `options`, so options repeat freely -
    which is what lets a league answer "is the imitation better than the
    personalities" rather than only "which four personalities cooperate best".
    Colours and the opening player are decided by `setup_seats`, the same code
    the UI uses, so a league game and a played game are laid out the same way.

    Returns a list of games, each a list of four seat rows as `play_match`
    returns them - the game's rows, not a flat list, because four seats is the
    unit that was played.

    With `records=None` no file is touched at all, which is how the tests use
    it.

    `paired_rng` switches the randomness from one stream for the whole league
    to five per game, derived from `(seed, game_index)`. The default is the
    original single stream and is left exactly as it was: every game reads from
    wherever the previous one stopped, so a model that draws a different number
    of times moves every later game's seat draw, colours and opening player.
    That is fine for a league whose question is simply which option is strongest
    on average, and fatal for one whose question is whether *these two* models
    differ - they never meet the same board twice. In paired mode the setup of a
    given game index is a function of the seed alone, so two runs that differ
    only in one seat's model meet the same opponents, the same colours and the
    same opening player in every game. See `paired_streams`.

    `subject` names one seat that every game is guaranteed to have, which is what
    makes an evaluation of a single model possible: without it the model is in
    about 41% of games in an eight-option pool and every other game is dead
    weight. It requires `paired_rng` - see `validate_subject` - and it is
    deliberately allowed to also appear in `options`, because seating `hunter`
    against a pool that contains `hunter` is reproducing the training condition
    rather than making a mistake. `subject_draw` draws the seat from the setup
    stream, so two runs with different subjects meet the same board in every
    game.
    """
    options = league_options() if options is None else tuple(options)
    if not options:
        raise ValueError("run_league needs at least one option to draw from")
    validate_subject(subject, options, paired_rng)
    if checkpoint_dir is None:
        checkpoint_dir = seats_mod.IMITATION_CHECKPOINT_DIR
    rng = random.Random(seed)
    rows = []
    for i in range(games):
        if paired_rng:
            game_rng, seat_rngs = paired_streams(seed, i)
        else:
            game_rng, seat_rngs = rng, None
        g = Game(game_rng)
        if subject is None:
            keys = [game_rng.choice(options) for _ in range(4)]
        else:
            _seat, keys = subject_draw(game_rng, options, subject)
        g.setup_seats(keys, [None] * 4, game_rng, seat_rngs=seat_rngs,
                      checkpoint_dir=checkpoint_dir,
                      device=device, mode=mode)
        standings = play_match(g, game_rng, on_move, seat_rngs=seat_rngs)
        rows.append(standings)
        if records is not None:
            records.record([(k, rem) for k, _c, rem, _rk, _p in standings])
        if on_game is not None:
            on_game(i + 1, standings)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI 比賽")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--reset", action="store_true",
                    help="先清空排行榜再開始")
    ap.add_argument("--dry", action="store_true", help="不寫 records.json")
    ap.add_argument("--pool", default=None,
                    help="逗號分隔,無空白。項目可為 preset 或單一 AI 名稱:"
                         " preset 有 all(%d 個)、no_imitation(%d 個人格)、"
                         " imitation_only(%d 個模仿檢查點)、rl_only(%d 個訓練"
                         "結果);名稱為 "
                         % (len(POOL_PRESETS["all"]),
                            len(POOL_PRESETS["no_imitation"]),
                            len(POOL_PRESETS["imitation_only"]),
                            len(POOL_PRESETS["rl_only"]))
                         + ", ".join(seats_mod.automated_options()) + "。"
                         "展開後取聯集並去重,每個不同的 AI 權重相等(各 1/n);"
                         "重複的項目不增加權重。順序固定為選項的既定順序,"
                         "與輸入順序無關。不給則等同 all。")
    ap.add_argument("--out", default=None,
                    help="批量評測輸出檔(JSON);預設寫到 data/match/ 下,"
                         "不寫入 records.json")
    ap.add_argument("--mode", default="argmax", choices=("argmax", "softmax"),
                    help="模仿選項的選步方式")
    ap.add_argument("--device", default=None, help="模仿選項的 torch device")
    ap.add_argument("--paired-rng", action="store_true",
                    help="每局用五條由 (種子, 局序) 派生的獨立流（setup 流 + "
                         "每席位流），使只更換某席位模型時每局開局完全一致。"
                         "預設關閉，維持原有單一隨機流。")
    ap.add_argument("--gzip", action="store_true",
                    help="批次輸出直接以 .json.gz 取代未壓縮檔（不保留兩份）；"
                         "gzip 標頭 mtime=0，相同內容產生相同位元組。")
    ap.add_argument("--subject", default=None,
                    help="指定每局必有一席的選項；該席位由 setup 流抽出，其餘三席"
                         "自 --pool 有放回抽。必須同時使用 --paired-rng。")
    args = ap.parse_args(argv)

    pool_raw = args.pool
    try:
        pool = expand_pool(pool_raw)
        validate_subject(args.subject, pool, args.paired_rng)
    except ValueError as exc:
        # A bad pool or a bad subject is a usage error, not a crash: argparse
        # prints the usage and exits 2, which is what a typo deserves.
        ap.error(str(exc))

    records = None
    if not args.dry and not args.paired_rng:
        records = Records()
        if args.reset:
            records.reset()
            print("排行榜已清空")
    if args.paired_rng and not args.dry:
        # A paired batch is a different measurement from the leaderboard's: its
        # games are laid out by per-game streams rather than by one stream the
        # league walks down, and folding it in would average two designs whose
        # seat draws are not comparable. The batch file carries the per-game rows
        # the paired statistics need, so nothing is lost by not folding.
        print("成對模式：不寫入排行榜（--paired-rng），結果只寫入批次檔")
    rows = run_league(args.games, args.seed, records, options=pool,
                      mode=args.mode, device=args.device,
                      paired_rng=args.paired_rng)

    flat = [r for game in rows for r in game]
    summary = summarise(flat)
    appearances = {r["option"]: r["appearances"] for r in summary}

    print("%d 局完成（每局 4 席）" % args.games)
    print("  --pool 原始字串 : %s" % (pool_raw if pool_raw else "(未指定 → all)"))
    print("  展開後的池      : %s" % ", ".join(pool))
    print("  種子            : %d" % args.seed)
    print("  隨機流模式      : %s" % rng_mode_of(args.paired_rng, args.subject))
    if args.subject:
        print("  指定席位(subject): %s" % args.subject)
        if subject_in_pool(args.subject, pool):
            print("    注意: subject 也在池內，對手席可能抽到同一個，"
                  "故同一局可能兩席同選項（與訓練條件一致）")
    print("  局數            : %d" % args.games)
    print("  席次總數        : %d" % len(flat))
    print()
    print("  各選項席次:")
    for key in pool:
        print("    %-12s %4d 席" % (key, appearances.get(key, 0)))
    print()
    print("  選項            席次   平均分   平均餘格   分數標準誤")
    for row in summary:
        print("    %-12s %5d %8.3f %10.1f %11.4f"
              % (row["option"], row["appearances"], row["avg_points"],
                 row["avg_remaining"], row["points_stderr"]))

    out_path = args.out or default_out_path(args.seed, args.games, pool_raw)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        "pool_raw": pool_raw,
        "pool": pool,
        "pool_size": len(pool),
        "seed": args.seed,
        "games": args.games,
        "seats_per_game": 4,
        "mode": args.mode,
        # Travels with the numbers on purpose: a paired batch and a leaderboard
        # batch lay their games out differently and must never be averaged
        # together, and a paired batch from another version of the derivation
        # must not be averaged with this one either.
        "rng_mode": rng_mode_of(args.paired_rng, args.subject),
        # The designated seat, so a batch says which model it is about. It may
        # also appear among the opponents - that is the training condition for a
        # personality subject - so this is not recoverable from `games_detail`
        # after the fact.
        "subject": args.subject,
        "appearances": appearances,
        "summary": summary,
        "games_detail": [[[k, c, r, rk, p] for k, c, r, rk, p in game]
                         for game in rows],
    }
    out_path = write_batch(out_path, payload, compress=args.gzip)
    print("批量評測輸出已寫入 %s" % out_path)

    if records is not None:
        print()
        for key, games, pts, rem in records.rows(order=pool):
            print("  %-12s %3d 席次　平均 %.2f 分　平均餘 %.1f 格"
                  % (key, games, pts, rem))
        print("排行榜已寫入 %s" % records.path)
    else:
        print("排行榜未寫入（--dry）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
