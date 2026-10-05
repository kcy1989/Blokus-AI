"""Session-wide checks that the suite leaves the project's own data alone.

The repository keeps its datasets and checkpoints under `data/`, and
`records.json` beside it. Both are gitignored, so nothing in a commit would
notice a test that wrote to them - and a test that *re-writes* a shard, or drops
a stray file next to one, is exactly the kind of thing that gets written once by
a debugging attempt and then quietly depended on.

Two things are watched:

    data/h*      the dataset and checkpoint directories
    records.json the cumulative leaderboard

`data/humanlog` and `data/match` are **excluded on purpose**. Both are products
of the running application rather than of the tests: a human game appends to
`data/humanlog/`, and an evaluation run drops a batch into `data/match/`. The
whole reason `data/humanlog` needs naming is that it matches `h*` - `gamelog.LOG_DIR`
is `data/humanlog` and the guard's own pattern would otherwise sweep it in.

**Known limitation.** `records.json` is written by playing a game, not only by
the tests. A human who plays - or an evaluation run started without `--dry` -
while the suite is running will make this guard fail on a change it did not
cause. The failure message says so. Watching the leaderboard is worth that
because it is the one guarded path an ordinary interactive session touches; the
alternative, dropping it, would leave the most-used writable file unwatched.

A same-size rewrite that also preserved `mtime_ns` would not be caught. That is
the stated contract rather than an oversight: the snapshot is
`{path: (size, mtime_ns)}` because it has to be cheap enough to take twice
around a five-minute suite, and a writer that restores mtime deliberately is
not the accident this is looking for.

Under `pytest -n 8` the comparison runs only in the controller. xdist gives each
worker a `workerinput` attribute on its config and the controller has none, so
that is the test used here: a worker took its own "before" snapshot at session
start and compared it against a controller-side file that had already moved,
which would report every file as changed.
"""
import glob
import os

import pytest

import gamelog
import records

_HERE = os.path.dirname(os.path.abspath(__file__))

# Where the guarded files live. Overridable so the guard can be pointed at a
# scratch tree - that is how it is verified to actually fail, since the real
# data/ must not be written to in order to prove the guard works.
PROJECT_ROOT = os.environ.get("BLOKUS_DATA_GUARD_ROOT") or \
    os.path.dirname(_HERE)

# Directories under `data/` to watch, as patterns relative to PROJECT_ROOT.
GUARD_PATTERNS = ("data/h*",)

# ...and single files.
GUARD_FILES = ("records.json",)

# Matched by a pattern but deliberately not watched; see the module docstring.
GUARD_EXCLUDED = ("data/humanlog", "data/match")


def guard_roots(project_root=None):
    """The existing paths this session watches, relative to `project_root`.

    Resolved at session start rather than hard-coded, so a dataset directory
    added later is covered without editing this file, and so a checkout without
    `data/` simply watches nothing instead of raising.
    """
    project_root = PROJECT_ROOT if project_root is None else project_root
    roots = []
    for pattern in GUARD_PATTERNS:
        for path in sorted(glob.glob(os.path.join(project_root, pattern))):
            rel = os.path.relpath(path, project_root)
            if rel in GUARD_EXCLUDED or not os.path.isdir(path):
                continue
            roots.append(rel)
    roots.extend(f for f in GUARD_FILES
                 if os.path.exists(os.path.join(project_root, f)))
    return tuple(roots)


def snapshot(root):
    """`{path relative to root: (size, mtime_ns)}` for every file under `root`.

    A pure function of the filesystem: it reads, sorts and returns a fresh dict,
    and never writes. A missing `root` yields an empty snapshot rather than
    raising, so a checkout with no data compares equal to itself.

    Files, not directories: a directory's mtime moves when an unrelated entry
    appears anywhere beneath it, which would report every dataset as changed
    when a single shard was added.
    """
    out = {}
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:                 # vanished mid-walk
                continue
            out[os.path.relpath(full, root)] = (st.st_size, st.st_mtime_ns)
    return out


def snapshot_roots(roots, project_root=None):
    """`snapshot` over several roots, keyed by each file's project-relative path.

    Two roots can hold the same relative name, so the key has to carry which
    root it came from - otherwise `records.json` and a shard of the same name
    would collide into one entry and half the comparison would be vacuous.
    """
    project_root = PROJECT_ROOT if project_root is None else project_root
    out = {}
    for rel in roots:
        root = os.path.join(project_root, rel)
        if os.path.isfile(root):
            st = os.stat(root)
            out[rel] = (st.st_size, st.st_mtime_ns)
            continue
        for sub, value in snapshot(root).items():
            out[os.path.join(rel, sub)] = value
    return out


def diff_snapshots(before, after):
    """One line per difference, as `(kind, path, detail)` triples.

    Returns an empty list when nothing moved. `kind` is one of
    `"added" / "removed" / "rewrote"`, kept as data rather than a finished string
    so a caller can render it however it likes - and so the unit tests can assert
    on the kind without matching prose.
    """
    out = []
    for path in sorted(set(before) | set(after)):
        b, a = before.get(path), after.get(path)
        if b == a:
            continue
        if b is None:
            out.append(("added", path, "%d bytes" % a[0]))
        elif a is None:
            out.append(("removed", path, "%d bytes" % b[0]))
        elif b[0] == a[0]:
            out.append(("rewrote", path,
                        "size unchanged at %d bytes, mtime_ns %d -> %d"
                        % (b[0], b[1], a[1])))
        else:
            out.append(("rewrote", path, "%d -> %d bytes" % (b[0], a[0])))
    return out


def render_changes(changes):
    """The failure text, including what to do about a `records.json` change."""
    lines = ["%s %s (%s)" % (kind, path, detail) for kind, path, detail in changes]
    touched = [p for _k, p, _d in changes if p == "records.json"]
    if touched:
        lines.append(
            "If records.json changed and you were playing a game or ran an "
            "evaluation without --dry while the suite ran, that is the cause "
            "and not a test. This guard exists to catch tests writing to it.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the session hooks
# --------------------------------------------------------------------------

_BEFORE = None


def _is_worker(config):
    """True inside an xdist worker; False in the controller and under -n 0."""
    return hasattr(config, "workerinput")


def pytest_sessionstart(session):
    global _BEFORE
    if _is_worker(session.config):
        return
    _BEFORE = snapshot_roots(guard_roots())


def pytest_sessionfinish(session, exitstatus):
    if _is_worker(session.config) or _BEFORE is None:
        return
    changes = diff_snapshots(_BEFORE, snapshot_roots(guard_roots()))
    if not changes:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line(
            "the test suite modified files it must not write to "
            "(%d change%s):" % (len(changes), "" if len(changes) == 1 else "s"),
            red=True, bold=True)
        for line in render_changes(changes).splitlines():
            reporter.write_line("  " + line, red=True)
    session.exitstatus = 1

# --------------------------------------------------------------------------
# the repository is not writable from a test
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _the_repository_is_not_writable_from_a_test(tmp_path_factory):
    """Point the two default output paths at scratch space for the whole run.

    `data/` and `records.json` are gitignored, so a write to either is invisible
    in a diff: nothing would show up in a commit, and the leaderboard would
    simply be wrong afterwards. The guard above *detects* that by comparing
    snapshots; this makes it impossible instead, which is the stronger property
    and the one that does not depend on catching it.

    Two mechanisms rather than one, because the two defaults are bound
    differently:

      * `Records.__init__` takes `path=RECORDS_PATH` as a **default argument**,
        evaluated when the function is defined. Rebinding `records.RECORDS_PATH`
        at runtime would not move the file - the trap this very repository
        documents in `seats.build_brain`. So the default tuple itself is
        replaced. That also composes with the tests that redirect the leaderboard
        by swapping `__init__` wholesale, because `monkeypatch` puts the original
        function back and the patched defaults ride along on it.
      * `gamelog.LOG_DIR` is a module constant read inside
        `default_log_path`, so rebinding it is enough.

    The session guard above is left in place. Detection and prevention are not
    the same tool, and the guard also covers a human playing while the suite
    runs - which this fixture cannot prevent, because that write is not a test's.
    """
    scratch = tmp_path_factory.mktemp("not-the-repository")
    log_dir = scratch / "humanlog"
    log_dir.mkdir()
    leaderboard = scratch / "records.json"

    original_defaults = records.Records.__init__.__defaults__
    original_log_dir = gamelog.LOG_DIR
    records.Records.__init__.__defaults__ = (str(leaderboard),)
    gamelog.LOG_DIR = str(log_dir)
    try:
        yield
    finally:
        records.Records.__init__.__defaults__ = original_defaults
        gamelog.LOG_DIR = original_log_dir
