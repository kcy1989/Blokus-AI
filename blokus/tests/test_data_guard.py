"""The data/ guard itself: does `snapshot` see what it is supposed to see?

Every test here builds its own tree under `tmp_path` and points the guard at it
with `project_root=`. Nothing reads or writes the repository's real `data/` -
which is the point of making the root a parameter, and the reason this file
exists rather than a line in conftest.py: a guard that can only be exercised
against the thing it guards cannot be tested without writing to it.

The four detection cases the guard has to catch are additions, removals,
in-place rewrites that keep the size, and renames. The third is why the snapshot
carries `mtime_ns` and not just the size, and it is the one a size-only check
would pass straight through.
"""
import os

from conftest import (GUARD_EXCLUDED, diff_snapshots, guard_roots,
                      render_changes, snapshot, snapshot_roots)


def write(path, data=b"x", mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))
    return path


# --------------------------------------------------------- snapshot itself

def test_snapshot_is_empty_for_a_missing_root(tmp_path):
    assert snapshot(str(tmp_path / "nope")) == {}


def test_snapshot_is_empty_for_an_empty_directory(tmp_path):
    (tmp_path / "d").mkdir()
    assert snapshot(str(tmp_path / "d")) == {}


def test_snapshot_keys_are_relative_and_values_are_size_and_mtime(tmp_path):
    write(str(tmp_path / "d" / "sub" / "a.bin"), b"abc", mtime=1_700_000_000_000_000_000)
    snap = snapshot(str(tmp_path / "d"))
    assert list(snap) == [os.path.join("sub", "a.bin")]
    size, mtime_ns = snap[os.path.join("sub", "a.bin")]
    assert size == 3
    assert mtime_ns == 1_700_000_000_000_000_000


def test_snapshot_walks_recursively(tmp_path):
    for i in range(3):
        write(str(tmp_path / "d" / ("deep%d" % i) / "f.bin"), b"y" * (i + 1))
    assert len(snapshot(str(tmp_path / "d"))) == 3


def test_snapshot_reads_nothing_and_returns_a_fresh_dict(tmp_path):
    write(str(tmp_path / "d" / "a.bin"), b"abc")
    root = str(tmp_path / "d")
    first = snapshot(root)
    first["injected"] = (0, 0)
    assert "injected" not in snapshot(root)


def test_snapshot_does_not_report_directories(tmp_path):
    """A directory's mtime moves when anything is added beneath it, so
    including directories would report a whole dataset as changed when one
    shard appeared."""
    write(str(tmp_path / "d" / "a.bin"), b"abc", mtime=1_000_000_000_000_000_000)
    before = snapshot(str(tmp_path / "d"))
    write(str(tmp_path / "d" / "b.bin"), b"abc")
    after = snapshot(str(tmp_path / "d"))
    assert diff_snapshots(before, after) == [
        ("added", "b.bin", "3 bytes")]


# --------------------------------------------------------- what it detects

def test_it_detects_an_added_file(tmp_path):
    write(str(tmp_path / "d" / "a.bin"), b"abc")
    before = snapshot(str(tmp_path / "d"))
    write(str(tmp_path / "d" / "new.bin"), b"hello")
    changes = diff_snapshots(before, snapshot(str(tmp_path / "d")))
    assert changes == [("added", "new.bin", "5 bytes")]


def test_it_detects_a_removed_file(tmp_path):
    write(str(tmp_path / "d" / "a.bin"), b"abc")
    write(str(tmp_path / "d" / "b.bin"), b"defg")
    before = snapshot(str(tmp_path / "d"))
    os.remove(str(tmp_path / "d" / "b.bin"))
    changes = diff_snapshots(before, snapshot(str(tmp_path / "d")))
    assert changes == [("removed", "b.bin", "4 bytes")]


def test_it_detects_an_in_place_rewrite_that_keeps_the_size(tmp_path):
    """The case a size-only snapshot would miss.

    `mtime_ns` is set explicitly to two values a nanosecond apart, because a
    rewrite fast enough to land inside one filesystem timestamp tick would
    legitimately look unchanged - which is the stated limit of the format, not
    something this test should pretend to beat.
    """
    p = write(str(tmp_path / "d" / "a.bin"), b"abc", mtime=1_700_000_000_000_000_000)
    before = snapshot(str(tmp_path / "d"))
    write(p, b"xyz", mtime=1_700_000_000_000_000_001)
    changes = diff_snapshots(before, snapshot(str(tmp_path / "d")))
    assert len(changes) == 1
    kind, path, detail = changes[0]
    assert kind == "rewrote"
    assert path == "a.bin"
    assert "size unchanged" in detail and "mtime_ns" in detail


def test_it_detects_a_size_changing_rewrite(tmp_path):
    p = write(str(tmp_path / "d" / "a.bin"), b"abc")
    before = snapshot(str(tmp_path / "d"))
    write(p, b"much longer contents")
    changes = diff_snapshots(before, snapshot(str(tmp_path / "d")))
    assert changes == [("rewrote", "a.bin", "3 -> 20 bytes")]


def test_it_detects_a_rename_as_a_removal_and_an_addition(tmp_path):
    write(str(tmp_path / "d" / "before.bin"), b"abc")
    before = snapshot(str(tmp_path / "d"))
    os.rename(str(tmp_path / "d" / "before.bin"),
              str(tmp_path / "d" / "after.bin"))
    changes = diff_snapshots(before, snapshot(str(tmp_path / "d")))
    assert changes == [("added", "after.bin", "3 bytes"),
                       ("removed", "before.bin", "3 bytes")]


def test_it_reports_nothing_when_nothing_moved(tmp_path):
    write(str(tmp_path / "d" / "a.bin"), b"abc")
    root = str(tmp_path / "d")
    assert diff_snapshots(snapshot(root), snapshot(root)) == []


# --------------------------------------------------------- which roots

def test_guard_roots_watches_the_dataset_directories_and_the_leaderboard(tmp_path):
    for name in ("h0", "h1", "hb1", "hb2", "hc1", "hc2"):
        (tmp_path / "data" / name).mkdir(parents=True)
    write(str(tmp_path / "records.json"), b"{}")
    roots = guard_roots(str(tmp_path))
    for name in ("h0", "h1", "hb1", "hb2", "hc1", "hc2"):
        assert os.path.join("data", name) in roots
    assert "records.json" in roots


def test_guard_roots_excludes_the_applications_own_output_directories(tmp_path):
    """`data/humanlog` matches `h*` and would otherwise be swept in.

    A human game appends there and an evaluation run drops a batch into
    `data/match`; watching either would make the guard fail on ordinary use of
    the application rather than on a test that writes where it should not.
    """
    (tmp_path / "data" / "h0").mkdir(parents=True)
    (tmp_path / "data" / "humanlog").mkdir(parents=True)
    (tmp_path / "data" / "match").mkdir(parents=True)
    roots = guard_roots(str(tmp_path))
    assert os.path.join("data", "humanlog") not in roots
    assert os.path.join("data", "match") not in roots
    assert os.path.join("data", "h0") in roots
    for excluded in GUARD_EXCLUDED:
        assert not any(r.endswith(excluded) for r in roots)


def test_guard_roots_is_empty_on_a_checkout_without_data(tmp_path):
    assert guard_roots(str(tmp_path)) == ()


def test_guard_roots_omits_a_leaderboard_that_is_not_there(tmp_path):
    (tmp_path / "data" / "h0").mkdir(parents=True)
    assert "records.json" not in guard_roots(str(tmp_path))


# --------------------------------------------------------- several roots

def test_snapshot_roots_keys_by_project_relative_path(tmp_path):
    """Two roots holding the same relative name must not collapse into one
    entry, or half the comparison is vacuous."""
    for root in ("data/hb1", "data/hc1"):
        write(str(tmp_path / root / "manifest.json"), b"{}")
    write(str(tmp_path / "records.json"), b"{}")
    snap = snapshot_roots(("data/hb1", "data/hc1", "records.json"),
                          str(tmp_path))
    assert sorted(snap) == [
        os.path.join("data", "hb1", "manifest.json"),
        os.path.join("data", "hc1", "manifest.json"),
        "records.json",
    ]


def test_a_change_in_any_root_shows_up(tmp_path):
    for root in ("data/hb1", "data/hc1"):
        write(str(tmp_path / root / "manifest.json"), b"{}")
    roots = ("data/hb1", "data/hc1")
    before = snapshot_roots(roots, str(tmp_path))
    write(str(tmp_path / "data" / "hc1" / "manifest.json"), b'{"a": 1}')
    changes = diff_snapshots(before, snapshot_roots(roots, str(tmp_path)))
    assert len(changes) == 1
    assert changes[0][1] == os.path.join("data", "hc1", "manifest.json")


# --------------------------------------------------------- the message

def test_the_failure_message_names_every_change(tmp_path):
    changes = [("added", os.path.join("data", "hb1", "x.npz"), "5 bytes"),
               ("removed", os.path.join("data", "hc1", "y.npz"), "7 bytes")]
    text = render_changes(changes)
    assert os.path.join("data", "hb1", "x.npz") in text
    assert os.path.join("data", "hc1", "y.npz") in text
    assert "added" in text and "removed" in text


def test_the_failure_message_explains_a_records_json_change(tmp_path):
    """The documented false positive has to be explained where it is seen, or
    the guard just looks broken."""
    text = render_changes([("rewrote", "records.json", "1 -> 2 bytes")])
    assert "records.json" in text
    assert "--dry" in text


def test_the_failure_message_stays_quiet_about_data_changes(tmp_path):
    text = render_changes([("added", os.path.join("data", "hb1", "x.npz"),
                            "5 bytes")])
    assert "without --dry" not in text


# ------------------------------------------------------------- xdist

def test_workers_are_told_apart_from_the_controller():
    """xdist puts `workerinput` on every worker config and none on the
    controller's; that attribute is the only thing separating them here.

    A worker that took its own snapshot would compare it against a
    controller-side view that has already moved, and report every guarded file
    as changed.
    """
    from conftest import _is_worker

    class WorkerConfig:
        workerinput = {"workerid": "/gw0"}

    class ControllerConfig:
        pass

    assert _is_worker(WorkerConfig()) is True
    assert _is_worker(ControllerConfig()) is False