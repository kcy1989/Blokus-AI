"""The repository cannot be written to from a test run.

`records.json` is gitignored, so a test that writes to it leaves no trace in a
diff. Nothing would show up in a commit, the leaderboard would simply be wrong
afterwards, and the damage would be found - if at all - much later, by a number
that does not reproduce.

Two layers deal with this and they are not the same layer:

  * the session fixture in `conftest.py` **redirects** the default leaderboard
    path to scratch space, so the write cannot happen;
  * the snapshot guard in `conftest.py` **detects** a change to the watched
    paths afterwards, which also covers a write the fixture cannot prevent -
    a person playing a game, or an evaluation started without `--dry`, while
    the suite runs.

The first is what this file tests. The second is `tests/test_data_guard.py`.
"""
import os

import records


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _inside_repo(path):
    root = _repo_root()
    return os.path.abspath(path) == root or \
        os.path.abspath(path).startswith(root + os.sep)


def test_a_default_leaderboard_is_not_the_repositorys():
    """`Records()` with no argument must not be the repository's own file.

    The interesting part is *how* the redirection is done. `path=RECORDS_PATH` is
    a default argument, evaluated once when `Records.__init__` is defined, so
    rebinding `records.RECORDS_PATH` would leave the default pointing at the
    repository and this test would pass or fail for the wrong reason. The
    assertion below is on the object the default argument actually produces,
    which is the only thing that can tell the two apart.
    """
    r = records.Records()
    assert not _inside_repo(r.path), r.path


def test_the_sandbox_is_actually_used(tmp_path_factory):
    """The redirection is real, not merely "not the repository".

    A fixture that pointed the default at a second in-repository file would
    satisfy the test above while still writing into a checkout. It has to be
    under the pytest tmp directory, which is outside by construction.
    """
    r = records.Records()
    # The shipped default really is the repository's own file, so the test above
    # is not passing because there was never anything to redirect.
    shipped = os.path.join(os.path.dirname(os.path.abspath(records.__file__)),
                           "records.json")
    assert _inside_repo(shipped)
    assert records.Records.__init__.__defaults__ != (shipped,), \
        "the fixture must have replaced the default argument"
    # and the replacement is pytest's own scratch, which is outside a checkout
    # by construction rather than by a path that happens to look outside
    base = os.path.abspath(str(tmp_path_factory.getbasetemp()))
    assert not _inside_repo(base)
    assert os.path.abspath(r.path).startswith(base + os.sep), r.path


def test_an_explicit_path_still_wins(tmp_path):
    """Only the *default* moved. Naming a file is still how you choose one.

    Without this, the fixture would look like a change of behaviour rather than a
    change of default, and every test that redirects the leaderboard by passing
    a path would be relying on something the fixture had quietly taken away.
    """
    chosen = tmp_path / "mine.json"
    chosen.write_text("{}", encoding="utf-8")
    r = records.Records(str(chosen))
    r.record([("a", 0), ("b", 5), ("c", 9), ("d", 12)])
    assert chosen.exists()
    import json
    assert json.loads(chosen.read_text(encoding="utf-8"))["a"]["games"] == 1
