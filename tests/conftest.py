import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # GUI tests need no display

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import make_video  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Slow tests. tests/slow_tests.txt lists the tests that took long in a measured full run (real SAM 2 checkpoint, worker
# processes that import torch, real encodes that are decoded again, ...). They get the marker "slow":
#
#     python -m pytest -m "not slow"          quick check while working
#     python -m pytest                        everything (do this before a push)
#     python -m pytest --record-slow 3        everything, then rewrite the list with the tests that took >= 3 s
#
# A test that is not in the list counts as fast, so re-record after adding slow tests. The list is a convenience for the
# quick run only; it never removes a test from the full run.
SLOW_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slow_tests.txt")
_durations: dict = {}


def pytest_addoption(parser):
    parser.addoption("--record-slow", type=float, default=None, metavar="SECONDS",
                     help="after the run, rewrite tests/slow_tests.txt with the tests that took at least SECONDS (run the whole suite)")


def _slow_ids() -> set:
    try:
        with open(SLOW_LIST, encoding="utf-8") as f:
            return {ln.split("  #")[0].strip() for ln in f if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()


def pytest_collection_modifyitems(config, items):
    slow = _slow_ids()
    for item in items:
        if item.nodeid.replace("\\", "/") in slow:
            item.add_marker(pytest.mark.slow)


def pytest_runtest_logreport(report):
    _durations[report.nodeid.replace("\\", "/")] = _durations.get(report.nodeid.replace("\\", "/"), 0.0) + report.duration


def pytest_sessionfinish(session, exitstatus):
    limit = session.config.getoption("--record-slow")
    if limit is None:
        return
    if session.config.getoption("-m") or session.config.getoption("-k") or len(_durations) < 50:
        print("\n--record-slow ignored: it needs a run of the whole suite (no -m / -k / single files)")
        return
    slow = sorted(((d, n) for n, d in _durations.items() if d >= limit), reverse=True)
    total, slow_total = sum(_durations.values()), sum(d for d, _ in slow)
    with open(SLOW_LIST, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Tests that took >= {limit:g} s in a full run (setup + call + teardown). Written by: python -m pytest --record-slow {limit:g}\n"
                f"# {len(slow)} of {len(_durations)} tests, {slow_total:.0f} of {total:.0f} s. Quick run: python -m pytest -m \"not slow\"\n")
        for d, n in slow:
            f.write(f"{n}  # {d:.1f} s\n")
    print(f"\nwrote {SLOW_LIST}: {len(slow)} of {len(_durations)} tests are slow ({slow_total:.0f} of {total:.0f} s)")


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("vid") / "synthetic.mp4")
    make_video(p)
    return p


@pytest.fixture(scope="session")
def video_noaudio(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("vid") / "synthetic_noaudio.mp4")
    make_video(p, audio=False)
    return p
