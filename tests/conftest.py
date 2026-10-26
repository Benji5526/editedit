import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # GUI tests need no display

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import make_video  # noqa: E402


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
