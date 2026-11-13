"""Detections marked "not a target": the file, the matching, building an exclusion from a clicked box, and the pipeline leaving
those boxes out when only the mosaic is rendered again (no new detection run)."""
import json
import os

import numpy as np
import pytest

from src.detection import Det
from src.exclusions import exclusion_for, is_excluded, load_exclusions, save_exclusions
from src.pipeline import Config, Pipeline, config_hash, render_hash
from src.video_io import FrameReader, probe_video

CS = 40


def test_file_roundtrip_and_matching(tmp_path):
    p = str(tmp_path / "x" / "e.json")
    save_exclusions(p, "fp", [{"frame_start": 10, "frame_end": 20, "box": [100, 100, 200, 200], "cls": "A"}])
    ex = load_exclusions(p, "fp")
    assert ex == [{"frame_start": 10, "frame_end": 20, "box": [100.0, 100.0, 200.0, 200.0], "cls": "A"}]
    assert "fp" in open(p, encoding="utf-8").read() and str(tmp_path) not in open(p, encoding="utf-8").read()
    with pytest.raises(ValueError):
        load_exclusions(p, "other video")
    assert is_excluded(ex, 15, (120, 120, 160, 160), "A") and not is_excluded(ex, 21, (120, 120, 160, 160), "A")
    assert not is_excluded(ex, 15, (300, 300, 340, 340), "A") and not is_excluded(ex, 15, (120, 120, 160, 160), "B")
    assert is_excluded(ex, 15, (120, 120, 160, 160))                       # a region without a class: by place only
    with pytest.raises(ValueError):
        save_exclusions(p, "fp", [{"frame_start": 5, "frame_end": 1, "box": [0, 0, 1, 1]}])


def test_exclusion_follows_the_clicked_box_through_a_flickering_track():
    def boxes_at(f):                                                       # a slowly moving box, missing on frames 30-49
        if 10 <= f <= 80 and not (30 <= f < 50):
            return [(100 + f, 100, 160 + f, 150, "A"), (500, 300, 560, 350, "A")]
        return [(500, 300, 560, 350, "A")] if f <= 90 else []
    ex = exclusion_for(boxes_at, 60, 190, 120, 200)
    assert (ex["frame_start"], ex["frame_end"], ex["cls"]) == (10, 80, "A")
    assert ex["box"][0] < 110 and ex["box"][2] > 240 and ex["box"][2] < 500     # covers the whole path, not the other box
    assert exclusion_for(boxes_at, 60, 20, 20, 200) is None                    # clicked beside every box
    other = exclusion_for(boxes_at, 60, 530, 320, 200)
    assert (other["frame_start"], other["frame_end"]) == (0, 90)


class TwoBoxes:
    def detect(self, bgr):
        return [Det(200.0, 20.0, 240.0, 50.0, 0.9, "FAKE"), Det(40.0, 100.0, 90.0, 150.0, 0.9, "FAKE")]


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def cfg_for(tmp, video, name, **extra):
    return Config(input=video, output=str(tmp / name), points=None, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                  chunk_size=CS, analysis_width=320, preset="ultrafast", crf=18, fail_closed=True, block=32,
                  extra={"keep_temp": True, **extra})


def test_excluded_box_is_left_out_by_a_render_only_rerun(video, tmp_path):
    src = frames(video)
    a = cfg_for(tmp_path, video, "a.mp4")
    assert Pipeline(a, None, TwoBoxes()).run() == 0
    before = frames(a.output)
    # analysis frames are 320 wide, the video 640: the first box is at (400..480, 40..100) in full-resolution pixels
    ex = [{"frame_start": 0, "frame_end": 79, "box": [380, 20, 500, 120], "cls": "FAKE"}]
    b = cfg_for(tmp_path, video, "a.mp4", det_exclusions=ex)
    b.resume = True
    assert config_hash(a, probe_video(video)) == config_hash(b, probe_video(video)) and render_hash(a) != render_hash(b)

    class Never:
        def detect(self, bgr):
            raise AssertionError("the detector must not run again")
    assert Pipeline(b, None, Never()).run() == 0
    after = frames(b.output)

    def diff(x, y, box):
        x1, y1, x2, y2 = box
        return float(np.abs(x[y1:y2, x1:x2].astype(int) - y[y1:y2, x1:x2].astype(int)).mean())
    first, second = (405, 45, 475, 95), (90, 210, 170, 290)
    assert diff(before[40], src[40], first) > 3 and diff(before[120], src[120], first) > 3     # was mosaicked
    assert diff(after[40], src[40], first) < 3                                                 # excluded range: original picture
    assert diff(after[120], src[120], first) > 3                                               # after the range: still mosaicked
    assert diff(after[40], src[40], second) > 3                                                # the other box is untouched


# ---- desktop app: click a wrongly detected box of the last run
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from src.app.job_runner import ProcessSettings, build_args  # noqa: E402
from test_gui import open_clip, qapp, win  # noqa: E402,F401


class FakeMasks:
    created, points_sha = "now", None

    def frame(self, f):                                    # analysis size 320x180 for the 640x360 clip
        return {"shape": (180, 320), "masks": {}, "boxes": [(200.0, 20.0, 240.0, 50.0, "FAKE", False)] if f < 100 else [],
                "fc_regions": [], "fc_full": False}


def test_marking_a_box_in_the_app_writes_the_file_and_is_passed_to_the_job(win, tmp_path):
    win.last_settings = ProcessSettings(output=str(tmp_path / "o.mp4"), log_dir=str(tmp_path / "logs"), work_dir=str(tmp_path / "work"))
    open_clip(win)
    win.act_exclude.setChecked(True)                       # no kept run: refused with a message
    assert not win.act_exclude.isChecked() and "지난 실행" in win.statusBar().currentMessage()
    win._job_masks = FakeMasks()
    win._set_checked(win.act_last, True)
    win.act_exclude.setChecked(True)
    assert win.act_exclude.isChecked()
    win.goto(30)
    n_points = len(win.store.points)
    win.on_click(10, 10, 1)                                # beside the box
    assert win.exclusions == [] and "탐지 상자가 없습니다" in win.statusBar().currentMessage()
    win.on_click(440, 70, 1)                               # inside the box (full-resolution pixels)
    assert len(win.store.points) == n_points and not win.dirty          # no click point was added
    (ex,) = win.exclusions
    assert (ex["frame_start"], ex["frame_end"], ex["cls"]) == (0, 99, "FAKE") and ex["box"][0] < 400 and ex["box"][2] > 480
    path = win._exclusions_file()
    assert os.path.isfile(path) and os.path.basename(path) == win.info.fingerprint + ".json"
    assert "clip" not in open(path, encoding="utf-8").read()                # no video name in the record
    st = win.last_settings
    st.exclusions = path
    a = build_args(win.info.path, None, st, "c.flag")
    assert a[a.index("--exclusions") + 1] == path
    win.on_click(440, 70, 0)                               # right click takes it back
    assert win.exclusions == [] and json.load(open(path, encoding="utf-8"))["exclusions"] == []
