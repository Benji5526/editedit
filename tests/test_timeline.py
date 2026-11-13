"""Timeline: the track widget's geometry and data handling, the per-frame summary of a kept run, and the wiring in the app."""
import os

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent

from src.app.timeline import GUTTER, MAX_OBJECT_ROWS, ThumbnailLoader, TimelineView, format_time, nice_step, runs_of
from src.mask_cache import JobMasks
from src.video_io import probe_video
from test_gui import _widget_pos, open_clip, pump, qapp, shown_code, win  # noqa: F401  (fixtures)
from test_mask_preview import run_job


# ------------------------------------------------------------------ pure helpers
def test_runs_of():
    assert runs_of([]) == [] and runs_of([False, False]) == []
    assert runs_of([True, True, False, True]) == [(0, 2), (3, 4)]
    assert runs_of(np.array([0, 1, 1, 1, 0, 0, 1], bool)) == [(1, 4), (6, 7)]
    assert runs_of([True] * 5) == [(0, 5)]


def test_ruler_steps_and_time_format():
    assert nice_step(0.3) == 0.5 and nice_step(1) == 1 and nice_step(7) == 10 and nice_step(100) == 120 and nice_step(99999) == 7200
    assert format_time(0) == "0:00" and format_time(75) == "1:15" and format_time(3725) == "1:02:05" and format_time(2.5, True) == "0:02.5"


# ------------------------------------------------------------------ the widget
def test_frame_position_mapping_and_seek(qapp):
    tv = TimelineView()
    tv.resize(1000, 200)
    tv.set_video(161, 30.0)
    assert tv.frame_at(tv.x_of(0)) == 0 and tv.frame_at(tv.x_of(160)) == 160 and tv.frame_at(tv.x_of(77)) == 77
    assert tv.frame_at(-50) == 0 and tv.frame_at(5000) == 160                     # clamped to the video
    assert tv.x_of(0) == GUTTER
    got = []
    tv.seek_requested.connect(got.append)

    def click(x):
        tv.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(x, 40), QPointF(x, 40), Qt.MouseButton.LeftButton,
                                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    click(tv.x_of(100))
    assert got == [100]
    tv.setEnabled(False)                                                          # no video / locked: clicks do nothing
    click(tv.x_of(20))
    assert got == [100]
    empty = TimelineView()
    empty.resize(600, 100)
    empty.seek_requested.connect(got.append)
    assert empty.frame_at(300) == 0 and not empty.grab().isNull()                 # paints its hint without a video


def test_rows_follow_the_data(qapp):
    tv = TimelineView()
    tv.resize(900, 200)
    tv.set_video(160, 30.0)
    assert tv.rows() == []
    h0 = tv.height()
    tv.set_objects([(1, QColor("#ffc800"), [12]), (2, QColor("#50ff50"), [95, 120])])
    assert [k for k, _ in tv.rows()] == ["object:1", "object:2"] and tv.height() > h0
    tv.set_run_data({"masks": {1: np.r_[np.zeros(12, bool), np.ones(58, bool), np.zeros(90, bool)]},
                     "detections": np.zeros(160, bool), "fail_closed": np.r_[np.zeros(70, bool), np.ones(10, bool), np.zeros(80, bool)]})
    assert tv._presence == {1: [(12, 70)]} and tv._fc_runs == [(70, 80)] and tv._det_runs == []
    assert [k for k, _ in tv.rows()] == ["object:1", "object:2", "detections", "fail_closed"]
    tv.set_flags([(70, 85, "warn")])
    assert tv.rows()[-1][0] == "flags"
    tv.set_frame(60)
    assert not tv.grab().isNull()                                                 # everything paints
    tv.set_run_data(None)
    assert [k for k, _ in tv.rows()] == ["object:1", "object:2", "flags"] and tv._presence == {}
    tv.set_objects([(i, QColor("#888888"), []) for i in range(1, MAX_OBJECT_ROWS + 4)])
    kinds = [k for k, _ in tv.rows()]
    assert kinds.count("more") == 1 and sum(k.startswith("object:") for k in kinds) == MAX_OBJECT_ROWS
    tv.set_video(10, 25.0)                                                        # a new video clears run data and flags
    assert tv._flags == [] and not tv._has_run_data


def test_thumbnail_loader_reads_small_frames(qapp, video):
    info = probe_video(video)
    loader = ThumbnailLoader(info, count=4, height=40)
    got = []
    loader.signals.done.connect(lambda path, images: got.append((path, images)))
    loader.run()                                                                  # synchronously, on this thread
    assert got and got[0][0] == info.path and len(got[0][1]) == 4
    imgs = [i for i in got[0][1] if i is not None]
    assert len(imgs) == 4 and imgs[0].height() == 40 and 68 <= imgs[0].width() <= 72      # 640x360 scaled to 40 high
    cancelled = ThumbnailLoader(info, count=4)
    cancelled.cancelled = True
    cancelled.signals.done.connect(lambda *a: got.append("late"))
    cancelled.run()
    assert "late" not in got                                                      # a cancelled loader stays silent


# ------------------------------------------------------------------ summary of a kept run
def test_job_summary_matches_the_per_frame_view(video, tmp_path):
    jd, _ = run_job(video, tmp_path)
    jm = JobMasks(jd)
    s = jm.summary()
    assert set(s["masks"]) == {1} and s["masks"][1].shape == (160,) and s["detections"].all()
    for f in (0, 5, 12, 30, 69, 70, 75, 85, 86, 130, 159):                        # same answer as unpacking that frame's mask
        fr = jm.frame(f)
        assert bool(s["masks"][1][f]) == (1 in fr["masks"]), f
        assert bool(s["fail_closed"][f]) == (bool(fr["fc_regions"]) or fr["fc_full"]), f
    assert not s["masks"][1][:12].any() and s["masks"][1][20:60].all()            # before the click / while visible
    assert not s["masks"][1][70:86].any() and s["fail_closed"][70:80].all()       # hidden: no mask, fail-closed instead
    os.remove(os.path.join(jd, "masks_00002.npz"))                                # a missing chunk simply stays empty
    s2 = JobMasks(jd).summary()
    assert not s2["masks"][1][80:120].any() and not s2["detections"][80:120].any() and s2["masks"][1][120:].any()
    assert JobMasks(jd).summary(should_stop=lambda: True)["masks"] == {}          # stopped before the first chunk


# ------------------------------------------------------------------ the app
def test_app_timeline_follows_frames_clicks_flags_and_runs(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    tv = win.timeline_view
    assert not tv.isEnabled() and tv.parent() is win.timeline
    open_clip(win)
    assert tv.isEnabled() and tv._n == 160
    assert pump(lambda: len(tv._thumbs) == 16, timeout=60)                        # thumbnails arrive from the background
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    assert tv._frame == 30
    tv.seek_requested.emit(90)                                                    # clicking the timeline moves the preview
    assert win.frame_idx == 90 and pump(lambda: shown_code(win) == 90)
    win.view.handle_click(*_widget_pos(win, 300, 150), 1)
    win.new_object()
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)
    assert [(o, fr) for o, _, fr in tv._objects] == [(1, [90]), (2, [90])]        # one track per object with its clicked frames
    assert not tv._has_run_data                                                   # nothing processed yet

    jd, pts = run_job(win.test_video, tmp_path)                                   # a kept run of this video
    win.dirty = False
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"))
    win._load_run_summary()
    assert pump(lambda: tv._has_run_data, timeout=60)
    assert tv._presence[1][0][0] == 12 and tv._fc_runs and tv._det_runs           # mask from the click on, fail-closed while hidden
    assert win.load_review(str(tmp_path / "logs" / "clip_review.json"), quiet=True)
    assert tv._flags and all(len(f) == 3 for f in tv._flags) and "flags" in [k for k, _ in tv.rows()]
    assert not win.grab().isNull()


def test_kept_chunks_can_be_deleted_while_the_timeline_shows_them(win, tmp_path):
    """The timeline reads the mask cache in the background; deleting the job folder must not be blocked by that (Windows keeps a
    folder alive while one of its files is open)."""
    from src.app.job_runner import ProcessSettings
    open_clip(win)
    jd, pts = run_job(win.test_video, tmp_path)
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"))
    win._try_load_points(pts, quiet=True)
    win.act_last.setChecked(True)
    for _ in range(5):                                                            # start the reader and delete right away, repeatedly
        win._load_run_summary()
        win.act_last.setChecked(True)
        win.delete_intermediate()
        assert not os.path.isdir(jd), os.listdir(jd)
        assert not win.timeline_view._has_run_data and not win.act_last.isChecked() and win._job_masks is None
        if _ < 4:
            jd, _pts = run_job(win.test_video, tmp_path)
