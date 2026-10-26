"""Desktop GUI tests on Qt's offscreen platform (real widgets, real FFmpeg frames, no display needed)."""
import os
import shutil
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from helpers import W, H, read_frame_code  # noqa: E402
from src.app.frame_server import FrameServer  # noqa: E402
from src.app.geometry import fit_rect  # noqa: E402
from src.app.main_window import MainWindow, default_points_path  # noqa: E402
from src.points_io import load_points  # noqa: E402
from src.video_io import probe_video  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def shown_code(win):
    img = win.view._image
    assert img is not None
    img = img.convertToFormat(QImage.Format.Format_BGR888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    return read_frame_code(arr)


@pytest.fixture()
def win(qapp, video, tmp_path):
    v = str(tmp_path / "clip.mp4")
    shutil.copy(video, v)
    w = MainWindow()
    w.show()
    w.test_video = v
    w.errors = []
    w.show_error = lambda text: w.errors.append(text)
    w.ask_save_changes = lambda: "discard"
    w.ask_confirm = lambda text: True
    yield w
    w.dirty = False
    w.close()
    QApplication.processEvents()


def open_clip(w, path=None):
    w.open_video(path or w.test_video)
    assert pump(lambda: w.info is not None and w.view._image is not None and w.isEnabled())


def test_starts_without_video(win):
    assert win.info is None and "no video" in win.pos_label.text()
    assert not win.slider.isEnabled() and all(not b.isEnabled() for b in win._nav_buttons)
    win.step(1)
    win.toggle_play()                       # must be harmless
    win.view.handle_click(10, 10, 1)
    assert win.store.points == []


def test_open_video_shows_first_frame(win):
    open_clip(win)
    assert (win.info.width, win.info.height, win.info.nb_frames) == (W, H, 160)
    assert win.slider.maximum() == 159 and "frame 0/159" in win.pos_label.text()
    assert shown_code(win) == 0
    assert "640x360" in win.info_label.text() and win.slider.isEnabled()


def test_stepping_and_jumping_show_the_right_frame(win):
    open_clip(win)
    for _ in range(3):
        win.step(1)
    assert pump(lambda: shown_code(win) == 3) and win.frame_idx == 3
    win.step(-2)                                              # frame 1 was coalesced away earlier: loaded on demand
    assert pump(lambda: shown_code(win) == 1) and win.slider.value() == 1
    win.step(1)
    assert pump(lambda: shown_code(win) == 2)
    win.step(-1)                                              # frame 1 is cached now: shown at once, no worker round trip
    assert shown_code(win) == 1
    win.goto(100)                                             # jump = seek
    assert pump(lambda: shown_code(win) == 100)
    win.step_seconds(1)
    assert win.frame_idx == 130 and pump(lambda: shown_code(win) == 130)
    win.slider.setValue(37)                                   # slider drives the same path
    assert pump(lambda: shown_code(win) == 37) and "frame 37/159" in win.pos_label.text()
    win.goto(-50)
    assert win.frame_idx == 0
    win.goto(10_000)
    assert win.frame_idx == 159 and pump(lambda: shown_code(win) == 159)
    assert "t = " in win.pos_label.text()


def test_playback_advances_and_stops(win):
    open_clip(win)
    win.toggle_play()
    assert win.playing and win.play_btn.text() == "Pause"
    assert pump(lambda: win.frame_idx > 15)
    win.toggle_play()
    assert not win.playing
    stopped_at = win.frame_idx
    pump(lambda: False, 0.3)
    assert win.frame_idx == stopped_at
    assert pump(lambda: shown_code(win) == win.frame_idx)      # what is shown is the frame the counter says


def test_click_maps_to_frame_coordinates_with_letterbox(win):
    open_clip(win)
    win.resize(1100, 900)
    QApplication.processEvents()
    win.goto(12)
    assert pump(lambda: shown_code(win) == 12)
    v = win.view
    fit = fit_rect(v.width(), v.height(), W, H)
    assert fit.oy > 5 or fit.ox > 5                           # there is a letterbox in this window size
    wx, wy = fit.ox + 320 * fit.scale, fit.oy + 180 * fit.scale
    assert v.handle_click(wx, wy, 1)
    assert v.handle_click(fit.ox + 100 * fit.scale, fit.oy + 50 * fit.scale, 0)
    p0, p1 = win.store.points
    assert (p0["frame_idx"], p0["label"]) == (12, 1) and (p0["x"], p0["y"]) == pytest.approx((320, 180), abs=0.01)
    assert p1["label"] == 0 and (p1["x"], p1["y"]) == pytest.approx((100, 50), abs=0.01)
    bar_x = fit.ox - 2 if fit.ox > 2 else v.width() / 2
    bar_y = fit.oy - 2 if fit.oy > 2 else None
    assert not v.handle_click(bar_x if bar_y is None else v.width() / 2, bar_y if bar_y is not None else v.height() / 2, 1)
    assert len(win.store.points) == 2 and win.dirty and win.windowTitle().startswith("* ")
    # resizing the window does not move saved points
    win.resize(700, 500)
    QApplication.processEvents()
    fit2 = fit_rect(v.width(), v.height(), W, H)
    assert v.handle_click(fit2.ox + 320 * fit2.scale, fit2.oy + 180 * fit2.scale, 1)
    assert win.store.points[-1]["x"] == pytest.approx(320, abs=0.01)


def test_save_and_reload_points(win, tmp_path):
    open_clip(win)
    win.goto(20)
    pump(lambda: shown_code(win) == 20)
    win.view.handle_click(*[c for c in _widget_pos(win, 200, 150)], 1)
    win.new_object()
    win.goto(40)
    win.view.handle_click(*_widget_pos(win, 500, 100), 1)
    win.view.handle_click(*_widget_pos(win, 50, 300), 0)
    assert win.save_points() and not win.dirty
    path = default_points_path(win.test_video)
    assert os.path.isfile(path)
    prompts = load_points(path, probe_video(win.test_video))
    assert prompts[1][20] == ([(pytest.approx(200, abs=0.01), pytest.approx(150, abs=0.01))], [1])
    assert prompts[2][40][1] == [1, 0]

    w2 = MainWindow()
    w2.show_error = lambda t: None
    try:
        open_clip(w2, win.test_video)
        assert len(w2.store.points) == 3 and not w2.dirty                   # auto-loaded
        assert [o for o in w2.store.objects] == [1, 2]
        assert "loaded 3 points" in w2.statusBar().currentMessage()
    finally:
        w2.close()


def _widget_pos(win, fx, fy):
    f = fit_rect(win.view.width(), win.view.height(), W, H)
    return f.ox + fx * f.scale, f.oy + fy * f.scale


def test_points_of_another_file_are_ignored(win, video_noaudio, tmp_path):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)
    assert win.save_points()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = str(other_dir / "clip.mp4")
    shutil.copy(video_noaudio, other)
    shutil.copy(default_points_path(win.test_video), default_points_path(other))   # points for a different file
    w2 = MainWindow()
    w2.show_error = lambda t: None
    try:
        open_clip(w2, other)
        assert w2.store.points == [] and "ignored" in w2.statusBar().currentMessage()
    finally:
        w2.close()


def test_negative_only_points_are_not_saved(win):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 100, 100), 0)
    assert win.save_points() is False
    assert win.errors and "only negative" in win.errors[0]
    assert not os.path.exists(default_points_path(win.test_video)) and win.dirty


def test_undo_clear_objects(win):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 10, 10), 1)
    win.view.handle_click(*_widget_pos(win, 20, 20), 1)
    win.undo_point()
    assert len(win.store.points) == 1
    win.clear_frame_points()
    assert win.store.points == []
    win.view.handle_click(*_widget_pos(win, 10, 10), 1)
    win.new_object()
    assert win.store.current == 2 and win.obj_list.count() == 2
    win.view.handle_click(*_widget_pos(win, 30, 30), 1)
    win.delete_object()                                      # confirm stubbed to True -> removes object 2
    assert {p["obj_id"] for p in win.store.points} == {1}


def test_closing_with_unsaved_points_can_be_cancelled(win):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 10, 10), 1)
    win.ask_save_changes = lambda: "cancel"
    assert win.close() is False and win.dirty
    win.ask_save_changes = lambda: "save"
    assert win.close() is True
    assert os.path.isfile(default_points_path(win.test_video))


def test_unreadable_video_reports_error_and_recovers(win, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    win.open_video(str(bad))
    assert pump(lambda: bool(win.errors))
    assert win.isEnabled() and win.info is None
    open_clip(win)                                          # still usable afterwards
    assert shown_code(win) == 0


def test_frame_server_rejects_out_of_range(qapp, video):
    info = probe_video(video)
    srv = FrameServer(info)
    got = []
    srv.failed.connect(lambda i, m: got.append((i, m)))
    try:
        srv.request(9999)
        assert pump(lambda: bool(got)) and got[0][0] == 9999
    finally:
        srv.close()
