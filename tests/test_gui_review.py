"""Review screen and re-processing in the desktop app (real widgets, real worker process; real SAM 2 where noted)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from helpers import ColorTracker, W, square_x  # noqa: E402
from src.app.job_runner import PROJECT_ROOT, ProcessSettings  # noqa: E402
from src.pipeline import Config, Pipeline  # noqa: E402
from src.review_data import seg_key, state_path_for  # noqa: E402
from src.video_io import FrameReader, probe_video  # noqa: E402
from test_gui import _widget_pos, open_clip, pump, qapp, shown_code, win  # noqa: E402,F401

CKPT = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_small.pt")


def settings_for(tmp_path, **kw):
    base = dict(output=str(tmp_path / "out" / "result.mp4"), chunk_size=40, analysis_width=320, fail_closed=True,
                work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"), keep_chunks=True)
    base.update(kw)
    return ProcessSettings(**base)


def finish(win, timeout=180):
    return pump(lambda: not win.job_running and win.last_result is not None and not win.plan_runner.is_running(), timeout)


def make_review_with_stand_in(win, tmp_path):
    """Click the square on frame 20, save the points, run the in-process pipeline with the colour tracker, return the review path."""
    open_clip(win)
    win.goto(20)
    assert pump(lambda: shown_code(win) == 20)
    win.view.handle_click(*_widget_pos(win, square_x(20) + 30, 180), 1)
    assert win.save_points()
    cfg = Config(input=win.test_video, output=str(tmp_path / "o.mp4"), points=win.points_path, work_dir=str(tmp_path / "w"),
                 log_dir=str(tmp_path / "l"), chunk_size=40, analysis_width=320, preset="ultrafast", extra={"keep_temp": False})
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    return str(tmp_path / "l" / "clip_review.json")


def test_review_table_filters_navigation_and_reviewed_marks(win, tmp_path):
    path = make_review_with_stand_in(win, tmp_path)
    assert win.load_review(path)
    rp = win.review_panel
    types = [rp.table.item(r, 2).text() for r in range(rp.table.rowCount())]
    assert "tracking_lost_mid_frame" in types and "tracking_recovered" in types and "chunk_boundary_resumed" not in types   # info hidden
    assert "consistent" in rp.banner.text() and win.slider._flags                                # ticks on the slider
    rp.sev.setCurrentIndex(2)                                                                    # everything incl. info
    assert "chunk_boundary_resumed" in [rp.table.item(r, 2).text() for r in range(rp.table.rowCount())]
    rp.sev.setCurrentIndex(1)                                                                    # errors only
    assert rp.table.rowCount() == 0
    rp.sev.setCurrentIndex(0)

    row = [rp.table.item(r, 2).text() for r in range(rp.table.rowCount())].index("tracking_lost_mid_frame")
    rp.table.selectRow(row)                                                                      # selecting a flag jumps to its frame
    assert pump(lambda: win.frame_idx == 70 and shown_code(win) == 70)
    win.goto(0)
    win.next_flag()
    assert win.frame_idx == 70 and pump(lambda: shown_code(win) == 70) and rp.current_segment()["type"] == "tracking_lost_mid_frame"
    win.next_flag()
    assert win.frame_idx == 80                                                                   # boundary re-seed warning
    win.next_flag()
    assert win.frame_idx == 86
    win.prev_flag()
    assert win.frame_idx == 80
    win.prev_flag()
    assert win.frame_idx == 70
    win.goto(159)
    win.next_flag()
    assert "No more flags" in win.statusBar().currentMessage() and win.frame_idx == 159

    # mark the lost flag reviewed: sidecar file, table, ticks, 'hide reviewed'
    ticks_before = len(win.slider._flags)
    win.goto(70)
    rp.select_row_for(70)
    rp.toggle_reviewed()
    assert os.path.isfile(state_path_for(path)) and len(win.slider._flags) < ticks_before
    rp.hide_done.setChecked(True)
    assert "tracking_lost_mid_frame" not in [rp.table.item(r, 2).text() for r in range(rp.table.rowCount())]
    # reload: the mark is still there
    assert win.load_review(path)
    assert win.review.state.is_reviewed(next(s for s in win.review.segments if s["type"] == "tracking_lost_mid_frame"))


def test_staleness_banner_follows_point_edits(win, tmp_path):
    path = make_review_with_stand_in(win, tmp_path)
    win.load_review(path)
    assert "consistent" in win.review_panel.banner.text()
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)                                       # unsaved edit
    assert "unsaved point edits" in win.review_panel.banner.text()
    assert win.save_points()
    assert "points file changed" in win.review_panel.banner.text()
    win.undo_point()
    assert win.save_points()
    assert "consistent" in win.review_panel.banner.text()                                       # identical file again => consistent


def test_review_of_another_video_is_rejected(win, video_noaudio, tmp_path):
    path = make_review_with_stand_in(win, tmp_path)
    win2_path = str(tmp_path / "other.mp4")
    import shutil
    shutil.copy(video_noaudio, win2_path)
    win.open_video(win2_path)
    assert pump(lambda: win.info is not None and os.path.basename(win.info.path) == "other.mp4" and win.view._image is not None)
    assert win.load_review(path) is False and "different video" in win.errors[-1]
    assert win.review is None


def test_reprocess_plan_reuses_everything_when_nothing_changed(win, tmp_path):
    open_clip(win)
    st = settings_for(tmp_path, fail_closed=False)                                              # NudeNet only
    assert win.start_processing(st) and finish(win)
    assert win.last_result["status"] == "completed_no_flags" and win.review is not None
    texts = []
    win.ask_confirm = lambda t: texts.append(t) or True
    win.reprocess()
    assert pump(lambda: bool(texts), 60) and "nothing changed" in texts[0].lower() and "160 of 160 frames are reused" in texts[0]
    assert finish(win)
    assert win.last_result["status"] == "completed_no_flags" and win.review.meta["reused_chunks"] == 4
    assert win.review.meta["reprocessed_chunks"] == []


def test_reprocess_without_kept_chunks_says_so(win, tmp_path):
    open_clip(win)
    st = settings_for(tmp_path, fail_closed=False, keep_chunks=False)
    assert win.start_processing(st) and finish(win)
    texts = []
    win.ask_confirm = lambda t: texts.append(t) or False                                        # user declines
    win.reprocess()
    assert pump(lambda: bool(texts), 60) and "no kept intermediate chunks" in texts[0].lower() and "All 160 frames" in texts[0]
    assert not win.job_running


def test_delete_intermediate_chunks(win, tmp_path):
    open_clip(win)
    assert win.start_processing(settings_for(tmp_path, fail_closed=False)) and finish(win)
    from src.app.job_runner import job_dir_for_video
    jd = job_dir_for_video(win.last_settings.work_dir, win.info.path, win.info.fingerprint)
    assert os.path.isdir(jd)
    texts = []
    win.ask_confirm = lambda t: texts.append(t) or True
    win.delete_intermediate()
    assert not os.path.isdir(jd) and "Delete" in texts[0] and os.path.isfile(win.last_result["output"])


def tex(frame, y, x, b=16):
    y0, x0 = y // b * b, x // b * b
    return frame[y0:y0 + b, x0:x0 + b].astype(float).reshape(-1, 3).std(axis=0).mean()


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_real_sam2_review_reclick_partial_reprocess_end_to_end(win, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    open_clip(win)
    win.goto(20)
    assert pump(lambda: shown_code(win) == 20)
    win.view.handle_click(*_widget_pos(win, square_x(20) + 30, 180), 1)
    assert win.save_points()
    st = settings_for(tmp_path, sam2_checkpoint=CKPT, use_nudenet=False)
    assert win.start_processing(st) and finish(win, 240)
    first = win.last_result
    assert first["status"] in ("completed_no_flags", "completed_with_review_required"), first
    assert win.review is not None and "consistent" in win.review_panel.banner.text()
    lost = [s for s in win.review.segments if s["type"] == "tracking_lost_mid_frame"]
    assert lost and lost[0]["start_frame"] == 70                                                # the occlusion was flagged
    # the reviewer looks at the flag and marks it as reviewed
    win.next_flag()
    while win.frame_idx != 70:
        win.next_flag()
    win.review_panel.select_row_for(70)
    win.review_panel.toggle_reviewed()
    assert win.review.state.is_reviewed(lost[0])
    src = probe_video(win.test_video)
    with FrameReader(src) as r:
        src_frames = list(r)
    with FrameReader(probe_video(first["output"])) as r:
        before = list(r)
    xc = square_x(100) + 30                                                                     # object position at frame 100
    print("texture ratio at frame 100 before re-click:", round(tex(before[100], 170, xc) / tex(src_frames[100], 170, xc), 2))

    # re-click the object where it reappears (frame 90), save: the log is now out of date
    win.goto(90)
    assert pump(lambda: shown_code(win) == 90)
    win.view.handle_click(*_widget_pos(win, square_x(90) + 30, 180), 1)
    assert win.save_points() and "points file changed" in win.review_panel.banner.text()
    texts = []
    win.ask_confirm = lambda t: texts.append(t) or True
    win.last_result = None
    win.reprocess()
    assert pump(lambda: bool(texts), 60)
    assert "80 of 160 frames are reused" in texts[0] and "chunks 3-4" in texts[0]               # chunks 1-2 reused, 3-4 re-processed
    assert finish(win, 240)
    res = win.last_result
    assert res["status"] in ("completed_no_flags", "completed_with_review_required"), res

    meta = win.review.meta
    assert meta["reused_chunks"] == 2 and meta["reprocessed_chunks"] == [2, 3]
    assert "consistent" in win.review_panel.banner.text()                                       # log, points and output agree again
    lost2 = [s for s in win.review.segments if s["type"] == "tracking_lost_mid_frame"]
    assert lost2 and seg_key(lost2[0]) == seg_key(lost[0]) and win.review.state.is_reviewed(lost2[0])   # chunk 2 unchanged: mark kept
    assert all(not win.review.state.is_reviewed(s) for s in win.review.segments if s["start_frame"] >= 80)

    # the correction is in the final video: the object is mosaicked at frame 100 now
    with FrameReader(probe_video(res["output"])) as r:
        after = list(r)
    ratio = tex(after[100], 170, xc) / tex(src_frames[100], 170, xc)
    print("texture ratio at frame 100 after re-click:", round(ratio, 2))
    assert ratio < 0.5, "the re-clicked object is not mosaicked at frame 100"
    # (SAM 2 results vary between runs: in some runs frame 100 is already covered before the re-click, so this pixel check
    #  proves the correction reaches the output, while the IoU gain from a re-click is measured in test_sam2_real.py)
    mid = np.abs(after[30].astype(int) - before[30].astype(int)).mean()
    assert mid < 0.5, "a reused chunk changed"
