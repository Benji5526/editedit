"""Desktop app driving the real worker process: start, progress, cancel, resume, failure, closing while running."""
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.app.job_runner import PROJECT_ROOT, ProcessSettings  # noqa: E402
from src.video_io import probe_video  # noqa: E402
from test_gui import _widget_pos, open_clip, pump, qapp, win  # noqa: E402,F401  (fixtures + helpers shared with the GUI tests)

CKPT = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_small.pt")


def settings_for(tmp_path, **kw):
    base = dict(output=str(tmp_path / "out" / "result.mp4"), chunk_size=40, analysis_width=320, fail_closed=False,
                work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"))
    base.update(kw)
    return ProcessSettings(**base)


def finish(win, timeout=120):
    return pump(lambda: not win.job_running and win.last_result is not None, timeout)


def test_nudenet_only_job_runs_to_completion(win, tmp_path):
    open_clip(win)
    opened = []
    win.open_path = opened.append
    assert win.start_processing(settings_for(tmp_path))
    assert win.job_running and win.process_dock.isVisible()
    assert not win.act_open_video.isEnabled() and not win.act_start.isEnabled() and win.act_cancel.isEnabled()
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)         # editing is locked while the job runs
    assert win.store.points == [] and "locked" in win.statusBar().currentMessage()
    assert finish(win)
    res = win.last_result
    assert res["status"] == "completed_no_flags" and res["exit_code"] == 0 and not res["cancelled_by_user"]
    assert os.path.isfile(res["output"]) and probe_video(res["output"]).nb_frames == 160
    assert win.panel.bar.value() == 1000 and win.panel.out_btn.isVisibleTo(win.panel)
    assert "No review flags" in win.panel.summary.text()
    assert win.act_open_video.isEnabled() and win.act_start.isEnabled() and not win.act_cancel.isEnabled()
    win.panel.out_btn.click()
    assert opened == [res["output"]]


def test_cancel_then_resume_through_the_gui(win, tmp_path):
    open_clip(win)
    st = settings_for(tmp_path, chunk_size=10)                   # 16 chunks: plenty of time to cancel in the middle
    assert win.start_processing(st)
    assert pump(lambda: win.panel.bar.value() > 100)             # some chunks are done
    win.cancel_processing()
    assert finish(win)
    res = win.last_result
    assert res["status"] == "cancelled" and res["cancelled_by_user"] and res["exit_code"] == 3
    assert not os.path.exists(st.output) and not os.path.exists(st.output + ".partial")
    assert win.panel.resume_btn.isVisibleTo(win.panel) and not win.panel.out_btn.isVisibleTo(win.panel)
    assert "Cancelled" in win.panel.summary.text()
    win.last_result = None
    win.panel.resume_btn.click()                                 # same settings + --resume
    assert win.job_running and win.last_settings.resume
    assert finish(win)
    assert win.last_result["status"] == "completed_no_flags" and probe_video(st.output).nb_frames == 160


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_job_with_clicked_points_uses_real_sam2(win, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    open_clip(win)
    win.goto(20)
    pump(lambda: win.frame_idx == 20)
    win.view.handle_click(*_widget_pos(win, 270, 180), 1)
    assert win.dirty
    win.ask_save_changes = lambda: "save"
    st = settings_for(tmp_path, sam2_checkpoint=CKPT, use_nudenet=False, fail_closed=True, max_frames=80)
    assert win.start_processing(st) is False and "Save the points first" in win.errors[-1]   # unsaved points are refused
    assert win.save_points()
    assert win.start_processing(st)
    assert finish(win, 240)
    res = win.last_result
    assert res["status"] in ("completed_no_flags", "completed_with_review_required"), res
    assert probe_video(res["output"]).nb_frames == 80
    assert os.path.isfile(res["review_log"])


def test_missing_checkpoint_is_refused_before_starting(win, tmp_path):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)
    assert win.save_points()
    ok = win.start_processing(settings_for(tmp_path, sam2_checkpoint=str(tmp_path / "missing.pt")))
    assert ok is False and not win.job_running and "checkpoint" in win.errors[-1]


def test_broken_checkpoint_fails_visibly_and_offers_resume(win, tmp_path):
    open_clip(win)
    win.view.handle_click(*_widget_pos(win, 100, 100), 1)
    assert win.save_points()
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    assert win.start_processing(settings_for(tmp_path, sam2_checkpoint=str(bad), use_nudenet=False))
    assert finish(win)
    assert win.last_result["status"] == "failed" and win.panel.summary.text().startswith("FAILED")
    assert not os.path.exists(win.last_settings.output) and win.panel.resume_btn.isVisibleTo(win.panel)
    assert win.act_start.isEnabled()                              # the app is usable again


def test_closing_while_a_job_runs_cancels_it(win, tmp_path):
    open_clip(win)
    assert win.start_processing(settings_for(tmp_path, chunk_size=10))
    assert pump(lambda: win.panel.bar.value() > 50)
    win.ask_confirm = lambda t: False
    assert win.close() is False and win.job_running              # user said no: keeps running
    win.ask_confirm = lambda t: True
    assert win.close() is True and not win.job_running
    assert win.last_result["status"] == "cancelled"


def test_nothing_to_do_is_refused(win, tmp_path):
    open_clip(win)
    assert win.start_processing(settings_for(tmp_path, use_nudenet=False)) is False
    assert "Nothing to do" in win.errors[-1] and not win.job_running
