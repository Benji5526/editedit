"""The desktop app and the job registry: interrupted jobs are found, offered for resume, and listed in the history."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.app.job_runner import ProcessSettings  # noqa: E402
from src.app.jobs_dialog import JobsDialog, job_rows  # noqa: E402
from src.app.main_window import MainWindow  # noqa: E402
from src.job_manager import JobDB, db_path_for  # noqa: E402
from test_gui import open_clip, pump, qapp, win  # noqa: E402,F401


def settings_for(tmp_path, **kw):
    base = dict(output=str(tmp_path / "out" / "result.mp4"), chunk_size=10, analysis_width=320, fail_closed=False,
                work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"), keep_chunks=True)
    base.update(kw)
    return ProcessSettings(**base)


def finish(win, timeout=120):
    return pump(lambda: not win.job_running and win.last_result is not None and not win.plan_runner.is_running(), timeout)


def test_cancelled_job_is_offered_for_resume_and_listed(win, tmp_path):
    open_clip(win)
    assert win.resumable_job is None and not win.act_resume_job.isEnabled()
    st = settings_for(tmp_path)
    assert win.start_processing(st)
    assert pump(lambda: win.panel.bar.value() > 150)
    win.cancel_processing()
    assert finish(win)
    assert win.last_result["status"] == "cancelled"

    j = win.resumable_job
    assert j is not None and j["status"] == "cancelled" and 0 < j["chunks_done"] < 16 and j["n_chunks"] == 16
    assert win.act_resume_job.isEnabled()
    assert "Interrupted job found" in win.statusBar().currentMessage()

    jobs = JobDB(db_path_for(st.log_dir)).jobs_for_input(win.info.fingerprint)
    rows = job_rows(jobs)
    assert rows[0][1] == "cancelled" and rows[0][2] == f"{j['chunks_done']}/16"
    dlg = JobsDialog(win, jobs, "clip.mp4")
    assert dlg.table.rowCount() == 1 and dlg.table.item(0, 1).text() == "cancelled"

    # 'Resume interrupted job' = plan first, then run with the reusable chunks
    texts = []
    win.ask_confirm = lambda t: texts.append(t) or True
    win.act_resume_job.trigger()
    assert pump(lambda: bool(texts), 60) and "frames are reused" in texts[0]
    win.last_result = None
    assert finish(win)
    assert win.last_result["status"] == "completed_no_flags"
    assert win.resumable_job is None and not win.act_resume_job.isEnabled()
    statuses = [r[1] for r in job_rows(JobDB(db_path_for(st.log_dir)).jobs_for_input(win.info.fingerprint))]
    assert statuses == ["completed_no_flags", "cancelled"]
    assert job_rows(JobDB(db_path_for(st.log_dir)).jobs_for_input(win.info.fingerprint))[0][3] != "0"     # chunks were reused


def test_interrupted_job_is_found_again_when_the_video_is_reopened(win, tmp_path):
    open_clip(win)
    st = settings_for(tmp_path)
    assert win.start_processing(st)
    assert pump(lambda: win.panel.bar.value() > 150)
    win.cancel_processing()
    assert finish(win)

    w2 = MainWindow()                                          # "restarted app": a fresh window that knows only the log folder
    w2.show_error = lambda t: None
    w2.last_settings = st
    try:
        open_clip(w2, win.test_video)
        assert w2.resumable_job is not None and "Interrupted job found" in w2.statusBar().currentMessage()
    finally:
        w2.close()


def test_no_registry_no_problem(win, tmp_path):
    open_clip(win)
    win.last_settings = settings_for(tmp_path)                 # log dir that has no jobs.sqlite
    assert win.check_resumable() is None and not win.act_resume_job.isEnabled()
    dlg = JobsDialog(win, [], "clip.mp4")
    assert dlg.table.rowCount() == 0
