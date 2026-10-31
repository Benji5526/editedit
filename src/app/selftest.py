"""Diagnostic mode of the desktop app: open a video, run a processing job through the normal GUI code path, write a report, quit.

    VideoCensor.exe <video> --selftest-job <report.json>

It drives exactly what the buttons drive (open_video, start_processing, JobRunner, the worker exe, the progress panel); only the clicks are
replaced. If a points file sits next to the video (<video>.points.json) the job uses SAM 2 and the checkpoint from the models folder,
otherwise it runs the NudeNet-only job. Used by tools/verify_build.py and the tests; not an end-user feature.
"""
from __future__ import annotations

import json
import os
import sys
import time

from src.paths import app_root, is_frozen, worker_command


def install(app, win, video: str, report_path: str, timeout: float = 900.0):
    from PySide6.QtCore import QTimer
    from src.video_io import probe_video
    from .job_runner import ProcessSettings
    from .main_window import default_points_path
    from .process_dialog import default_output, find_default_checkpoint

    errors = []
    win.show_error = lambda text: errors.append(text)
    win.ask_confirm = lambda text: True
    win.ask_save_changes = lambda: "discard"
    state = {"t0": time.time(), "started": False, "done": False, "mode": None}

    def write(extra: dict):
        rep = {"frozen": is_frozen(), "app_root": app_root(), "mode": state["mode"], "errors": errors, "seconds": round(time.time() - state["t0"], 1)}
        rep.update(extra)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=str)
        state["done"] = True
        app.quit()

    def tick():
        if state["done"]:
            return
        if time.time() - state["t0"] > timeout:
            write({"status": "timeout"})
            return
        if win.info is None:
            return                                                  # still analysing the video
        if not state["started"]:
            state["started"] = True
            sam2 = os.path.isfile(default_points_path(win.info.path))
            state["mode"] = "sam2" if sam2 else "nudenet_only"
            st = ProcessSettings(sam2_checkpoint=find_default_checkpoint(), output=default_output(win.info.path), chunk_size=40,
                                 analysis_width=320, use_nudenet=True if not sam2 else False, fail_closed=True, keep_chunks=False)
            if not win.start_processing(st):
                write({"status": "not_started"})
            return
        if not win.job_running and win.last_result is not None:
            res = dict(win.last_result)
            out = res.get("output")
            frames = None
            if out and os.path.isfile(out):
                frames = probe_video(out).nb_frames
            write({"status": res.get("status"), "exit_code": res.get("exit_code"), "output": out, "output_frames": frames,
                   "source_frames": win.info.nb_frames, "review_log": res.get("review_log"), "reason": res.get("reason"),
                   "panel_summary": win.panel.summary.text(), "panel_progress": win.panel.bar.value(),
                   "review_loaded": win.review is not None, "flags_shown": win.review_panel.table.rowCount() if win.review else None,
                   "worker": os.path.basename(worker_command()[0])})

    t = QTimer(win)
    t.timeout.connect(tick)
    t.start(250)
    win._selftest_timer = t
    return t
