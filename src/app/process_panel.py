"""Progress panel shown while (and after) a processing job runs."""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .job_runner import overall_fraction

STAGE_TEXT = {"loading_models": "Loading models ...", "processing": "Starting ...", "analyze": "Reading + detecting",
              "track": "Tracking (SAM 2)", "render": "Applying mosaic + encoding"}


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ProcessPanel(QWidget):
    cancel_requested = Signal()
    resume_requested = Signal()
    open_output_requested = Signal()
    open_logs_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stage = QLabel("Idle")
        self.detail = QLabel("")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(90)
        self.cancel_btn = QPushButton("Cancel")
        self.resume_btn = QPushButton("Resume")
        self.out_btn = QPushButton("Open result")
        self.logs_btn = QPushButton("Open logs folder")
        self.cancel_btn.clicked.connect(self.cancel_requested)
        self.resume_btn.clicked.connect(self.resume_requested)
        self.out_btn.clicked.connect(self.open_output_requested)
        self.logs_btn.clicked.connect(self.open_logs_requested)
        row = QHBoxLayout()
        for b in (self.cancel_btn, self.resume_btn, self.out_btn, self.logs_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        for w in (self.stage, self.bar, self.detail, self.summary, self.log):
            lay.addWidget(w)
        lay.addLayout(row)
        self._f0: Optional[float] = None
        self._t0 = 0.0
        self.reset()

    def reset(self):
        self.bar.setValue(0)
        self.stage.setText("Idle")
        self.detail.setText("")
        self.summary.setText("")
        self.log.clear()
        self.cancel_btn.setEnabled(False)
        for b in (self.resume_btn, self.out_btn):
            b.setVisible(False)
        self._f0 = None

    def begin(self):
        self.reset()
        self._t0 = time.time()
        self.cancel_btn.setEnabled(True)
        self.stage.setText(STAGE_TEXT["loading_models"])

    def add_log(self, line: str):
        self.log.appendPlainText(line)

    def on_progress(self, ev: dict):
        kind = ev.get("event")
        if kind == "stage":
            self.stage.setText(STAGE_TEXT.get(ev.get("stage"), str(ev.get("stage"))))
            return
        f = overall_fraction(ev)
        if f is None:
            return
        if kind == "start":
            self._f0 = f
            self.detail.setText(f"{ev.get('total_frames')} frames in {ev.get('chunks')} chunks"
                                + (f" (resuming at chunk {ev['resume_from_chunk'] + 1})" if ev.get("resume_from_chunk") else ""))
        if kind == "progress":
            self.stage.setText(f"Chunk {ev['chunk'] + 1}/{ev['chunks']}: {STAGE_TEXT.get(ev['stage'], ev['stage'])}")
        if kind == "chunk_done":
            self.detail.setText(f"chunk {ev['chunk'] + 1}/{ev['chunks']} done  -  {ev.get('events', 0)} review events"
                                f"  -  RSS {ev.get('rss_mb', '?')} MB, GPU peak {ev.get('cuda_peak_mb', '?')} MB")
        self.bar.setValue(int(f * 1000))
        base = self._f0 or 0.0
        if f > base + 0.005:
            elapsed = time.time() - self._t0
            self.bar.setFormat(f"{f * 100:.1f}%   elapsed {format_eta(elapsed)}   remaining ~{format_eta(elapsed * (1 - f) / (f - base))}")
        else:
            self.bar.setFormat(f"{f * 100:.1f}%")

    def on_finished(self, res: dict):
        self.cancel_btn.setEnabled(False)
        status = res.get("status") or "failed"
        text = {"completed_no_flags": "Finished. No review flags (this does not prove nothing was missed - spot-check the result).",
                "completed_with_review_required": f"Finished with review required: {res.get('error_events', 0)} error and "
                                                  f"{res.get('warn_events', 0)} warning events. Check the review log before using the video.",
                "cancelled": "Cancelled. Completed chunks are kept; Resume continues from them.",
                "failed": "FAILED: " + str(res.get("reason") or "unknown error")}.get(status, status)
        if res.get("problems"):
            text += "\n" + "; ".join(res["problems"])
        self.summary.setText(text)
        self.summary.setStyleSheet("color: #c0392b" if status == "failed" else ("color: #d9822b" if status != "completed_no_flags" else ""))
        self.stage.setText(status.replace("_", " "))
        if status in ("completed_no_flags", "completed_with_review_required"):
            self.bar.setValue(1000)
            self.out_btn.setVisible(True)
        self.resume_btn.setVisible(status in ("cancelled", "failed"))
