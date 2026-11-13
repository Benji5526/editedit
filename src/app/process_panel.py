"""Progress panel shown while (and after) a processing job runs."""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .ko_text import STATUS_TEXT  # noqa: F401  (also imported from here by the main window)
from .job_runner import overall_fraction

STAGE_TEXT = {"loading_models": "모델 불러오는 중 ...", "processing": "시작하는 중 ...", "finalizing": "조각을 하나로 합치고, 소리를 넣고, 결과 영상을 검증하는 중 ...", "analyze": "영상 읽기 + 자동 탐지",
              "track": "대상 추적 (SAM 2)", "render": "모자이크 입히기 + 영상 저장"}


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
        self.stage = QLabel("대기 중")
        self.detail = QLabel("")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(70)
        self.log.setMaximumHeight(130)                   # keep the panel compact so the preview above keeps its room
        self.cancel_btn = QPushButton("취소")
        self.resume_btn = QPushButton("이어하기")
        self.out_btn = QPushButton("결과 영상 열기")
        self.logs_btn = QPushButton("기록 폴더 열기")
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
        self.stage.setText("대기 중")
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
        if kind == "warning":
            self.add_log("경고: " + str(ev.get("text")))
            return
        if kind == "stage":
            self.stage.setText(STAGE_TEXT.get(ev.get("stage"), str(ev.get("stage"))))
            return
        f = overall_fraction(ev)
        if f is None:
            return
        if kind == "start":
            self._f0 = f
            self.detail.setText(f"{ev.get('total_frames')}프레임, {ev.get('chunks')}개 조각"
                                + (f" - 모자이크 설정 변경: {ev['rerender_chunks']}개 조각은 모자이크만 다시 적용 (추적 재사용)"
                                   if ev.get("rerender_chunks") else
                                   f" - 클릭 지점 변경: 조각 1-{ev['reused_chunks']} 재사용, 나머지 다시 처리"
                                   if ev.get("points_changed") and ev.get("reused_chunks") else
                                   f" - 클릭 지점 변경: 전체 다시 처리" if ev.get("points_changed") else
                                   f" (조각 {ev['resume_from_chunk'] + 1}부터 이어하기)" if ev.get("resume_from_chunk") else ""))
        if kind == "progress":
            self.stage.setText(f"조각 {ev['chunk'] + 1}/{ev['chunks']}: {STAGE_TEXT.get(ev['stage'], ev['stage'])}")
        if kind == "chunk_done":
            self.detail.setText(f"조각 {ev['chunk'] + 1}/{ev['chunks']} 완료  -  검수 이벤트 {ev.get('events', 0)}건"
                                f"  -  메모리 {ev.get('rss_mb', '?')} MB, GPU 최대 {ev.get('cuda_peak_mb', '?')} MB")
        self.bar.setValue(int(f * 1000))
        base = self._f0 or 0.0
        if f > base + 0.005:
            elapsed = time.time() - self._t0
            self.bar.setFormat(f"{f * 100:.1f}%   경과 {format_eta(elapsed)}   남은 시간 약 {format_eta(elapsed * (1 - f) / (f - base))}")
        else:
            self.bar.setFormat(f"{f * 100:.1f}%")

    def on_finished(self, res: dict):
        self.cancel_btn.setEnabled(False)
        status = res.get("status") or "failed"
        text = {"completed_no_flags": "완료. 검수할 항목이 없습니다. 다만 놓친 곳이 없다는 보장은 아니므로 결과 영상을 직접 확인하세요.",
                "completed_with_review_required": f"완료, 검수 필요: 오류 {res.get('error_events', 0)}건, "
                                                  f"경고 {res.get('warn_events', 0)}건. 결과 영상을 쓰기 전에 '검수' 탭의 항목을 확인하세요.",
                "cancelled": "취소했습니다. 끝난 조각은 보관되어 있으니 '이어하기'를 누르면 그 뒤부터 계속합니다.",
                "failed": "실패: " + str(res.get("reason") or "알 수 없는 오류")}.get(status, status)
        if res.get("problems"):
            text += "\n" + "; ".join(res["problems"])
        self.summary.setText(text)
        self.summary.setStyleSheet("color: #ff6b6b" if status == "failed" else ("color: #d9822b" if status != "completed_no_flags" else ""))
        self.stage.setText(STATUS_TEXT.get(status, status.replace("_", " ")))
        if status in ("completed_no_flags", "completed_with_review_required"):
            self.bar.setValue(1000)
            self.out_btn.setVisible(True)
        self.resume_btn.setVisible(status in ("cancelled", "failed"))
