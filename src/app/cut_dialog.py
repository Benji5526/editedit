"""'결과 영상 자르기' window: mark ranges of a finished video on a timeline and export the video without them.

It has its own preview and timeline and works on a video FILE only (src.video_edit); the points, masks and jobs of the main window
are never touched. The export runs on a background thread."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
                               QVBoxLayout)

from src.video_edit import CutCancelled, export_cut, frames_kept, normalize_ranges
from src.video_io import VideoInfo

from .frame_server import FrameServer
from .timeline import ThumbnailLoader, TimelineView, format_time
from .video_view import VideoView


def default_cut_output(video_path: str) -> str:
    stem, _ = os.path.splitext(video_path)
    return stem + "_cut.mp4"


class _CutSignals(QObject):
    progress = Signal(int, int)
    done = Signal(object, str)        # result dict or None, error text ("" = ok, "cancelled" = stopped by the user)


class CutTask(QRunnable):
    def __init__(self, info: VideoInfo, remove: List[Tuple[int, int]], out_path: str, encoder: str = "libx264", preset: str = "medium"):
        super().__init__()
        self.info, self.remove, self.out_path, self.encoder, self.preset = info, list(remove), out_path, encoder, preset
        self.signals = _CutSignals()
        self.cancel = False

    def run(self):
        try:
            res = export_cut(self.info.path, self.remove, self.out_path, encoder=self.encoder, preset=self.preset, info=self.info,
                             progress=lambda d, t: self.signals.progress.emit(d, t), cancel_check=lambda: self.cancel)
            self.signals.done.emit(res, "")
        except CutCancelled:
            self.signals.done.emit(None, "cancelled")
        except Exception as e:  # noqa: BLE001
            self.signals.done.emit(None, f"{type(e).__name__}: {e}")


class CutDialog(QDialog):
    def __init__(self, parent, info: VideoInfo):
        super().__init__(parent)
        self.setWindowTitle(f"결과 영상 자르기 - {os.path.basename(info.path)}")
        self.resize(1100, 760)
        self.info = info
        self.frame_idx = -1
        self.mark_in: Optional[int] = None
        self.mark_out: Optional[int] = None
        self.remove: List[Tuple[int, int]] = []          # [start, end) frame ranges that will be cut out
        self.task: Optional[CutTask] = None
        self.result: Optional[dict] = None
        self.last_error = ""

        self.view = VideoView()
        self.view.message = ""
        self.view.set_video(info.width, info.height)
        self.server = FrameServer(info)
        self.server.frame_ready.connect(self._frame_ready)
        self.timeline = TimelineView()
        self.timeline.set_video(info.nb_frames, info.fps)
        self.timeline.seek_requested.connect(self.goto)
        self.pos = QLabel("")
        self.pos.setObjectName("timecode")

        def btn(text, fn, tip=""):
            b = QPushButton(text)
            b.clicked.connect(fn)
            b.setToolTip(tip)
            return b

        nav = QHBoxLayout()
        nav.addWidget(self.pos)
        nav.addStretch(1)
        for text, step in (("-1초", -round(info.fps)), ("-10", -10), ("<", -1), (">", 1), ("+10", 10), ("+1초", round(info.fps))):
            nav.addWidget(btn(text, lambda _=False, s=step: self.goto(self.frame_idx + s)))
        nav.addStretch(1)

        marks = QHBoxLayout()
        self.in_btn = btn("구간 시작 지정 (I)", self.set_in, "현재 프레임을 잘라낼 구간의 시작으로 지정합니다.")
        self.out_btn = btn("구간 끝 지정 (O)", self.set_out, "현재 프레임을 잘라낼 구간의 끝(이 프레임까지 포함)으로 지정합니다.")
        self.add_btn = btn("선택 구간을 삭제 목록에 추가", self.add_selection)
        self.head_btn = btn("처음~현재 앞부분 잘라내기", self.cut_head, "영상 처음부터 현재 프레임 바로 앞까지를 삭제 목록에 넣습니다.")
        self.tail_btn = btn("현재~끝 뒷부분 잘라내기", self.cut_tail, "현재 프레임부터 영상 끝까지를 삭제 목록에 넣습니다.")
        for b in (self.in_btn, self.out_btn, self.add_btn, self.head_btn, self.tail_btn):
            marks.addWidget(b)
        self.sel_label = QLabel("")
        self.sel_label.setObjectName("hint")

        self.cut_list = QListWidget()
        self.cut_list.setMaximumHeight(110)
        self.del_btn = btn("선택한 항목을 목록에서 빼기", self.remove_selected)
        self.clear_btn = btn("목록 비우기", self.clear_cuts)
        lrow = QHBoxLayout()
        lrow.addWidget(self.del_btn)
        lrow.addWidget(self.clear_btn)
        lrow.addStretch(1)
        self.summary = QLabel("")

        self.out = QLineEdit(default_cut_output(info.path))
        orow = QHBoxLayout()
        orow.addWidget(QLabel("결과 파일"))
        orow.addWidget(self.out, 1)
        orow.addWidget(btn("...", self._browse))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.status = QLabel("삭제할 구간을 지정한 뒤 내보내기를 누르세요. 원본 파일은 바뀌지 않습니다.")
        self.status.setWordWrap(True)
        self.export_btn = btn("내보내기", self.start_export)
        self.export_btn.setObjectName("play")
        self.cancel_btn = btn("취소", self.cancel_export)
        self.cancel_btn.setEnabled(False)
        self.close_btn = btn("닫기", self.close)
        brow = QHBoxLayout()
        brow.addStretch(1)
        for b in (self.cancel_btn, self.export_btn, self.close_btn):
            brow.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(nav)
        lay.addWidget(self.timeline)
        lay.addLayout(marks)
        lay.addWidget(self.sel_label)
        lay.addWidget(QLabel("삭제 목록 (내보낼 때 잘려 나가는 구간):"))
        lay.addWidget(self.cut_list)
        lay.addLayout(lrow)
        lay.addWidget(self.summary)
        lay.addLayout(orow)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addLayout(brow)
        for key, fn in (("I", self.set_in), ("O", self.set_out), ("Left", lambda: self.goto(self.frame_idx - 1)),
                        ("Right", lambda: self.goto(self.frame_idx + 1))):
            QShortcut(QKeySequence(key), self, activated=fn)

        self._thumbs = ThumbnailLoader(info)
        self._thumbs.setAutoDelete(False)
        self._thumbs.signals.done.connect(lambda path, images: self.timeline.set_thumbnails(images) if path == self.info.path else None)
        QThreadPool.globalInstance().start(self._thumbs)
        self._refresh()
        self.goto(0)

    # ---- navigation
    def goto(self, idx: int):
        idx = int(min(max(0, idx), self.info.nb_frames - 1))
        if idx == self.frame_idx:
            return
        self.frame_idx = idx
        self.timeline.set_frame(idx)
        self.pos.setText(f"프레임 {idx}/{self.info.nb_frames - 1}   t = {self.info.timestamp_of(idx):.3f}초")
        self.server.request(idx)

    def _frame_ready(self, idx: int, img):
        if idx == self.frame_idx:
            self.view.set_image(img)

    # ---- marking
    def _span(self, s: int, e: int) -> str:
        """Text for the frame range [s, e)."""
        fps = self.info.fps
        return f"{format_time(s / fps, True)} ~ {format_time(e / fps, True)}  (프레임 {s}-{e - 1}, {e - s}프레임)"

    def set_in(self):
        self.mark_in = self.frame_idx
        if self.mark_out is not None and self.mark_out < self.mark_in:
            self.mark_out = None
        self._refresh()

    def set_out(self):
        self.mark_out = self.frame_idx
        if self.mark_in is not None and self.mark_in > self.mark_out:
            self.mark_in = None
        self._refresh()

    def selection(self) -> Optional[Tuple[int, int]]:
        """The marked range as [start, end), or None while it is incomplete."""
        if self.mark_in is None or self.mark_out is None:
            return None
        return self.mark_in, self.mark_out + 1

    def add_selection(self):
        sel = self.selection()
        if sel is not None:
            self._add(sel)
            self.mark_in = self.mark_out = None
            self._refresh()

    def cut_head(self):
        if self.frame_idx > 0:
            self._add((0, self.frame_idx))
            self._refresh()

    def cut_tail(self):
        self._add((self.frame_idx, self.info.nb_frames))
        self._refresh()

    def _add(self, rng: Tuple[int, int]):
        self.remove = normalize_ranges(self.remove + [rng], self.info.nb_frames)

    def remove_selected(self):
        row = self.cut_list.currentRow()
        if 0 <= row < len(self.remove):
            del self.remove[row]
            self._refresh()

    def clear_cuts(self):
        self.remove = []
        self._refresh()

    def _refresh(self):
        n, fps = self.info.nb_frames, self.info.fps
        self.cut_list.clear()
        for s, e in self.remove:
            self.cut_list.addItem(self._span(s, e))
        sel = self.selection()
        if sel is not None:
            self.sel_label.setText("선택한 구간: " + self._span(*sel))
        elif self.mark_in is not None:
            self.sel_label.setText(f"구간 시작: 프레임 {self.mark_in}. 끝 지점으로 이동해 '구간 끝 지정'을 누르세요.")
        elif self.mark_out is not None:
            self.sel_label.setText(f"구간 끝: 프레임 {self.mark_out}. 시작 지점으로 이동해 '구간 시작 지정'을 누르세요.")
        else:
            self.sel_label.setText("구간을 지정하려면 시작 지점과 끝 지점에서 각각 버튼을 누르세요 (단축키 I / O).")
        kept = frames_kept(n, self.remove)
        self.summary.setText(f"원본 {n}프레임 ({format_time(n / fps, True)})  →  결과 {kept}프레임 ({format_time(kept / fps, True)}),  "
                             f"{n - kept}프레임 삭제")
        self.timeline.set_cuts(self.remove, sel)
        busy = self.task is not None
        self.add_btn.setEnabled(sel is not None and not busy)
        self.export_btn.setEnabled(not busy and 0 < kept < n)
        for b in (self.in_btn, self.out_btn, self.head_btn, self.tail_btn, self.del_btn, self.clear_btn):
            b.setEnabled(not busy)

    # ---- export
    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "결과 파일", self.out.text(), "MP4 (*.mp4)")
        if p:
            self.out.setText(p)

    def problems(self) -> List[str]:
        out = self.out.text().strip()
        probs = []
        if not out:
            probs.append("결과 파일을 지정하세요.")
        elif os.path.abspath(out) == os.path.abspath(self.info.path):
            probs.append("결과 파일이 원본 영상과 같으면 안 됩니다.")
        kept = frames_kept(self.info.nb_frames, self.remove)
        if kept <= 0:
            probs.append("남는 프레임이 없습니다. 삭제 목록을 줄이세요.")
        elif kept == self.info.nb_frames:
            probs.append("삭제할 구간이 없습니다.")
        return probs

    def confirm_overwrite(self, path: str) -> bool:
        B = QMessageBox.StandardButton
        return QMessageBox.question(self, "결과 영상 자르기", f"{path}\n파일이 이미 있습니다. 덮어쓸까요?", B.Yes | B.No) == B.Yes

    def start_export(self) -> bool:
        if self.task is not None:
            return False
        probs = self.problems()
        if probs:
            self.status.setText("\n".join(probs))
            self.status.setStyleSheet("color: #ff6b6b")
            return False
        out = self.out.text().strip()
        if os.path.exists(out) and not self.confirm_overwrite(out):
            return False
        self.result, self.last_error = None, ""
        task = CutTask(self.info, self.remove, out, encoder="libx264")
        task.setAutoDelete(False)
        task.signals.progress.connect(self._on_progress)
        task.signals.done.connect(self._on_done)
        self.task = task
        self.bar.setValue(0)
        self.status.setStyleSheet("")
        self.status.setText("내보내는 중 ... (영상과 오디오를 다시 인코딩합니다)")
        self.cancel_btn.setEnabled(True)
        self._refresh()
        QThreadPool.globalInstance().start(task)
        return True

    def cancel_export(self):
        if self.task is not None:
            self.task.cancel = True
            self.cancel_btn.setEnabled(False)
            self.status.setText("취소하는 중 ...")

    def _on_progress(self, done: int, total: int):
        self.bar.setValue(int(1000 * done / max(1, total)))

    def _on_done(self, res, err: str):
        self.task = None
        self.cancel_btn.setEnabled(False)
        if err == "cancelled":
            self.status.setText("취소했습니다. 결과 파일은 만들어지지 않았습니다.")
            self.bar.setValue(0)
        elif err:
            self.last_error = err
            self.status.setStyleSheet("color: #ff6b6b")
            self.status.setText("내보내기에 실패했습니다: " + err)
        else:
            self.result = res
            self.bar.setValue(1000)
            self.status.setStyleSheet("color: #5cc46a")
            self.status.setText(f"완료: {self.out.text().strip()}\n{res['frames']}프레임 ({format_time(res['duration'], True)}), "
                                f"{res['removed']}프레임 삭제, 프레임 수 확인됨."
                                + ("" if res["audio"] else " 오디오 없음."))
        self._refresh()

    def closeEvent(self, e):
        if self.task is not None:
            self.task.cancel = True
        self._thumbs.cancelled = True
        self.server.close()
        e.accept()
