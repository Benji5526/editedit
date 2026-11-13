"""Editor-style timeline under the preview: a time ruler, a strip of thumbnails, one track per object (its clicks and, after a
run, the frames where it has a mask), a track for detections, a track for the review flags, and the playhead.
Click or drag anywhere on the tracks to seek. The widget only draws what it is given; it never reads files itself
(thumbnails come from ThumbnailLoader, which runs ffmpeg off the UI thread)."""
from __future__ import annotations

import subprocess
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

GUTTER = 64            # width of the row labels on the left
RULER_H = 18
THUMB_H = 46
ROW_H = 16
ROW_GAP = 3
MAX_OBJECT_ROWS = 8
SEV_COLOR = {"error": "#ff6b6b", "warn": "#e0a030", "info": "#6a8fb8"}


def runs_of(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """[start, end) runs of True values."""
    a = np.asarray(flags, dtype=bool)
    if a.size == 0 or not a.any():
        return []
    d = np.diff(np.concatenate(([0], a.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def nice_step(seconds_per_label: float) -> float:
    """Smallest 'round' interval (seconds) that is at least ``seconds_per_label``."""
    for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if s >= seconds_per_label:
            return float(s)
    return 7200.0


def format_time(seconds: float, with_fraction: bool = False) -> str:
    s = max(0.0, float(seconds))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    text = f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
    return text + (f".{int((s - int(s)) * 10)}" if with_fraction else "")


class TimelineView(QWidget):
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(False)
        self._n = 0
        self._fps = 30.0
        self._frame = 0
        self._thumbs: List[QImage] = []
        self._objects: List[Tuple[int, QColor, List[int]]] = []          # (obj_id, colour, frames with clicks)
        self._presence: Dict[int, List[Tuple[int, int]]] = {}            # obj_id -> runs of frames with a mask
        self._det_runs: List[Tuple[int, int]] = []
        self._fc_runs: List[Tuple[int, int]] = []
        self._has_run_data = False
        self._flags: List[Tuple[int, int, str]] = []                     # (start frame, end frame, severity)
        self._cuts: List[Tuple[int, int]] = []                           # [start, end) ranges that will be cut out
        self._selection: Optional[Tuple[int, int]] = None                # [start, end) range being marked
        self._show_cuts = False
        self._highlight: Optional[Tuple[int, int]] = None                # [start, end) range marked on the ruler and the strip
        self._update_height()

    # ---- data
    def set_video(self, nb_frames: int, fps: float):
        self._n, self._fps, self._frame = max(0, int(nb_frames)), float(fps) or 30.0, 0
        self._thumbs = []
        self.set_run_data(None)
        self._flags = []
        self._highlight = None
        self.update()

    def set_frame(self, idx: int):
        if idx != self._frame:
            self._frame = idx
            self.update()

    def set_thumbnails(self, images: List[QImage]):
        self._thumbs = list(images)
        self.update()

    def set_objects(self, objects: List[Tuple[int, QColor, List[int]]]):
        self._objects = list(objects)
        self._update_height()
        self.update()

    def set_run_data(self, data: Optional[dict]):
        """Per-frame results of the last run: {"masks": {obj_id: bool array}, "detections": bool array, "fail_closed": bool array}
        (each as long as the video), or None when there is no kept run."""
        self._has_run_data = data is not None
        self._presence = {int(k): runs_of(v) for k, v in ((data or {}).get("masks") or {}).items()}
        self._det_runs = runs_of((data or {}).get("detections", []))
        self._fc_runs = runs_of((data or {}).get("fail_closed", []))
        self._update_height()
        self.update()

    def set_cuts(self, cuts: List[Tuple[int, int]], selection: Optional[Tuple[int, int]] = None):
        """Ranges to be removed (red) and the range currently being marked (blue); used by the cutting window."""
        self._cuts, self._selection, self._show_cuts = list(cuts), selection, True
        self._update_height()
        self.update()

    def set_highlight(self, rng: Optional[Tuple[int, int]]):
        """Mark a frame range [start, end) (the frames of a before/after comparison), or None."""
        self._highlight = rng
        self.update()

    def set_flags(self, flags: List[Tuple[int, int, str]]):
        self._flags = list(flags)
        self._update_height()
        self.update()

    # ---- layout
    def rows(self) -> List[Tuple[str, str]]:
        """(kind, label) of the tracks below the thumbnails, top to bottom."""
        out = [("object:%d" % oid, f"대상 {oid}") for oid, _, _ in self._objects[:MAX_OBJECT_ROWS]]
        if len(self._objects) > MAX_OBJECT_ROWS:
            out.append(("more", f"+{len(self._objects) - MAX_OBJECT_ROWS}개"))
        if self._has_run_data:
            out.append(("detections", "탐지"))
            out.append(("fail_closed", "안전 모드"))
        if self._flags:
            out.append(("flags", "검수"))
        if self._show_cuts:
            out.append(("cuts", "삭제"))
        return out

    def _update_height(self):
        self.setFixedHeight(RULER_H + THUMB_H + 4 + len(self.rows()) * (ROW_H + ROW_GAP) + 6)

    def content_rect(self) -> QRectF:
        return QRectF(GUTTER, 0, max(1.0, self.width() - GUTTER - 8), self.height())

    def x_of(self, frame: float) -> float:
        c = self.content_rect()
        return c.left() + (frame / max(1, self._n - 1)) * c.width() if self._n > 1 else c.left()

    def frame_at(self, x: float) -> int:
        c = self.content_rect()
        if self._n <= 1:
            return 0
        return int(round(min(1.0, max(0.0, (x - c.left()) / c.width())) * (self._n - 1)))

    # ---- input
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._n > 0 and self.isEnabled():
            self.seek_requested.emit(self.frame_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton and self._n > 0 and self.isEnabled():
            self.seek_requested.emit(self.frame_at(e.position().x()))

    # ---- drawing
    def paintEvent(self, _):
        p = QPainter(self)
        c = self.content_rect()
        p.setFont(QFont("Malgun Gothic", 8))
        muted, text = QColor("#9aa0aa"), QColor("#e6e7ea")
        if self._n <= 0:
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "영상을 열면 타임라인이 표시됩니다")
            return
        # ruler
        duration = self._n / self._fps
        step = nice_step(duration / max(1.0, c.width() / 90.0))
        p.setPen(QPen(QColor("#4a4e58"), 1))
        t = 0.0
        while t <= duration + 1e-6:
            x = self.x_of(t * self._fps)
            p.setPen(QPen(QColor("#4a4e58"), 1))
            p.drawLine(QPointF(x, RULER_H - 5), QPointF(x, RULER_H))
            p.setPen(muted)
            p.drawText(QPointF(x + 3, RULER_H - 6), format_time(t, with_fraction=step < 1))
            t += step
        # thumbnails
        strip = QRectF(c.left(), RULER_H, c.width(), THUMB_H)
        p.fillRect(strip, QColor("#101114"))
        if self._thumbs:
            w = strip.width() / len(self._thumbs)
            for i, img in enumerate(self._thumbs):
                if img is None or img.isNull():
                    continue
                cell = QRectF(strip.left() + i * w, strip.top(), w, strip.height())
                src_w = min(img.width(), img.height() * cell.width() / cell.height())       # centre crop to the cell's shape
                src_h = min(img.height(), img.width() * cell.height() / cell.width())
                p.drawImage(cell, img, QRectF((img.width() - src_w) / 2, (img.height() - src_h) / 2, src_w, src_h))
        for a, b in self._cuts:                                                               # removed parts are dimmed on the strip
            x0, x1 = self.x_of(a), self.x_of(max(a, b - 1))
            p.fillRect(QRectF(x0, strip.top(), max(2.0, x1 - x0), strip.height()), QColor(20, 0, 0, 170))
        p.setPen(muted)
        p.drawText(QRectF(0, RULER_H, GUTTER - 6, THUMB_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "영상")
        # tracks
        y = RULER_H + THUMB_H + 4
        colours = {oid: col for oid, col, _ in self._objects}
        clicks = {oid: frames for oid, _, frames in self._objects}
        for kind, label in self.rows():
            row = QRectF(c.left(), y, c.width(), ROW_H)
            p.fillRect(row, QColor("#24262c"))
            p.setPen(muted)
            p.drawText(QRectF(0, y, GUTTER - 6, ROW_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)
            if kind.startswith("object:"):
                oid = int(kind.split(":")[1])
                col = colours.get(oid, QColor("#888"))
                fill = QColor(col)
                fill.setAlpha(150)
                for a, b in self._presence.get(oid, []):
                    self._bar(p, row, a, b, fill)
                p.setPen(QPen(QColor("white"), 1))
                p.setBrush(col)
                for f in clicks.get(oid, []):                                                 # a diamond at every clicked frame
                    x, cy = self.x_of(f), row.center().y()
                    p.drawPolygon([QPointF(x, cy - 5), QPointF(x + 4, cy), QPointF(x, cy + 5), QPointF(x - 4, cy)])
                p.setBrush(Qt.BrushStyle.NoBrush)
            elif kind == "detections":
                for a, b in self._det_runs:
                    self._bar(p, row, a, b, QColor(60, 255, 255, 150))
            elif kind == "fail_closed":
                for a, b in self._fc_runs:
                    self._bar(p, row, a, b, QColor(255, 107, 107, 170))
            elif kind == "flags":
                for a, b, sev in self._flags:
                    self._bar(p, row, a, b + 1, QColor(SEV_COLOR.get(sev, "#999")))
            elif kind == "cuts":
                for a, b in self._cuts:
                    self._bar(p, row, a, b, QColor(255, 107, 107, 220))
                if self._selection is not None:
                    self._bar(p, row, self._selection[0], self._selection[1], QColor(76, 141, 255, 220))
            y += ROW_H + ROW_GAP
        if self._highlight is not None:                                                       # compared range: a band over everything
            x0, x1 = self.x_of(self._highlight[0]), self.x_of(max(self._highlight[0], self._highlight[1] - 1))
            band = QRectF(x0, 0, max(3.0, x1 - x0), self.height())
            p.fillRect(band, QColor(76, 141, 255, 55))
            p.setPen(QPen(QColor(76, 141, 255), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(band)
        # playhead
        x = self.x_of(self._frame)
        p.setPen(QPen(text, 2))
        p.drawLine(QPointF(x, 2), QPointF(x, self.height() - 2))
        p.setBrush(text)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 8)])

    def _bar(self, p: QPainter, row: QRectF, a: int, b: int, colour: QColor):
        x0, x1 = self.x_of(a), self.x_of(max(a, b - 1))
        p.fillRect(QRectF(x0, row.top() + 2, max(2.0, x1 - x0), row.height() - 4), colour)


class _ThumbSignals(QObject):
    done = Signal(str, list)        # video path, [QImage or None, ...]


class ThumbnailLoader(QRunnable):
    """Grabs ``count`` small frames spread over the video with ffmpeg (one fast seek each), off the UI thread.
    Thumbnails are approximate on purpose: the seek goes to the nearest keyframe."""

    def __init__(self, info, count: int = 16, height: int = THUMB_H * 2):
        super().__init__()
        self.info, self.count, self.height = info, count, height
        self.signals = _ThumbSignals()
        self.cancelled = False

    def run(self):
        from src.video_io import find_tool
        info = self.info
        h = self.height // 2 * 2
        w = max(2, int(round(info.width * h / max(1, info.height))) // 2 * 2)
        images: list = []
        try:
            ffmpeg = find_tool("ffmpeg")
            for i in range(self.count):
                if self.cancelled:
                    return
                t = info.duration * (i + 0.5) / self.count
                cmd = [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", info.path, "-frames:v", "1", "-an",
                       "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
                try:
                    out = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=0x08000000).stdout
                except Exception:  # noqa: BLE001
                    out = b""
                if len(out) >= w * h * 3:
                    images.append(QImage(out[:w * h * 3], w, h, w * 3, QImage.Format.Format_RGB888).copy())
                else:
                    images.append(None)
        finally:
            if not self.cancelled:
                self.signals.done.emit(info.path, images)


class _SummarySignals(QObject):
    done = Signal(str, object)      # video path, summary dict or None


class RunSummaryLoader(QRunnable):
    """Reads the per-frame summary of a kept run (src.mask_cache.JobMasks.summary) off the UI thread."""

    def __init__(self, video_path: str, job_dir: str):
        super().__init__()
        self.video_path, self.job_dir = video_path, job_dir
        self.signals = _SummarySignals()
        self.cancelled = False
        self.finished = threading.Event()        # set when run() has returned: no file of the job folder is open any more

    def run(self):
        data = None
        try:
            from src.mask_cache import JobMasks
            jm = JobMasks(self.job_dir)
            if jm.available_chunks():
                data = jm.summary(should_stop=lambda: self.cancelled)
        except Exception:  # noqa: BLE001  (no kept run: nothing to show)
            data = None
        finally:
            self.finished.set()
        if not self.cancelled:
            self.signals.done.emit(self.video_path, data)

    def stop(self, timeout: float = 10.0) -> bool:
        """Cancel and wait until the job folder is no longer being read (needed before it is deleted or rewritten: an open file
        makes that fail on Windows). Returns False when the loader never started or did not stop in time."""
        self.cancelled = True
        return self.finished.wait(timeout)
