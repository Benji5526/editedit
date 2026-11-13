"""Preview widget: draws the frame letterboxed, overlays the clicked points, reports clicks in frame coordinates."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .geometry import Fit, fit_rect, frame_to_widget, widget_to_frame

COLORS = ["#ffc800", "#50ff50", "#3c78ff", "#ff3cff", "#3cffff", "#c8c83c", "#7878ff", "#ffc8c8", "#5aa05a"]


def object_color(obj_id: int) -> QColor:
    return QColor(COLORS[(obj_id - 1) % len(COLORS)])


def mask_image(mask: np.ndarray, color: QColor, alpha: int = 110) -> QImage:
    """Translucent image of a bool mask in one colour (drawn stretched over the frame)."""
    h, w = mask.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)                         # premultiplied ARGB32 = B, G, R, A bytes on little endian
    k = alpha / 255.0
    arr[mask] = (int(color.blue() * k), int(color.green() * k), int(color.red() * k), alpha)
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_ARGB32_Premultiplied).copy()


class VideoView(QWidget):
    clicked = Signal(float, float, int)    # frame x, frame y (full-res px), label (1 = positive, 0 = negative)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image: Optional[QImage] = None
        self._frame_size = (0, 0)
        self._points_for_frame: Callable[[], List[dict]] = lambda: []
        self._current_obj: Callable[[], int] = lambda: 1
        self.message = "영상을 여세요 (Ctrl+O)"
        self._overlay: Optional[dict] = None

    def set_overlay(self, overlay: Optional[dict]):
        """Masks and boxes to draw over the frame, or None. Keys (all optional): size (w, h) of the coordinate space of boxes and
        regions, masks [QImage], boxes [(x1, y1, x2, y2, text, filled)], regions [(x1, y1, x2, y2)], full (bool), note (str)."""
        self._overlay = overlay
        self.update()

    def overlay(self) -> Optional[dict]:
        return self._overlay

    def set_video(self, width: int, height: int):
        self._frame_size = (width, height)
        self._image = None
        self.update()

    def set_provider(self, points_for_frame: Callable[[], List[dict]], current_obj: Callable[[], int]):
        self._points_for_frame, self._current_obj = points_for_frame, current_obj

    def set_image(self, img: Optional[QImage]):
        self._image = img
        self.update()

    def fit(self) -> Optional[Fit]:
        return fit_rect(self.width(), self.height(), *self._frame_size)

    # ---- input
    def mousePressEvent(self, e):
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        pos = e.position()
        self.handle_click(pos.x(), pos.y(), 1 if e.button() == Qt.MouseButton.LeftButton else 0)

    def handle_click(self, wx: float, wy: float, label: int) -> bool:
        """Widget position -> frame coordinates; clicks on the letterbox bars are ignored."""
        if self._frame_size[0] <= 0:
            return False
        pt = widget_to_frame(self.fit(), wx, wy, *self._frame_size)
        if pt is None:
            return False
        self.clicked.emit(pt[0], pt[1], label)
        return True

    # ---- drawing
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101014"))
        fit = self.fit()
        if fit is None or self._frame_size[0] <= 0:
            p.setPen(QColor("#9aa0a6"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        target = QRectF(fit.ox, fit.oy, fit.w, fit.h)
        if self._image is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(target, self._image)
        else:
            p.fillRect(target, QColor("#1b1b22"))
            p.setPen(QColor("#9aa0a6"))
            p.drawText(target, Qt.AlignmentFlag.AlignCenter, "프레임 불러오는 중...")
        ov = self._overlay
        if ov:
            for img in ov.get("masks") or []:
                p.drawImage(target, img)
            sw, sh = ov.get("size") or self._frame_size
            if sw > 0 and sh > 0:
                def rect(b):
                    return QRectF(fit.ox + b[0] / sw * fit.w, fit.oy + b[1] / sh * fit.h, (b[2] - b[0]) / sw * fit.w, (b[3] - b[1]) / sh * fit.h)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setFont(QFont("Segoe UI", 8))
                for b in ov.get("boxes") or []:
                    p.setPen(QPen(QColor("#3cffff"), 2, Qt.PenStyle.DashLine if b[5] else Qt.PenStyle.SolidLine))
                    r = rect(b)
                    p.drawRect(r)
                    p.drawText(QPointF(r.left() + 2, r.top() - 3), str(b[4]))
                p.setPen(QPen(QColor("#ff5050"), 2, Qt.PenStyle.DotLine))
                for b in ov.get("regions") or []:
                    p.drawRect(rect(b))
            if ov.get("full"):
                p.setPen(QPen(QColor("#ff5050"), 4))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(target.adjusted(2, 2, -2, -2))
            if ov.get("note"):
                p.setFont(QFont("Segoe UI", 9))
                box = QRectF(target.left() + 6, target.top() + 6, min(target.width() - 12, 640), 22)
                p.fillRect(box, QColor(0, 0, 0, 150))
                p.setPen(QColor("white"))
                p.drawText(box.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, ov["note"])
        cur = self._current_obj()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        for pt in self._points_for_frame():
            xy = frame_to_widget(fit, pt["x"], pt["y"])
            c = object_color(pt["obj_id"])
            r = 8 if pt["obj_id"] == cur else 6
            center = QPointF(*xy)
            if pt["label"] == 1:
                p.setPen(QPen(QColor("white"), 2))
                p.setBrush(c)
                p.drawEllipse(center, r, r)
            else:
                p.setPen(QPen(c, 3))
                p.drawLine(QPointF(center.x() - r, center.y() - r), QPointF(center.x() + r, center.y() + r))
                p.drawLine(QPointF(center.x() - r, center.y() + r), QPointF(center.x() + r, center.y() - r))
            p.setPen(QPen(QColor("white")))
            p.drawText(QPointF(center.x() + r + 3, center.y() - r), str(pt["obj_id"]))
