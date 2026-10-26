"""Background frame provider for the preview. Never blocks the UI thread.

Frames come from FFmpeg (same decoder, rotation handling and numbering as the processing pipeline),
downscaled by FFmpeg to the preview size. Stepping forward / playback reuses one sequential reader;
jumps re-open the reader with a timestamp seek. Only the latest request is served (older ones are dropped).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from src.video_io import FrameReader, VideoError, VideoInfo

MAX_SKIP = 30   # frames to read-and-discard instead of re-seeking


def preview_size(info: VideoInfo, max_w: int = 1280):
    w = min(info.width, max_w)
    h = max(1, int(round(info.height * w / info.width)))
    return w, h


def bgr_to_qimage(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


class FrameServer(QObject):
    frame_ready = Signal(int, QImage)
    failed = Signal(int, str)

    def __init__(self, info: VideoInfo, max_w: int = 1280):
        super().__init__()
        self.info = info
        self.size = preview_size(info, max_w)
        self._cv = threading.Condition()
        self._want: Optional[int] = None
        self._stop = False
        self._reader: Optional[FrameReader] = None
        self._pos = -1
        self._thread = threading.Thread(target=self._run, name="FrameServer", daemon=True)
        self._thread.start()

    def request(self, idx: int):
        with self._cv:
            self._want = int(idx)
            self._cv.notify()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=5)
        self._drop_reader()

    # ---- worker thread
    def _drop_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._pos = -1

    def _run(self):
        while True:
            with self._cv:
                while self._want is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                idx, self._want = self._want, None
            try:
                arr = self._fetch(idx)
                if arr is None:
                    self.failed.emit(idx, f"could not read frame {idx}")
                else:
                    self.frame_ready.emit(idx, bgr_to_qimage(arr))
            except (VideoError, OSError, ValueError) as e:
                self._drop_reader()
                self.failed.emit(idx, str(e))

    def _fetch(self, idx: int) -> Optional[np.ndarray]:
        if not (0 <= idx < self.info.nb_frames):
            return None
        if self._reader is not None and self._pos < idx <= self._pos + MAX_SKIP:
            arr = None
            while self._pos < idx:
                arr = self._reader.read()
                if arr is None:
                    self._drop_reader()
                    break
                self._pos += 1
            if arr is not None:
                return arr
        self._drop_reader()
        self._reader = FrameReader(self.info, start_frame=idx, scale_to=self.size)
        arr = self._reader.read()
        if arr is None:
            self._drop_reader()
            return None
        self._pos = idx
        return arr
