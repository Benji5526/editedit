"""Main window: open a video, move through frames, click targets, save/load the points file."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget)

from src.points_io import PointStore, PointsError, load_points, save_points
from src.video_io import VideoInfo, is_hdr, probe_video

from .frame_server import FrameServer
from .video_view import VideoView, object_color

log = logging.getLogger("app")
VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts);;All files (*.*)"
CACHE_FRAMES = 40


class _ProbeSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _ProbeTask(QRunnable):
    """ffprobe incl. packet timestamps can take seconds on a long file: keep it off the UI thread."""

    def __init__(self, path: str):
        super().__init__()
        self.path, self.signals = path, _ProbeSignals()

    def run(self):
        try:
            self.signals.done.emit(probe_video(self.path))
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(str(e))


def default_points_path(video_path: str) -> str:
    return os.path.splitext(video_path)[0] + ".points.json"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Censor")
        self.resize(1280, 800)
        self.info: Optional[VideoInfo] = None
        self.server: Optional[FrameServer] = None
        self.store = PointStore()
        self.points_path: Optional[str] = None
        self.dirty = False
        self.frame_idx = -1
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._awaiting = False
        self._probe_task: Optional[_ProbeTask] = None
        self._play_step = 1
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play_tick)

        self.view = VideoView()
        self.view.set_provider(self._points_on_frame, lambda: self.store.current)
        self.view.clicked.connect(self.on_click)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._slider_changed)
        self.pos_label = QLabel("no video")
        self.pos_label.setMinimumWidth(260)

        self._nav_buttons = []

        def btn(text, tip, fn):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setEnabled(False)
            self._nav_buttons.append(b)
            return b

        self.play_btn = btn("Play", "Space", self.toggle_play)
        row = QHBoxLayout()
        for b in (btn("-1 s", "Ctrl+Left", lambda: self.step_seconds(-1)), btn("-10", "Shift+Left", lambda: self.step(-10)),
                  btn("<", "Left", lambda: self.step(-1)), self.play_btn, btn(">", "Right", lambda: self.step(1)),
                  btn("+10", "Shift+Right", lambda: self.step(10)), btn("+1 s", "Ctrl+Right", lambda: self.step_seconds(1))):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.pos_label)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.slider)
        lay.addLayout(row)
        self.setCentralWidget(central)
        self._build_side_panel()
        self._build_menu()
        self.statusBar().showMessage("Open a video to start (Ctrl+O). Left-click = this is the object, right-click = this is NOT.")

    # ------------------------------------------------------------------ UI construction
    def _act(self, text, shortcut, fn):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(fn)
        return a

    def _build_menu(self):
        act = self._act
        m = self.menuBar().addMenu("&File")
        m.addAction(act("Open video...", "Ctrl+O", self.open_video_dialog))
        m.addAction(act("Open points...", "Ctrl+L", self.open_points_dialog))
        m.addAction(act("Save points", "Ctrl+S", self.save_points))
        m.addAction(act("Save points as...", "Ctrl+Shift+S", self.save_points_as))
        m.addSeparator()
        m.addAction(act("Quit", "Ctrl+Q", self.close))
        e = self.menuBar().addMenu("&Edit")
        e.addAction(act("Undo last point on this frame", "U", self.undo_point))
        e.addAction(act("Delete this object's points on this frame", "X", self.clear_frame_points))
        # navigation shortcuts are window-level actions so they work wherever the focus is
        for key, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)),
                        ("Shift+Left", lambda: self.step(-10)), ("Shift+Right", lambda: self.step(10)),
                        ("Ctrl+Left", lambda: self.step_seconds(-1)), ("Ctrl+Right", lambda: self.step_seconds(1)),
                        ("Space", self.toggle_play)):
            self.addAction(act("", key, fn))

    def _build_side_panel(self):
        dock = QDockWidget("Targets", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        w = QWidget()
        v = QVBoxLayout(w)
        self.info_label = QLabel("No video loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.info_label)
        v.addWidget(QLabel("Objects (select the one you are marking):"))
        self.obj_list = QListWidget()
        self.obj_list.currentRowChanged.connect(self._object_selected)
        v.addWidget(self.obj_list, 1)
        for text, fn in (("New object", self.new_object), ("Delete object", self.delete_object),
                         ("Undo last point (U)", self.undo_point), ("Clear points on this frame (X)", self.clear_frame_points)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            v.addWidget(b)
        self.hint = QLabel("Tracking of an object starts at its earliest click: click it on the first frame where it is visible.")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        dock.setWidget(w)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._refresh_objects()

    # ------------------------------------------------------------------ dialogs (overridable for tests)
    def show_error(self, text: str):
        log.error(text)
        QMessageBox.critical(self, "Video Censor", text)

    def ask_save_changes(self) -> str:
        """'save' | 'discard' | 'cancel'"""
        B = QMessageBox.StandardButton
        r = QMessageBox.question(self, "Unsaved points", "The points have unsaved changes.", B.Save | B.Discard | B.Cancel)
        return {B.Save: "save", B.Discard: "discard"}.get(r, "cancel")

    def ask_confirm(self, text: str) -> bool:
        B = QMessageBox.StandardButton
        return QMessageBox.question(self, "Video Censor", text, B.Yes | B.No) == B.Yes

    def _maybe_save_before_leaving(self) -> bool:
        if not self.dirty:
            return True
        choice = self.ask_save_changes()
        if choice == "save":
            return self.save_points()
        return choice == "discard"

    # ------------------------------------------------------------------ opening a video
    def open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self.open_video(path)

    def open_video(self, path: str):
        if not self._maybe_save_before_leaving():
            return
        self._stop_play()
        self.statusBar().showMessage(f"Analysing {os.path.basename(path)} ...")
        self.setEnabled(False)
        task = _ProbeTask(path)
        task.signals.done.connect(self._video_probed)
        task.signals.failed.connect(self._probe_failed)
        self._probe_task = task            # keep the signal object alive until it fires
        QThreadPool.globalInstance().start(task)

    def _probe_failed(self, msg: str):
        self.setEnabled(True)
        self.statusBar().showMessage("Could not open the video.")
        self.show_error(f"Could not open the video:\n{msg}")

    def _video_probed(self, info: VideoInfo):
        self.setEnabled(True)
        if self.server is not None:
            self.server.close()
        self.info = info
        self.store = PointStore()
        self.points_path = default_points_path(info.path)
        self.dirty = False
        self._cache.clear()
        self.server = FrameServer(info)
        self.server.frame_ready.connect(self._frame_ready)
        self.server.failed.connect(self._frame_failed)
        self.view.set_video(info.width, info.height)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, info.nb_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        for b in self._nav_buttons:
            b.setEnabled(True)
        self._show_info()
        self._refresh_objects()
        self._update_title()
        msg = f"Opened {os.path.basename(info.path)}"
        if os.path.isfile(self.points_path):
            msg += self._try_load_points(self.points_path, quiet=True)
        self.statusBar().showMessage(msg)
        self.frame_idx = -1
        self.goto(0)

    def _show_info(self):
        i = self.info
        lines = [f"<b>{os.path.basename(i.path)}</b>", f"{i.width}x{i.height}, {i.fps:.3f} fps, {i.nb_frames} frames ({i.duration:.1f} s)",
                 f"codec {i.codec}, {i.pix_fmt}", f"audio: {i.audio_codec if i.has_audio else 'none'}"]
        warn = []
        if i.is_vfr:
            warn.append(f"Variable frame rate ({i.vfr_detail}). Processing rejects VFR unless allowed; times shown here are real packet times.")
        if is_hdr(i):
            warn.append("HDR / high bit depth: processing converts to 8-bit SDR.")
        if i.rotation:
            warn.append(f"Rotation metadata {i.rotation} deg is applied (frames shown as they will be processed).")
        if warn:
            lines.append("<span style='color:#d9822b'>" + "<br>".join(warn) + "</span>")
        self.info_label.setText("<br>".join(lines))

    # ------------------------------------------------------------------ frames
    def _points_on_frame(self):
        return [p for p in self.store.points if p["frame_idx"] == self.frame_idx]

    def goto(self, idx: int):
        if self.info is None:
            return
        idx = int(min(max(0, idx), self.info.nb_frames - 1))
        if idx == self.frame_idx:
            return
        self.frame_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._update_pos_label()
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.view.set_image(self._cache[idx])
            self._awaiting = False
        else:
            self._awaiting = True
            self.server.request(idx)
        self.view.update()

    def _frame_ready(self, idx: int, img):
        self._cache[idx] = img
        while len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)
        if idx == self.frame_idx:
            self.view.set_image(img)
            self._awaiting = False

    def _frame_failed(self, idx: int, msg: str):
        self._awaiting = False
        self._stop_play()
        self.statusBar().showMessage(f"Frame {idx}: {msg}")
        log.warning("frame %d failed: %s", idx, msg)

    def _slider_changed(self, v: int):
        self.goto(v)

    def step(self, n: int):
        if self.info is not None:
            self.goto(self.frame_idx + n)

    def step_seconds(self, s: float):
        if self.info is not None:
            self.step(int(round(s * self.info.fps)))

    def _update_pos_label(self):
        i = self.info
        self.pos_label.setText(f"frame {self.frame_idx}/{i.nb_frames - 1}   t = {i.timestamp_of(self.frame_idx):.3f} s")

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.info is None:
            return
        if self.playing:
            self._stop_play()
            return
        if self.frame_idx >= self.info.nb_frames - 1:
            self.goto(0)
        self.playing = True
        self.play_btn.setText("Pause")
        self._play_step = max(1, int(round(self.info.fps / 30.0)))
        self.timer.start(int(1000 * self._play_step / self.info.fps))

    def _stop_play(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("Play")

    def _play_tick(self):
        if self.info is None or self._awaiting:
            return                                  # never queue frames faster than they can be decoded
        if self.frame_idx >= self.info.nb_frames - 1:
            self._stop_play()
            return
        self.goto(self.frame_idx + self._play_step)

    # ------------------------------------------------------------------ points
    def on_click(self, fx: float, fy: float, label: int):
        if self.info is None:
            return
        self._stop_play()
        self.store.add(self.frame_idx, fx, fy, label)
        self._points_changed()

    def _points_changed(self):
        self.dirty = True
        self._refresh_objects()
        self._update_title()
        self.view.update()

    def new_object(self):
        oid = self.store.new_object()
        self._refresh_objects()
        self.statusBar().showMessage(f"New object {oid}: left-click it in the video")

    def delete_object(self):
        n = sum(1 for p in self.store.points if p["obj_id"] == self.store.current)
        if n and not self.ask_confirm(f"Delete object {self.store.current} and its {n} points?"):
            return
        removed = self.store.delete_object()
        if removed:
            self._points_changed()
        else:
            self._refresh_objects()

    def undo_point(self):
        if self.info is not None and self.store.undo(self.frame_idx):
            self._points_changed()

    def clear_frame_points(self):
        if self.info is not None and self.store.clear_frame(self.frame_idx):
            self._points_changed()

    def _object_selected(self, row: int):
        if row < 0:
            return
        oid = self.obj_list.item(row).data(Qt.ItemDataRole.UserRole)
        if oid != self.store.current:
            self.store.select(oid)
            self.view.update()

    def _refresh_objects(self):
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        ids = sorted(set(self.store.objects) | {p["obj_id"] for p in self.store.points})
        for oid in ids:
            pts = [p for p in self.store.points if p["obj_id"] == oid]
            first = min((p["frame_idx"] for p in pts), default=None)
            text = f"Object {oid}: {sum(1 for p in pts if p['label'] == 1)} +, {sum(1 for p in pts if p['label'] == 0)} -"
            text += f"  (starts at frame {first})" if first is not None else "  (no points yet)"
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, oid)
            pm = QPixmap(14, 14)
            pm.fill(object_color(oid))
            it.setIcon(QIcon(pm))
            self.obj_list.addItem(it)
            if oid == self.store.current:
                self.obj_list.setCurrentItem(it)
        self.obj_list.blockSignals(False)

    # ------------------------------------------------------------------ save / load points
    def save_points(self) -> bool:
        if self.info is None:
            return False
        if not self.points_path:
            return self.save_points_as()
        return self._write_points(self.points_path)

    def save_points_as(self) -> bool:
        if self.info is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save points", self.points_path or "", "Points (*.json)")
        return self._write_points(path) if path else False

    def _write_points(self, path: str) -> bool:
        probs = self.store.valid()
        if probs:
            self.show_error("Points not saved:\n" + "\n".join(probs) + "\nAdd a positive (left-click) point for each of them.")
            return False
        try:
            save_points(path, self.info, self.store.points)
        except OSError as e:
            self.show_error(f"Could not write {path}:\n{e}")
            return False
        self.points_path = path
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {len(self.store.points)} points to {path}")
        return True

    def open_points_dialog(self):
        if self.info is None:
            self.show_error("Open a video first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open points", os.path.dirname(self.info.path), "Points (*.json)")
        if path:
            self.statusBar().showMessage(self._try_load_points(path, quiet=False).strip(" -"))

    def _try_load_points(self, path: str, quiet: bool) -> str:
        try:
            prompts = load_points(path, self.info)
        except (PointsError, OSError, ValueError, KeyError) as e:
            if not quiet:
                self.show_error(f"Could not use this points file:\n{e}")
            return f" - points file ignored: {e}"
        if self.dirty and not self._maybe_save_before_leaving():
            return ""
        store = PointStore()
        for oid, frames in prompts.items():
            for fr, (pts, labs) in frames.items():
                for (x, y), lab in zip(pts, labs):
                    store.points.append({"obj_id": oid, "frame_idx": fr, "x": x, "y": y, "label": lab})
        ids = sorted({p["obj_id"] for p in store.points})
        if ids:
            store.objects, store.current = ids, ids[0]
        self.store, self.points_path, self.dirty = store, path, False
        self._refresh_objects()
        self._update_title()
        self.view.update()
        return f" - loaded {len(store.points)} points from {os.path.basename(path)}"

    def _update_title(self):
        name = os.path.basename(self.info.path) if self.info else ""
        self.setWindowTitle(f"{'* ' if self.dirty else ''}{name + ' - ' if name else ''}Video Censor")

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, e):
        if not self._maybe_save_before_leaving():
            e.ignore()
            return
        self._stop_play()
        if self.server is not None:
            self.server.close()
        e.accept()
