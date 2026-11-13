"""Processing settings dialog. Defaults are the safety-first ones; the last used values are remembered."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QSlider, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from src.mosaic import apply_mosaic, recommended_block
from src.video_io import nvenc_available, read_frame_at

from .job_runner import PROJECT_ROOT, ProcessSettings

LAST_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "config", "last_settings.json")


def find_default_checkpoint() -> str:
    found = sorted(glob.glob(os.path.join(PROJECT_ROOT, "models", "sam2.1_*.pt")))
    return found[0] if found else ""


def default_output(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(PROJECT_ROOT, "output", f"{stem}_censored.mp4")


def load_last_settings() -> ProcessSettings:
    st = ProcessSettings()
    try:
        with open(LAST_SETTINGS_PATH, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(st, k) and k not in ("output", "resume", "max_frames"):
                    setattr(st, k, v)
    except (OSError, ValueError):
        pass
    return st


def save_last_settings(st: ProcessSettings):
    try:
        os.makedirs(os.path.dirname(LAST_SETTINGS_PATH), exist_ok=True)
        with open(LAST_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(st), f, indent=1)
    except OSError:
        pass


class ProcessDialog(QDialog):
    def __init__(self, parent, video_path: str, needs_sam2: bool, n_frames: int, defaults: Optional[ProcessSettings] = None, info=None):
        super().__init__(parent)
        self.info = info
        self._frame = None
        self.setWindowTitle("Process video")
        self.video_path, self.needs_sam2, self.n_frames = video_path, needs_sam2, n_frames
        st = defaults or load_last_settings()
        if not st.sam2_checkpoint:
            st.sam2_checkpoint = find_default_checkpoint()
        st.output = default_output(video_path)
        form = QFormLayout()

        def path_row(edit: QLineEdit, browse):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(edit, 1)
            b = QPushButton("...")
            b.clicked.connect(browse)
            h.addWidget(b)
            return w

        self.ckpt = QLineEdit(st.sam2_checkpoint)
        self.ckpt.setEnabled(needs_sam2)
        form.addRow("SAM 2 checkpoint", path_row(self.ckpt, self._browse_ckpt))
        self.out = QLineEdit(st.output)
        form.addRow("Output file", path_row(self.out, self._browse_out))

        self.chunk = QSpinBox()
        self.chunk.setRange(10, 1000)
        self.chunk.setValue(st.chunk_size)
        self.chunk.setToolTip("Frames per chunk. 120-240 recommended; larger = fewer boundaries but more RAM/VRAM.")
        form.addRow("Chunk size (frames)", self.chunk)
        self.aw = QComboBox()
        for w in (1280, 1920, 2560, 3840):
            self.aw.addItem(str(w), w)
        self.aw.setCurrentText(str(st.analysis_width))
        form.addRow("Analysis width (px)", self.aw)

        self.nude = QCheckBox("Also detect with NudeNet (supported classes only)")
        self.nude.setChecked(st.use_nudenet)
        form.addRow(self.nude)
        self.every = QSpinBox()
        self.every.setRange(1, 60)
        self.every.setValue(st.detect_every)
        form.addRow("  detect every N frames (1 = all)", self.every)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.05, 0.95)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(st.det_threshold)
        form.addRow("  threshold", self.thr)
        self.tiles = QSpinBox()
        self.tiles.setRange(1, 4)
        self.tiles.setValue(st.nudenet_tiles)
        self.tiles.setToolTip("1 = whole frame only. N > 1 also scans an N x N grid of overlapping crops, so small regions get more "
                              "resolution. Slower: about 1 + N x N detector runs per frame.")
        form.addRow("  tiles (N x N extra crops, 1 = off)", self.tiles)
        self.nmodel = QLineEdit(st.nudenet_model)
        self.nmodel.setPlaceholderText("bundled 320 px model")
        self.nmodel.setToolTip("Leave empty for the model that ships with NudeNet. To use a larger one (e.g. 640m), download its "
                               ".onnx into models/ and select it here, then set its input size below.")
        form.addRow("  model file (.onnx, optional)", path_row(self.nmodel, self._browse_nmodel))
        self.nres = QComboBox()
        for r in (320, 640):
            self.nres.addItem(str(r), r)
        self.nres.setCurrentText(str(st.nudenet_res) if st.nudenet_model else "320")
        self.nres.setToolTip("Input size the selected model was exported for. Ignored (320) when no model file is selected.")
        form.addRow("  model input size (px)", self.nres)
        self.nmodel.textChanged.connect(lambda t: self.nres.setEnabled(bool(t.strip())))
        self.nres.setEnabled(bool(st.nudenet_model))

        self.fc = QCheckBox("Fail-closed: cover uncertain frames conservatively")
        self.fc.setChecked(st.fail_closed)
        form.addRow(self.fc)
        self.fcm = QComboBox()
        self.fcm.addItems(["bbox", "full"])
        self.fcm.setCurrentText(st.fail_closed_mode)
        form.addRow("  uncertain region", self.fcm)
        form.addRow(QLabel("  Fail-closed can cover more than the target (up to whole frames). It does not catch objects no model sees."))

        self.style = QComboBox()
        self.style.addItems(["pixelate", "blur", "solid"])
        self.style.setCurrentText(st.mosaic_style)
        self.style.setToolTip("pixelate and solid destroy the content; blur is the weakest")
        form.addRow("Mosaic style", self.style)
        self.block_auto = QCheckBox("Auto: choose the strength from the video resolution")
        self.block_auto.setChecked(st.block_auto and info is not None)
        self.block_auto.setEnabled(info is not None)
        self.block_auto.setToolTip("About 1/60 of the longer side (1920 -> 32, 3840 -> 64). A rule of thumb: check the preview.")
        form.addRow(self.block_auto)
        self.block = QSpinBox()
        self.block.setRange(4, 256)
        self.block.setValue(st.block)
        self.block_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_slider.setRange(4, 256)
        self.block_slider.setValue(st.block)
        sw = QWidget()
        sh = QHBoxLayout(sw)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.block_slider, 1)
        sh.addWidget(self.block)
        form.addRow("Mosaic strength (block size / blur, px)", sw)
        self.preview = QLabel("")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(180 if info is not None else 0)
        self.preview.setStyleSheet("background: #222; color: #bbb")
        form.addRow(self.preview)
        self.block_slider.valueChanged.connect(self.block.setValue)
        self.block.valueChanged.connect(self.block_slider.setValue)
        self.block.valueChanged.connect(self._update_preview)
        self.style.currentTextChanged.connect(self._update_preview)
        self.block_auto.toggled.connect(self._auto_toggled)
        self._auto_toggled(self.block_auto.isChecked())
        self.pad = QSpinBox()
        self.pad.setRange(0, 500)
        self.pad.setValue(int(st.mask_pad_px))
        form.addRow("Mask padding (px)", self.pad)
        self.audio = QCheckBox("Keep audio")
        self.audio.setChecked(st.keep_audio)
        form.addRow(self.audio)
        self.enc = QComboBox()
        self.enc.addItem("libx264")
        if nvenc_available():
            self.enc.addItem("h264_nvenc")
        self.enc.setCurrentText(st.encoder if self.enc.findText(st.encoder) >= 0 else "libx264")
        form.addRow("Encoder", self.enc)
        self.test_n = QSpinBox()
        self.test_n.setRange(0, max(1, n_frames))
        self.test_n.setSpecialValueText("all frames")
        self.test_n.setValue(0)
        form.addRow("Only the first N frames (test run)", self.test_n)
        self.keep = QCheckBox("Keep intermediate chunks (fast re-processing after re-clicks; needs disk space ~ output size)")
        self.keep.setChecked(st.keep_chunks)
        form.addRow(self.keep)
        self.resume = QCheckBox("Resume an interrupted job (same input and settings; changed points only redo the affected chunks)")
        form.addRow(self.resume)

        self.error = QLabel("")
        self.error.setStyleSheet("color: #c0392b")
        self.error.setWordWrap(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.error)
        lay.addWidget(bb)
        self.setMinimumWidth(560)

    def _auto_toggled(self, on: bool):
        if on and self.info is not None:
            self.block.setValue(recommended_block(self.info.width, self.info.height))
        self.block.setEnabled(not on)
        self.block_slider.setEnabled(not on)
        self._update_preview()

    def _update_preview(self, *_):
        """Show the chosen style and strength on a real frame of this video (an ellipse in the middle stands in for a target).
        Same code as the worker (src.mosaic.apply_mosaic), applied at the video's full resolution, then cropped for display."""
        if self.info is None:
            return
        try:
            if self._frame is None:
                self._frame = read_frame_at(self.info, min(self.info.nb_frames - 1, max(0, self.info.nb_frames // 10)))
            frame = self._frame.copy()
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), bool)
            cy, cx = h // 2, w // 2
            ry, rx = max(8, h // 8), max(8, h // 10)
            yy, xx = np.ogrid[:h, :w]
            mask[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = True
            apply_mosaic(frame, mask, self.style.currentText(), self.block.value())
            y0, y1, x0, x1 = max(0, cy - 2 * ry), min(h, cy + 2 * ry), max(0, cx - 3 * rx), min(w, cx + 3 * rx)
            crop = np.ascontiguousarray(frame[y0:y1, x0:x1][:, :, ::-1])             # decoded frames are BGR
            img = QImage(crop.data, crop.shape[1], crop.shape[0], crop.strides[0], QImage.Format.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(520, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:  # noqa: BLE001
            self.preview.setText(f"preview unavailable: {e}")

    def _browse_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(self, "SAM 2 checkpoint", os.path.join(PROJECT_ROOT, "models"), "Checkpoint (*.pt)")
        if p:
            self.ckpt.setText(p)

    def _browse_nmodel(self):
        p, _ = QFileDialog.getOpenFileName(self, "NudeNet model", os.path.join(PROJECT_ROOT, "models"), "ONNX model (*.onnx)")
        if p:
            self.nmodel.setText(p)

    def _browse_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "Output file", self.out.text(), "MP4 (*.mp4)")
        if p:
            self.out.setText(p)

    def settings(self) -> ProcessSettings:
        return ProcessSettings(
            sam2_checkpoint=self.ckpt.text().strip(), output=self.out.text().strip(), chunk_size=self.chunk.value(),
            analysis_width=int(self.aw.currentData()), use_nudenet=self.nude.isChecked(), detect_every=self.every.value(),
            det_threshold=round(self.thr.value(), 3), nudenet_tiles=self.tiles.value(), nudenet_model=self.nmodel.text().strip(),
            nudenet_res=int(self.nres.currentData()) if self.nmodel.text().strip() else 320, fail_closed=self.fc.isChecked(), fail_closed_mode=self.fcm.currentText(),
            block=self.block.value(), block_auto=self.block_auto.isChecked(), mosaic_style=self.style.currentText(), mask_pad_px=float(self.pad.value()), keep_audio=self.audio.isChecked(),
            max_frames=self.test_n.value() or None, resume=self.resume.isChecked(), encoder=self.enc.currentText(),
            keep_chunks=self.keep.isChecked())

    def problems(self) -> list:
        return validate_settings(self.settings(), self.video_path, self.needs_sam2)

    def _accept(self):
        probs = self.problems()
        if probs:
            self.error.setText("\n".join(probs))
            return
        save_last_settings(self.settings())
        self.accept()


def validate_settings(st: ProcessSettings, video_path: str, needs_sam2: bool) -> list:
    probs = []
    if needs_sam2 and not os.path.isfile(st.sam2_checkpoint):
        probs.append("SAM 2 checkpoint not found. Download one into models/ (see models/README.md) and select it.")
    if not st.output:
        probs.append("Choose an output file.")
    elif os.path.abspath(st.output) == os.path.abspath(video_path):
        probs.append("The output must not be the input file.")
    elif not os.path.isdir(os.path.dirname(os.path.abspath(st.output))):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(st.output)), exist_ok=True)
        except OSError as e:
            probs.append(f"Cannot create the output folder: {e}")
    if not needs_sam2 and not st.use_nudenet:
        probs.append("Nothing to do: no targets were clicked and NudeNet is off.")
    if st.use_nudenet:
        if st.nudenet_model and not os.path.isfile(st.nudenet_model):
            probs.append("NudeNet model file not found. Clear the field to use the bundled model, or select an existing .onnx file.")
        if not st.nudenet_model and st.nudenet_res != 320:
            probs.append("The bundled NudeNet model only works at 320 px. Select a model file for another input size.")
        if not (1 <= st.nudenet_tiles <= 4):
            probs.append("NudeNet tiles must be 1-4.")
    if not (10 <= st.chunk_size <= 1000):
        probs.append("Chunk size must be 10-1000 frames.")
    return probs
