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
            saved = json.load(f)
            for k, v in saved.items():
                if hasattr(st, k) and k not in ("output", "resume", "max_frames", "settings_rev"):
                    setattr(st, k, v)
            if (saved.get("settings_rev") or 1) < 2:           # saved before the GPU options existed: take the new default once
                st.encoder = "auto"
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
        self.setWindowTitle("영상 처리 설정")
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
        form.addRow("SAM 2 체크포인트", path_row(self.ckpt, self._browse_ckpt))
        self.out = QLineEdit(st.output)
        form.addRow("결과 파일", path_row(self.out, self._browse_out))

        self.chunk = QSpinBox()
        self.chunk.setRange(10, 1000)
        self.chunk.setValue(st.chunk_size)
        self.chunk.setToolTip("청크 하나의 프레임 수입니다. 120-240을 권장합니다. 클수록 청크 경계는 줄지만 RAM/VRAM을 더 씁니다.")
        form.addRow("청크 크기 (프레임)", self.chunk)
        self.aw = QComboBox()
        for w in (1280, 1920, 2560, 3840):
            self.aw.addItem(str(w), w)
        self.aw.setCurrentText(str(st.analysis_width))
        form.addRow("분석 너비 (px)", self.aw)

        self.nude = QCheckBox("NudeNet으로도 탐지 (지원하는 종류만)")
        self.nude.setChecked(st.use_nudenet)
        form.addRow(self.nude)
        self.every = QSpinBox()
        self.every.setRange(1, 60)
        self.every.setValue(st.detect_every)
        form.addRow("  N프레임마다 탐지 (1 = 매 프레임)", self.every)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.05, 0.95)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(st.det_threshold)
        form.addRow("  임계값", self.thr)
        self.tiles = QSpinBox()
        self.tiles.setRange(1, 4)
        self.tiles.setValue(st.nudenet_tiles)
        self.tiles.setToolTip("1 = 프레임 전체만 봅니다. N이 1보다 크면 겹치는 N x N 조각도 추가로 살펴서 작은 영역을 더 높은 "
                              "해상도로 봅니다. 느려집니다: 프레임당 탐지를 약 1 + N x N 번 실행합니다.")
        form.addRow("  타일 (N x N 조각 추가, 1 = 끔)", self.tiles)
        self.nmodel = QLineEdit(st.nudenet_model)
        self.nmodel.setPlaceholderText("기본 내장 320 px 모델")
        self.nmodel.setToolTip("비워 두면 NudeNet에 들어 있는 기본 모델을 씁니다. 더 큰 모델(예: 640m)을 쓰려면 .onnx 파일을 "
                               "models/ 폴더에 받아 여기서 고르고, 아래에서 입력 크기를 맞춰 주세요.")
        form.addRow("  모델 파일 (.onnx, 선택)", path_row(self.nmodel, self._browse_nmodel))
        self.nres = QComboBox()
        for r in (320, 640):
            self.nres.addItem(str(r), r)
        self.nres.setCurrentText(str(st.nudenet_res) if st.nudenet_model else "320")
        self.nres.setToolTip("선택한 모델이 만들어진 입력 크기입니다. 모델 파일을 고르지 않으면 무시되고 320으로 고정됩니다.")
        form.addRow("  모델 입력 크기 (px)", self.nres)
        self.nmodel.textChanged.connect(lambda t: self.nres.setEnabled(bool(t.strip())))
        self.nres.setEnabled(bool(st.nudenet_model))
        self.ndev = QComboBox()
        for text, val in (("자동 (GPU가 되면 GPU, 아니면 CPU)", "auto"), ("GPU만 (안 되면 시작하지 않음)", "cuda"), ("CPU", "cpu")):
            self.ndev.addItem(text, val)
        self.ndev.setCurrentIndex(max(0, self.ndev.findData(st.nudenet_device)))
        self.ndev.setToolTip("NudeNet 탐지를 어디서 실행할지 정합니다. 큰 모델(640m)과 타일은 GPU에서 훨씬 빠릅니다. "
                             "자동일 때 GPU를 못 쓰면 처리 로그에 이유가 남습니다.")
        form.addRow("  탐지 실행 장치", self.ndev)

        self.fc = QCheckBox("안전 모드(fail-closed): 불확실한 프레임은 넓게 가리기")
        self.fc.setChecked(st.fail_closed)
        form.addRow(self.fc)
        self.fcm = QComboBox()
        self.fcm.addItems(["bbox", "full"])
        self.fcm.setCurrentText(st.fail_closed_mode)
        form.addRow("  불확실할 때 가릴 범위", self.fcm)
        form.addRow(QLabel("  안전 모드는 대상보다 넓게(최대 화면 전체) 가릴 수 있습니다. 모델이 아예 못 본 대상은 잡지 못합니다."))

        self.style = QComboBox()
        self.style.addItems(["pixelate", "blur", "solid"])
        self.style.setCurrentText(st.mosaic_style)
        self.style.setToolTip("pixelate(픽셀)와 solid(단색)는 내용을 완전히 없앱니다. blur(흐림)가 가장 약합니다.")
        form.addRow("모자이크 형태", self.style)
        self.block_auto = QCheckBox("자동: 영상 해상도에 맞춰 강도 선택")
        self.block_auto.setChecked(st.block_auto and info is not None)
        self.block_auto.setEnabled(info is not None)
        self.block_auto.setToolTip("긴 변의 약 1/60입니다 (1920 -> 32, 3840 -> 64). 어림값이므로 미리보기로 확인하세요.")
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
        form.addRow("모자이크 강도 (블록 크기 / 흐림, px)", sw)
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
        form.addRow("마스크 여백 (px)", self.pad)
        self.audio = QCheckBox("오디오 유지")
        self.audio.setChecked(st.keep_audio)
        form.addRow(self.audio)
        self.enc = QComboBox()
        self.enc.addItem("libx264")
        if nvenc_available():
            self.enc.addItem("h264_nvenc")
        self.enc.insertItem(0, "auto")
        self.enc.setCurrentText(st.encoder if self.enc.findText(st.encoder) >= 0 else "auto")
        self.enc.setToolTip("auto = GPU 인코더(h264_nvenc)가 이 PC에서 동작하면 GPU, 아니면 CPU(libx264). "
                            "GPU 인코더는 4K에서 약 4배 빠르고, 화질은 libx264와 같게 맞춰져 있습니다.")
        form.addRow("인코더", self.enc)
        self.test_n = QSpinBox()
        self.test_n.setRange(0, max(1, n_frames))
        self.test_n.setSpecialValueText("전체 프레임")
        self.test_n.setValue(0)
        form.addRow("앞의 N프레임만 처리 (시험 실행)", self.test_n)
        self.keep = QCheckBox("중간 청크 보관 (수정 후 빠른 재처리, 결과 영상 크기만큼 디스크 필요)")
        self.keep.setChecked(st.keep_chunks)
        form.addRow(self.keep)
        self.resume = QCheckBox("이 영상의 이전 실행을 이어하기 / 재사용 (포인트를 바꾸면 해당 청크부터 다시 처리, "
                                "모자이크 형태·강도·여백·인코더만 바꾸면 모자이크만 다시 적용)")
        self.resume.setToolTip("이전 실행의 중간 청크가 있어야 합니다 (\"중간 청크 보관\"). 그 밖의 설정"
                               "(청크 크기, 분석 너비, NudeNet, 안전 모드 켬/끔, 모델)은 같아야 합니다.")
        form.addRow(self.resume)

        self.error = QLabel("")
        self.error.setStyleSheet("color: #ff6b6b")
        self.error.setWordWrap(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("시작")
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
            self.preview.setText(f"미리보기를 표시할 수 없습니다: {e}")

    def _browse_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(self, "SAM 2 체크포인트", os.path.join(PROJECT_ROOT, "models"), "체크포인트 (*.pt)")
        if p:
            self.ckpt.setText(p)

    def _browse_nmodel(self):
        p, _ = QFileDialog.getOpenFileName(self, "NudeNet 모델", os.path.join(PROJECT_ROOT, "models"), "ONNX 모델 (*.onnx)")
        if p:
            self.nmodel.setText(p)

    def _browse_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "결과 파일", self.out.text(), "MP4 (*.mp4)")
        if p:
            self.out.setText(p)

    def settings(self) -> ProcessSettings:
        return ProcessSettings(
            sam2_checkpoint=self.ckpt.text().strip(), output=self.out.text().strip(), chunk_size=self.chunk.value(),
            analysis_width=int(self.aw.currentData()), use_nudenet=self.nude.isChecked(), detect_every=self.every.value(),
            det_threshold=round(self.thr.value(), 3), nudenet_tiles=self.tiles.value(), nudenet_model=self.nmodel.text().strip(),
            nudenet_res=int(self.nres.currentData()) if self.nmodel.text().strip() else 320, nudenet_device=self.ndev.currentData(),
            fail_closed=self.fc.isChecked(), fail_closed_mode=self.fcm.currentText(),
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
        probs.append("SAM 2 체크포인트를 찾을 수 없습니다. models/ 폴더에 받아서(models/README.md 참고) 선택하세요.")
    if not st.output:
        probs.append("결과 파일을 지정하세요.")
    elif os.path.abspath(st.output) == os.path.abspath(video_path):
        probs.append("결과 파일이 원본 영상과 같으면 안 됩니다.")
    elif not os.path.isdir(os.path.dirname(os.path.abspath(st.output))):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(st.output)), exist_ok=True)
        except OSError as e:
            probs.append(f"결과 폴더를 만들 수 없습니다: {e}")
    if not needs_sam2 and not st.use_nudenet:
        probs.append("할 일이 없습니다: 클릭한 대상이 없고 NudeNet도 꺼져 있습니다.")
    if st.use_nudenet:
        if st.nudenet_model and not os.path.isfile(st.nudenet_model):
            probs.append("NudeNet 모델 파일을 찾을 수 없습니다. 칸을 비워 기본 모델을 쓰거나, 있는 .onnx 파일을 선택하세요.")
        if not st.nudenet_model and st.nudenet_res != 320:
            probs.append("기본 NudeNet 모델은 320 px에서만 동작합니다. 다른 입력 크기를 쓰려면 모델 파일을 선택하세요.")
        if not (1 <= st.nudenet_tiles <= 4):
            probs.append("NudeNet 타일은 1-4여야 합니다.")
    if not (10 <= st.chunk_size <= 1000):
        probs.append("청크 크기는 10-1000프레임이어야 합니다.")
    return probs
