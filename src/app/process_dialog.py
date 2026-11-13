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
                               QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

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

        def combo(items, current):
            """Combo box that shows Korean text and keeps the program's own value as item data."""
            c = QComboBox()
            for text, val in items:
                c.addItem(text, val)
            c.setCurrentIndex(max(0, c.findData(current)))
            return c

        def section(title):
            lab = QLabel(title)
            lab.setStyleSheet("font-weight: bold; padding-top: 6px")
            form.addRow(lab)

        section("파일")
        self.ckpt = QLineEdit(st.sam2_checkpoint)
        self.ckpt.setEnabled(needs_sam2)
        self.ckpt.setToolTip("클릭한 대상을 프레임마다 따라가는 추적 AI(SAM 2)의 모델 파일입니다. 이름에 small이 들어간 파일은 빠르고, "
                             "base_plus는 느리지만 경계를 더 정확하게 잡습니다. 클릭한 대상이 없으면 쓰지 않습니다.")
        form.addRow("추적 모델 파일 (SAM 2)", path_row(self.ckpt, self._browse_ckpt))
        self.out = QLineEdit(st.output)
        self.out.setToolTip("모자이크가 적용된 영상을 저장할 위치입니다. 원본 영상은 바뀌지 않습니다.")
        form.addRow("결과 영상 저장 위치", path_row(self.out, self._browse_out))

        section("모자이크 모양")
        self.style = combo((("픽셀 모자이크 (네모 칸으로 뭉개기)", "pixelate"), ("흐림 (블러, 가장 약함)", "blur"),
                            ("단색으로 덮기 (검정)", "solid")), st.mosaic_style)
        self.style.setToolTip("픽셀 모자이크와 단색 덮기는 원래 내용을 알아볼 수 없게 없앱니다. 흐림은 윤곽이 남아 가장 약합니다.")
        form.addRow("가리는 방식", self.style)
        self.block_auto = QCheckBox("셀 크기 자동 (영상 해상도에 맞춰 정함)")
        self.block_auto.setChecked(st.block_auto and info is not None)
        self.block_auto.setEnabled(info is not None)
        self.block_auto.setToolTip("화면 긴 변의 약 1/60로 정합니다 (풀HD 1920 → 32, 4K 3840 → 64). 어림값이므로 아래 미리보기로 "
                                   "확인하세요. 아래 값을 직접 바꾸면 자동은 꺼집니다.")
        form.addRow(self.block_auto)
        self.block = QSpinBox()
        self.block.setRange(4, 256)
        self.block.setSingleStep(2)
        self.block.setSuffix(" px")
        self.block.setValue(st.block)
        self.block_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_slider.setRange(4, 256)
        self.block_slider.setValue(st.block)
        tip = ("모자이크 네모 한 칸의 한 변 길이입니다 (원본 영상의 픽셀 기준, 4~256). 클수록 칸이 커져서 더 거칠게 가려집니다. "
               "'흐림'을 고르면 흐림의 세기로 쓰입니다. 대상마다 다르게 하려면 오른쪽 '속성' 창에서 대상별로 정할 수 있습니다.")
        self.block.setToolTip(tip)
        self.block_slider.setToolTip(tip)
        sw = QWidget()
        sh = QHBoxLayout(sw)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(QLabel("작게"))
        sh.addWidget(self.block_slider, 1)
        sh.addWidget(QLabel("크게"))
        sh.addWidget(self.block)
        form.addRow("모자이크 셀 크기", sw)
        self.preview = QLabel("")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(180 if info is not None else 0)
        self.preview.setStyleSheet("background: #222; color: #bbb")
        self.preview.setToolTip("이 영상의 실제 한 장면에 지금 설정을 적용한 모습입니다 (가운데 타원이 대상 자리).")
        form.addRow(self.preview)
        self._setting_auto = False
        self.block_slider.valueChanged.connect(self.block.setValue)
        self.block.valueChanged.connect(self.block_slider.setValue)
        self.block.valueChanged.connect(self._block_changed)
        self.style.currentIndexChanged.connect(self._update_preview)
        self.block_auto.toggled.connect(self._auto_toggled)
        self._auto_toggled(self.block_auto.isChecked())
        self.pad = QSpinBox()
        self.pad.setRange(0, 500)
        self.pad.setSuffix(" px")
        self.pad.setValue(int(st.mask_pad_px))
        self.pad.setToolTip("추적으로 찾은 영역보다 바깥으로 이만큼 더 넓게 가립니다 (원본 영상의 픽셀 기준). "
                            "가장자리가 비치면 늘리세요.")
        form.addRow("가리는 영역 여백", self.pad)

        section("자동 탐지 (NudeNet)")
        self.nude = QCheckBox("노출 부위를 자동으로 찾아서 함께 가리기")
        self.nude.setChecked(st.use_nudenet)
        self.nude.setToolTip("탐지 AI(NudeNet)가 프레임마다 노출 부위를 찾아 네모 영역으로 가립니다. 이 모델이 아는 종류만 찾고, "
                             "손이나 다른 물체를 잘못 잡을 수도 있습니다. 클릭한 대상의 추적과 함께 쓸 수 있습니다.")
        form.addRow(self.nude)
        self.every = QSpinBox()
        self.every.setRange(1, 60)
        self.every.setValue(st.detect_every)
        self.every.setToolTip("1이면 모든 프레임에서 탐지합니다. 2 이상이면 그만큼 건너뛰어 빨라지지만, 빠르게 움직이는 장면에서 "
                              "놓칠 수 있습니다.")
        form.addRow("  탐지 간격 (1 = 모든 프레임)", self.every)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.05, 0.95)
        self.thr.setSingleStep(0.05)
        self.thr.setValue(st.det_threshold)
        self.thr.setToolTip("탐지 AI가 매긴 확신 점수(0~1)가 이 값 이상인 것만 가립니다. 낮추면 더 많이 잡지만 잘못 잡는 것도 늘고, "
                            "높이면 잘못 잡는 것은 줄지만 놓치는 것이 늘어납니다.")
        form.addRow("  탐지 기준 점수 (낮을수록 많이 잡음)", self.thr)
        self.tiles = QSpinBox()
        self.tiles.setRange(1, 4)
        self.tiles.setValue(st.nudenet_tiles)
        self.tiles.setToolTip("1이면 화면 전체만 한 번 봅니다. 2 이상이면 화면을 N x N으로 나눈 부분들도 확대해서 추가로 살펴, "
                              "작게 찍힌 대상을 더 잘 찾습니다. 프레임당 탐지를 약 1 + N x N 번 하므로 그만큼 느려집니다.")
        form.addRow("  화면 분할 탐지 (1 = 사용 안 함)", self.tiles)
        self.nmodel = QLineEdit(st.nudenet_model)
        self.nmodel.setPlaceholderText("비워 두면 기본 모델 (320 px)")
        self.nmodel.setToolTip("비워 두면 NudeNet에 들어 있는 기본 모델을 씁니다. 더 정확한 큰 모델(예: 640m)을 쓰려면 .onnx 파일을 "
                               "models 폴더에 넣고 여기서 고른 뒤, 아래 입력 크기를 그 모델에 맞추세요.")
        form.addRow("  탐지 모델 파일 (.onnx)", path_row(self.nmodel, self._browse_nmodel))
        self.nres = QComboBox()
        for r in (320, 640):
            self.nres.addItem(f"{r} px", r)
        self.nres.setCurrentIndex(max(0, self.nres.findData(st.nudenet_res if st.nudenet_model else 320)))
        self.nres.setToolTip("고른 모델이 학습된 입력 크기입니다 (640m 모델은 640). 모델 파일을 고르지 않으면 320으로 고정됩니다.")
        form.addRow("  탐지 모델 입력 크기", self.nres)
        self.nmodel.textChanged.connect(lambda t: self.nres.setEnabled(bool(t.strip())))
        self.nres.setEnabled(bool(st.nudenet_model))
        self.ndev = combo((("자동 (그래픽카드를 쓸 수 있으면 그래픽카드, 아니면 CPU)", "auto"),
                           ("그래픽카드(GPU)만 (쓸 수 없으면 시작하지 않음)", "cuda"), ("CPU만", "cpu")), st.nudenet_device)
        self.ndev.setToolTip("탐지를 어디에서 계산할지 정합니다. 큰 모델(640m)과 화면 분할 탐지는 그래픽카드에서 훨씬 빠릅니다. "
                             "자동일 때 그래픽카드를 못 쓰면 처리 기록에 이유가 남습니다.")
        form.addRow("  탐지 계산 장치", self.ndev)

        section("안전 모드")
        self.fc = QCheckBox("확실하지 않은 프레임은 넓게 가리기")
        self.fc.setChecked(st.fail_closed)
        self.fc.setToolTip("놓치는 것보다 많이 가리는 쪽을 택합니다. 추적이 끊기거나 탐지되던 것이 갑자기 사라지면, 마지막으로 있던 자리를 넓혀서 계속 가립니다. "
                           "대상보다 넓게 가려질 수 있습니다. AI가 처음부터 못 본 대상은 이 기능으로도 가릴 수 없습니다.")
        form.addRow(self.fc)
        self.fcm = combo((("마지막으로 있던 자리 주변을 넓게", "bbox"), ("화면 전체", "full")), st.fail_closed_mode)
        form.addRow("  확실하지 않을 때 가릴 범위", self.fcm)

        section("처리 방식")
        self.chunk = QSpinBox()
        self.chunk.setRange(10, 1000)
        self.chunk.setValue(st.chunk_size)
        self.chunk.setToolTip("긴 영상을 이 프레임 수만큼의 조각으로 나눠 차례로 처리합니다. 120~240을 권장합니다. "
                              "크게 하면 조각 사이 경계가 줄지만 메모리(RAM과 그래픽 메모리)를 더 씁니다.")
        form.addRow("조각 크기 (한 번에 처리할 프레임 수)", self.chunk)
        self.aw = QComboBox()
        for w in (1280, 1920, 2560, 3840):
            self.aw.addItem(f"{w} px", w)
        self.aw.setCurrentIndex(max(0, self.aw.findData(st.analysis_width)))
        self.aw.setToolTip("대상을 찾고 추적할 때 쓰는 축소 화면의 가로 크기입니다. 클수록 작은 대상을 잘 찾지만 느려지고 메모리를 더 씁니다. "
                           "결과 영상의 해상도는 이 값과 상관없이 원본 그대로입니다.")
        form.addRow("분석용 화면 가로 크기", self.aw)
        self.audio = QCheckBox("소리(오디오)를 결과 영상에 그대로 넣기")
        self.audio.setChecked(st.keep_audio)
        form.addRow(self.audio)
        enc_items = [("자동 (그래픽카드를 쓸 수 있으면 그래픽카드)", "auto"), ("CPU로 저장 (libx264, 느림)", "libx264")]
        if nvenc_available():
            enc_items.append(("그래픽카드로 저장 (h264_nvenc, 빠름)", "h264_nvenc"))
        self.enc = combo(enc_items, st.encoder if st.encoder in [v for _, v in enc_items] else "auto")
        self.enc.setToolTip("결과 영상을 압축해 저장하는 방식(인코더)입니다. 그래픽카드 방식은 4K에서 약 4배 빠르고, "
                            "화질은 CPU 방식과 같게 맞춰져 있습니다.")
        form.addRow("영상 저장 방식 (인코더)", self.enc)
        self.test_n = QSpinBox()
        self.test_n.setRange(0, max(1, n_frames))
        self.test_n.setSpecialValueText("영상 전체")
        self.test_n.setValue(0)
        self.test_n.setToolTip("설정을 시험해 볼 때, 영상 앞부분의 이 프레임 수만 처리합니다. 0이면 영상 전체를 처리합니다.")
        form.addRow("앞부분만 시험 처리 (프레임 수)", self.test_n)
        self.keep = QCheckBox("중간 조각 파일 보관 (고친 뒤 다시 처리할 때 빨라짐)")
        self.keep.setChecked(st.keep_chunks)
        self.keep.setToolTip("처리하면서 만든 조각 파일과 추적 결과를 지우지 않고 둡니다. 클릭 지점이나 모자이크 설정을 고친 뒤 다시 처리할 때 "
                             "바뀐 부분만 다시 하므로 빨라집니다. 결과 영상 크기만큼 디스크 공간을 더 씁니다. '지난 실행의 마스크 보기'에도 필요합니다.")
        form.addRow(self.keep)
        self.resume = QCheckBox("이 영상의 이전 처리 결과 이어서 쓰기 (바뀐 부분만 다시 처리)")
        self.resume.setToolTip("이전에 '중간 조각 파일 보관'을 켜고 처리한 결과가 있어야 합니다. 클릭 지점을 바꿨으면 그 조각부터 다시 처리하고, "
                               "가리는 방식·셀 크기·여백·저장 방식만 바꿨으면 모자이크만 다시 입힙니다. 그 밖의 설정"
                               "(조각 크기, 분석용 화면 크기, 자동 탐지, 안전 모드, 모델)은 이전과 같아야 합니다.")
        form.addRow(self.resume)

        self.error = QLabel("")
        self.error.setStyleSheet("color: #ff6b6b")
        self.error.setWordWrap(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("시작")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        body = QWidget()
        body.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.error)
        lay.addWidget(bb)
        self.setMinimumWidth(760)
        self.resize(800, 900)

    def _auto_toggled(self, on: bool):
        if on and self.info is not None:
            self._setting_auto = True
            try:
                self.block.setValue(recommended_block(self.info.width, self.info.height))
            finally:
                self._setting_auto = False
        self._update_preview()

    def _block_changed(self, *_):
        """The cell size was changed. By hand (slider or number) = the user wants this size, so 'automatic' goes off."""
        if not self._setting_auto and self.block_auto.isChecked():
            self.block_auto.setChecked(False)           # -> _auto_toggled -> preview
        else:
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
            apply_mosaic(frame, mask, self.style.currentData(), self.block.value())
            y0, y1, x0, x1 = max(0, cy - 2 * ry), min(h, cy + 2 * ry), max(0, cx - 3 * rx), min(w, cx + 3 * rx)
            crop = np.ascontiguousarray(frame[y0:y1, x0:x1][:, :, ::-1])             # decoded frames are BGR
            img = QImage(crop.data, crop.shape[1], crop.shape[0], crop.strides[0], QImage.Format.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(520, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:  # noqa: BLE001
            self.preview.setText(f"미리보기를 표시할 수 없습니다: {e}")

    def _browse_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(self, "추적 모델 파일 (SAM 2) 고르기", os.path.join(PROJECT_ROOT, "models"), "SAM 2 모델 파일 (*.pt)")
        if p:
            self.ckpt.setText(p)

    def _browse_nmodel(self):
        p, _ = QFileDialog.getOpenFileName(self, "탐지 모델 파일 (NudeNet) 고르기", os.path.join(PROJECT_ROOT, "models"), "ONNX 모델 파일 (*.onnx)")
        if p:
            self.nmodel.setText(p)

    def _browse_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "결과 영상 저장 위치", self.out.text(), "MP4 영상 (*.mp4)")
        if p:
            self.out.setText(p)

    def settings(self) -> ProcessSettings:
        return ProcessSettings(
            sam2_checkpoint=self.ckpt.text().strip(), output=self.out.text().strip(), chunk_size=self.chunk.value(),
            analysis_width=int(self.aw.currentData()), use_nudenet=self.nude.isChecked(), detect_every=self.every.value(),
            det_threshold=round(self.thr.value(), 3), nudenet_tiles=self.tiles.value(), nudenet_model=self.nmodel.text().strip(),
            nudenet_res=int(self.nres.currentData()) if self.nmodel.text().strip() else 320, nudenet_device=self.ndev.currentData(),
            fail_closed=self.fc.isChecked(), fail_closed_mode=self.fcm.currentData(),
            block=self.block.value(), block_auto=self.block_auto.isChecked(), mosaic_style=self.style.currentData(), mask_pad_px=float(self.pad.value()), keep_audio=self.audio.isChecked(),
            max_frames=self.test_n.value() or None, resume=self.resume.isChecked(), encoder=self.enc.currentData(),
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
        probs.append("추적 모델 파일(SAM 2)을 찾을 수 없습니다. models 폴더에 내려받은 뒤(models/README.md 참고) 선택하세요.")
    if not st.output:
        probs.append("결과 영상을 저장할 위치를 정하세요.")
    elif os.path.abspath(st.output) == os.path.abspath(video_path):
        probs.append("결과 영상을 원본 영상과 같은 파일로 저장할 수 없습니다. 다른 이름을 정하세요.")
    elif not os.path.isdir(os.path.dirname(os.path.abspath(st.output))):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(st.output)), exist_ok=True)
        except OSError as e:
            probs.append(f"결과 폴더를 만들 수 없습니다: {e}")
    if not needs_sam2 and not st.use_nudenet:
        probs.append("가릴 것이 없습니다. 영상에서 대상을 클릭하거나 자동 탐지를 켜세요.")
    if st.use_nudenet:
        if st.nudenet_model and not os.path.isfile(st.nudenet_model):
            probs.append("탐지 모델 파일을 찾을 수 없습니다. 칸을 비워 기본 모델을 쓰거나, 실제로 있는 .onnx 파일을 선택하세요.")
        if not st.nudenet_model and st.nudenet_res != 320:
            probs.append("기본 탐지 모델은 입력 크기 320 px에서만 동작합니다. 다른 크기를 쓰려면 탐지 모델 파일을 선택하세요.")
        if not (1 <= st.nudenet_tiles <= 4):
            probs.append("화면 분할 탐지는 1~4 사이여야 합니다.")
    if not (10 <= st.chunk_size <= 1000):
        probs.append("조각 크기는 10~1000프레임 사이여야 합니다.")
    return probs
