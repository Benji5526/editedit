"""Main window: open a video, move through frames, click targets, save/load the points file."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Optional

import dataclasses
import hashlib
import json
import time

import cv2
import numpy as np

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QToolBar,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from src.corrections import save_correction
from src.points_io import (OBJECT_STYLES, PointStore, PointsError, load_mask_prompts, load_object_styles, load_points, mask_digest,
                           save_points)
from src.review import ReviewError
from src.review_data import ReviewData, load_review_data
from src.video_io import VideoInfo, is_hdr, probe_video

from .cut_dialog import CutDialog
from .frame_server import FrameServer
from src.job_manager import JobDB, db_path_for
from src.mask_cache import JobMasks

from .job_runner import PROJECT_ROOT, JobRunner, ProcessSettings, build_args, job_dir_for_video
from .jobs_dialog import JobsDialog
from .mask_preview import MaskPreview, preview_args
from .process_dialog import ProcessDialog, find_default_checkpoint, load_last_settings, validate_settings
from .process_panel import STATUS_TEXT, ProcessPanel
from .review_panel import FlagSlider, ReviewPanel
from .timeline import RunSummaryLoader, ThumbnailLoader, TimelineView
from .video_view import VideoView, mask_image, object_color

log = logging.getLogger("app")
VIDEO_FILTER = "영상 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts);;모든 파일 (*.*)"
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
        self.runner = JobRunner(self)
        self.plan_runner = JobRunner(self)
        self._plan: Optional[dict] = None
        self._plan_settings: Optional[ProcessSettings] = None
        self.review: Optional[ReviewData] = None
        self.resumable_job: Optional[dict] = None
        self.job_running = False
        self.last_settings: Optional[ProcessSettings] = None
        self.last_result: Optional[dict] = None
        # mask overlays: live preview of the clicked frame (SAM 2 in a helper process) and the kept masks of the last run
        self.preview = MaskPreview(self)
        self.preview_command = None            # tests: replaces the worker command line of the preview process
        self._live_masks: dict = {}            # frame -> (signature of the clicks on it, {obj_id: mask}, {obj_id: reason skipped})
        self._preview_reqs: dict = {}          # request id -> (frame, signature)
        self._preview_asked = None             # (frame, signature) of the newest request
        self._job_masks: Optional[JobMasks] = None
        # brush: hand-drawn masks. _brush_base remembers what a mask looked like before the first stroke (for the correction record)
        self._brush_base: dict = {}            # (obj_id, frame) -> (mask at edit size or None, source)
        self._brush_undo: list = []            # [(obj_id, frame, mask before the stroke or None)]
        self._stroke_key = None
        self._stroke_mask: Optional[np.ndarray] = None
        self.corrections_saved: list = []      # paths of the correction records written by the last save
        self._compare: Optional[dict] = None   # before/after comparison of a correction (see start_compare)

        self.view = VideoView()
        self.view.set_provider(self._points_on_frame, lambda: self.store.current)
        self.view.clicked.connect(self.on_click)
        self.view.stroke_started.connect(self._stroke_started)
        self.view.stroke.connect(self._stroke_segment)
        self.view.stroke_finished.connect(self._stroke_finished)

        self.slider = FlagSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._slider_changed)
        self.pos_label = QLabel("영상 없음")
        self.pos_label.setObjectName("timecode")
        self.pos_label.setMinimumWidth(260)

        self._nav_buttons = []

        def btn(text, tip, fn):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setEnabled(False)
            self._nav_buttons.append(b)
            return b

        self.play_btn = btn("재생", "Space", self.toggle_play)
        self.play_btn.setObjectName("play")
        row = QHBoxLayout()                              # transport: timecode on the left, controls in the middle
        row.addWidget(self.pos_label)
        row.addStretch(1)
        for b in (btn("-1초", "Ctrl+Left", lambda: self.step_seconds(-1)), btn("-10", "Shift+Left", lambda: self.step(-10)),
                  btn("<", "Left", lambda: self.step(-1)), self.play_btn, btn(">", "Right", lambda: self.step(1)),
                  btn("+10", "Shift+Right", lambda: self.step(10)), btn("+1초", "Ctrl+Right", lambda: self.step_seconds(1))):
            row.addWidget(b)
        row.addStretch(1)
        balance = QWidget()                              # same width as the timecode so the controls stay centred
        balance.setMinimumWidth(260)
        row.addWidget(balance)

        self.timeline = QFrame()                         # bottom strip: the frame slider with the review-flag ticks
        self.timeline.setObjectName("timeline")
        tl = QVBoxLayout(self.timeline)
        tl.setContentsMargins(10, 6, 10, 10)
        self.timeline_title = QLabel("타임라인")
        tl.addWidget(self.timeline_title)
        tl.addWidget(self.slider)
        self.timeline_view = TimelineView()              # thumbnails, one track per object, detections, review flags
        self.timeline_view.setEnabled(False)
        self.timeline_view.seek_requested.connect(self.goto)
        tl.addWidget(self.timeline_view)
        self._thumb_loader: Optional[ThumbnailLoader] = None
        self._summary_loader: Optional[RunSummaryLoader] = None

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.addWidget(self.view, 1)
        lay.addLayout(row)
        lay.addWidget(self.timeline)
        self.setCentralWidget(central)
        self._build_side_panel()
        self._build_process_dock()
        self._build_review_dock()
        self._build_menu()
        self.plan_runner.progress.connect(self._on_plan_event)
        self.plan_runner.job_finished.connect(self._on_plan_finished)
        self.runner.progress.connect(self._on_job_progress)
        self.runner.log_line.connect(self.panel.add_log)
        self.runner.job_finished.connect(self._on_job_finished)
        self.preview.masks_ready.connect(self._on_preview_masks)
        self.preview.request_failed.connect(self._on_preview_failed)
        self.preview.state_changed.connect(self._on_preview_state)
        self.preview.track_ready.connect(self._on_track_ready)
        self.statusBar().showMessage("영상을 열어 시작하세요 (Ctrl+O). 왼쪽 클릭 = 이것이 대상, 오른쪽 클릭 = 이것은 대상이 아님.")

    # ------------------------------------------------------------------ UI construction
    def _act(self, text, shortcut, fn):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(fn)
        return a

    def _build_menu(self):
        act = self._act
        m = self.menuBar().addMenu("파일(&F)")
        self.act_open_video = act("영상 열기...", "Ctrl+O", self.open_video_dialog)
        self.act_open_points = act("포인트 열기...", "Ctrl+L", self.open_points_dialog)
        m.addAction(self.act_open_video)
        m.addAction(self.act_open_points)
        self.act_open_review = act("검수 로그 열기...", "Ctrl+Shift+O", self.open_review_dialog)
        m.addAction(self.act_open_review)
        m.addAction(act("포인트 저장", "Ctrl+S", self.save_points))
        m.addAction(act("포인트 다른 이름으로 저장...", "Ctrl+Shift+S", self.save_points_as))
        m.addSeparator()
        m.addAction(act("종료", "Ctrl+Q", self.close))
        pm = self.menuBar().addMenu("처리(&P)")
        self.act_start = act("처리 시작...", "Ctrl+R", self.start_processing_dialog)
        self.act_cancel = act("처리 취소", "Ctrl+.", self.cancel_processing)
        self.act_cancel.setEnabled(False)
        pm.addAction(self.act_start)
        pm.addAction(self.act_cancel)
        self.act_reprocess = act("수정한 포인트로 다시 처리...", "Ctrl+Shift+R", self.reprocess)
        self.act_cleanup = act("중간 청크 삭제...", None, self.delete_intermediate)
        self.act_resume_job = act("중단된 작업 이어하기...", None, self.reprocess)
        self.act_resume_job.setEnabled(False)
        self.act_history = act("작업 기록...", None, self.show_job_history)
        pm.addAction(self.act_reprocess)
        pm.addAction(self.act_resume_job)
        pm.addAction(self.act_history)
        pm.addAction(self.act_cleanup)
        vm = self.menuBar().addMenu("보기(&V)")
        self.act_live = QAction("클릭하면서 마스크 미리보기 (SAM 2)", self)
        self.act_live.setCheckable(True)
        self.act_live.setShortcut(QKeySequence("P"))
        self.act_live.setToolTip("처리하기 전에, 현재 프레임의 클릭으로 SAM 2가 무엇을 잡는지 보여 줍니다.")
        self.act_live.toggled.connect(self.toggle_live_preview)
        self.act_last = QAction("지난 실행의 마스크 보기", self)
        self.act_last.setCheckable(True)
        self.act_last.setShortcut(QKeySequence("K"))
        self.act_last.setToolTip("지난 실행이 프레임마다 남긴 추적 마스크, 탐지 박스, 안전 모드 영역을 보여 줍니다.")
        self.act_last.toggled.connect(self.toggle_last_masks)
        vm.addAction(self.act_live)
        vm.addAction(self.act_last)
        self.act_brush = QAction("브러시로 마스크 수정", self)
        self.act_brush.setCheckable(True)
        self.act_brush.setShortcut(QKeySequence("G"))
        self.act_brush.setToolTip("선택한 객체의 마스크를 이 프레임에서 직접 칠해 고칩니다. 왼쪽 드래그 = 칠하기, 오른쪽 드래그 = 지우기.")
        self.act_brush.toggled.connect(self.toggle_brush)
        self.act_undo_stroke = self._act("마지막 붓질 되돌리기", "Ctrl+Z", self.undo_stroke)
        self.act_clear_mask = self._act("이 프레임에서 그린 마스크 지우기", None, self.clear_drawn_mask)
        self.act_compare = self._act("수정 전후 비교 (이 프레임부터 시험 추적)", "C", self.start_compare)
        self.act_compare.setToolTip("선택한 객체를 이 프레임부터 짧게 추적해 보고, 수정 전과 나란히 비교합니다. 영상은 바뀌지 않습니다.")
        e = self.menuBar().addMenu("편집(&E)")
        e.addAction(act("이 프레임의 마지막 점 되돌리기", "U", self.undo_point))
        e.addAction(act("이 프레임에서 이 객체의 점 지우기", "X", self.clear_frame_points))
        e.addSeparator()
        e.addAction(self.act_brush)
        e.addAction(self.act_undo_stroke)
        e.addAction(self.act_clear_mask)
        e.addAction(self.act_compare)
        e.addSeparator()
        self.act_cut = act("결과 영상 자르기...", "Ctrl+T", self.open_cut_dialog)
        self.act_cut.setToolTip("처리가 끝난 영상에서 구간을 잘라내 새 파일로 내보냅니다. 원본과 추적 결과는 바뀌지 않습니다.")
        e.addAction(self.act_cut)
        # navigation shortcuts are window-level actions so they work wherever the focus is
        for key, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)),
                        ("Shift+Left", lambda: self.step(-10)), ("Shift+Right", lambda: self.step(10)),
                        ("Ctrl+Left", lambda: self.step_seconds(-1)), ("Ctrl+Right", lambda: self.step_seconds(1)),
                        ("Space", self.toggle_play), ("M", self.next_flag), ("B", self.prev_flag),
                        ("R", lambda: self.review_panel.toggle_reviewed())):
            self.addAction(act("", key, fn))
        self._build_toolbar()

    def _build_toolbar(self):
        """Top bar with the everyday actions; the same QActions as the menus, so their enabled state stays in sync."""
        tb = QToolBar("도구", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toolbar = tb
        self.act_save = self._act("포인트 저장", None, self.save_points)
        for a in (self.act_open_video, self.act_open_points, self.act_save):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_live)
        tb.addAction(self.act_last)
        tb.addAction(self.act_brush)
        tb.addAction(self.act_compare)
        for a, short in ((self.act_live, "마스크 미리보기"), (self.act_last, "지난 실행 마스크"), (self.act_brush, "브러시"),
                         (self.act_compare, "수정 전후 비교")):
            a.setIconText(short)                         # shorter labels on the toolbar; the menus keep the full text
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background: transparent")
        tb.addWidget(spacer)
        tb.addAction(self.act_cancel)
        self.start_btn = QToolButton()                   # the accent button on the right: start the mosaic job
        self.start_btn.setObjectName("primary")
        self.start_btn.setDefaultAction(self.act_start)
        tb.addWidget(self.start_btn)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

    def _build_side_panel(self):
        # ---- left: the objects being marked
        dock = QDockWidget("대상", self)
        self.targets_dock = dock
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("객체 (지금 표시할 객체를 선택):"))
        self.obj_list = QListWidget()
        self.obj_list.currentRowChanged.connect(self._object_selected)
        v.addWidget(self.obj_list, 1)
        for text, fn in (("새 객체", self.new_object), ("객체 삭제", self.delete_object),
                         ("마지막 점 되돌리기 (U)", self.undo_point), ("이 프레임의 점 지우기 (X)", self.clear_frame_points)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            v.addWidget(b)
        self.hint = QLabel("객체 추적은 가장 이른 클릭에서 시작합니다. 객체가 처음 보이는 프레임에서 클릭하세요.\n"
                           "왼쪽 클릭 = 대상, 오른쪽 클릭 = 대상 아님.")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        dock.setWidget(w)
        dock.setMinimumWidth(260)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # ---- right: properties of the video and of the selected object
        pdock = QDockWidget("속성", self)
        self.props_dock = pdock
        pdock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        pw = QWidget()
        pv = QVBoxLayout(pw)
        self.info_label = QLabel("불러온 영상이 없습니다.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pv.addWidget(self.info_label)
        # per-object mosaic settings ("작업 설정" = whatever is chosen in the processing dialog)
        self.style_box = QGroupBox("선택한 객체의 모자이크")
        form = QFormLayout(self.style_box)
        self.obj_style = QComboBox()
        self.obj_style.addItem("작업 설정", None)
        for name in OBJECT_STYLES:
            self.obj_style.addItem(name, name)
        self.obj_style.setToolTip("pixelate(픽셀)와 solid(단색)는 내용을 완전히 없앱니다. blur(흐림)가 가장 약합니다.")
        form.addRow("형태", self.obj_style)
        self.obj_block = QSpinBox()
        self.obj_block.setRange(0, 256)
        self.obj_block.setSpecialValueText("작업 설정")
        self.obj_block.setToolTip("블록 크기(픽셀) 또는 흐림 강도, 원본 해상도 픽셀 기준 2-256. 0 = 작업 설정의 강도.")
        form.addRow("강도 (px)", self.obj_block)
        self.obj_pad = QSpinBox()
        self.obj_pad.setRange(-1, 500)
        self.obj_pad.setSpecialValueText("작업 설정")
        self.obj_pad.setToolTip("모자이크가 이 객체의 마스크 바깥으로 얼마나 넓게 덮는지, 원본 해상도 픽셀 기준. -1 = 작업 설정의 여백.")
        form.addRow("여백 (px)", self.obj_pad)
        self.obj_style.currentIndexChanged.connect(self._style_edited)
        self.obj_block.valueChanged.connect(self._style_edited)
        self.obj_pad.valueChanged.connect(self._style_edited)
        pv.addWidget(self.style_box)
        note = QLabel("탐지(NudeNet) 영역과 안전 모드 영역은 항상 작업 설정을 따릅니다. 영역이 겹치면 더 강한 쪽이 적용됩니다.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        pv.addWidget(note)
        # brush: paint the mask of the selected object on the current frame
        self.brush_box = QGroupBox("브러시 (선택한 객체, 현재 프레임)")
        bform = QFormLayout(self.brush_box)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(2, 300)
        self.brush_size.setValue(24)
        self.brush_size.setToolTip("붓의 반지름, 원본 해상도 픽셀 기준.")
        self.brush_size.valueChanged.connect(lambda *_: self._apply_brush_mode())
        bform.addRow("붓 크기 (px)", self.brush_size)
        self.brush_undo_btn = QPushButton("마지막 붓질 되돌리기 (Ctrl+Z)")
        self.brush_undo_btn.clicked.connect(self.undo_stroke)
        self.brush_clear_btn = QPushButton("이 프레임에서 그린 마스크 지우기")
        self.brush_clear_btn.clicked.connect(self.clear_drawn_mask)
        bform.addRow(self.brush_undo_btn)
        bform.addRow(self.brush_clear_btn)
        bnote = QLabel("도구 막대의 '브러시로 마스크 수정'(G)을 켜고 왼쪽 드래그로 칠하고 오른쪽 드래그로 지웁니다. "
                       "그린 프레임에서는 클릭 대신 그린 마스크에서 추적을 시작합니다.")
        bnote.setObjectName("hint")
        bnote.setWordWrap(True)
        bform.addRow(bnote)
        pv.addWidget(self.brush_box)
        # before/after comparison of a correction
        self.compare_box = QGroupBox("수정 전후 비교")
        cform = QFormLayout(self.compare_box)
        self.compare_frames = QSpinBox()
        self.compare_frames.setRange(5, 300)
        self.compare_frames.setValue(60)
        self.compare_frames.setToolTip("현재 프레임부터 몇 프레임을 시험 추적해서 비교할지 정합니다.")
        cform.addRow("비교할 프레임 수", self.compare_frames)
        self.compare_btn = QPushButton("이 프레임부터 비교 시작 (C)")
        self.compare_btn.clicked.connect(self.start_compare)
        cform.addRow(self.compare_btn)
        self.compare_mode = QComboBox()
        for text, val in (("차이 (초록 = 늘어남, 빨강 = 줄어듦)", "diff"), ("수정 전", "before"), ("수정 후", "after")):
            self.compare_mode.addItem(text, val)
        self.compare_mode.currentIndexChanged.connect(lambda *_: self._update_overlay())
        cform.addRow("보기", self.compare_mode)
        self.compare_info = QLabel("고친 프레임에서 비교를 시작하면, 그 뒤 프레임들이 어떻게 달라지는지 미리 볼 수 있습니다.")
        self.compare_info.setObjectName("hint")
        self.compare_info.setWordWrap(True)
        self.compare_jump = QPushButton("가장 많이 달라진 프레임으로 이동")
        self.compare_jump.clicked.connect(self.goto_biggest_change)
        self.compare_keep = QPushButton("수정 유지")
        self.compare_keep.clicked.connect(self.keep_correction)
        self.compare_revert = QPushButton("수정 취소 (되돌리기)")
        self.compare_revert.clicked.connect(self.revert_correction)
        cform.addRow(self.compare_jump)
        crow = QHBoxLayout()
        crow.addWidget(self.compare_keep)
        crow.addWidget(self.compare_revert)
        cform.addRow(crow)
        cform.addRow(self.compare_info)               # below the buttons, so keep / revert stay in view
        pv.addWidget(self.compare_box)
        self._refresh_compare_panel()
        pv.addStretch(1)
        scroll = QScrollArea()                           # the panel is taller than a small window: let it scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(pw)
        pdock.setWidget(scroll)
        pdock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, pdock)
        self._refresh_objects()

    def _build_process_dock(self):
        self.panel = ProcessPanel()
        self.panel.cancel_requested.connect(self.cancel_processing)
        self.panel.resume_requested.connect(self.resume_processing)
        self.panel.open_output_requested.connect(lambda: self.open_path(self._result_output()))
        self.panel.open_logs_requested.connect(lambda: self.open_path(os.path.join(PROJECT_ROOT, "logs")))
        self.process_dock = QDockWidget("처리", self)
        self.process_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.process_dock.setWidget(self.panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.process_dock)
        self.process_dock.hide()

    def _build_review_dock(self):
        self.review_panel = ReviewPanel()
        self.review_panel.segment_selected.connect(self._flag_selected)
        self.review_panel.next_requested.connect(self.next_flag)
        self.review_panel.prev_requested.connect(self.prev_flag)
        self.review_panel.reprocess_requested.connect(self.reprocess)
        self.review_panel.reviewed_changed.connect(self._update_flag_ticks)
        self.review_panel.sev.currentIndexChanged.connect(lambda *_: self._update_flag_ticks())
        self.review_dock = QDockWidget("검수", self)
        self.review_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.review_dock.setWidget(self.review_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.review_dock)
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)
        self.tabifyDockWidget(self.props_dock, self.review_dock)
        self.props_dock.raise_()

    # ------------------------------------------------------------------ review screen
    def open_review_dialog(self):
        if self.info is None:
            self.show_error("먼저 영상을 여세요.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "검수 로그 열기", os.path.join(PROJECT_ROOT, "logs"), "검수 로그 (*.json)")
        if path:
            self.load_review(path)

    def load_review(self, path: str, quiet: bool = False) -> bool:
        if self.info is None:
            return False
        try:
            data = load_review_data(path, self.info)
        except ReviewError as e:
            if not quiet:
                self.show_error(str(e))
            return False
        self.review = data
        self.review_panel.set_data(data, self._stale_reasons())
        self._update_flag_ticks()
        self.review_dock.raise_()
        msg = f"검수 로그를 불러왔습니다: 플래그 {len(data.filtered('warn'))}건 (경고 이상), 미확인 {data.unreviewed_count()}건"
        if data.dropped_marks:
            msg += f"; 해당 청크가 다시 처리되어 '확인' 표시 {data.dropped_marks}건이 초기화되었습니다"
        self.statusBar().showMessage(msg)
        return True

    def _stale_reasons(self) -> list:
        if self.review is None:
            return []
        pp = self.points_path if self.points_path and os.path.isfile(self.points_path) else None
        reasons = self.review.staleness(pp, bool(self.store.points or self.store.masks))
        if self.dirty:
            reasons.append("저장하지 않은 포인트 수정이 있습니다")
        return reasons

    def _refresh_stale_banner(self):
        if self.review is not None:
            self.review_panel.set_banner(self._stale_reasons())

    def _update_flag_ticks(self):
        flags, segs = [], []
        if self.review is not None:
            floor = self.review_panel.sev.currentData() or "warn"
            for s in self.review.filtered(floor):
                if not self.review.state.is_reviewed(s):
                    flags += [(s["start_frame"], s["severity"]), (s["end_frame"], s["severity"])]
                    segs.append((s["start_frame"], s["end_frame"], s["severity"]))
        self.slider.set_flags(flags)
        self.timeline_view.set_flags(segs)

    def _goto_flag(self, frame: int, obj: Optional[int]):
        self.goto(frame)
        if obj is not None and self.store.select(obj):
            self._refresh_objects()
            self.view.update()

    def _flag_selected(self, seg: dict):
        self._goto_flag(seg["start_frame"], seg.get("object_id"))

    def _step_flag(self, forward: bool):
        if self.review is None or self.info is None:
            self.statusBar().showMessage("불러온 검수 로그가 없습니다 (파일 > 검수 로그 열기).")
            return
        nav = self.review.navigator(self.review_panel.sev.currentData() or "warn")
        tgt = nav.next_after(self.frame_idx) if forward else nav.prev_before(self.frame_idx)
        if tgt is None:
            self.statusBar().showMessage("그 방향에는 더 이상 플래그가 없습니다.")
            return
        self._goto_flag(tgt, nav.object_at(tgt))
        self.review_panel.select_row_for(tgt)

    def next_flag(self):
        self._step_flag(True)

    def prev_flag(self):
        self._step_flag(False)

    # ------------------------------------------------------------------ re-processing after edits
    def reprocess(self):
        """Re-run with the edited points. A dry run (--plan-only) first tells what is reused and what is recomputed."""
        if self.info is None or self.job_running or self.plan_runner.is_running():
            return
        if self.dirty:
            choice = self.ask_save_changes()
            if choice == "cancel" or (choice == "save" and not self.save_points()):
                return
        settings = self.last_settings
        if settings is None:
            dlg = ProcessDialog(self, self.info.path, self._points_for_job() is not None, self.info.nb_frames, info=self.info)
            dlg.resume.setChecked(True)
            if not dlg.exec():
                return
            settings = dlg.settings()
        settings = dataclasses.replace(settings, resume=True)
        points = self._points_for_job()
        probs = validate_settings(settings, self.info.path, points is not None)
        if probs:
            self.show_error("\n".join(probs))
            return
        self._plan, self._plan_settings = None, settings
        cancel_file = self.plan_runner.new_cancel_file()
        self.statusBar().showMessage("재사용할 수 있는 부분을 확인하는 중 ...")
        self.plan_runner.start(build_args(self.info.path, points, settings, cancel_file, plan_only=True), cancel_file)

    def _on_plan_event(self, ev: dict):
        if ev.get("event") == "plan":
            self._plan = ev

    def _on_plan_finished(self, res: dict):
        plan = self._plan
        if plan is None:
            self.show_error("재사용할 수 있는 부분을 확인하지 못했습니다: " + str(res.get("reason") or "알 수 없는 오류"))
            return
        n, total = plan["total_chunks"], plan["total_frames"]
        render_only = plan.get("frames_render_only") or 0
        if plan["can_resume"] and render_only:
            full = total - plan["frames_reused"] - render_only
            text = (f"모자이크 또는 인코더 설정이 바뀌었습니다.\n\n전체 {total}프레임 중 {plan['frames_reused']}프레임은 그대로 재사용하고, "
                    f"{render_only}프레임은 모자이크만 다시 적용합니다 (추적·탐지 없음). "
                    f"{full}프레임은 처음부터 다시 처리합니다.\n시작할까요?")
        elif plan["can_resume"] and plan["reusable_chunks"]:
            text = (f"전체 {total}프레임 중 {plan['frames_reused']}프레임은 재사용하고, "
                    f"{total - plan['frames_reused']}프레임은 다시 처리합니다 (청크 {plan['reusable_chunks'] + 1}-{n}).\n시작할까요?"
                    if plan["reusable_chunks"] < n else "바뀐 것이 없습니다. 최종 영상만 다시 만듭니다.\n시작할까요?")
        else:
            why = ("보관된 중간 청크가 없습니다" if not plan["job_found"] else
                   "설정이나 영상이 이전 실행과 달라 이전 결과를 쓸 수 없습니다" if not plan["can_resume"] else "재사용할 수 있는 청크가 없습니다")
            text = f"{why}.\n\n전체 {total}프레임을 처리합니다. 시작할까요?"
        if self.ask_confirm(text):
            self.start_processing(dataclasses.replace(self._plan_settings, resume=bool(plan["can_resume"])))

    # ------------------------------------------------------------------ job registry
    def _log_dir(self) -> str:
        return (self.last_settings.log_dir if self.last_settings and self.last_settings.log_dir else "") or os.path.join(PROJECT_ROOT, "logs")

    def _job_db(self) -> Optional[JobDB]:
        path = db_path_for(self._log_dir())
        return JobDB(path) if os.path.isfile(path) else None

    def check_resumable(self) -> Optional[dict]:
        """Look for a job of this video that stopped before finishing (cancelled, failed, crashed) and tell the user."""
        self.resumable_job = None
        try:
            db = self._job_db()
            if self.info is not None and db is not None:
                self.resumable_job = db.latest_resumable(self.info.fingerprint)
        except Exception:  # noqa: BLE001  (the registry is only a convenience)
            log.warning("job registry unavailable", exc_info=True)
        j = self.resumable_job
        self.act_resume_job.setEnabled(j is not None and not self.job_running)
        if j is not None:
            total = j.get("n_chunks") or "?"
            done = j["chunks_done"] + (0 if j["chunks_done"] else (j.get("reused_chunks") or 0))
            self.statusBar().showMessage(self.statusBar().currentMessage() + f"  |  중단된 작업이 있습니다 ({j['status']}, 청크 {done}/{total} 완료, "
                                         f"{j.get('created')}): 처리 > 중단된 작업 이어하기")
        return j

    def show_job_history(self):
        if self.info is None:
            return
        try:
            db = self._job_db()
            jobs = db.jobs_for_input(self.info.fingerprint) if db is not None else []
        except Exception:  # noqa: BLE001
            jobs = []
        JobsDialog(self, jobs, os.path.basename(self.info.path)).exec()

    def delete_intermediate(self):
        if self.info is None or self.job_running:
            return
        work = self.last_settings.work_dir if self.last_settings else ""
        jd = job_dir_for_video(work, self.info.path, self.info.fingerprint)
        if not os.path.isdir(jd):
            self.statusBar().showMessage("이 영상의 중간 청크가 없습니다.")
            return
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(jd) for f in fs) / 2 ** 20
        if self.ask_confirm(f"중간 청크 {size:.0f} MB를 삭제할까요?\n결과 영상과 로그는 남지만, 다시 처리할 때는 처음부터 시작합니다."):
            import shutil
            self._release_job_folder()                     # an open cache file would keep the folder alive on Windows
            shutil.rmtree(jd, ignore_errors=True)
            if os.path.isdir(jd):
                self.statusBar().showMessage("중간 청크를 모두 지우지 못했습니다. 다른 프로그램이 파일을 쓰고 있는지 확인하세요.")
            else:
                self.statusBar().showMessage(f"중간 청크 {size:.0f} MB를 삭제했습니다.")

    # ------------------------------------------------------------------ processing jobs
    def open_path(self, path: Optional[str]):
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _result_output(self) -> Optional[str]:
        if self.last_result and self.last_result.get("output"):
            return self.last_result["output"]
        return self.last_settings.output if self.last_settings else None

    def _set_job_running(self, running: bool):
        self.job_running = running
        self.act_start.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_open_video.setEnabled(not running)
        self.act_open_points.setEnabled(not running)
        self.act_reprocess.setEnabled(not running)
        self.act_cleanup.setEnabled(not running)
        self.act_resume_job.setEnabled(not running and self.resumable_job is not None)
        self.style_box.setEnabled(not running)
        if running and self.act_live.isChecked():          # the job needs the GPU: the preview process is stopped
            self.act_live.setChecked(False)
        self.act_live.setEnabled(not running)
        if running and self.act_brush.isChecked():         # masks are locked like the points while a job runs
            self.act_brush.setChecked(False)
        for w in (self.act_brush, self.act_undo_stroke, self.act_clear_mask, self.brush_box, self.act_compare, self.compare_box):
            w.setEnabled(not running)
        if running:
            self._end_compare()

    def start_processing_dialog(self):
        if self.info is None:
            self.show_error("먼저 영상을 여세요.")
            return
        if self.job_running:
            return
        if self.dirty:
            choice = self.ask_save_changes()
            if choice == "cancel":
                return
            if choice == "save" and not self.save_points():
                return
        needs_sam2 = self._points_for_job() is not None
        dlg = ProcessDialog(self, self.info.path, needs_sam2, self.info.nb_frames, info=self.info)
        if dlg.exec():
            self.start_processing(dlg.settings())

    def _points_for_job(self) -> Optional[str]:
        """The saved points file to hand to the worker, or None when nothing was clicked."""
        if (self.store.points or self.store.masks) and self.points_path and os.path.isfile(self.points_path) and not self.store.valid():
            return self.points_path
        return None

    def start_processing(self, settings: ProcessSettings) -> bool:
        """Start the worker process. Returns False (with an error shown) when it cannot start."""
        if self.info is None or self.job_running:
            return False
        if self.dirty:
            self.show_error("먼저 포인트를 저장하세요 (Ctrl+S). 처리는 저장된 파일을 읽습니다.")
            return False
        points = self._points_for_job()
        probs = validate_settings(settings, self.info.path, points is not None)
        if probs:
            self.show_error("\n".join(probs))
            return False
        show_last = self.act_last.isChecked()
        self._release_job_folder()                         # the worker deletes or rewrites this video's job folder
        self._set_checked(self.act_last, show_last)        # shown again when the job has finished
        self.last_settings, self.last_result = settings, None
        cancel_file = self.runner.new_cancel_file()
        self.panel.begin()
        self.process_dock.show()
        self._set_job_running(True)
        self._stop_play()
        self.statusBar().showMessage("처리 중 ... (처리 > 처리 취소로 중단, 완료된 청크는 보관됩니다)")
        if not self.runner.start(build_args(self.info.path, points, settings, cancel_file), cancel_file):
            self._set_job_running(False)
            return False
        return True

    def cancel_processing(self):
        if self.job_running:
            self.panel.cancel_btn.setEnabled(False)
            self.panel.stage.setText("취소하는 중 - 현재 단계를 마무리하고 있습니다 ...")
            self.runner.cancel()

    def resume_processing(self):
        if self.last_settings is not None and not self.job_running:
            self.start_processing(dataclasses.replace(self.last_settings, resume=True))

    def _on_job_progress(self, ev: dict):
        self.panel.on_progress(ev)

    def _on_job_finished(self, res: dict):
        self.last_result = res
        self._set_job_running(False)
        self.panel.on_finished(res)
        self.statusBar().showMessage(f"처리 결과 {STATUS_TEXT.get(res.get('status'), res.get('status'))}: " + (res.get("reason") or res.get("output") or ""))
        if res.get("review_log") and os.path.isfile(res["review_log"]):
            self.load_review(res["review_log"], quiet=True)
        self.check_resumable()
        self._load_run_summary()                           # timeline tracks: where each object has a mask in the new run
        if self.act_last.isChecked():                      # show the masks of the run that just finished
            self.toggle_last_masks(True)

    # ------------------------------------------------------------------ cutting the result
    def open_cut_dialog(self, path: Optional[str] = None, modal: bool = True):
        """Open the cutting window for a finished video: ``path``, else the last result, else a file the user picks."""
        if not path:
            out = self._result_output()
            start = out if out and os.path.isfile(out) else os.path.join(PROJECT_ROOT, "output")
            if out and os.path.isfile(out) and self.ask_confirm(f"방금 처리한 결과 영상을 자를까요?\n{out}\n\n'아니요'를 누르면 다른 파일을 고릅니다."):
                path = out
            else:
                path, _ = QFileDialog.getOpenFileName(self, "자를 영상 선택", start, VIDEO_FILTER)
        if not path:
            return None
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                info = probe_video(path)
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as ex:  # noqa: BLE001
            self.show_error(f"영상을 열지 못했습니다:\n{ex}")
            return None
        dlg = CutDialog(self, info)
        self.cut_dialog = dlg
        if modal:
            dlg.exec()
        else:
            dlg.show()
        return dlg

    # ------------------------------------------------------------------ timeline data (loaded off the UI thread)
    def _start_thumbnails(self):
        if self._thumb_loader is not None:
            self._thumb_loader.cancelled = True
        loader = ThumbnailLoader(self.info)
        loader.setAutoDelete(False)                        # the signal object must outlive run(); we keep the reference
        loader.signals.done.connect(self._thumbnails_ready)
        self._thumb_loader = loader
        QThreadPool.globalInstance().start(loader)

    def _thumbnails_ready(self, path: str, images: list):
        if self.info is not None and path == self.info.path:
            self.timeline_view.set_thumbnails(images)

    def _release_job_folder(self):
        """Stop everything in the app that reads the job folder (the timeline's summary loader, the kept-mask overlay), so the
        folder can be deleted or rewritten by a job."""
        loader, self._summary_loader = self._summary_loader, None
        if loader is not None and QThreadPool.globalInstance().tryTake(loader):
            loader.finished.set()                          # it never started
        elif loader is not None:
            loader.stop()
        self.timeline_view.set_run_data(None)
        if self.act_last.isChecked():
            self._set_checked(self.act_last, False)
        self._job_masks = None
        self._update_overlay()

    def _load_run_summary(self):
        if self._summary_loader is not None:
            self._summary_loader.cancelled = True
        self.timeline_view.set_run_data(None)
        if self.info is None:
            return
        work = self.last_settings.work_dir if self.last_settings else ""
        jd = job_dir_for_video(work, self.info.path, self.info.fingerprint)
        if not os.path.isfile(os.path.join(jd, "job.json")):
            return
        loader = RunSummaryLoader(self.info.path, jd)
        loader.setAutoDelete(False)
        loader.signals.done.connect(self._run_summary_ready)
        self._summary_loader = loader
        QThreadPool.globalInstance().start(loader)

    def _run_summary_ready(self, path: str, data):
        if self.info is not None and path == self.info.path:
            self.timeline_view.set_run_data(data)

    # ------------------------------------------------------------------ mask overlays
    def _set_checked(self, action, on: bool):
        action.blockSignals(True)
        action.setChecked(on)
        action.blockSignals(False)

    def toggle_live_preview(self, on: bool):
        self._end_compare()                            # a comparison in progress lives in the preview process
        self.preview.stop()
        self._live_masks.clear()
        self._preview_reqs.clear()
        self._preview_asked = None
        if on:
            problem = None
            st = self.last_settings or load_last_settings()
            ckpt = st.sam2_checkpoint or find_default_checkpoint()
            if self.info is None:
                problem = "먼저 영상을 여세요."
            elif self.job_running:
                problem = "작업이 실행 중입니다. 작업이 끝나면 마스크 미리보기를 다시 쓸 수 있습니다."
            elif self.preview_command is None and not os.path.isfile(ckpt):
                problem = "SAM 2 체크포인트를 찾을 수 없습니다. models/ 폴더에 받거나(models/README.md 참고) 처리 > 처리 시작에서 선택하세요."
            if problem:
                self._set_checked(self.act_live, False)
                self.show_error(problem)
            else:
                self.preview.start(preview_args(self.info.path, ckpt, st.analysis_width, st.device), command=self.preview_command)
        self._update_overlay()

    def toggle_last_masks(self, on: bool):
        self._job_masks = None
        if on:
            try:
                if self.info is None:
                    raise FileNotFoundError("no video")
                work = self.last_settings.work_dir if self.last_settings else ""
                jm = JobMasks(job_dir_for_video(work, self.info.path, self.info.fingerprint))
                if not jm.available_chunks():
                    raise FileNotFoundError("no masks")
                self._job_masks = jm
            except (OSError, ValueError, KeyError):
                self._set_checked(self.act_last, False)
                self.statusBar().showMessage("이 영상으로 보관된 마스크가 없습니다. 먼저 \"중간 청크 보관\"을 켜고 처리하세요.")
        self._update_overlay()

    def _points_sha(self) -> Optional[str]:
        try:
            with open(self.points_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (OSError, TypeError):
            return None

    def _make_overlay(self, shape, masks: dict, boxes=(), regions=(), full=False, note="") -> dict:
        return {"size": (shape[1], shape[0]), "masks": [mask_image(m, object_color(oid)) for oid, m in sorted(masks.items())],
                "boxes": list(boxes), "regions": list(regions), "full": full, "note": note}

    # ------------------------------------------------------------------ brush: hand-drawn masks
    def _edit_shape(self) -> tuple:
        """(height, width) drawn masks are edited at: the preview size."""
        w, h = self.server.size
        return int(h), int(w)

    def _apply_brush_mode(self):
        self.view.set_brush(self.act_brush.isChecked(), float(self.brush_size.value()))

    def toggle_brush(self, on: bool):
        if on and self.info is None:
            self._set_checked(self.act_brush, False)
            self.show_error("먼저 영상을 여세요.")
            on = False
        if on:
            self._stop_play()
            self.statusBar().showMessage("브러시: 왼쪽 드래그 = 칠하기, 오른쪽 드래그 = 지우기. 미리보기나 지난 실행의 마스크가 있으면 거기서 시작합니다.")
        self._apply_brush_mode()
        self._update_overlay()

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        shape = self._edit_shape()
        if mask.shape == shape:
            return mask.astype(bool)
        return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    def _brush_start_mask(self, oid: int, frame: int):
        """What the mask of this object on this frame looks like before the user touches it: (mask at edit size or None, source)."""
        drawn = self.store.get_mask(oid, frame)
        if drawn is not None:
            return self._resize_mask(drawn), "drawn"
        hit = self._live_masks.get(frame)
        if hit is not None and oid in hit[1]:
            return self._resize_mask(hit[1][oid]), "preview"
        if self._job_masks is not None:
            fr = self._job_masks.frame(frame)
            if fr is not None and oid in fr["masks"]:
                return self._resize_mask(fr["masks"][oid]), "last_run"
        return None, "none"

    def _stroke_started(self, label: int):
        if self.info is None or self.job_running or not self.act_brush.isChecked():
            self._stroke_key = None
            return
        key = (self.store.current, self.frame_idx)
        base, source = self._brush_start_mask(*key)
        if key not in self._brush_base:
            self._brush_base[key] = (None if base is None else base.copy(), source)
        prev = self.store.get_mask(*key)
        self._brush_undo.append((key[0], key[1], None if prev is None else prev.copy()))
        del self._brush_undo[:-50]
        self._stroke_key = key
        self._stroke_mask = (np.zeros(self._edit_shape(), dtype=np.uint8) if base is None else base.astype(np.uint8))

    def _stroke_segment(self, x0: float, y0: float, x1: float, y1: float, label: int):
        if self._stroke_key is None or self._stroke_mask is None or self._stroke_key != (self.store.current, self.frame_idx):
            return
        h, w = self._stroke_mask.shape
        kx, ky = w / self.info.width, h / self.info.height
        r = max(1, int(round(self.brush_size.value() * kx)))
        p0, p1 = (int(round(x0 * kx)), int(round(y0 * ky))), (int(round(x1 * kx)), int(round(y1 * ky)))
        cv2.line(self._stroke_mask, p0, p1, int(label), thickness=2 * r, lineType=cv2.LINE_8)
        cv2.circle(self._stroke_mask, p1, r, int(label), thickness=-1)
        self.store.set_mask(self._stroke_key[0], self._stroke_key[1], self._stroke_mask.astype(bool))
        self._update_overlay()

    def _stroke_finished(self):
        if self._stroke_key is not None:
            self._stroke_key, self._stroke_mask = None, None
            self._points_changed()

    def undo_stroke(self):
        if self.job_running or not self._brush_undo:
            return
        oid, frame, prev = self._brush_undo.pop()
        self.store.set_mask(oid, frame, prev)
        self._points_changed()
        self.statusBar().showMessage(f"객체 {oid}, 프레임 {frame}의 붓질을 되돌렸습니다.")

    def clear_drawn_mask(self):
        key = (self.store.current, self.frame_idx)
        if self.job_running or self.info is None or self.store.get_mask(*key) is None:
            return
        self._brush_undo.append((key[0], key[1], self.store.get_mask(*key).copy()))
        self.store.set_mask(key[0], key[1], None)
        self._points_changed()
        self.statusBar().showMessage(f"객체 {key[0]}, 프레임 {key[1]}의 그린 마스크를 지웠습니다. 이 프레임은 다시 클릭으로 추적합니다.")

    def _write_corrections(self):
        """After a successful save: one record per mask that was painted since the last save (what it was before, what it is now)."""
        self.corrections_saved = []
        for (oid, frame), (base, source) in sorted(self._brush_base.items()):
            after = self.store.get_mask(oid, frame)
            if after is None:
                continue
            after = self._resize_mask(after)
            if base is not None and base.shape == after.shape and np.array_equal(base, after):
                continue
            try:
                self.corrections_saved.append(save_correction(self._log_dir(), self.info.fingerprint, (self.info.width, self.info.height), oid,
                                                              frame, self.info.timestamp_of(frame), base, after, source))
            except Exception:  # noqa: BLE001  (the record is a by-product: it must never make saving the points fail)
                log.warning("could not write a correction record", exc_info=True)
        self._brush_base.clear()

    # ------------------------------------------------------------------ before/after comparison of a correction
    def _compare_prompts(self, oid: int, start: int, n: int, corrected: bool) -> dict:
        """The object's prompts in [start, start + n) as offsets from ``start``. corrected=True: as they are now (drawn masks
        win over clicks). corrected=False: as they were before the brush touched the start frame."""
        out = {}
        for k in range(start, start + n):
            drawn = self.store.get_mask(oid, k)
            if not corrected and k == start:
                base = self._brush_base.get((oid, k))
                drawn = base[0] if base is not None and base[1] == "drawn" else (None if base is not None else drawn)
                if base is None and drawn is not None:
                    drawn = None                       # a mask loaded from the file: "before" means without it
            if drawn is not None:
                out[k - start] = {"mask": drawn}
                continue
            pts = [p for p in self.store.points if p["obj_id"] == oid and p["frame_idx"] == k]
            if pts and any(p["label"] == 1 for p in pts):
                out[k - start] = {"points": [[p["x"], p["y"]] for p in pts], "labels": [p["label"] for p in pts]}
        return out

    @staticmethod
    def _prompt_sig(prompts: dict) -> str:
        items = [[off, mask_digest(pr["mask"])] if pr.get("mask") is not None else [off, pr["points"], pr["labels"]]
                 for off, pr in sorted(prompts.items())]
        return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]

    def start_compare(self) -> bool:
        """Trial-track the selected object from the current frame and compare with what was there before."""
        if self.info is None or self.job_running:
            return False
        oid, start = self.store.current, self.frame_idx
        n = min(int(self.compare_frames.value()), self.info.nb_frames - start)
        after_prompts = self._compare_prompts(oid, start, n, corrected=True)
        if 0 not in after_prompts:
            self.show_error("이 프레임에는 이 객체의 클릭이나 그린 마스크가 없습니다.\n고친 프레임(또는 클릭한 프레임)으로 이동한 뒤 비교를 시작하세요.")
            return False
        if not self.preview.is_on():
            self.act_live.setChecked(True)             # the trial tracking runs in the mask-preview process
            if not self.preview.is_on():
                return False
        self._end_compare()
        before, source = None, "none"
        jm = self._job_masks
        if jm is None:
            try:
                work = self.last_settings.work_dir if self.last_settings else ""
                jm = JobMasks(job_dir_for_video(work, self.info.path, self.info.fingerprint))
            except (OSError, ValueError, KeyError):
                jm = None
        if jm is not None:
            frames = [jm.frame(k) for k in range(start, start + n)]
            if any(fr is not None for fr in frames):
                before, source = [None if fr is None else fr["masks"].get(oid) for fr in frames], "last_run"
        cmp = {"oid": oid, "start": start, "n": n, "sig": self._prompt_sig(after_prompts), "after": None, "before": before,
               "before_source": source, "pending": {}, "ready": False, "stats": None}
        if before is None:
            before_prompts = self._compare_prompts(oid, start, n, corrected=False)
            if 0 in before_prompts and self._prompt_sig(before_prompts) != cmp["sig"]:
                cmp["pending"][self.preview.request_track(oid, start, n, before_prompts)] = "before"
                cmp["before_source"] = "trial"
        cmp["pending"][self.preview.request_track(oid, start, n, after_prompts)] = "after"
        self._compare = cmp
        self.statusBar().showMessage(f"비교: 객체 {oid}를 프레임 {start}부터 {n}프레임 시험 추적하는 중 ...")
        self._refresh_compare_panel()
        self._update_overlay()
        return True

    def _on_track_ready(self, rid: int, frame: int, masks: list):
        cmp = self._compare
        if cmp is None or rid not in cmp["pending"]:
            return
        cmp[cmp["pending"].pop(rid)] = masks
        if cmp["pending"]:
            return
        shape = next((m.shape for m in cmp["after"] if m is not None), None)
        stats = []
        for i in range(len(cmp["after"])):
            a = cmp["after"][i]
            b = cmp["before"][i] if cmp["before"] is not None and i < len(cmp["before"]) else None
            if a is not None and b is not None and b.shape != a.shape:
                b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                cmp["before"][i] = b
            a_px, b_px = (0 if a is None else int(a.sum())), (0 if b is None else int(b.sum()))
            if a is None or b is None:
                union, inter = max(a_px, b_px), 0
            else:
                union, inter = int((a | b).sum()), int((a & b).sum())
            stats.append({"after_px": a_px, "before_px": b_px, "iou": 1.0 if union == 0 else inter / union})
        cmp.update(ready=True, stats=stats, shape=shape, n=len(cmp["after"]))
        self.timeline_view.set_highlight((cmp["start"], cmp["start"] + cmp["n"]))
        self.statusBar().showMessage("비교 준비 완료: 프레임을 넘기며 수정 전후를 확인한 뒤 '수정 유지' 또는 '수정 취소'를 고르세요.")
        self._refresh_compare_panel()
        self._update_overlay()

    def _refresh_compare_panel(self):
        cmp = self._compare
        ready = cmp is not None and cmp["ready"]
        for w in (self.compare_mode, self.compare_jump, self.compare_keep, self.compare_revert):
            w.setEnabled(ready)
        if cmp is None:
            self.compare_info.setText("고친 프레임에서 비교를 시작하면, 그 뒤 프레임들이 어떻게 달라지는지 미리 볼 수 있습니다.")
        elif not ready:
            self.compare_info.setText(f"객체 {cmp['oid']}, 프레임 {cmp['start']}부터 {cmp['n']}프레임 시험 추적 중 ...")
        else:
            st = cmp["stats"]
            src = {"last_run": "지난 실행의 마스크", "trial": "고치기 전 상태로 시험 추적한 결과", "none": "없음 (비교 대상 없이 수정 후만 표시)"}[cmp["before_source"]]
            lost = sum(1 for x in st if x["after_px"] == 0)
            text = (f"객체 {cmp['oid']}, 프레임 {cmp['start']}-{cmp['start'] + cmp['n'] - 1} ({cmp['n']}프레임).\n수정 전 기준: {src}.\n"
                    f"평균 면적: 수정 후 {sum(x['after_px'] for x in st) // max(1, len(st))} px")
            if cmp["before"] is not None:
                changed = sum(1 for x in st if x["iou"] < 0.9)
                text += (f", 수정 전 {sum(x['before_px'] for x in st) // max(1, len(st))} px.\n"
                         f"달라진 프레임(겹침 90% 미만): {changed}개.")
            else:
                text += "."
            if lost:
                text += f"\n수정 후 마스크가 없는 프레임: {lost}개."
            text += "\n시험 추적은 추정입니다. 실제 결과는 저장 후 다시 처리해야 확정됩니다."
            self.compare_info.setText(text)

    def _end_compare(self):
        if self._compare is not None:
            self._compare = None
            self.timeline_view.set_highlight(None)
            self._refresh_compare_panel()

    def goto_biggest_change(self):
        cmp = self._compare
        if cmp is None or not cmp["ready"]:
            return
        st = cmp["stats"]
        i = min(range(len(st)), key=lambda k: st[k]["iou"]) if cmp["before"] is not None else min(range(len(st)), key=lambda k: st[k]["after_px"])
        self.goto(cmp["start"] + i)

    def keep_correction(self):
        if self._compare is not None:
            self._end_compare()
            self._update_overlay()
            self.statusBar().showMessage("수정을 유지합니다. 포인트를 저장하고 다시 처리하면 영상에 반영됩니다.")

    def revert_correction(self):
        cmp = self._compare
        if cmp is None:
            return
        key = (cmp["oid"], cmp["start"])
        base = self._brush_base.get(key)
        if self.store.get_mask(*key) is None:
            self.statusBar().showMessage("이 프레임에는 되돌릴 그린 마스크가 없습니다 (클릭만 있는 프레임).")
        elif base is None and not self.ask_confirm("이 프레임의 그린 마스크는 이전에 저장된 것입니다. 지울까요?"):
            return
        else:
            self._brush_undo.append((key[0], key[1], self.store.get_mask(*key).copy()))
            self.store.set_mask(key[0], key[1], base[0] if base is not None and base[1] == "drawn" else None)
            self._brush_base.pop(key, None)             # back to where it started: no correction record for it
            self.statusBar().showMessage(f"객체 {key[0]}, 프레임 {key[1]}의 수정을 취소했습니다.")
        self._end_compare()
        self._points_changed()

    def _compare_overlay(self) -> Optional[dict]:
        """The overlay of the comparison for the current frame, or None when the frame is outside the compared range."""
        cmp = self._compare
        if cmp is None:
            return None
        i = self.frame_idx - cmp["start"]
        if not (0 <= i < cmp["n"]):
            return None
        if not cmp["ready"]:
            return {"note": "비교: 시험 추적 중 ..."}
        a = cmp["after"][i]
        b = cmp["before"][i] if cmp["before"] is not None and i < len(cmp["before"]) else None
        shape = cmp["shape"] or (self.info.height, self.info.width)
        zero = np.zeros(shape, dtype=bool)
        a = zero if a is None else a
        has_before = cmp["before"] is not None
        b = zero if b is None else b
        mode = self.compare_mode.currentData() if has_before else "after"
        col = object_color(cmp["oid"])
        st = cmp["stats"][i]
        head = f"[비교 {i + 1}/{cmp['n']}] "
        if mode == "before":
            imgs, note = [mask_image(b, col, 130)], head + f"수정 전: {st['before_px']} px"
        elif mode == "after":
            imgs = [mask_image(a, col, 130)]
            note = head + f"수정 후: {st['after_px']} px" + ("" if has_before else "  (비교할 수정 전 마스크가 없습니다)")
        else:
            imgs = [mask_image(a & b, col, 90), mask_image(a & ~b, QColor("#35d07f"), 190), mask_image(b & ~a, QColor("#ff5050"), 190)]
            note = head + f"초록 = 수정으로 늘어난 부분, 빨강 = 줄어든 부분.  수정 전 {st['before_px']} px → 수정 후 {st['after_px']} px"
        return {"size": (shape[1], shape[0]), "masks": imgs, "note": note, "boxes": [], "regions": [], "full": False}

    def _update_overlay(self):
        """Decide what to draw over the current frame, then add the brush hint when the brush is on."""
        cmp_overlay = self._compare_overlay()
        if cmp_overlay is not None:
            self.view.set_overlay(cmp_overlay)
            return
        self._update_overlay_base()
        if self.info is not None and self.frame_idx >= 0 and self.act_brush.isChecked():
            ov = dict(self.view.overlay() or {})
            hint = f"[브러시: 객체 {self.store.current}] 왼쪽 드래그 = 칠하기, 오른쪽 드래그 = 지우기."
            ov["note"] = hint + ("  " + ov["note"] if ov.get("note") else "")
            self.view.set_overlay(ov)

    def _update_overlay_base(self):
        """Hand-drawn masks of this frame first, else the live preview where the frame has clicks, else the last run's masks."""
        if self.info is None or self.frame_idx < 0:
            self.view.set_overlay(None)
            return
        f = self.frame_idx
        drawn = {oid: frames[f] for oid, frames in self.store.masks.items() if f in frames}
        if drawn:
            shape = next(iter(drawn.values())).shape
            ov = self._make_overlay(shape, drawn, note="그린 마스크: 이 프레임에서는 클릭 대신 이 마스크에서 추적을 시작합니다.")
            self.view.set_overlay(ov)
            return
        pts = [p for p in self.store.points if p["frame_idx"] == f]
        if self.preview.is_on() and pts:
            sig = tuple(sorted((p["obj_id"], round(p["x"], 2), round(p["y"], 2), p["label"]) for p in pts))
            hit = self._live_masks.get(f)
            if hit is not None and hit[0] == sig:
                note = "이 프레임의 클릭 미리보기 (여백 적용 전 마스크)."
                if hit[2]:
                    note += "  표시 안 됨: " + "; ".join(f"객체 {o} - {r}" for o, r in sorted(hit[2].items()))
                shape = next(iter(hit[1].values())).shape if hit[1] else (self.info.height, self.info.width)
                self.view.set_overlay(self._make_overlay(shape, hit[1], note=note))
                return
            self.view.set_overlay({"note": "마스크 계산 중 ..." if self.preview.is_ready() else "마스크 미리보기용 모델을 불러오는 중 ..."})
            if self._preview_asked != (f, sig):
                objects: dict = {}
                for p in pts:
                    o = objects.setdefault(p["obj_id"], {"points": [], "labels": []})
                    o["points"].append([p["x"], p["y"]])
                    o["labels"].append(p["label"])
                rid = self.preview.request(f, objects)
                self._preview_reqs[rid] = (f, sig)
                self._preview_asked = (f, sig)
            return
        if self._job_masks is not None:
            fr = self._job_masks.frame(f)
            stale = self.dirty or (self._job_masks.points_sha is not None and self._job_masks.points_sha != self._points_sha())
            tail = "  그 실행 이후 포인트가 바뀌었습니다." if stale else ""
            if fr is None:
                self.view.set_overlay({"note": "지난 실행에 이 프레임의 마스크가 없습니다." + tail})
            else:
                note = f"지난 실행 ({self._job_masks.created}): 여백 적용 전 마스크, 탐지 박스, 안전 모드 영역(빨강)." + tail
                self.view.set_overlay(self._make_overlay(fr["shape"], fr["masks"], fr["boxes"], fr["fc_regions"], fr["fc_full"], note))
            return
        if self.preview.is_on():
            self.view.set_overlay({"note": "마스크 미리보기 켜짐: 이 프레임에서 대상을 클릭하면 SAM 2가 잡는 영역이 보입니다."})
            return
        self.view.set_overlay(None)

    def _on_preview_masks(self, rid: int, frame: int, masks: dict, skipped: dict):
        req = self._preview_reqs.pop(rid, None)
        if req is None:
            return
        self._live_masks[frame] = (req[1], masks, skipped)
        while len(self._live_masks) > 60:
            self._live_masks.pop(next(iter(self._live_masks)))
        if frame == self.frame_idx:
            self._update_overlay()

    def _on_preview_failed(self, rid: int, reason: str):
        if self._compare is not None and rid in self._compare["pending"]:
            self._end_compare()
            self.statusBar().showMessage("비교용 시험 추적에 실패했습니다: " + reason)
            self.show_error("비교용 시험 추적에 실패했습니다:\n" + reason)
            self._update_overlay()
            return
        self._preview_reqs.pop(rid, None)
        self._preview_asked = None
        self.statusBar().showMessage("이 프레임의 마스크 미리보기에 실패했습니다: " + reason)
        self.view.set_overlay({"note": "마스크 미리보기 실패: " + reason})

    def _on_preview_state(self, state: str, detail: str):
        if state == "failed":
            self._set_checked(self.act_live, False)
            self.preview.stop()
            self.show_error("마스크 미리보기를 시작하지 못했습니다:\n" + detail)
        elif state == "loading":
            self.statusBar().showMessage("마스크 미리보기: 모델을 불러오는 중 (몇 초 걸립니다) ...")
        elif state == "ready":
            self.statusBar().showMessage("마스크 미리보기 준비 완료: 대상을 클릭하면 이 프레임에서 SAM 2가 잡는 영역이 보입니다.")
        if state != "off":
            self._update_overlay()

    # ------------------------------------------------------------------ dialogs (overridable for tests)
    def show_error(self, text: str):
        log.error(text)
        QMessageBox.critical(self, "Video Censor", text)

    def ask_save_changes(self) -> str:
        """'save' | 'discard' | 'cancel'"""
        B = QMessageBox.StandardButton
        r = QMessageBox.question(self, "저장하지 않은 포인트", "포인트에 저장하지 않은 변경이 있습니다.", B.Save | B.Discard | B.Cancel)
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
        path, _ = QFileDialog.getOpenFileName(self, "영상 열기", "", VIDEO_FILTER)
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
        self.statusBar().showMessage("영상을 열지 못했습니다.")
        self.show_error(f"영상을 열지 못했습니다:\n{msg}")

    def _video_probed(self, info: VideoInfo):
        self.setEnabled(True)
        if self.server is not None:
            self.server.close()
        self.info = info
        self._set_checked(self.act_live, False)          # overlays belong to the previous video
        self._set_checked(self.act_last, False)
        self.preview.stop()
        self._live_masks.clear()
        self._preview_reqs.clear()
        self._preview_asked, self._job_masks = None, None
        self._set_checked(self.act_brush, False)
        self._end_compare()
        self.view.set_brush(False)
        self._brush_base.clear()
        self._brush_undo.clear()
        self._stroke_key = None
        self.view.set_overlay(None)
        self.review = None
        self.review_panel.set_data(None)
        self.slider.set_flags([])
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
        self.timeline_view.set_video(info.nb_frames, info.fps)
        self.timeline_view.setEnabled(True)
        self._start_thumbnails()
        self._load_run_summary()
        for b in self._nav_buttons:
            b.setEnabled(True)
        self._show_info()
        self._refresh_objects()
        self._update_title()
        msg = f"{os.path.basename(info.path)} 열림"
        if os.path.isfile(self.points_path):
            msg += self._try_load_points(self.points_path, quiet=True)
        self.statusBar().showMessage(msg)
        self.check_resumable()
        self.frame_idx = -1
        self.goto(0)

    def _show_info(self):
        i = self.info
        lines = [f"<b>{os.path.basename(i.path)}</b>", f"{i.width}x{i.height}, {i.fps:.3f} fps, {i.nb_frames}프레임 ({i.duration:.1f}초)",
                 f"코덱 {i.codec}, {i.pix_fmt}", f"오디오: {i.audio_codec if i.has_audio else '없음'}"]
        warn = []
        if i.is_vfr:
            warn.append(f"가변 프레임률 ({i.vfr_detail}). 허용하지 않으면 처리가 거부됩니다. 여기 표시되는 시간은 실제 패킷 시간입니다.")
        if is_hdr(i):
            warn.append("HDR / 고비트 영상: 처리하면 8비트 SDR로 변환됩니다.")
        if i.rotation:
            warn.append(f"회전 정보 {i.rotation}도를 적용했습니다 (처리될 모습 그대로 표시).")
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
        self.timeline_view.set_frame(idx)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.view.set_image(self._cache[idx])
            self._awaiting = False
        else:
            self._awaiting = True
            self.server.request(idx)
        self._update_overlay()
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
        self.statusBar().showMessage(f"프레임 {idx}: {msg}")
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
        self.pos_label.setText(f"프레임 {self.frame_idx}/{i.nb_frames - 1}   t = {i.timestamp_of(self.frame_idx):.3f}초")

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
        self.play_btn.setText("일시정지")
        self._play_step = max(1, int(round(self.info.fps / 30.0)))
        self.timer.start(int(1000 * self._play_step / self.info.fps))

    def _stop_play(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("재생")

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
        if self.job_running:
            self.statusBar().showMessage("작업이 실행 중입니다. 끝나거나 취소될 때까지 포인트를 바꿀 수 없습니다.")
            return
        self._stop_play()
        self.store.add(self.frame_idx, fx, fy, label)
        self._points_changed()

    def _points_changed(self):
        self.dirty = True
        cmp = self._compare
        if cmp is not None and self._prompt_sig(self._compare_prompts(cmp["oid"], cmp["start"], cmp["n"], corrected=True)) != cmp["sig"]:
            self._end_compare()                        # the prompts were edited: the comparison no longer describes them
        self._refresh_stale_banner()
        self._refresh_objects()
        self._update_title()
        self._update_overlay()
        self.view.update()

    def new_object(self):
        oid = self.store.new_object()
        self._refresh_objects()
        self.statusBar().showMessage(f"새 객체 {oid}: 영상에서 대상을 왼쪽 클릭하세요")

    def delete_object(self):
        n = sum(1 for p in self.store.points if p["obj_id"] == self.store.current)
        if n and not self.ask_confirm(f"객체 {self.store.current}와(과) 그 점 {n}개를 삭제할까요?"):
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
        self._load_style_widgets()

    def _load_style_widgets(self):
        """Show the mosaic settings of the current object without triggering an edit."""
        st = self.store.styles.get(self.store.current, {})
        for w in (self.obj_style, self.obj_block, self.obj_pad):
            w.blockSignals(True)
        self.obj_style.setCurrentIndex(max(0, self.obj_style.findData(st.get("style"))))
        self.obj_block.setValue(int(st.get("block", 0)))
        self.obj_pad.setValue(int(round(st["pad_px"])) if "pad_px" in st else -1)
        for w in (self.obj_style, self.obj_block, self.obj_pad):
            w.blockSignals(False)

    def _style_edited(self, *_):
        if self.job_running:
            self._load_style_widgets()
            return
        if self.obj_block.value() == 1:                    # 0 means "job setting", the smallest real strength is 2
            self.obj_block.setValue(2)
            return
        oid = self.store.current
        changed = self.store.set_style(oid, "style", self.obj_style.currentData())
        changed |= self.store.set_style(oid, "block", self.obj_block.value() or None)
        changed |= self.store.set_style(oid, "pad_px", float(self.obj_pad.value()) if self.obj_pad.value() >= 0 else None)
        if changed:
            self._points_changed()

    def _refresh_objects(self):
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        ids = self.store.object_ids()
        for oid in ids:
            pts = [p for p in self.store.points if p["obj_id"] == oid]
            drawn_frames = sorted(self.store.masks.get(oid, {}))
            first = min([p["frame_idx"] for p in pts] + drawn_frames, default=None)
            text = f"객체 {oid}: {sum(1 for p in pts if p['label'] == 1)} +, {sum(1 for p in pts if p['label'] == 0)} -"
            if drawn_frames:
                text += f", 그린 마스크 {len(drawn_frames)}"
            text += f"  (프레임 {first}부터)" if first is not None else "  (아직 점 없음)"
            st = self.store.styles.get(oid)
            if st:
                text += "  [" + ", ".join(([st["style"]] if "style" in st else []) + ([f"{st['block']} px"] if "block" in st else [])
                                          + ([f"여백 {st['pad_px']:g}"] if "pad_px" in st else [])) + "]"
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, oid)
            pm = QPixmap(14, 14)
            pm.fill(object_color(oid))
            it.setIcon(QIcon(pm))
            self.obj_list.addItem(it)
            if oid == self.store.current:
                self.obj_list.setCurrentItem(it)
        self.obj_list.blockSignals(False)
        self.timeline_view.set_objects([(oid, object_color(oid), sorted({p["frame_idx"] for p in self.store.points if p["obj_id"] == oid}
                                                                         | set(self.store.masks.get(oid, {}))))
                                        for oid in ids])
        self._load_style_widgets()

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
        path, _ = QFileDialog.getSaveFileName(self, "포인트 저장", self.points_path or "", "포인트 (*.json)")
        return self._write_points(path) if path else False

    def _write_points(self, path: str) -> bool:
        probs = self.store.valid()
        if probs:
            self.show_error("포인트를 저장하지 않았습니다:\n" + "\n".join(probs) + "\n각 항목에 긍정(왼쪽 클릭) 점을 하나 이상 추가하세요.")
            return False
        try:
            save_points(path, self.info, self.store.points, styles=self.store.styles, masks=self.store.masks)
        except OSError as e:
            self.show_error(f"{path}에 쓸 수 없습니다:\n{e}")
            return False
        self.points_path = path
        self.dirty = False
        self._write_corrections()
        self._update_title()
        self._refresh_stale_banner()
        self.statusBar().showMessage(f"점 {len(self.store.points)}개를 {path}에 저장했습니다")
        return True

    def open_points_dialog(self):
        if self.info is None:
            self.show_error("먼저 영상을 여세요.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "포인트 열기", os.path.dirname(self.info.path), "포인트 (*.json)")
        if path:
            self.statusBar().showMessage(self._try_load_points(path, quiet=False).strip(" -"))

    def _try_load_points(self, path: str, quiet: bool) -> str:
        try:
            prompts = load_points(path, self.info)
            styles = load_object_styles(path)
            drawn = load_mask_prompts(path, self.info)
        except (PointsError, OSError, ValueError, KeyError) as e:
            if not quiet:
                self.show_error(f"이 포인트 파일을 쓸 수 없습니다:\n{e}")
            return f" - 포인트 파일 무시됨: {e}"
        if self.dirty and not self._maybe_save_before_leaving():
            return ""
        store = PointStore()
        for oid, frames in prompts.items():
            for fr, (pts, labs) in frames.items():
                for (x, y), lab in zip(pts, labs):
                    store.points.append({"obj_id": oid, "frame_idx": fr, "x": x, "y": y, "label": lab})
        store.masks = drawn
        ids = sorted({p["obj_id"] for p in store.points} | set(drawn))
        if ids:
            store.objects, store.current = ids, ids[0]
        self._brush_base.clear()
        self._brush_undo.clear()
        store.styles = {k: v for k, v in styles.items() if k in ids}
        self.store, self.points_path, self.dirty = store, path, False
        self._refresh_objects()
        self._update_title()
        self.view.update()
        return f" - {os.path.basename(path)}에서 점 {len(store.points)}개를 불러왔습니다"

    def _update_title(self):
        name = os.path.basename(self.info.path) if self.info else ""
        self.setWindowTitle(f"{'* ' if self.dirty else ''}{name + ' - ' if name else ''}Video Censor")

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, e):
        if self.job_running:
            if not self.ask_confirm("처리 작업이 실행 중입니다. 취소하고 종료할까요? (완료된 청크는 이어하기용으로 보관됩니다)"):
                e.ignore()
                return
            self.cancel_processing()
            end = time.time() + self.runner.grace_ms / 1000 + 10
            while self.job_running and time.time() < end:
                QApplication.processEvents()
                time.sleep(0.02)
        if not self._maybe_save_before_leaving():
            e.ignore()
            return
        self._stop_play()
        self.preview.stop()
        for loader in (self._thumb_loader, self._summary_loader):
            if loader is not None:
                loader.cancelled = True
        if self.server is not None:
            self.server.close()
        e.accept()
