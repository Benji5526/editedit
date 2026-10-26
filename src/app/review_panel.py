"""Review screen widgets: flagged-segment table with filters, and a slider that shows where the flags are."""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from src.review_data import ReviewData

SEV_COLOR = {"error": "#d64545", "warn": "#e0a030", "info": "#6a8fb8"}
HEADERS = ["Done", "Sev", "Type", "Obj", "Frames", "Time (s)", "Count", "Action"]


class FlagSlider(QSlider):
    """Frame slider that marks flagged frames (red = error, amber = warning)."""

    def __init__(self, *a):
        super().__init__(*a)
        self._flags: List[Tuple[int, str]] = []

    def set_flags(self, flags: List[Tuple[int, str]]):
        self._flags = flags
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._flags or self.maximum() <= 0:
            return
        p = QPainter(self)
        w = max(1, self.width() - 16)
        for frame, sev in self._flags:
            x = 8 + int(w * frame / self.maximum())
            p.fillRect(QRect(x - 1, 0, 3, 7), QColor(SEV_COLOR.get(sev, "#999")))


class ReviewPanel(QWidget):
    segment_selected = Signal(dict)
    next_requested = Signal()
    prev_requested = Signal()
    reprocess_requested = Signal()
    reviewed_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: Optional[ReviewData] = None
        self.banner = QLabel("No review log loaded.")
        self.banner.setWordWrap(True)
        self.sev = QComboBox()
        for text, val in (("warnings + errors", "warn"), ("errors only", "error"), ("everything (incl. info)", "info")):
            self.sev.addItem(text, val)
        self.obj = QComboBox()
        self.obj.addItem("all objects", None)
        self.hide_done = QCheckBox("hide reviewed")
        self.summary = QLabel("")
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.prev_btn = QPushButton("Prev flag (B)")
        self.next_btn = QPushButton("Next flag (M)")
        self.done_btn = QPushButton("Toggle reviewed (R)")
        self.reproc_btn = QPushButton("Re-process with edited points")
        row1 = QHBoxLayout()
        for w in (self.sev, self.obj, self.hide_done):
            row1.addWidget(w)
        row2 = QHBoxLayout()
        for b in (self.prev_btn, self.next_btn, self.done_btn):
            row2.addWidget(b)
        lay = QVBoxLayout(self)
        for w in (self.banner,):
            lay.addWidget(w)
        lay.addLayout(row1)
        lay.addWidget(self.summary)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.detail)
        lay.addLayout(row2)
        lay.addWidget(self.reproc_btn)
        self.sev.currentIndexChanged.connect(self.refresh)
        self.obj.currentIndexChanged.connect(self.refresh)
        self.hide_done.toggled.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.cellChanged.connect(self._cell_changed)
        self.prev_btn.clicked.connect(self.prev_requested)
        self.next_btn.clicked.connect(self.next_requested)
        self.done_btn.clicked.connect(self.toggle_reviewed)
        self.reproc_btn.clicked.connect(self.reprocess_requested)
        self._rows: List[dict] = []
        self._loading = False

    # ---- data
    def set_data(self, data: Optional[ReviewData], stale: Optional[List[str]] = None):
        self.data = data
        self._loading = True
        self.obj.clear()
        self.obj.addItem("all objects", None)
        if data is not None:
            for oid in data.object_ids():
                self.obj.addItem(f"object {oid}", oid)
        self._loading = False
        self.set_banner(stale)
        self.refresh()

    def set_banner(self, stale: Optional[List[str]]):
        if self.data is None:
            self.banner.setText("No review log loaded. Process a video or use File > Open review log.")
            self.banner.setStyleSheet("")
        elif stale:
            self.banner.setText("Out of date: " + "; ".join(stale))
            self.banner.setStyleSheet("color: #c0392b; font-weight: bold")
        else:
            m = self.data.meta
            self.banner.setText(f"Log of run {m.get('run_id', '?')} ({m.get('created', '')}); consistent with the current points and output."
                                + (f" Chunks 1-{m['reused_chunks']} were reused from an earlier run." if m.get("reused_chunks") else ""))
            self.banner.setStyleSheet("color: #2e7d32")

    def refresh(self, *_):
        if self._loading:
            return
        self._loading = True
        self.table.setRowCount(0)
        self._rows = []
        if self.data is not None:
            rows = self.data.filtered(self.sev.currentData(), self.obj.currentData(), self.hide_done.isChecked())
            self._rows = rows
            self.table.setRowCount(len(rows))
            for r, s in enumerate(rows):
                done = QTableWidgetItem()
                done.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                done.setCheckState(Qt.CheckState.Checked if self.data.state.is_reviewed(s) else Qt.CheckState.Unchecked)
                fr = f"{s['start_frame']}" if s["end_frame"] == s["start_frame"] else f"{s['start_frame']}-{s['end_frame']}"
                ts = s.get("start_timestamp")
                vals = [None, s["severity"], s["type"], "" if s.get("object_id") is None else str(s["object_id"]), fr,
                        "" if ts is None else f"{ts:.2f}", str(s["event_count"]), ", ".join(s.get("actions", []))]
                self.table.setItem(r, 0, done)
                for c in range(1, len(HEADERS)):
                    it = QTableWidgetItem(vals[c])
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if c == 1:
                        it.setForeground(QColor(SEV_COLOR.get(s["severity"], "#999")))
                    self.table.setItem(r, c, it)
            self.table.resizeColumnsToContents()
            left = self.data.unreviewed_count(self.sev.currentData())
            self.summary.setText(f"{len(rows)} flags shown, {left} not yet reviewed (flags are heuristics; 'reviewed' = a human looked)")
        else:
            self.summary.setText("")
        self._loading = False

    # ---- interaction
    def current_segment(self) -> Optional[dict]:
        r = self.table.currentRow()
        return self._rows[r] if 0 <= r < len(self._rows) else None

    def select_row_for(self, frame: int):
        """Highlight the first row whose segment starts at/before ``frame`` and contains it (used by next/prev)."""
        for r, s in enumerate(self._rows):
            if s["start_frame"] <= frame <= s["end_frame"]:
                self._loading = True
                self.table.selectRow(r)
                self._loading = False
                self._show_detail(s)
                return

    def _selected(self):
        if self._loading:
            return
        s = self.current_segment()
        if s is not None:
            self._show_detail(s)
            self.segment_selected.emit(s)

    def _show_detail(self, s: dict):
        self.detail.setText(f"{s['type']} (object {s.get('object_id')}): {s.get('first_detail', '')}")

    def _cell_changed(self, row: int, col: int):
        if self._loading or col != 0 or self.data is None or not (0 <= row < len(self._rows)):
            return
        checked = self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        self.data.state.set_reviewed(self._rows[row], checked)
        self._after_reviewed_change()

    def toggle_reviewed(self):
        s = self.current_segment()
        if s is None or self.data is None:
            return
        self.data.state.set_reviewed(s, not self.data.state.is_reviewed(s))
        self._after_reviewed_change()

    def _after_reviewed_change(self):
        cur = self.table.currentRow()
        self.refresh()
        if 0 <= cur < self.table.rowCount():
            self._loading = True
            self.table.selectRow(cur)
            self._loading = False
        self.reviewed_changed.emit()
