"""Read-only history of the processing jobs the registry knows for the current video."""
from __future__ import annotations

import os
from typing import List

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout)

HEADERS = ["Started", "Status", "Chunks", "Reused", "Points", "Output", "Error"]


def job_rows(jobs: List[dict]) -> List[List[str]]:
    rows = []
    for j in jobs:
        total = j.get("n_chunks") or "?"
        rows.append([j.get("created") or "", j.get("status") or "", f"{j.get('chunks_done', 0)}/{total}",
                     str(j.get("reused_chunks") or 0), (j.get("points_sha") or "")[:8],
                     os.path.basename(j["output_path"]) if j.get("output_path") else "", (j.get("error") or "")[:160]])
    return rows


class JobsDialog(QDialog):
    def __init__(self, parent, jobs: List[dict], video_name: str):
        super().__init__(parent)
        self.setWindowTitle(f"Processing jobs - {video_name}")
        rows = job_rows(jobs)
        self.table = QTableWidget(len(rows), len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        note = QLabel("Newest first. This list is an index kept in logs/jobs.sqlite; whether chunks can really be reused is checked "
                      "from the files in temp/ when a job is resumed." if rows else "No jobs recorded for this video yet.")
        note.setWordWrap(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(note)
        lay.addWidget(self.table, 1)
        lay.addWidget(bb)
        self.resize(900, 380)
