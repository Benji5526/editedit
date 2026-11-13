"""Read-only history of the processing jobs the registry knows for the current video."""
from __future__ import annotations

import os
from typing import List

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .ko_text import status_text

HEADERS = ["시작한 시각", "상태", "끝난 조각", "다시 쓴 조각", "클릭 지점 파일 표식", "결과 영상", "오류 내용"]


def job_rows(jobs: List[dict]) -> List[List[str]]:
    rows = []
    for j in jobs:
        total = j.get("n_chunks") or "?"
        rows.append([j.get("created") or "", status_text(j.get("status")), f"{j.get('chunks_done', 0)}/{total}",
                     str(j.get("reused_chunks") or 0), (j.get("points_sha") or "")[:8],
                     os.path.basename(j["output_path"]) if j.get("output_path") else "", (j.get("error") or "")[:160]])
    return rows


class JobsDialog(QDialog):
    def __init__(self, parent, jobs: List[dict], video_name: str):
        super().__init__(parent)
        self.setWindowTitle(f"처리 작업 기록 - {video_name}")
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
        note = QLabel("최신 작업이 위에 있습니다. 이 목록은 logs/jobs.sqlite에 보관되는 색인이며, 조각을 실제로 재사용할 수 있는지는 "
                      "작업을 이어할 때 temp/ 폴더의 파일로 다시 확인합니다." if rows else "이 영상으로 기록된 작업이 아직 없습니다.")
        note.setWordWrap(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(note)
        lay.addWidget(self.table, 1)
        lay.addWidget(bb)
        self.resize(900, 380)
