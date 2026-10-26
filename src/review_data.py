"""Review log as data for the review screen: loading, filtering, 'reviewed' marks, and staleness checks.

Consistency model (log <-> points <-> output video):
  * every run writes ``run_id``, ``points_sha`` (sha256 of the points file it used) and ``output`` (path, size, mtime) into the review log meta;
  * the app compares them with the current points file and output file and tells the user when the log describes an older state;
  * 'reviewed' marks live in a sidecar file next to the log. When a re-run replaces the log, marks are kept only for flags in chunks
    that were REUSED (their content is unchanged); flags in re-processed chunks start unreviewed again, because they were recomputed.
All flags are heuristics (see README): 'reviewed' means a human looked, not that the frame is correct.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .pipeline import file_sha
from .review import SEV, ReviewError, ReviewNavigator
from .video_io import VideoInfo


def seg_key(seg: dict) -> str:
    return f"{seg['type']}|{seg.get('object_id')}|{seg['start_frame']}"


def state_path_for(review_path: str) -> str:
    base, _ = os.path.splitext(review_path)
    return base + "_state.json"


class ReviewState:
    def __init__(self, path: str):
        self.path = path
        self.run_id: Optional[str] = None
        self.reviewed: Dict[str, dict] = {}

    @classmethod
    def load(cls, path: str) -> "ReviewState":
        st = cls(path)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            st.run_id, st.reviewed = d.get("run_id"), dict(d.get("reviewed", {}))
        except (OSError, ValueError):
            pass
        return st

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "reviewed": self.reviewed}, f, indent=1)
        os.replace(tmp, self.path)

    def is_reviewed(self, seg: dict) -> bool:
        return seg_key(seg) in self.reviewed

    def set_reviewed(self, seg: dict, value: bool = True, note: str = ""):
        k = seg_key(seg)
        if value:
            self.reviewed[k] = {"time": time.strftime("%F %T"), "note": note}
        else:
            self.reviewed.pop(k, None)
        self.save()

    def rebase(self, meta: dict, segments: List[dict]) -> int:
        """Adopt a new run: keep marks only for flags inside reused chunks. Returns how many marks were dropped."""
        if self.run_id == meta.get("run_id"):
            return 0
        first_recomputed = (meta.get("reused_chunks") or 0) * (meta.get("chunk_size") or 0)
        keep = {seg_key(s) for s in segments if s["start_frame"] < first_recomputed}
        before = len(self.reviewed)
        self.reviewed = {k: v for k, v in self.reviewed.items() if k in keep} if self.run_id else {}
        self.run_id = meta.get("run_id")
        self.save()
        return before - len(self.reviewed)


class ReviewData:
    def __init__(self, path: str, doc: dict):
        self.path, self.doc = path, doc
        self.meta: dict = doc.get("meta", {})
        self.segments: List[dict] = doc.get("segments", [])
        self.events: List[dict] = doc.get("events", [])
        self.state = ReviewState.load(state_path_for(path))
        self.dropped_marks = self.state.rebase(self.meta, self.segments)

    def object_ids(self) -> List[int]:
        return sorted({s["object_id"] for s in self.segments if s.get("object_id") is not None})

    def filtered(self, min_severity: str = "info", object_id: Optional[int] = None, hide_reviewed: bool = False) -> List[dict]:
        out = []
        for s in self.segments:
            if s["start_frame"] < 0 or SEV.get(s["severity"], 0) < SEV[min_severity]:
                continue
            if object_id is not None and s.get("object_id") not in (None, object_id):
                continue
            if hide_reviewed and self.state.is_reviewed(s):
                continue
            out.append(s)
        return sorted(out, key=lambda s: (s["start_frame"], -SEV.get(s["severity"], 0), s["type"]))

    def navigator(self, min_severity: str = "warn") -> ReviewNavigator:
        return ReviewNavigator(self.segments, min_severity)

    def unreviewed_count(self, min_severity: str = "warn") -> int:
        return sum(1 for s in self.filtered(min_severity) if not self.state.is_reviewed(s))

    def staleness(self, points_path: Optional[str], has_points: bool) -> List[str]:
        """Human-readable reasons why this log may not describe the current points/output. Empty = consistent."""
        out = []
        cur = file_sha(points_path) if points_path else None
        old = self.meta.get("points_sha")
        if has_points and old is None and cur is not None:
            out.append("this log was made without clicked points, but points exist now")
        elif old is not None and cur != old:
            out.append("the points file changed after this log was made - re-process to refresh the flags and the video")
        o = self.meta.get("output")
        if o:
            try:
                st = os.stat(o["path"])
                if st.st_size != o["size"] or abs(st.st_mtime - o["mtime"]) > 2:
                    out.append("the output video was modified after this log was made")
            except OSError:
                out.append("the output video this log describes no longer exists")
        elif self.meta.get("output", "missing") is None:
            out.append("the run that wrote this log did not produce a verified output video")
        return out


def load_review_data(path: str, info: VideoInfo) -> ReviewData:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise ReviewError(f"cannot read the review log: {e}") from e
    if "segments" not in doc:
        raise ReviewError("not a review log (no 'segments')")
    fp = doc.get("meta", {}).get("input", {}).get("fingerprint")
    if fp and fp != info.fingerprint:
        raise ReviewError("this review log belongs to a different video file (fingerprint mismatch)")
    return ReviewData(path, doc)
