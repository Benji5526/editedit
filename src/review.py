"""Navigation over the review log (logs/<name>_review.json) for pick_points.py --review.

The review log is made of heuristics; jumping to a flag means "look here", not "this is wrong".
"""
from __future__ import annotations

import bisect
import json
from typing import Dict, List, Optional

from .video_io import VideoInfo

SEV = {"info": 0, "warn": 1, "error": 2}


class ReviewError(ValueError):
    pass


class ReviewNavigator:
    def __init__(self, segments: List[dict], min_severity: str = "warn"):
        floor = SEV[min_severity]
        self.segments = [s for s in segments if SEV.get(s["severity"], 0) >= floor and s["start_frame"] >= 0]
        frames = set()
        for s in self.segments:
            frames.add(s["start_frame"])
            frames.add(s["end_frame"])
        self.frames: List[int] = sorted(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def next_after(self, frame: int) -> Optional[int]:
        i = bisect.bisect_right(self.frames, frame)
        return self.frames[i] if i < len(self.frames) else None

    def prev_before(self, frame: int) -> Optional[int]:
        i = bisect.bisect_left(self.frames, frame)
        return self.frames[i - 1] if i > 0 else None

    def index_of(self, frame: int) -> int:
        """1-based position of the nearest flagged frame at or before ``frame`` (0 = before the first)."""
        return bisect.bisect_right(self.frames, frame)

    def segments_at(self, frame: int) -> List[dict]:
        return [s for s in self.segments if s["start_frame"] <= frame <= s["end_frame"]]

    def object_at(self, frame: int) -> Optional[int]:
        """Object id of the most severe segment touching this frame, if it names one."""
        segs = [s for s in self.segments_at(frame) if s.get("object_id") is not None]
        if not segs:
            return None
        return int(max(segs, key=lambda s: SEV.get(s["severity"], 0))["object_id"])

    def describe(self, frame: int, limit: int = 3) -> List[str]:
        out = []
        for s in sorted(self.segments_at(frame), key=lambda s: -SEV.get(s["severity"], 0))[:limit]:
            obj = f" obj {s['object_id']}" if s.get("object_id") is not None else ""
            span = f"{s['start_frame']}-{s['end_frame']}" if s["end_frame"] != s["start_frame"] else f"{s['start_frame']}"
            out.append(f"[{s['severity']}] {s['type']}{obj} @{span}")
        return out


def load_review(path: str, info: VideoInfo, min_severity: str = "warn") -> ReviewNavigator:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    fp = doc.get("meta", {}).get("input", {}).get("fingerprint")
    if fp and fp != info.fingerprint:
        raise ReviewError("review log belongs to a different video file (fingerprint mismatch)")
    if "segments" not in doc:
        raise ReviewError("not a review log (no 'segments')")
    return ReviewNavigator(doc["segments"], min_severity)
