"""Detections the user marked as "not a target" (false positives of the detector, e.g. a hand).

An exclusion is a frame range and a box in full-resolution pixels: {"frame_start", "frame_end", "box": [x1, y1, x2, y2], "cls"}.
A detection (or a conservative region that came from one) whose centre lies inside the box on a frame of the range is not
mosaicked. Exclusions are a render-only setting: the detector does not run again when they change. The file holds no video
name or path, only the video's fingerprint, so the entries can later serve as negative examples for training a detector."""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

VERSION = 1


def normalize(items) -> List[dict]:
    out = []
    for it in items or []:
        a, b = int(it["frame_start"]), int(it["frame_end"])
        box = [float(v) for v in it["box"]]
        if len(box) != 4 or a > b or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"bad exclusion {it!r}")
        e = {"frame_start": a, "frame_end": b, "box": [round(v, 1) for v in box]}
        if it.get("cls"):
            e["cls"] = str(it["cls"])
        out.append(e)
    return sorted(out, key=lambda e: (e["frame_start"], e["frame_end"], e["box"]))


def save_exclusions(path: str, fingerprint: str, items) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "fingerprint": fingerprint, "exclusions": normalize(items)}, f, indent=1)
    os.replace(tmp, path)


def load_exclusions(path: str, fingerprint: Optional[str] = None) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise ValueError(f"unsupported exclusions file version {doc.get('version')!r}")
    if fingerprint and doc.get("fingerprint") and doc["fingerprint"] != fingerprint:
        raise ValueError("the exclusions file belongs to a different video")
    return normalize(doc.get("exclusions"))


def is_excluded(items: Sequence[dict], frame: int, box, cls: Optional[str] = None) -> bool:
    """box in full-resolution pixels. cls None = a region without a class (conservative mosaic): matched by place only."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for e in items:
        if e["frame_start"] <= frame <= e["frame_end"] and (cls is None or not e.get("cls") or e["cls"] == cls):
            b = e["box"]
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
    return False


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)


def exclusion_for(boxes_at, frame: int, x: float, y: float, total_frames: int, max_gap: int = 90, grow: float = 0.15) -> Optional[dict]:
    """Build the exclusion for the detection the user clicked.

    boxes_at(frame) -> [(x1, y1, x2, y2, cls)] in full-resolution pixels (the last run's detections), or None when unknown.
    The clicked box is followed backwards and forwards through the frames while a box of the same class overlaps the previous
    one (gaps of up to max_gap frames are bridged, the detector flickers). The exclusion covers all those boxes, grown a little."""
    start = [b for b in (boxes_at(frame) or []) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if not start:
        return None
    first = min(start, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))          # the smallest box under the click
    cls = first[4]
    lo = hi = frame
    union = list(first[:4])
    for step in (-1, 1):
        cur, f, miss = first, frame + step, 0
        while 0 <= f < total_frames and miss <= max_gap:
            cand = [b for b in (boxes_at(f) or []) if b[4] == cls and (_iou(b, cur) > 0.05 or
                                                                      (cur[0] <= (b[0] + b[2]) / 2 <= cur[2] and cur[1] <= (b[1] + b[3]) / 2 <= cur[3]))]
            if cand:
                cur = max(cand, key=lambda b: _iou(b, cur))
                union = [min(union[0], cur[0]), min(union[1], cur[1]), max(union[2], cur[2]), max(union[3], cur[3])]
                lo, hi, miss = min(lo, f), max(hi, f), 0
            else:
                miss += 1
            f += step
    w, h = union[2] - union[0], union[3] - union[1]
    return {"frame_start": lo, "frame_end": hi, "cls": cls,
            "box": [union[0] - grow * w, union[1] - grow * h, union[2] + grow * w, union[3] + grow * h]}
