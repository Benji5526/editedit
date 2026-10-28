"""Review log + heuristics that decide which frames are uncertain.

All rules here are HEURISTICS. A mask that disappears may be a tracking failure or the object may
really have left the frame / been fully occluded; the code cannot tell these apart reliably.
Events therefore never prove an error - they mark places a human should look at.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .mosaic import Box, mask_bbox

# event types
LOST_MID_FRAME = "tracking_lost_mid_frame"
LOST_AT_EDGE = "tracking_lost_at_edge"
RECOVERED = "tracking_recovered"
AREA_JUMP = "mask_area_jump"
DET_NO_MASK = "detection_without_mask"
CHUNK_RESUME_OK = "chunk_boundary_resumed"
CHUNK_RESUME_STALE = "chunk_boundary_resumed_from_stale_mask"
CHUNK_RESUME_FAILED = "chunk_boundary_resume_failed"
OBJECT_ABANDONED = "tracking_abandoned"
MODEL_ERROR = "model_error"
FRAME_ERROR = "frame_error"
COUNT_MISMATCH = "frame_count_mismatch"
OUTPUT_INVALID = "output_verification_failed"
FAIL_CLOSED_APPLIED = "fail_closed_mosaic_applied"
TIMESTAMP_WARNING = "timestamp_warning"
AUDIO_REENCODED = "audio_reencoded"

SEVERITY_ORDER = {"info": 0, "warn": 1, "error": 2}


@dataclass
class Event:
    frame: int
    type: str
    severity: str = "warn"
    object_id: Optional[int] = None
    timestamp: Optional[float] = None
    detail: str = ""
    action: str = "logged"   # what the program did about it
    extra: dict = field(default_factory=dict)


class QCLog:
    def __init__(self, ts_fn: Optional[Callable[[int], float]] = None):
        self.events: List[Event] = []
        self._ts = ts_fn

    def add(self, frame: int, type: str, severity: str = "warn", object_id: Optional[int] = None,
            detail: str = "", action: str = "logged", **extra) -> Event:
        ts = self._ts(frame) if (self._ts and frame >= 0) else None
        ev = Event(frame=int(frame), type=type, severity=severity, object_id=object_id,
                   timestamp=None if ts is None else round(float(ts), 6), detail=detail, action=action, extra=extra)
        self.events.append(ev)
        return ev

    def extend(self, events: Iterable[Event]):
        self.events.extend(events)

    def segments(self, gap: int = 1) -> List[dict]:
        """Merge consecutive-frame events of the same type/object into review segments."""
        groups: Dict[Tuple[str, Optional[int]], List[Event]] = {}
        for e in sorted(self.events, key=lambda e: e.frame):
            groups.setdefault((e.type, e.object_id), []).append(e)
        segs = []
        for (typ, obj), evs in groups.items():
            cur = [evs[0]]
            for e in evs[1:]:
                if e.frame - cur[-1].frame <= gap:
                    cur.append(e)
                else:
                    segs.append(self._seg(typ, obj, cur))
                    cur = [e]
            segs.append(self._seg(typ, obj, cur))
        segs.sort(key=lambda s: (s["start_frame"], s["type"]))
        return segs

    @staticmethod
    def _seg(typ, obj, evs: List[Event]) -> dict:
        sev = max((e.severity for e in evs), key=lambda s: SEVERITY_ORDER[s])
        return {"type": typ, "object_id": obj, "severity": sev,
                "start_frame": evs[0].frame, "end_frame": evs[-1].frame, "event_count": len(evs),
                "start_timestamp": evs[0].timestamp, "end_timestamp": evs[-1].timestamp,
                "actions": sorted({e.action for e in evs}), "first_detail": evs[0].detail}

    def to_dict(self, meta: Optional[dict] = None) -> dict:
        return {"meta": meta or {},
                "note": "Flags are heuristics. 'tracking_lost_*' may be a real disappearance/occlusion; absence of flags is NOT proof that nothing was missed.",
                "summary": self.summary(), "segments": self.segments(), "events": [asdict(e) for e in self.events]}

    def summary(self) -> dict:
        s: Dict[str, int] = {}
        for e in self.events:
            s[e.type] = s.get(e.type, 0) + 1
        return s

    def save(self, json_path: str, csv_path: Optional[str] = None, meta: Optional[dict] = None):
        _atomic_write(json_path, json.dumps(self.to_dict(meta), indent=1, ensure_ascii=False))
        if csv_path:
            tmp = csv_path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["frame", "timestamp", "type", "severity", "object_id", "action", "detail"])
                for e in sorted(self.events, key=lambda e: (e.frame, e.type)):
                    w.writerow([e.frame, e.timestamp, e.type, e.severity, e.object_id, e.action, e.detail])
            os.replace(tmp, csv_path)

    @classmethod
    def from_events(cls, events: List[dict], ts_fn=None) -> "QCLog":
        q = cls(ts_fn)
        for d in events:
            q.events.append(Event(**d))
        return q


def _atomic_write(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


@dataclass
class QCParams:
    min_area_frac: float = 5e-5      # mask smaller than this fraction of the frame counts as "no mask"
    jump_ratio: float = 2.0          # area change factor between adjacent frames that is suspicious
    edge_margin_frac: float = 0.02   # last bbox within this margin of the border => likely left the frame
    min_jump_area_frac: float = 1e-4 # ignore jumps when both areas are tiny (noise)
    edge_shrink_ratio: float = 0.7   # last area <= ratio * recent max counts as "shrinking"
    recover_warn_frames: int = 5     # a "recovery" after a longer loss is unverifiable (SAM 2 may have latched onto something else)
    far_ratio: float = 1.5           # reappearing farther than this many box diagonals from where it vanished


@dataclass
class Verdict:
    """Per object/frame outcome of the monitor."""
    status: str                       # "ok" | "lost" | "not_started"
    uncertain: Optional[str] = None   # reason when the result should not be trusted
    region: Optional[Box] = None      # where to apply fail-closed mosaic (analysis coordinates)


class ObjectMonitor:
    """Stateful per-object watcher for mask area and disappearance. State is serialisable so a
    restarted job continues with the same knowledge."""

    def __init__(self, shape: Tuple[int, int], params: Optional[QCParams] = None):
        self.shape = shape
        self.p = params or QCParams()
        self.state: Dict[int, dict] = {}

    # --- persistence
    def state_dict(self) -> dict:
        return {str(k): v for k, v in self.state.items()}

    def load_state(self, d: dict):
        self.state = {int(k): v for k, v in d.items()}

    def _min_area(self) -> float:
        return self.p.min_area_frac * self.shape[0] * self.shape[1]

    def _touches_edge(self, bbox: Box) -> bool:
        h, w = self.shape
        mx, my = self.p.edge_margin_frac * w, self.p.edge_margin_frac * h
        return bbox[0] <= mx or bbox[1] <= my or bbox[2] >= w - mx or bbox[3] >= h - my

    def observe(self, frame: int, obj: int, mask: Optional[np.ndarray], log: QCLog) -> Verdict:
        st = self.state.setdefault(obj, {"active": False, "last_area": 0, "last_bbox": None,
                                         "lost_since": None, "last_good_frame": None, "area_hist": []})
        area = int(mask.sum()) if mask is not None else 0
        present = area >= self._min_area() and area > 0
        if present:
            bb = mask_bbox(mask)
            verdict = Verdict("ok")
            if st["lost_since"] is not None:
                lost_n = frame - st["lost_since"]
                lb = st["last_bbox"]
                diag = max(1.0, float(np.hypot(lb[2] - lb[0], lb[3] - lb[1]))) if lb else 1.0
                dist = float(np.hypot((bb[0] + bb[2]) / 2 - (lb[0] + lb[2]) / 2, (bb[1] + bb[3]) / 2 - (lb[1] + lb[3]) / 2)) if lb else 0.0
                far = dist > self.p.far_ratio * diag
                notes = []
                if lost_n > self.p.recover_warn_frames:
                    notes.append(f"long loss ({lost_n} frames)")
                if far:
                    notes.append(f"reappeared {dist:.0f}px from where it vanished ({dist / diag:.1f} box diagonals)")
                log.add(frame, RECOVERED, "warn" if notes else "info", obj,
                        f"mask back after {lost_n} frames" + (": " + "; ".join(notes) + " - verify it is the same object" if notes else ""),
                        action="resumed_normal_unverified" if notes else "resumed_normal")
                st["lost_since"] = None
            elif st["active"] and st["last_area"] > 0:
                ratio = area / st["last_area"]
                big = max(area, st["last_area"]) >= self.p.min_jump_area_frac * self.shape[0] * self.shape[1]
                if big and (ratio > self.p.jump_ratio or ratio < 1.0 / self.p.jump_ratio):
                    lb = st["last_bbox"]
                    region = (min(bb[0], lb[0]), min(bb[1], lb[1]), max(bb[2], lb[2]), max(bb[3], lb[3]))
                    log.add(frame, AREA_JUMP, "warn", obj, f"area {st['last_area']} -> {area} (x{ratio:.2f})",
                            action="flagged")
                    verdict = Verdict("ok", uncertain=AREA_JUMP, region=region)
            st.update(active=True, last_area=area, last_bbox=list(bb), last_good_frame=frame)
            st["area_hist"] = (st.get("area_hist", []) + [area])[-6:]
            return verdict
        # no usable mask
        if not st["active"]:
            return Verdict("not_started")
        if st["lost_since"] is None:
            st["lost_since"] = frame
            hist = st.get("area_hist", [])
            # "left the frame" needs BOTH: bbox at the border AND a shrinking mask (sliding out). A mask that
            # vanishes at full size, or a bbox that merely touches the border, stays "mid-frame" (suspicious).
            shrinking = len(hist) >= 2 and hist[-1] <= self.p.edge_shrink_ratio * max(hist)
            edge = st["last_bbox"] is not None and self._touches_edge(tuple(st["last_bbox"])) and shrinking
            log.add(frame, LOST_AT_EDGE if edge else LOST_MID_FRAME, "info" if edge else "warn", obj,
                    f"mask vanished (last area {st['last_area']}px)" + (", mask was shrinking at the frame border -> possibly left the frame" if edge else ""),
                    action="flagged")
            st["lost_edge"] = bool(edge)
        edge = st.get("lost_edge", False)
        return Verdict("lost", uncertain=None if edge else LOST_MID_FRAME,
                       region=tuple(st["last_bbox"]) if st["last_bbox"] else None)

    def mark_abandoned(self, frame: int, obj: int, log: QCLog, reason: str):
        st = self.state.get(obj)
        if st and st["active"]:
            st["active"] = False
            log.add(frame, OBJECT_ABANDONED, "warn", obj, reason, action="object_no_longer_tracked")

    def frames_lost(self, frame: int, obj: int) -> int:
        st = self.state.get(obj)
        if not st or st["lost_since"] is None:
            return 0
        return frame - st["lost_since"] + 1


def detection_mask_mismatch(det_boxes: List[Tuple[Box, float, str]], masks: List[np.ndarray], frame: int,
                            log: QCLog, min_cover: float = 0.1) -> int:
    """Log detections whose box is (almost) not covered by any tracked mask. Not an error by itself:
    NudeNet and the user-selected object may legitimately differ. Both are mosaicked either way."""
    n = 0
    for (box, score, cls) in det_boxes:
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        covered = 0
        for m in masks:
            covered = max(covered, int(m[y1:y2, x1:x2].sum()))
        if covered / area < min_cover:
            log.add(frame, DET_NO_MASK, "info", None, f"{cls} score={score:.2f} box={box} not covered by any tracked mask",
                    action="mosaicked_from_detection_box", cls=cls, score=round(float(score), 3))
            n += 1
    return n
