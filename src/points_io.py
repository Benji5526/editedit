"""Selection file written by pick_points.py and read by censor_video.py.

Coordinates are in pixels of the DECODED (auto-rotated) full-resolution frame, i.e. the same frames
that FFmpeg delivers to the processing pipeline. Everything needed to detect a mismatch with the
video being processed (fingerprint, size, fps, frame count, rotation) is stored next to the points.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from .video_io import VideoInfo

VERSION = 1

# obj_id -> frame_idx -> (points[(x, y)], labels[int])
Prompts = Dict[int, Dict[int, Tuple[List[Tuple[float, float]], List[int]]]]


class PointsError(ValueError):
    pass


def video_block(info: VideoInfo) -> dict:
    return {"path": info.path, "fingerprint": info.fingerprint, "file_size": info.file_size,
            "width": info.width, "height": info.height, "fps": info.fps, "fps_rational": info.fps_rational,
            "nb_frames": info.nb_frames, "rotation": info.rotation, "is_vfr": info.is_vfr}


def save_points(path: str, info: VideoInfo, points: List[dict], names: Dict[int, str] = None):
    """points: dicts with obj_id, frame_idx, x, y, label (1 positive / 0 negative)."""
    rows = []
    for p in sorted(points, key=lambda p: (p["obj_id"], p["frame_idx"])):
        rows.append({"obj_id": int(p["obj_id"]), "frame_idx": int(p["frame_idx"]),
                     "timestamp": round(info.timestamp_of(int(p["frame_idx"])), 6),
                     "x": round(float(p["x"]), 2), "y": round(float(p["y"]), 2),
                     "x_norm": round(float(p["x"]) / info.width, 6), "y_norm": round(float(p["y"]) / info.height, 6),
                     "label": int(p["label"])})
    doc = {"version": VERSION, "video": video_block(info),
           "coordinate_space": "decoded full-resolution frame pixels (after rotation), origin top-left",
           "objects": {str(k): v for k, v in (names or {}).items()}, "points": rows}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_points(path: str, info: VideoInfo, allow_fingerprint_mismatch: bool = False) -> Prompts:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise PointsError(f"unsupported points file version {doc.get('version')}")
    v = doc["video"]
    if (v["width"], v["height"]) != (info.width, info.height):
        raise PointsError(f"points were picked on {v['width']}x{v['height']} frames but the video decodes to "
                          f"{info.width}x{info.height} (rotation {v.get('rotation')} vs {info.rotation}). Re-pick the points.")
    if v["fingerprint"] != info.fingerprint and not allow_fingerprint_mismatch:
        raise PointsError("points file was created for a different video file (fingerprint mismatch). "
                          "Re-pick, or pass --allow-points-mismatch if you know it is the same footage.")
    if v["nb_frames"] != info.nb_frames:
        raise PointsError(f"frame count differs: points {v['nb_frames']} vs video {info.nb_frames}")
    prompts: Prompts = {}
    for p in doc["points"]:
        oid, fr, lab = int(p["obj_id"]), int(p["frame_idx"]), int(p["label"])
        x, y = float(p["x"]), float(p["y"])
        if lab not in (0, 1):
            raise PointsError(f"invalid label {lab} (must be 0 or 1)")
        if not (0 <= fr < info.nb_frames):
            raise PointsError(f"frame_idx {fr} out of range 0..{info.nb_frames - 1}")
        if not (0 <= x < info.width and 0 <= y < info.height):
            raise PointsError(f"point ({x},{y}) outside frame {info.width}x{info.height}")
        pts, labs = prompts.setdefault(oid, {}).setdefault(fr, ([], []))
        pts.append((x, y))
        labs.append(lab)
    for oid, frames in prompts.items():
        for fr, (pts, labs) in frames.items():
            if 1 not in labs:
                raise PointsError(f"object {oid} frame {fr} has only negative points; add at least one positive point")
    return prompts


class PointStore:
    """All clicks, independent of any GUI (shared by pick_points.py and the desktop app)."""

    def __init__(self):
        self.points: List[dict] = []     # {obj_id, frame_idx, x, y, label}
        self.objects: List[int] = [1]
        self.current = 1

    def add(self, frame_idx: int, x: float, y: float, label: int):
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        self.points.append({"obj_id": self.current, "frame_idx": int(frame_idx), "x": float(x), "y": float(y), "label": int(label)})

    def new_object(self) -> int:
        self.current = max(self.objects + [p["obj_id"] for p in self.points]) + 1
        self.objects.append(self.current)
        return self.current

    def select(self, obj_id: int) -> bool:
        if obj_id in self.objects or any(p["obj_id"] == obj_id for p in self.points):
            self.current = obj_id
            return True
        return False

    def undo(self, frame_idx: int) -> bool:
        for i in range(len(self.points) - 1, -1, -1):
            p = self.points[i]
            if p["frame_idx"] == frame_idx and p["obj_id"] == self.current:
                del self.points[i]
                return True
        return False

    def clear_frame(self, frame_idx: int) -> int:
        n = len(self.points)
        self.points = [p for p in self.points if not (p["frame_idx"] == frame_idx and p["obj_id"] == self.current)]
        return n - len(self.points)

    def delete_object(self) -> int:
        n = len(self.points)
        self.points = [p for p in self.points if p["obj_id"] != self.current]
        if self.current in self.objects and len(self.objects) > 1:
            self.objects.remove(self.current)
        self.current = self.objects[0]
        return n - len(self.points)

    def valid(self) -> List[str]:
        """Problems that would make censor_video.py reject the file."""
        probs = []
        groups: Dict[tuple, List[int]] = {}
        for p in self.points:
            groups.setdefault((p["obj_id"], p["frame_idx"]), []).append(p["label"])
        for (o, f), labs in groups.items():
            if 1 not in labs:
                probs.append(f"object {o} frame {f}: only negative points")
        return probs
