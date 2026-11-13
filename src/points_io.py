"""Selection file written by pick_points.py and read by censor_video.py.

Coordinates are in pixels of the DECODED (auto-rotated) full-resolution frame, i.e. the same frames
that FFmpeg delivers to the processing pipeline. Everything needed to detect a mismatch with the
video being processed (fingerprint, size, fps, frame count, rotation) is stored next to the points.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from .video_io import VideoInfo

VERSION = 1            # clicks only
VERSION_STYLES = 2     # clicks + per-object mosaic settings ("object_styles"); written only when an object has its own settings
VERSION_MASKS = 3      # ... + hand-drawn masks ("mask_prompts"); written only when there is at least one
SUPPORTED_VERSIONS = (VERSION, VERSION_STYLES, VERSION_MASKS)
OBJECT_STYLES = ("pixelate", "blur", "solid")          # same names as src.mosaic.STYLES (kept here so this module stays light)

# obj_id -> {"style": str, "block": int, "pad_px": float}; every key is optional, a missing key means "use the job's setting"
ObjectStyles = Dict[int, dict]

# obj_id -> frame_idx -> (points[(x, y)], labels[int])
Prompts = Dict[int, Dict[int, Tuple[List[Tuple[float, float]], List[int]]]]


# obj_id -> frame_idx -> bool mask (any resolution with the video's aspect ratio; the pipeline resizes it to its analysis size).
# A drawn mask is a prompt like a click: on that frame the tracker starts from exactly this mask instead of from the clicks.
MaskPrompts = Dict[int, Dict[int, np.ndarray]]


class PointsError(ValueError):
    pass


def encode_mask(mask: np.ndarray) -> dict:
    m = np.ascontiguousarray(mask, dtype=bool)
    return {"shape": [int(m.shape[0]), int(m.shape[1])],
            "data": base64.b64encode(zlib.compress(np.packbits(m.reshape(-1)).tobytes(), 6)).decode("ascii")}


def decode_mask(d: dict) -> np.ndarray:
    h, w = int(d["shape"][0]), int(d["shape"][1])
    if h <= 0 or w <= 0 or h * w > 64_000_000:
        raise PointsError(f"invalid mask shape {d.get('shape')}")
    try:
        packed = np.frombuffer(zlib.decompress(base64.b64decode(d["data"])), dtype=np.uint8)
        return np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)
    except Exception as e:  # noqa: BLE001
        raise PointsError(f"unreadable mask data ({type(e).__name__}: {e})") from e


def mask_digest(mask: np.ndarray) -> str:
    """Short content hash of a mask (shape + pixels); part of the per-chunk prompt signature."""
    m = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(repr(m.shape).encode() + np.packbits(m.reshape(-1)).tobytes()).hexdigest()[:16]


def load_mask_prompts(path: str, info: Optional[VideoInfo] = None) -> MaskPrompts:
    """Hand-drawn masks of a points file ({} when it has none). With ``info`` the frame numbers are checked against the video."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    out: MaskPrompts = {}
    for d in doc.get("mask_prompts") or []:
        oid, fr = int(d["obj_id"]), int(d["frame_idx"])
        if info is not None and not (0 <= fr < info.nb_frames):
            raise PointsError(f"mask of object {oid}: frame_idx {fr} out of range 0..{info.nb_frames - 1}")
        m = decode_mask(d)
        if not m.any():
            raise PointsError(f"mask of object {oid} on frame {fr} is empty")
        out.setdefault(oid, {})[fr] = m
    return out


def video_block(info: VideoInfo) -> dict:
    return {"path": info.path, "fingerprint": info.fingerprint, "file_size": info.file_size,
            "width": info.width, "height": info.height, "fps": info.fps, "fps_rational": info.fps_rational,
            "nb_frames": info.nb_frames, "rotation": info.rotation, "is_vfr": info.is_vfr}


def normalize_styles(styles) -> ObjectStyles:
    """Validate per-object mosaic settings and drop everything that is unset. Raises PointsError on a bad value."""
    out: ObjectStyles = {}
    for oid, st in (styles or {}).items():
        if not isinstance(st, dict):
            raise PointsError(f"object {oid}: mosaic settings must be an object, not {type(st).__name__}")
        clean = {}
        unknown = set(st) - {"style", "block", "pad_px"}
        if unknown:
            raise PointsError(f"object {oid}: unknown mosaic setting(s) {sorted(unknown)}")
        if st.get("style") is not None:
            if st["style"] not in OBJECT_STYLES:
                raise PointsError(f"object {oid}: unknown mosaic style {st['style']!r}; choose one of {OBJECT_STYLES}")
            clean["style"] = st["style"]
        if st.get("block") is not None:
            b = int(st["block"])
            if not (2 <= b <= 256):
                raise PointsError(f"object {oid}: mosaic strength {b} is outside 2-256")
            clean["block"] = b
        if st.get("pad_px") is not None:
            pad = float(st["pad_px"])
            if not (0 <= pad <= 500):
                raise PointsError(f"object {oid}: mask padding {pad} is outside 0-500")
            clean["pad_px"] = pad
        if clean:
            out[int(oid)] = clean
    return out


def load_object_styles(path: str) -> ObjectStyles:
    """Per-object mosaic settings of a points file ({} for a version 1 file)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return normalize_styles(doc.get("object_styles"))


def save_points(path: str, info: VideoInfo, points: List[dict], names: Dict[int, str] = None, styles: ObjectStyles = None,
                masks: MaskPrompts = None):
    """points: dicts with obj_id, frame_idx, x, y, label (1 positive / 0 negative).
    styles: per-object mosaic settings; only those of objects that have points or masks are written.
    masks: hand-drawn masks {obj_id: {frame_idx: bool array}}; empty ones are dropped.
    The file version is the lowest that can hold the content: 1 (clicks only), 2 (+ settings), 3 (+ masks)."""
    rows = []
    for p in sorted(points, key=lambda p: (p["obj_id"], p["frame_idx"])):
        rows.append({"obj_id": int(p["obj_id"]), "frame_idx": int(p["frame_idx"]),
                     "timestamp": round(info.timestamp_of(int(p["frame_idx"])), 6),
                     "x": round(float(p["x"]), 2), "y": round(float(p["y"]), 2),
                     "x_norm": round(float(p["x"]) / info.width, 6), "y_norm": round(float(p["y"]) / info.height, 6),
                     "label": int(p["label"])})
    mask_rows = []
    for oid in sorted(masks or {}):
        for fr in sorted(masks[oid]):
            m = np.asarray(masks[oid][fr], dtype=bool)
            if not (0 <= int(fr) < info.nb_frames):
                raise PointsError(f"mask of object {oid}: frame_idx {fr} out of range 0..{info.nb_frames - 1}")
            if m.ndim == 2 and m.any():
                mask_rows.append({"obj_id": int(oid), "frame_idx": int(fr), "timestamp": round(info.timestamp_of(int(fr)), 6), **encode_mask(m)})
    have = {r["obj_id"] for r in rows} | {r["obj_id"] for r in mask_rows}
    styles = {k: v for k, v in normalize_styles(styles).items() if k in have}
    doc = {"version": VERSION_MASKS if mask_rows else VERSION_STYLES if styles else VERSION, "video": video_block(info),
           "coordinate_space": "decoded full-resolution frame pixels (after rotation), origin top-left",
           "objects": {str(k): v for k, v in (names or {}).items()}, "points": rows}
    if styles:
        doc["object_styles"] = {str(k): styles[k] for k in sorted(styles)}
    if mask_rows:
        doc["mask_prompts"] = mask_rows
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_points(path: str, info: VideoInfo, allow_fingerprint_mismatch: bool = False) -> Prompts:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") not in SUPPORTED_VERSIONS:
        raise PointsError(f"unsupported points file version {doc.get('version')}")
    normalize_styles(doc.get("object_styles"))          # a bad per-object setting is rejected here, before any processing
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
    for d in doc.get("mask_prompts") or []:                 # an object may consist of drawn masks only: it still has to be tracked
        oid, fr = int(d["obj_id"]), int(d["frame_idx"])
        if not (0 <= fr < info.nb_frames):
            raise PointsError(f"mask of object {oid}: frame_idx {fr} out of range 0..{info.nb_frames - 1}")
        prompts.setdefault(oid, {})
    return prompts


class PointStore:
    """All clicks, independent of any GUI (shared by pick_points.py and the desktop app)."""

    def __init__(self):
        self.points: List[dict] = []     # {obj_id, frame_idx, x, y, label}
        self.objects: List[int] = [1]
        self.current = 1
        self.styles: ObjectStyles = {}   # per-object mosaic settings (see normalize_styles)
        self.masks: MaskPrompts = {}     # hand-drawn masks: obj_id -> frame_idx -> bool array

    def set_mask(self, obj_id: int, frame_idx: int, mask: Optional[np.ndarray]):
        """Store (or with None / an empty mask: remove) the drawn mask of an object on a frame."""
        if mask is None or not np.asarray(mask).any():
            frames = self.masks.get(obj_id, {})
            frames.pop(int(frame_idx), None)
            if not frames:
                self.masks.pop(obj_id, None)
        else:
            self.masks.setdefault(obj_id, {})[int(frame_idx)] = np.asarray(mask, dtype=bool)

    def get_mask(self, obj_id: int, frame_idx: int) -> Optional[np.ndarray]:
        return self.masks.get(obj_id, {}).get(int(frame_idx))

    def object_ids(self) -> List[int]:
        """Every object that exists: listed, clicked or drawn."""
        return sorted(set(self.objects) | {p["obj_id"] for p in self.points} | set(self.masks))

    def set_style(self, obj_id: int, key: str, value) -> bool:
        """Set (or with None: clear) one mosaic setting of an object. Returns True when something changed."""
        st = dict(self.styles.get(obj_id, {}))
        if value is None:
            st.pop(key, None)
        else:
            st[key] = value
        new = normalize_styles({obj_id: st}).get(obj_id, {})
        if new == self.styles.get(obj_id, {}):
            return False
        if new:
            self.styles[obj_id] = new
        else:
            self.styles.pop(obj_id, None)
        return True

    def add(self, frame_idx: int, x: float, y: float, label: int):
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        self.points.append({"obj_id": self.current, "frame_idx": int(frame_idx), "x": float(x), "y": float(y), "label": int(label)})

    def new_object(self) -> int:
        self.current = max(self.objects + [p["obj_id"] for p in self.points] + list(self.masks)) + 1
        self.objects.append(self.current)
        return self.current

    def select(self, obj_id: int) -> bool:
        if obj_id in self.objects or obj_id in self.masks or any(p["obj_id"] == obj_id for p in self.points):
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
        self.styles.pop(self.current, None)
        n += len(self.masks.pop(self.current, {}))
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
            if 1 not in labs and self.get_mask(o, f) is None:     # a drawn mask on that frame is a prompt of its own
                probs.append(f"object {o} frame {f}: only negative points")
        return probs
