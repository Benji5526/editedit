"""NudeNet auxiliary detector (supported classes only) + temporal gap filling.

NudeNet 3.4.x facts this module is written against (verified against the installed package source):
  * bundled model: 320n.onnx (320x320 input). A 640 model can be passed with ``model_path`` +
    ``resolution=640`` but is NOT bundled.
  * ``NudeDetector.detect`` hard-codes a 0.2 score floor and 0.25 NMS score threshold, so a lower
    user threshold would have no effect through that API. This module therefore reuses the loaded
    ONNX session but performs its own post-processing with a configurable threshold.
  * the package installs plain ``onnxruntime`` (CPU). Detection runs on CPU unless the user installs
    ``onnxruntime-gpu`` and passes providers.
Lowering the threshold increases recall AND false positives; it does not make misses impossible.
NudeNet cannot detect arbitrary objects - user-picked targets are SAM 2's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_CLASSES = [
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
]


def supported_labels() -> List[str]:
    import nudenet.nudenet as nn
    return list(getattr(nn, "__labels"))


def nudenet_version() -> str:
    try:
        from importlib.metadata import version
        return version("nudenet")
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: str
    filled: bool = False   # True when synthesised by temporal gap filling (not seen by the model)

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class NudenetDetector:
    def __init__(self, threshold: float = 0.25, nms_iou: float = 0.45, classes: Optional[Sequence[str]] = None,
                 model_path: Optional[str] = None, resolution: int = 320, tiles: int = 1, providers=None):
        import onnxruntime
        from nudenet import NudeDetector

        labels = supported_labels()
        classes = list(classes) if classes else list(DEFAULT_CLASSES)
        bad = [c for c in classes if c not in labels]
        if bad:
            raise ValueError(f"unsupported NudeNet classes {bad}; supported: {labels}")
        self.labels = labels
        self.class_ids = np.array([labels.index(c) for c in classes])
        self.classes = classes
        self.threshold, self.nms_iou, self.resolution, self.tiles = float(threshold), float(nms_iou), int(resolution), int(tiles)
        det = NudeDetector(model_path=model_path, inference_resolution=resolution)
        if providers:  # NudeDetector ignores providers; rebuild session when asked
            import os
            import nudenet
            path = model_path or os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")
            det.onnx_session = onnxruntime.InferenceSession(path, providers=providers)
        self.session = det.onnx_session
        self.input_name = det.input_name
        out_dim = self.session.get_outputs()[0].shape
        if len(out_dim) != 3 or out_dim[1] != 4 + len(labels):
            raise RuntimeError(f"unexpected NudeNet output shape {out_dim} for {len(labels)} labels")

    def _infer(self, bgr: np.ndarray) -> List[Det]:
        h, w = bgr.shape[:2]
        side = max(h, w)
        padded = cv2.copyMakeBorder(bgr, 0, side - h, 0, side - w, cv2.BORDER_CONSTANT)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (self.resolution, self.resolution), (0, 0, 0), swapRB=True, crop=False)
        out = self.session.run(None, {self.input_name: blob})[0]
        rows = np.transpose(np.squeeze(out, 0))               # (N, 4+C)
        scores = rows[:, 4:][:, self.class_ids]               # only the classes we care about
        best = scores.argmax(1)
        conf = scores[np.arange(len(rows)), best]
        keep = conf >= self.threshold
        if not keep.any():
            return []
        rows, best, conf = rows[keep], best[keep], conf[keep]
        k = side / self.resolution
        cx, cy, bw, bh = rows[:, 0] * k, rows[:, 1] * k, rows[:, 2] * k, rows[:, 3] * k
        x1, y1 = np.clip(cx - bw / 2, 0, w), np.clip(cy - bh / 2, 0, h)
        x2, y2 = np.clip(cx + bw / 2, 0, w), np.clip(cy + bh / 2, 0, h)
        dets = [Det(float(a), float(b), float(c), float(d), float(s), self.classes[int(i)])
                for a, b, c, d, s, i in zip(x1, y1, x2, y2, conf, best) if c > a and d > b]
        return dets

    def _nms(self, dets: List[Det]) -> List[Det]:
        if not dets:
            return dets
        boxes = [[d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1] for d in dets]
        idx = cv2.dnn.NMSBoxes(boxes, [d.score for d in dets], self.threshold, self.nms_iou)
        return [dets[int(i)] for i in np.array(idx).reshape(-1)]

    def detect(self, bgr: np.ndarray) -> List[Det]:
        """Detect on a BGR uint8 frame. tiles>1 additionally scans a tiles x tiles grid of overlapping
        crops so small regions get more input resolution (slower)."""
        dets = self._infer(bgr)
        if self.tiles > 1:
            h, w = bgr.shape[:2]
            tw, th = int(w / self.tiles * 1.3), int(h / self.tiles * 1.3)
            xs = np.linspace(0, w - tw, self.tiles).astype(int)
            ys = np.linspace(0, h - th, self.tiles).astype(int)
            for y0 in ys:
                for x0 in xs:
                    for d in self._infer(np.ascontiguousarray(bgr[y0:y0 + th, x0:x0 + tw])):
                        dets.append(Det(d.x1 + x0, d.y1 + y0, d.x2 + x0, d.y2 + y0, d.score, d.cls))
        return self._nms(dets)


class _Track:
    __slots__ = ("cls", "obs")

    def __init__(self, cls: str):
        self.cls = cls
        self.obs: Dict[int, Det] = {}

    @property
    def last_frame(self) -> int:
        return max(self.obs)


def fill_detections(per_frame: List[Optional[List[Det]]], start_frame: int, carry: List[dict],
                    max_gap: int, hold: int, match_iou: float = 0.15) -> Tuple[List[List[Det]], List[dict]]:
    """Temporal gap filling inside one chunk.

    per_frame[i] are detections for global frame start_frame+i (None = detector not run on that frame).
    carry: tracks alive at the end of the previous chunk, [{cls, frame, box, score}].
    Gaps of <= max_gap frames between two observations of the same (IoU-matched) track are filled by
    linear box interpolation; after the last observation a box is held for ``hold`` frames.
    Returns (frames with detections incl. filled ones, carry for next chunk).
    This only bridges short misses; it cannot recover an object the detector never sees.
    """
    n = len(per_frame)
    tracks: List[_Track] = []
    for c in carry:
        t = _Track(c["cls"])
        t.obs[c["frame"]] = Det(*c["box"], c["score"], c["cls"])
        tracks.append(t)
    for i, dets in enumerate(per_frame):
        if not dets:
            continue
        f = start_frame + i
        taken = set()
        for d in sorted(dets, key=lambda d: -d.score):
            best, best_iou = None, match_iou
            for t in tracks:
                if t.cls != d.cls or id(t) in taken:
                    continue
                lf = t.last_frame
                if f - lf > max_gap + 1:
                    continue
                v = iou(t.obs[lf].box, d.box)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is None:
                best = _Track(d.cls)
                tracks.append(best)
            taken.add(id(best))
            best.obs[f] = d
    out: List[List[Det]] = [list(d) if d else [] for d in per_frame]
    new_carry = []
    end = start_frame + n
    for t in tracks:
        fr = sorted(t.obs)
        for a, b in zip(fr, fr[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                da, db = t.obs[a], t.obs[b]
                for g in range(a + 1, b):
                    if start_frame <= g < end:
                        u = (g - a) / (b - a)
                        box = [da.box[j] * (1 - u) + db.box[j] * u for j in range(4)]
                        # take the larger of the two boxes' extents so interpolation never shrinks coverage
                        box = [min(box[0], da.x1, db.x1), min(box[1], da.y1, db.y1),
                               max(box[2], da.x2, db.x2), max(box[3], da.y2, db.y2)] if gap <= 2 else box
                        out[g - start_frame].append(Det(*box, min(da.score, db.score), t.cls, filled=True))
        last = fr[-1]
        last_det = t.obs[last]
        for g in range(last + 1, last + 1 + hold):
            if start_frame <= g < end and not _has_cover(out[g - start_frame], last_det):
                out[g - start_frame].append(Det(*last_det.box, last_det.score, t.cls, filled=True))
        if last + hold >= end - 1 and last >= start_frame - 1 - max_gap:
            new_carry.append({"cls": t.cls, "frame": last, "box": list(last_det.box), "score": last_det.score})
    return out, new_carry


def _has_cover(dets: List[Det], ref: Det) -> bool:
    return any(d.cls == ref.cls and iou(d.box, ref.box) > 0.3 for d in dets)
