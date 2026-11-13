"""NudeNet auxiliary detector (supported classes only) + temporal gap filling.

NudeNet 3.4.x facts this module is written against (verified against the installed package source):
  * bundled model: 320n.onnx (320x320 input). A 640 model can be passed with ``model_path`` +
    ``resolution=640`` but is NOT bundled.
  * ``NudeDetector.detect`` hard-codes a 0.2 score floor and 0.25 NMS score threshold, so a lower
    user threshold would have no effect through that API. This module therefore reuses the loaded
    ONNX session but performs its own post-processing with a configurable threshold.
  * the package depends on plain ``onnxruntime`` (CPU). This project installs ``onnxruntime-gpu`` instead (the build for
    the CUDA major version PyTorch ships, see requirements.txt) and chooses the device itself (``make_session``): "auto"
    uses the GPU when it works and says so when it does not, "cuda" fails instead of falling back.
Lowering the threshold increases recall AND false positives; it does not make misses impossible.
NudeNet cannot detect arbitrary objects - user-picked targets are SAM 2's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_CLASSES = ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]
# faces: NudeNet finds whole faces (a box). "below_eyes" / "above_eyes" cut that box at the eye line; the eye line comes from a
# face landmark model (YuNet, run by OpenCV) when its file is there, else from fixed proportions of the box (rough).
FACE_CLASSES = ["FACE_FEMALE", "FACE_MALE"]
FACE_MODES = ("off", "full", "below_eyes", "above_eyes")
FACE_BELOW, FACE_ABOVE = "FACE_BELOW_EYES", "FACE_ABOVE_EYES"
EXACT_CLASSES = {FACE_BELOW, FACE_ABOVE}         # boxes of these classes are final: the renderer must not grow them (eyes!)
YUNET_NAME = "face_detection_yunet_2023mar.onnx"


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
    poly: Optional[list] = None   # exact region as a polygon [[x, y], ...] (rotated face parts); the box is then its bounding box

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


DEVICES = ("auto", "cuda", "cpu")


def make_session(path: str, device: str = "auto", providers=None):
    """ONNX Runtime session for ``path`` on the requested device. Returns (session, provider actually used, note).

    device: "cpu"; "cuda" (fails when the GPU cannot be used - never a silent fallback); "auto" (GPU when it works, else CPU with
    a note saying why). The session is always created with an explicit provider list: with onnxruntime-gpu installed, the
    library's default would try TensorRT first.
    The CUDA build needs the CUDA 12 / cuDNN 9 libraries in the process; PyTorch ships them, and ``preload_dlls`` loads them from
    there when PyTorch has not been imported yet (a NudeNet-only job)."""
    import onnxruntime
    if device not in DEVICES:
        raise ValueError(f"unknown detector device {device!r}; choose one of {DEVICES}")
    if providers:                                              # explicit list (tests, special setups): used as given
        sess = onnxruntime.InferenceSession(path, providers=list(providers))
        return sess, sess.get_providers()[0], ""
    note = ""
    if device in ("auto", "cuda"):
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            note = "this onnxruntime build has no CUDA provider (install onnxruntime-gpu built for the CUDA version PyTorch uses)"
        else:
            have_torch = False
            try:
                import torch  # noqa: F401  (loads the CUDA 12 / cuDNN 9 libraries it ships into this process; also works in the packaged worker)
                have_torch = True
            except Exception:  # noqa: BLE001
                pass
            try:
                # with torch imported the libraries are already loaded, and preload_dlls would only print a notice on stdout
                # (which carries the progress protocol for the desktop app)
                if not have_torch and hasattr(onnxruntime, "preload_dlls"):
                    onnxruntime.preload_dlls()
            except Exception as e:  # noqa: BLE001
                note = f"could not preload the CUDA libraries ({type(e).__name__}: {e})"
            try:
                sess = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                if sess.get_providers()[0] == "CUDAExecutionProvider":
                    return sess, "CUDAExecutionProvider", ""
                note = note or "the CUDA provider could not be initialised (CUDA 12 / cuDNN 9 libraries not found?)"
            except Exception as e:  # noqa: BLE001
                note = f"the CUDA provider failed ({type(e).__name__}: {e})"
        if device == "cuda":
            raise RuntimeError("NudeNet was asked to run on the GPU but cannot: " + note + ". Use --nudenet-device cpu or auto.")
    sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, "CPUExecutionProvider", note


class NudenetDetector:
    def __init__(self, threshold: float = 0.25, nms_iou: float = 0.45, classes: Optional[Sequence[str]] = None,
                 model_path: Optional[str] = None, resolution: int = 320, tiles: int = 1, providers=None, device: str = "cpu",
                 face_mode: str = "off", face_model: Optional[str] = None):
        import os
        import nudenet

        labels = supported_labels()
        classes = list(classes) if classes else list(DEFAULT_CLASSES)
        if face_mode not in FACE_MODES:
            raise ValueError(f"unknown face mode {face_mode!r}; choose one of {FACE_MODES}")
        self.face_mode = face_mode
        self.landmarks = None
        if face_mode != "off":
            classes += [c for c in FACE_CLASSES if c not in classes]
            if face_mode != "full" and face_model:
                self.landmarks = FaceLandmarks(face_model)
        bad = [c for c in classes if c not in labels]
        if bad:
            raise ValueError(f"unsupported NudeNet classes {bad}; supported: {labels}")
        self.labels = labels
        self.class_ids = np.array([labels.index(c) for c in classes])
        self.classes = classes
        self.threshold, self.nms_iou, self.resolution, self.tiles = float(threshold), float(nms_iou), int(resolution), int(tiles)
        path = model_path or os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NudeNet model not found: {path}")
        # the library's own NudeDetector is not used: it creates its session with every available provider and ignores a choice
        self.session, self.provider, self.device_note = make_session(path, device, providers)
        self.input_name = self.session.get_inputs()[0].name
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
        dets = self._nms(dets)
        if self.face_mode in ("below_eyes", "above_eyes"):
            faces = [d for d in dets if d.cls in FACE_CLASSES]
            marks = self.landmarks.detect(bgr) if self.landmarks is not None else []
            dets = [d for d in dets if d.cls not in FACE_CLASSES] + face_parts(faces, marks, self.face_mode, bgr.shape[1], bgr.shape[0])
        return dets


def round_face_part(poly, cls: str, n: int = 24) -> list:
    """Half-ellipse version of a face-part rectangle [[x, y] x 4] (corner order as made by face_parts: for FACE_BELOW the first
    two corners are on the eye side, for FACE_ABOVE the last two). The edge next to the eyes stays straight - rounding it would
    either cover the eyes or uncover the nose - and the far side (chin / forehead) becomes a curve."""
    p = [np.array(q, float) for q in poly]
    if cls == FACE_ABOVE:
        e0, e1, far = p[3], p[2], p[0]
    else:
        e0, e1, far = p[0], p[1], p[3]
    c, a, b = (e0 + e1) / 2, (e1 - e0) / 2, far - e0
    return [[float(v[0]), float(v[1])] for v in (c + a * np.cos(t) + b * np.sin(t) for t in np.linspace(0.0, np.pi, n + 1))]


class FaceLandmarks:
    """Eye positions of the faces in a frame, from the YuNet face detector that OpenCV runs (cv2.FaceDetectorYN). Local file,
    no network. detect() returns [(x1, y1, x2, y2, (rx, ry), (lx, ly), score, (nose_x, nose_y))] in frame pixels."""

    def __init__(self, model_path: str, score: float = 0.6, max_side: int = 960):
        import os
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"face landmark model not found: {model_path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("this OpenCV build has no FaceDetectorYN")
        self.max_side = int(max_side)
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), float(score), 0.3, 200)

    def detect(self, bgr: np.ndarray) -> list:
        h, w = bgr.shape[:2]
        k = min(1.0, self.max_side / max(h, w))
        small = bgr if k == 1.0 else cv2.resize(bgr, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        out = []
        for f in (faces if faces is not None else []):
            f = [float(v) / k for v in f[:14]] + [float(f[14])]
            out.append((f[0], f[1], f[0] + f[2], f[1] + f[3], (f[4], f[5]), (f[6], f[7]), f[14], (f[8], f[9])))
        return out


def face_parts(faces: List[Det], marks: list, mode: str, width: int, height: int) -> List[Det]:
    """Turn whole-face boxes into the part below or above the eyes (class FACE_BELOW / FACE_ABOVE).

    faces: whole-face boxes from the detector. marks: FaceLandmarks.detect() results (may be empty).
    A face with landmarks gets a region that follows the head's rotation: a rectangle aligned with the line through the eyes
    (a polygon in Det.poly), starting 0.3 x the eye distance below (above) the eyes so the eyes themselves stay visible. Its size
    comes from face proportions (eye distance), not from a face outline - it is a rectangle, not the contour of the face.
    A face without landmarks (profile, small, landmark model missing) is cut horizontally at fixed proportions of its box -
    rough, and wrong for tilted heads. A landmark face that the detector did not report is used too."""
    out: List[Det] = []
    used = set()

    def from_marks(m, score, box):
        re, le = np.array(m[4], float), np.array(m[5], float)
        mid = (re + le) / 2
        iod = float(np.hypot(*(le - re)))
        bw, bh = box[2] - box[0], box[3] - box[1]
        size = max(iod, 0.45 * min(bw, bh), 1.0)                 # a turned head shortens the eye distance: do not shrink with it
        u = (le - re) / max(iod, 1e-6)                           # along the eyes
        v = np.array([-u[1], u[0]])                              # across: towards the chin ...
        nose = np.array(m[7], float) if len(m) > 7 else mid + np.array([0.0, 1.0])
        if float(np.dot(nose - mid, v)) < 0:                     # ... on the side where the nose is
            v = -v
        half = 1.25 * size
        t0, t1 = (0.3 * size, 2.2 * size) if mode == "below_eyes" else (-1.6 * size, -0.3 * size)
        poly = [mid - half * u + t0 * v, mid + half * u + t0 * v, mid + half * u + t1 * v, mid - half * u + t1 * v]
        poly = [[float(min(max(q[0], 0), width)), float(min(max(q[1], 0), height))] for q in poly]
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        if max(xs) - min(xs) >= 2 and max(ys) - min(ys) >= 2:
            out.append(Det(min(xs), min(ys), max(xs), max(ys), score, FACE_BELOW if mode == "below_eyes" else FACE_ABOVE, poly=poly))

    for d in faces:
        w, h = d.x2 - d.x1, d.y2 - d.y1
        hit = None
        for j, m in enumerate(marks):
            cx, cy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
            if j not in used and d.x1 <= cx <= d.x2 and d.y1 <= cy <= d.y2:
                hit = j
                break
        if hit is not None:
            used.add(hit)
            m = marks[hit]
            from_marks(m, d.score, (min(d.x1, m[0]), min(d.y1, m[1]), max(d.x2, m[2]), max(d.y2, m[3])))
            continue
        ey, iod = d.y1 + 0.40 * h, 0.42 * w
        xa, xb = max(0.0, d.x1 - 0.06 * w), min(float(width), d.x2 + 0.06 * w)
        if mode == "below_eyes":
            ya, yb, cls = ey + 0.3 * iod, min(float(height), d.y2 + 0.12 * h), FACE_BELOW
        else:
            ya, yb, cls = max(0.0, d.y1 - 0.12 * h), ey - 0.3 * iod, FACE_ABOVE
        if yb - ya >= 2 and xb - xa >= 2:
            out.append(Det(xa, ya, xb, yb, d.score, cls))
    for j, m in enumerate(marks):
        if j not in used:
            from_marks(m, float(m[6]), m[:4])
    return out


class _Track:
    __slots__ = ("cls", "obs", "seen")

    def __init__(self, cls: str):
        self.cls = cls
        self.obs: Dict[int, Det] = {}
        self.seen = 0            # real detections of this track so far, including earlier chunks

    @property
    def last_frame(self) -> int:
        return max(self.obs)


def fill_detections(per_frame: List[Optional[List[Det]]], start_frame: int, carry: List[dict],
                    max_gap: int, hold: int, match_iou: float = 0.15, lost_hold: int = 0, lost: Optional[list] = None,
                    min_obs: int = 3) -> Tuple[List[List[Det]], List[dict]]:
    """Temporal gap filling inside one chunk.

    per_frame[i] are detections for global frame start_frame+i (None = detector not run on that frame).
    carry: tracks alive at the end of the previous chunk, [{cls, frame, box, score}].
    Gaps of <= max_gap frames between two observations of the same (IoU-matched) track are filled by
    linear box interpolation; after the last observation a box is held for ``hold`` frames.
    Returns (frames with detections incl. filled ones, carry for next chunk).
    This only bridges short misses; it cannot recover an object the detector never sees.

    lost_hold > 0 with a ``lost`` list: a track that was really detected at least ``min_obs`` times and then stops being
    detected is reported for up to ``lost_hold`` frames after its last detection (after the ``hold`` frames), or until a
    detection of the same class shows up near its last box. Each entry appended to ``lost`` is
    {frame, box, cls, last_seen, first}; ``first`` marks the first frame of the episode, and gets ``recovered_after`` (gap
    length in frames) when the region is detected again inside the same chunk. The caller decides what to do
    with it (review flag, conservative mosaic). It works forward in time only, so it also covers gaps that are longer
    than max_gap and gaps that cross a chunk boundary, which interpolation cannot.
    """
    n = len(per_frame)
    tracks: List[_Track] = []
    for c in carry:
        t = _Track(c["cls"])
        t.obs[c["frame"]] = Det(*c["box"], c["score"], c["cls"])
        t.seen = int(c.get("n", 1))
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
                if lost_hold > 0:                 # the same region found again after a longer gap keeps its history, so a
                    for t in tracks:              # region that flickers is still "known" when it drops out the next time
                        if t.cls == d.cls and 0 < f - t.last_frame <= lost_hold and _seen_again([d], t.obs[t.last_frame]):
                            best.seen = max(best.seen, t.seen)
                tracks.append(best)
            taken.add(id(best))
            best.obs[f] = d
            best.seen += 1
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
                        out[g - start_frame].append(Det(*box, min(da.score, db.score), t.cls, filled=True,
                                                        poly=_mid_poly(da, db, u)))
        last = fr[-1]
        last_det = t.obs[last]
        for g in range(last + 1, last + 1 + hold):
            if start_frame <= g < end and not _has_cover(out[g - start_frame], last_det):
                out[g - start_frame].append(Det(*last_det.box, last_det.score, t.cls, filled=True, poly=last_det.poly))
        watching = False                          # still inside the "lost" window when this chunk ends
        if lost_hold > 0 and lost is not None and t.seen >= min_obs:
            watching = True
            head = None
            for g in range(max(last + 1, start_frame), last + 1 + lost_hold):
                if g >= end:
                    break
                if _seen_again(per_frame[g - start_frame], last_det):
                    watching = False
                    if head is not None:
                        head["recovered_after"] = g - last - 1
                    break
                if g > last + hold:
                    lost.append({"frame": g, "box": list(last_det.box), "cls": t.cls, "last_seen": last,
                                 "first": g == last + hold + 1, "poly": last_det.poly})
                    if lost[-1]["first"]:
                        head = lost[-1]
            else:
                watching = False                  # the whole window has passed
        if (last + hold >= end - 1 and last >= start_frame - 1 - max_gap) or watching:
            new_carry.append({"cls": t.cls, "frame": last, "box": list(last_det.box), "score": last_det.score, "n": t.seen})
    return out, new_carry


def _mid_poly(da: Det, db: Det, u: float) -> Optional[list]:
    """Polygon for a gap-filled frame between two polygon detections (same number of corners: interpolated; else the nearer one)."""
    if da.poly is None or db.poly is None:
        return da.poly if u < 0.5 else db.poly
    if len(da.poly) != len(db.poly):
        return da.poly if u < 0.5 else db.poly
    return [[a[0] * (1 - u) + b[0] * u, a[1] * (1 - u) + b[1] * u] for a, b in zip(da.poly, db.poly)]


def _seen_again(dets: Optional[List[Det]], ref: Det) -> bool:
    """A real detection of the same class near the place where ``ref`` was last seen (overlapping it, or with its centre
    within one box size around it)."""
    if not dets:
        return False
    w, h = ref.x2 - ref.x1, ref.y2 - ref.y1
    for d in dets:
        if d.cls != ref.cls or d.filled:
            continue
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        if iou(d.box, ref.box) > 0.05 or (ref.x1 - w <= cx <= ref.x2 + w and ref.y1 - h <= cy <= ref.y2 + h):
            return True
    return False


def _has_cover(dets: List[Det], ref: Det) -> bool:
    return any(d.cls == ref.cls and iou(d.box, ref.box) > 0.3 for d in dets)
