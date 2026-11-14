"""Labels from a pair of videos: the original and a version of it that somebody already mosaicked.

Where the two differ is where a person decided to cover something. Comparing them frame by frame gives regions (boxes in the
original's full-resolution pixels) without anybody clicking - material for training a detector, including for things no
model here detects (bodily fluids). Nothing leaves the machine; the result holds no file names or paths, only the original's
fingerprint.

What this can and cannot tell:
  * a region says "this was covered", not what it was (face, genitalia, fluid ...): one class, "censor"
  * everything else that differs is found too: a logo, subtitles, a colour grade, a crop. A change in the same place on
    almost every frame is reported as a "static place" (it may be an overlay, or a subject that does not move) and is only
    left out on request; a frame that differs almost everywhere is skipped
  * both videos are lossy: small differences are compression noise, so only clearly changed, reasonably large areas count
  * the two must show the same frames. A constant offset (a few frames cut at the start) is searched for; cuts in the middle,
    speed changes or a different frame rate are not handled - the report says how well the frames match
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .video_io import FrameReader, VideoInfo, probe_video, read_frame_at

VERSION = 1
Box = Tuple[int, int, int, int]


def analysis_size(info: VideoInfo, width: int = 960) -> Tuple[int, int]:
    w = min(width, info.width)
    h = int(round(info.height * w / info.width / 2)) * 2
    return w, h


def diff_regions(a: np.ndarray, b: np.ndarray, threshold: int = 22, min_area_frac: float = 0.0004) -> Tuple[List[Box], float]:
    """Changed regions between two BGR frames of the same size. Returns (boxes, fraction of the frame that changed)."""
    d = cv2.absdiff(a, b).max(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), 2.0)                        # compression noise is fine-grained; a mosaic changes whole blocks
    m = (d > threshold).astype(np.uint8)
    k = max(3, int(round(a.shape[1] / 160)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k)))
    changed = float(m.mean())
    boxes: List[Box] = []
    min_area = min_area_frac * m.size
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes, changed


def find_offset(orig: VideoInfo, cens: VideoInfo, size: Tuple[int, int], max_offset: int = 90, samples: int = 5) -> Tuple[int, float]:
    """Constant frame offset k so that censored frame i + k shows original frame i. Returns (k, mean difference at that offset).
    Compared on small grey frames at a few places; the median over the frame ignores the mosaicked parts."""
    small = (size[0] // 4, size[1] // 4)

    def grey(info, idx):
        return cv2.cvtColor(read_frame_at(info, idx, scale_to=small), cv2.COLOR_BGR2GRAY).astype(np.int16)

    n = min(orig.nb_frames, cens.nb_frames)
    picks = [int(n * (i + 1) / (samples + 1)) for i in range(samples)]
    refs = {p: grey(orig, p) for p in picks}
    zero = float(np.mean([np.median(np.abs(refs[p] - grey(cens, p))) for p in picks if p < cens.nb_frames] or [1e9]))
    if zero <= 2.0:                                              # the frames already line up: no need to try 180 offsets
        return 0, zero
    best, best_cost = 0, zero
    for k in sorted(range(-max_offset, max_offset + 1), key=abs):
        if k == 0:
            continue
        cost, used = 0.0, 0
        for p in picks:
            if 0 <= p + k < cens.nb_frames:
                cost += float(np.median(np.abs(refs[p] - grey(cens, p + k))))
                used += 1
        if used and cost / used < best_cost:
            best, best_cost = k, cost / used
        if best_cost == 0:
            break
    return best, best_cost


def compare(original: str, censored: str, width: int = 960, max_offset: int = 90, threshold: int = 22,
            max_frames: Optional[int] = None, progress: Optional[Callable[[int, int], None]] = None,
            offset: Optional[int] = None, drop_static: bool = False) -> dict:
    """Compare the two videos and return the label document (see module text). Frames are read once, in order."""
    o, c = probe_video(original), probe_video(censored)
    notes = []
    if abs(o.fps - c.fps) > 0.01:
        raise ValueError(f"different frame rates ({o.fps:.3f} and {c.fps:.3f}): the frames cannot be matched one to one")
    if abs(o.width / o.height - c.width / c.height) > 0.01:
        raise ValueError(f"different picture shapes ({o.width}x{o.height} and {c.width}x{c.height}): one of them was cropped")
    if (o.width, o.height) != (c.width, c.height):
        notes.append(f"different sizes ({o.width}x{o.height} and {c.width}x{c.height}): compared at a common small size")
    size = analysis_size(o, width)
    if offset is None:
        offset, match_cost = find_offset(o, c, size, max_offset) if max_offset > 0 else (0, 0.0)
    else:
        match_cost = -1.0
    start_o, start_c = max(0, -offset), max(0, offset)
    n = min(o.nb_frames - start_o, c.nb_frames - start_c)
    if max_frames:
        n = min(n, max_frames)
    kx, ky = o.width / size[0], o.height / size[1]
    frames: Dict[int, List[Box]] = {}
    changed_frac, skipped_global, noise = [], [], []
    heat = np.zeros((size[1] // 8, size[0] // 8), np.int32)
    with FrameReader(o, start_frame=start_o, max_frames=n, scale_to=size) as ro, \
            FrameReader(c, start_frame=start_c, max_frames=n, scale_to=size) as rc:
        for i in range(n):
            a, b = ro.read(), rc.read()
            if a is None or b is None:
                notes.append(f"stopped at frame {i}: one of the videos ended early")
                n = i
                break
            boxes, changed = diff_regions(a, b, threshold)
            changed_frac.append(changed)
            if i % 30 == 0:
                noise.append(float(np.median(cv2.absdiff(a, b))))
            if changed > 0.45:                                   # nearly everything differs: a grade, a fade, a wrong match
                skipped_global.append(start_o + i)
            elif boxes:
                frames[start_o + i] = boxes
                for x1, y1, x2, y2 in boxes:
                    heat[y1 // 8:max(y1 // 8 + 1, y2 // 8), x1 // 8:max(x1 // 8 + 1, x2 // 8)] += 1
            if progress and i % 60 == 0:
                progress(i, n)
    # A place that is changed on almost every frame may be an overlay (logo, subtitle band) - or a subject that does not move
    # in front of a fixed camera. The comparison cannot tell, so such places are only reported; they are removed from the
    # labels when the caller says so (drop_static) after looking at the report.
    static = (heat >= 0.9 * max(1, n)).astype(np.uint8)
    overlays: List[Box] = []
    if static.any():
        contours, _ = cv2.findContours(static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlays = [(x * 8, y * 8, (x + w) * 8, (y + h) * 8) for x, y, w, h in (cv2.boundingRect(k) for k in contours)]

        def inside(bx):
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            return any(v[0] <= cx <= v[2] and v[1] <= cy <= v[3] and (bx[2] - bx[0]) * (bx[3] - bx[1]) <= 1.5 * (v[2] - v[0]) * (v[3] - v[1])
                       for v in overlays)
        if drop_static:
            frames = {f: [bx for bx in bs if not inside(bx)] for f, bs in frames.items()}
            frames = {f: bs for f, bs in frames.items() if bs}
    full = lambda bx: [round(bx[0] * kx), round(bx[1] * ky), round(bx[2] * kx), round(bx[3] * ky)]   # noqa: E731
    return {
        "version": VERSION, "fingerprint": o.fingerprint, "size": [o.width, o.height], "fps": o.fps_rational, "class": "censor",
        "offset": offset, "frames_compared": n, "first_frame": start_o,
        "frames": {str(f): [full(bx) for bx in bs] for f, bs in sorted(frames.items())},
        "stats": {"frames_with_regions": len(frames), "regions": sum(len(v) for v in frames.values()),
                  "frames_skipped_global_change": len(skipped_global),
                  "static_places": [full(v) for v in overlays], "static_places_dropped": bool(drop_static and overlays),
                  "median_changed_fraction": float(np.median(changed_frac)) if changed_frac else 0.0,
                  "background_difference": float(np.median(noise)) if noise else 0.0,      # 0-255; high = poor match or a grade
                  "offset_match_cost": match_cost},
        "notes": notes,
    }


def save_labels(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def load_labels(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise ValueError(f"unsupported label file version {doc.get('version')!r}")
    return doc
