"""Synthetic test footage + a colour-threshold stand-in for SAM 2.

The stand-in (ColorTracker) is a TEST DOUBLE: it finds a magenta square by colour. It exists only to
exercise the pipeline's chunking / QC / fail-closed / resume logic without a GPU or checkpoint. It says
nothing about how well real SAM 2 tracks.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List

import cv2
import numpy as np

from src.tracking import ChunkMasks, Init, Tracker, TrackerError, pack_mask

W, H, FPS = 640, 360, 30
SQ = 60
HIDDEN = range(70, 86)       # object not drawn: occlusion that straddles the chunk boundary at 80 (chunk size 40)
ENTER = 10
SPEED = 5


def square_x(t: int) -> int:
    return -30 + SPEED * (t - ENTER)


_NOISE = np.random.default_rng(0).integers(-25, 25, (H, W, 3))


def draw_frame(t: int, with_square: bool = True) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([(xx * 255 // W), (yy * 255 // H), ((xx + yy) * 255 // (W + H))], -1)
    img = np.clip(img * 0.6 + 60 + _NOISE, 0, 255).astype(np.uint8)   # static texture: visible blockiness when mosaicked
    # frame-number code: 12 bits as 10px cells, top-left, away from the square's path (y 150..210)
    for b in range(12):
        v = 255 if (t >> b) & 1 else 0
        img[4:14, 4 + b * 10: 14 + b * 10] = v
    if with_square and ENTER <= t and t not in HIDDEN:
        x0 = square_x(t)
        y0 = 150
        xa, xb = max(0, x0), min(W, x0 + SQ)
        if xb > xa:
            img[y0:y0 + SQ, xa:xb] = (255, 0, 255)
            # texture inside so a mosaic visibly changes the pixels
            img[y0 + 10:y0 + 50:6, xa:xb] = (0, 0, 0)
    return img


def read_frame_code(img: np.ndarray) -> int:
    v = 0
    for b in range(12):
        cell = img[4:14, 4 + b * 10: 14 + b * 10].mean()
        if cell > 127:
            v |= 1 << b
    return v


def make_video(path: str, n_frames: int = 160, audio: bool = True, vfr: bool = False, rotate: int = 0, fps="30"):
    """fps may be a fraction such as "60000/1001"."""
    from fractions import Fraction
    dur = n_frames / float(Fraction(str(fps)))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur}"]
    vf = r"select=mod(n\,5)" if vfr else None   # drop every 5th frame => irregular intervals
    cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-g", "30",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if vfr:
        cmd += ["-vf", vf, "-fps_mode", "vfr"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for t in range(n_frames):
        p.stdin.write(draw_frame(t).tobytes())
    p.stdin.close()
    err = p.stderr.read().decode()
    if p.wait() != 0:
        raise RuntimeError(err)
    if rotate:
        rot = path + ".rot.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotate), "-i", path, "-c", "copy", rot], check=True)
        os.replace(rot, path)


class ColorTracker(Tracker):
    """Finds the magenta square in the analysis JPEGs. Records what it was asked to do."""

    def __init__(self, shape_hw, fail_chunks=(), crash_chunks=(), empty_chunks=()):
        self.shape_hw = shape_hw
        self.calls: List[dict] = []
        self.fail_chunks, self.crash_chunks, self.empty_chunks = set(fail_chunks), set(crash_chunks), set(empty_chunks)
        self.n_calls = 0

    def describe(self):
        return {"tracker": "ColorTracker (test double)"}

    def track_chunk(self, frame_dir, n_frames, inits):
        k = self.n_calls
        self.n_calls += 1
        self.calls.append({"n_frames": n_frames, "inits": [(i.obj_id, i.local_frame, i.source,
                                                           None if i.mask is None else int(i.mask.sum()), i.points) for i in inits]})
        if k in self.fail_chunks:
            raise TrackerError("simulated model failure")
        if k in self.crash_chunks:
            raise RuntimeError("simulated crash")
        start = {}
        for i in inits:
            start[i.obj_id] = min(start.get(i.obj_id, 10 ** 9), i.local_frame)
        out: ChunkMasks = {o: {} for o in start}
        if k in self.empty_chunks:
            return out
        for li in range(n_frames):
            img = cv2.imdecode(np.fromfile(os.path.join(frame_dir, f"{li}.jpg"), np.uint8), cv2.IMREAD_COLOR)   # unicode-safe
            m = (img[..., 0] > 170) & (img[..., 1] < 100) & (img[..., 2] > 170)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            if n > 1 and stats[1:, cv2.CC_STAT_AREA].max() >= 150:     # keep the largest blob, drop noise specks
                m = lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            else:
                m = np.zeros_like(m)
            for o, s0 in start.items():
                if li >= s0:
                    out[o][li] = pack_mask(m)
        return out
