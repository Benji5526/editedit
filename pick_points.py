"""Click-to-select targets for censor_video.py (OpenCV window).

    python pick_points.py input/video.mp4 -o points.json

Frames come from FFmpeg (same decoder, rotation handling and frame numbering as the processing step);
saved coordinates are in full-resolution decoded-frame pixels no matter how small the window is.

Keys
  left click   positive point (this IS the object)      right click  negative point (this is NOT)
  n            new object (next id)                     1-9          switch to object id
  a / d        previous / next frame                    q / e        -10 / +10 frames
  z / c        -1 s / +1 s                              trackbar     jump anywhere
  u            undo last point of this frame            x            delete current object's points on this frame
  X            delete current object entirely           s            save
  Esc          quit (press twice if unsaved changes)
  Review mode (--review logs/<name>_review.json):
  m / b        jump to next / previous flagged frame (start and end of each flagged segment); the object named by
               the flag becomes the current object. Fix a lost object by left-clicking it on a later frame, save, and re-run censor_video.py with the updated points file.

Tracking of an object starts at the earliest frame where you clicked it. Pick the first frame where the
object is visible (frames before your first click are not tracked). Add more clicks on later frames to
correct drift.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.points_io import PointStore, PointsError, save_points, load_points  # noqa: F401  (PointStore re-exported)
from src.review import ReviewError, ReviewNavigator, load_review
from src.video_io import VideoInfo, probe_video, read_frame_at

COLORS = [(0, 200, 255), (80, 255, 80), (255, 120, 60), (255, 60, 255), (60, 255, 255), (200, 200, 60),
          (120, 120, 255), (255, 200, 200), (90, 160, 90)]


class Picker:
    def __init__(self, info: VideoInfo, out_path: str, display_width: int, review: Optional[ReviewNavigator] = None):
        self.info, self.out = info, out_path
        self.review = review
        self.scale = min(1.0, display_width / info.width)
        self.dw, self.dh = int(info.width * self.scale), int(info.height * self.scale)
        self.store = PointStore()
        self.frame_idx = 0
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.dirty = False
        self.msg = ""
        self.win = "pick_points"
        self._tb_lock = False

    def load_existing(self):
        if os.path.isfile(self.out):
            try:
                pr = load_points(self.out, self.info)
            except (PointsError, KeyError, ValueError) as e:
                print(f"existing {self.out} not loaded: {e}")
                return
            for oid, frames in pr.items():
                for fr, (pts, labs) in frames.items():
                    for (x, y), l in zip(pts, labs):
                        self.store.points.append({"obj_id": oid, "frame_idx": fr, "x": x, "y": y, "label": l})
            ids = sorted({p["obj_id"] for p in self.store.points})
            if ids:
                self.store.objects = ids
                self.store.current = ids[0]
            print(f"loaded {len(self.store.points)} points from {self.out}")

    def get_frame(self, idx: int) -> np.ndarray:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        full = read_frame_at(self.info, idx, scale_to=(self.dw, self.dh))
        self.cache[idx] = full
        while len(self.cache) > 12:
            self.cache.popitem(last=False)
        return full

    def render(self) -> np.ndarray:
        img = self.get_frame(self.frame_idx).copy()
        for p in self.store.points:
            if p["frame_idx"] != self.frame_idx:
                continue
            c = COLORS[(p["obj_id"] - 1) % len(COLORS)]
            xy = (int(p["x"] * self.scale), int(p["y"] * self.scale))
            if p["label"] == 1:
                cv2.circle(img, xy, 7, c, -1)
                cv2.circle(img, xy, 9, (255, 255, 255), 1)
            else:
                cv2.line(img, (xy[0] - 7, xy[1] - 7), (xy[0] + 7, xy[1] + 7), c, 3)
                cv2.line(img, (xy[0] - 7, xy[1] + 7), (xy[0] + 7, xy[1] - 7), c, 3)
            cv2.putText(img, str(p["obj_id"]), (xy[0] + 10, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        n_here = sum(1 for p in self.store.points if p["frame_idx"] == self.frame_idx and p["obj_id"] == self.store.current)
        bar = (f"frame {self.frame_idx}/{self.info.nb_frames - 1}  t={self.info.timestamp_of(self.frame_idx):.3f}s  "
               f"obj {self.store.current} ({n_here} pts here, {len(self.store.points)} total){'  *unsaved*' if self.dirty else ''}  {self.msg}")
        lines = [bar]
        if self.review is not None:
            lines.append(f"review: flag {self.review.index_of(self.frame_idx)}/{len(self.review)}  (m next, b prev)")
            lines += self.review.describe(self.frame_idx)
        cv2.rectangle(img, (0, 0), (img.shape[1], 8 + 20 * len(lines)), (0, 0, 0), -1)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (6, 19 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255) if k < 2 else (80, 200, 255), 1, cv2.LINE_AA)
        return img

    def goto(self, idx: int):
        self.frame_idx = int(min(max(0, idx), self.info.nb_frames - 1))
        self._tb_lock = True
        cv2.setTrackbarPos("frame", self.win, self.frame_idx)
        self._tb_lock = False

    def on_mouse(self, event, x, y, flags, _):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        fx = min(max(x / self.scale, 0), self.info.width - 1)
        fy = min(max(y / self.scale, 0), self.info.height - 1)
        self.store.add(self.frame_idx, fx, fy, 1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        self.dirty = True
        self.msg = ""

    def save(self):
        probs = self.store.valid()
        if probs:
            self.msg = "NOT saved: " + probs[0]
            print("NOT saved:", "; ".join(probs))
            return
        save_points(self.out, self.info, self.store.points)
        self.dirty = False
        self.msg = f"saved {len(self.store.points)} points"
        print(self.msg, "->", self.out)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        cv2.createTrackbar("frame", self.win, 0, max(1, self.info.nb_frames - 1),
                           lambda v: None if self._tb_lock else self._tb(v))
        self.load_existing()
        quit_armed = False
        fps = max(1, int(round(self.info.fps)))
        while True:
            cv2.imshow(self.win, self.render())
            k = cv2.waitKeyEx(30)
            if k == -1:
                continue
            ch = chr(k) if 0 <= k < 128 else ""
            if k == 27:
                if self.dirty and not quit_armed:
                    self.msg = "unsaved! press Esc again to quit, s to save"
                    quit_armed = True
                    continue
                break
            quit_armed = False
            if ch == "d" or k == 2555904:
                self.goto(self.frame_idx + 1)
            elif ch == "a" or k == 2424832:
                self.goto(self.frame_idx - 1)
            elif ch == "e":
                self.goto(self.frame_idx + 10)
            elif ch == "q":
                self.goto(self.frame_idx - 10)
            elif ch == "c":
                self.goto(self.frame_idx + fps)
            elif ch == "z":
                self.goto(self.frame_idx - fps)
            elif ch == "n":
                self.msg = f"new object {self.store.new_object()}"
            elif ch in "123456789" and ch:
                self.msg = "" if self.store.select(int(ch)) else f"object {ch} does not exist (press n)"
            elif ch == "u":
                self.dirty |= self.store.undo(self.frame_idx)
            elif ch == "x":
                self.dirty |= self.store.clear_frame(self.frame_idx) > 0
            elif ch == "X":
                self.dirty |= self.store.delete_object() > 0
            elif ch == "s":
                self.save()
            elif ch in ("m", "b") and self.review is not None:
                tgt = self.review.next_after(self.frame_idx) if ch == "m" else self.review.prev_before(self.frame_idx)
                if tgt is None:
                    self.msg = "no more flags in that direction"
                else:
                    self.goto(tgt)
                    obj = self.review.object_at(tgt)
                    self.msg = f"-> obj {obj}" if obj is not None and self.store.select(obj) else ""
            elif ch in ("m", "b"):
                self.msg = "start with --review <review.json> to use m/b"
        cv2.destroyAllWindows()

    def _tb(self, v):
        self.frame_idx = int(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Click targets in a video; writes a JSON for censor_video.py")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="points.json")
    ap.add_argument("--display-width", type=int, default=1600, help="window width (coordinates are saved in full-res pixels)")
    ap.add_argument("--review", help="review log JSON from censor_video.py: m/b jump between flagged frames")
    ap.add_argument("--review-min-severity", choices=["info", "warn", "error"], default="warn")
    a = ap.parse_args(argv)
    info = probe_video(a.video)
    nav = None
    if a.review:
        try:
            nav = load_review(a.review, info, a.review_min_severity)
        except (ReviewError, OSError, ValueError) as e:
            print(f"cannot use review log: {e}")
            return 1
        print(f"review: {len(nav)} flagged frames (severity >= {a.review_min_severity})")
    print(f"{info.width}x{info.height} {info.fps_rational} fps, {info.nb_frames} frames, rotation={info.rotation}, VFR={info.is_vfr}")
    if info.is_vfr:
        print(f"WARNING: variable frame rate ({info.vfr_detail}). Timestamps come from real packet times; frame numbers are decode order.")
    Picker(info, a.out, a.display_width, nav).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
