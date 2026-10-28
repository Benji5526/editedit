"""Isolate where the resident memory (RSS) grows during chunked processing.

    python tools/leak_probe.py sam2      # SAM 2 only: the same chunk of JPEGs tracked repeatedly
    python tools/leak_probe.py nudenet   # NudeNet only
    python tools/leak_probe.py frames    # reading full-resolution frames from the video and discarding them
"""
import gc
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
proc = psutil.Process()


def rss():
    return proc.memory_info().rss / 2 ** 20


def table(name, vals):
    d = np.diff(vals)
    print(f"{name}: first={vals[0]:.0f} MB last={vals[-1]:.0f} MB growth={vals[-1] - vals[0]:+.0f} MB over {len(vals) - 1} iterations "
          f"(slope {np.polyfit(np.arange(len(vals)), vals, 1)[0]:+.1f} MB/iter; last-10 mean change {d[-10:].mean():+.1f})", flush=True)


def probe_sam2(iters, frames=61):
    from src.tracking import Init, SAM2Tracker
    tr = SAM2Tracker(os.path.join(ROOT, "models", "sam2.1_hiera_small.pt"), device="cuda")
    d = tempfile.mkdtemp()
    base = cv2.imread(os.path.join(ROOT, "samples", "soak", "_probe.jpg"))
    for i in range(frames):
        cv2.imwrite(os.path.join(d, f"{i}.jpg"), np.roll(base, i * 3, axis=1))
    vals = []
    for it in range(iters):
        tr.track_chunk(d, frames, [Init(1, 0, [(640.0, 360.0)], [1])])
        gc.collect()
        vals.append(rss())
        if it % 5 == 0:
            print(f"  sam2 iter {it}: rss={vals[-1]:.0f} MB", flush=True)
    table("SAM 2 track_chunk", vals)
    shutil.rmtree(d, ignore_errors=True)


def probe_nudenet(iters, per=60):
    from src.detection import NudenetDetector
    det = NudenetDetector(0.25)
    img = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
    vals = []
    for it in range(iters):
        for _ in range(per):
            det.detect(img)
        vals.append(rss())
    table("NudeNet detect x%d per iter" % per, vals)


def probe_frames(iters, per=60):
    from src.video_io import FrameReader, probe_video
    info = probe_video(sys.argv[2])
    vals = []
    r = FrameReader(info)
    for it in range(iters):
        for _ in range(per):
            f = r.read()
            assert f is not None
            del f
        vals.append(rss())
    r.close()
    table("full-res FrameReader x%d per iter" % per, vals)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "sam2":
        probe_sam2(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif which == "nudenet":
        probe_nudenet(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif which == "frames":
        probe_frames(19)
