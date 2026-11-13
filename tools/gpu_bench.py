"""Measure where the GPU helps on this machine: decoding (software vs -hwaccel), encoding (libx264 vs h264_nvenc) and NudeNet
(CPU vs CUDA). Uses a synthetic clip unless a video is given. Prints a small report; nothing is changed.

    python tools/gpu_bench.py                       # synthetic 3840x2160 60 fps, 240 frames
    python tools/gpu_bench.py --video my.mp4 --frames 300
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_io import FrameReader, SegmentWriter, find_tool, nvenc_available, probe_video  # noqa: E402


def make_clip(path: str, frames: int, size: str = "3840x2160", fps: int = 60):
    """A moving test pattern with film-grain-like noise, encoded like a camera file (H.264 high bitrate)."""
    cmd = [find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}", "-frames:v", str(frames),
           "-vf", "noise=alls=12:allf=t", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-g", str(fps), path]
    subprocess.run(cmd, check=True)


def read_all(info, n, scale_to=None, hwaccel=False, digest=False):
    t = time.perf_counter()
    h = hashlib.md5()
    count = 0
    with FrameReader(info, max_frames=n, scale_to=scale_to, hwaccel=hwaccel) as r:
        for f in r:
            count += 1
            if digest and count % 10 == 1:
                h.update(f.tobytes())
    return (time.perf_counter() - t) / max(1, count) * 1000, count, h.hexdigest()


def encode(info, frames_src, n, encoder, out):
    w = SegmentWriter(out, info, encoder, 16, "medium")
    t = time.perf_counter()
    for i in range(n):
        w.write(frames_src[i % len(frames_src)])
    w.close()
    return (time.perf_counter() - t) / n * 1000, os.path.getsize(out) / 2 ** 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--analysis-width", type=int, default=1920)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="gpu_bench_")
    try:
        video = a.video
        if not video:
            video = os.path.join(tmp, "clip.mp4")
            print("making a synthetic 3840x2160 60 fps clip ...")
            make_clip(video, a.frames)
        info = probe_video(video)
        n = min(a.frames, info.nb_frames)
        aw = min(info.width, a.analysis_width)
        ah = int(round(info.height * aw / info.width))
        print(f"video: {info.width}x{info.height} {info.fps:.2f} fps {info.codec}, {n} frames\n")

        print("decoding (ms per frame, lower is better)")
        for label, scale in (("full resolution", None), (f"scaled to {aw}x{ah}", (aw, ah))):
            sw, c1, d1 = read_all(info, n, scale, hwaccel=False, digest=True)
            hw, c2, d2 = read_all(info, n, scale, hwaccel=True, digest=True)
            same = "identical frames" if (c1 == c2 and d1 == d2) else "FRAMES DIFFER"
            print(f"  {label:22s} software {sw:6.1f}   hwaccel {hw:6.1f}   ({sw / hw:4.2f}x, {same})")

        print("\nencoding of full-resolution frames (ms per frame; size of the result)")
        with FrameReader(info, max_frames=8) as r:
            src = [f.copy() for f in r]
        encoders = ["libx264"] + (["h264_nvenc"] if nvenc_available() else [])
        base = None
        for enc in encoders:
            ms, mb = encode(info, src, min(n, 120), enc, os.path.join(tmp, enc + ".mp4"))
            base = base or ms
            print(f"  {enc:12s} {ms:6.1f} ms   {mb:7.1f} MiB   ({base / ms:4.2f}x)")
        if len(encoders) == 1:
            print("  h264_nvenc is not available in this ffmpeg")

        print("\nNudeNet (ms per frame at the analysis size, incl. pre- and post-processing)")
        try:
            from src.detection import NudenetDetector
            img = np.ascontiguousarray(np.resize(src[0], (ah, aw, 3))) if src[0].shape[:2] != (ah, aw) else src[0]
            import cv2
            img = cv2.resize(src[0], (aw, ah))
            models = [("bundled 320n", {})]
            m640 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "640m.onnx")
            if os.path.isfile(m640):
                models.append(("640m", dict(model_path=m640, resolution=640)))
            for name, kw in models:
                row = []
                for dev in ("cpu", "auto"):
                    d = NudenetDetector(device=dev, **kw)
                    for _ in range(3):
                        d.detect(img)
                    t = time.perf_counter()
                    for _ in range(20):
                        d.detect(img)
                    row.append(((time.perf_counter() - t) / 20 * 1000, d.provider.replace("ExecutionProvider", "")))
                print(f"  {name:14s} " + "   ".join(f"{p} {ms:6.1f}" for ms, p in row) + f"   ({row[0][0] / row[1][0]:4.2f}x)")
        except Exception as e:  # noqa: BLE001
            print("  skipped:", type(e).__name__, e)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
