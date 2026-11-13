"""Mask preview server: answers "what would SAM 2 select for these clicks on this frame?" while the user is still clicking.

Started by the desktop app as ``censor_video.py <video> --preview-server --sam2-checkpoint ...`` (the same worker executable as a
processing job, so the app itself never loads torch). It keeps the model loaded and reads one JSON request per line on stdin:

    {"id": 7, "frame": 120, "objects": {"1": {"points": [[x, y], ...], "labels": [1, 0, ...]}}}      full-resolution pixels
    {"cmd": "quit"}

and answers on stdout with lines starting with '@@PREVIEW ':

    {"event": "loading"}  {"event": "ready", "analysis_size": [w, h], "tracker": {...}}
    {"event": "masks", "id": 7, "frame": 120, "shape": [h, w], "masks": {"1": "<base64 of zlib(bit-packed mask)>"}, "skipped": {...}}
    {"event": "error", "id": 7, "reason": "..."}        (the server keeps running)
    {"event": "fatal", "reason": "..."}                 (model could not be loaded; the server exits)

The mask of a frame is computed exactly like the first frame of a processing chunk: the analysis-size frame is written as a JPEG,
the clicks are given to the tracker on that frame, and the tracker's mask for that frame is returned. It therefore shows what the
job will get ON THE CLICKED FRAME; how well the object is followed on later frames can only be seen after processing.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import tempfile
import zlib
from typing import Dict, Tuple

import numpy as np

PREVIEW_PREFIX = "@@PREVIEW "


def analysis_size(width: int, height: int, analysis_width: int) -> Tuple[int, int]:
    """Same rounding as the pipeline."""
    sc = min(1.0, analysis_width / width)
    return int(round(width * sc)), int(round(height * sc))


def preview_masks(tracker, info, size: Tuple[int, int], frame: int, objects: dict, work_dir: str) -> Tuple[Dict[int, np.ndarray], dict]:
    """Masks (bit-packed, analysis size) for the clicks of one frame. Returns (masks, skipped {obj_id: reason})."""
    from .pipeline import write_jpeg
    from .tracking import Init
    from .video_io import read_frame_at

    aw, ah = size
    sx, sy = aw / info.width, ah / info.height
    inits, skipped = [], {}
    for key, o in objects.items():
        oid = int(key)
        pts, labs = o.get("points") or [], [int(v) for v in (o.get("labels") or [])]
        if len(pts) != len(labs) or not pts:
            skipped[oid] = "no points"
        elif 1 not in labs:
            skipped[oid] = "needs at least one positive (left-click) point"
        else:
            inits.append(Init(oid, 0, [(float(x) * sx, float(y) * sy) for x, y in pts], labs, source="points"))
    if not inits:
        return {}, skipped
    if not (0 <= int(frame) < info.nb_frames):
        raise ValueError(f"frame {frame} is outside the video")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    write_jpeg(os.path.join(work_dir, "0.jpg"), read_frame_at(info, int(frame), scale_to=(aw, ah)))
    out = tracker.track_chunk(work_dir, 1, inits)
    masks = {oid: frames[0] for oid, frames in out.items() if 0 in frames}
    for it in inits:
        if it.obj_id not in masks:
            skipped[it.obj_id] = "the model returned no mask"
    return masks, skipped


def encode_mask(packed: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(packed).tobytes(), 1)).decode("ascii")


def decode_mask(text: str, shape: Tuple[int, int]) -> np.ndarray:
    packed = np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype=np.uint8)
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def serve(input_path: str, checkpoint: str, config=None, device: str = "cuda", analysis_width: int = 1920,
          stdin=None, stdout=None, tracker=None) -> int:
    """Run the request loop. ``tracker`` replaces SAM 2 (tests)."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout

    def emit(ev: dict):
        stdout.write(PREVIEW_PREFIX + json.dumps(ev) + "\n")
        stdout.flush()

    work = tempfile.mkdtemp(prefix="vc_preview_")
    try:
        emit({"event": "loading"})
        try:
            from .video_io import probe_video
            info = probe_video(input_path)
            if tracker is None:
                from .tracking import SAM2Tracker
                tracker = SAM2Tracker(checkpoint, config, device=device)
        except Exception as e:  # noqa: BLE001
            emit({"event": "fatal", "reason": f"{type(e).__name__}: {e}"})
            return 1
        size = analysis_size(info.width, info.height, analysis_width)
        emit({"event": "ready", "analysis_size": list(size), "tracker": tracker.describe()})
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                emit({"event": "error", "id": None, "reason": "request is not JSON"})
                continue
            if req.get("cmd") == "quit":
                break
            rid = req.get("id")
            try:
                masks, skipped = preview_masks(tracker, info, size, req["frame"], req.get("objects") or {}, os.path.join(work, "f"))
                emit({"event": "masks", "id": rid, "frame": int(req["frame"]), "shape": [size[1], size[0]],
                      "masks": {str(k): encode_mask(v) for k, v in masks.items()}, "skipped": {str(k): v for k, v in skipped.items()}})
            except Exception as e:  # noqa: BLE001  (one bad request must not end the session)
                emit({"event": "error", "id": rid, "reason": f"{type(e).__name__}: {e}"})
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
