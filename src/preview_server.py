"""Mask preview server: answers "what would SAM 2 select for these clicks on this frame?" while the user is still clicking.

Started by the desktop app as ``censor_video.py <video> --preview-server --sam2-checkpoint ...`` (the same worker executable as a
processing job, so the app itself never loads torch). It keeps the model loaded and reads one JSON request per line on stdin:

    {"id": 7, "frame": 120, "objects": {"1": {"points": [[x, y], ...], "labels": [1, 0, ...]}}}      full-resolution pixels
    {"cmd": "quit"}

and answers on stdout with lines starting with '@@PREVIEW ':

    {"event": "loading"}  {"event": "ready", "analysis_size": [w, h], "tracker": {...}}
    {"event": "masks", "id": 7, "frame": 120, "shape": [h, w], "masks": {"1": "<base64 of zlib(bit-packed mask)>"}, "skipped": {...}}
    {"event": "error", "id": 7, "reason": "..."}        (the server keeps running)

A second request runs a TRIAL TRACKING of one object over a short range, to compare a correction before it is processed:

    {"cmd": "track", "id": 8, "obj_id": 1, "frame": 120, "frames": 60,
     "prompts": {"0": {"mask": "<base64 zlib bits>", "shape": [h, w]}, "14": {"points": [[x, y]], "labels": [1]}}}
    -> {"event": "track", "id": 8, "obj_id": 1, "frame": 120, "frames": 60, "shape": [h, w], "masks": ["<...>" or null, ...]}

The keys of "prompts" are offsets from "frame"; offset 0 must be there. The frames are tracked like one processing chunk that
starts at "frame". That is an estimate of what a job would produce: a real job reaches these frames with the memory of the
frames before them, and its chunk boundaries fall elsewhere.
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
            skipped[oid] = "점이 없습니다"
        elif 1 not in labs:
            skipped[oid] = "긍정(왼쪽 클릭) 점이 하나 이상 필요합니다"
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
            skipped[it.obj_id] = "모델이 마스크를 만들지 못했습니다"
    return masks, skipped


MAX_TRIAL_FRAMES = 300


def trial_track(tracker, info, size: Tuple[int, int], obj_id: int, frame: int, frames: int, prompts: dict, work_dir: str) -> list:
    """Track one object over ``frames`` frames starting at ``frame``, from the prompts given as offsets from ``frame``
    ({offset: {"points", "labels"} or {"mask": bool array}}). Returns one bit-packed mask (analysis size) or None per frame."""
    import cv2
    from .pipeline import write_jpeg
    from .tracking import Init
    from .video_io import FrameReader

    aw, ah = size
    sx, sy = aw / info.width, ah / info.height
    frame, frames = int(frame), int(frames)
    if not (0 <= frame < info.nb_frames):
        raise ValueError(f"frame {frame} is outside the video")
    if not (1 <= frames <= MAX_TRIAL_FRAMES):
        raise ValueError(f"frames must be 1-{MAX_TRIAL_FRAMES}")
    n = min(frames, info.nb_frames - frame)
    inits = []
    for key, pr in sorted(prompts.items(), key=lambda kv: int(kv[0])):
        off = int(key)
        if not (0 <= off < n):
            continue
        if pr.get("mask") is not None:
            m = np.asarray(pr["mask"], dtype=bool)
            if m.shape != (ah, aw):
                m = cv2.resize(m.astype(np.uint8), (aw, ah), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not m.any():
                raise ValueError(f"the mask at offset {off} is empty")
            inits.append(Init(int(obj_id), off, mask=m, source="mask_prompt"))
        else:
            pts, labs = pr.get("points") or [], [int(v) for v in (pr.get("labels") or [])]
            if not pts or len(pts) != len(labs) or 1 not in labs:
                raise ValueError(f"the prompt at offset {off} needs at least one positive point")
            inits.append(Init(int(obj_id), off, [(float(x) * sx, float(y) * sy) for x, y in pts], labs, source="points"))
    if not inits or inits[0].local_frame != 0:
        raise ValueError("the first frame needs a prompt (clicks or a drawn mask) for this object")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    with FrameReader(info, start_frame=frame, max_frames=n, scale_to=(aw, ah)) as r:
        for i in range(n):
            fa = r.read()
            if fa is None:
                raise ValueError(f"could not read frame {frame + i}")
            write_jpeg(os.path.join(work_dir, f"{i}.jpg"), fa)
    out = tracker.track_chunk(work_dir, n, inits).get(int(obj_id), {})
    return [out.get(i) for i in range(n)]


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
            if req.get("cmd") == "track":
                try:
                    prompts = {}
                    for key, pr in (req.get("prompts") or {}).items():
                        prompts[key] = ({"mask": decode_mask(pr["mask"], tuple(pr["shape"]))} if pr.get("mask") is not None else pr)
                    masks = trial_track(tracker, info, size, req["obj_id"], req["frame"], req["frames"], prompts, os.path.join(work, "t"))
                    emit({"event": "track", "id": rid, "obj_id": int(req["obj_id"]), "frame": int(req["frame"]), "frames": len(masks),
                          "shape": [size[1], size[0]], "masks": [None if m is None else encode_mask(m) for m in masks]})
                except Exception as e:  # noqa: BLE001
                    emit({"event": "error", "id": rid, "reason": f"{type(e).__name__}: {e}"})
                continue
            try:
                masks, skipped = preview_masks(tracker, info, size, req["frame"], req.get("objects") or {}, os.path.join(work, "f"))
                emit({"event": "masks", "id": rid, "frame": int(req["frame"]), "shape": [size[1], size[0]],
                      "masks": {str(k): encode_mask(v) for k, v in masks.items()}, "skipped": {str(k): v for k, v in skipped.items()}})
            except Exception as e:  # noqa: BLE001  (one bad request must not end the session)
                emit({"event": "error", "id": rid, "reason": f"{type(e).__name__}: {e}"})
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
