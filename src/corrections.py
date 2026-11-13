"""Correction records: what a mask looked like before the user repainted it, and after.

One small .npz per saved correction in ``<log_dir>/corrections/``. Nothing reads these files yet; they are the raw material for
later features (suggesting padding from repeated edits, training a detector on corrected masks). They are derived from the footage
and stay on this machine: the folder is inside ``logs/``, which the repository ignores.

Fields: before, after (bit-packed bool masks of ``shape``), obj_id, frame_idx, timestamp_s, video_fingerprint, video_size,
before_source ("drawn" = an earlier drawn mask, "preview" = SAM 2's mask for the clicks, "last_run" = the mask of the last
processing run, "none" = started from nothing), added_px / removed_px / before_px / after_px, created (local time), app version.
The video's path and name are NOT stored; the fingerprint links a record to its video.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

RECORD_VERSION = 1


def corrections_dir(log_dir: str) -> str:
    return os.path.join(log_dir, "corrections")


def save_correction(log_dir: str, fingerprint: str, video_size, obj_id: int, frame_idx: int, timestamp_s: float,
                    before: Optional[np.ndarray], after: np.ndarray, before_source: str) -> str:
    """Write one record and return its path. ``before`` None means the user started from an empty mask."""
    after = np.asarray(after, dtype=bool)
    before = np.zeros_like(after) if before is None else np.asarray(before, dtype=bool)
    if before.shape != after.shape:
        raise ValueError(f"before {before.shape} and after {after.shape} differ")
    d = corrections_dir(log_dir)
    os.makedirs(d, exist_ok=True)
    name = f"{fingerprint[:10]}_o{int(obj_id)}_f{int(frame_idx):07d}_{time.strftime('%Y%m%d_%H%M%S')}"
    path = os.path.join(d, name + ".npz")
    k = 1
    while os.path.exists(path):
        k += 1
        path = os.path.join(d, f"{name}_{k}.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, version=RECORD_VERSION, before=np.packbits(before.reshape(-1)), after=np.packbits(after.reshape(-1)),
                            shape=np.array(after.shape), obj_id=int(obj_id), frame_idx=int(frame_idx), timestamp_s=float(timestamp_s),
                            video_fingerprint=str(fingerprint), video_size=np.array([int(video_size[0]), int(video_size[1])]),
                            before_source=str(before_source), added_px=int((after & ~before).sum()), removed_px=int((before & ~after).sum()),
                            before_px=int(before.sum()), after_px=int(after.sum()), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.replace(tmp, path)
    return path


def load_correction(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(v) for v in z["shape"])
        n = shape[0] * shape[1]
        out = {k: z[k][()] if z[k].ndim == 0 else z[k] for k in z.files if k not in ("before", "after", "shape")}
        out = {k: (v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v) for k, v in out.items()}
        out["shape"] = shape
        out["before"] = np.unpackbits(z["before"], count=n).reshape(shape).astype(bool)
        out["after"] = np.unpackbits(z["after"], count=n).reshape(shape).astype(bool)
    return out


def list_corrections(log_dir: str, fingerprint: Optional[str] = None) -> List[str]:
    d = corrections_dir(log_dir)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, n) for n in os.listdir(d)
                  if n.endswith(".npz") and (fingerprint is None or n.startswith(fingerprint[:10] + "_")))
