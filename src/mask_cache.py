"""Per-chunk mask cache (masks_NNNNN.npz): what the analysis stage produced and the render stage works from.

Only numpy/json here, so the desktop app can read the cache without importing the processing pipeline.

Layout of one file: ``meta`` (JSON string: frame range, analysis-frame shape, object ids, detections per frame incl. gap-filled
ones, fail-closed regions/flags, the analysis-stage review events) and per object ``p<id>`` (bool per frame: has a mask) plus
``m<id>`` (the bit-packed masks of the frames that have one, in order; see tracking.pack_mask). Masks, boxes and regions are in
analysis-frame coordinates. The files are derived from the footage: they stay in the job's work folder and are deleted with it.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

CACHE_VERSION = 1


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def cache_path(job_dir: str, ci: int) -> str:
    return os.path.join(job_dir, f"masks_{ci:05d}.npz")


def save_chunk_cache(job_dir: str, rj, shape: Tuple[int, int]) -> None:
    """Write everything the render stage of one chunk needs (``rj`` is the pipeline's RenderJob) to masks_NNNNN.npz."""
    arrays = {}
    for oid, seq in rj.packed.items():
        present = np.array([m is not None for m in seq], dtype=bool)
        arrays[f"p{oid}"] = present
        if present.any():
            arrays[f"m{oid}"] = np.stack([m for m in seq if m is not None])
    meta = {"version": CACHE_VERSION, "chunk": rj.ci, "start": rj.s, "end": rj.e, "shape": [int(shape[0]), int(shape[1])],
            "obj_ids": [int(o) for o in rj.packed], "filled": [[asdict(d) for d in ds] for ds in rj.filled],
            "fc_regions": [[[int(v) for v in r] for r in rs] for rs in rj.fc_regions], "fc_full": [bool(v) for v in rj.fc_full],
            "fc_reason": list(rj.fc_reason), "events": rj.a_events, "detections": rj.chunk_dets}
    arrays["meta"] = np.array(json.dumps(meta, default=_json_default))
    path = cache_path(job_dir, rj.ci)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_chunk_cache(job_dir: str, ci: int) -> dict:
    """Read masks_NNNNN.npz back: {"meta": dict, "packed": {obj_id: [packed mask or None per frame]}}."""
    with np.load(cache_path(job_dir, ci), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][()]))
        if meta.get("version") != CACHE_VERSION:
            raise ValueError(f"unsupported mask cache version {meta.get('version')}")
        n = meta["end"] - meta["start"]
        packed = {}
        for oid in meta["obj_ids"]:
            present = z[f"p{oid}"]
            rows = z[f"m{oid}"] if f"m{oid}" in z.files else None
            seq, k = [], 0
            for i in range(n):
                if present[i]:
                    seq.append(rows[k])
                    k += 1
                else:
                    seq.append(None)
            packed[int(oid)] = seq
    return {"meta": meta, "packed": packed}


def unpack(p: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return np.unpackbits(p, count=shape[0] * shape[1]).reshape(shape).astype(bool)


class JobMasks:
    """Random access by frame to the cached analysis results of a job's kept chunks (for showing them in the app).

    ``frame(idx)`` returns None when that chunk has no cache, else a dict in ANALYSIS-frame coordinates:
    masks {obj_id: bool array}, boxes [(x1, y1, x2, y2, class, filled)], fc_regions [(x1, y1, x2, y2)], fc_full bool.
    These are the raw results before the render stage grows them (mask padding, box expansion, temporal union)."""

    def __init__(self, job_dir: str, keep_chunks: int = 2):
        self.job_dir = job_dir
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
            job = json.load(f)
        self.chunk_size = int(job["config"]["chunk_size"])
        self.total_frames = int(job["total_frames"])
        self.analysis_size = tuple(job.get("analysis_size") or (0, 0))       # (width, height)
        self.points_sha = job.get("points_sha")
        self.created = job.get("created")
        self.run_id = job.get("run_id")
        self._chunks: "OrderedDict[int, Optional[dict]]" = OrderedDict()
        self._keep = keep_chunks

    def available_chunks(self) -> int:
        n = 0
        while os.path.isfile(cache_path(self.job_dir, n)):
            n += 1
        return n

    def _chunk(self, ci: int) -> Optional[dict]:
        if ci in self._chunks:
            self._chunks.move_to_end(ci)
            return self._chunks[ci]
        try:
            c = load_chunk_cache(self.job_dir, ci) if os.path.isfile(cache_path(self.job_dir, ci)) else None
        except Exception:  # noqa: BLE001  (an unreadable cache is simply not shown)
            c = None
        self._chunks[ci] = c
        while len(self._chunks) > self._keep:
            self._chunks.popitem(last=False)
        return c

    def summary(self, should_stop=None) -> dict:
        """What the kept run has for every frame, without unpacking any mask:
        {"masks": {obj_id: bool array}, "detections": bool array, "fail_closed": bool array}, each ``total_frames`` long.
        "masks" is True where the object has a non-empty mask. Frames of chunks without a cache stay False.
        ``should_stop()`` is polled between chunks (returns the partial result)."""
        n = self.total_frames
        masks: Dict[int, np.ndarray] = {}
        det, fc = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        ci = 0
        while ci * self.chunk_size < n:
            if should_stop is not None and should_stop():
                break
            path = cache_path(self.job_dir, ci)
            if os.path.isfile(path):
                try:
                    with np.load(path, allow_pickle=False) as z:
                        meta = json.loads(str(z["meta"][()]))
                        s0, e0 = int(meta["start"]), min(n, int(meta["end"]))
                        for oid in meta["obj_ids"]:
                            present = z[f"p{oid}"][: e0 - s0]
                            has = np.zeros(e0 - s0, dtype=bool)
                            if f"m{oid}" in z.files and present.any():
                                has[np.flatnonzero(present)] = z[f"m{oid}"].any(axis=1)[: int(present.sum())]
                            masks.setdefault(int(oid), np.zeros(n, dtype=bool))[s0:e0] = has
                        det[s0:e0] = [bool(ds) for ds in meta["filled"][: e0 - s0]]
                        fc[s0:e0] = [bool(r) or bool(f) for r, f in zip(meta["fc_regions"], meta["fc_full"])][: e0 - s0]
                except Exception:  # noqa: BLE001  (an unreadable cache is simply not shown)
                    pass
            ci += 1
        return {"masks": masks, "detections": det, "fail_closed": fc}

    def frame(self, idx: int) -> Optional[dict]:
        if not (0 <= idx < self.total_frames) or self.chunk_size <= 0:
            return None
        c = self._chunk(idx // self.chunk_size)
        if c is None:
            return None
        meta = c["meta"]
        i = idx - meta["start"]
        if not (0 <= i < meta["end"] - meta["start"]):
            return None
        shape = tuple(meta["shape"])
        masks: Dict[int, np.ndarray] = {}
        for oid, seq in c["packed"].items():
            if seq[i] is not None:
                m = unpack(seq[i], shape)
                if m.any():                       # a lost object is stored as an empty mask: nothing to show
                    masks[oid] = m
        boxes = [(d["x1"], d["y1"], d["x2"], d["y2"], d["cls"], bool(d.get("filled"))) for d in meta["filled"][i]]
        return {"shape": shape, "masks": masks, "boxes": boxes, "fc_regions": [tuple(r) for r in meta["fc_regions"][i]],
                "fc_full": bool(meta["fc_full"][i])}
