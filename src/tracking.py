"""SAM 2 video tracking, one chunk at a time.

How SAM 2 is used (verified against sam2 1.1.0 on PyPI, ``SAM2VideoPredictor``):
  * ``init_state(video_path=<dir of JPEGs named 0.jpg,1.jpg,...>)`` loads the frames of ONE chunk.
  * ``add_new_points_or_box(state, frame_idx, obj_id, points, labels, clear_old_points, normalize_coords)``
    prompts an object with positive(1)/negative(0) clicks (pixel coordinates of the loaded frames).
  * ``add_new_mask(state, frame_idx, obj_id, mask)`` prompts an object with a bool 2-D mask.
  * ``propagate_in_video(state, start_frame_idx)`` yields ``(frame_idx, obj_ids, video_res_masks)``;
    masks are logits, >0 is foreground. It ALSO yields frames before an object's prompt frame
    (running without memory); those outputs are meaningless and are discarded here.
  * ``reset_state(state)`` frees the chunk.

Chunk boundaries: SAM 2's internal memory bank (memory features of previously tracked frames) is NOT
carried between chunks; only a mask is. Each chunk after the first starts with ONE overlap frame
(the previous chunk's last frame) prompted with the last mask via ``add_new_mask``. This re-seeds the
object but is not equal to continuing with the full memory: appearance history from before the
boundary is lost, so drift/ID confusion risk is higher right after a boundary. That is a known
limitation, logged per boundary by the pipeline.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class TrackerError(RuntimeError):
    pass


def pack_mask(m: np.ndarray) -> np.ndarray:
    return np.packbits(m.reshape(-1))


def unpack_mask(p: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return np.unpackbits(p, count=shape[0] * shape[1]).reshape(shape).astype(bool)


@dataclass
class Init:
    """How to (re)seed one object inside a chunk. Coordinates are in analysis-frame pixels."""
    obj_id: int
    local_frame: int
    points: Optional[List[Tuple[float, float]]] = None
    labels: Optional[List[int]] = None
    mask: Optional[np.ndarray] = None     # bool (H, W); used when points is None
    source: str = "points"                # "points" | "carry" | "carry_stale"


# obj_id -> local_frame -> packed bool mask. Only frames >= the object's first prompt frame are present.
ChunkMasks = Dict[int, Dict[int, np.ndarray]]


class Tracker:
    """Interface so the pipeline can be tested with a stand-in without a GPU."""
    shape_hw: Tuple[int, int]

    def track_chunk(self, frame_dir: str, n_frames: int, inits: List[Init]) -> ChunkMasks:
        raise NotImplementedError

    def describe(self) -> dict:
        return {}

    def close(self):
        pass


SAM21_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml", "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml", "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def guess_config(checkpoint: str) -> str:
    name = os.path.basename(checkpoint).lower()
    for key, cfg in SAM21_CONFIGS.items():
        if key in name and "sam2.1" in name:
            return cfg
    raise TrackerError(f"cannot infer SAM 2.1 config from '{name}'; pass --sam2-config explicitly")


class SAM2Tracker(Tracker):
    def __init__(self, checkpoint: Optional[str], config: Optional[str] = None, device: str = "cuda",
                 offload_video_to_cpu: bool = True, offload_state_to_cpu: bool = False, shape_hw=None,
                 _random_weights_api_test: bool = False):
        """``_random_weights_api_test`` builds the network WITHOUT weights. Output is meaningless; it exists
        only so tests can exercise the real SAM 2 call sequence when no checkpoint is available."""
        import torch
        if _random_weights_api_test:
            checkpoint = None
            if not config:
                raise TrackerError("config is required for the random-weights API test")
        elif not checkpoint or not os.path.isfile(checkpoint):
            raise TrackerError(f"SAM 2 checkpoint not found: {checkpoint} (see models/README.md)")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise TrackerError("CUDA requested but torch.cuda.is_available() is False; refusing to silently fall back to CPU. "
                               "Use --device cpu explicitly (very slow).")
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        self.device = device
        self.config = config or guess_config(checkpoint)
        self.checkpoint = checkpoint or "<random weights: API test only>"
        self.predictor = build_sam2_video_predictor(self.config, checkpoint, device=device)
        self.offload_video, self.offload_state = offload_video_to_cpu, offload_state_to_cpu
        self.shape_hw = shape_hw

    def describe(self) -> dict:
        t = self.torch
        try:
            from importlib.metadata import version
            sv = version("sam2")
        except Exception:  # noqa: BLE001
            sv = "unknown"
        return {"sam2": sv, "torch": t.__version__, "cuda": t.version.cuda, "device": self.device,
                "gpu": t.cuda.get_device_name(0) if self.device.startswith("cuda") else None,
                "checkpoint": os.path.basename(self.checkpoint), "config": self.config}

    def track_chunk(self, frame_dir: str, n_frames: int, inits: List[Init]) -> ChunkMasks:
        torch = self.torch
        out: ChunkMasks = {i.obj_id: {} for i in inits}
        start_of = {}
        for i in inits:
            start_of[i.obj_id] = min(start_of.get(i.obj_id, 10 ** 9), i.local_frame)
        state = None
        try:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else _nullctx()
            with torch.inference_mode(), ctx:
                state = self.predictor.init_state(frame_dir, offload_video_to_cpu=self.offload_video,
                                                  offload_state_to_cpu=self.offload_state)
                if state["num_frames"] != n_frames:
                    raise TrackerError(f"SAM 2 loaded {state['num_frames']} frames, expected {n_frames}")
                # masks first, then clicks, so a refinement click on a later frame builds on the seeded mask
                for it in sorted(inits, key=lambda i: (i.mask is None, i.local_frame)):
                    if it.mask is not None and it.points is None:
                        self.predictor.add_new_mask(state, frame_idx=it.local_frame, obj_id=it.obj_id, mask=it.mask)
                    else:
                        self.predictor.add_new_points_or_box(
                            state, frame_idx=it.local_frame, obj_id=it.obj_id,
                            points=np.asarray(it.points, dtype=np.float32), labels=np.asarray(it.labels, dtype=np.int32),
                            clear_old_points=True, normalize_coords=True)
                first = min(start_of.values())
                for fidx, obj_ids, logits in self.predictor.propagate_in_video(state, start_frame_idx=first):
                    for k, oid in enumerate(obj_ids):
                        if fidx < start_of.get(int(oid), 10 ** 9):
                            continue  # output before the object's first prompt: meaningless
                        m = (logits[k, 0] > 0.0).cpu().numpy()
                        out[int(oid)][int(fidx)] = pack_mask(m)
                        if self.shape_hw is None:
                            self.shape_hw = m.shape
        except TrackerError:
            raise
        except Exception as e:  # noqa: BLE001  (includes torch.cuda.OutOfMemoryError)
            raise TrackerError(f"{type(e).__name__}: {e}") from e
        finally:
            try:
                if state is not None:
                    self.predictor.reset_state(state)
            finally:
                del state
                gc.collect()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        return out


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
