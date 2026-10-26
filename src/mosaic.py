"""Mask utilities and mosaic (pixelation). Pure numpy/OpenCV, no model dependencies.

Convention: masks are 2-D bool arrays, boxes are (x1, y1, x2, y2) with x2/y2 exclusive.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def merge_masks(masks: Iterable[Optional[np.ndarray]], shape: Tuple[int, int]) -> np.ndarray:
    """Logical OR of all masks. None entries are skipped; a union can never lose a member."""
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        if m is None:
            continue
        if m.shape != shape:
            raise ValueError(f"mask shape {m.shape} != {shape}")
        out |= m
    return out


def mask_bbox(mask: np.ndarray) -> Optional[Box]:
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def clip_box(box: Sequence[float], shape: Tuple[int, int]) -> Optional[Box]:
    h, w = shape
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box: Sequence[float], ratio: float, pad_px: float, shape: Tuple[int, int]) -> Optional[Box]:
    """Grow a box by ``ratio`` of its size on each side plus ``pad_px`` pixels, clipped to the image."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return clip_box((x1 - w * ratio - pad_px, y1 - h * ratio - pad_px,
                     x2 + w * ratio + pad_px, y2 + h * ratio + pad_px), shape)


def boxes_to_mask(boxes: Iterable[Optional[Sequence[float]]], shape: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for b in boxes:
        if b is None:
            continue
        cb = clip_box(b, shape)
        if cb:
            m[cb[1]:cb[3], cb[0]:cb[2]] = True
    return m


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow the mask by ``px`` pixels (elliptical kernel). px<=0 returns a copy."""
    if px <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def temporal_union(masks: Sequence[Optional[np.ndarray]], idx: int, radius: int) -> Optional[np.ndarray]:
    """Union of masks in [idx-radius, idx+radius]; keeps the region from collapsing for one noisy frame."""
    lo, hi = max(0, idx - radius), min(len(masks), idx + radius + 1)
    acc = None
    for m in masks[lo:hi]:
        if m is None:
            continue
        acc = m.copy() if acc is None else (acc | m)
    return acc


def upscale_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    up = cv2.resize(mask.astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return up > 127


def pixelate_region(frame: np.ndarray, mask: np.ndarray, block: int) -> np.ndarray:
    """Pixelate ``frame`` in place where ``mask`` is True. The block grid is anchored to the image
    origin so blocks do not shimmer as the mask moves. Returns the same array."""
    if frame.shape[:2] != mask.shape:
        raise ValueError(f"frame {frame.shape[:2]} and mask {mask.shape} differ")
    bb = mask_bbox(mask)
    if bb is None:
        return frame
    block = max(2, int(block))
    h, w = mask.shape
    x1, y1 = bb[0] // block * block, bb[1] // block * block
    x2, y2 = min(w, -(-bb[2] // block) * block), min(h, -(-bb[3] // block) * block)
    region = frame[y1:y2, x1:x2]
    rh, rw = region.shape[:2]
    small = cv2.resize(region, (-(-rw // block), -(-rh // block)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    sub = mask[y1:y2, x1:x2]
    region[sub] = pix[sub]
    return frame
