"""Widget <-> video-frame coordinate mapping. Pure functions (no Qt), unit-tested.

The frame is drawn centred and uniformly scaled inside the widget (letterboxed). Clicks in the
letterbox bars are not on the image and are rejected instead of being clamped onto the border, so a
stray click can never be saved as a point on the video.
Coordinates are in Qt logical pixels on both sides; frame coordinates are full-resolution decoded pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Fit:
    scale: float      # widget px per frame px
    ox: float         # left edge of the drawn image in the widget
    oy: float         # top edge
    w: float          # drawn size
    h: float


def fit_rect(widget_w: float, widget_h: float, frame_w: int, frame_h: int) -> Optional[Fit]:
    if widget_w <= 0 or widget_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    scale = min(widget_w / frame_w, widget_h / frame_h)
    w, h = frame_w * scale, frame_h * scale
    return Fit(scale, (widget_w - w) / 2.0, (widget_h - h) / 2.0, w, h)


def widget_to_frame(fit: Optional[Fit], x: float, y: float, frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
    """Frame coordinates of a widget position, or None when it is outside the drawn image."""
    if fit is None:
        return None
    if not (fit.ox <= x < fit.ox + fit.w and fit.oy <= y < fit.oy + fit.h):
        return None
    fx = (x - fit.ox) / fit.scale
    fy = (y - fit.oy) / fit.scale
    return min(max(fx, 0.0), frame_w - 1e-6), min(max(fy, 0.0), frame_h - 1e-6)


def frame_to_widget(fit: Optional[Fit], fx: float, fy: float) -> Optional[Tuple[float, float]]:
    if fit is None:
        return None
    return fit.ox + fx * fit.scale, fit.oy + fy * fit.scale
