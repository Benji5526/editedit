"""Real SAM 2.1 checkpoint on the synthetic clip (skipped when models/sam2.1_hiera_small.pt is absent).

Measures how well real SAM 2 follows the magenta square against the known ground truth, across chunk
boundaries and an occlusion. The square on a noisy background is an EASY target: this shows the
integration works with real weights, not how SAM 2 behaves on real footage.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sam2.1_hiera_small.pt")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isfile(CKPT)), reason="needs CUDA + checkpoint")

from helpers import ENTER, H, HIDDEN, SQ, W, square_x  # noqa: E402
from src.pipeline import Config, Pipeline  # noqa: E402
from src.points_io import save_points  # noqa: E402
from src.tracking import SAM2Tracker, unpack_mask  # noqa: E402
from src.video_io import FrameReader, probe_video  # noqa: E402


def gt_mask(t, shape):
    m = np.zeros((H, W), bool)
    x0 = square_x(t)
    xa, xb = max(0, x0), min(W, x0 + SQ)
    if ENTER <= t not in HIDDEN and xb > xa:
        m[150:150 + SQ, xa:xb] = True
    import cv2
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def test_real_sam2_tracks_square_across_chunks(video, tmp_path):
    info = probe_video(video)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 20, "x": square_x(20) + 30, "y": 180, "label": 1}])
    tracker = SAM2Tracker(CKPT, device="cuda")
    # capture the raw per-frame masks the pipeline gets from the tracker
    got = {}
    orig = tracker.track_chunk
    chunk_starts = []

    def spy(frame_dir, n, inits):
        out = orig(frame_dir, n, inits)
        chunk_starts.append((n, [(i.obj_id, i.local_frame, i.source) for i in inits]))
        for o, fr in out.items():
            for li, p in fr.items():
                got[(len(chunk_starts) - 1, li)] = unpack_mask(p, (180, 320))
        return out

    tracker.track_chunk = spy
    cfg = Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", fail_closed=True, extra={"keep_temp": False})
    rc = Pipeline(cfg, tracker, None).run()
    assert rc in (0, 2)
    ious = {}
    for (c, li), m in got.items():
        t = c * 40 + li - (1 if c > 0 else 0)
        if c > 0 and li == 0:
            continue                                    # overlap frame belongs to the previous chunk
        g = gt_mask(t, (180, 320))
        if g.any():
            ious[t] = (m & g).sum() / max(1, (m | g).sum())
    pre = [ious[t] for t in sorted(ious) if t < 70]
    post = [ious[t] for t in sorted(ious) if t >= 86]
    print(f"IoU before occlusion (frames 20-69, crosses chunk boundary at 40): mean={np.mean(pre):.3f} min={np.min(pre):.3f}")
    print(f"IoU after the 16-frame occlusion (frames 86+): mean={np.mean(post):.3f}")
    print("post-occlusion IoU by frame:", {t: round(float(ious[t]), 2) for t in sorted(ious) if t >= 86 and t % 4 == 2})
    review = json.load(open(str(tmp_path / "l" / "synthetic_review.json")))
    print("review summary:", review["summary"])
    print("recovery events:", [(e["frame"], e["severity"], e["detail"]) for e in review["events"] if e["type"] == "tracking_recovered"])
    # what the integration must achieve with real weights: follow the square across a chunk boundary ...
    assert len(pre) == 50 and np.mean(pre) > 0.8
    # ... and make the occlusion visible in the review log
    assert any(e["type"] == "tracking_lost_mid_frame" for e in review["events"])
    # KNOWN LIMITATION (documented in README): SAM 2 small does not re-find the object after a full occlusion
    # during which it moved. The pipeline cannot fix that; it must not report the later "recovery" as verified.
    for e in review["events"]:
        if e["type"] == "tracking_recovered":
            assert e["severity"] == "warn"


def test_reclick_after_occlusion_fixes_tracking(video, tmp_path):
    """The review workflow: the log flags the occlusion, the user clicks the object again on a later frame, re-run."""
    info = probe_video(video)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 20, "x": square_x(20) + 30, "y": 180, "label": 1},
                            {"obj_id": 1, "frame_idx": 90, "x": square_x(90) + 30, "y": 180, "label": 1}])
    tracker = SAM2Tracker(CKPT, device="cuda")
    got = {}
    orig = tracker.track_chunk
    count = [0]

    def spy(frame_dir, n, inits):
        out = orig(frame_dir, n, inits)
        c = count[0]
        count[0] += 1
        P = n - 40
        for o, fr in out.items():
            for li, p in fr.items():
                if li >= P:                                  # skip prefix/overlap frames
                    got[c * 40 + li - P] = unpack_mask(p, (180, 320))
        return out

    tracker.track_chunk = spy
    cfg = Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", fail_closed=True, extra={"keep_temp": False})
    assert Pipeline(cfg, tracker, None).run() in (0, 2)
    post = [t for t in range(90, 143) if t in got and gt_mask(t, (180, 320)).any()]
    ious = [(got[t] & gt_mask(t, (180, 320))).sum() / max(1, (got[t] | gt_mask(t, (180, 320))).sum()) for t in post]
    print(f"IoU after re-click at 90 (frames 90-142): mean={np.mean(ious):.3f} min={np.min(ious):.3f} n={len(ious)}")
    assert len(ious) > 40 and np.mean(ious) > 0.8
