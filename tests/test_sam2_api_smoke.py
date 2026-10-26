"""Exercises the REAL SAM 2 code path (build, init_state, add_new_points_or_box, add_new_mask,
propagate_in_video, reset_state) with RANDOMLY INITIALISED weights.

This proves the call signatures/tensor shapes used in src/tracking.py match the installed sam2 package.
It proves NOTHING about tracking quality - that needs the official checkpoint (see models/README.md).
"""
import os

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

from helpers import ColorTracker, H, W, draw_frame, make_video  # noqa: E402
from src.pipeline import Config, Pipeline  # noqa: E402
from src.points_io import save_points  # noqa: E402
from src.tracking import Init, SAM2Tracker, unpack_mask  # noqa: E402
from src.video_io import probe_video  # noqa: E402

CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@pytest.fixture(scope="module")
def tracker():
    return SAM2Tracker(None, CFG, device="cuda", _random_weights_api_test=True)


def test_track_chunk_real_api(tracker, tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(9):
        cv2.imwrite(str(d / f"{i}.jpg"), cv2.resize(draw_frame(30 + i), (320, 180)))
    carry = np.zeros((180, 320), bool)
    carry[70:110, 100:140] = True
    inits = [Init(1, 0, mask=carry, source="carry"),
             Init(2, 3, points=[(150.0, 90.0), (10.0, 10.0)], labels=[1, 0])]
    out = tracker.track_chunk(str(d), 9, inits)
    assert sorted(out) == [1, 2]
    assert sorted(out[1]) == list(range(9))            # carried object: every frame
    assert sorted(out[2]) == list(range(3, 9))         # prompted at local frame 3: nothing before it
    m = unpack_mask(out[1][0], (180, 320))
    assert m.shape == (180, 320) and m.dtype == bool
    d2 = tracker.describe()
    assert d2["device"] == "cuda" and "RTX" in d2["gpu"] or d2["gpu"]
    # state really released between chunks
    before = torch.cuda.memory_allocated()
    tracker.track_chunk(str(d), 9, inits)
    assert torch.cuda.memory_allocated() <= before + 8 * 2 ** 20


def test_pipeline_with_real_sam2_code_path(tracker, tmp_path):
    v = str(tmp_path / "v.mp4")
    make_video(v, n_frames=100)
    info = probe_video(v)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = Config(input=v, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", extra={"keep_temp": False})
    rc = Pipeline(cfg, tracker, None).run()
    assert rc in (0, 2)                                # random weights: review events are expected, crashes are not
    assert probe_video(cfg.output).nb_frames == 100
