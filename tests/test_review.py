import json

import numpy as np
import pytest

from helpers import ColorTracker, H, W
from pick_points import Picker
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.review import ReviewError, ReviewNavigator, load_review
from src.video_io import probe_video

SEGS = [
    {"type": "tracking_lost_mid_frame", "object_id": 1, "severity": "warn", "start_frame": 70, "end_frame": 70},
    {"type": "tracking_recovered", "object_id": 1, "severity": "warn", "start_frame": 86, "end_frame": 86},
    {"type": "chunk_boundary_resumed", "object_id": 1, "severity": "info", "start_frame": 40, "end_frame": 40},
    {"type": "fail_closed_mosaic_applied", "object_id": None, "severity": "info", "start_frame": 70, "end_frame": 85},
    {"type": "output_verification_failed", "object_id": None, "severity": "error", "start_frame": -1, "end_frame": -1},
    {"type": "model_error", "object_id": None, "severity": "error", "start_frame": 120, "end_frame": 130},
]


def test_navigation_default_skips_info_and_negative_frames():
    n = ReviewNavigator(SEGS)
    assert n.frames == [70, 86, 120, 130] and len(n) == 4
    assert n.next_after(0) == 70 and n.next_after(70) == 86 and n.next_after(130) is None
    assert n.prev_before(86) == 70 and n.prev_before(70) is None and n.prev_before(1000) == 130
    assert n.index_of(69) == 0 and n.index_of(70) == 1 and n.index_of(100) == 2


def test_min_severity_info_includes_everything_real():
    n = ReviewNavigator(SEGS, "info")
    assert n.frames == [40, 70, 85, 86, 120, 130]
    assert ReviewNavigator(SEGS, "error").frames == [120, 130]


def test_object_and_description_at_frame():
    n = ReviewNavigator(SEGS, "info")
    assert n.object_at(70) == 1 and n.object_at(100) is None and n.object_at(125) is None
    d = n.describe(70)
    assert d[0].startswith("[warn] tracking_lost_mid_frame obj 1") and any("fail_closed" in x for x in d)


def test_load_review_rejects_other_video(tmp_path, video):
    info = probe_video(video)
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"meta": {"input": {"fingerprint": "deadbeef"}}, "segments": []}))
    with pytest.raises(ReviewError):
        load_review(str(p), info)
    p.write_text(json.dumps({"meta": {}}))
    with pytest.raises(ReviewError):
        load_review(str(p), info)


def test_review_from_real_pipeline_log_and_picker_overlay(video, tmp_path):
    info = probe_video(video)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", extra={"keep_temp": False})
    Pipeline(cfg, ColorTracker((180, 320)), None).run()
    nav = load_review(str(tmp_path / "l" / "synthetic_review.json"), info)      # fingerprint comes from the pipeline's meta
    assert 70 in nav.frames and 86 in nav.frames and nav.object_at(70) == 1
    pk = Picker(info, str(tmp_path / "pp.json"), 320, nav)
    pk.frame_idx = 70
    img = pk.render()                                                          # draws the review banner without a window
    assert img.shape == (int(H * pk.scale), int(W * pk.scale), 3)
    plain = Picker(info, str(tmp_path / "pp.json"), 320).render()
    assert not np.array_equal(img[:60], plain[:60])
