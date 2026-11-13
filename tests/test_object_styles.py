"""Per-object mosaic settings (style, strength, padding): stored in the points file (version 2), applied as separate layers by the
render stage, and render-only - changing them re-renders from the mask cache without tracking again."""
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker
from src.detection import Det
from src.mosaic import apply_layers, apply_mosaic, layer_order
from src.pipeline import Config, Pipeline, plan_reprocess, render_hash
from src.points_io import PointStore, PointsError, load_object_styles, load_points, normalize_styles, save_points
from src.video_io import FrameReader, probe_video

P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}
SQUARE = (slice(165, 195), slice(85, 115))       # inside the square at frame 30 (x 70-130, y 150-210)
BOX = (slice(50, 90), slice(410, 470))           # inside the fake detection box (analysis 200-240 x 20-50 -> full size x2)


class BoxDetector:
    def detect(self, bgr):
        return [Det(200.0, 20.0, 240.0, 50.0, 0.9, "FAKE")]


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": True})
    base.update(kw)
    return Config(**base)


def run(cfg, detector=None):
    tr, evs = ColorTracker((180, 320)), []
    rc = Pipeline(cfg, tr, detector, progress=evs.append).run()
    return rc, tr, evs


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def max_diff(a, b):
    assert len(a) == len(b) == 160
    return max(float(np.abs(x.astype(int) - y.astype(int)).mean()) for x, y in zip(a, b))


def points_file(video, tmp, name, styles=None, pts=(P12,)):
    p = str(tmp / name)
    save_points(p, probe_video(video), list(pts), styles=styles)
    return p


# ------------------------------------------------------------------ points file
def test_points_file_keeps_version_1_without_styles_and_round_trips_with_them(video, tmp_path):
    info = probe_video(video)
    plain = points_file(video, tmp_path, "plain.json")
    assert json.load(open(plain))["version"] == 1 and load_object_styles(plain) == {}
    assert json.load(open(points_file(video, tmp_path, "empty.json", styles={1: {}})))["version"] == 1

    styled = points_file(video, tmp_path, "styled.json", styles={1: {"style": "solid", "block": 48, "pad_px": 30}, 7: {"style": "blur"}})
    doc = json.load(open(styled))
    assert doc["version"] == 2 and doc["object_styles"] == {"1": {"style": "solid", "block": 48, "pad_px": 30.0}}   # object 7 has no points
    assert load_object_styles(styled) == {1: {"style": "solid", "block": 48, "pad_px": 30.0}}
    assert load_points(styled, info) == load_points(plain, info)                    # the clicks are the same


@pytest.mark.parametrize("bad", [{"style": "sparkle"}, {"block": 1}, {"block": 999}, {"pad_px": -3}, {"colour": "red"}, "solid"])
def test_bad_object_settings_are_rejected(video, tmp_path, bad):
    with pytest.raises(PointsError):
        normalize_styles({1: bad})
    p = points_file(video, tmp_path, "p.json")
    doc = json.load(open(p))
    doc.update(version=2, object_styles={"1": bad})
    json.dump(doc, open(p, "w"))
    with pytest.raises(PointsError):
        load_points(p, probe_video(video))                                          # refused before any processing


def test_point_store_settings():
    s = PointStore()
    assert s.set_style(1, "style", "blur") and s.styles == {1: {"style": "blur"}}
    assert not s.set_style(1, "style", "blur")                                      # no change
    assert s.set_style(1, "block", 40) and s.set_style(1, "style", None) and s.styles == {1: {"block": 40}}
    assert s.set_style(1, "block", None) and s.styles == {}
    with pytest.raises(PointsError):
        s.set_style(1, "block", 1000)
    s.set_style(1, "style", "solid")
    s.add(0, 1.0, 1.0, 1)
    s.delete_object()
    assert s.styles == {}


# ------------------------------------------------------------------ layers
def test_layers_strongest_wins_and_single_layer_equals_apply_mosaic():
    assert layer_order({("solid", 8): 1, ("pixelate", 64): 1, ("blur", 99): 1, ("pixelate", 16): 1}) == \
        [("blur", 99), ("pixelate", 16), ("pixelate", 64), ("solid", 8)]
    rng = np.random.default_rng(1)
    img = rng.integers(60, 255, (120, 160, 3), dtype=np.uint8)
    a, b = np.zeros((120, 160), bool), np.zeros((120, 160), bool)
    a[20:80, 20:100] = True
    b[50:110, 70:150] = True
    out = apply_layers(img.copy(), {("solid", 16): a, ("blur", 16): b})
    assert out[a].max() == 0                                                        # solid wins the overlap, whatever the dict order
    out2 = apply_layers(img.copy(), {("blur", 16): b, ("solid", 16): a})
    assert np.array_equal(out, out2)
    only_b = b & ~a
    assert not np.array_equal(out[only_b], img[only_b]) and out[only_b].max() > 0   # the rest of b is blurred, not black
    assert np.array_equal(out[~(a | b)], img[~(a | b)])                             # nothing outside the masks changes
    assert np.array_equal(apply_layers(img.copy(), {("pixelate", 8): a}), apply_mosaic(img.copy(), a, "pixelate", 8))


# ------------------------------------------------------------------ pipeline
def test_render_hash_depends_on_object_settings(video, tmp_path):
    cfg = cfg_for(tmp_path, video, None)
    assert render_hash(cfg, {}) == render_hash(cfg) == render_hash(cfg, None)       # kept chunks of unstyled jobs stay valid
    assert render_hash(cfg, {1: {"style": "solid"}}) != render_hash(cfg)
    assert render_hash(cfg, {1: {"style": "solid"}}) != render_hash(cfg, {2: {"style": "solid"}})


@pytest.mark.parametrize("own,job", [({"style": "solid"}, {"mosaic_style": "solid"}), ({"block": 40}, {"block": 40}),
                                     ({"pad_px": 30}, {"mask_pad_px": 30}),
                                     ({"style": "blur", "block": 24, "pad_px": 4}, {"mosaic_style": "blur", "block": 24, "mask_pad_px": 4})])
def test_object_setting_equals_the_same_job_setting_when_it_is_the_only_region(video, tmp_path, own, job):
    """One tracked object and no detector: giving the object its own setting must produce the same video as making that the job's."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ca = cfg_for(tmp_path / "a", video, points_file(video, tmp_path / "a", "p.json", styles={1: own}))
    cb = cfg_for(tmp_path / "b", video, points_file(video, tmp_path / "b", "p.json"), **job)
    assert run(ca)[0] == run(cb)[0]
    assert max_diff(frames(ca.output), frames(cb.output)) < 0.5
    meta = json.load(open(os.path.join(ca.log_dir, "synthetic_review.json"), encoding="utf-8"))["meta"]
    assert set(meta["object_styles"]["1"]) == set(own)


def test_object_and_detection_get_different_looks(video, tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "styled").mkdir()
    plain = cfg_for(tmp_path / "plain", video, points_file(video, tmp_path / "plain", "p.json"))
    styled = cfg_for(tmp_path / "styled", video, points_file(video, tmp_path / "styled", "p.json", styles={1: {"style": "solid"}}))
    assert run(plain, BoxDetector())[0] == run(styled, BoxDetector())[0]
    src, a, b = frames(video)[30], frames(plain.output)[30], frames(styled.output)[30]
    assert a[SQUARE].mean() > 60 and b[SQUARE].mean() < 25                           # the object: pixelated vs black
    assert float(np.abs(a[BOX].astype(int) - b[BOX].astype(int)).mean()) < 1.0       # the detection box looks the same in both ...
    assert float(np.abs(src[BOX].astype(int) - b[BOX].astype(int)).mean()) > 3.0     # ... and is still mosaicked
    assert b[BOX].mean() > 40


def test_changing_an_object_setting_rerenders_from_the_cache(video, tmp_path):
    (tmp_path / "inc").mkdir()
    cfg = cfg_for(tmp_path / "inc", video, points_file(video, tmp_path / "inc", "p.json"))
    rc0, tr0, _ = run(cfg)
    assert rc0 in (0, 2) and tr0.n_calls == 4
    p2 = points_file(video, tmp_path / "inc", "p2.json", styles={1: {"style": "solid", "pad_px": 20}})
    cfg2 = cfg_for(tmp_path / "inc", video, p2, resume=True)
    plan = plan_reprocess(cfg2)
    assert plan["can_resume"] and plan["render_only_chunks"] == 4 and plan["first_changed_chunk"] is None
    rc, tr, evs = run(cfg2)
    assert rc == rc0 and tr.n_calls == 0 and evs[0]["rerender_chunks"] == 4 and evs[0]["points_changed"] is False
    assert frames(cfg2.output)[30][SQUARE].mean() < 25

    (tmp_path / "scratch").mkdir()
    ref = cfg_for(tmp_path / "scratch", video, points_file(video, tmp_path / "scratch", "p.json", styles={1: {"style": "solid", "pad_px": 20}}))
    assert run(ref)[0] == rc
    assert max_diff(frames(ref.output), frames(cfg2.output)) < 0.5

    rc, tr, evs = run(cfg_for(tmp_path / "inc", video, p2, resume=True))            # same settings again: everything is reused
    assert tr.n_calls == 0 and evs[0]["reused_chunks"] == 4 and evs[0]["rerender_chunks"] == 0


def test_two_objects_with_their_own_settings(video, tmp_path):
    """Both objects resolve to the same square with the stand-in tracker, so the two layers overlap completely: the stronger wins."""
    pts = [P12, {"obj_id": 2, "frame_idx": 12, "x": 12, "y": 182, "label": 1}]
    cfg = cfg_for(tmp_path, video, points_file(video, tmp_path, "p.json", pts=pts, styles={1: {"style": "blur"}, 2: {"style": "solid"}}))
    assert run(cfg)[0] in (0, 2)
    assert frames(cfg.output)[30][SQUARE].mean() < 25
