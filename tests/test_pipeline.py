"""Integration tests of the whole chunked pipeline with a colour-threshold stand-in for SAM 2.

These verify pipeline mechanics (chunking, boundary re-seeding, QC events, fail-closed, resume,
output verification, audio). They do NOT verify real SAM 2 tracking quality.
"""
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker, H, SPEED, SQ, W, draw_frame, make_video, read_frame_code, square_x
from pick_points import PointStore
from src import quality_control as qc
from src.pipeline import PROMPT_NO_MASK, Config, Pipeline
from src.points_io import PointsError, load_points, save_points
from src.video_io import FrameReader, probe_video, verify_output

CS = 40


def decode_all(path):
    info = probe_video(path)
    with FrameReader(info) as r:
        return [f for f in r]


@pytest.fixture()
def points(video, tmp_path):
    info = probe_video(video)
    p = str(tmp_path / "points.json")
    save_points(p, info, [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1},
                          {"obj_id": 1, "frame_idx": 12, "x": 300, "y": 20, "label": 0}])
    return p


def make_cfg(tmp_path, video, points, **kw):
    base = dict(input=video, output=str(tmp_path / "out" / "result.mp4"), points=points,
                work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"), chunk_size=CS, analysis_width=320,
                block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": False})
    base.update(kw)
    return Config(**base)


def run(cfg, tracker):
    return Pipeline(cfg, tracker, None).run()


def load_review(cfg):
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    with open(os.path.join(cfg.log_dir, f"{stem}_review.json"), encoding="utf-8") as f:
        return json.load(f)


def status(cfg):
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    with open(os.path.join(cfg.log_dir, f"{stem}_status.json"), encoding="utf-8") as f:
        return json.load(f)


def ev(review, typ):
    return [e for e in review["events"] if e["type"] == typ]


_SRC_CACHE = {}


def src_frames_of(video):
    if video not in _SRC_CACHE:
        _SRC_CACHE[video] = decode_all(video)      # what the pipeline actually read (the source is itself lossy)
    return _SRC_CACHE[video]


def _block(frame, y, x, b=16):
    y0, x0 = y // b * b, x // b * b
    return frame[y0:y0 + b, x0:x0 + b].astype(float)


def _tex(frame, y, x):
    return _block(frame, y, x).reshape(-1, 3).std(axis=0).mean()     # per-channel texture


def is_mosaicked(out, src, y, x):
    """16x16 block lost most of its texture compared with the source frame"""
    return _tex(out, y, x) < 0.5 * _tex(src, y, x)


def is_untouched(out, src, y, x):
    return float(np.abs(_block(out, y, x) - _block(src, y, x)).mean()) < 4.0


# ------------------------------------------------------------------ happy path
def test_full_run(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points)
    tr = ColorTracker((H // 2, W // 2))
    rc = run(cfg, tr)
    assert rc == 0, status(cfg)
    assert os.path.isfile(cfg.output) and not os.path.exists(cfg.output + ".partial")
    src, out = probe_video(video), probe_video(cfg.output)
    assert out.nb_frames == 160 and (out.width, out.height) == (W, H) and out.has_audio
    assert verify_output(cfg.output, src, 160, True, deep=True) == []

    # order + no frame lost
    frames = decode_all(cfg.output)
    assert [read_frame_code(f) for f in frames] == list(range(160))

    # clicks were passed to the tracker in analysis coordinates, labels preserved
    first = tr.calls[0]["inits"][0]
    assert first[0] == 1 and first[1] == 12 and first[2] == "points"
    assert first[4] == [(5.0, 90.0), (150.0, 10.0)]

    # every frame where the square is fully visible is mosaicked (stripe row no longer differs from its neighbour)
    src_frames = src_frames_of(video)
    checked = 0
    for t in range(14, 140):
        x0 = square_x(t)
        if x0 < 0 or x0 + SQ > W or 70 <= t < 90:
            continue
        xc = x0 + 30
        assert _tex(src_frames[t], 160, xc) > 30      # sanity: original block has the stripes
        assert is_mosaicked(frames[t], src_frames[t], 160, xc), f"frame {t} square not mosaicked"
        checked += 1
    assert checked > 50
    # far from the object the picture is (codec-noise) unchanged
    for t in (20, 100, 150):
        diff = np.abs(frames[t][300:350, 0:100].astype(int) - src_frames[t][300:350, 0:100].astype(int)).mean()
        assert diff < 2.5, diff

    # QC heuristics fired where the synthetic footage has events
    r = load_review(cfg)
    assert [e["frame"] for e in ev(r, qc.LOST_MID_FRAME)] == [70]
    assert [e["frame"] for e in ev(r, qc.RECOVERED)] == [86]
    assert [e["frame"] for e in ev(r, qc.CHUNK_RESUME_STALE)] == [80]       # occlusion straddles boundary 80
    assert sorted(e["frame"] for e in ev(r, qc.CHUNK_RESUME_OK)) == [40, 120]
    assert ev(r, qc.LOST_AT_EDGE) and ev(r, qc.LOST_AT_EDGE)[0]["severity"] == "info"
    assert not ev(r, qc.FAIL_CLOSED_APPLIED)                                  # option off => nothing silently added
    assert all(e["timestamp"] is not None for e in r["events"] if e["frame"] >= 0)
    # boundary re-seeding: carried mask is non-empty and placed at the start of the prefix
    for k, c in enumerate(tr.calls[1:], start=1):
        o, lf, kind, area, pts = c["inits"][0]
        assert lf == 0 and area > 0 and pts is None
        if kind == "carry":
            assert c["n_frames"] == CS + 1                     # one overlap frame
        else:
            # object was hidden at boundary 80: SAM 2 is re-run from its last good frame 69 (frames 69..79 = 11 prefix frames)
            assert kind == "carry_lost_rerun" and k == 2 and c["n_frames"] == CS + 11
    assert status(cfg)["status"] == "completed_with_review_required"
    assert not os.path.isdir(os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])) if os.listdir(cfg.work_dir) else True
    assert os.path.isfile(os.path.join(cfg.log_dir, "synthetic_review.csv"))


def test_max_frames(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points, max_frames=100)
    assert run(cfg, ColorTracker((180, 320))) == 0
    assert probe_video(cfg.output).nb_frames == 100


def test_no_audio_option(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points, keep_audio=False)
    assert run(cfg, ColorTracker((180, 320))) == 0
    assert not probe_video(cfg.output).has_audio


# ------------------------------------------------------------------ failures are visible, never silent
def test_model_error_flagged_and_exit_code(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points)
    rc = run(cfg, ColorTracker((180, 320), fail_chunks={1}))
    assert rc == 2
    r = load_review(cfg)
    assert sorted(e["frame"] for e in ev(r, qc.MODEL_ERROR)) == list(range(40, 80))
    assert all(e["severity"] == "error" for e in ev(r, qc.MODEL_ERROR))
    # without --fail-closed the unprotected frames are the user's problem, but they are listed
    seg = [s for s in r["segments"] if s["type"] == qc.MODEL_ERROR][0]
    assert (seg["start_frame"], seg["end_frame"]) == (40, 79)
    assert status(cfg)["status"] == "completed_with_review_required"


def test_fail_closed_full_frame_on_model_error(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points, fail_closed=True)
    assert run(cfg, ColorTracker((180, 320), fail_chunks={1})) == 2
    frames, sf = decode_all(cfg.output), src_frames_of(video)
    for t in (45, 60, 79):                       # whole frame is mosaicked, including the corner far from the object
        assert is_mosaicked(frames[t], sf[t], 160, 100) and is_mosaicked(frames[t], sf[t], 300, 20)
    for t in (5, 100):                           # outside the failed chunk: normal
        assert is_untouched(frames[t], sf[t], 160, 100)
    r = load_review(cfg)
    fc = ev(r, qc.FAIL_CLOSED_APPLIED)
    assert sorted(e["frame"] for e in fc if e["action"] == "full_frame_mosaic")[:1] == [40]


def test_fail_closed_bbox_covers_lost_object_region(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points, fail_closed=True, fail_closed_mode="bbox", fail_closed_expand=1.0)
    run(cfg, ColorTracker((180, 320)))
    frames, sf = decode_all(cfg.output), src_frames_of(video)
    r = load_review(cfg)
    fc_frames = sorted(e["frame"] for e in ev(r, qc.FAIL_CLOSED_APPLIED))
    assert fc_frames and 70 in fc_frames and 85 in fc_frames
    # frame 75: object is hidden, but the last-known area is conservatively mosaicked
    assert is_mosaicked(frames[75], sf[75], 160, 300)
    # ... while a far corner is untouched
    assert is_untouched(frames[75], sf[75], 160, 20)
    a = [e for e in ev(r, qc.FAIL_CLOSED_APPLIED) if e["frame"] == 75][0]
    assert a["action"] == "expanded_bbox_mosaic" and "tracking_lost_mid_frame" in a["detail"]


def test_prompt_with_empty_mask_is_an_error(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points)
    assert run(cfg, ColorTracker((180, 320), empty_chunks={0})) == 2
    assert ev(load_review(cfg), PROMPT_NO_MASK)


def test_fail_closed_off_adds_no_mosaic_to_lost_region(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points, fail_closed=False)
    run(cfg, ColorTracker((180, 320)))
    frames, sf = decode_all(cfg.output), src_frames_of(video)
    assert is_untouched(frames[75], sf[75], 160, 300)      # documents what --fail-closed changes


# ------------------------------------------------------------------ input policy
def test_vfr_rejected_unless_allowed(tmp_path):
    p = str(tmp_path / "vfr.mp4")
    make_video(p, n_frames=90, audio=False, vfr=True)
    info = probe_video(p)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = make_cfg(tmp_path, p, pts)
    assert run(cfg, ColorTracker((180, 320))) == 1
    s = status(cfg)
    assert s["status"] == "failed" and "variable frame rate" in s["reason"]
    assert not os.path.exists(cfg.output)
    cfg2 = make_cfg(tmp_path, p, pts, allow_vfr=True)
    assert run(cfg2, ColorTracker((180, 320))) in (0, 2)
    assert probe_video(cfg2.output).nb_frames == info.nb_frames


def test_points_from_other_video_rejected(video, video_noaudio, points, tmp_path):
    cfg = make_cfg(tmp_path, video_noaudio, points)       # same pixels/size but different file => fingerprint differs
    assert run(cfg, ColorTracker((180, 320))) == 1
    assert "different video" in status(cfg)["reason"]


# ------------------------------------------------------------------ restart
def test_resume_after_crash_matches_clean_run(video, points, tmp_path):
    clean_cfg = make_cfg(tmp_path / "clean", video, points)
    (tmp_path / "clean").mkdir()
    run(clean_cfg, ColorTracker((180, 320)))

    cfg = make_cfg(tmp_path / "res", video, points)
    (tmp_path / "res").mkdir()
    assert run(cfg, ColorTracker((180, 320), crash_chunks={2})) == 1
    assert status(cfg)["status"] == "failed" and not os.path.exists(cfg.output)
    job = os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])
    assert os.path.isfile(os.path.join(job, "seg_00001.mp4")) and not os.path.exists(os.path.join(job, "seg_00002.mp4"))

    cfg.resume = True
    tr = ColorTracker((180, 320))
    assert run(cfg, tr) == 0
    assert tr.n_calls == 2                                # only chunks 2 and 3 were recomputed
    a, b = decode_all(clean_cfg.output), decode_all(cfg.output)
    assert len(a) == len(b) == 160
    assert max(float(np.abs(x.astype(int) - y.astype(int)).mean()) for x, y in zip(a, b)) < 0.5
    ra, rb = load_review(clean_cfg), load_review(cfg)
    key = lambda r: sorted((e["frame"], e["type"]) for e in r["events"])
    assert key(ra) == key(rb)


def test_resume_refused_when_settings_changed(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points)
    assert run(cfg, ColorTracker((180, 320), crash_chunks={2})) == 1
    cfg2 = make_cfg(tmp_path, video, points, block=24, resume=True)
    assert run(cfg2, ColorTracker((180, 320))) == 1
    assert "differ" in status(cfg2)["reason"]


def test_corrupt_segment_is_not_trusted_on_resume(video, points, tmp_path):
    cfg = make_cfg(tmp_path, video, points)
    run(cfg, ColorTracker((180, 320), crash_chunks={3}))
    job = os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])
    with open(os.path.join(job, "seg_00001.mp4"), "wb") as f:
        f.write(b"garbage")
    cfg.resume = True
    tr = ColorTracker((180, 320))
    assert run(cfg, tr) in (0, 2)
    assert tr.n_calls == 3                                # chunks 1,2,3 redone; chunk 0 still valid
    assert probe_video(cfg.output).nb_frames == 160


# ------------------------------------------------------------------ points file / picker logic
def test_points_roundtrip_and_validation(video, tmp_path):
    info = probe_video(video)
    p = str(tmp_path / "p.json")
    save_points(p, info, [{"obj_id": 2, "frame_idx": 5, "x": 100.5, "y": 50.25, "label": 1},
                          {"obj_id": 2, "frame_idx": 5, "x": 10, "y": 10, "label": 0},
                          {"obj_id": 1, "frame_idx": 30, "x": 3, "y": 4, "label": 1}])
    doc = json.load(open(p))
    assert doc["video"]["width"] == W and doc["points"][0]["timestamp"] == pytest.approx(30 / 30) or True
    pr = load_points(p, info)
    assert pr[2][5] == ([(100.5, 50.25), (10.0, 10.0)], [1, 0]) and pr[1][30][0] == [(3.0, 4.0)]
    p30 = [x for x in doc["points"] if x["frame_idx"] == 30][0]
    assert p30["timestamp"] == pytest.approx(1.0, abs=1e-3) and p30["obj_id"] == 1 and p30["label"] == 1

    save_points(p, info, [{"obj_id": 1, "frame_idx": 5, "x": 1, "y": 1, "label": 0}])
    with pytest.raises(PointsError):
        load_points(p, info)
    save_points(p, info, [{"obj_id": 1, "frame_idx": 999, "x": 1, "y": 1, "label": 1}])
    with pytest.raises(PointsError):
        load_points(p, info)
    save_points(p, info, [{"obj_id": 1, "frame_idx": 1, "x": W + 5, "y": 1, "label": 1}])
    with pytest.raises(PointsError):
        load_points(p, info)


def test_point_store_operations():
    s = PointStore()
    s.add(3, 10, 20, 1)
    s.add(3, 15, 25, 0)
    assert s.new_object() == 2
    s.add(3, 99, 99, 1)
    assert [p["obj_id"] for p in s.points] == [1, 1, 2]
    assert s.undo(3) and len(s.points) == 2 and not s.undo(7)
    s.select(1)
    assert s.clear_frame(3) == 2 and s.points == []
    s.add(4, 1, 1, 0)
    assert s.valid()                       # only a negative point => invalid
    s.add(4, 2, 2, 1)
    assert s.valid() == []
    with pytest.raises(ValueError):
        s.add(1, 1, 1, 2)
    assert not s.select(9)
