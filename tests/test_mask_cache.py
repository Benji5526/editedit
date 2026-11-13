"""Mask cache: the analysis results of every chunk are kept next to its segment, so changing a render-only setting (mosaic style,
strength, padding, encoder) re-renders from the cache without running the tracker or the detector again. The result must equal a
from-scratch run with the new settings (colour-threshold tracker stand-in, fake detector)."""
import json
import os

import numpy as np

from helpers import ColorTracker
from src.detection import Det
from src.pipeline import RENDER_ONLY, Config, Pipeline, config_hash, load_chunk_cache, plan_reprocess, render_hash
from src.points_io import save_points
from src.tracking import unpack_mask
from src.video_io import FrameReader, probe_video

CS = 40
P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}
P90 = {"obj_id": 1, "frame_idx": 90, "x": 400, "y": 180, "label": 1}
NEW = dict(block=32, mosaic_style="solid", mask_pad_px=20, det_expand=0.5, fail_closed_expand=0.5, crf=20)


class FakeDetector:
    """Reports one fixed box on every third frame (so gap filling has work to do). Counts its calls. The frame number is read from
    the code drawn into the synthetic footage, so the result depends on the frame only, not on how often the detector was called."""

    def __init__(self):
        self.calls = 0

    def detect(self, bgr):
        self.calls += 1
        k = bgr.shape[1] / 640.0                                        # analysis frames are scaled down
        t = sum(1 << b for b in range(12) if bgr[int(5 * k):int(13 * k), int((5 + b * 10) * k):int((13 + b * 10) * k)].mean() > 127)
        return [Det(200.0, 20.0, 240.0, 50.0, 0.9, "FAKE")] if t % 3 == 0 else []


def mk_points(video, tmp, name, pts):
    p = str(tmp / name)
    save_points(p, probe_video(video), pts)
    return p


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=CS, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, fail_closed=True,
                extra={"keep_temp": True})
    base.update(kw)
    return Config(**base)


def run(cfg, **kw):
    tr, det, evs = ColorTracker((180, 320)), FakeDetector(), []
    rc = Pipeline(cfg, tr, det, progress=evs.append, **kw).run()
    return rc, tr, det, evs


def review(cfg):
    return json.load(open(os.path.join(cfg.log_dir, "synthetic_review.json"), encoding="utf-8"))


def event_keys(doc):
    return sorted((e["frame"], e["type"], e["object_id"], e["action"], e["detail"]) for e in doc["events"])


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def max_diff(a, b):
    assert len(a) == len(b) == 160
    return max(float(np.abs(x.astype(int) - y.astype(int)).mean()) for x, y in zip(a, b))


def job_dir(cfg):
    return os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])


def detections_csv(cfg):
    return open(os.path.join(cfg.log_dir, "synthetic_detections.csv"), encoding="utf-8-sig").read()


def scratch(tmp_path, video, pts, **kw):
    d = tmp_path / "scratch"
    d.mkdir()
    cfg = cfg_for(d, video, mk_points(video, d, "p.json", pts), **kw)
    rc = run(cfg)[0]
    return rc, cfg


# ------------------------------------------------------------------ hashes
def test_render_only_settings_do_not_change_the_analysis_hash(video, tmp_path):
    info = probe_video(video)
    base = cfg_for(tmp_path, video, None)
    for key, value in dict(NEW, temporal_radius=0, det_pad_px=3.0, fail_closed_mode="full", encoder="h264_nvenc", preset="fast").items():
        assert key in RENDER_ONLY
        other = cfg_for(tmp_path, video, None, **{key: value})
        assert config_hash(other, info) == config_hash(base, info), key
        assert render_hash(other) != render_hash(base), key
    for key, value in dict(jump_ratio=3.0, det_threshold=0.5, fail_closed=False, analysis_width=160, chunk_size=80, det_hold=9).items():
        other = cfg_for(tmp_path, video, None, **{key: value})
        assert config_hash(other, info) != config_hash(base, info), key
        assert render_hash(other) == render_hash(base), key


# ------------------------------------------------------------------ cache content
def test_cache_holds_the_masks_and_detections_of_every_chunk(video, tmp_path):
    cfg = cfg_for(tmp_path, video, mk_points(video, tmp_path, "p.json", [P12]))
    assert run(cfg)[0] in (0, 2)
    jd = job_dir(cfg)
    assert sorted(n for n in os.listdir(jd) if n.startswith("masks_")) == [f"masks_{i:05d}.npz" for i in range(4)]
    c = load_chunk_cache(jd, 0)
    meta = c["meta"]
    assert (meta["start"], meta["end"], meta["shape"], meta["obj_ids"]) == (0, 40, [180, 320], [1])
    assert c["packed"][1][5] is None                                   # before the click at frame 12: no mask
    m = unpack_mask(c["packed"][1][30], (180, 320))                    # square is 60 px at 640 wide -> 30 px at analysis size
    assert 600 < int(m.sum()) < 1100
    assert len(meta["filled"]) == 40 and any(d["cls"] == "FAKE" for ds in meta["filled"] for d in ds)
    assert any(d["filled"] for ds in meta["filled"] for d in ds)       # gap-filled boxes are cached too
    lost = load_chunk_cache(jd, 1)["meta"]                             # object hidden from frame 70: fail-closed regions are cached
    assert any(lost["fc_regions"][i] for i in range(30, 40)) and any(e["type"] == "tracking_lost_mid_frame" for e in lost["events"])
    assert not any(e["type"] == "fail_closed_mosaic_applied" for e in lost["events"])   # render-stage events are not part of the cache


# ------------------------------------------------------------------ render-only re-run
def test_changed_mosaic_settings_rerender_without_tracking_or_detection(video, tmp_path):
    (tmp_path / "inc").mkdir()
    p = mk_points(video, tmp_path / "inc", "p.json", [P12])
    cfg = cfg_for(tmp_path / "inc", video, p)
    rc0, tr0, det0, _ = run(cfg)
    assert rc0 in (0, 2) and tr0.n_calls == 4 and det0.calls == 160
    old_frames = frames(cfg.output)
    state_mtimes = {n: os.path.getmtime(os.path.join(job_dir(cfg), n)) for n in os.listdir(job_dir(cfg)) if n.startswith(("state_", "masks_"))}

    cfg2 = cfg_for(tmp_path / "inc", video, p, resume=True, **NEW)
    plan = plan_reprocess(cfg2)
    assert plan["can_resume"] and plan["reusable_chunks"] == 0 and plan["render_only_chunks"] == 4 and plan["frames_render_only"] == 160
    assert "only their mosaic is redone" in plan["reason"]
    rc, tr, det, evs = run(cfg2)
    assert rc == rc0 and tr.n_calls == 0 and det.calls == 0            # neither model ran
    assert evs[0]["reused_chunks"] == 0 and evs[0]["rerender_chunks"] == 4 and evs[0]["points_changed"] is False
    assert [e["timing"]["from_mask_cache"] for e in evs if e["event"] == "chunk_done"] == [True] * 4
    for n, t in state_mtimes.items():                                  # tracking state and cache were not rewritten
        assert os.path.getmtime(os.path.join(job_dir(cfg2), n)) == t
    doc = review(cfg2)
    assert doc["meta"]["rerendered_chunks"] == 4 and doc["meta"]["reused_chunks"] == 0 and doc["meta"]["config"]["block"] == 32
    new_frames = frames(cfg2.output)
    assert max_diff(old_frames, new_frames) > 0.5                      # the picture did change

    rc_ref, cfg_ref = scratch(tmp_path, video, [P12], **NEW)
    assert rc_ref == rc
    assert event_keys(review(cfg_ref)) == event_keys(doc)
    assert detections_csv(cfg_ref) == detections_csv(cfg2)
    assert max_diff(frames(cfg_ref.output), new_frames) < 0.5

    rc, tr, det, evs = run(cfg_for(tmp_path / "inc", video, p, resume=True, **NEW))     # same settings again: nothing to do
    assert rc == rc0 and tr.n_calls == 0 and evs[0]["reused_chunks"] == 4 and evs[0]["rerender_chunks"] == 0


def test_changed_settings_and_changed_points_together(video, tmp_path):
    (tmp_path / "inc").mkdir()
    cfg = cfg_for(tmp_path / "inc", video, mk_points(video, tmp_path / "inc", "old.json", [P12]))
    assert run(cfg)[0] in (0, 2)
    cfg2 = cfg_for(tmp_path / "inc", video, mk_points(video, tmp_path / "inc", "new.json", [P12, P90]), resume=True, **NEW)
    plan = plan_reprocess(cfg2)
    assert plan["render_only_chunks"] == 2 and plan["reusable_chunks"] == 0 and plan["first_changed_chunk"] == 2
    rc, tr, det, evs = run(cfg2)
    assert rc in (0, 2) and tr.n_calls == 2 and det.calls == 80        # chunks 0-1 from the cache, chunks 2-3 in full
    assert evs[0]["rerender_chunks"] == 2 and evs[0]["reused_chunks"] == 0 and evs[0]["points_changed"] is True
    rc_ref, cfg_ref = scratch(tmp_path, video, [P12, P90], **NEW)
    assert rc_ref == rc and event_keys(review(cfg_ref)) == event_keys(review(cfg2))
    assert max_diff(frames(cfg_ref.output), frames(cfg2.output)) < 0.5


def test_object_clicked_later_does_not_break_cached_chunks(video, tmp_path):
    (tmp_path / "inc").mkdir()
    cfg = cfg_for(tmp_path / "inc", video, mk_points(video, tmp_path / "inc", "old.json", [P12]))
    assert run(cfg)[0] in (0, 2)
    pts = [P12, {"obj_id": 2, "frame_idx": 130, "x": 300, "y": 180, "label": 1}]
    rc, tr, _, evs = run(cfg_for(tmp_path / "inc", video, mk_points(video, tmp_path / "inc", "new.json", pts), resume=True, block=32))
    assert rc in (0, 2) and tr.n_calls == 1 and evs[0]["rerender_chunks"] == 3


def test_interrupted_rerender_continues_where_it_stopped(video, tmp_path):
    (tmp_path / "inc").mkdir()
    p = mk_points(video, tmp_path / "inc", "p.json", [P12])
    assert run(cfg_for(tmp_path / "inc", video, p))[0] in (0, 2)
    done = []

    def cancel_after_two():
        return len(done) >= 2

    tr, det = ColorTracker((180, 320)), FakeDetector()
    rc = Pipeline(cfg_for(tmp_path / "inc", video, p, resume=True, **NEW), tr, det,
                  progress=lambda ev: done.append(ev) if ev["event"] == "chunk_done" else None, cancel_check=cancel_after_two).run()
    assert rc == 3 and tr.n_calls == 0 and 2 <= len(done) < 4

    rc, tr, det, evs = run(cfg_for(tmp_path / "inc", video, p, resume=True, **NEW))
    assert rc in (0, 2) and tr.n_calls == 0 and det.calls == 0
    assert evs[0]["reused_chunks"] == len(done) and evs[0]["rerender_chunks"] == 4 - len(done)
    rc_ref, cfg_ref = scratch(tmp_path, video, [P12], **NEW)
    assert max_diff(frames(cfg_ref.output), frames(str(tmp_path / "inc" / "out.mp4"))) < 0.5


def test_going_back_to_the_old_settings_rerenders_again(video, tmp_path):
    p = mk_points(video, tmp_path, "p.json", [P12])
    assert run(cfg_for(tmp_path, video, p))[0] in (0, 2)
    first = frames(str(tmp_path / "out.mp4"))
    assert run(cfg_for(tmp_path, video, p, resume=True, block=48))[0] in (0, 2)
    rc, tr, _, evs = run(cfg_for(tmp_path, video, p, resume=True))
    assert rc in (0, 2) and tr.n_calls == 0 and evs[0]["rerender_chunks"] == 4
    assert max_diff(first, frames(str(tmp_path / "out.mp4"))) < 0.5


def test_unreadable_cache_fails_instead_of_guessing(video, tmp_path):
    p = mk_points(video, tmp_path, "p.json", [P12])
    cfg = cfg_for(tmp_path, video, p)
    assert run(cfg)[0] in (0, 2)
    with open(os.path.join(job_dir(cfg), "masks_00001.npz"), "wb") as f:
        f.write(b"not a cache")
    rc = run(cfg_for(tmp_path, video, p, resume=True, block=32))[0]
    assert rc == 1
    status = json.load(open(os.path.join(cfg.log_dir, "synthetic_status.json"), encoding="utf-8"))
    assert "mask cache of chunk 1" in status["reason"]


def test_missing_cache_file_falls_back_to_full_processing_from_there(video, tmp_path):
    p = mk_points(video, tmp_path, "p.json", [P12])
    cfg = cfg_for(tmp_path, video, p)
    assert run(cfg)[0] in (0, 2)
    os.remove(os.path.join(job_dir(cfg), "masks_00002.npz"))
    rc, tr, _, evs = run(cfg_for(tmp_path, video, p, resume=True, block=32))
    assert rc in (0, 2) and evs[0]["rerender_chunks"] == 2 and tr.n_calls == 2
