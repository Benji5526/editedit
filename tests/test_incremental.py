"""Re-processing after the clicked points changed: which chunks are reused, which are invalidated, and that the
incremental result equals a from-scratch run with the same points (colour-threshold tracker stand-in)."""
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker
from src.pipeline import Config, Pipeline, first_changed_chunk, prompt_signatures
from src.points_io import save_points
from src.video_io import FrameReader, probe_video

CS = 40
P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}


def mk_points(video, tmp, name, pts):
    p = str(tmp / name)
    save_points(p, probe_video(video), pts)
    return p


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=CS, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18,
                extra={"keep_temp": True})
    base.update(kw)
    return Config(**base)


def run(cfg, tracker=None, **kw):
    tr = tracker or ColorTracker((180, 320))
    evs = []
    rc = Pipeline(cfg, tr, None, progress=evs.append, **kw).run()
    return rc, tr, evs


def review(cfg):
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    return json.load(open(os.path.join(cfg.log_dir, f"{stem}_review.json"), encoding="utf-8"))


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def event_keys(doc, below=None):
    return sorted((e["frame"], e["type"], e["object_id"]) for e in doc["events"] if below is None or 0 <= e["frame"] < below)


def job_dir(cfg):
    return os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])


# ------------------------------------------------------------------ pure logic
def test_prompt_signatures_and_first_changed_chunk():
    chunks = [(0, 40), (40, 80), (80, 120), (120, 160)]
    base = {1: {12: ([(10.0, 180.0)], [1])}}
    a = prompt_signatures(base, chunks)
    assert a[0] != a[1] and a[1] == a[2] == a[3]                     # only chunk 0 contains a click
    add90 = {1: {12: ([(10.0, 180.0)], [1]), 90: ([(400.0, 180.0)], [1])}}
    b = prompt_signatures(add90, chunks)
    assert first_changed_chunk(a, b) == 2
    assert first_changed_chunk(a, a) == 4
    moved = {1: {12: ([(10.5, 180.0)], [1])}}
    assert first_changed_chunk(a, prompt_signatures(moved, chunks)) == 0
    neg = {1: {12: ([(10.0, 180.0)], [1]), 50: ([(200.0, 180.0), (5.0, 5.0)], [1, 0])}}
    assert first_changed_chunk(a, prompt_signatures(neg, chunks)) == 1
    new_obj_late = {1: {12: ([(10.0, 180.0)], [1])}, 2: {130: ([(1.0, 1.0)], [1])}}
    assert first_changed_chunk(a, prompt_signatures(new_obj_late, chunks)) == 3
    assert first_changed_chunk(None, a) == 0 and first_changed_chunk([], a) == 0
    assert first_changed_chunk(a[:2], a) == 2                         # old job covered fewer chunks


# ------------------------------------------------------------------ pipeline
def test_unchanged_points_reuse_everything(video, tmp_path):
    p1 = mk_points(video, tmp_path, "p1.json", [P12])
    cfg = cfg_for(tmp_path, video, p1)
    assert run(cfg)[0] == 0
    base_events = event_keys(review(cfg))
    first_frames = frames(cfg.output)

    p2 = mk_points(video, tmp_path, "other_name.json", [P12])         # same content, different path
    cfg2 = cfg_for(tmp_path, video, p2, resume=True)
    rc, tr, evs = run(cfg2)
    assert rc == 0 and tr.n_calls == 0
    assert evs[0]["reused_chunks"] == 4 and evs[0]["points_changed"] is False
    assert event_keys(review(cfg2)) == base_events
    assert max(float(np.abs(a.astype(int) - b.astype(int)).mean()) for a, b in zip(first_frames, frames(cfg2.output))) < 0.01


@pytest.mark.parametrize("extra_clicks,expect_reused", [
    ([{"obj_id": 1, "frame_idx": 90, "x": 400, "y": 180, "label": 1}], 2),    # chunk 2: chunks 0,1 reused
    ([{"obj_id": 1, "frame_idx": 50, "x": 200, "y": 180, "label": 1},
      {"obj_id": 1, "frame_idx": 50, "x": 5, "y": 5, "label": 0}], 1),        # positive + negative click in chunk 1
    ([{"obj_id": 2, "frame_idx": 130, "x": 300, "y": 180, "label": 1}], 3),   # new object in the last chunk
    ([{"obj_id": 1, "frame_idx": 12, "x": 12, "y": 170, "label": 1}], 0),     # refinement in chunk 0: nothing reusable
])
def test_incremental_equals_from_scratch(video, tmp_path, extra_clicks, expect_reused):
    (tmp_path / "inc").mkdir()
    (tmp_path / "scratch").mkdir()
    p_old = mk_points(video, tmp_path / "inc", "old.json", [P12])
    cfg = cfg_for(tmp_path / "inc", video, p_old)
    assert run(cfg)[0] == 0
    seg_mtimes = {n: os.path.getmtime(os.path.join(job_dir(cfg), n)) for n in os.listdir(job_dir(cfg)) if n.startswith("seg_")}
    old_doc = review(cfg)

    new_pts = [P12, *extra_clicks]
    p_new = mk_points(video, tmp_path / "inc", "new.json", new_pts)
    cfg_inc = cfg_for(tmp_path / "inc", video, p_new, resume=True)
    rc, tr, evs = run(cfg_inc)
    assert rc in (0, 2)
    assert evs[0]["reused_chunks"] == expect_reused and evs[0]["points_changed"] is True
    assert tr.n_calls == 4 - expect_reused                              # only invalidated chunks hit the tracker
    for ci in range(expect_reused):                                      # reused segments were not rewritten
        n = f"seg_{ci:05d}.mp4"
        assert os.path.getmtime(os.path.join(job_dir(cfg_inc), n)) == seg_mtimes[n]
    inc_doc = review(cfg_inc)
    assert inc_doc["meta"]["reused_chunks"] == expect_reused
    assert inc_doc["meta"]["reprocessed_chunks"] == list(range(expect_reused, 4))
    assert inc_doc["meta"]["run_id"] != old_doc["meta"]["run_id"] and inc_doc["meta"]["points_sha"] != old_doc["meta"]["points_sha"]
    assert inc_doc["meta"]["output"]["size"] == os.path.getsize(cfg_inc.output)
    # events of reused chunks are exactly the old ones
    assert event_keys(inc_doc, below=expect_reused * CS) == event_keys(old_doc, below=expect_reused * CS)

    # ground truth: a clean run with the new points only
    p_ref = mk_points(video, tmp_path / "scratch", "new.json", new_pts)
    cfg_ref = cfg_for(tmp_path / "scratch", video, p_ref)
    rc_ref, _, _ = run(cfg_ref)
    assert rc_ref == rc
    assert event_keys(review(cfg_ref)) == event_keys(inc_doc)
    a, b = frames(cfg_ref.output), frames(cfg_inc.output)
    assert len(a) == len(b) == 160
    assert max(float(np.abs(x.astype(int) - y.astype(int)).mean()) for x, y in zip(a, b)) < 0.5


def test_changed_settings_are_still_refused(video, tmp_path):
    p1 = mk_points(video, tmp_path, "p1.json", [P12])
    assert run(cfg_for(tmp_path, video, p1))[0] == 0
    rc, _, _ = run(cfg_for(tmp_path, video, p1, block=24, resume=True))
    assert rc == 1
    status = json.load(open(os.path.join(str(tmp_path / "logs"), "synthetic_status.json")))
    assert "differ" in status["reason"]


def test_keep_temp_toggle_does_not_break_resume(video, tmp_path):
    p1 = mk_points(video, tmp_path, "p1.json", [P12])
    assert run(cfg_for(tmp_path, video, p1))[0] == 0
    rc, tr, _ = run(cfg_for(tmp_path, video, p1, resume=True, extra={"keep_temp": False}))
    assert rc == 0 and tr.n_calls == 0
    assert not os.path.isdir(os.path.join(str(tmp_path / "work"), os.listdir(str(tmp_path / "work"))[0])) if os.listdir(str(tmp_path / "work")) else True


def test_without_kept_chunks_a_resume_starts_fresh(video, tmp_path):
    p1 = mk_points(video, tmp_path, "p1.json", [P12])
    assert run(cfg_for(tmp_path, video, p1, extra={"keep_temp": False}))[0] == 0          # chunks deleted after success
    p2 = mk_points(video, tmp_path, "p2.json", [P12, {"obj_id": 1, "frame_idx": 90, "x": 400, "y": 180, "label": 1}])
    rc, tr, evs = run(cfg_for(tmp_path, video, p2, resume=True))
    assert rc in (0, 2) and tr.n_calls == 4 and evs[0]["reused_chunks"] == 0


def test_interrupted_incremental_run_can_be_resumed_again(video, tmp_path):
    p1 = mk_points(video, tmp_path, "p1.json", [P12])
    assert run(cfg_for(tmp_path, video, p1))[0] == 0
    p2 = mk_points(video, tmp_path, "p2.json", [P12, {"obj_id": 1, "frame_idx": 50, "x": 200, "y": 180, "label": 1},
                                                  {"obj_id": 1, "frame_idx": 50, "x": 5, "y": 5, "label": 0}])
    rc, _, _ = run(cfg_for(tmp_path, video, p2, resume=True), tracker=ColorTracker((180, 320), crash_chunks={1}))
    assert rc == 1                                                    # crashed in the 2nd re-processed chunk
    rc, tr, evs = run(cfg_for(tmp_path, video, p2, resume=True))      # same points as the crashed run: continues from there
    assert rc in (0, 2) and evs[0]["reused_chunks"] == 2 and tr.n_calls == 2 and probe_video(str(tmp_path / "out.mp4")).nb_frames == 160
