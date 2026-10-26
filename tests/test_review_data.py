import json
import os
import time

import pytest

from helpers import ColorTracker
from src.pipeline import Config, Pipeline, file_sha
from src.points_io import save_points
from src.review import ReviewError
from src.review_data import ReviewData, ReviewState, load_review_data, seg_key, state_path_for
from src.video_io import probe_video


def seg(type_, start, end=None, obj=1, sev="warn", count=1):
    return {"type": type_, "object_id": obj, "severity": sev, "start_frame": start, "end_frame": start if end is None else end,
            "event_count": count, "start_timestamp": start / 30, "end_timestamp": (start if end is None else end) / 30,
            "actions": ["flagged"], "first_detail": "x"}


SEGS = [seg("tracking_lost_mid_frame", 70), seg("tracking_recovered", 86), seg("chunk_boundary_resumed", 40, sev="info"),
        seg("model_error", 120, 130, obj=None, sev="error", count=11), seg("mask_area_jump", 100, obj=2)]


def make(tmp_path, run_id="r1", reused=0, segs=SEGS, **meta):
    doc = {"meta": {"run_id": run_id, "reused_chunks": reused, "chunk_size": 40, "points_sha": None, "output": None, **meta},
           "segments": segs, "events": []}
    p = tmp_path / "x_review.json"
    p.write_text(json.dumps(doc))
    return str(p), doc


def test_filtering(tmp_path):
    p, doc = make(tmp_path)
    d = ReviewData(p, doc)
    assert [s["start_frame"] for s in d.filtered("warn")] == [70, 86, 100, 120]
    assert [s["start_frame"] for s in d.filtered("error")] == [120]
    assert [s["start_frame"] for s in d.filtered("info")] == [40, 70, 86, 100, 120]
    assert [s["start_frame"] for s in d.filtered("warn", object_id=2)] == [100, 120]          # flags without an object stay visible
    assert d.object_ids() == [1, 2]
    d.state.set_reviewed(SEGS[0])
    assert [s["start_frame"] for s in d.filtered("warn", hide_reviewed=True)] == [86, 100, 120]
    assert d.unreviewed_count("warn") == 3


def test_reviewed_marks_persist_and_toggle(tmp_path):
    p, doc = make(tmp_path)
    d = ReviewData(p, doc)
    d.state.set_reviewed(SEGS[1], True, note="looked fine")
    assert os.path.isfile(state_path_for(p))
    d2 = ReviewData(p, doc)
    assert d2.state.is_reviewed(SEGS[1]) and not d2.state.is_reviewed(SEGS[0])
    assert d2.state.reviewed[seg_key(SEGS[1])]["note"] == "looked fine"
    d2.state.set_reviewed(SEGS[1], False)
    assert not ReviewData(p, doc).state.is_reviewed(SEGS[1])


def test_new_run_keeps_marks_only_for_reused_chunks(tmp_path):
    p, doc = make(tmp_path, run_id="r1")
    d = ReviewData(p, doc)
    for s in SEGS:
        d.state.set_reviewed(s)
    assert len(d.state.reviewed) == 5
    # a re-run re-processed chunks 3-4 (frames >= 80): marks for flags starting at 80+ are dropped, earlier ones kept
    p2, doc2 = make(tmp_path, run_id="r2", reused=2)
    d2 = ReviewData(p2, doc2)
    kept = {k.split("|")[2] for k in d2.state.reviewed}
    assert kept == {"40", "70"} and d2.dropped_marks == 3
    # loading the same run again changes nothing
    assert ReviewData(p2, doc2).dropped_marks == 0 and len(ReviewData(p2, doc2).state.reviewed) == 2
    # a flag that no longer exists is not kept either
    p3, doc3 = make(tmp_path, run_id="r3", reused=4, segs=[SEGS[0]])
    assert {k.split("|")[2] for k in ReviewData(p3, doc3).state.reviewed} == {"70"}


def test_full_reprocess_resets_all_marks(tmp_path):
    p, doc = make(tmp_path, run_id="r1")
    d = ReviewData(p, doc)
    d.state.set_reviewed(SEGS[0])
    p2, doc2 = make(tmp_path, run_id="r2", reused=0)
    assert ReviewData(p2, doc2).state.reviewed == {}


def test_staleness(tmp_path):
    pts = tmp_path / "p.json"
    pts.write_text("points v1")
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x" * 100)
    meta = {"points_sha": file_sha(str(pts)), "output": {"path": str(out), "size": 100, "mtime": out.stat().st_mtime}}
    p, doc = make(tmp_path, **meta)
    d = ReviewData(p, doc)
    assert d.staleness(str(pts), True) == []
    pts.write_text("points v2")
    assert any("points file changed" in r for r in d.staleness(str(pts), True))
    pts.write_text("points v1")
    out.write_bytes(b"y" * 120)
    assert any("output video was modified" in r for r in d.staleness(str(pts), True))
    out.unlink()
    assert any("no longer exists" in r for r in d.staleness(str(pts), True))
    # log made without points, points exist now
    p2, doc2 = make(tmp_path, run_id="r9", points_sha=None, output={"path": str(tmp_path / "o2"), "size": 0, "mtime": 0})
    assert any("without clicked points" in r for r in ReviewData(p2, doc2).staleness(str(pts), True))


def test_load_rejects_wrong_video_and_garbage(tmp_path, video):
    info = probe_video(video)
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ReviewError):
        load_review_data(str(bad), info)
    bad.write_text(json.dumps({"meta": {}}))
    with pytest.raises(ReviewError):
        load_review_data(str(bad), info)
    bad.write_text(json.dumps({"meta": {"input": {"fingerprint": "other"}}, "segments": []}))
    with pytest.raises(ReviewError):
        load_review_data(str(bad), info)
    with pytest.raises(ReviewError):
        load_review_data(str(tmp_path / "missing.json"), info)


def test_pipeline_log_carries_consistency_metadata(video, tmp_path):
    info = probe_video(video)
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", extra={"keep_temp": False})
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    d = load_review_data(str(tmp_path / "l" / "synthetic_review.json"), info)
    m = d.meta
    assert m["points_sha"] == file_sha(pts) and m["run_id"] and m["chunk_size"] == 40 and m["reused_chunks"] == 0
    assert m["reprocessed_chunks"] == [0, 1, 2, 3] and m["output"]["size"] == os.path.getsize(cfg.output)
    assert d.staleness(pts, True) == []
    save_points(pts, info, [{"obj_id": 1, "frame_idx": 13, "x": 10, "y": 180, "label": 1}])        # user edits the points
    assert any("points file changed" in r for r in d.staleness(pts, True))
    time.sleep(0.01)
