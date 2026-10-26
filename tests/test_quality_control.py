import csv
import json

import numpy as np

from src import quality_control as qc


def mk(shape, x1, y1, x2, y2):
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


SHAPE = (360, 640)


def run(masks, **kw):
    log = qc.QCLog(lambda f: f / 30)
    mon = qc.ObjectMonitor(SHAPE, qc.QCParams(**kw))
    verdicts = [mon.observe(i, 1, m, log) for i, m in enumerate(masks)]
    return log, mon, verdicts


def types(log):
    return [e.type for e in log.events]


def test_stable_mask_has_no_events():
    log, _, v = run([mk(SHAPE, 100, 100, 160, 160)] * 10)
    assert log.events == [] and all(x.status == "ok" for x in v)


def test_lost_mid_frame_flagged_then_recovered():
    m = mk(SHAPE, 100, 100, 160, 160)
    log, mon, v = run([m, m, None, None, None, m])
    assert types(log) == [qc.LOST_MID_FRAME, qc.RECOVERED]
    assert v[2].status == "lost" and v[2].uncertain == qc.LOST_MID_FRAME and v[2].region == (100, 100, 160, 160)
    assert mon.frames_lost(5, 1) == 0                   # episode closed
    assert "3 frames" in log.events[1].detail           # lost at frame 2, back at 5
    _, mon2, _ = run([m, None, None, None])
    assert mon2.frames_lost(3, 1) == 3


def test_lost_at_edge_is_info_and_not_uncertain():
    sliding = [mk(SHAPE, 580, 100, 640, 160), mk(SHAPE, 600, 100, 640, 160), mk(SHAPE, 615, 100, 640, 160),
               mk(SHAPE, 625, 100, 640, 160), None]       # slides out through the right border
    log, _, v = run(sliding)
    assert types(log) == [qc.LOST_AT_EDGE] and log.events[0].severity == "info"
    assert v[4].status == "lost" and v[4].uncertain is None


def test_vanishing_at_full_size_next_to_border_is_still_suspicious():
    m = mk(SHAPE, 580, 100, 640, 160)            # touches the border but did not shrink
    log, _, v = run([m, m, m, None])
    assert types(log) == [qc.LOST_MID_FRAME] and v[3].uncertain == qc.LOST_MID_FRAME


def test_area_jump_flagged_with_union_region():
    small, big = mk(SHAPE, 100, 100, 140, 140), mk(SHAPE, 100, 100, 220, 220)
    log, _, v = run([small, big])
    assert types(log) == [qc.AREA_JUMP]
    assert v[1].uncertain == qc.AREA_JUMP and v[1].region == (100, 100, 220, 220)


def test_tiny_noise_jump_ignored():
    a, b = mk(SHAPE, 10, 10, 14, 14), mk(SHAPE, 10, 10, 20, 20)    # 16px -> 100px but both tiny
    log, _, _ = run([a, b], min_jump_area_frac=1e-3)
    assert log.events == []


def test_not_started_object_is_silent():
    log, _, v = run([None, None])
    assert log.events == [] and v[0].status == "not_started"


def test_state_roundtrip_keeps_lost_episode():
    m = mk(SHAPE, 100, 100, 160, 160)
    log = qc.QCLog()
    mon = qc.ObjectMonitor(SHAPE)
    mon.observe(0, 1, m, log)
    mon.observe(1, 1, None, log)
    saved = json.loads(json.dumps(mon.state_dict()))
    mon2 = qc.ObjectMonitor(SHAPE)
    mon2.load_state(saved)
    log2 = qc.QCLog()
    v = mon2.observe(2, 1, None, log2)
    assert v.status == "lost" and log2.events == []          # same episode, no duplicate event
    mon2.observe(3, 1, m, log2)
    assert types(log2) == [qc.RECOVERED]


def test_abandon():
    m = mk(SHAPE, 100, 100, 160, 160)
    log, mon, _ = run([m, None])
    mon.mark_abandoned(5, 1, log, "gave up")
    assert log.events[-1].type == qc.OBJECT_ABANDONED
    assert mon.observe(6, 1, None, log).status == "not_started"


def test_detection_mask_mismatch():
    log = qc.QCLog()
    covered = mk(SHAPE, 100, 100, 200, 200)
    n = qc.detection_mask_mismatch([((110, 110, 190, 190), 0.9, "A"), ((400, 50, 450, 90), 0.8, "B")], [covered], 7, log)
    assert n == 1 and log.events[0].extra["cls"] == "B" and log.events[0].frame == 7


def test_segments_merge_consecutive_frames():
    log = qc.QCLog(lambda f: f / 30)
    for f in (3, 4, 5, 9):
        log.add(f, qc.AREA_JUMP, object_id=1)
    log.add(4, qc.MODEL_ERROR, "error")
    segs = log.segments()
    jump = [s for s in segs if s["type"] == qc.AREA_JUMP]
    assert [(s["start_frame"], s["end_frame"]) for s in jump] == [(3, 5), (9, 9)]
    assert [s for s in segs if s["type"] == qc.MODEL_ERROR][0]["severity"] == "error"


def test_save_json_csv(tmp_path):
    log = qc.QCLog(lambda f: f / 30)
    log.add(10, qc.LOST_MID_FRAME, object_id=2, detail="x", action="flagged")
    log.add(-1, qc.OUTPUT_INVALID, "error", detail="count")
    j, c = tmp_path / "r.json", tmp_path / "r.csv"
    log.save(str(j), str(c), {"a": 1})
    d = json.loads(j.read_text(encoding="utf-8"))
    assert d["summary"][qc.LOST_MID_FRAME] == 1 and d["meta"] == {"a": 1} and "heuristic" in d["note"].lower()
    rows = list(csv.reader(c.open(encoding="utf-8-sig")))
    assert rows[0][:3] == ["frame", "timestamp", "type"] and len(rows) == 3
    assert not (tmp_path / "r.json.tmp").exists()
    back = qc.QCLog.from_events(d["events"])
    assert len(back.events) == 2


def test_recovery_after_long_loss_or_far_away_is_warn():
    a, b = mk(SHAPE, 100, 100, 160, 160), mk(SHAPE, 100, 100, 160, 160)
    log, _, _ = run([a] + [None] * 8 + [b])                       # same place, but 8 frames lost
    rec = [e for e in log.events if e.type == qc.RECOVERED][0]
    assert rec.severity == "warn" and "long loss" in rec.detail and rec.action == "resumed_normal_unverified"
    far = mk(SHAPE, 400, 100, 460, 160)
    log, _, _ = run([a, None, far])                               # 2 frames lost, but 300px away
    rec = [e for e in log.events if e.type == qc.RECOVERED][0]
    assert rec.severity == "warn" and "from where it vanished" in rec.detail
    log, _, _ = run([a, None, a])
    assert [e for e in log.events if e.type == qc.RECOVERED][0].severity == "info"
