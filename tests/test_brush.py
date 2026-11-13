"""Hand-drawn masks: stored in the points file (version 3), given to the tracker as a mask prompt on their frame, part of the
per-chunk prompt signature (so an edit re-processes from its chunk on), and recorded as before/after corrections."""
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker
from src.corrections import list_corrections, load_correction, save_correction
from src.mask_cache import JobMasks
from src.pipeline import Config, Pipeline, plan_reprocess, prompt_signatures
from src.points_io import (PointStore, PointsError, decode_mask, encode_mask, load_mask_prompts, load_object_styles, load_points,
                           mask_digest, save_points)
from src.video_io import probe_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "sam2.1_hiera_small.pt")
P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}
EDIT = (360, 640)                                    # masks are drawn at the preview size (here: the clip's own size)


def square_mask(frame: int, shape=EDIT, grow: int = 0) -> np.ndarray:
    """The magenta square of the synthetic clip at ``frame`` (x from -30 + 5 * (frame - 10), y 150-210), optionally grown."""
    h, w = shape
    k = w / 640.0
    x0 = -30 + 5 * (frame - 10)
    m = np.zeros(shape, bool)
    m[max(0, int((150 - grow) * k)):int((210 + grow) * k), max(0, int((x0 - grow) * k)):max(0, int((x0 + 60 + grow) * k))] = True
    return m


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": True})
    base.update(kw)
    return Config(**base)


def run(cfg):
    tr, evs = ColorTracker((180, 320)), []
    return Pipeline(cfg, tr, None, progress=evs.append).run(), tr, evs


# ------------------------------------------------------------------ file format
def test_mask_encoding_and_digest():
    m = square_mask(30)
    d = encode_mask(m)
    assert d["shape"] == [360, 640] and len(d["data"]) < 600                        # a simple mask is a few hundred bytes
    assert np.array_equal(decode_mask(d), m)
    assert mask_digest(m) == mask_digest(m.copy()) != mask_digest(square_mask(31))
    assert mask_digest(m) != mask_digest(m.reshape(180, 1280))                      # same bits, other shape
    for bad in ({"shape": [0, 10], "data": d["data"]}, {"shape": [360, 640], "data": "!!"}, {"shape": [99999, 99999], "data": d["data"]}):
        with pytest.raises(PointsError):
            decode_mask(bad)


def test_points_file_version_3_round_trip(video, tmp_path):
    info = probe_video(video)
    plain = str(tmp_path / "plain.json")
    save_points(plain, info, [P12], masks={1: {30: np.zeros(EDIT, bool)}})           # an empty mask is not written
    assert json.load(open(plain))["version"] == 1 and load_mask_prompts(plain) == {}
    p = str(tmp_path / "m.json")
    save_points(p, info, [P12], styles={2: {"style": "solid"}}, masks={1: {30: square_mask(30)}, 2: {50: square_mask(50)}})
    doc = json.load(open(p))
    assert doc["version"] == 3 and [(r["obj_id"], r["frame_idx"]) for r in doc["mask_prompts"]] == [(1, 30), (2, 50)]
    assert doc["mask_prompts"][0]["timestamp"] == pytest.approx(1.0)
    masks = load_mask_prompts(p, info)
    assert set(masks) == {1, 2} and np.array_equal(masks[1][30], square_mask(30)) and np.array_equal(masks[2][50], square_mask(50))
    prompts = load_points(p, info)
    assert prompts[1] == {12: ([(10.0, 180.0)], [1])} and prompts[2] == {}           # object 2 exists through its mask alone
    assert load_object_styles(p) == {2: {"style": "solid"}}                          # settings of a mask-only object are kept
    with pytest.raises(PointsError):
        save_points(p, info, [P12], masks={1: {9999: square_mask(30)}})
    doc["mask_prompts"][0]["frame_idx"] = 9999
    json.dump(doc, open(p, "w"))
    with pytest.raises(PointsError):
        load_points(p, info)


def test_point_store_masks():
    s = PointStore()
    s.set_mask(1, 30, square_mask(30))
    s.set_mask(3, 40, square_mask(40))
    assert s.get_mask(1, 30) is not None and s.get_mask(1, 31) is None and s.object_ids() == [1, 3]
    assert s.new_object() == 4 and s.select(3)
    s.add(30, 5.0, 5.0, 0)                                                           # only a negative click on a frame ...
    assert s.valid() == ["object 3 frame 30: only negative points"]
    s.set_mask(3, 30, square_mask(30))                                               # ... is fine when the frame has a drawn mask
    assert s.valid() == []
    s.set_mask(3, 30, np.zeros(EDIT, bool))                                          # an empty mask removes it
    assert s.get_mask(3, 30) is None and 3 in s.masks
    assert s.delete_object() == 2 and 3 not in s.masks                               # 1 click + 1 mask removed
    s.set_mask(1, 30, None)
    assert s.masks == {}


def test_drawn_masks_are_part_of_the_chunk_signatures():
    chunks = [(0, 40), (40, 80), (80, 120), (120, 160)]
    prompts = {1: {12: ([(10.0, 180.0)], [1])}}
    base = prompt_signatures(prompts, chunks)
    assert prompt_signatures(prompts, chunks, {}) == base == prompt_signatures(prompts, chunks, None)     # old files keep their signatures
    a = prompt_signatures(prompts, chunks, {1: {90: square_mask(90)}})
    assert a[:2] == base[:2] and a[2] != base[2] and a[3] == base[3]
    b = prompt_signatures(prompts, chunks, {1: {90: square_mask(90, grow=4)}})
    assert b[2] != a[2]                                                              # repainting the same frame changes its chunk


# ------------------------------------------------------------------ pipeline
def test_drawn_mask_is_given_to_the_tracker_on_its_frame(video, tmp_path):
    info = probe_video(video)
    p = str(tmp_path / "p.json")
    pts = [P12, {"obj_id": 1, "frame_idx": 50, "x": 5, "y": 5, "label": 1}]          # frame 50 also has a click: the mask wins
    save_points(p, info, pts, masks={1: {50: square_mask(50)}, 2: {90: square_mask(90)}})
    rc, tr, _ = run(cfg_for(tmp_path, video, p))
    assert rc in (0, 2) and tr.n_calls == 4
    by_source = {}
    for ci, call in enumerate(tr.calls):
        for oid, local, source, area, points in call["inits"]:
            by_source.setdefault(source, []).append((ci, oid, local, area, points))
    assert by_source["points"] == [(0, 1, 12, None, [(5.0, 90.0)])]                  # the click on frame 50 was not used
    m50, m90 = by_source["mask_prompt"]
    assert m50[:3] == (1, 1, 11) and m50[4] is None                                  # chunk 1, object 1, frame 50 = prefix 1 + 10
    prefix = tr.calls[2]["n_frames"] - 40                                            # > 1 here: object 1 was hidden at the boundary
    assert m90[:3] == (2, 2, prefix + 10)                                            # the mask-only object starts at its mask (frame 90)
    assert 800 < m50[3] < 1000 and 800 < m90[3] < 1000                               # 60x60 px drawn at 640 wide = 30x30 at analysis size
    jm = JobMasks(os.path.join(tmp_path / "work", os.listdir(tmp_path / "work")[0]))
    assert 2 not in jm.frame(60)["masks"] and 2 in jm.frame(100)["masks"]            # object 2 is tracked from its mask frame on


def test_adding_a_drawn_mask_reprocesses_from_its_chunk(video, tmp_path):
    info = probe_video(video)
    p1 = str(tmp_path / "p1.json")
    save_points(p1, info, [P12])
    assert run(cfg_for(tmp_path, video, p1))[0] in (0, 2)
    p2 = str(tmp_path / "p2.json")
    save_points(p2, info, [P12], masks={1: {90: square_mask(90)}})
    cfg2 = cfg_for(tmp_path, video, p2, resume=True)
    plan = plan_reprocess(cfg2)
    assert plan["can_resume"] and plan["reusable_chunks"] == 2 and plan["first_changed_chunk"] == 2
    rc, tr, evs = run(cfg2)
    assert rc in (0, 2) and tr.n_calls == 2 and evs[0]["reused_chunks"] == 2 and evs[0]["points_changed"] is True
    rc, tr, evs = run(cfg_for(tmp_path, video, p2, resume=True))                    # unchanged: everything reused
    assert tr.n_calls == 0 and evs[0]["reused_chunks"] == 4


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_real_sam2_follows_a_drawn_mask_where_one_click_selected_too_little(video, tmp_path):
    """One click in the middle of the striped square makes SAM 2 select a single stripe. Painting the whole square on that frame
    must give exactly that mask on the frame and a mask of the whole square on the frames after it."""
    from src.tracking import SAM2Tracker
    info = probe_video(video)
    tracker = SAM2Tracker(CKPT)
    click = {"obj_id": 1, "frame_idx": 30, "x": 100.0, "y": 180.0, "label": 1}

    def job(name, **save_kw):
        d = tmp_path / name
        d.mkdir()
        p = str(d / "p.json")
        save_points(p, info, [click], **save_kw)
        cfg = Config(input=video, output=str(d / "out.mp4"), points=p, work_dir=str(d / "work"), log_dir=str(d / "logs"), chunk_size=40,
                     analysis_width=640, max_frames=70, preset="ultrafast", extra={"keep_temp": True})
        assert Pipeline(cfg, tracker, None).run() in (0, 2)
        return JobMasks(os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0]))

    def iou(a, b):
        return float((a & b).sum()) / max(1, int((a | b).sum()))

    clicked = job("click")
    painted = job("paint", masks={1: {30: square_mask(30)}})
    gt = {f: square_mask(f) for f in (30, 35, 45, 60)}
    assert clicked.frame(30)["masks"][1].sum() < 0.3 * gt[30].sum()                 # the problem: a stripe, not the square
    assert iou(painted.frame(30)["masks"][1], gt[30]) > 0.97                        # on its frame the drawn mask is used as it is
    for f in (35, 45, 60):                                                          # ... and tracking continues from it, across the chunk boundary at 40
        assert iou(painted.frame(f)["masks"][1], gt[f]) > 0.8, (f, iou(painted.frame(f)["masks"][1], gt[f]))
        assert iou(painted.frame(f)["masks"][1], gt[f]) > iou(clicked.frame(f)["masks"].get(1, np.zeros(EDIT, bool)), gt[f]) + 0.3


# ------------------------------------------------------------------ correction records
def test_correction_records(tmp_path):
    before, after = square_mask(30), square_mask(30, grow=6)
    p = save_correction(str(tmp_path), "abcdef0123456789", (640, 360), 2, 30, 1.0, before, after, "preview")
    assert os.path.dirname(p) == os.path.join(str(tmp_path), "corrections") and os.path.basename(p).startswith("abcdef0123_o2_f0000030_")
    r = load_correction(p)
    assert np.array_equal(r["before"], before) and np.array_equal(r["after"], after) and r["shape"] == EDIT
    assert (r["obj_id"], r["frame_idx"], r["before_source"], r["video_fingerprint"]) == (2, 30, "preview", "abcdef0123456789")
    assert r["added_px"] == int(after.sum() - before.sum()) and r["removed_px"] == 0 and r["before_px"] == int(before.sum())
    p2 = save_correction(str(tmp_path), "abcdef0123456789", (640, 360), 2, 30, 1.0, None, after, "none")      # same second: no overwrite
    assert p2 != p and load_correction(p2)["before_px"] == 0 and load_correction(p2)["added_px"] == int(after.sum())
    save_correction(str(tmp_path), "ffffffffffffffff", (640, 360), 1, 5, 0.1, None, after, "none")
    assert len(list_corrections(str(tmp_path))) == 3 and len(list_corrections(str(tmp_path), "abcdef0123456789")) == 2
    assert list_corrections(str(tmp_path / "nowhere")) == []
    with pytest.raises(ValueError):
        save_correction(str(tmp_path), "x" * 16, (640, 360), 1, 1, 0.0, np.zeros((10, 10), bool), after, "drawn")


# ------------------------------------------------------------------ the brush in the app
from test_gui import _widget_pos, default_points_path, open_clip, pump, qapp, shown_code, win  # noqa: E402,F401  (fixtures)


def drag(win, points, label=1):
    """Press at the first frame position, move through the rest, release."""
    v = win.view
    assert v.handle_stroke_press(*_widget_pos(win, *points[0]), label)
    for p in points[1:]:
        v.handle_stroke_move(*_widget_pos(win, *p))
    v.handle_stroke_release()


def test_brush_paints_erases_undoes_and_saves(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    win.last_settings = ProcessSettings(log_dir=str(tmp_path / "logs"))
    win.act_brush.setChecked(True)                                                   # needs a video
    assert not win.act_brush.isChecked() and win.errors == ["먼저 영상을 여세요."]
    open_clip(win)
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    win.brush_size.setValue(10)
    win.act_brush.setChecked(True)
    assert win.view.brush_mode and win.view.brush_radius == 10 and "브러시" in win.view.overlay()["note"]
    win.view.handle_click(*_widget_pos(win, 300, 100), 1)                            # in brush mode a click places no point
    assert win.store.points == []

    drag(win, [(80, 160), (120, 160), (120, 200)])                                   # an L-shaped stroke on object 1
    m = win.store.get_mask(1, 30)
    assert m is not None and m.shape == (360, 640) and win.dirty
    assert m[160, 100] and m[180, 120] and m[160, 72] and not m[160, 60] and not m[100, 300]      # along the path, radius 10
    assert "그린 마스크 1" in win.obj_list.item(0).text() and win.timeline_view._objects[0][2] == [30]
    ov = win.view.overlay()
    assert len(ov["masks"]) == 1 and "그린 마스크" in ov["note"] and ov["masks"][0].pixelColor(100, 160).alpha() > 0
    area = int(m.sum())

    drag(win, [(120, 180), (120, 200)], label=0)                                     # right-drag erases
    m2 = win.store.get_mask(1, 30)
    assert not m2[190, 120] and m2[160, 90] and int(m2.sum()) < area
    win.undo_stroke()                                                                # back to the L
    assert int(win.store.get_mask(1, 30).sum()) == area
    win.undo_stroke()                                                                # back to nothing
    assert win.store.get_mask(1, 30) is None and win.store.masks == {}
    win.undo_stroke()                                                                # nothing left to undo: no error
    drag(win, [(80, 160), (120, 160)])
    win.goto(31)                                                                     # another frame has its own mask
    assert win.store.get_mask(1, 31) is None and not win.view.overlay().get("masks")
    drag(win, [(200, 100), (210, 100)])
    assert sorted(win.store.masks[1]) == [30, 31]
    win.clear_drawn_mask()
    assert sorted(win.store.masks[1]) == [30]
    win.undo_stroke()                                                                # clearing can be undone too
    assert sorted(win.store.masks[1]) == [30, 31]
    win.clear_drawn_mask()

    assert win.save_points() and not win.dirty and win.errors == ["먼저 영상을 여세요."]   # an object with a drawn mask needs no click
    path = default_points_path(win.test_video)
    assert json.load(open(path))["version"] == 3
    saved = load_mask_prompts(path, probe_video(win.test_video))
    assert sorted(saved[1]) == [30] and np.array_equal(saved[1][30], win.store.get_mask(1, 30))
    assert len(win.corrections_saved) == 1                                           # one record: frame 30, started from nothing
    rec = load_correction(win.corrections_saved[0])
    assert (rec["obj_id"], rec["frame_idx"], rec["before_source"], rec["before_px"]) == (1, 30, "none", 0)
    assert np.array_equal(rec["after"], saved[1][30]) and rec["video_fingerprint"] == win.info.fingerprint
    assert os.path.dirname(win.corrections_saved[0]) == os.path.join(str(tmp_path / "logs"), "corrections")
    assert win.save_points() and win.corrections_saved == []                         # nothing painted since: no new record

    win.store.masks.clear()
    assert win._try_load_points(path, quiet=True) and sorted(win.store.masks[1]) == [30] and not win.dirty
    win.goto(30)
    drag(win, [(300, 300), (320, 300)])                                              # repaint a loaded mask: the record starts from it
    assert win.save_points() and len(win.corrections_saved) == 1
    rec = load_correction(win.corrections_saved[0])
    assert rec["before_source"] == "drawn" and rec["before_px"] == int(saved[1][30].sum()) and rec["added_px"] > 0 and rec["removed_px"] == 0

    assert win._points_for_job() == path                                             # a file with only drawn masks is a job input
    win._set_job_running(True)                                                       # locked while a job runs
    assert not win.act_brush.isChecked() and not win.view.brush_mode and not win.brush_box.isEnabled()
    before = win.store.get_mask(1, 30).copy()
    win.undo_stroke()
    assert np.array_equal(win.store.get_mask(1, 30), before)
    win._set_job_running(False)
    assert win.act_brush.isEnabled() and not win.view.grab().isNull()


def test_brush_starts_from_the_mask_that_is_shown(win, tmp_path):
    """With the live preview on, the first stroke edits SAM 2's mask instead of starting from nothing, and the correction record
    says so. (Stand-in preview server: the 'model' finds the magenta square.)"""
    from src.app.job_runner import ProcessSettings
    from test_mask_preview import fake_server_command
    win.last_settings = ProcessSettings(log_dir=str(tmp_path / "logs"))
    open_clip(win)
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    win.preview_command = fake_server_command(tmp_path, win.test_video)
    win.act_live.setChecked(True)
    win.view.handle_click(*_widget_pos(win, 100, 180), 1)
    assert pump(lambda: bool((win.view.overlay() or {}).get("masks")), timeout=60)
    shown = win._live_masks[30][1][1]                                                # the previewed mask of object 1 (analysis size)
    win.act_brush.setChecked(True)
    win.brush_size.setValue(8)
    drag(win, [(140, 180), (150, 180)])                                              # add a bit to the right of the square
    m = win.store.get_mask(1, 30)
    assert m[180, 100] and m[180, 148] and int(m.sum()) > int(win._resize_mask(shown).sum())   # the previewed mask plus the stroke
    assert "그린 마스크" in win.view.overlay()["note"]                                # the drawn mask is what is shown now
    assert win.save_points() and len(win.corrections_saved) == 1
    rec = load_correction(win.corrections_saved[0])
    assert rec["before_source"] == "preview" and rec["removed_px"] == 0 and 0 < rec["added_px"] < 600
    assert rec["before_px"] == int(win._resize_mask(shown).sum())
