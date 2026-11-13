"""Before/after comparison of a correction: the preview server tracks one object over a short range from the corrected frame
(trial tracking), and the app shows that next to what was there before, with keep / revert."""
import io
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker
from src.mask_cache import JobMasks
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.preview_server import MAX_TRIAL_FRAMES, PREVIEW_PREFIX, decode_mask, encode_mask, serve, trial_track
from src.tracking import pack_mask, unpack_mask
from src.video_io import probe_video
from test_brush import square_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "sam2.1_hiera_small.pt")


def events(text):
    return [json.loads(ln[len(PREVIEW_PREFIX):]) for ln in text.splitlines() if ln.startswith(PREVIEW_PREFIX)]


# ------------------------------------------------------------------ trial tracking (stand-in tracker)
def test_trial_track_builds_one_chunk_from_the_prompts(video, tmp_path):
    info = probe_video(video)
    tr = ColorTracker((180, 320))
    prompts = {"0": {"mask": square_mask(30)}, "14": {"points": [[200.0, 180.0], [5.0, 5.0]], "labels": [1, 0]}, "99": {"points": [[1, 1]], "labels": [1]}}
    masks = trial_track(tr, info, (320, 180), 7, 30, 20, prompts, str(tmp_path / "t"))
    assert len(masks) == 20 and all(m is not None for m in masks)
    assert 600 < unpack_mask(masks[5], (180, 320)).sum() < 1100                      # the stand-in finds the square on every frame
    call = tr.calls[0]
    assert call["n_frames"] == 20 and len(call["inits"]) == 2                        # the prompt at offset 99 is outside the range
    (o0, l0, s0, area0, p0), (o1, l1, s1, area1, p1) = call["inits"]
    assert (o0, l0, s0, p0) == (7, 0, "mask_prompt", None) and 800 < area0 < 1000    # drawn at 640x360, resized to 320x180
    assert (o1, l1, s1, area1, p1) == (7, 14, "points", None, [(100.0, 90.0), (2.5, 2.5)])
    assert len(trial_track(tr, info, (320, 180), 1, 150, 60, {"0": {"mask": square_mask(100)}}, str(tmp_path / "t"))) == 10     # clipped at the end of the video
    for bad_kwargs, text in ((dict(frame=5000, frames=10, prompts=prompts), "outside the video"),
                             (dict(frame=30, frames=MAX_TRIAL_FRAMES + 1, prompts=prompts), "frames must be"),
                             (dict(frame=30, frames=10, prompts={"3": prompts["14"]}), "first frame needs a prompt"),
                             (dict(frame=30, frames=10, prompts={"0": {"points": [[5, 5]], "labels": [0]}}), "positive point"),
                             (dict(frame=30, frames=10, prompts={"0": {"mask": np.zeros((360, 640), bool)}}), "empty")):
        with pytest.raises(ValueError, match=text):
            trial_track(tr, info, (320, 180), 1, work_dir=str(tmp_path / "t"), **bad_kwargs)


def test_server_answers_track_requests_between_previews(video):
    wire_mask = {"mask": encode_mask(pack_mask(square_mask(30))), "shape": [360, 640]}
    reqs = [json.dumps({"cmd": "track", "id": 1, "obj_id": 3, "frame": 30, "frames": 12, "prompts": {"0": wire_mask}}),
            json.dumps({"id": 2, "frame": 30, "objects": {"1": {"points": [[100.0, 180.0]], "labels": [1]}}}),
            json.dumps({"cmd": "track", "id": 3, "obj_id": 3, "frame": 30, "frames": 5, "prompts": {}}),
            json.dumps({"cmd": "quit"})]
    out = io.StringIO()
    assert serve(video, "unused", analysis_width=320, stdin=io.StringIO("\n".join(reqs) + "\n"), stdout=out, tracker=ColorTracker((180, 320))) == 0
    evs = events(out.getvalue())
    assert [e["event"] for e in evs] == ["loading", "ready", "track", "masks", "error"]
    t = evs[2]
    assert (t["id"], t["obj_id"], t["frame"], t["frames"], t["shape"]) == (1, 3, 30, 12, [180, 320]) and len(t["masks"]) == 12
    assert decode_mask(t["masks"][0], (180, 320)).sum() > 600
    assert evs[4]["id"] == 3 and "first frame needs a prompt" in evs[4]["reason"]


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_real_sam2_trial_tracking_matches_the_job_within_the_same_chunk(video, tmp_path):
    """What the comparison shows as 'after' must be what processing produces, as far as that can hold: for the frames of the
    chunk in which the corrected frame is the object's first prompt, the trial and the job see the same history."""
    from src.tracking import SAM2Tracker
    info = probe_video(video)
    tracker = SAM2Tracker(CKPT)
    drawn = square_mask(30)
    trial = trial_track(tracker, info, (640, 360), 1, 30, 25, {"0": {"mask": drawn}}, str(tmp_path / "t"))
    trial = [unpack_mask(m, (360, 640)) for m in trial]

    p = str(tmp_path / "p.json")
    save_points(p, info, [], masks={1: {30: drawn}})
    cfg = Config(input=video, output=str(tmp_path / "out.mp4"), points=p, work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"),
                 chunk_size=40, analysis_width=640, max_frames=60, preset="ultrafast", extra={"keep_temp": True})
    assert Pipeline(cfg, tracker, None).run() in (0, 2)
    job = JobMasks(os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0]))

    def iou(a, b):
        return float((a & b).sum()) / max(1, int((a | b).sum()))

    assert iou(trial[0], drawn) > 0.97                                               # on the corrected frame: the drawn mask itself
    for f in range(30, 40):                                                          # same chunk as the job (frames 0-39)
        assert iou(trial[f - 30], job.frame(f)["masks"][1]) > 0.95, f
    for f in range(40, 55):                                                          # past the job's chunk boundary: still the same object
        assert iou(trial[f - 30], job.frame(f)["masks"][1]) > 0.8, f
        assert iou(trial[f - 30], square_mask(f)) > 0.8, f


# ------------------------------------------------------------------ the comparison in the app
import sys  # noqa: E402

from test_brush import drag  # noqa: E402
from test_gui import _widget_pos, open_clip, pump, qapp, shown_code, win  # noqa: E402,F401  (fixtures)
from test_mask_preview import run_job  # noqa: E402

# Stand-in "model" for the app tests: a drawn mask is echoed on every frame from its prompt on (so a correction visibly changes
# the result); click prompts fall back to finding the magenta square.
ECHO_SERVER = """
import sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from helpers import ColorTracker
from src.preview_server import serve
from src.tracking import pack_mask


class EchoTracker(ColorTracker):
    def track_chunk(self, frame_dir, n_frames, inits):
        out = super().track_chunk(frame_dir, n_frames, inits)
        for it in inits:
            if it.mask is not None:
                for li in range(it.local_frame, n_frames):
                    out[it.obj_id][li] = pack_mask(it.mask)
        return out


sys.exit(serve(sys.argv[1], "unused", analysis_width=320, tracker=EchoTracker((180, 320))))
"""


def echo_server_command(tmp_path, video):
    script = tmp_path / "echo_preview_server.py"
    script.write_text(ECHO_SERVER.format(root=ROOT, tests=os.path.join(ROOT, "tests")), encoding="utf-8")
    return [sys.executable.replace("pythonw.exe", "python.exe"), str(script), video]


def paint_big_blob(win):
    """A mask clearly larger than the square at frame 30 (x 70-130, y 150-210)."""
    win.brush_size.setValue(30)
    win.act_brush.setChecked(True)
    drag(win, [(60, 140), (140, 140), (140, 220), (60, 220), (60, 140), (100, 180)])
    win.act_brush.setChecked(False)


def test_compare_with_the_last_run_then_keep(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    open_clip(win)
    jd, pts = run_job(win.test_video, tmp_path)                                      # a kept run: the "before"
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"))
    assert win._try_load_points(pts, quiet=True)
    win.preview_command = echo_server_command(tmp_path, win.test_video)
    assert not win.compare_keep.isEnabled() and not win.compare_mode.isEnabled()
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    assert win.start_compare() is False and "클릭이나 그린 마스크가 없습니다" in win.errors[-1]    # nothing to track from on this frame
    paint_big_blob(win)
    drawn = win.store.get_mask(1, 30)
    win.compare_frames.setValue(20)
    assert not win.preview.is_on()
    assert win.start_compare() and win.preview.is_on() and win.act_live.isChecked()  # the preview process is started for it
    assert "시험 추적 중" in win.compare_info.text() and "시험 추적 중" in win.view.overlay()["note"]
    assert pump(lambda: win._compare is not None and win._compare["ready"], timeout=90), win.errors
    cmp = win._compare
    assert (cmp["oid"], cmp["start"], cmp["n"], cmp["before_source"]) == (1, 30, 20, "last_run")
    assert len(cmp["after"]) == 20 and all(m is not None for m in cmp["after"]) and len(cmp["stats"]) == 20
    assert cmp["stats"][0]["after_px"] > 2 * cmp["stats"][0]["before_px"] > 0 and cmp["stats"][0]["iou"] < 0.6
    assert "지난 실행의 마스크" in win.compare_info.text() and "달라진 프레임(겹침 90% 미만): 20개" in win.compare_info.text()
    assert win.timeline_view._highlight == (30, 50) and win.compare_keep.isEnabled()

    ov = win.view.overlay()                                                          # frame 30, "difference" view
    assert len(ov["masks"]) == 3 and "[비교 1/20]" in ov["note"] and "초록" in ov["note"]
    kept, added, removed = ov["masks"]
    assert kept.pixelColor(50, 90).alpha() > 0 and added.pixelColor(22, 62).alpha() > 0 and removed.pixelColor(50, 90).alpha() == 0
    win.compare_mode.setCurrentIndex(win.compare_mode.findData("before"))
    assert len(win.view.overlay()["masks"]) == 1 and "수정 전" in win.view.overlay()["note"]
    win.compare_mode.setCurrentIndex(win.compare_mode.findData("after"))
    assert "수정 후" in win.view.overlay()["note"]
    win.compare_mode.setCurrentIndex(win.compare_mode.findData("diff"))
    win.goto(45)                                                                     # later in the range: the square has moved on
    assert "[비교 16/20]" in win.view.overlay()["note"]
    assert win.view.overlay()["masks"][2].pixelColor(96, 90).alpha() > 0             # red: where the old mask was and the new one is not
    win.goto(60)                                                                     # outside the compared range: the normal overlays
    assert "비교" not in ((win.view.overlay() or {}).get("note") or "")
    win.goto_biggest_change()
    assert 30 <= win.frame_idx < 50 and not win.view.grab().isNull()

    win.keep_correction()
    assert win._compare is None and win.timeline_view._highlight is None and not win.compare_keep.isEnabled()
    assert np.array_equal(win.store.get_mask(1, 30), drawn) and win.dirty             # the correction stays, nothing else changed


def test_compare_without_a_run_tracks_the_old_prompt_too_and_revert_restores_it(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "no_work"), log_dir=str(tmp_path / "logs"))
    open_clip(win)
    win.preview_command = echo_server_command(tmp_path, win.test_video)
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    win.view.handle_click(*_widget_pos(win, 100, 180), 1)                            # the frame has a click ...
    paint_big_blob(win)                                                              # ... and then a drawn mask
    win.compare_frames.setValue(10)
    assert win.start_compare()
    assert pump(lambda: win._compare is not None and win._compare["ready"], timeout=90), win.errors
    cmp = win._compare
    assert cmp["before_source"] == "trial" and len(cmp["before"]) == 10              # no kept run: the click alone was tracked as "before"
    assert cmp["stats"][0]["after_px"] > cmp["stats"][0]["before_px"] > 0
    assert "고치기 전 상태로 시험 추적" in win.compare_info.text() and len(win.view.overlay()["masks"]) == 3

    drag_mask = win.store.get_mask(1, 30).copy()
    win.act_brush.setChecked(True)                                                   # painting again changes the prompt: the comparison is dropped
    drag(win, [(300, 300), (310, 300)])
    win.act_brush.setChecked(False)
    assert win._compare is None and win.timeline_view._highlight is None
    assert not np.array_equal(win.store.get_mask(1, 30), drag_mask)

    assert win.start_compare()
    assert pump(lambda: win._compare is not None and win._compare["ready"], timeout=90)
    win.revert_correction()                                                          # the frame had no drawn mask before: it is removed
    assert win._compare is None and win.store.get_mask(1, 30) is None and (1, 30) not in win._brush_base
    assert win.save_points() and win.corrections_saved == []                         # a reverted correction leaves no record
    win.undo_stroke()                                                                # and revert itself can be undone
    assert win.store.get_mask(1, 30) is not None


def test_compare_on_a_clicked_frame_without_a_drawn_mask(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "no_work"), log_dir=str(tmp_path / "logs"))
    open_clip(win)
    win.preview_command = echo_server_command(tmp_path, win.test_video)
    win.goto(150)
    pump(lambda: shown_code(win) == 150)
    win.view.handle_click(*_widget_pos(win, 300, 180), 1)
    win.compare_frames.setValue(60)
    assert win.start_compare()
    assert pump(lambda: win._compare is not None and win._compare["ready"], timeout=90), win.errors
    cmp = win._compare
    assert cmp["n"] == 10 and cmp["before_source"] == "none" and cmp["before"] is None   # clipped at the end; nothing to compare with
    assert len(win.view.overlay()["masks"]) == 1 and "비교할 수정 전 마스크가 없습니다" in win.view.overlay()["note"]
    assert "없음" in win.compare_info.text()
    win.revert_correction()                                                          # nothing drawn here: nothing is removed
    assert win._compare is None and len(win.store.points) == 1
    win._set_job_running(True)
    assert not win.compare_box.isEnabled() and not win.act_compare.isEnabled()
    win._set_job_running(False)
