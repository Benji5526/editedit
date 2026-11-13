"""Mask preview: (1) the preview server computes the mask of one frame's clicks exactly like the first frame of a processing
chunk, (2) the app shows it while clicking, (3) the app shows the kept masks of the last run.
The stand-in tracker is used everywhere except in the one test that starts the real worker with the real SAM 2 checkpoint."""
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from helpers import ColorTracker
from src.detection import Det
from src.mask_cache import JobMasks
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.preview_server import PREVIEW_PREFIX, analysis_size, decode_mask, encode_mask, preview_masks, serve
from src.tracking import pack_mask
from src.video_io import probe_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "sam2.1_hiera_small.pt")
CLICK30 = [100.0, 180.0]                 # inside the square at frame 30 (x 70-130, y 150-210)


def events(text):
    return [json.loads(ln[len(PREVIEW_PREFIX):]) for ln in text.splitlines() if ln.startswith(PREVIEW_PREFIX)]


# ------------------------------------------------------------------ server logic (no process, no GPU)
def test_analysis_size_matches_the_pipeline():
    assert analysis_size(3840, 2160, 1920) == (1920, 1080) and analysis_size(640, 360, 320) == (320, 180)
    assert analysis_size(640, 360, 1920) == (640, 360)                               # never upscaled


def test_mask_encoding_round_trip():
    m = np.zeros((180, 320), bool)
    m[40:70, 100:140] = True
    assert np.array_equal(decode_mask(encode_mask(pack_mask(m)), (180, 320)), m)


def test_preview_masks_for_one_frame(video, tmp_path):
    info = probe_video(video)
    tr = ColorTracker((180, 320))
    objects = {"1": {"points": [CLICK30], "labels": [1]}, "2": {"points": [[5, 5]], "labels": [0]}, "3": {"points": [], "labels": []}}
    masks, skipped = preview_masks(tr, info, (320, 180), 30, objects, str(tmp_path / "f"))
    assert set(masks) == {1} and set(skipped) == {2, 3} and "긍정" in skipped[2]
    m = decode_mask(encode_mask(masks[1]), (180, 320))
    assert 600 < int(m.sum()) < 1100 and m[90, 50]                                   # the 30 px square at analysis size
    call = tr.calls[0]
    assert call["n_frames"] == 1 and call["inits"][0][:3] == (1, 0, "points")        # one frame, prompted on it like a chunk's first frame
    assert call["inits"][0][4] == [(50.0, 90.0)]                                     # clicks scaled to analysis coordinates
    assert preview_masks(tr, info, (320, 180), 30, {"2": {"points": [[5, 5]], "labels": [0]}}, str(tmp_path / "f"))[0] == {}
    assert tr.n_calls == 1                                                           # nothing to ask the model
    with pytest.raises(ValueError):
        preview_masks(tr, info, (320, 180), 5000, {"1": {"points": [CLICK30], "labels": [1]}}, str(tmp_path / "f"))


def test_server_loop_answers_requests_and_survives_bad_ones(video):
    reqs = [json.dumps({"id": 1, "frame": 30, "objects": {"1": {"points": [CLICK30], "labels": [1]}}}), "this is not json",
            json.dumps({"id": 2, "frame": 99999, "objects": {"1": {"points": [CLICK30], "labels": [1]}}}),
            json.dumps({"id": 3, "frame": 5, "objects": {"1": {"points": [CLICK30], "labels": [1]}}}),     # square not in the frame yet
            json.dumps({"cmd": "quit"}), json.dumps({"id": 4, "frame": 30, "objects": {}})]
    out = io.StringIO()
    assert serve(video, "unused", analysis_width=320, stdin=io.StringIO("\n".join(reqs) + "\n"), stdout=out, tracker=ColorTracker((180, 320))) == 0
    evs = events(out.getvalue())
    assert [e["event"] for e in evs] == ["loading", "ready", "masks", "error", "error", "masks"]      # nothing after quit
    assert evs[1]["analysis_size"] == [320, 180]
    assert evs[2]["id"] == 1 and evs[2]["shape"] == [180, 320] and decode_mask(evs[2]["masks"]["1"], (180, 320)).sum() > 600
    assert evs[4]["id"] == 2 and "outside the video" in evs[4]["reason"]
    assert evs[5]["id"] == 3 and decode_mask(evs[5]["masks"]["1"], (180, 320)).sum() == 0              # an empty mask is reported as it is


def test_server_reports_a_model_that_cannot_be_loaded(video, tmp_path):
    out = io.StringIO()
    assert serve(video, str(tmp_path / "missing.pt"), stdin=io.StringIO(""), stdout=out) == 1
    assert [e["event"] for e in events(out.getvalue())] == ["loading", "fatal"]


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_real_worker_previews_a_click_with_sam2(video):
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "censor_video.py"), video, "--preview-server", "--sam2-checkpoint", CKPT,
                          "--analysis-width", "640"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, encoding="utf-8", cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1", TQDM_DISABLE="1"))
    try:
        p.stdin.write(json.dumps({"id": 1, "frame": 30, "objects": {"1": {"points": [CLICK30], "labels": [1]}}}) + "\n")
        p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        p.stdin.flush()
        out, _ = p.communicate(timeout=240)
    finally:
        if p.poll() is None:
            p.kill()
    evs = events(out)
    assert [e["event"] for e in evs] == ["loading", "ready", "masks"], evs
    assert evs[1]["tracker"]["checkpoint"] == "sam2.1_hiera_small.pt" and evs[2]["shape"] == [360, 640]
    m = decode_mask(evs[2]["masks"]["1"], tuple(evs[2]["shape"]))
    assert m.sum() > 100 and m[180, 100]                                             # a mask at the click (how much SAM 2 selects is its call)
    assert not m[20, 600]                                                            # ... and not the whole frame
    assert p.returncode == 0


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_preview_equals_what_the_job_gets_on_the_clicked_frame(video, tmp_path):
    """The point of the preview: the mask it shows for a click is the mask the processing job produces on that frame."""
    from src.tracking import SAM2Tracker
    info = probe_video(video)
    tracker = SAM2Tracker(CKPT)
    click = {"obj_id": 1, "frame_idx": 30, "x": CLICK30[0], "y": CLICK30[1], "label": 1}
    masks, skipped = preview_masks(tracker, info, (640, 360), 30, {"1": {"points": [CLICK30], "labels": [1]}}, str(tmp_path / "f"))
    assert not skipped
    shown = decode_mask(encode_mask(masks[1]), (360, 640))

    pts = str(tmp_path / "p.json")
    save_points(pts, info, [click])
    cfg = Config(input=video, output=str(tmp_path / "out.mp4"), points=pts, work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"),
                 chunk_size=40, analysis_width=640, max_frames=40, preset="ultrafast", extra={"keep_temp": True})
    assert Pipeline(cfg, tracker, None).run() in (0, 2)
    job = JobMasks(os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])).frame(30)["masks"][1]
    iou = (shown & job).sum() / max(1, (shown | job).sum())
    assert shown.sum() > 100 and iou > 0.95, (int(shown.sum()), int(job.sum()), float(iou))


# ------------------------------------------------------------------ kept masks of a run
class BoxDetector:
    def detect(self, bgr):
        return [Det(200.0, 20.0, 240.0, 50.0, 0.9, "FAKE")]


def run_job(video, tmp):
    pts = str(tmp / "p.json")
    save_points(pts, probe_video(video), [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = Config(input=video, output=str(tmp / "out.mp4"), points=pts, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"), chunk_size=40,
                 analysis_width=320, preset="ultrafast", fail_closed=True, extra={"keep_temp": True})
    assert Pipeline(cfg, ColorTracker((180, 320)), BoxDetector()).run() in (0, 2)
    return os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0]), pts


def test_job_masks_gives_the_cached_results_of_any_frame(video, tmp_path):
    jd, _ = run_job(video, tmp_path)
    jm = JobMasks(jd)
    assert jm.available_chunks() == 4 and jm.chunk_size == 40 and jm.total_frames == 160 and jm.analysis_size == (320, 180)
    fr = jm.frame(30)
    assert fr["shape"] == (180, 320) and set(fr["masks"]) == {1} and 600 < fr["masks"][1].sum() < 1100
    assert fr["boxes"] == [(200.0, 20.0, 240.0, 50.0, "FAKE", False)] and not fr["fc_regions"] and not fr["fc_full"]
    assert jm.frame(5)["masks"] == {}                                                # before the click
    lost = jm.frame(75)                                                              # object hidden: no mask, a fail-closed region
    assert lost["masks"] == {} and len(lost["fc_regions"]) == 1
    assert jm.frame(130)["masks"][1].sum() > 600                                     # another chunk
    assert jm.frame(160) is None and jm.frame(-1) is None
    os.remove(os.path.join(jd, "masks_00001.npz"))
    assert JobMasks(jd).frame(50) is None and JobMasks(jd).available_chunks() == 1
    with pytest.raises(OSError):
        JobMasks(str(tmp_path / "nowhere"))


# ------------------------------------------------------------------ the app
from test_gui import _widget_pos, open_clip, pump, qapp, shown_code, win  # noqa: E402,F401  (fixtures)

FAKE_SERVER = """
import sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from helpers import ColorTracker
from src.preview_server import serve
sys.exit(serve(sys.argv[1], 'unused', analysis_width=320, tracker=ColorTracker((180, 320))))
"""


def fake_server_command(tmp_path, video):
    script = tmp_path / "fake_preview_server.py"
    script.write_text(FAKE_SERVER.format(root=ROOT, tests=os.path.join(ROOT, "tests")), encoding="utf-8")
    exe = sys.executable.replace("pythonw.exe", "python.exe")
    return [exe, str(script), video]


def test_app_shows_the_mask_of_a_click_and_updates_it(win, tmp_path):
    open_clip(win)
    win.goto(30)
    pump(lambda: shown_code(win) == 30)
    win.preview_command = fake_server_command(tmp_path, win.test_video)
    win.act_live.setChecked(True)
    assert win.preview.is_on() and win.errors == []
    assert "대상을 클릭" in win.view.overlay()["note"]                          # no clicks on this frame yet
    win.view.handle_click(*_widget_pos(win, *CLICK30), 1)
    assert pump(lambda: bool((win.view.overlay() or {}).get("masks")), timeout=60), win.view.overlay()
    ov = win.view.overlay()
    assert ov["size"] == (320, 180) and len(ov["masks"]) == 1 and "이 프레임의 클릭 미리보기" in ov["note"]
    img = ov["masks"][0]
    assert img.pixelColor(50, 90).alpha() > 0 and img.pixelColor(5, 5).alpha() == 0  # tinted where the square is, clear elsewhere
    assert not win.view.grab().isNull()                                              # painting the overlay works

    win.new_object()                                                                 # a second object with only a negative click
    win.view.handle_click(*_widget_pos(win, 300, 40), 0)
    assert pump(lambda: "객체 2" in ((win.view.overlay() or {}).get("note") or ""), timeout=30)
    assert len(win.view.overlay()["masks"]) == 1

    win.goto(31)                                                                     # a frame without clicks: nothing to preview
    assert not win.view.overlay().get("masks")
    win.goto(30)                                                                     # back: the answer is remembered, no new request
    assert len(win.view.overlay()["masks"]) == 1

    win._set_job_running(True)                                                       # a job takes the GPU: the preview stops
    assert not win.act_live.isChecked() and not win.preview.is_on() and not win.act_live.isEnabled()
    win._set_job_running(False)
    assert win.act_live.isEnabled() and win.view.overlay() is None


def test_app_reports_a_preview_that_cannot_start(win, tmp_path):
    open_clip(win)
    win.preview_command = [sys.executable.replace("pythonw.exe", "python.exe"), "-c", "import sys; sys.exit(3)"]
    win.act_live.setChecked(True)
    assert pump(lambda: bool(win.errors), timeout=30)
    assert "시작하지 못했습니다" in win.errors[0] and not win.act_live.isChecked() and not win.preview.is_on()


def test_app_needs_a_video_and_a_checkpoint_for_the_preview(win, monkeypatch):
    win.act_live.setChecked(True)
    assert win.errors == ["먼저 영상을 여세요."] and not win.act_live.isChecked()
    open_clip(win)
    from src.app import main_window as mw
    from src.app.job_runner import ProcessSettings
    monkeypatch.setattr(mw, "load_last_settings", lambda: ProcessSettings(sam2_checkpoint="Z:/nope.pt"))
    win.act_live.setChecked(True)
    assert "체크포인트를 찾을 수 없습니다" in win.errors[-1] and not win.act_live.isChecked() and not win.preview.is_on()


def test_app_shows_the_masks_of_the_last_run(win, tmp_path):
    from src.app.job_runner import ProcessSettings
    open_clip(win)
    win.act_last.setChecked(True)                                                    # nothing kept yet
    assert not win.act_last.isChecked() and "보관된 마스크가 없습니다" in win.statusBar().currentMessage()

    _, pts = run_job(win.test_video, tmp_path)
    win.last_settings = ProcessSettings(work_dir=str(tmp_path / "work"))
    assert win._try_load_points(pts, quiet=True)
    win.act_last.setChecked(True)
    assert win.act_last.isChecked()
    win.goto(30)
    ov = win.view.overlay()
    assert len(ov["masks"]) == 1 and len(ov["boxes"]) == 1 and "지난 실행" in ov["note"] and "바뀌었습니다" not in ov["note"]
    win.goto(75)
    ov = win.view.overlay()
    assert not ov["masks"] and len(ov["regions"]) == 1                               # lost object: fail-closed area
    assert not win.view.grab().isNull()
    win.view.handle_click(*_widget_pos(win, 300, 100), 1)                            # edited since the run
    assert "포인트가 바뀌었습니다" in win.view.overlay()["note"]
    win.act_last.setChecked(False)
    assert win.view.overlay() is None
