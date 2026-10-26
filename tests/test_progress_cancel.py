"""Pipeline-level progress events and safe cancellation (colour-threshold tracker stand-in, no GPU needed)."""
import json
import os

import numpy as np
import pytest

from helpers import ColorTracker
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.video_io import FrameReader, probe_video

CS = 40


@pytest.fixture()
def points(video, tmp_path):
    p = str(tmp_path / "points.json")
    save_points(p, probe_video(video), [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    return p


def cfg_for(tmp_path, video, points, **kw):
    base = dict(input=video, output=str(tmp_path / "out" / "result.mp4"), points=points, work_dir=str(tmp_path / "work"),
                log_dir=str(tmp_path / "logs"), chunk_size=CS, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast",
                crf=18, extra={"keep_temp": False})
    base.update(kw)
    return Config(**base)


def status(cfg):
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    return json.load(open(os.path.join(cfg.log_dir, f"{stem}_status.json"), encoding="utf-8"))


def decode(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def test_progress_events_are_ordered_and_bounded(video, points, tmp_path):
    evs = []
    cfg = cfg_for(tmp_path, video, points)
    assert Pipeline(cfg, ColorTracker((180, 320)), None, progress=evs.append).run() == 0
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "start" and kinds.count("chunk_done") == 4
    start = evs[0]
    assert (start["total_frames"], start["chunks"], start["done_frames"], start["resume_from_chunk"]) == (160, 4, 0, 0)
    stages = {e["stage"] for e in evs if e["event"] == "progress"}
    assert stages == {"analyze", "track", "render"}
    done_prog = [e["done_frames"] for e in evs if e["event"] == "progress"]
    assert done_prog == sorted(done_prog)                          # never goes backwards
    assert all(0 <= e["chunk_frac"] <= 1 for e in evs if e["event"] == "progress")
    assert [e["done_frames"] for e in evs if e["event"] == "chunk_done"] == [40, 80, 120, 160]
    # within a chunk the stages come in order
    first_chunk = [e["stage"] for e in evs if e["event"] == "progress" and e["chunk"] == 0]
    assert first_chunk.index("track") > first_chunk.index("analyze") and first_chunk[-1] == "render"


def test_broken_progress_listener_does_not_break_the_job(video, points, tmp_path):
    def bad(_):
        raise RuntimeError("listener bug")
    cfg = cfg_for(tmp_path, video, points)
    assert Pipeline(cfg, ColorTracker((180, 320)), None, progress=bad).run() == 0
    assert probe_video(cfg.output).nb_frames == 160


def test_cancel_before_start_is_clean(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    assert Pipeline(cfg, ColorTracker((180, 320)), None, cancel_check=lambda: True).run() == 3
    assert status(cfg)["status"] == "cancelled" and not os.path.exists(cfg.output)


def test_cancel_between_chunks_then_resume_matches_clean_run(video, points, tmp_path):
    (tmp_path / "clean").mkdir()
    (tmp_path / "res").mkdir()
    clean = cfg_for(tmp_path / "clean", video, points)
    Pipeline(clean, ColorTracker((180, 320)), None).run()

    cfg = cfg_for(tmp_path / "res", video, points)
    flag = {"stop": False}
    rc = Pipeline(cfg, ColorTracker((180, 320)), None,
                  progress=lambda e: flag.update(stop=True) if e["event"] == "chunk_done" and e["chunk"] == 1 else None,
                  cancel_check=lambda: flag["stop"]).run()
    assert rc == 3 and status(cfg)["status"] == "cancelled" and not os.path.exists(cfg.output)
    assert not os.path.exists(cfg.output + ".partial")
    job = os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])
    assert os.path.isfile(os.path.join(job, "seg_00001.mp4")) and not os.path.exists(os.path.join(job, "seg_00002.mp4"))

    cfg.resume = True
    tr = ColorTracker((180, 320))
    evs = []
    assert Pipeline(cfg, tr, None, progress=evs.append).run() == 0
    assert tr.n_calls == 2 and evs[0]["resume_from_chunk"] == 2 and evs[0]["done_frames"] == 80
    a, b = decode(clean.output), decode(cfg.output)
    assert len(a) == len(b) == 160
    assert max(float(np.abs(x.astype(int) - y.astype(int)).mean()) for x, y in zip(a, b)) < 0.5


def test_cancel_in_the_middle_of_a_chunk_discards_that_chunk(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    flag = {"stop": False}

    def on_progress(e):
        if e["event"] == "progress" and e["chunk"] == 1 and e["stage"] == "render" and e["chunk_frac"] > 0.8:
            flag["stop"] = True

    assert Pipeline(cfg, ColorTracker((180, 320)), None, progress=on_progress, cancel_check=lambda: flag["stop"]).run() == 3
    job = os.path.join(cfg.work_dir, os.listdir(cfg.work_dir)[0])
    done = json.load(open(os.path.join(job, "job.json")))["chunks"]
    assert sorted(done) == ["0"]                                  # chunk 1 was in flight: not recorded as done
    cfg.resume = True
    tr = ColorTracker((180, 320))
    assert Pipeline(cfg, tr, None).run() == 0
    assert tr.n_calls == 3 and probe_video(cfg.output).nb_frames == 160
    assert not os.path.isdir(job)                                 # job directory is removed after a successful finish
