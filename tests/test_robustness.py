"""Things that only break on real, long, messy inputs: non-ASCII paths, fractional frame rates, many tiny chunks."""
import json
import os
import shutil

import numpy as np
import pytest

from helpers import ColorTracker, make_video
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.video_io import FrameReader, probe_video

P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out" / "결과 영상.mp4"), points=points, work_dir=str(tmp / "작업 폴더"),
                log_dir=str(tmp / "로그"), chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18,
                extra={"keep_temp": False})
    base.update(kw)
    return Config(**base)


def test_korean_and_space_paths_end_to_end(video, tmp_path):
    folder = tmp_path / "한글 폴더 (테스트)"
    folder.mkdir()
    vid = str(folder / "내 영상 #1.mp4")
    shutil.copy(video, vid)
    pts = str(folder / "클릭 모음.json")
    save_points(pts, probe_video(vid), [P12])
    cfg = cfg_for(tmp_path, vid, pts)
    flag = {"stop": False}
    # cancel after chunk 1 and resume, so the overlap/prefix/ring files are written AND read back with non-ASCII directory names
    rc = Pipeline(cfg, ColorTracker((180, 320)), None, cancel_check=lambda: flag["stop"],
                  progress=lambda e: flag.update(stop=True) if e["event"] == "chunk_done" and e["chunk"] == 1 else None).run()
    assert rc == 3
    cfg.resume = True
    tr = ColorTracker((180, 320))
    assert Pipeline(cfg, tr, None).run() == 0, open(os.path.join(cfg.log_dir, "내 영상 #1_status.json"), encoding="utf-8").read()
    assert tr.n_calls == 2 and probe_video(cfg.output).nb_frames == 160
    doc = json.load(open(os.path.join(cfg.log_dir, "내 영상 #1_review.json"), encoding="utf-8"))
    assert doc["meta"]["input"]["path"].endswith("내 영상 #1.mp4")


@pytest.mark.parametrize("fps,chunk", [("60000/1001", 7), ("30000/1001", 13), ("24000/1001", 11)])
def test_fractional_frame_rates_keep_a_cfr_grid_over_many_odd_chunks(tmp_path, fps, chunk):
    vid = str(tmp_path / "ntsc.mp4")
    make_video(vid, n_frames=160, fps=fps)
    info = probe_video(vid)
    assert not info.is_vfr and info.fps_rational == fps
    pts = str(tmp_path / "p.json")
    save_points(pts, info, [P12])
    cfg = cfg_for(tmp_path, vid, pts, chunk_size=chunk)
    rc = Pipeline(cfg, ColorTracker((180, 320)), None).run()
    assert rc in (0, 2), open(os.path.join(cfg.log_dir, "ntsc_status.json")).read()                  # includes verify_output (CFR grid check)
    out = probe_video(cfg.output)
    assert out.nb_frames == 160 and not out.is_vfr
    assert abs(out.fps - info.fps) / info.fps < 1e-4          # the container may write the same rate as another fraction (e.g. 1200000/20021)
    dt = 1.0 / info.fps
    dev = np.abs((np.asarray(out.timestamps) - out.timestamps[0]) - np.arange(160) * dt)
    print(f"{fps} chunk={chunk}: max timestamp deviation {dev.max() * 1000:.4f} ms = {dev.max() / dt:.4f} frame")
    assert dev.max() < 5e-6, f"timestamps drift by {dev.max() * 1000:.4f} ms ({dev.max() / dt:.4f} frames)"        # ffprobe prints 1 microsecond
    assert abs(out.duration - 160 * dt) < 1.5 * dt


def test_frame_exact_timescale_values():
    from src.video_io import frame_exact_timescale
    for fr in ("60000/1001", "30000/1001", "24000/1001", "30/1", "25/1", "60/1", "50/1", "120/1", "1/1", "12/1"):
        n, d = (int(x) for x in fr.split("/"))
        ts = frame_exact_timescale(fr)
        assert ts >= 10000 or n >= 10000
        assert (ts * d) % n == 0, f"{fr}: a frame would last {ts * d / n} ticks"        # integer ticks per frame
        assert ts < 2 ** 31
