"""--plan-only over the CLI protocol, and incremental re-processing with the real SAM 2 checkpoint."""
import os

import numpy as np
import pytest

from helpers import ENTER, H, HIDDEN, SQ, W, square_x
from src.app.job_runner import PROJECT_ROOT, ProcessSettings, build_args
from src.points_io import save_points
from src.video_io import probe_video
from test_job_runner import cli_args, run_cli

CKPT = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_small.pt")


def plan_of(evs):
    return next(e for e in evs if e["event"] == "plan")


def test_build_args_keep_chunks_and_plan_only():
    st = ProcessSettings(output="o.mp4", keep_chunks=True)
    a = build_args("v.mp4", None, st, "c.flag", plan_only=True)
    assert "--keep-temp" in a and "--plan-only" in a
    assert "--keep-temp" not in build_args("v.mp4", None, ProcessSettings(output="o.mp4", keep_chunks=False), "c.flag")
    assert "--plan-only" not in build_args("v.mp4", None, st, "c.flag")


def test_plan_only_describes_reuse_without_processing(video, tmp_path):
    # nothing kept yet
    rc, evs, _, err = run_cli(cli_args(video, tmp_path, "--plan-only", "--resume"))
    assert rc == 0, err[-400:]
    p = plan_of(evs)
    assert p["job_found"] is False and p["reusable_chunks"] == 0 and "no kept intermediate chunks" in p["reason"]
    assert evs[-1]["status"] == "planned" and not os.path.exists(str(tmp_path / "out.mp4"))      # nothing was processed

    # a finished run that kept its chunks
    rc, _, _, err = run_cli(cli_args(video, tmp_path, "--keep-temp"))
    assert rc == 0, err[-400:]
    rc, evs, _, _ = run_cli(cli_args(video, tmp_path, "--plan-only", "--resume", "--keep-temp"))
    p = plan_of(evs)
    assert p["can_resume"] and p["reusable_chunks"] == 4 and p["frames_reused"] == 160 and p["first_changed_chunk"] is None
    assert "nothing changed" in p["reason"]

    # other settings => the old chunks cannot be used
    rc, evs, _, _ = run_cli(cli_args(video, tmp_path, "--plan-only", "--resume", "--block", "48"))
    p = plan_of(evs)
    assert p["job_found"] and p["can_resume"] is False and "from scratch" in p["reason"]


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_real_sam2_reclick_reuses_chunks_and_improves_iou(video, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import cv2
    from src.pipeline import Config, Pipeline
    from src.tracking import SAM2Tracker, unpack_mask

    info = probe_video(video)
    p1 = str(tmp_path / "p1.json")
    save_points(p1, info, [{"obj_id": 1, "frame_idx": 20, "x": square_x(20) + 30, "y": 180, "label": 1}])
    p2 = str(tmp_path / "p2.json")
    save_points(p2, info, [{"obj_id": 1, "frame_idx": 20, "x": square_x(20) + 30, "y": 180, "label": 1},
                           {"obj_id": 1, "frame_idx": 90, "x": square_x(90) + 30, "y": 180, "label": 1}])
    tracker = SAM2Tracker(CKPT, device="cuda")
    got = {}
    calls = [0]
    orig = tracker.track_chunk

    def spy(frame_dir, n, inits):
        out = orig(frame_dir, n, inits)
        calls[0] += 1
        P = n - 40
        for o, fr in out.items():
            for li, pm in fr.items():
                if li >= P:
                    got[(calls[0], li - P)] = unpack_mask(pm, (180, 320))
        return out

    tracker.track_chunk = spy

    def gt(t):
        m = np.zeros((H, W), bool)
        x0 = square_x(t)
        xa, xb = max(0, x0), min(W, x0 + SQ)
        if ENTER <= t not in HIDDEN and xb > xa:
            m[150:150 + SQ, xa:xb] = True
        return cv2.resize(m.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST).astype(bool)

    def iou(m, t):
        g = gt(t)
        return (m & g).sum() / max(1, (m | g).sum())

    def cfg_for(points, resume):
        return Config(input=video, output=str(tmp_path / "o.mp4"), points=points, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                      chunk_size=40, analysis_width=320, preset="ultrafast", fail_closed=True, resume=resume, extra={"keep_temp": True})

    assert Pipeline(cfg_for(p1, False), tracker, None).run() in (0, 2)
    first_masks = {(c - 1) * 40 + li: m for (c, li), m in got.items()}
    iou_first = [iou(first_masks[t], t) for t in range(90, 143) if gt(t).any()]
    got.clear()
    calls[0] = 0
    evs = []
    assert Pipeline(cfg_for(p2, True), tracker, None, progress=evs.append).run() in (0, 2)
    assert evs[0]["reused_chunks"] == 2 and evs[0]["points_changed"] and calls[0] == 2          # only chunks 3 and 4 hit SAM 2
    second_masks = {(c - 1) * 40 + 80 + li: m for (c, li), m in got.items()}
    iou_second = [iou(second_masks[t], t) for t in range(90, 143) if gt(t).any()]
    print(f"\nIoU frames 90-142: before re-click mean={np.mean(iou_first):.3f}  after re-click mean={np.mean(iou_second):.3f}")
    assert np.mean(iou_second) > 0.8 and np.mean(iou_second) >= np.mean(iou_first) - 0.05
