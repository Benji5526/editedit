import json
import os

import cv2
import numpy as np
import pytest

from helpers import ColorTracker
from src.app.job_runner import ProcessSettings, build_args
from src.mosaic import STYLES, apply_mosaic, blur_region, solid_region
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.video_io import FrameReader, probe_video


def textured(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def rect(shape, x1, y1, x2, y2):
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


def hf_energy(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_blur_changes_only_the_mask_and_removes_detail():
    f = textured()
    orig = f.copy()
    m = rect((120, 160), 40, 30, 120, 90)
    blur_region(f, m, 16)
    assert np.array_equal(f[~m], orig[~m])
    assert hf_energy(f[40:80, 50:110]) < 0.05 * hf_energy(orig[40:80, 50:110])


def test_blur_strength_matters_and_edges_work():
    a, b = textured(), textured()
    m = rect((120, 160), 0, 0, 160, 120)                                   # whole frame, touching all borders
    blur_region(a, m, 4)
    blur_region(b, m, 24)
    assert hf_energy(b) < hf_energy(a)
    c = textured()
    assert blur_region(c, np.zeros((120, 160), bool), 16) is c and np.array_equal(c, textured())      # empty mask: no-op
    with pytest.raises(ValueError):
        blur_region(c, np.zeros((5, 5), bool), 16)


def test_solid_leaves_nothing_of_the_original():
    f = textured()
    orig = f.copy()
    m = rect((120, 160), 10, 10, 60, 60)
    solid_region(f, m)
    assert (f[m] == 0).all() and np.array_equal(f[~m], orig[~m])
    with pytest.raises(ValueError):
        solid_region(f, np.zeros((5, 5), bool))


def test_dispatch_and_unknown_style():
    assert STYLES == ("pixelate", "blur", "solid")
    m = rect((120, 160), 20, 20, 100, 80)
    outs = {}
    for style in STYLES:
        f = textured()
        assert apply_mosaic(f, m, style, 16) is f
        outs[style] = f
    assert not np.array_equal(outs["pixelate"], outs["blur"]) and (outs["solid"][m] == 0).all()
    with pytest.raises(ValueError, match="unknown mosaic style"):
        apply_mosaic(textured(), m, "emboss", 16)


def test_build_args_carries_the_style():
    a = build_args("v.mp4", None, ProcessSettings(output="o.mp4", mosaic_style="solid"), "c.flag")
    assert a[a.index("--mosaic-style") + 1] == "solid"


# ------------------------------------------------------------------ through the pipeline
P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}


def run_style(video, tmp_path, style):
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(video), [P12])
    cfg = Config(input=video, output=str(tmp_path / f"{style}.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, mosaic_style=style,
                 extra={"keep_temp": False})
    return cfg, Pipeline(cfg, ColorTracker((180, 320)), None).run()


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def test_pipeline_applies_the_selected_style(video, tmp_path):
    src = frames(video)
    t = 100                                                                 # the square is at x = 420..480, y = 150..210
    region = (slice(160, 200), slice(430, 470))
    energies = {}
    for style in STYLES:
        cfg, rc = run_style(video, tmp_path, style)
        assert rc == 0
        out = frames(cfg.output)
        energies[style] = hf_energy(out[t][region])
        # everything far from the object is untouched, whatever the style
        assert np.abs(out[t][300:350, 0:100].astype(int) - src[t][300:350, 0:100].astype(int)).mean() < 2.5
    assert energies["solid"] < 5 and float(frames(str(tmp_path / "solid.mp4"))[t][region].mean()) < 12     # black
    assert energies["blur"] < 0.3 * hf_energy(src[t][region])
    assert energies["pixelate"] < 0.3 * hf_energy(src[t][region])


def test_unknown_style_fails_cleanly(video, tmp_path):
    cfg, rc = run_style(video, tmp_path, "emboss")
    assert rc == 1
    st = json.load(open(os.path.join(cfg.log_dir, "synthetic_status.json")))
    assert st["status"] == "failed" and "mosaic style" in st["reason"] and not os.path.exists(cfg.output)


def test_style_change_invalidates_kept_chunks(video, tmp_path):
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(video), [P12])

    def cfg(style, resume=False):
        return Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                      chunk_size=40, analysis_width=320, preset="ultrafast", mosaic_style=style, resume=resume, extra={"keep_temp": True})
    assert Pipeline(cfg("pixelate"), ColorTracker((180, 320)), None).run() == 0
    assert Pipeline(cfg("solid", resume=True), ColorTracker((180, 320)), None).run() == 1          # chunks were rendered with another style
