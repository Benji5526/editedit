"""Outline of the mosaic on detected regions: box or ellipse (render-only), half-ellipse for face parts."""
import numpy as np

from src.detection import Det, FACE_ABOVE, FACE_BELOW, round_face_part
from src.pipeline import Config, Pipeline, config_hash, render_hash
from src.video_io import FrameReader, probe_video


def test_half_ellipse_keeps_the_eye_edge_and_rounds_the_far_side():
    poly = [[100, 100], [200, 100], [200, 180], [100, 180]]                 # FACE_BELOW: eye edge first (y = 100)
    r = np.array(round_face_part(poly, FACE_BELOW))
    assert r[:, 1].min() >= 100 - 1e-6 and abs(r[:, 1].max() - 180) < 1e-6  # nothing above the eye edge, reaches the chin edge
    assert r[0].tolist() == [200.0, 100.0] and np.allclose(r[-1], [100, 100])
    assert abs(r[:, 0].min() - 100) < 1e-6 and abs(r[:, 0].max() - 200) < 1e-6
    up = np.array(round_face_part(poly, FACE_ABOVE))                        # FACE_ABOVE: eye edge is the last two corners (y = 180)
    assert up[:, 1].max() <= 180 + 1e-6 and abs(up[:, 1].min() - 100) < 1e-6


class Boxes:
    def detect(self, bgr):
        return [Det(100.0, 40.0, 180.0, 100.0, 0.9, "FAKE"),
                Det(220.0, 90.0, 290.0, 150.0, 0.9, FACE_BELOW, poly=[[220, 90], [290, 90], [290, 150], [220, 150]])]


def frames(path):
    with FrameReader(probe_video(path)) as r:
        return list(r)


def test_ellipse_leaves_the_corners_and_is_a_render_only_setting(video, tmp_path):
    def cfg(shape, **kw):
        return Config(input=video, output=str(tmp_path / "o.mp4"), points=None, work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"),
                      chunk_size=40, analysis_width=320, preset="ultrafast", crf=18, block=32, det_shape=shape, det_expand=0.0, det_pad_px=0.0,
                      max_frames=40, extra={"keep_temp": True}, **kw)
    src = frames(video)[:40]
    a = cfg("box")
    assert Pipeline(a, None, Boxes()).run() == 0
    box = frames(a.output)
    b = cfg("ellipse", resume=True)
    assert config_hash(a, probe_video(video)) == config_hash(b, probe_video(video)) and render_hash(a) != render_hash(b)

    class Never:
        def detect(self, bgr):
            raise AssertionError("the detector must not run again")
    assert Pipeline(b, None, Never()).run() == 0
    oval = frames(b.output)

    def diff(x, y, x1, y1, x2, y2):                                         # full-resolution pixels (analysis x 2)
        return float(np.abs(x[y1:y2, x1:x2].astype(int) - y[y1:y2, x1:x2].astype(int)).mean())
    f = 20
    centre, corner = (250, 120, 310, 160), (202, 82, 214, 92)
    assert diff(box[f], src[f], *centre) > 3 and diff(oval[f], src[f], *centre) > 3          # both cover the middle
    assert diff(box[f], src[f], *corner) > 3 and diff(oval[f], src[f], *corner) < 3          # only the box covers the corner
    eye_edge, chin_corner = (500, 182, 520, 192), (442, 288, 452, 298)                        # face part: straight at the eyes
    assert diff(oval[f], src[f], *eye_edge) > 3 and diff(box[f], src[f], *chin_corner) > 3 and diff(oval[f], src[f], *chin_corner) < 3
