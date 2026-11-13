"""A NudeNet box that was there and suddenly is not: the gap is flagged for review, and with fail-closed the last place stays
covered until the box is back (or det_lost_hold frames have passed). Detector-only job, fake detector."""
import json
import os

from src.detection import Det
from src.pipeline import Config, Pipeline

CS = 40


class GapDetector:
    """One fixed box on every frame except frames [a, b) - a gap longer than det_fill_gap that crosses a chunk boundary."""

    def __init__(self, a=70, b=96):
        self.a, self.b = a, b

    def detect(self, bgr):
        k = bgr.shape[1] / 640.0
        t = sum(1 << b for b in range(12) if bgr[int(5 * k):int(13 * k), int((5 + b * 10) * k):int((13 + b * 10) * k)].mean() > 127)
        return [] if self.a <= t < self.b else [Det(200.0, 20.0, 240.0, 50.0, 0.9, "FAKE")]


def run(tmp, video, det, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=None, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=CS, analysis_width=320, preset="ultrafast", crf=18, fail_closed=True)
    base.update(kw)
    cfg = Config(**base)
    rc = Pipeline(cfg, None, det).run()
    doc = json.load(open(os.path.join(cfg.log_dir, "synthetic_review.json"), encoding="utf-8"))
    return rc, doc["events"]


def of(events, typ):
    return [e for e in events if e["type"] == typ]


def test_gap_is_covered_and_flagged_with_fail_closed(video, tmp_path):
    rc, events = run(tmp_path, video, GapDetector())
    lost = of(events, "detection_lost")
    assert [e["frame"] for e in lost] == [73]                             # last seen 69, frames 70-72 are the ordinary hold
    assert lost[0]["severity"] == "warn" and lost[0]["extra"]["last_seen"] == 69
    assert lost[0]["action"] == "expanded_bbox_mosaic_up_to_60_frames"
    covered = [e["frame"] for e in of(events, "fail_closed_mosaic_applied")]
    assert covered == list(range(73, 96))                                 # up to the frame before the box is back, across the chunk
    assert all("detection_lost" in e["detail"] for e in of(events, "fail_closed_mosaic_applied"))
    assert rc == 0


def test_gap_inside_one_chunk_that_was_covered_is_only_info(video, tmp_path):
    rc, events = run(tmp_path, video, GapDetector(50, 62))
    lost = of(events, "detection_lost")
    assert [(e["frame"], e["severity"]) for e in lost] == [(53, "info")] and "12 frames" in lost[0]["detail"]
    assert [e["frame"] for e in of(events, "fail_closed_mosaic_applied")] == list(range(53, 62))


def test_window_ends_after_det_lost_hold(video, tmp_path):
    rc, events = run(tmp_path, video, GapDetector(70, 160), det_lost_hold=20)
    assert [e["frame"] for e in of(events, "detection_lost")] == [73]
    assert [e["frame"] for e in of(events, "fail_closed_mosaic_applied")] == list(range(73, 90))      # 69 + 20


def test_without_fail_closed_the_gap_is_flagged_but_not_covered(video, tmp_path):
    rc, events = run(tmp_path, video, GapDetector(), fail_closed=False)
    lost = of(events, "detection_lost")
    assert [(e["frame"], e["severity"], e["action"]) for e in lost] == [(73, "warn", "flagged_only")]
    assert of(events, "fail_closed_mosaic_applied") == []


def test_switched_off(video, tmp_path):
    rc, events = run(tmp_path, video, GapDetector(), det_lost_hold=0)
    assert of(events, "detection_lost") == [] and of(events, "fail_closed_mosaic_applied") == []
