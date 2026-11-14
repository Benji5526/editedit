"""Labels from an original / already-mosaicked pair of videos."""
import numpy as np

from src.detection import Det
from src.pair_labels import compare, diff_regions, load_labels, save_labels
from src.pipeline import Config, Pipeline


def test_diff_regions_finds_the_covered_block_and_ignores_noise():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    a = np.repeat(np.repeat(a[::8, ::8], 8, 0), 8, 1)                       # blocky picture
    b = np.clip(a.astype(int) + rng.integers(-4, 5, a.shape), 0, 255).astype(np.uint8)       # "compression noise"
    assert diff_regions(a, b)[0] == []
    b[100:180, 300:420] = 128                                               # a covered region
    boxes, changed = diff_regions(a, b)
    assert len(boxes) == 1 and 0.03 < changed < 0.08
    x1, y1, x2, y2 = boxes[0]
    assert abs(x1 - 300) <= 12 and abs(y1 - 100) <= 12 and abs(x2 - 420) <= 12 and abs(y2 - 180) <= 12


class OneBox:
    def detect(self, bgr):
        return [Det(100.0, 40.0, 180.0, 100.0, 0.9, "FAKE")]


def test_compare_recovers_the_regions_of_a_mosaicked_copy(video, tmp_path):
    cfg = Config(input=video, output=str(tmp_path / "cens.mp4"), points=None, work_dir=str(tmp_path / "work"), log_dir=str(tmp_path / "logs"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", crf=18, block=32, mosaic_style="solid", det_expand=0.0,
                 det_pad_px=0.0, max_frames=80)
    assert Pipeline(cfg, None, OneBox()).run() == 0
    doc = compare(video, cfg.output, width=640, max_offset=0)
    assert doc["frames_compared"] == 80 and doc["offset"] == 0 and doc["class"] == "censor"
    assert doc["stats"]["frames_with_regions"] >= 76 and doc["stats"]["frames_skipped_global_change"] == 0
    x1, y1, x2, y2 = doc["frames"]["40"][0]                                 # the box is (200..360, 80..200) in full-resolution pixels
    assert abs(x1 - 200) <= 16 and abs(y1 - 80) <= 16 and abs(x2 - 360) <= 16 and abs(y2 - 200) <= 16
    assert len(doc["stats"]["static_places"]) == 1 and not doc["stats"]["static_places_dropped"]      # it never moves: reported, kept
    assert compare(video, cfg.output, width=640, max_offset=0, drop_static=True)["stats"]["frames_with_regions"] == 0
    p = str(tmp_path / "l" / "labels.json")
    save_labels(p, doc)
    text = open(p, encoding="utf-8").read()
    assert load_labels(p)["frames"] == doc["frames"] and "cens.mp4" not in text and "synthetic" not in text and str(tmp_path) not in text
