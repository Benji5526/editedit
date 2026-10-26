import numpy as np
import pytest

from src.detection import Det, NudenetDetector, DEFAULT_CLASSES, fill_detections, iou, supported_labels


def d(x1=10, y1=10, x2=50, y2=50, score=0.8, cls="A"):
    return Det(x1, y1, x2, y2, score, cls)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_gap_is_bridged_and_flagged_filled():
    per = [[d()], [], [], [d(x1=13, x2=53)], []]
    out, carry = fill_detections(per, 100, [], max_gap=4, hold=0)
    assert [len(x) for x in out] == [1, 1, 1, 1, 0]
    assert out[1][0].filled and out[2][0].filled and not out[0][0].filled


def test_gap_longer_than_max_not_bridged():
    per = [[d()], [], [], [], [], [d()]]
    out, _ = fill_detections(per, 0, [], max_gap=2, hold=0)
    assert [len(x) for x in out] == [1, 0, 0, 0, 0, 1]


def test_hold_after_last_detection_and_carry():
    per = [[d()], [], [], [], []]
    out, carry = fill_detections(per, 0, [], max_gap=2, hold=2)
    assert [len(x) for x in out] == [1, 1, 1, 0, 0]
    # a detection at the last frame must be carried so the next chunk can keep holding it
    out2, carry2 = fill_detections([[], [d()]], 0, [], 2, 3)
    assert carry2 and carry2[0]["frame"] == 1
    nxt, _ = fill_detections([[], [], []], 2, carry2, 2, 3)
    assert [len(x) for x in nxt] == [1, 1, 1]


def test_carry_bridges_into_next_chunk():
    carry = [{"cls": "A", "frame": 9, "box": [10, 10, 50, 50], "score": 0.9}]
    out, _ = fill_detections([[], [], [d()]], 10, carry, max_gap=4, hold=0)
    assert [len(x) for x in out] == [1, 1, 1]


def test_different_class_not_merged():
    out, _ = fill_detections([[d(cls="A")], [], [d(cls="B")]], 0, [], 4, 0)
    assert [len(x) for x in out] == [1, 0, 1]


def test_none_frames_when_detect_every_gt_1():
    out, _ = fill_detections([[d()], None, None, [d()]], 0, [], 4, 0)
    assert [len(x) for x in out] == [1, 1, 1, 1]


def test_default_classes_are_real_nudenet_labels():
    labels = supported_labels()
    assert len(labels) == 18
    assert set(DEFAULT_CLASSES) <= set(labels)


def test_unsupported_class_rejected():
    with pytest.raises(ValueError):
        NudenetDetector(classes=["BICYCLE"])


def test_real_model_runs_on_neutral_images():
    """API smoke test with the bundled 320n model: runs, returns a list, boxes stay inside the image.
    Says nothing about detection quality."""
    det = NudenetDetector(threshold=0.05, tiles=2)
    img = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
    for o in out:
        assert 0 <= o.x1 < o.x2 <= 640 and 0 <= o.y1 < o.y2 <= 360 and o.cls in DEFAULT_CLASSES
    assert det.detect(np.zeros((100, 300, 3), np.uint8)) is not None   # non-square input
