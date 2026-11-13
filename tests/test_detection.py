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


# ---- a box the detector stops finding ("lost"): reported forward in time so the caller can flag / keep covering it
def lost_run(per, lost_hold, start=0, carry=None, hold=1, max_gap=2):
    lost = []
    out, carry = fill_detections(per, start, carry or [], max_gap=max_gap, hold=hold, lost_hold=lost_hold, lost=lost)
    return out, carry, lost


def test_gap_longer_than_max_gap_is_reported_until_the_box_is_detected_again():
    per = [[d()], [d()], [d()]] + [[] for _ in range(8)] + [[d(x1=14, x2=54)], [d(x1=14, x2=54)]]
    out, _, lost = lost_run(per, lost_hold=30)
    assert [L["frame"] for L in lost] == [4, 5, 6, 7, 8, 9, 10]          # frame 3 is the ordinary hold, 11 is the re-detection
    assert [L["first"] for L in lost] == [True] + [False] * 6
    assert all(L["last_seen"] == 2 and L["box"] == [10, 10, 50, 50] for L in lost)


def test_lost_window_ends_after_lost_hold_frames():
    per = [[d()], [d()], [d()]] + [[] for _ in range(20)]
    _, carry, lost = lost_run(per, lost_hold=6)
    assert [L["frame"] for L in lost] == [4, 5, 6, 7, 8]                # last seen 2 -> frames 3..8, 3 is the hold
    assert carry == []


def test_a_box_seen_only_once_or_twice_is_not_reported():
    per = [[d()], [d()]] + [[] for _ in range(10)]
    assert lost_run(per, lost_hold=6)[2] == []


def test_lost_window_continues_in_the_next_chunk_and_is_reported_once():
    c1 = [[d()], [d()], [d()], [], [], []]
    _, carry, lost1 = lost_run(c1, lost_hold=8)
    assert [L["frame"] for L in lost1] == [4, 5] and lost1[0]["first"] and carry and carry[0]["n"] == 3
    _, carry2, lost2 = lost_run([[] for _ in range(6)], lost_hold=8, start=6, carry=carry)
    assert [L["frame"] for L in lost2] == [6, 7, 8, 9, 10] and not any(L["first"] for L in lost2)
    assert carry2 == []
    # ... and stops in the next chunk as soon as the box is back
    _, carry3, lost3 = lost_run([[], [d()], [d()], [], [], []], lost_hold=8, start=6, carry=carry)
    assert [L["frame"] for L in lost3] == [6, 10, 11]                 # 6: still lost; 7-8 back (keeps its history); lost again
    assert [L["first"] for L in lost3] == [False, True, False]


def test_recovery_inside_the_chunk_is_noted_on_the_first_entry():
    per = [[d()], [d()], [d()]] + [[] for _ in range(5)] + [[d()]]
    _, _, lost = lost_run(per, lost_hold=30)
    assert lost[0]["first"] and lost[0]["recovered_after"] == 5
    _, _, lost = lost_run([[d()], [d()], [d()]] + [[] for _ in range(5)], lost_hold=30)
    assert "recovered_after" not in lost[0]


def test_a_detection_somewhere_else_does_not_end_the_lost_window():
    far = d(x1=400, y1=400, x2=440, y2=440)
    per = [[d()], [d()], [d()]] + [[far] for _ in range(6)]
    _, _, lost = lost_run(per, lost_hold=5)
    assert [L["frame"] for L in lost] == [4, 5, 6, 7]


def test_no_lost_reports_unless_asked():
    per = [[d()], [d()], [d()]] + [[] for _ in range(10)]
    out, carry = fill_detections(per, 0, [], max_gap=2, hold=1)
    assert [len(x) for x in out] == [1, 1, 1, 1] + [0] * 9


# ---- faces: whole-face boxes cut at the eye line
from src.detection import EXACT_CLASSES, FACE_ABOVE, FACE_BELOW, face_parts  # noqa: E402


def test_face_below_eyes_uses_the_landmarks_and_leaves_the_eyes_out():
    face = Det(100, 100, 200, 240, 0.9, "FACE_FEMALE")
    marks = [(102, 104, 198, 236, (130, 150), (170, 154), 0.95)]          # box, right eye, left eye, score
    (below,) = face_parts([face], marks, "below_eyes", 640, 360)
    assert below.cls == FACE_BELOW and below.cls in EXACT_CLASSES
    assert below.y1 > 154 + 5 and below.y2 >= 240 and below.x1 <= 100 and below.x2 >= 200     # starts under the lower eye
    (above,) = face_parts([face], marks, "above_eyes", 640, 360)
    assert above.cls == FACE_ABOVE and above.y2 < 150 - 5 and above.y1 <= 100                  # ends over the upper eye


def test_face_without_landmarks_is_cut_by_proportion_and_extra_landmark_faces_are_used():
    face = Det(100, 100, 200, 240, 0.9, "FACE_MALE")
    (below,) = face_parts([face], [], "below_eyes", 640, 360)
    assert 100 + 0.40 * 140 < below.y1 < 100 + 0.60 * 140
    other = (400, 50, 460, 130, (418, 80), (442, 80), 0.9)                 # a face only the landmark model found
    parts = face_parts([face], [other], "below_eyes", 640, 360)
    assert len(parts) == 2 and any(p.x1 > 380 and p.y1 > 80 for p in parts)


def test_face_part_follows_a_head_lying_on_its_side():
    # eyes one above the other (head rotated 90 degrees), nose to the right of them: "below the eyes" is to the right
    marks = [(100, 100, 240, 200, (150, 130), (150, 170), 0.9, (175, 150))]
    (below,) = face_parts([], marks, "below_eyes", 640, 360)
    assert below.poly and len(below.poly) == 4 and below.x1 > 150 + 5 and below.x2 > 220       # starts right of the eyes
    assert below.y1 < 110 and below.y2 > 190                                                  # and spans the face's width
    (above,) = face_parts([], marks, "above_eyes", 640, 360)
    assert above.x2 < 150 - 5


def test_gap_filling_keeps_the_polygon():
    a = Det(10, 10, 50, 50, 0.9, "FACE_BELOW_EYES", poly=[[10, 10], [50, 10], [50, 50], [10, 50]])
    b = Det(20, 10, 60, 50, 0.9, "FACE_BELOW_EYES", poly=[[20, 10], [60, 10], [60, 50], [20, 50]])
    out, _ = fill_detections([[a], [], [], [], [b], []], 0, [], max_gap=4, hold=1)
    assert out[2][0].filled and out[2][0].poly[0] == [15.0, 10.0] and out[5][0].poly == b.poly
