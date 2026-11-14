"""Overlapping detection regions are merged into one shape."""
import numpy as np
import cv2

from src.mosaic import merge_overlapping


def blob_count(m):
    return cv2.connectedComponents(m.astype(np.uint8))[0] - 1


def test_two_overlapping_ellipses_become_one_convex_shape():
    m = np.zeros((200, 300), np.uint8)
    cv2.ellipse(m, (100, 80), (50, 30), 0, 0, 360, 1, -1)
    cv2.ellipse(m, (170, 120), (50, 30), 0, 0, 360, 1, -1)
    out = merge_overlapping(m)
    assert blob_count(out) == 1 and (out >= m).all() and out.sum() > m.sum()      # never less, and the notch is filled
    assert m[80, 160] == 0 and out[98, 150] == 1 and out[70, 150] == 1            # the dent between the two is covered
    hull = cv2.convexHull(cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    assert abs(cv2.contourArea(hull) - out.sum()) / out.sum() < 0.03              # the result is convex


def test_separate_regions_and_single_shapes_are_left_alone():
    m = np.zeros((200, 300), np.uint8)
    cv2.ellipse(m, (60, 60), (40, 25), 0, 0, 360, 1, -1)
    m[120:170, 200:280] = 1
    out = merge_overlapping(m)
    assert blob_count(out) == 2 and abs(int(out.sum()) - int(m.sum())) <= 0.02 * m.sum()
    assert merge_overlapping(np.zeros((10, 10), np.uint8)).sum() == 0
