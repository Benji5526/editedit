import numpy as np
import pytest

from src.mosaic import (boxes_to_mask, clip_box, dilate_mask, expand_box, mask_bbox, merge_masks,
                        pixelate_region, temporal_union, upscale_mask)


def sq(shape, x1, y1, x2, y2):
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


def test_merge_keeps_every_object():
    shape = (100, 200)
    a, b, c = sq(shape, 5, 5, 20, 20), sq(shape, 100, 60, 120, 90), sq(shape, 150, 10, 160, 15)
    m = merge_masks([a, None, b, c], shape)
    assert (m & a).sum() == a.sum() and (m & b).sum() == b.sum() and (m & c).sum() == c.sum()
    assert m.sum() == a.sum() + b.sum() + c.sum()


def test_merge_rejects_wrong_shape():
    with pytest.raises(ValueError):
        merge_masks([np.zeros((3, 3), bool)], (4, 4))


def test_bbox_and_empty():
    assert mask_bbox(np.zeros((10, 10), bool)) is None
    assert mask_bbox(sq((50, 60), 10, 20, 30, 40)) == (10, 20, 30, 40)


def test_expand_box_clips():
    assert expand_box((10, 10, 20, 20), 0.5, 2, (100, 100)) == (3, 3, 27, 27)
    assert expand_box((0, 0, 10, 10), 1.0, 0, (15, 15)) == (0, 0, 15, 15)
    assert clip_box((200, 200, 300, 300), (100, 100)) is None


def test_boxes_to_mask():
    m = boxes_to_mask([(1, 1, 4, 4), None, (50, 50, 60, 60)], (10, 10))
    assert m.sum() == 9


def test_dilate_grows_by_pixels():
    m = sq((100, 100), 40, 40, 60, 60)
    d = dilate_mask(m, 5)
    assert mask_bbox(d) == (35, 35, 65, 65)
    assert (d & m).sum() == m.sum()
    assert dilate_mask(m, 0).sum() == m.sum()


def test_temporal_union_prevents_sudden_shrink():
    big, small = sq((50, 50), 10, 10, 40, 40), sq((50, 50), 20, 20, 25, 25)
    seq = [big, small, small, None, small]
    u = temporal_union(seq, 1, 1)
    assert (u & big).sum() == big.sum()           # neighbour's big mask is kept
    assert temporal_union([None, None], 0, 1) is None
    assert temporal_union(seq, 4, 0).sum() == small.sum()


def test_upscale_mask():
    m = sq((10, 20), 2, 2, 6, 6)
    up = upscale_mask(m, (40, 80))
    assert up.shape == (40, 80)
    assert abs(int(up.sum()) - 16 * 16) < 16 * 8  # ~4x4 -> ~16x16 with soft edge


def test_pixelate_only_inside_mask_and_blocky():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    orig = frame.copy()
    mask = sq((120, 160), 33, 21, 97, 83)
    pixelate_region(frame, mask, 16)
    assert np.array_equal(frame[~mask], orig[~mask])               # untouched outside
    assert not np.array_equal(frame[mask], orig[mask])
    # inside a fully-masked, grid-aligned block all pixels are identical
    blk = frame[32:48, 48:64]
    assert (blk == blk[0, 0]).all()


def test_pixelate_grid_is_anchored_to_image_origin():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    f1, f2 = base.copy(), base.copy()
    pixelate_region(f1, sq((100, 100), 20, 20, 70, 70), 10)
    pixelate_region(f2, sq((100, 100), 23, 27, 66, 69), 10)   # smaller, shifted mask
    inter = sq((100, 100), 30, 30, 60, 60)
    assert np.array_equal(f1[inter], f2[inter])               # same blocks => no shimmer when mask moves


def test_pixelate_empty_mask_noop_and_shape_check():
    f = np.full((10, 10, 3), 7, np.uint8)
    pixelate_region(f, np.zeros((10, 10), bool), 4)
    assert (f == 7).all()
    with pytest.raises(ValueError):
        pixelate_region(f, np.zeros((5, 5), bool), 4)
