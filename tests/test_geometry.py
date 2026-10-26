import pytest

from src.app.geometry import fit_rect, frame_to_widget, widget_to_frame

FW, FH = 3840, 2160


def test_wide_widget_has_side_bars():
    f = fit_rect(1600, 600, FW, FH)
    assert f.scale == pytest.approx(600 / 2160)
    assert f.oy == pytest.approx(0) and f.ox == pytest.approx((1600 - FW * f.scale) / 2)
    assert f.ox > 0


def test_tall_widget_has_top_bottom_bars():
    f = fit_rect(800, 900, FW, FH)
    assert f.scale == pytest.approx(800 / 3840) and f.ox == pytest.approx(0) and f.oy > 0


def test_centre_maps_to_centre_and_round_trip():
    for w, h in [(1600, 600), (800, 900), (1280, 720), (333, 777)]:
        f = fit_rect(w, h, FW, FH)
        c = widget_to_frame(f, w / 2, h / 2, FW, FH)
        assert c == pytest.approx((FW / 2, FH / 2), abs=1e-6)
        for fx, fy in [(10, 20), (1920, 1080), (3800, 2100)]:
            wx, wy = frame_to_widget(f, fx, fy)
            assert widget_to_frame(f, wx, wy, FW, FH) == pytest.approx((fx, fy), abs=1e-6)


def test_click_in_letterbox_is_rejected_not_clamped():
    f = fit_rect(1600, 600, FW, FH)
    assert widget_to_frame(f, f.ox - 1, 300, FW, FH) is None          # left bar
    assert widget_to_frame(f, f.ox + f.w + 1, 300, FW, FH) is None    # right bar
    assert widget_to_frame(f, 800, -1, FW, FH) is None
    f2 = fit_rect(800, 900, FW, FH)
    assert widget_to_frame(f2, 400, f2.oy - 1, FW, FH) is None        # top bar


def test_image_edges_are_inside_the_frame():
    f = fit_rect(1280, 720, FW, FH)
    x, y = widget_to_frame(f, 0, 0, FW, FH)
    assert (x, y) == pytest.approx((0, 0))
    x, y = widget_to_frame(f, 1279.999, 719.999, FW, FH)
    assert 0 <= x < FW and 0 <= y < FH


def test_resize_changes_scale_but_not_the_frame_point():
    p_small = widget_to_frame(fit_rect(640, 360, FW, FH), 320, 90, FW, FH)
    p_big = widget_to_frame(fit_rect(1920, 1080, FW, FH), 960, 270, FW, FH)
    assert p_small == pytest.approx(p_big)


def test_degenerate_sizes():
    assert fit_rect(0, 100, FW, FH) is None and fit_rect(100, 100, 0, 0) is None
    assert widget_to_frame(None, 1, 1, FW, FH) is None
