"""Cutting a finished video: range arithmetic, and that the exported file contains exactly the kept frames in order
(checked with the frame-number code drawn into the synthetic footage)."""
import os

import pytest

from helpers import read_frame_code
from src.video_edit import CutCancelled, CutError, build_filter, export_cut, frames_kept, keep_ranges, normalize_ranges
from src.video_io import FrameReader, VideoError, probe_video


def codes(path):
    with FrameReader(probe_video(path)) as r:
        return [read_frame_code(f) for f in r]


# ------------------------------------------------------------------ ranges
def test_normalize_and_keep_ranges():
    assert normalize_ranges([(50, 60), (10, 20), (15, 30), (30, 35), (90, 80), (-5, 3), (150, 999)], 160) == [(0, 3), (10, 35), (50, 60), (150, 160)]
    assert normalize_ranges([], 160) == [] and normalize_ranges([(5, 5)], 160) == []
    assert keep_ranges(160, []) == [(0, 160)]
    assert keep_ranges(160, [(30, 60), (100, 110)]) == [(0, 30), (60, 100), (110, 160)]
    assert keep_ranges(160, [(0, 40)]) == [(40, 160)] and keep_ranges(160, [(120, 160)]) == [(0, 120)]      # trimming the ends
    assert keep_ranges(160, [(0, 160)]) == [] and keep_ranges(160, [(0, 100), (90, 500)]) == []
    assert frames_kept(160, [(30, 60), (100, 110)]) == 120 and frames_kept(160, [(50, 40)]) == 160


def test_filter_graph(video, video_noaudio):
    info = probe_video(video)
    g, v, a = build_filter(info, [(40, 160)], True)
    assert "concat" not in g and v == "[v0]" and a == "[a0]" and "trim=start_frame=40:end_frame=160" in g and "atrim=start=1.333333" in g
    g, v, a = build_filter(info, [(0, 30), (60, 100)], True)
    assert g.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]") and (v, a) == ("[v]", "[a]")
    g, v, a = build_filter(probe_video(video_noaudio), [(0, 30), (60, 100)], False)
    assert g.endswith("[v0][v1]concat=n=2:v=1:a=0[v]") and a is None and "atrim" not in g


# ------------------------------------------------------------------ export
def test_export_removes_exactly_the_given_frames(video, tmp_path):
    out = str(tmp_path / "sub" / "cut.mp4")
    seen = []
    res = export_cut(video, [(30, 60), (100, 110)], out, preset="ultrafast", progress=lambda d, t: seen.append((d, t)))
    assert res == {"frames": 120, "removed": 40, "duration": pytest.approx(4.0), "pieces": 3, "audio": True}
    assert codes(out) == list(range(0, 30)) + list(range(60, 100)) + list(range(110, 160))        # the right frames, in order
    info = probe_video(out)
    assert info.nb_frames == 120 and (info.width, info.height) == (640, 360) and info.fps == pytest.approx(30.0) and not info.is_vfr
    assert info.has_audio and info.duration == pytest.approx(4.0, abs=0.08)                       # audio was cut to the same length
    assert seen and seen[-1] == (120, 120) and all(t == 120 for _, t in seen)
    assert not os.path.exists(out + ".partial")


def test_trim_the_ends_and_video_without_audio(video, video_noaudio, tmp_path):
    out = str(tmp_path / "trim.mp4")
    assert export_cut(video, [(0, 40), (120, 160)], out, preset="ultrafast")["pieces"] == 1
    assert codes(out) == list(range(40, 120))
    out2 = str(tmp_path / "noaudio.mp4")
    res = export_cut(video_noaudio, [(10, 150)], out2, preset="ultrafast")
    assert res["frames"] == 20 and res["audio"] is False and not probe_video(out2).has_audio
    assert codes(out2) == list(range(0, 10)) + list(range(150, 160))


def test_refused_cuts_leave_no_file(video, tmp_path):
    out = str(tmp_path / "x.mp4")
    with pytest.raises(CutError, match="nothing would be left"):
        export_cut(video, [(0, 160)], out)
    with pytest.raises(CutError, match="nothing to cut"):
        export_cut(video, [], out)
    with pytest.raises(CutError, match="must not be the input"):
        export_cut(video, [(0, 10)], video)
    with pytest.raises(CutError, match="too many pieces"):
        export_cut(video, [(i, i + 1) for i in range(1, 1000, 2)], out, info=_long(probe_video(video)))   # 500 kept pieces
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


def _long(info):
    import copy
    big = copy.copy(info)
    big.nb_frames = 1000
    return big


def test_cancel_and_encoder_failure_remove_the_partial_file(video, tmp_path):
    out = str(tmp_path / "c.mp4")
    with pytest.raises(CutCancelled):
        export_cut(video, [(0, 10)], out, preset="ultrafast", cancel_check=lambda: True)
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")
    with pytest.raises(VideoError):
        export_cut(video, [(0, 10)], out, encoder="no_such_encoder")
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


# ------------------------------------------------------------------ the cutting window
from test_gui import open_clip, pump, qapp, win  # noqa: E402,F401  (fixtures)


def make_dialog(qapp, video, tmp_path):
    import shutil
    from src.app.cut_dialog import CutDialog
    src = str(tmp_path / "result.mp4")
    shutil.copy(video, src)
    d = CutDialog(None, probe_video(src))
    d.confirm_overwrite = lambda path: True
    d.show()
    return d, src


def test_cut_window_marks_ranges_and_exports(qapp, video, tmp_path):
    from src.app.cut_dialog import default_cut_output
    d, src = make_dialog(qapp, video, tmp_path)
    try:
        assert d.out.text() == default_cut_output(src) and d.out.text().endswith("result_cut.mp4")
        assert not d.export_btn.isEnabled() and not d.add_btn.isEnabled()            # nothing to cut yet
        assert pump(lambda: len(d.timeline._thumbs) == 16, timeout=60)
        d.goto(30)
        d.set_in()
        assert "구간 시작: 프레임 30" in d.sel_label.text() and d.selection() is None
        d.timeline.seek_requested.emit(59)                                           # seeking on the timeline moves the preview
        assert d.frame_idx == 59
        d.set_out()
        assert d.selection() == (30, 60) and d.add_btn.isEnabled() and d.timeline._selection == (30, 60)
        d.add_selection()
        assert d.remove == [(30, 60)] and d.selection() is None and d.cut_list.count() == 1 and "프레임 30-59, 30프레임" in d.cut_list.item(0).text()
        d.goto(150)
        d.cut_tail()
        d.goto(10)
        d.cut_head()
        assert d.remove == [(0, 10), (30, 60), (150, 160)] and d.timeline._cuts == d.remove
        assert "결과 110프레임" in d.summary.text() and "50프레임 삭제" in d.summary.text() and d.export_btn.isEnabled()
        d.cut_list.setCurrentRow(0)
        d.remove_selected()                                                          # take the head cut back
        assert d.remove == [(30, 60), (150, 160)] and "결과 120프레임" in d.summary.text()
        d.goto(100)                                                                  # out before in: the older mark is dropped
        d.set_in()
        d.goto(90)
        d.set_out()
        assert d.mark_in is None and d.mark_out == 90
        assert not d.grab().isNull()

        assert d.start_export() and d.task is not None and not d.export_btn.isEnabled() and d.cancel_btn.isEnabled()
        assert pump(lambda: d.task is None, timeout=120)
        assert d.result == {"frames": 120, "removed": 40, "duration": pytest.approx(4.0), "pieces": 2, "audio": True}, d.last_error
        assert "완료" in d.status.text() and d.bar.value() == 1000 and d.export_btn.isEnabled()
        assert codes(d.out.text()) == list(range(0, 30)) + list(range(60, 150))
        assert probe_video(src).nb_frames == 160                                     # the input file is untouched
    finally:
        d.close()


def test_cut_window_refuses_bad_exports_and_can_cancel(qapp, video, tmp_path):
    d, src = make_dialog(qapp, video, tmp_path)
    try:
        assert d.start_export() is False and "삭제할 구간이 없습니다" in d.status.text()
        d.goto(0)
        d.cut_tail()
        assert d.remove == [(0, 160)] and not d.export_btn.isEnabled()
        assert d.start_export() is False and "남는 프레임이 없습니다" in d.status.text()
        d.clear_cuts()
        d.goto(40)
        d.cut_head()
        d.out.setText(src)
        assert d.start_export() is False and "원본 영상과 같으면" in d.status.text()
        d.out.setText("")
        assert d.start_export() is False and "결과 파일을 지정" in d.status.text()
        out = str(tmp_path / "c.mp4")
        d.out.setText(out)
        assert d.start_export()
        d.cancel_export()
        assert pump(lambda: d.task is None, timeout=60)
        assert (d.result is None and "취소" in d.status.text() and not os.path.exists(out)) or d.result is not None   # a tiny clip may finish first
        assert not os.path.exists(out + ".partial")
    finally:
        d.close()


def test_menu_opens_the_cut_window_for_a_file(win, video, tmp_path):
    import shutil
    src = str(tmp_path / "done.mp4")
    shutil.copy(video, src)
    assert win.act_cut.text() == "결과 영상 자르기..."
    d = win.open_cut_dialog(src, modal=False)
    try:
        assert d is win.cut_dialog and d.info.nb_frames == 160 and "done.mp4" in d.windowTitle()
    finally:
        d.close()
    assert win.open_cut_dialog(str(tmp_path / "missing.mp4"), modal=False) is None and "영상을 열지 못했습니다" in win.errors[-1]
