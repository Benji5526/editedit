import os
import subprocess

import numpy as np
import pytest

from helpers import FPS, H, W, make_video, read_frame_code
from src.video_io import (FrameReader, VideoError, classify_vfr, probe_video, read_frame_at, verify_output)


def test_probe_cfr(video):
    i = probe_video(video)
    assert (i.width, i.height, i.nb_frames) == (W, H, 160)
    assert i.fps == FPS and not i.is_vfr and i.has_audio and i.rotation == 0
    assert len(i.timestamps) == 160 and abs(i.timestamp_of(30) - 1.0) < 1e-3


def test_probe_missing_file():
    with pytest.raises(VideoError):
        probe_video("nope.mp4")


def test_classify_vfr_pure():
    cfr = [i / 30 for i in range(100)]
    assert classify_vfr(cfr, 30, 30)[0] is False
    jittered = [i / 30 + (0.0004 if i % 2 else 0) for i in range(100)]
    assert classify_vfr(jittered, 30, 30)[0] is False          # sub-10% jitter is not VFR
    vfr = [i / 30 for i in range(50)] + [50 / 30 + 0.5 + i / 15 for i in range(50)]
    assert classify_vfr(vfr, 30, 22)[0] is True
    assert classify_vfr(None, 30, 30)[0] is True               # no pts => cannot trust


def test_vfr_file_detected(tmp_path):
    p = str(tmp_path / "vfr.mp4")
    make_video(p, n_frames=90, audio=False, vfr=True)
    i = probe_video(p)
    assert i.is_vfr, i.vfr_detail


def test_reader_frame_numbers_in_order_and_complete(video):
    i = probe_video(video)
    codes = []
    with FrameReader(i) as r:
        for f in r:
            codes.append(read_frame_code(f))
    assert codes == list(range(160))


def test_reader_scaled(video):
    i = probe_video(video)
    with FrameReader(i, scale_to=(320, 180), max_frames=3) as r:
        fs = list(r)
    assert len(fs) == 3 and fs[0].shape == (180, 320, 3)


@pytest.mark.parametrize("n", [0, 1, 29, 30, 31, 77, 159])
def test_seek_is_frame_exact(video, n):
    i = probe_video(video)
    assert read_frame_code(read_frame_at(i, n)) == n


def test_seek_start_then_sequential(video):
    i = probe_video(video)
    with FrameReader(i, start_frame=77, max_frames=5) as r:
        assert [read_frame_code(f) for f in r] == [77, 78, 79, 80, 81]


def test_rotation_metadata_gives_rotated_decoded_size(tmp_path):
    p = str(tmp_path / "rot.mp4")
    make_video(p, n_frames=30, audio=False, rotate=90)
    i = probe_video(p)
    assert i.rotation in (90, 270), i.rotation
    assert (i.width, i.height) == (H, W)               # decoded frames are swapped => reader/picker agree
    with FrameReader(i, max_frames=1) as r:
        assert r.read().shape == (W, H, 3)


def test_verify_output_detects_problems(video, tmp_path):
    src = probe_video(video)
    assert verify_output(video, src, 160, expect_audio=True) == []
    assert any("count" in p for p in verify_output(video, src, 161, expect_audio=True))
    assert verify_output(str(tmp_path / "missing.mp4"), src, 160, True) == ["output file missing or empty"]


def test_verify_detects_missing_audio(video, video_noaudio):
    src = probe_video(video)
    probs = verify_output(video_noaudio, src, 160, expect_audio=True)
    assert any("audio" in p for p in probs)


def test_verify_detects_truncated(video, tmp_path):
    src = probe_video(video)
    short = str(tmp_path / "short.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-frames:v", "100", "-c", "copy", short], check=True)
    probs = verify_output(short, src, 160, expect_audio=True)
    assert any("count" in p for p in probs)
