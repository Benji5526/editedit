"""Audio integrity (bit-identical stream copy, AAC fallback), disk-space preflight, sleep prevention."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from helpers import ColorTracker
from src import quality_control as qc
from src import system_guard
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.system_guard import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, disk_check, keep_awake
from src.video_io import audio_md5, probe_video, verify_output

P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}
GB = 2 ** 30


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "w"), log_dir=str(tmp / "l"),
                chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": False})
    base.update(kw)
    return Config(**base)


def points_for(video, tmp):
    p = str(tmp / "p.json")
    save_points(p, probe_video(video), [P12])
    return p


def status(cfg):
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    return json.load(open(os.path.join(cfg.log_dir, f"{stem}_status.json"), encoding="utf-8"))


# ------------------------------------------------------------------ audio
def test_output_audio_is_bit_identical_to_the_source(video, tmp_path):
    cfg = cfg_for(tmp_path, video, points_for(video, tmp_path))
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    a, b = audio_md5(video), audio_md5(cfg.output)
    assert a and a == b
    assert probe_video(cfg.output).audio_codec == probe_video(video).audio_codec == "aac"


def test_verification_catches_replaced_audio(video, tmp_path):
    other = str(tmp_path / "other_audio.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=5.3",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", other], check=True)
    src = probe_video(video)
    n = probe_video(other).nb_frames
    assert verify_output(other, src, n, True, audio_identical=False) == []                       # durations look fine...
    probs = verify_output(other, src, n, True, audio_identical=True)
    assert any("audio stream differs" in p for p in probs)                                       # ...but the content is not the source's


@pytest.mark.parametrize("codec", ["pcm_alaw", "adpcm_ima_wav"])
def test_audio_codec_that_mp4_cannot_hold_falls_back_to_aac_with_a_warning(video, tmp_path, codec):
    src = str(tmp_path / f"{codec}.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-c:v", "copy", "-c:a", codec, "-ar", "8000" if codec == "pcm_alaw" else "44100", src], check=True)
    assert probe_video(src).audio_codec == codec
    cfg = cfg_for(tmp_path, src, points_for(src, tmp_path))
    rc = Pipeline(cfg, ColorTracker((180, 320)), None).run()
    assert rc in (0, 2), status(cfg)                                                             # not a failure: the video is intact
    out = probe_video(cfg.output)
    assert out.has_audio and out.audio_codec == "aac" and out.nb_frames == 160
    doc = json.load(open(os.path.join(cfg.log_dir, f"{codec}_review.json"), encoding="utf-8"))
    ev = [e for e in doc["events"] if e["type"] == qc.AUDIO_REENCODED]
    assert len(ev) == 1 and ev[0]["severity"] == "warn" and codec in ev[0]["detail"]
    assert status(cfg)["status"] == "completed_with_review_required"


# ------------------------------------------------------------------ sleep prevention
@pytest.mark.skipif(sys.platform != "win32", reason="Windows power API")
def test_keep_awake_sets_and_clears_the_system_required_flag():
    import ctypes
    f = ctypes.windll.kernel32.SetThreadExecutionState
    f.restype = ctypes.c_uint32
    with keep_awake() as active:
        assert active is True
        state = f(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)             # returns the previous state of this thread
        assert state & ES_SYSTEM_REQUIRED
    cleared = f(ES_CONTINUOUS)
    assert not (cleared & ES_SYSTEM_REQUIRED)                      # the request ended with the context
    f(ES_CONTINUOUS)


def test_keep_awake_survives_exceptions():
    with pytest.raises(RuntimeError):
        with keep_awake():
            raise RuntimeError("boom")


# ------------------------------------------------------------------ disk space
def fake_usage(monkeypatch, free_by_prefix):
    class U:
        def __init__(self, free):
            self.free = free
    monkeypatch.setattr(system_guard.shutil, "disk_usage", lambda p: U(free_by_prefix(p)))


def test_disk_check_levels(monkeypatch, tmp_path):
    out, work = str(tmp_path / "o" / "r.mp4"), str(tmp_path / "w")
    src = 10 * GB                                                   # same drive: needs 2 x 1.5 x 10 = 30 GB
    fake_usage(monkeypatch, lambda p: 100 * GB)
    r = disk_check(src, out, work)
    assert r["problems"] == [] and r["warnings"] == [] and r["same_drive"]
    fake_usage(monkeypatch, lambda p: 45 * GB)
    assert disk_check(src, out, work)["warnings"] and not disk_check(src, out, work)["problems"]
    fake_usage(monkeypatch, lambda p: 20 * GB)
    r = disk_check(src, out, work)
    assert r["problems"] and "not enough free disk space" in r["problems"][0] and "--skip-disk-check" in r["problems"][0]


def test_disk_check_separate_drives_are_checked_separately(monkeypatch):
    monkeypatch.setattr(system_guard, "_drive", lambda p: "d:" if str(p).startswith("D") else "c:")
    monkeypatch.setattr(system_guard, "_existing_parent", lambda p: p)
    fake_usage(monkeypatch, lambda p: 5 * GB if str(p).startswith("C") else 500 * GB)
    r = disk_check(10 * GB, "D:\\out\\r.mp4", "C:\\work")
    assert not r["same_drive"] and len(r["problems"]) == 1 and "c:" in r["problems"][0]          # only the work drive is short


def test_pipeline_refuses_to_start_without_space_and_can_be_overridden(video, tmp_path, monkeypatch):
    fake_usage(monkeypatch, lambda p: 1)                              # one byte free
    cfg = cfg_for(tmp_path, video, points_for(video, tmp_path))
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 1
    st = status(cfg)
    assert st["status"] == "failed" and "not enough free disk space" in st["reason"] and not os.path.exists(cfg.output)
    cfg2 = cfg_for(tmp_path, video, points_for(video, tmp_path), skip_disk_check=True)
    assert Pipeline(cfg2, ColorTracker((180, 320)), None).run() == 0 and probe_video(cfg2.output).nb_frames == 160


def test_skip_disk_check_does_not_change_the_settings_hash(video, tmp_path):
    from src.pipeline import config_hash
    info = probe_video(video)
    a = cfg_for(tmp_path, video, None)
    b = cfg_for(tmp_path, video, None, skip_disk_check=True)
    assert config_hash(a, info) == config_hash(b, info)
