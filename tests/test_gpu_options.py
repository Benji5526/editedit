"""GPU options: where NudeNet runs (never a silent fallback), the encoder arguments (h264_nvenc at a quality matched to
libx264), "auto" choices, and that the app passes them on."""
import json
import os

import numpy as np
import pytest

from src.app.job_runner import ProcessSettings, build_args
from src.detection import DEVICES, NudenetDetector, make_session
from src.pipeline import Config, config_hash, render_hash
from src.video_io import NVENC_QP_OFFSET, SegmentWriter, count_packets, encoder_args, nvenc_usable, probe_video, resolve_encoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bundled_model():
    import nudenet
    return os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")


def cuda_works() -> bool:
    try:
        return make_session(bundled_model(), "auto")[1] == "CUDAExecutionProvider"
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ detector device
def test_detector_device_choices():
    assert DEVICES == ("auto", "cuda", "cpu")
    sess, provider, note = make_session(bundled_model(), "cpu")
    assert provider == "CPUExecutionProvider" and note == "" and sess.get_providers() == ["CPUExecutionProvider"]
    sess, provider, note = make_session(bundled_model(), "auto")
    assert provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert (note == "") == (provider == "CUDAExecutionProvider")              # on the CPU, "auto" says why
    with pytest.raises(ValueError):
        make_session(bundled_model(), "gpu")
    d = NudenetDetector()                                                      # the default of the class stays the CPU
    assert d.provider == "CPUExecutionProvider" and d.detect(np.zeros((180, 320, 3), np.uint8)) == []
    with pytest.raises(FileNotFoundError):
        NudenetDetector(model_path=os.path.join(ROOT, "models", "no_such_model.onnx"))


def test_explicit_cuda_never_falls_back_silently():
    if cuda_works():
        assert NudenetDetector(device="cuda").provider == "CUDAExecutionProvider"
    else:
        with pytest.raises(RuntimeError, match="cannot"):
            NudenetDetector(device="cuda")


@pytest.mark.skipif(not cuda_works(), reason="needs a working CUDA provider for onnxruntime")
def test_gpu_and_cpu_give_the_same_detector_output():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    cpu, gpu = NudenetDetector(device="cpu", threshold=0.01), NudenetDetector(device="cuda", threshold=0.01)
    blob_in = {cpu.input_name: np.ascontiguousarray(rng.random((1, 3, 320, 320), dtype=np.float32))}
    a, b = cpu.session.run(None, blob_in)[0], gpu.session.run(None, blob_in)[0]
    assert a.shape == b.shape and float(np.abs(a - b).max()) < 0.05 * max(1.0, float(np.abs(a).max()))   # same network, float noise only
    da, db = cpu.detect(img), gpu.detect(img)
    assert len(da) == len(db) and all(x.cls == y.cls and abs(x.score - y.score) < 0.02 for x, y in zip(da, db))


# ------------------------------------------------------------------ encoder
def test_encoder_arguments_and_auto():
    assert encoder_args("libx264", 16, "medium") == ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]
    nv = encoder_args("h264_nvenc", 16, "medium")
    assert nv[:4] == ["-c:v", "h264_nvenc", "-rc", "constqp"] and nv[nv.index("-qp") + 1] == str(16 + NVENC_QP_OFFSET)
    assert "-cq" not in nv and "-b:v" not in nv                                 # the old arguments ignored the quality value
    assert resolve_encoder("libx264") == "libx264" and resolve_encoder("h264_nvenc") == "h264_nvenc"
    assert resolve_encoder("auto") == ("h264_nvenc" if nvenc_usable() else "libx264")


@pytest.mark.skipif(not nvenc_usable(), reason="needs a working h264_nvenc")
def test_nvenc_quality_setting_changes_the_output(video, tmp_path):
    """Regression: with the old arguments every quality value produced the same file."""
    info = probe_video(video)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (info.height, info.width, 3), dtype=np.uint8) for _ in range(4)]
    sizes = {}
    for crf in (12, 30):
        p = str(tmp_path / f"q{crf}.mp4")
        w = SegmentWriter(p, info, "h264_nvenc", crf, "medium")
        for i in range(24):
            w.write(frames[i % 4])
        w.close()
        assert count_packets(p) == 24
        sizes[crf] = os.path.getsize(p)
    assert sizes[12] > 1.5 * sizes[30], sizes


def test_hashes_for_the_gpu_options(video, tmp_path):
    info = probe_video(video)
    base = Config(input=video, output=str(tmp_path / "o.mp4"))
    assert render_hash(Config(input=video, encoder="h264_nvenc")) != render_hash(base)
    assert config_hash(Config(input=video, encoder="h264_nvenc"), info) == config_hash(base, info)          # encoder: render-only
    a = Config(input=video, extra={"nudenet": True})
    b = Config(input=video, extra={"nudenet": True, "nudenet_provider": "CUDAExecutionProvider"})
    assert config_hash(a, info) == config_hash(b, info)                         # which device ran the detector is recorded, not hashed


# ------------------------------------------------------------------ app
def test_app_passes_the_gpu_options(tmp_path, monkeypatch):
    st = ProcessSettings(output="o.mp4")
    assert st.encoder == "auto" and st.nudenet_device == "auto"
    a = build_args("v.mp4", None, st, "c.flag")
    assert a[a.index("--encoder") + 1] == "auto" and a[a.index("--nudenet-device") + 1] == "auto"
    b = build_args("v.mp4", None, ProcessSettings(output="o.mp4", encoder="libx264", nudenet_device="cpu"), "c.flag")
    assert b[b.index("--encoder") + 1] == "libx264" and b[b.index("--nudenet-device") + 1] == "cpu"
    assert "--nudenet-device" not in build_args("v.mp4", None, ProcessSettings(output="o.mp4", use_nudenet=False), "c.flag")

    from src.app import process_dialog as pd
    path = tmp_path / "last_settings.json"
    monkeypatch.setattr(pd, "LAST_SETTINGS_PATH", str(path))
    path.write_text(json.dumps({"encoder": "libx264", "chunk_size": 200}), encoding="utf-8")       # saved before the GPU options existed
    old = pd.load_last_settings()
    assert old.encoder == "auto" and old.chunk_size == 200 and old.nudenet_device == "auto"
    pd.save_last_settings(ProcessSettings(encoder="libx264", nudenet_device="cpu"))                 # an explicit choice made now is kept
    kept = pd.load_last_settings()
    assert kept.encoder == "libx264" and kept.nudenet_device == "cpu"


def test_settings_dialog_has_the_gpu_choices(tmp_path):
    from PySide6.QtWidgets import QApplication
    from src.app.process_dialog import ProcessDialog
    app = QApplication.instance() or QApplication([])
    d = ProcessDialog(None, str(tmp_path / "v.mp4"), False, 100, ProcessSettings())
    assert d.enc.itemData(0) == "auto" and d.enc.currentData() == "auto" and d.enc.findData("libx264") >= 0
    assert [d.ndev.itemData(i) for i in range(d.ndev.count())] == ["auto", "cuda", "cpu"] and d.ndev.currentData() == "auto"
    d.ndev.setCurrentIndex(d.ndev.findData("cpu"))
    d.enc.setCurrentIndex(d.enc.findData("libx264"))
    st = d.settings()
    assert st.nudenet_device == "cpu" and st.encoder == "libx264"
    del app


def test_targets_become_detector_classes_and_face_mode():
    from src.app.job_runner import detector_classes
    st = ProcessSettings(output="o.mp4")
    assert detector_classes(st) == ["MALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_EXPOSED"]
    a = build_args("v.mp4", None, st, "c.flag")
    assert a[a.index("--det-classes") + 1:a.index("--det-classes") + 3] == detector_classes(st) and "--face" not in a
    st = ProcessSettings(output="o.mp4", det_male=False, det_female=False, face_mode="below_eyes")
    a = build_args("v.mp4", None, st, "c.flag")
    assert a[a.index("--face") + 1] == "below_eyes" and "FACE_FEMALE" in a and "MALE_GENITALIA_EXPOSED" not in a
    a = build_args("v.mp4", None, ProcessSettings(output="o.mp4", det_male=False, det_female=False), "c.flag")
    assert "--no-nudenet" in a and "--det-classes" not in a               # nothing ticked = detector off
