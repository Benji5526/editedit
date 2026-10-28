"""Final merge options (faststart is opt-in), HDD warning, and the new progress events."""
import os
import struct

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from helpers import ColorTracker
from src import system_guard
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.system_guard import disk_check, media_type
from src.video_io import probe_video

P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}
GB = 2 ** 30


def top_level_atoms(path):
    """Names of the MP4 top-level boxes in file order."""
    names = []
    with open(path, "rb") as f:
        size_total = os.path.getsize(path)
        pos = 0
        while pos + 8 <= size_total:
            f.seek(pos)
            size, name = struct.unpack(">I4s", f.read(8))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size == 0:
                size = size_total - pos
            names.append(name.decode("latin1"))
            pos += size
    return names


def cfg_for(tmp, video, **kw):
    pts = str(tmp / "p.json")
    if not os.path.exists(pts):
        save_points(pts, probe_video(video), [P12])
    base = dict(input=video, output=str(tmp / "out.mp4"), points=pts, work_dir=str(tmp / "w"), log_dir=str(tmp / "l"),
                chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": False})
    base.update(kw)
    return Config(**base)


def test_faststart_is_off_by_default_and_opt_in(video, tmp_path):
    cfg = cfg_for(tmp_path, video)
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    a = top_level_atoms(cfg.output)
    assert a.index("mdat") < a.index("moov"), a                         # index at the end: no second pass over the whole file
    cfg2 = cfg_for(tmp_path, video, faststart=True, output=str(tmp_path / "fast.mp4"))
    assert Pipeline(cfg2, ColorTracker((180, 320)), None).run() == 0
    b = top_level_atoms(cfg2.output)
    assert b.index("moov") < b.index("mdat"), b                         # index first (web streaming)
    assert probe_video(cfg.output).nb_frames == probe_video(cfg2.output).nb_frames == 160


def test_faststart_does_not_change_the_settings_hash(video, tmp_path):
    from src.pipeline import config_hash
    info = probe_video(video)
    assert config_hash(cfg_for(tmp_path, video), info) == config_hash(cfg_for(tmp_path, video, faststart=True), info)


def test_hdd_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(system_guard.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * GB})())
    seen = []
    monkeypatch.setattr(system_guard, "media_type", lambda p: seen.append(p) or "HDD")
    r = disk_check(1 * GB, str(tmp_path / "o" / "r.mp4"), str(tmp_path / "w"))
    assert r["problems"] == [] and any("hard disk drive" in w and "work folder" in w for w in r["warnings"])
    assert len(seen) == 2
    monkeypatch.setattr(system_guard, "media_type", lambda p: "SSD")
    assert disk_check(1 * GB, str(tmp_path / "o" / "r.mp4"), str(tmp_path / "w"))["warnings"] == []
    monkeypatch.setattr(system_guard, "media_type", lambda p: pytest.fail("must not be asked"))
    assert disk_check(1 * GB, str(tmp_path / "o" / "r.mp4"), str(tmp_path / "w"), check_media=False)["warnings"] == []


@pytest.mark.skipif(os.name != "nt", reason="Windows drive query")
def test_media_type_real_query_is_safe_and_cached():
    system_guard._MEDIA_CACHE.clear()
    first = media_type(os.getcwd())
    assert first in ("SSD", "HDD", "")
    assert media_type(os.getcwd()) == first and len(system_guard._MEDIA_CACHE) == 1       # second call hits the cache
    assert media_type("") == "" or media_type("relative\\path") in ("SSD", "HDD", "")


def test_pipeline_reports_warnings_and_the_finalizing_stage(video, tmp_path, monkeypatch):
    monkeypatch.setattr(system_guard, "media_type", lambda p: "HDD")
    import src.pipeline as pl
    monkeypatch.setattr(pl, "disk_check", lambda *a, **k: system_guard.disk_check(*a, **dict(k, check_media=True)))
    evs = []
    cfg = cfg_for(tmp_path, video)
    assert Pipeline(cfg, ColorTracker((180, 320)), None, progress=evs.append).run() == 0
    warns = [e for e in evs if e["event"] == "warning"]
    assert warns and "hard disk drive" in warns[0]["text"]
    stages = [e["stage"] for e in evs if e["event"] == "stage"]
    assert stages == ["finalizing"]
    assert max(i for i, e in enumerate(evs) if e["event"] == "chunk_done") < evs.index(next(e for e in evs if e["event"] == "stage"))


def test_progress_panel_shows_warnings_and_the_finalizing_stage():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from src.app.process_panel import ProcessPanel
    app = QApplication.instance() or QApplication([])
    p = ProcessPanel()
    p.on_progress({"event": "warning", "text": "the work folder is on a hard disk drive (HDD)"})
    assert "WARNING: the work folder is on a hard disk drive" in p.log.toPlainText()
    p.on_progress({"event": "stage", "stage": "finalizing"})
    assert "Merging chunks" in p.stage.text()
    del app


def test_overall_progress_is_monotonic_with_overlapping_stages(video, tmp_path):
    evs = []
    cfg = cfg_for(tmp_path, video)
    assert Pipeline(cfg, ColorTracker((180, 320)), None, progress=evs.append).run() == 0
    vals = [e["overall"] for e in evs if "overall" in e]
    assert vals and vals == sorted(vals) and 0 <= vals[0] and vals[-1] == 1.0         # never goes backwards, ends at 100 %
    from src.app.job_runner import overall_fraction
    assert [overall_fraction(e) for e in evs if "overall" in e] == vals                # the GUI uses exactly this value
