"""Syntax/import checks for every module and entry point, and JSON <-> CSV consistency of the review logs."""
import compileall
import csv
import importlib
import json
import os
import pkgutil
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

import src  # noqa: E402
from helpers import ColorTracker  # noqa: E402
from src import quality_control as qc  # noqa: E402
from src.pipeline import Config, Pipeline  # noqa: E402
from src.points_io import save_points  # noqa: E402
from src.video_io import probe_video  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_python_file_compiles():
    for target in ("src", "tests", "app.py", "censor_video.py", "pick_points.py"):
        p = os.path.join(ROOT, target)
        ok = compileall.compile_dir(p, quiet=2, force=True) if os.path.isdir(p) else compileall.compile_file(p, quiet=2, force=True)
        assert ok, f"syntax error in {target}"


def test_every_module_imports():
    names = [m.name for m in pkgutil.walk_packages(src.__path__, "src.")]
    assert {"src.pipeline", "src.job_manager", "src.review_data", "src.app.main_window", "src.app.review_panel", "src.app.job_runner"} <= set(names)
    for n in names:
        importlib.import_module(n)


@pytest.mark.parametrize("script", ["censor_video.py", "pick_points.py"])
def test_help_runs(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--help"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0 and "usage:" in r.stdout


def test_cli_rejects_bad_arguments():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "censor_video.py"), "x.mp4", "--mosaic-style", "emboss"], capture_output=True,
                       text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = subprocess.run([sys.executable, os.path.join(ROOT, "censor_video.py"), "x.mp4", "--chunk-size", "0"], capture_output=True,
                       text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 2


# ------------------------------------------------------------------ review log formats
def test_json_and_csv_hold_the_same_events(video, tmp_path):
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(video), [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    cfg = Config(input=video, output=str(tmp_path / "o.mp4"), points=pts, work_dir=str(tmp_path / "w"), log_dir=str(tmp_path / "l"),
                 chunk_size=40, analysis_width=320, preset="ultrafast", fail_closed=True, extra={"keep_temp": False})
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    doc = json.load(open(str(tmp_path / "l" / "synthetic_review.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(str(tmp_path / "l" / "synthetic_review.csv"), encoding="utf-8-sig", newline="")))
    assert list(rows[0]) == ["frame", "timestamp", "type", "severity", "object_id", "action", "detail"]
    assert len(rows) == len(doc["events"]) > 3
    key = lambda e: (int(e["frame"]), e["type"], e["severity"], str(e["object_id"] if e["object_id"] is not None else ""), e["action"], e["detail"])
    assert sorted(key(e) for e in doc["events"]) == sorted(key(r) for r in rows)
    # every event belongs to a segment, and segments only contain real event types
    for e in doc["events"]:
        assert any(s["type"] == e["type"] and s["object_id"] == e["object_id"] and s["start_frame"] <= e["frame"] <= s["end_frame"]
                   for s in doc["segments"]), e
    assert sum(doc["summary"].values()) == len(doc["events"]) and "heuristic" in doc["note"].lower()
    assert all(e["timestamp"] is not None for e in doc["events"] if e["frame"] >= 0)
    # identification of the input and of the run
    m = doc["meta"]
    assert m["input"]["fingerprint"] == probe_video(video).fingerprint and m["run_id"] and m["points_sha"]
    # detections and status files parse too
    det = list(csv.reader(open(str(tmp_path / "l" / "synthetic_detections.csv"), encoding="utf-8-sig", newline="")))
    assert det[0][:3] == ["frame", "timestamp", "class"]
    st = json.load(open(str(tmp_path / "l" / "synthetic_status.json")))
    assert st["status"].startswith("completed") and st["review_log"].endswith("synthetic_review.json")


def test_csv_survives_commas_quotes_and_newlines(tmp_path):
    log = qc.QCLog(lambda f: f / 30)
    detail = 'mask, "vanished"\nsecond line; unicode: 검수'
    log.add(7, qc.LOST_MID_FRAME, object_id=2, detail=detail, action="flagged")
    log.add(-1, qc.OUTPUT_INVALID, "error", detail="count mismatch")
    jp, cp = str(tmp_path / "r.json"), str(tmp_path / "r.csv")
    log.save(jp, cp, {"k": "v"})
    rows = list(csv.DictReader(open(cp, encoding="utf-8-sig", newline="")))
    assert rows[1]["detail"] == detail and rows[1]["frame"] == "7" and rows[0]["frame"] == "-1"
    assert json.load(open(jp, encoding="utf-8"))["events"][0]["detail"] in (detail, "count mismatch")
