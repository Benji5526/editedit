"""What the packaged application depends on: where it finds its data, its worker and ffmpeg, and which libraries each process loads."""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src import paths
from src.video_io import VideoError, find_tool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ app root and worker command
def test_source_checkout_paths():
    assert not paths.is_frozen()
    assert paths.app_root() == ROOT
    cmd = paths.worker_command()
    assert cmd[-1] == os.path.join(ROOT, "censor_video.py") and os.path.isfile(cmd[-1]) and os.path.isfile(cmd[0])


def test_frozen_paths_follow_the_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "VideoCensor.exe"))
    assert paths.is_frozen() and paths.app_root() == str(tmp_path)
    assert paths.worker_command() == [str(tmp_path / paths.WORKER_EXE)]                  # a second exe next to the GUI, no python involved
    assert all(d.startswith(str(tmp_path)) for d in paths.bundled_tool_dirs())


def test_pythonw_is_replaced_by_python_for_the_worker(monkeypatch, tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert paths.worker_command()[0] == str(tmp_path / "python.exe")


def test_job_runner_starts_whatever_worker_command_says(monkeypatch):
    """The packaged app starts VideoCensorWorker.exe through the same code path: worker_command() + the job's arguments."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from src.app import job_runner
    app = QApplication.instance() or QApplication([])
    code = ("import sys, json; print('@@PROGRESS ' + json.dumps({'event': 'echo', 'argv': sys.argv[1:]})); "
            "print('@@PROGRESS ' + json.dumps({'event': 'finished', 'status': 'completed_no_flags'}))")
    monkeypatch.setattr(job_runner, "worker_command", lambda: [sys.executable, "-c", code])
    r = job_runner.JobRunner()
    got, done = [], []
    r.progress.connect(got.append)
    r.job_finished.connect(done.append)
    assert r.start(["video.mp4", "--progress-json"], "cancel.flag")
    end = time.time() + 30
    while not done and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert done and done[0]["status"] == "completed_no_flags"
    assert got[0]["event"] == "echo" and got[0]["argv"] == ["video.mp4", "--progress-json"]
    del app


def test_unstartable_worker_is_reported_with_its_path(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from src.app import job_runner
    app = QApplication.instance() or QApplication([])
    missing = str(tmp_path / "VideoCensorWorker.exe")
    monkeypatch.setattr(job_runner, "worker_command", lambda: [missing])
    r = job_runner.JobRunner()
    done = []
    r.job_finished.connect(done.append)
    r.start([], "c.flag")
    end = time.time() + 10
    while not done and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert done and done[0]["status"] == "failed" and missing in done[0]["reason"]
    del app


# ------------------------------------------------------------------ ffmpeg lookup
def test_find_tool_prefers_the_folder_next_to_the_application(monkeypatch, tmp_path):
    bindir = tmp_path / "ffmpeg" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "ffmpeg.exe").write_bytes(b"")
    monkeypatch.setattr(paths, "app_root", lambda: str(tmp_path))
    assert find_tool("ffmpeg") == str(bindir / "ffmpeg.exe")
    assert find_tool("ffprobe") != str(bindir / "ffprobe.exe")                          # not present there: falls back to the PATH


def test_find_tool_error_explains_where_to_put_ffmpeg(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(paths, "app_root", lambda: str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(VideoError, match="ffmpeg.bin"):
        find_tool("ffmpeg")


# ------------------------------------------------------------------ what each process imports (the packaging excludes rely on it)
def run_py(code, **env):
    e = dict(os.environ, QT_QPA_PLATFORM="offscreen", **env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=e, timeout=120)


def test_gui_process_does_not_load_the_heavy_libraries():
    r = run_py("import sys; import src.app.main_window, src.app.process_dialog, src.app.review_panel, src.review_data, src.job_manager; "
               "print(sorted(m for m in ('torch', 'torchvision', 'sam2', 'onnxruntime', 'nudenet') if m in sys.modules))")
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip() == "[]", r.stdout           # so the GUI exe can be built without torch/onnxruntime


def test_worker_process_does_not_load_the_gui():
    r = run_py("import sys; import censor_video, src.pipeline, src.tracking, src.detection; "
               "print(sorted(m for m in ('PySide6', 'shiboken6') if m in sys.modules))")
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip() == "[]", r.stdout


# ------------------------------------------------------------------ app.py self test (also used to verify a build)
def test_app_selftest_report(tmp_path):
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "app.py"), "--selftest-file", str(out)], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, QT_QPA_PLATFORM="offscreen"), timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["frozen"] is False and rep["app_root"] == ROOT and rep["worker_exists"] and rep["title"] == "Video Censor"
    assert rep["gpu_libs_in_gui_process"] is False
    assert rep["ffmpeg"] and not rep["ffmpeg"].startswith("NOT FOUND") and not rep["ffprobe"].startswith("NOT FOUND")


def test_app_starts_without_console_streams(tmp_path):
    """A windowed exe has no stdout/stderr (both are None): the app must not crash on that."""
    out = tmp_path / "rep.json"
    code = ("import sys, runpy; sys.stdout = None; sys.stderr = None; sys.argv = ['app.py', '--selftest-file', r'%s']; "
            "runpy.run_path('app.py', run_name='__main__')" % out)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                       timeout=120)
    assert out.exists(), (r.returncode, r.stdout[-300:], r.stderr[-300:])
    assert json.loads(out.read_text(encoding="utf-8"))["title"] == "Video Censor"


# ------------------------------------------------------------------ the GUI-driven job self test (the same code path the packaged GUI uses)
def _run_job_selftest(video, report):
    return subprocess.run([sys.executable, os.path.join(ROOT, "app.py"), video, "--selftest-job", report], capture_output=True, text=True, cwd=ROOT,
                          env=dict(os.environ, QT_QPA_PLATFORM="offscreen", TQDM_DISABLE="1"), timeout=600)


def _cleanup_stem(stem):
    import glob
    import shutil
    for pat in (f"output/{stem}_censored.mp4*", f"logs/{stem}*"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            os.remove(f)
    for d in glob.glob(os.path.join(ROOT, "temp", f"{stem}_*")):
        shutil.rmtree(d, ignore_errors=True)


def test_gui_job_selftest_nudenet_only(video, tmp_path):
    import shutil
    stem = "selftest_clip_nn"
    vid = str(tmp_path / f"{stem}.mp4")
    shutil.copy(video, vid)
    rep = str(tmp_path / "rep.json")
    try:
        r = _run_job_selftest(vid, rep)
        assert os.path.isfile(rep), (r.returncode, r.stderr[-500:])
        d = json.loads(open(rep, encoding="utf-8").read())
        assert d["mode"] == "nudenet_only" and d["status"] == "completed_no_flags", d
        assert d["output_frames"] == d["source_frames"] == 160 and d["panel_progress"] == 1000 and d["errors"] == []
        assert d["review_loaded"] is True and d["frozen"] is False and d["worker"].lower().startswith("python")
    finally:
        _cleanup_stem(stem)


def test_gui_job_selftest_with_sam2(video, tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = os.path.join(ROOT, "models", "sam2.1_hiera_small.pt")
    if not (torch.cuda.is_available() and os.path.isfile(ckpt)):
        pytest.skip("needs CUDA and the SAM 2.1 checkpoint")
    import shutil
    from src.points_io import save_points
    from src.video_io import probe_video
    stem = "selftest_clip_sam"
    vid = str(tmp_path / f"{stem}.mp4")
    shutil.copy(video, vid)
    save_points(str(tmp_path / f"{stem}.points.json"), probe_video(vid), [{"obj_id": 1, "frame_idx": 20, "x": 50, "y": 180, "label": 1}])
    rep = str(tmp_path / "rep.json")
    try:
        r = _run_job_selftest(vid, rep)
        assert os.path.isfile(rep), (r.returncode, r.stderr[-500:])
        d = json.loads(open(rep, encoding="utf-8").read())
        assert d["mode"] == "sam2" and d["status"] in ("completed_no_flags", "completed_with_review_required"), d
        assert d["output_frames"] == 160 and d["errors"] == [] and d["review_loaded"] is True and d["flags_shown"] is not None
    finally:
        _cleanup_stem(stem)
