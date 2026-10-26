"""Worker-process protocol: argument building, progress parsing, the real censor_video.py CLI, and the JobRunner
(including force-killing a worker tree that ignores the cancel request)."""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.app.job_runner import (PROGRESS_PREFIX, PROJECT_ROOT, JobRunner, ProcessSettings, build_args, child_python,  # noqa: E402
                                overall_fraction, parse_progress_line)
from src.app.process_dialog import validate_settings  # noqa: E402
from src.app.process_panel import format_eta  # noqa: E402
from src.points_io import save_points  # noqa: E402
from src.video_io import probe_video  # noqa: E402

CKPT = os.path.join(PROJECT_ROOT, "models", "sam2.1_hiera_small.pt")
CLI = os.path.join(PROJECT_ROOT, "censor_video.py")


def test_parse_progress_line():
    assert parse_progress_line(PROGRESS_PREFIX + '{"event": "start", "total_frames": 5}') == {"event": "start", "total_frames": 5}
    assert parse_progress_line("  " + PROGRESS_PREFIX + '{"event": "x"}\r\n') == {"event": "x"}
    for bad in ("plain log line", PROGRESS_PREFIX + "{not json", PROGRESS_PREFIX + "[1]", PROGRESS_PREFIX + '{"no_event": 1}', ""):
        assert parse_progress_line(bad) is None


def test_tqdm_bars_are_filtered_from_the_log():
    from src.app.job_runner import is_noise_line
    assert is_noise_line("propagate in video:  45%|#####   | 5/11 [00:00<00:00, 39.00it/s]")
    assert is_noise_line("frame loading (JPEG): 100%|##########| 11/11 [00:00<00:00, 32.62it/s]")
    assert not is_noise_line("2026-10-26 21:55:19 INFO chunk 16/16 frames 150-159 done in 1.1s")
    assert not is_noise_line("ValueError: could not convert | to it/s")      # needs the bar shape ('s]'), not just a pipe


def test_overall_fraction():
    assert overall_fraction({"event": "start", "total_frames": 100, "done_frames": 40}) == pytest.approx(0.4)
    ev = {"event": "progress", "total_frames": 100, "done_frames": 40, "chunk_frames": 20, "chunk_frac": 0.5}
    assert overall_fraction(ev) == pytest.approx(0.5)
    assert overall_fraction({"event": "chunk_done", "total_frames": 100, "done_frames": 100}) == 1.0
    assert overall_fraction({"event": "progress", "total_frames": 0}) is None and overall_fraction({"event": "stage"}) is None


def test_format_eta():
    assert format_eta(None) == "--" and format_eta(75) == "1:15" and format_eta(3725) == "1:02:05"


def test_build_args_defaults_and_options():
    st = ProcessSettings(sam2_checkpoint="m.pt", output="o.mp4", chunk_size=240, fail_closed=True, fail_closed_mode="full",
                         use_nudenet=False, keep_audio=False, max_frames=600, resume=True, work_dir="w", log_dir="l")
    a = build_args("v.mp4", "p.json", st, "c.flag")
    assert a[0] == "v.mp4" and "--progress-json" in a and a[a.index("--cancel-file") + 1] == "c.flag"
    assert a[a.index("--points") + 1] == "p.json" and a[a.index("--sam2-checkpoint") + 1] == "m.pt"
    assert a[a.index("--chunk-size") + 1] == "240" and a[a.index("--fail-closed-mode") + 1] == "full"
    for flag in ("--fail-closed", "--no-nudenet", "--no-audio", "--resume"):
        assert flag in a
    assert a[a.index("--max-frames") + 1] == "600" and a[a.index("--work-dir") + 1] == "w" and a[a.index("--log-dir") + 1] == "l"
    b = build_args("v.mp4", None, ProcessSettings(output="o.mp4", fail_closed=False), "c.flag")
    assert "--points" not in b and "--sam2-checkpoint" not in b and "--fail-closed" not in b and "--no-nudenet" not in b


def test_validate_settings(tmp_path):
    v = str(tmp_path / "v.mp4")
    ok = ProcessSettings(sam2_checkpoint=str(tmp_path / "missing.pt"), output=str(tmp_path / "sub" / "o.mp4"))
    assert any("checkpoint" in p for p in validate_settings(ok, v, True))
    assert validate_settings(ok, v, False) == []                              # NudeNet only: no checkpoint needed
    assert any("Nothing to do" in p for p in validate_settings(ProcessSettings(output=ok.output, use_nudenet=False), v, False))
    assert any("input file" in p for p in validate_settings(ProcessSettings(output=v), v, False))
    assert any("Chunk size" in p for p in validate_settings(ProcessSettings(output=ok.output, chunk_size=5), v, False))
    assert os.path.isdir(str(tmp_path / "sub"))                               # output folder is created


def test_child_python_prefers_console_interpreter(monkeypatch, tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert child_python() == str(tmp_path / "python.exe")


# ---------------------------------------------------------------- the real CLI over its progress protocol
def run_cli(args, cancel_after_chunk=None, cancel_file=None, timeout=240):
    import tempfile
    # stderr goes to a file: SAM 2's progress bars would fill a pipe nobody reads and block the child
    with tempfile.TemporaryFile("w+", encoding="utf-8") as errf:
        p = subprocess.Popen([sys.executable, CLI, *args], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=errf,
                             text=True, encoding="utf-8")
        events, other = [], []
        try:
            for line in p.stdout:
                ev = parse_progress_line(line)
                if ev is None:
                    other.append(line)
                    continue
                events.append(ev)
                if cancel_after_chunk is not None and ev["event"] == "chunk_done" and ev["chunk"] == cancel_after_chunk:
                    open(cancel_file, "w").close()
            rc = p.wait(timeout=timeout)
        finally:
            if p.poll() is None:
                p.kill()
        errf.seek(0)
        return rc, events, other, errf.read()


def cli_args(video, tmp_path, *extra):
    return [video, "--progress-json", "-o", str(tmp_path / "out.mp4"), "--work-dir", str(tmp_path / "w"), "--log-dir",
            str(tmp_path / "l"), "--chunk-size", "40", "--preset", "ultrafast", "--analysis-width", "320", *extra]


def test_cli_nudenet_only_emits_protocol_and_finishes(video, tmp_path):
    rc, evs, other, err = run_cli(cli_args(video, tmp_path))
    assert rc == 0, err[-500:]
    kinds = [e["event"] for e in evs]
    assert kinds[:2] == ["stage", "stage"] and "start" in kinds and kinds.count("chunk_done") == 4 and kinds[-1] == "finished"
    fin = evs[-1]
    assert fin["exit_code"] == 0 and fin["status"] == "completed_no_flags" and os.path.isfile(fin["output"])
    assert probe_video(fin["output"]).nb_frames == 160
    assert not any(l.strip() for l in other), other           # stdout carries only protocol lines when --progress-json is set


def test_cli_cancel_file_stops_safely_with_exit_code_3(video, tmp_path):
    flag = str(tmp_path / "cancel.flag")
    rc, evs, _, err = run_cli(cli_args(video, tmp_path, "--cancel-file", flag), cancel_after_chunk=0, cancel_file=flag)
    assert rc == 3, err[-500:]
    fin = evs[-1]
    assert fin["event"] == "finished" and fin["status"] == "cancelled" and fin["exit_code"] == 3
    assert not os.path.exists(str(tmp_path / "out.mp4")) and not os.path.exists(str(tmp_path / "out.mp4.partial"))
    # and it is resumable with identical settings
    os.remove(flag)
    rc2, evs2, _, err2 = run_cli(cli_args(video, tmp_path, "--cancel-file", flag, "--resume"))
    assert rc2 == 0, err2[-500:]
    assert [e for e in evs2 if e["event"] == "start"][0]["resume_from_chunk"] >= 1
    assert probe_video(str(tmp_path / "out.mp4")).nb_frames == 160


def test_cli_reports_model_setup_failure(video, tmp_path):
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(video), [{"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}])
    rc, evs, _, _ = run_cli(cli_args(video, tmp_path, "--points", pts, "--sam2-checkpoint", str(tmp_path / "nope.pt"), "--no-nudenet"))
    assert rc == 1 and evs[-1]["event"] == "finished" and evs[-1]["status"] == "failed" and "checkpoint" in evs[-1]["reason"]


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs the SAM 2.1 checkpoint")
def test_cli_real_sam2_over_the_protocol(video, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(video), [{"obj_id": 1, "frame_idx": 20, "x": 240 + 30, "y": 180, "label": 1}])
    rc, evs, _, err = run_cli(cli_args(video, tmp_path, "--points", pts, "--sam2-checkpoint", CKPT, "--no-nudenet", "--fail-closed",
                                       "--max-frames", "80"))
    assert rc in (0, 2), err[-800:]
    chunk_done = [e for e in evs if e["event"] == "chunk_done"]
    assert len(chunk_done) == 2 and all("cuda_peak_mb" in e for e in chunk_done)
    assert evs[-1]["status"] in ("completed_no_flags", "completed_with_review_required") and os.path.isfile(evs[-1]["output"])


# ---------------------------------------------------------------- JobRunner
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_runner_force_kills_worker_tree_that_ignores_cancel(qapp, tmp_path):
    psutil = pytest.importorskip("psutil")
    pidfile = tmp_path / "child.pid"
    child = "import time; time.sleep(120)"
    parent = ("import subprocess, sys, time; "
              f"c = subprocess.Popen([sys.executable, '-c', {child!r}]); open({str(pidfile)!r}, 'w').write(str(c.pid)); "
              "print('@@PROGRESS {\"event\": \"stage\", \"stage\": \"processing\"}', flush=True); time.sleep(120)")
    r = JobRunner(grace_seconds=1.0)
    got, logs, done = [], [], []
    r.progress.connect(got.append)
    r.log_line.connect(logs.append)
    r.job_finished.connect(done.append)
    assert r.start([], str(tmp_path / "c.flag"), command=[sys.executable, "-c", parent])
    assert pump(lambda: pidfile.exists() and got)                   # worker is up and talking
    grandchild = int(pidfile.read_text())
    assert psutil.pid_exists(grandchild)
    r.cancel()
    assert pump(lambda: bool(done), 20)
    res = done[0]
    assert res["cancelled_by_user"] and res["forced_kill"] and res["status"] == "cancelled"
    assert pump(lambda: not psutil.pid_exists(grandchild), 10), "grandchild process survived the kill"
    assert any("terminating" in l for l in logs) and not r.is_running()


def test_runner_reports_worker_that_dies_silently(qapp, tmp_path):
    r = JobRunner(grace_seconds=5)
    done = []
    r.job_finished.connect(done.append)
    r.start([], str(tmp_path / "c.flag"), command=[sys.executable, "-c", "import sys; print('boom'); sys.exit(7)"])
    assert pump(lambda: bool(done), 20)
    assert done[0]["status"] == "failed" and done[0]["exit_code"] == 7 and "without reporting" in done[0]["reason"]


def test_runner_reports_missing_executable(qapp, tmp_path):
    r = JobRunner()
    done = []
    r.job_finished.connect(done.append)
    r.start([], str(tmp_path / "c.flag"), command=[str(tmp_path / "no_such_program.exe")])
    assert pump(lambda: bool(done), 10)
    assert done[0]["status"] == "failed" and "could not start" in done[0]["reason"]
