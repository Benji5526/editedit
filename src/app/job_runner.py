"""Runs censor_video.py as a separate process so the UI never blocks and a crash/OOM cannot take the app down.

Protocol (see censor_video.py --progress-json): stdout lines starting with '@@PROGRESS ' carry JSON events;
everything else is plain log text. Cancel = a cancel file the pipeline polls at safe points (completed chunks
stay valid for --resume); if it does not stop within the grace period the whole process tree is killed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

PROGRESS_PREFIX = "@@PROGRESS "
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREATE_NO_WINDOW = 0x08000000


def is_noise_line(line: str) -> bool:
    """tqdm progress bars (SAM 2 prints them to stderr) would flood the log view."""
    return ("it/s]" in line or "s/it]" in line) and "|" in line


def parse_progress_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        ev = json.loads(line[len(PROGRESS_PREFIX):])
    except ValueError:
        return None
    return ev if isinstance(ev, dict) and "event" in ev else None


@dataclass
class ProcessSettings:
    sam2_checkpoint: str = ""
    output: str = ""
    chunk_size: int = 120
    analysis_width: int = 1920
    use_nudenet: bool = True
    detect_every: int = 1
    det_threshold: float = 0.25
    fail_closed: bool = True
    fail_closed_mode: str = "bbox"
    block: int = 32
    mosaic_style: str = "pixelate"
    mask_pad_px: float = 24.0
    keep_audio: bool = True
    max_frames: Optional[int] = None
    resume: bool = False
    encoder: str = "libx264"
    device: str = "cuda"
    allow_vfr: bool = False
    keep_chunks: bool = True    # keep intermediate chunks after success: enables fast re-processing after re-clicks (uses disk)
    work_dir: str = ""          # empty = project default (temp/)
    log_dir: str = ""           # empty = project default (logs/)


def build_args(video: str, points_path: Optional[str], st: ProcessSettings, cancel_file: str, plan_only: bool = False) -> List[str]:
    """Arguments for censor_video.py (without the interpreter and script path)."""
    a = [video, "--progress-json", "--cancel-file", cancel_file, "-o", st.output,
         "--chunk-size", str(st.chunk_size), "--analysis-width", str(st.analysis_width),
         "--detect-every", str(st.detect_every), "--det-threshold", str(st.det_threshold),
         "--block", str(st.block), "--mosaic-style", st.mosaic_style, "--mask-pad-px", str(st.mask_pad_px), "--encoder", st.encoder, "--device", st.device]
    if points_path:
        a += ["--points", points_path, "--sam2-checkpoint", st.sam2_checkpoint]
    if not st.use_nudenet:
        a.append("--no-nudenet")
    if st.fail_closed:
        a += ["--fail-closed", "--fail-closed-mode", st.fail_closed_mode]
    if not st.keep_audio:
        a.append("--no-audio")
    if st.max_frames:
        a += ["--max-frames", str(st.max_frames)]
    if st.resume:
        a.append("--resume")
    if st.allow_vfr:
        a.append("--allow-vfr")
    if st.keep_chunks:
        a.append("--keep-temp")
    if plan_only:
        a.append("--plan-only")
    if st.work_dir:
        a += ["--work-dir", st.work_dir]
    if st.log_dir:
        a += ["--log-dir", st.log_dir]
    return a


def job_dir_for_video(work_dir: str, video_path: str, fingerprint: str) -> str:
    """Same location the pipeline uses for a video's intermediate chunks (src.pipeline.job_dir_for)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(work_dir or os.path.join(PROJECT_ROOT, "temp"), f"{stem}_{fingerprint[:10]}")


def overall_fraction(ev: dict) -> Optional[float]:
    """0..1 overall progress from a progress/start/chunk_done event."""
    if ev.get("overall") is not None:                     # the pipeline's monotonic value (analysis and render overlap)
        return min(1.0, float(ev["overall"]))
    total = ev.get("total_frames") or 0
    if total <= 0:
        return None
    kind = ev.get("event")
    if kind == "progress":
        return min(1.0, (ev.get("done_frames", 0) + ev.get("chunk_frac", 0.0) * ev.get("chunk_frames", 0)) / total)
    if kind in ("start", "chunk_done"):
        return min(1.0, ev.get("done_frames", 0) / total)
    return None


def child_python() -> str:
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":     # pythonw has no usable stdout for the child
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(cand):
            return cand
    return exe


def kill_tree(pid: int):
    """The venv python.exe is a launcher that starts the real interpreter as a child: kill the whole tree."""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=15)
    except Exception:  # noqa: BLE001
        pass


class JobRunner(QObject):
    progress = Signal(dict)
    log_line = Signal(str)
    job_finished = Signal(dict)

    def __init__(self, parent=None, grace_seconds: float = 60.0):
        super().__init__(parent)
        self.grace_ms = int(grace_seconds * 1000)
        self._proc: Optional[QProcess] = None
        self._buf = ""
        self._finished_event: Optional[dict] = None
        self._cancel_file: Optional[str] = None
        self._cancel_requested = False
        self._forced = False
        self._t0 = 0.0
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.force_kill)

    def is_running(self) -> bool:
        return self._proc is not None

    def new_cancel_file(self) -> str:
        fd, path = tempfile.mkstemp(prefix="vc_cancel_", suffix=".flag")
        os.close(fd)
        os.remove(path)                       # the file must not exist until cancel is requested
        return path

    def start(self, args: List[str], cancel_file: str, command: Optional[List[str]] = None) -> bool:
        """Start censor_video.py with ``args`` (``command`` replaces the whole command line; used by tests)."""
        if self._proc is not None:
            return False
        self._cancel_file, self._cancel_requested, self._forced = cancel_file, False, False
        self._buf, self._finished_event, self._t0 = "", None, time.time()
        p = QProcess(self)
        p.setWorkingDirectory(PROJECT_ROOT)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("TQDM_DISABLE", "1")
        p.setProcessEnvironment(env)
        try:                                   # no console window flashing when started from the GUI
            def mod(a):
                a.flags |= CREATE_NO_WINDOW
            p.setCreateProcessArgumentsModifier(mod)
        except (AttributeError, TypeError):
            pass
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self._proc = p
        if command:
            p.start(command[0], list(command[1:]))
        else:
            p.start(child_python(), [os.path.join(PROJECT_ROOT, "censor_video.py")] + args)
        return True

    def cancel(self):
        """Ask the job to stop at its next safe point; force-kill after the grace period."""
        if self._proc is None or self._cancel_requested:
            return
        self._cancel_requested = True
        if self._cancel_file:
            try:
                open(self._cancel_file, "w").close()
            except OSError:
                pass
        self._kill_timer.start(self.grace_ms)
        self.log_line.emit("cancel requested - stopping at the next safe point ...")

    def force_kill(self):
        if self._proc is None:
            return
        self._forced = True
        self.log_line.emit("job did not stop in time - terminating the process tree")
        pid = int(self._proc.processId())
        if pid:
            kill_tree(pid)
        self._proc.kill()

    # ---- process callbacks
    def _read(self):
        if self._proc is None:
            return
        self._buf += bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self._handle_line(ln.rstrip("\r"))

    def _handle_line(self, ln: str):
        ev = parse_progress_line(ln)
        if ev is None:
            if ln.strip() and not is_noise_line(ln):
                self.log_line.emit(ln)
            return
        if ev["event"] == "finished":
            self._finished_event = ev
        self.progress.emit(ev)

    def _on_error(self, err):
        if err == QProcess.ProcessError.FailedToStart and self._proc is not None:
            self._finish({"exit_code": -1, "status": "failed", "reason": f"could not start the worker process ({child_python()})"})

    def _on_finished(self, code, _status):
        self._read()
        if self._buf.strip():
            self._handle_line(self._buf)
            self._buf = ""
        res = dict(self._finished_event or {})
        res.setdefault("exit_code", int(code))
        if self._finished_event is None:
            res.update(status="failed", reason="the worker process ended without reporting a result"
                       + (" (killed after cancel timeout)" if self._forced else f" (exit code {code})"))
            if self._cancel_requested:
                res["status"] = "cancelled"
                res["reason"] = "cancelled by the user (process was terminated)"
        self._finish(res)

    def _finish(self, res: dict):
        if self._proc is None:
            return
        self._kill_timer.stop()
        p, self._proc = self._proc, None
        p.deleteLater()
        if self._cancel_file and os.path.exists(self._cancel_file):
            try:
                os.remove(self._cancel_file)
            except OSError:
                pass
        res.update(cancelled_by_user=self._cancel_requested, forced_kill=self._forced, seconds=round(time.time() - self._t0, 1))
        self.job_finished.emit(res)
