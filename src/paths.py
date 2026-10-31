"""Where things live, for a source checkout and for the packaged (PyInstaller) application.

Packaged layout (onedir):   VideoCensor\\VideoCensor.exe          the desktop app (no console window)
                            VideoCensor\\VideoCensorWorker.exe    the processing worker (console subsystem, started hidden by the app)
                            VideoCensor\\_internal\\...            Python, libraries (read-only, replaced by every build)
                            VideoCensor\\models, logs, temp, output, input, config, ffmpeg   YOUR data, next to the exe
User data never goes into ``_internal``: it would vanish with the next build, and the bundle may be read-only.
"""
from __future__ import annotations

import os
import sys
from typing import List

WORKER_EXE = "VideoCensorWorker.exe"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> str:
    """Folder that holds models/, logs/, temp/, output/, config/: the folder of the exe when packaged, else the project folder."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker_command() -> List[str]:
    """Command line (without arguments) that starts the processing worker."""
    if is_frozen():
        return [os.path.join(app_root(), WORKER_EXE)]
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":          # pythonw has no usable stdout for the child
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(cand):
            exe = cand
    return [exe, os.path.join(app_root(), "censor_video.py")]


def bundled_tool_dirs() -> List[str]:
    """Places searched for ffmpeg/ffprobe before the PATH: a folder the user drops next to the application."""
    root = app_root()
    return [os.path.join(root, "ffmpeg", "bin"), os.path.join(root, "ffmpeg"), os.path.join(root, "bin")]
