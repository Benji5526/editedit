"""Client for the mask preview server (src/preview_server.py): a long-lived worker process that keeps SAM 2 loaded and returns
the mask for the clicks of one frame. Only the newest request matters: while one is being computed, a newer one replaces any
request still waiting."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from src.paths import worker_command
import numpy as np

from src.preview_server import PREVIEW_PREFIX, decode_mask, encode_mask

from .job_runner import CREATE_NO_WINDOW, PROJECT_ROOT, kill_tree


def preview_args(video: str, checkpoint: str, analysis_width: int, device: str = "cuda") -> List[str]:
    return [video, "--preview-server", "--sam2-checkpoint", checkpoint, "--analysis-width", str(analysis_width), "--device", device]


class MaskPreview(QObject):
    # request id, frame, {obj_id: bool mask (analysis size)}, {obj_id: reason skipped}; "object" because Qt cannot carry int-keyed dicts
    masks_ready = Signal(int, int, object, object)
    state_changed = Signal(str, str)             # state ("off" | "loading" | "ready" | "failed"), detail text
    request_failed = Signal(int, str)            # request id, reason
    track_ready = Signal(int, int, object)       # request id, first frame, [bool mask (analysis size) or None per frame]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._buf = ""
        self.state = "off"
        self._next_id = 0
        self._busy = False
        self._waiting: Optional[dict] = None
        self._queue: List[dict] = []             # trial-tracking requests: sent in order, never replaced

    def is_on(self) -> bool:
        return self._proc is not None

    def is_ready(self) -> bool:
        return self.state == "ready"

    def start(self, args: List[str], command: Optional[List[str]] = None) -> bool:
        """Start the server with ``args`` (``command`` replaces the whole command line; used by tests)."""
        if self._proc is not None:
            return False
        p = QProcess(self)
        p.setWorkingDirectory(PROJECT_ROOT)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)     # stderr (library chatter) is not protocol
        env = QProcessEnvironment.systemEnvironment()
        for k, v in (("PYTHONUNBUFFERED", "1"), ("PYTHONIOENCODING", "utf-8"), ("TQDM_DISABLE", "1")):
            env.insert(k, v)
        p.setProcessEnvironment(env)
        try:
            def mod(a):
                a.flags |= CREATE_NO_WINDOW
            p.setCreateProcessArgumentsModifier(mod)
        except (AttributeError, TypeError):
            pass
        p.readyReadStandardOutput.connect(self._read)
        p.readyReadStandardError.connect(lambda: p.readAllStandardError())        # drain so the pipe never fills up
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self._proc, self._buf, self._busy, self._waiting, self._queue = p, "", False, None, []
        cmd = list(command) if command else worker_command() + args
        self._set_state("loading", "모델을 불러오는 중 ...")
        p.start(cmd[0], cmd[1:])
        return True

    def stop(self):
        p, self._proc = self._proc, None
        self._busy, self._waiting, self._queue = False, None, []
        if p is not None:
            try:
                p.finished.disconnect(self._on_finished)
                p.errorOccurred.disconnect(self._on_error)
            except (RuntimeError, TypeError):
                pass
            if p.state() != QProcess.ProcessState.NotRunning:
                try:
                    p.write(b'{"cmd": "quit"}\n')
                    p.closeWriteChannel()
                except Exception:  # noqa: BLE001
                    pass
                if not p.waitForFinished(1500):
                    pid = int(p.processId())
                    if pid:
                        kill_tree(pid)             # the venv python.exe is a launcher: kill the real interpreter too
                    p.kill()
                    p.waitForFinished(3000)
            p.deleteLater()
        self._set_state("off", "")

    def request(self, frame: int, objects: Dict[int, dict]) -> int:
        """objects: {obj_id: {"points": [[x, y], ...], "labels": [...]}} in full-resolution pixels. Returns the request id."""
        self._next_id += 1
        req = {"id": self._next_id, "frame": int(frame), "objects": {str(k): v for k, v in objects.items()}}
        if self._proc is None:
            return self._next_id
        if self._busy or self.state != "ready":
            self._waiting = req                    # newest wins
        else:
            self._send(req)
        return self._next_id

    def request_track(self, obj_id: int, frame: int, frames: int, prompts: Dict[int, dict]) -> int:
        """Trial tracking of one object over ``frames`` frames from ``frame``. prompts: {offset: {"points", "labels"} or
        {"mask": bool array}}. Returns the request id (0 when the server is off)."""
        if self._proc is None:
            return 0
        self._next_id += 1
        wire = {}
        for off, pr in prompts.items():
            if pr.get("mask") is not None:
                m = np.ascontiguousarray(pr["mask"], dtype=bool)
                wire[str(int(off))] = {"mask": encode_mask(np.packbits(m.reshape(-1))), "shape": [int(m.shape[0]), int(m.shape[1])]}
            else:
                wire[str(int(off))] = {"points": pr["points"], "labels": pr["labels"]}
        req = {"cmd": "track", "id": self._next_id, "obj_id": int(obj_id), "frame": int(frame), "frames": int(frames), "prompts": wire}
        if self._busy or self.state != "ready":
            self._queue.append(req)
        else:
            self._send(req)
        return self._next_id

    # ---- internals
    def _send(self, req: dict):
        self._busy = True
        self._proc.write((json.dumps(req) + "\n").encode("utf-8"))

    def _set_state(self, state: str, detail: str):
        self.state = state
        self.state_changed.emit(state, detail)

    def _read(self):
        if self._proc is None:
            return
        self._buf += bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            ln = ln.strip()
            if ln.startswith(PREVIEW_PREFIX):
                try:
                    self._handle(json.loads(ln[len(PREVIEW_PREFIX):]))
                except ValueError:
                    pass

    def _handle(self, ev: dict):
        kind = ev.get("event")
        if kind == "ready":
            self._set_state("ready", "")
        elif kind == "fatal":
            self._set_state("failed", str(ev.get("reason")))
        elif kind in ("masks", "error", "track"):
            self._busy = False
            if kind == "track":
                shape = tuple(ev["shape"])
                self.track_ready.emit(int(ev["id"]), int(ev["frame"]), [None if m is None else decode_mask(m, shape) for m in ev["masks"]])
            elif kind == "masks":
                shape = tuple(ev["shape"])
                masks = {int(k): decode_mask(v, shape) for k, v in ev["masks"].items()}
                self.masks_ready.emit(int(ev["id"]), int(ev["frame"]), masks, {int(k): v for k, v in (ev.get("skipped") or {}).items()})
            else:
                self.request_failed.emit(int(ev.get("id") or 0), str(ev.get("reason")))
        if self.state == "ready" and not self._busy and self._proc is not None:
            if self._queue:
                self._send(self._queue.pop(0))
            elif self._waiting is not None:
                req, self._waiting = self._waiting, None
                self._send(req)

    def _on_error(self, err):
        if err == QProcess.ProcessError.FailedToStart and self._proc is not None:
            self._proc = None
            self._set_state("failed", "작업 프로세스를 시작하지 못했습니다")

    def _on_finished(self, code, _status):
        if self._proc is None:
            return
        self._read()
        self._proc = None
        self._busy, self._waiting, self._queue = False, None, []
        if self.state != "failed":
            self._set_state("failed", f"미리보기 프로세스가 종료되었습니다 (종료 코드 {code})")
