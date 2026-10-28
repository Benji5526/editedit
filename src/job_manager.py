"""SQLite registry of processing jobs (one row per run, one row per chunk).

The registry answers "which jobs exist for this video, how far did they get, can I resume?". It is an INDEX, not the source
of truth: whether chunks are really reusable is decided from the files in the job directory (job.json + segment/state/qc files,
validated on resume). A wrong or missing registry entry can therefore never make the pipeline reuse a bad chunk.
A failure to open or write the registry never stops a processing job (it is logged and ignored).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

ACTIVE = "running"
RESUMABLE_STATES = ("cancelled", "failed", "interrupted")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, created TEXT, updated TEXT, pid INTEGER, pid_ctime REAL,
  input_path TEXT, input_fp TEXT, points_sha TEXT, config_hash TEXT, model_sig TEXT, versions TEXT,
  total_frames INTEGER, chunk_size INTEGER, n_chunks INTEGER, reused_chunks INTEGER DEFAULT 0,
  status TEXT, error TEXT, output_path TEXT, review_log TEXT, job_dir TEXT, resume INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_by_input ON jobs(input_fp, created);
CREATE TABLE IF NOT EXISTS chunks (
  job_id TEXT, idx INTEGER, status TEXT, start_frame INTEGER, end_frame INTEGER, seconds REAL, rss_mb INTEGER,
  cuda_peak_mb INTEGER, error TEXT, updated TEXT, PRIMARY KEY (job_id, idx)
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def process_ctime(pid: int) -> Optional[float]:
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:  # noqa: BLE001
        return None


def pid_alive(pid: Optional[int], ctime: Optional[float] = None) -> bool:
    """Is the process that started a job still running? Windows reuses process ids quickly, so when the creation time was recorded
    the process must also have been created at that time; a different process that merely got the same id does not count."""
    if not pid:
        return False
    try:
        import psutil
        if not psutil.pid_exists(int(pid)):
            return False
        if ctime is not None:
            return abs(psutil.Process(int(pid)).create_time() - ctime) < 2.0
        return True
    except Exception:  # noqa: BLE001
        return False


class JobDB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
            if "pid_ctime" not in cols:                                   # registry created by an older version
                c.execute("ALTER TABLE jobs ADD COLUMN pid_ctime REAL")

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")          # the GUI reads while the worker writes
            yield c
            c.commit()
        finally:
            c.close()

    # ---- writes (worker process)
    def start_job(self, job_id: str, **f) -> None:
        cols = ["job_id", "created", "updated", "pid", "pid_ctime", "status"] + list(f)
        vals = [job_id, _now(), _now(), os.getpid(), process_ctime(os.getpid()), ACTIVE] + [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO jobs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)

    def update_job(self, job_id: str, **f) -> None:
        sets = ", ".join(f"{k}=?" for k in f) + (", " if f else "") + "updated=?"
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()] + [_now(), job_id]
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", vals)

    def set_chunk(self, job_id: str, idx: int, status: str, **f) -> None:
        cols = ["job_id", "idx", "status", "updated"] + list(f)
        vals = [job_id, idx, status, _now()] + list(f.values())
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO chunks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
            c.execute("UPDATE jobs SET updated=? WHERE job_id=?", (_now(), job_id))

    def finish_job(self, job_id: str, status: str, error: Optional[str] = None, output_path: Optional[str] = None,
                   review_log: Optional[str] = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=?, error=?, output_path=COALESCE(?, output_path), review_log=COALESCE(?, review_log), "
                      "updated=?, pid=NULL, pid_ctime=NULL WHERE job_id=?", (status, error, output_path, review_log, _now(), job_id))
            if status in ("cancelled", "failed", "interrupted"):       # a chunk that was in flight did not finish
                c.execute("UPDATE chunks SET status='failed', error=COALESCE(error, ?), updated=? WHERE job_id=? AND status='running'",
                          (f"job {status}", _now(), job_id))

    # ---- reads (GUI, CLI)
    def recover_interrupted(self) -> int:
        """Jobs still marked running whose process is gone (crash, kill, power loss) become 'interrupted'."""
        n = 0
        with self._conn() as c:
            for r in c.execute("SELECT job_id, pid, pid_ctime FROM jobs WHERE status=?", (ACTIVE,)).fetchall():
                if not pid_alive(r["pid"], r["pid_ctime"]):
                    c.execute("UPDATE jobs SET status='interrupted', error=COALESCE(error,'worker process disappeared'), pid=NULL, pid_ctime=NULL, updated=? "
                              "WHERE job_id=?", (_now(), r["job_id"]))
                    c.execute("UPDATE chunks SET status='failed', error='worker process disappeared', updated=? WHERE job_id=? AND status='running'",
                              (_now(), r["job_id"]))
                    n += 1
        return n

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._with_counts(c, dict(r)) if r else None

    @staticmethod
    def _with_counts(c, job: dict) -> dict:
        rows = c.execute("SELECT status, COUNT(*) AS n FROM chunks WHERE job_id=? GROUP BY status", (job["job_id"],)).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        job["chunks_done"] = counts.get("done", 0)
        job["chunks_failed"] = counts.get("failed", 0)
        job["chunks_running"] = counts.get("running", 0)
        return job

    def jobs_for_input(self, input_fp: str, limit: int = 20) -> List[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM jobs WHERE input_fp=? ORDER BY created DESC, rowid DESC LIMIT ?", (input_fp, limit)).fetchall()
            return [self._with_counts(c, dict(r)) for r in rows]

    def chunks_of(self, job_id: str) -> List[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM chunks WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()]

    def latest_resumable(self, input_fp: str) -> Optional[dict]:
        """Newest job for this video that stopped before finishing but completed at least one chunk."""
        self.recover_interrupted()
        for j in self.jobs_for_input(input_fp):
            if j["status"] == ACTIVE:
                return None                                            # something is running right now
            if j["status"] in RESUMABLE_STATES and j["chunks_done"] + (j.get("reused_chunks") or 0) > 0:
                return j
            if j["status"].startswith("completed"):
                return None                                            # the newest job finished: nothing to resume
        return None


def file_signature(path: str) -> str:
    """Cheap identity of a (large) model file: name, size and sha256 of its first and last MiB."""
    import hashlib
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return f"{os.path.basename(path)}:{size}:{h.hexdigest()[:16]}"


def db_path_for(log_dir: str) -> str:
    return os.path.join(log_dir, "jobs.sqlite")
