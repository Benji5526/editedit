"""SQLite job registry: the registry itself, its integration in the pipeline, and model-signature validation on resume."""
import json
import os
import sqlite3

import pytest

from helpers import ColorTracker, make_video
from src.job_manager import JobDB, db_path_for, file_signature
from src.pipeline import Config, Pipeline
from src.points_io import save_points
from src.video_io import probe_video

DEAD_PID = 2_000_000_000


@pytest.fixture()
def db(tmp_path):
    return JobDB(str(tmp_path / "jobs.sqlite"))


def mk_job(db, job_id, fp="fp1", status=None, pid=None, **kw):
    db.start_job(job_id, input_path="/x/v.mp4", input_fp=fp, total_frames=160, chunk_size=40, resume=0, **kw)
    if pid is not None:
        db.update_job(job_id, pid=pid)
    if status and status != "running":
        db.finish_job(job_id, status)


# ------------------------------------------------------------------ registry
def test_job_and_chunk_lifecycle(db):
    mk_job(db, "j1")
    db.update_job("j1", config_hash="cfg", points_sha="pts", n_chunks=4, reused_chunks=1, versions={"sam2": "1.1.0"})
    db.set_chunk("j1", 0, "done", start_frame=0, end_frame=40, seconds=1.5, rss_mb=500, cuda_peak_mb=600)
    db.set_chunk("j1", 1, "running", start_frame=40, end_frame=80)
    j = db.get_job("j1")
    assert j["status"] == "running" and j["chunks_done"] == 1 and j["chunks_running"] == 1 and j["n_chunks"] == 4
    assert json.loads(j["versions"]) == {"sam2": "1.1.0"} and j["config_hash"] == "cfg" and j["points_sha"] == "pts"
    db.set_chunk("j1", 1, "done", start_frame=40, end_frame=80, seconds=2.0)
    db.finish_job("j1", "completed_no_flags", None, output_path="o.mp4", review_log="r.json")
    j = db.get_job("j1")
    assert j["status"] == "completed_no_flags" and j["output_path"] == "o.mp4" and j["review_log"] == "r.json" and j["pid"] is None
    assert [c["status"] for c in db.chunks_of("j1")] == ["done", "done"]


def test_a_chunk_in_flight_is_marked_failed_when_the_job_ends_unfinished(db):
    mk_job(db, "j1")
    db.set_chunk("j1", 0, "done", start_frame=0, end_frame=40)
    db.set_chunk("j1", 1, "running", start_frame=40, end_frame=80)
    db.finish_job("j1", "cancelled", "cancelled by the user")
    cs = db.chunks_of("j1")
    assert [c["status"] for c in cs] == ["done", "failed"] and "cancelled" in cs[1]["error"]
    assert db.get_job("j1")["error"] == "cancelled by the user"


def test_crashed_worker_is_detected_as_interrupted(db):
    mk_job(db, "alive", fp="a", pid=os.getpid())
    mk_job(db, "dead", fp="b", pid=DEAD_PID)
    db.set_chunk("dead", 0, "done", start_frame=0, end_frame=40)
    db.set_chunk("dead", 1, "running", start_frame=40, end_frame=80)
    assert db.recover_interrupted() == 1
    assert db.get_job("alive")["status"] == "running"
    d = db.get_job("dead")
    assert d["status"] == "interrupted" and "disappeared" in d["error"] and d["chunks_done"] == 1 and d["chunks_failed"] == 1


def test_a_reused_process_id_does_not_keep_a_dead_job_alive(db):
    """Windows hands a crashed worker's process id to another process quickly; the creation time tells them apart."""
    mk_job(db, "ghost", fp="g", pid=os.getpid())                      # an id that exists right now...
    db.update_job("ghost", pid_ctime=1.0)                             # ...but whose process was created at another time (long ago)
    assert db.recover_interrupted() == 1 and db.get_job("ghost")["status"] == "interrupted"
    mk_job(db, "real", fp="r")                                        # a real job records this process's own creation time
    assert db.get_job("real")["pid_ctime"] is not None and db.recover_interrupted() == 0


def test_registry_from_an_older_version_is_migrated(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.sqlite")
    c = sqlite3.connect(path)
    c.executescript("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, created TEXT, updated TEXT, pid INTEGER, input_path TEXT, input_fp TEXT, "
                    "points_sha TEXT, config_hash TEXT, model_sig TEXT, versions TEXT, total_frames INTEGER, chunk_size INTEGER, n_chunks INTEGER, "
                    "reused_chunks INTEGER DEFAULT 0, status TEXT, error TEXT, output_path TEXT, review_log TEXT, job_dir TEXT, resume INTEGER DEFAULT 0);"
                    "INSERT INTO jobs (job_id, status, pid, input_fp) VALUES ('old', 'running', 2000000000, 'fp');")
    c.commit()
    c.close()
    db = JobDB(path)                                                  # opens, adds the column, keeps the rows
    assert db.recover_interrupted() == 1 and db.get_job("old")["status"] == "interrupted"


def test_latest_resumable_rules(db):
    assert db.latest_resumable("none") is None
    mk_job(db, "a", fp="f")
    db.set_chunk("a", 0, "done", start_frame=0, end_frame=40)
    db.finish_job("a", "cancelled")
    assert db.latest_resumable("f")["job_id"] == "a"
    mk_job(db, "b", fp="g", status="failed")                                  # failed before finishing any chunk: nothing to resume
    assert db.latest_resumable("g") is None
    mk_job(db, "c", fp="f")
    db.set_chunk("c", 0, "done", start_frame=0, end_frame=40)
    db.finish_job("c", "completed_no_flags")                                  # the newest job for the video finished: not resumable
    assert db.latest_resumable("f") is None
    mk_job(db, "d", fp="h", pid=os.getpid())                                  # a job is running right now
    db.set_chunk("d", 0, "done", start_frame=0, end_frame=40)
    assert db.latest_resumable("h") is None
    mk_job(db, "e", fp="i", pid=DEAD_PID)                                     # crashed: recovered and offered for resume
    db.set_chunk("e", 0, "done", start_frame=0, end_frame=40)
    assert db.latest_resumable("i")["status"] == "interrupted"


def test_reused_chunks_count_as_progress(db):
    mk_job(db, "r")
    db.update_job("r", reused_chunks=2)
    db.finish_job("r", "failed", "x")
    assert db.latest_resumable("fp1")["job_id"] == "r"


def test_jobs_for_input_newest_first_and_wal_reader(db):
    for i in range(3):
        mk_job(db, f"j{i}", fp="same")
    assert [j["job_id"] for j in db.jobs_for_input("same")] == ["j2", "j1", "j0"]
    reader = JobDB(db.path)                                                   # a second connection (the GUI) sees the writer's data
    assert reader.get_job("j1")["input_fp"] == "same"


def test_file_signature_detects_changes(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(os.urandom(3 << 20))
    s1 = file_signature(str(f))
    assert s1 == file_signature(str(f)) and s1.startswith("m.pt:")
    data = bytearray(f.read_bytes())
    data[100] ^= 0xFF
    f.write_bytes(bytes(data))
    assert file_signature(str(f)) != s1
    f.write_bytes(os.urandom((3 << 20) + 5))
    assert file_signature(str(f)) != s1


# ------------------------------------------------------------------ pipeline integration
P12 = {"obj_id": 1, "frame_idx": 12, "x": 10, "y": 180, "label": 1}


def cfg_for(tmp, video, points, **kw):
    base = dict(input=video, output=str(tmp / "out.mp4"), points=points, work_dir=str(tmp / "work"), log_dir=str(tmp / "logs"),
                chunk_size=40, analysis_width=320, block=16, mask_pad_px=10, preset="ultrafast", crf=18, extra={"keep_temp": True})
    base.update(kw)
    return Config(**base)


@pytest.fixture()
def points(video, tmp_path):
    p = str(tmp_path / "p.json")
    save_points(p, probe_video(video), [P12])
    return p


def registry(cfg):
    return JobDB(db_path_for(cfg.log_dir))


def test_pipeline_records_a_completed_job(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    jobs = registry(cfg).jobs_for_input(probe_video(video).fingerprint)
    assert len(jobs) == 1
    j = jobs[0]
    assert j["status"] == "completed_no_flags" or j["status"] == "completed_with_review_required"
    assert j["chunks_done"] == 4 and j["n_chunks"] == 4 and j["total_frames"] == 160 and j["chunk_size"] == 40
    assert j["output_path"] == os.path.abspath(cfg.output) and os.path.isfile(j["review_log"]) and j["pid"] is None
    assert j["points_sha"] and j["config_hash"] and j["error"] is None
    assert [c["status"] for c in registry(cfg).chunks_of(j["job_id"])] == ["done"] * 4


def test_pipeline_records_cancel_crash_and_resume(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    fp = probe_video(video).fingerprint
    flag = {"stop": False}
    rc = Pipeline(cfg, ColorTracker((180, 320)), None, cancel_check=lambda: flag["stop"],
                  progress=lambda e: flag.update(stop=True) if e["event"] == "chunk_done" and e["chunk"] == 1 else None).run()
    assert rc == 3
    j = registry(cfg).latest_resumable(fp)
    assert j["status"] == "cancelled" and j["chunks_done"] == 2

    cfg.resume = True
    assert Pipeline(cfg, ColorTracker((180, 320), crash_chunks={0}), None).run() == 1                     # crash in the first resumed chunk
    j2 = registry(cfg).jobs_for_input(fp)[0]
    assert j2["status"] == "failed" and "simulated crash" in j2["error"] and j2["reused_chunks"] == 2
    failed = [c for c in registry(cfg).chunks_of(j2["job_id"]) if c["status"] == "failed"]
    assert [c["idx"] for c in failed] == [2] and "RuntimeError" in failed[0]["error"]
    assert registry(cfg).latest_resumable(fp)["job_id"] == j2["job_id"]                                   # still resumable (2 chunks reused)

    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    jobs = registry(cfg).jobs_for_input(fp)
    assert [j["status"].startswith("completed") for j in jobs] == [True, False, False]
    assert jobs[0]["reused_chunks"] == 2 and jobs[0]["chunks_done"] == 4
    assert registry(cfg).latest_resumable(fp) is None


def test_rejected_input_leaves_a_failed_record(tmp_path):
    p = str(tmp_path / "vfr.mp4")
    make_video(p, n_frames=90, audio=False, vfr=True)
    pts = str(tmp_path / "p.json")
    save_points(pts, probe_video(p), [P12])
    cfg = cfg_for(tmp_path, p, pts)
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 1
    j = registry(cfg).jobs_for_input(probe_video(p).fingerprint)[0]
    assert j["status"] == "failed" and "variable frame rate" in j["error"] and j["chunks_done"] == 0


def test_a_broken_registry_never_stops_processing(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    os.makedirs(cfg.log_dir)
    os.makedirs(db_path_for(cfg.log_dir))                                     # a directory where the database file should be
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    assert probe_video(cfg.output).nb_frames == 160


def test_registry_write_failures_are_swallowed(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points)
    p = Pipeline(cfg, ColorTracker((180, 320)), None)
    orig = JobDB.set_chunk

    def boom(self, *a, **k):
        raise sqlite3.OperationalError("database is locked")
    JobDB.set_chunk = boom
    try:
        assert p.run() == 0
    finally:
        JobDB.set_chunk = orig


def test_changed_model_file_refuses_resume(video, points, tmp_path):
    cfg = cfg_for(tmp_path, video, points, extra={"keep_temp": True, "sam2_ckpt_sig": "m.pt:100:aaaa"})
    assert Pipeline(cfg, ColorTracker((180, 320)), None).run() == 0
    same = cfg_for(tmp_path, video, points, resume=True, extra={"keep_temp": True, "sam2_ckpt_sig": "m.pt:100:aaaa"})
    tr = ColorTracker((180, 320))
    assert Pipeline(same, tr, None).run() == 0 and tr.n_calls == 0
    other = cfg_for(tmp_path, video, points, resume=True, extra={"keep_temp": True, "sam2_ckpt_sig": "m.pt:100:bbbb"})
    assert Pipeline(other, ColorTracker((180, 320)), None).run() == 1
    status = json.load(open(os.path.join(str(tmp_path / "logs"), "synthetic_status.json")))
    assert "differ" in status["reason"]
