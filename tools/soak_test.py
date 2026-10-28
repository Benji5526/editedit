"""Long-run soak test on synthetic 4K60 footage: memory over time, hard kill + resume, output verification.

    python tools/soak_test.py --minutes 3            # about 10,800 frames at 3840x2160, 60 fps
    python tools/soak_test.py --minutes 40 --keep    # the full-size case (hours)

Everything lands in samples/soak/ (not tracked). The report is printed and saved as samples/soak/soak_report.json.
It measures a synthetic clip (a magenta box on a noisy grey background that wraps around the frame); it says nothing about
detection quality on real footage.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.job_manager import JobDB, db_path_for  # noqa: E402
from src.points_io import save_points  # noqa: E402
from src.video_io import audio_md5, probe_video, read_frame_at, verify_output  # noqa: E402

W, H, FPS = 3840, 2160, 60
BOX = 300
SPEED = 200            # px per second, the box wraps at x = 3400


def box_x(t_frame: int) -> int:
    return int((t_frame / FPS * SPEED) % 3400)


def make_video(path: str, seconds: int):
    fc = (f"color=c=0x303030:s={W}x{H}:r={FPS}:d={seconds}[bg];color=c=0xff00ff:s={BOX}x{BOX}:r={FPS}:d={seconds}[box];"
          f"[bg][box]overlay=x='mod(t*{SPEED}\\,3400)':y=900:eval=frame,noise=alls=12:allf=t[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-filter_complex", fc, "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
           "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", "-b:a", "128k", path]
    subprocess.run(cmd, check=True)


def kill_tree(pid: int):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def run_job(args, env, kill_after_chunk=None, label=""):
    """Run censor_video.py, collect progress events; optionally kill the whole process tree after a chunk."""
    errf = open(os.path.join(args.dir, f"stderr_{label}.txt"), "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "censor_video.py"), *args.cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=errf,
                         text=True, encoding="utf-8", env=env)
    events, killed = [], False
    t0 = time.time()
    for line in p.stdout:
        if not line.startswith("@@PROGRESS "):
            continue
        ev = json.loads(line[len("@@PROGRESS "):])
        events.append(ev)
        if ev["event"] == "chunk_done":
            done = ev["chunk"] + 1
            if done % 10 == 0 or done == ev["chunks"]:
                print(f"[{label}] chunk {done}/{ev['chunks']}  rss={ev.get('rss_mb')} MB  gpu_peak={ev.get('cuda_peak_mb')} MB  "
                      f"{ev['seconds']}s/chunk  elapsed {time.time() - t0:.0f}s", flush=True)
            if kill_after_chunk is not None and ev["chunk"] + 1 >= kill_after_chunk and not killed:
                print(f"[{label}] >>> killing the worker process tree after chunk {done}", flush=True)
                kill_tree(p.pid)
                killed = True
    rc = p.wait()
    errf.close()
    return rc, events, killed, time.time() - t0


def slope(values):
    if len(values) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(values)), np.asarray(values, float), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3)
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--kill-at-fraction", type=float, default=0.3, help="hard-kill the worker after this fraction of the chunks")
    ap.add_argument("--sam2-checkpoint", default=os.path.join(ROOT, "models", "sam2.1_hiera_small.pt"))
    ap.add_argument("--no-nudenet", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the intermediate chunks and the generated video")
    a = ap.parse_args()

    a.dir = os.path.join(ROOT, "samples", "soak")
    os.makedirs(a.dir, exist_ok=True)
    seconds = int(a.minutes * 60)
    video = os.path.join(a.dir, f"soak_4k60_{seconds}s.mp4")
    if not os.path.isfile(video):
        print(f"generating {seconds} s of 4K60 test footage ...", flush=True)
        t = time.time()
        make_video(video, seconds)
        print(f"  done in {time.time() - t:.0f} s, {os.path.getsize(video) / 2 ** 30:.2f} GB", flush=True)
    info = probe_video(video)
    print(f"source: {info.width}x{info.height} {info.fps_rational} fps, {info.nb_frames} frames, {info.duration:.1f} s, vfr={info.is_vfr}", flush=True)
    assert (info.width, info.height, info.nb_frames) == (W, H, seconds * FPS)

    pts = os.path.join(a.dir, "soak_points.json")
    f0 = 30
    save_points(pts, info, [{"obj_id": 1, "frame_idx": f0, "x": box_x(f0) + BOX / 2, "y": 900 + BOX / 2, "label": 1}])
    out = os.path.join(a.dir, "soak_out.mp4")
    work, logs = os.path.join(a.dir, "work"), os.path.join(a.dir, "logs")
    for d in (work, logs):
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
    if os.path.exists(out):
        os.remove(out)
    a.cmd = [video, "--progress-json", "--points", pts, "--sam2-checkpoint", a.sam2_checkpoint, "-o", out, "--work-dir", work, "--log-dir", logs,
             "--chunk-size", str(a.chunk_size), "--fail-closed", "--preset", "veryfast", "--keep-temp"]
    if a.no_nudenet:
        a.cmd.append("--no-nudenet")
    env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_DISABLE="1")
    n_chunks = -(-info.nb_frames // a.chunk_size)
    kill_after = max(2, int(n_chunks * a.kill_at_fraction))
    report = {"seconds": seconds, "frames": info.nb_frames, "chunks": n_chunks, "chunk_size": a.chunk_size,
              "nudenet": not a.no_nudenet, "started": time.strftime("%F %T")}

    # ---- run 1: killed hard
    rc1, ev1, killed, dt1 = run_job(a, env, kill_after_chunk=kill_after, label="run1")
    cd1 = [e for e in ev1 if e["event"] == "chunk_done"]
    report["run1"] = {"killed": killed, "exit_code": rc1, "chunks_done": len(cd1), "seconds": round(dt1)}
    assert killed and not os.path.exists(out), "run 1 should have been killed before producing a result"
    db = JobDB(db_path_for(logs))
    j = db.latest_resumable(info.fingerprint)
    report["registry_after_kill"] = {"status": j and j["status"], "chunks_done": j and j["chunks_done"], "chunks_failed": j and j["chunks_failed"]}
    print("registry after the kill:", report["registry_after_kill"], flush=True)
    assert j is not None and j["status"] == "interrupted"

    # ---- run 2: resume
    a.cmd.append("--resume")
    rc2, ev2, _, dt2 = run_job(a, env, label="run2")
    cd2 = [e for e in ev2 if e["event"] == "chunk_done"]
    start2 = next(e for e in ev2 if e["event"] == "start")
    fin = ev2[-1]
    report["run2"] = {"exit_code": rc2, "resumed_from_chunk": start2["resume_from_chunk"], "reused_chunks": start2["reused_chunks"],
                      "chunks_done": len(cd2), "seconds": round(dt2), "status": fin.get("status"),
                      "error_events": fin.get("error_events"), "warn_events": fin.get("warn_events")}
    print("run 2:", report["run2"], flush=True)
    assert rc2 in (0, 2) and os.path.isfile(out), fin

    # ---- memory over time (each run is its own process)
    mem = {}
    for name, cds in (("run1", cd1), ("run2", cd2)):
        rss = [e.get("rss_mb") for e in cds if e.get("rss_mb")]
        gpu = [e.get("cuda_peak_mb") for e in cds if e.get("cuda_peak_mb") is not None]
        mem[name] = {"chunks": len(cds), "rss_first_mb": rss[0] if rss else None, "rss_last_mb": rss[-1] if rss else None,
                     "rss_max_mb": max(rss) if rss else None, "rss_slope_mb_per_chunk": round(slope(rss), 3),
                     "gpu_peak_max_mb": max(gpu) if gpu else None, "gpu_peak_first_mb": gpu[0] if gpu else None, "gpu_peak_last_mb": gpu[-1] if gpu else None,
                     "chunk_seconds_first": cds[0]["seconds"] if cds else None, "chunk_seconds_last": cds[-1]["seconds"] if cds else None}
        # growth across the whole run, relative to the steady state reached after the first 10 chunks
        if len(rss) > 20:
            mem[name]["rss_growth_after_chunk_10_mb"] = rss[-1] - rss[10]
    report["memory"] = mem
    print("memory:", json.dumps(mem, indent=1), flush=True)

    # ---- output verification
    t = time.time()
    problems = verify_output(out, info, info.nb_frames, True, deep=True, audio_identical=True)
    oinfo = probe_video(out)
    dt = 1.0 / info.fps
    dev = np.abs((np.asarray(oinfo.timestamps) - oinfo.timestamps[0]) - np.arange(len(oinfo.timestamps)) * dt)
    report["output"] = {"frames": oinfo.nb_frames, "size": [oinfo.width, oinfo.height], "duration": oinfo.duration, "source_duration": info.duration,
                        "audio_identical": audio_md5(video) == audio_md5(out), "max_timestamp_deviation_ms": round(float(dev.max()) * 1000, 4),
                        "verify_problems": problems, "verify_seconds": round(time.time() - t)}
    print("output:", report["output"], flush=True)

    # ---- chunks written before the kill were reused, not recomputed
    jobs = db.jobs_for_input(info.fingerprint)
    report["jobs"] = [{"status": x["status"], "chunks_done": x["chunks_done"], "reused": x["reused_chunks"], "error": x["error"]} for x in jobs]

    # ---- spot checks against the analytic position of the box (far from where it wraps/leaves the frame)
    spots = []
    for fr in np.linspace(f0 + 100, info.nb_frames - 100, 6).astype(int):
        bx = box_x(int(fr))
        if bx > 3400 - 400 or bx < 150:
            continue
        src_f, out_f = read_frame_at(info, int(fr)), read_frame_at(oinfo, int(fr))
        inside = np.s_[900 + 60:900 + BOX - 60, bx + 60:bx + BOX - 60]
        outside = np.s_[100:400, 100:600] if bx > 800 else np.s_[1500:1800, 100:600]
        d_in = float(np.abs(out_f[inside].astype(int) - src_f[inside].astype(int)).mean())
        d_out = float(np.abs(out_f[outside].astype(int) - src_f[outside].astype(int)).mean())
        spots.append({"frame": int(fr), "box_x": bx, "inside_diff": round(d_in, 1), "outside_diff": round(d_out, 2)})
    report["spot_checks"] = spots
    print("spot checks:", spots, flush=True)

    report["finished"] = time.strftime("%F %T")
    with open(os.path.join(a.dir, "soak_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    ok = not problems and report["output"]["audio_identical"] and report["output"]["max_timestamp_deviation_ms"] < 1.0
    print("SOAK RESULT:", "PASS" if ok else "FAIL", flush=True)
    if not a.keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
