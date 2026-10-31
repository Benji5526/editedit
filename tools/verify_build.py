"""Smoke-test the built application folder:  python tools/verify_build.py [--dist dist/VideoCensor] [--no-job]

1. layout: both exes, _internal, documents, data folders
2. VideoCensor.exe --selftest-file: starts on the real desktop, reports frozen/app_root/worker/ffmpeg and that the GUI process stays free of torch
3. VideoCensorWorker.exe --help
4. a real job with the worker exe (SAM 2 small + NudeNet + fail-closed on the synthetic clip): protocol events, exit code, frame count,
   bit-identical audio, review log, no model errors.  Needs models/sam2.1_hiera_small.pt (copied into the build if it is missing there).
Run it from the project virtual environment (it uses src.video_io to inspect the result). Everything it creates is under dist/.../output and logs.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.video_io import audio_md5, probe_video  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(ROOT, "dist", "VideoCensor"))
    ap.add_argument("--no-job", action="store_true", help="skip the real processing job")
    a = ap.parse_args()
    dist = os.path.abspath(a.dist)
    gui, worker = os.path.join(dist, "VideoCensor.exe"), os.path.join(dist, "VideoCensorWorker.exe")

    # 1. layout
    check("GUI exe and worker exe exist", os.path.isfile(gui) and os.path.isfile(worker))
    check("_internal folder exists", os.path.isdir(os.path.join(dist, "_internal")))
    for d in ("models", "input", "output", "logs", "temp", "config", os.path.join("ffmpeg", "bin")):
        check(f"data folder {d}", os.path.isdir(os.path.join(dist, d)))
    for f in ("README.md", "READ_ME_FIRST.txt", "THIRD_PARTY_NOTICES.md"):
        check(f"document {f}", os.path.isfile(os.path.join(dist, f)))
    check("no model weights, videos or FFmpeg were bundled", not any(
        fn.lower().endswith((".pt", ".pth", ".mp4", ".mov", ".mkv")) or fn.lower() in ("ffmpeg.exe", "ffprobe.exe")
        for _, _, fs in os.walk(dist) for fn in fs))
    size_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dist) for f in fs) / 2 ** 30
    print(f"      build size: {size_gb:.2f} GB", flush=True)

    # 2. GUI self test (real desktop window, not offscreen)
    rep_path = os.path.join(tempfile.mkdtemp(), "selftest.json")
    t = time.time()
    p = subprocess.run([gui, "--selftest-file", rep_path], cwd=dist, timeout=180)
    rep = json.load(open(rep_path, encoding="utf-8")) if os.path.isfile(rep_path) else {}
    check("GUI exe starts and writes its self-test report", p.returncode == 0 and bool(rep), f"{time.time() - t:.1f} s, exit {p.returncode}")
    check("GUI knows it is packaged and uses the exe folder", rep.get("frozen") is True and os.path.normcase(rep.get("app_root", "")) == os.path.normcase(dist))
    check("GUI finds the worker exe", rep.get("worker_exists") is True)
    check("GUI window title", rep.get("title") == "Video Censor", str(rep.get("title")))
    check("GUI process does not load torch", rep.get("gpu_libs_in_gui_process") is False)
    check("GUI finds ffmpeg and ffprobe", all(rep.get(k) and not str(rep[k]).startswith("NOT FOUND") for k in ("ffmpeg", "ffprobe")),
          f"ffmpeg={rep.get('ffmpeg')}")

    # 3. worker --help, in an environment without any Python on PATH (what a user's machine looks like)
    env_min = minimal_env()
    r = subprocess.run([worker, "--help"], capture_output=True, text=True, cwd=dist, timeout=180, env=env_min)
    check("worker --help", r.returncode == 0 and "usage:" in r.stdout and "--progress-json" in r.stdout, f"exit {r.returncode}")

    # 4. real job
    if a.no_job:
        return summary()
    ckpt_src, ckpt = os.path.join(ROOT, "models", "sam2.1_hiera_small.pt"), os.path.join(dist, "models", "sam2.1_hiera_small.pt")
    if not os.path.isfile(ckpt):
        if not os.path.isfile(ckpt_src):
            check("checkpoint available for the job test", False, "models/sam2.1_hiera_small.pt not found")
            return summary()
        shutil.copy2(ckpt_src, ckpt)
        print("      copied the SAM 2 checkpoint into the build's models folder for this test", flush=True)
    video, pts = os.path.join(ROOT, "samples", "synthetic.mp4"), os.path.join(ROOT, "samples", "points.json")
    out = os.path.join(dist, "output", "verify_synthetic.mp4")
    if os.path.exists(out):
        os.remove(out)
    cmd = [worker, video, "--progress-json", "--points", pts, "--sam2-checkpoint", os.path.join("models", "sam2.1_hiera_small.pt"),
           "--chunk-size", "40", "--analysis-width", "320", "--fail-closed", "--preset", "ultrafast", "-o", out]
    t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=dist, timeout=900,
                          env=dict(env_min, PYTHONIOENCODING="utf-8", TQDM_DISABLE="1"))
    events = []
    for line in proc.stdout.splitlines():
        if line.startswith("@@PROGRESS "):
            events.append(json.loads(line[len("@@PROGRESS "):]))
    kinds = [e["event"] for e in events]
    fin = events[-1] if events else {}
    check("worker job finishes", proc.returncode in (0, 2) and fin.get("event") == "finished" and fin.get("status", "").startswith("completed"),
          f"exit {proc.returncode}, status {fin.get('status')}, {time.time() - t:.0f} s; stderr tail: {proc.stderr[-200:].strip() if proc.returncode not in (0, 2) else ''}")
    check("worker emits the progress protocol", "start" in kinds and kinds.count("chunk_done") == 4, f"{kinds.count('chunk_done')} chunk_done events")
    if os.path.isfile(out):
        src, res = probe_video(video), probe_video(out)
        check("result has all frames and the source size", res.nb_frames == src.nb_frames and (res.width, res.height) == (src.width, src.height),
              f"{res.nb_frames} frames {res.width}x{res.height}")
        check("result audio is bit-identical to the source", audio_md5(video) == audio_md5(out))
    else:
        check("result file exists", False)
    review = os.path.join(dist, "logs", "synthetic_review.json")
    if os.path.isfile(review):
        doc = json.load(open(review, encoding="utf-8"))
        errs = {k: v for k, v in doc["summary"].items() if k in ("model_error", "detector_error", "frame_error", "frame_count_mismatch",
                                                                  "output_verification_failed", "prompt_produced_no_mask")}
        check("review log written, no model/frame errors", not errs, f"summary {doc['summary']}")
        check("the real SAM 2 ran in the packaged worker (lost/recovered events as in the source run)",
              any(k.startswith("tracking_") or k.startswith("chunk_boundary") for k in doc["summary"]), "")
    else:
        check("review log written", False)
    gui_job_checks(dist, gui, video, pts, env_min)
    return summary()


def minimal_env():
    """The current environment minus every PATH entry that holds a python.exe and minus Python variables."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON") and k.upper() not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env["PATH"] = os.pathsep.join(d for d in os.environ.get("PATH", "").split(os.pathsep) if d and not os.path.isfile(os.path.join(d, "python.exe")))
    return env


def gui_job_checks(dist, gui, video, pts, env_min):
    """The GUI exe runs a job through its normal path: open_video -> start_processing -> JobRunner -> VideoCensorWorker.exe -> review panel."""
    work = tempfile.mkdtemp()
    for mode in ("nudenet_only", "sam2"):
        stem = f"verify_gui_{mode}"
        vid = os.path.join(work, stem + ".mp4")
        shutil.copy(video, vid)
        if mode == "sam2":
            shutil.copy(pts, os.path.join(work, stem + ".points.json"))
        rep_path = os.path.join(work, stem + ".json")
        t = time.time()
        p = subprocess.run([gui, vid, "--selftest-job", rep_path], cwd=dist, timeout=1200, env=dict(env_min, TQDM_DISABLE="1"))
        rep = json.load(open(rep_path, encoding="utf-8")) if os.path.isfile(rep_path) else {}
        ok = (p.returncode == 0 and str(rep.get("status", "")).startswith("completed") and rep.get("mode") == mode
              and rep.get("worker") == "VideoCensorWorker.exe" and rep.get("frozen") is True and not rep.get("errors"))
        check(f"GUI exe runs a {mode} job through the worker exe", ok, f"{time.time() - t:.0f} s, status {rep.get('status')}, errors {rep.get('errors')}, reason {rep.get('reason')}")
        check(f"GUI {mode}: result has all frames, panel at 100 %, review log loaded",
              rep.get("output_frames") == rep.get("source_frames") and rep.get("panel_progress") == 1000 and rep.get("review_loaded") is True,
              f"{rep.get('output_frames')}/{rep.get('source_frames')} frames, progress {rep.get('panel_progress')}")
        if mode == "sam2" and rep.get("review_log") and os.path.isfile(rep["review_log"]):
            doc = json.load(open(rep["review_log"], encoding="utf-8"))
            tr = doc.get("meta", {}).get("tracker", {})
            check("review log names the SAM 2 version in the packaged build", "unknown" not in json.dumps(tr).lower(), json.dumps(tr)[:160])


def summary():
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
