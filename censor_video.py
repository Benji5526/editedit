"""Safety-first streaming video mosaic. See README.md.

    python censor_video.py input.mp4 --points points.json --sam2-checkpoint models/sam2.1_hiera_small.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading

from src.job_manager import file_signature
from src.pipeline import Config, Pipeline, plan_reprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track user-selected objects with SAM 2 and (optionally) detect NudeNet classes, "
                    "then mosaic them in a long video using chunked streaming. Flags are heuristics; "
                    "always review the log. Exit code: 0 ok, 2 finished with error-level review events, 1 failed, 3 cancelled (resumable).")
    p.add_argument("input", help="input video")
    p.add_argument("-o", "--output", help="final output path (default output/<name>_censored.mp4)")
    p.add_argument("--points", help="selection JSON from pick_points.py (SAM 2 targets)")
    p.add_argument("--config", help="JSON file with default option values (see config.example.json); CLI wins")
    g = p.add_argument_group("streaming")
    g.add_argument("--chunk-size", type=int, default=120, help="frames per chunk, 120-240 recommended (default 120)")
    g.add_argument("--analysis-width", type=int, default=1920, help="width of the downscaled frames used for tracking/detection")
    g.add_argument("--max-frames", type=int, help="process only the first N frames (testing)")
    g.add_argument("--work-dir", default="temp")
    g.add_argument("--log-dir", default="logs")
    g.add_argument("--resume", action="store_true", help="continue an interrupted run (validates completed chunks first)")
    g.add_argument("--keep-temp", action="store_true", help="keep job directory after success")
    g.add_argument("--progress-json", action="store_true",
                   help="print machine-readable progress lines ('@@PROGRESS {json}') on stdout (used by the desktop app)")
    g.add_argument("--plan-only", action="store_true",
                   help="print what a --resume run would reuse (event 'plan') and exit without processing")
    g.add_argument("--cancel-file", help="stop safely at the next checkpoint when this file appears (exit code 3, resumable)")
    g = p.add_argument_group("SAM 2")
    g.add_argument("--sam2-checkpoint", help="e.g. models/sam2.1_hiera_small.pt (official download, see models/README.md)")
    g.add_argument("--sam2-config", help="e.g. configs/sam2.1/sam2.1_hiera_s.yaml (inferred from checkpoint name if omitted)")
    g.add_argument("--device", default="cuda", help="cuda (default) or cpu; never silently falls back")
    g.add_argument("--max-stale-frames", type=int, default=600, help="stop following a lost object after this many frames without a mask")
    g.add_argument("--stale-prefix-frames", type=int, default=90,
                   help="at a chunk boundary, re-run SAM 2 from a lost object's last good frame if it is at most N frames back "
                        "(extra RAM/time for that chunk; 0 disables)")
    g.add_argument("--jump-ratio", type=float, default=2.0, help="adjacent-frame mask area ratio flagged as suspicious")
    g = p.add_argument_group("NudeNet")
    g.add_argument("--no-nudenet", action="store_true", help="disable the auxiliary detector")
    g.add_argument("--detect-every", type=int, default=1, help="run NudeNet every N frames (1 = every frame)")
    g.add_argument("--det-threshold", type=float, default=0.25, help="score threshold; lower = more detections AND more false positives")
    g.add_argument("--det-classes", nargs="+", help="NudeNet class names (default: exposed classes)")
    g.add_argument("--det-expand", type=float, default=0.25, help="box growth per side as fraction of box size")
    g.add_argument("--det-pad-px", type=float, default=12.0, help="extra box padding in full-res pixels")
    g.add_argument("--det-fill-gap", type=int, default=6, help="bridge detection gaps up to N frames")
    g.add_argument("--det-hold", type=int, default=3, help="keep a box N frames after the last detection")
    g.add_argument("--nudenet-model", help="path to a different NudeNet onnx (e.g. 640m)")
    g.add_argument("--nudenet-res", type=int, default=320, help="inference resolution of that model")
    g.add_argument("--nudenet-tiles", type=int, default=1, help="also scan a NxN grid of crops (slower, finds smaller regions)")
    g = p.add_argument_group("mosaic")
    g.add_argument("--block", type=int, default=32, help="mosaic strength in full-res pixels: block size (pixelate) or 2 x sigma (blur)")
    g.add_argument("--mosaic-style", choices=["pixelate", "blur", "solid"], default="pixelate",
                   help="pixelate (default), blur (weakest) or solid black")
    g.add_argument("--mask-pad-px", type=float, default=24.0, help="grow SAM 2 masks by this many full-res pixels")
    g.add_argument("--temporal-radius", type=int, default=2, help="union masks of +-N neighbouring frames")
    g = p.add_argument_group("fail-closed (may cover more than the target)")
    g.add_argument("--fail-closed", action="store_true", help="mosaic conservatively when results are uncertain")
    g.add_argument("--fail-closed-mode", choices=["bbox", "full"], default="bbox",
                   help="bbox: expanded last-known box; full: whole frame. Model/detector errors always use the whole frame")
    g.add_argument("--fail-closed-expand", type=float, default=1.0, help="bbox growth per side (fraction of size)")
    g.add_argument("--fail-closed-max-frames", type=int, default=180, help="max frames of conservative mosaic per loss episode")
    g = p.add_argument_group("output")
    g.add_argument("--encoder", default="libx264", choices=["libx264", "h264_nvenc"])
    g.add_argument("--crf", type=int, default=16)
    g.add_argument("--preset", default="medium")
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--deep-verify", action="store_true", help="decode the whole output to count frames (slow)")
    g.add_argument("--allow-vfr", action="store_true", help="accept variable-frame-rate input (timing becomes constant-rate)")
    g.add_argument("--allow-hdr", action="store_true", help="accept HDR/10-bit input (converted to 8-bit SDR)")
    g.add_argument("--faststart", action="store_true", help="put the MP4 index at the start of the result (web streaming); rewrites the whole file once more")
    g.add_argument("--skip-disk-check", action="store_true", help="do not refuse to start when the free disk space looks too small")
    g.add_argument("--allow-points-mismatch", action="store_true", help="accept points from a different file with same size/frames")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config, encoding="utf-8") as f:
            defaults = {k.replace("-", "_"): v for k, v in json.load(f).items() if not k.startswith("_")}
        valid = {a.dest for a in parser._actions}
        unknown = set(defaults) - valid
        if unknown:
            parser.error(f"unknown keys in config: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    a = parser.parse_args(argv)
    if not (1 <= a.chunk_size <= 1000):
        parser.error("--chunk-size must be positive (120-240 recommended)")
    if a.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if not a.points and a.no_nudenet:
        parser.error("nothing to do: give --points and/or leave NudeNet enabled")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        input=a.input, output=a.output or "", points=a.points, work_dir=a.work_dir, log_dir=a.log_dir,
        chunk_size=a.chunk_size, analysis_width=a.analysis_width, max_frames=a.max_frames,
        max_stale_frames=a.max_stale_frames, stale_prefix_frames=a.stale_prefix_frames, jump_ratio=a.jump_ratio,
        detect_every=a.detect_every, det_threshold=a.det_threshold, det_expand=a.det_expand, det_pad_px=a.det_pad_px,
        det_fill_gap=a.det_fill_gap, det_hold=a.det_hold, block=a.block, mosaic_style=a.mosaic_style, mask_pad_px=a.mask_pad_px,
        temporal_radius=a.temporal_radius, fail_closed=a.fail_closed, fail_closed_mode=a.fail_closed_mode,
        fail_closed_expand=a.fail_closed_expand, fail_closed_max_frames=a.fail_closed_max_frames,
        encoder=a.encoder, crf=a.crf, preset=a.preset, keep_audio=not a.no_audio, deep_verify=a.deep_verify,
        allow_vfr=a.allow_vfr, allow_hdr=a.allow_hdr, allow_points_mismatch=a.allow_points_mismatch, skip_disk_check=a.skip_disk_check, faststart=a.faststart, resume=a.resume,
        extra={"keep_temp": a.keep_temp, "nudenet": not a.no_nudenet, "det_classes": a.det_classes,
               "nudenet_model": a.nudenet_model, "nudenet_res": a.nudenet_res, "nudenet_tiles": a.nudenet_tiles,
               "sam2_ckpt_sig": file_signature(a.sam2_checkpoint) if a.sam2_checkpoint and os.path.isfile(a.sam2_checkpoint) else "",
               "device": a.device},
    )
    emit_lock = threading.Lock()

    def emit(ev: dict):
        if a.progress_json:
            with emit_lock:                                   # progress comes from the analysis and the render thread
                sys.stdout.write("@@PROGRESS " + json.dumps(ev) + "\n")
                sys.stdout.flush()

    cancel_check = (lambda: os.path.exists(a.cancel_file)) if a.cancel_file else None
    if a.plan_only:
        try:
            plan = plan_reprocess(cfg)
        except Exception as e:  # noqa: BLE001
            emit({"event": "finished", "exit_code": 1, "status": "failed", "reason": f"cannot plan: {e}"})
            return 1
        emit({"event": "plan", **plan})
        emit({"event": "finished", "exit_code": 0, "status": "planned"})
        if not a.progress_json:
            print(json.dumps(plan, indent=1))
        return 0
    tracker = detector = None
    emit({"event": "stage", "stage": "loading_models"})
    try:
        if a.points:
            from src.tracking import SAM2Tracker, TrackerError
            if not a.sam2_checkpoint:
                parser.error("--sam2-checkpoint is required with --points")
            tracker = SAM2Tracker(a.sam2_checkpoint, a.sam2_config, device=a.device)
            logging.info("SAM 2: %s", tracker.describe())
        if not a.no_nudenet:
            from src.detection import NudenetDetector, nudenet_version
            detector = NudenetDetector(a.det_threshold, classes=a.det_classes, model_path=a.nudenet_model,
                                       resolution=a.nudenet_res, tiles=a.nudenet_tiles)
            logging.info("NudeNet %s classes=%s threshold=%.2f (CPU onnxruntime unless configured otherwise)",
                         nudenet_version(), detector.classes, a.det_threshold)
    except Exception as e:  # noqa: BLE001
        logging.error("model setup failed: %s", e)
        emit({"event": "finished", "exit_code": 1, "status": "failed", "reason": f"model setup failed: {e}"})
        return 1
    emit({"event": "stage", "stage": "processing"})
    rc = Pipeline(cfg, tracker, detector, progress=emit, cancel_check=cancel_check).run()
    stem = os.path.splitext(os.path.basename(a.input))[0]
    status = {}
    try:
        with open(os.path.join(a.log_dir, f"{stem}_status.json"), encoding="utf-8") as f:
            status = json.load(f)
    except (OSError, ValueError):
        pass
    emit({"event": "finished", "exit_code": rc, **{k: status.get(k) for k in ("status", "reason", "output", "review_log", "error_events", "warn_events", "problems")}})
    if rc == 0 and not a.progress_json:
        print("finished. Review flags can still exist (warn/info) - open the review log.")
    return rc

if __name__ == "__main__":
    sys.exit(main())
