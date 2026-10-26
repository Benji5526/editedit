# video-censor-pipeline

Safety-first, chunked, restartable mosaic tool for long (e.g. 40 min 4K60) videos on a local Windows + NVIDIA PC.

* **SAM 2** tracks objects/body parts you click (`pick_points.py`).
* **NudeNet** adds an auxiliary detector for the classes NudeNet itself supports (not arbitrary objects).
* Everything uncertain is **logged** (JSON + CSV); with `--fail-closed` uncertain frames are additionally mosaicked conservatively.

> **No AI model finds every instance in every frame.** This tool is built to *notice and record* likely failures and to cover some
> of them conservatively. It cannot prove that nothing was missed. Always spot-check the output, starting with the segments in the
> review log. Review flags are heuristics (see [Limitations](#limitations)).

## Install (Windows 11)
Requirements: Python 3.10+ (tested 3.12), NVIDIA GPU + driver, FFmpeg and FFprobe on `PATH`, Git (optional).
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # match your driver: pytorch.org
pip install -r requirements.txt
pip uninstall -y opencv-python-headless opencv-python ; pip install opencv-python==5.0.0.93        # nudenet installs the headless build; the picker needs the GUI build
```
Then download a SAM 2.1 checkpoint into `models/` – see [models/README.md](models/README.md). The checkpoint is **not** included.

Check the environment: `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print `True`.
`censor_video.py` refuses to silently fall back to CPU.

## Desktop app (in progress: stages 1-3 of 6)
```powershell
python app.py [video]        # or double-click run_app.bat (no console window)
```
**Stage 1-2 (selection):** open a video (analysed in the background), preview with frame stepping (+-1, +-10, +-1 s, slider, play/pause), click targets
(left = positive, right = negative, several objects with their own colour/ID, undo, clear, delete), save/load the points file
(`<video>.points.json` next to the video, auto-loaded when it matches the file's fingerprint, size, fps and frame count), unsaved-change prompts.
Clicks are converted from widget to full-resolution frame coordinates with letterbox handling (clicks on the black bars are ignored, not clamped);
the preview is decoded by FFmpeg (same decoder, rotation and frame numbering as the processing pipeline) and never blocks the UI thread.

**Stage 3 (processing):** `Process > Start processing... (Ctrl+R)` opens the settings (SAM 2 checkpoint, output, chunk size, analysis width, NudeNet,
fail-closed, mosaic block/padding, audio, encoder, test run on the first N frames, resume). The job runs in a **separate worker process**
(`censor_video.py --progress-json`), so the UI stays responsive and a crash/out-of-memory in the worker cannot take the app down. The panel shows the stage
(reading+detecting / tracking / mosaic+encoding), chunk n/N, overall progress, elapsed and estimated remaining time, per-chunk memory (RSS, GPU peak) and the log.
* **Cancel** (`Ctrl+.`): the worker stops at the next safe point (between frames/stages), discards the chunk in flight, keeps completed chunks and writes status
  `cancelled` (exit code 3). If it does not stop within 60 s the whole process tree is killed. **Resume** continues from the completed chunks (same input, points and settings).
* **Failure**: a failed job shows the reason in red, leaves no final file (a failed verification keeps `*.partial`, which is not a valid result) and offers Resume.
* **Result**: `completed_no_flags` or `completed_with_review_required` (error/warning counts; check the review log before using the video), with buttons to open the result and the logs folder.
* While a job runs, opening files and editing points are locked; closing the window asks to cancel the job first.
* Before starting, unsaved points must be saved (the worker reads the saved file); a missing checkpoint, an unusable output path or "nothing to do" is refused up front.

**Not yet in the app (still command line):** the review screen (use `pick_points.py --review` below), re-processing only the affected chunks after re-clicks
(a changed points file means a fresh run today), the SQLite job database (state lives in `temp/<job>/job.json` and `logs/*_status.json`), EXE packaging.
Worker protocol (stdout lines `@@PROGRESS {json}`: `stage`, `start`, `progress`, `chunk_done`, `finished`) is documented in `src/app/job_runner.py`.

## Usage (command line)
1. **Pick targets** (OpenCV window; left click = this is the object, right click = this is NOT; `n` new object, `a/d` ±1 frame, `q/e` ±10, `z/c` ±1 s, trackbar, `u` undo, `s` save – full key list in `pick_points.py --help`/docstring):
   ```powershell
   python pick_points.py input\video.mp4 -o points.json
   ```
   Click on the **first frame where the object is visible**; tracking of an object starts at its earliest click (frames before it are not tracked).
   Re-click on later frames where tracking drifts. Coordinates are saved in full-resolution decoded-frame pixels together with the
   video fingerprint, size, fps, frame count and rotation; `censor_video.py` rejects a points file that does not match the video.
2. **Short test run first** (first 600 frames ≈ 10 s at 60 fps):
   ```powershell
   python censor_video.py input\video.mp4 --points points.json --sam2-checkpoint models\sam2.1_hiera_small.pt --max-frames 600 --chunk-size 120 --fail-closed -o output\test.mp4
   ```
3. **Full run (40 min 4K60, safety-first)**:
   ```powershell
   python censor_video.py input\video.mp4 --points points.json --sam2-checkpoint models\sam2.1_hiera_small.pt --config config.example.json -o output\video_censored.mp4
   ```
   Interrupted? Re-run the same command plus `--resume` (completed chunks are validated first; changed settings/points/input are refused).
   NudeNet only (no clicked targets): leave out `--points`. Clicked targets only: add `--no-nudenet`.
4. **Review and fix**: read the review log (see below), watch the flagged segments in the output, and fix lost objects with the review mode of the picker:
   ```powershell
   python pick_points.py input\video.mp4 -o points.json --review logs\video_review.json
   ```
   `m` / `b` jump to the next / previous flagged frame (start and end of each flagged segment, severity warn and above; `--review-min-severity info` for all). The object named
   by the flag becomes the current object, and the banner lists the flags at that frame. Left-click the object on a frame where it is visible again, press `s`, then re-run
   `censor_video.py` with the same `points.json` (a changed points file means a fresh run, not `--resume`). On the synthetic clip this workflow took the IoU of the frames after
   a full occlusion from ~0.1 to a mean of 0.93 (see Tests).

Exit code: `0` finished, `2` finished but error-level review events exist, `1` failed (no final file is produced), `3` cancelled (resumable).

## Outputs
| file | content |
|---|---|
| `output/<name>_censored.mp4` | final video; only exists after verification passed (frames, size, CFR timestamps, duration, audio) |
| `output/<name>_censored.mp4.partial` | kept when verification failed – **not a valid result** |
| `logs/<name>_review.json` / `.csv` | every event + merged `segments` (type, object, start/end frame and time, action taken) |
| `logs/<name>_detections.csv` | NudeNet detections per frame: class, score, box, and whether the box was *filled* by temporal hold |
| `logs/<name>_status.json` | `completed_with_review_required`, `completed_no_flags` or `failed` + reason |
| `logs/<name>.log` | timestamps, chunk progress, RSS/GPU memory per chunk, errors |

Review JSON from PowerShell: `python -c "import json;d=json.load(open('logs/video_review.json'));print(d['summary']);[print(s) for s in d['segments'] if s['severity']!='info']"`
(or open the CSV in a spreadsheet; sort by `severity`, `type`).

### Event types (all heuristics)
| type | meaning | severity |
|---|---|---|
| `tracking_lost_mid_frame` | a tracked mask vanished and its last box was not sliding out of the frame | warn |
| `tracking_lost_at_edge` | mask vanished while shrinking at the frame border (probably left the frame) | info |
| `tracking_recovered` | a lost object has a mask again. **warn** when it was lost for more than 5 frames or reappeared far from where it vanished: the mask may belong to something else | info / warn |
| `mask_area_jump` | mask area changed by more than `--jump-ratio` (default 2×) between adjacent frames | warn |
| `detection_without_mask` | a NudeNet box overlaps no tracked mask (only logged while a clicked object is tracked). Both are mosaicked anyway | info |
| `chunk_boundary_resumed` / `_from_stale_mask` / `_resume_failed` | how tracking was re-seeded at a chunk boundary (`_from_stale_mask` = object was hidden, SAM 2 re-run from its last good frame) | info / warn / error |
| `prompt_produced_no_mask` | SAM 2 returned nothing for your click | error |
| `tracking_abandoned` | no mask for `--max-stale-frames`; object no longer followed | warn |
| `model_error`, `detector_error`, `frame_error` | inference/IO failure in a chunk/frame | error |
| `frame_count_mismatch`, `output_verification_failed` | output integrity problems | error |
| `fail_closed_mosaic_applied` | `--fail-closed` added a conservative mosaic here (reason + mode in detail) | info |

`tracking_lost_*` does **not** mean something was missed (the object may be occluded or off-screen) and the *absence* of events
does **not** mean nothing was missed.

## Fail-closed (`--fail-closed`)
Off by default. When on, frames whose result is not trustworthy get an extra mosaic:

* object lost mid-frame → last known box grown by `--fail-closed-expand` (default +100 % per side), up to `--fail-closed-max-frames` per episode (`--fail-closed-mode full` = whole frame instead)
* `mask_area_jump` → union box of previous and current mask, grown the same way
* SAM 2 error in a chunk, or NudeNet error on a frame → **whole frame(s)**

**This will cover more than the target** (e.g. a whole second of the screen when an object is merely occluded) and it is not a fix for
detection failures: an object that the models never see produces no event and no fail-closed mosaic. Use it to reduce exposure risk, not to eliminate it.
Without the flag the same events are logged but the frames pass through as computed.

## How it works
* `video_io.py` – FFprobe metadata, real packet timestamps, VFR heuristic, FFmpeg raw-frame reader (`-fps_mode passthrough`, auto-rotation) and per-chunk encoder; audio is stream-copied from the source at the end.
* Per chunk (`--chunk-size`, default 120; 120–240 recommended): the analysis reader (frames downscaled by FFmpeg to `--analysis-width`, default 1920) writes JPEGs for SAM 2 and feeds NudeNet; SAM 2 tracks; QC runs; then a **second reader streams full-resolution frames one by one**, the mosaic is applied at full resolution and encoded to a segment file. Masks are kept bit-packed. Per-chunk temp data is deleted. Memory is bounded by the chunk size, not the video length (measured: see below).
* The final file is made by concatenating segments (stream copy) + original audio into `*.partial`, verifying it, and only then renaming it.
* Mosaic area = union of (all objects' masks over ±`--temporal-radius` frames, grown by `--mask-pad-px`) ∪ (NudeNet boxes grown by `--det-expand`/`--det-pad-px`, with short gaps bridged and boxes held `--det-hold` frames) ∪ fail-closed regions. A union cannot lose one of its members. The pixel grid is anchored to the image so blocks do not shimmer as masks move.
* NudeNet: model output is post-processed here with a configurable `--det-threshold` (the library's own `detect()` hard-codes a 0.2/0.25 floor). Lowering the threshold gives more detections and more false positives, not guaranteed recall. `--nudenet-tiles 2` additionally scans overlapping crops (small regions get more resolution; slower). It runs on CPU unless you install `onnxruntime-gpu` yourself.

### Chunk boundaries and SAM 2 state (important)
SAM 2's internal memory bank cannot be carried across separately initialised chunks. Each chunk after the first therefore starts with one **overlap frame**
(the previous chunk's last frame) prompted with the previous chunk's last mask via `add_new_mask`. That re-seeds the object but is **not** the same as
uninterrupted tracking: the appearance history is lost at every boundary, so drift and identity swaps are more likely right after one.

If an object has **no mask at the boundary**, a stale mask is *not* injected (that would produce a mask in the wrong place that looks like a successful
recovery). Instead the chunk is extended backwards to the frame where the object was last seen (up to `--stale-prefix-frames`, default 90, kept in an on-disk
ring buffer), the last good mask is placed there, and SAM 2 runs through the hidden frames like it would inside one chunk, so SAM 2's own visibility
decision stands. This costs extra RAM/time for that chunk (all frames are loaded by SAM 2; ~12.6 MB each on the host). If the object was lost for longer than
that, it is not re-seeded (`chunk_boundary_resume_failed`, error) and stays flagged as lost until you re-click it. Larger `--chunk-size` means fewer boundaries but more RAM/VRAM.

## Input policy
* **VFR** input is rejected (re-encoding at a constant rate would shift timing/audio). `--allow-vfr` accepts it: frame count is preserved, timing becomes constant-rate at the nominal fps, audio may drift. Detection is a heuristic on packet timestamps.
* **HDR / 10-bit** input is rejected unless `--allow-hdr` (output is 8-bit SDR).
* Rotation metadata is applied by FFmpeg for both the picker and the pipeline, so click coordinates match.
* Output is H.264 (`libx264`, or `--encoder h264_nvenc`), 8-bit 4:2:0, colour tags copied for bt709/bt601; other source codecs/containers are re-encoded.

## Tests
```powershell
python -m pytest tests -q
```
* `test_pipeline.py` uses a **colour-threshold stand-in for SAM 2** (named test double in `tests/helpers.py`) on synthetic footage to check chunking, boundary re-seeding, QC events,
  fail-closed, resume, audio and output verification.
* `test_sam2_api_smoke.py` drives the real SAM 2 code with random weights (API compatibility only).
* `test_sam2_real.py` runs the **real `sam2.1_hiera_small.pt`** on the synthetic clip (skipped when the checkpoint is absent) and scores the masks against ground truth.
  Measured on the synthetic clip (a flat magenta 60 px square on a noisy background, 40-frame chunks, RTX 4070 Ti):

  | frames | what happens | result |
  |---|---|---|
  | 20-69 | clicked at 20, tracked across the boundary at frame 40 | mean IoU 0.91 (min 0.89) |
  | 70-85 | object hidden for 16 frames (boundary at 80 falls inside) | `tracking_lost_mid_frame` at 70, 16 frames of fail-closed mosaic |
  | 86-118 | object reappears (and moved 80 px meanwhile) | SAM 2 produced a mask again at 86 (`tracking_recovered`, **warn**) but at the wrong/lagging place: IoU 0.0-0.1 |
  | 122-143 | after the next chunk boundary re-seeded from that mask | IoU 0.84-0.95 |

  Takeaway: the square is an easy target and SAM 2 small followed it well until it was occluded. After a full occlusion it re-found the object late and wrongly; the
  review log shows the loss and flags the recovery for verification, but the frames 86-118 were mosaicked in the wrong place (the real target was uncovered).
  Re-click the object after occlusions (the log tells you where; use `pick_points.py --review`): clicking it again at frame 90 raised the IoU of frames 90-142 to a mean of 0.93 (min 0.82).
  Expect real footage to be harder than this.

Neither the test double nor the synthetic clip says anything about NudeNet's detection quality or about SAM 2 on real footage.

## Limitations
* Real SAM 2 tracking was verified only on the synthetic clip above. On your footage it is **unverified**; occlusion (shown above to fail after a full occlusion with movement), fast motion, scale change and re-entry are weak spots.
* SAM 2 `sam2._C` (hole-filling post-processing) is not built on Windows with the PyPI wheel; SAM 2 runs without it (upstream says this rarely matters).
* Tracking only runs forward from your first click; there is no backward pass.
* "Left the frame" vs "tracking failed" is a heuristic (border + shrinking mask) and can be wrong either way.
* NudeNet's bundled model is small (320 px input) and CPU-only here; it misses things. It never replaces clicking targets.
* Detection/tracking run on downscaled frames (`--analysis-width`); very small targets may be lost.
* Audio is copied untouched; sync is verified by duration comparison, not by listening. Source files with a video/audio start-time offset are only checked via durations.
* `--resume` relies on frame-exact FFmpeg seeking; it checks one frame fingerprint at the resume point and refuses if it differs.
* Speed was deliberately not optimised (two decodes of the source, JPEG round-trip, CPU NudeNet).
