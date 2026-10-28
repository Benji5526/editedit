# video-censor-pipeline

Safety-first, chunked, restartable mosaic tool for long (e.g. 40 min 4K60) videos on a local Windows + NVIDIA PC.

* **SAM 2** tracks objects/body parts you click (`pick_points.py`).
* **NudeNet** adds an auxiliary detector for the classes NudeNet itself supports (not arbitrary objects).
* Everything uncertain is **logged** (JSON + CSV); with `--fail-closed` uncertain frames are additionally mosaicked conservatively.

> **No AI model finds every instance in every frame.** This tool is built to *notice and record* likely failures and to cover some
> of them conservatively. It cannot prove that nothing was missed. Always spot-check the output, starting with the segments in the
> review log. Review flags are heuristics (see [Limitations](#limitations)).

## Project layout
```
app.py, run_app.bat        desktop app launcher (double-click run_app.bat)
censor_video.py            command-line processing (also the worker process the app starts)
pick_points.py             command-line click picker with --review
config.example.json        safety-first default options for a long 4K60 video
requirements.txt
src/
  video_io.py              FFprobe/FFmpeg: probing, timestamps, frame reader, segment writer, mux, output verification
  tracking.py              SAM 2 per chunk (+ re-seeding at chunk boundaries)
  detection.py             NudeNet auxiliary detector + temporal gap filling
  mosaic.py                mask utilities and pixelation
  quality_control.py       review events, heuristics, review log JSON/CSV
  pipeline.py              chunked, restartable pipeline; progress, cancel, incremental re-processing
  points_io.py, review.py, review_data.py    clicked-points file, review navigation, review marks/consistency checks
  app/                     PySide6 desktop app (main window, preview, job runner, dialogs, review panel)
tests/                     pytest suite (unit, pipeline with a stand-in tracker, GUI offscreen, real SAM 2)
models/                    checkpoints go here (not tracked; see models/README.md)
samples/                   synthetic test videos, example points and logs (not tracked; see samples/README.md)
input/  output/            your videos and results (kept empty, not tracked)
logs/   temp/              run logs, review logs, status; intermediate chunks (not tracked)
```

## Install (Windows 11)
**Supported and tested: native Windows 11** (Python 3.12.10, RTX 4070 Ti 12 GB, driver with CUDA 13.3, PyTorch 2.11 `cu128` wheels, SAM 2 1.1.0, NudeNet 3.4.2,
PySide6 6.12, FFmpeg 9.0.1). **WSL2 is not tested** (the desktop app needs a display; GPU pass-through and the Windows-only `taskkill`/`run_app.bat` helpers were not tried), so no WSL2 instructions are given.
Requirements: Python 3.10+, NVIDIA GPU + driver, FFmpeg and FFprobe on `PATH`, Git (optional). Pick the PyTorch wheel that matches your driver from
<https://pytorch.org/get-started/locally/>; do not install a guessed combination. SAM 2's own install notes: <https://github.com/facebookresearch/sam2>.
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # match your driver: pytorch.org
pip install -r requirements.txt
pip uninstall -y opencv-python-headless opencv-python ; pip install opencv-python==5.0.0.93        # nudenet installs the headless build; the picker needs the GUI build
```
Then download a SAM 2.1 checkpoint into `models/` – see [models/README.md](models/README.md). The checkpoint is **not** included.

Check the environment: `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print `True`.
Everything runs locally: no video, frame, point or log is sent anywhere, and no external service is required (only the one-time model download below needs internet).
Logs and review files contain local file paths and file names; keep them private (the repository ignores `logs/`, `temp/`, `output/`, `input/`, `samples/`, `models/` and `backups/`).
`censor_video.py` refuses to silently fall back to CPU.

## Desktop app (in progress: stages 1-4 of 6)
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

**Stage 4 (review, re-click, re-process):** the **Review** tab (next to Targets) lists the flagged segments of the last run's review log (opened automatically when a job
finishes, or via `File > Open review log`): filter by severity/object, hide reviewed, select a flag to jump to its frame and object (`M`/`B` = next/previous flag, `R` = mark reviewed),
red/amber ticks on the frame slider show where the unreviewed flags are. Re-click the object on a frame where it is visible, save, then `Process > Re-process with edited points`
(`Ctrl+Shift+R`): a dry run first shows how many frames are reused and how many are recomputed, then the job runs.

*Re-processing after edits.* Tracking runs forward only and every chunk is re-seeded from the previous one, so a chunk depends on the clicks in itself and in earlier chunks only.
Each job stores a signature of the clicks per chunk (`temp/<job>/job.json`); a re-run reuses all chunks before the first changed one (segments, tracker state, detector state and
review events), recomputes everything after it, and refuses to reuse anything when the settings, the video or the code version differ. Intermediate chunks are kept after success
(option "Keep intermediate chunks", disk use about the size of the output; `Process > Delete intermediate chunks`) - without them a re-process starts from scratch.

*Review log consistency.* Every run writes `run_id`, the sha256 of the points file it used and the size/mtime of the output video into the review log. The Review tab shows a red
banner when the current points file or the output video no longer match the log. "Reviewed" marks are stored next to the log (`*_review_state.json`); after a re-process they are kept
only for flags in reused chunks and reset for recomputed ones. "Reviewed" means a human looked at it, not that the frame is correct.

*Job registry (SQLite).* Every run is recorded in `logs/jobs.sqlite` (job id = run id, input/points/settings/model fingerprints, library versions, per-chunk status
`running/done/failed` with seconds and memory, output and review-log paths, status and error). When a video is opened the app looks for a job that stopped before finishing
(cancelled, failed, or crashed - a job still marked `running` whose process is gone becomes `interrupted`) and offers `Process > Resume interrupted job`; `Process > Job history`
lists all jobs of the video. **The registry is only an index**: whether chunks are reusable is decided from the files in `temp/<job>/` (job.json hashes, segment frame counts,
state and QC files, a frame fingerprint at the resume point), so a wrong or missing registry row can never make the pipeline reuse a bad chunk, and a registry that cannot be
opened or written never stops a job. Resuming also requires the same model file (name, size and a hash of its first/last MiB are part of the settings hash).

**Not yet:** EXE packaging, a negative-only click on a frame (every clicked frame needs a positive point).
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
   `censor_video.py` with the same `points.json` and `--resume --keep-temp`: only the chunks from the first changed click onwards are recomputed (see "Re-processing after edits").

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

Names used in the project brief map to these types: tracking_lost = `tracking_lost_mid_frame`/`tracking_lost_at_edge`; mask_area_jump = `mask_area_jump`;
detector_tracker_mismatch = `detection_without_mask`; chunk_boundary_failure = `chunk_boundary_resume_failed`; frame_read_error/frame_write_error = `frame_error` (the
detail says which reader/writer) and `model_error`/`detector_error`; output_validation_failure = `output_verification_failed` and `frame_count_mismatch`. The JSON holds the
fields frame, timestamp, type, severity, object_id, detail (detections, tracking result, reason), action (what the program did) and meta (input fingerprint, run id, points hash,
output); `tests/test_imports_and_logs.py` checks that the CSV and the JSON contain the same events. "Reviewed" status is kept next to the log (`*_review_state.json`), not inside it.

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
* Per chunk (`--chunk-size`, default 120; 120–240 recommended) there are two stages. **Stage 1** (main thread): the analysis reader (frames downscaled by FFmpeg to `--analysis-width`, default 1920) writes JPEGs for SAM 2 and feeds NudeNet; SAM 2 tracks; QC runs. **Stage 2** (render thread): a **second reader streams full-resolution frames one by one** (read ahead on a background thread), the mosaic is applied at full resolution and encoded to a segment file, which is verified. The two stages **overlap**: stage 2 of chunk c runs while stage 1 of chunk c+1 runs, which is safe because tracking depends on the previous chunk's *tracking state*, not on its rendered video. A chunk only counts as done (job.json, registry) after its segment is verified and its state, snapshotted at the end of *its own* stage 1, is saved; cancel and errors in either stage propagate to the job and a chunk in flight is simply redone on resume. Masks are kept bit-packed. Per-chunk temp data is deleted. Memory is bounded by the chunk size, not the video length (see "Long runs").
* The final file is made by concatenating segments (stream copy) + original audio into `*.partial`, verifying it, and only then renaming it.
* Mosaic style (`--mosaic-style`, `--block` = strength in full-resolution px): `pixelate` (default, block grid anchored to the image), `solid` (black; nothing of the original survives) or
  `blur` (Gaussian, sigma = block/2; the weakest of the three, prefer pixelate or solid for sensitive content).
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

## Long runs: speed, memory, disk (measured)
Hardware: RTX 4070 Ti 12 GB, 16 cores / 24 threads, 32 GB RAM, Windows 11. Footage: synthetic 3840x2160, 60 fps, H.264 (`tools/soak_test.py`); SAM 2.1 small + NudeNet on every frame
(`--detect-every 1`) + `--fail-closed`, `libx264 veryfast`, 60-frame chunks, analysis width 1920. A chunk is 60 frames (1 s).

| stage (per 60-frame chunk) | first version | now | what changed |
|---|---|---|---|
| reading full-resolution frames (4K) | 63 ms/frame | 20 ms/frame (7 ms visible with read-ahead) | unbuffered pipe, read into a preallocated array (no extra 25 MB copy), prefetch thread |
| SAM 2 frame loading | 2.2 s | 0.5 s | 8 threads, tensors bit-identical to SAM 2's own loader |
| SAM 2 propagation | 2.6 s | 2.6 s | unchanged (GPU bound, ~23 frames/s for the small model) |
| NudeNet detection (CPU, 1920 px) | 0.5 s | 0.5 s | not a bottleneck: 8 ms/frame |
| render stage | 5.5 s | 2.5 s | see reading. `h264_nvenc` gave no gain in the first version (the bottleneck was the reader, not the encoder); it was not re-measured after the fixes and the overlap |
| **wall time per chunk** | **12.9 s (4.7 fps)** | **6.2 s (10 fps)** | the two stages now overlap |

At ~10 fps a 40 minute 4K60 video (144,000 frames) takes roughly **4 hours** with this configuration; that is an extrapolation, not a measurement of a 40 minute run. A 3 minute run is in the
table below. Per-stage times of every chunk are in the `chunk_done` events, `temp/<job>/job.json` and the log (`tools/leak_probe.py` isolates memory growth per component).
`--analysis-width 1280` is a little faster (about 8%) but gives NudeNet and the tracker fewer pixels; SAM 2 itself works at 1024 px.

**Memory.** Resident memory is bounded: it rises in steps to a high-water mark and then stays flat (80 identical SAM 2 chunks: one +0.5 GB step, then flat). It does not grow with the video
length. Rule of thumb for the tracker's input: about 12.6 MB per frame of (chunk size + up to `--stale-prefix-frames` re-run frames) on top of ~1.8 GB base, e.g. 120+90 frames = ~2.6 GB.
Chunks that re-run a lost object are the ones that raise the mark. GPU memory peaked at about 0.7 GB for the small model. Nothing checks free RAM before a run: close memory hungry programs or lower `--chunk-size`.

**Disk.** Before starting, free space is compared with an estimate (1.5 x the source size for the intermediate chunks and 1.5 x for the result; doubled when both are on one drive); below the estimate the job
refuses to start (`--skip-disk-check` overrides), below twice the estimate it warns. Intermediate chunks are kept after success only with `--keep-temp` (the app's "Keep intermediate chunks").
A disk that fills up mid-run fails the job with the reason; completed chunks stay valid for `--resume`.

**Hours-long jobs.** The worker asks Windows not to go to sleep while it runs (the display may still turn off). Progress, ETA and per-chunk memory are reported continuously; a killed or crashed worker is
recorded as `interrupted` in the job registry and can be resumed.

**Output integrity.** The result is verified before it is renamed: frame count, size, CFR timestamps (<0.5 frame), duration, audio duration, and the audio packets must be **bit-identical** to the source
(MD5 of the stream-copied packets). Audio codecs MP4 cannot hold (e.g. `pcm_alaw`, `adpcm_ima_wav`, `wmav2`) are re-encoded to AAC 256 kbit/s with an `audio_reencoded` warning. Segment timestamps use a container
timescale that makes every frame an exact number of ticks, so concatenating thousands of chunks of 59.94/29.97/23.976 fps footage does not drift (an earlier 90 kHz timescale lost half a tick per odd-sized chunk).
Non-ASCII paths (Korean folder or file names, spaces) work: OpenCV's `imwrite` fails silently on them on Windows, so frames are written with Python file I/O.

**Soak test (`python tools/soak_test.py --minutes 3`)**, same hardware and settings as above, 120-frame chunks: synthetic 3 minute 3840x2160 60 fps clip (10,800 frames, 90 chunks, 12 GB source,
AAC audio) with a magenta box that bounces across the frame; SAM 2 small + NudeNet on every frame + `--fail-closed`. The worker process tree was **killed hard after chunk 27** (simulated crash), then resumed.

| check | result |
|---|---|
| job registry after the kill | `interrupted`, 27 chunks done, 2 in-flight chunks marked failed |
| resume | reused 27 chunks (validated from their files), processed the other 63, finished `completed_no_flags`; 0 error events, 0 model errors |
| resident memory (RSS) over 90 chunks | 1,862-2,095 MB; slope -3.5 MB/chunk before the kill, -0.2 MB/chunk after (no growth); levels 2.04 -> 2.09 -> 2.01 GB in steps, then flat |
| GPU memory peak | 697 MB, constant |
| output | 10,800 frames, 3840x2160, duration 180.0 s (= source), CFR timestamps within 0.0003 ms, **audio bit-identical to the source**, whole file decoded (frame count verified, 87 s) |
| mosaic at the box, 8 sampled frames (3 from chunks done before the kill, 5 after the resume) | the 32 px block at the box lost all texture (std 5-6 -> 0.0) while a background block kept it (x0.65-1.11, codec variation on a noise texture) |
| speed | 8.4 fps steady (14.3 s per 120-frame chunk: reading+JPEG 2.2, NudeNet 2.8, SAM 2 8.2, QC 0.2, render 7.4 s), slower than the 10 fps above because this source is a 530 Mbit/s noise clip and everything is on an HDD |
| wall time | 42.5 min for 3 minutes of footage in total, including the first run up to the kill, the final merge and the checks |

What this does **not** show: a 40 minute run, real footage, tracking quality, or behaviour when the disk or RAM is nearly full. The drive that held the source, the chunks and the result in this run is a 5400 rpm-class
**hard disk (HDD)**: the final merge of the 12 GB result ran at ~20 MB/s (reading the chunks and writing the result on the same disk); merge plus verification took ~16 minutes, with the `+faststart` rewrite that was still on by default in this run (now opt-in, see below).
Put `--work-dir` on an SSD if you have one; the job warns when the work or output folder is on an HDD. `+faststart` (`--faststart`) rewrites the whole result a second time and is only useful for progressive web streaming, so it is off by default.


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
  | 86-142 | object reappears (and moved 80 px meanwhile) | SAM 2 finds it again at frame 86: IoU 0.84-0.95, mean 0.92 (identical in 3 runs); `tracking_recovered` is logged as **warn** ("verify it is the same object") |

  Takeaway: on this easy target, re-running SAM 2 from the object's last good frame through the hidden frames (see "Chunk boundaries") recovers the object right when it reappears.
  An earlier version that injected the stale mask at the boundary instead produced a mask in the wrong place (IoU about 0 for the rest of the clip) - that is why the current design exists.
  The log still marks the recovery as unverified, because the code cannot tell "same object" from "something that looks similar". Expect real footage to be harder than this.
  (A first version of this table was wrong: the measuring code assumed a 1-frame prefix and compared one chunk's masks with ground truth shifted by 10 frames. Fixed and re-measured.)

  `test_incremental.py` / `test_plan_and_sam2_incremental.py` check re-processing after re-clicks: only the chunks from the first changed click onwards are recomputed, and the
  result equals a from-scratch run with the same points. With the real checkpoint, adding a click at frame 90 reused 2 of 4 chunks; the IoU of frames 90-142 stayed at about 0.93
  because the object was already tracked correctly there - the re-click mechanics are verified, an accuracy gain from re-clicking is not demonstrated on this clip.

Neither the test double nor the synthetic clip says anything about NudeNet's detection quality or about SAM 2 on real footage.

## Limitations
* Real SAM 2 tracking was verified only on synthetic clips (a flat box on a noisy background). On your footage it is **unverified**; occlusion, fast motion, scale change, look-alike objects and re-entry are the usual weak spots, and a recovery after a long loss is always logged as "verify it is the same object".
* SAM 2 `sam2._C` (hole-filling post-processing) is not built on Windows with the PyPI wheel; SAM 2 runs without it (upstream says this rarely matters).
* Tracking only runs forward from your first click; there is no backward pass.
* "Left the frame" vs "tracking failed" is a heuristic (border + shrinking mask) and can be wrong either way.
* NudeNet's bundled model is small (320 px input) and CPU-only here; it misses things. It never replaces clicking targets.
* Detection/tracking run on downscaled frames (`--analysis-width`); very small targets may be lost.
* Audio is stream-copied and verified bit-identical to the source (unless AAC re-encoding was needed); A/V sync is checked through durations and the untouched audio timeline, not by listening. A source whose audio starts at a different time than its video is not corrected.
* `--resume` relies on frame-exact FFmpeg seeking; it checks one frame fingerprint at the resume point and refuses if it differs.
* Speed: about 10 fps for 4K60 with SAM 2 + per-frame NudeNet on this PC (see "Long runs"); the source is decoded twice (analysis and render), frames take a JPEG round trip into SAM 2, NudeNet runs on the CPU. A 40 minute video has not been run end to end.
