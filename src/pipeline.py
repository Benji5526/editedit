"""Chunked, restartable, fail-closed-capable mosaic pipeline.

Per chunk:  analysis frames -> JPEGs on disk (SAM 2 input) + NudeNet on each frame
            -> SAM 2 tracking (re-seeded from the previous chunk's last mask)
            -> QC heuristics -> render on FULL-resolution frames -> encode a segment
After all chunks: concat segments + original audio -> verify -> atomic rename to the final name.

Only one chunk of analysis data is ever held; full-resolution frames are streamed one at a time.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import quality_control as qc
from .job_manager import JobDB, db_path_for
from .detection import Det, fill_detections
from .mosaic import (STYLES, apply_mosaic, boxes_to_mask, dilate_mask, expand_box, mask_bbox,
                     temporal_union, upscale_mask)
from .points_io import Prompts, load_points
from .tracking import Init, Tracker, TrackerError, pack_mask, unpack_mask
from .video_io import (FrameReader, SegmentWriter, VideoError, VideoInfo, concat_and_mux, count_packets,
                       is_hdr, probe_video, verify_output)

CODE_VERSION = "1"
PROMPT_NO_MASK = "prompt_produced_no_mask"
DET_ERROR = "detector_error"

log = logging.getLogger("censor")


class JobFailed(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    """Raised at a safe point when the user asked to stop. Completed chunks stay valid for --resume."""


@dataclass
class Config:
    input: str = ""
    output: str = ""
    points: Optional[str] = None
    work_dir: str = "temp"
    log_dir: str = "logs"
    chunk_size: int = 120
    analysis_width: int = 1920
    max_frames: Optional[int] = None
    # tracking
    max_stale_frames: int = 600          # stop following a lost object after this many frames without a mask
    stale_prefix_frames: int = 90        # at a chunk boundary, re-run SAM 2 from a lost object's last good frame if that is <= N frames back
    jump_ratio: float = 2.0
    # detection
    detect_every: int = 1
    det_threshold: float = 0.25
    det_expand: float = 0.25             # box growth per side, fraction of box size
    det_pad_px: float = 12.0             # extra px (full-res units)
    det_fill_gap: int = 6
    det_hold: int = 3
    # mosaic
    block: int = 32                      # strength: block size (pixelate) or 2 x sigma (blur), full-res pixels
    mosaic_style: str = "pixelate"       # pixelate | blur | solid
    mask_pad_px: float = 24.0            # full-res pixels
    temporal_radius: int = 2
    # fail-closed
    fail_closed: bool = False
    fail_closed_mode: str = "bbox"       # bbox | full
    fail_closed_expand: float = 1.0
    fail_closed_max_frames: int = 180    # per loss episode
    # output
    encoder: str = "libx264"
    crf: int = 16
    preset: str = "medium"
    keep_audio: bool = True
    deep_verify: bool = False
    allow_vfr: bool = False
    allow_hdr: bool = False
    allow_points_mismatch: bool = False
    resume: bool = False
    extra: dict = field(default_factory=dict)   # recorded in the job file (e.g. detector settings)


NON_HASHED = {"resume", "deep_verify", "work_dir", "log_dir", "output", "input", "points"}
NON_HASHED_EXTRA = {"keep_temp"}


def config_hash(cfg: Config, info: VideoInfo) -> str:
    """Hash of everything that changes chunk results EXCEPT the clicked points (those are tracked per chunk, see
    prompt_signatures) and pure bookkeeping options. A mismatch makes a job non-resumable."""
    d = {k: v for k, v in asdict(cfg).items() if k not in NON_HASHED}
    d["extra"] = {k: v for k, v in (cfg.extra or {}).items() if k not in NON_HASHED_EXTRA}
    d["input_fp"] = info.fingerprint
    d["code"] = CODE_VERSION
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:24]


def file_sha(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prompt_signatures(prompts: Prompts, chunks) -> List[str]:
    """One signature per chunk over the clicks that fall into it.

    Tracking only runs forward and every chunk is re-seeded from the previous one, so the result of chunk c depends on the
    clicks in chunks 0..c and on nothing later. Changing a click therefore invalidates its own chunk and everything after it,
    and nothing before it."""
    sigs = []
    for s_, e_ in chunks:
        items = []
        for oid in sorted(prompts):
            for fr in sorted(prompts[oid]):
                if s_ <= fr < e_:
                    pts, labs = prompts[oid][fr]
                    items.append([oid, fr, [[round(x, 2), round(y, 2)] for x, y in pts], list(labs)])
        sigs.append(hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16])
    return sigs


def job_dir_for(cfg: Config, info: VideoInfo) -> str:
    stem = os.path.splitext(os.path.basename(cfg.input))[0]
    return os.path.join(cfg.work_dir, f"{stem}_{info.fingerprint[:10]}")      # per input; settings are checked via config_hash


def plan_reprocess(cfg: Config) -> dict:
    """What a '--resume' run would reuse, without touching anything (cheap: probe + points + job.json)."""
    info = probe_video(cfg.input)
    total = min(info.nb_frames, cfg.max_frames) if cfg.max_frames else info.nb_frames
    chunks = [(s, min(s + cfg.chunk_size, total)) for s in range(0, total, cfg.chunk_size)]
    prompts = load_points(cfg.points, info, cfg.allow_points_mismatch) if cfg.points else {}
    sigs = prompt_signatures(prompts, chunks)
    jd = job_dir_for(cfg, info)
    plan = {"total_chunks": len(chunks), "total_frames": total, "job_found": False, "can_resume": False, "reusable_chunks": 0,
            "frames_reused": 0, "first_changed_chunk": None, "reason": ""}
    jp = os.path.join(jd, "job.json")
    if not os.path.isfile(jp):
        plan["reason"] = "no kept intermediate chunks were found, so everything will be processed"
        return plan
    plan["job_found"] = True
    with open(jp, encoding="utf-8") as f:
        old = json.load(f)
    if old.get("config_hash") != config_hash(cfg, info):
        plan["reason"] = "the settings or the video differ from the earlier run, so everything will be processed from scratch"
        return plan
    plan["can_resume"] = True
    limit = first_changed_chunk(old.get("chunk_prompt_sigs"), sigs)
    plan["first_changed_chunk"] = limit if limit < len(chunks) else None
    good = 0
    for ci in range(min(limit, len(chunks))):
        c = old.get("chunks", {}).get(str(ci))
        if not (c and c.get("status") == "done" and all(os.path.isfile(os.path.join(jd, f"{k}_{ci:05d}.{x}"))
                                                         for k, x in (("seg", "mp4"), ("state", "json"), ("qc", "json")))):
            break
        good = ci + 1
    plan["reusable_chunks"] = good
    plan["frames_reused"] = chunks[good - 1][1] if good else 0
    plan["reason"] = ("nothing changed: only the final video is rebuilt" if good == len(chunks) else
                      f"chunks 1-{good} are reused, the rest is re-processed" if good else "no chunk can be reused")
    return plan


def first_changed_chunk(old_sigs: Optional[List[str]], new_sigs: List[str]) -> int:
    """Index of the first chunk whose clicks differ (len(new_sigs) when nothing changed)."""
    if not old_sigs:
        return 0
    for i, new in enumerate(new_sigs):
        if i >= len(old_sigs) or old_sigs[i] != new:
            return i
    return len(new_sigs)


def _atomic_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _mem_snapshot() -> dict:
    out = {}
    try:
        import psutil
        out["rss_mb"] = round(psutil.Process().memory_info().rss / 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda_alloc_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20)
            out["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    return out


class Pipeline:
    def __init__(self, cfg: Config, tracker: Optional[Tracker], detector=None, progress=None, cancel_check=None):
        """progress(dict) receives machine-readable progress events; cancel_check() -> bool is polled at safe points."""
        self.cfg, self.tracker, self.detector = cfg, tracker, detector
        self.stem = os.path.splitext(os.path.basename(cfg.input))[0]
        self.progress, self.cancel_check = progress, cancel_check
        self._ci = self._nchunks = self._done_frames = 0
        self.db: Optional[JobDB] = None
        self.run_id: Optional[str] = None

    def _db_call(self, name: str, *a, **k):
        """Registry writes are best effort: a broken registry must never stop or corrupt a job."""
        if self.db is None or self.run_id is None:
            return
        try:
            getattr(self.db, name)(self.run_id, *a, **k)
        except Exception:  # noqa: BLE001
            log.warning("job registry write failed (%s)", name, exc_info=True)

    # ------------------------------------------------------------------ progress / cancel
    def _emit(self, **ev):
        if self.progress is not None:
            try:
                self.progress(ev)
            except Exception:  # noqa: BLE001  (a broken listener must never break processing)
                log.exception("progress callback failed")

    def _check_cancel(self):
        if self.cancel_check is not None and self.cancel_check():
            raise JobCancelled("cancelled by the user")

    _STAGE = {"analyze": (0.0, 0.5), "track": (0.5, 0.1), "render": (0.6, 0.4)}

    def _tick(self, stage: str, i: int, n: int):
        self._check_cancel()
        base, span = self._STAGE[stage]
        self._emit(event="progress", stage=stage, chunk=self._ci, chunks=self._nchunks,
                   chunk_frac=round(base + span * i / max(1, n), 4), done_frames=self._done_frames,
                   chunk_frames=n, total_frames=self.total)

    # ------------------------------------------------------------------ setup
    def _check_input(self, info: VideoInfo):
        cfg = self.cfg
        if info.is_vfr and not cfg.allow_vfr:
            raise JobFailed(f"input looks like variable frame rate ({info.vfr_detail}). This tool re-encodes at a constant "
                            f"{info.fps_rational} fps, which would shift timing/audio sync. Convert the input to CFR "
                            f"yourself or pass --allow-vfr to accept linearised timing (frame count is preserved).")
        if is_hdr(info) and not cfg.allow_hdr:
            raise JobFailed(f"input is HDR/high-bit-depth (pix_fmt={info.pix_fmt}, trc={info.color_trc}); processing is 8-bit SDR "
                            f"and would change the look. Pass --allow-hdr to accept the conversion.")
        if not info.timestamps and cfg.resume:
            raise JobFailed("resume needs reliable packet timestamps for frame-exact seeking; this file has none")
        if info.nb_frames <= 0:
            raise JobFailed("could not determine frame count")
        if cfg.mosaic_style not in STYLES:
            raise JobFailed(f"unknown mosaic style {cfg.mosaic_style!r}; choose one of {STYLES}")

    def run(self) -> int:
        cfg = self.cfg
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.work_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(cfg.log_dir, f"{self.stem}.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        log.setLevel(logging.INFO)
        status_path = os.path.join(cfg.log_dir, f"{self.stem}_status.json")
        try:
            self.db = JobDB(db_path_for(cfg.log_dir))
            self.db.recover_interrupted()
        except Exception:  # noqa: BLE001
            self.db = None
            log.warning("job registry unavailable; continuing without it", exc_info=True)
        try:
            return self._run(status_path)
        except JobCancelled as e:
            self._db_call("finish_job", "cancelled", str(e))
            log.warning("JOB CANCELLED: %s", e)
            _atomic_json(status_path, {"status": "cancelled", "reason": str(e), "time": time.strftime("%F %T"),
                                       "note": "completed chunks are kept; run again with --resume to continue"})
            return 3
        except (JobFailed, VideoError) as e:
            self._db_call("finish_job", "failed", str(e)[:1000])
            log.error("JOB FAILED: %s", e)
            _atomic_json(status_path, {"status": "failed", "reason": str(e), "time": time.strftime("%F %T")})
            return 1
        except Exception as e:  # noqa: BLE001
            self._db_call("finish_job", "failed", f"{type(e).__name__}: {e}"[:1000])
            log.exception("JOB FAILED (unexpected)")
            _atomic_json(status_path, {"status": "failed", "reason": f"{type(e).__name__}: {e}", "time": time.strftime("%F %T")})
            return 1
        finally:
            log.removeHandler(fh)
            fh.close()

    # ------------------------------------------------------------------ main
    def _run(self, status_path: str) -> int:
        cfg = self.cfg
        t_start = time.time()
        info = probe_video(cfg.input)
        log.info("input %s %dx%d %s fps nb_frames=%d dur=%.2fs rot=%d vfr=%s (%s) audio=%s pix_fmt=%s",
                 info.path, info.width, info.height, info.fps_rational, info.nb_frames, info.duration, info.rotation,
                 info.is_vfr, info.vfr_detail, info.has_audio, info.pix_fmt)
        self.run_id = uuid.uuid4().hex[:12]
        self._db_call("start_job", input_path=info.path, input_fp=info.fingerprint,
                      total_frames=min(info.nb_frames, cfg.max_frames) if cfg.max_frames else info.nb_frames,
                      chunk_size=cfg.chunk_size, output_path=os.path.abspath(cfg.output) if cfg.output else None,
                      job_dir=None, resume=int(cfg.resume), model_sig=(cfg.extra or {}).get("sam2_ckpt_sig"))
        self._check_input(info)
        W, H = info.width, info.height
        total = min(info.nb_frames, cfg.max_frames) if cfg.max_frames else info.nb_frames
        sc = min(1.0, cfg.analysis_width / W)
        aw, ah = int(round(W * sc)), int(round(H * sc))
        self.aw, self.ah, self.sx, self.sy = aw, ah, aw / W, ah / H
        self.shape = (ah, aw)
        self.info, self.total = info, total

        prompts: Prompts = {}
        if cfg.points:
            prompts = load_points(cfg.points, info, cfg.allow_points_mismatch)
        if prompts and self.tracker is None:
            raise JobFailed("points were given but no tracker is available")
        if not prompts and self.detector is None:
            raise JobFailed("nothing to do: no points and NudeNet disabled")
        self.prompts = prompts
        self.obj_ids = sorted(prompts)

        cs = cfg.chunk_size
        chunks = [(s, min(s + cs, total)) for s in range(0, total, cs)]
        h = config_hash(cfg, info)
        self._points_changed, self._reused = False, 0
        self._sigs = prompt_signatures(prompts, chunks)
        job_dir = job_dir_for(cfg, info)
        self.job_dir = job_dir
        job_path = os.path.join(job_dir, "job.json")
        self.points_sha = file_sha(cfg.points)
        job = {"config_hash": h, "input": info.to_json_dict(), "config": asdict(cfg), "total_frames": total,
               "chunks": {}, "analysis_size": [aw, ah], "created": time.strftime("%F %T"),
               "chunk_prompt_sigs": self._sigs, "points_sha": self.points_sha, "run_id": self.run_id}
        start_ci = 0
        self.monitor = qc.ObjectMonitor(self.shape, qc.QCParams(jump_ratio=cfg.jump_ratio))
        self.objstate: Dict[int, dict] = {}
        self.last_good: Dict[int, np.ndarray] = {}
        self.det_carry: List[dict] = []
        self.det_records: List[dict] = []
        self.prev_last_analysis: Optional[np.ndarray] = None
        events: List[dict] = []
        ts_fn = info.timestamp_of

        if cfg.resume and os.path.isfile(job_path):
            start_ci = self._try_resume(job, job_path, chunks)
        else:
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir)
            os.makedirs(job_dir)
        if start_ci == 0 and not os.path.isdir(job_dir):
            os.makedirs(job_dir)
        _atomic_json(job_path, job)
        self._db_call("update_job", config_hash=h, points_sha=self.points_sha, n_chunks=len(chunks), reused_chunks=start_ci,
                      job_dir=os.path.abspath(job_dir), versions=self.tracker.describe() if self.tracker else None)
        for ci_done in range(start_ci):
            c = job["chunks"].get(str(ci_done), {})
            self._db_call("set_chunk", ci_done, "done", start_frame=c.get("start"), end_frame=c.get("end"), seconds=c.get("seconds"),
                          rss_mb=c.get("rss_mb"), cuda_peak_mb=c.get("cuda_peak_mb"))
        if start_ci:
            if start_ci < len(chunks):
                log.info("resuming at chunk %d/%d (frame %d)", start_ci, len(chunks), chunks[start_ci][0])
            else:
                log.info("all %d chunks are valid and reused; only the final video is rebuilt", len(chunks))
            for ci in range(start_ci):
                with open(os.path.join(job_dir, f"qc_{ci:05d}.json"), encoding="utf-8") as f:
                    events += json.load(f)["events"]
        self.events_so_far = events

        s0 = chunks[start_ci][0] if start_ci < len(chunks) else total
        self._nchunks, self._done_frames = len(chunks), s0
        self._emit(event="start", total_frames=total, chunks=len(chunks), chunk_size=cfg.chunk_size,
                   resume_from_chunk=start_ci, done_frames=s0, width=W, height=H, fps=info.fps,
                   reused_chunks=self._reused, points_changed=self._points_changed)
        self.ring = os.path.join(job_dir, "ring")
        shutil.rmtree(self.ring, ignore_errors=True)
        os.makedirs(self.ring)
        cap = cfg.stale_prefix_frames if self.tracker is not None else 0
        a_start = max(0, s0 - 1 - cap) if s0 > 0 else 0
        a_reader = f_reader = None
        if start_ci < len(chunks):
            a_reader = FrameReader(info, start_frame=a_start, max_frames=total - a_start, scale_to=(aw, ah))
            f_reader = FrameReader(info, start_frame=s0, max_frames=total - s0)
        try:
            if s0 > 0 and start_ci < len(chunks):
                for g in range(a_start, s0):      # rebuild the ring buffer; the last frame must match the checkpoint
                    fa = a_reader.read()
                    if fa is None:
                        raise JobFailed("resume: could not read analysis frames before the resume point")
                    if g < s0 - 1 and cap:
                        cv2.imwrite(os.path.join(self.ring, f"{g}.jpg"), fa, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    self.prev_last_analysis = fa
                want = job["chunks"][str(start_ci - 1)]["last_frame_md5"]
                if self.prev_last_analysis is None or hashlib.md5(self.prev_last_analysis.tobytes()).hexdigest() != want:
                    raise JobFailed("resume: seeked analysis frame does not match the checkpoint (seek not frame-exact). "
                                    "Re-run without --resume.")
            for ci in range(start_ci, len(chunks)):
                s, e = chunks[ci]
                t0 = time.time()
                self._ci, self._done_frames = ci, s
                self._check_cancel()
                self._db_call("set_chunk", ci, "running", start_frame=s, end_frame=e)
                try:
                    chunk_events = self._process_chunk(ci, s, e, a_reader, f_reader, prompts)
                except JobCancelled:
                    raise
                except BaseException as ex:
                    self._db_call("set_chunk", ci, "failed", start_frame=s, end_frame=e, error=f"{type(ex).__name__}: {ex}"[:500])
                    raise
                self._done_frames = e
                events += chunk_events
                mem = _mem_snapshot()
                job["chunks"][str(ci)] = {"status": "done", "start": s, "end": e, "frames": e - s,
                                          "segment": f"seg_{ci:05d}.mp4", "seconds": round(time.time() - t0, 1), **mem,
                                          "last_frame_md5": hashlib.md5(self.prev_last_analysis.tobytes()).hexdigest()}
                _atomic_json(job_path, job)
                self._db_call("set_chunk", ci, "done", start_frame=s, end_frame=e, seconds=round(time.time() - t0, 1),
                              rss_mb=mem.get("rss_mb"), cuda_peak_mb=mem.get("cuda_peak_mb"))
                self._emit(event="chunk_done", chunk=ci, chunks=len(chunks), start=s, end=e, done_frames=e, total_frames=total,
                           seconds=round(time.time() - t0, 2), events=len(chunk_events), **mem)
                el = time.time() - t_start
                done = e
                log.info("chunk %d/%d frames %d-%d done in %.1fs  mem=%s  eta=%.0fs", ci + 1, len(chunks), s, e - 1,
                         time.time() - t0, mem, el / max(1, done - (chunks[start_ci][0])) * (total - done) if done > chunks[start_ci][0] else 0)
        finally:
            for r in (a_reader, f_reader):
                if r is not None:
                    r.close()

        return self._finalize(info, chunks, job, events, status_path, t_start)

    # ------------------------------------------------------------------ resume
    def _try_resume(self, job: dict, job_path: str, chunks) -> int:
        with open(job_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config_hash") != job["config_hash"]:
            raise JobFailed("resume refused: settings or input differ from the earlier run (config hash mismatch). "
                            "Only the clicked points may change between a run and its resume. Re-run without --resume to start over.")
        job.update({k: old[k] for k in ("created",) if k in old})
        limit = first_changed_chunk(old.get("chunk_prompt_sigs"), job["chunk_prompt_sigs"])
        self._points_changed = limit < len(chunks)
        if self._points_changed:
            log.info("points changed: chunks 0-%d are reused, chunk %d onwards is re-processed", limit - 1, limit)
        ok = 0
        for ci in range(min(limit, len(chunks))):
            c = old["chunks"].get(str(ci))
            seg = os.path.join(self.job_dir, f"seg_{ci:05d}.mp4")
            st = os.path.join(self.job_dir, f"state_{ci:05d}.json")
            qcf = os.path.join(self.job_dir, f"qc_{ci:05d}.json")
            if not (c and c.get("status") == "done" and all(os.path.isfile(p) for p in (seg, st, qcf))):
                break
            try:
                if count_packets(seg) != chunks[ci][1] - chunks[ci][0]:
                    log.warning("chunk %d segment frame count mismatch; redoing from here", ci)
                    break
            except Exception as e:  # noqa: BLE001
                log.warning("chunk %d segment unreadable (%s); redoing from here", ci, e)
                break
            job["chunks"][str(ci)] = c
            ok = ci + 1
        self._reused = ok
        if ok:
            self._load_state(ok - 1)
        # drop anything after the last valid chunk (incomplete work is never treated as done)
        for name in os.listdir(self.job_dir):
            if name.startswith(("seg_", "state_", "qc_", "frames_")):
                try:
                    idx = int(name.split("_")[1].split(".")[0])
                except ValueError:
                    continue
                if idx >= ok:
                    p = os.path.join(self.job_dir, name)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        return ok

    def _save_state(self, ci: int, chunk_events: List[dict]):
        st = {"monitor": self.monitor.state_dict(), "objstate": {str(k): v for k, v in self.objstate.items()},
              "det_carry": self.det_carry, "det_records_tail": []}
        _atomic_json(os.path.join(self.job_dir, f"state_{ci:05d}.json"), st)
        arrs = {f"m{k}": v for k, v in self.last_good.items()}
        if self.prev_last_analysis is not None:
            pass
        np.savez(os.path.join(self.job_dir, f"state_{ci:05d}.npz"), **arrs)
        _atomic_json(os.path.join(self.job_dir, f"qc_{ci:05d}.json"),
                     {"events": chunk_events, "detections": self._chunk_dets})

    def _load_state(self, ci: int):
        with open(os.path.join(self.job_dir, f"state_{ci:05d}.json"), encoding="utf-8") as f:
            st = json.load(f)
        self.monitor.load_state(st["monitor"])
        self.objstate = {int(k): v for k, v in st["objstate"].items()}
        self.det_carry = st["det_carry"]
        z = np.load(os.path.join(self.job_dir, f"state_{ci:05d}.npz"))
        self.last_good = {int(k[1:]): z[k] for k in z.files}
        for i in range(ci + 1):
            with open(os.path.join(self.job_dir, f"qc_{i:05d}.json"), encoding="utf-8") as f:
                self.det_records += json.load(f).get("detections", [])

    # ------------------------------------------------------------------ chunk
    def _process_chunk(self, ci: int, s: int, e: int, a_reader: FrameReader, f_reader: FrameReader,
                       prompts: Prompts) -> List[dict]:
        cfg, info = self.cfg, self.info
        n = e - s
        qlog = qc.QCLog(info.timestamp_of)
        frame_dir = os.path.join(self.job_dir, f"frames_{ci:05d}")
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.makedirs(frame_dir)
        has_overlap = s > 0
        jpg_q = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # ---- 0. plan boundary re-seeding (needs the monitor state from the previous chunk)
        inits: List[Init] = []
        started_here: Dict[int, int] = {}      # obj -> local frame where tracking starts in this chunk
        seeds = []                              # (oid, kind, last_good_frame, mask)
        for oid in self.obj_ids:
            st = self.objstate.setdefault(oid, {"known": False, "retired": False})
            if has_overlap and st["known"] and not st["retired"]:
                seeds.append(self._plan_carry(oid, s, qlog))
        seeds = [x for x in seeds if x is not None]
        lost_seeds = [x for x in seeds if x[1] == "lost"]
        g0 = min([x[2] for x in lost_seeds] + [s - 1]) if has_overlap else s
        P = s - g0                              # prefix frames before the chunk (>=1 when has_overlap)
        L = P
        for oid, kind, lg, mask in seeds:
            if kind == "ok":
                inits.append(Init(oid, P - 1, mask=mask, source="carry"))
                started_here[oid] = P - 1
            elif kind == "lost":
                inits.append(Init(oid, lg - g0, mask=mask, source="carry_lost_rerun"))
                started_here[oid] = lg - g0
            else:                               # "unseeded": too old to re-run; stays flagged as lost
                started_here[oid] = 0

        # ---- 1. analysis frames -> disk (chunk dir + ring buffer), detection inline
        for j, g in enumerate(range(g0, s)):
            src = os.path.join(self.ring, f"{g}.jpg")
            if g == s - 1:
                cv2.imwrite(os.path.join(frame_dir, f"{j}.jpg"), self.prev_last_analysis, jpg_q)
            elif os.path.isfile(src):
                shutil.copyfile(src, os.path.join(frame_dir, f"{j}.jpg"))
            else:
                raise JobFailed(f"ring buffer is missing analysis frame {g} needed to re-run a lost object")
        raw_dets: List[Optional[List[Det]]] = [None] * n
        det_failed = [False] * n
        last = None
        use_ring = self.tracker is not None and cfg.stale_prefix_frames > 0
        for i in range(n):
            if i % 4 == 0:
                self._tick("analyze", i, n)
            fa = a_reader.read()
            if fa is None:
                qlog.add(s + i, qc.FRAME_ERROR, "error", None, "analysis reader ended early", action="job_aborted")
                raise JobFailed(f"analysis reader returned only {i} of {n} frames in chunk {ci} (frame {s + i}); "
                                f"decoded frame count differs from container packet count")
            ok_, buf = cv2.imencode(".jpg", fa, jpg_q)
            data = buf.tobytes()
            with open(os.path.join(frame_dir, f"{L + i}.jpg"), "wb") as fh:
                fh.write(data)
            if use_ring:
                with open(os.path.join(self.ring, f"{s + i}.jpg"), "wb") as fh:
                    fh.write(data)
            if self.detector is not None and (s + i) % cfg.detect_every == 0:
                try:
                    raw_dets[i] = self.detector.detect(fa)
                except Exception as ex:  # noqa: BLE001
                    det_failed[i] = True
                    qlog.add(s + i, DET_ERROR, "error", None, f"{type(ex).__name__}: {ex}",
                             action="fail_closed_full_frame" if cfg.fail_closed else "none")
            last = fa
        self.prev_last_analysis = last
        for name in os.listdir(self.ring):      # keep only the last stale_prefix_frames+1 frames
            if int(name.split(".")[0]) < e - 1 - cfg.stale_prefix_frames:
                os.remove(os.path.join(self.ring, name))

        # ---- 2. tracking
        for oid in self.obj_ids:
            st = self.objstate[oid]
            for k in sorted(prompts[oid]):
                if s <= k < e:
                    pts, labs = prompts[oid][k]
                    inits.append(Init(oid, L + k - s, [(x * self.sx, y * self.sy) for x, y in pts], list(labs), source="points"))
                    started_here.setdefault(oid, L + k - s)
                    st.update(known=True, retired=False)
        self._tick("track", 0, n)
        model_error: Optional[str] = None
        masks: Dict[int, Dict[int, np.ndarray]] = {}
        if inits:
            try:
                masks = self.tracker.track_chunk(frame_dir, L + n, inits)
            except TrackerError as ex:
                model_error = str(ex)
                log.error("tracker error in chunk %d: %s", ci, ex)
                for i in range(n):
                    qlog.add(s + i, qc.MODEL_ERROR, "error", None, model_error[:300],
                             action="fail_closed_full_frame" if cfg.fail_closed else "none_original_passthrough_for_tracked_objects")
        shutil.rmtree(frame_dir, ignore_errors=True)
        self._tick("track", n, n)

        # ---- 3. QC per frame (sequential: the monitor is stateful)
        fc_regions: List[List[tuple]] = [[] for _ in range(n)]
        fc_full = [False] * n
        fc_reason: List[Optional[str]] = [None] * n
        packed: Dict[int, List[Optional[np.ndarray]]] = {oid: [None] * n for oid in self.obj_ids}
        frame_has_obj = [False] * n
        shape = self.shape
        for i in range(n):
            f = s + i
            if det_failed[i] or model_error:
                fc_full[i] = True
                fc_reason[i] = DET_ERROR if det_failed[i] else qc.MODEL_ERROR
            ok_masks = []
            for oid in self.obj_ids:
                st = self.objstate[oid]
                if oid not in started_here and not (st["known"] and not st["retired"]):
                    continue
                if st["retired"]:
                    continue
                li = L + i
                if li < started_here.get(oid, 10 ** 9):
                    continue
                if model_error:
                    continue
                pm = masks.get(oid, {}).get(li)
                m = unpack_mask(pm, shape) if pm is not None else None
                v = self.monitor.observe(f, oid, m, qlog)
                if v.status == "not_started":
                    if li == started_here.get(oid) and f in self.prompts[oid]:
                        qlog.add(f, PROMPT_NO_MASK, "error", oid, "SAM 2 returned an empty mask for the clicked prompt",
                                 action="flagged_no_region_known")
                    continue
                frame_has_obj[i] = True
                packed[oid][i] = pm
                if v.status == "ok" and m is not None:
                    ok_masks.append(m)
                    self.last_good[oid] = pm
                    self.objstate[oid]["last_good_frame"] = f
                if v.uncertain and cfg.fail_closed and v.region is not None:
                    lost_n = self.monitor.frames_lost(f, oid)
                    if v.uncertain == qc.AREA_JUMP or lost_n <= cfg.fail_closed_max_frames:
                        fc_regions[i].append(v.region)
                        fc_reason[i] = fc_reason[i] or v.uncertain
                if v.status == "lost" and self.monitor.frames_lost(f, oid) >= cfg.max_stale_frames:
                    self.monitor.mark_abandoned(f, oid, qlog, f"mask absent for {cfg.max_stale_frames} frames; no longer re-seeding")
                    self.objstate[oid]["retired"] = True
            if frame_has_obj[i] and raw_dets[i]:
                boxes = [((int(d.x1), int(d.y1), int(np.ceil(d.x2)), int(np.ceil(d.y2))), d.score, d.cls) for d in raw_dets[i]]
                qc.detection_mask_mismatch(boxes, ok_masks, f, qlog)

        # ---- detection gap filling + records
        filled, self.det_carry = fill_detections(raw_dets, s, self.det_carry, cfg.det_fill_gap, cfg.det_hold)
        self._chunk_dets = []
        for i, ds in enumerate(filled):
            for d in ds:
                self._chunk_dets.append({"frame": s + i, "timestamp": round(info.timestamp_of(s + i), 6), "class": d.cls,
                                         "score": round(d.score, 4), "box": [round(v, 1) for v in d.box], "filled": d.filled})
        self.det_records += self._chunk_dets

        # ---- 4. render at full resolution
        seg_path = os.path.join(self.job_dir, f"seg_{ci:05d}.mp4")
        writer = SegmentWriter(seg_path, info, cfg.encoder, cfg.crf, cfg.preset)
        pad_an = int(round(cfg.mask_pad_px * self.sx))
        det_pad_an = cfg.det_pad_px * self.sx
        fc_run: Dict[str, int] = {}
        try:
            for i in range(n):
                if i % 4 == 0:
                    self._tick("render", i, n)
                full = f_reader.read()
                if full is None:
                    qlog.add(s + i, qc.FRAME_ERROR, "error", None, "full-resolution reader ended early", action="job_aborted")
                    raise JobFailed(f"full-resolution reader returned only {i} of {n} frames in chunk {ci}")
                if fc_full[i]:
                    m_an = np.ones(shape, dtype=bool)
                else:
                    seqs = []
                    for oid in self.obj_ids:
                        if any(p is not None for p in packed[oid]):
                            lo, hi = max(0, i - cfg.temporal_radius), min(n, i + cfg.temporal_radius + 1)
                            for j in range(lo, hi):
                                if packed[oid][j] is not None:
                                    seqs.append(unpack_mask(packed[oid][j], shape))
                    m_an = np.zeros(shape, dtype=bool)
                    for m in seqs:
                        m_an |= m
                    m_an = dilate_mask(m_an, pad_an)
                    dboxes = [expand_box(d.box, cfg.det_expand, det_pad_an, shape) for d in filled[i]]
                    if dboxes:
                        m_an |= boxes_to_mask(dboxes, shape)
                    if fc_regions[i]:
                        fb = [expand_box(r, cfg.fail_closed_expand, 0, shape) for r in fc_regions[i]]
                        if cfg.fail_closed_mode == "full":
                            m_an[:] = True
                        else:
                            m_an |= boxes_to_mask(fb, shape)
                if (fc_full[i] or fc_regions[i]) and cfg.fail_closed:
                    qlog.add(s + i, qc.FAIL_CLOSED_APPLIED, "info", None,
                             f"reason={fc_reason[i]} mode={'full' if fc_full[i] else cfg.fail_closed_mode}",
                             action="full_frame_mosaic" if (fc_full[i] or cfg.fail_closed_mode == "full") else "expanded_bbox_mosaic")
                m_full = upscale_mask(m_an, (info.height, info.width))
                apply_mosaic(full, m_full, cfg.mosaic_style, cfg.block)
                writer.write(full)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        got = count_packets(seg_path)
        if got != n:
            qlog.add(s, qc.COUNT_MISMATCH, "error", None, f"chunk {ci}: wrote {n} frames, segment has {got}", action="job_aborted")
            raise JobFailed(f"segment {ci} has {got} frames, expected {n}")

        chunk_events = [asdict(ev) for ev in qlog.events]
        self._save_state(ci, chunk_events)
        del packed, masks, raw_dets, filled
        gc.collect()
        return chunk_events

    def _plan_carry(self, oid: int, s: int, qlog: qc.QCLog):
        """Decide how to re-seed an object at the chunk boundary (see tracking.py docstring).
        Returns (oid, kind, last_good_frame, mask) or None. kind: ok | lost | unseeded."""
        cfg = self.cfg
        mon = self.monitor.state.get(oid, {})
        gm = self.last_good.get(oid)
        lgf = self.objstate[oid].get("last_good_frame")
        if gm is None or lgf is None:
            qlog.add(s, qc.CHUNK_RESUME_FAILED, "error", oid, "no mask available to re-seed tracking at chunk boundary",
                     action="object_not_tracked_in_chunk")
            return None
        age = (s - 1) - lgf
        mask = unpack_mask(gm, self.shape)
        if mon.get("lost_since") is None and age == 0:
            qlog.add(s, qc.CHUNK_RESUME_OK, "info", oid, "re-seeded from last mask of previous chunk", action="resumed")
            return (oid, "ok", lgf, mask)
        if age <= cfg.stale_prefix_frames:
            qlog.add(s, qc.CHUNK_RESUME_STALE, "warn", oid,
                     f"object had no mask at the boundary; SAM 2 re-run from its last good frame {lgf} ({age} frames back) "
                     f"through the hidden frames - it may or may not find the object again",
                     action="rerun_from_last_good_frame")
            return (oid, "lost", lgf, mask)
        qlog.add(s, qc.CHUNK_RESUME_FAILED, "error", oid,
                 f"object lost for {age} frames at the boundary (> --stale-prefix-frames {cfg.stale_prefix_frames}); not re-seeded. "
                 f"It stays flagged as lost; re-click it on a later frame with pick_points.py",
                 action="object_not_tracked_in_chunk")
        return (oid, "unseeded", lgf, mask)

    # ------------------------------------------------------------------ finish
    def _finalize(self, info: VideoInfo, chunks, job, events: List[dict], status_path: str, t_start: float) -> int:
        cfg = self.cfg
        qlog = qc.QCLog.from_events(events, info.timestamp_of)
        total = self.total
        segs = [os.path.join(self.job_dir, f"seg_{ci:05d}.mp4") for ci in range(len(chunks))]
        out_final = cfg.output or os.path.join("output", f"{self.stem}_censored.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out_final)), exist_ok=True)
        partial = out_final + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        concat_and_mux(segs, info, partial, os.path.join(self.job_dir, "concat.txt"), cfg.keep_audio)
        problems = verify_output(partial, info, total, expect_audio=cfg.keep_audio and info.has_audio, deep=cfg.deep_verify,
                                 audio_tol=0.25 if not cfg.max_frames else max(0.5, 2.0))
        if cfg.max_frames and info.has_audio:
            problems = [p for p in problems if "audio" not in p]   # audio is not trimmed in --max-frames test runs
        for p in problems:
            qlog.add(-1, qc.OUTPUT_INVALID, "error", None, p, action="output_not_finalised")
        meta = {"input": info.to_json_dict(), "total_frames": total, "chunks": len(chunks), "chunk_size": cfg.chunk_size,
                "fail_closed": cfg.fail_closed, "tracker": self.tracker.describe() if self.tracker else None,
                "config": asdict(cfg), "run_id": self.run_id, "points_sha": self.points_sha,
                "points_path": os.path.abspath(cfg.points) if cfg.points else None,
                "reused_chunks": self._reused, "reprocessed_chunks": list(range(self._reused, len(chunks))),
                "created": time.strftime("%F %T"),
                "output": None if problems else {"path": os.path.abspath(out_final), "size": os.path.getsize(partial),
                                                 "mtime": round(os.path.getmtime(partial), 3)}}
        review_json = os.path.join(cfg.log_dir, f"{self.stem}_review.json")
        qlog.save(review_json, os.path.join(cfg.log_dir, f"{self.stem}_review.csv"), meta)
        with open(os.path.join(cfg.log_dir, f"{self.stem}_detections.csv"), "w", encoding="utf-8-sig", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["frame", "timestamp", "class", "score", "x1", "y1", "x2", "y2", "filled_by_temporal_hold"])
            for d in self.det_records:
                w.writerow([d["frame"], d["timestamp"], d["class"], d["score"], *d["box"], d["filled"]])
        if problems:
            _atomic_json(status_path, {"status": "failed", "reason": "output verification failed", "problems": problems,
                                       "partial_file": partial, "time": time.strftime("%F %T")})
            log.error("output verification FAILED: %s (kept as %s)", problems, partial)
            self._db_call("finish_job", "failed", "output verification failed: " + "; ".join(problems)[:900],
                          review_log=os.path.abspath(review_json))
            return 1
        os.replace(partial, out_final)
        errors = sum(1 for e in qlog.events if e.severity == "error")
        warns = sum(1 for e in qlog.events if e.severity == "warn")
        status = "completed_with_review_required" if (errors or warns) else "completed_no_flags"
        _atomic_json(status_path, {"status": status, "output": os.path.abspath(out_final), "review_log": os.path.abspath(review_json),
                                   "error_events": errors, "warn_events": warns, "frames": total,
                                   "seconds": round(time.time() - t_start, 1), "summary": qlog.summary(),
                                   "note": "no flags does not prove nothing was missed; spot-check the output"})
        log.info("DONE %s status=%s errors=%d warns=%d", out_final, status, errors, warns)
        self._db_call("finish_job", status, None, output_path=os.path.abspath(out_final), review_log=os.path.abspath(review_json))
        shutil.rmtree(self.job_dir, ignore_errors=True) if not cfg.extra.get("keep_temp") else None
        return 2 if errors else 0
