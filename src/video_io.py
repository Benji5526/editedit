"""FFmpeg/FFprobe based video I/O: probing, sequential frame reading, segment writing, muxing, verification.

All frame access goes through FFmpeg (never loads a whole video). Frames are decoded with
``-fps_mode passthrough`` so frame index i always means "i-th decoded frame", for both the
analysis (downscaled) reader and the full-resolution reader.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class VideoError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VideoError(f"'{name}' not found on PATH. Install FFmpeg (includes ffprobe) and retry.")
    return path


def _run(cmd: List[str], timeout: Optional[float] = None) -> str:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW)
    if p.returncode != 0:
        raise VideoError(f"command failed ({p.returncode}): {' '.join(cmd[:4])} ...\n{p.stderr.decode(errors='replace')[-2000:]}")
    return p.stdout.decode(errors="replace")


@dataclass
class VideoInfo:
    path: str
    width: int                # display size (after rotation, = size of decoded frames)
    height: int
    fps_rational: str         # e.g. "60/1"
    fps: float
    nb_frames: int            # packet count (fast); verified against the decode count while processing
    duration: float
    rotation: int
    pix_fmt: str
    codec: str
    colorspace: str
    color_trc: str
    is_vfr: bool
    vfr_detail: str
    has_audio: bool
    audio_codec: str = ""
    audio_duration: float = 0.0
    video_start_time: float = 0.0
    audio_start_time: float = 0.0
    file_size: int = 0
    fingerprint: str = ""
    timestamps: Optional[List[float]] = field(default=None, repr=False)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d.pop("timestamps", None)
        return d

    def timestamp_of(self, frame_idx: int) -> float:
        """Presentation time of a frame from real packet timestamps (falls back to idx/fps)."""
        if self.timestamps and 0 <= frame_idx < len(self.timestamps):
            return float(self.timestamps[frame_idx])
        return frame_idx / self.fps


def file_fingerprint(path: str) -> str:
    """Cheap identity: size + sha256 of first and last 1 MiB."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:32]


def _decoded_size(path: str) -> Tuple[int, int]:
    """Size of an actually decoded (auto-rotated) frame; avoids guessing from rotation metadata."""
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-c:v", "bmp", "-"]
    p = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if p.returncode != 0 or not p.stdout:
        raise VideoError("could not decode first frame: " + p.stderr.decode(errors="replace")[-500:])
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise VideoError("could not decode first frame image")
    return img.shape[1], img.shape[0]


def read_packet_timestamps(path: str) -> Optional[List[float]]:
    out = _run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    vals = []
    for line in out.splitlines():
        line = line.strip().strip(",")
        if not line or line == "N/A":
            return None  # missing pts -> cannot trust timestamps
        vals.append(float(line))
    return sorted(vals)


def classify_vfr(ts: Optional[List[float]], r_fps: float, avg_fps: float) -> Tuple[bool, str]:
    """Heuristic VFR check. Per-delta deviation >10% from the median counts as irregular."""
    reasons = []
    if r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) / r_fps > 0.01:
        reasons.append(f"r_frame_rate={r_fps:.4f} differs from avg_frame_rate={avg_fps:.4f}")
    if ts is None:
        reasons.append("packet pts unavailable")
    elif len(ts) > 2:
        d = np.diff(np.asarray(ts))
        med = float(np.median(d))
        if med <= 0:
            reasons.append("non-increasing timestamps")
        else:
            bad = np.abs(d - med) > 0.10 * med
            if bad.mean() > 0.001:
                reasons.append(f"{bad.sum()} of {len(d)} frame intervals deviate >10% from median {med*1000:.3f} ms")
    return (bool(reasons), "; ".join(reasons) if reasons else "constant frame rate")


def _frac(s: str) -> Fraction:
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def probe_video(path: str, with_timestamps: bool = True) -> VideoInfo:
    if not os.path.isfile(path):
        raise VideoError(f"input not found: {path}")
    raw = _run([find_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    meta = json.loads(raw)
    vs = [s for s in meta["streams"] if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not vs:
        raise VideoError("no video stream")
    v = vs[0]
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if "rotate" in (v.get("tags") or {}):
        rotation = int(float(v["tags"]["rotate"]))
    rotation %= 360

    ts = read_packet_timestamps(path) if with_timestamps else None
    r_fps = float(_frac(v.get("r_frame_rate", "0/1")))
    avg_fps = float(_frac(v.get("avg_frame_rate", "0/1")))
    is_vfr, vfr_detail = classify_vfr(ts, r_fps, avg_fps)
    nominal = _frac(v.get("avg_frame_rate", "0/1")) if not is_vfr or r_fps == 0 else _frac(v.get("r_frame_rate"))
    if nominal == 0:
        nominal = _frac(v.get("r_frame_rate", "30/1")) or Fraction(30)
    nb = len(ts) if ts else int(v.get("nb_frames") or 0)
    duration = float(v.get("duration") or meta.get("format", {}).get("duration") or 0)

    aud = [s for s in meta["streams"] if s.get("codec_type") == "audio"]
    w, h = _decoded_size(path)
    return VideoInfo(
        path=os.path.abspath(path), width=w, height=h,
        fps_rational=f"{nominal.numerator}/{nominal.denominator}", fps=float(nominal),
        nb_frames=nb, duration=duration, rotation=rotation,
        pix_fmt=v.get("pix_fmt", ""), codec=v.get("codec_name", ""),
        colorspace=v.get("color_space", "") or "", color_trc=v.get("color_transfer", "") or "",
        is_vfr=is_vfr, vfr_detail=vfr_detail, has_audio=bool(aud),
        audio_codec=aud[0].get("codec_name", "") if aud else "",
        audio_duration=float(aud[0].get("duration") or 0) if aud else 0.0,
        video_start_time=float(v.get("start_time") or 0), audio_start_time=float(aud[0].get("start_time") or 0) if aud else 0.0,
        file_size=os.path.getsize(path), fingerprint=file_fingerprint(path), timestamps=ts,
    )


def color_tags(info: VideoInfo) -> Tuple[str, str, str]:
    """(swscale matrix name, ffmpeg -colorspace tag, trc tag) for the 8-bit SDR output."""
    cs = (info.colorspace or "").lower()
    if cs in ("bt470bg", "smpte170m"):
        return "bt601", cs, "smpte170m"
    if cs in ("bt2020nc", "bt2020c"):
        return "bt2020", cs, info.color_trc or "bt709"
    return "bt709", "bt709", "bt709"  # default / unspecified / bt709


def is_hdr(info: VideoInfo) -> bool:
    return info.color_trc in ("smpte2084", "arib-std-b67") or "10" in info.pix_fmt or "12" in info.pix_fmt


class FrameReader:
    """Sequential BGR24 reader. scale_to=(w,h) lets FFmpeg downscale (analysis frames)."""

    def __init__(self, info: VideoInfo, start_frame: int = 0, max_frames: Optional[int] = None,
                 scale_to: Optional[Tuple[int, int]] = None, hwaccel: bool = False):
        self.info = info
        self.out_w, self.out_h = scale_to or (info.width, info.height)
        self.frame_bytes = self.out_w * self.out_h * 3
        self.start_frame = start_frame
        cmd = [find_tool("ffmpeg"), "-v", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", "auto"]
        if start_frame > 0:
            # Seek just before the target frame's real timestamp; the first decoded frame with
            # pts >= ss is the target. Needs reliable timestamps, so callers must check.
            t0 = info.timestamp_of(start_frame)
            t_prev = info.timestamp_of(start_frame - 1)
            cmd += ["-ss", f"{(t0 + t_prev) / 2:.6f}"]
        cmd += ["-i", info.path, "-map", "0:v:0", "-an", "-sn", "-fps_mode", "passthrough"]
        if max_frames is not None:
            cmd += ["-frames:v", str(max_frames)]
        if scale_to:
            cmd += ["-vf", f"scale={self.out_w}:{self.out_h}:flags=area"]
        cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        # bufsize=0: unbuffered pipe. Reading a 25 MB 4K frame through a BufferedReader and copying it again cost ~64 ms per frame;
        # reading straight into a freshly allocated array takes ~18 ms (measured, see tools/ and the README).
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, creationflags=CREATE_NO_WINDOW)
        self.frames_read = 0

    def read(self) -> Optional[np.ndarray]:
        """Next frame as a new, writable BGR array (the caller owns it), or None at the end of the stream."""
        arr = np.empty((self.out_h, self.out_w, 3), np.uint8)
        mv = memoryview(arr).cast("B")
        got, total = 0, self.frame_bytes
        while got < total:
            k = self._p.stdout.readinto(mv[got:])
            if not k:
                break
            got += k
        if got == 0:
            return None
        if got < total:
            raise VideoError(f"truncated frame after {self.frames_read} frames: {self._stderr()}")
        self.frames_read += 1
        return arr

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def _stderr(self) -> str:
        try:
            self._p.wait(timeout=2)
            return self._p.stderr.read().decode(errors="replace")[-800:]
        except Exception:
            return ""

    def close(self):
        try:
            if self._p.stdout:
                self._p.stdout.close()
            if self._p.poll() is None:
                self._p.kill()
            self._p.wait(timeout=5)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class PrefetchReader:
    """Wraps a FrameReader and reads ahead on a background thread (a few frames), so decoding/piping the next 4K frame overlaps with
    the caller's processing of the current one. Same interface as FrameReader.read(); frame order and error behaviour are unchanged
    (a read error is raised from read() at the position where it happened). Memory is bounded by ``depth`` frames."""

    _END = object()

    def __init__(self, reader: "FrameReader", depth: int = 3):
        import queue
        import threading
        self._reader, self._q, self._stop = reader, queue.Queue(maxsize=max(1, depth)), threading.Event()
        self._done = False
        self.out_w, self.out_h = reader.out_w, reader.out_h
        self._t = threading.Thread(target=self._run, name="PrefetchReader", daemon=True)
        self._t.start()

    def _put(self, item) -> bool:
        import queue
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            while not self._stop.is_set():
                f = self._reader.read()
                if f is None:
                    self._put(self._END)
                    return
                if not self._put(f):
                    return
        except BaseException as e:  # noqa: BLE001  (delivered to the consumer in order)
            self._put(e)

    @property
    def frames_read(self) -> int:
        return self._reader.frames_read

    def read(self) -> Optional[np.ndarray]:
        if self._done:
            return None
        item = self._q.get()
        if item is self._END:
            self._done = True
            return None
        if isinstance(item, BaseException):
            self._done = True
            raise item
        return item

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def close(self):
        self._stop.set()
        self._reader.close()            # also unblocks a read that is waiting on the pipe
        self._t.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_frame_at(info: VideoInfo, frame_idx: int, scale_to=None) -> np.ndarray:
    with FrameReader(info, start_frame=frame_idx, max_frames=1, scale_to=scale_to) as r:
        f = r.read()
    if f is None:
        raise VideoError(f"could not read frame {frame_idx}")
    return f


def nvenc_available() -> bool:
    try:
        out = _run([find_tool("ffmpeg"), "-hide_banner", "-encoders"])
        return "h264_nvenc" in out
    except Exception:
        return False


def frame_exact_timescale(fps_rational: str) -> int:
    """Container timescale in which one frame lasts a whole number of ticks.

    Segments are concatenated by stream copy, so their durations add up in container ticks. With the usual 90000 a 59.94 fps frame
    lasts 1501.5 ticks, every segment with an odd frame count is rounded by half a tick, and the error grows with the number of
    chunks (about 7 ms after 1200 odd-sized chunks, nearly half a frame). Using a multiple of the fps numerator (>= 10000 for
    player compatibility) makes every frame an exact number of ticks, so no rounding happens at all."""
    n, d = (int(x) for x in fps_rational.split("/"))
    k = max(1, -(-10000 // n))
    return n * k


class SegmentWriter:
    """Encodes BGR24 frames to a video-only mp4 segment (CFR at the nominal fps, no audio)."""

    def __init__(self, path: str, info: VideoInfo, encoder: str = "libx264", crf: int = 16, preset: str = "medium"):
        self.path = path
        matrix, cs_tag, trc_tag = color_tags(info)
        vf = f"scale=out_color_matrix={matrix}:out_range=tv,format=yuv420p"
        cmd = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
               "-r", info.fps_rational, "-i", "-", "-an", "-vf", vf]
        if encoder == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-cq", str(crf), "-preset", "p5", "-rc", "vbr", "-b:v", "0"]
        else:
            cmd += ["-c:v", encoder, "-crf", str(crf), "-preset", preset]
        cmd += ["-colorspace", cs_tag, "-color_primaries", "bt709" if cs_tag == "bt709" else cs_tag,
                "-color_trc", trc_tag, "-pix_fmt", "yuv420p", "-video_track_timescale", str(frame_exact_timescale(info.fps_rational)), path]
        self._cmd = cmd
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
        self.frames_written = 0
        self._shape = (info.height, info.width, 3)

    def write(self, frame: np.ndarray):
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise VideoError(f"frame shape/dtype mismatch: {frame.shape} {frame.dtype} != {self._shape}")
        try:
            self._p.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))      # no extra 25 MB copy
        except (BrokenPipeError, OSError) as e:
            raise VideoError(f"encoder pipe broke at frame {self.frames_written}: {self._p.stderr.read().decode(errors='replace')[-800:]}") from e
        self.frames_written += 1

    def close(self):
        try:
            self._p.stdin.close()
        except Exception:
            pass
        err = self._p.stderr.read().decode(errors="replace")
        rc = self._p.wait()
        if rc != 0:
            raise VideoError(f"encoder failed ({rc}): {err[-1000:]}")

    def abort(self):
        try:
            self._p.kill()
            self._p.wait(timeout=5)
        except Exception:
            pass


def count_packets(path: str) -> int:
    ts = read_packet_timestamps(path)
    if ts is not None:
        return len(ts)
    out = _run([find_tool("ffprobe"), "-v", "error", "-count_packets", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    return int(out.strip().strip(",") or 0)


def audio_md5(path: str) -> Optional[str]:
    """MD5 over the first audio stream's packets, read without decoding (stream copy). Equal hashes mean the audio data is bit-identical."""
    out = _run([find_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"])
    line = out.strip().splitlines()[-1] if out.strip() else ""
    return line.split("=", 1)[1] if line.startswith("MD5=") else None


def concat_and_mux(segments: List[str], source: Optional[VideoInfo], out_path: str, list_path: str,
                   copy_audio: bool = True) -> str:
    """Concatenate encoded segments (stream copy) and attach the original audio untouched.

    Returns "copy" (audio packets copied bit-for-bit), "aac" (the source audio codec cannot be stored in MP4, so it was re-encoded to AAC)
    or "none" (no audio attached)."""
    with open(list_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write("file '" + os.path.abspath(s).replace("\\", "/").replace("'", "'\\''") + "'\n")
    base = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    tail = ["-movflags", "+faststart", "-f", "mp4", out_path]
    if not (copy_audio and source is not None and source.has_audio):
        _run(base + ["-map", "0:v:0", "-c", "copy"] + tail)
        return "none"
    try:
        _run(base + ["-i", source.path, "-map", "0:v:0", "-map", "1:a", "-c", "copy"] + tail)
        return "copy"
    except VideoError:
        # most often: the source audio codec has no MP4 mapping. Keep the audio, change only its codec.
        _run(base + ["-i", source.path, "-map", "0:v:0", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k"] + tail)
        return "aac"


def verify_output(path: str, src: VideoInfo, expected_frames: int, expect_audio: bool, deep: bool = False,
                  audio_tol: float = 0.25, audio_identical: bool = False) -> List[str]:
    """Return list of problems (empty = verified). Checks frames, size, duration, CFR timestamps, audio."""
    problems: List[str] = []
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return ["output file missing or empty"]
    try:
        out = probe_video(path)
    except Exception as e:  # noqa: BLE001
        return [f"output cannot be probed: {e}"]
    if (out.width, out.height) != (src.width, src.height):
        problems.append(f"resolution {out.width}x{out.height} != {src.width}x{src.height}")
    if out.nb_frames != expected_frames:
        problems.append(f"packet/frame count {out.nb_frames} != expected {expected_frames}")
    if deep:
        n = 0
        with FrameReader(out, scale_to=(64, 36)) as r:
            for _ in r:
                n += 1
        if n != expected_frames:
            problems.append(f"decoded frame count {n} != expected {expected_frames}")
    if out.timestamps and len(out.timestamps) > 1:
        dt = 1.0 / src.fps
        dev = np.abs((np.asarray(out.timestamps) - out.timestamps[0]) - np.arange(len(out.timestamps)) * dt)
        if dev.max() > 0.5 * dt:
            problems.append(f"output timestamps deviate from CFR grid by up to {dev.max()*1000:.1f} ms")
    exp_dur = expected_frames / src.fps
    if abs(out.duration - exp_dur) > 2.0 / src.fps + 0.05 and out.duration > 0:
        problems.append(f"video duration {out.duration:.3f}s vs expected {exp_dur:.3f}s")
    if expect_audio:
        if not out.has_audio:
            problems.append("audio stream missing in output")
        else:
            a_ref = src.audio_duration or src.duration
            if a_ref and abs(out.audio_duration - a_ref) > audio_tol:
                problems.append(f"audio duration {out.audio_duration:.3f}s vs source {a_ref:.3f}s")
            if out.duration and abs(out.audio_duration - out.duration) > max(audio_tol, abs((src.audio_duration or 0) - src.duration) + audio_tol):
                problems.append(f"audio/video duration differ by {abs(out.audio_duration - out.duration):.3f}s (A/V sync risk)")
            if audio_identical:
                try:
                    if audio_md5(src.path) != audio_md5(path):
                        problems.append("audio stream differs from the source (expected bit-identical stream copy)")
                except VideoError as e:
                    problems.append(f"could not compare the audio streams: {e}")
    return problems
