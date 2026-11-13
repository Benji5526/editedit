"""Cut a finished video: remove frame ranges (or trim the ends) and export the rest as a new file.

This works on any video file and never touches the tracking data of a job, which is why it is meant for the RESULT of the mosaic
processing: cutting the source first would shift every frame number the clicked points and the mask cache refer to.

The cut is frame-accurate: the kept ranges are selected by frame number (ffmpeg ``trim=start_frame/end_frame``), the matching audio
by time (``atrim``), and the pieces are joined with ``concat``. Video and audio are re-encoded (H.264 + AAC); the picture is not
bit-identical to the input. The result is written to ``<out>.partial``, its frame count is checked, and only then renamed.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

from .video_io import CREATE_NO_WINDOW, VideoError, VideoInfo, color_tags, count_packets, find_tool, frame_exact_timescale, probe_video

Range = Tuple[int, int]          # [start, end) in frames
MAX_PIECES = 200                 # keeps the ffmpeg command line well below the Windows limit


class CutError(ValueError):
    pass


class CutCancelled(RuntimeError):
    pass


def normalize_ranges(ranges: Sequence[Range], total: int) -> List[Range]:
    """Clip to the video, drop empty ranges, sort and merge overlapping or touching ones."""
    out: List[Range] = []
    for s, e in sorted((max(0, int(s)), min(int(total), int(e))) for s, e in ranges):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def keep_ranges(total: int, remove: Sequence[Range]) -> List[Range]:
    """The frame ranges that stay when ``remove`` is cut out of a video of ``total`` frames."""
    keep, pos = [], 0
    for s, e in normalize_ranges(remove, total):
        if s > pos:
            keep.append((pos, s))
        pos = e
    if pos < total:
        keep.append((pos, total))
    return keep


def frames_kept(total: int, remove: Sequence[Range]) -> int:
    return sum(e - s for s, e in keep_ranges(total, remove))


def build_filter(info: VideoInfo, keep: Sequence[Range], with_audio: bool) -> Tuple[str, str, Optional[str]]:
    """ffmpeg filter graph for the kept ranges. Returns (graph, video label, audio label or None)."""
    def t(frame: int) -> float:
        return info.timestamp_of(frame) if frame < info.nb_frames else info.duration + 1.0

    parts, labels = [], []
    for i, (s, e) in enumerate(keep):
        parts.append(f"[0:v]trim=start_frame={s}:end_frame={e},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
        if with_audio:
            parts.append(f"[0:a]atrim=start={t(s):.6f}:end={t(e):.6f},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    if len(keep) == 1:
        return ";".join(parts), "[v0]", ("[a0]" if with_audio else None)
    parts.append("".join(labels) + f"concat=n={len(keep)}:v=1:a={1 if with_audio else 0}[v]" + ("[a]" if with_audio else ""))
    return ";".join(parts), "[v]", ("[a]" if with_audio else None)


def export_cut(src_path: str, remove: Sequence[Range], out_path: str, encoder: str = "libx264", crf: int = 16, preset: str = "medium",
               progress: Optional[Callable[[int, int], None]] = None, cancel_check: Optional[Callable[[], bool]] = None,
               info: Optional[VideoInfo] = None) -> dict:
    """Write ``src_path`` without the frame ranges in ``remove`` to ``out_path``.
    progress(done_frames, total_frames) is called while encoding; cancel_check() -> True stops and removes the partial file.
    Returns {"frames": kept, "removed": removed, "duration": seconds, "pieces": n, "audio": bool}."""
    info = info or probe_video(src_path)
    if os.path.abspath(out_path) == os.path.abspath(src_path):
        raise CutError("the output must not be the input file")
    keep = keep_ranges(info.nb_frames, remove)
    kept = sum(e - s for s, e in keep)
    if kept <= 0:
        raise CutError("nothing would be left: the removed ranges cover the whole video")
    if kept == info.nb_frames:
        raise CutError("nothing to cut: no frame range is removed")
    if len(keep) > MAX_PIECES:
        raise CutError(f"too many pieces ({len(keep)}); at most {MAX_PIECES} are supported in one export")
    graph, vlabel, alabel = build_filter(info, keep, info.has_audio)
    matrix, cs_tag, trc_tag = color_tags(info)
    partial = out_path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", src_path, "-filter_complex", graph, "-map", vlabel]
    if alabel:
        cmd += ["-map", alabel, "-c:a", "aac", "-b:a", "256k"]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-cq", str(crf), "-preset", "p5", "-rc", "vbr", "-b:v", "0"]
    else:
        cmd += ["-c:v", encoder, "-crf", str(crf), "-preset", preset]
    cmd += ["-pix_fmt", "yuv420p", "-colorspace", cs_tag, "-color_primaries", "bt709" if cs_tag == "bt709" else cs_tag, "-color_trc", trc_tag,
            "-fps_mode", "passthrough", "-video_track_timescale", str(frame_exact_timescale(info.fps_rational)),
            "-progress", "pipe:1", "-nostats", "-f", "mp4", partial]
    err_file = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, creationflags=CREATE_NO_WINDOW)
    try:
        for raw in p.stdout:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("frame=") and progress is not None:
                try:
                    progress(min(kept, int(line.split("=", 1)[1])), kept)
                except ValueError:
                    pass
            if cancel_check is not None and cancel_check():
                p.kill()
                p.wait(timeout=10)
                raise CutCancelled("cancelled")
        rc = p.wait()
        if rc != 0:
            err_file.seek(0)
            raise VideoError(f"ffmpeg failed ({rc}): {err_file.read().decode(errors='replace')[-800:]}")
        got = count_packets(partial)
        if got != kept:
            raise VideoError(f"the cut video has {got} frames, expected {kept}; it was not saved")
        os.replace(partial, out_path)
    except BaseException:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)
        try:
            os.remove(partial)
        except OSError:
            pass
        raise
    finally:
        err_file.close()
    if progress is not None:
        progress(kept, kept)
    return {"frames": kept, "removed": info.nb_frames - kept, "duration": kept / info.fps, "pieces": len(keep), "audio": bool(alabel)}
