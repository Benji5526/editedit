"""Guards for jobs that run for hours: keep Windows awake, and check disk space before starting."""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from typing import Dict, List

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
OUTPUT_FACTOR = 1.5      # output (and the kept intermediate chunks) are assumed to be up to 1.5x the source size at the default quality


@contextmanager
def keep_awake():
    """Stop Windows from going to sleep while a job runs (the display may still turn off). No-op on other platforms.
    The request belongs to the calling thread and ends with the context or the process."""
    set_state = None
    if sys.platform == "win32":
        try:
            import ctypes
            set_state = ctypes.windll.kernel32.SetThreadExecutionState
            set_state.restype = ctypes.c_uint32
            set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception:  # noqa: BLE001  (never let a power-management call stop a job)
            set_state = None
    try:
        yield set_state is not None
    finally:
        if set_state is not None:
            try:
                set_state(ES_CONTINUOUS)
            except Exception:  # noqa: BLE001
                pass


def _drive(path: str) -> str:
    return os.path.splitdrive(os.path.abspath(path))[0].lower() or os.path.abspath(path).split(os.sep)[1:2][0]


def _existing_parent(path: str) -> str:
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def disk_check(source_bytes: int, output_path: str, work_dir: str) -> Dict[str, object]:
    """Estimate the space a job needs and compare with what is free.

    Needed: the intermediate chunks in ``work_dir`` and the final video next to ``output_path`` (each about OUTPUT_FACTOR x the source;
    summed when both are on the same drive). The estimate is rough - re-encoded 4K can be larger or smaller than the source - so
    running short is reported as a failure only when free space is below the estimate, and as a warning below twice the estimate."""
    need_each = int(source_bytes * OUTPUT_FACTOR)
    out_dir, work = _existing_parent(os.path.dirname(os.path.abspath(output_path))), _existing_parent(work_dir)
    same = _drive(out_dir) == _drive(work)
    needs = {work: need_each * (2 if same else 1)}
    if not same:
        needs[out_dir] = need_each
    problems: List[str] = []
    warnings: List[str] = []
    free_info = {}
    for path, need in needs.items():
        free = shutil.disk_usage(path).free
        free_info[path] = free
        gb = lambda b: f"{b / 2 ** 30:.1f} GB"
        if free < need:
            problems.append(f"not enough free disk space on {_drive(path) or path}: about {gb(need)} are needed for the intermediate chunks "
                            f"and the result, {gb(free)} are free. Free some space, choose another work/output folder, or pass --skip-disk-check.")
        elif free < 2 * need:
            warnings.append(f"free space on {_drive(path) or path} is only {gb(free)} (about {gb(need)} are needed)")
    return {"problems": problems, "warnings": warnings, "need": needs, "free": free_info, "same_drive": same}
