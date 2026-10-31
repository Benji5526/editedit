"""Build the Windows application folder:  python tools/build_exe.py [--no-clean]

Result: dist/VideoCensor/ with VideoCensor.exe (GUI), VideoCensorWorker.exe (processing), _internal/ (libraries) and the data folders
(models, input, output, logs, temp, config, ffmpeg/bin) plus README.md, READ_ME_FIRST.txt, THIRD_PARTY_NOTICES.md next to the exe.
Model weights, FFmpeg and videos are NOT copied into the build.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")
APP = os.path.join(DIST, "VideoCensor")


def folder_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", action="store_true", help="reuse PyInstaller's cache from the previous build")
    a = ap.parse_args()
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed: pip install -r requirements-build.txt")
        return 1
    t0 = time.time()
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", DIST, "--workpath", BUILD,
           os.path.join(ROOT, "packaging", "video_censor.spec")]
    if not a.no_clean:
        cmd.insert(3, "--clean")
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f"PyInstaller failed with exit code {rc}")
        return rc
    for exe in ("VideoCensor.exe", "VideoCensorWorker.exe"):
        if not os.path.isfile(os.path.join(APP, exe)):
            print(f"missing {exe} in the build")
            return 1

    # data folders and documents next to the exe
    for d in ("models", "input", "output", "logs", "temp", "config", os.path.join("ffmpeg", "bin")):
        os.makedirs(os.path.join(APP, d), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "models", "README.md"), os.path.join(APP, "models", "README.md"))
    shutil.copy2(os.path.join(ROOT, "config.example.json"), os.path.join(APP, "config", "config.example.json"))
    shutil.copy2(os.path.join(ROOT, "README.md"), os.path.join(APP, "README.md"))
    shutil.copy2(os.path.join(ROOT, "packaging", "READ_ME_FIRST.txt"), os.path.join(APP, "READ_ME_FIRST.txt"))
    subprocess.call([sys.executable, os.path.join(ROOT, "tools", "collect_licenses.py"), os.path.join(APP, "THIRD_PARTY_NOTICES.md")])
    shutil.copy2(os.path.join(APP, "THIRD_PARTY_NOTICES.md"), os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"))
    print(f"\nbuilt {APP}\n  size {folder_size(APP) / 2 ** 30:.2f} GB, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
