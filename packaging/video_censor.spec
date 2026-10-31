# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: ONE folder (onedir) with TWO executables that share one set of libraries.
#   VideoCensor.exe        desktop app, windowed (no console). Does not contain torch/onnxruntime/NudeNet: it never imports them.
#   VideoCensorWorker.exe  processing worker (console subsystem; the app starts it hidden and reads its stdout)
# Build:  python tools/build_exe.py        (do not call pyinstaller on this file directly: the post-build step creates the data folders)
# onedir, not onefile: a onefile exe would unpack several GB of CUDA libraries into %TEMP% on every start.
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# ---- libraries only the worker needs
worker_datas = (collect_data_files("sam2")          # the hydra model configs (configs/sam2.1/*.yaml) are looked up as package data
                + collect_data_files("hydra") + collect_data_files("nudenet")   # nudenet: the bundled 320n.onnx model
                + copy_metadata("sam2") + copy_metadata("nudenet") + copy_metadata("hydra-core")   # importlib.metadata.version(...) at run time
                + copy_metadata("torch") + copy_metadata("tqdm"))
worker_hidden = collect_submodules("sam2") + collect_submodules("hydra") + ["omegaconf", "iopath", "onnxruntime", "psutil", "PIL"]

GUI_EXCLUDES = ["torch", "torchvision", "sam2", "onnxruntime", "nudenet", "hydra", "omegaconf", "iopath",
                "tkinter", "matplotlib", "IPython", "pytest", "scipy", "pandas"]
WORKER_EXCLUDES = ["PySide6", "shiboken6", "tkinter", "matplotlib", "IPython", "pytest", "scipy", "pandas"]

gui = Analysis([os.path.join(ROOT, "app.py")], pathex=[ROOT], binaries=[], datas=copy_metadata("PySide6-Essentials"), hiddenimports=[],
               excludes=GUI_EXCLUDES, noarchive=False)
worker = Analysis([os.path.join(ROOT, "censor_video.py")], pathex=[ROOT], binaries=[], datas=worker_datas, hiddenimports=worker_hidden,
                  excludes=WORKER_EXCLUDES, noarchive=False)

MERGE((gui, "app", "VideoCensor"), (worker, "censor_video", "VideoCensorWorker"))

gui_pyz = PYZ(gui.pure)
gui_exe = EXE(gui_pyz, gui.scripts, [], exclude_binaries=True, name="VideoCensor", console=False, upx=False, disable_windowed_traceback=False)
worker_pyz = PYZ(worker.pure)
worker_exe = EXE(worker_pyz, worker.scripts, [], exclude_binaries=True, name="VideoCensorWorker", console=True, upx=False)

coll = COLLECT(gui_exe, gui.binaries, gui.datas, worker_exe, worker.binaries, worker.datas, strip=False, upx=False, name="VideoCensor")
