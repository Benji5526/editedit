"""Shared by the training scripts. Everything stays on this machine: Ultralytics is told not to go online, and every path it
writes to is inside training/ (its own global settings file is not relied on)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")          # datasets made by make_dataset.py (images of the footage: never committed)
RUNS_DIR = os.path.join(HERE, "runs")          # training runs and exported models
CLASSES = ["censor"]                           # one class: "somebody decided to cover this" (see src/pair_labels.py)

os.environ.setdefault("YOLO_OFFLINE", "true")              # no update checks, no downloads, no analytics
os.environ.setdefault("YOLO_AUTOINSTALL", "false")         # never pip-install anything by itself
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "disabled")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def need_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        sys.exit("Ultralytics is not installed in this Python. Run the training scripts with the training environment:\n"
                 r"  .venv-train\Scripts\python.exe training\<script>.py ...")
