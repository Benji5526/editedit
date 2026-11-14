"""Train a detector on a dataset made by make_dataset.py and export it for the mosaic app.

    .venv-train\\Scripts\\python.exe training\\train.py NAME [--base yolov8m.pt | yolov8n.yaml] [--epochs 100] [--imgsz 640]

--base: a weights file to start from (far better with little data; the file must already be on this machine - nothing is
downloaded) or a model description like yolov8n.yaml to start from nothing (needs a lot of data; fine for a dry run).
Writes training/runs/NAME/<run>/ with the usual Ultralytics results, plus
    model.onnx          the model for the mosaic app ("탐지 모델 파일")
    model.labels.json   its class names and input size (the app reads it next to the onnx)
    report.json         the numbers on the validation part
The numbers say how the model does on the validation frames of THIS dataset. They are no promise for other footage."""
import argparse
import json
import os
import shutil

from common import CLASSES, DATA_DIR, RUNS_DIR, need_ultralytics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset name (training/data/NAME)")
    ap.add_argument("--base", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--run", default="train")
    a = ap.parse_args(argv)
    need_ultralytics()
    from ultralytics import YOLO
    data = os.path.join(DATA_DIR, a.name, "data.yaml")
    if not os.path.isfile(data):
        raise SystemExit(f"no dataset {data}: run make_dataset.py first")
    if a.base.endswith(".pt") and not os.path.isfile(a.base):
        raise SystemExit(f"base weights not found: {a.base}\nNothing is downloaded automatically. Put the file there, or use a "
                         f"model description such as yolov8n.yaml to train from nothing.")
    model = YOLO(a.base)
    project = os.path.join(RUNS_DIR, a.name)
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project=project, name=a.run, exist_ok=True,
                workers=0, plots=False, pretrained=a.base.endswith(".pt"), verbose=False)
    run_dir = os.path.join(project, a.run)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    m = best.val(data=data, imgsz=a.imgsz, device=a.device, project=project, name=a.run + "_val", exist_ok=True, workers=0, plots=False,
                 verbose=False)
    onnx = best.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=False, dynamic=False)
    shutil.copy2(onnx, os.path.join(run_dir, "model.onnx"))
    with open(os.path.join(run_dir, "model.labels.json"), "w", encoding="utf-8") as fh:
        json.dump({"labels": CLASSES, "resolution": a.imgsz}, fh)
    report = {"dataset": a.name, "base": os.path.basename(a.base), "epochs": a.epochs, "imgsz": a.imgsz,
              "precision": float(m.box.mp), "recall": float(m.box.mr), "map50": float(m.box.map50), "map50_95": float(m.box.map),
              "model": os.path.join(run_dir, "model.onnx")}
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
