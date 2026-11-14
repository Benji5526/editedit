"""Compare detector models on the validation part of a dataset, the way the mosaic app would use them.

    python training/evaluate.py NAME MODEL.onnx [MODEL2.onnx ...] [--threshold 0.25] [--standard]

Each model is run through the app's own detector code (src/detection.py), so thresholds, tiling and box handling are the same
as in a real job. --standard adds the NudeNet model that ships with the app (its usual two classes) as a reference.
For every model:
  missed regions   labelled regions that no detection touches            (what would stay uncovered - the number that matters)
  extra boxes      detections that touch no labelled region              (what would be covered needlessly; on background
                                                                          images every box is extra)
A model is only better when it misses fewer regions without many more extra boxes. The numbers describe THIS validation set."""
import argparse
import glob
import json
import os

import cv2

from common import DATA_DIR

from src.detection import NudenetDetector  # noqa: E402


def touches(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def evaluate(detector, images_dir: str, labels_dir: str) -> dict:
    regions = missed = extra = boxes = images = 0
    for path in sorted(glob.glob(os.path.join(images_dir, "*.jpg"))):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        truth = []
        lab = os.path.join(labels_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.isfile(lab):
            for line in open(lab, encoding="utf-8").read().split("\n"):
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    truth.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        dets = [d.box for d in detector.detect(img)]
        images += 1
        regions += len(truth)
        boxes += len(dets)
        missed += sum(1 for t in truth if not any(touches(t, d) for d in dets))
        extra += sum(1 for d in dets if not any(touches(t, d) for t in truth))
    return {"images": images, "regions": regions, "missed_regions": missed, "missed_share": round(missed / regions, 4) if regions else None,
            "detections": boxes, "extra_boxes": extra, "extra_per_image": round(extra / images, 3) if images else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("models", nargs="*")
    ap.add_argument("--standard", action="store_true")
    ap.add_argument("--standard-model", help="another NudeNet onnx for --standard (e.g. models/640m.onnx)")
    ap.add_argument("--standard-res", type=int, default=320)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    base = os.path.join(DATA_DIR, a.name)
    images, labels = os.path.join(base, "images", "val"), os.path.join(base, "labels", "val")
    if not glob.glob(os.path.join(images, "*.jpg")):
        raise SystemExit(f"no validation images in {images}")
    out = {}
    if a.standard:
        det = NudenetDetector(a.threshold, model_path=a.standard_model, resolution=a.standard_res, device=a.device)
        out["standard NudeNet" + (f" ({os.path.basename(a.standard_model)})" if a.standard_model else "")] = evaluate(det, images, labels)
    for m in a.models:
        det = NudenetDetector(a.threshold, model_path=m, device=a.device)
        out[os.path.relpath(m, os.path.dirname(base))] = evaluate(det, images, labels)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
