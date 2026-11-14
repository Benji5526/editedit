"""Turn label files into a training dataset (images + YOLO label files).

    python training/make_dataset.py NAME --video ORIGINAL [--video ORIGINAL2 ...] [--every 15] [--val-part 0.2]

For every ORIGINAL video the label file logs/pair_labels/<fingerprint>.json is used (made by tools/pair_labels.py from the
original and its mosaicked copy). Frames are taken from the ORIGINAL (the model must learn what the thing looks like, not
what a mosaic looks like):
  * every --every-th frame that has regions            -> an image with boxes
  * about --background times as many frames without    -> an image with an empty label file (teaches "nothing here")
The validation part is cut out of each video as whole time blocks (neighbouring frames are nearly identical: mixing them
between training and validation would make the numbers look better than they are).

Writes training/data/NAME/{images,labels}/{train,val}/ and data.yaml. The image files are frames of the footage: they stay in
training/data (git-ignored). File names are fingerprint + frame number, never the video's name."""
import argparse
import json
import os
import random

import cv2

from common import CLASSES, DATA_DIR, ROOT

from src.pair_labels import load_labels  # noqa: E402
from src.video_io import FrameReader, probe_video  # noqa: E402


def pick_frames(doc: dict, every: int, background: float, seed: int = 0):
    labelled = sorted(int(f) for f in doc["frames"])
    pos = labelled[::max(1, every)]
    first, n = doc.get("first_frame", 0), doc["frames_compared"]
    has = set(labelled)
    # background frames: compared, no region, and not right next to a labelled frame (the mosaic may start a frame late)
    free = [f for f in range(first, first + n, max(1, every)) if not any((f + d) in has for d in range(-5, 6))]
    random.Random(seed).shuffle(free)
    neg = sorted(free[:int(round(len(pos) * background))])
    return pos, neg


def val_blocks(total: int, part: float, blocks: int = 5):
    """Frame ranges that go to validation: `blocks` equal stretches spread over the video, together `part` of it."""
    if part <= 0:
        return []
    size = max(1, int(total * part / blocks))
    step = total / blocks
    return [(int(i * step + (step - size) / 2), int(i * step + (step - size) / 2) + size) for i in range(blocks)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--video", action="append", required=True, help="an ORIGINAL video whose label file exists (repeatable)")
    ap.add_argument("--labels-dir", default=os.path.join(ROOT, "logs", "pair_labels"))
    ap.add_argument("--every", type=int, default=15, help="take every N-th labelled frame (neighbouring frames are near duplicates)")
    ap.add_argument("--background", type=float, default=0.3, help="frames without regions, as a share of the labelled ones")
    ap.add_argument("--val-part", type=float, default=0.2)
    ap.add_argument("--width", type=int, default=1280, help="stored image width (the detector trains at 640; more keeps small things)")
    a = ap.parse_args(argv)
    out = os.path.join(DATA_DIR, a.name)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}          # images, boxes, background images
    for path in a.video:
        info = probe_video(path)
        lab = os.path.join(a.labels_dir, info.fingerprint + ".json")
        if not os.path.isfile(lab):
            raise SystemExit(f"no label file for this video ({lab}). Run tools/pair_labels.py on it and its mosaicked copy first.")
        doc = load_labels(lab)
        W, H = doc["size"]
        pos, neg = pick_frames(doc, a.every, a.background)
        wanted = {f: True for f in pos}
        wanted.update({f: False for f in neg})
        blocks = val_blocks(info.nb_frames, a.val_part)
        w = min(a.width, W)
        h = int(round(H * w / W / 2)) * 2
        last = max(wanted) if wanted else -1
        with FrameReader(info, max_frames=last + 1, scale_to=(w, h)) as r:
            for f in range(last + 1):
                img = r.read()
                if img is None:
                    break
                if f not in wanted:
                    continue
                split = "val" if any(s <= f < e for s, e in blocks) else "train"
                stem = f"{info.fingerprint[:12]}_{f:07d}"
                cv2.imwrite(os.path.join(out, "images", split, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for x1, y1, x2, y2 in (doc["frames"].get(str(f), []) if wanted[f] else []):
                    lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                with open(os.path.join(out, "labels", split, stem + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines))
                c = counts[split]
                c[0] += 1
                c[1] += len(lines)
                c[2] += 0 if lines else 1
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {out.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n" +
                 "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    summary = {"dataset": out, "videos": len(a.video),
               **{k: {"images": v[0], "boxes": v[1], "background_images": v[2]} for k, v in counts.items()}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if counts["val"][0] == 0 or counts["train"][0] == 0:
        print("WARNING: one of the two parts is empty - the dataset is too small to train or to judge a model on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
