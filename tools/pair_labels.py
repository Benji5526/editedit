"""Make training labels from an original video and an already mosaicked version of it (see src/pair_labels.py).

    python tools/pair_labels.py ORIGINAL CENSORED [-o labels.json]

Writes logs/pair_labels/<fingerprint of the original>.json by default (no file names inside) and prints a summary: how well
the frames match, how many frames have regions, what was left out. Read the summary before trusting the labels."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pair_labels import compare, save_labels  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("censored")
    ap.add_argument("-o", "--output")
    ap.add_argument("--width", type=int, default=960, help="width the frames are compared at")
    ap.add_argument("--max-offset", type=int, default=90, help="search a constant frame offset of up to N frames (0 = none)")
    ap.add_argument("--offset", type=int, help="use this frame offset instead of searching")
    ap.add_argument("--threshold", type=int, default=22, help="pixel difference (0-255) that counts as changed")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--drop-static", action="store_true",
                    help="leave out places that are changed on almost every frame (logo, subtitle band). Only after checking the "
                         "'static_places' of a first run: a subject that does not move looks the same")
    a = ap.parse_args(argv)
    doc = compare(a.original, a.censored, a.width, a.max_offset, a.threshold, a.max_frames, offset=a.offset, drop_static=a.drop_static,
                  progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr))
    print(file=sys.stderr)
    out = a.output or os.path.join(ROOT, "logs", "pair_labels", doc["fingerprint"] + ".json")
    save_labels(out, doc)
    print(json.dumps({"labels": out, "offset": doc["offset"], "frames_compared": doc["frames_compared"], **doc["stats"],
                      "notes": doc["notes"]}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
