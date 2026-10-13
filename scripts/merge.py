"""Merge several YOLO datasets into one with a unified label set.

What it handles:
  - different class names/IDs per source -> remapped via class_map
  - polygon (segmentation) labels mixed in -> converted to bounding boxes
  - wildly different dataset sizes -> capped with max_images
  - each source's own train/val/test split -> pooled and re-split with a fixed seed
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_class_names(src_dir: Path):
    names = yaml.safe_load(open(src_dir / "data.yaml"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def image_label_pairs(src_dir: Path):
    pairs = []
    for img in src_dir.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS or "images" not in img.parts:
            continue
        parts = list(img.parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        pairs.append((img, Path(*parts).with_suffix(".txt")))
    return sorted(pairs)


def to_bbox(vals):
    """YOLO box (4 values) passes through; polygon (x1 y1 x2 y2 ...) -> enclosing box."""
    if len(vals) == 4:
        return vals
    if len(vals) < 6 or len(vals) % 2:
        return None
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def remap(label_path: Path, src_names, class_map, unified):
    out = []
    if not label_path.exists():
        return out
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        name = src_names[int(parts[0])]
        target = class_map.get(name, class_map.get("*"))
        if target is None:
            continue
        box = to_bbox([float(v) for v in parts[1:]])
        if box is None or box[2] <= 0 or box[3] <= 0:
            continue
        out.append(f"{unified.index(target)} " + " ".join(f"{v:.6f}" for v in box))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/merged")
    p.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-empty", action="store_true",
                   help="keep images with no matching labels (as negatives)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    unified = cfg["classes"]
    rng = random.Random(args.seed)

    items, per_source = [], {}
    for src in cfg["sources"]:
        bad = set(src["class_map"].values()) - set(unified)
        if bad:
            raise SystemExit(f"{src['name']}: class_map targets {bad} not in classes")
        d = Path(args.raw) / src["name"]
        if not d.exists():
            print(f"[warn] {src['name']}: missing {d}, skipping")
            continue
        names = source_class_names(d)
        pairs = image_label_pairs(d)
        rng.shuffle(pairs)
        cap, kept = src.get("max_images"), 0
        for img, lbl in pairs:
            lines = remap(lbl, names, src["class_map"], unified)
            if not lines and not args.keep_empty:
                continue
            items.append((img, lines, src["name"]))
            kept += 1
            if cap and kept >= cap:
                break
        per_source[src["name"]] = {"found": len(pairs), "kept": kept}
        print(f"[source] {src['name']}: kept {kept} of {len(pairs)} images")

    if not items:
        raise SystemExit("No images collected. Did you run download.py?")

    rng.shuffle(items)
    fr = [float(x) for x in args.split.split(",")]
    n = len(items)
    n_train, n_val = int(n * fr[0]), int(n * fr[1])
    splits = {"train": items[:n_train],
              "val": items[n_train:n_train + n_val],
              "test": items[n_train + n_val:]}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    box_counts = {}
    for split, subset in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        c = Counter()
        for i, (img, lines, name) in enumerate(subset):
            stem = f"{name}_{i:06d}"
            shutil.copy2(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            c.update(unified[int(l.split()[0])] for l in lines)
        box_counts[split] = {k: c.get(k, 0) for k in unified}

    yaml.safe_dump({"path": str(out.resolve()), "train": "images/train",
                    "val": "images/val", "test": "images/test", "names": unified},
                   open(out / "data.yaml", "w"), sort_keys=False)

    stats = {"images": {k: len(v) for k, v in splits.items()},
             "boxes_per_class": box_counts, "sources": per_source}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
