"""Evaluate on the held-out test split; write per-class precision/recall/mAP to results/."""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--data", default="data/merged/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    m = YOLO(args.weights).val(data=args.data, split="test", imgsz=args.imgsz,
                               project="runs", name="eval", exist_ok=True)
    box = m.box

    rows = [{"class": m.names[int(c)], "precision": float(box.p[i]), "recall": float(box.r[i]),
             "mAP50": float(box.ap50[i]), "mAP50-95": float(box.ap[i])}
            for i, c in enumerate(box.ap_class_index)]
    rows.append({"class": "all", "precision": float(box.mp), "recall": float(box.mr),
                 "mAP50": float(box.map50), "mAP50-95": float(box.map)})

    out = Path("results")
    out.mkdir(exist_ok=True)
    json.dump(rows, open(out / "metrics.json", "w"), indent=2)

    md = ["| Class | Precision | Recall | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|"]
    md += [f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | "
           f"{r['mAP50']:.3f} | {r['mAP50-95']:.3f} |" for r in rows]
    (out / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    for name in ["confusion_matrix_normalized.png", "PR_curve.png", "BoxPR_curve.png"]:
        f = Path(m.save_dir) / name
        if f.exists():
            shutil.copy2(f, out / name)


if __name__ == "__main__":
    main()
