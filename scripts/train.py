"""Fine-tune a YOLO detector on the merged dataset and copy best weights to weights/best.pt."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/merged/data.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="pretrained starting checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs", name="safeplacescan", exist_ok=True, seed=42)

    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, "weights/best.pt")
    print("Saved weights/best.pt")


if __name__ == "__main__":
    main()
