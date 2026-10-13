"""Download every source dataset in configs/datasets.yaml from Roboflow (YOLOv8 format)."""
import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (in .env or your environment).")

    cfg = yaml.safe_load(open(args.config))
    rf = Roboflow(api_key=api_key)

    for src in cfg["sources"]:
        dest = Path(args.out) / src["name"]
        r = src.get("roboflow")
        if dest.exists():
            print(f"[skip] {src['name']}: already at {dest}")
            continue
        if not r:
            print(f"[local] {src['name']}: no roboflow entry, put YOLO data at {dest}")
            continue
        print(f"[download] {src['name']} <- {r['workspace']}/{r['project']} v{r['version']}")
        project = rf.workspace(r["workspace"]).project(r["project"])
        project.version(r["version"]).download("yolov8", location=str(dest))


if __name__ == "__main__":
    main()
