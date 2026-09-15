#!/usr/bin/env python3
"""Prepare the latest two-video capture for Gaussian training."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

from fish_pipeline.fresh_mac import camera_report, choose_boxes, extract, masks, prepare_reconstruction, reconstruct
from fish_pipeline.common import write_json

SUBMISSION = Path(__file__).resolve().parents[1]
SOURCE = SUBMISSION / "source" / "Latest source"
WORK = SUBMISSION / "work" / "latest"


def show_preview(path: Path) -> None:
    if path.exists():
        print(f"Mask preview: {path}")
        if sys.stdin.isatty():
            subprocess.run(["open", str(path)], check=False)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    prepared = WORK / "brush_dataset" / "adapter.json"
    manifest = WORK / "dataset" / "manifest.json"
    if prepared.exists() and manifest.exists() and (WORK / "images").exists() and (WORK / "masks").exists():
        print("The reviewed work/latest image and mask set is already prepared.")
        show_preview(WORK / "mask_preview.jpg")
        return

    videos = sorted(SOURCE.glob("Kro_normal*.MOV"))
    if not videos:
        raise FileNotFoundError(f"Put the final Kro_normal videos in {SOURCE}")

    stage = WORK / "preprocess_new"
    if stage.exists():
        raise FileExistsError(f"A partial run exists at {stage}. Rename it before starting again.")
    stage.mkdir()
    frames = extract(videos, stage, count=82)
    boxes = choose_boxes(stage, frames)
    masks(stage, frames, boxes)
    show_preview(stage / "mask_preview.jpg")
    if sys.stdin.isatty():
        answer = input("Check the fish and background. Press Enter to continue, or type q to stop: ")
        if answer.strip().lower() == "q":
            raise SystemExit("Stopped after mask review.")

    reconstruction = reconstruct(stage, frames)
    report = camera_report(reconstruction, frames)
    write_json(stage / "camera_report.json", report)
    print(json.dumps(report, indent=2))
    if report["issues"]:
        raise RuntimeError("Camera checks failed. See work/latest/preprocess_new/camera_report.json")
    prepare_reconstruction(stage, reconstruction, frames, resolution=512)

    shutil.move(stage / "images", WORK / "raw_frames")
    shutil.move(stage / "dataset" / "images", WORK / "images")
    for name in ("masks", "dataset", "brush_dataset", "selected_sparse"):
        destination = WORK / name
        if destination.exists():
            raise FileExistsError(f"{destination} already exists")
        shutil.move(stage / name, destination)
    for name in ("frames.json", "boxes.json", "mask_preview.jpg", "camera_report.json"):
        shutil.move(stage / name, WORK / name)

    manifest_path = WORK / "dataset" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    for frame in data["frames"]:
        frame["image"] = "../images/" + Path(frame["image"]).name
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    stage.rmdir()
    print("\nPreprocessing complete. You can now run scripts/train.py.")


if __name__ == "__main__":
    main()
