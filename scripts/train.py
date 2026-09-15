#!/usr/bin/env python3
"""Train the final video model from the prepared frames and masks."""

from pathlib import Path
import json
import shutil
import time

from fish_pipeline.mac import evaluate_mac, train_mac

SUBMISSION = Path(__file__).resolve().parents[1]
WORK = SUBMISSION / "work" / "latest"
OUTPUTS = SUBMISSION / "outputs"


def main() -> None:
    dataset = WORK / "brush_dataset"
    manifest = WORK / "dataset" / "manifest.json"
    if not dataset.exists() or not manifest.exists():
        raise FileNotFoundError("Prepared data is missing. Run scripts/preprocess.py first.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = WORK / f"training_{stamp}"
    config = {
        "steps": 6000,
        "resolution": 512,
        "export_every": 6000,
        "eval_every": 6000,
        "sh_degree": 3,
        "max_splats": 50000,
        "growth_stop_iter": 5000,
        "lr_mean": 0.00002,
        "lr_mean_end": 0.000001,
    }
    result = train_mac(dataset, run_dir, config)
    model = Path(result["model"])
    destination = OUTPUTS / "models" / "new_run_model.ply"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(model, destination)
    evaluate_mac(manifest, run_dir / "eval_6000", WORK / f"evaluation_{stamp}")
    (OUTPUTS / "latest_run.json").write_text(
        json.dumps({"model": str(destination.relative_to(SUBMISSION)), "run": str(run_dir.relative_to(SUBMISSION))}, indent=2)
        + "\n"
    )
    print(f"\nTraining complete. Model saved to {destination}")


if __name__ == "__main__":
    main()
