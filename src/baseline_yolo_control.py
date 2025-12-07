#!/usr/bin/env python
"""
Baseline control script for your custom bottle object.

What it does:
  1) Evaluates a COCO-pretrained YOLO model on your dataset (optional zero-shot).
  2) Trains YOLO on the dataset described by your data.yaml.
  3) Evaluates the trained model.
  4) Saves the key metrics to JSON for later comparison
     (e.g., when you switch from real-only to real+synthetic).

Edit ONLY the "==== USER CONFIG ====" section below as you
create new datasets (real-only, real+synthetic, new test set, etc.).
"""

from pathlib import Path
from datetime import datetime
import json

from ultralytics import YOLO

# ================================================================
# ======================  USER CONFIG  ===========================
# ================================================================

# Workspace root (where you keep datasets / runs)
WORKSPACE_ROOT = Path(r"C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline")

# Which dataset to use for THIS experiment:
#   e.g. real_only.yaml, real_plus_synth.yaml, synth_only.yaml, etc.
DATA_YAML = WORKSPACE_ROOT / "data" / "bottle01" / "bottle01_real_only.yaml"
# Later you can point this to:
# DATA_YAML = WORKSPACE_ROOT / "data" / "my_bottle_real_plus_synth.yaml"

# COCO-pretrained weights (from C:\yolo_play)
PRETRAINED_WEIGHTS = Path(r"C:\yolo_play\yolo11n.pt")   # or yolov8n.pt

# Where to put results for this run
RUNS_PARENT = WORKSPACE_ROOT / "runs"
RUN_NAME = f"baseline_real_only_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Basic training / eval settings
IMG_SIZE = 640        # image size
EPOCHS = 50           # training epochs
BATCH_SIZE = 16       # batch size
DEVICE = "0"          # "0" for first GPU, "cpu" for CPU-only
TEST_SPLIT = "val"    # "val" or "test" (must exist in your data.yaml)

# If True, skip the zero-shot eval of the COCO-pretrained model
SKIP_PRETRAINED_EVAL = False

# ================================================================
# =====================  END USER CONFIG  ========================
# ================================================================


def metrics_to_dict(tag, metrics):
    """
    Convert Ultralytics 'metrics' object into a plain dict.
    'tag' is a label like 'pretrained' or 'trained_real_only'.
    """
    d = {
        "tag": tag,
        "map50_95": float(metrics.box.map),        # mAP@[0.5:0.95]
        "map50": float(metrics.box.map50),         # mAP@0.5
        "map75": float(metrics.box.map75),         # mAP@0.75
        "precision": float(metrics.box.p.mean()),  # mean precision over classes
        "recall": float(metrics.box.r.mean()),     # mean recall over classes
        "per_class_map": [float(x) for x in metrics.box.maps],  # list of mAP per class
    }
    return d


def main():
    # Compute run directory
    run_dir = RUNS_PARENT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== USING DATASET YAML: {DATA_YAML} ===")
    print(f"=== RESULTS WILL BE SAVED UNDER: {run_dir} ===")

    # 1) Load base model (typically COCO-pretrained)
    print(f"\n=== Loading base model: {PRETRAINED_WEIGHTS} ===")
    model = YOLO(PRETRAINED_WEIGHTS)

    metrics_summary = {}

    # 2) Zero-shot evaluation (COCO-pretrained on your dataset, no training)
    if not SKIP_PRETRAINED_EVAL:
        print("\n=== Step 1: Evaluating pretrained model (zero-shot baseline) ===")
        pretrained_metrics = model.val(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            split=TEST_SPLIT,
            device=DEVICE
        )

        pretrained_dict = metrics_to_dict("pretrained", pretrained_metrics)
        metrics_summary["pretrained"] = pretrained_dict

        print("\nZero-shot baseline on your dataset:")
        print(f"  mAP50-95: {pretrained_dict['map50_95']:.4f}")
        print(f"  mAP50:    {pretrained_dict['map50']:.4f}")
        print(f"  Precision:{pretrained_dict['precision']:.4f}")
        print(f"  Recall:   {pretrained_dict['recall']:.4f}")
    else:
        print("\n[Skipping zero-shot evaluation of the pretrained model]")

    # 3) Train on the dataset (whatever DATA_YAML points to)
    print("\n=== Step 2: Training on dataset from data.yaml ===")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RUNS_PARENT),  # parent folder; Ultralytics adds 'name'
        name=RUN_NAME,
        exist_ok=True
    )

    # After training, model now points to best weights
    print("\n=== Step 3: Evaluating trained model on the same split ===")
    trained_metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        split=TEST_SPLIT,
        device=DEVICE
    )

    # You can rename tag depending on what DATA_YAML is (real-only, real+synthetic, etc.)
    trained_tag = "trained_real_only"
    trained_dict = metrics_to_dict(trained_tag, trained_metrics)
    metrics_summary[trained_tag] = trained_dict

    print("\nControl baseline (after training):")
    print(f"  mAP50-95: {trained_dict['map50_95']:.4f}")
    print(f"  mAP50:    {trained_dict['map50']:.4f}")
    print(f"  Precision:{trained_dict['precision']:.4f}")
    print(f"  Recall:   {trained_dict['recall']:.4f}")

    # 4) Save metrics to JSON for later comparison
    out_json = run_dir / "baseline_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "data_yaml": str(DATA_YAML),
                "pretrained_weights": str(PRETRAINED_WEIGHTS),
                "img_size": IMG_SIZE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
                "test_split": TEST_SPLIT,
                "metrics": metrics_summary,
            },
            f,
            indent=4
        )

    print(f"\nMetrics saved to: {out_json.resolve()}")
    print("\nTo compare experiments, just point DATA_YAML at a different yaml "
          "(e.g., real_only vs real_plus_synth), re-run this script, and "
          "compare the generated JSON files.")


if __name__ == "__main__":
    main()
