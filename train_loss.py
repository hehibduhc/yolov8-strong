"""Run YOLOv8-seg training sequentially with multiple segmentation loss settings.

Usage:
    python train_loss.py \
        --model yolov8n-seg.pt \
        --data ultralytics/cfg/datasets/coco8-seg.yaml \
        --epochs 100 \
        --imgsz 640 \
        --batch 16
"""

from __future__ import annotations

import argparse
from copy import deepcopy

from ultralytics import YOLO


LOSS_EXPERIMENTS = [
    {
        "seg_loss_type": "bce_dice",
        "dice_weight": 1.0,
        "name_suffix": "bce_dice",
    },
    {
        "seg_loss_type": "bce_tversky",
        "tversky_weight": 1.0,
        "tversky_alpha": 0.7,
        "tversky_beta": 0.3,
        "name_suffix": "bce_tversky",
    },
    {
        "seg_loss_type": "bce_dice_boundary",
        "dice_weight": 1.0,
        "boundary_weight": 0.2,
        "name_suffix": "bce_dice_boundary",
    },
]


def build_parser() -> argparse.ArgumentParser:
    """Build command line arguments for sequential loss experiments."""
    parser = argparse.ArgumentParser(description="Sequential YOLOv8-seg training with different segmentation losses")
    parser.add_argument("--model", type=str, required=True, help="Model path, e.g. yolov8n-seg.pt or *.yaml")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Training device, e.g. 0 or cpu")
    parser.add_argument("--project", type=str, default="runs/segment", help="Project directory")
    parser.add_argument("--name-prefix", type=str, default="crack_seg", help="Experiment name prefix")
    parser.add_argument("--pretrained", type=str, default="true", help="Whether to use pretrained weights (true/false)")
    parser.add_argument("--amp", type=str, default="true", help="Enable AMP (true/false)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def str2bool(value: str) -> bool:
    """Convert common string bool inputs to bool."""
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def main() -> None:
    """Run three trainings sequentially using different segmentation loss settings."""
    args = build_parser().parse_args()

    base_train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "pretrained": str2bool(args.pretrained),
        "amp": str2bool(args.amp),
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
    }

    for exp in LOSS_EXPERIMENTS:
        train_args = deepcopy(base_train_args)
        train_args["name"] = f"{args.name_prefix}_{exp['name_suffix']}"
        train_args.update({k: v for k, v in exp.items() if k != "name_suffix"})

        print("\n" + "=" * 80)
        print(f"Start training: {train_args['name']}")
        print(f"seg_loss_type={train_args['seg_loss_type']}")
        print("=" * 80)

        # 每个实验重新初始化模型，确保三组损失从同一起点比较
        model = YOLO(args.model)
        model.train(**train_args)


if __name__ == "__main__":
    main()
