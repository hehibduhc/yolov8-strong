from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils import YAML


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_paths = [
        root / "ultralytics/cfg/models/v8/yolov8n-seg-pconv-c2ffaster.yaml",
        root / "ultralytics/cfg/models/v8/yolov8n-seg-gsneck.yaml",
        root / "ultralytics/cfg/models/v8/yolov8n-seg-ema.yaml",
        root / "ultralytics/cfg/models/v8/yolov8n-seg-pconv-gsneck-ema.yaml",
    ]

    x = torch.randn(1, 3, 640, 640)
    for mp in model_paths:
        print(f"\n=== Checking {mp} ===")
        model = YOLO(str(mp))
        model.info()
        with torch.no_grad():
            _ = model.model(x)
        print("Forward pass OK")

    cfg = YAML.load(str(root / "ultralytics/cfg/experiments/segment-mpdiou.yaml"))
    mp_model = YOLO(str(model_paths[-1]))
    for k, v in cfg.items():
        setattr(mp_model.model.args, k, v)
    criterion = mp_model.model.init_criterion()
    print(f"Criterion built: {criterion.__class__.__name__}")
    print(f"IoU loss type: {criterion.bbox_loss.iou_type}")
    assert criterion.bbox_loss.iou_type == "mpdiou", "MPDIoU path was not selected from cfg"


if __name__ == "__main__":
    main()
