#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch

from ultralytics import YOLO


def describe_structure(obj, prefix="output"):
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__} len={len(obj)}")
        for i, item in enumerate(obj):
            describe_structure(item, prefix=f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        print(f"{prefix}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe_structure(v, prefix=f"{prefix}[{k}]")
    else:
        print(f"{prefix}: {type(obj).__name__}")


def main():
    cfgs = [
        "ultralytics/cfg/models/v8/yolov8n-seg-spd-backbone.yaml",
        "ultralytics/cfg/models/v8/yolov8n-seg-dsc-backbone.yaml",
        "ultralytics/cfg/models/v8/yolov8n-seg-spd-dsc-backbone.yaml",
    ]

    x = torch.randn(1, 3, 640, 640)
    for cfg in cfgs:
        cfg_path = Path(cfg)
        print(f"\n===== Checking {cfg_path} =====")
        model = YOLO(str(cfg_path))
        model.info()
        model.model.eval()
        with torch.no_grad():
            y = model.model(x)
        describe_structure(y)


if __name__ == "__main__":
    main()
