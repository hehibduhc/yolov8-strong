from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch validate YOLO best.pt weights and export metrics to Excel.")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="One or more best.pt paths. Example: --weights runs/segment/exp1/weights/best.pt runs/segment/exp2/weights/best.pt",
    )
    parser.add_argument(
        "--glob",
        dest="weight_globs",
        nargs="*",
        default=[],
        help="Glob patterns for best.pt. Example: --glob 'runs/segment/*/weights/best.pt'",
    )
    parser.add_argument(
        "--data",
        default=r"ultralytics/cfg/datasets/crack_seg.yaml",
        help="Dataset config path passed to model.val().",
    )
    parser.add_argument("--device", default="0", help="Device passed to model.val(), such as 0, 0,1 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="Optional IoU threshold.")
    parser.add_argument("--split", default="val", help="Dataset split for validation.")
    parser.add_argument("--plots", action="store_true", help="Whether to save validation plots.")
    parser.add_argument("--output", default="result.xlsx", help="Output Excel file path.")
    return parser.parse_args()


def resolve_weight_paths(weight_args: list[str], weight_globs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for weight in weight_args:
        candidates.append(Path(weight))
    for pattern in weight_globs:
        candidates.extend(Path(match) for match in glob.glob(pattern, recursive=True))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in sorted((path.expanduser().resolve() for path in candidates), key=lambda p: str(p)):
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Weight file not found: {path}")
        seen.add(path)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No valid weight files found. Use --weights and/or --glob to specify best.pt files.")
    return unique_paths


def infer_experiment_name(weight_path: Path) -> str:
    if weight_path.parent.name == "weights" and weight_path.parent.parent.name:
        return weight_path.parent.parent.name
    return weight_path.parent.name


def normalize_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def collect_model_stats(model: Any, imgsz: int) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    params = get_num_params(model)
    gflops = round(get_flops(model, imgsz=imgsz), 3)
    return {
        "parameters": int(params),
        "GFLOPs": gflops,
    }


def collect_speed_stats(validator: Any) -> dict[str, Any]:
    inference_ms = normalize_scalar(getattr(validator, "speed", {}).get("inference"))
    inference_ms = float(inference_ms) if inference_ms is not None else None
    fps = round(1000.0 / inference_ms, 3) if inference_ms and inference_ms > 0 else None
    return {
        "inference_ms": round(inference_ms, 3) if inference_ms is not None else None,
        "FPS": fps,
    }


def build_val_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    val_kwargs: dict[str, Any] = {
        "data": args.data,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
        "plots": args.plots,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    if args.iou is not None:
        val_kwargs["iou"] = args.iou
    return val_kwargs


def collect_result_row(weight_path: Path, run_name: str, validator: Any, model_stats: dict[str, Any]) -> dict[str, Any]:
    metrics = validator.metrics
    box_p, box_r, box_map50, box_map5095, mask_p, mask_r, mask_map50, mask_map5095 = metrics.mean_results()
    speed_stats = collect_speed_stats(validator)
    return {
        "name": run_name,
        "weight_path": str(weight_path),
        "Class": "all",
        "Images": int(validator.seen),
        "Instances": int(metrics.nt_per_class.sum()),
        "parameters": model_stats["parameters"],
        "GFLOPs": model_stats["GFLOPs"],
        "inference_ms": speed_stats["inference_ms"],
        "FPS": speed_stats["FPS"],
        "Box(P)": normalize_scalar(box_p),
        "Box(R)": normalize_scalar(box_r),
        "Box(mAP50)": normalize_scalar(box_map50),
        "Box(mAP50-95)": normalize_scalar(box_map5095),
        "Mask(P)": normalize_scalar(mask_p),
        "Mask(R)": normalize_scalar(mask_r),
        "Mask(mAP50)": normalize_scalar(mask_map50),
        "Mask(mAP50-95)": normalize_scalar(mask_map5095),
        "IoU_fg": normalize_scalar(getattr(validator, "iou_crack", None)),
        "IoU_bg": normalize_scalar(getattr(validator, "iou_bg", None)),
        "mIoU": normalize_scalar(getattr(validator, "miou", None)),
    }


def print_run_summary(result_row: dict[str, Any]) -> None:
    params = result_row["parameters"]
    gflops = result_row["GFLOPs"]
    inference_ms = result_row["inference_ms"]
    fps = result_row["FPS"]
    inference_ms_text = f"{inference_ms:.3f}" if inference_ms is not None else "N/A"
    fps_text = f"{fps:.3f}" if fps is not None else "N/A"
    print(
        f"[Extra] parameters={params:,}, GFLOPs={gflops:.3f}, "
        f"inference(ms/image)={inference_ms_text}, FPS={fps_text}"
    )


def save_excel(result_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result_rows)
    ordered_columns = [
        "name",
        "weight_path",
        "Class",
        "Images",
        "Instances",
        "parameters",
        "GFLOPs",
        "inference_ms",
        "FPS",
        "Box(P)",
        "Box(R)",
        "Box(mAP50)",
        "Box(mAP50-95)",
        "Mask(P)",
        "Mask(R)",
        "Mask(mAP50)",
        "Mask(mAP50-95)",
        "IoU_fg",
        "IoU_bg",
        "mIoU",
    ]
    df = df[ordered_columns]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)


def main() -> None:
    args = parse_args()
    weight_paths = resolve_weight_paths(args.weights, args.weight_globs)
    val_kwargs = build_val_kwargs(args)

    from ultralytics import YOLO

    result_rows: list[dict[str, Any]] = []

    for weight_path in weight_paths:
        run_name = infer_experiment_name(weight_path)
        print(f"\n[Val] {run_name}: {weight_path}")
        model = YOLO(str(weight_path))
        model_stats = collect_model_stats(model.model, imgsz=args.imgsz)
        validator_cls = model._smart_load("validator")
        validator = validator_cls(args={**model.overrides, **{"rect": True}, **val_kwargs, "mode": "val"}, _callbacks=model.callbacks)
        validator(model=model.model)
        result_row = collect_result_row(weight_path, run_name, validator, model_stats)
        print_run_summary(result_row)
        result_rows.append(result_row)

    output_path = Path(args.output).expanduser().resolve()
    save_excel(result_rows, output_path)
    print(f"\nSaved validation summary to: {output_path}")


if __name__ == "__main__":
    main()

r"""python val_local.py ^
  --weights runs\segment\backbone-conv-HWD-module\yolov8-seg-hwd-backbone-4down\weights\best.pt^
  --data ultralytics/cfg/datasets/crack_seg.yaml ^
  --device 0 ^
  --split test
  --output result.xlsx

  python val_local.py ^
  --weights runs\segment\0315_paper_ablation\yolov8-seg_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-mdka-sadilation_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-mdka-sadilation-sppf-replk-full_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-spd13_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-spd13-mdka-sadilation_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-spd13-mdka-sadilation-sppf-replk-full_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-spd13-sppf-replk-full_seed43\weights\best.pt
  runs\segment\0315_paper_ablation\yolov8-seg-sppf-replk-full_seed43\weights\best.pt^
  --data ultralytics/cfg/datasets/crack_seg_paper.yaml ^
  --device 0 ^
  --split test
  --output result.xlsx

"""
