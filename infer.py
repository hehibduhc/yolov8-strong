from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.files import increment_path


@dataclass
class PolygonROI:
    points: list[list[int]]

    @property
    def x1(self) -> int:
        return min(point[0] for point in self.points)

    @property
    def y1(self) -> int:
        return min(point[1] for point in self.points)

    @property
    def x2(self) -> int:
        return max(point[0] for point in self.points) + 1

    @property
    def y2(self) -> int:
        return max(point[1] for point in self.points) + 1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contour(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)


@dataclass
class InstanceRecord:
    instance_id: int
    area: int
    bbox_xyxy: list[int]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.next_id = 1

    def new(self) -> int:
        instance_id = self.next_id
        self.parent[instance_id] = instance_id
        self.next_id += 1
        return instance_id

    def find(self, x: int) -> int:
        parent = self.parent.setdefault(x, x)
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if ra < rb:
            self.parent[rb] = ra
            return ra
        self.parent[ra] = rb
        return rb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROI tiled inference for YOLOv8-seg crack instance segmentation")
    parser.add_argument("--weights", type=str, default="yolov8n-seg.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="121212.jpg", help="input large image path")
    parser.add_argument(
        "--polygon",
        action="append",
        default=[],
        help="Polygon in 'x1,y1 x2,y2 x3,y3 ...' format. Repeat this argument for multiple polygons.",
    )
    parser.add_argument(
        "--polygon-file",
        type=str,
        default=None,
        help="JSON file containing polygons. Supports [[[x,y],...], ...] or [{'points': [[x,y], ...]}, ...].",
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="Legacy rectangle ROI in x1,y1,x2,y2 format. Repeat this argument for multiple ROIs.",
    )
    parser.add_argument(
        "--roi-file",
        type=str,
        default=None,
        help="JSON file containing ROI list. Supports [[x1,y1,x2,y2], ...] or [{'x1':..,'y1':..,'x2':..,'y2':..}, ...].",
    )
    parser.add_argument("--tile-size", type=int, default=1280, help="tile size")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap ratio in [0, 1)")
    parser.add_argument("--imgsz", type=int, default=1280, help="per-tile inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="max instances per tile")
    parser.add_argument("--device", type=str, default=None, help="inference device, e.g. 0 or cpu")
    parser.add_argument("--project", type=str, default="runs/segment_infer", help="output root dir")
    parser.add_argument("--name", type=str, default="exp", help="output run name")
    parser.add_argument("--exist-ok", action="store_true", help="reuse output dir if it exists")
    parser.add_argument("--merge-kernel", type=int, default=5, help="dilation kernel size for cross-tile merge")
    parser.add_argument("--min-area", type=int, default=20, help="min area to keep an instance")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    parser.add_argument("--select-max-side", type=int, default=1600, help="max preview side for ROI selection")
    parser.add_argument("--preview-grid", type=int, default=500, help="grid spacing in preview image")
    return parser


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def build_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile-size must be > 0")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError("invalid tile-size and overlap combination")

    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def clamp_point(x: int, y: int, width: int, height: int) -> list[int]:
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    return [x, y]


def clamp_roi(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> PolygonROI:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return PolygonROI(points=[[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]])


def parse_roi_text(roi_text: str, width: int, height: int) -> PolygonROI:
    parts = [p.strip() for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid ROI '{roi_text}'. Expected format: x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(part) for part in parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI '{roi_text}'. Require x2>x1 and y2>y1")
    return clamp_roi(x1, y1, x2, y2, width, height)


def parse_polygon_text(polygon_text: str, width: int, height: int) -> PolygonROI:
    raw_points = [segment.strip() for segment in polygon_text.replace(";", " ").split() if segment.strip()]
    if len(raw_points) < 3:
        raise ValueError(
            f"Invalid polygon '{polygon_text}'. Expected at least 3 points in 'x1,y1 x2,y2 x3,y3' format"
        )

    points: list[list[int]] = []
    for raw_point in raw_points:
        xy = [part.strip() for part in raw_point.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Invalid point '{raw_point}' in polygon '{polygon_text}'")
        points.append(clamp_point(int(xy[0]), int(xy[1]), width, height))

    contour = np.array(points, dtype=np.int32)
    if cv2.contourArea(contour) <= 0:
        raise ValueError(f"Invalid polygon '{polygon_text}'. Polygon area must be > 0")
    return PolygonROI(points=points)


def load_rois_from_args(args: argparse.Namespace, width: int, height: int) -> list[PolygonROI]:
    rois: list[PolygonROI] = []
    for polygon_text in args.polygon:
        rois.append(parse_polygon_text(polygon_text, width, height))

    for roi_text in args.roi:
        rois.append(parse_roi_text(roi_text, width, height))

    if args.polygon_file:
        polygon_path = Path(args.polygon_file)
        payload = json.loads(polygon_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Polygon file must contain a list")
        for item in payload:
            if isinstance(item, dict):
                points = item.get("points")
                if not isinstance(points, list):
                    raise ValueError("Polygon dict items must contain a 'points' list")
                rois.append(
                    PolygonROI(points=[clamp_point(point[0], point[1], width, height) for point in points])
                )
            elif isinstance(item, list):
                if len(item) == 4 and all(isinstance(value, (int, float)) for value in item):
                    rois.append(clamp_roi(item[0], item[1], item[2], item[3], width, height))
                else:
                    rois.append(
                        PolygonROI(points=[clamp_point(point[0], point[1], width, height) for point in item])
                    )
            else:
                raise ValueError("Polygon file items must be polygon point lists, polygon dicts, or legacy ROI boxes")

    if args.roi_file:
        roi_path = Path(args.roi_file)
        payload = json.loads(roi_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("ROI file must contain a list")
        for item in payload:
            if isinstance(item, dict):
                rois.append(clamp_roi(item["x1"], item["y1"], item["x2"], item["y2"], width, height))
            elif isinstance(item, list) and len(item) == 4:
                rois.append(clamp_roi(item[0], item[1], item[2], item[3], width, height))
            else:
                raise ValueError("ROI file items must be [x1,y1,x2,y2] or dicts with x1,y1,x2,y2")

    for roi in rois:
        if len(roi.points) < 3 or cv2.contourArea(np.array(roi.points, dtype=np.int32)) <= 0:
            raise ValueError(f"Invalid polygon points: {roi.points}")

    return rois


def create_roi_preview(image: np.ndarray, max_side: int, grid_step: int) -> np.ndarray:
    preview = image.copy()
    height, width = preview.shape[:2]

    if grid_step > 0:
        for x in range(0, width, grid_step):
            cv2.line(preview, (x, 0), (x, height - 1), (0, 255, 255), 2)
            cv2.putText(
                preview,
                str(x),
                (min(x + 8, width - 80), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for y in range(0, height, grid_step):
            cv2.line(preview, (0, y), (width - 1, y), (255, 255, 0), 2)
            cv2.putText(
                preview,
                str(y),
                (10, min(y + 30, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        preview = cv2.resize(preview, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return preview


def prompt_rois(width: int, height: int) -> list[PolygonROI]:
    print("Input one polygon per line in 'x1,y1 x2,y2 x3,y3 ...' format.")
    print("Press Enter on an empty line to finish.")
    print(f"Valid image range: x in [0, {width - 1}], y in [0, {height - 1}]")

    rois: list[PolygonROI] = []
    while True:
        prompt = f"Polygon #{len(rois) + 1}: "
        polygon_text = input(prompt).strip()
        if not polygon_text:
            break
        rois.append(parse_polygon_text(polygon_text, width, height))

    if not rois:
        raise RuntimeError("No polygon provided.")
    return rois


def select_rois(image: np.ndarray, max_side: int) -> list[PolygonROI]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))

    if scale < 1.0:
        preview = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = image.copy()

    window_name = "Select ROIs"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    print("Select multiple polygons as rectangles, then press ENTER. Press ESC when done.")
    rois = cv2.selectROIs(window_name, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    selected: list[PolygonROI] = []
    for x, y, w, h in rois:
        if w <= 0 or h <= 0:
            continue
        x1 = int(round(x / scale))
        y1 = int(round(y / scale))
        x2 = int(round((x + w) / scale))
        y2 = int(round((y + h) / scale))
        selected.append(clamp_roi(x1, y1, x2, y2, width, height))

    if not selected:
        raise RuntimeError("No region selected.")
    return selected


def get_rois(image: np.ndarray, args: argparse.Namespace, save_dir: Path) -> list[PolygonROI]:
    height, width = image.shape[:2]
    rois = load_rois_from_args(args, width, height)
    if rois:
        return rois

    preview_path = save_dir / f"{Path(args.source).stem}_roi_preview.jpg"
    preview = create_roi_preview(image, args.select_max_side, args.preview_grid)
    imwrite_unicode(preview_path, preview)
    print(f"Polygon preview saved to: {preview_path}")
    return prompt_rois(width, height)


def build_polygon_mask(roi: PolygonROI, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted = np.array([[point[0] - x1, point[1] - y1] for point in roi.points], dtype=np.int32)
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 255)
    return mask


def merge_patch_instance(
    global_map: np.ndarray,
    patch_mask: np.ndarray,
    x1: int,
    y1: int,
    union_find: UnionFind,
    kernel: np.ndarray | None,
) -> None:
    mask_bool = patch_mask > 0
    if not mask_bool.any():
        return

    y2 = y1 + patch_mask.shape[0]
    x2 = x1 + patch_mask.shape[1]
    target_region = global_map[y1:y2, x1:x2]

    expanded = cv2.dilate(patch_mask, kernel, iterations=1) > 0 if kernel is not None else mask_bool
    overlap_ids = np.unique(target_region[expanded])
    overlap_ids = [union_find.find(int(v)) for v in overlap_ids if v > 0]

    if overlap_ids:
        target_id = overlap_ids[0]
        for overlap_id in overlap_ids[1:]:
            target_id = union_find.union(target_id, overlap_id)
    else:
        target_id = union_find.new()

    target_region[mask_bool] = target_id


def remap_instances(global_map: np.ndarray, union_find: UnionFind) -> np.ndarray:
    remapped = np.zeros_like(global_map, dtype=np.int32)
    root_to_new: dict[int, int] = {}
    next_id = 1

    instance_ids = np.unique(global_map)
    instance_ids = instance_ids[instance_ids > 0]
    for instance_id in instance_ids:
        root = union_find.find(int(instance_id))
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        remapped[global_map == instance_id] = root_to_new[root]

    return remapped


def filter_instances(instance_map: np.ndarray, min_area: int) -> tuple[np.ndarray, list[InstanceRecord]]:
    filtered = np.zeros_like(instance_map, dtype=np.uint16)
    records: list[InstanceRecord] = []
    next_id = 1

    instance_ids, counts = np.unique(instance_map, return_counts=True)
    for instance_id, area in zip(instance_ids, counts):
        if instance_id == 0 or area < min_area:
            continue

        mask = instance_map == instance_id
        ys, xs = np.where(mask)
        filtered[mask] = next_id
        records.append(
            InstanceRecord(
                instance_id=next_id,
                area=int(area),
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            )
        )
        next_id += 1

    return filtered, records


def color_for_instance(instance_id: int) -> np.ndarray:
    rng = np.random.default_rng(instance_id)
    return rng.integers(0, 256, size=3, dtype=np.uint8)


def render_overlay(image: np.ndarray, instance_map: np.ndarray, rois: list[PolygonROI], alpha: float) -> np.ndarray:
    overlay = image.copy()
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    for instance_id in instance_ids:
        mask = instance_map == instance_id
        color = color_for_instance(int(instance_id)).astype(np.float32)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.astype(np.int32).tolist(), 2)

    for roi in rois:
        cv2.polylines(overlay, [roi.contour()], isClosed=True, color=(0, 255, 255), thickness=3)

    return overlay.astype(np.uint8)


def predict_selected_rois(
    model: YOLO,
    image: np.ndarray,
    rois: list[PolygonROI],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[InstanceRecord]]:
    global_map = np.zeros(image.shape[:2], dtype=np.int32)
    union_find = UnionFind()
    kernel = None

    if args.merge_kernel > 1:
        kernel_size = args.merge_kernel if args.merge_kernel % 2 == 1 else args.merge_kernel + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    total_tiles = 0
    for roi in rois:
        total_tiles += len(build_starts(roi.width, args.tile_size, args.overlap)) * len(build_starts(roi.height, args.tile_size, args.overlap))

    tile_index = 0
    for roi_index, roi in enumerate(rois, start=1):
        x_starts = build_starts(roi.width, args.tile_size, args.overlap)
        y_starts = build_starts(roi.height, args.tile_size, args.overlap)
        print(
            f"Polygon {roi_index}: bbox=({roi.x1}, {roi.y1}) -> ({roi.x2}, {roi.y2}), "
            f"vertices={len(roi.points)}, tiles={len(x_starts) * len(y_starts)}"
        )

        for dy in y_starts:
            for dx in x_starts:
                tile_index += 1
                x1 = roi.x1 + dx
                y1 = roi.y1 + dy
                x2 = min(x1 + args.tile_size, roi.x2)
                y2 = min(y1 + args.tile_size, roi.y2)
                polygon_mask = build_polygon_mask(roi, x1, y1, x2, y2)
                if not polygon_mask.any():
                    print(f"  tile {tile_index}/{total_tiles} at (x={x1}, y={y1}) -> skipped (outside polygon)")
                    continue

                patch = image[y1:y2, x1:x2].copy()
                patch[polygon_mask == 0] = 0

                predict_kwargs = {
                    "source": patch,
                    "imgsz": args.imgsz,
                    "conf": args.conf,
                    "iou": args.iou,
                    "retina_masks": True,
                    "max_det": args.max_det,
                    "verbose": False,
                }
                if args.device:
                    predict_kwargs["device"] = args.device

                result = model.predict(**predict_kwargs)[0]
                instance_count = len(result.boxes) if result.boxes is not None else 0
                print(f"  tile {tile_index}/{total_tiles} at (x={x1}, y={y1}) -> {instance_count} instances")

                if result.masks is None:
                    continue

                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    patch_mask = (mask > 0.5).astype(np.uint8)
                    patch_mask[polygon_mask == 0] = 0
                    merge_patch_instance(global_map, patch_mask, x1, y1, union_find, kernel)

    merged_map = remap_instances(global_map, union_find)
    return filter_instances(merged_map, args.min_area)


def save_outputs(
    image_path: Path,
    image: np.ndarray,
    rois: list[PolygonROI],
    instance_map: np.ndarray,
    records: list[InstanceRecord],
    save_dir: Path,
    alpha: float,
) -> None:
    stem = image_path.stem
    binary_mask = (instance_map > 0).astype(np.uint8) * 255
    overlay = render_overlay(image, instance_map, rois, alpha)
    roi_only = image.copy()
    roi_only[instance_map == 0] = 0

    overlay_path = save_dir / f"{stem}_overlay.jpg"
    binary_path = save_dir / f"{stem}_binary.png"
    instance_path = save_dir / f"{stem}_instance_map.png"
    roi_only_path = save_dir / f"{stem}_roi_only.png"
    json_path = save_dir / f"{stem}_result.json"

    imwrite_unicode(overlay_path, overlay)
    imwrite_unicode(binary_path, binary_mask)
    imwrite_unicode(instance_path, instance_map)
    imwrite_unicode(roi_only_path, roi_only)

    payload = {
        "image_path": str(image_path),
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "num_polygons": len(rois),
        "polygons": [asdict(roi) for roi in rois],
        "num_instances": len(records),
        "instances": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {overlay_path}")
    print(f"Saved: {binary_path}")
    print(f"Saved: {instance_path}")
    print(f"Saved: {roi_only_path}")
    print(f"Saved: {json_path}")


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.source)
    if not image_path.is_file():
        raise FileNotFoundError(f"source is not a valid image file: {args.source}")

    print("Assumption: the same crack instance overlaps or nearly touches in adjacent tiles, so masks can be merged by overlap.")
    image = imread_unicode(image_path)
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
    rois = get_rois(image, args, save_dir)
    print(f"Selected polygon count: {len(rois)}")
    model = YOLO(args.weights)
    instance_map, records = predict_selected_rois(model, image, rois, args)
    print(f"Final instance count: {len(records)}")
    save_outputs(image_path, image, rois, instance_map, records, save_dir, args.alpha)
    print(f"Output dir: {save_dir}")


if __name__ == "__main__":
    main()
