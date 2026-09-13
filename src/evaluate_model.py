from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from ultralytics import RTDETR, YOLO

from common import (
    load_config,
    model_setting,
    project_path,
    require_dataset,
    resolve_device,
    save_json,
    synchronize,
)
from count_metrics import counting_metrics
from dataset import CocoDetectionDataset, collate_fn
from matching import match_boxes
from models import build_torchvision_model
from visualize_errors import draw_error_map

MODELS = ["yolo11", "rtdetr", "fasterrcnn", "retinanet", "ssd"]
MAX_DETECTIONS_PER_IMAGE = 1000


def load_coco(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    anns: dict[int, list[list[float]]] = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        anns.setdefault(int(ann["image_id"]), []).append([x, y, x + w, y + h])
    images = sorted(data["images"], key=lambda x: x["id"])
    return images, anns


def append_summary(path: Path, row: dict) -> None:
    """Обновляет строку модели в CSV вместо бесконечного добавления дублей."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "config",
        "images",
        "input_size",
        "precision_iou50",
        "recall_iou50",
        "f1_iou50",
        "mAP50",
        "mAP50_95",
        "count_MAE",
        "count_RMSE",
        "count_bias",
        "count_accuracy_within_5pct",
        "mean_relative_count_error",
        "dense_count_MAE",
        "latency_ms",
        "FPS",
        "max_detections_per_image",
        "tp",
        "fp",
        "fn",
        "weights",
    ]

    existing_rows: list[dict] = []
    if path.exists():
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                existing_rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            existing_rows = []

    # Оставляем результаты других моделей. Текущую модель перезаписываем.
    existing_rows = [r for r in existing_rows if r.get("model") != str(row.get("model"))]
    existing_rows.append({k: row.get(k, "") for k in fields})

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in existing_rows])


def basic_update(gt, pred, totals) -> None:
    matches, misses, false_pos = match_boxes(
        np.asarray(gt, dtype=float).reshape(-1, 4),
        np.asarray(pred, dtype=float).reshape(-1, 4),
        0.5,
    )
    totals["tp"] += len(matches)
    totals["fn"] += len(misses)
    totals["fp"] += len(false_pos)


def save_worst(worst, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("error_*"):
        if old.is_file():
            old.unlink()

    for rank, item in enumerate(sorted(worst, key=lambda x: x[0], reverse=True), 1):
        _, image_path, gt, pred = item
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rendered, _ = draw_error_map(image, gt, pred, 0.5)
        cv2.imwrite(str(out_dir / f"error_{rank}_{image_path.name}"), rendered)


def xyxy_to_coco_predictions(
    image_id: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
) -> list[dict]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return []

    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    valid &= boxes[:, 2] > boxes[:, 0]
    valid &= boxes[:, 3] > boxes[:, 1]
    boxes = boxes[valid]
    scores = scores[valid]

    order = np.argsort(-scores)[:max_detections]
    boxes = boxes[order]
    scores = scores[order]

    result: list[dict] = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        result.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return result


def compute_dense_coco_map(
    coco_path: Path,
    predictions: list[dict],
    image_ids: list[int],
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
) -> tuple[float, float]:
    """
    Считает AP напрямую из массива precision COCOeval.

    COCOeval.summarize() жёстко ожидает maxDets=100 и возвращает -1 для
    общего mAP, если задать [1, 10, 1000]. Здесь summarize() не используется,
    поэтому AP корректно считается при 1000 детекций на изображение.
    """
    if not predictions:
        return 0.0, 0.0

    # pycocotools много печатает в stdout — скрываем служебный вывод.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(coco_path))
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [int(x) for x in image_ids]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, int(max_detections)]
        evaluator.evaluate()
        evaluator.accumulate()

    # precision: [IoU thresholds, recall thresholds, classes, areas, maxDets]
    precision = evaluator.eval["precision"]
    all_area = 0
    one_class = 0
    max_det_index = len(evaluator.params.maxDets) - 1

    p_all = precision[:, :, one_class, all_area, max_det_index]
    valid_all = p_all[p_all > -1]
    map_50_95 = float(valid_all.mean()) if valid_all.size else 0.0

    iou_thresholds = np.asarray(evaluator.params.iouThrs)
    iou50_index = int(np.argmin(np.abs(iou_thresholds - 0.50)))
    p50 = precision[iou50_index, :, one_class, all_area, max_det_index]
    valid_50 = p50[p50 > -1]
    map_50 = float(valid_50.mean()) if valid_50.size else 0.0

    return map_50, map_50_95


def configure_torchvision_detection_limit(model, max_detections: int) -> None:
    """Поднимает стандартный лимит 100 детекций у torchvision-моделей."""
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "detections_per_img"):
        model.roi_heads.detections_per_img = max_detections
    if hasattr(model, "detections_per_img"):
        model.detections_per_img = max_detections
    if hasattr(model, "topk_candidates"):
        model.topk_candidates = max(int(model.topk_candidates), max_detections)


def evaluate_ultralytics(args, cfg, device, images, gt_by_id, coco_path: Path):
    weights = project_path(args.weights)
    model = YOLO(str(weights)) if args.model == "yolo11" else RTDETR(str(weights))
    input_size = int(model_setting(cfg, args.model, "input_size", 640))
    test_root = project_path(cfg["paths"]["dataset_root"]) / "images" / "test"
    selected = images[: args.limit] if args.limit else images
    paths = [str(test_root / x["file_name"]) for x in selected]

    gt_counts, pred_counts, dense_gt, dense_pred = [], [], [], []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    latencies: list[float] = []
    worst = []
    coco_predictions: list[dict] = []
    selected_ids = [int(x["id"]) for x in selected]

    if paths:
        # Прогрев CUDA, чтобы первый запуск не портил среднюю задержку.
        model.predict(
            paths[0],
            imgsz=input_size,
            conf=0.001,
            iou=0.7,
            max_det=MAX_DETECTIONS_PER_IMAGE,
            device=str(device),
            verbose=False,
        )

    predictions = model.predict(
        paths,
        imgsz=input_size,
        conf=0.001,
        iou=0.7,
        max_det=MAX_DETECTIONS_PER_IMAGE,
        device=str(device),
        stream=True,
        verbose=False,
    )

    for info, result in zip(selected, predictions):
        gt = gt_by_id.get(int(info["id"]), [])
        boxes_all = [] if result.boxes is None else result.boxes.xyxy.detach().cpu().numpy()
        scores_all = [] if result.boxes is None else result.boxes.conf.detach().cpu().numpy()
        boxes_array = np.asarray(boxes_all, dtype=np.float32).reshape(-1, 4)
        scores_array = np.asarray(scores_all, dtype=np.float32).reshape(-1)

        coco_predictions.extend(
            xyxy_to_coco_predictions(int(info["id"]), boxes_array, scores_array)
        )

        keep = scores_array >= float(cfg["confidence_threshold"])
        pred = boxes_array[keep].reshape(-1, 4)
        gt_counts.append(len(gt))
        pred_counts.append(len(pred))
        if len(gt) >= int(cfg["dense_scene_threshold"]):
            dense_gt.append(len(gt))
            dense_pred.append(len(pred))

        basic_update(gt, pred, totals)

        speed = result.speed or {}
        # Реальная обработка модели: preprocessing + inference + postprocessing.
        latency = sum(float(speed.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess"))
        if latency > 0:
            latencies.append(latency)

        error = abs(len(gt) - len(pred))
        worst.append((error, Path(result.path), gt, pred))
        worst = sorted(worst, key=lambda x: x[0], reverse=True)[:3]

    map50, map5095 = compute_dense_coco_map(
        coco_path,
        coco_predictions,
        selected_ids,
        MAX_DETECTIONS_PER_IMAGE,
    )
    return gt_counts, pred_counts, dense_gt, dense_pred, totals, latencies, worst, map50, map5095


def evaluate_torchvision(args, cfg, device, coco_path: Path):
    input_size = int(model_setting(cfg, args.model, "input_size", 640))
    architecture = cfg["models"][args.model]["name"]
    model = build_torchvision_model(architecture, int(cfg["num_classes"]), input_size).to(device)
    checkpoint = torch.load(project_path(args.weights), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    configure_torchvision_detection_limit(model, MAX_DETECTIONS_PER_IMAGE)
    model.eval()

    images_dir = project_path(cfg["paths"]["dataset_root"]) / "images" / "test"
    ds = CocoDetectionDataset(str(images_dir), str(coco_path), train=False)
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    gt_counts, pred_counts, dense_gt, dense_pred = [], [], [], []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    latencies: list[float] = []
    worst = []
    coco_predictions: list[dict] = []
    selected_ids: list[int] = []
    base_ds = ds.dataset if isinstance(ds, Subset) else ds
    warmed_up = False

    for idx, (batch_images, targets) in enumerate(loader):
        image = batch_images[0].to(device, non_blocking=True)

        if not warmed_up:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                _ = model([image])[0]
            synchronize(device)
            warmed_up = True

        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model([image])[0]
        synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)

        target = targets[0]
        labels = output["labels"].detach().cpu()
        one_class = labels == 1
        boxes_all = output["boxes"].detach().cpu()[one_class].numpy()
        scores_all = output["scores"].detach().cpu()[one_class].numpy()

        actual_idx = ds.indices[idx] if isinstance(ds, Subset) else idx
        image_info = base_ds.images[actual_idx]
        image_id = int(image_info["id"])
        selected_ids.append(image_id)
        coco_predictions.extend(
            xyxy_to_coco_predictions(image_id, boxes_all, scores_all)
        )

        keep = scores_all >= float(cfg["confidence_threshold"])
        pred_boxes = np.asarray(boxes_all, dtype=np.float32).reshape(-1, 4)[keep]
        gt_boxes = target["boxes"].cpu().numpy()

        gt_counts.append(len(gt_boxes))
        pred_counts.append(len(pred_boxes))
        if len(gt_boxes) >= int(cfg["dense_scene_threshold"]):
            dense_gt.append(len(gt_boxes))
            dense_pred.append(len(pred_boxes))

        basic_update(gt_boxes, pred_boxes, totals)

        image_path = base_ds.images_dir / image_info["file_name"]
        error = abs(len(gt_boxes) - len(pred_boxes))
        worst.append((error, image_path, gt_boxes, pred_boxes))
        worst = sorted(worst, key=lambda x: x[0], reverse=True)[:3]

    map50, map5095 = compute_dense_coco_map(
        coco_path,
        coco_predictions,
        selected_ids,
        MAX_DETECTIONS_PER_IMAGE,
    )
    return gt_counts, pred_counts, dense_gt, dense_pred, totals, latencies, worst, map50, map5095


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="configs/windows_full.yaml")
    parser.add_argument("--limit", type=int, default=0, help="0 = весь test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_dataset(cfg)
    device = resolve_device(cfg.get("device", "auto"))
    coco_path = project_path(cfg["paths"]["coco_test"])
    images, gt_by_id = load_coco(coco_path)

    if args.model in {"yolo11", "rtdetr"}:
        values = evaluate_ultralytics(args, cfg, device, images, gt_by_id, coco_path)
    else:
        values = evaluate_torchvision(args, cfg, device, coco_path)

    gt_counts, pred_counts, dense_gt, dense_pred, totals, latencies, worst, map50, map5095 = values

    counts = counting_metrics(gt_counts, pred_counts)
    dense = counting_metrics(dense_gt, dense_pred) if dense_gt else {"count_MAE": float("nan")}
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    latency = float(np.mean(latencies)) if latencies else float("nan")

    result = {
        "model": args.model,
        "config": str(project_path(args.config)),
        "images": len(gt_counts),
        "input_size": int(model_setting(cfg, args.model, "input_size", 640)),
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "mAP50": map50,
        "mAP50_95": map5095,
        **counts,
        "dense_count_MAE": dense["count_MAE"],
        "latency_ms": latency,
        "FPS": 1000.0 / latency if latency > 0 else float("nan"),
        "weights": str(project_path(args.weights)),
        "device": str(device),
        "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
    }

    out_dir = project_path(cfg["paths"]["runs"]) / "evaluation" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, out_dir / "metrics.json")
    save_worst(worst, out_dir / "errors")
    append_summary(project_path("report/experiment_results.csv"), result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Ошибки: {out_dir / 'errors'}")
    print(f"Общая таблица: {project_path('report/experiment_results.csv')}")


if __name__ == "__main__":
    main()
