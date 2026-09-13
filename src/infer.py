from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torchvision.transforms import functional as F
from ultralytics import RTDETR, YOLO

from common import load_config, model_setting, project_path, resolve_device
from models import build_torchvision_model


def render_boxes(image, boxes, scores, count: int):
    out = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{float(score):.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    cv2.putText(out, f"Products: {count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--model", choices=["yolo11", "rtdetr", "fasterrcnn", "retinanet", "ssd"], default="yolo11")
    p.add_argument("--config", default="configs/windows_full.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output", default="demo/result.jpg")
    args = p.parse_args()

    weights = project_path(args.weights)
    source = project_path(args.source)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.model in {"yolo11", "rtdetr"}:
        model = YOLO(str(weights)) if args.model == "yolo11" else RTDETR(str(weights))
        result = model.predict(str(source), conf=args.conf, verbose=False)[0]
        rendered = result.plot()
        count = 0 if result.boxes is None else len(result.boxes)
        cv2.putText(rendered, f"Products: {count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
    else:
        cfg = load_config(args.config)
        device = resolve_device(cfg.get("device", "auto"))
        input_size = int(model_setting(cfg, args.model, "input_size", 640))
        architecture = cfg["models"][args.model]["name"]
        model = build_torchvision_model(architecture, int(cfg["num_classes"]), input_size).to(device)
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        image_bgr = cv2.imread(str(source))
        if image_bgr is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {source}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(image_rgb).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model([tensor])[0]
        keep = pred["scores"] >= args.conf
        boxes = pred["boxes"][keep].detach().cpu().numpy()
        scores = pred["scores"][keep].detach().cpu().numpy()
        count = len(boxes)
        rendered = render_boxes(image_bgr, boxes, scores, count)

    cv2.imwrite(str(output), rendered)
    print(f"products={count}; saved={output}")


if __name__ == "__main__":
    main()
