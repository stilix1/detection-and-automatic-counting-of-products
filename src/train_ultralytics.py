from __future__ import annotations

import argparse
from ultralytics import YOLO, RTDETR

from common import (
    load_config,
    model_setting,
    project_path,
    require_dataset,
    resolve_device,
    seed_everything,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/windows_quick.yaml")
    p.add_argument("--model", required=True, choices=["yolo11", "rtdetr"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    require_dataset(cfg)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    spec = cfg["models"][args.model]
    imgsz = int(model_setting(cfg, args.model, "input_size", 640))
    batch = int(model_setting(cfg, args.model, "batch_size", 1))
    workers = int(model_setting(cfg, args.model, "workers", 2))
    amp = bool(model_setting(cfg, args.model, "amp", device.type == "cuda"))
    runs_dir = project_path(cfg["paths"]["runs"])
    data_yaml = project_path(cfg["paths"]["yolo_yaml"])

    print(f"model={args.model}; device={device}; imgsz={imgsz}; batch={batch}; workers={workers}; amp={amp}")
    print(f"results={runs_dir}")

    model = YOLO(spec["weights"]) if args.model == "yolo11" else RTDETR(spec["weights"])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs or int(cfg["epochs"]),
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        seed=int(cfg["seed"]),
        project=str(runs_dir),
        name=args.model,
        pretrained=True,
        single_cls=True,
        device=str(device),
        cache=False,
        amp=amp,
        fraction=float(cfg.get("train_fraction", 1.0)),
        plots=True,
        resume=args.resume,
        patience=10,
    )


if __name__ == "__main__":
    main()
