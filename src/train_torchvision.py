from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import (
    empty_device_cache,
    ensure_dir,
    load_config,
    model_setting,
    project_path,
    require_dataset,
    resolve_device,
    seed_everything,
)
from dataset import CocoDetectionDataset, collate_fn
from models import build_torchvision_model


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int, use_amp: bool) -> float:
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"epoch {epoch}")
    for images, targets in bar:
        images = [x.to(device, non_blocking=True) for x in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().item())
        total += value
        bar.set_postfix(loss=f"{value:.4f}")
    return total / max(len(loader), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/windows_quick.yaml")
    p.add_argument("--model", required=True, choices=["fasterrcnn", "retinanet", "ssd"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    require_dataset(cfg)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    input_size = int(model_setting(cfg, args.model, "input_size", 640))
    batch_size = int(model_setting(cfg, args.model, "batch_size", 1))
    workers = int(model_setting(cfg, args.model, "workers", 2))
    use_amp = device.type == "cuda" and bool(model_setting(cfg, args.model, "amp", True))

    images_dir = project_path(cfg["paths"]["dataset_root"]) / "images" / "train"
    ds = CocoDetectionDataset(str(images_dir), str(project_path(cfg["paths"]["coco_train"])), train=True)
    max_images = cfg.get("max_train_images")
    if max_images:
        ds = Subset(ds, range(min(int(max_images), len(ds))))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    name = cfg["models"][args.model]["name"]
    print(
        f"model={args.model}; device={device}; images={len(ds)}; input={input_size}; "
        f"batch={batch_size}; workers={workers}; amp={use_amp}"
    )
    model = build_torchvision_model(name, int(cfg["num_classes"]), input_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    total_epochs = args.epochs or int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = make_scaler(use_amp)
    out = ensure_dir(Path(cfg["paths"]["runs"]) / args.model)

    start_epoch = 1
    best = float("inf")
    last_path = out / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best = float(checkpoint.get("best_loss", checkpoint.get("loss", best)))
        print(f"Продолжаю с эпохи {start_epoch}")

    for epoch in range(start_epoch, total_epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scaler, device, epoch, use_amp)
        scheduler.step()
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "best_loss": min(best, loss),
            "model_key": args.model,
            "architecture": name,
            "input_size": input_size,
            "num_classes": int(cfg["num_classes"]),
            "config": str(project_path(args.config)),
        }
        torch.save(payload, last_path)
        if loss < best:
            best = loss
            payload["best_loss"] = best
            torch.save(payload, out / "best.pt")
        empty_device_cache(device)
        print(f"epoch={epoch} mean_loss={loss:.5f}; best={best:.5f}; saved={out}")


if __name__ == "__main__":
    main()
