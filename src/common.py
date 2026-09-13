from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def model_setting(cfg: dict[str, Any], model_name: str, key: str, default: Any = None) -> Any:
    spec = cfg.get("models", {}).get(model_name, {})
    if key in spec:
        return spec[key]
    return cfg.get(key, default)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str = "auto") -> torch.device:
    if value != "auto":
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. Запусти 00_CHECK_PC.bat и проверь драйвер NVIDIA.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def require_dataset(cfg: dict[str, Any]) -> None:
    dataset_root = project_path(cfg["paths"]["dataset_root"])
    yolo_yaml = project_path(cfg["paths"]["yolo_yaml"])
    # A prepared dataset may be copied from macOS. Rewrite its absolute root path for this PC.
    if yolo_yaml.exists():
        data = yaml.safe_load(yolo_yaml.read_text(encoding="utf-8")) or {}
        expected = str(dataset_root)
        if str(data.get("path", "")) != expected:
            data["path"] = expected
            ordered = {
                "path": data["path"],
                "train": data.get("train", "images/train"),
                "val": data.get("val", "images/val"),
                "test": data.get("test", "images/test"),
                "names": data.get("names", {0: "product"}),
            }
            yolo_yaml.write_text(yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"Исправлен путь датасета в {yolo_yaml}: {expected}")
    required = [
        dataset_root,
        yolo_yaml,
        project_path(cfg["paths"]["coco_train"]),
        project_path(cfg["paths"]["coco_test"]),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Датасет не подготовлен. Не найдены:\n- " + "\n- ".join(missing)
            + "\nЗапусти START_WINDOWS.bat и выбери подготовку датасета."
        )


def ensure_dir(path: str | Path) -> Path:
    p = project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def timed_call(fn, device: torch.device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start
