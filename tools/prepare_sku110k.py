"""Prepare SKU-110K as both COCO and YOLO datasets.

Accepted input:
1) --annotations path/to/annotations.csv (the script creates an 80/10/10 split), or
2) --annotations path/to/annotations_folder containing train/val/test CSV files.

CSV may have a header or use the original SKU-110K eight-column layout:
image_name,x1,y1,x2,y2,class,image_width,image_height
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import os
from pathlib import Path

import pandas as pd
from PIL import Image

RAW_COLUMNS = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]
ALIASES = {
    "image_name": ["image_name", "image", "filename", "image_id"],
    "x1": ["x1", "xmin", "x_min"],
    "y1": ["y1", "ymin", "y_min"],
    "x2": ["x2", "xmax", "x_max"],
    "y2": ["y2", "ymax", "y_max"],
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not any(str(c).lower() in ALIASES["image_name"] for c in df.columns):
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 5:
            raise ValueError(f"В {path} меньше пяти столбцов")
        df.columns = RAW_COLUMNS[: df.shape[1]]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, variants in ALIASES.items():
        result[key] = next((name for name in variants if name in df.columns), "")
        if not result[key]:
            raise ValueError(f"Не найден столбец {key}. Есть: {list(df.columns)}")
    return result


def find_split_csvs(folder: Path) -> dict[str, Path]:
    csvs = list(folder.glob("*.csv"))
    found: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        aliases = (split, "validation" if split == "val" else split)
        match = next((p for p in csvs if any(a in p.stem.lower() for a in aliases)), None)
        if match:
            found[split] = match
    return found


def locate_image(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    for split in ("train", "val", "test", "images"):
        candidate = root / split / name
        if candidate.exists():
            return candidate
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Не найдено изображение: {name}")


def materialize(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def write_split(split: str, df: pd.DataFrame, images_root: Path, out: Path, mode: str) -> None:
    cols = resolve_columns(df)
    image_names = list(dict.fromkeys(df[cols["image_name"]].astype(str).tolist()))
    image_out = out / "images" / split
    label_out = out / "labels" / split
    ann_out = out / "annotations"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "product"}]}
    ann_id = 1
    grouped = df.groupby(cols["image_name"], sort=False)

    for image_id, name in enumerate(image_names, 1):
        source = locate_image(images_root, name)
        target = image_out / Path(name).name
        materialize(source, target, mode)
        with Image.open(source) as image:
            width, height = image.size
        coco["images"].append({
            "id": image_id,
            "file_name": target.name,
            "width": width,
            "height": height,
        })
        labels: list[str] = []
        rows = grouped.get_group(name) if name in grouped.groups else grouped.get_group(str(name))
        for _, row in rows.iterrows():
            x1, y1, x2, y2 = [float(row[cols[k]]) for k in ("x1", "y1", "x2", "y2")]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1
            labels.append(
                f"0 {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
                f"{bw / width:.8f} {bh / height:.8f}"
            )
        (label_out / f"{target.stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    (ann_out / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"{split}: images={len(image_names)}, boxes={len(coco['annotations'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Папка с изображениями SKU-110K")
    parser.add_argument("--annotations", required=True, help="CSV или папка с CSV")
    parser.add_argument("--output", default="data/sku110k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy",
                        help="copy безопаснее; hardlink экономит место, если исходники на том же диске")
    args = parser.parse_args()

    random.seed(args.seed)
    images_root = Path(args.images).expanduser().resolve()
    annotations = Path(args.annotations).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"Не найдена папка изображений: {images_root}")
    if not annotations.exists():
        raise FileNotFoundError(f"Не найдена разметка: {annotations}")

    if annotations.is_dir():
        split_files = find_split_csvs(annotations)
        if set(split_files) != {"train", "val", "test"}:
            raise FileNotFoundError(
                "В папке разметки должны быть CSV для train, val и test. "
                f"Найдено: {split_files}"
            )
        split_frames = {split: read_csv_auto(path) for split, path in split_files.items()}
    else:
        df = read_csv_auto(annotations)
        cols = resolve_columns(df)
        names = list(dict.fromkeys(df[cols["image_name"]].astype(str).tolist()))
        random.shuffle(names)
        n = len(names)
        split_names = {
            "train": set(names[: int(0.8 * n)]),
            "val": set(names[int(0.8 * n): int(0.9 * n)]),
            "test": set(names[int(0.9 * n):]),
        }
        split_frames = {
            split: df[df[cols["image_name"]].astype(str).isin(selected)].copy()
            for split, selected in split_names.items()
        }

    out.mkdir(parents=True, exist_ok=True)
    yaml_text = "\n".join([
        f"path: {out}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: product",
    ])
    (out / "sku110k.yaml").write_text(yaml_text, encoding="utf-8")

    for split in ("train", "val", "test"):
        write_split(split, split_frames[split], images_root, out, args.mode)

    print(f"\nГотово: {out}")


if __name__ == "__main__":
    main()
