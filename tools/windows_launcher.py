from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
QUICK = ROOT / "configs" / "windows_quick.yaml"
FULL = ROOT / "configs" / "windows_full.yaml"
MODELS = {
    "1": ("yolo11", "YOLO11"),
    "2": ("rtdetr", "RT-DETR-L"),
    "3": ("fasterrcnn", "Faster R-CNN"),
    "4": ("retinanet", "RetinaNet"),
    "5": ("ssd", "SSDLite"),
}


def clean_path(value: str) -> str:
    return value.strip().strip('"').strip("'")


def run(*parts: str):
    cmd = [str(PYTHON), *map(str, parts)]
    print("\nЗАПУСК:", " ".join(f'"{x}"' if " " in x else x for x in cmd), "\n")
    subprocess.run(cmd, cwd=ROOT, check=False)
    input("\nНажми Enter, чтобы вернуться в меню...")


def dataset_ready() -> bool:
    return (ROOT / "data" / "sku110k" / "sku110k.yaml").exists()


def choose_model():
    for key, (_, title) in MODELS.items():
        print(f"{key} — {title}")
    choice = input("Номер модели: ").strip()
    return MODELS.get(choice)


def prepare():
    print("\nМожно перетащить папку или файл прямо в это окно.")
    images = clean_path(input("Путь к папке SKU110K\\images: "))
    annotations = clean_path(input("Путь к папке annotations или CSV: "))
    mode = input("Режим: 1 — копировать (безопасно), 2 — hardlink (экономит место): ").strip()
    mode = "hardlink" if mode == "2" else "copy"
    run("tools/prepare_sku110k.py", "--images", images, "--annotations", annotations,
        "--output", "data/sku110k", "--mode", mode)


def train(config: Path):
    selected = choose_model()
    if not selected:
        print("Неверный номер."); return
    model, _ = selected
    if not dataset_ready():
        print("\nСначала подготовь датасет: пункт 2.")
        input("Enter..."); return
    script = "src/train_ultralytics.py" if model in {"yolo11", "rtdetr"} else "src/train_torchvision.py"
    run(script, "--config", str(config), "--model", model)


def latest_weight(model: str):
    candidates = list(ROOT.glob(f"runs_windows_*/**/{model}*/**/best.pt"))
    candidates += list(ROOT.glob(f"runs_windows_*/{model}/best.pt"))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def ask_weight(model: str):
    latest = latest_weight(model)
    if latest:
        answer = input(f"Enter — взять последние веса:\n{latest}\nИли перетащи другой best.pt: ").strip()
        return Path(clean_path(answer)) if answer else latest
    return Path(clean_path(input("Перетащи файл best.pt: ")))


def choose_config(weight: Path):
    return QUICK if "quick" in str(weight).lower() else FULL


def evaluate():
    selected = choose_model()
    if not selected: return
    model, _ = selected
    weight = ask_weight(model)
    if not weight.exists():
        print("Файл весов не найден:", weight); input("Enter..."); return
    limit = input("Сколько test-изображений проверить? Enter или 0 = все; для пробы 100: ").strip() or "0"
    run("src/evaluate_model.py", "--model", model, "--weights", str(weight),
        "--config", str(choose_config(weight)), "--limit", limit)


def infer():
    selected = choose_model()
    if not selected: return
    model, _ = selected
    weight = ask_weight(model)
    image = Path(clean_path(input("Перетащи фотографию полки: ")))
    if not weight.exists() or not image.exists():
        print("Не найден файл весов или изображения."); input("Enter..."); return
    output = ROOT / "demo" / f"result_{model}.jpg"
    run("src/infer.py", "--model", model, "--weights", str(weight),
        "--source", str(image), "--output", str(output),
        "--config", str(choose_config(weight)))
    if output.exists():
        os.startfile(output)


def open_results():
    for path in [ROOT / "runs_windows_quick", ROOT / "runs_windows_full", ROOT / "report", ROOT / "demo"]:
        path.mkdir(parents=True, exist_ok=True)
    os.startfile(ROOT)


def main():
    os.chdir(ROOT)
    while True:
        print("\n" + "=" * 66)
        print(" ПОДСЧЕТ ТОВАРОВ НА ПОЛКЕ")
        print("=" * 66)
        print("1 — проверить CUDA и видеокарту")
        print("2 — подготовить SKU-110K")
        #print("3 — быстрый тест одной модели (2 эпохи)")
        print("3 — полное обучение одной модели (3 эпохи)")
        print("4 — оценить модель и построить карту ошибок")
        print("5 — посчитать товары на своей фотографии")
        print("6 — открыть папку результатов")
        print("0 — выход")
        choice = input("Выбор: ").strip()
        if choice == "1": run("tools/check_environment.py")
        elif choice == "2": prepare()
        #elif choice == "3": train(QUICK)
        elif choice == "3": train(FULL)
        elif choice == "4": evaluate()
        elif choice == "5": infer()
        elif choice == "6": open_results()
        elif choice == "0": return
        else: print("Неверный пункт.")


if __name__ == "__main__":
    main()
