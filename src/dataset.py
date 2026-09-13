from __future__ import annotations
from pathlib import Path
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    """Minimal COCO detection dataset for one or more classes."""
    def __init__(self, images_dir: str, annotation_file: str, train: bool = False):
        self.images_dir = Path(images_dir)
        self.train = train
        with open(annotation_file, encoding="utf-8") as f:
            coco = json.load(f)
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_path = self.images_dir / info["file_name"]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        anns = self.annotations.get(info["id"], [])
        boxes, labels, areas, crowds = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann.get("category_id", 1)))
            areas.append(float(ann.get("area", w * h)))
            crowds.append(int(ann.get("iscrowd", 0)))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        image = F.to_tensor(image)
        if self.train and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            width = image.shape[2]
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
