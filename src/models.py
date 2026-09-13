from __future__ import annotations

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_torchvision_model(name: str, num_classes: int, input_size: int = 640):
    """Build detectors. num_classes is foreground classes; background is added here."""
    total_classes = int(num_classes) + 1

    if name == "fasterrcnn_resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            min_size=input_size,
            max_size=input_size,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, total_classes)
        return model

    if name == "retinanet_resnet50_fpn_v2":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            min_size=input_size,
            max_size=input_size,
        )
        old = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            old.conv[0][0].in_channels,
            old.num_anchors,
            total_classes,
            norm_layer=torch.nn.BatchNorm2d,
        )
        return model

    if name == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=total_classes,
        )

    raise ValueError(f"Unknown model: {name}")
