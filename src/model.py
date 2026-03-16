import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src import config


def build_fasterrcnn(
    num_classes=config.NUM_CLASSES, pretrained=True, trainable_layers=3
):
    try:
        weights = "DEFAULT" if pretrained else None
        kwargs = {
            "weights": weights,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        # Only set trainable_backbone_layers when weights are loaded
        if weights is not None:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = fasterrcnn_resnet50_fpn(**kwargs)
    except:
        kwargs = {
            "pretrained": pretrained,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if pretrained:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = fasterrcnn_resnet50_fpn(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_fasterrcnn_v2(
    num_classes=config.NUM_CLASSES, pretrained=True, trainable_layers=3
):
    try:
        weights = "DEFAULT" if pretrained else None
        kwargs = {
            "weights": weights,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if weights is not None:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = fasterrcnn_resnet50_fpn_v2(**kwargs)
    except:
        kwargs = {
            "pretrained": pretrained,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if pretrained:
            kwargs["trainable_backbone_layers"] = 4
        model = fasterrcnn_resnet50_fpn(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_retinanet(
    num_classes=config.NUM_CLASSES, pretrained=True, trainable_layers=3
):
    try:
        weights = "DEFAULT" if pretrained else None
        kwargs = {
            "weights": weights,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if weights is not None:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = retinanet_resnet50_fpn(**kwargs)
    except:
        kwargs = {
            "pretrained": pretrained,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if pretrained:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = retinanet_resnet50_fpn(**kwargs)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )
    return model


def build_retinanet_v2(
    num_classes=config.NUM_CLASSES, pretrained=True, trainable_layers=4
):
    try:
        weights = "DEFAULT" if pretrained else None
        kwargs = {
            "weights": weights,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if weights is not None:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = retinanet_resnet50_fpn_v2(**kwargs)
    except:
        kwargs = {
            "pretrained": pretrained,
            "min_size": config.IMAGE_SIZE,
            "max_size": config.IMAGE_SIZE,
        }
        if pretrained:
            kwargs["trainable_backbone_layers"] = trainable_layers
        model = retinanet_resnet50_fpn_v2(**kwargs)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )
    return model


def build_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True):
    if model_name == "fasterrcnn":
        return build_fasterrcnn(num_classes, pretrained, trainable_layers=3)
    elif model_name == "fasterrcnnv2":
        return build_fasterrcnn_v2(num_classes, pretrained, trainable_layers=3)
    elif model_name == "retinanet":
        return build_retinanet(num_classes, pretrained, trainable_layers=3)
    elif model_name == "retinanetv2":
        return build_retinanet_v2(num_classes, pretrained, trainable_layers=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(path, model_name, device):
    model = build_model(model_name, pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
