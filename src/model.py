import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src import config


def build_fasterrcnn(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = config.IMAGE_SIZE,
    max_size: int = config.IMAGE_SIZE,
    version: str = "v1",
) -> FasterRCNN:
    """Build Faster R-CNN model with ResNet-50 FPN backbone."""
    if version == "v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def build_retinanet(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = config.IMAGE_SIZE,
    max_size: int = config.IMAGE_SIZE,
    version: str = "v1",
) -> RetinaNet:
    """Build RetinaNet model with ResNet-50 FPN backbone."""
    if version == "v2":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = retinanet_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = retinanet_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace the classification head
    # RetinaNet head needs: in_channels, num_anchors, num_classes
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )

    return model


def build_model(
    model_name: str,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """Build detection model by name."""

    if model_name == "fasterrcnn":
        return build_fasterrcnn(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif model_name == "fasterrcnn_v2":
        return build_fasterrcnn(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            version="v2",
        )
    elif model_name == "retinanet":
        return build_retinanet(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif model_name == "retinanet_v2":
        return build_retinanet(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            version="v2",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: fasterrcnn, fasterrcnn_v2, retinanet, retinanet_v2")


def load_single_model(checkpoint_path: str, model_name: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    model = build_model(model_name=model_name, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def load_models(model_names, fold, device):
    """Load models."""
    models = []
    for name in model_names:
        checkpoint_path = config.get_checkpoint_path(name, fold)
        model = load_single_model(checkpoint_path, name, device)
        models.append(model)
    return models


def get_model_info(model: nn.Module) -> dict:
    """Return model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


if __name__ == "__main__":
    print("Testing model builds...\n")

    for model_name in ["fasterrcnn", "retinanet"]:
        print(f"=== {model_name.upper()} ===")
        model = build_model(model_name=model_name)
        info = get_model_info(model)

        print(f"Total params:     {info['total_params']:,}")
        print(f"Trainable params: {info['trainable_params']:,}")

        # Test forward pass
        model.eval()
        dummy_image = [torch.rand(3, 512, 512)]
        with torch.no_grad():
            output = model(dummy_image)

        print(f"Output keys: {list(output[0].keys())}")
        print()
