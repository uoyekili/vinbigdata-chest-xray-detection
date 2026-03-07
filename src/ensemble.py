"""
Ensemble utilities using Weighted Boxes Fusion (WBF).
Based on ideas from ZFTurbo's 2nd place solution.
"""

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from typing import List, Dict, Tuple

from src import config


def normalize_boxes(boxes: np.ndarray, image_size: int) -> np.ndarray:
    """Normalize boxes to [0, 1] range."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes.astype(float)
    boxes[:, [0, 2]] /= image_size
    boxes[:, [1, 3]] /= image_size
    return np.clip(boxes, 0, 1)


def denormalize_boxes(boxes: np.ndarray, image_size: int) -> np.ndarray:
    """Denormalize boxes from [0, 1] to pixel coordinates."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes.astype(float)
    boxes[:, [0, 2]] *= image_size
    boxes[:, [1, 3]] *= image_size
    return boxes


def ensemble_predictions(
    predictions_list: List[Dict],
    weights: List[float] = None,
    iou_thr: float = config.WBF_IOU_THR,
    skip_box_thr: float = config.WBF_SKIP_BOX_THR,
    image_size: int = config.IMAGE_SIZE,
) -> Dict:
    """
    Ensemble predictions from multiple models using Weighted Boxes Fusion.
    
    Args:
        predictions_list: List of prediction dicts, each containing:
            - boxes: (N, 4) array in pixel coordinates
            - scores: (N,) array
            - labels: (N,) array
        weights: Weight for each model (default: equal weights)
        iou_thr: IoU threshold for WBF
        skip_box_thr: Minimum score threshold
        image_size: Image size for normalization
    
    Returns:
        Dict with ensembled boxes, scores, labels
    """
    if not predictions_list:
        return {"boxes": np.array([]), "scores": np.array([]), "labels": np.array([])}
    
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    # Prepare boxes for WBF (normalized to [0, 1])
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for pred in predictions_list:
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        
        if len(boxes) > 0:
            norm_boxes = normalize_boxes(boxes.copy(), image_size)
            boxes_list.append(norm_boxes.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(labels.tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
    
    # Check if all empty
    if all(len(b) == 0 for b in boxes_list):
        return {"boxes": np.array([]), "scores": np.array([]), "labels": np.array([])}
    
    # Apply WBF
    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    
    # Denormalize boxes back to pixel coordinates
    merged_boxes = denormalize_boxes(merged_boxes, image_size)
    
    return {
        "boxes": merged_boxes,
        "scores": merged_scores,
        "labels": merged_labels.astype(int),
    }


def apply_tta(model, image: torch.Tensor, device: torch.device) -> List[Dict]:
    """
    Apply Test Time Augmentation (horizontal flip).
    
    Args:
        model: Detection model
        image: (C, H, W) tensor
        device: torch device
    
    Returns:
        List of predictions (original + flipped)
    """
    model.eval()
    predictions = []
    img_w = image.shape[2]
    
    # Original prediction
    with torch.no_grad():
        output = model([image.to(device)])[0]
    
    predictions.append({
        "boxes": output["boxes"].cpu().numpy(),
        "scores": output["scores"].cpu().numpy(),
        "labels": output["labels"].cpu().numpy(),
    })
    
    # Horizontal flip
    img_flipped = torch.flip(image, dims=[2])
    with torch.no_grad():
        output_flip = model([img_flipped.to(device)])[0]
    
    boxes_flip = output_flip["boxes"].cpu().numpy()
    if len(boxes_flip) > 0:
        # Flip boxes back
        boxes_flip[:, [0, 2]] = img_w - boxes_flip[:, [2, 0]]
    
    predictions.append({
        "boxes": boxes_flip,
        "scores": output_flip["scores"].cpu().numpy(),
        "labels": output_flip["labels"].cpu().numpy(),
    })
    
    return predictions


def predict_single_model(
    model,
    image: torch.Tensor,
    device: torch.device,
    use_tta: bool = True,
) -> Dict:
    """
    Get prediction from a single model (with optional TTA).
    
    Args:
        model: Detection model
        image: (C, H, W) tensor
        device: torch device
        use_tta: Whether to use Test Time Augmentation
    
    Returns:
        Dict with boxes, scores, labels
    """
    if use_tta:
        predictions = apply_tta(model, image, device)
        return ensemble_predictions(predictions, image_size=image.shape[1])
    else:
        model.eval()
        with torch.no_grad():
            output = model([image.to(device)])[0]
        return {
            "boxes": output["boxes"].cpu().numpy(),
            "scores": output["scores"].cpu().numpy(),
            "labels": output["labels"].cpu().numpy(),
        }


def ensemble_multi_model(
    models: List,
    image: torch.Tensor,
    device: torch.device,
    weights: List[float] = None,
    use_tta: bool = True,
) -> Dict:
    """
    Ensemble predictions from multiple models.
    
    Args:
        models: List of detection models
        image: (C, H, W) tensor
        device: torch device
        weights: Weight for each model
        use_tta: Whether to use TTA for each model
    
    Returns:
        Dict with ensembled boxes, scores, labels
    """
    all_predictions = []
    
    for model in models:
        pred = predict_single_model(model, image, device, use_tta=use_tta)
        all_predictions.append(pred)
    
    return ensemble_predictions(all_predictions, weights=weights, image_size=image.shape[1])


def filter_predictions(
    predictions: Dict,
    conf_threshold: float = config.FINAL_CONF_THRESHOLD,
) -> Dict:
    """Filter predictions by confidence threshold."""
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    if len(scores) == 0:
        return predictions
    
    mask = scores >= conf_threshold
    return {
        "boxes": boxes[mask],
        "scores": scores[mask],
        "labels": labels[mask],
    }


def format_prediction_string(
    predictions: Dict,
    no_finding_label: int = 14,
) -> str:
    """
    Format predictions to submission string format.
    
    Format: "label score x1 y1 x2 y2 label score x1 y1 x2 y2 ..."
    If no predictions, return "14 1.0 0 0 1 1" (No finding)
    """
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    
    if len(boxes) == 0:
        return f"{no_finding_label} 1.0 0 0 1 1"
    
    result_parts = []
    for box, score, label in zip(boxes, scores, labels):
        # Convert label back: 1-14 -> 0-13
        class_id = int(label) - 1 if label > 0 else 0
        result_parts.append(
            f"{class_id} {score:.4f} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}"
        )
    
    return " ".join(result_parts)
