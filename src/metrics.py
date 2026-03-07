"""
Metrics for object detection evaluation.
Computes mAP (mean Average Precision) at various IoU thresholds.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Args:
        box1, box2: Boxes in format [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_ap(
    predictions: List[Tuple],
    ground_truths: Dict,
    iou_threshold: float = 0.4,
) -> float:
    """
    Compute Average Precision for a single class.

    Args:
        predictions: List of (image_id, box, score)
        ground_truths: Dict mapping image_id to list of boxes
        iou_threshold: IoU threshold for matching

    Returns:
        AP value
    """
    if not predictions:
        return 0.0

    # Sort by score descending
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

    nd = len(predictions)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # Track matched GTs per image
    gt_matched = {
        img_id: np.zeros(len(boxes), dtype=bool)
        for img_id, boxes in ground_truths.items()
    }

    for i, (img_id, pred_box, score) in enumerate(predictions):
        if img_id not in ground_truths:
            fp[i] = 1
            continue

        gt_boxes = ground_truths[img_id]
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        # Find best IoU match
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            if not gt_matched[img_id][best_gt_idx]:
                tp[i] = 1
                gt_matched[img_id][best_gt_idx] = True
            else:
                fp[i] = 1  # Duplicate detection
        else:
            fp[i] = 1

    # Compute Precision/Recall
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)

    n_pos = sum(len(boxes) for boxes in ground_truths.values())
    if n_pos == 0:
        return 0.0

    recall = tp_cumsum / n_pos
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # PASCAL VOC 11-point interpolation AP
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recall >= t
        if np.any(mask):
            ap += np.max(precision[mask])

    return ap / 11.0


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.4,
    score_threshold: float = 0.0,
) -> float:
    """
    Compute mAP (mean Average Precision) over all classes.

    Args:
        predictions: List of dicts with keys: boxes, scores, labels
        targets: List of dicts with keys: boxes, labels
        iou_threshold: IoU threshold for matching
        score_threshold: Minimum score threshold (already filtered if using ensemble)

    Returns:
        mAP value
    """
    class_preds = defaultdict(list)
    class_gts = defaultdict(lambda: defaultdict(list))

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Collect predictions
        boxes = pred.get("boxes", [])
        scores = pred.get("scores", [])
        labels = pred.get("labels", [])

        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                class_preds[int(label)].append((img_idx, box, float(score)))

        # Collect ground truths
        gt_boxes = target.get("boxes", [])
        gt_labels = target.get("labels", [])

        for box, label in zip(gt_boxes, gt_labels):
            class_gts[int(label)][img_idx].append(box)

    # Compute AP for each class
    aps = []
    all_classes = set(class_gts.keys())

    for cls_id in all_classes:
        preds = class_preds[cls_id]
        gts = class_gts[cls_id]
        ap = compute_ap(preds, gts, iou_threshold)
        aps.append(ap)

    if not aps:
        return 0.0

    return float(np.mean(aps))


def compute_map_range(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_thresholds: List of IoU thresholds (default: [0.4, 0.5, 0.75])

    Returns:
        Dict mapping threshold name to mAP value
    """
    if iou_thresholds is None:
        iou_thresholds = [0.4, 0.5, 0.75]

    results = {}
    for iou_thr in iou_thresholds:
        mAP = compute_map(predictions, targets, iou_threshold=iou_thr)
        results[f"mAP@{iou_thr}"] = mAP

    # Average mAP
    results["mAP_avg"] = np.mean(list(results.values()))

    return results