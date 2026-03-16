import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.logger import get_logger


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def match_predictions(
    pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
):
    tp_idx = []
    fp_idx = []
    fn_idx = []
    matched_gt = set()

    sorted_idx = np.argsort(-pred_scores)

    for p_idx in sorted_idx:
        best_iou = 0
        best_g_idx = -1

        for g_idx, gt_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue

            iou = compute_iou(pred_boxes[p_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx

        if best_iou >= iou_threshold and best_g_idx != -1:
            if pred_labels[p_idx] == gt_labels[best_g_idx]:
                tp_idx.append(p_idx)
                matched_gt.add(best_g_idx)
            else:
                fp_idx.append(p_idx)
        else:
            fp_idx.append(p_idx)

    fn_idx = [i for i in range(len(gt_labels)) if i not in matched_gt]
    return tp_idx, fp_idx, fn_idx


def build_confusion_matrix(predictions_list, targets_list, class_names, iou_threshold):
    num_classes = len(class_names)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    for predictions, targets in zip(predictions_list, targets_list):
        pred_boxes = predictions["boxes"]
        pred_labels = predictions["labels"]
        pred_scores = predictions["scores"]
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        tp_idx, fp_idx, fn_idx = match_predictions(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
        )

        for idx in tp_idx:
            label = int(pred_labels[idx])
            cm[label, label] += 1

        for idx in fp_idx:
            label = int(pred_labels[idx])
            cm[label, 0] += 1

        for idx in fn_idx:
            label = int(gt_labels[idx])
            cm[0, label] += 1

    return cm


def _draw_box_on_image(img, box, label, color, y_offset=0):
    """Helper function to draw a single box on image"""
    x_min, y_min, x_max, y_max = box
    h, w = img.shape[:2]
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(w, int(x_max))
    y_max = min(h, int(y_max))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    text = (
        config.CLASS_NAMES[int(label) - 1]
        if 1 <= int(label) <= len(config.CLASS_NAMES)
        else f"Class {label}"
    )
    text = f"{text}"
    cv2.putText(
        img, text, (x_min, y_min - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )


def save_case_metadata(image_id, predictions, targets, metadata_dir):
    """Save metadata as JSON with image_id filename"""
    metadata = {
        "image_id": str(image_id),
        "ground_truth": {
            "boxes": (
                targets["boxes"].tolist()
                if isinstance(targets["boxes"], np.ndarray)
                else targets["boxes"]
            ),
            "labels": (
                targets["labels"].tolist()
                if isinstance(targets["labels"], np.ndarray)
                else targets["labels"]
            ),
            "class_names": (
                [config.CLASS_NAMES[int(l) - 1] for l in targets["labels"].tolist()]
                if targets["labels"].shape[0] > 0
                else []
            ),
        },
        "predictions": {
            "boxes": (
                predictions["boxes"].tolist()
                if predictions["boxes"].shape[0] > 0
                else []
            ),
            "scores": (
                predictions["scores"].tolist()
                if predictions["scores"].shape[0] > 0
                else []
            ),
            "labels": (
                predictions["labels"].tolist()
                if predictions["labels"].shape[0] > 0
                else []
            ),
            "class_names": (
                [config.CLASS_NAMES[int(l) - 1] for l in predictions["labels"].tolist()]
                if predictions["labels"].shape[0] > 0
                else []
            ),
        },
    }

    metadata_path = os.path.join(metadata_dir, f"{image_id}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_case_visualization(
    image_id, predictions, targets, case_dir, dataset_dir
):
    """Save visualization image for a single case - side-by-side GT (left) and Pred (right)"""
    img_path = os.path.join(dataset_dir, f"{image_id}.png")
    img_original = cv2.imread(img_path)

    # Create two copies - one for GT, one for Predictions
    img_gt = img_original.copy()
    img_pred = img_original.copy()

    # Draw ground truth boxes in blue on left image
    for box, label in zip(targets["boxes"], targets["labels"]):
        _draw_box_on_image(img_gt, box, label, (255, 0, 0), 20)

    # Draw prediction boxes in green on right image
    for box, score, label in zip(
        predictions["boxes"], predictions["scores"], predictions["labels"]
    ):
        x_min, y_min, x_max, y_max = box
        h, w = img_pred.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        cv2.rectangle(img_pred, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        text = (
            config.CLASS_NAMES[int(label) - 1]
            if 1 <= int(label) <= len(config.CLASS_NAMES)
            else f"Class {label}"
        )
        cv2.putText(
            img_pred,
            f"{text} {score:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Concatenate images horizontally
    img_combined = cv2.hconcat([img_gt, img_pred])

    output_img_path = os.path.join(case_dir, f"{image_id}.png")
    cv2.imwrite(output_img_path, img_combined)


def compute_metrics_per_class(
    predictions_list, targets_list, class_names, iou_threshold
):
    """Compute per-class metrics"""
    num_classes = len(class_names)
    metrics_per_class = {}

    for class_id in range(1, num_classes + 1):
        tp = fp = fn = 0

        for predictions, targets in zip(predictions_list, targets_list):
            pred_boxes = predictions["boxes"]
            pred_labels = predictions["labels"]
            pred_scores = predictions["scores"]
            gt_boxes = targets["boxes"]
            gt_labels = targets["labels"]

            # Match all predictions to ground truth
            tp_idx, fp_idx, fn_idx = match_predictions(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
            )

            # Count for current class
            for idx in tp_idx:
                if pred_labels[idx] == class_id:
                    tp += 1
            for idx in fp_idx:
                if pred_labels[idx] == class_id:
                    fp += 1
            for idx in fn_idx:
                if gt_labels[idx] == class_id:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics_per_class[class_id] = {
            "class_name": class_names[class_id - 1],
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return metrics_per_class


def compute_ap(predictions_list, targets_list, class_id, iou_threshold):
    """
    Compute Average Precision (AP) for a specific class
    """
    # Collect all predictions and ground truths for this class
    all_pred_scores = []
    all_pred_matched = []
    all_gt_count = 0

    for predictions, targets in zip(predictions_list, targets_list):
        pred_boxes = predictions["boxes"]
        pred_labels = predictions["labels"]
        pred_scores = predictions["scores"]
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        # Count ground truth objects for this class
        for gt_label in gt_labels:
            if gt_label == class_id:
                all_gt_count += 1

        # Match predictions
        tp_idx, fp_idx, fn_idx = match_predictions(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
        )

        # Record predictions for this class
        for idx in range(len(pred_labels)):
            if pred_labels[idx] == class_id:
                is_tp = idx in tp_idx
                all_pred_scores.append(pred_scores[idx])
                all_pred_matched.append(is_tp)

    if all_gt_count == 0:
        return 0.0

    if len(all_pred_scores) == 0:
        return 0.0

    # Sort by scores in descending order
    sorted_indices = np.argsort(-np.array(all_pred_scores))
    all_pred_matched = np.array(all_pred_matched)[sorted_indices]

    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(all_pred_matched)
    fp_cumsum = np.cumsum(1 - all_pred_matched)

    # Compute precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / all_gt_count

    # Compute AP (area under PR curve)
    ap = 0.0
    for i in range(len(precision)):
        if i == 0 or recall[i] != recall[i - 1]:
            ap += precision[i] * (recall[i] - (recall[i - 1] if i > 0 else 0))

    return ap


def compute_map(predictions_list, targets_list, class_names, iou_threshold):
    """
    Compute mean Average Precision (mAP) across all classes
    """
    num_classes = len(class_names)
    aps = {}

    for class_id in range(1, num_classes + 1):
        ap = compute_ap(predictions_list, targets_list, class_id, iou_threshold)
        aps[class_id] = {
            "class_name": class_names[class_id - 1],
            "ap": float(ap),
        }

    # Calculate mean AP
    mean_ap = np.mean([ap["ap"] for ap in aps.values()])

    return mean_ap, aps


def save_map_metrics(mean_ap, aps, metrics_dir):
    """
    Save mAP metrics to JSON file
    """
    map_summary = {
        "map": float(mean_ap),
        "per_class_ap": aps,
    }
    map_json_path = os.path.join(metrics_dir, "map_metrics.json")
    with open(map_json_path, "w") as f:
        json.dump(map_summary, f, indent=2)


def save_metrics_summary(metrics_per_class, metrics_dir):
    summary = {
        "overall": {
            "total_tp": sum(m["tp"] for m in metrics_per_class.values()),
            "total_fp": sum(m["fp"] for m in metrics_per_class.values()),
            "total_fn": sum(m["fn"] for m in metrics_per_class.values()),
        },
        "per_class": metrics_per_class,
    }

    # Add overall precision, recall, f1
    total_tp = summary["overall"]["total_tp"]
    total_fp = summary["overall"]["total_fp"]
    total_fn = summary["overall"]["total_fn"]

    summary["overall"]["precision"] = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    summary["overall"]["recall"] = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    )
    summary["overall"]["f1"] = (
        2
        * (summary["overall"]["precision"] * summary["overall"]["recall"])
        / (summary["overall"]["precision"] + summary["overall"]["recall"])
        if (summary["overall"]["precision"] + summary["overall"]["recall"]) > 0
        else 0
    )

    summary_path = os.path.join(metrics_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def save_confusion_matrix_csv(cm, class_names, metrics_dir):
    csv_path = os.path.join(metrics_dir, "confusion_matrix.csv")
    labels = ["Background"] + class_names
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(csv_path)


def save_confusion_matrix_png(cm, class_names, metrics_dir):
    _, ax = plt.subplots(figsize=(14, 12))
    labels = ["Background"] + class_names
    
    # Create custom annotations - only show non-zero values
    annot_array = cm.astype(str)
    annot_array[cm == 0] = ""
    
    sns.heatmap(
        cm,
        annot=annot_array,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    plt.tight_layout()

    png_path = os.path.join(metrics_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()


def evaluate(
    predictions_list,
    targets_list,
    images_dir,
    metadatas_dir,
    metrics_dir,
    image_ids=None,
    dataset_dir=None,
    iou_threshold=None,
):
    logger = get_logger()

    # Use provided iou_threshold or fallback to config default
    if iou_threshold is None:
        iou_threshold = config.IOU_THRESHOLD

    # Use fallback for image_ids if not provided
    if image_ids is None:
        image_ids = list(range(len(predictions_list)))

    logger.info(f"Saving {len(predictions_list)} visualizations and metadatas...")

    # Save per-case visualizations and metadata
    for predictions, targets, image_id in zip(
        predictions_list, targets_list, image_ids
    ):
        # Save metadata
        save_case_metadata(image_id, predictions, targets, metadatas_dir)

        # Save visualization (composite image: GT | Pred)
        save_case_visualization(image_id, predictions, targets, images_dir, dataset_dir)

    logger.info(f"Saved {len(predictions_list)} visualizations and metadatas")

    # Compute and save metrics
    logger.info("Computing per-class metrics...")
    metrics_per_class = compute_metrics_per_class(
        predictions_list, targets_list, config.CLASS_NAMES, iou_threshold
    )

    logger.info("Saving metrics_summary.json...")
    save_metrics_summary(metrics_per_class, metrics_dir)

    logger.info("Building confusion matrix...")
    cm = build_confusion_matrix(
        predictions_list, targets_list, config.CLASS_NAMES, iou_threshold
    )

    logger.info("Saving confusion_matrix.csv...")
    save_confusion_matrix_csv(cm, config.CLASS_NAMES, metrics_dir)

    logger.info("Saving confusion_matrix.png...")
    save_confusion_matrix_png(cm, config.CLASS_NAMES, metrics_dir)

    # Compute and save mAP
    logger.info(f"Computing mAP (IOU threshold: {iou_threshold})...")
    mean_ap, aps = compute_map(
        predictions_list, targets_list, config.CLASS_NAMES, iou_threshold
    )

    logger.info("Saving mAP metrics...")
    save_map_metrics(mean_ap, aps, metrics_dir)
    logger.info(f"mAP: {mean_ap:.4f}")

    logger.info(f"All metrics and visualizations saved to {os.path.dirname(images_dir)}")
