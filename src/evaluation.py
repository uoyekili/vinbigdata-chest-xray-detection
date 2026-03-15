import os
import json
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.logger import get_logger


def draw_boxes(image, boxes, labels, scores=None, class_names=None):
    img = image.copy()
    h, w = image.shape[:2]

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = box
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        text = class_names[int(label) - 1] if class_names else f"Class {label}"
        if scores is not None:
            text += f" {scores[idx]:.2f}"

        cv2.putText(
            img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    return img


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
    pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5
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


def build_confusion_matrix(
    predictions_list, targets_list, class_names, iou_threshold=0.5
):
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


def plot_confusion_matrix(cm, class_names, output_path):
    _, ax = plt.subplots(figsize=(12, 10))
    labels = ["Background"] + class_names
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_prediction_metadata(case_id, predictions, targets, image_id, output_dir):
    """Save metadata for a single prediction case as JSON - DEPRECATED"""
    case_dir = os.path.join(output_dir, f"case_{case_id}")
    os.makedirs(case_dir, exist_ok=True)

    metadata = {
        "image_id": str(image_id),
        "case_id": case_id,
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
            "num_objects": (
                len(targets["labels"]) if targets["labels"].shape[0] > 0 else 0
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
            "num_detections": (
                len(predictions["labels"]) if predictions["labels"].shape[0] > 0 else 0
            ),
        },
    }

    metadata_path = os.path.join(case_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return case_dir


def save_case_metadata(case_id, predictions, targets, image_id, case_dir):
    """Save metadata for a single prediction case as JSON with detailed info"""
    metadata = {
        "case_id": case_id,
        "patient_id": str(image_id),
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
            "num_objects": (
                int(len(targets["labels"])) if targets["labels"].shape[0] > 0 else 0
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
            "num_detections": (
                int(len(predictions["labels"]))
                if predictions["labels"].shape[0] > 0
                else 0
            ),
        },
    }

    metadata_path = os.path.join(case_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_case_visualization(
    case_id, image, predictions, targets, case_dir, dataset_dir
):
    """Save visualization image for a single case"""
    img_path = os.path.join(dataset_dir, f"{case_id}.png")

    if not os.path.exists(img_path):
        # Try alternative naming
        img_path = os.path.join(dataset_dir, f"case_{case_id}.png")

    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
    else:
        # Use provided image or create blank canvas if needed
        h, w = 1024, 1024
        if image is not None:
            h, w = image.shape[:2]
        else:
            image = np.ones((h, w, 3), dtype=np.uint8) * 128

    # Draw ground truth boxes in blue
    img_with_boxes = image.copy()
    for box, label in zip(targets["boxes"], targets["labels"]):
        x_min, y_min, x_max, y_max = box
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
        cv2.rectangle(
            img_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2
        )  # Blue
        text = (
            config.CLASS_NAMES[int(label) - 1]
            if 1 <= int(label) <= len(config.CLASS_NAMES)
            else f"Class {label}"
        )
        cv2.putText(
            img_with_boxes,
            f"GT: {text}",
            (x_min, y_min - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Draw prediction boxes in green
    for box, score, label in zip(
        predictions["boxes"], predictions["scores"], predictions["labels"]
    ):
        x_min, y_min, x_max, y_max = box
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
        cv2.rectangle(
            img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )  # Green
        text = (
            config.CLASS_NAMES[int(label) - 1]
            if 1 <= int(label) <= len(config.CLASS_NAMES)
            else f"Class {label}"
        )
        cv2.putText(
            img_with_boxes,
            f"Pred: {text} {score:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Save as visualization
    output_img_path = os.path.join(case_dir, f"{case_id}.png")
    cv2.imwrite(output_img_path, img_with_boxes)


def compute_metrics_per_class(
    predictions_list, targets_list, class_names, iou_threshold=0.5
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

    return summary


def save_metrics_table(metrics_per_class, metrics_dir):
    csv_path = os.path.join(metrics_dir, "metrics_table.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Class ID", "Class Name", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        )

        for class_id in sorted(metrics_per_class.keys()):
            m = metrics_per_class[class_id]
            writer.writerow(
                [
                    class_id,
                    m["class_name"],
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                ]
            )


def save_per_class_metrics(metrics_per_class, metrics_dir):
    csv_path = os.path.join(metrics_dir, "per_class_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Class ID", "Class Name", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        )

        for class_id in sorted(metrics_per_class.keys()):
            m = metrics_per_class[class_id]
            writer.writerow(
                [
                    class_id,
                    m["class_name"],
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                ]
            )


def save_confusion_matrix_csv(cm, class_names, metrics_dir):
    csv_path = os.path.join(metrics_dir, "confusion_matrix.csv")

    labels = ["Background"] + class_names
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(csv_path)


def save_confusion_matrix_png(cm, class_names, metrics_dir):
    """Save confusion_matrix.png"""
    _, ax = plt.subplots(figsize=(14, 12))
    labels = ["Background"] + class_names
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    png_path = os.path.join(metrics_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()


def evaluate(
    predictions_list,
    targets_list,
    eval_base_dir,
    cases_dir,
    metrics_dir,
    image_ids=None,
    df_test=None,
    dataset_dir=None,
):
    logger = get_logger()

    # Use fallback for image_ids if not provided
    if image_ids is None:
        image_ids = list(range(len(predictions_list)))

    # Use config dataset_dir if not provided
    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    logger.info(f"Saving {len(predictions_list)} cases...")

    # Save per-case visualizations and metadata
    for case_id, (predictions, targets, image_id) in enumerate(
        zip(predictions_list, targets_list, image_ids)
    ):
        patient_id = str(image_id)
        case_subdir = os.path.join(cases_dir, patient_id)
        os.makedirs(case_subdir, exist_ok=True)

        # Save metadata
        save_case_metadata(patient_id, predictions, targets, image_id, case_subdir)

        # Save visualization
        save_case_visualization(
            patient_id, None, predictions, targets, case_subdir, dataset_dir
        )

    logger.info(f"✓ Saved {len(predictions_list)} cases")

    # Compute and save metrics
    logger.info("Computing per-class metrics...")
    metrics_per_class = compute_metrics_per_class(
        predictions_list, targets_list, config.CLASS_NAMES
    )

    logger.info("Saving metrics_summary.json...")
    summary = save_metrics_summary(metrics_per_class, metrics_dir)

    logger.info("Saving metrics_table.csv...")
    save_metrics_table(metrics_per_class, metrics_dir)

    logger.info("Saving per_class_metrics.csv...")
    save_per_class_metrics(metrics_per_class, metrics_dir)

    logger.info("Building confusion matrix...")
    cm = build_confusion_matrix(
        predictions_list, targets_list, config.CLASS_NAMES, iou_threshold=0.5
    )

    logger.info("Saving confusion_matrix.csv...")
    save_confusion_matrix_csv(cm, config.CLASS_NAMES, metrics_dir)

    logger.info("Saving confusion_matrix.png...")
    save_confusion_matrix_png(cm, config.CLASS_NAMES, metrics_dir)

    logger.info(f"✓ All metrics saved to {metrics_dir}")

    return summary
