"""Evaluation on local test set with comprehensive analysis and visualization."""

import argparse
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import seaborn as sns

from src import config
from src.dataset import create_test_dataloader
from src.model import load_models
from src.metrics import compute_map, compute_ap
from src.ensemble import ensemble_multi_model, filter_predictions, predict_single_model
from src.utils import get_logger, ensure_dir, find_trained_models


def load_ground_truth(csv_path: str):
    """Load ground truth from CSV."""
    df = pd.read_csv(csv_path)
    gt_dict = {}
    
    for image_id in df["image_id"].unique():
        rows = df[df["image_id"] == image_id]
        boxes = []
        labels = []
        
        for _, row in rows.iterrows():
            if "class_id" in row and row["class_id"] == 14:
                continue
            boxes.append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
            labels.append(int(row["class_id"]) + 1)
        
        gt_dict[image_id] = {
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        }
    
    return gt_dict


def load_image(image_id: str, image_dir: str):
    """Load and return image from PNG file."""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def draw_boxes_on_image(image, boxes, labels, scores=None, color=(0, 255, 0), label_names=None, thickness=2, score_threshold=None):
    """Draw bounding boxes on image.
    
    Args:
        image: numpy array (H, W, 3) RGB
        boxes: nx4 array [x_min, y_min, x_max, y_max]
        labels: class labels
        scores: confidence scores (optional)
        color: (R, G, B)
        label_names: list of class names
        thickness: line thickness
        score_threshold: only draw boxes with score >= threshold (optional)
    
    Returns:
        image with boxes drawn
    """
    img_copy = image.copy()
    h, w = image.shape[:2]
    
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        # Skip boxes below confidence threshold
        if score_threshold is not None and scores is not None:
            if scores[idx] < score_threshold:
                continue
        
        x_min, y_min, x_max, y_max = box
        
        # Clip to image bounds
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw label
        label = int(label)
        if label_names:
            text = label_names[label - 1]
        else:
            text = f"Class {label}"
        
        if scores is not None:
            text += f" {scores[idx]:.2f}"
        
        cv2.putText(img_copy, text, (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_copy


def visualize_image_predictions(image_id: str, image, predictions, ground_truth, 
                               output_path: str, class_names: list, score_threshold=None):
    """Create side-by-side visualization of predictions vs ground truth.
    
    Args:
        image_id: image identifier
        image: numpy array RGB
        predictions: dict with 'boxes', 'labels', 'scores'
        ground_truth: dict with 'boxes', 'labels'
        output_path: where to save the result
        class_names: list of class names
        score_threshold: confidence threshold for predictions
    """
    h, w = image.shape[:2]
    
    # Draw predictions vs ground truth side-by-side
    img_pred = draw_boxes_on_image(image, predictions["boxes"], predictions["labels"],
                                  predictions["scores"], color=(0, 255, 0), 
                                  label_names=class_names, thickness=2, score_threshold=score_threshold)
    img_gt = draw_boxes_on_image(image, ground_truth["boxes"], ground_truth["labels"],
                                color=(255, 0, 0), label_names=class_names, thickness=2)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_gt)
    axes[0].set_title(f"Ground Truth ({len(ground_truth['boxes'])} boxes)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Count filtered predictions
    filtered_count = len([s for s in predictions['scores'] if score_threshold is None or s >= score_threshold]) if len(predictions['scores']) > 0 else 0
    axes[1].imshow(img_pred)
    axes[1].set_title(f"Predictions ({filtered_count} boxes", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def apply_wbf(boxes, scores, labels, image_width=1024, image_height=1024):
    """Apply Weighted Boxes Fusion to merge overlapping boxes.
    
    Args:
        boxes: nx4 array [x_min, y_min, x_max, y_max]
        scores: confidence scores
        labels: class labels
        image_width: image width (for normalization)
        image_height: image height (for normalization)
    
    Returns:
        keep_indices: indices of boxes to keep after WBF
    """
    if len(boxes) == 0:
        return np.array([])
    
    # Normalize coordinates to [0, 1]
    boxes_normalized = boxes.copy().astype(float)
    boxes_normalized[:, [0, 2]] /= image_width
    boxes_normalized[:, [1, 3]] /= image_height
    boxes_normalized = np.clip(boxes_normalized, 0, 1)
    
    # Apply WBF
    try:
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            [boxes_normalized.tolist()],
            [scores.tolist()],
            [labels.tolist()],
            iou_thr=config.WBF_IOU_THR,
            skip_box_thr=config.WBF_SKIP_BOX_THR,
        )
        
        # Denormalize coordinates
        merged_boxes = np.array(merged_boxes)
        merged_boxes[:, [0, 2]] *= image_width
        merged_boxes[:, [1, 3]] *= image_height
        merged_scores = np.array(merged_scores)
        merged_labels = np.array(merged_labels, dtype=np.int64)
        
        return merged_boxes, merged_scores, merged_labels
    except Exception as e:
        # If WBF fails, return original
        return boxes, scores, labels


def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores, 
                                     gt_boxes, gt_labels, iou_threshold=0.5):
    """Match predictions to ground truth and classify as TP/FP.
    
    Returns:
        - tp_indices: indices of true positives in predictions
        - fp_indices: indices of false positives in predictions
        - fn_indices: indices of false negatives in ground truth
        - matched_pairs: list of (pred_idx, gt_idx, iou) for matched boxes
    """
    tp_indices = []
    fp_indices = []
    matched_pairs = []
    matched_gt = set()
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            # Check if label matches
            if pred_labels[pred_idx] == gt_labels[best_gt_idx]:
                tp_indices.append(pred_idx)
                matched_pairs.append((pred_idx, best_gt_idx, best_iou))
                matched_gt.add(best_gt_idx)
            else:
                fp_indices.append(pred_idx)
        else:
            fp_indices.append(pred_idx)
    
    fn_indices = [i for i in range(len(gt_labels)) if i not in matched_gt]
    
    return tp_indices, fp_indices, fn_indices, matched_pairs


def evaluate(fold: int = 0, use_ensemble: bool = True):
    """Evaluate on local test set with comprehensive analysis."""
    logger = get_logger()
    device = config.DEVICE
    
    # Find trained models
    available_models = find_trained_models(fold)
    if not available_models:
        raise RuntimeError(f"No trained models found for fold {fold}")
    
    # Load GT
    gt_dict = load_ground_truth(config.HOLDOUT_TEST_CSV)
    image_ids = list(gt_dict.keys())
    logger.info(f"Test set: {len(image_ids)} images")
    
    loader = create_test_dataloader(image_ids, config.TRAIN_PNG_DIR, batch_size=1)
    all_predictions = []
    all_predictions_wbf = []  # After WBF
    all_targets = []
    image_results = {}
    all_scores = []
    image_details = {}  # For per-image detailed analysis
    
    # Run inference
    if use_ensemble and len(available_models) > 1:
        logger.info(f"Evaluating ENSEMBLE ({len(available_models)} models)...")
        models = load_models(available_models, fold, device)
        mode = "ensemble"
        
        pbar = tqdm(loader, desc="Evaluation")
        for images, ids, orig_ws, orig_hs in pbar:
            image = images[0]
            image_id = ids[0]
            pred = ensemble_multi_model(models, image, device)
            pred = filter_predictions(pred, config.FINAL_CONF_THRESHOLD)
            
            # Apply WBF to merge overlapping boxes
            if len(pred["boxes"]) > 0:
                boxes_wbf, scores_wbf, labels_wbf = apply_wbf(
                    pred["boxes"], pred["scores"], pred["labels"],
                    image_width=int(orig_ws[0]) if hasattr(orig_ws[0], 'item') else orig_ws[0],
                    image_height=int(orig_hs[0]) if hasattr(orig_hs[0], 'item') else orig_hs[0]
                )
                pred_nms = {
                    "boxes": boxes_wbf,
                    "labels": labels_wbf,
                    "scores": scores_wbf,
                }
            else:
                pred_nms = pred.copy()
            
            all_predictions.append(pred)
            all_predictions_wbf.append(pred_nms)
            all_targets.append(gt_dict[image_id])
            image_results[image_id] = (len(pred["boxes"]), len(gt_dict[image_id]["boxes"]))
            all_scores.extend(pred["scores"])
            
            # Store detailed info for later visualization
            image_details[image_id] = {
                "predictions": pred,
                "ground_truth": gt_dict[image_id],
                "orig_w": int(orig_ws[0]) if hasattr(orig_ws[0], 'item') else orig_ws[0],
                "orig_h": int(orig_hs[0]) if hasattr(orig_hs[0], 'item') else orig_hs[0],
            }
    else:
        model_name = available_models[0]
        logger.info(f"Evaluating SINGLE model ({model_name})...")
        model = load_models([model_name], fold, device)[0]
        mode = f"single_{model_name}"
        
        pbar = tqdm(loader, desc="Evaluation")
        for images, ids, orig_ws, orig_hs in pbar:
            image = images[0]
            image_id = ids[0]
            pred = predict_single_model(model, image, device)
            pred = filter_predictions(pred, config.FINAL_CONF_THRESHOLD)
            
            # Apply WBF to merge overlapping boxes
            if len(pred["boxes"]) > 0:
                boxes_wbf, scores_wbf, labels_wbf = apply_wbf(
                    pred["boxes"], pred["scores"], pred["labels"],
                    image_width=int(orig_ws[0]) if hasattr(orig_ws[0], 'item') else orig_ws[0],
                    image_height=int(orig_hs[0]) if hasattr(orig_hs[0], 'item') else orig_hs[0]
                )
                pred_nms = {
                    "boxes": boxes_wbf,
                    "labels": labels_wbf,
                    "scores": scores_wbf,
                }
            else:
                pred_nms = pred.copy()
            
            all_predictions.append(pred)
            all_predictions_wbf.append(pred_nms)
            all_targets.append(gt_dict[image_id])
            image_results[image_id] = (len(pred["boxes"]), len(gt_dict[image_id]["boxes"]))
            all_scores.extend(pred["scores"])
            
            # Store detailed info for later visualization
            image_details[image_id] = {
                "predictions": pred,
                "ground_truth": gt_dict[image_id],
                "orig_w": orig_ws[0],
                "orig_h": orig_hs[0],
            }
    
    # Compute metrics
    mAP = compute_map(all_predictions, all_targets, iou_threshold=0.4)
    
    # Compute per-class AP
    class_preds = defaultdict(list)
    class_gts = defaultdict(lambda: defaultdict(list))
    
    for img_idx, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            class_preds[int(label)].append((img_idx, box, float(score)))
        for box, label in zip(target["boxes"], target["labels"]):
            class_gts[int(label)][img_idx].append(box)
    
    per_class_ap = {}
    for cls_id in sorted(class_gts.keys()):
        ap = compute_ap(class_preds[cls_id], class_gts[cls_id])
        per_class_ap[cls_id] = ap
    
    # ====================================================================
    # Compute Confusion Matrix and ROC Curves for all predictions
    # ====================================================================
    all_true_labels = []
    all_pred_labels = []
    all_pred_scores = []
    
    for img_idx, (pred, target) in enumerate(zip(all_predictions_wbf, all_targets)):
        # Match predictions to ground truth
        tp_idx, fp_idx, fn_idx, _ = match_predictions_to_ground_truth(
            pred["boxes"], pred["labels"], pred["scores"],
            target["boxes"], target["labels"], iou_threshold=0.4
        )
        
        # Add TP
        for idx in tp_idx:
            all_true_labels.append(int(pred["labels"][idx]))
            all_pred_labels.append(int(pred["labels"][idx]))
            all_pred_scores.append(float(pred["scores"][idx]))
        
        # Add FP
        for idx in fp_idx:
            all_true_labels.append(0)  # Background/negative
            all_pred_labels.append(int(pred["labels"][idx]))
            all_pred_scores.append(float(pred["scores"][idx]))
        
        # Add FN
        for idx in fn_idx:
            all_true_labels.append(int(target["labels"][idx]))
            all_pred_labels.append(0)  # Undetected (background)
            all_pred_scores.append(0.0)
    
    # Print results
    logger.info("=" * 60)
    logger.info(f"mAP@0.4 ({mode}): {mAP:.4f}")
    logger.info("=" * 60)
    logger.info("Per-class AP:")
    for cls_id, ap in sorted(per_class_ap.items()):
        class_name = config.CLASS_NAMES[cls_id - 1]
        logger.info(f"  {class_name:25s}: {ap:.4f}")
    
    # Create organized output directories
    analysis_dir = ensure_dir(f"{config.OUTPUT_DIR}/analysis")
    metrics_dir = ensure_dir(f"{analysis_dir}/metrics")
    charts_dir = ensure_dir(f"{analysis_dir}/charts")
    images_dir = ensure_dir(f"{analysis_dir}/images")
    tp_dir = ensure_dir(f"{images_dir}/true_positives")
    fp_dir = ensure_dir(f"{images_dir}/false_positives")
    fn_dir = ensure_dir(f"{images_dir}/false_negatives")
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    
    # ====================================================================
    # 1. Save detailed metrics reports
    # ====================================================================
    
    # Per-class metrics
    class_metrics_data = []
    for cls_id in sorted(per_class_ap.keys()):
        class_name = config.CLASS_NAMES[cls_id - 1]
        class_metrics_data.append({
            "class_id": cls_id,
            "class_name": class_name,
            "average_precision": per_class_ap[cls_id],
            "prediction_count": len(class_preds.get(cls_id, [])),
            "ground_truth_count": sum(len(boxes) for boxes in class_gts[cls_id].values()),
        })
    
    class_metrics_df = pd.DataFrame(class_metrics_data)
    class_metrics_df.to_csv(f"{metrics_dir}/per_class_metrics.csv", index=False)
    logger.info(f"  ✓ per_class_metrics.csv")
    
    # Overall metrics
    overall_metrics = {
        "Metric": ["mAP@0.4", "Total Images", "Total Predictions (After WBF)", "Total Ground Truth Boxes",
                  "Avg Predictions per Image (After WBF)", "Avg GT Boxes per Image", "Avg Confidence Score"],
        "Value": [
            f"{mAP:.4f}",
            len(image_ids),
            sum(len(p["boxes"]) for p in all_predictions_wbf),
            sum(len(t["boxes"]) for t in all_targets),
            f"{np.mean([len(p['boxes']) for p in all_predictions_wbf]):.2f}",
            f"{np.mean([len(t['boxes']) for t in all_targets]):.2f}",
            f"{np.mean(all_scores):.4f}" if all_scores else "N/A",
        ]
    }
    pd.DataFrame(overall_metrics).to_csv(f"{metrics_dir}/overall_metrics.csv", index=False)
    logger.info(f"  ✓ overall_metrics.csv")
    
    # ====================================================================
    # 2. Advanced Visualizations
    # ====================================================================
    pred_counts = [len(p["boxes"]) for p in all_predictions_wbf]
    gt_counts = [len(t["boxes"]) for t in all_targets]
    
    # 2.1 Distribution of Number of Boxes (Violin Plot)
    # -------------------------------------------------
    data_counts = []
    for c in pred_counts:
        data_counts.append({"Type": "Prediction", "Count": c})
    for c in gt_counts:
        data_counts.append({"Type": "Ground Truth", "Count": c})
    df_counts = pd.DataFrame(data_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_counts, x="Type", y="Count", hue="Type", palette="muted", split=True, ax=ax)
    ax.set_title("Distribution of Box Counts per Image")
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/box_count_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ charts/box_count_distribution.png")

    # 2.2 Scatter: GT vs Predictions per image (Hexbin Plot)
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add jitter to avoid overlapping points
    jitter_gt = np.array(gt_counts) + np.random.normal(0, 0.1, len(gt_counts))
    jitter_pred = np.array(pred_counts) + np.random.normal(0, 0.1, len(pred_counts))
    
    hb = ax.hexbin(jitter_gt, jitter_pred, gridsize=20, cmap='Blues', mincnt=1)
    ax.set_xlabel("Ground Truth Boxes")
    ax.set_ylabel("Predicted Boxes")
    ax.set_title("Prediction vs Ground Truth Count Correlation")
    
    # Perfect match line
    max_val = max(max(gt_counts), max(pred_counts)) if gt_counts else 5
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Perfect Match")
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/box_count_correlation.png", dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ charts/box_count_correlation.png")
    
    # 2.3 Confidence Score Distribution (KDE Plot)
    # --------------------------------------------
    if all_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(all_scores, bins=30, kde=True, color="teal", ax=ax)
        mean_score = np.mean(all_scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        ax.set_xlabel("Confidence Score")
        ax.set_title("Distribution of Confidence Scores")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/confidence_distribution.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ charts/confidence_distribution.png")
    
    # 2.4 Class Distribution: GT vs Predictions (Side-by-side Bar)
    # ------------------------------------------------------------
    class_dist_gt = defaultdict(int)
    class_dist_pred = defaultdict(int)
    for target in all_targets:
        for label in target["labels"]:
            class_dist_gt[int(label)] += 1
    for pred in all_predictions_wbf:
        for label in pred["labels"]:
            class_dist_pred[int(label)] += 1
            
    dist_data = []
    all_classes = sorted(list(set(class_dist_gt.keys()) | set(class_dist_pred.keys())))
    
    for cls_id in all_classes:
        cls_name = config.CLASS_NAMES[cls_id - 1]
        dist_data.append({"Class": cls_name, "Count": class_dist_gt[cls_id], "Source": "Ground Truth"})
        dist_data.append({"Class": cls_name, "Count": class_dist_pred[cls_id], "Source": "Prediction"})
        
    df_dist = pd.DataFrame(dist_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_dist, x="Class", y="Count", hue="Source", palette="viridis", ax=ax)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Class Frequency: Ground Truth vs Predictions")
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/class_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ charts/class_distribution.png")

    
    # ====================================================================
    # 3. Per-class AP bar chart
    # ====================================================================
    # Prepare dataframe for Seaborn
    ap_data = []
    for i, name in enumerate(config.CLASS_NAMES):
        ap = per_class_ap.get(i+1, 0)
        # Assign color group based on AP value
        if ap >= 0.5:
            color_group = "Good"
        elif ap >= 0.3:
            color_group = "Fair"
        else:
            color_group = "Poor"
        ap_data.append({"Class": name, "AP": ap, "Performance": color_group})
    df_ap = pd.DataFrame(ap_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    # Use hue to assign colors based on performance group
    color_map = {"Good": "#2ecc71", "Fair": "#f1c40f", "Poor": "#e74c3c"}
    
    sns.barplot(data=df_ap, x="Class", y="AP", hue="Performance", palette=color_map, ax=ax, legend=False)
    
    # Add threshold lines
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Target: AP=0.5")
    ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.5, label="Baseline: AP=0.3")
    
    # Add values on top of bars
    for i, v in enumerate(df_ap["AP"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(f"Per-Class Average Precision (mAP={mAP:.4f})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/class_performance.png", dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ charts/class_performance.png")
    
    # ====================================================================
    # 3.5. Confusion Matrix Heatmap
    # ====================================================================
    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(len(config.CLASS_NAMES) + 1)))
        
        fig, ax = plt.subplots(figsize=(16, 14))
        class_labels = ['Background'] + [n[:15] for n in config.CLASS_NAMES]
        
        # Create custom annotations - only show non-zero values
        annot_array = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    annot_array[i, j] = f'{int(cm[i, j])}'
                else:
                    annot_array[i, j] = ''
        
        sns.heatmap(cm, annot=annot_array, fmt='', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix (Count)")
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/confusion_matrix.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ charts/confusion_matrix.png")
        
        # Analyze common errors - find top misclassification pairs
        error_pairs = []
        for true_label_idx in range(len(config.CLASS_NAMES) + 1):
            for pred_label_idx in range(len(config.CLASS_NAMES) + 1):
                if true_label_idx != pred_label_idx and cm[true_label_idx, pred_label_idx] > 0:
                    true_name = 'Background' if true_label_idx == 0 else config.CLASS_NAMES[true_label_idx - 1]
                    pred_name = 'Background' if pred_label_idx == 0 else config.CLASS_NAMES[pred_label_idx - 1]
                    error_pairs.append({
                        'true_label': true_name,
                        'predicted_label': pred_name,
                        'count': int(cm[true_label_idx, pred_label_idx])
                    })
        
        error_df = pd.DataFrame(error_pairs).sort_values('count', ascending=False)
        if len(error_df) > 0:
            error_df.to_csv(f"{metrics_dir}/common_errors.csv", index=False)
            logger.info(f"  ✓ metrics/common_errors.csv")
    
    # ====================================================================
    # 3.6. ROC Curves for Each Class
    # ====================================================================
    if len(all_true_labels) > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Compute ROC for each class
        roc_data = []
        for cls_id in range(1, len(config.CLASS_NAMES) + 1):
            # Binary classification: cls_id vs rest
            binary_true = np.array([1 if label == cls_id else 0 for label in all_true_labels])
            # Use max score for this class or 0 if not predicted as this class
            # This is a simplification; ideally we need scores for all classes for each box
            binary_scores = np.array([score if label == cls_id else 0.0 for label, score in zip(all_pred_labels, all_pred_scores)])
            
            if len(np.unique(binary_true)) > 1:  # Only if both classes present
                fpr, tpr, _ = roc_curve(binary_true, binary_scores)
                roc_auc = auc(fpr, tpr)
                roc_data.append((cls_id, fpr, tpr, roc_auc))
                
                ax.plot(fpr, tpr, lw=2, label=f'{config.CLASS_NAMES[cls_id-1][:20]} (AUC={roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - Multi-class Detection")
        ax.legend(loc="lower right", fontsize=10, ncol=2)
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/roc_curves.png", dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ charts/roc_curves.png")
        
        # Save ROC metrics
        roc_metrics = []
        for cls_id, fpr, tpr, roc_auc in roc_data:
            roc_metrics.append({
                "class_id": cls_id,
                "class_name": config.CLASS_NAMES[cls_id - 1],
                "auc_score": roc_auc,
            })
        pd.DataFrame(roc_metrics).to_csv(f"{metrics_dir}/roc_metrics.csv", index=False)
        logger.info(f"  ✓ metrics/roc_metrics.csv")
    
    # ====================================================================
    # 4. Detailed image-level analysis with TP/FP/FN
    # ===================================================================
    image_analysis_records = []
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    # Track images by type for visualization
    tp_images = []  # (img_id, tp_count, fp_count, fn_count)
    fp_images = []  # Images with only FP
    fn_images = []  # Images with FN
    
    for img_id, (pred_nms, target) in zip(image_ids, zip(all_predictions_wbf, all_targets)):
        tp_idx, fp_idx, fn_idx, matched = match_predictions_to_ground_truth(
            pred_nms["boxes"], pred_nms["labels"], pred_nms["scores"],
            target["boxes"], target["labels"]
        )
        
        tp_count += len(tp_idx)
        fp_count += len(fp_idx)
        fn_count += len(fn_idx)
        
        image_analysis_records.append({
            "image_id": img_id,
            "true_positives": len(tp_idx),
            "false_positives": len(fp_idx),
            "false_negatives": len(fn_idx),
            "total_predictions": len(pred_nms["boxes"]),
            "total_ground_truth": len(target["boxes"]),
            "avg_confidence": np.mean(pred_nms["scores"]) if len(pred_nms["scores"]) > 0 else 0.0,
        })
        
        # Categorize images for visualization
        if len(tp_idx) > 0:
            tp_images.append((img_id, len(tp_idx), len(fp_idx), len(fn_idx)))
        if len(fp_idx) > 0 and len(tp_idx) == 0:
            fp_images.append((img_id, len(tp_idx), len(fp_idx), len(fn_idx)))
        if len(fn_idx) > 0:
            fn_images.append((img_id, len(tp_idx), len(fp_idx), len(fn_idx)))
    
    # Visualize sample images from each category
    # True Positives - ALL
    for sample_idx, (img_id, tp_len, fp_len, fn_len) in enumerate(tp_images):
        image = load_image(img_id, config.TRAIN_PNG_DIR)
        if image is not None:
            pred = all_predictions_wbf[image_ids.index(img_id)]
            target = all_targets[image_ids.index(img_id)]
            visualize_image_predictions(
                img_id, image, pred, target,
                f"{tp_dir}/tp_{sample_idx+1:05d}_{img_id}.png",
                config.CLASS_NAMES,
                score_threshold=config.FINAL_CONF_THRESHOLD
            )
    
    # False Positives - ALL
    for sample_idx, (img_id, tp_len, fp_len, fn_len) in enumerate(fp_images):
        image = load_image(img_id, config.TRAIN_PNG_DIR)
        if image is not None:
            pred = all_predictions_wbf[image_ids.index(img_id)]
            target = all_targets[image_ids.index(img_id)]
            visualize_image_predictions(
                img_id, image, pred, target,
                f"{fp_dir}/fp_{sample_idx+1:05d}_{img_id}.png",
                config.CLASS_NAMES,
                score_threshold=config.FINAL_CONF_THRESHOLD
            )
    
    # False Negatives - ALL
    for sample_idx, (img_id, tp_len, fp_len, fn_len) in enumerate(fn_images):
        image = load_image(img_id, config.TRAIN_PNG_DIR)
        if image is not None:
            pred = all_predictions_wbf[image_ids.index(img_id)]
            target = all_targets[image_ids.index(img_id)]
            visualize_image_predictions(
                img_id, image, pred, target,
                f"{fn_dir}/fn_{sample_idx+1:05d}_{img_id}.png",
                config.CLASS_NAMES,
                score_threshold=config.FINAL_CONF_THRESHOLD
            )
    
    logger.info(f"  ✓ Visualized {len(tp_images)} True Positive samples")
    logger.info(f"  ✓ Visualized {len(fp_images)} False Positive samples")
    logger.info(f"  ✓ Visualized {len(fn_images)} False Negative samples")
    
    # Save detailed image analysis
    image_analysis_df = pd.DataFrame(image_analysis_records)
    image_analysis_df = image_analysis_df.sort_values("false_positives", ascending=False)
    image_analysis_df.to_csv(f"{metrics_dir}/image_level_analysis.csv", index=False)
    logger.info(f"  ✓ metrics/image_level_analysis.csv")
    
    # Save detection statistics
    stats_data = {
        "Detection Type": ["True Positives", "False Positives", "False Negatives", "Total Predictions (After WBF)"],
        "Count": [tp_count, fp_count, fn_count, sum(len(p["boxes"]) for p in all_predictions_wbf)],
        "Percentage": [
            f"{tp_count / (tp_count + fp_count + fn_count) * 100:.2f}%" if (tp_count + fp_count + fn_count) > 0 else "0%",
            f"{fp_count / (tp_count + fp_count + fn_count) * 100:.2f}%" if (tp_count + fp_count + fn_count) > 0 else "0%",
            f"{fn_count / (tp_count + fp_count + fn_count) * 100:.2f}%" if (tp_count + fp_count + fn_count) > 0 else "0%",
            "-"
        ]
    }
    pd.DataFrame(stats_data).to_csv(f"{metrics_dir}/detection_statistics.csv", index=False)
    logger.info(f"  ✓ metrics/detection_statistics.csv")
    
    # ====================================================================
    # 5. Detection statistics pie chart
    # ====================================================================
    if tp_count + fp_count + fn_count > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = [tp_count, fp_count, fn_count]
        labels = [f'True Positives\n({tp_count})', f'False Positives\n({fp_count})', f'False Negatives\n({fn_count})']
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        explode = (0.05, 0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 11})
        ax.set_title('Detection Performance Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/detection_breakdown.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ charts/detection_breakdown.png")
    
    # ====================================================================
    # 6. Worst and best predictions
    # ====================================================================
    results_df = pd.DataFrame([
        {"image_id": img_id, "predictions": num_pred, "ground_truth": num_gt}
        for img_id, (num_pred, num_gt) in image_results.items()
    ])
    results_df["diff"] = results_df["predictions"] - results_df["ground_truth"]
    results_df["error_abs"] = np.abs(results_df["diff"])
    results_df = results_df.sort_values("error_abs", ascending=False)
    results_df.to_csv(f"{metrics_dir}/image_results.csv", index=False)
    logger.info(f"  ✓ metrics/image_results.csv")
    
    # Show worst and best predictions
    logger.info("\nWORST PREDICTIONS (top 10):")
    logger.info("-" * 60)
    for _, row in results_df.head(10).iterrows():
        logger.info(f"  {row['image_id']:30s} | Pred: {int(row['predictions']):2d} | GT: {int(row['ground_truth']):2d} | Diff: {int(row['diff']):+2d}")
    
    logger.info("\nBEST PREDICTIONS (top 10):")
    logger.info("-" * 60)
    for _, row in results_df.tail(10).iterrows():
        logger.info(f"  {row['image_id']:30s} | Pred: {int(row['predictions']):2d} | GT: {int(row['ground_truth']):2d} | Diff: {int(row['diff']):+2d}")
    
    # ====================================================================
    # 7. Summary statistics
    # ====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(image_ids)}")
    logger.info(f"Total predictions: {sum(pred_counts)}")
    logger.info(f"Total ground truth boxes: {sum(gt_counts)}")
    logger.info(f"Avg predictions/image: {np.mean(pred_counts):.2f}")
    logger.info(f"Avg GT boxes/image: {np.mean(gt_counts):.2f}")
    logger.info(f"Images with detections: {sum(1 for c in pred_counts if c > 0)}/{len(image_ids)}")
    logger.info(f"Detection error (MAE): {np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts))):.2f}")
    if all_scores:
        logger.info(f"Avg confidence score: {np.mean(all_scores):.4f}")
        logger.info(f"Min confidence score: {np.min(all_scores):.4f}")
        logger.info(f"Max confidence score: {np.max(all_scores):.4f}")
        logger.info(f"Std confidence score: {np.std(all_scores):.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DETECTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"True Positives: {tp_count}")
    logger.info(f"False Positives: {fp_count}")
    logger.info(f"False Negatives: {fn_count}")
    if tp_count + fp_count > 0:
        precision = tp_count / (tp_count + fp_count)
        logger.info(f"Precision: {precision:.4f}")
    if tp_count + fn_count > 0:
        recall = tp_count / (tp_count + fn_count)
        logger.info(f"Recall: {recall:.4f}")
    logger.info("=" * 60)
    
    logger.info(f"\n✓ Comprehensive analysis saved to:")
    logger.info(f"  📊 Charts: {charts_dir}/")
    logger.info(f"  📈 Metrics: {metrics_dir}/")
    logger.info(f"  🖼️  Images: {images_dir}/")
    logger.info(f"     └─ True Positives: {tp_dir}/")
    logger.info(f"     └─ False Positives: {fp_dir}/")
    logger.info(f"     └─ False Negatives: {fn_dir}/")
    logger.info("=" * 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--single", action="store_true", help="Use single model instead of ensemble")
    args = parser.parse_args()
    
    logger = get_logger()
    logger.info("Starting evaluation...")
    evaluate(fold=args.fold, use_ensemble=not args.single)
    logger.info("Evaluation complete!")
