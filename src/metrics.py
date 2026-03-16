import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_map(predictions, targets, iou_threshold=0.4):
    class_preds = defaultdict(list)
    class_gts = defaultdict(lambda: defaultdict(list))

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        boxes = pred.get("boxes", [])
        scores = pred.get("scores", [])
        labels = pred.get("labels", [])

        for box, score, label in zip(boxes, scores, labels):
            class_preds[int(label)].append((img_idx, box, float(score)))

        gt_boxes = target.get("boxes", [])
        gt_labels = target.get("labels", [])

        for box, label in zip(gt_boxes, gt_labels):
            class_gts[int(label)][img_idx].append(box)

    if not class_preds:
        return 0.0

    aps = []
    for class_id in class_gts.keys():
        preds = sorted(class_preds[class_id], key=lambda x: x[2], reverse=True)

        if not preds:
            continue

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = {
            img_id: np.zeros(len(boxes))
            for img_id, boxes in class_gts[class_id].items()
        }

        for i, (img_id, pred_box, score) in enumerate(preds):
            if img_id not in class_gts[class_id]:
                fp[i] = 1
                continue

            gt_boxes = class_gts[class_id][img_id]
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            best_iou = 0.0
            best_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx != -1:
                if not gt_matched[img_id][best_idx]:
                    tp[i] = 1
                    gt_matched[img_id][best_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        n_pos = sum(len(boxes) for boxes in class_gts[class_id].values())

        if n_pos == 0:
            continue

        recall = tp_cumsum / n_pos
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        ap = (
            sum(
                np.max(precision[recall >= t])
                for t in np.arange(0, 1.1, 0.1)
                if np.any(recall >= t)
            )
            / 11.0
        )
        aps.append(ap)

    return np.mean(aps) if aps else 0.0
