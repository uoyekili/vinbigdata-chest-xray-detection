import numpy as np
from ensemble_boxes import weighted_boxes_fusion as wbf_lib


def weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, iou_thr, skip_box_thr, weights=None
):
    if len(boxes_list) == 0 or all(len(b) == 0 for b in boxes_list):
        return np.zeros((0, 4)), np.array([]), np.array([])

    if weights is None:
        weights = [1] * len(boxes_list)

    boxes_for_fusion = []
    scores_for_fusion = []
    labels_for_fusion = []

    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        if len(boxes) == 0:
            boxes_for_fusion.append([])
            scores_for_fusion.append([])
            labels_for_fusion.append([])
        else:
            boxes_converted = [[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes]
            boxes_for_fusion.append(boxes_converted)
            scores_for_fusion.append(
                scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
            )
            labels_for_fusion.append(
                labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
            )

    try:
        fused = wbf_lib(
            boxes_for_fusion,
            scores_for_fusion,
            labels_for_fusion,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        if len(fused[0]) == 0:
            return np.zeros((0, 4)), np.array([]), np.array([])

        return (
            np.array(fused[0]),
            np.array(fused[1]),
            np.array(fused[2], dtype=np.int64),
        )
    except:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=np.int64)
