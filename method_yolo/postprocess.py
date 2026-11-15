"""Post-processing for YOLO layout region detections.

Performs:
1. Cross-class NMS to resolve overlapping detections
2. Minimum box area filtering
3. Clipping bounding boxes to page boundaries and image dimensions
4. Converting Ultralytics results to Prediction objects
"""
from typing import List, Dict, Any, Optional
import numpy as np

from common.schema import Prediction, LABELS
from common.page_boundary import clip_bbox_to_boundary


def postprocess_yolo_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    page_boundary: List[int],
    img_w: int,
    img_h: int,
    config: Dict[str, Any],
) -> List[Prediction]:
    """Post-process raw YOLO detections into validated Prediction objects.

    Args:
        boxes: (N, 4) array of [x_min, y_min, x_max, y_max] in pixels.
        scores: (N,) array of confidence scores in [0, 1].
        class_ids: (N,) array of integer class IDs.
        page_boundary: [x_min, y_min, x_max, y_max] estimated page boundary.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        config: YOLO configuration dict.

    Returns:
        List of filtered and clipped Prediction objects.
    """
    if len(boxes) == 0:
        return []

    min_area = config.get("min_box_area", 200)
    review_thresh = config.get("review_threshold", 0.4)
    iou_thresh = config.get("iou_nms", 0.45)

    # 1. Apply cross-class Non-Maximum Suppression (NMS)
    keep_indices = _cross_class_nms(boxes, scores, iou_thresh)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]

    predictions: List[Prediction] = []

    for box, score, cid in zip(boxes, scores, class_ids):
        cid = int(cid)
        if cid < 0 or cid >= len(LABELS):
            continue
        label = LABELS[cid]

        # Clip box to page boundary and image
        raw_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        clipped = clip_bbox_to_boundary(raw_box, page_boundary, img_w, img_h)
        if clipped is None:
            continue

        bw = clipped[2] - clipped[0]
        bh = clipped[3] - clipped[1]
        area = bw * bh
        if area < min_area:
            continue

        conf = float(score)
        pred = Prediction(
            label=label,
            confidence=round(conf, 4),
            bbox=clipped,
            reason="yolo_detection",
            needs_review=conf < review_thresh,
        )
        predictions.append(pred)

    return predictions


def _cross_class_nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """Perform Non-Maximum Suppression across all classes to eliminate heavily overlapping boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
