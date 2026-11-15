"""Unit tests for the evaluation metric calculations."""
import numpy as np
from evaluation.evaluate import compute_iou, compute_ap, evaluate_predictions


def test_compute_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 0, 15, 10]
    # Intersection: 5*10 = 50. Union: 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    assert abs(compute_iou(box1, box2) - 1.0 / 3.0) < 1e-4

    # No overlap
    box3 = [20, 20, 30, 30]
    assert compute_iou(box1, box3) == 0.0

    # Perfect overlap
    assert compute_iou(box1, box1) == 1.0


def test_compute_ap():
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ap = compute_ap(recalls, precisions)
    assert abs(ap - 1.0) < 1e-4


def test_evaluate_predictions_mock():
    from common.schema import Prediction

    preds = [{
        "image": "img1",
        "preds": [
            Prediction("main_text", 0.9, [10, 10, 100, 100]),
            Prediction("header", 0.8, [10, 0, 100, 10]),
        ],
    }]

    gts = [{
        "image": "img1",
        "gt": [
            ("main_text", [10, 10, 100, 100]),
            ("header", [10, 0, 100, 10]),
        ],
    }]

    metrics = evaluate_predictions(preds, gts, iou_threshold=0.5)
    assert metrics["main_text"]["AP50"] == 1.0
    assert metrics["header"]["AP50"] == 1.0
    assert metrics["macro_mAP50"] == 1.0
