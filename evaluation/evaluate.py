"""Evaluation suite for manuscript layout region detection.

Computes precision, recall, AP@0.5, and macro mAP@0.5 per method and split.
Computes IoU and PR curves from scratch.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common.schema import LABELS, Prediction
from common.io_utils import load_image, list_images
import yaml


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union (IoU) of two boxes [x_min, y_min, x_max, y_max]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / float(max(union, 1e-6))


def load_ground_truth(label_file: Path, img_w: int, img_h: int) -> List[Tuple[str, List[int]]]:
    """Load YOLO-format ground truth annotations and convert to pixel coordinates."""
    if not label_file.exists():
        return []

    gt = []
    lines = label_file.read_text().strip().split("\n")
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        if cid < 0 or cid >= len(LABELS):
            continue
        label = LABELS[cid]
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x0 = int(round((cx - bw / 2.0) * img_w))
        y0 = int(round((cy - bh / 2.0) * img_h))
        x1 = int(round((cx + bw / 2.0) * img_w))
        y1 = int(round((cy + bh / 2.0) * img_h))
        gt.append((label, [max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)]))

    return gt


def compute_ap(
    recalls: np.ndarray, precisions: np.ndarray
) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_interp = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p_interp / 11.0
    return float(ap)


def evaluate_predictions(
    all_predictions: List[Dict[str, Any]],
    all_ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate predictions against ground truths for a dataset split.

    Args:
        all_predictions: List of dicts per image: {'image': stem, 'preds': [Prediction, ...]}
        all_ground_truths: List of dicts per image: {'image': stem, 'gt': [(label, bbox), ...]}
        iou_threshold: IoU overlap threshold for matching (default 0.5).

    Returns:
        Dictionary of metrics per class and overall.
    """
    gt_by_img = {item["image"]: item["gt"] for item in all_ground_truths}
    pred_by_img = {item["image"]: item["preds"] for item in all_predictions}

    results = {}
    class_aps = []

    for target_class in LABELS:
        # Collect all predictions for this class across all images
        class_preds = []
        total_gt = 0

        # Match status tracker for each ground truth box
        gt_matched = {img: [False] * len(gt_by_img.get(img, [])) for img in gt_by_img}

        for img, gts in gt_by_img.items():
            for g_label, _ in gts:
                if g_label == target_class:
                    total_gt += 1

        for img, preds in pred_by_img.items():
            for p in preds:
                if p.label == target_class:
                    class_preds.append((img, p.confidence, p.bbox))

        # Sort predictions by confidence descending
        class_preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, (img, conf, p_bbox) in enumerate(class_preds):
            gts = gt_by_img.get(img, [])
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, (g_label, g_bbox) in enumerate(gts):
                if g_label == target_class:
                    iou = compute_iou(p_bbox, g_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not gt_matched[img][best_gt_idx]:
                    tp[i] = 1.0
                    gt_matched[img][best_gt_idx] = True
                else:
                    fp[i] = 1.0  # duplicate detection
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / float(max(total_gt, 1))
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

        ap = compute_ap(recalls, precisions) if total_gt > 0 else 0.0
        final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

        results[target_class] = {
            "AP50": round(ap, 4),
            "precision": round(final_precision, 4),
            "recall": round(final_recall, 4),
            "support": total_gt,
            "detections": len(class_preds),
        }

        # Include classes with non-zero support in mAP calculation
        if total_gt > 0:
            class_aps.append(ap)

    macro_map = float(np.mean(class_aps)) if class_aps else 0.0
    results["macro_mAP50"] = round(macro_map, 4)

    return results


def run_split_evaluation(
    method: str,
    split_name: str,
    img_dir: Path,
    label_dir: Path,
    config: dict,
) -> Dict[str, Any]:
    """Run full evaluation of a method on a dataset split."""
    from inference import get_predictor

    predict_fn = get_predictor(method, config)
    img_files = sorted(list_images(str(img_dir)))

    all_preds = []
    all_gts = []

    for img_path in img_files:
        img, _ = load_image(str(img_path))
        h, w = img.shape[:2]
        stem = img_path.stem

        # Run predictions
        preds = predict_fn(image=img, config=config)
        all_preds.append({"image": stem, "preds": preds})

        # Load GT
        gt_file = label_dir / f"{stem}.txt"
        gt = load_ground_truth(gt_file, w, h)
        all_gts.append({"image": stem, "gt": gt})

    return evaluate_predictions(all_preds, all_gts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate manuscript layout detectors")
    parser.add_argument("--method", choices=["yolo"], default="yolo")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    splits = [
        ("val", Path("dataset/images/val"), Path("dataset/labels/val")),
        ("test_synthetic", Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
        ("real_test", Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
    ]

    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    print(f"==================================================")
    print(f"Evaluating Method: {args.method}")
    print(f"==================================================")

    for split_name, img_dir, label_dir in splits:
        print(f"Evaluating {split_name} ({len(list(img_dir.glob('*.jpg')))} images)...")
        res = run_split_evaluation(args.method, split_name, img_dir, label_dir, config)
        summary[split_name] = res
        print(f"  mAP@0.5: {res['macro_mAP50']:.4f}")
        for cls_name in LABELS:
            c_res = res[cls_name]
            print(f"    {cls_name:10s}: AP50={c_res['AP50']:.4f}  P={c_res['precision']:.4f}  R={c_res['recall']:.4f}  (support: {c_res['support']})")

    # Save results JSON
    json_path = report_dir / "results.json"
    existing = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except Exception:
            pass
    existing[args.method] = summary
    json_path.write_text(json.dumps(existing, indent=2))

    # Write Markdown results table
    md_path = report_dir / "results.md"
    _write_markdown_report(existing, md_path)
    print(f"\nSaved evaluation reports to:")
    print(f"  {json_path}")
    print(f"  {md_path}")


def _write_markdown_report(all_results: dict, out_path: Path):
    """Write comprehensive markdown evaluation table."""
    lines = [
        "# Manuscript Layout Region Detection — Evaluation Results",
        "",
        "Evaluation across dataset splits using **AP@0.5** and **macro mAP@0.5**.",
        "",
    ]

    for method, splits in all_results.items():
        lines.append(f"## Method: `{method}`")
        lines.append("")
        lines.append("| Split | macro mAP@0.5 | header AP (N) | footer AP (N) | main_text AP (N) | side_text AP (N) | filler AP (N) |")
        lines.append("|---|---|---|---|---|---|---|")

        for split_name, res in splits.items():
            m_map = res.get("macro_mAP50", 0.0) or 0.0
            def _fmt(cls_res):
                if cls_res is None:
                    return "N/A (0)"
                ap = cls_res.get("AP50", 0.0) or 0.0
                sup = cls_res.get("support", 0) or 0
                return f"{ap:.3f} ({sup})"
            h_info = _fmt(res.get('header'))
            f_info = _fmt(res.get('footer'))
            m_info = _fmt(res.get('main_text'))
            s_info = _fmt(res.get('side_text'))
            fl_info = _fmt(res.get('filler'))
            lines.append(f"| **{split_name}** | **{m_map:.3f}** | {h_info} | {f_info} | {m_info} | {s_info} | {fl_info} |")

        lines.append("")

    lines.append("*Note: `real_test` results are indicative with low support (9 pages total, draft boxes).*")
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
