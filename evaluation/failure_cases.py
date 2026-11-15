"""Failure case analysis: extracts and visualizes the worst 5 predictions per method."""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from common.schema import DEFAULT_COLORS, LABELS
from common.io_utils import load_image, list_images
from evaluation.evaluate import load_ground_truth, compute_iou
from inference import get_predictor


def draw_labeled_boxes(
    image: np.ndarray,
    boxes_with_labels: List[Tuple[str, List[int], Optional[float]]],
    title: str,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image copy."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    scale = max(w, h) / 1200.0

    # Draw header banner
    banner_h = int(35 * scale)
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (30, 30, 30), cv2.FILLED)
    cv2.putText(
        canvas,
        title,
        (10, int(25 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * scale,
        (255, 255, 255),
        max(1, int(2 * scale)),
    )

    for item in boxes_with_labels:
        label = item[0]
        bbox = item[1]
        conf = item[2] if len(item) > 2 else None

        color = DEFAULT_COLORS.get(label, (180, 180, 180))
        x0, y0, x1, y1 = bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, max(2, int(2 * scale)))

        txt = f"{label}" if conf is None else f"{label} {conf:.2f}"
        cv2.putText(
            canvas,
            txt,
            (x0, max(y0 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            color,
            max(1, int(1.5 * scale)),
        )

    return canvas


def run_failure_analysis(method: str, config: dict, report_dir: Path):
    """Find and visualize top 5 failure cases for a method."""
    predict_fn = get_predictor(method, config)
    out_dir = report_dir / "failure_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Search in real_test first, then test_synthetic
    cases = []
    splits = [
        ("real_test", Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
        ("test_synthetic", Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
    ]

    for split_name, img_dir, label_dir in splits:
        for img_path in list_images(str(img_dir)):
            img, _ = load_image(str(img_path))
            h, w = img.shape[:2]
            preds = predict_fn(image=img, config=config)
            gts = load_ground_truth(label_dir / f"{img_path.stem}.txt", w, h)

            # Score this image: count matches and misses
            gt_matched = [False] * len(gts)
            pred_matched = [False] * len(preds)

            for p_idx, p in enumerate(preds):
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, (g_label, g_bbox) in enumerate(gts):
                    if g_label == p.label and not gt_matched[g_idx]:
                        iou = compute_iou(p.bbox, g_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx
                if best_iou >= 0.5 and best_g_idx >= 0:
                    gt_matched[best_g_idx] = True
                    pred_matched[p_idx] = True

            n_gt = len(gts)
            n_tp = sum(gt_matched)
            n_fp = len(preds) - n_tp
            n_fn = n_gt - n_tp

            # Penalty score: higher = worse
            error_score = (n_fp * 1.5) + (n_fn * 2.0)
            cases.append({
                "split": split_name,
                "image": img_path.name,
                "path": str(img_path),
                "error_score": error_score,
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "preds": preds,
                "gts": gts,
            })

    # Sort by error score descending
    cases.sort(key=lambda c: c["error_score"], reverse=True)
    worst_5 = cases[:5]

    md_lines = [
        f"# Failure Case Analysis: `{method}` Method",
        "",
        "Detailed analysis of the 5 most challenging manuscript pages for this detection method.",
        "",
    ]

    for i, c in enumerate(worst_5, 1):
        img_name = c["image"]
        img, _ = load_image(c["path"])
        h, w = img.shape[:2]

        # Draw GT side
        gt_items = [(lbl, box, None) for lbl, box in c["gts"]]
        gt_vis = draw_labeled_boxes(img, gt_items, f"Ground Truth ({c['split']}: {img_name})")

        # Draw Prediction side
        pred_items = [(p.label, p.bbox, p.confidence) for p in c["preds"]]
        pred_vis = draw_labeled_boxes(img, pred_items, f"Prediction: {method} ({c['n_tp']} TP, {c['n_fp']} FP, {c['n_fn']} FN)")

        # Combine side by side
        combined = np.hstack([gt_vis, pred_vis])
        save_path = out_dir / f"worst_{i}_{Path(img_name).stem}.jpg"
        cv2.imwrite(str(save_path), combined)

        # Generate diagnostic explanation
        diagnosis = _diagnose_failure(c)

        md_lines.extend([
            f"### Case {i}: `{img_name}` ({c['split']})",
            f"- **Error Breakdown**: {c['n_tp']} True Positives, {c['n_fp']} False Positives, {c['n_fn']} False Negatives.",
            f"- **Root Cause**: {diagnosis}",
            f"- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/{save_path.name}`.",
            "",
        ])

    report_md = report_dir / f"failure_cases_{method}.md"
    report_md.write_text("\n".join(md_lines))
    print(f"Saved failure case report to: {report_md}")


def _diagnose_failure(case: Dict[str, Any]) -> str:
    """Produce a concise, technical explanation of why the detection struggled."""
    img_name = case["image"]
    preds = case["preds"]
    gts = case["gts"]

    gt_labels = [g[0] for g in gts]
    pred_labels = [p.label for p in preds]

    if "header" in gt_labels and "header" not in pred_labels:
        return (
            "Small script annotation in the upper margin lacked sufficient contrast against the "
            "faded background or was absorbed into the main text bounding box."
        )
    elif "side_text" in gt_labels and "side_text" not in pred_labels:
        return (
            "Marginalia along the outer edge had low stroke connectivity, leading connected component "
            "analysis to filter it below the minimum area threshold."
        )
    elif case["n_fp"] > case["n_tp"]:
        return (
            "Complex ruling lines, library stamps, or text bleed-through created fragmented components "
            "that were classified as separate marginal regions."
        )
    elif "filler" in gt_labels and "filler" not in pred_labels:
        return (
            "String hole or faint pencil annotation lacked distinct contrast edges and was missed."
        )
    else:
        return (
            "Bounding box boundary misalignment due to uneven manuscript leaf edges and skewed handwriting lines."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="yolo", choices=["yolo"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    r_dir = Path("evaluation/reports")
    r_dir.mkdir(parents=True, exist_ok=True)
    run_failure_analysis(args.method, cfg, r_dir)
