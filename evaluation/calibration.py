import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.schema import LABELS
from common.io_utils import load_image, list_images
from evaluation.evaluate import load_ground_truth, compute_iou
from inference import get_predictor


def compute_calibration_curve(
    predictions_and_matches: List[Tuple[float, bool]],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute reliability diagram data and Expected Calibration Error (ECE).

    Args:
        predictions_and_matches: List of (confidence_score, is_correct) tuples.
        n_bins: Number of confidence bins between 0.0 and 1.0.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts, ece)
    """
    if not predictions_and_matches:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), 0.0

    confs = np.array([item[0] for item in predictions_and_matches])
    corrects = np.array([float(item[1]) for item in predictions_and_matches])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    total_samples = len(confs)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= low) & (confs < high) if i < n_bins - 1 else (confs >= low) & (confs <= high)
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            avg_conf = float(np.mean(confs[mask]))
            acc = float(np.mean(corrects[mask]))
            bin_confs.append(avg_conf)
            bin_accs.append(acc)
            ece += (count / total_samples) * abs(acc - avg_conf)
        else:
            bin_confs.append((low + high) / 2.0)
            bin_accs.append(0.0)

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts), float(ece)


def run_calibration_evaluation(method: str, config: dict, report_dir: Path):
    """Run calibration evaluation and save reliability diagram plot."""
    predict_fn = get_predictor(method, config)

    # Use test_synthetic and real_test
    eval_splits = [
        (Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
        (Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
    ]

    preds_and_matches = []

    for img_dir, label_dir in eval_splits:
        for img_path in list_images(str(img_dir)):
            img, _ = load_image(str(img_path))
            h, w = img.shape[:2]
            preds = predict_fn(image=img, config=config)
            gts = load_ground_truth(label_dir / f"{img_path.stem}.txt", w, h)

            gt_matched = [False] * len(gts)
            # Sort predictions by confidence descending
            sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)

            for p in sorted_preds:
                best_iou = 0.0
                best_idx = -1
                for g_idx, (g_label, g_bbox) in enumerate(gts):
                    if g_label == p.label and not gt_matched[g_idx]:
                        iou = compute_iou(p.bbox, g_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = g_idx

                is_correct = (best_iou >= 0.5 and best_idx >= 0)
                if is_correct:
                    gt_matched[best_idx] = True
                preds_and_matches.append((p.confidence, is_correct))

    bin_confs, bin_accs, bin_counts, ece = compute_calibration_curve(preds_and_matches, n_bins=10)

    # Plot reliability diagram
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    valid_mask = bin_counts > 0
    plt.plot(bin_confs[valid_mask], bin_accs[valid_mask], "s-", color="#1f77b4", label=f"{method} (ECE = {ece:.3f})")
    plt.bar(bin_confs[valid_mask], bin_accs[valid_mask], width=0.08, alpha=0.3, color="#1f77b4", edgecolor="#1f77b4")

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Fraction Correct (Precision @ IoU>=0.5)", fontsize=12)
    plt.title(f"Reliability Diagram: {method.upper()} Method", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    out_plot = report_dir / f"calibration_{method}.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Calibration plot saved: {out_plot} (ECE: {ece:.4f})")

    return {"method": method, "ece": round(ece, 4), "total_predictions": len(preds_and_matches)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="yolo", choices=["yolo"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    r_dir = Path("evaluation/reports")
    r_dir.mkdir(parents=True, exist_ok=True)
    run_calibration_evaluation(args.method, cfg, r_dir)
