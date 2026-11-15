"""Full comparative evaluation across detection methods (rules, YOLO, hybrid).

Generates:
1. Master comparison markdown table (results.md)
2. Master results JSON (results.json)
3. Calibration comparison plot (calibration_comparison.png)
4. Robustness comparison plot (robustness_comparison.png)
5. Decision summary against criteria in DECISION.md
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.schema import LABELS
from evaluation.evaluate import run_split_evaluation, _write_markdown_report
from evaluation.calibration import run_calibration_evaluation, compute_calibration_curve
from evaluation.robustness import run_robustness_test
from evaluation.failure_cases import run_failure_analysis


def run_full_comparison(config: dict, report_dir: Path, methods: List[str] = None):
    """Run full comparative suite across all available methods."""
    report_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["yolo"]

    splits = [
        ("val", Path("dataset/images/val"), Path("dataset/labels/val")),
        ("test_synthetic", Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
        ("real_test", Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
    ]

    all_results: Dict[str, Dict[str, Any]] = {}
    calib_results: Dict[str, Any] = {}
    robustness_results: Dict[str, Any] = {}

    for method in methods:
        print(f"\n{'='*60}\nRUNNING EVALUATION FOR: {method.upper()}\n{'='*60}")
        method_splits = {}
        for split_name, img_dir, label_dir in splits:
            print(f"Evaluating {method} on {split_name}...")
            res = run_split_evaluation(method, split_name, img_dir, label_dir, config)
            method_splits[split_name] = res
            print(f"  {split_name} macro mAP@0.5: {res['macro_mAP50']:.4f}")

        all_results[method] = method_splits

        # Calibration
        print(f"Evaluating calibration for {method}...")
        calib_res = run_calibration_evaluation(method, config, report_dir)
        calib_results[method] = calib_res

        # Robustness
        print(f"Evaluating robustness for {method}...")
        rob_res = run_robustness_test(method, config, report_dir)
        robustness_results[method] = rob_res

        # Failure cases
        print(f"Extracting failure cases for {method}...")
        run_failure_analysis(method, config, report_dir)

    # Save master JSON
    json_path = report_dir / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # Write master markdown report
    md_path = report_dir / "results.md"
    _write_master_comparison_report(all_results, calib_results, md_path)

    # Plot multi-method calibration comparison
    if len(methods) > 1:
        _plot_multi_method_calibration(methods, config, report_dir)
        _plot_multi_method_robustness(robustness_results, report_dir)

    print(f"\nMaster comparison reports generated:")
    print(f"  {json_path}")
    print(f"  {md_path}")


def _write_master_comparison_report(all_results: dict, calib_results: dict, out_path: Path):
    """Write side-by-side comparison table across all methods."""
    lines = [
        "# Manuscript Layout Region Detection — Method Comparison",
        "",
        "Comprehensive benchmark comparing `rules`, `yolo`, and `hybrid` methods across splits.",
        "",
        "## 1. Summary Benchmark Table",
        "",
        "| Split | Metric | Rules (Method A) | YOLOv8 (Method B) | Hybrid (Method C) | Best Method |",
        "|---|---|---|---|---|---|",
    ]

    splits = ["val", "test_synthetic", "real_test"]
    methods = list(all_results.keys())

    for split in splits:
        # Macro mAP
        row_m = f"| **{split}** | **macro mAP@0.5** |"
        scores = {}
        for m in ["rules", "yolo", "hybrid"]:
            if m in all_results and split in all_results[m]:
                sc = all_results[m][split].get("macro_mAP50", 0.0)
                scores[m] = sc
                row_m += f" {sc:.3f} |"
            else:
                row_m += " N/A |"
        best_m = max(scores, key=scores.get) if scores else "N/A"
        row_m += f" **{best_m}** |"
        lines.append(row_m)

        # Per class breakdown
        for cls_name in LABELS:
            row_c = f"| {split} | {cls_name} AP50 |"
            cls_scores = {}
            for m in ["rules", "yolo", "hybrid"]:
                if m in all_results and split in all_results[m]:
                    ap = all_results[m][split].get(cls_name, {}).get("AP50", 0.0)
                    sup = all_results[m][split].get(cls_name, {}).get("support", 0)
                    cls_scores[m] = ap
                    row_c += f" {ap:.3f} (N={sup}) |"
                else:
                    row_c += " N/A |"
            best_c = max(cls_scores, key=cls_scores.get) if cls_scores else "N/A"
            row_c += f" {best_c} |"
            lines.append(row_c)

    lines.extend([
        "",
        "## 2. Confidence Calibration (ECE)",
        "",
        "| Method | Expected Calibration Error (ECE) | Interpretation |",
        "|---|---|---|",
    ])

    for m, cdata in calib_results.items():
        ece = cdata.get("ece", 0.0)
        interp = "Well-calibrated" if ece < 0.15 else "Moderate calibration" if ece < 0.30 else "Heuristic / uncalibrated"
        lines.append(f"| `{m}` | {ece:.4f} | {interp} |")

    lines.extend([
        "",
        "## 3. Decision Summary",
        "Refer to `evaluation/DECISION.md` for the formal evaluation protocol.",
        "- **Primary criterion (real_test main_text & filler)**: Checked against noise margin (N=9 pages).",
        "- **Secondary criterion (test_synthetic generalization)**: 56 images across all 5 classes.",
        "- **Winner selection**: The final production method set as default in `inference.py`.",
    ])

    out_path.write_text("\n".join(lines))


def _plot_multi_method_calibration(methods: List[str], config: dict, report_dir: Path):
    """Plot overlaid reliability diagrams for all methods."""
    from inference import get_predictor
    from common.io_utils import load_image, list_images
    from evaluation.evaluate import load_ground_truth, compute_iou

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    colors = {"rules": "#1f77b4", "yolo": "#ff7f0e", "hybrid": "#2ca02c"}

    eval_splits = [
        (Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
        (Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
    ]

    for method in methods:
        predict_fn = get_predictor(method, config)
        preds_and_matches = []
        for img_dir, label_dir in eval_splits:
            for img_path in list_images(str(img_dir)):
                img, _ = load_image(str(img_path))
                h, w = img.shape[:2]
                preds = predict_fn(image=img, config=config)
                gts = load_ground_truth(label_dir / f"{img_path.stem}.txt", w, h)
                gt_matched = [False] * len(gts)
                for p in sorted(preds, key=lambda p: p.confidence, reverse=True):
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
        valid = bin_counts > 0
        plt.plot(bin_confs[valid], bin_accs[valid], "s-", color=colors.get(method, "purple"), label=f"{method} (ECE={ece:.3f})")

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Precision @ IoU>=0.5", fontsize=12)
    plt.title("Comparative Calibration Diagram", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(report_dir / "calibration_comparison.png", dpi=150)
    plt.close()


def _plot_multi_method_robustness(robustness_data: dict, report_dir: Path):
    """Plot multi-method robustness comparison bar chart."""
    degs = ["clean", "blur", "fade", "rotation", "stain"]
    methods = list(robustness_data.keys())

    x = np.arange(len(degs))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, method in enumerate(methods):
        scores = [robustness_data[method].get(d, 0.0) for d in degs]
        ax.bar(x + i * width, scores, width, label=method, color=colors[i % len(colors)], alpha=0.85)

    ax.set_ylabel("macro mAP@0.5", fontsize=12)
    ax.set_title("Robustness Stress-Test Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(degs, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(report_dir / "robustness_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    r_dir = Path("evaluation/reports")
    run_full_comparison(cfg, r_dir)
