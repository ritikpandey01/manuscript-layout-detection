import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.io_utils import load_image, list_images
from evaluation.evaluate import load_ground_truth, evaluate_predictions
from inference import get_predictor


def apply_degradation(image: np.ndarray, deg_type: str, rng: np.random.RandomState) -> np.ndarray:
    """Apply a synthetic degradation to simulate real-world manuscript damage."""
    h, w = image.shape[:2]
    out = image.copy()

    if deg_type == "blur":
        # Gaussian blur
        ksize = 9
        out = cv2.GaussianBlur(out, (ksize, ksize), 2.5)

    elif deg_type == "fade":
        # Contrast reduction / ink fading
        alpha = 0.4  # contrast reduction
        beta = 120   # brightening background
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    elif deg_type == "rotation":
        # Slight page tilt (+/- 3 degrees)
        angle = rng.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif deg_type == "stain":
        # Simulated ink/water stain blotches
        stain_mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(2):
            cx = rng.randint(int(0.2 * w), int(0.8 * w))
            cy = rng.randint(int(0.2 * h), int(0.8 * h))
            axes = (rng.randint(w // 10, w // 4), rng.randint(h // 6, h // 3))
            cv2.ellipse(stain_mask, (cx, cy), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        stain_mask = cv2.GaussianBlur(stain_mask, (0, 0), sigmaX=15, sigmaY=15)
        stain_color = np.array([30, 60, 90], dtype=np.float32)  # brownish stain
        out_f = out.astype(np.float32)
        factor = np.clip(stain_mask[..., None] * 0.45, 0, 0.7)
        out_f = out_f * (1 - factor) + stain_color * factor
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    return out


def run_robustness_test(method: str, config: dict, report_dir: Path):
    """Run robustness evaluation across blur, fade, rotation, and stain."""
    predict_fn = get_predictor(method, config)
    rng = np.random.RandomState(42)

    # Use real_test and test_synthetic
    eval_images = []
    for split_dir, l_dir in [
        (Path("dataset/real_test/images"), Path("dataset/real_test/labels_draft")),
        (Path("dataset/images/test_synthetic"), Path("dataset/labels/test_synthetic")),
    ]:
        for p in list_images(str(split_dir)):
            eval_images.append((p, l_dir / f"{p.stem}.txt"))

    degradations = ["clean", "blur", "fade", "rotation", "stain"]
    scores = {}

    print(f"Running robustness test for {method} across {len(eval_images)} images...")

    for deg in degradations:
        all_preds = []
        all_gts = []

        for img_path, gt_file in eval_images:
            img, _ = load_image(str(img_path))
            h, w = img.shape[:2]

            deg_img = apply_degradation(img, deg, rng) if deg != "clean" else img
            preds = predict_fn(image=deg_img, config=config)
            gts = load_ground_truth(gt_file, w, h)

            all_preds.append({"image": img_path.stem, "preds": preds})
            all_gts.append({"image": img_path.stem, "gt": gts})

        metrics = evaluate_predictions(all_preds, all_gts)
        m_map = metrics.get("macro_mAP50", 0.0)
        scores[deg] = m_map
        print(f"  {deg:10s} -> macro mAP@0.5 = {m_map:.4f}")

    # Plot drop chart
    clean_score = scores["clean"]
    degs_to_plot = ["blur", "fade", "rotation", "stain"]
    drops = [clean_score - scores[d] for d in degs_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(degs_to_plot, [scores[d] for d in degs_to_plot], color="#2ca02c", alpha=0.8, label="Degraded mAP50")
    ax.axhline(clean_score, color="red", linestyle="--", linewidth=1.5, label=f"Clean Baseline ({clean_score:.3f})")

    for bar, d in zip(bars, degs_to_plot):
        drop_pct = ((clean_score - scores[d]) / max(clean_score, 1e-4)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{scores[d]:.3f}\n(-{drop_pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim([0.0, max(clean_score * 1.3, 1.0)])
    ax.set_ylabel("macro mAP@0.5", fontsize=12)
    ax.set_title(f"Robustness Stress-Test: {method.upper()} Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_chart = report_dir / f"robustness_{method}.png"
    plt.savefig(out_chart, dpi=150)
    plt.close()
    print(f"Robustness chart saved: {out_chart}")

    # Save scores to JSON
    json_path = report_dir / f"robustness_{method}.json"
    json_path.write_text(json.dumps(scores, indent=2))

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="yolo", choices=["yolo"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    r_dir = Path("evaluation/reports")
    r_dir.mkdir(parents=True, exist_ok=True)
    run_robustness_test(args.method, cfg, r_dir)
