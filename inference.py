#!/usr/bin/env python3
"""Manuscript Layout Region Detection — Production CLI entry point.

Usage:
    python inference.py --input ./data/test_images --output ./results
    python inference.py --input ./data/test_images/page.jpg --output ./results
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Callable, Dict, Any

import yaml

from common.schema import ImageResult, Prediction
from common.io_utils import load_image, list_images, ensure_output_dir
from common.output import save_annotated_image, save_json, save_combined_json


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        print(f"Warning: Config file not found: {p}, using defaults")
        return {"method": "yolo", "review_threshold": 0.4}
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_predictor(*args, **kwargs) -> Callable:
    """Import and return the YOLO prediction function."""
    from method_yolo.predict import predict
    return predict


def process_image(
    image_path: Path,
    predict_fn: Callable,
    config: dict,
    method: str,
    annotated_dir: Path,
    json_dir: Path,
) -> ImageResult:
    """Process a single image through the layout detection pipeline."""
    image, resolved_path = load_image(str(image_path))
    h, w = image.shape[:2]

    # Run detection
    predictions = predict_fn(image, config)

    # Apply review threshold
    review_threshold = config.get("review_threshold", 0.4)
    for pred in predictions:
        if pred.confidence < review_threshold:
            pred.needs_review = True

    # Build result schema
    result = ImageResult(
        image=resolved_path.name,
        width=w,
        height=h,
        method=method,
        predictions=predictions,
    )

    # Save outputs
    stem = resolved_path.stem
    save_annotated_image(
        image, predictions, str(annotated_dir / f"{stem}_annotated.png"),
        colors=_parse_colors(config),
    )
    save_json(result, str(json_dir / f"{stem}.json"))

    return result


def _parse_colors(config: dict) -> dict:
    """Parse color config lists to tuples."""
    colors_cfg = config.get("colors", {})
    return {k: tuple(v) if isinstance(v, list) else v for k, v in colors_cfg.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection (YOLOv8 Production Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --input ./data/test_images --output ./results
  python inference.py --input ./data/test_images/02_a.jpg --output ./results
""",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single image or a directory of images",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--method", default="yolo", choices=["yolo"],
        help="Detection method (default: yolo)",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    method = "yolo"

    # Verify weights existence before batch processing
    weights_path = Path(config.get("yolo", {}).get("weights_path", "method_yolo/weights/best.pt"))
    if not weights_path.exists():
        print(f"\nERROR: YOLO weights not found at: {weights_path}")
        print("Please check method_yolo/KAGGLE_STEPS.md on how to place weights.")
        sys.exit(1)

    print("Manuscript Layout Region Detection (YOLOv8)")
    print(f"  Method: {method}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Weights: {weights_path}\n")

    # Resolve input images
    try:
        image_paths = list_images(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process")

    # Setup output directories
    base_dir, annotated_dir, json_dir = ensure_output_dir(args.output)

    # Load predictor
    predict_fn = get_predictor(config)

    # Process images
    results: List[ImageResult] = []
    start_time = time.time()

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {img_path.name} ... ", end="", flush=True)
        try:
            result = process_image(
                img_path, predict_fn, config, method, annotated_dir, json_dir
            )
            results.append(result)
            n_preds = len(result.predictions)
            print(f"{n_preds} region(s) detected")
        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - start_time

    # Save combined predictions
    combined_path = save_combined_json(results, str(base_dir / "predictions.json"))

    # Summary
    print(f"\n{'='*50}")
    print(f"Done! Processed {len(results)}/{len(image_paths)} images in {elapsed:.1f}s")
    print(f"  Annotated images: {annotated_dir}/")
    print(f"  JSON results:     {json_dir}/")
    print(f"  Combined JSON:    {combined_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
