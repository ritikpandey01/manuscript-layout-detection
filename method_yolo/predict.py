"""Inference predictor for YOLO-based manuscript layout detection (METHOD B)."""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from common.schema import Prediction
from common.page_boundary import estimate_page_boundary
from method_yolo.postprocess import postprocess_yolo_detections

# Cached model instance
_MODEL = None
_LOADED_WEIGHTS = None


def get_model(weights_path: str):
    """Load and cache the YOLO model.

    Raises:
        FileNotFoundError: If weights file does not exist.
        ImportError: If ultralytics is not installed.
    """
    global _MODEL, _LOADED_WEIGHTS

    wp = Path(weights_path).resolve()
    if not wp.exists():
        err_msg = (
            f"\n{'='*70}\n"
            f"ERROR: YOLO weights not found at: {weights_path}\n\n"
            f"To use method_yolo:\n"
            f"1. Run the training notebook method_yolo/kaggle_train.ipynb on Kaggle.\n"
            f"   (See method_yolo/KAGGLE_STEPS.md for step-by-step instructions)\n"
            f"2. Download outputs.zip and extract best.pt into:\n"
            f"   {wp.parent}/best.pt\n"
            f"3. Re-run inference with --method yolo\n"
            f"{'='*70}\n"
        )
        print(err_msg, file=sys.stderr)
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        err_msg = (
            "\nERROR: 'ultralytics' is not installed. "
            "Install it via: pip install ultralytics\n"
        )
        print(err_msg, file=sys.stderr)
        raise ImportError(err_msg)

    if _MODEL is None or _LOADED_WEIGHTS != str(wp):
        _MODEL = YOLO(str(wp))
        _LOADED_WEIGHTS = str(wp)

    return _MODEL


def predict(image: np.ndarray, config: dict) -> List[Prediction]:
    """Run YOLO inference on a single image.

    Args:
        image: BGR image as numpy array.
        config: Full configuration dictionary.

    Returns:
        List of Prediction objects.
    """
    yolo_cfg = config.get("yolo", {})
    weights_path = yolo_cfg.get("weights_path", "method_yolo/weights/best.pt")
    imgsz = yolo_cfg.get("imgsz", 1024)
    conf = yolo_cfg.get("conf", 0.25)
    page_cfg = config.get("page_boundary", {})

    model = get_model(weights_path)

    h, w = image.shape[:2]
    page_boundary = estimate_page_boundary(image, page_cfg)

    # Run Ultralytics inference
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return []

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy()

    return postprocess_yolo_detections(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        page_boundary=page_boundary,
        img_w=w,
        img_h=h,
        config=yolo_cfg,
    )
