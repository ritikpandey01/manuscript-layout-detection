"""Output generation: annotated images and JSON files.

Creates annotated copies of images (never modifies originals) and writes
JSON metadata in the required format.
"""
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from common.schema import ImageResult, Prediction, DEFAULT_COLORS


def save_annotated_image(
    image: np.ndarray,
    predictions: List[Prediction],
    output_path: str,
    colors: Optional[dict] = None,
) -> Path:
    """Draw detection boxes on a COPY of the image and save it.

    Args:
        image: Original BGR image (not modified).
        predictions: List of predictions to draw.
        output_path: Where to save the annotated image.
        colors: Optional color override dict {label: (B, G, R)}.

    Returns:
        Path to the saved annotated image.
    """
    color_map = colors or DEFAULT_COLORS
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Scale font/thickness to image size
    scale_factor = max(w, h) / 1500.0
    font_scale = max(0.4, 0.6 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    box_thickness = max(2, int(3 * scale_factor))

    for pred in predictions:
        color = color_map.get(pred.label, (200, 200, 200))
        x_min, y_min, x_max, y_max = pred.bbox

        # Draw bounding box
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, box_thickness)

        # Prepare label text
        label_text = f"{pred.label} {pred.confidence:.2f}"
        if pred.needs_review:
            label_text += " [REVIEW]"

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Place label above box, or inside if at top edge
        label_y = y_min - 5 if y_min > text_h + 10 else y_min + text_h + 5
        label_x = x_min

        cv2.rectangle(
            annotated,
            (label_x, label_y - text_h - baseline),
            (label_x + text_w, label_y + baseline),
            color,
            cv2.FILLED,
        )
        # Text in contrasting color
        cv2.putText(
            annotated,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return out_path


def save_json(result: ImageResult, output_path: str) -> Path:
    """Save per-image detection results as JSON.

    Args:
        result: ImageResult for a single image.
        output_path: Where to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")
    return out_path


def save_combined_json(
    results: List[ImageResult], output_path: str
) -> Path:
    """Save combined predictions for all images.

    Args:
        results: List of ImageResult for all processed images.
        output_path: Where to save the combined JSON file.

    Returns:
        Path to the saved JSON file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined = [r.to_dict() for r in results]
    out_path.write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    return out_path
