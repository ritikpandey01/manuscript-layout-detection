"""Page boundary estimation for manuscript images.

Estimates the actual page/leaf extent within the photograph.
All detection boxes are clipped to this boundary.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


def estimate_page_boundary(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Estimate the page/leaf boundary within the image.

    Tries multiple methods in order:
    1. Edge-based: Canny edges + contour finding for clear page borders
    2. Color-based: Separate page from background by dominant color
    3. Fallback: Full image dimensions

    Args:
        image: BGR image as numpy array.
        config: Optional config dict with page_boundary settings.

    Returns:
        Bounding box [x_min, y_min, x_max, y_max] of the page region.
    """
    cfg = config or {}
    method = cfg.get("method", "auto")
    h, w = image.shape[:2]
    full_image = [0, 0, w, h]

    if method == "full_image":
        return full_image

    if method == "auto":
        # Check if color-based segmentation finds a distinct page on a contrasting background
        cb = _color_based_boundary(image, cfg)
        if cb is not None:
            cb_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
            # If the color-based page covers between 15% and 88% of the image, it is a true page object
            if 0.15 <= (cb_area / float(h * w)) <= 0.88:
                return cb

        eb = _edge_based_boundary(image, cfg)
        if eb is not None:
            eb_area = (eb[2] - eb[0]) * (eb[3] - eb[1])
            if (eb_area / float(h * w)) < 0.90:
                return eb
            # If edge covers >90% of image, return color-based if it found any boundary
            if cb is not None:
                return cb
            return eb

        if cb is not None:
            return cb

    elif method == "edge":
        boundary = _edge_based_boundary(image, cfg)
        if boundary is not None:
            return boundary

    elif method == "color":
        boundary = _color_based_boundary(image, cfg)
        if boundary is not None:
            return boundary

    return full_image


def _edge_based_boundary(
    image: np.ndarray, config: Dict[str, Any]
) -> Optional[List[int]]:
    """Detect page boundary using Canny edges and contour finding."""
    h, w = image.shape[:2]
    min_area_ratio = config.get("min_page_area_ratio", 0.3)
    canny_low = config.get("edge_canny_low", 50)
    canny_high = config.get("edge_canny_high", 150)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate edges to connect nearby edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find largest contour that covers enough area
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best = cnt

    if best is None or best_area < (h * w * min_area_ratio):
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    padding = config.get("margin_padding", 5)
    return _pad_and_clip([x, y, x + bw, y + bh], w, h, padding)


def _color_based_boundary(
    image: np.ndarray, config: Dict[str, Any]
) -> Optional[List[int]]:
    """Detect page boundary by separating page from background color.

    Works well when the background is a distinct solid color (e.g., dark
    scanner background vs light manuscript page).
    """
    h, w = image.shape[:2]
    min_area_ratio = config.get("min_page_area_ratio", 0.3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's thresholding to separate foreground/background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine if the page is lighter or darker than background
    # Sample corners to estimate background intensity
    corner_size = min(h, w) // 10
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    bg_mean = np.mean([c.mean() for c in corners])
    page_mean = gray[h//4:3*h//4, w//4:3*w//4].mean()

    if page_mean > bg_mean:
        # Page is lighter than background
        mask = binary
    else:
        # Page is darker (e.g., dark palm leaf on light background)
        mask = cv2.bitwise_not(binary)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest contour
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < (h * w * min_area_ratio):
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    padding = config.get("margin_padding", 5)
    return _pad_and_clip([x, y, x + bw, y + bh], w, h, padding)


def _pad_and_clip(
    bbox: List[int], width: int, height: int, padding: int
) -> List[int]:
    """Apply negative padding (shrink) and clip to image bounds."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min + padding)
    y_min = max(0, y_min + padding)
    x_max = min(width, x_max - padding)
    y_max = min(height, y_max - padding)
    # Ensure valid box
    if x_min >= x_max or y_min >= y_max:
        return [0, 0, width, height]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def clip_bbox_to_boundary(
    bbox: List[int],
    boundary: List[int],
    image_width: int,
    image_height: int,
) -> Optional[List[int]]:
    """Clip a bbox to both the page boundary and the image dimensions.

    Args:
        bbox: [x_min, y_min, x_max, y_max] to clip.
        boundary: Page boundary [x_min, y_min, x_max, y_max].
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Clipped bbox, or None if the box becomes degenerate (zero area).
    """
    x_min = max(bbox[0], boundary[0], 0)
    y_min = max(bbox[1], boundary[1], 0)
    x_max = min(bbox[2], boundary[2], image_width)
    y_max = min(bbox[3], boundary[3], image_height)

    if x_min >= x_max or y_min >= y_max:
        return None

    return [int(x_min), int(y_min), int(x_max), int(y_max)]
