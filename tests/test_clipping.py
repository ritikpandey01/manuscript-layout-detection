"""Unit tests for bounding box clipping and page boundary estimation."""
import numpy as np
from common.page_boundary import (
    estimate_page_boundary,
    clip_bbox_to_boundary,
)


def test_clip_bbox_to_boundary_normal():
    boundary = [50, 50, 950, 650]
    bbox = [40, 60, 960, 600]
    clipped = clip_bbox_to_boundary(bbox, boundary, 1000, 700)
    assert clipped == [50, 60, 950, 600]


def test_clip_bbox_to_boundary_outside():
    boundary = [100, 100, 500, 500]
    # Box completely outside to the left
    bbox = [10, 10, 80, 80]
    clipped = clip_bbox_to_boundary(bbox, boundary, 1000, 1000)
    assert clipped is None


def test_estimate_page_boundary_synthetic():
    # Create synthetic image with dark border around a light center
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[50:350, 60:540] = 200  # light page

    boundary = estimate_page_boundary(img, {"method": "auto"})
    assert len(boundary) == 4
    x0, y0, x1, y1 = boundary
    assert 0 <= x0 < x1 <= 600
    assert 0 <= y0 < y1 <= 400
    # Boundary should capture most of the light page
    assert x0 <= 70 and x1 >= 530
    assert y0 <= 60 and y1 >= 340
