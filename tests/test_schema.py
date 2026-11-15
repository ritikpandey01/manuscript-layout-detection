"""Unit tests for common schema and data types."""
import pytest
from common.schema import (
    Prediction,
    ImageResult,
    LABELS,
    BBOX_FORMAT,
    validate_bbox_in_image,
    clip_bbox,
)


def test_valid_prediction():
    pred = Prediction(
        label="main_text",
        confidence=0.92,
        bbox=[100, 50, 800, 600],
        reason="largest_dense_block",
        needs_review=False,
    )
    assert pred.label == "main_text"
    assert pred.confidence == 0.92
    assert pred.bbox == [100, 50, 800, 600]
    d = pred.to_dict()
    assert d["label"] == "main_text"
    assert d["confidence"] == 0.92
    assert d["bbox"] == [100, 50, 800, 600]
    assert d["needs_review"] is False


def test_invalid_label():
    with pytest.raises(ValueError, match="Invalid label"):
        Prediction(label="invalid_class", confidence=0.8, bbox=[10, 10, 50, 50])


def test_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence"):
        Prediction(label="main_text", confidence=1.5, bbox=[10, 10, 50, 50])
    with pytest.raises(ValueError, match="Confidence"):
        Prediction(label="main_text", confidence=-0.1, bbox=[10, 10, 50, 50])


def test_invalid_bbox_coordinates():
    # x_min >= x_max
    with pytest.raises(ValueError, match="x_min"):
        Prediction(label="main_text", confidence=0.8, bbox=[50, 10, 50, 100])
    # y_min >= y_max
    with pytest.raises(ValueError, match="y_min"):
        Prediction(label="main_text", confidence=0.8, bbox=[10, 100, 50, 50])
    # negative coords
    with pytest.raises(ValueError, match="non-negative"):
        Prediction(label="main_text", confidence=0.8, bbox=[-5, 10, 50, 50])


def test_image_result_serialization():
    pred = Prediction(
        label="header",
        confidence=0.85,
        bbox=[50, 10, 500, 60],
        reason="top_margin",
    )
    result = ImageResult(
        image="sample.jpg",
        width=1000,
        height=700,
        method="yolo",
        predictions=[pred],
    )
    d = result.to_dict()
    assert d["image"] == "sample.jpg"
    assert d["width"] == 1000
    assert d["height"] == 700
    assert d["bbox_format"] == BBOX_FORMAT
    assert d["method"] == "yolo"
    assert len(d["predictions"]) == 1
    assert d["predictions"][0]["label"] == "header"


def test_bbox_validation_and_clipping():
    assert validate_bbox_in_image([10, 10, 90, 90], 100, 100) is True
    assert validate_bbox_in_image([10, 10, 110, 90], 100, 100) is False

    clipped = clip_bbox([-10, 5, 150, 80], 100, 100)
    assert clipped == [0, 5, 100, 80]
    assert validate_bbox_in_image(clipped, 100, 100) is True
