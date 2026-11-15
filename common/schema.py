"""Core data types and validation for manuscript layout detection.

All bounding boxes use pixel coordinates in [x_min, y_min, x_max, y_max] format.
All coordinates are integers. Boxes must satisfy x_min < x_max and y_min < y_max.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json

# The five allowed labels
LABELS = ["header", "footer", "main_text", "side_text", "filler"]

# Default annotation colors (BGR for OpenCV)
DEFAULT_COLORS = {
    "header":    (255, 100, 100),   # blue-ish
    "footer":    (100, 255, 100),   # green
    "main_text": (100, 100, 255),   # red
    "side_text": (255, 255, 100),   # cyan
    "filler":    (128, 128, 128),   # gray
}

BBOX_FORMAT = "[x_min,y_min,x_max,y_max]"


@dataclass
class Prediction:
    """A single detected region."""
    label: str
    confidence: float
    bbox: List[int]  # [x_min, y_min, x_max, y_max] in pixels
    reason: str = ""
    needs_review: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate prediction fields."""
        if self.label not in LABELS:
            raise ValueError(f"Invalid label '{self.label}'. Must be one of {LABELS}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence {self.confidence} must be in [0.0, 1.0]")
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 elements, got {len(self.bbox)}")
        x_min, y_min, x_max, y_max = self.bbox
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be < y_max ({y_max})")
        if any(v < 0 for v in self.bbox):
            raise ValueError(f"Bbox values must be non-negative: {self.bbox}")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [int(v) for v in self.bbox],
            "reason": self.reason,
            "needs_review": self.needs_review,
        }


@dataclass
class ImageResult:
    """Detection results for a single image."""
    image: str  # filename (not full path)
    width: int
    height: int
    method: str
    predictions: List[Prediction] = field(default_factory=list)
    bbox_format: str = BBOX_FORMAT

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict matching required output schema."""
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "bbox_format": self.bbox_format,
            "method": self.method,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def validate_bbox_in_image(bbox: List[int], width: int, height: int) -> bool:
    """Check if bbox is fully inside image dimensions."""
    x_min, y_min, x_max, y_max = bbox
    return (0 <= x_min < x_max <= width) and (0 <= y_min < y_max <= height)


def clip_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """Clip bbox to image dimensions. Returns clipped bbox."""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(x_min + 1, min(x_max, width))
    y_max = max(y_min + 1, min(y_max, height))
    return [int(x_min), int(y_min), int(x_max), int(y_max)]
