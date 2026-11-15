"""Image I/O utilities for manuscript layout detection.

Handles loading images (jpg/png/tif), batch processing, and path resolution.
Never modifies or writes to input directories.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_image(path: str) -> Tuple[np.ndarray, Path]:
    """Load an image from disk.

    Args:
        path: Path to the image file (relative or absolute).

    Returns:
        Tuple of (image as BGR numpy array, resolved Path object).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image (unsupported format or corrupt): {p}")

    return img, p


def list_images(path: str) -> List[Path]:
    """List image files from a path.

    If path is a file, returns [path].
    If path is a directory, returns all image files in it (non-recursive).

    Args:
        path: Path to a single image or a directory of images.

    Returns:
        Sorted list of Path objects for image files.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If path is a file but not an image, or directory is empty.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Not a supported image file: {p} "
                f"(supported: {', '.join(sorted(IMAGE_EXTENSIONS))})"
            )
        return [p]

    if p.is_dir():
        images = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No image files found in directory: {p}")
        return images

    raise ValueError(f"Path is neither a file nor a directory: {p}")


def get_image_hash(path: str) -> str:
    """Get SHA-256 hash of an image file for integrity verification.

    Used in tests to verify originals are not modified.
    """
    import hashlib
    p = Path(path).resolve()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ensure_output_dir(output_path: str) -> Tuple[Path, Path, Path]:
    """Create output directory structure.

    Creates:
        output_path/annotated/
        output_path/json/

    Args:
        output_path: Base output directory.

    Returns:
        Tuple of (base_dir, annotated_dir, json_dir).
    """
    base = Path(output_path).resolve()
    annotated_dir = base / "annotated"
    json_dir = base / "json"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    return base, annotated_dir, json_dir
