"""Unit tests for batch image loading and path handling."""
from pathlib import Path
import pytest
from common.io_utils import list_images, load_image
from conftest import get_test_image_dir


def test_list_images_single_file():
    test_dir = get_test_image_dir()
    img_path = test_dir / "02_a.jpg"
    assert img_path.exists()
    images = list_images(str(img_path))
    assert len(images) == 1
    assert images[0].name == "02_a.jpg"


def test_list_images_directory():
    dir_path = get_test_image_dir()
    assert dir_path.exists()
    images = list_images(str(dir_path))
    assert len(images) == 9
    assert all(img.suffix.lower() in {".jpg", ".jpeg", ".png"} for img in images)


def test_load_image_properties():
    test_dir = get_test_image_dir()
    img_path = str(test_dir / "02_a.jpg")
    img, resolved = load_image(img_path)
    assert img is not None
    assert len(img.shape) == 3  # BGR
    assert resolved.name == "02_a.jpg"
