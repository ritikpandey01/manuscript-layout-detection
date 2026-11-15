"""Test ensuring original images and inputs are never modified."""
import hashlib
import sys
from pathlib import Path
import subprocess
import pytest
from conftest import get_test_image_dir


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_original_images_unmodified(tmp_path):
    test_dir = get_test_image_dir()
    assert test_dir.exists()

    # Record SHA-256 hashes of all input images before inference
    before_hashes = {p.name: _file_hash(p) for p in test_dir.glob("*.jpg")}
    assert len(before_hashes) > 0

    # Run inference into temporary output directory
    out_dir = tmp_path / "test_out"
    cmd = [
        sys.executable,
        "inference.py",
        "--input",
        str(test_dir),
        "--output",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference failed: {res.stderr}\nStdout: {res.stdout}"

    # Verify all input images are byte-identical
    after_hashes = {p.name: _file_hash(p) for p in test_dir.glob("*.jpg")}
    assert before_hashes == after_hashes, "Original images were modified!"
