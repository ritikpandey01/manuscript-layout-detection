"""Smoke tests for the CLI interface."""
import json
import sys
from pathlib import Path
import subprocess
from conftest import get_test_image_dir


def test_cli_single_image(tmp_path):
    test_dir = get_test_image_dir()
    img_path = test_dir / "02_a.jpg"
    out_dir = tmp_path / "single_out"
    cmd = [
        sys.executable,
        "inference.py",
        "--input",
        str(img_path),
        "--output",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}\nStdout: {res.stdout}"

    # Check annotated image exists
    annotated = out_dir / "annotated" / "02_a_annotated.png"
    assert annotated.exists()
    assert annotated.stat().st_size > 1000

    # Check per-image JSON exists
    img_json = out_dir / "json" / "02_a.json"
    assert img_json.exists()
    data = json.loads(img_json.read_text())
    assert data["image"] == "02_a.jpg"
    assert "predictions" in data

    # Check predictions.json exists
    comb_json = out_dir / "predictions.json"
    assert comb_json.exists()
    all_data = json.loads(comb_json.read_text())
    assert len(all_data) == 1


def test_cli_folder_relative_paths(tmp_path):
    test_dir = get_test_image_dir()
    out_dir = tmp_path / "batch_out"
    cmd = [
        sys.executable,
        "inference.py",
        "--input",
        str(test_dir),
        "--output",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}\nStdout: {res.stdout}"

    comb_json = out_dir / "predictions.json"
    assert comb_json.exists()
    all_data = json.loads(comb_json.read_text())
    assert len(all_data) == 9


def test_cli_yolo_graceful_failure_without_weights(tmp_path):
    test_dir = get_test_image_dir()
    img_path = test_dir / "02_a.jpg"
    out_dir = tmp_path / "yolo_out"
    custom_cfg = tmp_path / "test_cfg.yaml"
    custom_cfg.write_text("yolo:\n  weights_path: nonexistent_weights.pt\n")

    cmd = [
        sys.executable,
        "inference.py",
        "--input",
        str(img_path),
        "--output",
        str(out_dir),
        "--config",
        str(custom_cfg),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Checks that it printed error message directing user to Kaggle steps
    output_text = res.stderr + res.stdout
    assert "YOLO weights not found" in output_text
    assert "KAGGLE_STEPS.md" in output_text
