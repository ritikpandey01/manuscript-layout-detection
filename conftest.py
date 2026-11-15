"""Root conftest for pytest to set pythonpath."""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def get_test_image_dir() -> Path:
    """Return available test image directory (prefers data/test_images)."""
    p = Path("data/test_images")
    if p.exists() and any(p.glob("*.jpg")):
        return p
    return Path("dataset/real_test/images")
