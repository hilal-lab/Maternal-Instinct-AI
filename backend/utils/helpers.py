"""
Utility helpers — shared across the backend.
"""
from pathlib import Path


def get_project_root() -> Path:
    """Get the Maternal-Instinct-AI root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "backend" / "data"


def safe_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    return Path(name).name.replace("..", "").replace("/", "").replace("\\", "")
