"""Legacy data-script paths, kept relative to this repository."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DATA_DIRECTORY = PROJECT_ROOT / "data_diretory"
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data_diretory" / "processed"
