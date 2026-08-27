#!/usr/bin/env python
"""Train the frequency + IR + Raman variant in its own result directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train import main  # noqa: E402


if __name__ == "__main__":
    main(
        default_output_dir=ROOT / "results" / "ir_raman_transformer",
        default_modalities=("freq", "ir", "raman"),
    )
