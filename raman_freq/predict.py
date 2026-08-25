#!/usr/bin/env python
"""Predict SMILES using frequency + Raman spectra."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict import main  # noqa: E402


if __name__ == "__main__":
    main(
        default_checkpoint=ROOT / "results" / "raman_freq_transformer" / "best.pt",
        default_output=ROOT / "results" / "raman_freq_transformer" / "predictions.csv",
    )
