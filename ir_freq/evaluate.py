#!/usr/bin/env python
"""Evaluate the frequency + IR variant."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import main  # noqa: E402


if __name__ == "__main__":
    main(default_checkpoint=ROOT / "results" / "ir_freq_transformer" / "best.pt")
