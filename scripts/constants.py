"""Project-wide defaults.

All paths are derived from this file's location.  Command-line arguments in the
entry-point scripts can override them, and relative arguments are resolved from
the repository root (not from the caller's current working directory).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_diretory"
DATA_DIRECTORY = DATA_ROOT / "processed"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

DEVICE = "auto"
SEED = 42
SMILES_MAX_LENGTH = 32
DROPOUT_RATE = 0.0
LR = 1e-4
MODEL_NAME = "ir_raman_transformer"
ENCODER_NUM_LAYERS = 3
ENCODER_HIDDEN_DIMENSION = 128
ENCODER_N_HEADS = 4
DECODER_NUM_LAYERS = 6
DECODER_N_HEADS = 4
DECODER_HIDDEN_DIMENSION = 128
EMBEDDING_DIMENSION = 128
