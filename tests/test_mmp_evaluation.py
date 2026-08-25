import io
import json
import pickle
import zipfile

import pandas as pd
import pytest
import torch

from ir_raman_freq.evaluation import finalize_counts
from ir_raman_freq.prepare_mmp_evaluation import (
    QCForeverUnpickler,
    prepare_size,
    save_size_dataset,
)
from scripts.analyze_molecular_performance import (
    candidate_functional_group_hits,
    functional_group_recovery,
    rdkit_features,
)
from utils.tokenizers import SPETokenizerWrapper


class UnsupportedPickleObject:
    def __reduce__(self):
        return eval, ("1 + 1",)


def test_restricted_unpickler_blocks_unsupported_globals():
    payload = pickle.dumps(UnsupportedPickleObject())

    with pytest.raises(pickle.UnpicklingError, match="Blocked unsupported"):
        QCForeverUnpickler(io.BytesIO(payload)).load()


def test_prepare_size_truncates_and_saves_size_group(tmp_path):
    valid = {
        "index": "valid",
        "smiles": "CCCCCCCCCC",
        "freq": list(range(1, 86)),
        "IR": [0.5] * 85,
        "Raman": [0.25] * 85,
        "dipole": [0.1, 0.2, 0.3, 0.4],
        "vip": [7.5],
        "vea": [1.25],
        "homolumo": [4.5],
        "polar_aniso": 2.5,
        "polar_iso": 3.5,
        "deen": -0.75,
    }
    imaginary = {
        "index": "imaginary",
        "smiles": "CCCCCCCCCC",
        "freq": [-1.0, 2.0],
        "IR": [0.5, 0.5],
        "Raman": [0.25, 0.25],
    }
    archive_path = tmp_path / "mmp.zip"
    entry_name = "MMP1%/size10_withFL.pickle"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(entry_name, pickle.dumps([valid, imaginary]))

    tokenizer = SPETokenizerWrapper()
    with zipfile.ZipFile(archive_path) as archive:
        records, rejected, source_records = prepare_size(
            archive,
            entry_name,
            10,
            tokenizer,
            spectrum_length=81,
            smiles_max_length=32,
            overflow_policy="truncate",
            max_samples=None,
        )

    assert source_records == 2
    assert len(records) == 1
    assert rejected == {"imaginary_frequency": 1}
    assert records[0]["manifest"]["spectrum_truncated"] == 1
    assert records[0]["manifest"]["dipole_norm"] == 0.4
    assert records[0]["manifest"]["vip"] == 7.5

    save_size_dataset(tmp_path / "prepared", 10, records, spectrum_length=81)
    size_dir = tmp_path / "prepared" / "size10"
    freqs = torch.load(size_dir / "freqs.pt", weights_only=True)
    masks = torch.load(size_dir / "freq_attention_masks.pt", weights_only=True)
    assert freqs.shape == (1, 81)
    assert masks.sum().item() == 81
    assert (size_dir / "analysis_manifest.csv").is_file()


def test_prepare_size_can_reject_spectrum_overflow(tmp_path):
    molecule = {
        "smiles": "CCCCCCCCCC",
        "freq": list(range(1, 86)),
        "IR": [0.5] * 85,
        "Raman": [0.25] * 85,
    }
    archive_path = tmp_path / "mmp.zip"
    entry_name = "MMP1%/size10_withFL.pickle"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(entry_name, pickle.dumps([molecule]))

    with zipfile.ZipFile(archive_path) as archive:
        records, rejected, _ = prepare_size(
            archive,
            entry_name,
            10,
            SPETokenizerWrapper(),
            spectrum_length=81,
            smiles_max_length=32,
            overflow_policy="reject",
            max_samples=None,
        )

    assert not records
    assert rejected == {"spectrum_too_long": 1}


def test_finalize_counts_calculates_weighted_metrics():
    result = finalize_counts(
        {
            "samples": 2,
            "loss_sum": 6.0,
            "target_tokens": 3,
            "top1_correct": 1,
            "top3_correct": 1,
            "top5_correct": 2,
            "top1_valid": 1,
            "tanimoto_sum": 1.25,
            "tanimoto_values": [1.0, 0.25],
            "formula_correct": 1,
            "heavy_size_error_sum": 2.0,
            "heavy_size_error_count": 1,
            "truncated": 1,
        }
    )

    assert result["cross_entropy_loss"] == 2.0
    assert result["canonical_top_1_accuracy"] == 0.5
    assert result["canonical_top_5_accuracy"] == 1.0
    assert result["mean_top_1_tanimoto"] == 0.625
    assert result["median_top_1_tanimoto"] == 0.625
    assert result["mean_valid_top_1_heavy_atom_count_absolute_error"] == 2.0
    assert result["truncated_spectra_rate"] == 0.5


def test_functional_group_recovery_compares_target_and_beam_candidates():
    base = pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCO", "CC"],
            "predicted_canonical_smiles": ["CCO", "CC", "CCO"],
            "candidate_canonical_smiles_json": [
                json.dumps(["CCO"]),
                json.dumps(["CC", "CCO"]),
                json.dumps(["CCO"]),
            ],
            "heavy_size": [3, 3, 2],
        }
    )
    target = pd.DataFrame(
        [rdkit_features(value) for value in base["canonical_smiles"]]
    )
    predicted = pd.DataFrame(
        [rdkit_features(value) for value in base["predicted_canonical_smiles"]]
    ).add_prefix("predicted_")
    hits = candidate_functional_group_hits(base)

    result = functional_group_recovery(pd.concat([base, target, predicted, hits], axis=1))
    alcohol = result.set_index("functional_group").loc["alcohol"]

    assert alcohol["true_positive"] == 1
    assert alcohol["false_negative"] == 1
    assert alcohol["false_positive"] == 1
    assert alcohol["top1_precision"] == 0.5
    assert alcohol["top1_recall"] == 0.5
    assert alcohol["target_group_recall_at_3"] == 1.0
