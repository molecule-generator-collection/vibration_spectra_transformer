#!/usr/bin/env python
"""Convert the repository's QCForever pickle files into training tensors."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constants import DATA_DIRECTORY, DATA_ROOT, SEED, SMILES_MAX_LENGTH  # noqa: E402
from common import project_path, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DATA_ROOT), help="Directory containing size*_all.pickle")
    parser.add_argument("--output-dir", default=str(DATA_DIRECTORY))
    parser.add_argument("--min-heavy-size", type=int, default=1)
    parser.add_argument("--max-heavy-size", type=int, default=9)
    parser.add_argument("--max-samples", type=int, help="Optional cap for a quick experiment")
    parser.add_argument("--spectrum-length", type=int, help="Pad/truncate length; default is the observed maximum")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "Also save an analysis_manifest.csv in each split directory. "
            "The manifest preserves the exact tensor row order and selected "
            "molecular properties; normal hyperparameter runs omit it."
        ),
    )
    return parser.parse_args()


def split_indices(sizes: list[int], seed: int) -> dict[str, list[int]]:
    """Create deterministic 80/10/10 splits while preserving each heavy-size group."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, size in enumerate(sizes):
        groups[size].append(index)
    result = {"train": [], "valid": [], "test": []}
    for indices in groups.values():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_valid = max(1, round(n * 0.1)) if n >= 3 else 0
        if n_test + n_valid >= n:
            n_test, n_valid = 1, 1
        result["test"].extend(indices[:n_test])
        result["valid"].extend(indices[n_test:n_test + n_valid])
        result["train"].extend(indices[n_test + n_valid:])
    for indices in result.values():
        rng.shuffle(indices)
    return result


def optional_float(value: object) -> float | None:
    """Return a finite float when possible, otherwise an empty manifest value."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def sequence_value(value: object, index: int) -> float | None:
    """Safely select a scalar property from a list-like QCForever result."""
    if isinstance(value, (str, bytes)):
        return None
    try:
        return optional_float(value[index])  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return None


def analysis_metadata(
    molecule: dict,
    source_file: str,
    source_row: int,
    canonical_smiles: str,
    heavy_size: int,
) -> dict[str, object]:
    """Build stable molecule metadata without using SMILES as the row key."""
    return {
        "sample_id": f"{source_file}:{source_row}",
        "source_file": source_file,
        "source_row": source_row,
        "source_index": molecule.get("index", ""),
        "raw_smiles": molecule.get("smiles", ""),
        "canonical_smiles": canonical_smiles,
        "heavy_size": heavy_size,
        # QCForever list layouts confirmed for this dataset.
        "dipole_norm": sequence_value(molecule.get("dipole"), 3),
        "vip": sequence_value(molecule.get("vip"), 0),
        "vea": sequence_value(molecule.get("vea"), 0),
        "homolumo": sequence_value(molecule.get("homolumo"), 0),
        "polar_aniso": optional_float(molecule.get("polar_aniso")),
        "polar_iso": optional_float(molecule.get("polar_iso")),
        "deen": optional_float(molecule.get("deen")),
    }


def write_analysis_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty analysis manifest: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    from utils.tokenizers import SPETokenizerWrapper

    input_dir = project_path(args.input_dir)
    output_dir = project_path(args.output_dir)
    tokenizer = SPETokenizerWrapper()
    records: list[
        tuple[list[float], list[float], list[float], str, int, dict[str, object]]
    ] = []
    rejected = 0

    for heavy_size in range(args.min_heavy_size, args.max_heavy_size + 1):
        path = input_dir / f"size{heavy_size}_all.pickle"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            molecules = pickle.load(handle)
        for source_row, molecule in enumerate(molecules):
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
            try:
                freq = list(molecule["freq"])
                ir = list(molecule["IR"])
                raman = list(molecule["Raman"])
                mol = Chem.MolFromSmiles(molecule["smiles"])
                if mol is None or not freq or freq[0] < 0 or not (len(freq) == len(ir) == len(raman)):
                    raise ValueError
                smiles = Chem.MolToSmiles(mol)
                tokenizer([smiles])
                metadata = (
                    analysis_metadata(
                        molecule, path.name, source_row, smiles, heavy_size
                    )
                    if args.analysis
                    else {}
                )
                records.append((freq, ir, raman, smiles, heavy_size, metadata))
            except (KeyError, TypeError, ValueError, IndexError):
                rejected += 1
        if args.max_samples is not None and len(records) >= args.max_samples:
            break

    if len(records) < 3:
        raise ValueError("At least three valid molecules are required")
    observed_max = max(len(record[0]) for record in records)
    spectrum_length = args.spectrum_length or observed_max
    if spectrum_length < observed_max:
        print(f"Warning: spectra longer than {spectrum_length} will be truncated")
    splits = split_indices([record[4] for record in records], args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, indices in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        selected = [records[index] for index in indices]
        spectra = []
        masks = []
        for field in range(3):
            values = torch.full((len(selected), spectrum_length), -1.0, dtype=torch.float32)
            for row, record in enumerate(selected):
                sequence = torch.as_tensor(record[field][:spectrum_length], dtype=torch.float32)
                values[row, :len(sequence)] = sequence
            spectra.append(values)
        mask = spectra[0].ne(-1).to(torch.int8)
        encodings = tokenizer([record[3] for record in selected])
        torch.save(spectra[0], split_dir / "freqs.pt")
        torch.save(spectra[1], split_dir / "IRs.pt")
        torch.save(spectra[2], split_dir / "Ramans.pt")
        torch.save(mask, split_dir / "freq_attention_masks.pt")
        torch.save(encodings.input_ids, split_dir / "smiles_ids.pt")
        torch.save(encodings.attention_mask, split_dir / "smiles_attention_masks.pt")
        if args.analysis:
            write_analysis_manifest(
                split_dir / "analysis_manifest.csv",
                [record[5] for record in selected],
            )
        print(f"{split}: {len(selected):,} molecules")

    metadata = {
        "spectrum_max_length": spectrum_length,
        "smiles_max_length": SMILES_MAX_LENGTH,
        "smiles_vocab_size": tokenizer.vocab_size,
        "num_molecules": len(records),
        "rejected_molecules": rejected,
        "seed": args.seed,
        "analysis_manifests": args.analysis,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(f"Saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
