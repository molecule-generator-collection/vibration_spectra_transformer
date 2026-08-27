#!/usr/bin/env python
"""Prepare size-grouped MMP evaluation tensors from a ZIP archive.

The source archive contains one pickle list per heavy-atom size.  Pickle is an
executable format, so this script uses an allow-list unpickler that accepts only
the NumPy scalar objects observed in the QCForever database.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import project_path, write_json  # noqa: E402
from constants import SMILES_MAX_LENGTH  # noqa: E402


TENSOR_FILES = (
    "freqs.pt",
    "IRs.pt",
    "Ramans.pt",
    "freq_attention_masks.pt",
    "smiles_ids.pt",
    "smiles_attention_masks.pt",
)
SIZE_PATTERN = re.compile(r"(?:^|/)size(?P<size>\d+)_withFL\.pickle$")


class QCForeverUnpickler(pickle.Unpickler):
    """Restricted unpickler for the primitive QCForever record structure."""

    ALLOWED_GLOBALS = {
        ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str):  # noqa: ANN201
        try:
            return self.ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Blocked unsupported pickle global: {module}.{name}"
            ) from error


def restricted_pickle_load(handle: BinaryIO) -> object:
    return QCForeverUnpickler(handle).load()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="MMP1%%.zip archive")
    parser.add_argument(
        "--output-dir",
        default="data_diretory/mmp_evaluation",
        help="Directory for size*/ tensors and metadata.json",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint used to infer spectrum and SMILES lengths",
    )
    parser.add_argument(
        "--spectrum-length",
        type=int,
        default=81,
        help="Model input length when --checkpoint is omitted (default: 81)",
    )
    parser.add_argument("--min-heavy-size", type=int, default=10)
    parser.add_argument("--max-heavy-size", type=int)
    parser.add_argument(
        "--overflow-policy",
        choices=("truncate", "reject"),
        default="truncate",
        help="Truncate long spectra or exclude them from strict evaluation",
    )
    parser.add_argument(
        "--max-samples-per-size",
        type=int,
        help="Optional cap for smoke tests; applied after validation",
    )
    return parser.parse_args(argv)


def checkpoint_dimensions(path: Path) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    params = checkpoint["model_params"]
    return (
        int(params["spectrum_max_length"]),
        int(params["smiles_max_length"]),
        int(params["smiles_vocab_size"]),
    )


def archive_size_entries(archive: zipfile.ZipFile) -> dict[int, str]:
    entries: dict[int, str] = {}
    for name in archive.namelist():
        match = SIZE_PATTERN.search(name)
        if not match:
            continue
        size = int(match.group("size"))
        if size in entries:
            raise ValueError(f"Archive contains duplicate entries for heavy size {size}")
        entries[size] = name
    if not entries:
        raise ValueError("No size*_withFL.pickle entries were found in the archive")
    return entries


def finite_float_list(value: object) -> list[float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        result = [float(item) for item in value]  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError):
        return None
    if not result or not all(math.isfinite(item) for item in result):
        return None
    return result


def optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def sequence_float(value: object, index: int) -> float | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        return optional_float(value[index])  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return None


def encode_smiles(tokenizer, smiles: str, smiles_max_length: int):  # noqa: ANN001, ANN201
    tokens = tokenizer.tokenize(smiles).split()
    if not tokens:
        raise ValueError("empty token sequence")
    if len(tokens) + 2 > smiles_max_length:
        raise ValueError("SMILES exceeds checkpoint token length")
    if any(token not in tokenizer.VOCABS_INDICES for token in tokens):
        raise ValueError("SMILES contains an out-of-vocabulary token")

    input_ids = torch.zeros(smiles_max_length, dtype=torch.long)
    attention_mask = torch.zeros(smiles_max_length, dtype=torch.long)
    token_ids = [tokenizer.VOCABS_INDICES[token] for token in tokens]
    encoded = [tokenizer.VOCABS_INDICES["[BOS]"], *token_ids, tokenizer.VOCABS_INDICES["[EOS]"]]
    input_ids[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    attention_mask[: len(encoded)] = 1
    return input_ids, attention_mask


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_size_dataset(
    output_dir: Path,
    heavy_size: int,
    records: list[dict[str, object]],
    spectrum_length: int,
) -> None:
    size_dir = output_dir / f"size{heavy_size}"
    size_dir.mkdir(parents=True, exist_ok=True)
    count = len(records)
    freqs = torch.full((count, spectrum_length), -1.0, dtype=torch.float32)
    irs = torch.full_like(freqs, -1.0)
    ramans = torch.full_like(freqs, -1.0)
    spectrum_masks = torch.zeros((count, spectrum_length), dtype=torch.int8)

    for row, record in enumerate(records):
        used = min(len(record["freq"]), spectrum_length)  # type: ignore[arg-type]
        for tensor, key in ((freqs, "freq"), (irs, "ir"), (ramans, "raman")):
            tensor[row, :used] = torch.tensor(
                record[key][:used], dtype=torch.float32  # type: ignore[index]
            )
        spectrum_masks[row, :used] = 1

    torch.save(freqs, size_dir / TENSOR_FILES[0])
    torch.save(irs, size_dir / TENSOR_FILES[1])
    torch.save(ramans, size_dir / TENSOR_FILES[2])
    torch.save(spectrum_masks, size_dir / TENSOR_FILES[3])
    torch.save(
        torch.stack([record["smiles_ids"] for record in records]),  # type: ignore[list-item]
        size_dir / TENSOR_FILES[4],
    )
    torch.save(
        torch.stack([record["smiles_mask"] for record in records]),  # type: ignore[list-item]
        size_dir / TENSOR_FILES[5],
    )
    write_manifest(
        size_dir / "analysis_manifest.csv",
        [record["manifest"] for record in records],  # type: ignore[list-item]
    )


def prepare_size(
    archive: zipfile.ZipFile,
    entry_name: str,
    heavy_size: int,
    tokenizer,
    spectrum_length: int,
    smiles_max_length: int,
    overflow_policy: str,
    max_samples: int | None,
) -> tuple[list[dict[str, object]], Counter[str], int]:
    with archive.open(entry_name) as handle:
        molecules = restricted_pickle_load(handle)
    if not isinstance(molecules, list):
        raise ValueError(f"Expected a list in {entry_name}, got {type(molecules).__name__}")

    accepted: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    for source_row, molecule in enumerate(molecules):
        if max_samples is not None and len(accepted) >= max_samples:
            break
        if not isinstance(molecule, dict):
            rejected["not_a_record"] += 1
            continue
        try:
            freq = finite_float_list(molecule.get("freq"))
            ir = finite_float_list(molecule.get("IR"))
            raman = finite_float_list(molecule.get("Raman"))
            if freq is None or ir is None or raman is None:
                raise ValueError("invalid_spectrum")
            if not (len(freq) == len(ir) == len(raman)):
                raise ValueError("spectrum_length_mismatch")
            if any(value < 0 for value in freq):
                raise ValueError("imaginary_frequency")
            if len(freq) > spectrum_length and overflow_policy == "reject":
                raise ValueError("spectrum_too_long")

            molecule_object = Chem.MolFromSmiles(str(molecule["smiles"]))
            if molecule_object is None:
                raise ValueError("invalid_smiles")
            actual_size = molecule_object.GetNumHeavyAtoms()
            if actual_size != heavy_size:
                raise ValueError("heavy_size_mismatch")
            canonical_smiles = Chem.MolToSmiles(
                molecule_object, canonical=True, isomericSmiles=True
            )
            try:
                smiles_ids, smiles_mask = encode_smiles(
                    tokenizer, canonical_smiles, smiles_max_length
                )
            except (KeyError, IndexError, ValueError):
                raise ValueError("smiles_not_tokenizable") from None
        except KeyError:
            rejected["missing_required_field"] += 1
            continue
        except ValueError as error:
            rejected[str(error)] += 1
            continue

        accepted.append(
            {
                "freq": freq,
                "ir": ir,
                "raman": raman,
                "smiles_ids": smiles_ids,
                "smiles_mask": smiles_mask,
                "manifest": {
                    "sample_id": f"{entry_name}:{source_row}",
                    "source_file": entry_name,
                    "source_row": source_row,
                    "source_index": molecule.get("index", ""),
                    "raw_smiles": molecule.get("smiles", ""),
                    "canonical_smiles": canonical_smiles,
                    "heavy_size": heavy_size,
                    "original_spectrum_length": len(freq),
                    "used_spectrum_length": min(len(freq), spectrum_length),
                    "spectrum_truncated": int(len(freq) > spectrum_length),
                    "dipole_norm": sequence_float(molecule.get("dipole"), 3),
                    "vip": sequence_float(molecule.get("vip"), 0),
                    "vea": sequence_float(molecule.get("vea"), 0),
                    "homolumo": sequence_float(molecule.get("homolumo"), 0),
                    "polar_aniso": optional_float(molecule.get("polar_aniso")),
                    "polar_iso": optional_float(molecule.get("polar_iso")),
                    "deen": optional_float(molecule.get("deen")),
                },
            }
        )
    return accepted, rejected, len(molecules)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.min_heavy_size < 1:
        raise ValueError("--min-heavy-size must be at least 1")
    if args.max_heavy_size is not None and args.max_heavy_size < args.min_heavy_size:
        raise ValueError("--max-heavy-size must be >= --min-heavy-size")
    if args.spectrum_length < 1:
        raise ValueError("--spectrum-length must be at least 1")
    if args.max_samples_per_size is not None and args.max_samples_per_size < 1:
        raise ValueError("--max-samples-per-size must be at least 1")

    from utils.tokenizers import SPETokenizerWrapper

    input_path = project_path(args.input)
    output_dir = project_path(args.output_dir)
    spectrum_length = args.spectrum_length
    smiles_max_length = SMILES_MAX_LENGTH
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else None
    tokenizer = SPETokenizerWrapper()
    if checkpoint_path is not None:
        spectrum_length, smiles_max_length, checkpoint_vocab_size = checkpoint_dimensions(
            checkpoint_path
        )
        if checkpoint_vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint vocabulary size does not match the repository tokenizer: "
                f"{checkpoint_vocab_size} != {tokenizer.vocab_size}"
            )

    if (output_dir / "metadata.json").exists() or any(output_dir.glob("size*")):
        raise FileExistsError(
            f"Output directory already contains an MMP dataset: {output_dir}. "
            "Use a new --output-dir to avoid mixing evaluation protocols."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    size_summaries: list[dict[str, object]] = []
    total_accepted = 0
    total_rejected: Counter[str] = Counter()
    with zipfile.ZipFile(input_path) as archive:
        entries = archive_size_entries(archive)
        maximum = args.max_heavy_size or max(entries)
        selected_sizes = [
            size for size in sorted(entries) if args.min_heavy_size <= size <= maximum
        ]
        if not selected_sizes:
            raise ValueError("No archive entries match the requested heavy-size range")

        for heavy_size in selected_sizes:
            records, rejected, source_records = prepare_size(
                archive,
                entries[heavy_size],
                heavy_size,
                tokenizer,
                spectrum_length,
                smiles_max_length,
                args.overflow_policy,
                args.max_samples_per_size,
            )
            if records:
                save_size_dataset(output_dir, heavy_size, records, spectrum_length)
            truncated = sum(
                int(record["manifest"]["spectrum_truncated"])  # type: ignore[index]
                for record in records
            )
            summary = {
                "heavy_size": heavy_size,
                "source_records": source_records,
                "scanned_records": len(records) + sum(rejected.values()),
                "accepted": len(records),
                "truncated": truncated,
                "rejected": dict(sorted(rejected.items())),
            }
            size_summaries.append(summary)
            total_accepted += len(records)
            total_rejected.update(rejected)
            print(
                f"size {heavy_size}: accepted={len(records):,}; "
                f"truncated={truncated:,}; rejected={sum(rejected.values()):,}"
            )

    metadata = {
        "source_archive": str(input_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "spectrum_max_length": spectrum_length,
        "smiles_max_length": smiles_max_length,
        "smiles_vocab_size": tokenizer.vocab_size,
        "overflow_policy": args.overflow_policy,
        "min_heavy_size": args.min_heavy_size,
        "max_heavy_size": max(summary["heavy_size"] for summary in size_summaries),
        "max_samples_per_size": args.max_samples_per_size,
        "num_molecules": total_accepted,
        "rejected_molecules": sum(total_rejected.values()),
        "rejected_by_reason": dict(sorted(total_rejected.items())),
        "sizes": size_summaries,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(f"Saved {total_accepted:,} evaluation molecules to {output_dir}")


if __name__ == "__main__":
    main()
