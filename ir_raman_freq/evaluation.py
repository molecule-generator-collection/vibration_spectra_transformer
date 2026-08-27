#!/usr/bin/env python
"""Evaluate an IR + Raman + frequency checkpoint on size-grouped MMP data."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import choose_device, padding_mask, project_path, read_json, write_json  # noqa: E402
from evaluate import beam_search, canonicalize_smiles, load_model  # noqa: E402
from ir_raman_freq.prepare_mmp_evaluation import (  # noqa: E402
    TENSOR_FILES,
    checkpoint_dimensions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="results/ir_raman_transformer/best.pt",
        help="Trained ir_raman_freq checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default="data_diretory/mmp_evaluation",
        help="Output directory from prepare_mmp_evaluation.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ir_raman_transformer/mmp_evaluation",
    )
    parser.add_argument("--min-heavy-size", type=int, default=10)
    parser.add_argument("--max-heavy-size", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Number of candidates; must be at least 5 for Top-1/3/5",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save molecule-level candidate and similarity CSV files per size",
    )
    return parser.parse_args(argv)


def read_manifest(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(
            f"Manifest has {len(rows)} rows but tensors have {expected_rows}: {path}"
        )
    required = {
        "sample_id",
        "canonical_smiles",
        "heavy_size",
        "spectrum_truncated",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def load_size_dataset(size_dir: Path) -> TensorDataset:
    missing = [name for name in TENSOR_FILES if not (size_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing tensors in {size_dir}: {', '.join(missing)}"
        )
    tensors = [
        torch.load(size_dir / name, map_location="cpu", weights_only=True)
        for name in TENSOR_FILES
    ]
    lengths = {len(tensor) for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError(
            f"Tensor lengths differ in {size_dir}: {[len(tensor) for tensor in tensors]}"
        )
    return TensorDataset(*tensors)


def available_sizes(data_dir: Path, minimum: int, maximum: int | None) -> list[int]:
    sizes = []
    for path in data_dir.glob("size*"):
        if not path.is_dir() or not path.name[4:].isdigit():
            continue
        size = int(path.name[4:])
        if size >= minimum and (maximum is None or size <= maximum):
            sizes.append(size)
    if not sizes:
        raise ValueError("No prepared size directories match the requested range")
    return sorted(sizes)


def molecular_similarity(
    target: Chem.Mol,
    predicted: Chem.Mol | None,
    fingerprint_generator,
) -> tuple[float, int, int | None, int | None]:
    if predicted is None:
        return 0.0, 0, None, None
    target_fingerprint = fingerprint_generator.GetFingerprint(target)
    predicted_fingerprint = fingerprint_generator.GetFingerprint(predicted)
    tanimoto = DataStructs.TanimotoSimilarity(
        target_fingerprint, predicted_fingerprint
    )
    formula_match = int(
        rdMolDescriptors.CalcMolFormula(target)
        == rdMolDescriptors.CalcMolFormula(predicted)
    )
    predicted_heavy_size = predicted.GetNumHeavyAtoms()
    heavy_size_error = abs(target.GetNumHeavyAtoms() - predicted_heavy_size)
    return tanimoto, formula_match, predicted_heavy_size, heavy_size_error


def empty_counts() -> dict[str, object]:
    return {
        "samples": 0,
        "loss_sum": 0.0,
        "target_tokens": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
        "top1_valid": 0,
        "tanimoto_sum": 0.0,
        "tanimoto_values": [],
        "formula_correct": 0,
        "heavy_size_error_sum": 0.0,
        "heavy_size_error_count": 0,
        "truncated": 0,
    }


def merge_counts(target: dict[str, object], source: dict[str, object]) -> None:
    for key in (
        "samples",
        "loss_sum",
        "target_tokens",
        "top1_correct",
        "top3_correct",
        "top5_correct",
        "top1_valid",
        "tanimoto_sum",
        "formula_correct",
        "heavy_size_error_sum",
        "heavy_size_error_count",
        "truncated",
    ):
        target[key] += source[key]  # type: ignore[operator]
    target["tanimoto_values"].extend(source["tanimoto_values"])  # type: ignore[union-attr]


def finalize_counts(counts: dict[str, object]) -> dict[str, object]:
    samples = int(counts["samples"])
    tokens = int(counts["target_tokens"])
    valid = int(counts["top1_valid"])
    tanimoto_values = counts["tanimoto_values"]
    heavy_error_count = int(counts["heavy_size_error_count"])
    if not samples or not tokens:
        raise ValueError("Cannot finalize empty evaluation counts")
    return {
        "samples": samples,
        "cross_entropy_loss": float(counts["loss_sum"]) / tokens,
        "canonical_top_1_accuracy": int(counts["top1_correct"]) / samples,
        "canonical_top_3_accuracy": int(counts["top3_correct"]) / samples,
        "canonical_top_5_accuracy": int(counts["top5_correct"]) / samples,
        "top_1_valid_smiles_rate": valid / samples,
        "mean_top_1_tanimoto": float(counts["tanimoto_sum"]) / samples,
        "median_top_1_tanimoto": statistics.median(tanimoto_values),
        "molecular_formula_accuracy": int(counts["formula_correct"]) / samples,
        "mean_valid_top_1_heavy_atom_count_absolute_error": (
            float(counts["heavy_size_error_sum"]) / heavy_error_count
            if heavy_error_count
            else None
        ),
        "truncated_spectra": int(counts["truncated"]),
        "truncated_spectra_rate": int(counts["truncated"]) / samples,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows: list[dict[str, object]], include_header: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if include_header else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if include_header:
            writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate_size(
    heavy_size: int,
    dataset: TensorDataset,
    manifest: list[dict[str, str]],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    beam_size: int,
    save_predictions: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    bos_index = tokenizer.VOCABS_INDICES["[BOS]"]
    eos_index = tokenizer.VOCABS_INDICES["[EOS]"]
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    counts = empty_counts()
    prediction_rows: list[dict[str, object]] = []
    offset = 0

    for batch in tqdm(loader, desc=f"Evaluating size {heavy_size}"):
        freq, ir, raman, spectrum_mask, smiles_ids, smiles_mask = [
            item.to(device) for item in batch
        ]
        spectrum_padding = padding_mask(spectrum_mask)
        smiles_padding = padding_mask(smiles_mask)
        logits = model(
            freq,
            ir,
            raman,
            spectrum_padding,
            smiles_ids,
            smiles_padding,
        ).transpose(1, 2)
        counts["loss_sum"] += criterion(
            logits[:, :, :-1], smiles_ids[:, 1:].long()
        ).item()
        counts["target_tokens"] += smiles_ids[:, 1:].ne(0).sum().item()
        generated = beam_search(
            model,
            freq,
            ir,
            raman,
            spectrum_padding,
            bos_index,
            eos_index,
            beam_size=beam_size,
        ).cpu()
        truth_smiles = tokenizer.decode_for_moses(smiles_ids.cpu())

        for row, decoded_truth in enumerate(truth_smiles):
            manifest_row = manifest[offset + row]
            target_canonical = canonicalize_smiles(decoded_truth)
            manifest_canonical = canonicalize_smiles(manifest_row["canonical_smiles"])
            if target_canonical is None or target_canonical != manifest_canonical:
                raise ValueError(
                    "Manifest and tensor labels are misaligned at "
                    f"{manifest_row['sample_id']}"
                )
            candidates = tokenizer.decode_for_moses(generated[row])
            candidate_canonical = [canonicalize_smiles(value) for value in candidates]
            truth_rank = next(
                (
                    rank
                    for rank, value in enumerate(candidate_canonical, start=1)
                    if value == target_canonical
                ),
                None,
            )
            predicted_molecule = (
                Chem.MolFromSmiles(candidate_canonical[0])
                if candidate_canonical[0] is not None
                else None
            )
            target_molecule = Chem.MolFromSmiles(target_canonical)
            tanimoto, formula_match, predicted_size, size_error = molecular_similarity(
                target_molecule, predicted_molecule, fingerprint_generator
            )

            counts["samples"] += 1
            counts["top1_correct"] += int(truth_rank == 1)
            counts["top3_correct"] += int(
                truth_rank is not None and truth_rank <= 3
            )
            counts["top5_correct"] += int(
                truth_rank is not None and truth_rank <= 5
            )
            counts["top1_valid"] += int(candidate_canonical[0] is not None)
            counts["tanimoto_sum"] += tanimoto
            counts["tanimoto_values"].append(tanimoto)
            counts["formula_correct"] += formula_match
            if size_error is not None:
                counts["heavy_size_error_sum"] += size_error
                counts["heavy_size_error_count"] += 1
            counts["truncated"] += int(manifest_row["spectrum_truncated"])

            if save_predictions:
                prediction_rows.append(
                    {
                        **manifest_row,
                        "target_smiles": decoded_truth,
                        "predicted_smiles": candidates[0],
                        "predicted_canonical_smiles": candidate_canonical[0] or "",
                        "candidate_smiles_json": json.dumps(candidates, ensure_ascii=False),
                        "candidate_canonical_smiles_json": json.dumps(
                            [value or "" for value in candidate_canonical],
                            ensure_ascii=False,
                        ),
                        "truth_rank": truth_rank or "",
                        "top1_correct": int(truth_rank == 1),
                        "top3_correct": int(
                            truth_rank is not None and truth_rank <= 3
                        ),
                        "top5_correct": int(
                            truth_rank is not None and truth_rank <= 5
                        ),
                        "top1_valid": int(candidate_canonical[0] is not None),
                        "top1_tanimoto": tanimoto,
                        "molecular_formula_correct": formula_match,
                        "predicted_heavy_size": predicted_size or "",
                        "heavy_size_absolute_error": (
                            size_error if size_error is not None else ""
                        ),
                    }
                )
        offset += len(truth_smiles)

    return counts, prediction_rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.min_heavy_size < 1:
        raise ValueError("--min-heavy-size must be at least 1")
    if args.max_heavy_size is not None and args.max_heavy_size < args.min_heavy_size:
        raise ValueError("--max-heavy-size must be >= --min-heavy-size")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.beam_size < 5:
        raise ValueError("--beam-size must be at least 5")

    from utils.tokenizers import SPETokenizerWrapper

    checkpoint_path = project_path(args.checkpoint)
    data_dir = project_path(args.data_dir)
    output_dir = project_path(args.output_dir)
    metadata = read_json(data_dir / "metadata.json")
    device = choose_device(args.device)
    model = load_model(checkpoint_path, device)
    checkpoint_spectrum_length, checkpoint_smiles_length, checkpoint_vocab_size = (
        checkpoint_dimensions(checkpoint_path)
    )
    tokenizer = SPETokenizerWrapper()
    if list(model.encoder.input_modalities) != ["freq", "ir", "raman"]:
        raise ValueError(
            "Checkpoint is not an IR + Raman + frequency model: "
            f"{model.encoder.input_modalities}"
        )
    if int(metadata["spectrum_max_length"]) != checkpoint_spectrum_length:
        raise ValueError(
            "Prepared spectrum length does not match checkpoint: "
            f"{metadata['spectrum_max_length']} != {checkpoint_spectrum_length}"
        )
    if (
        int(metadata["smiles_max_length"]) != checkpoint_smiles_length
        or checkpoint_smiles_length != model.smiles_max_length
    ):
        raise ValueError(
            "Prepared SMILES length does not match checkpoint: "
            f"{metadata['smiles_max_length']} != {model.smiles_max_length}"
        )
    if (
        int(metadata["smiles_vocab_size"]) != checkpoint_vocab_size
        or checkpoint_vocab_size != tokenizer.vocab_size
    ):
        raise ValueError("Prepared tokenizer vocabulary does not match repository tokenizer")

    sizes = available_sizes(
        data_dir, args.min_heavy_size, args.max_heavy_size
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    overall_counts = empty_counts()
    size_results: list[dict[str, object]] = []
    combined_predictions_written = False

    for heavy_size in sizes:
        size_dir = data_dir / f"size{heavy_size}"
        dataset = load_size_dataset(size_dir)
        if dataset.tensors[0].shape[1] != int(metadata["spectrum_max_length"]):
            raise ValueError(f"Spectrum length differs from metadata in {size_dir}")
        manifest = read_manifest(size_dir / "analysis_manifest.csv", len(dataset))
        counts, prediction_rows = evaluate_size(
            heavy_size,
            dataset,
            manifest,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.beam_size,
            args.save_predictions,
        )
        merge_counts(overall_counts, counts)
        size_result = {"heavy_size": heavy_size, **finalize_counts(counts)}
        size_results.append(size_result)
        if args.save_predictions:
            write_rows(
                output_dir / "predictions" / f"size{heavy_size}.csv",
                prediction_rows,
            )
            append_rows(
                output_dir / "evaluation_molecules.csv",
                prediction_rows,
                include_header=not combined_predictions_written,
            )
            combined_predictions_written = combined_predictions_written or bool(
                prediction_rows
            )
        print(json.dumps(size_result, ensure_ascii=False))

    overall = finalize_counts(overall_counts)
    result = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "device": str(device),
        "beam_size": args.beam_size,
        "overflow_policy": metadata.get("overflow_policy"),
        "overall": overall,
        "by_heavy_size": size_results,
    }
    if args.save_predictions and combined_predictions_written:
        result["molecule_output"] = str(output_dir / "evaluation_molecules.csv")
    write_json(output_dir / "evaluation.json", result)
    write_rows(output_dir / "evaluation_by_heavy_size.csv", size_results)
    print(json.dumps({"overall": overall}, ensure_ascii=False, indent=2))
    print(f"Saved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
