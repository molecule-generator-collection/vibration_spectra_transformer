#!/usr/bin/env python
"""Evaluate canonical-SMILES Top-1/3/5 accuracy on a labelled data split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import choose_device, load_split, padding_mask, project_path  # noqa: E402
from constants import DATA_DIRECTORY, MODEL_NAME, RESULTS_DIRECTORY  # noqa: E402
from modules.models import SmilesPredictor  # noqa: E402


def parse_args(
    default_checkpoint: str | Path = RESULTS_DIRECTORY / MODEL_NAME / "best.pt",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--data-dir", default=str(DATA_DIRECTORY))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        help=(
            "JSON result path (default: evaluation_<split>.json next to the "
            "checkpoint)"
        ),
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "Save molecule-level predictions joined to the split's analysis "
            "manifest. The manifest is created by prepare_data.py --analysis."
        ),
    )
    parser.add_argument(
        "--manifest",
        help="Analysis manifest CSV (default: <data-dir>/<split>/analysis_manifest.csv)",
    )
    parser.add_argument(
        "--molecule-output",
        help=(
            "Molecule-level CSV path (default: evaluation_<split>_molecules.csv "
            "next to the checkpoint)"
        ),
    )
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> SmilesPredictor:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SmilesPredictor(checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit canonical isomeric SMILES, or None for an invalid SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def read_manifest(path: Path, expected_rows: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Analysis manifest not found: {path}\n"
            "Re-run: python scripts/prepare_data.py --analysis"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(
            f"Manifest has {len(rows)} rows but the dataset has {expected_rows}: {path}"
        )
    required = {"sample_id", "canonical_smiles", "heavy_size"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty molecule evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def beam_search(
    model: SmilesPredictor,
    freq: torch.Tensor,
    ir: torch.Tensor,
    raman: torch.Tensor,
    spectrum_padding: torch.Tensor,
    bos_index: int,
    eos_index: int,
    beam_size: int = 5,
) -> torch.Tensor:
    """Generate the highest-scoring complete token sequences for each sample."""
    memory = model.encoder.encode(freq, ir, raman, spectrum_padding)
    batch_size = freq.size(0)
    max_length = model.smiles_max_length
    vocab_size = model.decoder.fc_out.out_features

    sequences = torch.zeros(
        (batch_size, 1, max_length), dtype=torch.long, device=freq.device
    )
    sequences[:, :, 0] = bos_index
    scores = torch.zeros((batch_size, 1), device=freq.device)
    lengths = torch.zeros((batch_size, 1), dtype=torch.long, device=freq.device)
    finished = torch.zeros((batch_size, 1), dtype=torch.bool, device=freq.device)

    for position in range(1, max_length):
        current_beams = sequences.size(1)
        flat_sequences = sequences.reshape(batch_size * current_beams, max_length)
        expanded_memory = memory.repeat_interleave(current_beams, dim=0)
        expanded_spectrum_padding = spectrum_padding.repeat_interleave(current_beams, dim=0)
        decoder_padding = torch.zeros_like(flat_sequences, dtype=torch.bool)
        logits = model.decoder(
            expanded_memory,
            model.smiles_emb(flat_sequences),
            expanded_spectrum_padding,
            decoder_padding,
        )[:, position - 1, :]
        log_probabilities = torch.log_softmax(logits, dim=-1).reshape(
            batch_size, current_beams, vocab_size
        )

        # A completed beam is carried forward without changing its score.
        if finished.any():
            log_probabilities = log_probabilities.masked_fill(
                finished.unsqueeze(-1), float("-inf")
            )
            eos_scores = log_probabilities[:, :, eos_index]
            log_probabilities[:, :, eos_index] = torch.where(
                finished, torch.zeros_like(eos_scores), eos_scores
            )

        candidate_scores = scores.unsqueeze(-1) + log_probabilities
        candidate_lengths = (
            lengths.unsqueeze(-1) + (~finished).long().unsqueeze(-1)
        ).expand(-1, -1, vocab_size)
        ranking_scores = candidate_scores / candidate_lengths.clamp_min(1)
        next_beams = min(beam_size, current_beams * vocab_size)
        _, flat_indices = ranking_scores.reshape(batch_size, -1).topk(
            next_beams, dim=-1
        )
        scores = candidate_scores.reshape(batch_size, -1).gather(1, flat_indices)
        lengths = candidate_lengths.reshape(batch_size, -1).gather(1, flat_indices)
        parent_indices = flat_indices // vocab_size
        next_tokens = flat_indices % vocab_size
        sequences = sequences.gather(
            1, parent_indices.unsqueeze(-1).expand(-1, -1, max_length)
        )
        parent_finished = finished.gather(1, parent_indices)
        sequences[:, :, position] = next_tokens
        finished = parent_finished | next_tokens.eq(eos_index)
        if finished.all():
            break

    return sequences


def main(
    default_checkpoint: str | Path = RESULTS_DIRECTORY / MODEL_NAME / "best.pt",
) -> None:
    args = parse_args(default_checkpoint)
    from utils.tokenizers import SPETokenizerWrapper

    device = choose_device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    dataset = load_split(project_path(args.data_dir), args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(checkpoint_path, device)
    modalities = model.encoder.input_modalities
    tokenizer = SPETokenizerWrapper()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    bos_index = tokenizer.VOCABS_INDICES["[BOS]"]
    eos_index = tokenizer.VOCABS_INDICES["[EOS]"]
    total_loss = 0.0
    target_tokens = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    valid_top1 = 0
    count = 0
    molecule_results: list[dict[str, object]] = []
    manifest_rows: list[dict[str, str]] | None = None
    if args.analysis:
        manifest_path = (
            project_path(args.manifest)
            if args.manifest
            else project_path(args.data_dir) / args.split / "analysis_manifest.csv"
        )
        manifest_rows = read_manifest(manifest_path, len(dataset))
    RDLogger.DisableLog("rdApp.error")

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            freq, ir, raman, spectrum_mask, smiles_ids, smiles_mask = [item.to(device) for item in batch]
            spectrum_padding = padding_mask(spectrum_mask)
            smiles_padding = padding_mask(smiles_mask)
            logits = model(freq, ir, raman, spectrum_padding, smiles_ids, smiles_padding).transpose(1, 2)
            total_loss += criterion(logits[:, :, :-1], smiles_ids[:, 1:].long()).item()
            target_tokens += smiles_ids[:, 1:].ne(0).sum().item()

            generated = beam_search(
                model,
                freq,
                ir,
                raman,
                spectrum_padding,
                bos_index,
                eos_index,
            )
            truth = tokenizer.decode_for_moses(smiles_ids.cpu())
            generated_cpu = generated.cpu()
            for row, target_smiles in enumerate(truth):
                target_canonical = canonicalize_smiles(target_smiles)
                candidate_smiles = tokenizer.decode_for_moses(generated_cpu[row])
                candidate_canonical = [canonicalize_smiles(value) for value in candidate_smiles]
                valid_top1 += candidate_canonical[0] is not None
                top1_correct += target_canonical is not None and target_canonical in candidate_canonical[:1]
                top3_correct += target_canonical is not None and target_canonical in candidate_canonical[:3]
                top5_correct += target_canonical is not None and target_canonical in candidate_canonical[:5]
                if manifest_rows is not None:
                    manifest = manifest_rows[count + row]
                    manifest_canonical = canonicalize_smiles(manifest["canonical_smiles"])
                    if target_canonical != manifest_canonical:
                        raise ValueError(
                            "Analysis manifest is not aligned with the evaluation "
                            f"dataset at row {count + row}: target={target_canonical!r}, "
                            f"manifest={manifest_canonical!r}"
                        )
                    truth_rank = next(
                        (
                            rank
                            for rank, value in enumerate(candidate_canonical, start=1)
                            if target_canonical is not None and value == target_canonical
                        ),
                        None,
                    )
                    molecule_results.append(
                        {
                            **manifest,
                            "target_smiles": target_smiles,
                            "target_canonical_smiles": target_canonical or "",
                            "predicted_smiles": candidate_smiles[0],
                            "predicted_canonical_smiles": candidate_canonical[0] or "",
                            "candidate_smiles_json": json.dumps(
                                candidate_smiles, ensure_ascii=False
                            ),
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
                        }
                    )
            count += len(truth)

    result = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "samples": count,
        "input_modalities": modalities,
        "cross_entropy_loss": total_loss / target_tokens,
        "canonical_top_1_accuracy": top1_correct / count,
        "canonical_top_3_accuracy": top3_correct / count,
        "canonical_top_5_accuracy": top5_correct / count,
        "top_1_valid_smiles_rate": valid_top1 / count,
        "device": str(device),
    }
    if args.analysis:
        molecule_output = (
            project_path(args.molecule_output)
            if args.molecule_output
            else checkpoint_path.parent / f"evaluation_{args.split}_molecules.csv"
        )
        write_csv(molecule_output, molecule_results)
        result["molecule_output"] = str(molecule_output)
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(result_json)
    output = (
        project_path(args.output)
        if args.output
        else checkpoint_path.parent / f"evaluation_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result_json + "\n", encoding="utf-8")
    print(f"Saved evaluation result to {output}")


if __name__ == "__main__":
    main()
