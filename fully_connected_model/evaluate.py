#!/usr/bin/env python
"""Evaluate the fully connected baseline on a labelled data split."""

from __future__ import annotations

import argparse
import json

import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from fully_connected_model.common import (
    DEFAULT_DATA_DIRECTORY,
    DEFAULT_RESULTS_DIRECTORY,
    choose_device,
    load_checkpoint,
    load_split,
    project_path,
    top_k_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_RESULTS_DIRECTORY / "best.pt"))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIRECTORY))
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
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def main() -> None:
    args = parse_args()
    from utils.tokenizers import SPETokenizerWrapper

    device = choose_device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    model, _ = load_checkpoint(checkpoint_path, device)
    dataset = load_split(project_path(args.data_dir), args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    tokenizer = SPETokenizerWrapper()
    bos_index = tokenizer.VOCABS_INDICES["[BOS]"]
    eos_index = tokenizer.VOCABS_INDICES["[EOS]"]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0.0
    target_tokens = 0
    top1_correct = top3_correct = top5_correct = valid_top1 = count = 0
    RDLogger.DisableLog("rdApp.error")

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            freq, ir, raman, spectrum_mask, smiles_ids, _ = [item.to(device) for item in batch]
            logits = model(freq, ir, raman, spectrum_mask)
            total_loss += criterion(logits.transpose(1, 2), smiles_ids[:, 1:].long()).item()
            target_tokens += smiles_ids[:, 1:].ne(0).sum().item()
            generated, _ = top_k_sequences(logits, bos_index, eos_index, top_k=5)
            truth = tokenizer.decode_for_moses(smiles_ids.cpu())
            generated_cpu = generated.cpu()
            for row, target_smiles in enumerate(truth):
                target_canonical = canonicalize_smiles(target_smiles)
                candidate_smiles = tokenizer.decode_for_moses(generated_cpu[row])
                candidates = [canonicalize_smiles(value) for value in candidate_smiles]
                valid_top1 += candidates[0] is not None
                top1_correct += target_canonical is not None and target_canonical in candidates[:1]
                top3_correct += target_canonical is not None and target_canonical in candidates[:3]
                top5_correct += target_canonical is not None and target_canonical in candidates[:5]
            count += len(truth)

    result = {
        "model_type": "fully_connected_baseline",
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "samples": count,
        "cross_entropy_loss": total_loss / target_tokens,
        "canonical_top_1_accuracy": top1_correct / count,
        "canonical_top_3_accuracy": top3_correct / count,
        "canonical_top_5_accuracy": top5_correct / count,
        "top_1_valid_smiles_rate": valid_top1 / count,
        "device": str(device),
    }
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
