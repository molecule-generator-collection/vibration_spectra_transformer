#!/usr/bin/env python
"""Predict SMILES with the fully connected baseline from a spectra CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json

import torch

from fully_connected_model.common import (
    DEFAULT_RESULTS_DIRECTORY,
    ROOT,
    choose_device,
    load_checkpoint,
    project_path,
    top_k_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing freq, IR, and Raman columns")
    parser.add_argument("--checkpoint", default=str(DEFAULT_RESULTS_DIRECTORY / "best.pt"))
    parser.add_argument("--output", default=str(ROOT / "results" / "fully_connected_predictions.csv"))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_spectrum(value: str, column: str, row: int) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
        return [float(item) for item in parsed]
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"Invalid {column} list at CSV row {row}") from error


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    from utils.tokenizers import SPETokenizerWrapper

    input_path = project_path(args.input)
    output_path = project_path(args.output)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(project_path(args.checkpoint), device)
    tokenizer = SPETokenizerWrapper()
    bos_index = tokenizer.VOCABS_INDICES["[BOS]"]
    eos_index = tokenizer.VOCABS_INDICES["[EOS]"]
    spectrum_length = checkpoint["model_params"]["spectrum_max_length"]

    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"freq", "IR", "Raman"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain columns: {', '.join(sorted(required))}")

    output_rows = []
    for row_number, row in enumerate(rows, start=2):
        values = [parse_spectrum(row[column], column, row_number) for column in ("freq", "IR", "Raman")]
        if not values[0] or not (len(values[0]) == len(values[1]) == len(values[2])):
            raise ValueError(f"freq, IR, and Raman lengths must match and be non-empty at CSV row {row_number}")
        used = min(len(values[0]), spectrum_length)
        tensors = [torch.zeros((1, spectrum_length), dtype=torch.float32, device=device) for _ in range(3)]
        for tensor, sequence in zip(tensors, values):
            tensor[0, :used] = torch.tensor(sequence[:used], dtype=torch.float32, device=device)
        mask = torch.zeros((1, spectrum_length), dtype=torch.bool, device=device)
        mask[0, :used] = True
        with torch.inference_mode():
            logits = model(*tensors, mask)
            sequences, scores = top_k_sequences(logits, bos_index, eos_index, args.top_k)
        smiles = tokenizer.decode_for_moses(sequences[0].cpu())
        candidates = [
            {"smiles": value, "mean_log_probability": score}
            for value, score in zip(smiles, scores[0].cpu().tolist())
        ]
        result = dict(row)
        result["predicted_smiles"] = candidates[0]["smiles"]
        result["candidates_json"] = json.dumps(candidates, ensure_ascii=False)
        output_rows.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
