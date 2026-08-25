#!/usr/bin/env python
"""Identify candidate SMILES from vibrational spectra in a CSV file.

The required CSV columns are inferred from the checkpoint's input modalities.
Each spectrum cell is a JSON/Python-style numeric list, for example
``"[650.0, 712.3]"``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import choose_device, padding_mask, project_path  # noqa: E402
from constants import MODEL_NAME, RESULTS_DIRECTORY  # noqa: E402
from modules.models import SmilesPredictor  # noqa: E402


def parse_args(
    default_checkpoint: str | Path = RESULTS_DIRECTORY / MODEL_NAME / "best.pt",
    default_output: str | Path = "results/predictions.csv",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_spectrum(value: str, column: str, row: int) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
        return [float(item) for item in parsed]
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"Invalid {column} list at CSV row {row}") from error


def beam_search(model, freq, ir, raman, mask, tokenizer, top_k: int) -> list[tuple[str, float]]:
    if top_k < 1:
        raise ValueError("--top-k must be at least 1")
    bos = tokenizer.VOCABS_INDICES["[BOS]"]
    eos = tokenizer.VOCABS_INDICES["[EOS]"]
    max_length = model.smiles_max_length
    with torch.no_grad():
        memory = model.encoder.encode(freq, ir, raman, padding_mask(mask))
        beams = [([bos], 0.0, False)]
        for _ in range(1, max_length):
            candidates = []
            for tokens, score, ended in beams:
                if ended:
                    candidates.append((tokens, score, True))
                    continue
                decoder_ids = torch.zeros((1, max_length), dtype=torch.long, device=freq.device)
                decoder_ids[0, :len(tokens)] = torch.tensor(tokens, device=freq.device)
                embedded = model.smiles_emb(decoder_ids)
                logits = model.decoder(
                    memory, embedded, padding_mask(mask),
                    torch.zeros((1, max_length), dtype=torch.bool, device=freq.device),
                )[0, len(tokens) - 1]
                values, indices = torch.log_softmax(logits, dim=-1).topk(top_k)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((tokens + [index], score + value, index == eos))
            beams = sorted(candidates, key=lambda item: item[1] / len(item[0]), reverse=True)[:top_k]
            if all(item[2] for item in beams):
                break
    tensor_ids = [torch.tensor(tokens) for tokens, _, _ in beams]
    smiles = tokenizer.decode_for_moses(tensor_ids)
    return [(value, score / len(tokens)) for value, (tokens, score, _) in zip(smiles, beams)]


def main(
    default_checkpoint: str | Path = RESULTS_DIRECTORY / MODEL_NAME / "best.pt",
    default_output: str | Path = "results/predictions.csv",
) -> None:
    args = parse_args(default_checkpoint, default_output)
    from utils.tokenizers import SPETokenizerWrapper

    input_path = project_path(args.input)
    output_path = project_path(args.output)
    checkpoint_path = project_path(args.checkpoint)
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SmilesPredictor(checkpoint["model_params"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = SPETokenizerWrapper()
    spectrum_length = checkpoint["model_params"]["spectrum_max_length"]
    modalities = model.encoder.input_modalities
    column_names = {"freq": "freq", "ir": "IR", "raman": "Raman"}
    required_columns = [column_names[name] for name in modalities]

    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(required_columns).issubset(rows[0]):
        raise ValueError(f"Input CSV must contain columns: {', '.join(required_columns)}")

    output_rows = []
    for row_number, row in enumerate(rows, start=2):
        parsed_values = {
            name: parse_spectrum(row[column_names[name]], column_names[name], row_number)
            for name in modalities
        }
        lengths = {len(value) for value in parsed_values.values()}
        if not parsed_values["freq"] or len(lengths) != 1:
            raise ValueError(
                f"Required spectra must have matching, non-empty lengths at CSV row {row_number}"
            )
        used = min(len(parsed_values["freq"]), spectrum_length)
        tensors = [torch.full((1, spectrum_length), -1.0, device=device) for _ in range(3)]
        tensor_by_name = dict(zip(("freq", "ir", "raman"), tensors))
        for name, sequence in parsed_values.items():
            tensor_by_name[name][0, :used] = torch.tensor(
                sequence[:used], dtype=torch.float32, device=device
            )
        mask = torch.zeros((1, spectrum_length), dtype=torch.int8, device=device)
        mask[0, :used] = 1
        candidates = beam_search(model, *tensors, mask, tokenizer, args.top_k)
        result = dict(row)
        result["predicted_smiles"] = candidates[0][0]
        result["candidates_json"] = json.dumps(
            [{"smiles": smiles, "mean_log_probability": score} for smiles, score in candidates],
            ensure_ascii=False,
        )
        output_rows.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
