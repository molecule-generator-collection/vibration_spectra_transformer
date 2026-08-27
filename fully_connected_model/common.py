"""Shared helpers for the isolated fully connected baseline entry points."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.common import (  # noqa: E402,F401
    DistributedContext,
    choose_device,
    finalize_distributed,
    initialize_distributed,
    load_split,
    project_path,
    read_json,
    set_seed,
    write_json,
)

DEFAULT_DATA_DIRECTORY = ROOT / "data_diretory" / "processed"
DEFAULT_RESULTS_DIRECTORY = ROOT / "results" / "fully_connected_baseline"


def build_model_params(metadata: dict, args) -> dict:
    return {
        "spectrum_max_length": metadata["spectrum_max_length"],
        "smiles_max_length": metadata["smiles_max_length"],
        "smiles_vocab_size": metadata["smiles_vocab_size"],
        "hidden_dimension": args.hidden_dimension,
        "latent_dimension": args.latent_dimension,
        "encoder_num_layers": args.encoder_layers,
        "decoder_num_layers": args.decoder_layers,
        "dropout": args.dropout,
    }


def load_checkpoint(path: Path, device: torch.device):
    from fully_connected_model.model import FullyConnectedSmilesPredictor

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FullyConnectedSmilesPredictor(checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.inference_mode()
def top_k_sequences(
    logits: torch.Tensor,
    bos_index: int,
    eos_index: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine independent position probabilities into up to ``top_k`` sequences."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    batch_size, output_length, vocab_size = logits.shape
    beam_size = min(top_k, vocab_size)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    sequences = torch.full(
        (batch_size, 1, output_length + 1), 0, dtype=torch.long, device=logits.device
    )
    sequences[:, :, 0] = bos_index
    scores = torch.zeros((batch_size, 1), device=logits.device)
    lengths = torch.zeros((batch_size, 1), dtype=torch.long, device=logits.device)
    finished = torch.zeros((batch_size, 1), dtype=torch.bool, device=logits.device)

    for position in range(output_length):
        current_beams = scores.size(1)
        position_log_probs = log_probabilities[:, position].unsqueeze(1).expand(
            -1, current_beams, -1
        ).clone()
        if finished.any():
            position_log_probs.masked_fill_(finished.unsqueeze(-1), float("-inf"))
            padding_scores = position_log_probs[:, :, 0]
            position_log_probs[:, :, 0] = torch.where(
                finished, torch.zeros_like(padding_scores), padding_scores
            )

        candidate_scores = scores.unsqueeze(-1) + position_log_probs
        candidate_lengths = (
            lengths.unsqueeze(-1) + (~finished).long().unsqueeze(-1)
        ).expand(-1, -1, vocab_size)
        ranking_scores = candidate_scores / candidate_lengths.clamp_min(1)
        next_beams = min(beam_size, current_beams * vocab_size)
        _, flat_indices = ranking_scores.reshape(batch_size, -1).topk(next_beams, dim=-1)
        scores = candidate_scores.reshape(batch_size, -1).gather(1, flat_indices)
        lengths = candidate_lengths.reshape(batch_size, -1).gather(1, flat_indices)
        parent_indices = flat_indices // vocab_size
        next_tokens = flat_indices % vocab_size
        sequences = sequences.gather(
            1, parent_indices.unsqueeze(-1).expand(-1, -1, output_length + 1)
        )
        parent_finished = finished.gather(1, parent_indices)
        sequences[:, :, position + 1] = next_tokens
        finished = parent_finished | next_tokens.eq(eos_index)

    normalized_scores = scores / lengths.clamp_min(1)
    order = normalized_scores.argsort(dim=-1, descending=True)
    sequences = sequences.gather(
        1, order.unsqueeze(-1).expand(-1, -1, output_length + 1)
    )
    normalized_scores = normalized_scores.gather(1, order)
    return sequences, normalized_scores
