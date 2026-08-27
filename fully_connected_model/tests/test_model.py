import torch

from fully_connected_model.common import top_k_sequences
from fully_connected_model.model import FullyConnectedSmilesPredictor


def model_params() -> dict:
    return {
        "spectrum_max_length": 5,
        "smiles_max_length": 6,
        "smiles_vocab_size": 11,
        "hidden_dimension": 8,
        "latent_dimension": 4,
        "encoder_num_layers": 2,
        "decoder_num_layers": 2,
        "dropout": 0.0,
    }


def test_forward_and_generate_shapes() -> None:
    model = FullyConnectedSmilesPredictor(model_params()).eval()
    spectra = [torch.randn(3, 5) for _ in range(3)]
    mask = torch.ones(3, 5, dtype=torch.bool)
    logits = model(*spectra, mask)
    generated = model.generate(*spectra, mask, bos_index=9)
    assert logits.shape == (3, 5, 11)
    assert generated.shape == (3, 6)
    assert generated[:, 0].eq(9).all()


def test_padding_values_do_not_change_output() -> None:
    model = FullyConnectedSmilesPredictor(model_params()).eval()
    spectra = [torch.randn(2, 5) for _ in range(3)]
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    changed = [value.masked_fill(~mask, 12345.0) for value in spectra]
    assert torch.allclose(model(*spectra, mask), model(*changed, mask))


def test_top_k_sequences_include_bos_and_rank_candidates() -> None:
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 4.0
    logits[0, 1, 4] = 4.0  # EOS
    sequences, scores = top_k_sequences(logits, bos_index=3, eos_index=4, top_k=2)
    assert sequences.shape == (1, 2, 4)
    assert scores.shape == (1, 2)
    assert sequences[0, 0, :3].tolist() == [3, 2, 4]
    assert scores[0, 0] >= scores[0, 1]
