"""Non-autoregressive fully connected spectrum-to-SMILES baseline."""

from __future__ import annotations

import torch
from torch import nn


def _hidden_block(input_dimension: int, output_dimension: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dimension, output_dimension),
        nn.LayerNorm(output_dimension),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FullyConnectedEncoder(nn.Module):
    """Encode the three masked spectra into one fixed-size latent vector."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        spectrum_length = params["spectrum_max_length"]
        hidden_dimension = params["hidden_dimension"]
        latent_dimension = params["latent_dimension"]
        num_layers = params["encoder_num_layers"]
        dropout = params["dropout"]
        if num_layers < 1:
            raise ValueError("encoder_num_layers must be at least 1")

        input_dimension = 3 * spectrum_length
        if num_layers == 1:
            self.network = nn.Sequential(nn.Linear(input_dimension, latent_dimension))
        else:
            layers: list[nn.Module] = [_hidden_block(input_dimension, hidden_dimension, dropout)]
            for _ in range(num_layers - 2):
                layers.append(_hidden_block(hidden_dimension, hidden_dimension, dropout))
            layers.append(nn.Linear(hidden_dimension, latent_dimension))
            self.network = nn.Sequential(*layers)

    def forward(
        self,
        freq: torch.Tensor,
        ir: torch.Tensor,
        raman: torch.Tensor,
        spectrum_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, latent_dimension]``; mask uses 1/True for valid values."""
        valid = spectrum_mask.bool()
        spectra = torch.stack((freq, ir, raman), dim=1)
        spectra = spectra.masked_fill(~valid.unsqueeze(1), 0.0)
        return self.network(spectra.flatten(start_dim=1))

    def encode(
        self,
        freq: torch.Tensor,
        ir: torch.Tensor,
        raman: torch.Tensor,
        spectrum_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(freq, ir, raman, spectrum_mask)


class FullyConnectedDecoder(nn.Module):
    """Decode one latent vector into independent token logits for every position."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.output_length = params["smiles_max_length"] - 1
        self.hidden_dimension = params["hidden_dimension"]
        self.vocab_size = params["smiles_vocab_size"]
        latent_dimension = params["latent_dimension"]
        num_layers = params["decoder_num_layers"]
        dropout = params["dropout"]
        if self.output_length < 1:
            raise ValueError("smiles_max_length must be at least 2")
        if num_layers < 1:
            raise ValueError("decoder_num_layers must be at least 1")

        # The position projection avoids a very large hidden-to-(length*vocab)
        # matrix while remaining a purely fully connected decoder.
        position_dimension = self.output_length * self.hidden_dimension
        layers: list[nn.Module] = []
        current_dimension = latent_dimension
        for _ in range(num_layers - 1):
            layers.append(_hidden_block(current_dimension, self.hidden_dimension, dropout))
            current_dimension = self.hidden_dimension
        layers.append(nn.Linear(current_dimension, position_dimension))
        self.position_projection = nn.Sequential(*layers)
        self.vocabulary_head = nn.Linear(self.hidden_dimension, self.vocab_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        positions = self.position_projection(latent).reshape(
            latent.size(0), self.output_length, self.hidden_dimension
        )
        return self.vocabulary_head(positions)


class FullyConnectedSmilesPredictor(nn.Module):
    """MLP baseline with no attention, recurrence, convolution, or token embedding."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.model_params = dict(params)
        self.smiles_max_length = params["smiles_max_length"]
        self.encoder = FullyConnectedEncoder(params)
        self.decoder = FullyConnectedDecoder(params)

    def forward(
        self,
        freq: torch.Tensor,
        ir: torch.Tensor,
        raman: torch.Tensor,
        spectrum_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped ``[batch, smiles_max_length - 1, vocab_size]``."""
        return self.decoder(self.encoder(freq, ir, raman, spectrum_mask))

    @torch.inference_mode()
    def generate(
        self,
        freq: torch.Tensor,
        ir: torch.Tensor,
        raman: torch.Tensor,
        spectrum_mask: torch.Tensor,
        bos_index: int,
    ) -> torch.Tensor:
        predicted = self(freq, ir, raman, spectrum_mask).argmax(dim=-1)
        bos = torch.full(
            (predicted.size(0), 1), bos_index, dtype=torch.long, device=predicted.device
        )
        return torch.cat((bos, predicted), dim=1)
