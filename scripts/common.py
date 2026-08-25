from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset

try:
    from .constants import PROJECT_ROOT
except ImportError:  # Direct execution through scripts/train.py.
    from constants import PROJECT_ROOT


TENSOR_FILES = (
    "freqs.pt",
    "IRs.pt",
    "Ramans.pt",
    "freq_attention_masks.pt",
    "smiles_ids.pt",
    "smiles_attention_masks.pt",
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for single-process or torchrun-based training."""

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def all_ranks_true(self, value: bool) -> bool:
        if not self.enabled:
            return value
        flag = torch.tensor(int(value), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def sum(self, value: float) -> float:
        if not self.enabled:
            return value
        total = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return float(total.item())

    def broadcast_float(self, value: float, source: int = 0) -> float:
        if not self.enabled:
            return value
        result = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.broadcast(result, src=source)
        return float(result.item())


def initialize_distributed(requested_device: str) -> DistributedContext:
    """Initialize NCCL when launched by ``torchrun``; otherwise stay local."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(device=choose_device(requested_device))
    if requested_device == "cpu":
        raise ValueError("Multi-GPU training cannot be used with --device cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun requested multiple processes, but CUDA is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(
        device=torch.device("cuda", local_rank),
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def finalize_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data_dir: Path, split: str, require_labels: bool = True) -> TensorDataset:
    split_dir = data_dir / split
    names = TENSOR_FILES if require_labels else TENSOR_FILES[:4]
    missing = [str(split_dir / name) for name in names if not (split_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset is not prepared. Missing:\n  " + "\n  ".join(missing)
            + "\nRun: python scripts/prepare_data.py"
        )
    tensors = [torch.load(split_dir / name, map_location="cpu", weights_only=True) for name in names]
    lengths = {len(tensor) for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError(f"Tensor lengths differ in {split_dir}: {[len(t) for t in tensors]}")
    return TensorDataset(*tensors)


def build_model_params(metadata: dict, args) -> dict:
    return {
        "input_modalities": args.modalities,
        "encoder_num_layers": args.encoder_layers,
        "smiles_max_length": metadata["smiles_max_length"],
        "spectrum_max_length": metadata["spectrum_max_length"],
        "encoder_hidden_dimention": args.hidden_dimension,
        "encoder_n_heads": args.attention_heads,
        "encoder_dropout_rate": args.dropout,
        "decoder_hidden_dimention": args.hidden_dimension,
        "decoder_n_heads": args.attention_heads,
        "decoder_dropout_rate": args.dropout,
        "decoder_num_layers": args.decoder_layers,
        "smiles_vocab_size": metadata["smiles_vocab_size"],
        "embed_dimention": args.hidden_dimension,
        "embed_dropout_rate": args.dropout,
    }


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert 1=valid/0=padding masks to PyTorch's True=padding convention."""
    return ~attention_mask.bool()


def decode_ids(tokenizer, ids: Iterable[torch.Tensor]) -> list[str]:
    return tokenizer.decode_for_moses(ids)
