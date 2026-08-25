#!/usr/bin/env python
"""Train a vibrational-spectra-to-SMILES transformer using local data."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import DistributedContext, build_model_params, finalize_distributed, initialize_distributed, load_split, padding_mask, project_path, read_json, set_seed, write_json  # noqa: E402
from constants import DATA_DIRECTORY, DECODER_NUM_LAYERS, DROPOUT_RATE, ENCODER_HIDDEN_DIMENSION, ENCODER_N_HEADS, ENCODER_NUM_LAYERS, LR, MODEL_NAME, RESULTS_DIRECTORY, SEED  # noqa: E402
from modules.models import SmilesPredictor  # noqa: E402


def parse_args(
    default_output_dir: str | Path = RESULTS_DIRECTORY / MODEL_NAME,
    default_modalities: tuple[str, ...] = ("freq", "ir", "raman"),
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIRECTORY))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("freq", "ir", "raman"),
        default=list(default_modalities),
        help="Spectrum inputs used by the encoder",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--learning-rate",
        "--initial-learning-rate",
        dest="learning_rate",
        type=float,
        default=LR,
        help=f"Initial AdamW learning rate (default: {LR:g})",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hidden-dimension", type=int, default=ENCODER_HIDDEN_DIMENSION)
    parser.add_argument("--attention-heads", type=int, default=ENCODER_N_HEADS)
    parser.add_argument("--encoder-layers", type=int, default=ENCODER_NUM_LAYERS)
    parser.add_argument("--decoder-layers", type=int, default=DECODER_NUM_LAYERS)
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT_RATE,
        help=(
            "Dropout probability used by the encoder, decoder, and SMILES "
            f"embedding (default: {DROPOUT_RATE:g})"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def loss_for_batch(model, batch, device, criterion):
    freq, ir, raman, spectrum_mask, smiles_ids, smiles_mask = [item.to(device) for item in batch]
    logits = model(
        freq, ir, raman, padding_mask(spectrum_mask), smiles_ids, padding_mask(smiles_mask)
    ).transpose(1, 2)
    return criterion(logits[:, :, :-1], smiles_ids[:, 1:].long())


@torch.no_grad()
def validate(model, loader, device, criterion) -> float:
    model.eval()
    return sum(loss_for_batch(model, batch, device, criterion).item() for batch in loader) / len(loader)


def train(args: argparse.Namespace, distributed: DistributedContext) -> None:
    """Run training after device/distributed initialization."""

    device = distributed.device
    data_dir = project_path(args.data_dir)
    output_dir = project_path(args.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.enabled:
        torch.distributed.barrier()
    set_seed(args.seed)
    metadata = read_json(data_dir / "metadata.json")
    model_params = build_model_params(metadata, args)
    run_config = {
        **vars(args),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device_used": str(device),
        "distributed": distributed.enabled,
        "world_size": distributed.world_size,
        "global_batch_size": args.batch_size * distributed.world_size,
    }
    if distributed.is_main:
        write_json(output_dir / "run_config.json", run_config)
    train_data = load_split(data_dir, "train")
    valid_data = load_split(data_dir, "valid")
    train_sampler = (
        DistributedSampler(
            train_data,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed.enabled
        else None
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    # Validation is intentionally performed once on rank 0, without sampler
    # padding that could duplicate examples and bias the reported loss.
    valid_loader = (
        DataLoader(
            valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if distributed.is_main
        else None
    )
    if not len(train_loader) or not len(valid_data):
        raise ValueError("Training and validation splits must both be non-empty")

    raw_model = SmilesPredictor(model_params).to(device)
    model = (
        DistributedDataParallel(
            raw_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
        )
        if distributed.enabled
        else raw_model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    best_loss = float("inf")
    history = []
    if distributed.is_main:
        print(
            f"device={device}; gpus={distributed.world_size}; train={len(train_data):,}; "
            f"valid={len(valid_data):,}; modalities={'+'.join(args.modalities)}; "
            f"batch_per_gpu={args.batch_size}; global_batch={args.batch_size * distributed.world_size}; "
            f"initial_lr={args.learning_rate:g}; dropout={args.dropout:g}"
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        stop_reason = None
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, device, criterion)
            loss_value = loss.item()
            if not distributed.all_ranks_true(math.isfinite(loss_value)):
                stop_reason = (
                    f"non-finite training loss on at least one rank at epoch {epoch}, "
                    f"batch {batch_index}"
                )
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not distributed.all_ranks_true(torch.isfinite(grad_norm).item()):
                optimizer.zero_grad(set_to_none=True)
                stop_reason = (
                    f"non-finite gradient norm on at least one rank at epoch {epoch}, "
                    f"batch {batch_index}"
                )
                break
            optimizer.step()
            total += loss_value
        if stop_reason is not None:
            if distributed.is_main:
                write_json(
                    output_dir / "history.json",
                    {"epochs": history, "stopped_early": True, "stop_reason": stop_reason},
                )
                print(f"Stopping early: {stop_reason}")
                print("The last finite last.pt and best.pt checkpoints were kept unchanged.")
            break
        train_loss = distributed.sum(total) / (len(train_loader) * distributed.world_size)
        valid_loss = (
            validate(raw_model, valid_loader, device, criterion)
            if distributed.is_main
            else 0.0
        )
        valid_loss = distributed.broadcast_float(valid_loss)
        if not math.isfinite(train_loss) or not math.isfinite(valid_loss):
            stop_reason = (
                f"non-finite epoch loss at epoch {epoch}: "
                f"train_loss={train_loss}, valid_loss={valid_loss}"
            )
            if distributed.is_main:
                write_json(
                    output_dir / "history.json",
                    {"epochs": history, "stopped_early": True, "stop_reason": stop_reason},
                )
                print(f"Stopping early: {stop_reason}")
                print("The last finite last.pt and best.pt checkpoints were kept unchanged.")
            break
        if distributed.is_main:
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            checkpoint = {
                "model_state_dict": raw_model.state_dict(),
                "model_params": model_params,
                "training_config": run_config,
                "epoch": epoch,
                "valid_loss": valid_loss,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(checkpoint, output_dir / "best.pt")
            write_json(output_dir / "history.json", {"epochs": history})
            print(f"epoch {epoch:4d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

    if distributed.is_main:
        print(f"Best checkpoint: {output_dir / 'best.pt'}")


def main(
    default_output_dir: str | Path = RESULTS_DIRECTORY / MODEL_NAME,
    default_modalities: tuple[str, ...] = ("freq", "ir", "raman"),
) -> None:
    args = parse_args(default_output_dir, default_modalities)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be at least 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in the range [0, 1)")
    if len(args.modalities) != len(set(args.modalities)):
        raise ValueError("--modalities must not contain duplicates")
    # Frequency supplies both the vibrational positions and the shared padding
    # mask, so every supported variant requires it.
    if "freq" not in args.modalities:
        raise ValueError("--modalities must include freq")
    if args.hidden_dimension % args.attention_heads:
        raise ValueError("--hidden-dimension must be divisible by --attention-heads")
    distributed = initialize_distributed(args.device)
    try:
        train(args, distributed)
    finally:
        finalize_distributed(distributed)


if __name__ == "__main__":
    main()
