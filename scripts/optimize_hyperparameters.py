#!/usr/bin/env python
"""Tune one learning-rate/dropout pair shared by multiple model variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]

MODEL_COMMANDS = {
    "ir_raman": ("ir_raman_freq/train.py",),
    "ir": ("ir_freq/train.py",),
    "raman": ("raman_freq/train.py",),
    "fully_connected": ("-m", "fully_connected_model.train"),
}
DEFAULT_MODELS = tuple(MODEL_COMMANDS)


@dataclass(frozen=True)
class TrialParameters:
    learning_rate: float
    dropout: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_COMMANDS),
        default=list(DEFAULT_MODELS),
        help="Models sharing each trial's hyperparameters (default: all)",
    )
    parser.add_argument(
        "--search-method",
        choices=("random", "grid"),
        default="random",
        help="Hyperparameter search method (default: random)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of combinations for random search; ignored by grid search",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model and trial")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate-min", type=float, default=1e-5)
    parser.add_argument("--learning-rate-max", type=float, default=1e-3)
    parser.add_argument(
        "--learning-rate-grid-factor",
        type=float,
        default=10.0,
        help="Multiplicative LR step for grid search (default: 10)",
    )
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.3)
    parser.add_argument(
        "--dropout-grid-step",
        type=float,
        default=0.1,
        help="Additive dropout step for grid search (default: 0.1)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data_diretory/processed")
    parser.add_argument("--output-dir", default="results/hyperparameter_optimization")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and training seed")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep best.pt and last.pt for every tuning run (uses substantial disk space)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted study whose configuration is unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without training",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be at least 1")
    if args.search_method == "random" and args.trials < 1:
        raise ValueError("trials must be at least 1 for random search")
    if args.num_workers < 0:
        raise ValueError("num-workers must be at least 0")
    if not 0 < args.learning_rate_min <= args.learning_rate_max:
        raise ValueError("learning-rate bounds must satisfy 0 < min <= max")
    if args.learning_rate_grid_factor <= 1:
        raise ValueError("learning-rate-grid-factor must be greater than 1")
    if not 0 <= args.dropout_min <= args.dropout_max < 1:
        raise ValueError("dropout bounds must satisfy 0 <= min <= max < 1")
    if args.dropout_grid_step <= 0:
        raise ValueError("dropout-grid-step must be greater than 0")
    if len(args.models) != len(set(args.models)):
        raise ValueError("models must not contain duplicates")


def sample_trials(args: argparse.Namespace) -> list[TrialParameters]:
    """Use log-uniform LR and uniform dropout sampling, reproducibly."""
    rng = random.Random(args.seed)
    log_min = math.log(args.learning_rate_min)
    log_max = math.log(args.learning_rate_max)
    return [
        TrialParameters(
            learning_rate=math.exp(rng.uniform(log_min, log_max)),
            dropout=rng.uniform(args.dropout_min, args.dropout_max),
        )
        for _ in range(args.trials)
    ]


def multiplicative_grid(minimum: float, maximum: float, factor: float) -> list[float]:
    """Return a multiplicative grid with both requested bounds included."""
    values = [minimum]
    while values[-1] * factor < maximum and not math.isclose(
        values[-1] * factor, maximum
    ):
        values.append(values[-1] * factor)
    if not math.isclose(values[-1], maximum):
        values.append(maximum)
    return values


def additive_grid(minimum: float, maximum: float, step: float) -> list[float]:
    """Return an additive grid without accumulating floating-point drift."""
    values = [minimum]
    index = 1
    while minimum + index * step < maximum and not math.isclose(
        minimum + index * step, maximum
    ):
        values.append(minimum + index * step)
        index += 1
    if not math.isclose(values[-1], maximum):
        values.append(maximum)
    return values


def grid_trials(args: argparse.Namespace) -> list[TrialParameters]:
    learning_rates = multiplicative_grid(
        args.learning_rate_min,
        args.learning_rate_max,
        args.learning_rate_grid_factor,
    )
    dropouts = additive_grid(
        args.dropout_min,
        args.dropout_max,
        args.dropout_grid_step,
    )
    return [
        TrialParameters(learning_rate=learning_rate, dropout=dropout)
        for learning_rate in learning_rates
        for dropout in dropouts
    ]


def create_trials(args: argparse.Namespace) -> list[TrialParameters]:
    return grid_trials(args) if args.search_method == "grid" else sample_trials(args)


def model_command(
    model_name: str,
    params: TrialParameters,
    trial_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [sys.executable, *MODEL_COMMANDS[model_name]]
    command.extend(
        [
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(trial_dir / model_name),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            format(params.learning_rate, ".12g"),
            "--dropout",
            format(params.dropout, ".12g"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
        ]
    )
    return command


def best_validation_loss(history_path: Path) -> float:
    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    epochs = history.get("epochs")
    if history.get("stopped_early"):
        reason = history.get("stop_reason", "reason was not recorded")
        raise ValueError(f"Training stopped early: {reason}")
    if not isinstance(epochs, list):
        raise ValueError(f"Invalid or missing epochs list in {history_path}")
    if not epochs:
        raise ValueError(f"No epochs recorded in {history_path}")
    losses = [float(epoch["valid_loss"]) for epoch in epochs]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError(f"Non-finite validation loss in {history_path}")
    return min(losses)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def study_config(args: argparse.Namespace) -> dict:
    keys = (
        "models",
        "search_method",
        "epochs",
        "batch_size",
        "learning_rate_min",
        "learning_rate_max",
        "dropout_min",
        "dropout_max",
        "device",
        "data_dir",
        "seed",
        "num_workers",
    )
    config = {key: getattr(args, key) for key in keys}
    if args.search_method == "grid":
        config.update(
            learning_rate_grid_factor=args.learning_rate_grid_factor,
            dropout_grid_step=args.dropout_grid_step,
        )
    else:
        config["trials"] = args.trials
    return config


def load_or_initialize_study(output_dir: Path, args: argparse.Namespace) -> list[dict]:
    config_path = output_dir / "study_config.json"
    trials_path = output_dir / "trials.json"
    config = study_config(args)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"{output_dir} is not empty; choose another --output-dir or use --resume"
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume: missing {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            previous_config = json.load(handle)
        if previous_config != config:
            raise ValueError("Cannot resume because the study configuration has changed")
        if trials_path.is_file():
            with trials_path.open(encoding="utf-8") as handle:
                return json.load(handle)["trials"]
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    return []


def write_summaries(output_dir: Path, trials: list[dict]) -> None:
    write_json(output_dir / "trials.json", {"trials": trials})
    csv_path = output_dir / "trials.csv"
    model_names = sorted(
        {model for trial in trials for model in trial.get("model_valid_losses", {})}
    )
    fieldnames = [
        "trial",
        "status",
        "learning_rate",
        "dropout",
        "mean_valid_loss",
        "failed_model",
        "failure_reason",
        "returncode",
    ]
    fieldnames.extend(f"{model}_valid_loss" for model in model_names)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {key: trial.get(key) for key in fieldnames}
            for model, loss in trial.get("model_valid_losses", {}).items():
                row[f"{model}_valid_loss"] = loss
            writer.writerow(row)
    temporary.replace(csv_path)
    completed = [trial for trial in trials if trial["status"] == "completed"]
    if completed:
        best = min(completed, key=lambda trial: trial["mean_valid_loss"])
        write_json(
            output_dir / "best_hyperparameters.json",
            {
                "learning_rate": best["learning_rate"],
                "dropout": best["dropout"],
                "mean_valid_loss": best["mean_valid_loss"],
                "model_valid_losses": best["model_valid_losses"],
                "trial": best["trial"],
            },
        )


def remove_checkpoints(model_dir: Path) -> None:
    for name in ("best.pt", "last.pt"):
        path = model_dir / name
        if path.exists():
            path.unlink()


def run_trial(
    trial_number: int,
    total_trials: int,
    params: TrialParameters,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    trial_dir = output_dir / f"trial_{trial_number:04d}"
    losses: dict[str, float] = {}
    result = {
        "trial": trial_number,
        **asdict(params),
        "status": "running",
        "mean_valid_loss": None,
        "model_valid_losses": losses,
    }
    print(
        f"\ntrial {trial_number + 1}/{total_trials}: "
        f"learning_rate={params.learning_rate:.6g}, dropout={params.dropout:.6g}",
        flush=True,
    )
    for model_name in args.models:
        model_dir = trial_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        command = model_command(model_name, params, trial_dir, args)
        log_path = model_dir / "train.log"
        print(f"  training {model_name} (log: {log_path})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode:
            result["status"] = "failed"
            result["failed_model"] = model_name
            result["returncode"] = completed.returncode
            result["failure_reason"] = (
                f"Training process exited with return code {completed.returncode}"
            )
            print(f"  FAILED: {model_name}; inspect {log_path}", flush=True)
            return result
        try:
            loss = best_validation_loss(model_dir / "history.json")
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["status"] = "failed"
            result["failed_model"] = model_name
            result["failure_reason"] = str(error)
            print(
                f"  FAILED: {model_name}; {error}; inspect {log_path}",
                flush=True,
            )
            if not args.keep_checkpoints:
                remove_checkpoints(model_dir)
            return result
        losses[model_name] = loss
        print(f"  {model_name}: best_valid_loss={loss:.6f}", flush=True)
        if not args.keep_checkpoints:
            remove_checkpoints(model_dir)
    result["status"] = "completed"
    result["mean_valid_loss"] = sum(losses.values()) / len(losses)
    print(f"  mean_valid_loss={result['mean_valid_loss']:.6f}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    parameters = create_trials(args)
    print(
        f"search_method={args.search_method}; combinations={len(parameters)}; "
        f"models={len(args.models)}; total_training_runs={len(parameters) * len(args.models)}"
    )
    if args.dry_run:
        for number, params in enumerate(parameters):
            trial_dir = output_dir / f"trial_{number:04d}"
            for model_name in args.models:
                print(" ".join(model_command(model_name, params, trial_dir, args)))
        return

    trials = load_or_initialize_study(output_dir, args)
    finished_numbers = {trial["trial"] for trial in trials}
    for number, params in enumerate(parameters):
        if number in finished_numbers:
            print(f"skipping recorded trial {number + 1}/{len(parameters)}")
            continue
        trials.append(run_trial(number, len(parameters), params, output_dir, args))
        write_summaries(output_dir, trials)

    completed = [trial for trial in trials if trial["status"] == "completed"]
    if not completed:
        raise RuntimeError(f"No trial completed successfully; inspect logs under {output_dir}")
    best = min(completed, key=lambda trial: trial["mean_valid_loss"])
    print(
        f"\nBest: learning_rate={best['learning_rate']:.6g}, "
        f"dropout={best['dropout']:.6g}, mean_valid_loss={best['mean_valid_loss']:.6f}"
    )
    print(f"Summary: {output_dir / 'best_hyperparameters.json'}")


if __name__ == "__main__":
    main()
