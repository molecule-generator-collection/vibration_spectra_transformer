import json
import subprocess
from pathlib import Path

import pytest

from scripts.optimize_hyperparameters import (
    TrialParameters,
    additive_grid,
    best_validation_loss,
    create_trials,
    grid_trials,
    model_command,
    multiplicative_grid,
    parse_args,
    run_trial,
    sample_trials,
    validate_args,
    write_summaries,
)


def test_sampling_is_reproducible_and_within_bounds():
    args = parse_args(
        [
            "--trials",
            "5",
            "--learning-rate-min",
            "1e-5",
            "--learning-rate-max",
            "1e-3",
            "--dropout-min",
            "0.1",
            "--dropout-max",
            "0.4",
            "--seed",
            "7",
        ]
    )

    first = sample_trials(args)
    second = sample_trials(args)

    assert first == second
    assert all(1e-5 <= trial.learning_rate <= 1e-3 for trial in first)
    assert all(0.1 <= trial.dropout <= 0.4 for trial in first)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--trials", "0"],
        ["--learning-rate-min", "0"],
        ["--learning-rate-min", "1e-2", "--learning-rate-max", "1e-3"],
        ["--dropout-min", "-0.1"],
        ["--dropout-max", "1.0"],
        ["--learning-rate-grid-factor", "1.0"],
        ["--dropout-grid-step", "0"],
        ["--models", "ir", "ir"],
    ],
)
def test_invalid_search_arguments_are_rejected(arguments):
    with pytest.raises(ValueError):
        validate_args(parse_args(arguments))


def test_model_command_shares_the_same_hyperparameters(tmp_path):
    args = parse_args(["--epochs", "3", "--batch-size", "8", "--device", "cpu"])
    params = TrialParameters(learning_rate=2e-4, dropout=0.25)

    transformer = model_command("ir", params, tmp_path, args)
    fully_connected = model_command("fully_connected", params, tmp_path, args)

    for command in (transformer, fully_connected):
        assert command[command.index("--learning-rate") + 1] == "0.0002"
        assert command[command.index("--dropout") + 1] == "0.25"
        assert command[command.index("--epochs") + 1] == "3"
    assert "ir_freq/train.py" in transformer
    assert fully_connected[1:4] == ["-m", "fully_connected_model.train", "--data-dir"]


def test_grid_search_covers_cartesian_product_and_includes_bounds():
    args = parse_args(
        [
            "--search-method",
            "grid",
            "--learning-rate-min",
            "1e-5",
            "--learning-rate-max",
            "1e-3",
            "--learning-rate-grid-factor",
            "10",
            "--dropout-min",
            "0",
            "--dropout-max",
            "0.3",
            "--dropout-grid-step",
            "0.1",
        ]
    )

    trials = grid_trials(args)

    assert len(trials) == 12
    assert {trial.learning_rate for trial in trials} == {1e-5, 1e-4, 1e-3}
    assert {round(trial.dropout, 10) for trial in trials} == {0.0, 0.1, 0.2, 0.3}
    assert create_trials(args) == trials


def test_grids_include_maximum_when_range_is_not_evenly_divisible():
    assert multiplicative_grid(1e-5, 5e-4, 10) == [1e-5, 1e-4, 5e-4]
    assert additive_grid(0.05, 0.3, 0.1) == pytest.approx([0.05, 0.15, 0.25, 0.3])


def test_best_validation_loss_uses_best_epoch(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "epochs": [
                    {"epoch": 1, "valid_loss": 2.0},
                    {"epoch": 2, "valid_loss": 1.25},
                    {"epoch": 3, "valid_loss": 1.5},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert best_validation_loss(history) == 1.25


def test_best_validation_loss_rejects_early_stopped_training(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "epochs": [],
                "stopped_early": True,
                "stop_reason": "non-finite gradient norm at epoch 1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite gradient norm"):
        best_validation_loss(history)


def test_run_trial_records_empty_history_as_failure_and_continues(tmp_path, monkeypatch):
    args = parse_args(["--models", "ir", "--epochs", "1", "--device", "cpu"])

    def fake_training(command, **_kwargs):
        model_dir = Path(command[command.index("--output-dir") + 1])
        (model_dir / "history.json").write_text(
            json.dumps(
                {
                    "epochs": [],
                    "stopped_early": True,
                    "stop_reason": "non-finite loss at epoch 1",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_training)

    result = run_trial(
        0,
        1,
        TrialParameters(learning_rate=1e-3, dropout=0.2),
        tmp_path,
        args,
    )

    assert result["status"] == "failed"
    assert result["failed_model"] == "ir"
    assert "non-finite loss" in result["failure_reason"]


def test_summary_selects_smallest_mean_loss(tmp_path):
    trials = [
        {
            "trial": 0,
            "status": "completed",
            "learning_rate": 1e-4,
            "dropout": 0.1,
            "mean_valid_loss": 1.5,
            "model_valid_losses": {"ir": 1.4, "raman": 1.6},
        },
        {
            "trial": 1,
            "status": "completed",
            "learning_rate": 2e-4,
            "dropout": 0.2,
            "mean_valid_loss": 1.2,
            "model_valid_losses": {"ir": 1.1, "raman": 1.3},
        },
    ]

    write_summaries(Path(tmp_path), trials)

    best = json.loads((tmp_path / "best_hyperparameters.json").read_text())
    assert best["trial"] == 1
    assert best["learning_rate"] == 2e-4
    assert (tmp_path / "trials.csv").is_file()
