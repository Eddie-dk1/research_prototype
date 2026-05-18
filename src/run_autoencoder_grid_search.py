from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils import RESULTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_lstm_autoencoder.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "src" / "evaluate.py"

SEQUENCE_LENGTHS = [5, 6, 8]
THRESHOLD_PERCENTILES = [85, 90, 92, 95]
HIDDEN_SIZES = [16, 32]
EPOCHS = [10, 20, 30]
BATCH_SIZES = [64, 128]
LEARNING_RATES = [1e-3, 5e-4]
EARLY_STOPPING_PATIENCE = 5
SELECTION_METRIC = "f1"
EXPERIMENT_TAG = "stage8_full_grid"


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        command_text = " ".join(command)
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {command_text}")


def load_latest_experiment(experiment_tag: str) -> pd.Series:
    experiments_path = RESULTS_DIR / "autoencoder_experiments.csv"
    experiments = pd.read_csv(experiments_path)
    tagged = experiments[experiments["experiment_tag"] == experiment_tag].copy()
    if tagged.empty:
        raise RuntimeError(f"No experiments found for tag: {experiment_tag}")
    tagged = tagged.sort_values("run_label")
    return tagged.iloc[-1]


def build_configs() -> list[dict[str, float | int]]:
    configs: list[dict[str, float | int]] = []
    for sequence_length, hidden_size, epochs, batch_size, learning_rate in itertools.product(
        SEQUENCE_LENGTHS,
        HIDDEN_SIZES,
        EPOCHS,
        BATCH_SIZES,
        LEARNING_RATES,
    ):
        configs.append(
            {
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
    return configs


def main() -> None:
    configs = build_configs()
    rows: list[dict[str, float | int | str]] = []

    for index, config in enumerate(configs, start=1):
        config_tag = f"{EXPERIMENT_TAG}_{index:02d}"
        artifact_suffix = f"exp{index:02d}"
        print(
            f"[{index}/{len(configs)}] "
            f"seq={config['sequence_length']} "
            f"hidden={config['hidden_size']} "
            f"epochs={config['epochs']} "
            f"batch={config['batch_size']} "
            f"lr={config['learning_rate']}"
        )
        run_command(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--sequence-length",
                str(config["sequence_length"]),
                "--hidden-size",
                str(config["hidden_size"]),
                "--epochs",
                str(config["epochs"]),
                "--batch-size",
                str(config["batch_size"]),
                "--learning-rate",
                str(config["learning_rate"]),
                "--threshold-percentiles",
                *(str(value) for value in THRESHOLD_PERCENTILES),
                "--early-stopping-patience",
                str(EARLY_STOPPING_PATIENCE),
                "--selection-metric",
                SELECTION_METRIC,
                "--experiment-tag",
                config_tag,
                "--artifact-suffix",
                artifact_suffix,
            ]
        )
        latest = load_latest_experiment(config_tag)
        row = config | {
            "experiment_tag": config_tag,
            "epochs_completed": int(latest["epochs_completed"]),
            "best_epoch": int(latest["best_epoch"]),
            "selected_threshold_percentile": int(latest["selected_threshold_percentile"]),
            "sequence_anomaly_ratio": float(latest["sequence_anomaly_ratio"]),
            "precision": float(latest["precision"]),
            "recall": float(latest["recall"]),
            "f1": float(latest["f1"]),
            "roc_auc": float(latest["roc_auc"]),
            "validation_precision": float(latest["validation_precision"]),
            "validation_recall": float(latest["validation_recall"]),
            "validation_f1": float(latest["validation_f1"]),
            "validation_roc_auc": float(latest["validation_roc_auc"]),
            "stopped_early": int(latest["stopped_early"]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["f1", "roc_auc", "recall", "validation_f1", "validation_recall"],
        ascending=[False, False, False, False, False],
    )
    summary_path = RESULTS_DIR / "autoencoder_stage8_grid_results.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.iloc[0]
    print("\nBest configuration:")
    print(best.to_string())

    run_command(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--sequence-length",
            str(int(best["sequence_length"])),
            "--hidden-size",
            str(int(best["hidden_size"])),
            "--epochs",
            str(int(best["epochs"])),
            "--batch-size",
            str(int(best["batch_size"])),
            "--learning-rate",
            str(best["learning_rate"]),
            "--threshold-percentiles",
            *(str(value) for value in THRESHOLD_PERCENTILES),
            "--early-stopping-patience",
            str(EARLY_STOPPING_PATIENCE),
            "--selection-metric",
            SELECTION_METRIC,
            "--experiment-tag",
            f"{EXPERIMENT_TAG}_best",
            "--artifact-suffix",
            f"exp{len(configs) + 1:02d}",
        ]
    )
    run_command([sys.executable, str(EVALUATE_SCRIPT)])

    print(f"\nSaved stage 8 grid results to: {summary_path}")


if __name__ == "__main__":
    main()
