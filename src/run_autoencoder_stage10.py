from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from features import LSTM_FEATURE_COLUMNS
from utils import RESULTS_DIR
from utils import load_json
from utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_lstm_autoencoder.py"
REFERENCE_AUTOENCODER_PATH = RESULTS_DIR / "baseline_reference" / "metrics_autoencoder.json"
THRESHOLD_PERCENTILES = [85, 90, 92, 95]
SELECTION_METRIC = "f1"
EXPERIMENT_TAG_PREFIX = "stage10_practical"
SUMMARY_PATH = RESULTS_DIR / "autoencoder_stage10_results.csv"
REPORT_PATH = RESULTS_DIR / "autoencoder_stage10_summary.json"
BASE_BATCH_SIZE = 128
BASE_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
FEATURE_SET_NAME = "lstm_without_user_and_ip_identifiers"
REMOVED_FEATURE_COLUMNS = ["user_id_encoded", "source_ip_encoded", "destination_ip_encoded"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-6-epochs",
        type=int,
        default=30,
        help="Epoch count for experiment 6. Use 30 by default, or 50 for a longer run.",
    )
    return parser.parse_args()


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


def load_reference_metrics() -> dict[str, Any] | None:
    if not REFERENCE_AUTOENCODER_PATH.exists():
        return None
    return load_json(REFERENCE_AUTOENCODER_PATH).get("lstm_autoencoder")


def load_experiment_metadata(artifact_suffix: str) -> dict[str, Any]:
    metadata_path = RESULTS_DIR / f"autoencoder_metadata_{artifact_suffix}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    return load_json(metadata_path)


def validate_feature_set() -> list[str]:
    feature_columns = list(LSTM_FEATURE_COLUMNS)
    forbidden = sorted(set(feature_columns) & set(REMOVED_FEATURE_COLUMNS))
    if forbidden:
        forbidden_text = ", ".join(forbidden)
        raise ValueError(
            "Stage 10 expects user_id/source_ip/destination_ip to be removed from LSTM features, "
            f"but found: {forbidden_text}"
        )
    return feature_columns


def result_level(record: dict[str, Any]) -> str:
    recall = float(record["recall"])
    f1 = float(record["f1"])
    roc_auc = float(record["roc_auc"])

    if recall > 0.40 and f1 > 0.40 and roc_auc > 0.75:
        return "very_good"
    if recall >= 0.25 and f1 >= 0.25 and roc_auc > 0.65:
        return "real_improvement"
    return "limited"


def build_selection_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    level_rank = {
        "limited": 0.0,
        "real_improvement": 1.0,
        "very_good": 2.0,
    }[result_level(record)]
    return (
        level_rank,
        float(record["f1"]),
        float(record["roc_auc"]),
        float(record["recall"]),
        float(record["validation_f1"]),
    )


def choose_best(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=build_selection_key)


def run_experiment(
    *,
    experiment_number: int,
    goal: str,
    config: dict[str, Any],
    feature_columns: list[str],
    reference_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_tag = f"{EXPERIMENT_TAG_PREFIX}_exp{experiment_number:02d}"
    artifact_suffix = f"exp{experiment_number:02d}"

    print(
        f"\nExperiment {experiment_number}: "
        f"seq={config['sequence_length']} "
        f"hidden={config['hidden_size']} "
        f"epochs={config['epochs']} "
        f"batch={config['batch_size']} "
        f"lr={config['learning_rate']} "
        f"early_stop={config['early_stopping_patience']}"
    )
    print(f"Goal: {goal}")

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
            *(str(value) for value in config["threshold_percentiles"]),
            "--early-stopping-patience",
            str(config["early_stopping_patience"]),
            "--selection-metric",
            config["selection_metric"],
            "--experiment-tag",
            experiment_tag,
            "--artifact-suffix",
            artifact_suffix,
        ]
    )

    latest = load_latest_experiment(experiment_tag)
    metadata = load_experiment_metadata(artifact_suffix)
    split_sizes = metadata["split_sizes"]
    record = {
        "experiment_number": experiment_number,
        "experiment_tag": experiment_tag,
        "artifact_suffix": artifact_suffix,
        "goal": goal,
        "feature_set_name": FEATURE_SET_NAME,
        "feature_columns": "|".join(feature_columns),
        "removed_lstm_features": "|".join(REMOVED_FEATURE_COLUMNS),
        "sequence_length": int(config["sequence_length"]),
        "hidden_size": int(config["hidden_size"]),
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "early_stopping_enabled": int(config["early_stopping_patience"] > 0),
        "threshold_selection_method": "validation_percentile_search",
        "threshold_percentiles_checked": "|".join(str(value) for value in config["threshold_percentiles"]),
        "threshold_percentile": int(latest["selected_threshold_percentile"]),
        "threshold": float(latest["threshold"]),
        "precision": float(latest["precision"]),
        "recall": float(latest["recall"]),
        "f1": float(latest["f1"]),
        "roc_auc": float(latest["roc_auc"]),
        "validation_precision": float(latest["validation_precision"]),
        "validation_recall": float(latest["validation_recall"]),
        "validation_f1": float(latest["validation_f1"]),
        "validation_roc_auc": float(latest["validation_roc_auc"]),
        "train_size": int(split_sizes["train"]),
        "validation_size": int(split_sizes["validation"]),
        "test_size": int(split_sizes["test"]),
        "train_pool_size": int(latest["train_size"]),
        "epochs_completed": int(latest["epochs_completed"]),
        "best_epoch": int(latest["best_epoch"]),
        "stopped_early": int(latest["stopped_early"]),
        "result_level": result_level(latest.to_dict()),
    }

    if reference_metrics is not None:
        record["delta_recall_vs_reference"] = record["recall"] - float(reference_metrics["recall"])
        record["delta_f1_vs_reference"] = record["f1"] - float(reference_metrics["f1"])
        record["delta_roc_auc_vs_reference"] = record["roc_auc"] - float(reference_metrics["roc_auc"])

    return record


def main() -> None:
    args = parse_args()
    if args.experiment_6_epochs <= 0:
        raise ValueError("Experiment 6 epochs must be a positive integer.")

    feature_columns = validate_feature_set()
    reference_metrics = load_reference_metrics()
    records: list[dict[str, Any]] = []

    base_config = {
        "batch_size": BASE_BATCH_SIZE,
        "learning_rate": BASE_LEARNING_RATE,
        "threshold_percentiles": THRESHOLD_PERCENTILES,
        "selection_metric": SELECTION_METRIC,
    }

    experiment_1 = run_experiment(
        experiment_number=1,
        goal="Check whether dropping user_id/source_ip/destination_ip from LSTM features helps by itself.",
        config=base_config
        | {
            "sequence_length": 10,
            "hidden_size": 16,
            "epochs": 10,
            "early_stopping_patience": 0,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_1)

    experiment_2 = run_experiment(
        experiment_number=2,
        goal="Check whether a shorter sequence window of 5 helps.",
        config=base_config
        | {
            "sequence_length": 5,
            "hidden_size": 16,
            "epochs": 10,
            "early_stopping_patience": 0,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_2)

    experiment_3 = run_experiment(
        experiment_number=3,
        goal="Check whether sequence length 6 is a better compromise than 5.",
        config=base_config
        | {
            "sequence_length": 6,
            "hidden_size": 16,
            "epochs": 10,
            "early_stopping_patience": 0,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_3)

    best_window_record = choose_best(records[:3])
    experiment_4 = run_experiment(
        experiment_number=4,
        goal="Check whether increasing hidden_size to 32 improves capacity.",
        config=base_config
        | {
            "sequence_length": int(best_window_record["sequence_length"]),
            "hidden_size": 32,
            "epochs": 10,
            "early_stopping_patience": 0,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_4)

    best_through_experiment_4 = choose_best(records[:4])
    experiment_5 = run_experiment(
        experiment_number=5,
        goal="Check whether 20 epochs with early stopping improve the best setup so far.",
        config=base_config
        | {
            "sequence_length": int(best_through_experiment_4["sequence_length"]),
            "hidden_size": int(best_through_experiment_4["hidden_size"]),
            "epochs": 20,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_5)

    best_through_experiment_5 = choose_best(records[:5])
    experiment_6 = run_experiment(
        experiment_number=6,
        goal="Check whether longer training at lower learning rate still helps.",
        config=base_config
        | {
            "sequence_length": int(best_through_experiment_5["sequence_length"]),
            "hidden_size": int(best_through_experiment_5["hidden_size"]),
            "epochs": int(args.experiment_6_epochs),
            "learning_rate": 5e-4,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        feature_columns=feature_columns,
        reference_metrics=reference_metrics,
    )
    records.append(experiment_6)

    overall_best = choose_best(records)
    summary = pd.DataFrame(records).sort_values("experiment_number")
    summary.to_csv(SUMMARY_PATH, index=False)

    report_payload = {
        "stage": "stage10_practical_experiments",
        "feature_set_name": FEATURE_SET_NAME,
        "feature_columns": feature_columns,
        "removed_lstm_features": REMOVED_FEATURE_COLUMNS,
        "threshold_percentiles_checked": THRESHOLD_PERCENTILES,
        "selection_metric": SELECTION_METRIC,
        "experiment_6_epochs": int(args.experiment_6_epochs),
        "acceptance_criteria": {
            "real_improvement": {
                "recall_min": 0.25,
                "f1_min": 0.25,
                "roc_auc_min_exclusive": 0.65,
            },
            "very_good_result": {
                "recall_min_exclusive": 0.40,
                "f1_min_exclusive": 0.40,
                "roc_auc_min_exclusive": 0.75,
            },
        },
        "reference_metrics": reference_metrics,
        "best_experiment": overall_best,
        "recommended_final_config": {
            "sequence_length": int(overall_best["sequence_length"]),
            "hidden_size": int(overall_best["hidden_size"]),
            "epochs": int(overall_best["epochs"]),
            "batch_size": int(overall_best["batch_size"]),
            "learning_rate": float(overall_best["learning_rate"]),
            "early_stopping_patience": int(overall_best["early_stopping_patience"]),
            "threshold_percentiles": THRESHOLD_PERCENTILES,
            "selection_metric": SELECTION_METRIC,
        },
        "experiments": records,
    }
    save_json(report_payload, REPORT_PATH)

    print("\nStage 10 summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved stage 10 results to: {SUMMARY_PATH}")
    print(f"Saved stage 10 summary to: {REPORT_PATH}")
    print(
        "Best experiment: "
        f"exp{int(overall_best['experiment_number']):02d} "
        f"(f1={overall_best['f1']:.4f}, "
        f"recall={overall_best['recall']:.4f}, "
        f"roc_auc={overall_best['roc_auc']:.4f}, "
        f"level={overall_best['result_level']})"
    )


if __name__ == "__main__":
    main()
