from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is not installed. Run `pip install -r requirements.txt` before training the autoencoder."
    ) from error

from features import LSTM_FEATURE_COLUMNS
from features import build_sequences
from utils import PLOTS_DIR
from utils import PROCESSED_DIR
from utils import RESULTS_DIR
from utils import compute_classification_metrics
from utils import ensure_directories
from utils import load_json
from utils import save_json
from utils import set_seed


RANDOM_STATE = 42
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.2
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_THRESHOLD_PERCENTILES = list(range(80, 100))
DEFAULT_EARLY_STOPPING_PATIENCE = 5
BASELINE_REFERENCE_DIR = RESULTS_DIR / "baseline_reference"


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.encoder(inputs)
        repeated_hidden = hidden_state[-1].unsqueeze(1).repeat(1, inputs.size(1), 1)
        decoded_sequence, _ = self.decoder(repeated_hidden)
        return self.output_layer(decoded_sequence)


def get_device() -> torch.device:
    return torch.device("cpu")


def score_reconstruction_errors(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(sequences).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    errors: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            batch_errors = ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(batch_errors)

    if not errors:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(errors).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override the sequence length used for LSTM-specific sequence construction.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Hidden size for the LSTM autoencoder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for training and inference.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--threshold-percentiles",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLD_PERCENTILES,
        help="Candidate percentiles for validation threshold selection.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Number of epochs without validation improvement before stopping early.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "roc_auc"),
        default="f1",
        help="Validation metric used to keep the best epoch.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="",
        help="Optional label stored with experiment artifacts for grouped comparisons.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix for per-experiment artifacts, for example `exp01`.",
    )
    return parser.parse_args()


def build_metrics_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict:
    metrics = compute_classification_metrics(y_true, y_pred, scores)
    confusion = confusion_matrix(y_true, y_pred)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "confusion_matrix": confusion.tolist(),
    }


def select_best_threshold(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    candidate_percentiles: list[int],
) -> tuple[float, int, dict[str, float | list[list[int]]], list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []

    for percentile in candidate_percentiles:
        threshold = float(np.percentile(validation_scores, percentile))
        predictions = (validation_scores >= threshold).astype(np.int64)
        metrics = build_metrics_block(validation_labels, predictions, validation_scores)
        candidates.append(
            {
                "percentile": float(percentile),
                "threshold": threshold,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else np.nan,
                "pr_auc": float(metrics["pr_auc"]) if metrics["pr_auc"] is not None else np.nan,
            }
        )

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            candidate["f1"],
            candidate["recall"],
            -candidate["percentile"],
        ),
    )
    best_percentile = int(best_candidate["percentile"])
    best_threshold = float(best_candidate["threshold"])
    best_predictions = (validation_scores >= best_threshold).astype(np.int64)
    best_metrics = build_metrics_block(validation_labels, best_predictions, validation_scores)
    return best_threshold, best_percentile, best_metrics, candidates


def build_epoch_selection_key(
    metrics: dict[str, Any],
    threshold_percentile: int,
    selection_metric: str,
) -> tuple[float, float, float, float]:
    roc_auc = float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else float("-inf")
    if selection_metric == "roc_auc":
        return (
            roc_auc,
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(-threshold_percentile),
        )

    return (
        float(metrics["f1"]),
        float(metrics["recall"]),
        roc_auc,
        float(-threshold_percentile),
    )


def load_feature_config() -> tuple[int | None, list[str], list[str]]:
    feature_config_path = PROCESSED_DIR / "feature_config.json"
    feature_columns_path = PROCESSED_DIR / "feature_columns.json"

    sequence_length: int | None = None
    feature_columns: list[str] = []
    lstm_feature_columns = list(LSTM_FEATURE_COLUMNS)

    if feature_config_path.exists():
        config = load_json(feature_config_path)
        sequence_length = int(config.get("sequence_length")) if config.get("sequence_length") is not None else None
        configured_lstm_columns = config.get("feature_columns_lstm")
        if configured_lstm_columns:
            lstm_feature_columns = list(configured_lstm_columns)

    if feature_columns_path.exists():
        feature_columns = list(load_json(feature_columns_path))

    return sequence_length, feature_columns, lstm_feature_columns


def concatenate_sequence_batches(*batches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid_batches = [batch for batch in batches if len(batch["X"]) > 0]
    if not valid_batches:
        return {
            "X": np.empty((0, 0, 0), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "user_ids": np.empty((0,), dtype="<U1"),
            "timestamps": np.empty((0,), dtype="<U1"),
        }

    return {
        "X": np.concatenate([batch["X"] for batch in valid_batches], axis=0),
        "y": np.concatenate([batch["y"] for batch in valid_batches], axis=0),
        "user_ids": np.concatenate([batch["user_ids"] for batch in valid_batches], axis=0),
        "timestamps": np.concatenate([batch["timestamps"] for batch in valid_batches], axis=0),
    }


def load_lstm_sequences(
    sequence_length: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    train_events_path = PROCESSED_DIR / "events_train.csv"
    test_events_path = PROCESSED_DIR / "events_test.csv"
    if not train_events_path.exists() or not test_events_path.exists():
        raise FileNotFoundError(
            "Processed event files not found. Run `python3 src/preprocessing.py` first."
        )

    train_events = pd.read_csv(train_events_path)
    test_events = pd.read_csv(test_events_path)

    missing_columns = sorted(set(feature_columns) - set(train_events.columns))
    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(f"LSTM feature columns are missing from processed events: {missing_columns_text}")

    train_sequences = build_sequences(
        train_events,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
    )
    test_sequences = build_sequences(
        test_events,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
    )
    all_sequences = concatenate_sequence_batches(train_sequences, test_sequences)
    if len(all_sequences["X"]) == 0:
        raise ValueError("No sequences were generated for the requested LSTM configuration.")

    stats = {
        "train_sequence_count": int(len(train_sequences["X"])),
        "test_sequence_count": int(len(test_sequences["X"])),
        "sequence_count": int(len(all_sequences["X"])),
        "sequence_anomaly_count": int(all_sequences["y"].sum()),
        "sequence_anomaly_ratio": float(all_sequences["y"].mean()),
    }
    return all_sequences["X"].astype(np.float32), all_sequences["y"].astype(np.int64), stats


def load_reference_metrics() -> dict | None:
    reference_metrics_path = BASELINE_REFERENCE_DIR / "metrics_autoencoder.json"
    if not reference_metrics_path.exists():
        return None
    return load_json(reference_metrics_path).get("lstm_autoencoder")


def append_experiment_record(record: dict[str, object]) -> None:
    experiments_path = RESULTS_DIR / "autoencoder_experiments.csv"
    new_row = pd.DataFrame([record])

    if experiments_path.exists():
        existing = pd.read_csv(experiments_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(experiments_path, index=False)


def normalize_artifact_suffix(raw_suffix: str) -> str:
    artifact_suffix = raw_suffix.strip()
    if not artifact_suffix:
        return ""
    if "/" in artifact_suffix or "\\" in artifact_suffix:
        raise ValueError("Artifact suffix must not contain path separators.")
    return artifact_suffix


def with_artifact_suffix(path: Path, artifact_suffix: str) -> Path | None:
    if not artifact_suffix:
        return None
    return path.with_name(f"{path.stem}_{artifact_suffix}{path.suffix}")


def update_feature_config_lstm_columns(lstm_feature_columns: list[str]) -> None:
    feature_config_path = PROCESSED_DIR / "feature_config.json"
    if not feature_config_path.exists():
        return

    config = load_json(feature_config_path)
    if config.get("feature_columns_lstm") == lstm_feature_columns:
        return

    config["feature_columns_lstm"] = lstm_feature_columns
    save_json(config, feature_config_path)


def plot_reconstruction_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(scores[y_true == 0], bins=30, alpha=0.7, label="Normal", density=True)
    axis.hist(scores[y_true == 1], bins=30, alpha=0.7, label="Anomaly", density=True)
    axis.axvline(threshold, color="red", linestyle="--", label="Threshold")
    axis.set_title("Reconstruction Error Distribution")
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=["Normal", "Anomaly"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title("Confusion Matrix: LSTM-Autoencoder")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        scores,
        name="LSTM-Autoencoder",
        ax=axis,
    )
    axis.set_title("ROC Curve: LSTM-Autoencoder")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ensure_directories()
    set_seed(RANDOM_STATE)
    artifact_suffix = normalize_artifact_suffix(args.artifact_suffix)

    default_sequence_length, feature_columns, lstm_feature_columns = load_feature_config()
    selected_sequence_length = args.sequence_length or default_sequence_length
    if selected_sequence_length is None:
        raise ValueError("Sequence length is not configured. Run `python3 src/preprocessing.py` first.")
    if selected_sequence_length <= 0:
        raise ValueError("Sequence length must be a positive integer.")
    if args.hidden_size <= 0:
        raise ValueError("Hidden size must be a positive integer.")
    if args.epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.early_stopping_patience < 0:
        raise ValueError("Early stopping patience must be zero or a positive integer.")
    threshold_percentiles = sorted(set(args.threshold_percentiles))
    if not threshold_percentiles:
        raise ValueError("At least one threshold percentile must be provided.")
    early_stopping_enabled = args.early_stopping_patience > 0

    update_feature_config_lstm_columns(lstm_feature_columns)
    sequences, sequence_labels, sequence_stats = load_lstm_sequences(
        sequence_length=selected_sequence_length,
        feature_columns=lstm_feature_columns,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        sequence_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=sequence_labels,
    )

    X_train_inner, X_validation, y_train_inner, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_train_normal = X_train_inner[y_train_inner == 0]
    if len(X_train_normal) == 0:
        raise ValueError("No normal train sequences available for autoencoder training.")

    device = get_device()
    model = LSTMAutoencoder(input_size=sequences.shape[2], hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal).float()),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_threshold = 0.0
    best_threshold_percentile = 0
    best_validation_metrics: dict[str, Any] | None = None
    best_epoch_key: tuple[float, float, float, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        validation_errors = score_reconstruction_errors(
            model,
            X_validation,
            device=device,
            batch_size=args.batch_size,
        )
        epoch_threshold, epoch_percentile, epoch_validation_metrics, epoch_thresholds = select_best_threshold(
            validation_errors,
            y_validation,
            candidate_percentiles=threshold_percentiles,
        )
        epoch_key = build_epoch_selection_key(
            epoch_validation_metrics,
            epoch_percentile,
            selection_metric=args.selection_metric,
        )
        is_improved = best_epoch_key is None or epoch_key > best_epoch_key

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "validation_precision": float(epoch_validation_metrics["precision"]),
                "validation_recall": float(epoch_validation_metrics["recall"]),
                "validation_f1": float(epoch_validation_metrics["f1"]),
                "validation_roc_auc": float(epoch_validation_metrics["roc_auc"]),
                "validation_pr_auc": float(epoch_validation_metrics["pr_auc"]),
                "validation_threshold": float(epoch_threshold),
                "validation_threshold_percentile": int(epoch_percentile),
                "improved": int(is_improved),
            }
        )

        if is_improved:
            best_epoch = epoch
            best_threshold = float(epoch_threshold)
            best_threshold_percentile = int(epoch_percentile)
            best_validation_metrics = epoch_validation_metrics
            best_epoch_key = epoch_key
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            validation_thresholds = epoch_thresholds
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        completed_epochs = epoch
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {epoch_loss:.6f}, "
            f"val_f1: {epoch_validation_metrics['f1']:.4f}, "
            f"val_recall: {epoch_validation_metrics['recall']:.4f}, "
            f"val_roc_auc: {epoch_validation_metrics['roc_auc']:.4f}, "
            f"val_threshold_percentile: {epoch_percentile}"
        )

        if early_stopping_enabled and epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping triggered: "
                f"no validation improvement for {args.early_stopping_patience} epochs."
            )
            break

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Training finished without a valid best validation checkpoint.")

    model.load_state_dict(best_state_dict)
    threshold = best_threshold
    best_percentile = best_threshold_percentile
    validation_metrics = best_validation_metrics

    test_errors = score_reconstruction_errors(
        model,
        X_test,
        device=device,
        batch_size=args.batch_size,
    )
    test_predictions = (test_errors >= threshold).astype(np.int64)

    metrics_block = build_metrics_block(y_test, test_predictions, test_errors)
    metrics_payload = {
        "lstm_autoencoder": {
            **metrics_block,
            "threshold": threshold,
            "threshold_percentile": best_percentile,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "validation_metrics": validation_metrics,
            "best_epoch": best_epoch,
            "selection_metric": args.selection_metric,
        }
    }

    metrics_path = RESULTS_DIR / "metrics_autoencoder.json"
    metrics_experiment_path = with_artifact_suffix(metrics_path, artifact_suffix)
    save_json(metrics_payload, metrics_path)
    if metrics_experiment_path is not None:
        save_json(metrics_payload, metrics_experiment_path)

    model_path = RESULTS_DIR / "lstm_autoencoder.pt"
    torch.save(model.state_dict(), model_path)

    outputs_path = RESULTS_DIR / "autoencoder_outputs.csv"
    outputs = pd.DataFrame(
        {
            "sequence_index": np.arange(len(y_test)),
            "y_true": y_test,
            "reconstruction_error": test_errors,
            "pred": test_predictions,
        }
    )
    outputs.to_csv(outputs_path, index=False)

    metadata_path = RESULTS_DIR / "autoencoder_metadata.json"
    metadata_experiment_path = with_artifact_suffix(metadata_path, artifact_suffix)
    validation_anomaly_ratio = float(y_validation.mean()) if len(y_validation) else 0.0
    reference_metrics = load_reference_metrics()
    reference_comparison = None
    if reference_metrics is not None:
        reference_comparison = {
            "reference_precision": float(reference_metrics["precision"]),
            "reference_recall": float(reference_metrics["recall"]),
            "reference_f1": float(reference_metrics["f1"]),
            "reference_roc_auc": float(reference_metrics["roc_auc"]),
            "reference_pr_auc": float(reference_metrics["pr_auc"]) if reference_metrics.get("pr_auc") is not None else None,
            "delta_precision": float(metrics_block["precision"] - reference_metrics["precision"]),
            "delta_recall": float(metrics_block["recall"] - reference_metrics["recall"]),
            "delta_f1": float(metrics_block["f1"] - reference_metrics["f1"]),
            "delta_roc_auc": float(metrics_block["roc_auc"] - reference_metrics["roc_auc"]),
            "delta_pr_auc": (
                float(metrics_block["pr_auc"] - reference_metrics["pr_auc"])
                if reference_metrics.get("pr_auc") is not None and metrics_block["pr_auc"] is not None
                else None
            ),
        }

    metadata_payload = {
        "experiment_tag": args.experiment_tag,
        "artifact_suffix": artifact_suffix,
        "feature_columns": lstm_feature_columns,
        "all_feature_columns": feature_columns,
        "feature_count": len(lstm_feature_columns),
        "sequence_length": selected_sequence_length,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "threshold": threshold,
        "threshold_percentile": best_percentile,
        "threshold_percentiles_checked": threshold_percentiles,
        "threshold_selection_method": {
            "selection_split": "validation",
            "candidate_type": "reconstruction_error_percentile",
            "candidate_percentiles": threshold_percentiles,
            "best_epoch_metric": args.selection_metric,
            "epoch_tie_breakers": ["recall", "lower_threshold_percentile"],
        },
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_enabled": early_stopping_enabled,
        "selection_metric": args.selection_metric,
        "stopped_early": bool(completed_epochs < args.epochs),
        "split_sizes": {
            "train": int(len(X_train_inner)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "train_size": int(len(X_train)),
        "train_inner_size": int(len(X_train_inner)),
        "validation_size": int(len(X_validation)),
        "test_size": int(len(X_test)),
        "train_normal_sequences": int(len(X_train_normal)),
        "train_normal_count": int(len(X_train_normal)),
        "validation_anomaly_ratio": validation_anomaly_ratio,
        "sequence_stats": sequence_stats,
        "validation_metrics": validation_metrics,
        "reference_comparison": reference_comparison,
    }
    save_json(metadata_payload, metadata_path)
    if metadata_experiment_path is not None:
        save_json(metadata_payload, metadata_experiment_path)

    history_path = RESULTS_DIR / "autoencoder_history.csv"
    history_experiment_path = with_artifact_suffix(history_path, artifact_suffix)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False)
    if history_experiment_path is not None:
        history_frame.to_csv(history_experiment_path, index=False)

    validation_thresholds_path = RESULTS_DIR / "autoencoder_validation_thresholds.csv"
    pd.DataFrame(validation_thresholds).to_csv(validation_thresholds_path, index=False)

    distribution_path = PLOTS_DIR / "reconstruction_error_distribution.png"
    confusion_path = PLOTS_DIR / "confusion_matrix_autoencoder.png"
    roc_path = PLOTS_DIR / "roc_curve_autoencoder.png"
    distribution_experiment_path = with_artifact_suffix(distribution_path, artifact_suffix)
    confusion_experiment_path = with_artifact_suffix(confusion_path, artifact_suffix)
    roc_experiment_path = with_artifact_suffix(roc_path, artifact_suffix)

    plot_reconstruction_distribution(y_test, test_errors, threshold, distribution_path)
    plot_confusion_matrix(y_test, test_predictions, confusion_path)
    plot_roc_curve(y_test, test_errors, roc_path)
    if distribution_experiment_path is not None:
        plot_reconstruction_distribution(y_test, test_errors, threshold, distribution_experiment_path)
    if confusion_experiment_path is not None:
        plot_confusion_matrix(y_test, test_predictions, confusion_experiment_path)
    if roc_experiment_path is not None:
        plot_roc_curve(y_test, test_errors, roc_experiment_path)

    metrics_table = pd.DataFrame(
        [
            {
                "model": "lstm_autoencoder",
                "precision": metrics_block["precision"],
                "recall": metrics_block["recall"],
                "f1": metrics_block["f1"],
                "roc_auc": metrics_block["roc_auc"],
                "pr_auc": metrics_block["pr_auc"],
                "threshold_percentile": best_percentile,
            }
        ]
    )

    append_experiment_record(
        {
            "run_label": pd.Timestamp.now(tz="UTC").isoformat(),
            "experiment_tag": args.experiment_tag,
            "artifact_suffix": artifact_suffix,
            "sequence_length": selected_sequence_length,
            "feature_count": len(lstm_feature_columns),
            "feature_columns": "|".join(lstm_feature_columns),
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_sequence_count": sequence_stats["train_sequence_count"],
            "test_sequence_count": sequence_stats["test_sequence_count"],
            "sequence_count": sequence_stats["sequence_count"],
            "sequence_anomaly_count": sequence_stats["sequence_anomaly_count"],
            "sequence_anomaly_ratio": sequence_stats["sequence_anomaly_ratio"],
            "epochs_requested": args.epochs,
            "epochs_completed": completed_epochs,
            "best_epoch": best_epoch,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_enabled": int(early_stopping_enabled),
            "selection_metric": args.selection_metric,
            "stopped_early": int(completed_epochs < args.epochs),
            "train_size": int(len(X_train)),
            "train_inner_size": int(len(X_train_inner)),
            "train_normal_count": int(len(X_train_normal)),
            "validation_size": int(len(X_validation)),
            "validation_anomaly_ratio": validation_anomaly_ratio,
            "checked_threshold_percentiles": "|".join(str(value) for value in threshold_percentiles),
            "selected_threshold_percentile": best_percentile,
            "threshold": threshold,
            "precision": metrics_block["precision"],
            "recall": metrics_block["recall"],
            "f1": metrics_block["f1"],
            "roc_auc": metrics_block["roc_auc"],
            "pr_auc": metrics_block["pr_auc"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "reference_recall": reference_metrics["recall"] if reference_metrics is not None else np.nan,
            "reference_f1": reference_metrics["f1"] if reference_metrics is not None else np.nan,
            "reference_roc_auc": reference_metrics["roc_auc"] if reference_metrics is not None else np.nan,
            "reference_pr_auc": (
                reference_metrics["pr_auc"]
                if reference_metrics is not None and reference_metrics.get("pr_auc") is not None
                else np.nan
            ),
            "delta_recall": (
                metrics_block["recall"] - reference_metrics["recall"]
                if reference_metrics is not None
                else np.nan
            ),
            "delta_f1": (
                metrics_block["f1"] - reference_metrics["f1"]
                if reference_metrics is not None
                else np.nan
            ),
            "delta_roc_auc": (
                metrics_block["roc_auc"] - reference_metrics["roc_auc"]
                if reference_metrics is not None
                else np.nan
            ),
            "delta_pr_auc": (
                metrics_block["pr_auc"] - reference_metrics["pr_auc"]
                if reference_metrics is not None and reference_metrics.get("pr_auc") is not None
                else np.nan
            ),
        }
    )

    print(f"Saved autoencoder metrics to: {metrics_path}")
    if metrics_experiment_path is not None:
        print(f"Saved experiment metrics to: {metrics_experiment_path}")
    print(f"Saved validation threshold search to: {validation_thresholds_path}")
    print(f"Saved reconstruction distribution to: {distribution_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved ROC curve to: {roc_path}")
    if metadata_experiment_path is not None:
        print(f"Saved experiment metadata to: {metadata_experiment_path}")
    if history_experiment_path is not None:
        print(f"Saved experiment history to: {history_experiment_path}")
    if distribution_experiment_path is not None:
        print(f"Saved experiment reconstruction plot to: {distribution_experiment_path}")
    if confusion_experiment_path is not None:
        print(f"Saved experiment confusion matrix to: {confusion_experiment_path}")
    if roc_experiment_path is not None:
        print(f"Saved experiment ROC curve to: {roc_experiment_path}")
    print(
        "Validation threshold selection: "
        f"best_epoch={best_epoch}, "
        f"percentile={best_percentile}, "
        f"precision={validation_metrics['precision']:.4f}, "
        f"recall={validation_metrics['recall']:.4f}, "
        f"f1={validation_metrics['f1']:.4f}, "
        f"roc_auc={validation_metrics['roc_auc']:.4f}"
    )
    print("Autoencoder metrics:")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
