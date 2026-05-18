from __future__ import annotations

import os
from pathlib import Path

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

from utils import PLOTS_DIR
from utils import PROCESSED_DIR
from utils import RESULTS_DIR
from utils import compute_classification_metrics
from utils import ensure_directories
from utils import save_json
from utils import set_seed


RANDOM_STATE = 42
TEST_SIZE = 0.3
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 16
THRESHOLD_PERCENTILE = 95


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE) -> None:
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
    batch_size: int = BATCH_SIZE,
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
        "confusion_matrix": confusion.tolist(),
    }


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
    ensure_directories()
    set_seed(RANDOM_STATE)

    sequences_path = PROCESSED_DIR / "sequences.npy"
    labels_path = PROCESSED_DIR / "sequence_labels.npy"
    if not sequences_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Sequence files not found. Run `python3 src/preprocessing.py` first."
        )

    sequences = np.load(sequences_path).astype(np.float32)
    sequence_labels = np.load(labels_path).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        sequence_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=sequence_labels,
    )

    X_train_normal = X_train[y_train == 0]
    if len(X_train_normal) == 0:
        raise ValueError("No normal train sequences available for autoencoder training.")

    device = get_device()
    model = LSTMAutoencoder(input_size=sequences.shape[2], hidden_size=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal).float()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
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
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}/{EPOCHS} - loss: {epoch_loss:.6f}")

    train_normal_errors = score_reconstruction_errors(model, X_train_normal, device=device)
    threshold = float(np.percentile(train_normal_errors, THRESHOLD_PERCENTILE))

    test_errors = score_reconstruction_errors(model, X_test, device=device)
    test_predictions = (test_errors >= threshold).astype(np.int64)

    metrics_block = build_metrics_block(y_test, test_predictions, test_errors)
    metrics_payload = {
        "lstm_autoencoder": {
            **metrics_block,
            "threshold": threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "epochs": EPOCHS,
            "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
        }
    }

    metrics_path = RESULTS_DIR / "metrics_autoencoder.json"
    save_json(metrics_payload, metrics_path)

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
    save_json(
        {
            "threshold": threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "epochs": EPOCHS,
            "hidden_size": HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "train_normal_sequences": int(len(X_train_normal)),
        },
        metadata_path,
    )

    history_path = RESULTS_DIR / "autoencoder_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    distribution_path = PLOTS_DIR / "reconstruction_error_distribution.png"
    confusion_path = PLOTS_DIR / "confusion_matrix_autoencoder.png"
    roc_path = PLOTS_DIR / "roc_curve_autoencoder.png"

    plot_reconstruction_distribution(y_test, test_errors, threshold, distribution_path)
    plot_confusion_matrix(y_test, test_predictions, confusion_path)
    plot_roc_curve(y_test, test_errors, roc_path)

    metrics_table = pd.DataFrame(
        [
            {
                "model": "lstm_autoencoder",
                "precision": metrics_block["precision"],
                "recall": metrics_block["recall"],
                "f1": metrics_block["f1"],
                "roc_auc": metrics_block["roc_auc"],
            }
        ]
    )

    print(f"Saved autoencoder metrics to: {metrics_path}")
    print(f"Saved reconstruction distribution to: {distribution_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print("Autoencoder metrics:")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
