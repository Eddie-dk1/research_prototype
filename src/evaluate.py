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
import pandas as pd

from utils import PLOTS_DIR
from utils import RESULTS_DIR
from utils import ensure_directories
from utils import load_json


MODEL_NAME_MAP = {
    "isolation_forest": "Isolation Forest",
    "random_forest": "Random Forest",
    "lstm_autoencoder": "LSTM-Autoencoder",
}

MODEL_ORDER = [
    "Isolation Forest",
    "Random Forest",
    "LSTM-Autoencoder",
]


def build_summary_table(
    baseline_metrics: dict,
    autoencoder_metrics: dict,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for raw_name, metrics in baseline_metrics.items():
        rows.append(
            {
                "model": MODEL_NAME_MAP[raw_name],
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1", metrics.get("f1_score")),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            }
        )

    for raw_name, metrics in autoencoder_metrics.items():
        rows.append(
            {
                "model": MODEL_NAME_MAP[raw_name],
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1", metrics.get("f1_score")),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            }
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame["model"] = pd.Categorical(
        summary_frame["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    summary_frame = summary_frame.sort_values("model").reset_index(drop=True)
    return summary_frame


def plot_metric_bars(
    summary_frame: pd.DataFrame,
    metric_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    values = summary_frame[metric_column].astype(float)
    labels = summary_frame["model"].astype(str)
    colors = ["#4C78A8", "#54A24B", "#E45756"]

    bars = axis.bar(labels, values, color=colors)
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    ensure_directories()

    baseline_metrics_path = RESULTS_DIR / "metrics_baseline.json"
    autoencoder_metrics_path = RESULTS_DIR / "metrics_autoencoder.json"

    if not baseline_metrics_path.exists():
        raise FileNotFoundError(
            "Baseline metrics not found. Run `python3 src/train_baseline.py` first."
        )
    if not autoencoder_metrics_path.exists():
        raise FileNotFoundError(
            "Autoencoder metrics not found. Run `python3 src/train_lstm_autoencoder.py` first."
        )

    baseline_metrics = load_json(baseline_metrics_path)
    autoencoder_metrics = load_json(autoencoder_metrics_path)

    summary_frame = build_summary_table(baseline_metrics, autoencoder_metrics)

    summary_path = RESULTS_DIR / "metrics_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    f1_plot_path = PLOTS_DIR / "f1_score_comparison.png"
    roc_auc_plot_path = PLOTS_DIR / "roc_auc_comparison.png"
    pr_auc_plot_path = PLOTS_DIR / "pr_auc_comparison.png"

    plot_metric_bars(
        summary_frame,
        metric_column="f1",
        title="F1-score Comparison",
        y_label="F1-score",
        output_path=f1_plot_path,
    )
    plot_metric_bars(
        summary_frame,
        metric_column="roc_auc",
        title="ROC-AUC Comparison",
        y_label="ROC-AUC",
        output_path=roc_auc_plot_path,
    )
    plot_metric_bars(
        summary_frame,
        metric_column="pr_auc",
        title="PR-AUC Comparison",
        y_label="PR-AUC",
        output_path=pr_auc_plot_path,
    )

    print("Metrics summary:")
    print(summary_frame.to_string(index=False))
    print(f"Saved metrics summary to: {summary_path}")
    print(f"Saved F1-score comparison plot to: {f1_plot_path}")
    print(f"Saved ROC-AUC comparison plot to: {roc_auc_plot_path}")
    print(f"Saved PR-AUC comparison plot to: {pr_auc_plot_path}")


if __name__ == "__main__":
    main()
