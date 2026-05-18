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
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import PLOTS_DIR
from utils import PROCESSED_DIR
from utils import RESULTS_DIR
from utils import compute_classification_metrics
from utils import ensure_directories
from utils import save_json
from utils import set_seed


RANDOM_STATE = 42
TEST_SIZE = 0.3

META_COLUMNS = [
    "timestamp",
    "user_id",
    "label",
    "anomaly_type",
    "split",
]


def get_feature_columns(data_frame: pd.DataFrame) -> list[str]:
    return [column for column in data_frame.columns if column not in META_COLUMNS]


def build_metrics_block(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: pd.Series,
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


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=["Normal", "Anomaly"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curve(
    y_true: pd.Series,
    rf_scores: pd.Series,
    if_scores: pd.Series,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        rf_scores,
        name="Random Forest",
        ax=axis,
    )
    RocCurveDisplay.from_predictions(
        y_true,
        if_scores,
        name="Isolation Forest",
        ax=axis,
    )

    axis.set_title("ROC Curve: Baseline Models")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    ensure_directories()
    set_seed(RANDOM_STATE)

    baseline_dataset_path = PROCESSED_DIR / "baseline_dataset.csv"
    if not baseline_dataset_path.exists():
        raise FileNotFoundError(
            "Baseline dataset not found. Run `python3 src/preprocessing.py` first."
        )

    baseline_frame = pd.read_csv(baseline_dataset_path)
    feature_columns = get_feature_columns(baseline_frame)

    X = baseline_frame[feature_columns]
    y = baseline_frame["label"].astype(int)
    meta = baseline_frame[["timestamp", "user_id", "anomaly_type"]]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    anomaly_ratio = float(y_train.mean())

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    rf_scores = random_forest.predict_proba(X_test)[:, 1]
    rf_predictions = (rf_scores >= 0.5).astype(int)

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=anomaly_ratio,
        random_state=RANDOM_STATE,
    )
    X_train_normal = X_train[y_train == 0]
    isolation_forest.fit(X_train_normal)

    train_normal_scores = -isolation_forest.score_samples(X_train_normal)
    if_threshold = float(pd.Series(train_normal_scores).quantile(1 - anomaly_ratio))

    if_scores = -isolation_forest.score_samples(X_test)
    if_predictions = (if_scores >= if_threshold).astype(int)

    rf_metrics = build_metrics_block(y_test, rf_predictions, rf_scores)
    if_metrics = build_metrics_block(y_test, if_predictions, if_scores)

    metrics_payload = {
        "isolation_forest": {
            **if_metrics,
            "contamination": anomaly_ratio,
            "threshold": if_threshold,
        },
        "random_forest": {
            **rf_metrics,
            "n_estimators": 200,
            "max_depth": 10,
        },
    }
    metrics_path = RESULTS_DIR / "metrics_baseline.json"
    save_json(metrics_payload, metrics_path)

    outputs = pd.DataFrame(
        {
            "timestamp": meta_test["timestamp"].to_numpy(),
            "user_id": meta_test["user_id"].to_numpy(),
            "anomaly_type": meta_test["anomaly_type"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "rf_score": rf_scores,
            "rf_pred": rf_predictions,
            "iforest_score": if_scores,
            "iforest_pred": if_predictions,
        }
    )
    outputs_path = RESULTS_DIR / "baseline_outputs.csv"
    outputs.to_csv(outputs_path, index=False)

    summary_path = RESULTS_DIR / "baseline_training_summary.json"
    save_json(
        {
            "feature_columns": feature_columns,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "anomaly_ratio_train": anomaly_ratio,
            "random_state": RANDOM_STATE,
            "test_size_fraction": TEST_SIZE,
        },
        summary_path,
    )

    rf_confusion_path = PLOTS_DIR / "confusion_matrix_random_forest.png"
    if_confusion_path = PLOTS_DIR / "confusion_matrix_isolation_forest.png"
    roc_path = PLOTS_DIR / "roc_curve_baseline.png"

    plot_confusion_matrix(
        y_test,
        rf_predictions,
        "Confusion Matrix: Random Forest",
        rf_confusion_path,
    )
    plot_confusion_matrix(
        y_test,
        if_predictions,
        "Confusion Matrix: Isolation Forest",
        if_confusion_path,
    )
    plot_roc_curve(y_test, rf_scores, if_scores, roc_path)

    metrics_table = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "precision": rf_metrics["precision"],
                "recall": rf_metrics["recall"],
                "f1": rf_metrics["f1"],
                "roc_auc": rf_metrics["roc_auc"],
                "pr_auc": rf_metrics["pr_auc"],
            },
            {
                "model": "isolation_forest",
                "precision": if_metrics["precision"],
                "recall": if_metrics["recall"],
                "f1": if_metrics["f1"],
                "roc_auc": if_metrics["roc_auc"],
                "pr_auc": if_metrics["pr_auc"],
            },
        ]
    )

    print(f"Saved baseline metrics to: {metrics_path}")
    print(f"Saved Random Forest confusion matrix to: {rf_confusion_path}")
    print(f"Saved Isolation Forest confusion matrix to: {if_confusion_path}")
    print(f"Saved baseline ROC curve to: {roc_path}")
    print("Baseline metrics:")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
