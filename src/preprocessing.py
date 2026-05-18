from __future__ import annotations

import json

import numpy as np
import pandas as pd

from features import CATEGORICAL_COLUMNS
from features import LSTM_FEATURE_COLUMNS
from features import NUMERIC_COLUMNS
from features import build_feature_table
from features import build_sequences
from features import fit_category_encoders
from features import fit_numeric_scaler
from features import sort_events
from utils import PROCESSED_DIR
from utils import RAW_DIR
from utils import ensure_directories
from utils import save_json


SEQUENCE_LENGTH = 10
TRAIN_FRACTION = 0.7


def concatenate_sequence_batches(*batches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid_batches = [batch for batch in batches if len(batch["X"]) > 0]
    if not valid_batches:
        return {
            "X": np.empty((0, SEQUENCE_LENGTH, 0), dtype=np.float32),
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


def main() -> None:
    ensure_directories()

    input_path = RAW_DIR / "synthetic_logs.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "Raw data not found. Run `python3 src/generate_data.py` first."
        )

    raw_frame = pd.read_csv(input_path)
    sorted_frame = sort_events(raw_frame)

    split_index = int(len(sorted_frame) * TRAIN_FRACTION)
    train_raw = sorted_frame.iloc[:split_index].reset_index(drop=True)
    test_raw = sorted_frame.iloc[split_index:].reset_index(drop=True)

    # Fit encoders and scaler only on the training part to avoid data leakage.
    category_encoders = fit_category_encoders(train_raw, CATEGORICAL_COLUMNS)
    numeric_scaler = fit_numeric_scaler(train_raw, NUMERIC_COLUMNS)

    train_processed, feature_columns = build_feature_table(
        train_raw,
        encoders=category_encoders,
        scaler=numeric_scaler,
    )
    test_processed, _ = build_feature_table(
        test_raw,
        encoders=category_encoders,
        scaler=numeric_scaler,
    )

    train_processed["split"] = "train"
    test_processed["split"] = "test"

    baseline_dataset = pd.concat([train_processed, test_processed], axis=0).reset_index(drop=True)

    baseline_dataset_path = PROCESSED_DIR / "baseline_dataset.csv"
    baseline_dataset.to_csv(baseline_dataset_path, index=False)

    train_events_path = PROCESSED_DIR / "events_train.csv"
    test_events_path = PROCESSED_DIR / "events_test.csv"
    train_processed.to_csv(train_events_path, index=False)
    test_processed.to_csv(test_events_path, index=False)

    train_sequences = build_sequences(
        train_processed,
        feature_columns=feature_columns,
        sequence_length=SEQUENCE_LENGTH,
    )
    test_sequences = build_sequences(
        test_processed,
        feature_columns=feature_columns,
        sequence_length=SEQUENCE_LENGTH,
    )
    all_sequences = concatenate_sequence_batches(train_sequences, test_sequences)

    sequences_path = PROCESSED_DIR / "sequences.npy"
    sequence_labels_path = PROCESSED_DIR / "sequence_labels.npy"
    np.save(sequences_path, all_sequences["X"])
    np.save(sequence_labels_path, all_sequences["y"])

    train_sequences_path = PROCESSED_DIR / "sequences_train.npz"
    test_sequences_path = PROCESSED_DIR / "sequences_test.npz"
    np.savez_compressed(train_sequences_path, **train_sequences)
    np.savez_compressed(test_sequences_path, **test_sequences)

    feature_columns_path = PROCESSED_DIR / "feature_columns.json"
    with feature_columns_path.open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2, ensure_ascii=False)

    feature_config_path = PROCESSED_DIR / "feature_config.json"
    save_json(
        {
            "feature_columns": feature_columns,
            "feature_columns_lstm": LSTM_FEATURE_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "train_rows": int(len(train_processed)),
            "test_rows": int(len(test_processed)),
            "train_sequences": int(len(train_sequences["X"])),
            "test_sequences": int(len(test_sequences["X"])),
            "baseline_rows": int(len(baseline_dataset)),
        },
        feature_config_path,
    )

    print(f"Saved baseline dataset to: {baseline_dataset_path}")
    print(f"Saved sequences to: {sequences_path}")
    print(f"Saved sequence labels to: {sequence_labels_path}")
    print(f"Saved feature columns to: {feature_columns_path}")
    print(f"Baseline dataset shape: {baseline_dataset.shape}")
    print(f"Sequences shape: {all_sequences['X'].shape}")
    print(f"Sequence labels shape: {all_sequences['y'].shape}")


if __name__ == "__main__":
    main()
