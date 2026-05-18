from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLUMNS = [
    "user_id",
    "source_ip",
    "destination_ip",
    "event_type",
    "login_status",
    "protocol",
]

NUMERIC_COLUMNS = [
    "bytes_sent",
    "bytes_received",
    "hour",
]

PASSTHROUGH_COLUMNS = [
    "is_weekend",
]

LSTM_FEATURE_COLUMNS = [
    "bytes_sent",
    "bytes_received",
    "hour",
    "is_weekend",
    "event_type_encoded",
    "login_status_encoded",
    "protocol_encoded",
]


def sort_events(data_frame: pd.DataFrame) -> pd.DataFrame:
    frame = data_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["hour"] = frame["hour"].astype(int)
    frame["is_weekend"] = frame["is_weekend"].astype(int)
    frame["label"] = frame["label"].astype(int)
    return frame


def fit_category_encoders(
    train_frame: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    columns = categorical_columns or CATEGORICAL_COLUMNS
    encoders: dict[str, dict[str, int]] = {}

    for column in columns:
        unique_values = pd.Series(train_frame[column].astype(str).unique()).sort_values()
        encoders[column] = {value: index for index, value in enumerate(unique_values.tolist())}

    return encoders


def apply_category_encoders(
    data_frame: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=data_frame.index)

    for column, mapping in encoders.items():
        encoded[f"{column}_encoded"] = (
            data_frame[column].astype(str).map(mapping).fillna(-1).astype(np.int32)
        )

    return encoded


def fit_numeric_scaler(
    train_frame: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> StandardScaler:
    columns = numeric_columns or NUMERIC_COLUMNS
    scaler = StandardScaler()
    scaler.fit(train_frame[columns].astype(float))
    return scaler


def apply_numeric_scaler(
    data_frame: pd.DataFrame,
    scaler: StandardScaler,
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = numeric_columns or NUMERIC_COLUMNS
    scaled_values = scaler.transform(data_frame[columns].astype(float))
    scaled_frame = pd.DataFrame(
        scaled_values,
        index=data_frame.index,
        columns=columns,
    )
    return scaled_frame.astype(np.float32)


def build_feature_table(
    data_frame: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, list[str]]:
    frame = sort_events(data_frame)

    encoded_categorical = apply_category_encoders(frame, encoders)
    scaled_numeric = apply_numeric_scaler(frame, scaler)
    passthrough = frame[PASSTHROUGH_COLUMNS].astype(np.float32)

    feature_frame = pd.concat([scaled_numeric, passthrough, encoded_categorical], axis=1)
    feature_columns = feature_frame.columns.tolist()

    result = pd.concat(
        [
            frame[["timestamp", "user_id", "label", "anomaly_type"]].reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    return result, feature_columns


def build_sequences(
    data_frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 10,
) -> dict[str, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    user_ids: list[str] = []
    timestamps: list[str] = []

    sorted_frame = data_frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    sorted_frame["timestamp"] = pd.to_datetime(sorted_frame["timestamp"])

    for user_id, user_frame in sorted_frame.groupby("user_id"):
        if len(user_frame) < sequence_length:
            continue

        values = user_frame[feature_columns].to_numpy(dtype=np.float32)
        window_labels = user_frame["label"].to_numpy(dtype=np.int64)
        window_timestamps = user_frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

        for start_index in range(0, len(user_frame) - sequence_length + 1):
            stop_index = start_index + sequence_length

            sequences.append(values[start_index:stop_index])
            labels.append(int(window_labels[start_index:stop_index].max()))
            user_ids.append(str(user_id))
            timestamps.append(window_timestamps[stop_index - 1])

    if not sequences:
        return {
            "X": np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "user_ids": np.empty((0,), dtype="<U1"),
            "timestamps": np.empty((0,), dtype="<U1"),
        }

    return {
        "X": np.stack(sequences).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "user_ids": np.asarray(user_ids),
        "timestamps": np.asarray(timestamps),
    }
