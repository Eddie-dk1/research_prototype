from __future__ import annotations

import pandas as pd

from utils import RAW_DIR
from utils import RESULTS_DIR
from utils import ensure_directories


def build_user_profiles(normal_frame: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, float]]:
    typical_ips = (
        normal_frame.groupby("user_id")["source_ip"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    typical_event_types = (
        normal_frame.groupby("user_id")["event_type"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    typical_bytes_sent = (
        normal_frame.groupby("user_id")["bytes_sent"]
        .median()
        .to_dict()
    )
    return typical_ips, typical_event_types, typical_bytes_sent


def detect_probable_reason(
    row: pd.Series,
    typical_ips: dict[str, set[str]],
    typical_event_types: dict[str, set[str]],
    typical_bytes_sent: dict[str, float],
    high_traffic_threshold: float,
) -> str:
    user_id = str(row["user_id"])
    row_hour = int(row["hour"])
    bytes_sent = float(row["bytes_sent"])
    failed_logins = int(row.get("failed_logins", 0))
    source_ip = str(row["source_ip"])
    event_type = str(row["event_type"])

    if failed_logins >= 5 or (
        str(row["login_status"]) == "failure" and row["anomaly_type"] == "brute_force_login"
    ):
        return "много неуспешных входов"

    if row_hour <= 5 or row_hour >= 22:
        return "активность в нетипичное ночное время"

    user_median = float(typical_bytes_sent.get(user_id, 0.0))
    if bytes_sent >= high_traffic_threshold or (user_median > 0 and bytes_sent >= user_median * 4):
        return "резкий рост исходящего трафика"

    user_ips = typical_ips.get(user_id, set())
    if user_ips and source_ip not in user_ips:
        return "необычная смена IP-адреса"

    user_event_types = typical_event_types.get(user_id, set())
    if user_event_types and event_type not in user_event_types:
        return "нетипичный тип события"

    anomaly_type_to_reason = {
        "brute_force_login": "много неуспешных входов",
        "unusual_night_activity": "активность в нетипичное ночное время",
        "data_exfiltration": "резкий рост исходящего трафика",
        "unusual_ip_change": "необычная смена IP-адреса",
    }
    return anomaly_type_to_reason.get(str(row["anomaly_type"]), "нетипичный тип события")


def main() -> None:
    ensure_directories()

    input_path = RAW_DIR / "synthetic_logs.csv"
    output_path = RESULTS_DIR / "anomaly_examples.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Raw synthetic logs not found. Run `python3 src/generate_data.py` first."
        )

    data_frame = pd.read_csv(input_path)
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    data_frame = data_frame.sort_values("timestamp").reset_index(drop=True)

    normal_frame = data_frame[data_frame["label"] == 0].copy()
    anomaly_frame = data_frame[data_frame["label"] == 1].copy()

    typical_ips, typical_event_types, typical_bytes_sent = build_user_profiles(normal_frame)
    high_traffic_threshold = float(normal_frame["bytes_sent"].quantile(0.99))

    # Each anomalous event receives one most probable explanation.
    anomaly_frame["probable_reason"] = anomaly_frame.apply(
        detect_probable_reason,
        axis=1,
        typical_ips=typical_ips,
        typical_event_types=typical_event_types,
        typical_bytes_sent=typical_bytes_sent,
        high_traffic_threshold=high_traffic_threshold,
    )

    result_frame = anomaly_frame[
        [
            "user_id",
            "timestamp",
            "anomaly_type",
            "probable_reason",
            "bytes_sent",
            "source_ip",
            "event_type",
        ]
    ].copy()
    result_frame["timestamp"] = result_frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result_frame.to_csv(output_path, index=False)

    print(f"Saved anomaly examples to: {output_path}")
    print("First 10 anomaly examples:")
    print(result_frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
