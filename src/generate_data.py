from __future__ import annotations

import ipaddress

import numpy as np
import pandas as pd

from utils import RAW_DIR
from utils import ensure_directories
from utils import set_seed


NORMAL_ANOMALY_TYPE = "normal"
ANOMALY_TYPES = [
    "brute_force_login",
    "unusual_night_activity",
    "data_exfiltration",
    "unusual_ip_change",
]


def random_ip(rng: np.random.Generator, network: str) -> str:
    base_network = ipaddress.ip_network(network)
    host_index = int(rng.integers(10, base_network.num_addresses - 10))
    return str(base_network.network_address + host_index)


def build_user_profiles(
    user_ids: list[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    departments = ["it", "finance", "hr", "sales", "security"]
    roles = ["analyst", "engineer", "manager", "admin"]
    devices = ["laptop", "workstation", "server"]
    locations = ["office", "vpn", "remote"]

    profiles: dict[str, dict[str, object]] = {}
    for user_id in user_ids:
        usual_ips = sorted(
            {
                random_ip(rng, "10.10.0.0/16"),
                random_ip(rng, "10.20.0.0/16"),
            }
        )
        profiles[user_id] = {
            "department": rng.choice(departments, p=[0.28, 0.18, 0.14, 0.22, 0.18]),
            "user_role": rng.choice(roles, p=[0.34, 0.32, 0.24, 0.1]),
            "device_type": rng.choice(devices, p=[0.5, 0.35, 0.15]),
            "usual_source_ips": usual_ips,
            "rare_source_ip": random_ip(rng, "185.220.100.0/24"),
            "default_location": rng.choice(locations, p=[0.65, 0.2, 0.15]),
        }
    return profiles


def build_anomaly_schedule(
    n_events: int,
    anomaly_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_anomalies = int(round(n_events * anomaly_ratio))
    n_normals = n_events - n_anomalies

    anomaly_labels = rng.choice(
        ANOMALY_TYPES,
        size=n_anomalies,
        p=[0.28, 0.22, 0.3, 0.2],
    )
    full_schedule = np.concatenate(
        [np.repeat(NORMAL_ANOMALY_TYPE, n_normals), anomaly_labels]
    )
    rng.shuffle(full_schedule)
    return full_schedule


def make_timestamp(
    rng: np.random.Generator,
    start_timestamp: pd.Timestamp,
    anomaly_type: str,
) -> pd.Timestamp:
    day_offset = int(rng.integers(0, 60))

    if anomaly_type == "unusual_night_activity":
        hour = int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))
    else:
        hour = int(rng.choice(np.arange(8, 19)))

    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return (
        start_timestamp
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )


def generate_normal_event(
    rng: np.random.Generator,
    user_id: str,
    profile: dict[str, object],
    timestamp: pd.Timestamp,
) -> dict[str, object]:
    event_type = rng.choice(
        ["login", "file_access", "web_request", "dns_query", "email_access"],
        p=[0.24, 0.22, 0.28, 0.14, 0.12],
    )
    login_status = "success" if event_type == "login" else "not_applicable"
    if event_type == "login" and rng.random() < 0.06:
        login_status = "failure"

    source_ip = rng.choice(profile["usual_source_ips"])
    destination_pool = {
        "login": [random_ip(rng, "10.0.1.0/24"), random_ip(rng, "10.0.2.0/24")],
        "file_access": [random_ip(rng, "10.0.10.0/24"), random_ip(rng, "10.0.11.0/24")],
        "web_request": [random_ip(rng, "172.16.20.0/24"), random_ip(rng, "172.16.21.0/24")],
        "dns_query": [random_ip(rng, "10.0.53.0/24")],
        "email_access": [random_ip(rng, "172.16.30.0/24")],
    }
    destination_ip = rng.choice(destination_pool[event_type])

    protocol_map = {
        "login": rng.choice(["SSH", "HTTPS"]),
        "file_access": rng.choice(["SMB", "HTTPS"]),
        "web_request": "HTTPS",
        "dns_query": "DNS",
        "email_access": "HTTPS",
    }
    protocol = protocol_map[event_type]

    bytes_sent = float(rng.lognormal(mean=7.0, sigma=0.42))
    bytes_received = float(rng.lognormal(mean=7.3, sigma=0.38))
    session_duration = float(rng.integers(60, 3600))
    failed_logins = 1 if login_status == "failure" else 0
    distinct_destinations = int(rng.integers(1, 5))
    privileged_action = int(rng.choice([0, 1], p=[0.93, 0.07]))
    dst_service = {
        "login": "auth_server",
        "file_access": "fileshare",
        "web_request": "intranet",
        "dns_query": "dns_resolver",
        "email_access": "mail",
    }[event_type]

    return {
        "timestamp": timestamp,
        "user_id": user_id,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "event_type": event_type,
        "login_status": login_status,
        "bytes_sent": round(bytes_sent, 2),
        "bytes_received": round(bytes_received, 2),
        "protocol": protocol,
        "hour": int(timestamp.hour),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "label": 0,
        "anomaly_type": NORMAL_ANOMALY_TYPE,
        "department": profile["department"],
        "user_role": profile["user_role"],
        "device_type": profile["device_type"],
        "location": profile["default_location"],
        "dst_service": dst_service,
        "session_duration": round(session_duration, 2),
        "failed_logins": failed_logins,
        "distinct_destinations": distinct_destinations,
        "privileged_action": privileged_action,
    }


def generate_anomaly_event(
    rng: np.random.Generator,
    user_id: str,
    profile: dict[str, object],
    timestamp: pd.Timestamp,
    anomaly_type: str,
) -> dict[str, object]:
    event = generate_normal_event(rng, user_id, profile, timestamp)
    event["label"] = 1
    event["anomaly_type"] = anomaly_type

    if anomaly_type == "brute_force_login":
        event.update(
            {
                "event_type": "login",
                "login_status": "failure",
                "destination_ip": random_ip(rng, "10.0.1.0/24"),
                "protocol": rng.choice(["SSH", "RDP", "HTTPS"]),
                "bytes_sent": round(float(rng.lognormal(mean=6.4, sigma=0.28)), 2),
                "bytes_received": round(float(rng.lognormal(mean=6.2, sigma=0.25)), 2),
                "dst_service": "auth_server",
                "session_duration": round(float(rng.integers(15, 240)), 2),
                "failed_logins": int(rng.integers(6, 15)),
                "distinct_destinations": int(rng.integers(1, 3)),
                "privileged_action": 0,
            }
        )

    elif anomaly_type == "unusual_night_activity":
        event.update(
            {
                "event_type": rng.choice(["file_access", "web_request", "email_access"]),
                "login_status": "not_applicable",
                "bytes_sent": round(float(rng.lognormal(mean=8.1, sigma=0.45)), 2),
                "bytes_received": round(float(rng.lognormal(mean=8.0, sigma=0.4)), 2),
                "session_duration": round(float(rng.integers(1800, 10800)), 2),
                "distinct_destinations": int(rng.integers(4, 10)),
                "privileged_action": int(rng.choice([0, 1], p=[0.55, 0.45])),
                "location": rng.choice(["remote", "vpn"]),
                "dst_service": rng.choice(["fileshare", "intranet", "mail"]),
            }
        )

    elif anomaly_type == "data_exfiltration":
        event.update(
            {
                "event_type": "file_access",
                "login_status": "not_applicable",
                "destination_ip": random_ip(rng, "203.0.113.0/24"),
                "protocol": rng.choice(["HTTPS", "SFTP"]),
                "bytes_sent": round(float(rng.lognormal(mean=11.1, sigma=0.55)), 2),
                "bytes_received": round(float(rng.lognormal(mean=7.4, sigma=0.35)), 2),
                "session_duration": round(float(rng.integers(2400, 14000)), 2),
                "failed_logins": 0,
                "distinct_destinations": int(rng.integers(8, 18)),
                "privileged_action": 1,
                "location": rng.choice(["vpn", "remote"]),
                "dst_service": "external_storage",
            }
        )

    elif anomaly_type == "unusual_ip_change":
        event.update(
            {
                "event_type": "login",
                "login_status": "success",
                "source_ip": profile["rare_source_ip"],
                "destination_ip": random_ip(rng, "10.0.1.0/24"),
                "protocol": rng.choice(["HTTPS", "SSH"]),
                "bytes_sent": round(float(rng.lognormal(mean=7.1, sigma=0.4)), 2),
                "bytes_received": round(float(rng.lognormal(mean=7.1, sigma=0.35)), 2),
                "session_duration": round(float(rng.integers(120, 2400)), 2),
                "failed_logins": 0,
                "distinct_destinations": int(rng.integers(1, 4)),
                "privileged_action": int(rng.choice([0, 1], p=[0.75, 0.25])),
                "location": "external",
                "dst_service": "auth_server",
            }
        )

    event["hour"] = int(pd.Timestamp(event["timestamp"]).hour)
    event["is_weekend"] = int(pd.Timestamp(event["timestamp"]).dayofweek >= 5)
    return event


def generate_synthetic_logs(
    n_events: int = 30000,
    n_users: int = 100,
    anomaly_ratio: float = 0.065,
    seed: int = 42,
) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    user_ids = [f"user_{index:03d}" for index in range(1, n_users + 1)]
    user_profiles = build_user_profiles(user_ids, rng)
    anomaly_schedule = build_anomaly_schedule(n_events, anomaly_ratio, rng)

    start_timestamp = pd.Timestamp("2025-01-01 00:00:00")
    events: list[dict[str, object]] = []

    for anomaly_type in anomaly_schedule:
        user_id = str(rng.choice(user_ids))
        profile = user_profiles[user_id]
        timestamp = make_timestamp(rng, start_timestamp, str(anomaly_type))

        if anomaly_type == NORMAL_ANOMALY_TYPE:
            record = generate_normal_event(rng, user_id, profile, timestamp)
        else:
            record = generate_anomaly_event(rng, user_id, profile, timestamp, str(anomaly_type))

        events.append(record)

    data_frame = pd.DataFrame.from_records(events)
    data_frame = data_frame.sort_values("timestamp").reset_index(drop=True)
    return data_frame


def main() -> None:
    ensure_directories()

    output_path = RAW_DIR / "synthetic_logs.csv"
    data_frame = generate_synthetic_logs()
    data_frame.to_csv(output_path, index=False)

    print(f"Saved synthetic logs to: {output_path}")
    print(f"Dataset size: {data_frame.shape}")
    print("Label distribution:")
    print(data_frame["label"].value_counts(dropna=False).sort_index())
    print("Anomaly type distribution:")
    print(data_frame["anomaly_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
