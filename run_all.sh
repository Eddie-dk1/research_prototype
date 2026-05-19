#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SKIP_GENERATE=0
BEST_AUTOENCODER=0
WITH_STAGE10=0
STAGE10_EPOCHS=30

usage() {
  cat <<'EOF'
Usage:
  ./run_all.sh [options]

Options:
  --skip-generate      Skip synthetic data generation if raw data already exists.
  --best-autoencoder   Train the current best practical autoencoder config instead of defaults.
  --with-stage10       Run the full stage 10 experiment sequence after the standard pipeline.
  --stage10-epochs N   Set experiment 6 epochs for stage 10. Default: 30.
  -h, --help           Show this help.

Examples:
  ./run_all.sh
  ./run_all.sh --best-autoencoder
  ./run_all.sh --skip-generate --with-stage10 --stage10-epochs 50
EOF
}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

run_best_autoencoder() {
  "$PYTHON_BIN" src/train_lstm_autoencoder.py \
    --sequence-length 5 \
    --hidden-size 32 \
    --epochs 10 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --threshold-percentiles 85 90 92 95 \
    --early-stopping-patience 0 \
    --selection-metric f1
}

print_final_report() {
  PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas is required for the final report. Install it with: pip install pandas")
    sys.exit(0)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


project_root = Path(os.environ["PROJECT_ROOT"])
results_dir = project_root / "results"
plots_dir = results_dir / "plots"


def normalize_scalar(value, digits=4):
    if isinstance(value, bool):
        return value
    try:
        numeric_value = float(value)
    except Exception:
        return value
    return round(numeric_value, digits)


def format_number(value, digits=4):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def rounded_df(df, default_digits=4, column_digits=None):
    formatted = df.copy()
    numeric_columns = formatted.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        digits = default_digits
        if column_digits and column in column_digits:
            digits = column_digits[column]
        formatted[column] = formatted[column].round(digits)
    return formatted


def print_header(title):
    print(f"\n{title}")
    print("=" * len(title))


def print_warning(message):
    print(f"[!] {message}")


warnings = []


def add_warning(section, message):
    warnings.append((section, message))


def get_warnings(section):
    return [message for warning_section, message in warnings if warning_section == section]


def read_csv(path, section):
    if not path.exists():
        add_warning(section, f"File not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        add_warning(section, f"Failed to read CSV {path}: {exc}")
        return None


def read_json(path, section):
    if not path.exists():
        add_warning(section, f"File not found: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        add_warning(section, f"Failed to read JSON {path}: {exc}")
        return None


def metric_value(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                return None
    return None


def format_metric(name, value):
    if value is None:
        return f"{name}: n/a"
    return f"{name}: {value:.4f}"


def flatten_autoencoder_metrics(payload):
    if not isinstance(payload, dict):
        return {}
    if "lstm_autoencoder" in payload and isinstance(payload["lstm_autoencoder"], dict):
        return payload["lstm_autoencoder"]
    return payload


def print_json_metrics_table(title, metrics, ordered_keys):
    if title:
        print(title)
    rows = []
    for key in ordered_keys:
        if key in metrics and not isinstance(metrics[key], (dict, list)):
            rows.append({"metric": key, "value": normalize_scalar(metrics[key])})
    if not rows:
        print("No scalar metrics available.")
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def human_size(num_bytes):
    try:
        size = float(num_bytes)
    except Exception:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def directory_size(path):
    total = 0
    try:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
    except Exception:
        return None
    return total


def choose_f1_column(df):
    if df is None:
        return None
    if "f1" in df.columns:
        return "f1"
    if "f1_score" in df.columns:
        return "f1_score"
    return None


def best_model_interpretation(best_f1):
    if best_f1 is None:
        return "Model quality could not be interpreted."
    if best_f1 >= 0.90:
        return "Excellent performance on the current synthetic dataset."
    if best_f1 >= 0.70:
        return "Good performance, but further improvement is possible."
    return "Model quality requires further improvement."


def build_short_conclusion(best_row, autoencoder_metrics):
    if best_row is None:
        return "The short conclusion could not be generated because the metrics summary is unavailable."

    conclusions = []
    best_model = str(best_row.get("model", "Unknown"))
    best_f1 = metric_value(best_row, "f1", "f1_score")
    best_recall = metric_value(best_row, "recall")
    best_precision = metric_value(best_row, "precision")
    best_roc_auc = metric_value(best_row, "roc_auc")

    conclusions.append(f"Best model by F1-score: {best_model}.")
    conclusions.append(
        "The best model achieved "
        f"F1-score = {format_number(best_f1)}, "
        f"precision = {format_number(best_precision)}, "
        f"recall = {format_number(best_recall)}, "
        f"ROC-AUC = {format_number(best_roc_auc)}."
    )
    conclusions.append(best_model_interpretation(best_f1))

    auto_f1 = None
    auto_recall = None
    if isinstance(autoencoder_metrics, dict):
        auto_f1 = autoencoder_metrics.get("f1_score", autoencoder_metrics.get("f1"))
        auto_recall = autoencoder_metrics.get("recall")

    if auto_f1 is not None:
        try:
            auto_f1 = float(auto_f1)
        except Exception:
            auto_f1 = None
    if auto_recall is not None:
        try:
            auto_recall = float(auto_recall)
        except Exception:
            auto_recall = None

    if auto_f1 is not None and auto_recall is not None and (auto_f1 < 0.7 or auto_recall < 0.5):
        conclusions.append("LSTM-Autoencoder requires further tuning, especially threshold selection and recall improvement.")

    if best_recall is not None and best_recall < 0.7:
        conclusions.append("Recall of the best overall model is still limited, so missed anomalies remain a relevant risk.")

    return " ".join(conclusions)


metrics_summary_path = results_dir / "metrics_summary.csv"
metrics_autoencoder_path = results_dir / "metrics_autoencoder.json"
anomaly_examples_path = results_dir / "anomaly_examples.csv"
thresholds_path = results_dir / "autoencoder_validation_thresholds.csv"


def collect_report_data():
    data = {}
    data["metrics_summary_df"] = read_csv(metrics_summary_path, "metrics_summary")
    data["autoencoder_payload"] = read_json(metrics_autoencoder_path, "autoencoder")
    data["anomaly_examples_df"] = read_csv(anomaly_examples_path, "anomaly_examples")
    data["thresholds_df"] = read_csv(thresholds_path, "thresholds")

    metrics_summary_df = data["metrics_summary_df"]
    best_row = None
    f1_column = choose_f1_column(metrics_summary_df)
    if metrics_summary_df is not None and metrics_summary_df.empty:
        add_warning("metrics_summary", f"No rows found in {metrics_summary_path}")
    if metrics_summary_df is not None and not metrics_summary_df.empty and f1_column is not None:
        best_row = metrics_summary_df.sort_values(f1_column, ascending=False).iloc[0]
    elif metrics_summary_df is not None and not metrics_summary_df.empty and f1_column is None:
        add_warning("metrics_summary", f"No F1 column found in {metrics_summary_path}")

    autoencoder_metrics = flatten_autoencoder_metrics(data["autoencoder_payload"])
    if data["autoencoder_payload"] is not None and not autoencoder_metrics:
        add_warning("autoencoder", f"No usable LSTM-Autoencoder metrics found in {metrics_autoencoder_path}")

    png_files = []
    if plots_dir.exists():
        try:
            png_files = sorted(path.name for path in plots_dir.glob("*.png"))
        except Exception as exc:
            add_warning("plots", f"Failed to list PNG files in {plots_dir}: {exc}")
            png_files = []
        if not png_files:
            add_warning("plots", f"No PNG files found in {plots_dir}")
    else:
        add_warning("plots", f"Directory not found: {plots_dir}")

    data["best_row"] = best_row
    data["autoencoder_metrics"] = autoencoder_metrics
    data["png_files"] = png_files
    return data


def render_plain_report(data):
    metrics_summary_df = data["metrics_summary_df"]
    best_row = data["best_row"]
    autoencoder_metrics = data["autoencoder_metrics"]
    thresholds_df = data["thresholds_df"]
    anomaly_examples_df = data["anomaly_examples_df"]
    png_files = data["png_files"]

    print_header("FINAL PIPELINE REPORT")

    print_header("1. MODEL METRICS SUMMARY")
    for message in get_warnings("metrics_summary"):
        print_warning(message)
    if metrics_summary_df is not None and not metrics_summary_df.empty:
        print(rounded_df(metrics_summary_df).to_string(index=False))

    print_header("2. BEST MODEL")
    if best_row is None:
        print("Best model could not be determined.")
    else:
        best_model_name = best_row.get("model", "Unknown")
        print(f"Best model by F1-score: {best_model_name}")
        print(f"precision: {format_number(metric_value(best_row, 'precision'))}")
        print(f"recall: {format_number(metric_value(best_row, 'recall'))}")
        print(f"f1: {format_number(metric_value(best_row, 'f1', 'f1_score'))}")
        print(f"roc_auc: {format_number(metric_value(best_row, 'roc_auc'))}")
        print(f"pr_auc: {format_number(metric_value(best_row, 'pr_auc'))}")
        print(best_model_interpretation(metric_value(best_row, "f1", "f1_score")))

    print_header("3. LSTM-AUTOENCODER DETAILS")
    for message in get_warnings("autoencoder"):
        print_warning(message)
    if autoencoder_metrics:
        metrics_to_print = dict(autoencoder_metrics)
        metrics_to_print.setdefault("model", "LSTM-Autoencoder")
        print_json_metrics_table(
            "Current LSTM-Autoencoder metrics:",
            metrics_to_print,
            ["model", "precision", "recall", "f1", "f1_score", "roc_auc", "pr_auc", "threshold_percentile"],
        )
    else:
        print("LSTM-Autoencoder details are unavailable.")

    print_header("AUTOENCODER THRESHOLD SEARCH")
    for message in get_warnings("thresholds"):
        print_warning(message)
    if thresholds_df is not None and not thresholds_df.empty:
        print(rounded_df(thresholds_df.head(10)).to_string(index=False))

    print_header("4. FIRST 10 ANOMALY EXAMPLES")
    for message in get_warnings("anomaly_examples"):
        print_warning(message)
    if anomaly_examples_df is not None and not anomaly_examples_df.empty:
        preferred_columns = [
            "user_id",
            "timestamp",
            "anomaly_type",
            "probable_reason",
            "bytes_sent",
            "source_ip",
            "event_type",
        ]
        available_columns = [column for column in preferred_columns if column in anomaly_examples_df.columns]
        if not available_columns:
            available_columns = list(anomaly_examples_df.columns)
        anomaly_head = anomaly_examples_df[available_columns].head(10).copy()
        if "bytes_sent" in anomaly_head.columns:
            anomaly_head["bytes_sent"] = pd.to_numeric(anomaly_head["bytes_sent"], errors="coerce").round(2)
        print(rounded_df(anomaly_head, column_digits={"bytes_sent": 2}).to_string(index=False))

    print_header("5. GENERATED FILES")
    generated_targets = [
        results_dir / "metrics_summary.csv",
        results_dir / "metrics_baseline.json",
        results_dir / "metrics_autoencoder.json",
        results_dir / "anomaly_examples.csv",
        results_dir / "autoencoder_validation_thresholds.csv",
    ]
    for path in generated_targets:
        status = "OK" if path.exists() else "missing"
        size = human_size(path.stat().st_size) if path.exists() and path.is_file() else "n/a"
        print(f"- [{status}] {path.relative_to(project_root)} ({size})")
    plots_status = "OK" if plots_dir.exists() else "missing"
    plots_size = human_size(directory_size(plots_dir)) if plots_dir.exists() else "n/a"
    print(f"- [{plots_status}] results/plots/ ({plots_size})")

    print_header("PLOTS")
    for message in get_warnings("plots"):
        print_warning(message)
    for name in png_files:
        print(f"- {name}")

    print_header("6. SHORT CONCLUSION FOR REPORT")
    print(build_short_conclusion(best_row, autoencoder_metrics))


def render_rich_warning(console, message):
    console.print(Text(f"[!] {message}", style="yellow"))


def rich_table_from_dataframe(title, df, digits=4, column_digits=None):
    table = Table(title=title, header_style="bold cyan")
    for column in df.columns:
        table.add_column(str(column))
    formatted_df = rounded_df(df, default_digits=digits, column_digits=column_digits)
    for _, row in formatted_df.iterrows():
        rendered_row = []
        for value in row.tolist():
            if pd.isna(value):
                rendered_row.append("n/a")
            else:
                rendered_row.append(str(value))
        table.add_row(*rendered_row)
    return table


def render_rich_report(data):
    console = Console()
    metrics_summary_df = data["metrics_summary_df"]
    best_row = data["best_row"]
    autoencoder_metrics = data["autoencoder_metrics"]
    thresholds_df = data["thresholds_df"]
    anomaly_examples_df = data["anomaly_examples_df"]
    png_files = data["png_files"]

    console.print(
        Panel(
            "[bold cyan]FINAL PIPELINE REPORT[/bold cyan]\nCybersecurity Anomaly Detection Research Pipeline",
            border_style="bright_blue",
            expand=False,
        )
    )

    console.rule("[bold cyan]MODEL METRICS SUMMARY[/bold cyan]")
    for message in get_warnings("metrics_summary"):
        render_rich_warning(console, message)
    if metrics_summary_df is not None and not metrics_summary_df.empty:
        metrics_table = Table(header_style="bold cyan")
        metrics_table.add_column("Best", justify="center")
        metrics_table.add_column("Model")
        metrics_table.add_column("Precision", justify="right")
        metrics_table.add_column("Recall", justify="right")
        metrics_table.add_column("F1", justify="right")
        metrics_table.add_column("ROC-AUC", justify="right")
        metrics_table.add_column("PR-AUC", justify="right")

        best_index = best_row.name if best_row is not None else None
        for index, row in metrics_summary_df.iterrows():
            is_best = best_index is not None and index == best_index
            metrics_table.add_row(
                "Yes" if is_best else "",
                str(row.get("model", "Unknown")),
                format_number(metric_value(row, "precision")),
                format_number(metric_value(row, "recall")),
                format_number(metric_value(row, "f1", "f1_score")),
                format_number(metric_value(row, "roc_auc")),
                format_number(metric_value(row, "pr_auc")),
                style="bold green" if is_best else "",
            )
        console.print(metrics_table)

    console.rule("[bold cyan]BEST MODEL[/bold cyan]")
    if best_row is None:
        render_rich_warning(console, "Best model could not be determined.")
    else:
        best_f1 = metric_value(best_row, "f1", "f1_score")
        best_table = Table.grid(padding=(0, 2))
        best_table.add_column(style="bold cyan")
        best_table.add_column()
        best_table.add_row("Model", str(best_row.get("model", "Unknown")))
        best_table.add_row("Precision", format_number(metric_value(best_row, "precision")))
        best_table.add_row("Recall", format_number(metric_value(best_row, "recall")))
        best_table.add_row("F1", format_number(best_f1))
        best_table.add_row("ROC-AUC", format_number(metric_value(best_row, "roc_auc")))
        if "pr_auc" in best_row.index:
            best_table.add_row("PR-AUC", format_number(metric_value(best_row, "pr_auc")))
        best_table.add_row("Interpretation", best_model_interpretation(best_f1))
        console.print(Panel(best_table, border_style="green", expand=False))

    console.rule("[bold cyan]LSTM-AUTOENCODER DETAILS[/bold cyan]")
    for message in get_warnings("autoencoder"):
        render_rich_warning(console, message)
    if autoencoder_metrics:
        auto_table = Table(header_style="bold cyan")
        auto_table.add_column("Metric")
        auto_table.add_column("Value")
        auto_rows = []
        auto_rows.append(("model", autoencoder_metrics.get("model", "LSTM-Autoencoder")))
        for key in ["precision", "recall", "roc_auc", "pr_auc", "threshold_percentile"]:
            if key in autoencoder_metrics:
                auto_rows.append((key, autoencoder_metrics[key]))
        if "f1" in autoencoder_metrics:
            auto_rows.append(("f1", autoencoder_metrics["f1"]))
        elif "f1_score" in autoencoder_metrics:
            auto_rows.append(("f1", autoencoder_metrics["f1_score"]))
        ordered_keys = {"model", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold_percentile"}
        for key, value in auto_rows:
            if key in ordered_keys:
                rendered_value = str(value) if key == "model" else format_number(value)
                auto_table.add_row(key, rendered_value)
        console.print(auto_table)
    else:
        render_rich_warning(console, "LSTM-Autoencoder details are unavailable.")

    console.rule("[bold cyan]AUTOENCODER THRESHOLD SEARCH[/bold cyan]")
    for message in get_warnings("thresholds"):
        render_rich_warning(console, message)
    if thresholds_df is not None and not thresholds_df.empty:
        console.print(rich_table_from_dataframe("First 10 validation threshold rows", thresholds_df.head(10)))

    console.rule("[bold cyan]FIRST 10 ANOMALY EXAMPLES[/bold cyan]")
    for message in get_warnings("anomaly_examples"):
        render_rich_warning(console, message)
    if anomaly_examples_df is not None and not anomaly_examples_df.empty:
        preferred_columns = [
            "user_id",
            "timestamp",
            "anomaly_type",
            "probable_reason",
            "bytes_sent",
            "source_ip",
            "event_type",
        ]
        available_columns = [column for column in preferred_columns if column in anomaly_examples_df.columns]
        if not available_columns:
            available_columns = list(anomaly_examples_df.columns)
        anomaly_head = anomaly_examples_df[available_columns].head(10).copy()
        if "bytes_sent" in anomaly_head.columns:
            anomaly_head["bytes_sent"] = pd.to_numeric(anomaly_head["bytes_sent"], errors="coerce").round(2)
        console.print(
            rich_table_from_dataframe(
                "First 10 anomaly examples",
                anomaly_head,
                digits=4,
                column_digits={"bytes_sent": 2},
            )
        )

    console.rule("[bold cyan]GENERATED FILES[/bold cyan]")
    files_table = Table(header_style="bold cyan")
    files_table.add_column("Status")
    files_table.add_column("File")
    files_table.add_column("Size", justify="right")
    generated_targets = [
        results_dir / "metrics_summary.csv",
        results_dir / "metrics_baseline.json",
        results_dir / "metrics_autoencoder.json",
        results_dir / "anomaly_examples.csv",
        results_dir / "autoencoder_validation_thresholds.csv",
    ]
    for path in generated_targets:
        exists = path.exists()
        status = Text("OK", style="green") if exists else Text("missing", style="yellow")
        size = human_size(path.stat().st_size) if exists and path.is_file() else "n/a"
        files_table.add_row(status, str(path.relative_to(project_root)), size)
    plots_exists = plots_dir.exists()
    plots_status = Text("OK", style="green") if plots_exists else Text("missing", style="yellow")
    plots_size = human_size(directory_size(plots_dir)) if plots_exists else "n/a"
    if plots_exists:
        plots_size = f"{plots_size} ({len(png_files)} PNG)"
    files_table.add_row(plots_status, "results/plots/", plots_size)
    console.print(files_table)

    console.rule("[bold cyan]PLOTS[/bold cyan]")
    for message in get_warnings("plots"):
        render_rich_warning(console, message)
    if png_files:
        plots_table = Table(title="PNG files", header_style="bold cyan")
        plots_table.add_column("File")
        for name in png_files:
            plots_table.add_row(name)
        console.print(Panel(plots_table, border_style="blue"))

    console.rule("[bold cyan]SHORT CONCLUSION FOR REPORT[/bold cyan]")
    console.print(Panel(build_short_conclusion(best_row, autoencoder_metrics), border_style="magenta"))


report_data = collect_report_data()

if HAVE_RICH:
    render_rich_report(report_data)
else:
    print("rich is not installed. Install it with: pip install rich")
    render_plain_report(report_data)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --best-autoencoder)
      BEST_AUTOENCODER=1
      shift
      ;;
    --with-stage10)
      WITH_STAGE10=1
      shift
      ;;
    --stage10-epochs)
      STAGE10_EPOCHS="${2:-}"
      if [[ -z "$STAGE10_EPOCHS" ]]; then
        echo "Error: --stage10-epochs requires a value." >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

log_step "Project root: $PROJECT_ROOT"
log_step "Python: $PYTHON_BIN"

if [[ "$SKIP_GENERATE" -eq 0 ]]; then
  log_step "Generating synthetic data"
  "$PYTHON_BIN" src/generate_data.py
else
  log_step "Skipping synthetic data generation"
fi

log_step "Preprocessing data"
"$PYTHON_BIN" src/preprocessing.py

log_step "Training baseline models"
"$PYTHON_BIN" src/train_baseline.py

if [[ "$BEST_AUTOENCODER" -eq 1 ]]; then
  log_step "Training best practical autoencoder config"
  run_best_autoencoder
else
  log_step "Training default autoencoder config"
  "$PYTHON_BIN" src/train_lstm_autoencoder.py
fi

log_step "Building metrics summary"
"$PYTHON_BIN" src/evaluate.py

log_step "Generating anomaly explanations"
"$PYTHON_BIN" src/explain_anomalies.py

if [[ "$WITH_STAGE10" -eq 1 ]]; then
  log_step "Running stage 10 experiments"
  "$PYTHON_BIN" src/run_autoencoder_stage10.py --experiment-6-epochs "$STAGE10_EPOCHS"

  log_step "Restoring best practical autoencoder as final current result"
  run_best_autoencoder

  log_step "Refreshing metrics summary after best practical autoencoder"
  "$PYTHON_BIN" src/evaluate.py
fi

log_step "Pipeline finished"
print_final_report
