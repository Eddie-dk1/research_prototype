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
echo "Key outputs:"
echo "  results/metrics_summary.csv"
echo "  results/metrics_baseline.json"
echo "  results/metrics_autoencoder.json"
echo "  results/anomaly_examples.csv"
