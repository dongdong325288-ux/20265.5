#!/usr/bin/env bash
set -euo pipefail

export BAGEL_PROJECT_ROOT="${BAGEL_PROJECT_ROOT:-/path/to/Bagel-main}"
export MODEL_PATH="${MODEL_PATH:-/path/to/BAGEL-7B-MoT}"
export OUT_ROOT="${OUT_ROOT:-$PWD/_paper_rerun}"
export QUALITY_DEVICE="${QUALITY_DEVICE:-cuda}"

echo "[INFO] BAGEL_PROJECT_ROOT=$BAGEL_PROJECT_ROOT"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] QUALITY_DEVICE=$QUALITY_DEVICE"

python scripts/idea6_uqp_bagel_eval.py
python scripts/make_tradeoff_figure.py
python scripts/make_paper_figures_v2.py
