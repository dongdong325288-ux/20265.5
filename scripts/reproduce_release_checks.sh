#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Running paper-number check"
python scripts/check_paper_numbers.py

echo "[INFO] Regenerating Figure 2 assets"
python scripts/make_tradeoff_figure.py

echo "[INFO] Regenerating paper figure assets"
python scripts/make_paper_figures_v2.py

echo "[INFO] Release checks completed"
