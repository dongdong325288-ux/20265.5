$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "[INFO] Running paper-number check"
python scripts/check_paper_numbers.py

Write-Host "[INFO] Regenerating Figure 2 assets"
python scripts/make_tradeoff_figure.py

Write-Host "[INFO] Regenerating paper figure assets"
python scripts/make_paper_figures_v2.py

Write-Host "[INFO] Release checks completed"
