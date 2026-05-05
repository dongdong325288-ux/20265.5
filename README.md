# UQP-Bagel Paper Release

This repository is a cleaned reproducibility package for the paper:

`UQP-Bagel: Task-Aware Quality-Constrained Planning for Training-Free Acceleration of Native Unified Models`

The goal of this release is simple: let a reader verify that the paper's equations, tables, figures, and main claims are backed by concrete code and CSV files.

## What This Repository Lets You Check

This release is organized to make four questions easy to answer:

1. Which code produced the numbers reported in the paper?
2. Which CSV files back each table and figure?
3. Do the paper equations match the actual selection logic in code?
4. Can the figure-generation and paper-number checks be rerun from this release package?
5. Which dependency set is sufficient for lightweight audit versus full BAGEL reruns?

## Start Here

If you are reviewing the paper or checking formulas, this is the fastest path:

1. Read `docs/FORMULA_CODE_MAP.md`
2. Read `docs/FORMULA_AUDIT_NOTES.md`
3. Read `docs/TABLE_FIGURE_AUDIT.md`
4. Read `docs/EXPERIMENT_PROTOCOL.md`
5. Read `docs/FORMULA_WALKTHROUGH.md`
6. Inspect `paper/neurips_2026.tex`
7. Inspect `scripts/idea6_uqp_bagel_eval.py`
8. Run `python scripts/check_paper_numbers.py` or `scripts/reproduce_release_checks.ps1`

## Repository Layout

- `paper/`
  - NeurIPS paper source (`neurips_2026.tex`, style file, checklist)
- `scripts/`
  - main evaluation scripts for `idea5` and `idea6`
  - figure-generation scripts
  - paper-number consistency checker
  - Linux rerun helper script
- `results/data/`
  - aggregate CSVs used directly in the paper
  - row-level detail CSVs for generation and understanding
  - oracle-row CSVs for per-instance path choices
- `results/figures/`
  - figure assets used by the current draft
  - qualitative images and held-out prompt result
- `docs/`
  - formula/code mapping
  - formula walkthrough
  - formula audit
  - table/figure audit
  - experiment protocol

## Key Files

### Paper
- `paper/neurips_2026.tex`

### Main code
- `scripts/idea6_uqp_bagel_eval.py`
- `scripts/idea5_scope_flashu_eval.py`

### Formula and audit notes
- `docs/FORMULA_CODE_MAP.md`
- `docs/FORMULA_WALKTHROUGH.md`
- `docs/FORMULA_AUDIT_NOTES.md`
- `docs/TABLE_FIGURE_AUDIT.md`

### Main result CSVs
- `results/data/generation_oracle_overall.csv`
- `results/data/understanding_oracle_overall.csv`
- `results/data/compare_to_idea5.csv`

### Detailed result CSVs
- `results/data/generation_detail.csv`
- `results/data/generation_oracle_rows.csv`
- `results/data/understanding_detail.csv`
- `results/data/understanding_oracle_rows.csv`

## What You Can Reproduce Directly From This Release

These checks work directly from this repository:

- paper-number consistency check
- regeneration of Figure 2 assets
- regeneration of Figure 4 mosaic from released image tiles
- inspection of the exact CSV rows backing the reported results

Example:

```bash
python scripts/check_paper_numbers.py
python scripts/make_tradeoff_figure.py
python scripts/make_paper_figures_v2.py
```

Or use the helper scripts:

- Windows PowerShell: `scripts/reproduce_release_checks.ps1`
- Linux/macOS shell: `scripts/reproduce_release_checks.sh`

## What Still Depends on the Original BAGEL Environment

This repository is a paper release, not a full standalone fork of BAGEL.
Full GPU reruns of generation and understanding still depend on:

- the original BAGEL codebase
- BAGEL model weights
- the runtime environment used for the archived experiments

See `docs/EXPERIMENT_PROTOCOL.md` and `scripts/run_release_linux.sh` for the expected environment variables and rerun flow.

## Dependency Files

- `requirements-release.txt`: minimal dependencies for audit, figure regeneration, and paper-number checking
- `environment-release.yml`: conda environment for the lightweight release checks
- `requirements-full-rerun.txt`: stronger dependency set for BAGEL-side reruns on top of the original BAGEL environment

## Full Rerun Prerequisites

```bash
export BAGEL_PROJECT_ROOT=/path/to/Bagel-main
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export OUT_ROOT=/path/to/output_dir
export QUALITY_DEVICE=cuda
```

## Notes on Scope

This release is designed for auditability, not for claiming that every BAGEL experiment can be rerun from scratch on a clean machine with no external dependencies. The main promise is narrower and more concrete: a reader should be able to follow the path from paper statement -> formula -> code -> CSV -> figure asset with minimal friction.
