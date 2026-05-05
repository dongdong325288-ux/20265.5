# UQP-Bagel Paper Release

This repository is a cleaned reproducibility package for the paper:

`UQP-Bagel: Task-Aware Quality-Constrained Planning for Training-Free Acceleration of Native Unified Models`

It is organized to make three things easy to audit:

1. Which code produced the numbers reported in the paper.
2. Which CSV files back each table and figure.
3. How the paper equations map to the actual selection logic in code.

## What Is Included

- `paper/`
- `scripts/`
- `results/data/`
- `results/figures/`
- `docs/`

## Repository Scope

This package is a paper release, not a full standalone fork of BAGEL.
The archived scripts still depend on the original BAGEL codebase and model weights for full GPU reruns.

## Quick Audit Guide

1. Read `docs/FORMULA_CODE_MAP.md`
2. Read `docs/EXPERIMENT_PROTOCOL.md`
3. Inspect `scripts/idea6_uqp_bagel_eval.py` and `scripts/idea5_scope_flashu_eval.py`
4. Check `results/data/`
5. Run `python scripts/check_paper_numbers.py`

## Included Result Granularity

This release now contains:
- aggregate CSVs used directly in the paper
- row-level detail CSVs for generation and understanding
- oracle-row CSVs used to analyze per-instance path choices
- figure assets used in the current draft

## Full Rerun Prerequisites

```bash
export BAGEL_PROJECT_ROOT=/path/to/Bagel-main
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export OUT_ROOT=/path/to/output_dir
export QUALITY_DEVICE=cuda
```

See `docs/EXPERIMENT_PROTOCOL.md` and `scripts/run_release_linux.sh`.
