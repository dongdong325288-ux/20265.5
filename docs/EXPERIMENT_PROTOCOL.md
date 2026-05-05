# Experiment Protocol

## Goal

The paper studies whether native unified acceleration on BAGEL should be controlled as structure-first shortcut activation or minimum-cost feasible-path selection.

## Generation Evaluation

Prompt families:
- `fairy_cosplayer`
- `cyberpunk_alley`
- `snow_cabin`
- `product_camera`
- `corgi_ocean`

Held-out qualitative prompt:
- `tea_room`

Metrics:
- elapsed time
- speedup vs. 50-step baseline
- CLIP score
- semantic coverage
- relation coverage
- style coverage
- LPIPS to baseline

## Understanding Evaluation

Metrics:
- elapsed time
- speedup vs. `und_full`
- exact-match correctness / accuracy

Candidate budgets:
- `und_full`
- `und_med`
- `und_small`
- `und_tiny`
- `und_micro`

## Rerun Flow

```bash
export BAGEL_PROJECT_ROOT=/path/to/Bagel-main
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export OUT_ROOT=/path/to/output_dir
export QUALITY_DEVICE=cuda

python scripts/idea6_uqp_bagel_eval.py
python scripts/make_tradeoff_figure.py
python scripts/make_paper_figures_v2.py
```

## Release-Repo Notes

- `scripts/make_tradeoff_figure.py` reads from `results/data/` and writes to `results/figures/`.
- `scripts/make_paper_figures_v2.py` reads from `results/data/` and `results/figures/`, then rewrites:
  - `results/figures/tradeoff_summary_v2.{png,pdf}`
  - `results/figures/generation_oracle_mosaic.png`
  - `results/figures/understanding_source_grid.png`
- `scripts/figure4_make_sixth_prompt.py` is a remote rerun helper for the held-out qualitative prompt and assumes access to the original BAGEL environment plus model weights.


## Dependency Presets

- `requirements-release.txt` and `environment-release.yml` are enough for release-level auditing and figure regeneration.
- `requirements-full-rerun.txt` is closer to what the archived scripts need, but full reruns still assume access to the original BAGEL codebase and model weights.
- For Windows users who only want to verify numbers and regenerate paper assets, `scripts/reproduce_release_checks.ps1` is the shortest path.
