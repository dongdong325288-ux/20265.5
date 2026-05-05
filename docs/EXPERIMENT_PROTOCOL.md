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
