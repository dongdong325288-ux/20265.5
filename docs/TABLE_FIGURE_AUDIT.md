# Table and Figure Audit

## Scope

This audit checks whether the paper tables and figures are backed by the released CSV files and scripts in the current reproducibility package.

## Verdict

- Table 1 is consistent with the released aggregate CSV files.
- Table 2 is consistent with the released oracle-by-case and case-summary CSV files, up to paper-side rounding.
- Figure 2 is backed by the released CSV files and both figure-generation scripts now run successfully inside the release repository after a path fix.
- Figure 4 is backed by the released qualitative image assets, including the held-out `tea_room` prompt.
- A real reproducibility issue existed in the first release: two figure scripts still expected the old `scripts/data` and `scripts/figures` layout. This has now been fixed so they read from `results/data` and write to `results/figures`.

## Table 1 Audit

Paper values:
- Generation UQP-Bagel: `108.02s`, `1.24x`, `0.37`, `0.26`, `0.54`
- Understanding UQP-Bagel: `0.22s`, `2.37x`, `1.00`
- Prior deployable baseline: `112.52s`, `1.19x`, `0.36` and `0.29s`, `1.94x`, `1.00`

Backing files:
- `results/data/generation_oracle_overall.csv`
- `results/data/understanding_oracle_overall.csv`
- `results/data/compare_to_idea5.csv`

Released numbers:
- Generation oracle: `108.0236`, `1.2386`, `0.366044`, `0.262747`, `0.539188`
- Understanding oracle: `0.2221`, `2.3697`, `1.0`
- Prior deployable baseline: `112.5231`, `1.1912`, `0.361916`; `0.2941`, `1.9436`, `1.0`

Status:
- exact match up to the two-decimal display style used in the paper.

## Table 2 Audit

Paper values:
- `fast_15_cfg02`: `#=1`, `106.15s`, `1.24x`, `0.31`, `0.24`
- `uqp_sched16`: `#=2`, `107.47s`, `1.22x`, `0.39`, `0.27`
- `uqp_sched18`: `#=2`, `109.52s`, `1.26x`, `0.37`, `0.27`
- `und_full`: `0.52s`, `1.00x`, `1.00`
- `und_med`: `0.33s`, `1.56x`, `1.00`
- `und_small`: `0.29s`, `1.83x`, `1.00`
- `und_tiny`: `0.26s`, `2.05x`, `1.00`
- `und_micro`: `0.22s`, `2.37x`, `1.00`

Backing files:
- `results/data/generation_oracle_by_case.csv`
- `results/data/understanding_case_summary.csv`

Released numbers:
- `fast_15_cfg02`: `1`, `106.1515`, `1.2415`, `0.313631`, `0.238651`
- `uqp_sched16`: `2`, `107.4680`, `1.2182`, `0.391383`, `0.269685`
- `uqp_sched18`: `2`, `109.5153`, `1.2575`, `0.366911`, `0.267856`
- `und_full`: `0.5191`, `1.0000`, `1.0`
- `und_med`: `0.3332`, `1.5626`, `1.0`
- `und_small`: `0.2856`, `1.8289`, `1.0`
- `und_tiny`: `0.2553`, `2.0496`, `1.0`
- `und_micro`: `0.2221`, `2.3697`, `1.0`

Status:
- exact match up to rounding.

## Figure 2 Audit

Referenced paper asset:
- `results/figures/tradeoff_summary_v2.pdf`

Generation-side inputs:
- `results/data/generation_case_summary.csv`
- `results/data/compare_to_idea5.csv`

Understanding-side inputs:
- `results/data/understanding_case_summary.csv`
- `results/data/compare_to_idea5.csv`

Backing scripts:
- `scripts/make_tradeoff_figure.py`
- `scripts/make_paper_figures_v2.py`

Reproducibility note:
- Both scripts originally expected the old `scripts/data` and `scripts/figures` layout.
- They have been patched to use the release-repo layout:
  - read from `results/data`
  - write to `results/figures`
- After the fix, both scripts ran successfully.

Status:
- figure logic matches the released CSV files.
- the release package is now self-consistent for Figure 2 regeneration.

## Figure 4 Audit

Referenced paper asset:
- `results/figures/generation_oracle_mosaic.png`

Image tiles used:
- `results/figures/fairy_cosplayer__uqp_sched18.png`
- `results/figures/cyberpunk_alley__uqp_sched16.png`
- `results/figures/snow_cabin__uqp_sched18.png`
- `results/figures/product_camera__fast_15_cfg02.png`
- `results/figures/corgi_ocean__uqp_sched16.png`
- `results/figures/tea_room__uqp_sched18.png`

Supporting selection records:
- `results/data/generation_oracle_rows.csv`
- `results/data/tea_room_summary.json`

Status:
- the first five images come from the oracle-selected five-prompt suite.
- the sixth image is a held-out qualitative prompt and is explicitly described that way in the paper caption.
- this is consistent with the current narrative: five benchmark prompts plus one held-out qualitative example.

## Remaining caveat

The release package is now consistent for paper-number checking and figure regeneration. However, the full GPU rerun of generation and understanding still depends on the original BAGEL codebase, model weights, and remote execution environment, as already stated in the repository README.
