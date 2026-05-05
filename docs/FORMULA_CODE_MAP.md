# Formula to Code Map

## 1. Minimum-Cost Feasible Generation Path

Implemented in `scripts/idea6_uqp_bagel_eval.py`.

Relevant objects:
- `GEN_CASES`
- `GEN_CLIP_MARGIN`
- `GEN_SEM_MARGIN`
- `quality_constrained_fastest(rows, [("clip_score", GEN_CLIP_MARGIN), ("semantic_coverage", GEN_SEM_MARGIN)])`

## 2. Minimum-Cost Feasible Understanding Path

Implemented in `scripts/idea6_uqp_bagel_eval.py` and `scripts/idea5_scope_flashu_eval.py`.

Relevant objects:
- `UNDERSTANDING_CASES`
- `UND_EXTRA_CASES`
- `run_understanding_benchmark`
- `quality_constrained_fastest(rows, [("correct", 0.0)])`

## 3. Prior Deployable Baseline

Implemented in `scripts/idea5_scope_flashu_eval.py`.

Routing functions:
- `choose_generation_case`
- `choose_generation_oracle`
- `choose_understanding_case`
- `build_understanding_policy`

## 4. Figure Production

- Figure 2: `scripts/make_tradeoff_figure.py`, `scripts/make_paper_figures_v2.py`
- Figure 4: `scripts/make_paper_figures_v2.py`, `scripts/figure4_make_sixth_prompt.py`
