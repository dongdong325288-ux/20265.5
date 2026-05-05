# Formula Walkthrough

This note walks through the core equations in `paper/neurips_2026.tex` and explains how each one is realized in code.

## 1. Unified objective

Paper:
- generation: `pi_g*(x_g) = argmin C(pi)` subject to generation quality feasibility
- understanding: `pi_u*(x_u) = argmin C(pi)` subject to correctness feasibility

Code:
- both tasks use the same helper: `quality_constrained_fastest(rows, quality_specs)`
- the helper computes a feasible subset and then returns the row with minimum `elapsed_s`

Where to look:
- `scripts/idea6_uqp_bagel_eval.py`
- `quality_constrained_fastest(...)`

Interpretation:
- `C(pi)` is implemented operationally as `elapsed_s`
- feasibility is implemented as a task-specific filter, not as a soft score added to the objective

## 2. Generation candidate family

Paper:
- six generation paths are evaluated in total
- the planner selects among the five accelerated candidates while the 50-step baseline is kept as a reference

Code:
- baseline, `fast_15_cfg02`, `fast_20_cfg04`, and `flashu_cache15` come from `idea5.build_generation_table(...)`
- `uqp_sched16` and `uqp_sched18` are added in `GEN_CASES`
- actual planner candidate set excludes `baseline_50_cfg04`

Where to look:
- `GEN_CASES`
- `build_generation_table(...)` in `idea5_scope_flashu_eval.py`
- `gen_candidates = [r for r in gen_rows if r["case"] != "baseline_50_cfg04"]`

Interpretation:
- the baseline is part of the evaluated set but not part of the planner's accelerated search space

## 3. Generation feasibility set

Paper:
- keep paths whose CLIP is within `delta_clip = 0.004` of the best path for the prompt
- keep paths whose semantic coverage is within `delta_sem = 0.0015` of the best path for the prompt

Code:
- `GEN_CLIP_MARGIN = 0.004`
- `GEN_SEM_MARGIN = 0.0015`
- `quality_constrained_fastest(rows, [("clip_score", GEN_CLIP_MARGIN), ("semantic_coverage", GEN_SEM_MARGIN)])`

Where to look:
- `scripts/idea6_uqp_bagel_eval.py`

Interpretation:
- the implementation is exact with respect to the paper's current margins
- the reference value for each metric is the best metric value observed among candidates for that prompt

## 4. Generation minimization step

Paper:
- choose the fastest path inside the feasible generation set

Code:
- after feasibility filtering, the helper returns
  - `min(pool, key=lambda r: float(r["elapsed_s"]))`

Interpretation:
- this is the concrete realization of `argmin C(pi)`
- no secondary optimization term is used after feasibility filtering

## 5. Generation fallback rule

Paper:
- if no candidate survives, fall back to the globally fastest path

Code:
- `pool = feasible if feasible else rows`

Interpretation:
- this fallback is shared by the helper and therefore applies symmetrically to generation and understanding
- on the released prompt suite, this fallback is not the dominant regime

## 6. Understanding candidate family

Paper:
- five understanding budgets: `und_full`, `und_med`, `und_small`, `und_tiny`, `und_micro`

Code:
- the first three come from `idea5_scope_flashu_eval.py`
- `und_tiny` and `und_micro` are added in `UND_EXTRA_CASES`
- the dictionary is extended by `idea5.UNDERSTANDING_CASES.update(UND_EXTRA_CASES)`

Interpretation:
- the understanding planner is not selecting among schedules; it is selecting among progressively smaller visual budgets

## 7. Understanding feasibility set

Paper:
- keep only budgets with `Q_u(pi; x_u) = 1`
- here `Q_u` is exact-match correctness on the current question

Code:
- `quality_constrained_fastest(rows, [("correct", 0.0)])`

Interpretation:
- because the helper interprets margins as `best - margin`, and the best `correct` value is `1.0`, the feasible set becomes exactly the set of rows with `correct = 1`
- this is a clean implementation of the paper's equation

## 8. Understanding minimization step

Paper:
- among correctness-preserving budgets, choose the cheapest one

Code:
- same helper as generation, minimizing `elapsed_s`

Interpretation:
- this is the planner-level unification claim of the paper: task-specific feasibility test, shared minimum-cost selection rule

## 9. Prior deployable baseline

Paper:
- `SCOPE-FlashU rule` is the prior deployable baseline

Code:
- generation-side routing is in `choose_generation_case(...)`
- understanding-side routing is in `choose_understanding_case(...)` and `build_understanding_policy(...)`

Interpretation:
- this baseline is task-aware but not quality-constrained in the same planner sense as UQP-Bagel
- that distinction is central to the paper's empirical comparison

## 10. What is exact, and what is operational

Exact matches between paper and code:
- generation margins
- cheapest-feasible-path selection rule
- understanding correctness filter
- fallback behavior after wording fix in the paper

Operational choices that the paper abstracts over:
- `C(pi)` is concretely wall-clock elapsed time, not a symbolic complexity term
- generation quality is implemented through released proxies `clip_score` and `semantic_coverage`
- understanding quality is implemented as exact-match correctness on the released QA set

## 11. Practical reading advice

If you are checking the paper against code, the most direct file order is:
1. `paper/neurips_2026.tex`
2. `docs/FORMULA_CODE_MAP.md`
3. `docs/FORMULA_AUDIT_NOTES.md`
4. `scripts/idea6_uqp_bagel_eval.py`
5. `results/data/generation_oracle_rows.csv`
6. `results/data/understanding_oracle_rows.csv`
