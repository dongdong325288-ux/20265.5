# Formula Audit Notes

## Overall verdict

The paper's core optimization story matches the code:

- generation: choose the cheapest path that satisfies CLIP and semantic-coverage margins
- understanding: choose the cheapest path that preserves correctness

The implementation is faithful in the main regime used by the paper.

## Match 1: generation feasibility margins

Paper:
- `delta_clip = 0.004`
- `delta_sem = 0.0015`

Code:
- `GEN_CLIP_MARGIN = 0.004`
- `GEN_SEM_MARGIN = 0.0015`
- enforced inside `quality_constrained_fastest`

Status:
- exact match

## Match 2: generation selection rule

Paper:
- select the fastest path in the feasible generation set

Code:
- `return min(pool, key=lambda r: float(r["elapsed_s"]))`

Status:
- exact match for the feasible set

## Match 3: understanding selection rule

Paper:
- keep only correctness-preserving budgets and choose the cheapest one

Code:
- `quality_constrained_fastest(rows, [("correct", 0.0)])`

Status:
- matches the paper on the current benchmark, because the best per-question correctness is 1

## Important discrepancy A: fallback behavior exists in code

Paper wording:
- generation explicitly mentions a fallback to the globally fastest path when no candidate survives
- understanding is written as if only correctness-preserving paths are ever considered

Code behavior:
- `quality_constrained_fastest` falls back to `rows` whenever the feasible set is empty
- this applies to both generation and understanding

Implication:
- the code is slightly more permissive than the idealized understanding equation
- on the current data this does not change any result, because all oracle-selected understanding cases are correct

Recommended fix:
- either add one sentence in the understanding subsection that the same fallback rule is used if no budget preserves correctness,
- or split the helper into generation-specific and understanding-specific variants.

## Important discrepancy B: baseline is described as part of the generation candidate set, but not selected over in code

Paper wording:
- the generation candidate set is described as containing six paths, including the 50-step baseline

Code behavior:
- `gen_candidates = [r for r in gen_rows if r["case"] != "baseline_50_cfg04"]`
- baseline is used as reference and for metrics, but is excluded from route selection

Implication:
- the paper slightly overstates the candidate set used by the actual planner

Recommended fix:
- revise the text to say the experiment evaluates six total generation paths, while the planner selects among five accelerated candidates using the baseline as reference.

## Recommendation

The formulas are substantively correct.
The two issues above are wording / contract issues, not evidence that the method is implemented incorrectly.
