#!/usr/bin/env python3
"""
Same-trait detection comparison: Assistant Axis projection vs. persona-vector
projection, both correlated against each condition's own trait judge score,
across all three elicitation conditions (explicit, implicit_described,
implicit_contextual). Including explicit lets us check whether the relative
gap between the general Assistant Axis and the trait-specific vector changes
as elicitation moves from explicit to implicit - not just how the Axis does
on implicit prompts in isolation.

NOTE on scope: a true cross-trait discrimination test ("does the Axis fire
similarly on evil, sycophantic, AND hallucinating responses, i.e. is it
trait-agnostic?") is NOT computable from this data - each trait's implicit
CSV contains only that trait's own judge score for that trait's own prompts/
responses; there is no shared response set scored on all three traits to
correlate row-for-row. What IS computable, and what this script reports, is
each condition's Assistant-Axis-r side by side with its persona-vector-r: if
the Axis matches or beats the trait-specific vector's r on its OWN trait, the
general axis is at least as good a monitor for that specific implicit
condition, even without answering the separate discrimination question.

Reads the *_axis.csv outputs of project_persona_vectors_responses.py (one
file per trait x condition, each carrying its own trait's judge score plus
assistant_axis_projection).

Usage:
    uv run scripts/analyze_persona_vectors_projections.py
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

TRAITS = ["evil", "sycophantic", "hallucinating"]
CONDITIONS = ["explicit", "implicit_described", "implicit_contextual"]
# best explicit layer per trait, from WRITEUP_32B.md Section 2
BEST_LAYER = {"evil": 30, "sycophantic": 28, "hallucinating": 60}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default=str(Path(__file__).parent.parent / "results" / "persona_vectors_projections"),
        help="Directory containing this repo's own *_axis.csv outputs (never persona_vectors' tree)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.results_dir)

    rows = []
    for trait in TRAITS:
        for cond in CONDITIONS:
            path = eval_dir / f"{trait}_{cond}_axis.csv"
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            df = pd.read_csv(path)
            axis_r, axis_p = pearsonr(df["assistant_axis_projection"], df[trait])
            axis_rho, axis_rho_p = spearmanr(df["assistant_axis_projection"], df[trait])

            vec_col = f"Qwen3-32B_{trait}_response_avg_diff_proj_layer{BEST_LAYER[trait]}"
            vec_r = vec_rho = None
            if vec_col in df.columns:
                vec_r, _ = pearsonr(df[vec_col], df[trait])
                vec_rho, _ = spearmanr(df[vec_col], df[trait])

            rows.append({
                "trait": trait,
                "condition": cond,
                "n": len(df),
                "assistant_axis_pearson_r": round(axis_r, 3),
                "assistant_axis_spearman_rho": round(axis_rho, 3),
                "persona_vector_pearson_r": round(vec_r, 3) if vec_r is not None else "N/A",
                "persona_vector_spearman_rho": round(vec_rho, 3) if vec_rho is not None else "N/A",
            })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv("results_assistant_axis_vs_persona_vector.csv", index=False)
    print("\nWrote results_assistant_axis_vs_persona_vector.csv")

    print("\nNote: Assistant Axis projection is signed so HIGHER = more assistant-like, "
          "LOWER = more role-playing. Expect assistant_axis_* to be NEGATIVE (more trait "
          "expression -> lower projection), opposite sign convention from persona_vector_*.")
    print("\nSpearman rho is included alongside Pearson r as a robustness check: rho only "
          "requires a monotonic (not necessarily linear) relationship and is less sensitive "
          "to outliers/floor effects, both relevant given how skewed several conditions' "
          "judge-score distributions are. If rho and r tell the same story for a given cell, "
          "that's reassurance the r-based finding isn't a linearity/outlier artifact.")


if __name__ == "__main__":
    main()
