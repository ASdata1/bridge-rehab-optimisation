"""
"Worst-condition-first" greedy baseline -- a common real-world triage rule (fix the
worst-rated bridges first, regardless of cost-efficiency or traffic exposure) used
as the comparison point for the MILP-optimal plan.

Bridges are ranked purely by lowest_rating (worst first, ties broken by risk_untreated
descending), then funded greedily year by year until each year's budget runs out.
This is deliberately *not* cost- or criticality-aware -- that's the point of the
comparison: it's what a simple, defensible-sounding triage rule gets you, versus
what actually solving the allocation problem gets you for the same money.
"""

from __future__ import annotations

import pandas as pd


def worst_condition_first(df: pd.DataFrame, T: int, budget_per_year: list[float]) -> pd.DataFrame:
    ranked = df.sort_values(["lowest_rating", "risk_untreated"], ascending=[True, False]).copy()

    rows = []
    remaining = set(ranked.index)
    for t in range(1, T + 1):
        budget_left = budget_per_year[t - 1]
        for i in list(ranked.index):
            if i not in remaining:
                continue
            cost = ranked.loc[i, "cost_usd"]
            if cost <= budget_left:
                rows.append({"structure_number": ranked.loc[i, "structure_number"], "year": t, "selected": 1})
                remaining.discard(i)
                budget_left -= cost

    return pd.DataFrame(rows, columns=["structure_number", "year", "selected"])


def risk_years_achieved(df: pd.DataFrame, selection: pd.DataFrame, T: int) -> float:
    """Objective value (sum_i sum_s risk_reduction_i * (T-s+1) * x[i,s]) for a given
    selection, so the baseline and the MILP-optimal plan can be compared on exactly
    the same footing as the optimiser's own objective.
    """
    if selection.empty:
        return 0.0
    merged = selection.merge(df[["structure_number", "risk_reduction"]], on="structure_number")
    merged["weight"] = T - merged["year"] + 1
    return float((merged["risk_reduction"] * merged["weight"]).sum())
