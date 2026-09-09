"""
Budget sensitivity: how much is an extra dollar of capital budget actually worth?

Two different, deliberately-not-conflated numbers are produced here (see the
discussion with Aman in-conversation, and the README "Shadow price" section):

1. LP-relaxation shadow price (`lp_shadow_prices`) -- the exact dual value on each
   year's budget constraint from the *continuous* relaxation (x in [0,1]), via KKT /
   linear-programming duality. Because relaxing the integrality constraint can only
   enlarge the feasible region, the LP relaxation's optimal value is always an upper
   bound on the true (integer-constrained) achievable risk reduction for a minimising
   risk objective -- equivalently a *best case* on how much value an extra dollar of
   budget could buy, since it implicitly allows funding 30% of a bridge. It is exact
   for the relaxation, not for the real problem.

2. Empirical shadow price (`budget_sweep`) -- re-solves the actual MILP at a range of
   budget levels and reports the realised finite difference in risk-years achieved
   between adjacent budget levels. This is the real, integer-constrained answer, and
   it is a step function (a small budget increase sometimes buys nothing at all, until
   it's enough to fund one more whole bridge) rather than the smooth line the LP
   relaxation implies.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.optimise import solve_lp_relaxation, solve_milp


def lp_shadow_prices(df: pd.DataFrame, T: int, budget_per_year: Sequence[float]) -> pd.DataFrame:
    result = solve_lp_relaxation(df, T, budget_per_year)
    rows = []
    for t in range(1, T + 1):
        constraint = result.problem.constraints[f"budget_year_{t}"]
        rows.append({
            "year": t,
            "budget": budget_per_year[t - 1],
            # PuLP's constraint.pi is the dual value for a <= constraint under CBC.
            "shadow_price_per_usd": constraint.pi,
        })
    return pd.DataFrame(rows)


def budget_sweep(
    df: pd.DataFrame,
    T: int,
    budget_levels: Sequence[float],
    equal_per_year: bool = True,
) -> pd.DataFrame:
    """Re-solve the MILP at each total-budget level in `budget_levels` (split evenly
    across the T years) and report the achieved objective, plus the empirical
    marginal value of the budget step from the previous level.
    """
    if not equal_per_year:
        raise NotImplementedError("budget_sweep currently only supports equal_per_year=True")

    rows = []
    prev_total = None
    prev_obj = None
    for total_budget in budget_levels:
        per_year = [total_budget / T] * T
        result = solve_milp(df, T, per_year)
        obj = result.objective_value
        marginal = None
        if prev_total is not None and prev_obj is not None and total_budget > prev_total:
            marginal = (obj - prev_obj) / (total_budget - prev_total)
        rows.append({
            "total_budget": total_budget,
            "risk_years_reduced": obj,
            "bridges_treated": result.x["structure_number"].nunique(),
            "empirical_marginal_value_per_usd": marginal,
        })
        prev_total, prev_obj = total_budget, obj
    return pd.DataFrame(rows)
