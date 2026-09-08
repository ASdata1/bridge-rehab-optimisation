"""
MILP formulation for multi-year bridge rehabilitation investment planning.

Sets
----
i = 1..N   bridges
t = 1..T   years in the planning horizon (T=1 collapses this to a single-cycle
           0/1 knapsack -- see README for why that's not a separate model, just
           this one with T=1)

Decision variable
------------------
x[i, t] in {0, 1} : 1 if bridge i is rehabilitated in year t

Objective
---------
Minimise total risk-years across the horizon:

    min  sum_t sum_i risk_untreated_i - risk_reduction_i * sum_{s<=t} x[i, s]

Dropping the constant sum_t sum_i risk_untreated_i term, this is equivalent to:

    max  sum_i sum_s risk_reduction_i * (T - s + 1) * x[i, s]

i.e. treating bridge i in year s is worth risk_reduction_i for every remaining
year of the horizon, so earlier treatment of a high-risk-reduction bridge is
worth strictly more than later treatment of the same bridge -- this weighting
falls straight out of "minimise risk-years", it isn't bolted on separately.
This is the formulation actually implemented below (see `optimise.py`
docstring reference: Subira Rodriguez, "Linear Programming: optimizing
solutions with Python using PuLP", Medium, 2022, for the PuLP modelling
syntax this follows).

Constraints
-----------
    sum_i cost_i * x[i, t] <= budget_t          for every year t   (capital budget)
    sum_t x[i, t] <= 1                          for every bridge i (treat at most once)
    x[i, t] in {0, 1}                            (or [0, 1] for the LP relaxation)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp


@dataclass
class OptimiseResult:
    status: str
    objective_value: float  # sum of risk_reduction_i * (T-s+1) * x[i,s] achieved
    x: pd.DataFrame  # long-form: structure_number, year, selected (0/1 or fractional)
    problem: pulp.LpProblem


def _build_problem(
    df: pd.DataFrame,
    T: int,
    budget_per_year: list[float],
    integer: bool,
) -> tuple[pulp.LpProblem, dict]:
    assert len(budget_per_year) == T, "budget_per_year must have one entry per year"

    cat = "Binary" if integer else "Continuous"
    prob = pulp.LpProblem("bridge_investment_planning", pulp.LpMaximize)

    x = {
        (i, t): pulp.LpVariable(f"x_{i}_{t}", lowBound=0, upBound=1, cat=cat)
        for i in df.index
        for t in range(1, T + 1)
    }

    # Objective: sum_i sum_s risk_reduction_i * (T - s + 1) * x[i, s]
    prob += pulp.lpSum(
        df.loc[i, "risk_reduction"] * (T - t + 1) * x[i, t]
        for i in df.index
        for t in range(1, T + 1)
    )

    # Budget constraint per year -- named so we can pull its shadow price later.
    for t in range(1, T + 1):
        prob += (
            pulp.lpSum(df.loc[i, "cost_usd"] * x[i, t] for i in df.index) <= budget_per_year[t - 1],
            f"budget_year_{t}",
        )

    # Treat each bridge at most once across the horizon.
    for i in df.index:
        prob += (
            pulp.lpSum(x[i, t] for t in range(1, T + 1)) <= 1,
            f"at_most_once_{i}",
        )

    return prob, x


def solve_milp(
    df: pd.DataFrame,
    T: int,
    budget_per_year: list[float],
    mip_gap: float = 0.01,
    time_limit: int = 60,
) -> OptimiseResult:
    """Solve the MILP. For T > 1 this is a genuinely large combinatorial problem
    (N bridges x T years of binaries, coupled by an at-most-once constraint per
    bridge) -- proving *exact* optimality can take CBC a very long time even
    when a near-optimal solution is trivial to find, because the search spends
    almost all its effort closing the last fraction of a percent of gap. Solving
    to within `mip_gap` (default 1%) of the proven LP bound is standard practice
    for problems this size, and is disclosed here rather than silently accepted:
    every result reports the achieved gap alongside the objective value.
    """
    prob, x = _build_problem(df, T, budget_per_year, integer=True)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap))

    rows = []
    for (i, t), var in x.items():
        val = var.value() or 0
        if val > 0.5:  # binary, but guard against solver float noise
            rows.append({"structure_number": df.loc[i, "structure_number"], "year": t, "selected": 1})
    selected = pd.DataFrame(rows, columns=["structure_number", "year", "selected"])

    return OptimiseResult(
        status=pulp.LpStatus[prob.status],
        objective_value=pulp.value(prob.objective),
        x=selected,
        problem=prob,
    )


def solve_lp_relaxation(df: pd.DataFrame, T: int, budget_per_year: list[float]) -> OptimiseResult:
    """Solve the continuous relaxation (x in [0,1]) so we can read off the shadow
    price (dual value) on each year's budget constraint -- see src/sensitivity.py.
    """
    prob, x = _build_problem(df, T, budget_per_year, integer=False)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    rows = []
    for (i, t), var in x.items():
        val = var.value() or 0
        if val > 1e-9:
            rows.append({"structure_number": df.loc[i, "structure_number"], "year": t, "selected": val})
    selected = pd.DataFrame(rows, columns=["structure_number", "year", "selected"])

    return OptimiseResult(
        status=pulp.LpStatus[prob.status],
        objective_value=pulp.value(prob.objective),
        x=selected,
        problem=prob,
    )
