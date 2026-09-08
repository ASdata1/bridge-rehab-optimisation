import pandas as pd
import pytest

from src.baseline import risk_years_achieved, worst_condition_first
from src.optimise import solve_milp


@pytest.fixture
def toy_df():
    # 4 bridges, deliberately sized so the budget constraint actually binds and forces
    # a real choice (can't just fund everything).
    return pd.DataFrame({
        "structure_number": ["A", "B", "C", "D"],
        "lowest_rating": [3, 8, 4, 6],
        "risk_untreated": [50.0, 5.0, 40.0, 20.0],
        "risk_treated": [10.0, 4.0, 8.0, 5.0],
        "risk_reduction": [40.0, 1.0, 32.0, 15.0],
        "cost_usd": [1_000_000, 200_000, 900_000, 500_000],
    })


def test_budget_constraint_is_respected_single_year(toy_df):
    budget = [1_000_000]
    result = solve_milp(toy_df, T=1, budget_per_year=budget)
    spent = toy_df.set_index("structure_number").loc[result.x["structure_number"], "cost_usd"].sum()
    assert spent <= budget[0] + 1e-6


def test_budget_constraint_is_respected_per_year_multi_year(toy_df):
    T = 3
    budget = [900_000, 900_000, 900_000]
    result = solve_milp(toy_df, T=T, budget_per_year=budget)
    for t in range(1, T + 1):
        spent_t = toy_df.set_index("structure_number").loc[
            result.x.loc[result.x["year"] == t, "structure_number"], "cost_usd"
        ].sum()
        assert spent_t <= budget[t - 1] + 1e-6


def test_each_bridge_treated_at_most_once_across_horizon(toy_df):
    result = solve_milp(toy_df, T=3, budget_per_year=[2_000_000, 2_000_000, 2_000_000])
    counts = result.x["structure_number"].value_counts()
    assert (counts <= 1).all()


def test_selection_is_a_feasible_subset_of_input_bridges(toy_df):
    result = solve_milp(toy_df, T=1, budget_per_year=[1_500_000])
    assert set(result.x["structure_number"]).issubset(set(toy_df["structure_number"]))


def test_tight_budget_picks_the_highest_risk_reduction_per_dollar_bridge(toy_df):
    # With a budget that can only afford exactly one bridge, the optimiser should
    # pick whichever single bridge gives the most risk_reduction per dollar spent --
    # here that's C (32 reduction / $900k = 3.56e-5) not A (40 / $1M = 4.0e-5)...
    # actually check both fit only one at a time under a $900k cap: only C fits.
    result = solve_milp(toy_df, T=1, budget_per_year=[900_000])
    assert set(result.x["structure_number"]) == {"C"}


def test_zero_budget_selects_nothing(toy_df):
    result = solve_milp(toy_df, T=1, budget_per_year=[0])
    assert result.x.empty


def test_optimal_plan_is_never_worse_than_worst_condition_first_baseline(toy_df):
    T, budget = 1, [1_000_000]
    opt = solve_milp(toy_df, T, budget)
    baseline_sel = worst_condition_first(toy_df, T, budget)
    opt_val = opt.objective_value
    baseline_val = risk_years_achieved(toy_df, baseline_sel, T)
    assert opt_val >= baseline_val - 1e-6
