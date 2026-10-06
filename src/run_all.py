"""
Regenerate every file in data/processed/ from data/raw/RI25.csv, end to end.

This is the driver the pipeline was missing. `src/data_prep.py` cleans the raw
NBI extract, but the risk/cost scoring, the MILP-optimal plan, the
worst-condition-first baseline, the LP-relaxation shadow prices, and the budget
sweep were previously only ever run by hand -- so the committed results in
data/processed/ could not be reproduced from the repo alone. Run this to rebuild
all of them from scratch:

    python -m src.run_all

Then regenerate the figures that read these outputs:

    python -m src.make_figures

Runtime: a few minutes. The budget sweep re-solves the full 787-bridge x 5-year
MILP once per budget level; each solve stops at the 1% MIP gap (see
src/optimise.py), so risk-year totals can move by up to ~1% between runs or
solver versions.
"""

from __future__ import annotations

from pathlib import Path

from src.baseline import risk_years_achieved, worst_condition_first
from src.data_prep import clean, load_raw
from src.optimise import solve_milp
from src.risk_model import add_risk_and_cost
from src.sensitivity import budget_sweep, lp_shadow_prices
from src.spatial_analysis import main as spatial_analysis_main
from src.spatial_data import main as spatial_data_main

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Headline scenario: $100M of capital over a 5-year horizon, split evenly.
HORIZON_YEARS = 5
BUDGET_PER_YEAR = [20_000_000] * HORIZON_YEARS

# Total-budget levels for the sensitivity sweep (each split evenly across the 5 years).
BUDGET_SWEEP_LEVELS = [
    50_000_000,
    75_000_000,
    100_000_000,
    125_000_000,
    150_000_000,
    200_000_000,
]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean the raw NBI extract, then attach risk_untreated / risk_treated /
    #    risk_reduction / cost_usd. This is the file every step below consumes.
    #    Note: goes straight from load_raw() (dtype=str), so NBI codes like
    #    owner_code / functional_class keep their leading zeros ("01", not 1).
    df = add_risk_and_cost(clean(load_raw()))
    df.to_csv(PROCESSED_DIR / "bridges.csv", index=False, lineterminator="\n")
    print(f"[run_all] bridges.csv        {len(df)} bridges cleaned + scored")

    # 2. MILP-optimal plan for the headline $100M / 5-year scenario.
    opt = solve_milp(df, HORIZON_YEARS, BUDGET_PER_YEAR)
    opt.x.to_csv(PROCESSED_DIR / "optimal_selection.csv", index=False, lineterminator="\n")
    print(f"[run_all] optimal_selection  status={opt.status}, "
          f"{opt.x['structure_number'].nunique()} bridges, "
          f"{opt.objective_value:,.0f} risk-years avoided")

    # 3. Worst-condition-first baseline, same horizon and budget, scored on the
    #    optimiser's own objective so the two are directly comparable.
    base_sel = worst_condition_first(df, HORIZON_YEARS, BUDGET_PER_YEAR)
    base_sel.to_csv(PROCESSED_DIR / "baseline_selection.csv", index=False, lineterminator="\n")
    base_val = risk_years_achieved(df, base_sel, HORIZON_YEARS)
    print(f"[run_all] baseline_selection {base_sel['structure_number'].nunique()} bridges, "
          f"{base_val:,.0f} risk-years avoided")

    # 4. LP-relaxation shadow price (dual value) on each year's budget constraint
    #    -- the best-case upper bound on an extra dollar's marginal value.
    shadow = lp_shadow_prices(df, HORIZON_YEARS, BUDGET_PER_YEAR)
    shadow.to_csv(PROCESSED_DIR / "lp_shadow_prices.csv", index=False, lineterminator="\n")
    print(f"[run_all] lp_shadow_prices\n{shadow.to_string(index=False)}")

    # 5. Empirical budget sweep -- re-solves the MILP at each budget level and
    #    reports the realised marginal value of each step up.
    sweep = budget_sweep(df, HORIZON_YEARS, BUDGET_SWEEP_LEVELS)
    sweep.to_csv(PROCESSED_DIR / "budget_sweep.csv", index=False, lineterminator="\n")
    print(f"[run_all] budget_sweep\n{sweep.to_string(index=False)}")

    # 6. Geospatial chapter: decode coordinates and attach plan membership, then the
    #    county / Moran's I / hotspot statistics. Reads files written above.
    print("[run_all] spatial: bridges_geo.csv")
    spatial_data_main()
    print("[run_all] spatial: spatial_summary.csv")
    spatial_analysis_main()

    print(f"[run_all] done -- wrote 7 files to {PROCESSED_DIR}. "
          "Next: python -m src.make_figures")


if __name__ == "__main__":
    main()
