# Bridge Renewal Investment Planning

Given a fixed multi-year capital budget, which bridges in a real state highway network
should be prioritised for rehabilitation, in which year, to minimise network-wide risk
exposure — and what's the marginal value of an extra $1M of budget?

A mixed-integer optimisation model (PuLP/CBC) applied to Rhode Island's real bridge
inventory, built to demonstrate applied mathematical optimisation and Asset Investment
Planning specifically — see "Motivation" below.

## Headline result

Solving the allocation as a MILP instead of following the common "fix the worst-rated
bridges first" heuristic achieves **174% more risk reduction for the same $100M / 5-year
budget** — 15,719 risk-years avoided vs. 5,732 — and funds **161 bridges instead of 35**,
because the heuristic burns most of the budget on a handful of large, expensive,
badly-rated bridges, while the optimiser finds many smaller bridges that deliver more
risk reduction per dollar.

![Optimal vs baseline](figures/optimal_vs_baseline.png)

Budget has diminishing returns, as expected — each extra $25M buys less risk reduction
than the last:

![Budget vs risk reduction](figures/budget_vs_risk_reduction.png)

## Motivation

This project targets a specific line from a real graduate Data Scientist posting
(Arcadis, Data Science & Analytics Group): *"solving complex mathematical optimisation
challenges — principally, but not exclusively related to Asset Investment Planning."*
Rather than a generic optimisation demo, it's built around the exact decision that
phrase describes — which physical assets get a limited capital budget, and when — using
real government asset-condition data instead of synthetic numbers.

## Data

**Source:** FHWA National Bridge Inventory (NBI), 2025 delimited extract for Rhode
Island (`RI25.txt`), via
[fhwa.dot.gov/bridge/nbi/ascii2025.cfm](https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm).
787 bridges, 123 standard NBI fields per bridge (condition ratings, traffic, geometry,
inspection history, past improvement costs). This is real government data with real
messiness, not a synthetic dataset — see `src/data_prep.py` for the specific quirks
handled:

- Culverts are rated on a separate field (`CULVERT_COND_062`) instead of
  deck/superstructure/substructure, which show `"N"` (not applicable) for those
  structures. FHWA's own `LOWEST_RATING` field already resolves this correctly —
  verified against the raw condition fields for all 787 rows in
  `tests/test_data_prep.py`, rather than trusted blindly.
- 3 of 787 bridges have `ADT_029 == 0` on roads that plainly carry real traffic (e.g.
  "US 1 SB POST RD") — a data gap, not a literal zero. Imputed with the state median.
- `STRUCTURE_NUMBER_008` formatting is inconsistent across bridges (`"1RI0668"` vs.
  `"000000000000010"`) — kept as an opaque string identifier throughout, never parsed
  as a number.

14.0% of Rhode Island's bridges are rated Poor by FHWA's own classification, 61.1% Fair,
24.9% Good — see `notebooks/analysis.ipynb` for the full EDA.

## Methodology

### Risk score (`src/risk_model.py`)

A deliberately simple, fully disclosed proxy — not a calibrated structural
failure-probability model:

```
risk_i = deficiency_i × criticality_i × 100

deficiency_i  = (9 - lowest_rating_i) / 7      # NBI condition scale, normalised against
                                                # the worst rating actually observed (2)
criticality_i = log1p(adt_i) / log1p(adt_max)  # traffic is heavily right-skewed, so it's
                                                # log-scaled before normalising 0..1
```

If rehabilitated, a bridge's condition is assumed restored to rating 8 ("very good", one
below new — a realistic post-rehab state, not "as new"), capped so treatment can never
increase a bridge's risk (a real edge case caught by `tests/test_risk_model.py`: a
bridge already better than the assumed post-treatment state would otherwise show
`risk_treated > risk_untreated`).

### Cost model (`src/risk_model.py`)

```
cost_i = deck_area_sqm_i × 10.7639 × unit_rate_i
```

`unit_rate_i` is FHWA's own published 2024 **Bridge Replacement Unit Cost**
([fhwa.dot.gov/bridge/nbi/sd2024.cfm](https://www.fhwa.dot.gov/bridge/nbi/sd2024.cfm)):
**$738/ft²** for Rhode Island's National Highway System bridges specifically, **$382/ft²**
(national non-NHS average) for the rest. `deck_area_sqm` is FHWA's own computed field,
cross-checked against `length × width` in `tests/test_data_prep.py`.

This treats every intervention as full-replacement-equivalent cost, which is
deliberately conservative — a real capital programme would cost most rehabilitations
well below full replacement. Flagged explicitly here rather than papered over with an
invented discount factor this project has no data to justify (see "Where this is
going"). One reason it wasn't calibrated from this dataset's own
`TOTAL_IMP_COST_096` field: that field clusters suspiciously tightly around one
implied $/ft² rate across very different bridges, suggesting it's a formulaic default
rather than a bridge-specific bid — a real data quirk worth naming rather than quietly
using a misleading number.

### The optimisation model (`src/optimise.py`)

A mixed-integer program, decision variable `x[i,t] ∈ {0,1}` = 1 if bridge *i* is
rehabilitated in year *t* of a *T*-year horizon:

```
min  Σ_t Σ_i  risk_untreated_i − risk_reduction_i · Σ_{s≤t} x[i,s]

s.t. Σ_i cost_i · x[i,t] ≤ budget_t     for every year t   (capital budget)
     Σ_t x[i,t] ≤ 1                     for every bridge i (treat at most once)
     x[i,t] ∈ {0,1}
```

Expanding this shows it's equivalent to **maximising**
`Σ_i Σ_s risk_reduction_i · (T−s+1) · x[i,s]` — treating a bridge in year 1 is worth
`T` times its risk reduction, year 2 worth `T−1` times, and so on, which is what
minimising *risk-years* naturally rewards: earlier treatment of a high-value bridge is
worth strictly more than later treatment of the same bridge. `T=1` collapses this to a
plain 0/1 knapsack — the model wasn't scoped as two separate cases, `T` is just a
parameter, and the single-year case is the guaranteed-finishable fallback the multi-year
extension was built on top of.

Solved with PuLP + CBC ([Subira Rodriguez, "Linear Programming: optimizing solutions
with Python using PuLP", Medium,
2022](https://medium.com/@telmosubirar/linear-programming-optimizing-solutions-with-python-using-pulp-e0c4379696c8),
for the modelling syntax). At full scale (787 bridges × 5 years = 3,935 binary
variables) CBC can spend a very long time closing the last fraction of a percent of
optimality gap even when a near-optimal solution is trivial to find — solved to within
1% of the proven LP bound instead (`mip_gap=0.01`), standard practice for a problem this
size, and reported alongside every result rather than silently accepted.

### Shadow price of the budget — two different numbers, deliberately not conflated

1. **LP-relaxation shadow price** (`src/sensitivity.py::lp_shadow_prices`) — the exact
   dual value on each year's budget constraint from the *continuous* relaxation
   (`x ∈ [0,1]`), via LP duality (the same KKT reasoning as a textbook QP, just applied
   here). Because relaxing integrality can only enlarge the feasible region, this is a
   **best-case upper bound** on the true marginal value — it implicitly assumes 30% of a
   bridge can be funded.
2. **Empirical shadow price** (`src/sensitivity.py::budget_sweep`) — re-solves the
   actual MILP at a range of budget levels and takes the realised finite difference in
   risk-years avoided between adjacent levels. This is the real, integer-constrained
   answer: at $100M over 5 years, the LP relaxation says an extra dollar in year 1 is
   worth ~$0.00019 of risk-years; the real (integer) marginal value over that budget
   range is ~$0.00011–0.00012 — the relaxation's bound is real but optimistic, exactly
   as the theory predicts.

## Comparison baseline (`src/baseline.py`)

"Worst-condition-first": rank bridges purely by NBI condition rating (worst first),
fund greedily until each year's budget runs out. A common, defensible-sounding real
triage rule that ignores cost-efficiency and traffic exposure — which is exactly why it
leaves 174% of achievable risk reduction on the table for the same money (see
"Headline result").

## Repo structure

```
bridge-investment-optimisation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/RI25.csv            # FHWA NBI extract, Rhode Island, 2025
│   └── processed/              # all regenerated by: python -m src.run_all
│       ├── bridges.csv         # cleaned + risk/cost-scored
│       ├── optimal_selection.csv     # MILP plan for $100M / 5 years
│       ├── baseline_selection.csv    # worst-condition-first, same budget
│       ├── lp_shadow_prices.csv      # LP-relaxation dual on each year's budget
│       └── budget_sweep.csv          # risk-years avoided at 6 budget levels
├── src/
│   ├── data_prep.py            # load, clean, export
│   ├── risk_model.py           # risk_i, cost_i
│   ├── optimise.py             # MILP + LP relaxation
│   ├── baseline.py             # worst-condition-first heuristic
│   ├── sensitivity.py          # shadow prices + budget sweep
│   ├── make_figures.py
│   └── run_all.py              # raw extract -> every data/processed/ file, end to end
├── notebooks/analysis.ipynb    # EDA
├── tests/                      # 20 tests — data validation, risk model, MILP constraints
└── figures/
```

## How to run

```bash
pip install -r requirements.txt
python -m src.run_all         # raw extract -> bridges.csv, optimal/baseline plans,
                              #   shadow prices, budget sweep (re-solves the MILP
                              #   once per budget level -- a few minutes)
python -m pytest tests/ -v    # 20 tests
python -m src.make_figures    # regenerate figures/ from the outputs above
```

`python -m src.data_prep` runs just the cleaning step (raw extract ->
`bridges.csv` without the risk/cost columns) if that's all you need; `run_all`
supersedes it for a full regeneration. Because each MILP solve stops at the 1%
gap, risk-year totals can move by up to ~1% between runs or solver versions.

## Where this is going

- **Cost model refinement.** Every intervention currently costs full-replacement-
  equivalent; a more realistic model would scale cost with rehabilitation scope
  (deck overlay vs. full replacement) rather than treating them identically.
- **Degradation over time.** The current model treats an untreated bridge's risk as
  constant across the horizon; a fuller model would let condition (and therefore risk)
  worsen year over year absent intervention, which would likely sharpen the case for
  earlier treatment further.
- **Uncertainty.** Point-estimate condition ratings and costs, no confidence intervals
  or robustness analysis — a natural next step, not attempted here given the project's
  time budget.

## References

- Federal Highway Administration, [National Bridge Inventory (NBI) — 2025 ASCII/delimited files](https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm) — source dataset (Rhode Island extract, `RI25.txt`).
- Federal Highway Administration, [Bridge Replacement Unit Costs 2024](https://www.fhwa.dot.gov/bridge/nbi/sd2024.cfm) — cost model unit rates.
- Telmo Subira Rodriguez, ["Linear Programming: optimizing solutions with Python using PuLP"](https://medium.com/@telmosubirar/linear-programming-optimizing-solutions-with-python-using-pulp-e0c4379696c8), Medium, June 2022 — PuLP modelling syntax reference used in `src/optimise.py`.
