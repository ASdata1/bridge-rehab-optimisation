# Bridge Renewal Investment Planning

A highway agency has a fixed amount of money to spend over several years repairing
ageing bridges, and cannot fix them all. This project works out which bridges to
rehabilitate, and in which year, so that the road network carries as little risk as
possible while the work is done. It also measures how much extra benefit another
$1M of budget would buy.

## The problem

Every bridge is different. Some are in much worse condition than others. Some carry
tens of thousands of vehicles a day and some carry a few hundred. Some are large and
expensive to work on and some are small and cheap. A limited budget cannot cover
everything, so the agency has to choose.

A sensible-sounding way to choose is to fix the worst-rated bridges first. The
problem is that this ignores cost and traffic: it can spend the whole budget on a
few huge, expensive, badly-rated bridges and leave many cheaper bridges untouched
that together would have removed far more risk.

This project treats the choice as an optimisation problem instead, and compares the
result against the fix-the-worst-first rule.

## Motivation

This project comes out of the third-year mathematical programming module on my MORSE
BSc at Southampton. That course covered solving linear programmes with the simplex
and dual simplex methods and with interior point methods, handling convex problems
through KKT conditions, and solving integer problems with branch and bound and
knapsack methods, along with the idea of shadow prices. Here I extend that material:
I taught myself binary integer programming, the CBC solver, and the PuLP modelling
library from a Medium article on linear programming in Python, and applied them to a
real asset-investment decision. It is also an exercise in one of the many kinds of
problem a data scientist can be asked to solve.

## Data

The bridge data is the Federal Highway Administration's National Bridge Inventory,
2025 delimited extract for Rhode Island: 787 bridges, each with a condition rating, a
traffic count, and deck geometry. The cost rates are the FHWA's published 2024 bridge
replacement unit costs.

## How risk is scored

Each bridge gets a risk score built from two things:

- **Condition.** The inventory rates each bridge from 9, meaning new, down to 0,
  meaning failed. A worse rating gives a higher score.
- **Traffic.** A busy bridge matters more than a quiet one. Traffic counts vary
  enormously between bridges, so they are put on a compressed scale before use,
  otherwise a handful of motorway bridges would swamp everything else.

The score is condition multiplied by traffic, so a bridge scores high only if it is
both in poor condition and well used. A crumbling bridge on a quiet lane scores low,
and so does a perfect bridge on a motorway.

Rehabilitating a bridge is assumed to bring its condition back up to "very good", one
step below new, so its score drops but does not reach zero. The fall in score is the
risk reduction that rehabilitation buys for that bridge.

## How cost is estimated

The cost of working on a bridge is its deck area multiplied by a cost per square
foot, taken from the FHWA's 2024 replacement unit rates. Bridges on the main national
highway network use a higher rate than the rest. This treats every job as a full
replacement, which is deliberately on the high side, since in practice many bridges
would need less work than that. It keeps the cost figures consistent and
conservative.

## The optimisation method

For every bridge and every year of the plan there is one yes/no decision:
rehabilitate that bridge in that year, or not. Each decision is a variable that can
only take the value 0 or 1, and the objective and every constraint are linear in
those variables. A model with this shape, linear with variables restricted to 0 or 1,
is a **binary integer program**, or BIP. Because the budget is split across several
years and the model combines these 0/1 decisions with linear budget arithmetic, it is
also called a **mixed-integer linear program**, or MILP.

The objective is to minimise total risk-years: each bridge's risk score, added up
over every year of the plan that it still carries that risk. A bridge rehabilitated
early has its risk removed for more of the remaining years than one rehabilitated
late, so the model naturally treats the high-value bridges as early as the budget
allows. The constraints are that spending in any year cannot exceed that year's
budget, and that a bridge is rehabilitated at most once.

The model is built with PuLP and solved with CBC. An integer program cannot be solved
by the simplex method alone, because simplex would hand back a solution with
fractional decisions such as "rehabilitate 0.4 of a bridge". CBC uses **branch and
bound**. It first solves the relaxed problem in which the decisions are allowed to be
fractional, which is fast and gives a bound on the best result that is possible. It
then repeatedly splits the problem by forcing one fractional variable to 0 on one
branch and to 1 on the other, and discards any branch whose bound shows it cannot
beat the best whole-number solution found so far. At full size the model has 787
bridges over 5 years, close to 4,000 binary variables. Proving that a solution is
exactly optimal can take a very long time even after a good solution has been found,
so CBC is allowed to stop once it is provably within 1% of the best possible. This is
standard for a problem of this size, and every result is reported with the gap it was
solved to.

## What an extra dollar of budget is worth

This is answered two ways.

The first is the **shadow price** from the relaxed problem: the dual value on each
year's budget constraint when the decisions are allowed to be fractional. It is the
marginal value of one more dollar of budget in that year. Because allowing fractional
decisions can only ever help, this is a best case, an upper bound on what a real
dollar could be worth. For year one it works out at about 190 risk-years per extra
$1M, and it is lower in later years because a dollar spent later has fewer years left
to pay back.

The second is the **empirical value**: solve the real integer model at a range of
budget levels and measure how much extra risk reduction each step up actually
delivered. Near the $100M level this comes out at about 110 risk-years per extra $1M,
below the relaxed-problem bound, exactly as the theory says it must be, because in
reality a bridge cannot be part-funded.

## The comparison baseline

The optimised plan is compared against a simple rule an agency might realistically
use: rank the bridges from worst condition to best, and fund them in that order until
each year's budget runs out. This rule looks at condition only, and ignores both cost
and traffic.

## Results

All figures use a $100M budget spread evenly over 5 years.

| Approach | Bridges rehabilitated | Risk-years avoided |
|---|---|---|
| Worst-condition-first rule | 35 | 5,732 |
| Optimisation | 161 | 15,719 |

The same optimisation, run at a range of budgets:

| 5-year budget | Bridges treated | Risk-years avoided | Extra risk-years per additional $1M |
|---|---|---|---|
| $50M | 102 | 9,931 | — |
| $75M | 132 | 12,971 | 122 |
| $100M | 161 | 15,719 | 110 |
| $125M | 191 | 18,061 | 94 |
| $150M | 214 | 20,159 | 84 |
| $200M | 248 | 23,813 | 73 |

### Headline results

For the same $100M and the same 5 years, the optimisation avoids **174% more risk**
than the worst-condition-first rule, 15,719 risk-years against 5,732, and it does
this while rehabilitating **161 bridges instead of 35**. The rule spends most of the
budget on a small number of large, expensive, badly-rated bridges. The optimisation
instead finds many smaller bridges that each remove more risk per dollar, and gets
far more out of the same money.

More budget always helps, but each increment helps less than the one before. The
first $25M above $50M buys about 122 risk-years for every $1M; the step from $150M to
$200M buys about 73. The most cost-effective bridges are funded first, so every later
tranche of money goes on weaker options.

The two measures of a dollar's value agree in direction and differ in size as
expected. The relaxed-problem shadow price for year one is about 190 risk-years per
$1M. The real integer-constrained value near $100M is about 110. The difference is
the price of not being able to fund a fraction of a bridge.

![Optimal vs baseline](figures/optimal_vs_baseline.png)

![Budget vs risk reduction](figures/budget_vs_risk_reduction.png)

## How to run

```bash
pip install -r requirements.txt
python -m src.run_all        # raw data -> scored bridges, optimised plan, baseline, budget sweep
python -m pytest tests/ -v   # 20 tests
python -m src.make_figures   # regenerate the figures above
```

## Where this is going

- **Cost model refinement.** Every job is currently costed as a full replacement. A
  better model would scale the cost with the actual scope of work, from a deck
  overlay up to a full rebuild.
- **Degradation over time.** An untreated bridge's risk is held constant across the
  plan. A fuller model would let condition, and so risk, worsen year on year without
  intervention, which would likely make the case for early treatment stronger still.
- **Uncertainty.** Condition ratings and costs are treated as fixed numbers with no
  error bars. Adding uncertainty and testing how robust the plan is to it is a
  natural next step.
