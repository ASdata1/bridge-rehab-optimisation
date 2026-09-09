"""Generate the two headline figures for the README from the saved sensitivity /
comparison results (data/processed/budget_sweep.csv, optimal vs baseline selections).

Run: python -m src.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.baseline import risk_years_achieved, worst_condition_first
from src.optimise import solve_milp
from src.risk_model import add_risk_and_cost

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def budget_vs_risk_reduction() -> None:
    sweep = pd.read_csv("data/processed/budget_sweep.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        sweep["total_budget"] / 1e6, sweep["risk_years_reduced"],
        color=BLUE, linewidth=2, marker="o", markersize=6, markerfacecolor=BLUE,
        markeredgewidth=0,
    )
    ax.set_xlabel("5-year capital budget ($M)")
    ax.set_ylabel("Risk-years avoided")
    ax.set_title("Diminishing returns: each extra $25M buys less risk reduction than the last", fontsize=11, color=TEXT_PRIMARY, loc="left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", visible=False)

    # direct label the last point's marginal value for context
    last = sweep.iloc[-1]
    prev = sweep.iloc[-2]
    marg = last["empirical_marginal_value_per_usd"]
    ax.annotate(
        f"marginal value here:\n{marg*1e6:.2f} risk-years per $1M",
        xy=(last["total_budget"] / 1e6, last["risk_years_reduced"]),
        xytext=(-95, -28), textcoords="offset points",
        fontsize=9, color=TEXT_SECONDARY,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "budget_vs_risk_reduction.png", dpi=160)
    plt.close(fig)


def optimal_vs_baseline() -> None:
    df = pd.read_csv("data/processed/bridges.csv")
    df = add_risk_and_cost(df) if "risk_untreated" not in df.columns else df

    T, budget = 5, [20_000_000] * 5
    opt = solve_milp(df, T, budget)
    base_sel = worst_condition_first(df, T, budget)

    opt_val = opt.objective_value
    base_val = risk_years_achieved(df, base_sel, T)
    opt_bridges = opt.x["structure_number"].nunique()
    base_bridges = base_sel["structure_number"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))

    labels = ["Worst-condition-\nfirst (baseline)", "MILP-optimal"]

    ax = axes[0]
    vals = [base_val, opt_val]
    bars = ax.bar(labels, vals, color=[ORANGE, BLUE], width=0.55)
    ax.set_ylabel("Risk-years avoided")
    ax.set_title("Same $100M budget, same 5 years", fontsize=10, color=TEXT_PRIMARY, loc="left")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10, color=TEXT_PRIMARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    vals2 = [base_bridges, opt_bridges]
    bars2 = ax.bar(labels, vals2, color=[ORANGE, BLUE], width=0.55)
    ax.set_ylabel("Bridges treated")
    ax.set_title("Optimisation spreads the same money further", fontsize=10, color=TEXT_PRIMARY, loc="left")
    for b, v in zip(bars2, vals2):
        ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10, color=TEXT_PRIMARY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"Optimisation achieves {(opt_val-base_val)/base_val*100:.0f}% more risk reduction "
        "for the same budget",
        fontsize=12, color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "optimal_vs_baseline.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    budget_vs_risk_reduction()
    optimal_vs_baseline()
    print(f"Wrote figures to {FIG_DIR}")
