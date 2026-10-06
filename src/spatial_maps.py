"""
Static maps of Rhode Island bridges: where the risk is, and where each plan spends.

Plain longitude/latitude scatter (no basemap) with an equal aspect ratio scaled
by cos(latitude), so distances look right at Rhode Island's latitude.

Also writes figures/bridge_plan_map.html, an interactive folium map.

Run: python -m src.spatial_maps
"""

from __future__ import annotations

import html
import math

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.make_figures import BLUE, FIG_DIR, GRID, ORANGE, TEXT_PRIMARY, TEXT_SECONDARY
from src.spatial_data import PLAN_LABELS, PROCESSED_DIR

GREY = "#c9c8c3"
# Colour-blind-safe (Okabe-Ito) hues: blue / orange keep the repo's plan colours.
PLAN_COLOURS = {
    "both": "#009E73",
    "optimal only": BLUE,
    "baseline only": ORANGE,
    "neither": GREY,
}
YEAR_COLOURS = ["#fde725", "#7ad151", "#22a884", "#2a788e", "#414487"]  # viridis, years 1-5
DPI = 200


def _style(ax: plt.Axes, title: str) -> None:
    """Equal-scale lon/lat axes with a left-aligned title and no clutter."""
    ax.set_aspect(1 / math.cos(math.radians(41.6)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=DPI)
    plt.close(fig)


def _legend(ax: plt.Axes, labels: dict[str, str], colours: dict[str, str]) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=colours[k],
               markeredgewidth=0, label=text)
        for k, text in labels.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")


def risk_map(df: pd.DataFrame) -> None:
    """All bridges coloured by untreated risk (highest drawn last, on top)."""
    d = df.sort_values("risk_untreated")
    vmax = d["risk_untreated"].quantile(0.98)  # clip so a few extremes don't wash out the rest
    fig, ax = plt.subplots(figsize=(6.5, 7))
    sc = ax.scatter(d["lon"], d["lat"], c=d["risk_untreated"], cmap="YlOrRd", vmin=0, vmax=vmax,
                    s=14, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Untreated risk score (colour capped at 98th percentile)", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Untreated risk across Rhode Island's 787 bridges")
    _save(fig, "risk_map.png")


def plan_comparison(df: pd.DataFrame) -> None:
    """Two panels: optimal picks and baseline picks over greyed-out unselected bridges."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, col, colour, name in [
        (axes[0], "optimal_year", BLUE, "Optimal (MILP) plan"),
        (axes[1], "baseline_year", ORANGE, "Worst-condition-first plan"),
    ]:
        picked = df[col].notna()
        ax.scatter(df.loc[~picked, "lon"], df.loc[~picked, "lat"], s=8, color=GREY, edgecolors="none")
        ax.scatter(df.loc[picked, "lon"], df.loc[picked, "lat"], s=20, color=colour, edgecolors="none")
        _style(ax, f"{name}: {picked.sum()} bridges treated")
    axes[1].set_ylabel("")
    _save(fig, "plan_comparison.png")


def plan_difference(df: pd.DataFrame) -> None:
    """One panel, four categories: which plan treats each bridge."""
    fig, ax = plt.subplots(figsize=(6.5, 7))
    for plan in ["neither", "baseline only", "both", "optimal only"]:  # draw the small groups on top
        d = df[df["plan"] == plan]
        ax.scatter(d["lon"], d["lat"], s=8 if plan == "neither" else 26,
                   color=PLAN_COLOURS[plan], edgecolors="none")
    _style(ax, "Which plan treats each bridge")
    counts = df["plan"].value_counts()
    _legend(ax, {p: f"{p} ({counts.get(p, 0)})" for p in PLAN_LABELS}, PLAN_COLOURS)
    _save(fig, "plan_difference.png")


def optimal_by_year(df: pd.DataFrame) -> None:
    """Optimal-plan picks coloured by treatment year (1 = first)."""
    picked = df[df["optimal_year"].notna()]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.scatter(df["lon"], df["lat"], s=8, color=GREY, edgecolors="none")
    year_colours = {str(y): YEAR_COLOURS[y - 1] for y in range(1, 6)}
    for y in range(1, 6):
        d = picked[picked["optimal_year"] == y]
        ax.scatter(d["lon"], d["lat"], s=26, color=year_colours[str(y)], edgecolors="none")
    _style(ax, "Optimal plan: year each bridge is treated")
    _legend(ax, {str(y): f"Year {y} ({(picked['optimal_year'] == y).sum()})" for y in range(1, 6)},
            year_colours)
    _save(fig, "optimal_by_year.png")


def _popup(r: pd.Series) -> str:
    """HTML popup for one bridge. Text from the file is escaped so odd characters can't break the page."""
    def year(col: str) -> str:
        return f"year {int(r[col])}" if pd.notna(r[col]) else "not selected"

    adt = f"{r['adt']:,.0f}" + (" (imputed)" if r["adt_imputed"] else "")
    return (
        f"<b>{html.escape(r['structure_number'])}</b><br>"
        f"Route: {html.escape(str(r['route']))}<br>"
        f"Condition: {html.escape(str(r['bridge_condition']))} (lowest rating {int(r['lowest_rating'])})<br>"
        f"Traffic (ADT): {adt}<br>"
        f"Cost: ${r['cost_usd'] / 1e6:,.2f}M<br>"
        f"Optimal plan: {year('optimal_year')}<br>"
        f"Baseline plan: {year('baseline_year')}"
    )


def interactive_map(df: pd.DataFrame) -> None:
    """Folium map: points coloured by plan category, with toggleable optimal / baseline layers.

    A bridge picked by both plans sits in both layers (same colour), so either
    layer alone shows everything that plan treats.
    """
    m = folium.Map(tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m)  # base tiles; not a toggle
    layers = {
        "Optimal plan picks": df["optimal_year"].notna(),
        "Baseline plan picks": df["baseline_year"].notna(),
        "Not selected by either plan": df["plan"] == "neither",
    }
    for name, mask in layers.items():
        group = folium.FeatureGroup(name=f"{name} ({int(mask.sum())})", show=True)
        for _, r in df[mask].iterrows():
            colour = PLAN_COLOURS[r["plan"]]
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=3 if r["plan"] == "neither" else 6,
                color=colour, fill=True, fill_color=colour, fill_opacity=0.85, weight=1,
                popup=folium.Popup(_popup(r), max_width=280),
            ).add_to(group)
        group.add_to(m)
    m.fit_bounds([[df["lat"].min(), df["lon"].min()], [df["lat"].max(), df["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)

    counts = df["plan"].value_counts()
    rows = "".join(
        f'<div><span style="color:{PLAN_COLOURS[p]}">&#9679;</span> {p} ({counts.get(p, 0)})</div>'
        for p in PLAN_LABELS
    )
    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #ccc;border-radius:4px;font-size:13px">'
        f"<b>Which plan treats it</b>{rows}</div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    m.save(FIG_DIR / "bridge_plan_map.html")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "bridges_geo.csv", dtype={"structure_number": str})
    for fn in (risk_map, plan_comparison, plan_difference, optimal_by_year, interactive_map):
        fn(df)
        print(f"wrote {fn.__name__}")


if __name__ == "__main__":
    main()
