"""
Spatial statistics: where does each plan spend, is risk clustered, and where is
the risk that is left after each plan?

Writes data/processed/spatial_summary.csv (long format: section, plan, group,
metric, value) and prints a readable summary.

Run: python -m src.spatial_analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.spatial_data import PROCESSED_DIR, to_geodataframe

# Added by hand from the FIPS codes for Rhode Island; the NBI file only has the code.
COUNTY_NAMES = {
    "001": "Bristol",
    "003": "Kent",
    "005": "Newport",
    "007": "Providence",
    "009": "Washington",
}
PLANS = {"optimal": "optimal_year", "baseline": "baseline_year"}

K_NEIGHBOURS = 8
N_PERMUTATIONS = 999
HOTSPOT_FRACTION = 0.10  # top 10% of bridges by remaining risk
DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS_METRES = [2000, 3000, 5000]
SEED = 0


def remaining_risk(df: pd.DataFrame, year_col: str) -> pd.Series:
    """Risk left after a plan: treated bridges drop to risk_treated, the rest stay put."""
    return df["risk_treated"].where(df[year_col].notna(), df["risk_untreated"])


def county_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per plan and county: picks, spend, risk reduction, and shares vs share of untreated risk."""
    county = df["county_code"].map(COUNTY_NAMES)
    untreated_share = df.groupby(county)["risk_untreated"].sum()
    untreated_share = untreated_share / untreated_share.sum()
    rows = []
    for plan, col in PLANS.items():
        picked = df[df[col].notna()]
        g = picked.groupby(picked["county_code"].map(COUNTY_NAMES))
        t = pd.DataFrame(
            {
                "bridges_selected": g.size(),
                "spend_usd": g["cost_usd"].sum(),
                "risk_reduction": g["risk_reduction"].sum(),
            }
        ).reindex(list(COUNTY_NAMES.values()), fill_value=0)
        t["share_of_spend"] = t["spend_usd"] / t["spend_usd"].sum()
        t["share_of_risk_reduction"] = t["risk_reduction"] / t["risk_reduction"].sum()
        t["share_of_untreated_risk"] = untreated_share.reindex(t.index)
        t["bridges_in_county"] = county.value_counts().reindex(t.index)
        t.insert(0, "plan", plan)
        rows.append(t.rename_axis("county").reset_index())
    return pd.concat(rows, ignore_index=True)


def morans_i(
    values: np.ndarray,
    xy: np.ndarray,
    k: int = K_NEIGHBOURS,
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, float]:
    """Moran's I with k-nearest-neighbour, row-standardised weights and a permutation p-value.

    With row-standardised weights the sum of all weights equals n, so
    I = z'Wz / z'z where z is the mean-centred value. Each bridge's neighbours
    are its k closest bridges in metres, not a fixed distance, so dense
    Providence and sparse rural areas each get exactly k neighbours.
    The p-value is two-sided: the share of random reshuffles of the values
    whose I is at least as far from E[I] as the observed I.
    """
    n = len(values)
    # k + 1 because the nearest neighbour of each point is itself.
    idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    z = np.asarray(values, dtype=float) - np.mean(values)

    def stat(zz: np.ndarray) -> float:
        # row-standardised: each neighbour has weight 1/k, so the weighted sum is the neighbour average
        return float(zz @ zz[idx].mean(axis=1) / (zz @ zz))

    observed = stat(z)
    expected = -1 / (n - 1)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm - expected) >= abs(observed - expected)) + 1) / (n_perm + 1)
    return {"I": observed, "expected_I": expected, "p_value": float(p)}


def hotspot_labels(xy: np.ndarray, eps_m: float, min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """DBSCAN cluster labels (-1 = noise) for points in metres."""
    return DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xy)


def hotspot_summary(gdf: pd.DataFrame, year_col: str, eps_m: float) -> dict[str, object]:
    """Cluster the top 10% of bridges by remaining risk; report cluster sizes and noise."""
    rem = remaining_risk(gdf, year_col)
    n_top = int(round(HOTSPOT_FRACTION * len(gdf)))
    top = gdf.loc[rem.nlargest(n_top).index]
    labels = hotspot_labels(top[["x_m", "y_m"]].to_numpy(), eps_m)
    sizes = pd.Series(labels[labels >= 0]).value_counts().sort_values(ascending=False)
    return {
        "n_top": n_top,
        "n_clusters": len(sizes),
        "cluster_sizes": sizes.tolist(),
        "n_noise": int((labels == -1).sum()),
    }


def run(gdf: pd.DataFrame) -> pd.DataFrame:
    """Compute every statistic and return the long-format summary table."""
    xy = gdf[["x_m", "y_m"]].to_numpy()
    out: list[dict[str, object]] = []

    def add(section: str, plan: str, group: str, metric: str, value: object) -> None:
        out.append({"section": section, "plan": plan, "group": group, "metric": metric, "value": value})

    ct = county_table(gdf)
    for _, r in ct.iterrows():
        for m in ct.columns.drop(["plan", "county"]):
            add("county", r["plan"], r["county"], m, r[m])

    series = {"untreated": ("none", gdf["risk_untreated"])}
    for plan, col in PLANS.items():
        series[f"remaining after {plan}"] = (plan, remaining_risk(gdf, col))
    for label, (plan, s) in series.items():
        for metric, v in morans_i(s.to_numpy(), xy).items():
            add("morans_i", plan, label, metric, v)

    for plan, col in PLANS.items():
        for eps in DBSCAN_EPS_METRES:
            h = hotspot_summary(gdf, col, eps)
            add("hotspots", plan, f"eps={eps}", "n_clusters", h["n_clusters"])
            add("hotspots", plan, f"eps={eps}", "cluster_sizes", " ".join(map(str, h["cluster_sizes"])))
            add("hotspots", plan, f"eps={eps}", "n_noise", h["n_noise"])
            add("hotspots", plan, f"eps={eps}", "n_top", h["n_top"])
    return pd.DataFrame(out)


def print_summary(summary: pd.DataFrame, gdf: pd.DataFrame) -> None:
    """Readable console view of the summary table."""
    ct = county_table(gdf)
    pd.options.display.float_format = "{:,.3f}".format
    print("COUNTY TABLE (county names mapped by hand from FIPS codes, not read from the file)")
    for plan, t in ct.groupby("plan"):
        t = t.drop(columns="plan").set_index("county")
        t["spend_$M"] = t.pop("spend_usd") / 1e6
        print(f"\n[{plan}]")
        print(t.to_string())
        print(f"  total: {t['bridges_selected'].sum()} bridges, ${t['spend_$M'].sum():,.1f}M, "
              f"risk reduction {t['risk_reduction'].sum():,.1f}")

    m = summary[summary["section"] == "morans_i"].pivot(index="group", columns="metric", values="value")
    print(f"\nMORAN'S I (k={K_NEIGHBOURS} nearest neighbours, {N_PERMUTATIONS} permutations, two-sided)")
    print(m[["I", "expected_I", "p_value"]].to_string())

    h = summary[summary["section"] == "hotspots"]
    print(f"\nHOTSPOTS: DBSCAN on top {HOTSPOT_FRACTION:.0%} by remaining risk, min_samples={DBSCAN_MIN_SAMPLES}")
    for (plan, group), g in h.groupby(["plan", "group"], sort=False):
        v = g.set_index("metric")["value"]
        print(f"  {plan:9s} {group}: {v['n_clusters']} clusters, sizes [{v['cluster_sizes']}], "
              f"{v['n_noise']} of {v['n_top']} unclustered")


def main() -> None:
    df = pd.read_csv(
        PROCESSED_DIR / "bridges_geo.csv", dtype={"structure_number": str, "county_code": str}
    )
    gdf = to_geodataframe(df)
    summary = run(gdf)
    summary.to_csv(PROCESSED_DIR / "spatial_summary.csv", index=False, lineterminator="\n")
    print_summary(summary, gdf)
    print("\nwrote spatial_summary.csv")


if __name__ == "__main__":
    main()
