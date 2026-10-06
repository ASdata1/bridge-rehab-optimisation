"""
Attach decoded coordinates and plan membership to the processed bridge table.

The NBI stores positions as fixed-width digit strings, not decimal degrees:
  LAT_016   8 digits  DDMMSSss   (41302330 -> 41 deg 30 min 23.30 s N)
  LONG_017  9 digits  DDDMMSSss  (071193090 -> 71 deg 19 min 30.90 s W)
The last two digits are hundredths of a second. Longitude is west of
Greenwich, so it is negated. Structure numbers stay opaque strings (some have
leading zeros or stray whitespace) and are stripped before any join.

Run directly to write data/processed/bridges_geo.csv:
    python -m src.spatial_data
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "RI25.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

LAT_DIGITS = 2  # degree digits in LAT_016
LONG_DIGITS = 3  # degree digits in LONG_017

WGS84 = "EPSG:4326"  # lat/lon in degrees, for plotting
UTM19N = "EPSG:32619"  # metres, for distances (covers Rhode Island)

PLAN_LABELS = ["both", "optimal only", "baseline only", "neither"]


def dms_to_decimal(value: object, degree_digits: int) -> float:
    """Decode an NBI DMS string to decimal degrees (unsigned); NaN if malformed.

    A wrong-width or non-digit string would silently decode to a wrong place,
    so anything not exactly `degree_digits + 6` digits is rejected.
    """
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        return float("nan")
    if len(value) != degree_digits + 6:
        return float("nan")
    d = int(value[:degree_digits])
    m = int(value[degree_digits : degree_digits + 2])
    s = int(value[degree_digits + 2 :]) / 100  # SSss -> seconds with hundredths
    if m >= 60 or s >= 60:
        return float("nan")
    return d + m / 60 + s / 3600


def load_coordinates(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw NBI file as strings and return id, decoded lat/lon, county, route."""
    raw = pd.read_csv(path, dtype=str)
    out = pd.DataFrame(
        {
            "structure_number": raw["STRUCTURE_NUMBER_008"].str.strip(),
            "lat": raw["LAT_016"].str.strip().map(lambda v: dms_to_decimal(v, LAT_DIGITS)),
            "lon": -raw["LONG_017"].str.strip().map(lambda v: dms_to_decimal(v, LONG_DIGITS)),
            "county_code": raw["COUNTY_CODE_003"].str.strip(),
            "route": raw["FEATURES_DESC_006A"].str.strip().str.strip("'"),
        }
    )
    n_bad = int(out[["lat", "lon"]].isna().any(axis=1).sum())
    print(f"coordinates decoded: {len(out) - n_bad} ok, {n_bad} failed")
    return out


def _year_by_bridge(selection: pd.DataFrame) -> pd.Series:
    """Treatment year per selected bridge, indexed by stripped structure number."""
    sel = selection[selection["selected"] == 1].copy()
    sel["structure_number"] = sel["structure_number"].str.strip()
    return sel.set_index("structure_number")["year"]


def plan_category(optimal_year: pd.Series, baseline_year: pd.Series) -> pd.Series:
    """Label each bridge by which plan(s) treat it."""
    opt, base = optimal_year.notna(), baseline_year.notna()
    return pd.Series(
        np.select(
            [opt & base, opt, base],
            ["both", "optimal only", "baseline only"],
            default="neither",
        ),
        index=optimal_year.index,
    )


def build_bridges_geo(processed_dir: Path = PROCESSED_DIR, raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """bridges.csv + coordinates + county/route + optimal/baseline year and plan."""
    bridges = pd.read_csv(processed_dir / "bridges.csv", dtype={"structure_number": str})
    bridges["structure_number"] = bridges["structure_number"].str.strip()
    coords = load_coordinates(raw_path)
    df = bridges.merge(coords, on="structure_number", how="left", validate="one_to_one")

    for name in ("optimal", "baseline"):
        sel = pd.read_csv(processed_dir / f"{name}_selection.csv", dtype={"structure_number": str})
        df[f"{name}_year"] = df["structure_number"].map(_year_by_bridge(sel))
    df["plan"] = plan_category(df["optimal_year"], df["baseline_year"])
    return df


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Points in WGS84, plus x_m / y_m columns in UTM 19N for distance work."""
    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=WGS84
    )
    metric = gdf.geometry.to_crs(UTM19N)
    gdf["x_m"], gdf["y_m"] = metric.x, metric.y
    return gdf


def main() -> None:
    df = build_bridges_geo()
    df.to_csv(PROCESSED_DIR / "bridges_geo.csv", index=False, lineterminator="\n")
    print(f"wrote bridges_geo.csv: {len(df)} rows")
    print(df["plan"].value_counts().reindex(PLAN_LABELS).to_string())


if __name__ == "__main__":
    main()
