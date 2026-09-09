"""
Load and clean the Rhode Island extract of FHWA's National Bridge Inventory (NBI),
and export a tidy, model-ready table to data/processed/bridges.csv.

Source: FHWA National Bridge Inventory, 2025 delimited extract for Rhode Island
(RI25.txt / RI25.csv), https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm

This is real government data, not synthetic. It has the quirks real data has:
  - "N" (not applicable) in DECK_COND_058 / SUPERSTRUCTURE_COND_059 / SUBSTRUCTURE_COND_060
    for culverts, which are rated on CULVERT_COND_062 instead. FHWA's own LOWEST_RATING
    column already resolves this (verified below against the raw fields, not just trusted
    blindly), so it is used directly as the condition input.
  - A handful of bridges (3 of 787) have ADT_029 == 0 on roads that obviously carry
    real traffic (e.g. "US 1 SB POST RD") -- a data gap, not a literal zero. These are
    treated as missing and imputed with the state median ADT.
  - STRUCTURE_NUMBER_008 is inconsistently formatted across bridges (e.g. "1RI0668" vs
    "000000000000010") -- kept as an opaque string identifier, never parsed as a number.

Run directly to (re)generate the processed file:
    python -m src.data_prep
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "RI25.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "bridges.csv"

CONDITION_COLS = [
    "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060",
    "CULVERT_COND_062",
]


def _recompute_lowest_rating(df: pd.DataFrame) -> pd.Series:
    """Recompute FHWA's LOWEST_RATING from the raw condition columns, treating "N"
    (not applicable) as absent rather than zero. Used only to *validate* that we
    understand FHWA's own field correctly before relying on it -- see tests/test_data_prep.py.
    """
    numeric = df[CONDITION_COLS].apply(pd.to_numeric, errors="coerce")
    return numeric.min(axis=1)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()

    df["structure_number"] = df_raw["STRUCTURE_NUMBER_008"].str.strip()

    # Condition: use FHWA's own LOWEST_RATING (already resolves the deck/superstructure/
    # substructure vs. culvert split correctly -- confirmed in tests/test_data_prep.py).
    df["lowest_rating"] = pd.to_numeric(df_raw["LOWEST_RATING"], errors="coerce")
    df["bridge_condition"] = df_raw["BRIDGE_CONDITION"]  # G / F / P, kept for EDA/sanity checks

    # Traffic exposure. A handful of records have ADT_029 == 0 on roads that plainly carry
    # real traffic -- a data gap, not a literal reading. Treat 0 as missing and impute with
    # the state median so a handful of bad records don't get zero criticality weight.
    adt = pd.to_numeric(df_raw["ADT_029"], errors="coerce")
    n_zero_or_missing = int(((adt == 0) | adt.isna()).sum())
    adt_median = adt[adt > 0].median()
    df["adt"] = adt.replace(0, np.nan).fillna(adt_median)
    df["adt_imputed"] = (adt == 0) | adt.isna()

    # Cost driver: FHWA's own computed DECK_AREA (sqm) -- cross-checked in tests against
    # STRUCTURE_LEN_MT_049 x DECK_WIDTH_MT_052.
    df["deck_area_sqm"] = pd.to_numeric(df_raw["DECK_AREA"], errors="coerce")

    # National Highway System flag drives which FHWA-published unit cost applies
    # (see src/risk_model.py).
    df["is_nhs"] = df_raw["NATIONAL_NETWORK_110"] == "1"

    # Kept for EDA
    df["year_built"] = pd.to_numeric(df_raw["YEAR_BUILT_027"], errors="coerce")
    df["functional_class"] = df_raw["FUNCTIONAL_CLASS_026"]
    df["owner_code"] = df_raw["OWNER_022"]
    df["facility_carried"] = df_raw["FACILITY_CARRIED_007"].str.strip("' ")
    df["features_desc"] = df_raw["FEATURES_DESC_006A"].str.strip("' ")

    # Drop the small number of rows missing a condition rating or deck area entirely --
    # there's nothing the risk/cost model can do with those.
    before = len(df)
    df = df.dropna(subset=["lowest_rating", "deck_area_sqm"]).reset_index(drop=True)
    dropped = before - len(df)

    print(f"[data_prep] {before} raw rows -> {len(df)} after cleaning ({dropped} dropped for "
          f"missing condition/deck area); {n_zero_or_missing} ADT values imputed with median "
          f"{adt_median:,.0f}.")

    return df


def main() -> None:
    df_raw = load_raw()
    df_clean = clean(df_raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False, lineterminator="\n")
    print(f"[data_prep] wrote {len(df_clean)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
