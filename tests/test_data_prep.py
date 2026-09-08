import pandas as pd
import pytest

from src.data_prep import RAW_PATH, clean, load_raw


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


def test_raw_file_has_expected_shape(raw_df):
    # Rhode Island's 2025 NBI extract: 787 bridges, 123 NBI-standard columns.
    assert raw_df.shape == (787, 123)


def test_lowest_rating_matches_fhwas_own_field(raw_df):
    """FHWA's own LOWEST_RATING column should equal min(deck, superstructure,
    substructure, culvert) once "N" (not applicable) is treated as absent. This is
    the check that justifies trusting LOWEST_RATING directly in the risk model,
    instead of re-deriving it (and risking getting the culvert-vs-deck logic wrong).
    """
    cond_cols = ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062"]
    numeric = raw_df[cond_cols].apply(pd.to_numeric, errors="coerce")
    recomputed = numeric.min(axis=1)
    official = pd.to_numeric(raw_df["LOWEST_RATING"], errors="coerce")
    assert (recomputed == official).all()


def test_no_rows_dropped_for_this_extract(clean_df):
    # True for the RI25 extract specifically -- every bridge has a usable condition
    # rating and deck area. Documented as a fact about this dataset, not assumed in
    # general (a different state's extract could easily have gaps).
    assert len(clean_df) == 787


def test_zero_adt_bridges_are_imputed_not_zero(raw_df, clean_df):
    zero_adt_structures = raw_df.loc[pd.to_numeric(raw_df["ADT_029"], errors="coerce") == 0, "STRUCTURE_NUMBER_008"].str.strip()
    assert len(zero_adt_structures) == 3
    imputed_rows = clean_df[clean_df["structure_number"].isin(zero_adt_structures)]
    assert (imputed_rows["adt"] > 0).all()
    assert imputed_rows["adt_imputed"].all()


def test_deck_area_matches_length_times_width(raw_df):
    """Sanity-check FHWA's own DECK_AREA field against length x width, for the rows
    where both are populated -- confirms we understand the units (both metres).
    """
    length = pd.to_numeric(raw_df["STRUCTURE_LEN_MT_049"], errors="coerce")
    width = pd.to_numeric(raw_df["DECK_WIDTH_MT_052"], errors="coerce")
    area = pd.to_numeric(raw_df["DECK_AREA"], errors="coerce")
    mask = width > 0  # a handful of culverts have 0 recorded deck width
    implied = length[mask] * width[mask]
    assert (implied.sub(area[mask]).abs() < 0.5).mean() > 0.95  # matches within rounding for 95%+ of rows


def test_structure_number_is_never_parsed_as_numeric(clean_df):
    assert not pd.api.types.is_numeric_dtype(clean_df["structure_number"])
    # Formats genuinely differ across rows (e.g. "1RI0668" vs "000000000000010") --
    # if this were ever numeric, leading zeros would silently collapse and ids collide.
    assert clean_df["structure_number"].str.match(r"^[A-Za-z0-9]+$").all()
