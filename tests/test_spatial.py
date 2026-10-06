import math

import pandas as pd
import pytest

from src.spatial_data import (
    PLAN_LABELS,
    PROCESSED_DIR,
    build_bridges_geo,
    dms_to_decimal,
    to_geodataframe,
)


@pytest.fixture(scope="module")
def geo():
    return build_bridges_geo()


def test_decoder_known_values():
    assert dms_to_decimal("41302330", 2) == pytest.approx(41.50647, abs=1e-5)
    assert dms_to_decimal("071193090", 3) == pytest.approx(71.32525, abs=1e-5)


@pytest.mark.parametrize(
    "bad", ["", "4130233", "413023300", "4130233x", "41.02330", " 41302330", "41602330", "41306030", None, float("nan")]
)
def test_decoder_rejects_bad_input(bad):
    assert math.isnan(dms_to_decimal(bad, 2))


def test_all_bridges_decode(geo):
    assert len(geo) == 787
    assert geo[["lat", "lon"]].notna().all().all()


def test_points_inside_rhode_island_box(geo):
    assert geo["lat"].between(41.1, 42.05).all()
    assert geo["lon"].between(-71.9, -71.1).all()


def test_merge_is_one_row_per_bridge(geo):
    assert len(geo) == 787
    assert geo["structure_number"].is_unique


def test_plan_counts_match_selection_files(geo):
    for name in ("optimal", "baseline"):
        sel = pd.read_csv(PROCESSED_DIR / f"{name}_selection.csv")
        assert geo[f"{name}_year"].notna().sum() == (sel["selected"] == 1).sum()


def test_plan_categories_partition_bridges(geo):
    assert set(geo["plan"]) <= set(PLAN_LABELS)
    assert sum((geo["plan"] == p).sum() for p in PLAN_LABELS) == len(geo)


def test_geodataframe_has_metric_columns(geo):
    gdf = to_geodataframe(geo)
    assert gdf.crs.to_epsg() == 4326
    assert gdf[["x_m", "y_m"]].notna().all().all()
    assert gdf["x_m"].between(250_000, 400_000).all()  # Rhode Island in UTM 19N
