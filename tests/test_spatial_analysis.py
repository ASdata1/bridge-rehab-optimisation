import numpy as np
import pandas as pd
import pytest

from src.spatial_analysis import (
    PLANS,
    county_table,
    hotspot_labels,
    morans_i,
    remaining_risk,
)
from src.spatial_data import build_bridges_geo


@pytest.fixture(scope="module")
def geo():
    return build_bridges_geo()


def _grid(n: int = 12) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_morans_i_positive_on_clustered_grid():
    xy = _grid()
    res = morans_i(xy[:, 0], xy, n_perm=199)  # value rises smoothly left to right
    assert res["I"] > 0.5
    assert res["p_value"] < 0.05


def test_morans_i_near_zero_when_shuffled():
    xy = _grid()
    shuffled = np.random.default_rng(1).permutation(xy[:, 0])
    res = morans_i(shuffled, xy, n_perm=199)
    assert abs(res["I"]) < 0.15
    assert res["expected_I"] == pytest.approx(-1 / (len(xy) - 1))


def test_dbscan_labels_match_input_length():
    xy = _grid()
    assert len(hotspot_labels(xy, eps_m=1.5)) == len(xy)


def test_county_totals_match_overall_totals(geo):
    ct = county_table(geo)
    for plan, col in PLANS.items():
        t = ct[ct["plan"] == plan]
        picked = geo[geo[col].notna()]
        assert t["bridges_selected"].sum() == len(picked)
        assert t["spend_usd"].sum() == pytest.approx(picked["cost_usd"].sum())
        assert t["risk_reduction"].sum() == pytest.approx(picked["risk_reduction"].sum())
        assert t["share_of_spend"].sum() == pytest.approx(1)
    assert ct[ct["plan"] == "optimal"]["share_of_untreated_risk"].sum() == pytest.approx(1)


def test_remaining_risk_only_drops_for_treated_bridges(geo):
    rem = remaining_risk(geo, "optimal_year")
    treated = geo["optimal_year"].notna()
    assert (rem[~treated] == geo.loc[~treated, "risk_untreated"]).all()
    assert (rem[treated] == geo.loc[treated, "risk_treated"]).all()
