import pandas as pd
import pytest

from src.risk_model import add_risk_and_cost, criticality, deficiency


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "structure_number": ["A", "B", "C"],
        "lowest_rating": [9, 2, 5],   # best possible, worst observed, middle
        "adt": [100, 100, 50000],
        "deck_area_sqm": [100.0, 100.0, 100.0],
        "is_nhs": [False, False, True],
    })


def test_deficiency_bounds():
    d = deficiency(pd.Series([9, 2]))
    assert d.iloc[0] == pytest.approx(0.0)   # best rating -> zero deficiency
    assert d.iloc[1] == pytest.approx(1.0)   # worst observed rating -> full deficiency


def test_criticality_bounds():
    c = criticality(pd.Series([0, 236230]))  # min and max ADT observed in the RI extract
    assert c.iloc[0] == pytest.approx(0.0)
    assert c.iloc[1] == pytest.approx(1.0)


def test_risk_is_zero_only_at_best_condition_and_zero_traffic():
    df = pd.DataFrame({
        "structure_number": ["Z"], "lowest_rating": [9], "adt": [0],
        "deck_area_sqm": [100.0], "is_nhs": [False],
    })
    out = add_risk_and_cost(df)
    assert out["risk_untreated"].iloc[0] == pytest.approx(0.0)


def test_worse_condition_or_more_traffic_means_more_risk(sample_df):
    out = add_risk_and_cost(sample_df)
    # B (worst condition, same traffic as A) should have strictly higher risk than A.
    assert out.loc[1, "risk_untreated"] > out.loc[0, "risk_untreated"]
    # C (much higher traffic, better condition than B) should still register real risk.
    assert out.loc[2, "risk_untreated"] > 0


def test_treated_risk_never_exceeds_untreated_risk(sample_df):
    out = add_risk_and_cost(sample_df)
    assert (out["risk_treated"] <= out["risk_untreated"]).all()
    assert (out["risk_reduction"] >= 0).all()


def test_nhs_bridges_cost_more_per_sqm_than_non_nhs(sample_df):
    out = add_risk_and_cost(sample_df)
    # A and C have identical deck area; C is NHS and should cost more per the
    # FHWA-published unit rates (738 vs 382 $/sqft).
    assert out.loc[2, "cost_usd"] > out.loc[0, "cost_usd"]


def test_cost_is_positive_and_finite(sample_df):
    out = add_risk_and_cost(sample_df)
    assert (out["cost_usd"] > 0).all()
    assert out["cost_usd"].notna().all()
