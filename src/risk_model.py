"""
Risk and cost scoring for the bridge investment model.

These are deliberately simple, fully disclosed proxies -- not a calibrated structural
failure-probability model. A real engineering risk model would use inspection-level
element condition data, load ratings, and hydraulic/scour analysis; NBI's summary
condition rating is a coarse but standard and widely used stand-in in the academic and
practitioner asset-management literature.

Risk score
----------
risk_i = deficiency_i * criticality_i * 100

  deficiency_i    = (9 - lowest_rating_i) / 7
                    NBI condition ratings run 9 (new) down to 0 (failed); the worst
                    rating actually observed in this dataset is 2, so deficiency is
                    normalised against a 9..2 range -> 0 (rating 9) to 1 (rating 2).

  criticality_i   = log1p(adt_i) / log1p(adt_max)
                    Traffic volume (ADT) is heavily right-skewed (a handful of bridges
                    carry >100,000 vehicles/day, most carry a few thousand), so it is
                    log-scaled before normalising 0..1 -- otherwise a few high-traffic
                    bridges would swamp the ranking and everything else would look
                    equally low-risk by comparison.

If bridge i is rehabilitated, its condition is assumed to be restored to rating 8
("very good", one below new -- a realistic post-rehab state, not "as new"), giving a
residual deficiency of (9-8)/7 = 1/7 and a residual risk of:

  risk_treated_i  = (1/7) * criticality_i * 100

Cost model
----------
cost_i = deck_area_sqm_i * SQM_TO_SQFT * unit_rate_i

  unit_rate_i is FHWA's own published 2024 Bridge Replacement Unit Cost:
    - $738/sqft for National Highway System (NHS) bridges in Rhode Island specifically
      (FHWA, "Bridge Replacement Unit Costs 2024", https://www.fhwa.dot.gov/bridge/nbi/sd2024.cfm)
    - $382/sqft for non-NHS bridges (FHWA's national non-NHS average for the same year;
      no Rhode-Island-specific non-NHS figure is published)

This treats every intervention as a full replacement-equivalent cost, which is a
deliberately conservative simplification -- a real capital programme would cost a
partial rehabilitation well below full replacement for most bridges. Left as a stated
limitation (see README "Where this is going") rather than an invented discount factor
this project has no data to justify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SQM_TO_SQFT = 10.7639
NHS_UNIT_RATE_USD_PER_SQFT = 738.0
NON_NHS_UNIT_RATE_USD_PER_SQFT = 382.0

WORST_OBSERVED_RATING = 2
BEST_RATING = 9
POST_TREATMENT_RATING = 8


def deficiency(lowest_rating: pd.Series) -> pd.Series:
    span = BEST_RATING - WORST_OBSERVED_RATING
    return (BEST_RATING - lowest_rating) / span


def criticality(adt: pd.Series) -> pd.Series:
    """Log-scaled, 0..1-normalised traffic exposure, relative to the maximum ADT in
    the batch passed in. This is a batch-relative score, not an absolute constant per
    bridge -- always score the full state extract together (as data_prep.py -> the
    optimiser does), not a subset, or the scale will shift.
    """
    log_adt = np.log1p(adt)
    denom = log_adt.max()
    if not denom or pd.isna(denom) or denom == 0:
        return pd.Series(0.0, index=adt.index)
    return log_adt / denom


def risk_score(df: pd.DataFrame) -> pd.Series:
    return deficiency(df["lowest_rating"]) * criticality(df["adt"]) * 100


def residual_risk_score(df: pd.DataFrame) -> pd.Series:
    """Risk remaining immediately after a bridge is rehabilitated (condition -> 8).

    Capped at the bridge's *current* (untreated) risk: a bridge already in better
    condition than the assumed post-treatment state (rating 9, "new") would otherwise
    show a residual risk higher than its untreated risk, i.e. "treating" it would look
    like it makes things worse. Capping means such a bridge simply has zero available
    risk_reduction, so the optimiser correctly never selects it for funding.
    """
    post_deficiency = (BEST_RATING - POST_TREATMENT_RATING) / (BEST_RATING - WORST_OBSERVED_RATING)
    hypothetical = post_deficiency * criticality(df["adt"]) * 100
    return np.minimum(hypothetical, risk_score(df))


def rehab_cost(df: pd.DataFrame) -> pd.Series:
    unit_rate = np.where(df["is_nhs"], NHS_UNIT_RATE_USD_PER_SQFT, NON_NHS_UNIT_RATE_USD_PER_SQFT)
    return df["deck_area_sqm"] * SQM_TO_SQFT * unit_rate


def add_risk_and_cost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_untreated"] = risk_score(df)
    df["risk_treated"] = residual_risk_score(df)
    df["risk_reduction"] = df["risk_untreated"] - df["risk_treated"]
    df["cost_usd"] = rehab_cost(df)
    return df
