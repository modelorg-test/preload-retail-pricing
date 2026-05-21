"""Batch inference for the Dynamic Pricing Engine.

Scores each SKU-day combination and enforces hard margin floors by
category before publishing recommendations to the pricing engine.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import MIN_MARGIN_FLOORS, build_feature_matrix

logger = logging.getLogger(__name__)


def enforce_margin_floor(price: float, cost: float, category: str) -> float:
    """Clamp the recommended price to satisfy the minimum margin floor.

    Args:
        price: Model-recommended price.
        cost: Unit cost of the SKU.
        category: Product category key.

    Returns:
        Adjusted price satisfying the margin floor constraint.
    """
    floor_pct = MIN_MARGIN_FLOORS.get(category, 0.10)
    min_price = cost / (1.0 - floor_pct)
    return max(price, min_price)


def score_catalogue(
    model: lgb.Booster,
    skus: pd.DataFrame,
) -> pd.DataFrame:
    """Score the full catalogue and apply margin floor constraints.

    Args:
        model: Fitted LightGBM booster.
        skus: Raw SKU pricing DataFrame (same schema as training).

    Returns:
        DataFrame with optimal_price, elasticity_coefficient,
        and margin_impact columns.
    """
    df = build_feature_matrix(skus.copy())
    feature_cols = model.feature_name()
    X = df[feature_cols]

    log_demand_pred = model.predict(X)
    demand_pred = np.expm1(log_demand_pred)

    # Estimate elasticity coefficient via finite difference
    delta = 0.01
    df_up = df.copy()
    df_up["log_price"] = np.log1p(np.expm1(df["log_price"]) * (1 + delta))
    log_demand_up = model.predict(df_up[feature_cols])
    elasticity = (log_demand_up - log_demand_pred) / delta  # d(ln Q)/d(ln P)

    result = skus.copy()
    result["optimal_price"] = [
        enforce_margin_floor(
            float(skus.iloc[i]["base_price"]),
            float(skus.iloc[i].get("unit_cost", 0)),
            str(skus.iloc[i].get("category", "")),
        )
        for i in range(len(skus))
    ]
    result["elasticity_coefficient"] = elasticity
    result["margin_impact"] = (result["optimal_price"] - skus["base_price"]) / skus["base_price"]
    return result[["sku_id", "optimal_price", "elasticity_coefficient", "margin_impact"]]
