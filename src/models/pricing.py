"""Dynamic pricing model definition using GradientBoostingRegressor.

Predicts optimal price points that maximise revenue while
maintaining a minimum margin floor constraint.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import PriceRatioComputer

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "base_price", "competitor_price", "cost_of_goods",
    "units_sold_7d", "units_sold_30d", "inventory_days",
    "category_avg_price", "brand_premium_index",
    "seasonality_index", "customer_segment_score",
    "channel_mix_online_pct",
]

PASSTHROUGH_FEATURES = [
    "price_ratio", "margin_pct", "is_promotional",
    "days_since_last_promo",
]

# Minimum gross margin — prices below this floor are clamped
MARGIN_FLOOR = 0.15


def build_pricing_pipeline(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Build the pricing optimisation pipeline.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth per tree.
    learning_rate : float
        Shrinkage rate.
    random_state : int
        Random seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit regression pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("price_ratios", PriceRatioComputer()),
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    min_samples_leaf=20,
                    random_state=random_state,
                    loss="huber",
                    alpha=0.9,
                ),
            ),
        ]
    )

    logger.info(
        "Built pricing pipeline: n_estimators=%d, max_depth=%d",
        n_estimators, max_depth,
    )
    return pipeline


def apply_margin_floor(
    predicted_price: np.ndarray,
    cost_of_goods: np.ndarray,
    floor: float = MARGIN_FLOOR,
) -> np.ndarray:
    """Enforce the minimum margin floor on predicted prices.

    Parameters
    ----------
    predicted_price : np.ndarray
        Model-predicted optimal prices.
    cost_of_goods : np.ndarray
        Cost of goods for each product.
    floor : float
        Minimum gross margin (e.g., 0.15 = 15%).

    Returns
    -------
    np.ndarray
        Adjusted prices with floor enforced.
    """
    min_price = cost_of_goods / (1 - floor)
    return np.maximum(predicted_price, min_price)
