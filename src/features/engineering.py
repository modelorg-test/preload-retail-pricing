"""Feature engineering for the dynamic pricing model.

Computes demand-side features including price elasticity proxies,
seasonal indicators, competitor positioning, and promotional flags.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

PRICING_FEATURES = [
    "base_price",
    "competitor_price",
    "price_ratio",
    "cost_of_goods",
    "margin_pct",
    "units_sold_7d",
    "units_sold_30d",
    "inventory_days",
    "category_avg_price",
    "brand_premium_index",
    "seasonality_index",
    "is_promotional",
    "days_since_last_promo",
    "customer_segment_score",
    "channel_mix_online_pct",
]


class PriceRatioComputer(BaseEstimator, TransformerMixin):
    """Compute relative price positioning features.

    Creates ratio-based features that capture the product's price
    position relative to competitors and category averages.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PriceRatioComputer:
        """No-op fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived pricing ratio features."""
        result = X.copy()

        if "base_price" in result.columns and "competitor_price" in result.columns:
            result["price_gap"] = result["base_price"] - result["competitor_price"]
            result["price_gap_pct"] = (
                result["price_gap"] / result["competitor_price"].clip(lower=0.01)
            )

        if "base_price" in result.columns and "category_avg_price" in result.columns:
            result["category_position"] = (
                result["base_price"] / result["category_avg_price"].clip(lower=0.01)
            )

        if "base_price" in result.columns and "cost_of_goods" in result.columns:
            result["gross_margin"] = (
                (result["base_price"] - result["cost_of_goods"])
                / result["base_price"].clip(lower=0.01)
            )

        if "units_sold_7d" in result.columns and "units_sold_30d" in result.columns:
            result["demand_velocity"] = (
                result["units_sold_7d"] * 30 / 7
            ) / result["units_sold_30d"].clip(lower=1)

        return result
