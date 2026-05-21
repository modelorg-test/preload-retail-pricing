"""Feature engineering for the Dynamic Pricing Engine.

Constructs elasticity features from SKU price history, inventory
levels, competitor data, and demand forecasts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_MARGIN_FLOORS: dict[str, float] = {
    "electronics": 0.15,
    "fashion": 0.25,
    "home_garden": 0.20,
    "fmcg": 0.10,
}


def compute_price_delta(df: pd.DataFrame, base_price_col: str = "base_price") -> pd.DataFrame:
    """Compute percentage price delta relative to the SKU's rolling 7-day average.

    Args:
        df: DataFrame with base_price_col and sku_id columns.
        base_price_col: Column name for the current recommended price.

    Returns:
        DataFrame with price_delta column appended.
    """
    result = df.copy()
    rolling_avg = (
        df.groupby("sku_id")[base_price_col]
        .transform(lambda s: s.rolling(7, min_periods=1).mean().shift(1))
    )
    result["price_delta"] = (df[base_price_col] - rolling_avg) / rolling_avg.replace(0, np.nan)
    result["price_delta"] = result["price_delta"].fillna(0.0)
    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix for LightGBM training.

    Expected input columns: sku_id, base_price, competitor_price,
    demand_index, inventory_days, category, date.

    Args:
        df: Raw pricing DataFrame.

    Returns:
        DataFrame with engineered features and category encoding.
    """
    df = compute_price_delta(df)
    df["price_gap"] = df["base_price"] - df["competitor_price"]
    df["log_price"] = np.log1p(df["base_price"])
    df["log_demand"] = np.log1p(df["demand_index"])
    df["inventory_urgency"] = 1.0 / (df["inventory_days"].clip(lower=1))
    df["category_code"] = df["category"].astype("category").cat.codes
    return df
