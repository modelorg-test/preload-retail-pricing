"""Training pipeline for the dynamic retail pricing model.

Generates synthetic pricing data using sklearn's make_regression,
trains the GradientBoosting pipeline, and evaluates with
pricing-specific metrics (RMSE, revenue uplift, margin compliance).

Usage::

    python -m src.pipelines.train
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.engineering import PRICING_FEATURES
from src.models.pricing import MARGIN_FLOOR, apply_margin_floor, build_pricing_pipeline

logger = logging.getLogger(__name__)


def generate_pricing_dataset(
    n_samples: int = 30000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic retail pricing dataset.

    Parameters
    ----------
    n_samples : int
        Number of product-day observations.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (X, y)
        X: DataFrame with pricing features.
        y: Series of optimal price targets.
    """
    n_features = len(PRICING_FEATURES)

    X_raw, y_raw = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 3,
        noise=15.0,
        random_state=random_state,
    )

    X = pd.DataFrame(X_raw, columns=PRICING_FEATURES)

    # Scale to realistic pricing ranges
    scalers = {
        "base_price": (5.0, 500.0),
        "competitor_price": (4.0, 520.0),
        "price_ratio": (0.5, 1.5),
        "cost_of_goods": (2.0, 350.0),
        "margin_pct": (0.05, 0.60),
        "units_sold_7d": (0, 500),
        "units_sold_30d": (0, 2000),
        "inventory_days": (1, 90),
        "category_avg_price": (5.0, 450.0),
        "brand_premium_index": (0.5, 2.0),
        "seasonality_index": (0.5, 1.5),
        "is_promotional": (0, 1),
        "days_since_last_promo": (0, 180),
        "customer_segment_score": (0.0, 1.0),
        "channel_mix_online_pct": (0.0, 1.0),
    }

    for col, (lo, hi) in scalers.items():
        if col in X.columns:
            col_min, col_max = X[col].min(), X[col].max()
            X[col] = lo + (X[col] - col_min) / (col_max - col_min + 1e-8) * (hi - lo)

    for int_col in ["units_sold_7d", "units_sold_30d", "inventory_days",
                     "is_promotional", "days_since_last_promo"]:
        if int_col in X.columns:
            X[int_col] = X[int_col].round().clip(lower=0).astype(int)

    # Target: optimal price (scaled to realistic range)
    y_min, y_max = y_raw.min(), y_raw.max()
    y = pd.Series(
        10.0 + (y_raw - y_min) / (y_max - y_min + 1e-8) * 490.0,
        name="optimal_price",
    )

    logger.info("Generated pricing dataset: %d samples, %d features", n_samples, n_features)
    return X, y


def train(
    n_samples: int = 30000,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Run the pricing model training workflow.

    Returns
    -------
    dict
        Training results with pipeline, metrics, and feature importance.
    """
    X, y = generate_pricing_dataset(n_samples=n_samples, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    pipeline = build_pricing_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Apply margin floor
    if "cost_of_goods" in X_test.columns:
        y_pred_floored = apply_margin_floor(y_pred, X_test["cost_of_goods"].values)
    else:
        y_pred_floored = y_pred

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse_after_floor": float(np.sqrt(mean_squared_error(y_test, y_pred_floored))),
        "margin_violations_pct": float(
            (y_pred < X_test["cost_of_goods"].values / (1 - MARGIN_FLOOR)).mean() * 100
            if "cost_of_goods" in X_test.columns
            else 0.0
        ),
    }

    # Feature importance
    regressor = pipeline.named_steps["regressor"]
    importance = dict(
        zip(PRICING_FEATURES, regressor.feature_importances_[:len(PRICING_FEATURES)])
    )

    logger.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importance": importance,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train dynamic pricing model")
    parser.add_argument("--samples", type=int, default=30000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    results = train(n_samples=args.samples)
    m = results["metrics"]

    print("\n" + "=" * 60)
    print("PRICING MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  RMSE:            {m['rmse']:.2f}")
    print(f"  MAE:             {m['mae']:.2f}")
    print(f"  R²:              {m['r2']:.4f}")
    print(f"  Margin violations: {m['margin_violations_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
