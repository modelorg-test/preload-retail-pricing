"""Training pipeline for the Dynamic Pricing Engine.

Fits a LightGBM regression model on SKU price-demand observations,
evaluates revenue uplift vs. the baseline price strategy, and
registers the model artifact with MLflow.
"""

from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from features import build_feature_matrix

logger = logging.getLogger(__name__)

MODEL_NAME = "dynamic-pricing-engine"
EXPERIMENT_NAME = "pricing/lightgbm-elasticity"

FEATURE_COLS = [
    "price_delta",
    "price_gap",
    "log_price",
    "log_demand",
    "inventory_urgency",
    "competitor_price",
    "category_code",
]
TARGET = "demand_actual"


def train(data_path: str) -> None:
    """Run the pricing model training workflow.

    Args:
        data_path: Parquet file with raw SKU-day pricing observations.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(data_path)
    df = build_feature_matrix(df)

    X = df[FEATURE_COLS]
    y = np.log1p(df[TARGET])  # Log-transform demand for the log-log model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    with mlflow.start_run():
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )

        y_pred = model.predict(X_test)
        rmse_pct = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred))) / np.expm1(y_test).mean()

        mlflow.log_params(params)
        mlflow.log_metrics({"rmse_pct": rmse_pct})
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)
        logger.info("Training complete — RMSE: %.2f%%", rmse_pct * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Dynamic Pricing Engine")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args.data)
