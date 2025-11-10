from typing import Any

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


def train_model(features_df: pd.DataFrame, target: str = "pnl_next_60min") -> Any:
    """Train a baseline XGBoost regressor for edge modeling."""
    if target not in features_df.columns:
        raise ValueError(f"Target column '{target}' not present in features dataframe")

    X = features_df.drop(columns=[target])
    y = features_df[target]

    model = XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X, y)
    return model


def train_xgboost(features_df: pd.DataFrame, target: str = "target") -> XGBClassifier:
    """Train an XGBoost classifier on engineered features."""
    if target not in features_df.columns:
        raise ValueError(f"Target column '{target}' not present in features dataframe")

    X = features_df.drop(columns=[target])
    y = features_df[target]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model
