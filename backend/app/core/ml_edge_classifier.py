import joblib
from pathlib import Path
from typing import Optional

import pandas as pd
from xgboost import XGBClassifier


class MLEdgeClassifier:
    """ML Edge Classifier for loading and using the trained XGBoost model."""

    def __init__(self, model_path: Optional[Path] = None, confidence_threshold: float = 0.5):
        self.model_path = model_path or Path("models/model.joblib")
        self.confidence_threshold = confidence_threshold
        self.model: Optional[XGBClassifier] = None

    def load_model(self) -> None:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions on the given features DataFrame, filtered by confidence."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        numeric_features = features.select_dtypes(include=["number"]).drop(columns=["target"], errors="ignore")
        preds = self.model.predict(numeric_features)
        proba = self.model.predict_proba(numeric_features)[:, 1]
        # Filter by confidence
        filtered_preds = pd.Series([p if prob > self.confidence_threshold else 0 for p, prob in zip(preds, proba)], index=features.index)
        return filtered_preds

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make probability predictions on the given features DataFrame."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        numeric_features = features.select_dtypes(include=["number"]).drop(columns=["target"], errors="ignore")
        proba = self.model.predict_proba(numeric_features)
        return pd.DataFrame(proba, index=features.index, columns=["prob_0", "prob_1"])