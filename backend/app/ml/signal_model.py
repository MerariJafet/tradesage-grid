"""Lightweight ML pipeline to assist grid entries with probabilistic signals."""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:  # pragma: no cover - allow execution as a standalone script
    from ..core.grid_backtest import load_price_series
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_backtest import load_price_series  # type: ignore

DEFAULT_MODEL_PATH = Path("models/xgb_model.joblib")


@dataclass(slots=True)
class SignalPrediction:
    probability: float
    decision: bool
    threshold: float
    mode: str

    def as_dict(self) -> dict:
        return {
            "probability": self.probability,
            "decision": self.decision,
            "threshold": self.threshold,
            "mode": self.mode,
        }


def _expand_training_resources(patterns: Iterable[str]) -> List[Path]:
    resources: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            resources.extend(Path(match) for match in matches)
        else:
            resources.append(Path(pattern))
    return resources


def _build_feature_vector(window_slice: np.ndarray) -> np.ndarray:
    momentum = float(window_slice.sum())
    volatility = float(window_slice.std(ddof=1)) if window_slice.size > 1 else 0.0
    last_return = float(window_slice[-1])
    return np.concatenate([window_slice, np.array([momentum, volatility, last_return], dtype=np.float32)])


def prepare_features(
    prices: Sequence[float],
    *,
    window: int = 60,
    horizon: int = 5,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a price series into supervised learning samples."""

    arr = np.asarray(prices, dtype=np.float64)
    if arr.size <= window + horizon:
        return np.empty((0, window + 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    returns = np.diff(arr) / np.maximum(arr[:-1], 1e-8)
    features: List[np.ndarray] = []
    labels: List[float] = []

    limit = returns.size - horizon
    for idx in range(window, limit, stride):
        window_slice = returns[idx - window : idx].astype(np.float32)
        future_slice = returns[idx : idx + horizon]
        future_return = float(future_slice.sum())
        features.append(_build_feature_vector(window_slice))
        labels.append(1.0 if future_return > 0 else 0.0)

    if not features:
        return np.empty((0, window + 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.vstack(features), np.asarray(labels, dtype=np.float32)


def _latest_feature(prices: Sequence[float], *, window: int, horizon: int) -> np.ndarray | None:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size <= window + horizon:
        return None
    returns = np.diff(arr) / np.maximum(arr[:-1], 1e-8)
    window_slice = returns[-window:].astype(np.float32)
    return _build_feature_vector(window_slice)[None, :]


def train_model(
    resources: Iterable[Path],
    *,
    window: int = 60,
    horizon: int = 5,
    stride: int = 5,
    mode: str = "probability",
    limit: int | None = 20_000,
    test_size: float = 0.2,
    seed: int = 7,
) -> Tuple[XGBClassifier, dict]:
    """Train an XGBoost classifier and return metrics."""

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for path in resources:
        prices = load_price_series(path, limit=limit)
        if not prices:
            continue
        X, y = prepare_features(prices, window=window, horizon=horizon, stride=stride)
        if X.size == 0:
            continue
        all_features.append(X)
        all_labels.append(y)

    if not all_features:
        raise RuntimeError("Training aborted: no samples collected from provided datasets.")

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all if len(np.unique(y_all)) > 1 else None,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(np.float32)
    accuracy = float(accuracy_score(y_test, preds))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = math.nan

    metrics = {
        "samples": int(X_all.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "accuracy": accuracy,
        "auc": auc,
        "mode": mode,
        "window": window,
        "horizon": horizon,
        "stride": stride,
    }
    return model, metrics


class SignalModel:
    """Wrapper that loads a trained model and produces trading signals."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        mode: str = "probability",
        threshold: float = 0.55,
        window: int = 60,
        horizon: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.mode = mode
        self.threshold = threshold
        self.window = window
        self.horizon = horizon
        self.model: XGBClassifier | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(self, prices: Sequence[float]) -> SignalPrediction:
        if self.model is None:
            raise RuntimeError("SignalModel not loaded. Call load() first.")
        features = _latest_feature(prices, window=self.window, horizon=self.horizon)
        if features is None:
            return SignalPrediction(probability=0.5, decision=True, threshold=self.threshold, mode=self.mode)
        probability = float(self.model.predict_proba(features)[0][1])
        if self.mode == "binary":
            decision = probability >= 0.5
        else:
            decision = probability >= self.threshold
        return SignalPrediction(probability=probability, decision=decision, threshold=self.threshold, mode=self.mode)

    @classmethod
    def load_from_disk(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        *,
        mode: str = "probability",
        threshold: float = 0.55,
        window: int = 60,
        horizon: int = 5,
    ) -> "SignalModel":
        instance = cls(model_path=model_path, mode=mode, threshold=threshold, window=window, horizon=horizon)
        instance.load()
        return instance


def predict_signal(
    prices: Sequence[float],
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    mode: str = "probability",
    threshold: float = 0.55,
    window: int = 60,
    horizon: int = 5,
) -> SignalPrediction:
    model = SignalModel.load_from_disk(model_path, mode=mode, threshold=threshold, window=window, horizon=horizon)
    return model.predict(prices)


def _save_artifacts(model: XGBClassifier, metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = metrics | {"model_path": str(path)}
    meta_path = path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML signals to assist the grid engine")
    parser.add_argument("--train", nargs="+", help="Glob patterns or file paths with historical prices")
    parser.add_argument("--mode", choices=["binary", "probability"], default="probability")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train:
        print("⚠️ Provide at least one --train dataset path or glob pattern.")
        return

    resources = _expand_training_resources(args.train)
    try:
        model, metrics = train_model(
            resources,
            window=args.window,
            horizon=args.horizon,
            stride=args.stride,
            mode=args.mode,
            limit=args.limit,
            test_size=args.test_size,
            seed=args.seed,
        )
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    _save_artifacts(model, metrics | {"threshold": args.threshold, "mode": args.mode}, args.model_path)

    print("✅ Model trained and persisted")
    print(f"   Samples used: {metrics['samples']}")
    print(f"   Test accuracy: {metrics['accuracy']:.4f}")
    if not math.isnan(metrics["auc"]):
        print(f"   Test ROC-AUC: {metrics['auc']:.4f}")
    print(f"   Mode: {args.mode} | Threshold: {args.threshold:.2f}")
    print(f"   Window: {args.window} | Horizon: {args.horizon} | Stride: {args.stride}")
    print(f"   Artifact: {args.model_path}")


if __name__ == "__main__":
    main()
