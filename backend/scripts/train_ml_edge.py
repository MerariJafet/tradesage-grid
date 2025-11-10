import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.feature_engineering import engineer_features

DATA_PATH = Path("data/btc_1m_12months.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
REPORT_PATH = Path("walk_forward_report.txt")
DEFAULT_IMBALANCE = 0.1
DEFAULT_FUNDING_RATE = 0.0001


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_features(
    df: pd.DataFrame,
    imbalance: float = DEFAULT_IMBALANCE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
) -> pd.DataFrame:
    imbalance_series = pd.Series(imbalance, index=df.index)
    features = engineer_features(df, imbalance_series, funding_rate)
    non_numeric = features.select_dtypes(exclude=["number"]).columns
    if not non_numeric.empty:
        features = features.drop(columns=list(non_numeric))
    return features


def split_data(features: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Assuming scheme like "6m_train+3m_val+3m_forward" for 12 months
    total_len = len(features)
    train_end = int(total_len * 0.5)  # 6m / 12m
    val_end = int(total_len * 0.75)   # 3m more
    train = features.iloc[:train_end]
    val = features.iloc[train_end:val_end]
    forward = features.iloc[val_end:]
    return train, val, forward


def evaluate_on_period(features: pd.DataFrame, model: XGBClassifier) -> dict:
    X = features.drop(columns=["target"])
    y = features["target"]
    preds = model.predict(X)
    
    # Simulate PnL like in backtest
    future_close = features["close"].shift(-60)
    pnl = future_close - features["close"]
    trades = pnl[preds == 1].dropna()
    
    if len(trades) == 0:
        return {"win_rate": 0, "expectancy": 0, "profit_factor": 0, "sharpe_ratio": 0}
    
    win_rate = (trades > 0).mean()
    expectancy = trades.mean()
    
    winning = trades[trades > 0].sum()
    losing = abs(trades[trades < 0].sum())
    profit_factor = winning / losing if losing > 0 else float('inf')
    
    returns = trades / features.loc[trades.index, "close"]
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    return {
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML Edge model with walk-forward validation")
    parser.add_argument("--period", required=True, help="Period, e.g., 12months")
    parser.add_argument("--scheme", required=True, help="Scheme, e.g., 6m_train+3m_val+3m_forward")
    args = parser.parse_args()
    
    if args.period != "12months" or args.scheme != "6m_train+3m_val+3m_forward":
        raise ValueError("Unsupported period or scheme")
    
    df = load_dataset()
    features = build_features(df)
    
    train, val, forward = split_data(features, args.scheme)
    
    # Train on train
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Evaluate on forward
    forward_metrics = evaluate_on_period(forward, model)
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Write report
    report = f"""Walk-Forward Report - {args.scheme}
Period: {args.period}
Forward Metrics:
Win Rate: {forward_metrics['win_rate']:.1%}
Expectancy: ${forward_metrics['expectancy']:.2f}
Profit Factor: {forward_metrics['profit_factor']:.2f}
Sharpe Ratio: {forward_metrics['sharpe_ratio']:.2f}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("Report saved to walk_forward_report.txt")
    print(report)


if __name__ == "__main__":
    main()