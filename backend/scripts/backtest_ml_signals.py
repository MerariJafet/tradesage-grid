import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Ensure repository root is available for imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.train_real import (  # noqa: E402
    DEFAULT_FUNDING_RATE,
    DEFAULT_IMBALANCE,
    MODEL_PATH,
    build_features,
)


def load_model(model_path: Path = MODEL_PATH) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def generate_predictions(features_df: pd.DataFrame, model: XGBClassifier) -> pd.Series:
    feature_matrix = features_df.drop(columns=["target"], errors="ignore").copy()
    # Defensive drop of any non-numeric data that might have slipped in
    non_numeric = feature_matrix.select_dtypes(exclude=["number"]).columns
    if not non_numeric.empty:
        feature_matrix = feature_matrix.drop(columns=list(non_numeric))
    preds = model.predict(feature_matrix)
    return pd.Series(preds, index=features_df.index, name="signal")


def evaluate_trades(features_df: pd.DataFrame, signals: pd.Series) -> tuple[int, float, float, float, float]:
    combined = features_df.copy()
    combined["signal"] = signals

    future_close = combined["close"].shift(-60)
    pnl = future_close - combined["close"]
    combined["pnl"] = pnl

    trades = combined[combined["signal"] == 1].dropna(subset=["pnl"]).copy()
    trade_count = int(len(trades))
    if trade_count == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")

    win_rate = float((trades["pnl"] > 0).mean())
    expectancy = float(trades["pnl"].mean())
    
    # Calculate profit factor
    winning_trades = trades[trades["pnl"] > 0]
    losing_trades = trades[trades["pnl"] < 0]
    total_profit = winning_trades["pnl"].sum()
    total_loss = abs(losing_trades["pnl"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
    
    # Calculate Sharpe ratio (assuming daily returns, risk-free rate 0)
    # Convert to percentage returns
    returns = trades["pnl"] / trades["close"]
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else float("nan")
    
    return trade_count, win_rate, expectancy, profit_factor, sharpe_ratio


def main() -> None:
    features_df = build_features(DEFAULT_IMBALANCE, DEFAULT_FUNDING_RATE)
    if features_df.empty:
        raise ValueError("No features available. Ensure dataset generation and feature engineering succeeded.")

    model = load_model()
    signals = generate_predictions(features_df, model)
    trades, win_rate, expectancy, profit_factor, sharpe_ratio = evaluate_trades(features_df, signals)

    print(f"Trades ML: {trades}")
    if np.isnan(win_rate):
        print("Win Rate: N/A")
    else:
        print(f"Win Rate: {win_rate:.1%}")
    if np.isnan(expectancy):
        print("Expectancy: N/A")
    else:
        print(f"Expectancy: ${expectancy:.2f}")
    if np.isnan(profit_factor):
        print("Profit Factor: N/A")
    else:
        print(f"Profit Factor: {profit_factor:.2f}")
    if np.isnan(sharpe_ratio):
        print("Sharpe Ratio: N/A")
    else:
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
