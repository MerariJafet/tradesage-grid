import sys
from pathlib import Path

from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.train_real import (  # noqa: E402
    DEFAULT_FUNDING_RATE,
    DEFAULT_IMBALANCE,
    MODEL_PATH,
    build_features,
)

SYMBOLS = ["ETHUSDT", "BNBUSDT"]


def main() -> None:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    for symbol in SYMBOLS:
        dataset_path = Path("data") / f"{symbol.lower()}_1m_12months.csv"
        print(f"\n=== {symbol} ===")
        if not dataset_path.exists():
            print(f"Dataset missing: {dataset_path}")
            continue

        try:
            features = build_features(
                imbalance=DEFAULT_IMBALANCE,
                funding_rate=DEFAULT_FUNDING_RATE,
                dataset_path=dataset_path,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to engineer features for {symbol}: {exc}")
            continue

        feature_matrix = features.drop(columns=["target"], errors="ignore").copy()
        non_numeric = feature_matrix.select_dtypes(exclude=["number"]).columns
        if not non_numeric.empty:
            feature_matrix = feature_matrix.drop(columns=list(non_numeric))

        preds = model.predict(feature_matrix)
        features = features.copy()
        features["signal"] = preds

        future_close = features["close"].shift(-60)
        pnl = future_close - features["close"]
        features["pnl"] = pnl
        trades = features[features["signal"] == 1].dropna(subset=["pnl"]).copy()

        trade_count = len(trades)
        if trade_count == 0:
            print("Trades: 0")
            print("Win Rate: N/A")
            print("Expectancy: N/A")
            continue

        win_rate = float((trades["pnl"] > 0).mean())
        expectancy = float(trades["pnl"].mean())

        print(f"Trades: {trade_count}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Expectancy: ${expectancy:.2f}")


if __name__ == "__main__":
    main()
