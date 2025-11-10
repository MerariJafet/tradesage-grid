import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import websockets
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.train_real import DEFAULT_FUNDING_RATE, DEFAULT_IMBALANCE  # noqa: E402
from backend.app.ml.feature_engineering import engineer_features  # noqa: E402

MODEL_PATH = Path("models/xgboost_edge.model")
BUFFER_SIZE = 120


def load_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


async def live_trading(symbol: str = "BTCUSDT") -> None:
    model = load_model()
    uri = f"wss://fstream.binance.com/ws/{symbol.lower()}@depth10@100ms"
    print(f"LIVE TRADING INICIADO – {symbol}", flush=True)
    buffer: list[dict] = []

    async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
        while True:
            message = await ws.recv()
            payload = json.loads(message)
            bids = payload.get("b", [])
            asks = payload.get("a", [])
            if not bids or not asks:
                continue

            top_bid_prices = [float(level[0]) for level in bids[:5]]
            top_bid_sizes = [float(level[1]) for level in bids[:5]]
            top_ask_prices = [float(level[0]) for level in asks[:5]]
            top_ask_sizes = [float(level[1]) for level in asks[:5]]

            bid_volume = sum(top_bid_sizes)
            ask_volume = sum(top_ask_sizes)
            total_volume = bid_volume + ask_volume
            imbalance = 0.0 if total_volume == 0 else (bid_volume - ask_volume) / total_volume

            bar = {
                "open": top_bid_prices[0],
                "high": max(top_bid_prices),
                "low": min(top_bid_prices),
                "close": top_ask_prices[0],
                "volume": bid_volume + ask_volume,
            }
            buffer.append({**bar, "imbalance_rt": imbalance})
            if len(buffer) < 60:
                continue

            buffer = buffer[-BUFFER_SIZE:]
            df = pd.DataFrame(buffer)
            df_prices = df[["open", "high", "low", "close", "volume"]].copy()
            try:
                features = engineer_features(
                    df_prices,
                    imbalance=DEFAULT_IMBALANCE,
                    funding_rate=DEFAULT_FUNDING_RATE,
                )
            except Exception:  # pylint: disable=broad-except
                continue

            if features.empty:
                continue

            feature_matrix = features.drop(columns=["target"], errors="ignore")
            non_numeric = feature_matrix.select_dtypes(exclude=["number"]).columns
            if not non_numeric.empty:
                feature_matrix = feature_matrix.drop(columns=list(non_numeric))

            latest = feature_matrix.tail(1)
            if latest.empty:
                continue

            pred = model.predict(latest.values)
            if pred[0] == 1:
                timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                price = bar["close"]
                imbalance_str = f"{buffer[-1]['imbalance_rt']:+.3f}"
                print(f"[{timestamp}] SEÑAL BUY – Precio: {price:.2f} – Imbalance: {imbalance_str}", flush=True)


def main() -> None:
    asyncio.run(live_trading())


if __name__ == "__main__":
    main()
