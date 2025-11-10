import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import ccxt
import pandas as pd


def create_exchange() -> ccxt.binance:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def fetch_ohlcv(exchange: ccxt.binance, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> List[List[float]]:
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)


def parse_iso8601(date_str: str, fallback: Optional[datetime] = None) -> datetime:
    if not date_str:
        if fallback is None:
            raise ValueError("No date provided and no fallback available")
        return fallback
    if date_str.endswith("Z"):
        date_str = date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(date_str).astimezone(timezone.utc)


def download(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    if start >= end:
        raise ValueError("Start date must be before end date")

    exchange = create_exchange()
    start_ms = exchange.parse8601(start.isoformat())
    end_ms = exchange.parse8601(end.isoformat())
    all_rows: List[List[float]] = []
    fetch_ms = start_ms

    while fetch_ms < end_ms:
        batch = fetch_ohlcv(exchange, symbol, timeframe, fetch_ms)
        if not batch:
            break
        all_rows.extend(batch)
        last_timestamp = batch[-1][0]
        if last_timestamp <= fetch_ms:
            break
        fetch_ms = last_timestamp + exchange.parse_timeframe(timeframe) * 1000
        if fetch_ms >= end_ms:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        raise RuntimeError("No OHLCV data retrieved.")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] <= end_ms]
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def build_default_output(symbol: str, timeframe: str) -> Path:
    normalized = symbol.replace("/", "").lower()
    return Path("data") / f"{normalized}_{timeframe}_12months.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV history from Binance futures.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g., BTC/USDT")
    parser.add_argument("--timeframe", default="1m", help="Binance timeframe, e.g., 1m")
    parser.add_argument("--start", default="2024-10-25T00:00:00Z", help="ISO8601 start date")
    parser.add_argument("--end", default="2025-10-25T00:00:00Z", help="ISO8601 end date")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_dt = parse_iso8601(args.start)
    end_dt = parse_iso8601(args.end)
    output_path = args.output or build_default_output(args.symbol, args.timeframe)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = download(args.symbol, args.timeframe, start_dt, end_dt)
    df.to_csv(output_path, index=False)
    print(f"Guardado: {len(df)} filas -> {output_path}")


if __name__ == "__main__":
    main()
