import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


BINANCE_FUNDING_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"


class FundingTracker:
    """Fetch and store the latest funding rate observations."""

    def __init__(self) -> None:
        self.records: List[dict] = []

    def record(self, payload: dict) -> None:
        self.records.append(payload)

    def latest(self) -> Optional[dict]:
        return self.records[-1] if self.records else None

    def fetch_latest(self, symbol: str) -> dict:
        params = f"?symbol={symbol.upper()}&limit=1"
        try:
            with urlopen(BINANCE_FUNDING_ENDPOINT + params, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Error fetching funding rate for {symbol}: {exc}") from exc
        if not payload:
            raise RuntimeError(f"No funding data returned for {symbol}")
        point = payload[0]
        rate = float(point["fundingRate"])
        funding_time = int(point["fundingTime"]) / 1000
        return {
            "symbol": symbol.upper(),
            "funding_rate": rate,
            "funding_time": funding_time,
        }


def format_record(record: dict) -> Tuple[str, str]:
    timestamp = datetime.fromtimestamp(record["funding_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    humans = f"{record['funding_rate'] * 100:.4f}%"
    line = f"[{timestamp}] {record['symbol']} Funding Rate: {humans}"
    return line, humans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the latest Binance futures funding rate.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to query, default BTCUSDT")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = FundingTracker()
    record = tracker.fetch_latest(args.symbol)
    tracker.record(record)
    line, _ = format_record(record)
    print(line)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="ascii", errors="ignore") as sink:
            sink.write(line + "\n")


if __name__ == "__main__":
    main()
