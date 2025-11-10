"""Minimal CLI backtester to validate the GridEngine behaviour."""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - fallback when executed as script
    from .grid_config import GridConfig
    from .grid_engine import GridEngine
    from .risk_controller import RiskController, RiskLimitBreached
    from .trailing_manager import TrailingManager
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_config import GridConfig  # type: ignore
    from app.core.grid_engine import GridEngine  # type: ignore
    from app.core.risk_controller import RiskController, RiskLimitBreached  # type: ignore
    from app.core.trailing_manager import TrailingManager  # type: ignore


def load_price_series(path: Path, column: str = "close", limit: int | None = None) -> List[float]:
    if not path.exists():
        return generate_synthetic_series(limit or 720)

    if path.is_dir():
        parquet_prices = _load_parquet_prices(path, limit)
        if parquet_prices:
            return parquet_prices
        first_csv = next(path.glob("*.csv"), None)
        if first_csv:
            path = first_csv
        else:
            return generate_synthetic_series(limit or 720)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        prices: List[float] = []
        normalised_column = column.lower()
        for row in reader:
            value = _extract_price(row, normalised_column)
            if value is None:
                continue
            prices.append(value)
            if limit and len(prices) >= limit:
                break
    if not prices:
        return generate_synthetic_series(limit or 720)
    return prices


def _load_parquet_prices(path: Path, limit: int | None) -> List[float]:
    try:
        import duckdb
    except ImportError:  # pragma: no cover
        return []

    pattern = str(path / "**/*.parquet")
    con = duckdb.connect()
    try:
        candidates = ["mark_price", "close", "price"]
        for column in candidates:
            try:
                query = (
                    f"SELECT {column} FROM read_parquet('{pattern}') "
                    "ORDER BY timestamp"
                )
                if limit:
                    query += f" LIMIT {limit}"
                rows = con.execute(query).fetchall()
            except duckdb.ConversionException:
                continue
            except duckdb.BinderException:
                continue
            if rows:
                return [float(row[0]) for row in rows]
    finally:
        con.close()
    return []


def _extract_price(row: dict, column: str) -> float | None:
    keys = [column, column.upper(), column.capitalize(), "close"]
    for key in keys:
        if key in row and row[key]:
            try:
                return float(row[key])
            except ValueError:
                continue
    return None


def generate_synthetic_series(length: int) -> List[float]:
    base = 100.0
    volatility = 0.002
    values = [base]
    for _ in range(length - 1):
        drift = random.uniform(-volatility, volatility)
        next_price = max(values[-1] * (1 + drift), 1.0)
        values.append(next_price)
    return values


def run_backtest(price_series: Iterable[float], config: GridConfig) -> None:
    prices = list(price_series)
    if not prices:
        print("⚠️ No price data available. Aborting simulation.")
        return

    config = GridConfig(**config.as_dict())
    config.base_price = prices[0]

    trailing_manager = TrailingManager(config.trailing_pct)
    risk_controller = RiskController(
        initial_equity=config.capital,
        max_drawdown_pct=config.max_drawdown_pct,
        max_capital_fraction=1.0,
    )
    engine = GridEngine(config, trailing_manager=trailing_manager, risk_controller=risk_controller)

    print(f"🚀 Running grid backtest for {config.symbol} | levels={config.levels} spacing={config.spacing_pct}%")
    total_fills = 0

    for step, price in enumerate(prices):
        try:
            fills = engine.update_price(price)
        except RiskLimitBreached as exc:
            print(f"⛔ Risk limit triggered at step {step}: {exc}")
            break
        if not fills:
            continue
        total_fills += len(fills)
        for fill in fills:
            pnl = f"PNL {fill.pnl:+.2f}"
            print(
                f"[{fill.timestamp.isoformat()}] {fill.side.upper()} price={fill.price:.2f} "
                f"size={fill.size:.6f} -> equity={fill.equity:.2f} {pnl}"
            )

    summary = engine.state()
    equity = engine._compute_equity(prices[-1])  # internal helper OK for CLI reporting
    print("\n📈 Backtest completed")
    print(f"Trades executed: {total_fills}")
    print(f"Final position: {summary['position']:.6f}")
    print(f"Cash: {summary['cash']:.2f}")
    print(f"Realized PnL: {summary['realized_pnl']:.2f}")
    print(f"Equity (mark-to-market): {equity:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick validation harness for the GridEngine")
    parser.add_argument(
        "--data",
        type=Path,
        help="CSV or CSV.GZ file with at least a close column",
        default=Path("data/btc_1m_12months.csv.gz"),
    )
    parser.add_argument("--limit", type=int, default=2000, help="Maximum rows to ingest from the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    series = load_price_series(args.data.expanduser(), limit=args.limit)
    cfg = GridConfig()
    run_backtest(series, cfg)
