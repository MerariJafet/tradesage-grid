"""Extended grid backtester capable of processing multiple datasets and exporting reports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - allow running as standalone script
    from .grid_config import GridConfig
    from .grid_engine import GridEngine, ExecutionRecord
    from .pnl_report import PnLReport
    from .risk_controller import RiskController, RiskLimitBreached
    from .trailing_manager import TrailingManager
    from .grid_backtest import load_price_series, generate_synthetic_series
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_config import GridConfig  # type: ignore
    from app.core.grid_engine import GridEngine, ExecutionRecord  # type: ignore
    from app.core.pnl_report import PnLReport  # type: ignore
    from app.core.risk_controller import RiskController, RiskLimitBreached  # type: ignore
    from app.core.trailing_manager import TrailingManager  # type: ignore
    from app.core.grid_backtest import load_price_series, generate_synthetic_series  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch backtester for the Grid + Trailing engine")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing CSV or CSV.GZ datasets")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum rows to read from each dataset")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Output directory for reports")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to annotate within reports")
    parser.add_argument("--spacing", type=float, default=0.25, help="Grid spacing percentage")
    parser.add_argument("--levels", type=int, default=6, help="Number of levels above/below base price")
    parser.add_argument("--order-size", type=float, default=0.001, help="Order size per grid level")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital for the strategy")
    return parser.parse_args()


def discover_datasets(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    patterns = ["*.csv", "*.csv.gz"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))
    return files


def prepare_prices(resource: Path, limit: int) -> Tuple[str, List[float]]:
    if resource.exists():
        series = load_price_series(resource, limit=limit)
        return resource.name, series
    return "synthetic", generate_synthetic_series(limit or 720)


def run_simulation(prices: Iterable[float], cfg: GridConfig) -> Tuple[List[ExecutionRecord], str]:
    trailing_manager = TrailingManager(cfg.trailing_pct)
    risk_controller = RiskController(
        initial_equity=cfg.capital,
        max_drawdown_pct=cfg.max_drawdown_pct,
        max_capital_fraction=1.0,
    )
    engine = GridEngine(cfg, trailing_manager=trailing_manager, risk_controller=risk_controller)

    executions: List[ExecutionRecord] = []
    warning = ""
    for step, price in enumerate(prices):
        try:
            fills = engine.update_price(price)
        except RiskLimitBreached as exc:
            warning = f"Risk limit triggered at step {step}: {exc}"
            break
        executions.extend(fills)
    return executions, warning


def main() -> None:
    args = parse_args()

    datasets = discover_datasets(args.data_dir)
    if not datasets:
        print(f"⚠️ No datasets found in {args.data_dir}. Falling back to synthetic price series.")
        datasets = [args.data_dir / "synthetic.csv"]

    results: Dict[str, Dict[str, float]] = {}
    warnings: Dict[str, str] = {}

    base_config = GridConfig(
        symbol=args.symbol,
        spacing_pct=args.spacing,
        levels=args.levels,
        order_size=args.order_size,
        capital=args.capital,
    )

    for dataset_path in datasets:
        dataset_label, prices = prepare_prices(dataset_path, args.limit)
        if not prices:
            warnings[dataset_label] = "Dataset yielded no price series. Skipping."
            continue

        cfg = GridConfig(**base_config.as_dict())
        cfg.base_price = max(prices[0], 0.01)

        executions, notice = run_simulation(prices, cfg)
        if notice:
            warnings[dataset_label] = notice

        report = PnLReport(executions, cfg.capital, dataset_label)
        metrics = report.compute_pnl()
        results[dataset_label] = metrics.to_dict()
        report.export_reports(args.reports_dir, metrics)

        print(f"Dataset: {dataset_label}")
        print(f"  Trades: {metrics.total_trades}")
        print(f"  Net PnL: {metrics.net_pnl:.2f}")
        print(f"  Win rate: {metrics.win_rate:.2f}%")
        print(f"  Max drawdown: {metrics.max_drawdown_pct:.2f}%")
        if dataset_label in warnings:
            print(f"  Warning: {warnings[dataset_label]}")
        print()

    if not results:
        print("No successful simulations were recorded.")
        return

    summary_json = args.reports_dir / "pnl_metrics.json"
    summary_csv = args.reports_dir / "pnl_metrics.csv"
    print("Summary files generated:")
    print(f"  JSON: {summary_json}")
    print(f"  CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
