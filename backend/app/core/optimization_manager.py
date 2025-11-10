"""Parameter sweep utilities to optimise GridEngine configurations."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - allow running as standalone script
    from .grid_config import GridConfig
    from .grid_engine import ExecutionRecord, GridEngine
    from .grid_backtest import load_price_series, generate_synthetic_series
    from .grid_backtest_v2 import run_simulation  # reuse execution helper
    from .pnl_report import PnLReport
    from .plot_report import plot_drawdown, plot_pnl_curve, plot_histogram
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_config import GridConfig  # type: ignore
    from app.core.grid_engine import ExecutionRecord, GridEngine  # type: ignore
    from app.core.grid_backtest import load_price_series, generate_synthetic_series  # type: ignore
    from app.core.grid_backtest_v2 import run_simulation  # type: ignore
    from app.core.pnl_report import PnLReport  # type: ignore
    from app.core.plot_report import plot_drawdown, plot_pnl_curve, plot_histogram  # type: ignore


@dataclass(slots=True)
class CandidateResult:
    dataset: str
    config_key: str
    metrics: Dict[str, float]
    profit_factor: float
    drawdown_pct: float
    warning: str = ""

    def score(self) -> float:
        if math.isinf(self.profit_factor):
            return float("inf")
        penalty = self.drawdown_pct / 100.0
        return self.profit_factor - penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid parameter optimisation harness")
    parser.add_argument("--data", type=Path, default=Path("data/raw"), help="Directory with CSV or CSV.GZ data")
    parser.add_argument("--limit", type=int, default=3000, help="Maximum rows per dataset")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Directory for optimisation output")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol label for reports")
    parser.add_argument("--spacing", type=str, default="0.15,0.25,0.35", help="Comma separated spacing percentages")
    parser.add_argument("--levels", type=str, default="4,6,8", help="Comma separated level counts")
    parser.add_argument("--capital", type=str, default="5000,10000", help="Comma separated capital allocations")
    return parser.parse_args()


def discover_datasets(data_root: Path) -> List[Path]:
    if not data_root.exists():
        return []
    patterns = ["*.csv", "*.csv.gz"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_root.glob(pattern)))
    return files


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def ensure_prices(resource: Path, limit: int) -> Tuple[str, List[float]]:
    if resource.exists():
        series = load_price_series(resource, limit=limit)
        return resource.name, series
    return "synthetic", generate_synthetic_series(limit or 720)


def build_config(base: GridConfig, spacing: float, levels: int, capital: float, base_price: float) -> GridConfig:
    cfg = GridConfig(**base.as_dict())
    cfg.spacing_pct = spacing
    cfg.levels = levels
    cfg.capital = capital
    cfg.base_price = max(base_price, 0.01)
    cfg.order_size = base.order_size
    return cfg


def compute_profit_factor(metrics: Dict[str, float]) -> float:
    gross_profit = metrics.get("gross_profit", 0.0)
    gross_loss = metrics.get("gross_loss", 0.0)
    if math.isclose(gross_loss, 0.0):
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def run_optimisation(args: argparse.Namespace) -> Tuple[List[CandidateResult], Dict[str, CandidateResult]]:
    datasets = discover_datasets(args.data)
    if not datasets:
        datasets = [args.data / "synthetic.csv"]

    spacing_values = parse_float_list(args.spacing)
    level_values = parse_int_list(args.levels)
    capital_values = parse_float_list(args.capital)

    base_config = GridConfig(symbol=args.symbol)

    all_results: List[CandidateResult] = []
    best_per_dataset: Dict[str, CandidateResult] = {}

    for dataset_path in datasets:
        dataset_label, prices = ensure_prices(dataset_path, args.limit)
        if not prices:
            continue
        dataset_results: List[CandidateResult] = []

        for spacing in spacing_values:
            for levels in level_values:
                for capital in capital_values:
                    cfg = build_config(base_config, spacing, levels, capital, prices[0])
                    executions, warning = run_simulation(prices, cfg)
                    report_label = f"{dataset_label}|s={spacing}|l={levels}|c={capital}"
                    report = PnLReport(executions, cfg.capital, report_label)
                    metrics_model = report.compute_pnl()
                    metrics_dict = metrics_model.to_dict()
                    profit_factor = compute_profit_factor(metrics_dict)
                    candidate = CandidateResult(
                        dataset=dataset_label,
                        config_key=f"spacing={spacing}|levels={levels}|capital={capital}",
                        metrics=metrics_dict,
                        profit_factor=profit_factor,
                        drawdown_pct=metrics_dict["max_drawdown_pct"],
                        warning=warning,
                    )
                    dataset_results.append(candidate)

        dataset_results.sort(key=lambda item: item.score(), reverse=True)
        if dataset_results:
            best_per_dataset[dataset_label] = dataset_results[0]
            all_results.extend(dataset_results)

    return all_results, best_per_dataset


def generate_reports(
    args: argparse.Namespace,
    results: List[CandidateResult],
    best_map: Dict[str, CandidateResult],
) -> Dict[str, Path]:
    reports_dir = args.reports
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    summary_path = reports_dir / "optimization_summary.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        if not results:
            handle.write("No optimisation results were generated.\n")
            return {"summary": summary_path}

        handle.write("GRID OPTIMISATION SUMMARY\n")
        handle.write(f"Generated: {timestamp} UTC\n\n")
        for dataset, best in best_map.items():
            handle.write(f"Dataset: {dataset}\n")
            handle.write(f"  Best configuration: {best.config_key}\n")
            handle.write(f"  Profit factor: {best.profit_factor:.4f}\n")
            handle.write(f"  Net PnL: {best.metrics['net_pnl']:.2f}\n")
            handle.write(f"  Max drawdown: {best.metrics['max_drawdown_pct']:.2f}%\n")
            handle.write("\n")

    generated: Dict[str, Path] = {"summary": summary_path}

    for dataset, best in best_map.items():
        label = dataset.replace(" ", "_")
        plot_base = f"{label}_{timestamp}"
        pnl_path = plots_dir / f"pnl_curve_{plot_base}.png"
        dd_path = plots_dir / f"drawdown_{plot_base}.png"
        hist_path = plots_dir / f"pnl_hist_{plot_base}.png"

        cfg_template = GridConfig(symbol=args.symbol)
        spacing = float(best.config_key.split("|")[0].split("=")[1])
        levels = int(best.config_key.split("|")[1].split("=")[1])
        capital = float(best.config_key.split("|")[2].split("=")[1])
        dataset_path = args.data / dataset if dataset != "synthetic" else args.data / "synthetic.csv"
        _, prices = ensure_prices(dataset_path, args.limit)
        cfg = build_config(cfg_template, spacing, levels, capital, prices[0])
        executions, _ = run_simulation(prices, cfg)

        plot_pnl_curve(executions, cfg.capital, pnl_path)
        plot_drawdown(executions, cfg.capital, dd_path)
        plot_histogram(executions, hist_path)

        generated[f"pnl_{dataset}"] = pnl_path
        generated[f"drawdown_{dataset}"] = dd_path
        generated[f"hist_{dataset}"] = hist_path

    return generated


def main() -> None:
    args = parse_args()
    results, best_map = run_optimisation(args)
    outputs = generate_reports(args, results, best_map)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
