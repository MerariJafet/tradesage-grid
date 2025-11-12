"""Extended grid backtester capable of processing multiple datasets and exporting reports."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # pragma: no cover - allow running as standalone script
    from .grid_config import GridConfig
    from .grid_engine import GridEngine, ExecutionRecord
    from .pnl_report import PnLReport, PnLMetrics
    from .risk_controller import RiskController, RiskLimitBreached
    from .trailing_manager import TrailingManager
    from .grid_backtest import load_price_series, generate_synthetic_series
    from ..ml.signal_model import SignalModel
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_config import GridConfig  # type: ignore
    from app.core.grid_engine import GridEngine, ExecutionRecord  # type: ignore
    from app.core.pnl_report import PnLReport, PnLMetrics  # type: ignore
    from app.core.risk_controller import RiskController, RiskLimitBreached  # type: ignore
    from app.core.trailing_manager import TrailingManager  # type: ignore
    from app.core.grid_backtest import load_price_series, generate_synthetic_series  # type: ignore
    from app.ml.signal_model import SignalModel  # type: ignore


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
    parser.add_argument("--use-ml", action="store_true", help="Enable ML-assisted signal gating")
    parser.add_argument("--ml-mode", choices=["probability", "binary"], default="probability")
    parser.add_argument("--ml-threshold", type=float, default=0.55, help="Probability threshold when using probabilistic mode")
    parser.add_argument("--ml-window", type=int, default=60, help="Rolling window size for ML features")
    parser.add_argument("--ml-horizon", type=int, default=3, help="Prediction horizon for ML targets")
    parser.add_argument("--ml-model", type=Path, default=Path("models/xgb_model.joblib"), help="Path to the trained ML model artifact")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive grid adjustments driven by ML confidence")
    return parser.parse_args()


def discover_datasets(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    patterns = ["*.csv", "*.csv.gz"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))

    parquet_dirs = [p for p in data_dir.rglob("markprice") if p.is_dir()]
    return files + parquet_dirs


def prepare_prices(resource: Path, limit: int) -> Tuple[str, List[float]]:
    if resource.exists():
        series = load_price_series(resource, limit=limit)
        label = resource.name
        if resource.is_dir() and resource.parent.name:
            label = f"{resource.parent.name}_{resource.name}"
        return label, series
    return "synthetic", generate_synthetic_series(limit or 720)


def _profit_factor(metrics: PnLMetrics) -> float:
    gross_loss = metrics.gross_loss
    if gross_loss >= 0:
        return float("inf") if metrics.gross_profit > 0 else 1.0
    denominator = abs(gross_loss)
    if denominator < 1e-9:
        return float("inf")
    return metrics.gross_profit / denominator


def _write_metrics_csv(data: Dict[str, Dict[str, float]], path: Path) -> None:
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(next(iter(data.values())).keys())
    ordered_fields = ["dataset"] + [name for name in fieldnames if name != "dataset"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for dataset, metrics in data.items():
            row = {"dataset": dataset}
            row.update({k: v for k, v in metrics.items() if k != "dataset"})
            writer.writerow(row)


def _write_metrics_json(data: Dict[str, Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _equity_curve(executions: List[ExecutionRecord], starting_capital: float) -> List[float]:
    if not executions:
        return [starting_capital]
    curve = [exec.equity for exec in executions if exec.equity > 0]
    if curve:
        return curve
    cumulative = starting_capital
    points = [starting_capital]
    for exec in executions:
        cumulative += exec.pnl
        points.append(cumulative)
    return points


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def _save_adaptive_plot(
    dataset_label: str,
    baseline_execs: List[ExecutionRecord],
    ml_execs: List[ExecutionRecord],
    adaptive_execs: List[ExecutionRecord],
    starting_capital: float,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover
        return

    baseline_curve = _equity_curve(baseline_execs, starting_capital)
    ml_curve = _equity_curve(ml_execs, starting_capital)
    adaptive_curve = _equity_curve(adaptive_execs, starting_capital)

    if max(len(baseline_curve), len(ml_curve), len(adaptive_curve)) <= 1:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_curve, label="Baseline", linestyle="--", color="#7f7f7f")
    plt.plot(ml_curve, label="ML Enabled", linestyle="-", color="#1f77b4")
    plt.plot(adaptive_curve, label="Adaptive", linestyle="-", color="#d62728")
    plt.title(f"Adaptive Equity Curve: {dataset_label}")
    plt.xlabel("Execution Index")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    safe_name = _safe_label(dataset_label.lower())
    plt.tight_layout()
    plt.savefig(output_dir / f"adaptive_{safe_name}.png")
    plt.close()


def run_simulation(
    prices: Iterable[float],
    cfg: GridConfig,
    *,
    signal_model: SignalModel | None = None,
) -> Tuple[List[ExecutionRecord], str]:
    trailing_manager = TrailingManager(cfg.trailing_pct)
    risk_controller = RiskController(
        initial_equity=cfg.capital,
        max_drawdown_pct=cfg.max_drawdown_pct,
        max_capital_fraction=1.0,
    )
    engine = GridEngine(
        cfg,
        trailing_manager=trailing_manager,
        risk_controller=risk_controller,
        signal_model=signal_model,
    )

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

    if args.adaptive and not args.use_ml:
        print("⚠️ --adaptive requested without --use-ml. Adaptive mode requires ML predictions; flag will be ignored.")
        args.adaptive = False

    datasets = discover_datasets(args.data_dir)
    if not datasets:
        print(f"⚠️ No datasets found in {args.data_dir}. Falling back to synthetic price series.")
        datasets = [args.data_dir / "synthetic.csv"]

    warnings: Dict[str, str] = {}

    base_config = GridConfig(
        symbol=args.symbol,
        spacing_pct=args.spacing,
        levels=args.levels,
        order_size=args.order_size,
        capital=args.capital,
        ml_enabled=args.use_ml,
        ml_mode=args.ml_mode,
        ml_threshold=args.ml_threshold,
        ml_window=args.ml_window,
        ml_horizon=args.ml_horizon,
        adaptive_mode=args.adaptive,
    )

    if not args.use_ml:
        results: Dict[str, Dict[str, float]] = {}
        for dataset_path in datasets:
            dataset_label, prices = prepare_prices(dataset_path, args.limit)
            if not prices:
                warnings[dataset_label] = "Dataset yielded no price series. Skipping."
                continue

            cfg = GridConfig(**base_config.as_dict())
            cfg.ml_enabled = False
            cfg.adaptive_mode = False
            cfg.base_price = max(prices[0], 0.01)

            executions, notice = run_simulation(prices, cfg, signal_model=None)
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
        return

    try:
        signal_model = SignalModel.load_from_disk(
            args.ml_model,
            mode=args.ml_mode,
            threshold=args.ml_threshold,
            window=args.ml_window,
            horizon=args.ml_horizon,
        )
    except FileNotFoundError as exc:
        print(f"⚠️ ML requested but model not found: {exc}")
        signal_model = None

    baseline_metrics: Dict[str, Dict[str, float]] = {}
    ml_metrics: Dict[str, Dict[str, float]] = {}
    adaptive_metrics: Dict[str, Dict[str, float]] = {}

    plot_dir = args.reports_dir / "plots"

    for dataset_path in datasets:
        dataset_label, prices = prepare_prices(dataset_path, args.limit)
        if not prices:
            warnings[dataset_label] = "Dataset yielded no price series. Skipping."
            continue

        base_price = max(prices[0], 0.01)

        cfg_baseline = GridConfig(**base_config.as_dict())
        cfg_baseline.ml_enabled = False
        cfg_baseline.adaptive_mode = False
        cfg_baseline.base_price = base_price

        baseline_execs, baseline_notice = run_simulation(prices, cfg_baseline, signal_model=None)
        if baseline_notice:
            warnings[f"baseline::{dataset_label}"] = baseline_notice
        baseline_report = PnLReport(baseline_execs, cfg_baseline.capital, dataset_label)
        baseline_metric = baseline_report.compute_pnl()
        baseline_metrics[dataset_label] = baseline_metric.to_dict()

        cfg_ml = GridConfig(**base_config.as_dict())
        cfg_ml.ml_enabled = True
        cfg_ml.adaptive_mode = False
        cfg_ml.base_price = base_price

        ml_execs, ml_notice = run_simulation(prices, cfg_ml, signal_model=signal_model)
        if ml_notice:
            warnings[f"ml::{dataset_label}"] = ml_notice
        ml_report = PnLReport(ml_execs, cfg_ml.capital, dataset_label)
        ml_metric = ml_report.compute_pnl()
        ml_metrics[dataset_label] = ml_metric.to_dict()

        adaptive_execs: List[ExecutionRecord] = []
        adaptive_metric: PnLMetrics | None = None
        if args.adaptive:
            cfg_adaptive = GridConfig(**base_config.as_dict())
            cfg_adaptive.ml_enabled = True
            cfg_adaptive.adaptive_mode = True
            cfg_adaptive.base_price = base_price

            adaptive_execs, adaptive_notice = run_simulation(prices, cfg_adaptive, signal_model=signal_model)
            if adaptive_notice:
                warnings[f"adaptive::{dataset_label}"] = adaptive_notice
            adaptive_report = PnLReport(adaptive_execs, cfg_adaptive.capital, dataset_label)
            adaptive_metric = adaptive_report.compute_pnl()
            adaptive_metrics[dataset_label] = adaptive_metric.to_dict()

        print(f"Dataset: {dataset_label}")
        print("  Baseline")
        print(f"    Trades: {baseline_metric.total_trades}")
        print(f"    Net PnL: {baseline_metric.net_pnl:.4f}")
        print(f"    Win rate: {baseline_metric.win_rate:.2f}%")
        print(f"    Max drawdown: {baseline_metric.max_drawdown_pct:.2f}%")
        profit_factor_base = _profit_factor(baseline_metric)
        print(f"    Profit factor: {profit_factor_base:.4f}")

        profit_factor_ml = _profit_factor(ml_metric)
        print("  ML Enabled")
        print(f"    Trades: {ml_metric.total_trades}")
        print(f"    Net PnL: {ml_metric.net_pnl:.4f}")
        print(f"    Win rate: {ml_metric.win_rate:.2f}%")
        print(f"    Max drawdown: {ml_metric.max_drawdown_pct:.2f}%")
        print(f"    Profit factor: {profit_factor_ml:.4f}")

        print("  Δ ML vs Baseline")
        print(f"    ΔTrades: {ml_metric.total_trades - baseline_metric.total_trades}")
        print(f"    ΔNet PnL: {ml_metric.net_pnl - baseline_metric.net_pnl:.4f}")
        print(f"    ΔWin rate: {ml_metric.win_rate - baseline_metric.win_rate:.2f} pp")
        print(f"    ΔDrawdown: {ml_metric.max_drawdown_pct - baseline_metric.max_drawdown_pct:.2f} pp")
        print(f"    ΔProfit factor: {profit_factor_ml - profit_factor_base:.4f}")

        if adaptive_metric is not None:
            profit_factor_adaptive = _profit_factor(adaptive_metric)
            print("  Adaptive")
            print(f"    Trades: {adaptive_metric.total_trades}")
            print(f"    Net PnL: {adaptive_metric.net_pnl:.4f}")
            print(f"    Win rate: {adaptive_metric.win_rate:.2f}%")
            print(f"    Max drawdown: {adaptive_metric.max_drawdown_pct:.2f}%")
            print(f"    Profit factor: {profit_factor_adaptive:.4f}")

            print("  Δ Adaptive vs ML")
            print(f"    ΔTrades: {adaptive_metric.total_trades - ml_metric.total_trades}")
            print(f"    ΔNet PnL: {adaptive_metric.net_pnl - ml_metric.net_pnl:.4f}")
            print(f"    ΔWin rate: {adaptive_metric.win_rate - ml_metric.win_rate:.2f} pp")
            print(f"    ΔDrawdown: {adaptive_metric.max_drawdown_pct - ml_metric.max_drawdown_pct:.2f} pp")
            print(f"    ΔProfit factor: {profit_factor_adaptive - profit_factor_ml:.4f}")

            _save_adaptive_plot(
                dataset_label,
                baseline_execs,
                ml_execs,
                adaptive_execs,
                cfg_baseline.capital,
                plot_dir,
            )

        if f"baseline::{dataset_label}" in warnings:
            print(f"    Baseline warning: {warnings[f'baseline::{dataset_label}']}")
        if f"ml::{dataset_label}" in warnings:
            print(f"    ML warning: {warnings[f'ml::{dataset_label}']}")
        if f"adaptive::{dataset_label}" in warnings:
            print(f"    Adaptive warning: {warnings[f'adaptive::{dataset_label}']}")
        print()

    if baseline_metrics:
        _write_metrics_csv(baseline_metrics, args.reports_dir / "pnl_metrics_baseline.csv")
        _write_metrics_json(baseline_metrics, args.reports_dir / "pnl_metrics_baseline.json")
    if ml_metrics:
        _write_metrics_csv(ml_metrics, args.reports_dir / "pnl_metrics_ml.csv")
        _write_metrics_json(ml_metrics, args.reports_dir / "pnl_metrics_ml.json")
    if adaptive_metrics:
        _write_metrics_csv(adaptive_metrics, args.reports_dir / "pnl_metrics_adaptive.csv")
        _write_metrics_json(adaptive_metrics, args.reports_dir / "pnl_metrics_adaptive.json")

    if not ml_metrics:
        print("No ML-enabled simulations were recorded.")


if __name__ == "__main__":
    main()
