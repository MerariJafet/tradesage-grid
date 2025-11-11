"""PnL and drawdown report helpers for grid engine simulations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

try:  # pragma: no cover
    from .grid_engine import ExecutionRecord
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.core.grid_engine import ExecutionRecord  # type: ignore


@dataclass(slots=True)
class PnLMetrics:
    dataset: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    average_pnl: float
    max_drawdown_pct: float
    max_equity: float
    min_equity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_pnl": self.net_pnl,
            "average_pnl": self.average_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_equity": self.max_equity,
            "min_equity": self.min_equity,
        }


class PnLReport:
    """Compute aggregate PnL metrics and export structured reports."""

    def __init__(self, executions: Iterable[ExecutionRecord], starting_capital: float, dataset: str) -> None:
        self.executions: List[ExecutionRecord] = list(executions)
        self.starting_capital = starting_capital
        self.dataset = dataset

    def compute_pnl(self) -> PnLMetrics:
        total = len(self.executions)
        gross_profit = sum(max(exec.pnl, 0.0) for exec in self.executions)
        gross_loss = sum(min(exec.pnl, 0.0) for exec in self.executions)
        net_pnl = gross_profit + gross_loss
        average_pnl = net_pnl / total if total else 0.0
        winning = sum(1 for exec in self.executions if exec.pnl > 0)
        losing = sum(1 for exec in self.executions if exec.pnl < 0)
        breakeven = total - winning - losing
        win_rate = (winning / total * 100.0) if total else 0.0

        equity_curve = self._build_equity_curve()
        max_dd_pct, max_equity, min_equity = self.compute_drawdown(equity_curve)

        return PnLMetrics(
            dataset=self.dataset,
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            breakeven_trades=breakeven,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pnl=net_pnl,
            average_pnl=average_pnl,
            max_drawdown_pct=max_dd_pct,
            max_equity=max_equity,
            min_equity=min_equity,
        )

    def compute_drawdown(self, equity_curve: Iterable[float]) -> Tuple[float, float, float]:
        curve = list(equity_curve)
        if not curve:
            return 0.0, self.starting_capital, self.starting_capital

        peak = curve[0]
        max_drawdown = 0.0
        max_equity = curve[0]
        min_equity = curve[0]

        for equity in curve:
            if equity > peak:
                peak = equity
            drawdown = 0.0 if peak <= 0 else (peak - equity) / peak * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if equity > max_equity:
                max_equity = equity
            if equity < min_equity:
                min_equity = equity
        return max_drawdown, max_equity, min_equity

    def export_reports(self, output_dir: Path, metrics: PnLMetrics) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "pnl_metrics.json"
        csv_path = output_dir / "pnl_metrics.csv"

        existing: Dict[str, Any] = {}
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                try:
                    existing = json.load(handle)
                except json.JSONDecodeError:
                    existing = {}

        existing[self.dataset] = metrics.to_dict()
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2)

        rows = list(existing.values())
        fieldnames = list(metrics.to_dict().keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return {"json": json_path, "csv": csv_path}

    def _build_equity_curve(self) -> List[float]:
        if not self.executions:
            return []

        equity_values = [exec.equity for exec in self.executions if exec.equity > 0]
        if len(equity_values) == len(self.executions):
            return equity_values

        curve: List[float] = []
        cumulative = self.starting_capital
        for exec in self.executions:
            cumulative += exec.pnl
            curve.append(cumulative)
        return curve


def _load_baseline_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    results: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row.get("dataset")
            if not dataset:
                continue
            metrics: Dict[str, float] = {}
            for key, value in row.items():
                if key == "dataset":
                    continue
                metrics[key] = float(value) if value not in {"", None} else 0.0
            results[dataset] = metrics
    return results


def _parse_ml_summary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"ML summary text not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip() for line in handle]

    parsed: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset: str | None = None
    section: str | None = None
    for raw_line in lines:
        if not raw_line.strip():
            continue
        if raw_line.startswith("Dataset:"):
            dataset = raw_line.split(":", 1)[1].strip()
            parsed[dataset] = {"baseline": {}, "ml": {}, "delta": {}}
            section = None
            continue
        if dataset is None:
            continue
        stripped = raw_line.strip()
        if stripped == "Baseline":
            section = "baseline"
            continue
        if stripped == "ML Enabled":
            section = "ml"
            continue
        if stripped.startswith("Δ Performance"):
            section = "delta"
            continue
        if section is None or ":" not in stripped:
            continue
        key, value = [part.strip() for part in stripped.split(":", 1)]
        parsed[dataset][section][key] = _safe_parse_float(value)
    return parsed


def _safe_parse_float(value: str) -> float:
    cleaned = value.replace(" pp", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


def _profit_factor_from_metrics(metrics: Dict[str, float]) -> float:
    gross_loss = metrics.get("gross_loss", 0.0)
    gross_profit = metrics.get("gross_profit", 0.0)
    if gross_loss >= 0:
        return float("inf") if gross_profit > 0 else 1.0
    denom = abs(gross_loss)
    if denom < 1e-9:
        return float("inf")
    return gross_profit / denom


def compare_reports(baseline_csv: Path, summary_txt: Path) -> List[str]:
    baseline = _load_baseline_metrics(baseline_csv)
    summary = _parse_ml_summary(summary_txt)
    lines: List[str] = []

    for dataset, base_metrics in baseline.items():
        ml_metrics = summary.get(dataset, {}).get("ml", {})
        delta_metrics = summary.get(dataset, {}).get("delta", {})

        base_pf = _profit_factor_from_metrics(base_metrics)
        base_drawdown = base_metrics.get("max_drawdown_pct", math.nan)

        ml_pf = ml_metrics.get("Profit factor", math.nan)
        ml_drawdown = ml_metrics.get("Max drawdown", math.nan)
        delta_pf = delta_metrics.get("ΔProfit factor", math.nan)
        delta_dd = delta_metrics.get("ΔDrawdown", math.nan)

        lines.append(f"Dataset: {dataset}")
        lines.append(f"  Baseline Profit Factor: {base_pf if math.isfinite(base_pf) else float('inf'):.4f}")
        lines.append(f"  Baseline Max Drawdown: {base_drawdown:.4f} %")
        lines.append(f"  ML Profit Factor: {ml_pf:.4f}")
        lines.append(f"  ML Max Drawdown: {ml_drawdown:.4f} %")
        lines.append(f"  Δ Profit Factor: {delta_pf:.4f}")
        lines.append(f"  Δ Max Drawdown: {delta_dd:.4f} pp")
        lines.append("  Sharpe Ratio: N/A (requires high-frequency returns)")
        lines.append("")

    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PnL report utilities")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_CSV", "ML_SUMMARY"), help="Compare baseline vs ML run outputs")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.compare:
        baseline_path = Path(args.compare[0])
        summary_path = Path(args.compare[1])
        report_lines = compare_reports(baseline_path, summary_path)
        print("\n".join(report_lines))


if __name__ == "__main__":
    main()