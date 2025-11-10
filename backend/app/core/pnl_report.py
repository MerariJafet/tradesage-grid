"""PnL and drawdown report helpers for grid engine simulations."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

from .grid_engine import ExecutionRecord


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