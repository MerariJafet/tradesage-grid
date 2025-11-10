"""Visual reporting helpers for grid engine simulations."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")  # ensure headless compatibility
import matplotlib.pyplot as plt
import seaborn as sns

from .grid_engine import ExecutionRecord

sns.set_theme(style="darkgrid")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamps(executions: Iterable[ExecutionRecord]) -> List[datetime]:
    stamps: List[datetime] = []
    for index, exec in enumerate(executions):
        if exec.timestamp:
            stamps.append(exec.timestamp)
        else:
            stamps.append(datetime.utcnow())
    if not stamps:
        now = datetime.utcnow()
        return [now]
    return stamps


def plot_pnl_curve(executions: Iterable[ExecutionRecord], starting_capital: float, output_path: Path) -> None:
    executions = list(executions)
    _ensure_parent(output_path)

    cumulative = []
    total = starting_capital
    for exec in executions:
        total += exec.pnl
        cumulative.append(total)

    if not cumulative:
        cumulative = [starting_capital]
        timestamps = [datetime.utcnow()]
    else:
        timestamps = _timestamps(executions)

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, cumulative, label="Equity curve", color="#1f77b4")
    plt.axhline(starting_capital, linestyle="--", color="#ff7f0e", linewidth=1, label="Start capital")
    plt.title("Cumulative PnL")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(executions: Iterable[ExecutionRecord], starting_capital: float, output_path: Path) -> None:
    executions = list(executions)
    _ensure_parent(output_path)

    curve = []
    total = starting_capital
    for exec in executions:
        total += exec.pnl
        curve.append(total)

    if not curve:
        curve = [starting_capital]

    peaks = []
    drawdowns = []
    peak = curve[0]
    for value in curve:
        peak = max(peak, value)
        peaks.append(peak)
        if peak <= 0:
            drawdowns.append(0.0)
        else:
            drawdowns.append((peak - value) / peak * 100.0)

    timestamps = _timestamps(executions) if executions else [datetime.utcnow()]

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, drawdowns, color="#d62728", label="Drawdown %")
    plt.title("Drawdown Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_histogram(executions: Iterable[ExecutionRecord], output_path: Path) -> None:
    executions = list(executions)
    _ensure_parent(output_path)

    pnl_values = [exec.pnl for exec in executions]
    if not pnl_values:
        pnl_values = [0.0]

    plt.figure(figsize=(8, 4))
    sns.histplot(pnl_values, bins=20, kde=True, color="#2ca02c")
    plt.title("PnL Distribution per Trade")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
