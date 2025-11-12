"""Adaptive feedback loop utilizing recent PnL data for parameter tuning."""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PERSISTENCE_DIR = Path(__file__).resolve().parents[1] / "persistence"
DB_PATH = PERSISTENCE_DIR / "trades.db"


@dataclass
class FeedbackMetrics:
    mean_pnl: float
    volatility: float
    spacing_factor: float
    order_size_factor: float
    ml_threshold: float | None


class AdaptiveFeedbackLoop:
    """Read recent PnL from persistence storage and derive adjustments."""

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = max(1, window_size)

    def _fetch_recent_pnl(self) -> List[float]:
        if not DB_PATH.exists():
            return []
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT pnl FROM trades ORDER BY id DESC LIMIT ?", (self.window_size,)
                )
                rows = cursor.fetchall()
        except sqlite3.Error:
            return []
        return [float(row[0]) for row in rows if row and row[0] is not None]

    def compute_adjustments(self) -> Dict[str, float | None]:
        pnl_data = self._fetch_recent_pnl()
        if not pnl_data:
            return {
                "spacing": 1.0,
                "order_size": 1.0,
                "ml_threshold": None,
            }

        mean_pnl = statistics.mean(pnl_data)
        volatility = statistics.stdev(pnl_data) if len(pnl_data) > 1 else 0.0

        spacing_factor = 1.0 - max(-0.05, min(0.05, mean_pnl / 100.0))
        spacing_factor = max(0.8, min(1.2, spacing_factor))
        order_size_factor = 1.0 + max(-0.05, min(0.05, mean_pnl / 200.0))
        order_size_factor = max(0.8, min(1.2, order_size_factor))
        ml_threshold = 0.7 + (-0.05 if mean_pnl > 0 else 0.05)
        ml_threshold = max(0.50, min(0.80, ml_threshold))

        print(
            "[FEEDBACK] mean={:.4f} vol={:.4f} spacing={:.3f} size={:.3f} ml_threshold={:.2f}".format(
                mean_pnl, volatility, spacing_factor, order_size_factor, ml_threshold
            )
        )
        return {
            "spacing": spacing_factor,
            "order_size": order_size_factor,
            "ml_threshold": ml_threshold,
        }


if __name__ == "__main__":  # pragma: no cover
    loop = AdaptiveFeedbackLoop()
    loop.compute_adjustments()
