"""Comparative Backtesting Engine scaffolding for Sprint 9A.

This module will evolve to execute parallel simulations for three operating modes:
1) ML-driven
2) Adaptive (feedback loop only)
3) Hybrid (ML + adaptive adjustments)

Current implementation provides a lightweight aggregation helper so the sprint can
bootstrap quickly while more detailed comparative logic is designed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PERSISTENCE_DIR = Path(__file__).resolve().parents[1] / "core" / "persistence"
DB_PATH = PERSISTENCE_DIR / "trades.db"


@dataclass(slots=True)
class TradeRow:
	timestamp: datetime
	pnl: float
	exposure: float


@dataclass(slots=True)
class SignalRow:
	timestamp: datetime
	probability: float
	decision: bool


class ComparativeBacktester:
	"""Compute quick comparative aggregates using persisted trades and signals."""

	def __init__(self, db_path: Path | str = DB_PATH) -> None:
		self.db_path = Path(db_path)

	# ------------------------------------------------------------------
	# Data loading helpers
	# ------------------------------------------------------------------
	def _parse_timestamp(self, value: object) -> Optional[datetime]:
		if value is None:
			return None
		if isinstance(value, datetime):
			return value
		try:
			return datetime.fromisoformat(str(value))
		except ValueError:
			return None

	def _fetch_trades(self, conn: sqlite3.Connection) -> List[TradeRow]:
		try:
			cursor = conn.execute(
				"SELECT timestamp, pnl, exposure FROM trades ORDER BY timestamp ASC"
			)
		except sqlite3.Error:
			return []
		rows: List[TradeRow] = []
		for ts_raw, pnl, exposure in cursor.fetchall():
			ts = self._parse_timestamp(ts_raw)
			if ts is None:
				continue
			rows.append(
				TradeRow(timestamp=ts, pnl=float(pnl or 0.0), exposure=float(exposure or 0.0))
			)
		return rows

	def _fetch_signals(self, conn: sqlite3.Connection) -> List[SignalRow]:
		try:
			cursor = conn.execute(
				"SELECT created_at, probability, decision FROM ml_signals ORDER BY created_at ASC"
			)
		except sqlite3.Error:
			return []
		rows: List[SignalRow] = []
		for ts_raw, probability, decision in cursor.fetchall():
			ts = self._parse_timestamp(ts_raw)
			if ts is None:
				continue
			rows.append(
				SignalRow(timestamp=ts, probability=float(probability or 0.0), decision=bool(decision))
			)
		return rows

	def load_data(self) -> Tuple[List[TradeRow], List[SignalRow]]:
		if not self.db_path.exists():
			return [], []
		with sqlite3.connect(self.db_path) as conn:
			trades = self._fetch_trades(conn)
			signals = self._fetch_signals(conn)
		return trades, signals

	# ------------------------------------------------------------------
	# Aggregate helpers
	# ------------------------------------------------------------------
	def run_backtest(self) -> Dict[str, float]:
		trades, signals = self.load_data()
		adaptive_pnl = sum(trade.pnl for trade in trades)
		ml_probabilities = [row.probability for row in signals]
		ml_mean = float(sum(ml_probabilities) / len(ml_probabilities)) if ml_probabilities else 0.0
		combined_score = adaptive_pnl * ml_mean

		return {
			"adaptive_pnl": float(round(adaptive_pnl, 4)),
			"ml_signal_mean": float(round(ml_mean, 6)),
			"combined_score": float(round(combined_score, 4)),
		}


if __name__ == "__main__":  # pragma: no cover
	engine = ComparativeBacktester()
	print("\n=== Comparative Backtest Report ===")
	print(engine.run_backtest())
