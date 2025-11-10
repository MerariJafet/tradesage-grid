from __future__ import annotations

"""Utility helpers for risk-based position sizing.

This module centralizes the sizing logic shared across strategies and the
backtest engine so we avoid drifting implementations and recurring numerical
issues (e.g. underflow leading to ~1e-170 balances).
"""

from typing import Optional


DEFAULT_STARTING_BALANCE = 10_000.0
MIN_POSITION_SIZE = 0.001  # Hard floor to avoid zero-sized trades
MAX_POSITION_SIZE_PCT = 0.10  # Cap any single position to 10 % of balance
FALLBACK_STOP_DISTANCE_PCT = 0.001  # 0.1 % fallback when SL distance is 0


def calculate_position_size(
  balance: float,
  entry_price: float,
  stop_loss_price: float,
  risk_percent: float = 0.01,
  atr: Optional[float] = None,
  model_confidence: float = 1.0,
) -> float:
  """Compute position size with dynamic ATR-aware risk controls and model confidence."""

  effective_balance = balance if balance > 0 else DEFAULT_STARTING_BALANCE
  risk_amount = effective_balance * risk_percent

  stop_distance = abs(entry_price - stop_loss_price)
  if atr and atr > 0:
    stop_distance = max(stop_distance, 1.8 * atr)

  if stop_distance <= 0:
    stop_distance = entry_price * FALLBACK_STOP_DISTANCE_PCT

  raw_size = risk_amount / stop_distance if stop_distance else 0.0

  # Apply model confidence multiplier
  raw_size *= model_confidence

  max_notional_size = (effective_balance * MAX_POSITION_SIZE_PCT)
  max_size = max_notional_size / entry_price if entry_price else raw_size
  bounded_size = max(MIN_POSITION_SIZE, min(raw_size, max_size))

  return round(bounded_size, 6)
