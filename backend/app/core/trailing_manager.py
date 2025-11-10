"""Trailing stop management for grid-based executions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(slots=True)
class TrailingState:
    position_id: str
    direction: str
    entry_price: float
    trailing_pct: float
    peak_price: float
    stop_price: float

    def update_peak(self, price: float) -> None:
        if self.direction == "long" and price > self.peak_price:
            self.peak_price = price
            self.stop_price = self._calculate_stop(price)
        elif self.direction == "short" and price < self.peak_price:
            self.peak_price = price
            self.stop_price = self._calculate_stop(price)

    def _calculate_stop(self, reference_price: float) -> float:
        ratio = self.trailing_pct / 100.0
        if self.direction == "long":
            return reference_price * (1 - ratio)
        return reference_price * (1 + ratio)

    def should_exit(self, price: float) -> bool:
        if self.direction == "long":
            return price <= self.stop_price
        return price >= self.stop_price


class TrailingManager:
    """Lightweight registry that tracks trailing stops per position."""

    def __init__(self, default_trailing_pct: float) -> None:
        self.default_trailing_pct = default_trailing_pct
        self._states: Dict[str, TrailingState] = {}

    def register(
        self,
        position_id: str,
        entry_price: float,
        direction: str = "long",
        trailing_pct: Optional[float] = None,
    ) -> TrailingState:
        pct = trailing_pct if trailing_pct is not None else self.default_trailing_pct
        pct = max(pct, 0.01)
        state = TrailingState(
            position_id=position_id,
            direction=direction,
            entry_price=entry_price,
            trailing_pct=pct,
            peak_price=entry_price,
            stop_price=self._initial_stop(entry_price, direction, pct),
        )
        self._states[position_id] = state
        return state

    def update(self, position_id: str, price: float) -> Optional[TrailingState]:
        state = self._states.get(position_id)
        if not state:
            return None
        state.update_peak(price)
        return state

    def should_exit(self, position_id: str, price: float) -> bool:
        state = self._states.get(position_id)
        if not state:
            return False
        return state.should_exit(price)

    def remove(self, position_id: str) -> None:
        self._states.pop(position_id, None)

    def get(self, position_id: str) -> Optional[TrailingState]:
        return self._states.get(position_id)

    @staticmethod
    def _initial_stop(entry_price: float, direction: str, trailing_pct: float) -> float:
        ratio = trailing_pct / 100.0
        if direction == "long":
            return entry_price * (1 - ratio)
        return entry_price * (1 + ratio)
