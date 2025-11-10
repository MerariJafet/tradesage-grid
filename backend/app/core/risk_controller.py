"""Simple capital and drawdown risk guardrails for the grid engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class RiskLimitBreached(RuntimeError):
    """Raised when a safety rail is exceeded."""


@dataclass(slots=True)
class RiskSnapshot:
    equity: float
    drawdown_pct: float
    capital_in_use: float


class RiskController:
    """Tracks basic drawdown and capital usage constraints."""

    def __init__(
        self,
        *,
        initial_equity: float,
        max_drawdown_pct: float = 25.0,
        max_capital_fraction: float = 1.0,
    ) -> None:
        self.initial_equity = initial_equity
        self.max_drawdown_pct = max_drawdown_pct
        self.max_capital_fraction = max(0.0, min(1.0, max_capital_fraction))

        self._equity_peak = initial_equity
        self._equity = initial_equity
        self._capital_in_use = 0.0

    def allow_trade(self, notional: float) -> bool:
        """Return True when the trade keeps capital usage within bounds."""

        if notional <= 0:
            return False
        projected = self._capital_in_use + notional
        limit = self.initial_equity * self.max_capital_fraction
        return projected <= limit + 1e-8

    def register_fill(self, notional: float, is_long: bool) -> None:
        """Update capital in use based on executed fill."""

        delta = notional if is_long else -notional
        self._capital_in_use = max(self._capital_in_use + delta, 0.0)

    def update_equity(self, equity: float) -> RiskSnapshot:
        self._equity = equity
        if equity > self._equity_peak:
            self._equity_peak = equity
        drawdown = self.drawdown_pct
        if drawdown > self.max_drawdown_pct:
            raise RiskLimitBreached(
                f"Drawdown {drawdown:.2f}% exceeds limit {self.max_drawdown_pct:.2f}%"
            )
        return RiskSnapshot(equity=equity, drawdown_pct=drawdown, capital_in_use=self._capital_in_use)

    @property
    def drawdown_pct(self) -> float:
        if self._equity_peak <= 0:
            return 0.0
        return max(0.0, (1 - self._equity / self._equity_peak) * 100.0)

    @property
    def capital_in_use(self) -> float:
        return self._capital_in_use

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def equity_peak(self) -> float:
        return self._equity_peak

    def snapshot(self) -> RiskSnapshot:
        return RiskSnapshot(
            equity=self._equity,
            drawdown_pct=self.drawdown_pct,
            capital_in_use=self._capital_in_use,
        )

    def reset(self, initial_equity: Optional[float] = None) -> None:
        if initial_equity is not None:
            self.initial_equity = initial_equity
            self._equity = initial_equity
            self._equity_peak = initial_equity
        else:
            self._equity = self.initial_equity
            self._equity_peak = self.initial_equity
        self._capital_in_use = 0.0
